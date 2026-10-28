"""ArrayRecord access, with pinned bounded mirrors for non-native URLs."""

from __future__ import annotations

import io
import itertools
import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property

import grain.python as grain

from dew.batch import Batch
from dew.data.dataset import Loading, Ranges
from dew.data.storage import Store, local, on_gcs

from .codec import decode

_MAX_OPEN_SHARDS = Loading().threads


@dataclass
class _OpenedShard:
    reader: ArrayRecords
    reads: int = 0
    used: int = 0
    """When a read last asked for it, on the process's clock of reads (`ShardReaders._clock`)."""


class ShardReaders[K]:
    """At most `limit` shards' readers open at once, each opened by `open` for its key as a read
    first asks for it and closed as the least recent idle one makes room: the bound every physical
    shard's mirror is sized by (`_MAX_OPEN_SHARDS`, Loading's default read threads). Busy readers stay
    open until their reads drain; with as many readers as read threads, a thread asking for a new
    shard holds none, so at least one retained reader is idle and can release its mirror before the
    new open. More read threads than retained readers queue on missing shard opens until a reader
    drains. Readers retired by close still occupy that bound until their reads drain.

    The idle readers of every source a process reads are bounded together by the same count
    (`_trim`): a mixture of hundreds of sources keeps the files one source keeps, the least recently
    read closed as others are read, while each source's own reads never wait on another's."""

    _live: weakref.WeakSet[ShardReaders] = weakref.WeakSet()
    _clock = itertools.count()
    _trimming = threading.Lock()

    def __init__(self, open: Callable[[K], ArrayRecords], limit: int = _MAX_OPEN_SHARDS):
        self.open, self.limit = open, limit
        self._opened: OrderedDict[K, _OpenedShard] = OrderedDict()
        self._retired: list[_OpenedShard] = []
        self._lock = threading.Condition(threading.Lock())
        ShardReaders._live.add(self)

    def read(self, key: K, row: int) -> Batch | bytes | None:
        """Row `row` of shard `key`'s reader, opened for it where none is."""
        with self._lock:
            while (
                key not in self._opened
                and len(self._opened) + len(self._retired) >= self.limit
                and all(opened.reads for opened in self._opened.values())
            ):
                self._lock.wait()
            opening = key not in self._opened
            if opening:
                if len(self._opened) + len(self._retired) >= self.limit:
                    idle = next(held for held, opened in self._opened.items() if not opened.reads)
                    self._opened.pop(idle).reader.close()
                self._opened[key] = _OpenedShard(self.open(key))
            self._opened.move_to_end(key)
            opened_shard = self._opened[key]
            opened_shard.reads += 1
            opened_shard.used = next(ShardReaders._clock)
        try:
            return opened_shard.reader[row]
        finally:
            with self._lock:
                opened_shard.reads -= 1
                # A reopened shard owns a different reader from an evicted in-flight read.
                if not opened_shard.reads:
                    if self._opened.get(key) is not opened_shard:
                        self._retired.remove(opened_shard)
                        opened_shard.reader.close()
                    self._lock.notify_all()
            if opening:
                ShardReaders._trim()

    @classmethod
    def _trim(cls) -> None:
        """Close the least recently read idle readers of every source in the process past the bound
        one source keeps (`_MAX_OPEN_SHARDS`), as a read opens another: the idle readers then number
        at most that and the reads in flight. It holds one source's lock at a time, never while it
        waits for another's, so no read waits on a source it does not read."""
        with cls._trimming:
            idle = []
            for readers in list(cls._live):
                with readers._lock:
                    idle.extend((opened.used, id(readers), key, readers)
                                for key, opened in readers._opened.items() if not opened.reads)
            idle.sort(key=lambda entry: entry[:2])
            for _, _, key, readers in idle[:max(0, len(idle) - _MAX_OPEN_SHARDS)]:
                with readers._lock:
                    opened = readers._opened.get(key)
                    if opened is not None and not opened.reads:
                        readers._opened.pop(key).reader.close()
                        readers._lock.notify_all()

    def close(self) -> None:
        """Close every idle reader, retiring busy ones until their reads drain."""
        with self._lock:
            for opened in self._opened.values():
                if opened.reads:
                    self._retired.append(opened)
                else:
                    opened.reader.close()
            self._opened.clear()
            self._lock.notify_all()


class ArrayRecords:
    """Read shards concurrently, retaining as many readers as default read threads (`ShardReaders`),
    and as many idle readers as that with every other source the process reads.

    Shuffled indexed reads reuse the LRU across Loading's in-flight read width.
    Non-native shards need a mirror with room for that many physical shards.
    """

    def __init__(
        self,
        urls: Sequence[str],
        storage_options: Mapping[str, object] | None = None,
        *,
        record_codec: bool = False,
        counts: Sequence[int] | None = None,
    ):
        self.urls = tuple(urls)
        self.storage_options = storage_options
        self.record_codec = record_codec
        self.counts = None if counts is None else tuple(counts)
        if self.counts is not None and (
            len(self.counts) != len(self.urls) or any(count < 0 for count in self.counts)
        ):
            raise ValueError("ArrayRecord counts must name every physical shard with a nonnegative row count")
        self._pins = []
        self._shards = ShardReaders(self._shard)

    def _shard(self, shard: int) -> ArrayRecords:
        """Physical shard `shard`'s own reader, checked against its producer's row count."""
        assert self.counts is not None
        opened = ArrayRecords((self.urls[shard],), self.storage_options, record_codec=self.record_codec)
        if len(opened) != self.counts[shard]:
            opened.close()
            raise ValueError(f"ArrayRecord {self.urls[shard]} differs from its producer's row count")
        return opened

    @cached_property
    def ranges(self) -> Ranges:
        return Ranges(self.counts or ())

    @cached_property
    def reader(self):
        paths = []
        for url in self.urls:
            fs, path = Store(url, self.storage_options).resolve()
            if local(fs):
                paths.append(path)
            elif on_gcs(fs):
                paths.append("gs://" + path)
            else:
                fs, path = Store(url, self.storage_options).cached().resolve()
                pin = fs.open(path, "rb")
                if not isinstance(pin, io.BufferedReader) or not isinstance(pin.name, str):
                    pin.close()
                    raise ValueError(
                        f"ArrayRecord {url} needs mirror room for {_MAX_OPEN_SHARDS} physical shards "
                        "(Loading's default read threads)"
                    )
                self._pins.append(pin)
                paths.append(pin.name)
        return grain.ArrayRecordDataSource(paths)

    def __len__(self) -> int:
        if not self.urls:
            return 0
        return len(self.ranges) if self.counts is not None else len(self.reader)

    def __getitem__(self, index: int) -> Batch | bytes | None:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.counts is not None:
            shard, row = self.ranges.locate(index)
            return self._shards.read(shard, row)
        payload = self.reader[index]
        if not self.record_codec:
            return payload
        value = decode(payload)
        if not isinstance(value, (Mapping, bytes)) and value is not None:
            raise TypeError("an exported ArrayRecord source must hold records or bytes")
        return value

    def __getstate__(self) -> Mapping[str, object]:
        return {
            "urls": self.urls,
            "storage_options": self.storage_options,
            "record_codec": self.record_codec,
            "counts": self.counts,
        }

    def __setstate__(self, state: Mapping[str, object]) -> None:
        self.__dict__.update(state)
        self._pins = []
        self._shards = ShardReaders(self._shard)

    def close(self) -> None:
        self._shards.close()
        self.__dict__.pop("reader", None)
        for pin in self._pins:
            pin.close()
        self._pins.clear()
