"""ArrayRecord access, with pinned bounded mirrors for non-native URLs."""

from __future__ import annotations

import io
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
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


class ArrayRecords:
    """Read shards concurrently, retaining as many readers as default read threads.

    Shuffled indexed reads reuse the LRU across Loading's in-flight read width.
    Non-native shards need a mirror with room for that many physical shards.
    Busy readers stay open until their reads drain. With as many readers as
    read threads, a thread asking for a new shard holds none, so at least one
    retained reader is idle and can release its mirror before the new open.
    More read threads than retained readers queue on missing shard opens
    until a reader drains.
    Readers retired by close still occupy that bound until their reads drain.
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
        self._opened: OrderedDict[int, _OpenedShard] = OrderedDict()
        self._retired: list[_OpenedShard] = []
        self._lock = threading.Condition(threading.Lock())

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
            with self._lock:
                while (
                    shard not in self._opened
                    and len(self._opened) + len(self._retired) >= _MAX_OPEN_SHARDS
                    and all(opened.reads for opened in self._opened.values())
                ):
                    self._lock.wait()
                if shard not in self._opened:
                    if len(self._opened) + len(self._retired) >= _MAX_OPEN_SHARDS:
                        idle = next(key for key, opened in self._opened.items() if not opened.reads)
                        self._opened.pop(idle).reader.close()
                    opened = ArrayRecords(
                        (self.urls[shard],), self.storage_options, record_codec=self.record_codec
                    )
                    if len(opened) != self.counts[shard]:
                        opened.close()
                        raise ValueError(
                            f"ArrayRecord {self.urls[shard]} differs from its producer's row count"
                        )
                    self._opened[shard] = _OpenedShard(opened)
                self._opened.move_to_end(shard)
                opened_shard = self._opened[shard]
                opened_shard.reads += 1
            try:
                return opened_shard.reader[row]
            finally:
                with self._lock:
                    opened_shard.reads -= 1
                    # A reopened shard owns a different reader from an evicted in-flight read.
                    if not opened_shard.reads:
                        if self._opened.get(shard) is not opened_shard:
                            self._retired.remove(opened_shard)
                            opened_shard.reader.close()
                        self._lock.notify_all()
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
        self._opened = OrderedDict()
        self._retired = []
        self._lock = threading.Condition(threading.Lock())

    def close(self) -> None:
        with self._lock:
            for opened in self._opened.values():
                if opened.reads:
                    self._retired.append(opened)
                else:
                    opened.reader.close()
            self._opened.clear()
            self._lock.notify_all()
        self.__dict__.pop("reader", None)
        for pin in self._pins:
            pin.close()
        self._pins.clear()
