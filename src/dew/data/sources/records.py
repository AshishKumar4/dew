"""ArrayRecord access, with pinned bounded mirrors for non-native URLs."""

from __future__ import annotations

import bisect
import itertools
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from functools import cached_property

import grain.python as grain

from dew.data.dataset import Batch
from dew.data.storage import Store

from .codec import decode


class ArrayRecords:
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
        self._opened: OrderedDict[int, ArrayRecords] = OrderedDict()
        self._lock = threading.RLock()

    @cached_property
    def ends(self) -> tuple[int, ...]:
        return tuple(itertools.accumulate(self.counts or ()))

    @cached_property
    def reader(self):
        paths = []
        for url in self.urls:
            fs, path = Store(url, self.storage_options).resolve()
            protocols = (fs.protocol,) if isinstance(fs.protocol, str) else fs.protocol
            if "file" in protocols:
                paths.append(path)
            elif "gs" in protocols or "gcs" in protocols:
                paths.append("gs://" + path)
            else:
                fs, path = Store(url, self.storage_options).cached().resolve()
                pin = fs.open(path, "rb")
                if not isinstance(pin.name, str):
                    pin.close()
                    raise ValueError(f"ArrayRecord {url} needs a mirror large enough to hold one shard")
                self._pins.append(pin)
                paths.append(pin.name)
        return grain.ArrayRecordDataSource(paths)

    def __len__(self) -> int:
        if not self.urls:
            return 0
        return self.ends[-1] if self.counts is not None else len(self.reader)

    def __getitem__(self, index: int) -> Batch | bytes | None:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.counts is not None:
            shard = bisect.bisect_right(self.ends, index)
            with self._lock:
                if shard not in self._opened:
                    opened = ArrayRecords(
                        (self.urls[shard],), self.storage_options, record_codec=self.record_codec
                    )
                    if len(opened) != self.counts[shard]:
                        opened.close()
                        raise ValueError(
                            f"ArrayRecord {self.urls[shard]} differs from its producer's row count"
                        )
                    self._opened[shard] = opened
                    if len(self._opened) > 2:
                        _, removed = self._opened.popitem(last=False)
                        removed.close()
                self._opened.move_to_end(shard)
                return self._opened[shard][index - (self.ends[shard - 1] if shard else 0)]
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
        self._lock = threading.RLock()

    def close(self) -> None:
        for opened in self._opened.values():
            opened.close()
        self._opened.clear()
        self.__dict__.pop("reader", None)
        for pin in self._pins:
            pin.close()
        self._pins.clear()
