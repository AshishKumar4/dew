"""ArrayRecord access, with pinned bounded mirrors for non-native URLs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cached_property

import grain.python as grain

from dew.data.dataset import Batch
from dew.data.storage import Store

from .codec import decode


class ArrayRecords:
    def __init__(self, urls: Sequence[str], storage_options: Mapping[str, object] | None = None,
                 *, record_codec: bool = False):
        self.urls = tuple(urls)
        self.storage_options = storage_options
        self.record_codec = record_codec
        self._pins = []

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
                options = dict(self.storage_options or {})
                fs, path = Store(url if url.startswith("dewcache::") else "dewcache::" + url, options).resolve()
                pin = fs.open(path, "rb")
                if not isinstance(pin.name, str):
                    pin.close()
                    raise ValueError(f"ArrayRecord {url} needs a mirror large enough to hold one shard")
                self._pins.append(pin)
                paths.append(pin.name)
        return grain.ArrayRecordDataSource(paths)

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, index: int) -> Batch | bytes | None:
        data = self.reader[index]
        if not self.record_codec:
            return data
        value = decode(data)
        if not isinstance(value, (Mapping, bytes)) and value is not None:
            raise TypeError("an exported ArrayRecord source must hold records or bytes")
        return value

    def __getstate__(self) -> Mapping[str, object]:
        return {"urls": self.urls, "storage_options": self.storage_options, "record_codec": self.record_codec}

    def __setstate__(self, state: Mapping[str, object]) -> None:
        self.__dict__.update(state)
        self._pins = []

    def close(self) -> None:
        self.__dict__.pop("reader", None)
        for pin in self._pins:
            pin.close()
        self._pins.clear()
