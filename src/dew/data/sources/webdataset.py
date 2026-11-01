"""Sequential WebDataset tar shards with checkpointed record positions."""

from __future__ import annotations

import tarfile
from collections.abc import Mapping
from contextlib import ExitStack

import grain.python as grain

from dew.data.declared import Declared
from dew.data.storage import Store


class TarRows(grain.IterDataset):
    def __init__(self, url: str, storage_options: Mapping[str, object] | None = None,
                 declared: Declared | None = None):
        super().__init__()
        # The archive as its producer declared it where one did (`Declared`), else as its store holds it.
        self.url, self.storage_options, self.declared = url, storage_options, declared

    def __iter__(self) -> grain.DatasetIterator:
        return _TarIterator(self)


class _TarIterator(grain.DatasetIterator):
    def __init__(self, source: TarRows):
        super().__init__()
        self.source = source
        self.position = 0
        self.pending: tuple[str, str, bytes] | None = None
        self.opened = ExitStack()
        if source.declared is not None:
            self.file = self.opened.enter_context(source.declared.open())
        else:
            fs, path = Store(source.url, source.storage_options).resolve()
            self.file = self.opened.enter_context(fs.open(path, "rb"))
        self.archive = self.opened.enter_context(tarfile.TarFile.open(fileobj=self.file, mode="r|*"))
        self.members = iter(self.archive)

    def _member(self) -> tuple[str, str, bytes]:
        if self.pending is not None:
            value, self.pending = self.pending, None
            return value
        for member in self.members:
            if not member.isfile():
                continue
            key, dot, extension = member.name.rpartition(".")
            if not dot:
                raise ValueError(f"WebDataset member {member.name!r} has no extension")
            file = self.archive.extractfile(member)
            if file is None:
                raise OSError(f"WebDataset member {member.name!r} is unreadable")
            with file:
                return key, extension, file.read()
        raise StopIteration

    def __next__(self) -> dict[str, bytes | str]:
        key, extension, payload = self._member()
        row: dict[str, bytes | str] = {"__key__": key, extension: payload}
        while True:
            try:
                next_key, extension, payload = self._member()
            except StopIteration:
                break
            if next_key != key:
                self.pending = next_key, extension, payload
                break
            if extension in row:
                raise ValueError(f"WebDataset key {key!r} has two {extension!r} members")
            row[extension] = payload
        self.position += 1
        return row

    def get_state(self) -> dict[str, int]:
        return {"position": self.position}

    def set_state(self, state: Mapping[str, object]) -> None:
        position = state["position"]
        if not isinstance(position, int) or position < 0:
            raise ValueError("invalid WebDataset reader position")
        self.close()
        self.__init__(self.source)
        for _ in range(position):
            next(self)

    def close(self) -> None:
        self.opened.close()
