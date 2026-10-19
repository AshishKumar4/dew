"""Delta's Arrow dataset, pinned to the table version stated by its log."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

from .files import Files, Format


@dataclass(frozen=True)
class Delta(Files):
    version: int | None = None

    @cached_property
    def table(self):
        try:
            from deltalake import DeltaTable
        except ImportError as error:
            raise ImportError("reading Delta tables needs pip install 'dewml[delta]'") from error
        if not isinstance(self.url, str):
            raise ValueError("Delta takes one table URL")
        return DeltaTable(self.url, version=self.version, storage_options=self.storage_options)

    @cached_property
    def kind(self) -> Format:
        return "parquet"

    @cached_property
    def arrow(self):
        return self.table.to_pyarrow_dataset()

    def _shard_name(self, path: str) -> str:
        """The file as the table's log names it, where Delta's dataset reads it."""
        return path

    @cached_property
    def paths(self) -> tuple[str, ...]:
        return tuple(self.table.file_uris())

    def fingerprint(self) -> Mapping[str, object]:
        """The table at its resolved version, whose log names immutable files, read with the
        files' options."""
        return {
            "store": "delta",
            "uri": self.url,
            "version": self.table.version(),
            "files": self.table.file_uris(),
            **self._read_options,
        }
