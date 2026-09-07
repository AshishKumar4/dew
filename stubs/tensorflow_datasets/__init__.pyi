"""The public prepared-read surface used by Dew.

TFDS 4.9.10's logging decorator makes Pyright 1.1.406 infer a zero-argument
DatasetBuilder.as_data_source. These signatures follow core/dataset_builder.py
and core/read_only_builder.py. Records remain generic because TFDS features
and decoders determine their structure; they are not always image dictionaries.
"""

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Protocol, TypeAlias, TypeVar, overload

import tensorflow_datasets.core as core
from tensorflow_datasets.core import decode as decode, download as download, features as features
from tensorflow_datasets.core.decode.partial_decode import DecoderArg
from tensorflow_datasets.core.splits import SplitArg

__all__ = ["core", "decode", "download", "features", "builder_from_directory"]

Record = TypeVar("Record")
SplitTree: TypeAlias = SplitArg | list["SplitTree"] | tuple["SplitTree", ...] | dict[str, "SplitTree"]
DecoderTree: TypeAlias = DecoderArg | dict[str, "DecoderTree"]
SourceTree: TypeAlias = Sequence[Record] | list["SourceTree[Record]"] | tuple["SourceTree[Record]", ...] | dict[str, "SourceTree[Record]"]

class _FileInstruction(Protocol):
    @property
    def filename(self) -> str: ...

class _SplitInfo(Protocol):
    @property
    def file_instructions(self) -> Sequence[_FileInstruction]: ...

class _DatasetInfo(Protocol):
    @property
    def file_format(self) -> core.FileFormat | None: ...
    @property
    def splits(self) -> Mapping[str, _SplitInfo]: ...

class _PreparedBuilder(Protocol[Record]):
    @property
    def info(self) -> _DatasetInfo: ...

    @overload
    def as_data_source(
        self, split: None = None, *, decoders: DecoderTree | None = None,
        deserialize_method: decode.DeserializeMethod = ...,
        file_format: str | core.FileFormat | None = None,
    ) -> dict[str, Sequence[Record]]: ...
    @overload
    def as_data_source(
        self, split: SplitArg, *, decoders: DecoderTree | None = None,
        deserialize_method: decode.DeserializeMethod = ...,
        file_format: str | core.FileFormat | None = None,
    ) -> Sequence[Record]: ...
    @overload
    def as_data_source(
        self, split: list[SplitTree], *, decoders: DecoderTree | None = None,
        deserialize_method: decode.DeserializeMethod = ...,
        file_format: str | core.FileFormat | None = None,
    ) -> list[SourceTree[Record]]: ...
    @overload
    def as_data_source(
        self, split: tuple[SplitTree, ...], *, decoders: DecoderTree | None = None,
        deserialize_method: decode.DeserializeMethod = ...,
        file_format: str | core.FileFormat | None = None,
    ) -> tuple[SourceTree[Record], ...]: ...
    @overload
    def as_data_source(
        self, split: dict[str, SplitTree], *, decoders: DecoderTree | None = None,
        deserialize_method: decode.DeserializeMethod = ...,
        file_format: str | core.FileFormat | None = None,
    ) -> dict[str, SourceTree[Record]]: ...

def builder_from_directory(
    builder_dir: str | PathLike[str],
    file_format: str | core.FileFormat | None = None,
) -> _PreparedBuilder[object]: ...
