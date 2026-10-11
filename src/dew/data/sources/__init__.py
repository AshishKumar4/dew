"""Storage capabilities and the source boundary used by pipeline recipes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike, fspath
from typing import Protocol, runtime_checkable

from dew.data.dataset import Batch

from .base import Read, Shard, ShardManifest, Source
from .files import Files, Format, TokenFiles
from .hub import Hub
from .memory import Memory
from .tfds import TFDS


@runtime_checkable
class _ArrowRows(Protocol):
    def to_pylist(self) -> list[Batch]: ...


def source(value: object, *, format: Format | None = None, split: str | None = None,
           provision: bool = False) -> Source:
    if isinstance(value, Source):
        resolved = value
    elif isinstance(value, (str, PathLike)):
        url = fspath(value)
        if url.startswith("tfds://"):
            name, _, version = url[7:].partition(":")
            resolved = TFDS(name=name, version=version or None, split=split or "train")
        elif url.startswith("hf://") and not any(character in url for character in ("*", ".parquet", ".arrow")):
            repo, _, revision = url.removeprefix("hf://").removeprefix("datasets/").partition("@")
            resolved = Hub(repo=repo, revision=revision or None, split=split or "train")
        else:
            from dew.data.storage import Store

            fs, path = Store(url).resolve()
            token_file = path.rstrip("/") + f"/{split or 'train'}.bin"
            resolved = TokenFiles(url, split or "train") if format is None and fs.exists(token_file) else Files(url, format=format)
    elif isinstance(value, _ArrowRows):
        resolved = Memory(value.to_pylist())
    elif isinstance(value, (Mapping, Sequence)):
        resolved = Memory(value)
    else:
        raise TypeError(f"a source is a filesystem URL, named store, or records; got {type(value).__name__}")
    if provision and isinstance(resolved, (Files, Hub, TFDS)):
        resolved.provision()
    return resolved


__all__ = ["Files", "Hub", "Memory", "Read", "Shard", "ShardManifest", "Source", "TFDS"]
