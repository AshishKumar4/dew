"""Storage capabilities and the source boundary used by pipeline recipes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike, fspath

import pyarrow as pa

from .base import Read, Shard, ShardManifest, Source
from .files import Files, Format
from .hub import Hub
from .memory import Memory
from .tfds import TFDS


def source(value: object, *, format: Format | None = None, split: str | None = None,
           provision: bool = False) -> Source:
    if isinstance(value, Source):
        result = value
    elif isinstance(value, (str, PathLike)):
        url = fspath(value)
        if url.startswith("tfds://"):
            name, _, version = url[7:].partition(":")
            result = TFDS(name=name, version=version or None, split=split or "train")
        elif url.startswith("hf://") and not any(character in url for character in ("*", ".parquet", ".arrow")):
            repo, _, revision = url.removeprefix("hf://").removeprefix("datasets/").partition("@")
            result = Hub(repo=repo, revision=revision or None, split=split or "train")
        else:
            result = Files(url, format=format)
    elif isinstance(value, pa.Table):
        result = Memory(value.to_pylist())
    elif isinstance(value, (Mapping, Sequence)):
        result = Memory(value)
    else:
        raise TypeError(f"a source is a filesystem URL, named store, or records; got {type(value).__name__}")
    if provision and isinstance(result, (Files, Hub, TFDS)):
        result.provision()
    return result


__all__ = ["Files", "Hub", "Memory", "Read", "Shard", "ShardManifest", "Source", "TFDS"]
