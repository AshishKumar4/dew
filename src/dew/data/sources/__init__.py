"""Storage capabilities and the source boundary used by pipeline recipes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from os import PathLike, fspath
from typing import Protocol, runtime_checkable

from dew.data.dataset import Batch

from .base import Read, Shard, ShardManifest, Source
from .files import Files, Format, _inferred
from .grain import Grain, from_grain
from .hf import HF, from_hf
from .hub import Hub
from .iterator import Iterated, from_iter
from .memory import Memory
from .pytorch import Torch, from_torch
from .tfds import TFDS
from .tokens import TokenFiles


@runtime_checkable
class _ArrowRows(Protocol):
    def to_pylist(self) -> list[Batch]: ...


def source(
    value: object, *, format: Format | None = None, split: str | None = None, provision: bool = False
) -> Source:
    if isinstance(value, Source):
        resolved = value
    elif isinstance(value, (str, PathLike)):
        url = fspath(value)
        if url.startswith("tfds://"):
            name, _, version = url[7:].partition(":")
            resolved = TFDS(name=name, version=version or None, split=split or "train")
        elif url.startswith("hf://") and format is None and "*" not in url and _inferred(url) is None:
            # A repository's own URL; a file or glob in it, or any URL given a format, is read as files.
            repo, _, revision = url.removeprefix("hf://").removeprefix("datasets/").partition("@")
            resolved = Hub(repo=repo, revision=revision or None, split=split or "train")
        else:
            from dew.data.storage import Store

            fs, path = Store(url).resolve()
            token_file = path.rstrip("/") + f"/{split or 'train'}.bin"
            files = Files(url, format=format)
            committed = files.committed
            metadata = None if committed is None else committed.get("metadata")
            native_tokens = isinstance(metadata, Mapping) and metadata.get("token_store") is True
            metadata_path = path.rstrip("/") + "/meta.json"
            if format is None and not native_tokens and fs.exists(metadata_path):
                metadata = json.loads(fs.cat_file(metadata_path))
                native_tokens = (
                    isinstance(metadata, Mapping) and {"tokenizer", "dtype", "vocab_size"} <= metadata.keys()
                )
            resolved = (
                TokenFiles(url, split or "train")
                if format is None and (native_tokens or fs.exists(token_file))
                else files
            )
    elif isinstance(value, _ArrowRows):
        resolved = Memory(value.to_pylist())
    elif isinstance(value, (Mapping, Sequence)):
        resolved = Memory(value)
    else:
        raise TypeError(f"a source is a filesystem URL, named store, or records; got {type(value).__name__}")
    if provision and isinstance(resolved, (Files, Hub, TFDS, TokenFiles)):
        resolved.provision()
    return resolved


__all__ = [
    "HF",
    "TFDS",
    "Files",
    "Grain",
    "Hub",
    "Iterated",
    "Memory",
    "Read",
    "Shard",
    "ShardManifest",
    "Source",
    "TokenFiles",
    "Torch",
    "from_grain",
    "from_hf",
    "from_iter",
    "from_torch",
]
