"""Storage capabilities and the source boundary every chain reads through."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike, fspath

from .base import Read, Shard, ShardManifest, Source
from .concatenate import Concatenate
from .files import Files, Format, _inferred
from .grain import Grain, from_grain
from .hf import HF, from_hf
from .hub import Hub
from .iterator import Iterated, from_iter
from .memory import DataFrame, Memory
from .pytorch import Torch, from_torch
from .tfds import TFDS
from .tokens import _SPLIT as _ONE_SPLIT, TokenFiles


def source(
    value: object, *, format: Format | None = None, split: str | None = None, provision: bool = False
) -> Source:
    """The store `value` names: a `Source` as it is; a `tfds://name[:version]` or `hf://owner/repo[@revision]`
    URL as its TFDS or Hub dataset, `split` its split ("train" by default); any other URL as its files
    (`Files`, `format` naming their record format where the URL's suffix does not), or as a committed
    token export (`TokenFiles`) where its manifest says it is one; and records in memory (a sequence
    of rows, a mapping of columns or a data frame) as `Memory`. Files and records read what they
    hold, whatever split is named: a split is a Hub or TFDS URL's. With `provision`, the store builds
    the indexes its reads need first."""
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
            files = Files(url, format=format)
            committed = files.committed
            metadata = None if committed is None else committed.get("metadata")
            # A committed token export reads as its token store, unless a format names its files.
            tokens = format is None and isinstance(metadata, Mapping) and metadata.get("token_store") is True
            resolved = TokenFiles(url) if tokens else files
    elif isinstance(value, (DataFrame, Mapping, Sequence)):
        resolved = Memory(value)
    else:
        raise TypeError(f"a source is a filesystem URL, named store, or records; got {type(value).__name__}")
    if split not in (None, _ONE_SPLIT) and isinstance(resolved, TokenFiles):
        raise ValueError(f"{resolved.describe()} is a token export, which holds one split; split={split!r} "
                         "is an export of its own, its TokenFiles(url)")
    if provision and isinstance(resolved, (Files, Hub, TFDS, TokenFiles)):
        resolved.provision()
    return resolved


__all__ = [
    "HF",
    "TFDS",
    "Concatenate",
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
    "source",
]
