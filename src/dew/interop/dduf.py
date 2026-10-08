"""DDUF files: a diffusers pipeline directory packed into one uncompressed ZIP.

A `.dduf` file (huggingface_hub's `export_folder_as_dduf`) holds exactly the
files of a diffusers directory: model_index.json, and each component's config
and weights under its own folder. huggingface_hub's `read_dduf_file` checks
the archive (stored, not compressed, with model_index.json at its root) and
lists each entry's byte range. `unpacked` writes those entries once into Dew's
cache (`dew_cache_dir()/dduf/<key>`), then the pipeline loads from there as
the directory it was packed from (`Pretrained.load(..., dduf_file=)`), as
diffusers' own `from_pretrained(..., dduf_file=)` reads it.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath

from dew.cache import dew_cache_dir
from dew.files import staged

_CHUNK = 1 << 24


def _key(path: Path) -> str:
    """The cache key of one DDUF file: its resolved path, size and
    modification time, which stand in for a hash of its bytes.

    Hashing the contents would read the whole archive on every cache hit,
    the cost the cache exists to avoid. A Hub file resolves to its
    content-addressed blob, so for a published file the key is a content
    key. A local file relies on its mtime: rewriting it in place at the
    same size within one mtime tick, or `cp -p` of another file of the same
    size over it, reuses the stale unpack; deleting `dew_cache_dir()/dduf`
    clears it."""
    status = path.stat()
    identity = f"{path.resolve()}\0{status.st_size}\0{status.st_mtime_ns}"
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


def unpacked(path: str | os.PathLike[str]) -> Path:
    """The diffusers directory `path` packs, written once into Dew's cache.

    The directory is published with one rename once every entry is written,
    so an interrupted unpack leaves no partial directory to be read."""
    from huggingface_hub import read_dduf_file

    source = Path(path)
    target = Path(dew_cache_dir()) / "dduf" / _key(source)
    if (target / "model_index.json").is_file():
        return target
    entries = read_dduf_file(source)
    with staged(target) as staging, open(source, "rb") as archive:
        for name, entry in entries.items():
            # huggingface_hub 1.30.0's read_dduf_file checks each name with
            # its slashes stripped but keys the entry by the name as
            # written, so an absolute name or a '..' would land outside.
            parts = PurePosixPath(name)
            if parts.is_absolute() or ".." in parts.parts:
                raise ValueError(f"{source} names an entry {name!r} outside its own directory")
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            archive.seek(entry.offset)
            remaining = entry.length
            with open(destination, "wb") as out:
                while remaining:
                    chunk = archive.read(min(_CHUNK, remaining))
                    if not chunk:
                        raise ValueError(f"{source} ends inside its entry {name}")
                    out.write(chunk)
                    remaining -= len(chunk)
    return target
