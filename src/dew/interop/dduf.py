"""DDUF files: a diffusers pipeline directory packed into one uncompressed ZIP.

A `.dduf` file (`huggingface_hub.export_folder_as_dduf`) holds the files of a
diffusers directory: `model_index.json`, and each component's config and weights
under its own folder. Its entries and its structure are huggingface_hub's to read
and check: a local archive's by its public `read_dduf_file`, a Hub archive's
through a stream by that reader vendored for one (`dew.interop.dduf_stream`). An
entry's bytes are read by their range alone, so a load without weights
(`metadata`) reads no weight payload; `unpacked` writes every entry out.
"""

from __future__ import annotations

import base64
import hashlib
import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import IO

import jax
import numpy as np

from dew import _records
from dew._builds import dew_cache_dir
from dew._cache import hub_answer, recorded_hub_answer
from dew._files import staged, stat_identity
from dew.interop import sources
from dew.interop.safetensors_io import stored_spec

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_CHUNK = 1 << 24
Entries = Mapping[str, tuple[int, int]]
"""A DDUF archive's entries, each its data's offset in the archive and its length."""


def _key(path: Path) -> str:
    """The cache key of one DDUF file, its `stat_identity`; deleting
    `dew_cache_dir()/dduf` clears a stale unpack."""
    return hashlib.sha256("\0".join(map(str, stat_identity(path))).encode()).hexdigest()[:24]


def _contained(entries: Entries, label: str) -> Entries:
    """`entries`, refused where one names a path outside the directory it unpacks into, which the DDUF
    format's own checks leave (`read_dduf_file` admits `../x.json`)."""
    for name in entries:
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts:
            raise ValueError(f"{label} names an entry {name!r} outside its own directory")
    return entries


def _local_entries(path: Path) -> Entries:
    """A local DDUF archive's entries, its structure checked (huggingface_hub's public `read_dduf_file`)."""
    from huggingface_hub import read_dduf_file

    return _contained({name: (entry.offset, entry.length) for name, entry in read_dduf_file(path).items()},
                      str(path))


def _ranged(stream: IO[bytes], offset: int, length: int, label: str) -> bytes:
    """The `length` bytes of `stream` at `offset`, read by their range alone."""
    stream.seek(offset)
    held = stream.read(length)
    if len(held) != length:
        raise ValueError(f"{label} ends inside its entry")
    return held


def unpacked(path: str | os.PathLike[str]) -> Path:
    """The diffusers directory `path` packs, written once into Dew's cache.

    The directory is published with one rename once every entry is written,
    so an interrupted unpack leaves no partial directory to be read."""
    source = Path(path)
    target = Path(dew_cache_dir()) / "dduf" / _key(source)
    if (target / "model_index.json").is_file():
        return target
    with staged(target) as staging, open(source, "rb") as archive:
        for name, (offset, length) in _local_entries(source).items():
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with open(destination, "wb") as out:
                for start in range(0, length, _CHUNK):
                    piece = min(_CHUNK, length - start)
                    out.write(_ranged(archive, offset + start, piece, f"{source} {name}"))
    return target


def _scanned(stream: IO[bytes], entries: Entries, label: str) -> tuple[dict[str, str], dict[str, dict]]:
    """A DDUF archive's metadata entries (base64, every entry no weight) and each safetensors entry's
    tensors' shapes and dtype tags, from their headers: no weight payload read (`_ranged`,
    `sources.safetensors_header`)."""
    files: dict[str, str] = {}
    headers: dict[str, dict] = {}
    for name, (offset, length) in entries.items():
        if name.endswith(".safetensors"):
            header, _ = sources.safetensors_header(stream, offset=offset)
            headers[name] = {key: [list(_records.integers(_records.record(entry, key)["shape"], "shape")),
                                   _records.text(_records.record(entry, key)["dtype"], "dtype")]
                             for key, entry in header.items() if key != "__metadata__"}
        elif not name.endswith(_WEIGHT_SUFFIXES):
            files[name] = base64.b64encode(_ranged(stream, offset, length, f"{label} {name}")).decode("ascii")
    return files, headers


def metadata(
    source: sources.SourceReference, root: Path, filename: str
) -> tuple[Path, sources.StoredWeights]:
    """Extract metadata files and read safetensors headers from a DDUF file without unpacking or
    downloading weight payloads: its directory, its entries' local headers, its metadata entries and its
    safetensors headers are read by their byte ranges, a Hub archive's through a stream with no read-ahead
    (`HfFileSystem.open(cache_type="none")`), and a tensor read later (a sidecar's, a retained one's)
    by its own range."""
    selected = (Path(source.subfolder or "") / filename).as_posix()
    if os.path.isdir(source.name):
        local_path = source.file(root, filename)
        key = f"{_key(local_path)}-metadata"
        entries = _local_entries(local_path)
        with open(local_path, "rb") as stream:
            files, headers = _scanned(stream, entries, str(local_path))

        def local_tensor(file: str, tensor_name: str) -> np.ndarray:
            with open(local_path, "rb") as stream:
                return sources.read_safetensors_slice(stream, tensor_name, file, offset=entries[file][0])

        reader = local_tensor
    else:
        from huggingface_hub import HfFileSystem, constants, file_exists

        from dew.interop.dduf_stream import read_dduf_stream

        ref = f"{source.name}@{root.name}/{selected}"
        question = ("dduf metadata", source.name, root.name, selected)

        def fetch_remote() -> tuple[dict[str, list[int]], dict[str, str], dict[str, dict]]:
            if not file_exists(source.name, selected, revision=root.name):
                present = sorted(name for name in sources.repo_files(source.name, root)
                                 if name.endswith(Path(filename).suffix))
                raise FileNotFoundError(f"{source.name} at {root.name} has no {selected!r}; "
                                        f"its {Path(filename).suffix} files are {present}")
            with HfFileSystem().open(ref, "rb", cache_type="none") as stream:
                read = _contained(read_dduf_stream(stream), ref)
                return ({name: list(place) for name, place in read.items()},
                        *_scanned(stream, read, ref))

        if constants.HF_HUB_OFFLINE and recorded_hub_answer(question) is None:
            raise FileNotFoundError(f"{source.name} at {root.name}: {selected!r} metadata was never read "
                                    "online; run online once, or load with weights")
        recorded_entries, files, headers = hub_answer(question, fetch_remote).value
        entries = {str(name): (int(place[0]), int(place[1])) for name, place in recorded_entries.items()}
        digest = hashlib.sha256(f"{source.name}\0{root.name}\0{selected}".encode()).hexdigest()[:24]
        key = f"{digest}-metadata"

        def remote_tensor(file: str, tensor_name: str) -> np.ndarray:
            return sources.hub_tensor(ref, tensor_name, file,
                                      ("dduf sidecar", source.name, root.name, selected, file, tensor_name),
                                      offset=entries[file][0])

        reader = remote_tensor

    target = Path(dew_cache_dir()) / "dduf" / key
    if not (target / "model_index.json").is_file():
        with staged(target) as staging:
            for name, encoded in files.items():
                destination = staging / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(base64.b64decode(encoded))
    specs: dict[str, Mapping[str, jax.ShapeDtypeStruct]] = {
        file: {tname: stored_spec(tname, shape, tag, file) for tname, (shape, tag) in tensor_specs.items()}
        for file, tensor_specs in headers.items()
    }
    return target, sources.StoredWeights(source.name, target, entries=specs, tensor_reader=reader)
