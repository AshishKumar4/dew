"""DDUF files: a diffusers pipeline directory packed into one uncompressed ZIP.

A `.dduf` file (`huggingface_hub.export_folder_as_dduf`) holds the files of a
diffusers directory: `model_index.json`, and each component's config and weights
under its own folder. `_validated_members` checks the public DDUF archive contract
(stored entries, UNIX separators, one directory level, allowed suffixes, and
`model_index.json` component folders with config files — adapted from
`huggingface/huggingface_hub` v1.30.0 `serialization/_dduf.py`, Apache-2.0) over
stdlib `zipfile.ZipFile` so both `unpacked` (full extraction) and `metadata`
(header-only local or HTTP byte-range reads with `cache_type="none"`) share one
validator without private `huggingface_hub` imports.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import os
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import jax
import numpy as np

from dew import _records
from dew._builds import dew_cache_dir
from dew._cache import hub_answer, recorded_hub_answer
from dew._files import staged, stat_identity
from dew.interop import sources
from dew.interop.safetensors_io import stored_spec

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_ALLOWED_SUFFIXES = frozenset({".json", ".model", ".safetensors", ".txt"})
_REQUIRED_CONFIGS = (
    "config.json",
    "preprocessor_config.json",
    "scheduler_config.json",
    "tokenizer_config.json",
)
_CHUNK = 1 << 24
_SIDECAR_CACHE_LIMIT = 65536


def _key(path: Path) -> str:
    """The cache key of one DDUF file, its `stat_identity`; deleting
    `dew_cache_dir()/dduf` clears a stale unpack."""
    return hashlib.sha256("\0".join(map(str, stat_identity(path))).encode()).hexdigest()[:24]


def _validated_members(archive: zipfile.ZipFile, label: str) -> list[zipfile.ZipInfo]:
    """Validate DDUF archive structure and entry names over public `zipfile.ZipFile`."""
    from huggingface_hub.errors import DDUFCorruptedFileError

    members = archive.infolist()
    names: list[str] = []
    for member in members:
        name = member.filename
        if member.compress_type != zipfile.ZIP_STORED:
            raise DDUFCorruptedFileError("Data must not be compressed in DDUF file.")
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts:
            raise ValueError(f"{label} names an entry {name!r} outside its own directory")
        suffix = f".{name.rsplit('.', 1)[-1]}" if "." in name else ""
        if suffix not in _ALLOWED_SUFFIXES:
            raise DDUFCorruptedFileError(f"Invalid entry name in DDUF file: {name}")
        if "\\" in name or name.strip("/").count("/") > 1:
            raise DDUFCorruptedFileError(f"Invalid entry name in DDUF file: {name}")
        names.append(name)
    if "model_index.json" not in names:
        raise DDUFCorruptedFileError("Missing required 'model_index.json' entry in DDUF file.")
    index = json.loads(archive.read("model_index.json"))
    if not isinstance(index, dict):
        raise DDUFCorruptedFileError("Invalid 'model_index.json' content: expected a JSON object.")
    name_set = set(names)
    folders = {entry.split("/", 1)[0] for entry in names if "/" in entry}
    for folder in folders:
        if folder not in index:
            raise DDUFCorruptedFileError(f"Missing required entry '{folder}' in 'model_index.json'.")
        if not any(f"{folder}/{cfg}" in name_set for cfg in _REQUIRED_CONFIGS):
            raise DDUFCorruptedFileError(
                f"Missing required config file in folder '{folder}': expected one of {_REQUIRED_CONFIGS}."
            )
    return members


def unpacked(path: str | os.PathLike[str]) -> Path:
    """The diffusers directory `path` packs, written once into Dew's cache.

    The directory is published with one rename once every entry is written,
    so an interrupted unpack leaves no partial directory to be read."""
    source = Path(path)
    target = Path(dew_cache_dir()) / "dduf" / _key(source)
    if (target / "model_index.json").is_file():
        return target
    with staged(target) as staging, zipfile.ZipFile(source, "r") as archive:
        for member in _validated_members(archive, str(source)):
            destination = staging / member.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            remaining = member.file_size
            with archive.open(member, "r") as entry_stream, open(destination, "wb") as out:
                while remaining:
                    chunk = entry_stream.read(min(_CHUNK, remaining))
                    if not chunk:
                        raise ValueError(f"{source} ends inside its entry {member.filename}")
                    out.write(chunk)
                    remaining -= len(chunk)
    return target


def _scan_dduf_archive(
    archive: zipfile.ZipFile, label: str
) -> tuple[dict[str, str], dict[str, dict[str, tuple[list[int], str]]]]:
    """Read metadata entries and safetensors entry headers from a validated `ZipFile`."""
    files: dict[str, str] = {}
    headers: dict[str, dict[str, tuple[list[int], str]]] = {}
    for member in _validated_members(archive, label):
        name = member.filename
        if name.endswith(".safetensors"):
            with archive.open(member, "r") as entry_stream:
                header_len = int.from_bytes(entry_stream.read(8), "little")
                parsed = json.loads(entry_stream.read(header_len))
            headers[name] = {
                key: (
                    list(_records.integers(_records.record(entry, key)["shape"], "shape")),
                    _records.text(_records.record(entry, key)["dtype"], "dtype"),
                )
                for key, entry in parsed.items()
                if key != "__metadata__"
            }
        elif not name.endswith(_WEIGHT_SUFFIXES):
            payload = archive.read(member)
            if len(payload) != member.file_size:
                raise ValueError(f"{label} ends inside its entry {name}")
            files[name] = base64.b64encode(payload).decode("ascii")
    return files, headers


def _read_remote_dduf_tensor(
    repo: str, revision: str, archive_name: str, file: str, tensor_name: str
) -> np.ndarray:
    from huggingface_hub import HfFileSystem

    ref = f"{repo}@{revision}/{archive_name}"
    with (
        HfFileSystem().open(ref, "rb", cache_type="none") as stream,
        zipfile.ZipFile(stream, "r") as archive,
        archive.open(file, "r") as entry_stream,
    ):
        return sources.read_safetensors_slice(entry_stream, tensor_name, file)


def _cached_remote_dduf_sidecar(
    repo: str, revision: str, archive_name: str, file: str, tensor_name: str
) -> tuple[list[int], str, str]:
    from huggingface_hub import HfFileSystem

    ref = f"{repo}@{revision}/{archive_name}"
    with (
        HfFileSystem().open(ref, "rb", cache_type="none") as stream,
        zipfile.ZipFile(stream, "r") as archive,
        archive.open(file, "r") as entry_stream,
    ):
        header_len = int.from_bytes(entry_stream.read(8), "little")
        header = json.loads(entry_stream.read(header_len))
        entry = _records.record(header[tensor_name], tensor_name)
        bounds = _records.integers(entry["data_offsets"], "data_offsets")
        entry_stream.seek(8 + header_len + bounds[0])
        raw = entry_stream.read(bounds[1] - bounds[0])
    return list(_records.integers(entry["shape"], "shape")), _records.text(entry["dtype"], "dtype"), raw.hex()


def metadata(
    source: sources.SourceReference, root: Path, filename: str
) -> tuple[Path, sources.StoredWeights]:
    """Extract metadata files and read safetensors headers from a DDUF file without unpacking or
    downloading weight payloads."""
    selected = (Path(source.subfolder or "") / filename).as_posix()
    if os.path.isdir(source.name):
        local_path = source.file(root, filename)
        key = f"{_key(local_path)}-metadata"
        with zipfile.ZipFile(local_path, "r") as archive:
            files, headers = _scan_dduf_archive(archive, str(local_path))

        def local_tensor(file: str, tensor_name: str) -> np.ndarray:
            with zipfile.ZipFile(local_path, "r") as archive, archive.open(file, "r") as entry_stream:
                return sources.read_safetensors_slice(entry_stream, tensor_name, file)

        reader = local_tensor
    else:
        from huggingface_hub import HfFileSystem, constants, file_exists

        question = ("dduf metadata", source.name, root.name, selected)
        if constants.HF_HUB_OFFLINE:
            if recorded_hub_answer(question) is None:
                raise FileNotFoundError(
                    f"{source.name} at {root.name}: {selected!r} metadata was never read online; "
                    "run online once, or load with weights"
                )
        elif not file_exists(source.name, selected, revision=root.name):
            present = sorted(
                name for name in sources.repo_files(source.name, root) if name.endswith(Path(filename).suffix)
            )
            raise FileNotFoundError(
                f"{source.name} at {root.name} has no {selected!r}; "
                f"its {Path(filename).suffix} files are {present}"
            )
        digest = hashlib.sha256(f"{source.name}\0{root.name}\0{selected}".encode()).hexdigest()[:24]
        key = f"{digest}-metadata"

        def fetch_remote() -> tuple[dict[str, str], dict[str, dict[str, list[_records.JSON]]]]:
            ref = f"{source.name}@{root.name}/{selected}"
            with (
                HfFileSystem().open(ref, "rb", cache_type="none") as stream,
                zipfile.ZipFile(stream, "r") as archive,
            ):
                raw_files, raw_headers = _scan_dduf_archive(archive, f"{source.name}/{selected}")
            packed_headers: dict[str, dict[str, list[_records.JSON]]] = {
                name: {
                    k: [[int(dim) for dim in shape], str(tag)]
                    for k, (shape, tag) in specs.items()
                }
                for name, specs in raw_headers.items()
            }
            return raw_files, packed_headers

        recorded_files, recorded_headers = hub_answer(question, fetch_remote).value
        files = {str(k): str(v) for k, v in _records.record(recorded_files, "dduf files").items()}
        headers = {
            entry_name: {
                tname: (
                    list(_records.integers(tspec[0], "shape")),
                    _records.text(tspec[1], "dtype"),
                )
                for tname, tspec in _records.record(tensor_map, "dduf tensor map").items()
                if isinstance(tspec, list) and len(tspec) == 2
            }
            for entry_name, tensor_map in _records.record(recorded_headers, "dduf headers").items()
        }

        def remote_tensor(file: str, tensor_name: str) -> np.ndarray:
            spec = entries[file][tensor_name]
            nbytes = int(np.prod(spec.shape)) * np.dtype(spec.dtype).itemsize
            if nbytes <= _SIDECAR_CACHE_LIMIT:
                sidecar_q = ("dduf sidecar", source.name, root.name, selected, file, tensor_name)
                if constants.HF_HUB_OFFLINE and recorded_hub_answer(sidecar_q) is None:
                    raise FileNotFoundError(
                        f"{source.name} at {root.name}: {file}:{tensor_name} was never read online"
                    )
                shape, tag, hex_bytes = hub_answer(
                    sidecar_q,
                    functools.partial(
                        _cached_remote_dduf_sidecar, source.name, root.name, selected, file, tensor_name
                    ),
                ).value
                cached_spec = stored_spec(tensor_name, shape, tag, file)
                return (
                    np.frombuffer(bytes.fromhex(hex_bytes), dtype=cached_spec.dtype)
                    .reshape(cached_spec.shape)
                    .copy()
                )
            return _read_remote_dduf_tensor(source.name, root.name, selected, file, tensor_name)

        reader = remote_tensor

    target = Path(dew_cache_dir()) / "dduf" / key
    if not (target / "model_index.json").is_file():
        with staged(target) as staging:
            for name, encoded in files.items():
                destination = staging / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(base64.b64decode(encoded))
    entries: dict[str, Mapping[str, jax.ShapeDtypeStruct]] = {
        file: {
            tname: stored_spec(tname, shape, tag, file)
            for tname, (shape, tag) in tensor_specs.items()
        }
        for file, tensor_specs in headers.items()
    }
    return target, sources.StoredWeights(source.name, target, entries=entries, tensor_reader=reader)
