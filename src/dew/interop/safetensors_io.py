"""safetensors for Flax parameter trees.

A parameter tree is nested dicts, a safetensors file is a flat table of named
tensors. The two meet at the '/'-joined path, the same naming the Hugging Face
checkpoints use, so a tree written here opens in anything that reads
safetensors and a file read back keeps its nesting.

Only the container is handled here. No leaf is renamed, transposed or cast.
The names on disk are the module names in the tree.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import jax
import ml_dtypes
import numpy as np

SEPARATOR = "/"
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"

# Use the format tag, not a NumPy conversion through safetensors: its NumPy
# FP8 conversion looks for np.float8_e4m3fn, which NumPy does not expose.
# These dtypes describe the mapped bytes; none widens or dequantizes them.
_STORED_DTYPES = {
    "BOOL": np.dtype(np.bool_),
    "U8": np.dtype(np.uint8),
    "I8": np.dtype(np.int8),
    "U16": np.dtype("<u2"),
    "I16": np.dtype("<i2"),
    "U32": np.dtype("<u4"),
    "I32": np.dtype("<i4"),
    "U64": np.dtype("<u8"),
    "I64": np.dtype("<i8"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
    "C64": np.dtype("<c8"),
    "BF16": np.dtype(ml_dtypes.bfloat16),
    "F8_E4M3": np.dtype(ml_dtypes.float8_e4m3fn),
    "F8_E5M2": np.dtype(ml_dtypes.float8_e5m2),
    "F8_E4M3FNUZ": np.dtype(ml_dtypes.float8_e4m3fnuz),
    "F8_E5M2FNUZ": np.dtype(ml_dtypes.float8_e5m2fnuz),
}


def _safetensors():
    """The reader and NumPy writer share one lazy optional-package boundary."""
    try:
        import safetensors
        from safetensors import numpy as safetensors_numpy
    except ImportError as error:
        raise ImportError(
            "dew.interop needs safetensors: pip install dew-ml[interop]"
        ) from error
    return safetensors, safetensors_numpy


def _publish(
    tensors: Callable[[], dict[str, np.ndarray]],
    path,
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Publish a complete inode without truncating arrays mapped by readers.

    The temporary is on the destination filesystem, so replacement is atomic.
    The source arrays are borrowed, including maps of the destination itself;
    serialization finishes before its directory entry changes. A failed write
    or replacement leaves the old file intact and removes the temporary.
    """
    _, backend = _safetensors()
    destination = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=".dew-weights-", suffix=".safetensors"
    )
    try:
        os.close(descriptor)
        backend.save_file(
            tensors(), temporary, metadata=None if metadata is None else dict(metadata)
        )
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _leaf_name(path) -> str:
    names = []
    for entry in path:
        if not isinstance(entry, jax.tree_util.DictKey):
            raise TypeError(
                f"safetensors names come from dict keys, this tree has a "
                f"{type(entry).__name__} in its path"
            )
        if not isinstance(entry.key, str) or SEPARATOR in entry.key:
            raise ValueError(
                f"parameter key {entry.key!r} cannot go into a {SEPARATOR!r}-joined name"
            )
        names.append(entry.key)
    return SEPARATOR.join(names)


def _host_array(leaf) -> np.ndarray:
    """Host copy of a leaf. safetensors writes raw bytes, so it must be dense."""
    array = np.asarray(leaf)
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


def _flatten(params) -> dict[str, np.ndarray]:
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    return {_leaf_name(path): _host_array(leaf) for path, leaf in leaves}


def _unflatten(tensors: Mapping[str, np.ndarray]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for name, tensor in tensors.items():
        *branches, leaf = name.split(SEPARATOR)
        node = tree
        for branch in branches:
            node = node.setdefault(branch, {})
        node[leaf] = tensor
    return tree


def save_params(params, path) -> None:
    """Write a parameter tree to a safetensors file, one tensor per leaf."""
    _publish(lambda: _flatten(params), path)


def load_params(path) -> dict[str, Any]:
    """Read a safetensors file back into a nested parameter dict.

    Leaves are read-only views of the file in their stored dtype, so nothing
    is placed on a device until the caller asks for it.
    """
    tensors, _ = read_file(path)
    return _unflatten(tensors)


def _tensor_offsets(path: str, header: object) -> dict[str, int]:
    """Each tensor's byte offset from the start of the data region.

    safe_open already validated the container, so a malformed entry here is a
    file whose header the official parser accepted and then disagreed with;
    it fails by name rather than falling back to a whole-file copy.
    """
    if not isinstance(header, dict):
        raise ValueError(f"{path} is not a safetensors file: its header is not a table")
    offsets: dict[str, int] = {}
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(key, str):
            raise ValueError(f"{path} holds a malformed tensor name {key!r}")
        name: str = key
        if not isinstance(entry, dict):
            raise ValueError(f"tensor {name!r} in {path} has a malformed header entry")
        span = entry.get("data_offsets")
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError(f"tensor {name!r} in {path} has malformed data_offsets")
        start, end = span
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end < start
        ):
            raise ValueError(f"tensor {name!r} in {path} has malformed data_offsets")
        offsets[name] = start
    return offsets


def read_file(path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """A file's flat tensor table, names as stored, and its header metadata.

    One read-only memory map backs every array, so the tensors stay file
    backed after the reader closes and arrive in their stored dtype -
    bfloat16 included. Anything that wants float32 asks for it.
    """
    filename = os.fspath(path)
    tensors: dict[str, np.ndarray] = {}
    package, _ = _safetensors()
    with package.safe_open(filename, "np") as reader:
        with open(filename, "rb") as stream:
            length = int.from_bytes(stream.read(8), "little")
            header = json.loads(stream.read(length))
        mapping = np.memmap(filename, mode="r", dtype=np.uint8)
        metadata = reader.metadata() or {}
        offsets = _tensor_offsets(filename, header)
        names = reader.keys()
        for name in names:
            view = reader.get_slice(name)
            shape = tuple(view.get_shape())
            tag = view.get_dtype()
            try:
                dtype = _STORED_DTYPES[tag]
            except KeyError as error:
                raise ValueError(
                    f"tensor {name!r} in {filename} uses unsupported stored dtype {tag!r}"
                ) from error
            offset = offsets.get(name)
            if offset is None:
                raise ValueError(f"tensor {name!r} in {filename} has no header entry")
            tensors[name] = np.ndarray(
                shape, dtype=dtype, buffer=mapping, offset=8 + length + offset
            )
    return tensors, metadata


def write_file(
    tensors: Mapping[str, np.ndarray], path, metadata: Mapping[str, str]
) -> None:
    """Write a flat tensor table under the names given, with header metadata."""
    _publish(lambda: dict(tensors), path, metadata)


def save_hf_layout(params, config: Mapping[str, object], directory) -> None:
    """Write model.safetensors and config.json into `directory`.

    That pair is what a Hugging Face style loader looks for. The config is
    written as given. Dew does not translate its own config vocabulary into
    anyone else's.
    """
    os.makedirs(directory, exist_ok=True)
    save_params(params, os.path.join(directory, WEIGHTS_FILE))
    with open(os.path.join(directory, CONFIG_FILE), "w") as handle:
        json.dump(config, handle, indent=2)
