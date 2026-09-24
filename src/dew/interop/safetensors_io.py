"""Read and write Flax parameter trees as safetensors files.

A parameter tree is nested dicts, a safetensors file is a flat table of named
tensors. The two meet at the '/'-joined path, the same naming the Hugging Face
checkpoints use, so a tree written here opens in anything that reads
safetensors and a file read back keeps its nesting.

Only the container is handled here. No leaf is renamed, transposed or cast.
The names on disk are the module names in the tree.
"""

import json
import math
import os
import re
import secrets
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Callable, Collection, Mapping

import jax
import ml_dtypes
import numpy as np

from dew.nn.text_encoders import ParamTree, insert
from dew.records import JSON

SEPARATOR = "/"
WEIGHTS_FILE = "model.safetensors"
INDEX_FILE = "model.safetensors.index.json"
CONFIG_FILE = "config.json"
MAX_SHARD_SIZE = "5GB"
"""The shard size an export writes at most, as huggingface_hub's splitter
reads it; a tensor larger than that takes a shard of its own."""

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
    # The block-scale exponents of MX formats and DeepSeek-V4's `.scale`
    # tensors: one unsigned power-of-two exponent per byte.
    "F8_E8M0": np.dtype(ml_dtypes.float8_e8m0fnu),
}

PACKED_F4 = "F4"
"""safetensors' 4-bit float (e2m1), two elements per byte.

Its header states the element shape, and PyTorch writes a
`float4_e2m1fn_x2` tensor of shape [..., n] as F4 [..., 2n] over n bytes.
NumPy has no sub-byte dtype, so `read_file` returns such a tensor as that
byte view: uint8 with the last axis halved. Which nibble holds which element
is the producing codec's convention; safetensors fixes only the byte count.
"""


def _safetensors():
    """Import safetensors and return its reader and NumPy writer modules.

    The import is lazy and in one place, so the reader and the writer raise the
    same install message when the optional package is missing.
    """
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
    serialization finishes before its directory entry changes. Each array is
    made C-contiguous first: safetensors 0.8.0 writes an array's memory as if
    it were C-ordered, so a transposed view, or the column-major host view a
    TPU array can come back as, would be written transposed. A failed write
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
            {name: _host_array(array) for name, array in tensors().items()}, temporary, metadata=None if metadata is None else dict(metadata)
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
    """`leaf` on the host, C-contiguous: the one step every safetensors write takes."""
    array = np.asarray(leaf)
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


def _flatten(params) -> dict[str, np.ndarray]:
    """The leaves under their '/'-joined names; `_publish` brings each to the host."""
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    return {_leaf_name(path): leaf for path, leaf in leaves}


def _unflatten(tensors: Mapping[str, np.ndarray]) -> ParamTree:
    tree: ParamTree = {}
    for name, tensor in tensors.items():
        insert(tree, tuple(name.split(SEPARATOR)), tensor, name)
    return tree


def save_params(params, path) -> None:
    """Write a parameter tree to a safetensors file, one tensor per leaf."""
    _publish(lambda: _flatten(params), path)


def load_params(path) -> ParamTree:
    """Read a safetensors file back into a nested parameter dict.

    Leaves are read-only views of the file in their stored dtype, so nothing
    is placed on a device until the caller asks for it.
    """
    tensors, _ = read_file(path)
    return _unflatten(tensors)


def _tensor_offsets(path: str, header: object) -> dict[str, int]:
    """Return each tensor's byte offset from the start of the data region.

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


WEIGHT_STEMS = ("diffusion_pytorch_model", "model")
"""The weights file stems of a diffusers model and a transformers model."""


def weight_files(files: Collection[str], folder: str,
                 read_json: Callable[[str], JSON], *,
                 stems: tuple[str, ...] = WEIGHT_STEMS, suffix: str = ".safetensors") -> tuple[str, ...]:
    """Return the safetensors files that hold one checkpoint folder's weights.

    `files` are the repo-relative names a directory or a Hub listing holds,
    `folder` the component ('' for the root) and `read_json` reads one of
    those names. A `<stem>.safetensors.index.json` names its shards in
    `weight_map`, and those are the weights, as vLLM reads them
    (`filter_duplicate_safetensors_files`, weight_utils.py @d110c2f) and
    transformers does; otherwise `<stem>.safetensors` is. Anything else
    beside them is another format of the same weights and is never read:
    Mistral's `consolidated.safetensors`, a root single-file checkpoint, or
    a precision variant. A variant (`model.fp16.safetensors`,
    `model.safetensors.index.fp16.json`) is the name diffusers'
    `variant_compatible_siblings` (pipelines/pipeline_loading_utils.py:205,
    diffusers 0.34.0) leaves out of a load with `variant=None`; that module
    imports torch, so its rule is restated here rather than called.
    `stems` and `suffix` select another format by the same rule, as
    transformers' `pytorch_model.bin` and its index.
    Returns () when the folder holds no such weights.
    """
    prefix = f"{folder}/" if folder else ""
    for stem in stems:
        index = f"{prefix}{stem}{suffix}.index.json"
        if index in files:
            record = read_json(index)
            weight_map = record.get("weight_map") if isinstance(record, dict) else None
            shards = ([shard for shard in weight_map.values() if isinstance(shard, str)]
                      if isinstance(weight_map, dict) else [])
            if not isinstance(weight_map, dict) or len(shards) != len(weight_map):
                raise ValueError(f"{index} has no weight_map of tensor names to shard files")
            return tuple(sorted({prefix + shard for shard in shards}))
        single = f"{prefix}{stem}{suffix}"
        if single in files:
            return (single,)
    return ()


def _listing(folder: Path) -> set[str]:
    return {entry.name for entry in folder.iterdir() if entry.is_file()} if folder.is_dir() else set()


def _json_reader(folder: Path) -> Callable[[str], JSON]:
    return lambda name: json.loads((folder / name).read_text())


def read_weights(folder) -> dict[str, np.ndarray]:
    """Read one checkpoint folder's weights, as `weight_files` selects them.

    A tensor stored in two of the selected shards raises a ValueError naming
    it and both files, where a merge would keep whichever came last.
    Raises FileNotFoundError when the folder holds no safetensors weights.
    """
    folder = Path(folder)
    selected = weight_files(_listing(folder), "", _json_reader(folder))
    if not selected:
        raise FileNotFoundError(
            f"no safetensors weights in {folder}: expected "
            + " or ".join(f"{stem}.safetensors (or its .index.json)" for stem in WEIGHT_STEMS))
    tensors: dict[str, np.ndarray] = {}
    owner: dict[str, str] = {}
    for shard in selected:
        values, _ = read_file(folder / shard)
        for name, value in values.items():
            if name in owner:
                raise ValueError(f"tensor {name!r} is stored in both {owner[name]} and {shard} under {folder}")
            owner[name] = shard
            tensors[name] = value
    return tensors


def read_file(path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Return a file's flat tensor table, names as stored, and its header metadata.

    One read-only memory map backs every array, so the tensors stay file
    backed after the reader closes and arrive in their stored dtype -
    bfloat16 and the FP8 formats included, and F4 as its `PACKED_F4` byte
    view. Anything that wants float32 asks for it.
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
            if tag == PACKED_F4:
                if not shape or shape[-1] % 2:
                    raise ValueError(
                        f"tensor {name!r} in {filename} is F4 with shape {shape}; two "
                        "elements share a byte, so its last axis is even")
                shape, dtype = (*shape[:-1], shape[-1] // 2), np.dtype(np.uint8)
            else:
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


class LazyTensors(Mapping[str, np.ndarray]):
    """Named host tensors, each built only when it is read.

    `specs` gives every tensor's shape and dtype up front, which is all a
    shard plan needs; `build(name)` makes the tensor. A writer that reads one
    shard's tensors at a time holds one shard on the host, not the model.
    """

    def __init__(self, specs: Mapping[str, jax.ShapeDtypeStruct], build: Callable[[str], np.ndarray]):
        self.specs = specs
        self._build = build

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self.specs:
            raise KeyError(name)
        return self._build(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)


def write_index(directory: Path, weight_map: Mapping[str, str], total_size: int | None = None) -> None:
    """Publish `model.safetensors.index.json` naming each tensor's shard file."""
    index = directory / INDEX_FILE
    temporary = index.with_name(f".{index.name}.tmp")
    metadata = {} if total_size is None else {"total_size": total_size}
    temporary.write_text(json.dumps({"metadata": metadata, "weight_map": dict(weight_map)}, indent=2))
    os.replace(temporary, index)


_OWN_SHARD = re.compile(r"model-[0-9a-f]{8}-\d{5}-of-\d{5}\.safetensors")
"""The shard names `save_sharded` writes."""


def save_sharded(tensors: Mapping[str, np.ndarray], directory, max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
    """Write `tensors` as `model.safetensors`, or as numbered shards and their
    index when they exceed `max_shard_size`, reading one shard's tensors at a
    time.

    huggingface_hub's splitter plans the shards from sizes alone, so a
    `LazyTensors` table builds each shard's tensors as that shard is written
    and the host holds one shard of them at a time. A dense decoder export
    and a source-layout `Pretrained.export` are such tables. A quantized
    source's requantization, `save_pretrained_decoder` through the Gemma 4
    and GLM-5-next exporters, and DiffusionGemma's export adapter build their
    whole table first, and this writer then holds what it is given.

    An export replaces the one `directory` held without a moment at which a
    loader reads a mix of the two. Each shard takes a name no earlier export
    can hold, so no file an old index names is overwritten; the index (or,
    unsharded, `model.safetensors`) is published last, in one rename, and
    commits the export. Only then are the files the old export named and
    this one does not removed, with shards an interrupted export of this
    writer left unindexed. transformers reads a local `model.safetensors`
    before an index (modeling_utils.py:594-605, 5.16.1), so a crash while an
    export changes between one file and shards can leave Dew and transformers
    each reading a whole export, but not the same one. Any other file, a precision variant such as
    `model.fp16.safetensors` among them, is left as it is.
    """
    from huggingface_hub import split_state_dict_into_shards_factory

    folder = Path(directory)
    specs = (tensors.specs if isinstance(tensors, LazyTensors)
             else {name: jax.ShapeDtypeStruct(array.shape, array.dtype) for name, array in tensors.items()})
    split = split_state_dict_into_shards_factory(
        dict(specs), get_storage_size=lambda spec: math.prod(spec.shape) * np.dtype(spec.dtype).itemsize,
        filename_pattern="model{suffix}.safetensors", max_shard_size=max_shard_size)
    old_index = folder / INDEX_FILE
    listing = _listing(folder)
    # An interrupted export may leave both an index and a single file.
    previous = set(weight_files(listing, "", _json_reader(folder))) | ({WEIGHTS_FILE} & listing)
    token = secrets.token_hex(4)
    names = {filename: (filename.replace("model-", f"model-{token}-", 1) if split.is_sharded else filename)
             for filename in split.filename_to_tensors}
    for filename, members in split.filename_to_tensors.items():
        _publish(lambda members=members: {name: tensors[name] for name in members},
                 folder / names[filename], {"format": "pt"})
    if split.is_sharded:
        write_index(folder, {name: names[filename] for name, filename in split.tensor_to_filename.items()},
                    split.metadata["total_size"])
        stale = previous - set(names.values())
    else:
        old_index.unlink(missing_ok=True)
        stale = previous - {WEIGHTS_FILE}
    # Shards an interrupted export published and never indexed carry this
    # writer's own token pattern and are no one else's.
    stale |= {name for name in listing if _OWN_SHARD.fullmatch(name)} - set(names.values())
    for name in stale:
        (folder / name).unlink(missing_ok=True)


def write_file(
    tensors: Mapping[str, np.ndarray], path, metadata: Mapping[str, str]
) -> None:
    """Write a flat tensor table under the names given, with header metadata."""
    _publish(lambda: dict(tensors), path, metadata)


def save_hf_layout(params, config: Mapping[str, object], directory,
                   max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
    """Write the weights (`save_sharded`) and config.json into `directory`.

    That is what a Hugging Face style loader looks for. `params` is a flat
    table of named tensors, or a tree whose '/'-joined paths name them. The
    config is written as given. Dew does not translate its own config
    vocabulary into anyone else's.
    """
    os.makedirs(directory, exist_ok=True)
    if not isinstance(params, LazyTensors):
        # Leaves stay where they are; `save_sharded` brings one shard at a time to the host.
        leaves, _ = jax.tree_util.tree_flatten_with_path(params)
        params = {_leaf_name(path): leaf for path, leaf in leaves}
    save_sharded(params, directory, max_shard_size)
    with open(os.path.join(directory, CONFIG_FILE), "w") as handle:
        json.dump(config, handle, indent=2)
