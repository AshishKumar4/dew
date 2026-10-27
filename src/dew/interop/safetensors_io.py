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
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from pathlib import Path

import jax
import ml_dtypes
import numpy as np

from dew._records import JSON
from dew.files import replacing, write_atomically
from dew.interop.weights import ParamTree, insert

SEPARATOR = "/"
WEIGHTS_FILE = "model.safetensors"
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
            "dew.interop needs safetensors: pip install dewml[interop]"
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
    with replacing(path) as temporary:
        backend.save_file(
            {name: _host_array(array) for name, array in tensors().items()},
            os.fspath(temporary),
            metadata=None if metadata is None else dict(metadata),
        )
        if metadata is not None and len(metadata) > 1:
            _sort_metadata(os.fspath(temporary))


def _sort_metadata(path: str) -> None:
    """Write the header's metadata in key order, in place.

    safetensors keeps the metadata in a Rust HashMap, whose order differs from
    one write to the next, so the same table would be different bytes. The
    keys and values are the same text in another order, so the header keeps
    its length and the tensors their offsets. This works around
    safetensors/safetensors#584, which its PR #790 fixes; once a release
    includes that, safetensors writes the order itself and this goes.
    """
    with open(path, "r+b") as file:
        length = int.from_bytes(file.read(8), "little")
        raw = file.read(length)
        header = json.loads(raw)

        def serialized(table: dict) -> bytes:
            return json.dumps(table, separators=(",", ":"), ensure_ascii=False).encode()

        if serialized(header) != raw.rstrip(b" "):
            raise ValueError(f"{path}'s header is not one this writer reproduces, so its metadata "
                             "cannot be put in order")
        header["__metadata__"] = dict(sorted(header["__metadata__"].items()))
        file.seek(8)
        file.write(serialized(header).ljust(length))


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


def flatten(params) -> dict[str, np.ndarray]:
    """The leaves under their '/'-joined names; `_publish` brings each to the host."""
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    return {_leaf_name(path): leaf for path, leaf in leaves}


def unflatten(tensors: Mapping[str, np.ndarray]) -> ParamTree:
    tree: ParamTree = {}
    for name, tensor in tensors.items():
        insert(tree, tuple(name.split(SEPARATOR)), tensor, name)
    return tree


def save_params(params, path) -> None:
    """Write a parameter tree to a safetensors file, one tensor per leaf."""
    _publish(lambda: flatten(params), path)


def load_params(path) -> ParamTree:
    """Read a safetensors file back into a nested parameter dict.

    Leaves are read-only views of the file in their stored dtype, so nothing
    is placed on a device until the caller asks for it.
    """
    tensors, _ = read_file(path)
    return unflatten(tensors)


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


def folder_weights(folder: Path) -> tuple[str, ...]:
    """The names of the files in `folder` that hold its weights, by
    `weight_files`' rule; () when it holds none."""
    return weight_files(_listing(folder), "", _json_reader(folder))


def read_weights(folder) -> dict[str, np.ndarray]:
    """Read one checkpoint folder's weights, as `weight_files` selects them.

    A tensor stored in two of the selected shards raises a ValueError naming
    it and both files, where a merge would keep whichever came last.
    Raises FileNotFoundError when the folder holds no safetensors weights.
    """
    folder = Path(folder)
    selected = folder_weights(folder)
    if not selected:
        raise FileNotFoundError(
            f"no safetensors weights in {folder}: expected "
            + " or ".join(f"{stem}.safetensors (or its .index.json)" for stem in WEIGHT_STEMS))
    tensors: dict[str, np.ndarray] = {}
    owner: dict[str, str] = {}
    for shard in selected:
        values, _ = read_file(folder / shard)
        own(owner, values, shard, folder)
        tensors.update(values)
    return tensors


def own(owner: dict[str, str], names: Iterable[str], shard: str, where: Path) -> None:
    """Record in `owner` that `shard` holds each of `names`, refusing a
    tensor another of `where`'s shards already holds: a merge would keep
    whichever came last."""
    for name in names:
        if name in owner:
            raise ValueError(f"tensor {name!r} is stored in both {owner[name]} and {shard} under {where}")
        owner[name] = shard


def read_file(path, copy_on_write: bool = False) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Return a file's flat tensor table, names as stored, and its header metadata.

    One read-only memory map backs every array, so the tensors stay file
    backed after the reader closes and arrive in their stored dtype -
    bfloat16 and the FP8 formats included, and F4 as its `PACKED_F4` byte
    view. Anything that wants float32 asks for it. `copy_on_write` maps the
    file privately instead: the arrays are writable, and a page is copied
    only when written, never back to the file.
    """
    filename = os.fspath(path)
    package, _ = _safetensors()
    # One open file answers for the header, its validation and the bytes: `_publish`
    # replaces the pathname, so a reopened path can be another file. safe_open takes
    # only a path, so it validates the open file through its descriptor's name, which
    # every platform Dew installs on has (grain, a dependency, ships no Windows wheels).
    with open(filename, "rb") as stream, package.safe_open(f"/dev/fd/{stream.fileno()}", "np") as reader:
        length = int.from_bytes(stream.read(8), "little")
        metadata = reader.metadata() or {}
        stored = [(name, reader.get_slice(name)) for name in reader.offset_keys()]
        layout = [(name, tuple(view.get_shape()), view.get_dtype()) for name, view in stored]
        mapping = np.memmap(stream, mode="c" if copy_on_write else "r", dtype=np.uint8)
    tensors: dict[str, np.ndarray] = {}
    # safe_open has checked that the tensors tile the data in offset order, so
    # each one starts where the one before it ends.
    offset = 8 + length
    for name, shape, tag in layout:
        spec = stored_spec(name, shape, tag, filename)
        tensors[name] = np.ndarray(spec.shape, dtype=spec.dtype, buffer=mapping, offset=offset)
        offset += math.prod(spec.shape) * spec.dtype.itemsize
    return dict(sorted(tensors.items())), metadata


def stored_spec(name: str, shape: Sequence[int], tag: str, filename: str) -> jax.ShapeDtypeStruct:
    """The array a stored tensor reads as, by its header's shape and dtype tag:
    the tag's dtype, or for F4 its `PACKED_F4` byte view."""
    if tag == PACKED_F4:
        if not shape or shape[-1] % 2:
            raise ValueError(
                f"tensor {name!r} in {filename} is F4 with shape {tuple(shape)}; two "
                "elements share a byte, so its last axis is even")
        return jax.ShapeDtypeStruct((*shape[:-1], shape[-1] // 2), np.dtype(np.uint8))
    try:
        return jax.ShapeDtypeStruct(tuple(shape), _STORED_DTYPES[tag])
    except KeyError as error:
        raise ValueError(f"tensor {name!r} in {filename} uses unsupported stored dtype {tag!r}") from error


def header_specs(path) -> dict[str, jax.ShapeDtypeStruct]:
    """The arrays a safetensors file's tensors read as (`read_file`), from its header
    alone; no tensor's bytes are read."""
    filename = os.fspath(path)
    package, _ = _safetensors()
    with package.safe_open(filename, "np") as reader:
        names = reader.keys()
        views = [(name, reader.get_slice(name)) for name in names]
        return {name: stored_spec(name, view.get_shape(), view.get_dtype(), filename) for name, view in views}


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


class PayloadUnread(LookupError):
    """A tensor's values were asked of a load that reads none (`Pretrained.load(weights=False)`), which
    holds each stored tensor's shape and dtype alone: whatever asked computes on values it was not
    given, and reads the shapes instead (`tensor_specs`)."""


def unread(specs: Mapping[str, jax.ShapeDtypeStruct]) -> LazyTensors:
    """`specs`' tensors as a load without its weights holds them: their shapes and dtypes, and no value,
    reading one raising `PayloadUnread`."""
    def refuse(name: str) -> np.ndarray:
        raise PayloadUnread(f"{name}: a load without its weights reads no tensor's values, its shape and "
                            f"dtype alone ({specs[name].shape} {specs[name].dtype})")
    return LazyTensors(specs, refuse)


def tensor_specs(tensors: Mapping[str, np.ndarray]) -> Mapping[str, jax.ShapeDtypeStruct]:
    """Stored shape and dtype metadata without reading deferred tensor values."""
    return (tensors.specs if isinstance(tensors, LazyTensors)
            else {name: jax.ShapeDtypeStruct(array.shape, array.dtype) for name, array in tensors.items()})


def write_index(directory: Path, weight_map: Mapping[str, str], total_size: int | None = None,
                stem: str = "model") -> None:
    """Publish `<stem>.safetensors.index.json` naming each tensor's shard file."""
    metadata = {} if total_size is None else {"total_size": total_size}
    write_atomically(directory / f"{stem}.safetensors.index.json",
                     json.dumps({"metadata": metadata, "weight_map": dict(weight_map)}, indent=2))


def _replaced(stem: str) -> re.Pattern[str]:
    """The weight files an export under `stem` replaces: the single file, and shards named as
    `save_sharded` (`<stem>-<token>-00001-of-00002`) and transformers' and diffusers'
    `save_pretrained` (`<stem>-00001-of-00002`) name them."""
    escaped = re.escape(stem)
    return re.compile(rf"{escaped}\.safetensors|{escaped}-(?:[0-9a-f]{{8}}-)?\d{{5}}-of-\d{{5}}\.safetensors")


def save_sharded(
    tensors: Mapping[str, np.ndarray], directory, max_shard_size: int | str = MAX_SHARD_SIZE,
    stem: str = "model",
) -> None:
    """Write `tensors` as `<stem>.safetensors`, or as numbered shards and their
    index when they exceed `max_shard_size`, reading one shard's tensors at a
    time. `stem` is transformers' `model` or a diffusers model's
    `diffusion_pytorch_model` (`WEIGHT_STEMS`), as each library names its
    weights and reads them back.

    huggingface_hub's splitter plans the shards from sizes alone, so a
    `LazyTensors` table builds each shard's tensors as that shard is written
    and the host holds one shard of them at a time. A dense decoder export
    and a source-layout `Pretrained.export` are such tables. A quantized
    source's requantization, `PretrainedDecoder.from_model` through the
    GLM-5-next exporter, and DiffusionGemma's export adapter build their
    whole table first, and this writer then holds what it is given.

    An export to a directory that holds no weights yet, as `push_to_hub`
    stages one, names its shards as its library does,
    `<stem>-00001-of-00003.safetensors`, the only names Diffusers' Hub
    download keeps (`variant_compatible_siblings`,
    pipeline_loading_utils.py:239-289, 0.34.0).

    An export replaces the one `directory` held without a moment at which a
    loader reads a mix of the two. A replacement's shards carry a token of
    their own, so no file an earlier index named is overwritten or named
    again, and a reader still holding an earlier index finds its files gone
    rather than another export's under their names. A directory saved over
    in place so holds token names, which Diffusers' Hub download skips:
    publish it through `push_to_hub`.
    The index (or, unsharded, `<stem>.safetensors`) is published last, in one
    rename, and commits the export. Only then are the earlier weight files
    removed: the names `_replaced` matches that this export did not write,
    read off the directory's own listing. An old index's contents name
    nothing to delete, so one that names a path outside the directory, or a
    file of another name, removes nothing. transformers reads a local
    `model.safetensors` before an index (modeling_utils.py:594-605, 5.16.1),
    so a crash while an export changes between one file and shards can leave
    Dew and transformers each reading a whole export, but not the same one.
    Any other file, a precision variant such as `model.fp16.safetensors`
    among them, is left as it is.
    """
    from huggingface_hub import split_state_dict_into_shards_factory

    folder = Path(directory)
    specs = tensor_specs(tensors)
    split = split_state_dict_into_shards_factory(
        dict(specs), get_storage_size=lambda spec: math.prod(spec.shape) * np.dtype(spec.dtype).itemsize,
        filename_pattern=stem + "{suffix}.safetensors", max_shard_size=max_shard_size)
    old_index = folder / f"{stem}.safetensors.index.json"
    # An interrupted export may leave both an index and a single file.
    replaced = {name for name in _listing(folder) if _replaced(stem).fullmatch(name)}
    token = secrets.token_hex(4)
    replacing = split.is_sharded and bool(replaced)
    names = {filename: (filename.replace(f"{stem}-", f"{stem}-{token}-", 1) if replacing else filename)
             for filename in split.filename_to_tensors}
    for filename, members in split.filename_to_tensors.items():
        _publish(lambda members=members: {name: tensors[name] for name in members},
                 folder / names[filename], {"format": "pt"})
    if split.is_sharded:
        write_index(folder, {name: names[filename] for name, filename in split.tensor_to_filename.items()},
                    split.metadata["total_size"], stem)
    else:
        old_index.unlink(missing_ok=True)
    for name in replaced - set(names.values()):
        (folder / name).unlink(missing_ok=True)


def write_file(
    tensors: Mapping[str, np.ndarray], path, metadata: Mapping[str, str]
) -> None:
    """Write a flat tensor table under the names given, with header metadata."""
    _publish(lambda: dict(tensors), path, metadata)


def save_hf_layout(params, config: Mapping[str, object], directory,
                   max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
    """Write the weights and config.json into `directory`, the layout a Hugging Face style loader reads.

    `save_sharded` writes the weights, as one file or as shards of at most
    `max_shard_size`, and every file is replaced whole (`dew.files.replacing`), so a reader of the
    directory never reads one half written. `params` is a flat table of named tensors, or a tree
    whose '/'-joined paths name them. The config is written as given; Dew
    does not translate its own config vocabulary into anyone else's.
    """
    os.makedirs(directory, exist_ok=True)
    if not isinstance(params, LazyTensors):
        # Leaves stay where they are; `save_sharded` brings one shard at a time to the host.
        params = flatten(params)
    save_sharded(params, directory, max_shard_size)
    write_atomically(os.path.join(directory, CONFIG_FILE), json.dumps(config, indent=2))
