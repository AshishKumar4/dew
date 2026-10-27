"""Checkpoint precision, native tensor layouts and reversible source names."""
from __future__ import annotations

import dataclasses
from collections.abc import Callable, Container, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

from dew import _records
from dew.interop.unread import held
from dew.registry import dtype_name, resolve_dtype, storage_name

if TYPE_CHECKING:
    from flax import linen as nn

    from dew.interop.streaming import LazyTree, SourceLeaf, WeightLayout

type Tree[LeafT] = dict[str, LeafT | Tree[LeafT]]
type ParamTree = Tree[np.ndarray]
type TensorPath = Callable[[str], tuple[str, ...] | None]


def in_params(path: TensorPath) -> TensorPath:
    """`path`, whose leaves are parameters, rooted at `params`."""
    def rooted(name: str) -> tuple[str, ...] | None:
        leaf = path(name)
        return None if leaf is None else ("params", *leaf)
    return rooted


def check_shapes[PathT](loaded: Iterable[tuple[PathT, tuple[int, ...]]],
                        expected: Mapping[PathT, tuple[int, ...]]) -> None:
    """Check every supplied shape against its native slot, including multiple source aliases.
    Missing, unexpected and mismatched entries are reported together before parameters are read."""
    seen: set[PathT] = set()
    mismatched = []
    for path, shape in loaded:
        seen.add(path)
        if path in expected and shape != expected[path]:
            mismatched.append(f"{path} is {shape}, the module takes {expected[path]}")
    missing = sorted(set(expected) - seen, key=str)
    unexpected = sorted(seen - set(expected), key=str)
    if missing or unexpected or mismatched:
        raise ValueError(
            f"the checkpoint does not fit the model: missing {missing}, "
            f"unexpected {unexpected}, mismatched {sorted(mismatched)}")


def check_tree(variables: Mapping[str, object], module: nn.Module, *inputs) -> None:
    """Check every collection against the module's abstract initialization, without another
    copy of its parameters. Inputs may be arrays or jax.ShapeDtypeStruct values."""
    def shapes(tree) -> dict[str, tuple[int, ...]]:
        return {jax.tree_util.keystr(path, simple=True, separator="."): np.shape(leaf)
                for path, leaf in jax.tree_util.tree_leaves_with_path(tree)}

    expected = shapes(jax.eval_shape(module.init, jax.random.key(0), *inputs))
    check_shapes(shapes(dict(variables)).items(), expected)


AUTO = "auto"
"""The param_dtype that stores a checkpoint's parameters in its own dtype."""


def declared_dtype(config: Mapping[str, object], tensors: Mapping[str, np.ndarray]) -> str:
    """The compute dtype a checkpoint declares, which its packed FP8 or FP4
    payloads decode to under param_dtype 'auto': config.json's `dtype`
    (`torch_dtype` before transformers 5.0), else the dtype of the first
    floating tensor (transformers' `_get_dtype`, modeling_utils.py, 5.16.1).
    """
    stated = config.get("dtype", config.get("torch_dtype"))
    if stated is not None:
        storage = dtype_name(resolve_dtype(_records.text(stated, "dtype")))
        if storage is None:
            raise ValueError(f"dtype={stated!r} names no floating parameter storage")
        return storage
    for tensor in tensors.values():
        name = storage_name(tensor.dtype)
        if name is not None:
            return name
    raise ValueError("param_dtype 'auto' found neither a stated dtype nor a floating tensor a config "
                     "names (`dew.registry.DtypeName`) in the checkpoint")


def checkpoint_dtype(stored: np.dtype, param_dtype: str = "float32", *,
                     path: tuple[str, ...] = ()) -> np.dtype:
    """Floating payloads use parameter storage, or under `AUTO` keep the dtype
    each was stored in; integer and boolean state keeps its dtype."""
    if path and (path[0] == 'constants' or (path[0] == 'params' and path[-1] == 'head_bias')):
        param_dtype = 'float32'
    if param_dtype == AUTO:
        return stored
    dtype = resolve_dtype(param_dtype)
    if dtype is None or not jnp.issubdtype(dtype, jnp.floating):
        raise ValueError(f"param_dtype {param_dtype!r} must name floating parameter storage")
    return np.dtype(dtype) if jnp.issubdtype(stored, jnp.floating) else stored


def decoding_storage(model: nn.Module, variables: Mapping) -> Mapping:
    """`variables` stored as decoding reads them: each floating parameter
    stored wider than `model`'s compute dtype cast to it, but for those
    `checkpoint_dtype` keeps in fp32, as a checkpoint loaded at that storage
    holds them, and as HF's `torch_dtype` and vLLM's `dtype` store a model.
    Other collections, and a model that declares no compute dtype, keep
    theirs, and so do the layers a `SafetensorsBanks` streams, which read at
    the storage its own `param_dtype` states. A caller casts before its
    program, not inside it, so the program is the one a load at that storage
    runs."""
    compute = getattr(model, "dtype", None)
    if compute is None:
        return variables

    def stored(path: tuple[jax.tree_util.KeyEntry, ...], leaf):
        names = tuple(jax.tree_util.keystr((key,), simple=True) for key in path)
        dtype = np.dtype(leaf.dtype)
        if (names[0] != "params" or not jnp.issubdtype(dtype, jnp.floating)
                or dtype.itemsize <= jnp.dtype(compute).itemsize):
            return leaf
        target = checkpoint_dtype(dtype, dtype_name(compute), path=names)
        return leaf if target == dtype else leaf.astype(target)

    return jax.tree_util.tree_map_with_path(stored, variables)


def checkpoint_array(tensor, param_dtype: str = "float32") -> np.ndarray:
    """Cast a stored floating array directly, without an FP32 intermediate."""
    leaf = held(tensor)
    return leaf.astype(checkpoint_dtype(leaf.dtype, param_dtype), copy=False)


def indexed_path(tokens: Iterable[str], *, indexed: Container[str] | None = None) -> tuple[str, ...]:
    """A source tensor name's dotted tokens as Linen's path: each numeric token joins the token before it
    (`down_blocks.0` → `down_blocks_0`) where that token, as folded so far, is one of `indexed` (any token,
    where None; none, where empty). Unrestricted, consecutive digits all join it (`a.0.1` → `a_0_1`);
    with `indexed`, the second joins only where the folded token (`a_0`) is itself in it. A leading
    numeric token, or one after a token `indexed` leaves out, stays its own."""
    path: list[str] = []
    for token in tokens:
        if token.isdigit() and path and (indexed is None or path[-1] in indexed):
            path[-1] = f"{path[-1]}_{token}"
        else:
            path.append(token)
    return tuple(path)


def insert[LeafT](tree: Tree[LeafT], path: tuple[str, ...], leaf: LeafT, name: str) -> None:
    """Bind a leaf, refusing paths through leaves and unequal duplicate tensors.

    Lazy leaves cannot be compared without a read; source aliases must be
    checked before inserting them.
    """
    node = tree
    for key in path[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"{name} crosses the tensor already at {key!r}")
        node = child
    held = node.setdefault(path[-1], leaf)
    if held is not leaf and not (isinstance(held, np.ndarray) and isinstance(leaf, np.ndarray)
                                 and np.array_equal(held, leaf)):
        raise ValueError(f"{name} lands on {'/'.join(path)}, which another tensor already fills")


def source_alias(tensors: Mapping[str, np.ndarray], owners: dict[tuple[str, ...], str],
                 path: tuple[str, ...], name: str) -> None:
    """Compare source aliases before casting, since unequal values can round alike."""
    previous = owners.setdefault(path, name)
    if previous != name and not np.array_equal(tensors[previous], tensors[name]):
        raise ValueError(f"Two different source tensors {previous!r} and {name!r} map to {path}")


_TILE = 64
"""The side of the squares `swapped` copies: a 64x64 block of float32 is
16 KiB, so a block and its transpose stay in a core's cache."""


def swapped(values: np.ndarray) -> np.ndarray:
    """`values` with its last two axes swapped, C-ordered.

    numpy copies a whole transposed matrix with one strided pass, which
    misses the cache on every element: 0.39 GiB/s for a 10240x4096 float32
    kernel, where a contiguous copy runs at 9.7. Square tiles small enough
    for the cache copy at 3.5 GiB/s."""
    *lead, rows, columns = values.shape
    if values.size == 0:
        return np.empty((*lead, columns, rows), values.dtype)
    flat = np.ascontiguousarray(values).reshape(-1, rows, columns)
    out = np.empty((flat.shape[0], columns, rows), values.dtype)
    for row in range(0, rows, _TILE):
        for column in range(0, columns, _TILE):
            out[:, column:column + _TILE, row:row + _TILE] = (
                flat[:, row:row + _TILE, column:column + _TILE].swapaxes(-1, -2))
    return out.reshape(*lead, columns, rows)


def _copy(stored: np.ndarray, dtype: np.dtype, order: tuple[int, ...] | None) -> np.ndarray:
    """Cast in contiguous source order before copying the native layout, a
    swap of the last two axes (every linear kernel) in tiles (`swapped`)."""
    leaf = stored.astype(dtype, copy=False)
    if order is None:
        return leaf
    if order == (*range(leaf.ndim - 2), leaf.ndim - 1, leaf.ndim - 2):
        return swapped(leaf)
    return np.ascontiguousarray(leaf.transpose(order))


def _leaves[LeafT](tensors: Mapping[str, np.ndarray], path_of: TensorPath, param_dtype: str,
                   read: Callable[[np.ndarray, np.dtype, tuple[int, ...] | None], LeafT]
                   ) -> Iterator[tuple[str, tuple[str, ...], LeafT, tuple[int, ...] | None]]:
    """Choose native paths, layout and precision for an eager or mapped reader.

    Frozen constants stay FP32 independently of parameter precision.
    """
    for name, tensor in tensors.items():
        path = path_of(name)
        if path is None:
            continue
        stored = np.asarray(tensor)
        order, transpose = _orders(path, stored.ndim)
        dtype = checkpoint_dtype(stored.dtype, param_dtype, path=path)
        yield name, path, read(stored, dtype, order), transpose


def _orders(path: tuple[str, ...], rank: int) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    """The axis order a stored tensor of `rank` reads into its leaf at `path` by, and
    the order that writes the leaf back: a kernel's `[out, in, ...]` read as
    `[..., in, out]`; None for any other leaf."""
    if path[-1] != "kernel":
        return None, None
    order = (*range(2, rank), 1, 0)
    return order, tuple(int(axis) for axis in np.argsort(order))


def native_shape(path: tuple[str, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    """The native shape at `path` under the same source-axis permutation the parameter reader uses."""
    order, _ = _orders(path, len(shape))
    return shape if order is None else tuple(shape[axis] for axis in order)


def _stored_shape(path: tuple[str, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    """The shape of the source tensor a leaf of `shape` at `path` is read from (`_orders`' inverse):
    a kernel's axes as the source stores them, any other leaf's as they are."""
    order, _ = _orders(path, len(shape))
    return tuple(shape) if order is None else tuple(shape[order.index(axis)] for axis in range(len(shape)))


def native_layouts(variables: Mapping[str, Mapping[str, object]], name_of: Callable[[tuple[str, ...]], str],
                   path_of: TensorPath) -> tuple[WeightLayout, ...]:
    """The layouts that write every leaf of `variables` (its collections) back as the source tensor
    `name_of` names it from its path, for a tree built natively rather than read; `path_of`, the
    reader's own map, places each name, and a name it does not place back at its leaf is refused."""
    shapes = {}
    for collection, tree in variables.items():
        for path, leaf in flatten_dict(dict(tree)).items():
            name = name_of((collection, *path))
            if path_of(name) != (collection, *path) or not isinstance(leaf, (np.ndarray, jax.Array)):
                raise ValueError(f"{name} reads into {path_of(name)}, not an array at ({collection}, "
                                 f"{', '.join(path)})")
            shapes[name] = _stored_shape(path, leaf.shape)
    return source_layouts("", shapes, path_of, ())


def source_layouts(component: str, shapes: Mapping[str, tuple[int, ...]], path_of: TensorPath,
                   prefix: tuple[str, ...]) -> tuple[WeightLayout, ...]:
    """`record_layouts`' layouts, from the stored tensors' shapes alone: each
    tensor's leaf under `prefix` and the transpose that writes it back, named under
    `component` (as stored where it is empty), for a tree bound from saved variables
    rather than read (`components.bind_component`)."""
    return tuple(_layout(component, name, shape, path, prefix)
                 for name, shape in shapes.items() if (path := path_of(name)) is not None)


def _layout(component: str, name: str, shape: tuple[int, ...], path: tuple[str, ...],
            prefix: tuple[str, ...]) -> WeightLayout:
    """The layout that writes the leaf at `prefix + path` back as the `shape`d tensor `name`."""
    from dew.interop.streaming import WeightLayout

    return WeightLayout(f"{component}/{name}" if component else name, ((*prefix, *path),), tuple(shape),
                        _orders(path, len(shape))[1])


def translate_parameters(tensors: Mapping[str, np.ndarray], path_of: TensorPath,
                         param_dtype: str = "float32") -> ParamTree:
    """Read host-built parameters without creating deferred placement recipes.

    Repeated paths compare at their storage precision.
    """
    return placed(tensors, path_of, param_dtype)[0]


def placed(tensors: Mapping[str, np.ndarray], path_of: TensorPath, param_dtype: str = "float32"
           ) -> tuple[ParamTree, tuple[WeightLayout, ...]]:
    """`translate_parameters`' tree, with the layout that writes each tensor it placed back
    under its own name from its path in that tree; a tensor `path_of` leaves out has none.
    It is `record_layouts`' binding pass (`_bound`) with each tensor read whole, under no
    component."""
    return _bound("", tensors, path_of, (), param_dtype, _copy)


def _bound[LeafT](component: str, tensors: Mapping[str, np.ndarray], path_of: TensorPath,
                  prefix: tuple[str, ...], param_dtype: str,
                  read: Callable[[np.ndarray, np.dtype, tuple[int, ...] | None], LeafT]
                  ) -> tuple[dict, tuple[WeightLayout, ...]]:
    """The one binding pass, eager (`placed`) or mapped (`record_layouts`): each tensor
    `path_of` places, read by `read` into its leaf, source aliases sharing the leaf of the
    first (`_aliased`), and the layout that writes each back from the path it was read at,
    under `component` and `prefix` (`_layout`); `path_of` is asked once a tensor."""
    parameters: dict = {}
    layouts = []
    owners, locate = _aliased(tensors, path_of)
    for name, path, leaf, _ in _leaves(tensors, locate, param_dtype, read):
        if owners[path] == name:
            insert(parameters, path, leaf, name)
        layouts.append(_layout(component, name, np.shape(tensors[name]), path, prefix))
    return parameters, tuple(layouts)


def _aliased(tensors: Mapping[str, np.ndarray], path_of: TensorPath
             ) -> tuple[dict[tuple[str, ...], str], TensorPath]:
    """The first tensor each path holds, filled as the returned `path_of` is asked, which
    compares each later alias with it before any cast (`source_alias`)."""
    owners: dict[tuple[str, ...], str] = {}

    def locate(name: str) -> tuple[str, ...] | None:
        path = path_of(name)
        if path is not None:
            source_alias(tensors, owners, path, name)
        return path

    return owners, locate


def nested(layouts: tuple[WeightLayout, ...], depth: int, *keys: str) -> tuple[WeightLayout, ...]:
    """`layouts` read from a tree that holds theirs under `keys`, after the first `depth` keys
    of each path: a component's `params` in its variables, or a component inside its model's
    collections."""
    return tuple(dataclasses.replace(layout, paths=tuple((*path[:depth], *keys, *path[depth:])
                                                          for path in layout.paths))
                 for layout in layouts)


def record_layouts(component: str, tensors: Mapping[str, np.ndarray], path_of: TensorPath,
                   prefix: tuple[str, ...], *, param_dtype: str = "float32", lazy: bool = False
                   ) -> tuple[LazyTree, tuple[WeightLayout, ...]]:
    """Bind native leaves and the layouts that rebuild every source tensor.

    Equal source aliases share a leaf and keep separate export names. With
    `lazy`, linear kernels and untransposed tensors stay `SourceLeaf` recipes;
    each placement reads and casts only its shard. Scoring state can request
    FP32 separately from parameters.
    """
    from dew.interop.streaming import SourceLeaf, materialize

    def mapped(stored: np.ndarray, dtype: np.dtype, order: tuple[int, ...] | None) -> np.ndarray | SourceLeaf:
        return (_copy(stored, dtype, order) if order is not None and stored.ndim != 2
                else SourceLeaf((stored,), dtype, transposed=order is not None))

    parameters, layouts = _bound(component, tensors, path_of, prefix, param_dtype, mapped)
    return (parameters if lazy else materialize(parameters)), layouts
