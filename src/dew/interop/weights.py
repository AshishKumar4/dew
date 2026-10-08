"""Checkpoint precision, native tensor layouts and reversible source names."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING

import jax.numpy as jnp
import ml_dtypes
import numpy as np

from dew import records
from dew.registry import dtype_name, resolve_dtype

if TYPE_CHECKING:
    from dew.interop.streaming import LazyTree, SourceLeaf, WeightLayout

type Tree[LeafT] = dict[str, LeafT | Tree[LeafT]]
type ParamTree = Tree[np.ndarray]
type TensorPath = Callable[[str], tuple[str, ...] | None]


AUTO = "auto"
"""The param_dtype that stores a checkpoint's parameters in its own dtype."""


def auto_storage_dtype(config: Mapping[str, object], tensors: Mapping[str, np.ndarray]) -> str:
    """Return the storage dtype a checkpoint states, for param_dtype 'auto'.

    transformers' dtype='auto' rule (modeling_utils.py `_get_dtype`, 5.16.1):
    config.json's `dtype` (`torch_dtype` before 5.0), else the dtype of the
    first floating tensor. Packed FP8 or FP4 payloads are no storage dtype,
    so the first tensor stored in one is what a quantized checkpoint without
    a stated dtype resolves to. A diffusers pipeline states none, so its
    denoiser's tensors decide.
    """
    stated = config.get("dtype", config.get("torch_dtype"))
    if stated is not None:
        storage = dtype_name(resolve_dtype(records.text(stated, "dtype")))
        if storage is None:
            raise ValueError(f"dtype={stated!r} names no floating parameter storage")
        return storage
    storable = {np.dtype(np.float32): "float32", np.dtype(np.float16): "float16",
                np.dtype(ml_dtypes.bfloat16): "bfloat16"}
    for tensor in tensors.values():
        if tensor.dtype in storable:
            return storable[tensor.dtype]
    raise ValueError("param_dtype 'auto' found neither a stated dtype nor a float32, bfloat16 or "
                     "float16 tensor in the checkpoint")


def checkpoint_dtype(stored: np.dtype, param_dtype: str = "float32", *,
                     path: tuple[str, ...] = ()) -> np.dtype:
    """Floating payloads use parameter storage; integer and boolean state keeps its dtype."""
    if path and (path[0] == 'constants' or (path[0] == 'params' and path[-1] == 'head_bias')):
        param_dtype = 'float32'
    dtype = resolve_dtype(param_dtype)
    if dtype is None or not jnp.issubdtype(dtype, jnp.floating):
        raise ValueError(f"param_dtype {param_dtype!r} must name floating parameter storage")
    return np.dtype(dtype) if jnp.issubdtype(stored, jnp.floating) else stored


def checkpoint_array(tensor, param_dtype: str = "float32") -> np.ndarray:
    """Cast a stored floating array directly, without an FP32 intermediate."""
    leaf = np.asarray(tensor)
    return leaf.astype(checkpoint_dtype(leaf.dtype, param_dtype), copy=False)


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
        order = (*range(2, stored.ndim), 1, 0) if path[-1] == "kernel" else None
        transpose = None if order is None else tuple(int(axis) for axis in np.argsort(order))
        dtype = checkpoint_dtype(stored.dtype, param_dtype, path=path)
        yield name, path, read(stored, dtype, order), transpose


def translate_parameters(tensors: Mapping[str, np.ndarray], path_of: TensorPath,
                         param_dtype: str = "float32") -> ParamTree:
    """Read host-built parameters without creating deferred placement recipes.

    Repeated paths compare at their storage precision.
    """
    parameters: ParamTree = {}
    for name, path, leaf, _ in _leaves(tensors, path_of, param_dtype, _copy):
        insert(parameters, path, leaf, name)
    return parameters


def record_layouts(component: str, tensors: Mapping[str, np.ndarray], path_of: TensorPath,
                   prefix: tuple[str, ...], *, param_dtype: str = "float32", lazy: bool = False
                   ) -> tuple[LazyTree, tuple[WeightLayout, ...]]:
    """Bind native leaves and the layouts that rebuild every source tensor.

    Equal source aliases share a leaf and keep separate export names. With
    `lazy`, linear kernels and untransposed tensors stay `SourceLeaf` recipes;
    each placement reads and casts only its shard. Scoring state can request
    FP32 separately from parameters.
    """
    from dew.interop.streaming import SourceLeaf, WeightLayout, materialize

    parameters: LazyTree = {}
    layouts = []
    owners: dict[tuple[str, ...], str] = {}

    def locate(name: str) -> tuple[str, ...] | None:
        path = path_of(name)
        if path is not None:
            source_alias(tensors, owners, path, name)
        return path

    def mapped(stored: np.ndarray, dtype: np.dtype, order: tuple[int, ...] | None) -> np.ndarray | SourceLeaf:
        return (_copy(stored, dtype, order) if order is not None and stored.ndim != 2
                else SourceLeaf((stored,), dtype, transposed=order is not None))

    for name, path, leaf, transpose in _leaves(tensors, locate, param_dtype, mapped):
        layouts.append(WeightLayout(f"{component}/{name}", ((*prefix, *path),),
                                    tensors[name].shape, transpose))
        if owners[path] == name:
            insert(parameters, path, leaf, name)
    return (parameters if lazy else materialize(parameters)), tuple(layouts)
