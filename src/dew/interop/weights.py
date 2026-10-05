"""Checkpoint precision, native tensor layouts and reversible source names."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from dew.registry import resolve_dtype

if TYPE_CHECKING:
    from dew.interop.streaming import LazyTree, SourceLeaf, WeightLayout

type Tree[LeafT] = dict[str, LeafT | Tree[LeafT]]
type ParamTree = Tree[np.ndarray]
type TensorPath = Callable[[str], tuple[str, ...] | None]


def checkpoint_dtype(stored: np.dtype, param_dtype: str = "float32") -> np.dtype:
    """Floating payloads use parameter storage; integer and boolean state keeps its dtype."""
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


def _leaves(tensors: Mapping[str, np.ndarray], path_of: TensorPath, param_dtype: str
            ) -> Iterator[tuple[str, tuple[str, ...], np.ndarray | SourceLeaf, tuple[int, ...] | None]]:
    """Cast before the layout copy, keeping linear and untransposed leaves mapped.

    Convolutions need arbitrary axis permutations, so they are read whole.
    Frozen constants stay FP32 independently of parameter precision.
    """
    from dew.interop.streaming import SourceLeaf

    for name, tensor in tensors.items():
        path = path_of(name)
        if path is None:
            continue
        stored = np.asarray(tensor)
        order = (*range(2, stored.ndim), 1, 0) if path[-1] == "kernel" else None
        transpose = None if order is None else tuple(int(axis) for axis in np.argsort(order))
        dtype = checkpoint_dtype(stored.dtype, "float32" if path[0] == "constants" else param_dtype)
        leaf = (np.ascontiguousarray(stored.astype(dtype, copy=False).transpose(order))
                if order is not None and stored.ndim != 2
                else SourceLeaf((stored,), dtype, transposed=order is not None))
        yield name, path, leaf, transpose


def translate_parameters(tensors: Mapping[str, np.ndarray], path_of: TensorPath,
                         param_dtype: str = "float32") -> ParamTree:
    """Read native parameters, comparing repeated paths at their storage precision."""
    from dew.interop.streaming import SourceLeaf

    parameters: ParamTree = {}
    for name, path, leaf, _ in _leaves(tensors, path_of, param_dtype):
        insert(parameters, path, leaf.read() if isinstance(leaf, SourceLeaf) else leaf, name)
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
    from dew.interop.streaming import WeightLayout, materialize

    parameters: LazyTree = {}
    layouts = []
    owners: dict[tuple[str, ...], str] = {}

    def locate(name: str) -> tuple[str, ...] | None:
        path = path_of(name)
        if path is not None:
            source_alias(tensors, owners, path, name)
        return path

    for name, path, leaf, transpose in _leaves(tensors, locate, param_dtype):
        layouts.append(WeightLayout(f"{component}/{name}", ((*prefix, *path),),
                                    tensors[name].shape, transpose))
        if owners[path] == name:
            insert(parameters, path, leaf, name)
    return (parameters if lazy else materialize(parameters)), tuple(layouts)
