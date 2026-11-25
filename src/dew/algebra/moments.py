"""Central-moment declarations and algebra-level tiled consumer (`dew.algebra.derive`, docs/design/algebra.md
§1.4, §6.9): weighted and masked mean/variance and bilinear covariance walks derived through
Chan-Golub-LeVeque merge rules, producing finite variance and covariance across arbitrarily shifted data and
disjoint partials."""

from __future__ import annotations

import functools
import math
from typing import Literal

import jax
import jax.numpy as jnp

from dew.algebra import derive, program
from dew.algebra.forms import Derived

SATURATION = program.Saturation(growth=64, depth=4)
"""Derivation limits for central-moment walks."""


def weighted_moments(x: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Weighted `(mean, variance)` of `x` along its last axis under nonnegative weights `w`, zero where
    total weight is zero."""
    n = jnp.sum(w, axis=-1)
    safe_n = jnp.where(n > 0, n, jnp.ones_like(n))
    mean = jnp.sum(w * x, axis=-1) / safe_n
    diff = x - mean[..., None]
    m2 = jnp.sum(w * diff * diff, axis=-1)
    return mean, m2 / safe_n


def masked_moments(x: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Masked unweighted `(mean, variance)` of `x` along its last axis over boolean `mask`, zero where
    every entry is masked out."""
    n = jnp.sum(mask.astype(x.dtype), axis=-1)
    safe_n = jnp.where(n > 0, n, jnp.ones_like(n))
    mean = jnp.sum(jnp.where(mask, x, jnp.zeros_like(x)), axis=-1) / safe_n
    diff = x - mean[..., None]
    m2 = jnp.sum(jnp.where(mask, diff * diff, jnp.zeros_like(x)), axis=-1)
    return mean, m2 / safe_n


def weighted_covariance(x: jax.Array, y: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Weighted `(mean_x, mean_y, covariance)` of `(x, y)` along their last axis under nonnegative `w`."""
    n = jnp.sum(w, axis=-1)
    safe_n = jnp.where(n > 0, n, jnp.ones_like(n))
    mean_x = jnp.sum(w * x, axis=-1) / safe_n
    mean_y = jnp.sum(w * y, axis=-1) / safe_n
    dx = x - mean_x[..., None]
    dy = y - mean_y[..., None]
    cov = jnp.sum(w * dx * dy, axis=-1) / safe_n
    return mean_x, mean_y, cov


@functools.cache
def declaration(kind: Literal["weighted", "masked", "covariance"], rank: int = 2) -> derive.Reduction:
    """`derive.Reduction` for `kind` over `rank`-D arrays walked along their last axis."""
    axes = (*(f"d{i}" for i in range(rank - 1)), "walked")
    if kind == "weighted":
        return derive.Reduction(weighted_moments, {"x": axes, "w": axes}, walked="walked",
                                ranges={"w": (0.0, math.inf)})
    if kind == "masked":
        return derive.Reduction(masked_moments, {"x": axes, "mask": axes}, walked="walked")
    if kind == "covariance":
        return derive.Reduction(weighted_covariance, {"x": axes, "y": axes, "w": axes}, walked="walked",
                                ranges={"w": (0.0, math.inf)})
    raise ValueError(f"unknown central-moment declaration kind: {kind!r}")


@functools.cache
def derived(kind: Literal["weighted", "masked", "covariance"], rank: int = 2, *,
            accurate: bool = False) -> Derived:
    """Derived `(init, step, merge, finish)` forms for `declaration(kind, rank)`."""
    return derive.derive(declaration(kind, rank), accurate=accurate, saturation=SATURATION)


def _tiled(operands: dict[str, jax.Array], tile: int) -> dict[str, jax.Array]:
    """Broadcast `operands` to one rank >= 1 shape and reshape `(..., n_tiles * tile)` to
    `(n_tiles, ..., tile)` so scans and vmapped tree merges trace one tile body."""
    arrays = jnp.broadcast_arrays(*operands.values())
    common = dict(zip(operands, arrays, strict=True))
    shape = arrays[0].shape
    if not shape:
        raise ValueError("central-moment operands must have rank at least 1")
    length = shape[-1]
    if tile <= 0 or length <= 0 or length % tile != 0:
        raise ValueError(f"walked extent {length} must be a positive multiple of tile {tile}")
    n_tiles = length // tile
    leading = shape[:-1]
    return {name: jnp.moveaxis(value.reshape(*leading, n_tiles, tile), -2, 0)
            for name, value in common.items()}


def _run_chunked(kind: Literal["weighted", "masked", "covariance"], operands: dict[str, jax.Array], *,
                 tile: int, mode: Literal["walk", "merge"] = "merge"):
    tiled = _tiled(operands, tile)
    first = {name: tile_stack[0] for name, tile_stack in tiled.items()}
    rank = next(iter(first.values())).ndim
    forms = derived(kind, rank)
    shapes = {name: (value.shape, value.dtype) for name, value in first.items()}
    empty = forms.init(shapes)
    if mode == "walk":
        state, _ = jax.lax.scan(lambda carry, chunk: (forms.step(carry, chunk), None), empty, tiled)
        return forms.finish(state, first, shapes=shapes)
    partials = jax.vmap(lambda chunk: forms.step(empty, chunk))(tiled)
    while partials[0].shape[0] > 1:
        count = partials[0].shape[0]
        limit = 2 * (count // 2)
        left = jax.tree.map(lambda state, end=limit: state[:end:2], partials)
        right = jax.tree.map(lambda state, end=limit: state[1:end:2], partials)
        merged = jax.vmap(lambda a, b: forms.merge(a, b, shapes=shapes))(left, right)
        if count % 2:
            tail = jax.tree.map(lambda state: state[-1:], partials)
            merged = jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), merged, tail)
        partials = merged
    final = jax.tree.map(lambda state: state[0], partials)
    return forms.finish(final, first, shapes=shapes)


def _floating_dtype(*arrays: jax.Array):
    dtype = jnp.result_type(*arrays)
    return dtype if jnp.issubdtype(dtype, jnp.floating) else jnp.result_type(dtype, jnp.float32)


def chunked_moments(x: jax.Array, w: jax.Array, *, tile: int,
                    mode: Literal["walk", "merge"] = "merge") -> tuple[jax.Array, jax.Array]:
    """Compute weighted or boolean-masked `(mean, variance)` of `x` along its last axis in `tile`-wide chunks
    using derived Chan-Golub-LeVeque forms (`mode="walk"` via `lax.scan`, `mode="merge"` via vmapped
    pairwise tree reduction). Operands broadcast to a common shape of rank >= 1 and positive walked extent
    divisible by `tile`; non-boolean weights must be nonnegative and promote with `x` to a floating dtype."""
    x_arr, w_arr = jnp.asarray(x), jnp.asarray(w)
    if w_arr.dtype == jnp.bool_:
        dtype = _floating_dtype(x_arr)
        return _run_chunked("masked", {"x": x_arr.astype(dtype), "mask": w_arr}, tile=tile, mode=mode)
    dtype = _floating_dtype(x_arr, w_arr)
    return _run_chunked("weighted", {"x": x_arr.astype(dtype), "w": w_arr.astype(dtype)}, tile=tile,
                        mode=mode)


def chunked_covariance(x: jax.Array, y: jax.Array, w: jax.Array, *, tile: int,
                       mode: Literal["walk", "merge"] = "merge") -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute weighted `(mean_x, mean_y, covariance)` of `(x, y)` along their last axis in `tile`-wide
    chunks using derived Chan-Golub-LeVeque forms. Operands broadcast to a common shape of rank >= 1 and
    positive walked extent divisible by `tile`, with nonnegative `w` promoted to one floating dtype."""
    x_arr, y_arr, w_arr = jnp.asarray(x), jnp.asarray(y), jnp.asarray(w)
    dtype = _floating_dtype(x_arr, y_arr, w_arr)
    return _run_chunked("covariance",
                        {"x": x_arr.astype(dtype), "y": y_arr.astype(dtype), "w": w_arr.astype(dtype)},
                        tile=tile, mode=mode)
