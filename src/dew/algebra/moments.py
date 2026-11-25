"""Central-moment declarations and algebra-level tiled consumer (`dew.algebra.derive`, docs/design/algebra.md
§1.4, §6.9): weighted and masked mean/variance and bilinear covariance walks derived through
Chan-Golub-LeVeque merge rules, producing finite variance and covariance across arbitrarily shifted data and
disjoint partials."""

from __future__ import annotations

import functools
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
        return derive.Reduction(weighted_moments, {"x": axes, "w": axes}, walked="walked")
    if kind == "masked":
        return derive.Reduction(masked_moments, {"x": axes, "mask": axes}, walked="walked")
    if kind == "covariance":
        return derive.Reduction(weighted_covariance, {"x": axes, "y": axes, "w": axes}, walked="walked")
    raise ValueError(f"unknown central-moment declaration kind: {kind!r}")


@functools.cache
def derived(kind: Literal["weighted", "masked", "covariance"], rank: int = 2, *,
            accurate: bool = False) -> Derived:
    """Derived `(init, step, merge, finish)` forms for `declaration(kind, rank)`."""
    return derive.derive(declaration(kind, rank), accurate=accurate, saturation=SATURATION)


def _shaped(tile: dict[str, jax.Array]) -> dict[str, tuple[tuple[int, ...], object]]:
    return {name: (value.shape, value.dtype) for name, value in tile.items()}


def _tiles(operands: dict[str, jax.Array], tile: int) -> list[dict[str, jax.Array]]:
    length = next(iter(operands.values())).shape[-1]
    if tile <= 0 or length % tile != 0:
        raise ValueError(f"walked extent {length} must be a positive multiple of tile {tile}")
    return [{name: value[..., start:start + tile] for name, value in operands.items()}
            for start in range(0, length, tile)]


def _tree_merged(forms: Derived, tiles: list[dict[str, jax.Array]]):
    shapes = _shaped(tiles[0])
    empty = forms.init(shapes)
    partials = [forms.step(empty, chunk) for chunk in tiles]
    while len(partials) > 1:
        merged = [forms.merge(partials[i], partials[i + 1], shapes=shapes)
                  for i in range(0, len(partials) - 1, 2)]
        if len(partials) % 2:
            merged.append(partials[-1])
        partials = merged
    return forms.finish(partials[0], tiles[0], shapes=shapes)


def _walked(forms: Derived, tiles: list[dict[str, jax.Array]]):
    shapes = _shaped(tiles[0])
    state = forms.init(shapes)
    for chunk in tiles:
        state = forms.step(state, chunk)
    return forms.finish(state, tiles[0], shapes=shapes)


def _run_chunked(kind: Literal["weighted", "masked", "covariance"], operands: dict[str, jax.Array], *,
                 tile: int, mode: Literal["walk", "merge"] = "merge"):
    rank = next(iter(operands.values())).ndim
    forms = derived(kind, rank)
    chunks = _tiles(operands, tile)
    return _tree_merged(forms, chunks) if mode == "merge" else _walked(forms, chunks)


def chunked_moments(x: jax.Array, w: jax.Array, *, tile: int,
                    mode: Literal["walk", "merge"] = "merge") -> tuple[jax.Array, jax.Array]:
    """Compute weighted or boolean-masked `(mean, variance)` of `x` along its last axis in `tile`-wide chunks
    using derived Chan-Golub-LeVeque forms (`mode="walk"` for sequential tiles, `mode="merge"` for pairwise
    tree reduction of independent partial states)."""
    if w.dtype == jnp.bool_:
        return _run_chunked("masked", {"x": x, "mask": w}, tile=tile, mode=mode)
    return _run_chunked("weighted", {"x": x, "w": w}, tile=tile, mode=mode)


def chunked_covariance(x: jax.Array, y: jax.Array, w: jax.Array, *, tile: int,
                       mode: Literal["walk", "merge"] = "merge") -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute weighted `(mean_x, mean_y, covariance)` of `(x, y)` along their last axis in `tile`-wide
    chunks using derived Chan-Golub-LeVeque forms."""
    return _run_chunked("covariance", {"x": x, "y": y, "w": w}, tile=tile, mode=mode)
