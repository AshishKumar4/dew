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
from dew.algebra.ratios import Ratio, ratio
from dew.algebra.windows import Units
from dew.nn.derivatives import Protected, contracted, declared_tangent

SATURATION = program.Saturation(growth=64, depth=4)
"""Derivation limits for central-moment walks."""


def _weighted_moments_with_total(x: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Weighted `(mean, variance, total_weight)` of `x` along its last axis under nonnegative `w`."""
    n = jnp.sum(w, axis=-1)
    safe_n = jnp.where(n > 0, n, jnp.ones_like(n))
    mean = jnp.sum(w * x, axis=-1) / safe_n
    diff = x - mean[..., None]
    m2 = jnp.sum(w * diff * diff, axis=-1)
    return mean, m2 / safe_n, n


def weighted_moments(x: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Weighted `(mean, variance)` of `x` along its last axis under nonnegative weights `w`, zero where
    total weight is zero."""
    return _weighted_moments_with_total(x, w)[:2]


def masked_moments(x: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Masked unweighted `(mean, variance)` of `x` along its last axis over boolean `mask`, zero where
    every entry is masked out."""
    n = jnp.sum(mask.astype(x.dtype), axis=-1)
    safe_n = jnp.where(n > 0, n, jnp.ones_like(n))
    mean = jnp.sum(jnp.where(mask, x, jnp.zeros_like(x)), axis=-1) / safe_n
    diff = x - mean[..., None]
    m2 = jnp.sum(jnp.where(mask, diff * diff, jnp.zeros_like(x)), axis=-1)
    return mean, m2 / safe_n


def _weighted_cov_with_total(
    x: jax.Array, y: jax.Array, w: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Weighted `(mean_x, mean_y, covariance, total_weight)` of `(x, y)` along their last axis."""
    n = jnp.sum(w, axis=-1)
    safe_n = jnp.where(n > 0, n, jnp.ones_like(n))
    mean_x = jnp.sum(w * x, axis=-1) / safe_n
    mean_y = jnp.sum(w * y, axis=-1) / safe_n
    dx = x - mean_x[..., None]
    dy = y - mean_y[..., None]
    cov = jnp.sum(w * dx * dy, axis=-1) / safe_n
    return mean_x, mean_y, cov, n


def weighted_covariance(x: jax.Array, y: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Weighted `(mean_x, mean_y, covariance)` of `(x, y)` along their last axis under nonnegative `w`."""
    return _weighted_cov_with_total(x, y, w)[:3]


def _reduction(kind: Literal["weighted", "masked", "covariance"], rank: int, *,
               with_total: bool = False) -> derive.Reduction:
    """Build one `derive.Reduction` for `kind` over `rank`-D arrays walked along their last axis, optionally
    exposing total weight `n` as the final declared output (reusing the same reduction contract)."""
    axes = (*(f"d{i}" for i in range(rank - 1)), "walked")
    if kind == "weighted":
        fn = _weighted_moments_with_total if with_total else weighted_moments
        return derive.Reduction(fn, {"x": axes, "w": axes}, walked="walked", ranges={"w": (0.0, math.inf)})
    if kind == "masked":
        return derive.Reduction(masked_moments, {"x": axes, "mask": axes}, walked="walked")
    if kind == "covariance":
        fn = _weighted_cov_with_total if with_total else weighted_covariance
        return derive.Reduction(fn, {"x": axes, "y": axes, "w": axes}, walked="walked",
                                ranges={"w": (0.0, math.inf)})
    raise ValueError(f"unknown central-moment declaration kind: {kind!r}")


@functools.cache
def declaration(kind: Literal["weighted", "masked", "covariance"], rank: int = 2) -> derive.Reduction:
    """`derive.Reduction` for `kind` over `rank`-D arrays walked along their last axis."""
    return _reduction(kind, rank, with_total=False)


@functools.cache
def derived(kind: Literal["weighted", "masked", "covariance"], rank: int = 2, *,
            accurate: bool = False) -> Derived:
    """Derived `(init, step, merge, finish)` forms for `declaration(kind, rank)`."""
    return derive.derive(declaration(kind, rank), accurate=accurate, saturation=SATURATION)


@functools.cache
def _derived_with_total(kind: Literal["weighted", "covariance"], rank: int = 2) -> Derived:
    """Derived forms exposing total weight `n` as the final declared output (no extra reduction slot)."""
    return derive.derive(_reduction(kind, rank, with_total=True), accurate=False, saturation=SATURATION)


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


def _eval_derived(forms: Derived, tiled: dict[str, jax.Array], mode: Literal["walk", "merge"]):
    from dew.nn.sharding import explicit_spec

    first = {name: tile_stack[0] for name, tile_stack in tiled.items()}
    shapes = {name: (value.shape, value.dtype) for name, value in first.items()}
    empty = forms.init(shapes)
    placed = explicit_spec(first["x"])
    if placed is not None and placed.spec[-1] is None:
        state_spec = jax.sharding.NamedSharding(placed.mesh, jax.sharding.PartitionSpec(*placed.spec[:-1]))
        empty = jax.tree.map(lambda value: jax.sharding.reshard(value, state_spec), empty)
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


_MUL_DIV = Ratio(
    numerator=((1.0, (("v", 0), ("v", 1))),),
    denominator=((1.0, (("v", 2),)),),
    axis=0,
    reduced=True,
    units=Units(),
)
_COV_SHIFT_DIV = Ratio(
    numerator=((1.0, (("v", 0), ("v", 1), ("v", 2))),),
    denominator=((0.5, (("v", 3),)),),
    axis=0,
    reduced=True,
    units=Units(),
)
_COV_WEIGHT_TANGENT = Ratio(
    numerator=((1.0, (("v", 0), ("v", 1), ("v", 2))), (-1.0, (("v", 0), ("v", 3)))),
    denominator=((1.0, (("v", 4),)),),
    axis=0,
    reduced=True,
    units=Units(),
)


def _scaled_ratio(form: Ratio, *operands: jax.Array) -> jax.Array:
    """Evaluate a product-over-denominator ratio elementwise via `dew.algebra.ratios.ratio` on a leading
    singleton axis (`axis=0`), matching `means.finished`/`resumed` so products, differences, quotients, and
    higher-order derivatives stay in scaled double-float arithmetic."""
    return ratio([operand[None] for operand in operands], form)


def _cov_value_tangent(dval: jax.Array, dmean: jax.Array, other_c: jax.Array, tw: jax.Array,
                       sn: jax.Array) -> jax.Array:
    """`(dval - dmean) * other_c * tw / sn` via `ratio` over a 2-state axis holding `[dval, -dmean]`: one
    linear operand (valid under partial VJPs such as weight-only `grad(cov)(w)`), cancelling uniform shifts
    before dividing by `sn` (`0.5 * sn` summed twice equals `sn` exactly)."""
    packed = jnp.stack([dval, -jnp.broadcast_to(dmean[..., None], dval.shape)], axis=0)
    return ratio([packed, other_c[None], tw[None], sn[None]], _COV_SHIFT_DIV)


def _mean_tangent_tile(
    tx: jax.Array, ty: jax.Array, tw: jax.Array,
    dtx: jax.Array, dty: jax.Array, dtw: jax.Array,
    mu_x: jax.Array, mu_y: jax.Array, n: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Exact final-center linear tangent `(d_mu_x, d_mu_y, d_n)` of one tile."""
    pos = n[..., None] > 0
    sn = jnp.where(pos, n[..., None], 1.0)
    mu_x_pos = jnp.where(pos, mu_x[..., None], 0.0)
    mu_y_pos = jnp.where(pos, mu_y[..., None], 0.0)
    d_mx = jnp.sum(_scaled_ratio(_MUL_DIV, dtx, tw, sn) + _scaled_ratio(_MUL_DIV, dtw, tx - mu_x_pos, sn),
                   axis=-1)
    d_my = jnp.sum(_scaled_ratio(_MUL_DIV, dty, tw, sn) + _scaled_ratio(_MUL_DIV, dtw, ty - mu_y_pos, sn),
                   axis=-1)
    return d_mx, d_my, jnp.sum(dtw, axis=-1)


def _cov_tangent_tile(
    tx: jax.Array, ty: jax.Array, tw: jax.Array,
    dtx: jax.Array, dty: jax.Array, dtw: jax.Array,
    mu_x: jax.Array, mu_y: jax.Array, cov: jax.Array, n: jax.Array,
    d_mx: jax.Array, d_my: jax.Array,
) -> jax.Array:
    """Exact final-center linear covariance tangent `d_cov` of one tile, combining each value tangent with
    the mean recentering shift (`dtx - d_mx`, `dty - d_my`) inside `Ratio` before dividing by `sn`."""
    pos = n[..., None] > 0
    sn = jnp.where(pos, n[..., None], 1.0)
    cx = tx - mu_x[..., None]
    cy = ty - mu_y[..., None]
    cy_safe = jnp.where((tw > 0) | jnp.isfinite(cy), cy, 0.0)
    cx_safe = jnp.where((tw > 0) | jnp.isfinite(cx), cx, 0.0)
    cov_term = jnp.where(pos, cov[..., None], 0.0)
    d_cov_x = _cov_value_tangent(dtx, d_mx, cy_safe, tw, sn)
    d_cov_y = _cov_value_tangent(dty, d_my, cx_safe, tw, sn)
    d_cov_w = _scaled_ratio(_COV_WEIGHT_TANGENT, dtw, cx, cy, cov_term, sn)
    return jnp.sum(d_cov_x + d_cov_y + d_cov_w, axis=-1)


def _tile_array(value: jax.Array, shape: tuple[int, ...], tile: int) -> jax.Array:
    n_tiles = shape[-1] // tile
    return jnp.moveaxis(jnp.broadcast_to(value, shape).reshape(*shape[:-1], n_tiles, tile), -2, 0)


def _reduce_tiles(fn, zeros, tiles: tuple[jax.Array, ...], mode: Literal["walk", "merge"]):
    if mode == "walk":
        def step(acc, chunk):
            parts = fn(*chunk)
            return jax.tree.map(lambda prev, delta: prev + delta, acc, parts), None

        return jax.lax.scan(step, zeros, tiles)[0]
    return jax.tree.map(lambda part: jnp.sum(part, axis=0), jax.vmap(fn)(*tiles))


def _tiled_tangents(
    x: jax.Array, y: jax.Array, w: jax.Array,
    dx: jax.Array, dy: jax.Array, dw: jax.Array,
    mu_x: jax.Array, mu_y: jax.Array, cov: jax.Array, n: jax.Array,
    tile: int, mode: Literal["walk", "merge"],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    shape = tuple(int(size) for size in jnp.broadcast_shapes(x.shape, y.shape, w.shape))
    tiles = tuple(_tile_array(value, shape, tile) for value in (x, y, w, dx, dy, dw))
    d_mx, d_my, d_n = _reduce_tiles(
        lambda tx, ty, tw, dtx, dty, dtw: _mean_tangent_tile(tx, ty, tw, dtx, dty, dtw, mu_x, mu_y, n),
        (jnp.zeros_like(mu_x), jnp.zeros_like(mu_y), jnp.zeros_like(n)), tiles, mode)
    d_cov = _reduce_tiles(
        lambda tx, ty, tw, dtx, dty, dtw: _cov_tangent_tile(
            tx, ty, tw, dtx, dty, dtw, mu_x, mu_y, cov, n, d_mx, d_my),
        jnp.zeros_like(cov), tiles, mode)
    return d_mx, d_my, d_cov, d_n


def _chunked_weighted_primal(x: jax.Array, w: jax.Array, tile: int,
                             mode: Literal["walk", "merge"]) -> tuple[jax.Array, jax.Array, jax.Array]:
    tiled = _tiled({"x": x, "w": w}, tile)
    return _eval_derived(_derived_with_total("weighted", tiled["x"].ndim - 1), tiled, mode)


def _chunked_weighted_jvp(tile: int, mode: Literal["walk", "merge"], primals, tangents):
    x, w = primals
    dx, dw = tangents
    mean, var, n = _run_chunked_weighted(x, w, tile, mode)
    d_mean, _, d_var, d_n = _tiled_tangents(x, x, w, dx, dx, dw, mean, mean, var, n, tile, mode)
    return (mean, var, n), (d_mean, d_var, d_n)


_run_chunked_weighted = contracted("test_chunked_moments_holds_every_order")(
    declared_tangent(Protected(), _chunked_weighted_primal, _chunked_weighted_jvp, nondiff_argnums=(2, 3)))


def _chunked_cov_primal(
    x: jax.Array, y: jax.Array, w: jax.Array, tile: int, mode: Literal["walk", "merge"],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    tiled = _tiled({"x": x, "y": y, "w": w}, tile)
    return _eval_derived(_derived_with_total("covariance", tiled["x"].ndim - 1), tiled, mode)


def _chunked_cov_jvp(tile: int, mode: Literal["walk", "merge"], primals, tangents):
    x, y, w = primals
    dx, dy, dw = tangents
    mean_x, mean_y, cov, n = _run_chunked_cov(x, y, w, tile, mode)
    d_mx, d_my, d_cov, d_n = _tiled_tangents(x, y, w, dx, dy, dw, mean_x, mean_y, cov, n, tile, mode)
    return (mean_x, mean_y, cov, n), (d_mx, d_my, d_cov, d_n)


_run_chunked_cov = contracted("test_chunked_moments_holds_every_order")(
    declared_tangent(Protected(), _chunked_cov_primal, _chunked_cov_jvp, nondiff_argnums=(3, 4)))


def _floating_dtype(*arrays: jax.Array):
    dtype = jnp.result_type(*arrays)
    return dtype if jnp.issubdtype(dtype, jnp.floating) else jnp.result_type(dtype, jnp.float32)


def _working_dtype(dtype):
    """Accumulation dtype for central-moment walks: at least float32 so narrow 16-bit inputs do not trigger
    implicit widen-reduce-narrow casts inside `jnp.sum`."""
    return jnp.promote_types(dtype, jnp.float32)


def _cast_back(outputs, work, out_dtype):
    return outputs if work == out_dtype else jax.tree.map(lambda value: value.astype(out_dtype), outputs)


def chunked_moments(x: jax.Array, w: jax.Array, *, tile: int,
                    mode: Literal["walk", "merge"] = "merge") -> tuple[jax.Array, jax.Array]:
    """Compute weighted or boolean-masked `(mean, variance)` of `x` along its last axis in `tile`-wide chunks
    using derived Chan-Golub-LeVeque forms (`mode="walk"` via `lax.scan`, `mode="merge"` via vmapped
    pairwise tree reduction). Operands broadcast to a common shape of rank >= 1 and positive walked extent
    divisible by `tile`; non-boolean weights must be nonnegative and promote with `x` to a floating dtype
    (accumulated in at least float32 for narrow float16/bfloat16 inputs and cast back)."""
    x_arr, w_arr = jnp.asarray(x), jnp.asarray(w)
    if w_arr.dtype == jnp.bool_:
        out_dtype = _floating_dtype(x_arr)
        work = _working_dtype(out_dtype)
        tiled = _tiled({"x": x_arr.astype(work), "mask": w_arr}, tile)
        finished = _eval_derived(derived("masked", tiled["x"].ndim - 1), tiled, mode)
        return _cast_back(finished, work, out_dtype)
    out_dtype = _floating_dtype(x_arr, w_arr)
    work = _working_dtype(out_dtype)
    return _cast_back(_run_chunked_weighted(x_arr.astype(work), w_arr.astype(work), tile, mode)[:2],
                      work, out_dtype)


def chunked_covariance(x: jax.Array, y: jax.Array, w: jax.Array, *, tile: int,
                       mode: Literal["walk", "merge"] = "merge") -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute weighted `(mean_x, mean_y, covariance)` of `(x, y)` along their last axis in `tile`-wide
    chunks using derived Chan-Golub-LeVeque forms. Operands broadcast to a common shape of rank >= 1 and
    positive walked extent divisible by `tile`, with nonnegative `w` promoted to one floating dtype
    (accumulated in at least float32 for narrow float16/bfloat16 inputs and cast back)."""
    x_arr, y_arr, w_arr = jnp.asarray(x), jnp.asarray(y), jnp.asarray(w)
    out_dtype = _floating_dtype(x_arr, y_arr, w_arr)
    work = _working_dtype(out_dtype)
    return _cast_back(_run_chunked_cov(x_arr.astype(work), y_arr.astype(work), w_arr.astype(work), tile,
                                       mode)[:3], work, out_dtype)
