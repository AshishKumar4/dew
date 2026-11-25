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
from dew.nn.derivatives import Protected, contracted, declared_tangent

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


def _eval_derived(kind: Literal["weighted", "masked", "covariance"], tiled: dict[str, jax.Array],
                  mode: Literal["walk", "merge"]):
    first = {name: tile_stack[0] for name, tile_stack in tiled.items()}
    rank = next(iter(first.values())).ndim
    forms = derived(kind, rank)
    shapes = {name: (value.shape, value.dtype) for name, value in first.items()}
    empty = forms.init(shapes)
    if mode == "walk":
        state, _ = jax.lax.scan(lambda carry, chunk: (forms.step(carry, chunk), None), empty, tiled)
        return state, forms.finish(state, first, shapes=shapes)
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
    return final, forms.finish(final, first, shapes=shapes)


def _weighted_term(tw: jax.Array, operand: jax.Array, sn: jax.Array, pos: jax.Array) -> jax.Array:
    """Range-safe `(tw / sn) * operand`, zeroed where `not pos` or `tw == 0`."""
    nz = pos & (tw > 0)
    safe_operand = jnp.where(nz, operand, 0.0)
    safe_tw = jnp.where(nz, tw, 0.0)
    small_tw = (tw <= 1.0) & (sn > 1.0)
    tw_mf = jnp.where(small_tw, safe_tw, 0.0)
    sn_mf = jnp.where(small_tw, sn, 1.0)
    tw_df = jnp.where(small_tw, 0.0, safe_tw)
    sn_df = jnp.where(small_tw, 1.0, sn)
    return jnp.where(small_tw, (tw_mf * safe_operand) / sn_mf, (tw_df / sn_df) * safe_operand)


def _safe_tangent_div(dtw: jax.Array, numerator: jax.Array, sn: jax.Array) -> jax.Array:
    """Linear-in-`dtw` evaluation of `dtw * numerator / sn` without `0 * inf` in forward or reverse mode."""
    qmax = jnp.asarray(0.25 * jnp.finfo(numerator.dtype).max, numerator.dtype)
    hi = sn > 1.0
    sn_hi = jnp.where(hi, sn, 1.0)
    sn_lo = jnp.where(hi, 1.0, sn)
    num_hi = jnp.where(hi, numerator, 0.0)
    dtw_hi = jnp.where(hi, dtw, 0.0)
    num_lo = jnp.where(hi, 0.0, numerator)
    dtw_lo = jnp.where(hi, 0.0, dtw)
    safe_lo = jnp.abs(num_lo) <= sn_lo * qmax
    num_safe = jnp.where(safe_lo, num_lo, 0.0)
    num_big = jnp.where(safe_lo, 0.0, num_lo)
    dtw_safe = jnp.where(safe_lo, dtw_lo, 0.0)
    dtw_big = jnp.where(safe_lo, 0.0, dtw_lo)
    sn_big = jnp.where(safe_lo, 1.0, sn_lo)
    lo_term = dtw_safe * (num_safe / sn_lo) + (dtw_big * num_big) / sn_big
    return jnp.where(hi, dtw_hi * (num_hi / sn_hi), lo_term)


def _cov_tangent_tile(tx: jax.Array, ty: jax.Array, tw: jax.Array,
                      dtx: jax.Array, dty: jax.Array, dtw: jax.Array,
                      mu_x: jax.Array, mu_y: jax.Array, cov: jax.Array,
                      n: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Exact final-center linear tangent `(d_mu_x, d_mu_y, d_cov)` of one tile for both `n > 0` and
    `n == 0` boundaries without multi-tile zero-mass exponent overflow."""
    pos = n[..., None] > 0
    sn = jnp.where(pos, n[..., None], 1.0)
    cx = tx - mu_x[..., None]
    cy = ty - mu_y[..., None]
    d_mx = jnp.sum(_weighted_term(tw, dtx, sn, pos) + _safe_tangent_div(dtw, cx, sn), axis=-1)
    d_my = jnp.sum(_weighted_term(tw, dty, sn, pos) + _safe_tangent_div(dtw, cy, sn), axis=-1)
    pw_cy = _weighted_term(tw, cy, sn, pos)
    pw_cx = _weighted_term(tw, cx, sn, pos)
    hi = sn > 1.0
    sn_hi = jnp.where(hi, sn, 1.0)
    big_c = jnp.where(jnp.abs(cx) >= jnp.abs(cy), cx, cy)
    sml_c = jnp.where(jnp.abs(cx) >= jnp.abs(cy), cy, cx)
    c_dw_hi = (jnp.where(hi, big_c, 0.0) / sn_hi) * jnp.where(hi, sml_c, 0.0) - (
        jnp.where(hi, cov[..., None], 0.0) / sn_hi)
    safe_cx = jnp.where(hi, 0.0, cx)
    safe_cy = jnp.where(hi, 0.0, cy)
    c_dw_lo = safe_cx * safe_cy - jnp.where(hi, 0.0, cov[..., None])
    d_cov_w = jnp.where(hi, jnp.where(hi, dtw, 0.0) * c_dw_hi,
                        _safe_tangent_div(jnp.where(hi, 0.0, dtw), c_dw_lo, jnp.where(hi, 1.0, sn)))
    d_cov = jnp.sum(pw_cy * dtx + pw_cx * dty + d_cov_w, axis=-1)
    return d_mx, d_my, d_cov


def _tile_array(value: jax.Array, shape: tuple[int, ...], tile: int) -> jax.Array:
    n_tiles = shape[-1] // tile
    return jnp.moveaxis(jnp.broadcast_to(value, shape).reshape(*shape[:-1], n_tiles, tile), -2, 0)


def _total_weight(shape: tuple[int, ...], w: jax.Array, tile: int,
                  mode: Literal["walk", "merge"]) -> jax.Array:
    w_stack = _tile_array(w, shape, tile)
    if mode == "walk":
        zeros = jnp.zeros(shape[:-1], w.dtype)
        total, _ = jax.lax.scan(lambda acc, chunk: (acc + jnp.sum(chunk, axis=-1), None), zeros, w_stack)
        return total
    return jnp.sum(w_stack, axis=(0, -1))


def _tiled_tangents(x: jax.Array, y: jax.Array, w: jax.Array,
                    dx: jax.Array, dy: jax.Array, dw: jax.Array,
                    mu_x: jax.Array, mu_y: jax.Array, cov: jax.Array, n: jax.Array,
                    tile: int, mode: Literal["walk", "merge"]) -> tuple[jax.Array, jax.Array, jax.Array]:
    shape = jnp.broadcast_shapes(x.shape, y.shape, w.shape)
    tx_s = _tile_array(x, shape, tile)
    ty_s = _tile_array(y, shape, tile)
    tw_s = _tile_array(w, shape, tile)
    dtx_s = _tile_array(dx, shape, tile)
    dty_s = _tile_array(dy, shape, tile)
    dtw_s = _tile_array(dw, shape, tile)
    if mode == "walk":
        zeros = (jnp.zeros_like(mu_x), jnp.zeros_like(mu_y), jnp.zeros_like(cov))

        def step(acc, chunk):
            tx, ty, tw, dtx, dty, dtw = chunk
            d_mx, d_my, d_cov = _cov_tangent_tile(tx, ty, tw, dtx, dty, dtw, mu_x, mu_y, cov, n)
            return (acc[0] + d_mx, acc[1] + d_my, acc[2] + d_cov), None

        totals, _ = jax.lax.scan(step, zeros, (tx_s, ty_s, tw_s, dtx_s, dty_s, dtw_s))
        return totals
    d_mx, d_my, d_cov = jax.vmap(
        lambda tx, ty, tw, dtx, dty, dtw: _cov_tangent_tile(tx, ty, tw, dtx, dty, dtw, mu_x, mu_y, cov, n)
    )(tx_s, ty_s, tw_s, dtx_s, dty_s, dtw_s)
    return jnp.sum(d_mx, axis=0), jnp.sum(d_my, axis=0), jnp.sum(d_cov, axis=0)


def _chunked_weighted_primal(x: jax.Array, w: jax.Array, tile: int,
                             mode: Literal["walk", "merge"]) -> tuple[jax.Array, jax.Array]:
    _, finished = _eval_derived("weighted", _tiled({"x": x, "w": w}, tile), mode)
    return finished


def _chunked_weighted_jvp(tile: int, mode: Literal["walk", "merge"], primals, tangents):
    x, w = primals
    dx, dw = tangents
    mean, var = _run_chunked_weighted(x, w, tile, mode)
    shape = tuple(int(size) for size in jnp.broadcast_shapes(x.shape, w.shape))
    n = _total_weight(shape, w, tile, mode)
    d_mean, _, d_var = _tiled_tangents(x, x, w, dx, dx, dw, mean, mean, var, n, tile, mode)
    return (mean, var), (d_mean, d_var)


_run_chunked_weighted = contracted("test_chunked_moments_holds_every_order")(
    declared_tangent(Protected(), _chunked_weighted_primal, _chunked_weighted_jvp, nondiff_argnums=(2, 3)))


def _chunked_cov_primal(x: jax.Array, y: jax.Array, w: jax.Array, tile: int,
                        mode: Literal["walk", "merge"]) -> tuple[jax.Array, jax.Array, jax.Array]:
    _, finished = _eval_derived("covariance", _tiled({"x": x, "y": y, "w": w}, tile), mode)
    return finished


def _chunked_cov_jvp(tile: int, mode: Literal["walk", "merge"], primals, tangents):
    x, y, w = primals
    dx, dy, dw = tangents
    mean_x, mean_y, cov = _run_chunked_cov(x, y, w, tile, mode)
    shape = tuple(int(size) for size in jnp.broadcast_shapes(x.shape, y.shape, w.shape))
    n = _total_weight(shape, w, tile, mode)
    d_mx, d_my, d_cov = _tiled_tangents(x, y, w, dx, dy, dw, mean_x, mean_y, cov, n, tile, mode)
    return (mean_x, mean_y, cov), (d_mx, d_my, d_cov)


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
        _, finished = _eval_derived("masked", _tiled({"x": x_arr.astype(work), "mask": w_arr}, tile), mode)
        return _cast_back(finished, work, out_dtype)
    out_dtype = _floating_dtype(x_arr, w_arr)
    work = _working_dtype(out_dtype)
    return _cast_back(_run_chunked_weighted(x_arr.astype(work), w_arr.astype(work), tile, mode),
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
                                       mode), work, out_dtype)
