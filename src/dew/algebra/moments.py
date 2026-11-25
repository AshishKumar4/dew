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

from dew.algebra import derive, program, ratios, windows
from dew.algebra.forms import Derived
from dew.nn import float_bits
from dew.nn.derivatives import Protected, contracted, declared_tangent
from dew.nn.precision import power2

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


def _weighted_term(tw: jax.Array, operand: jax.Array, sn: jax.Array) -> jax.Array:
    """Range-safe `(tw / sn) * operand`, strictly linear in `operand` and preserving `tw` derivatives."""
    small_tw = (tw <= 1.0) & (sn > 1.0)
    tw_mf = jnp.where(small_tw, tw, 0.0)
    sn_mf = jnp.where(small_tw, sn, 1.0)
    tw_df = jnp.where(small_tw, 0.0, tw)
    sn_df = jnp.where(small_tw, 1.0, sn)
    return jnp.where(small_tw, (tw_mf * operand) / sn_mf, (tw_df / sn_df) * operand)


def _primal_exponent(value: jax.Array) -> jax.Array:
    """`floor(log2(|value|))` of primal `value` (`int32`, stopped for AD), 0 where zero or nonfinite."""
    plain = jax.lax.stop_gradient(value)
    _, biased, fraction, _, exponent = float_bits.float_bits(plain)
    valid = jnp.isfinite(plain) & ((biased != 0) | (fraction != 0))
    return jnp.where(valid, exponent.astype(jnp.int32), jnp.int32(0))


def _safe_tangent_div(dtw: jax.Array, numerator: jax.Array, sn: jax.Array) -> jax.Array:
    """Linear-in-`dtw` evaluation of `dtw * numerator / sn` using `dew.nn.precision.power2`: balances
    `k_dtw` and `q = power2(numerator, k_n) / u_den` so both factors move in the same direction (`>= 1` when
    `|R| >= 1`, `<= 1` when `|R| < 1`), halves `u_sn` when `k_num > 0` so `power2(dx, k_num - 1)` cannot
    overflow ahead of `/ u_den`, moves `/ u_sn` onto `scaled_dtw` when `e_sn >= 0` and `e_num <= minexp`,
    and puts positive shift onto zero `numerator` to avoid `inf * 0`."""
    minexp = int(jnp.finfo(numerator.dtype).minexp)
    e_sn = _primal_exponent(sn)
    u_sn = power2(sn, -e_sn)
    plain_num = jax.lax.stop_gradient(numerator)
    zero_num = plain_num == 0
    e_num = _primal_exponent(numerator)
    m_q = jnp.abs(power2(plain_num, -e_num)) / jax.lax.stop_gradient(u_sn)
    q_ge_1 = m_q >= 1.0
    e_total = -e_sn
    e_r = e_num + e_total - jnp.where(q_ge_1, jnp.int32(0), jnp.int32(1))
    need = -e_num + jnp.where(q_ge_1, jnp.int32(0), jnp.int32(1))
    drop = -e_num - jnp.where(q_ge_1, jnp.int32(1), jnp.int32(0))
    k_up = jnp.where(need > 0, need, jnp.minimum(e_total, jnp.int32(0)))
    k_dn = jnp.where(drop < 0, drop, jnp.maximum(e_total, jnp.int32(0)))
    k_num = jnp.where(zero_num, jnp.maximum(e_total, jnp.int32(0)), jnp.where(e_r >= 0, k_up, k_dn))
    k_dtw = e_total - k_num
    div_on_dtw = (~zero_num) & (e_r < 0) & (drop >= 0) & (e_total <= 0) & (e_num <= minexp)
    scaled_dtw = power2(dtw, k_dtw)
    k_n = jnp.where(k_num > 0, k_num - jnp.int32(1), k_num)
    u_den = jnp.where(k_num > 0, u_sn * 0.5, u_sn)
    on_dtw = (jnp.where(div_on_dtw, scaled_dtw, 0.0) / u_sn) * jnp.where(div_on_dtw, numerator, 0.0)
    on_num = (jnp.where(div_on_dtw, 0.0, scaled_dtw)
              * (power2(jnp.where(div_on_dtw, 0.0, numerator), k_n) / u_den))
    return jnp.where(div_on_dtw, on_dtw, on_num)


def _safe_tangent_quad(dtw: jax.Array, cx: jax.Array, cy: jax.Array, sn: jax.Array) -> jax.Array:
    """Range-safe `dtw * cx * cy / sn` via singleton `ratios.ratio_p`, keeping primal and higher-order
    mixed derivatives in one scaled-product operation without primal-only exponent scaling of `cx`."""
    form = ratios.Ratio(
        numerator=((1.0, (("v", 0), ("v", 1), ("v", 2))),),
        denominator=((1.0, (("v", 3),)),),
        axis=dtw.ndim,
        reduced=True,
        units=windows.Units(),
    )
    return ratios.ratio([dtw[..., None], cx[..., None], cy[..., None], sn[..., None]], form)


def _cov_tangent_tile(
    tx: jax.Array, ty: jax.Array, tw: jax.Array,
    dtx: jax.Array, dty: jax.Array, dtw: jax.Array,
    mu_x: jax.Array, mu_y: jax.Array, cov: jax.Array, n: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Exact final-center linear tangent `(d_mu_x, d_mu_y, d_cov_raw, sum_pw_cy, sum_pw_cx, d_n)` of one
    tile for both `n > 0` and `n == 0` boundaries without multi-tile zero-mass exponent overflow."""
    pos = n[..., None] > 0
    sn = jnp.where(pos, n[..., None], 1.0)
    mu_x_pos = jnp.where(pos, mu_x[..., None], 0.0)
    mu_y_pos = jnp.where(pos, mu_y[..., None], 0.0)
    cx = tx - mu_x[..., None]
    cy = ty - mu_y[..., None]
    d_mx = jnp.sum(_weighted_term(tw, dtx, sn) + _safe_tangent_div(dtw, tx - mu_x_pos, sn), axis=-1)
    d_my = jnp.sum(_weighted_term(tw, dty, sn) + _safe_tangent_div(dtw, ty - mu_y_pos, sn), axis=-1)
    pw_cy = _weighted_term(tw, jnp.where((tw > 0) | jnp.isfinite(cy), cy, 0.0), sn)
    pw_cx = _weighted_term(tw, jnp.where((tw > 0) | jnp.isfinite(cx), cx, 0.0), sn)
    cov_term = jnp.where(pos, cov[..., None], 0.0)
    limits = jnp.finfo(cx.dtype)
    half_max = jnp.asarray(0.5 * limits.max, cx.dtype)
    tiny = jnp.asarray(limits.tiny, cx.dtype)
    cx_sg, cy_sg = jax.lax.stop_gradient(cx), jax.lax.stop_gradient(cy)
    big_sg = jnp.maximum(jnp.abs(cx_sg), jnp.abs(cy_sg))
    sml_sg = jnp.minimum(jnp.abs(cx_sg), jnp.abs(cy_sg))
    prod_ovf = (big_sg > 1.0) & (sml_sg > half_max / jnp.maximum(big_sg, 1.0))
    prod_unf = (big_sg != 0) & (sml_sg != 0) & (big_sg * sml_sg < tiny)
    prod_split = prod_ovf | prod_unf
    cx_safe = jnp.where(prod_split, 0.0, cx)
    cy_safe = jnp.where(prod_split, 0.0, cy)
    d_cov_no = _safe_tangent_div(jnp.where(prod_split, 0.0, dtw),
                                 cx_safe * cy_safe - jnp.where(prod_split, 0.0, cov_term),
                                 jnp.where(prod_split, 1.0, sn))
    dtw_sp = jnp.where(prod_split, dtw, 0.0)
    sn_sp = jnp.where(prod_split, sn, 1.0)
    d_cov_sp = (_safe_tangent_quad(dtw_sp, jnp.where(prod_split, cx, 0.0),
                                   jnp.where(prod_split, cy, 0.0), sn_sp)
                - _safe_tangent_div(dtw_sp, jnp.where(prod_split, cov_term, 0.0), sn_sp))
    d_cov_w = jnp.where(prod_split, d_cov_sp, d_cov_no)
    d_cov_raw = jnp.sum(pw_cy * dtx + pw_cx * dty + d_cov_w, axis=-1)
    return d_mx, d_my, d_cov_raw, jnp.sum(pw_cy, axis=-1), jnp.sum(pw_cx, axis=-1), jnp.sum(dtw, axis=-1)


def _tile_array(value: jax.Array, shape: tuple[int, ...], tile: int) -> jax.Array:
    n_tiles = shape[-1] // tile
    return jnp.moveaxis(jnp.broadcast_to(value, shape).reshape(*shape[:-1], n_tiles, tile), -2, 0)


def _tiled_tangents(
    x: jax.Array, y: jax.Array, w: jax.Array,
    dx: jax.Array, dy: jax.Array, dw: jax.Array,
    mu_x: jax.Array, mu_y: jax.Array, cov: jax.Array, n: jax.Array,
    tile: int, mode: Literal["walk", "merge"],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    shape = tuple(int(size) for size in jnp.broadcast_shapes(x.shape, y.shape, w.shape))
    tx_s = _tile_array(x, shape, tile)
    ty_s = _tile_array(y, shape, tile)
    tw_s = _tile_array(w, shape, tile)
    dtx_s = _tile_array(dx, shape, tile)
    dty_s = _tile_array(dy, shape, tile)
    dtw_s = _tile_array(dw, shape, tile)
    if mode == "walk":
        zeros = (jnp.zeros_like(mu_x), jnp.zeros_like(mu_y), jnp.zeros_like(cov),
                 jnp.zeros_like(mu_y), jnp.zeros_like(mu_x), jnp.zeros_like(n))

        def step(acc, chunk):
            tx, ty, tw, dtx, dty, dtw = chunk
            parts = _cov_tangent_tile(tx, ty, tw, dtx, dty, dtw, mu_x, mu_y, cov, n)
            return tuple(prev + delta for prev, delta in zip(acc, parts, strict=True)), None

        d_mx, d_my, d_cov_raw, sum_pw_cy, sum_pw_cx, d_n = jax.lax.scan(
            step, zeros, (tx_s, ty_s, tw_s, dtx_s, dty_s, dtw_s))[0]
    else:
        d_mx_s, d_my_s, d_cov_s, sum_cy_s, sum_cx_s, d_n_s = jax.vmap(
            lambda tx, ty, tw, dtx, dty, dtw: _cov_tangent_tile(
                tx, ty, tw, dtx, dty, dtw, mu_x, mu_y, cov, n)
        )(tx_s, ty_s, tw_s, dtx_s, dty_s, dtw_s)
        d_mx, d_my, d_cov_raw = jnp.sum(d_mx_s, axis=0), jnp.sum(d_my_s, axis=0), jnp.sum(d_cov_s, axis=0)
        sum_pw_cy, sum_pw_cx, d_n = (jnp.sum(sum_cy_s, axis=0), jnp.sum(sum_cx_s, axis=0),
                                     jnp.sum(d_n_s, axis=0))
    d_cov = d_cov_raw - d_mx * sum_pw_cy - d_my * sum_pw_cx
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
