"""Central-moment declarations and algebra-level tiled consumer (`dew.algebra.derive`, docs/design/algebra.md
§1.4, §6.9): weighted and masked mean/variance and bilinear covariance walks derived through
Chan-Golub-LeVeque merge rules, producing finite variance and covariance across arbitrarily shifted data and
disjoint partials."""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import Literal

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero

from dew.algebra import derive, program, ratios, windows
from dew.algebra.forms import Derived
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


_MEAN_FORM = ratios.Ratio(
    ((1.0, (("v", 0), ("v", 1))),), ((1.0, (("v", 1),)),), 0, reduced=True, units=windows.Units())
"""Streamed weighted mean `sum(w * r) / sum(w)` along axis 0 (`_streamed_tangents`)."""

_COV_FORM = ratios.Ratio(
    ((1.0, (("v", 0), ("v", 1), ("v", 2))),), ((1.0, (("v", 2),)),), 0, reduced=True, units=windows.Units())
"""Streamed weighted covariance `sum(w * cx * cy) / sum(w)` along axis 0 (`_streamed_tangents`)."""

_MEAN_ZERO = ratios.Ratio(
    ((1.0, (("v", 0), ("v", 1))),), (), 0, reduced=True, units=windows.Units())
"""Streamed `sum(x * dw)` along axis 0 at the all-zero-weight (`n == 0`) boundary."""

_COV_ZERO = ratios.Ratio(
    ((1.0, (("v", 0), ("v", 1), ("v", 2))),), (), 0, reduced=True, units=windows.Units())
"""Streamed `sum(x * y * dw)` along axis 0 at the all-zero-weight (`n == 0`) boundary."""


def _ratio_jvp_only(primals: Sequence[jax.Array], tangents: Sequence,
                    form: ratios.Ratio) -> jax.Array | None:
    """Exact tangent of `ratios.ratio(primals, form)` along non-`SymbolicZero` `tangents`, or None if none
    move: every moving value and weight contribution cancels unlanded inside one `Ratio`, and omitting
    `SymbolicZero` operands keeps partial reverse-mode transposes strictly linear."""
    moving = [index for index, tangent in enumerate(tangents) if not isinstance(tangent, SymbolicZero)]
    if not moving:
        return None
    active_set = frozenset(moving)

    def evaluate(*active: jax.Array) -> jax.Array:
        iterator = iter(active)
        full = [next(iterator) if index in active_set else primal for index, primal in enumerate(primals)]
        return ratios.ratio(full, form)

    _, tangent = jax.jvp(evaluate, tuple(primals[index] for index in moving),
                         tuple(tangents[index] for index in moving))
    return tangent


def _move_tangent(tangent, shape: tuple[int, ...]):
    """Broadcast a live tangent to `shape` and move its walked axis to 0, leaving `SymbolicZero` intact."""
    if isinstance(tangent, SymbolicZero):
        return tangent
    return jnp.moveaxis(jnp.broadcast_to(tangent, shape), -1, 0)


def _mask_tangent(mask: jax.Array, tangent):
    """Zero a live tangent outside `mask`, leaving `SymbolicZero` intact."""
    return tangent if isinstance(tangent, SymbolicZero) else jnp.where(mask, tangent, 0.0)


def _centered_tangent(dv, d_mean: jax.Array | None, shape: tuple[int, ...]):
    """Centered tangent `dv - d_mean[None]` along axis 0, or `SymbolicZero` when neither moves."""
    if isinstance(dv, SymbolicZero):
        return dv if d_mean is None else jnp.broadcast_to(-d_mean[None], (shape[-1], *shape[:-1]))
    return dv if d_mean is None else dv - d_mean[None]


def _streamed_tangents(
    x: jax.Array, y: jax.Array, w: jax.Array,
    dx, dy, dw,
    mu_x: jax.Array, mu_y: jax.Array, cov: jax.Array, n: jax.Array,
) -> tuple:
    """Exact streamed tangents `(d_mx, d_my, d_cov, d_n)` sharing one broadcast/centering pass and one
    unlanded `ratios.Ratio` per output, handling both `n > 0` and `n == 0` boundaries and `SymbolicZero`."""
    shape = tuple(int(size) for size in jnp.broadcast_shapes(x.shape, y.shape, w.shape))
    x0 = jnp.moveaxis(jnp.broadcast_to(x, shape), -1, 0)
    y0 = jnp.moveaxis(jnp.broadcast_to(y, shape), -1, 0)
    w0 = jnp.moveaxis(jnp.broadcast_to(w, shape), -1, 0)
    dx0 = _move_tangent(dx, shape)
    dy0 = _move_tangent(dy, shape)
    dw0 = _move_tangent(dw, shape)
    pos = n > 0
    pos0 = pos[None]
    unit0 = jnp.zeros((shape[-1],) + (1,) * (len(shape) - 1), w0.dtype).at[0].set(1.0)
    w_pos = jnp.where(pos0, w0, unit0)
    dx_pos, dy_pos, dw_pos = (_mask_tangent(pos0, t) for t in (dx0, dy0, dw0))
    dw_zero = None if isinstance(dw0, SymbolicZero) else jnp.where(pos0, 0.0, dw0)

    def mean_tangent(vals0: jax.Array, d_vals_pos, mu: jax.Array) -> jax.Array | None:
        # Shift by constant `stop_gradient(mu)` so large-offset inputs do not cancel while preserving exact
        # higher-order derivatives of `sum(w * vals) / sum(w)`.
        shifted = vals0 - jax.lax.stop_gradient(mu)[None]
        d_pos = _ratio_jvp_only((shifted, w_pos), (d_vals_pos, dw_pos), _MEAN_FORM)
        if d_pos is None and dw_zero is None:
            return None
        zero_branch = 0.0 if dw_zero is None else ratios.ratio([vals0, dw_zero], _MEAN_ZERO)
        return jnp.where(pos, 0.0 if d_pos is None else d_pos, zero_branch)

    d_mx = mean_tangent(x0, dx_pos, mu_x)
    same_xy = x is y and dx is dy and mu_x is mu_y
    d_my = d_mx if same_xy else mean_tangent(y0, dy_pos, mu_y)
    cx0 = x0 - mu_x[None]
    cy0 = cx0 if same_xy else y0 - mu_y[None]
    dcx_pos = _mask_tangent(pos0, _centered_tangent(dx0, d_mx, shape))
    dcy_pos = dcx_pos if same_xy else _mask_tangent(pos0, _centered_tangent(dy0, d_my, shape))
    d_cov_pos = _ratio_jvp_only((cx0, cy0, w_pos), (dcx_pos, dcy_pos, dw_pos), _COV_FORM)
    if d_cov_pos is None and dw_zero is None:
        d_cov = SymbolicZero(jax.typeof(cov).to_tangent_aval())
    else:
        cov_zero = 0.0 if dw_zero is None else ratios.ratio([x0, y0, dw_zero], _COV_ZERO)
        d_cov = jnp.where(pos, 0.0 if d_cov_pos is None else d_cov_pos, cov_zero)
    d_mx_out = SymbolicZero(jax.typeof(mu_x).to_tangent_aval()) if d_mx is None else d_mx
    d_my_out = SymbolicZero(jax.typeof(mu_y).to_tangent_aval()) if d_my is None else d_my
    d_n_out = SymbolicZero(jax.typeof(n).to_tangent_aval()) if isinstance(dw0, SymbolicZero) else jnp.sum(
        dw0, axis=0)
    return d_mx_out, d_my_out, d_cov, d_n_out


def _chunked_weighted_primal(x: jax.Array, w: jax.Array, tile: int,
                             mode: Literal["walk", "merge"]) -> tuple[jax.Array, jax.Array, jax.Array]:
    tiled = _tiled({"x": x, "w": w}, tile)
    return _eval_derived(_derived_with_total("weighted", tiled["x"].ndim - 1), tiled, mode)


def _chunked_weighted_jvp(tile: int, mode: Literal["walk", "merge"], primals, tangents):
    x, w = primals
    dx, dw = tangents
    mean, var, n = _run_chunked_weighted(x, w, tile, mode)
    d_mean, _, d_var, d_n = _streamed_tangents(x, x, w, dx, dx, dw, mean, mean, var, n)
    return (mean, var, n), (d_mean, d_var, d_n)


_run_chunked_weighted = contracted("test_chunked_moments_holds_every_order")(
    declared_tangent(Protected(), _chunked_weighted_primal, _chunked_weighted_jvp,
                     nondiff_argnums=(2, 3), symbolic_zeros=True))


def _chunked_cov_primal(
    x: jax.Array, y: jax.Array, w: jax.Array, tile: int, mode: Literal["walk", "merge"],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    tiled = _tiled({"x": x, "y": y, "w": w}, tile)
    return _eval_derived(_derived_with_total("covariance", tiled["x"].ndim - 1), tiled, mode)


def _chunked_cov_jvp(tile: int, mode: Literal["walk", "merge"], primals, tangents):
    x, y, w = primals
    dx, dy, dw = tangents
    mean_x, mean_y, cov, n = _run_chunked_cov(x, y, w, tile, mode)
    d_mx, d_my, d_cov, d_n = _streamed_tangents(x, y, w, dx, dy, dw, mean_x, mean_y, cov, n)
    return (mean_x, mean_y, cov, n), (d_mx, d_my, d_cov, d_n)


_run_chunked_cov = contracted("test_chunked_moments_holds_every_order")(
    declared_tangent(Protected(), _chunked_cov_primal, _chunked_cov_jvp,
                     nondiff_argnums=(3, 4), symbolic_zeros=True))


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
