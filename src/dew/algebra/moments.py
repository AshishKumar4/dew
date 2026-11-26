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
from dew.nn.float_bits import stored_zero
from dew.nn.programs import holes

SATURATION = program.Saturation(growth=64, depth=4)
"""Derivation limits for central-moment walks."""


def _present(w: jax.Array, *values: jax.Array) -> jax.Array:
    """Keep a finite observation even at zero weight, for its one-sided derivative."""
    zero = stored_zero(w) if jnp.issubdtype(w.dtype, jnp.floating) else w == 0
    nonfinite = functools.reduce(jnp.logical_or, (~jnp.isfinite(value) for value in values))
    return ~(zero & nonfinite)


def _observed(present: jax.Array, *values: jax.Array) -> tuple[jax.Array, ...]:
    return tuple(jnp.where(present, value, jnp.zeros_like(value)) for value in values)


def _denominator(w: jax.Array, kept: jax.Array, present: jax.Array) -> jax.Array:
    # Broadcasting one weight across the reduction still counts it only once in a direct call.
    if w.shape[-1] != kept.shape[-1]:
        kept, = _observed(jnp.any(present, axis=-1, keepdims=True), w)
    count = jnp.sum(kept, axis=-1)
    return jnp.where(count > 0, count, jnp.ones_like(count))


def _mean(x, w, denominator):
    return jnp.sum(w * x, axis=-1) / denominator


def _variance(x, w, mean, denominator):
    diff = x - mean[..., None]
    return jnp.sum(w * diff * diff, axis=-1) / denominator


def _weighted_moments_with_total(x: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Weighted `(mean, variance, total_weight)` of `x` along its last axis under nonnegative `w`."""
    n = jnp.sum(w, axis=-1)
    safe_n = jnp.where(n > 0, n, jnp.ones_like(n))
    mean = _mean(x, w, safe_n)
    return mean, _variance(x, w, mean, safe_n), n


def weighted_moments(x: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Weighted `(mean, variance)` of `x` along its last axis under nonnegative weights `w`, zero where
    total weight is zero."""
    present = _present(w, x)
    x, kept = _observed(present, x, w)
    denominator = _denominator(w, kept, present)
    mean = _mean(x, kept, denominator)
    return mean, _variance(x, kept, mean, denominator)


def masked_moments(x: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Masked unweighted `(mean, variance)` of `x` along its last axis over boolean `mask`, zero where
    every entry is masked out."""
    n = jnp.sum(mask.astype(x.dtype), axis=-1)
    safe_n = jnp.where(n > 0, n, jnp.ones_like(n))
    mean = jnp.sum(jnp.where(mask, x, jnp.zeros_like(x)), axis=-1) / safe_n
    # Mask before any nonlinear operation can make its pullback multiply an inactive infinity.
    diff = jnp.where(mask, x - mean[..., None], jnp.zeros_like(x))
    m2 = jnp.sum(jax.lax.square(diff), axis=-1)
    return mean, m2 / safe_n


def _covariance(x, y, w, mean_x, mean_y, denominator):
    dx = x - mean_x[..., None]
    dy = y - mean_y[..., None]
    return jnp.sum(w * dx * dy, axis=-1) / denominator


def _weighted_cov_with_total(
    x: jax.Array, y: jax.Array, w: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Weighted `(mean_x, mean_y, covariance, total_weight)` of `(x, y)` along their last axis."""
    n = jnp.sum(w, axis=-1)
    safe_n = jnp.where(n > 0, n, jnp.ones_like(n))
    mean_x, mean_y = _mean(x, w, safe_n), _mean(y, w, safe_n)
    return mean_x, mean_y, _covariance(x, y, w, mean_x, mean_y, safe_n), n


def weighted_covariance(x: jax.Array, y: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Weighted `(mean_x, mean_y, covariance)` of `(x, y)` along their last axis under nonnegative `w`.

    A mean whose operand broadcasts with the weights alone keeps that shape and its marginal mask.
    The covariance excludes a missing pair, including its weight direction.
    """
    present = _present(w, x, y)
    paired_x, paired_y, kept = _observed(present, x, y, w)
    denominator = _denominator(w, kept, present)

    def marginal(value, paired):
        if jnp.broadcast_shapes(value.shape, w.shape) == paired.shape:
            return _mean(paired, kept, denominator)
        live = _present(w, value)
        observed, weights = _observed(live, value, w)
        return _mean(observed, weights, _denominator(w, weights, live))

    mean_x, mean_y = marginal(x, paired_x), marginal(y, paired_y)
    return mean_x, mean_y, _covariance(paired_x, paired_y, kept, mean_x, mean_y, denominator)


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


def _weighted_sum(*indices: int) -> ratios.Summed:
    return ratios.Summed(((1.0, tuple(("v", index) for index in indices)),), reduced=True)


def _safe_mass(weight: int) -> ratios.Summed:
    # Keep each appearance of the weight as the same Ratio operand so its transpose cancels before landing.
    return ratios.Summed((*_weighted_sum(weight, weight + 1).terms, (1.0, (("v", weight + 2),))),
                         reduced=True)


_MEAN_FORM = ratios.Ratio(_weighted_sum(0, 1).terms, _safe_mass(1).terms, 0,
                         reduced=True, units=windows.Units(), track_directions=True)


def _covariance_form(second: int, weight: int) -> ratios.Ratio:
    """Covariance about a fixed anchor: P/D - R Sx Sy/D³, where P sums w cx cy, Sx and Sy sum the
    weighted centers, D is the safe mass and R = 2D - sum(w). For positive mass R = D, so this is the
    usual centered covariance; at zero mass D = 1 and this is the declaration's denominator-one polynomial.
    One form holds both cases and all correction products until their sum lands."""
    mass = _safe_mass(weight)
    correction = ratios.Summed((*_weighted_sum(weight, weight + 3).terms,
                                (2.0, (("v", weight + 2),))), reduced=True)
    joint, sx, sy = _weighted_sum(0, second, weight), _weighted_sum(0, weight), _weighted_sum(second, weight)
    return ratios.Ratio((), mass.terms, 0, reduced=True, units=windows.Units(), power=3,
                        products=((1.0, (joint, mass, mass)), (-1.0, (sx, sy, correction))),
                        track_directions=True)


_COV_FORM = _covariance_form(1, 2)
_VAR_FORM = _covariance_form(0, 1)


def _ratio_jvp_only(primals: Sequence[jax.Array], tangents: Sequence,
                    form: ratios.Ratio, moving: tuple[int, ...]) -> jax.Array:
    """Take a ratio's derivative only in the moving operands, keeping its transpose linear."""
    live = frozenset(moving)
    operands, fill = holes(list(primals), [index in live for index in range(len(primals))])

    def evaluate(*active):
        return ratios.ratio(fill(active), form)

    _, tangent = jax.jvp(evaluate, tuple(operands), tuple(tangents[index] for index in moving))
    return tangent


def _move_tangent(tangent, shape: tuple[int, ...]):
    """Broadcast a live tangent to `shape` and move its walked axis to 0, leaving `SymbolicZero` intact."""
    if isinstance(tangent, SymbolicZero):
        return tangent
    return jnp.moveaxis(jnp.broadcast_to(tangent, shape), -1, 0)


def _streamed_tangents(
    x: jax.Array, y: jax.Array, w: jax.Array,
    dx, dy, dw,
    mu_x: jax.Array, mu_y: jax.Array, cov: jax.Array, n: jax.Array,
) -> tuple:
    """Exact streamed tangents `(d_mx, d_my, d_cov, d_n)` sharing one broadcast/centering pass and one
    unlanded `ratios.Ratio` per output, handling both `n > 0` and `n == 0` boundaries and `SymbolicZero`."""
    shape = tuple(int(size) for size in jnp.broadcast_shapes(x.shape, y.shape, w.shape))
    x0 = jnp.moveaxis(jnp.broadcast_to(x, shape), -1, 0)
    same_xy = x is y and dx is dy and mu_x is mu_y
    y0 = x0 if same_xy else jnp.moveaxis(jnp.broadcast_to(y, shape), -1, 0)
    w0 = jnp.moveaxis(jnp.broadcast_to(w, shape), -1, 0)
    dx0 = _move_tangent(dx, shape)
    dy0 = dx0 if same_xy else _move_tangent(dy, shape)
    dw0 = _move_tangent(dw, shape)
    pos = (n > 0)[None]
    unit0 = (jnp.arange(shape[-1]) == 0).reshape((shape[-1],) + (1,) * (len(shape) - 1)).astype(w0.dtype)
    present = pos.astype(w0.dtype)
    unit = jnp.where(pos, 0.0, unit0)
    sign = jnp.where(pos, jnp.ones_like(present), -jnp.ones_like(present))
    held = isinstance(dw0, SymbolicZero)
    fixed = tuple(SymbolicZero(jax.typeof(value).to_tangent_aval()) for value in (present, unit, sign))

    # A fixed anchor preserves translation invariance at every derivative order, without subtracting a
    # rounded mean tangent from each sample. A row that would overflow with this anchor uses zero instead.
    def centered(vals0: jax.Array, mu: jax.Array) -> jax.Array:
        anchor = jax.lax.stop_gradient(mu)[None]
        shifted = vals0 - anchor
        return jnp.where(jnp.any(jnp.isinf(shifted), axis=0, keepdims=True), vals0, shifted)

    def tangent(out: jax.Array, primals: tuple, directions: tuple, form: ratios.Ratio):
        moving = tuple(i for i, direction in enumerate(directions) if not isinstance(direction, SymbolicZero))
        if not moving:
            return SymbolicZero(jax.typeof(out).to_tangent_aval())
        return _ratio_jvp_only(primals, directions, form, moving)

    cx0 = centered(x0, mu_x)
    if same_xy:
        d_mx = d_my = tangent(mu_x, (cx0, w0, present, unit), (dx0, dw0, *fixed[:2]), _MEAN_FORM)
        d_cov = tangent(cov, (cx0, w0, present, unit, sign), (dx0, dw0, *fixed), _VAR_FORM)
    else:
        cy0 = centered(y0, mu_y)
        if not isinstance(dx0, SymbolicZero) and not isinstance(dy0, SymbolicZero):
            d_columns = jnp.stack((dx0, dy0), axis=-1)
        else:
            d_columns = None
        shared = d_columns is not None or (not held and isinstance(dx0, SymbolicZero)
                                           and isinstance(dy0, SymbolicZero))
        if shared:
            # The same linear form makes both means with shared weights, one column per mean.
            centers = jnp.stack((cx0, cy0), axis=-1)
            d_centers = (SymbolicZero(jax.typeof(centers).to_tangent_aval())
                         if d_columns is None else d_columns)
            weight = dw0 if held else dw0[..., None]
            moving = (1,) if d_columns is None else (0,) if held else (0, 1)
            means = _ratio_jvp_only((centers, w0[..., None], present[..., None], unit[..., None]),
                                    (d_centers, weight, *fixed[:2]), _MEAN_FORM, moving)
            d_mx, d_my = means[..., 0], means[..., 1]
        else:
            d_mx = tangent(mu_x, (cx0, w0, present, unit), (dx0, dw0, *fixed[:2]), _MEAN_FORM)
            d_my = tangent(mu_y, (cy0, w0, present, unit), (dy0, dw0, *fixed[:2]), _MEAN_FORM)
        d_cov = tangent(cov, (cx0, cy0, w0, present, unit, sign), (dx0, dy0, dw0, *fixed), _COV_FORM)
    d_n_out = SymbolicZero(jax.typeof(n).to_tangent_aval()) if held else jnp.sum(dw0, axis=0)
    return d_mx, d_my, d_cov, d_n_out


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


_run_chunked_weighted = contracted("test_the_weighted_moments_walk_holds_every_order")(
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


_run_chunked_cov = contracted("test_the_covariance_merge_holds_every_order")(
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
        if not tiled["x"].size:
            return _cast_back(masked_moments(tiled["x"][0], tiled["mask"][0]), work, out_dtype)
        finished = _eval_derived(derived("masked", tiled["x"].ndim - 1), tiled, mode)
        return _cast_back(finished, work, out_dtype)
    out_dtype = _floating_dtype(x_arr, w_arr)
    work = _working_dtype(out_dtype)
    x_arr, w_arr = _observed(_present(w_arr, x_arr), x_arr, w_arr)
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
    x_arr, y_arr, w_arr = _observed(_present(w_arr, x_arr, y_arr), x_arr, y_arr, w_arr)
    return _cast_back(_run_chunked_cov(x_arr.astype(work), y_arr.astype(work), w_arr.astype(work), tile,
                                       mode)[:3], work, out_dtype)
