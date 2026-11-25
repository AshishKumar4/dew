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


def _weighted_sum(*indices: int) -> ratios.Summed:
    return ratios.Summed(((1.0, tuple(("v", index) for index in indices)),), reduced=True)


_MEAN_FORM = ratios.Ratio(_weighted_sum(0, 1).terms, _weighted_sum(2).terms, 0,
                         reduced=True, units=windows.Units())


def _covariance_form(second: int, weight: int) -> ratios.Ratio:
    """Covariance about a fixed anchor: P/D - R Sx Sy/D³, where P sums w cx cy, Sx and Sy sum the
    weighted centers, D is the safe mass and R = 2D - sum(w). For positive mass R = D, so this is the
    usual centered covariance; at zero mass D = 1 and this is the declaration's denominator-one polynomial.
    One form holds both cases and all correction products until their sum lands."""
    mass, correction = _weighted_sum(weight + 1), _weighted_sum(weight + 2)
    joint, sx, sy = _weighted_sum(0, second, weight), _weighted_sum(0, weight), _weighted_sum(second, weight)
    return ratios.Ratio((), mass.terms, 0, reduced=True, units=windows.Units(), power=3,
                        products=((1.0, (joint, mass, mass)), (-1.0, (sx, sy, correction))))


_COV_FORM = _covariance_form(1, 2)
_VAR_FORM = _covariance_form(0, 1)


def _ratio_jvp_only(primals: Sequence[jax.Array], tangents: Sequence,
                    form: ratios.Ratio, moving: tuple[int, ...]) -> jax.Array:
    """Take a ratio's derivative only in the moving operands, keeping its transpose linear."""
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
    safe = jnp.where(pos, w0, unit0)
    # Form the two branches independently: computing 2D - T directly could overflow where D == T.
    correction = jnp.where(pos, w0, 2 * unit0 - w0)
    held = isinstance(dw0, SymbolicZero)
    dsafe = dw0 if held else jnp.where(pos, dw0, 0.0)
    dcorrection = dw0 if held else jnp.where(pos, dw0, -dw0)

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
    d_mx = tangent(mu_x, (cx0, w0, safe), (dx0, dw0, dsafe), _MEAN_FORM)
    if same_xy:
        d_my = d_mx
        d_cov = tangent(cov, (cx0, w0, safe, correction), (dx0, dw0, dsafe, dcorrection), _VAR_FORM)
    else:
        cy0 = centered(y0, mu_y)
        d_my = tangent(mu_y, (cy0, w0, safe), (dy0, dw0, dsafe), _MEAN_FORM)
        d_cov = tangent(cov, (cx0, cy0, w0, safe, correction),
                        (dx0, dy0, dw0, dsafe, dcorrection), _COV_FORM)
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
