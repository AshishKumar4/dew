"""Batch-dimension folding for `gemm._gemm` and `gemm._gemm_t` via `jax.custom_batching.custom_vmap`
(kernel-templates.md Appendix C, row 6104):
- `[B, M, K] @ [K, N]` folds into one dense product of `[B * M, K] @ [K, N]` rows.
- `[B, M, K] @ [B, K, N]` (or `[M, K] @ [B, K, N]`) folds into one grouped product with `group_sizes=[M] * B`.
- Unfoldable cases (already-grouped calls, paired/gathered/scattered/beside/chained) vmap the portable XLA
  branch directly so `vmap` never touches `platform_dependent` / Mosaic interpreter IO-effect `cond`s."""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import jax
import jax.numpy as jnp

from dew.nn.kernels.templates.axes import GROUPS


def _with_committed(fn: Callable, committed_device: Callable, *operands):
    """Resolve concrete committed placement before `custom_vmap` traces operands into an abstract jaxpr."""
    from jax.core import Tracer

    if not any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(operands)):
        device = committed_device(operands)
        with jax.default_device(device) if device is not None else contextlib.nullcontext():
            return fn(*operands)
    return fn(*operands)


def _broadcast_rows(array: jax.Array, axis_size: int, rows_each: int, is_batched: bool) -> jax.Array:
    """Fold a `[B, M, ...]` or `[M, ...]` row-leading array into `[B * M, ...]`."""
    if is_batched:
        return array.reshape(axis_size * rows_each, *array.shape[2:])
    expanded = jnp.broadcast_to(array[None], (axis_size, *array.shape))
    return expanded.reshape(axis_size * rows_each, *array.shape[1:])


def _fold_sides(arrays: tuple, axes_of: tuple, batched_flags, axis_size: int, rows_each: int
                ) -> tuple[tuple, tuple, bool] | None:
    """Fold dense side operands across a leading batch of size `axis_size`.

    Returns `(folded_arrays, folded_axes, needs_grouped)` or `None` if an axis ordering cannot be folded."""
    flags = tuple(batched_flags) if arrays else ()
    folded_arrays = []
    folded_axes = []
    needs_grouped = False
    for array, axes, is_batched in zip(arrays, axes_of, flags, strict=True):
        if 'rows' in axes:
            if axes[0] != 'rows':
                return None
            folded_arrays.append(_broadcast_rows(array, axis_size, rows_each, bool(is_batched)))
            folded_axes.append(axes)
        elif is_batched:
            needs_grouped = True
            folded_arrays.append(array)
            folded_axes.append((GROUPS, *axes))
        else:
            folded_arrays.append(array)
            folded_axes.append(axes)
    return tuple(folded_arrays), tuple(folded_axes), needs_grouped


def batched_gemm(dispatch_fn: Callable, run_gemm: Callable, portable_fn: Callable, *, dense: bool,
                 prologue_axes: tuple, epilogue_axes: tuple,
                 committed_device: Callable | None = None) -> Callable:
    """Wrap `_gemm`'s dispatched call with `custom_vmap` so dense `vmap` folds into one dense or grouped
    GEMM launch (row 6104)."""

    @jax.custom_batching.custom_vmap
    def dispatched(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, first, gather, scatter,
                   joined, beside, chained):
        return dispatch_fn(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, first, gather,
                           scatter, joined, beside, chained)

    @dispatched.def_vmap
    def vmap_rule(axis_size: int, in_batched, lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays,
                  paired, first, gather, scatter, joined, beside, chained):
        args = (lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, first, gather, scatter,
                joined, beside, chained)
        if not any(jax.tree.leaves(in_batched)):
            plain_out = dispatch_fn(*args)
            return plain_out, jax.tree.map(lambda _: False, plain_out)
        lhs_b, rhs_b, sizes_b, pro_b, epi_b, *_ = in_batched
        can_fold = (dense and not sizes_b and paired is None and first is None and gather is None
                    and scatter is None and joined is None and not beside and not chained)
        if can_fold:
            rows_each = lhs.shape[1] if lhs_b else lhs.shape[0]
            folded_pro = _fold_sides(prologue_arrays, prologue_axes, pro_b, axis_size, rows_each)
            folded_epi = _fold_sides(epilogue_arrays, epilogue_axes, epi_b, axis_size, rows_each)
            if folded_pro is not None and folded_epi is not None:
                pro_arrs, pro_ax, pro_grouped = folded_pro
                epi_arrs, epi_ax, epi_grouped = folded_epi
                grouped = bool(rhs_b) or pro_grouped or epi_grouped
                lhs_flat = _broadcast_rows(lhs, axis_size, rows_each, bool(lhs_b))
                if grouped:
                    rhs_in = rhs[:, 0] if rhs_b else jnp.broadcast_to(rhs, (axis_size, *rhs.shape[1:]))
                    sizes_in = jnp.full((axis_size,), rows_each, jnp.int32)
                else:
                    rhs_in = rhs
                    sizes_in = jnp.asarray([axis_size * rows_each], jnp.int32)
                flat_out = run_gemm(lhs_flat, rhs_in, sizes_in, pro_arrs, pro_ax, epi_arrs, epi_ax,
                                    dense=not grouped)
                batched_out = jax.tree.map(lambda y: y.reshape(axis_size, rows_each, *y.shape[1:]), flat_out)
                return batched_out, jax.tree.map(lambda _: True, batched_out)
        in_axes = jax.tree.map(lambda batched: 0 if batched else None, tuple(in_batched))
        mapped = jax.vmap(portable_fn, in_axes=in_axes)(*args)
        return mapped, jax.tree.map(lambda _: True, mapped)

    if committed_device is None:
        return dispatched
    return lambda *args: _with_committed(dispatched, committed_device, *args)


def batched_gemm_t(dispatch_fn: Callable, run_gemm_t: Callable, portable_fn: Callable, *, dense: bool,
                   prologue_axes: tuple, committed_device: Callable | None = None) -> Callable:
    """Wrap `_gemm_t`'s dispatched call with `custom_vmap` so vmapped dense kernel gradients fold into one
    grouped `gemm_t` launch (row 6104)."""

    @jax.custom_batching.custom_vmap
    def dispatched(lhs, rhs, group_sizes, prologue_arrays, tables):
        return dispatch_fn(lhs, rhs, group_sizes, prologue_arrays, tables)

    @dispatched.def_vmap
    def vmap_rule(axis_size: int, in_batched, lhs, rhs, group_sizes, prologue_arrays, tables):
        args = (lhs, rhs, group_sizes, prologue_arrays, tables)
        if not any(jax.tree.leaves(in_batched)):
            return dispatch_fn(*args), False
        lhs_b, rhs_b, sizes_b, pro_b, _ = in_batched
        if dense and not sizes_b and all(table is None for table in tables):
            rows_each = lhs.shape[1] if lhs_b else lhs.shape[0]
            folded_pro = _fold_sides(prologue_arrays, prologue_axes, pro_b, axis_size, rows_each)
            if folded_pro is not None:
                pro_arrs, pro_ax, _ = folded_pro
                lhs_flat = _broadcast_rows(lhs, axis_size, rows_each, bool(lhs_b))
                rhs_flat = _broadcast_rows(rhs, axis_size, rows_each, bool(rhs_b))
                sizes_in = jnp.full((axis_size,), rows_each, jnp.int32)
                grouped_out = run_gemm_t(lhs_flat, rhs_flat, sizes_in, pro_arrs, pro_ax, dense=False)
                return grouped_out[:, None, :, :], True
        in_axes = jax.tree.map(lambda batched: 0 if batched else None, tuple(in_batched))
        return jax.vmap(portable_fn, in_axes=in_axes)(*args), True

    if committed_device is None:
        return dispatched
    return lambda *args: _with_committed(dispatched, committed_device, *args)
