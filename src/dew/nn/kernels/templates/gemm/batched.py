"""Batch-dimension folding for `gemm._gemm` and `gemm._gemm_t` via `jax.custom_batching.custom_vmap`
(kernel-templates.md Appendix C, row 6104):
- `[B, M, K] @ [K, N]` folds into one dense product of `[B * M, K] @ [K, N]` rows.
- `[B, M, K] @ [B, K, N]` (or `[M, K] @ [B, K, N]`) folds into one grouped product with `group_sizes=[M] * B`.
- Unfoldable cases preserve named batched dispatch (`cutlass`, `xla`), raise `selection.KernelRefused` when a
  non-batchable Pallas kernel (`mosaic`, `triton`) is explicitly forced, and vmap `portable_fn` under auto."""

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

    Normalizes singleton `('groups', ...)` axes on dense sides (squeezing `(B, 1, ...)` -> `(B, ...)` when
    mapped and preserving `(1, ...)` for `_expand_groups` when unmapped). Returns
    `(folded_arrays, folded_axes, needs_grouped)` or `None` if an axis ordering cannot be folded."""
    flags = tuple(batched_flags) if arrays else ()
    folded_arrays = []
    folded_axes = []
    needs_grouped = False
    for array, axes, is_batched in zip(arrays, axes_of, flags, strict=True):
        batched = bool(is_batched)
        if GROUPS in axes:
            g_dim = 1 if batched else 0
            if axes[0] != GROUPS or 'rows' in axes or array.shape[g_dim] != 1:
                return None
            if batched:
                needs_grouped = True
                folded_arrays.append(array[:, 0])
            else:
                folded_arrays.append(array)
            folded_axes.append(axes)
        elif 'rows' in axes:
            if axes[0] != 'rows':
                return None
            folded_arrays.append(_broadcast_rows(array, axis_size, rows_each, batched))
            folded_axes.append(axes)
        elif batched:
            needs_grouped = True
            folded_arrays.append(array)
            folded_axes.append((GROUPS, *axes))
        else:
            folded_arrays.append(array)
            folded_axes.append(axes)
    return tuple(folded_arrays), tuple(folded_axes), needs_grouped


def _expand_groups(arrays: tuple, axes_of: tuple, axis_size: int) -> tuple:
    """Broadcast unmapped singleton `('groups', ...)` sides `(1, ...)` to `(axis_size, ...)` when folded into
    a grouped call of `axis_size` groups."""
    return tuple(
        jnp.broadcast_to(array, (axis_size, *array.shape[1:]))
        if (GROUPS in axes and array.shape[0] == 1) else array
        for array, axes in zip(arrays, axes_of, strict=True))


def _unfolded_vmap(dispatch_fn: Callable, portable_fn: Callable, in_batched, args: tuple, *,
                   interpret_on_cpu: bool = False):
    """Execute an unfoldable vmapped GEMM preserving named dispatch:
    - Named `cutlass` / `xla` and native (non-interpreted) `triton` run through `dispatch_fn` so CUTLASS FFI
      sequential batching (row 4389), Triton `pallas_call` batching, and route selection stay active
      (raising `KernelRefused` when the named route refuses the call).
    - Non-batchable forced kernels (`mosaic`, or CPU-interpreted `triton`) raise typed
      `selection.KernelRefused` rather than silently falling back to XLA.
    - Auto selection (`forced is None`) maps `portable_fn` cleanly."""
    from dew.nn.kernels import generation, selection, xla_products

    in_axes = jax.tree.map(lambda batched: 0 if batched else None, tuple(in_batched))
    chosen = selection.forced(generation.Choice.GROUPED_MATMUL)
    batchable = (chosen in ('cutlass', 'xla')
                 or (chosen == 'triton' and not xla_products.interpreted(interpret_on_cpu)))
    if batchable:
        mapped = jax.vmap(dispatch_fn, in_axes=in_axes)(*args)
    elif chosen is not None:
        raise selection.KernelRefused(
            generation.Choice.GROUPED_MATMUL, chosen,
            selection.Refusal('batched',
                              f"unfoldable vmapped GEMM cannot lower through kernel {chosen!r}",
                              'ledger 6104'),
            iter(()))
    else:
        mapped = jax.vmap(portable_fn, in_axes=in_axes)(*args)
    return mapped, jax.tree.map(lambda _: True, mapped)


def batched_gemm(dispatch_fn: Callable, run_gemm: Callable, portable_fn: Callable, *, dense: bool,
                 prologue_axes: tuple, epilogue_axes: tuple,
                 committed_device: Callable | None = None, interpret_on_cpu: bool = False) -> Callable:
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
                    pro_arrs = _expand_groups(pro_arrs, pro_ax, axis_size)
                    epi_arrs = _expand_groups(epi_arrs, epi_ax, axis_size)
                    rhs_in = rhs[:, 0] if rhs_b else jnp.broadcast_to(rhs, (axis_size, *rhs.shape[1:]))
                    sizes_in = jnp.full((axis_size,), rows_each, jnp.int32)
                else:
                    rhs_in = rhs
                    sizes_in = jnp.asarray([axis_size * rows_each], jnp.int32)
                flat_out = run_gemm(lhs_flat, rhs_in, sizes_in, pro_arrs, pro_ax, epi_arrs, epi_ax,
                                    dense=not grouped)
                batched_out = jax.tree.map(lambda y: y.reshape(axis_size, rows_each, *y.shape[1:]), flat_out)
                return batched_out, jax.tree.map(lambda _: True, batched_out)
        return _unfolded_vmap(dispatch_fn, portable_fn, in_batched, args, interpret_on_cpu=interpret_on_cpu)

    return _wrap_dispatched(dispatched, dispatch_fn, committed_device, interpret_on_cpu)


def batched_gemm_t(dispatch_fn: Callable, run_gemm_t: Callable, portable_fn: Callable, *, dense: bool,
                   prologue_axes: tuple, committed_device: Callable | None = None,
                   interpret_on_cpu: bool = False) -> Callable:
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
                pro_arrs = _expand_groups(pro_arrs, pro_ax, axis_size)
                lhs_flat = _broadcast_rows(lhs, axis_size, rows_each, bool(lhs_b))
                rhs_flat = _broadcast_rows(rhs, axis_size, rows_each, bool(rhs_b))
                sizes_in = jnp.full((axis_size,), rows_each, jnp.int32)
                grouped_out = run_gemm_t(lhs_flat, rhs_flat, sizes_in, pro_arrs, pro_ax, dense=False)
                return grouped_out[:, None, :, :], True
        return _unfolded_vmap(dispatch_fn, portable_fn, in_batched, args, interpret_on_cpu=interpret_on_cpu)

    return _wrap_dispatched(dispatched, dispatch_fn, committed_device, interpret_on_cpu)


def _batched_tracer(value) -> bool:
    while value is not None:
        if type(value).__name__ == 'BatchTracer':
            return True
        value = getattr(value, 'primal', None)
    return False


def _wrap_dispatched(dispatched: Callable, dispatch_fn: Callable, committed_device: Callable | None,
                     interpret_on_cpu: bool) -> Callable:
    """Bypass `custom_vmap` for non-vmapped CPU-interpreted calls so inner `OrderedIOEffect` tokens lower
    inside control flow (`lax.cond`, `scan`) without JAX's `custom_vmap_call` dropping effects."""
    def call(*args):
        from dew.nn.kernels import xla_products

        if (xla_products.interpreted(interpret_on_cpu)
                and not any(_batched_tracer(leaf) for leaf in jax.tree.leaves(args))):
            return dispatch_fn(*args)
        return dispatched(*args)

    if committed_device is None:
        return call
    return lambda *args: _with_committed(call, committed_device, *args)
