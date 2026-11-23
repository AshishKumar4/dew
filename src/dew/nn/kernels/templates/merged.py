"""A split walk's slices merged where its launch wrote them: one Mosaic GPU launch on NVIDIA sm80+ CUDA GPUs
(row 9795), one Pallas Triton launch on pre-Ampere CUDA GPUs (`platform="cuda"`), and the law's own XLA path
elsewhere and under differentiation.

A decode step's walk splits a row's keys across programs (`attention_decode`), each slice's output, maximum
and denominator padded to its launch's tile of rows. The law merges them (`attention_algebra.merged_state`:
`slice_state` of each, then `finished_merged`), which in XLA took the padded arrays' slices, their selects and
the law's total and output as separate launches around it. `merged_padded` takes the launch's padded arrays as
they are; on a CUDA GPU it lowers as one Pallas launch (Mosaic GPU on sm80+, Pallas Triton on pre-Ampere CUDA
GPUs) that reads them in place and runs the law's own functions on each row (`slice_state`,
`finished_merged`), and on non-CUDA platforms and under differentiation as the law's path on their slices
(`_value`), which its derivative always is (a `jax.custom_jvp` whose rule is `jax.jvp` of `_value`): a step
faster than main's merge at every decode cell measured, its outputs bit for bit the law's (M-ET-87i).

Each program merges a row's slices, a (batch, query, key head) row's, over a block of its channels
(`_channel_block`: all of them where one block holds the row's width, else blocks of it), each forming the
law's work on the row's states (its factors, total and maximum, which no channel changes) and storing its
block's outputs, the row's statistics stored by the first block alone; a row a program formed them once a
row, where programs each of a block of channels formed them a block (four times a row at a decode step of
one row on an A100, M-ET-87 at 7598a24655f). A call with an initial state merges it beside the slices on
the law's path (`merged_split`)."""
import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import api_util, dispatch
from jax._src.interpreters import partial_eval as pe
from jax._src.lib.mlir.dialects import llvm as mgpu_llvm, scf as mgpu_scf
from jax.custom_derivatives import SymbolicZero
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import fragmented_array as mgpu_fragments, utils as mgpu_utils
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import core as jcore
from jax.extend.mlir import ir as mgpu_ir
from jax.extend.mlir.dialects import arith as mgpu_arith, vector as mgpu_vector
from jax.interpreters import batching, mlir

from dew.algebra import windows
from dew.nn import programs
from dew.nn.derivatives import KernelGlue, contracted, declared_tangent
from dew.nn.kernels import generation, launches, options, tuning
from dew.nn.kernels.templates import attention_algebra as algebra, cost, emission, mosaic, report
from dew.nn.kernels.templates.attention import Initial, Kind, Param, kept_dtype, output_hooked, ungrouped_rows

_OUTPUT_KINDS = (Kind.HEAD, Kind.QUERY, Kind.OUTPUT)


def _split_extra(extra: Sequence, output_kinds: tuple[Kind, ...]) -> tuple[tuple, tuple]:
    """Split `extra` operands into optional `(init_m, init_t, init_acc)` and `output_params` arrays."""
    n_init = len(extra) - len(output_kinds)
    assert n_init in (0, 3), f"expected 0 or 3 initial-state operands, got {n_init}"
    return tuple(extra[:n_init]), tuple(extra[n_init:])


def output_hook_refusal(output_mod: Callable | None, output_params: tuple[Param, ...]) -> str | None:
    """Refusal reason when `output_mod`/`output_params` are inadmissible for fused decode/merge output hooks,
    or None when admissible (the single admission rule shared by `attention_decode.packable` and
    `merged_finished`, row 5385)."""
    if output_params and output_mod is None:
        return "output_params require an output_mod"
    bad = [param.kind for param in output_params if param.kind not in _OUTPUT_KINDS]
    if bad:
        return (f"merged_finished supports {[k.value for k in _OUTPUT_KINDS]} output_params, "
                f"got {[k.value for k in bad]}")
    return None


def _validate_output_hook(output_mod: Callable | None, output_params: tuple[Param, ...]) -> None:
    """Validate `output_mod`/`output_params` via `output_hook_refusal`."""
    if (reason := output_hook_refusal(output_mod, output_params)) is not None:
        raise ValueError(reason)


def _trimmed(padded, stacked, *, queries: int, width: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A split walk's padded outputs `padded` `[B, S, R, H, Dv]` and stacked maxima and denominators `stacked`
    `[2, B, H, S, R]` at their first `queries` rows and `width` columns: the slices' outputs, maxima and
    denominators the law merges (`attention_algebra.merged_state`), the one reading of their layout."""
    return padded[:, :, :queries, :, :width], stacked[0, ..., :queries], stacked[1, ..., :queries]


def _value(padded, stacked, *extra, units: windows.Units, queries: int, width: int,
           raw_scale: float | None = None, out_dtype=None, out_heads: int | None = None,
           output_mod: Callable | None = None, output_kinds: tuple[Kind, ...] = ()):
    """The law's merge of a split walk's padded slices (`_trimmed`), optional raw `Initial` state, and
    optional fused `output_mod`/`output_params`, delegated directly to `algebra.merged_state`. When
    `out_dtype` and `out_heads` are given (row 7926), finishes, ungroups, applies `output_mod` in
    promoted float precision (matching `template.padded_call`), and casts to `[B, T, out_heads, width]`."""
    init_args, out_args = _split_extra(extra, output_kinds)
    initial = Initial(*init_args) if init_args else None
    state = algebra.merged_state(units, *_trimmed(padded, stacked, queries=queries, width=width),
                                 initial=initial, raw_scale=raw_scale)
    if out_dtype is None:
        return tuple(state[:3])
    assert out_heads is not None
    ungrouped = ungrouped_rows(state.acc, out_heads)
    if output_mod is not None:
        params = tuple(
            Param(arg.astype(jnp.promote_types(arg.dtype, state.acc.dtype))
                  if jnp.issubdtype(arg.dtype, jnp.floating) else arg, kind)
            for arg, kind in zip(out_args, output_kinds, strict=True))
        ungrouped = jnp.swapaxes(output_hooked(jnp.swapaxes(ungrouped, 1, 2), output_mod, params), 1, 2)
    return (ungrouped.astype(out_dtype),)


def merged_split(spec: algebra.Forward, padded, stacked, *, queries: int, width: int,
                 initial: Initial | None = None) -> algebra.Partial:
    """A split walk's padded slices merged (`_trimmed`'s layout) with the call's `initial` state where it has
    one, through `merged_padded` (one launch on a GPU, normalizing `initial` inside the launch; row 9804)."""
    return merged_padded(spec, padded, stacked, queries=queries, width=width, initial=initial)


merged_padded_p = jcore.Primitive("merged_padded")
merged_padded_p.multiple_results = True


@merged_padded_p.def_abstract_eval
def _abstract(padded, stacked, *extra, units, queries, width, raw_scale=None, out_dtype=None,
              out_heads=None, output_mod=None, output_kinds=()):
    """The merge's result types, its value's (`_value`) as JAX types it: its shapes and dtypes, and the
    operands' shardings and manual axes (`shard_map`'s varying ones) as the law's own rules carry them.
    Types made fresh dropped them, so a scan of the law's output into this merge's was refused, its manual
    axes unmatched (the base review of 9eb04fc04, F1)."""
    value = functools.partial(_value, units=units, queries=queries, width=width, raw_scale=raw_scale,
                              out_dtype=out_dtype, out_heads=out_heads, output_mod=output_mod,
                              output_kinds=output_kinds)
    operands = (padded, stacked, *extra)
    return pe.abstract_eval_fun(value, *operands,
                                debug_info=api_util.debug_info("merged_padded", value, operands, {}))
merged_padded_p.def_impl(functools.partial(dispatch.apply_primitive, merged_padded_p))
mlir.register_lowering(merged_padded_p, mlir.lower_fun(_value, multiple_results=True))


def _batched(args, dims, **params):
    """A batch of splits merged as one launch (`merged_padded_p` again, kept for a GPU's lowering): the
    batch folded into the splits' leading rows, `padded`'s `[N B, ...]`, `stacked`'s `[2, N B, ...]`, any
    initial state's `[N B, ...]` and any output hook parameters, an operand the batch does not map (JAX's
    `not_mapped`) broadcast or kept shared along it; each result unfolded. Mapped by `jax.vmap` of the law's
    path, a batched split merge ran XLA's law, not its launch (base review of aa990591f, F9)."""
    padded, stacked, *extra = args
    size = next(arg.shape[dim] for arg, dim in zip(args, dims, strict=True) if dim is not None)
    padded = (jnp.broadcast_to(padded, (size, *padded.shape)) if dims[0] is None
              else jnp.moveaxis(padded, dims[0], 0))
    stacked = (jnp.broadcast_to(stacked, (size, *stacked.shape)) if dims[1] is None
               else jnp.moveaxis(stacked, dims[1], 0))
    # The extents folded and unfolded by name, so a batch of no splits (`size` 0) keeps its rows' count.
    batch = padded.shape[1]
    output_kinds = params.get("output_kinds", ())
    init_args, out_args = _split_extra(extra, output_kinds)
    init_dims, out_dims = _split_extra(dims[2:], output_kinds)
    if any(kind is Kind.HEAD and dim is not None for kind, dim in zip(output_kinds, out_dims, strict=True)):
        # Keep mapped `Kind.HEAD` scalar per example (rewriting it to `Kind.QUERY` changes its rank to
        # `[tokens, 1]` and breaks scalar-sensitive hooks, row 5385).
        mapped = jax.vmap(functools.partial(_value, **params), in_axes=dims)(*args)
        return mapped, (0,) * len(mapped)
    folded_extra = []
    for arg, dim in zip(init_args, init_dims, strict=True):
        moved = jnp.broadcast_to(arg, (size, *arg.shape)) if dim is None else jnp.moveaxis(arg, dim, 0)
        folded_extra.append(moved.reshape(size * batch, *moved.shape[2:]))
    for arg, dim, kind in zip(out_args, out_dims, output_kinds, strict=True):
        if kind is Kind.HEAD:
            folded_extra.append(arg)
        elif kind is Kind.QUERY:
            if dim is None:
                if arg.shape[0] == 1:
                    folded_extra.append(arg)
                else:
                    folded_extra.append(jnp.broadcast_to(
                        arg, (size, *arg.shape)).reshape(size * batch, *arg.shape[1:]))
            else:
                moved = jnp.moveaxis(arg, dim, 0)
                if moved.shape[1] == 1:
                    folded_extra.append(jnp.broadcast_to(
                        moved, (size, batch, *moved.shape[2:])).reshape(size * batch, *moved.shape[2:]))
                else:
                    folded_extra.append(moved.reshape(size * batch, *moved.shape[2:]))
        else:
            moved = jnp.broadcast_to(arg, (size, *arg.shape)) if dim is None else jnp.moveaxis(arg, dim, 0)
            folded_extra.append(moved.reshape(size * batch, *moved.shape[2:]))
    folded = merged_padded_p.bind(padded.reshape(size * batch, *padded.shape[2:]),
                                  jnp.moveaxis(stacked, 0, 1).reshape(2, size * batch, *stacked.shape[3:]),
                                  *folded_extra, **params)
    return tuple(merge.reshape(size, batch, *merge.shape[1:]) for merge in folded), (0,) * len(folded)


batching.primitive_batchers[merged_padded_p] = _batched


def merged_padded(spec: algebra.Forward, padded: jax.Array, stacked: jax.Array, *, queries: int,
                  width: int, initial: Initial | None = None) -> algebra.Partial:
    """A split walk's slices as its launch wrote them, padded (`padded` `[B, S, R, H, Dv]`, its maxima and
    denominators stacked `[2, B, H, S, R]`), merged with an optional raw `Initial` state (`[B, H, Q]` and
    `[B, H, Q, Dv]`, normalized inside the launch, row 9804) into one finished state of their first `queries`
    rows and `width` columns."""
    if algebra.finished_form(spec) is not spec:
        raise ValueError("a split walk's merge takes its finished form's states (`finished_form`)")
    extra = () if initial is None else (initial.m, initial.total, initial.acc)
    raw_scale = spec.scale if (initial is not None and spec.raw) else None
    return algebra.Partial(*_merged(algebra.units_of(spec), queries, width, raw_scale, None, None,
                                    None, (), padded, stacked, *extra))


def merged_finished(spec: algebra.Forward, padded: jax.Array, stacked: jax.Array, *, queries: int,
                    width: int, out_dtype, out_heads: int, initial: Initial | None = None,
                    output_mod: Callable | None = None,
                    output_params: tuple[Param, ...] = ()) -> jax.Array:
    """`merged_padded` finished, ungrouped to `[B, queries // (out_heads // KV), out_heads, width]`, passed
    through optional fused `output_mod`/`output_params` (row 5385), and cast to `out_dtype` inside the same
    launch (row 7926: eliminates post-merge finish/transpose/hook/cast XLA fusions on a split decode step)."""
    if algebra.finished_form(spec) is not spec:
        raise ValueError("a split walk's merge takes its finished form's states (`finished_form`)")
    _validate_output_hook(output_mod, output_params)
    init_extra = () if initial is None else (initial.m, initial.total, initial.acc)
    out_extra = tuple(param.value for param in output_params)
    out_kinds = tuple(param.kind for param in output_params)
    raw_scale = spec.scale if (initial is not None and spec.raw) else None
    (out,) = _merged(algebra.units_of(spec), queries, width, raw_scale, jnp.dtype(out_dtype), out_heads,
                     output_mod, out_kinds, padded, stacked, *init_extra, *out_extra)
    return out


def _launched(units, queries, width, raw_scale, out_dtype, out_heads, output_mod, output_kinds, *operands):
    """`merged_padded_p`'s value: one launch on a GPU (`_kernel`), the law's path elsewhere."""
    return tuple(merged_padded_p.bind(*operands, units=units, queries=queries, width=width,
                                      raw_scale=raw_scale, out_dtype=out_dtype, out_heads=out_heads,
                                      output_mod=output_mod, output_kinds=output_kinds))


def _tangent(units, queries, width, raw_scale, out_dtype, out_heads, output_mod, output_kinds,
             primals, tangents):
    """The law's path's value and tangent (`jax.jvp` of `_value`) along the operands that move alone, one
    whose tangent is a symbolic zero held as the law's own rule holds it (`means._merged_state_jvp`): a
    differentiated step merges as the law does. An instantiated zero for the outputs moved the law's ratio
    along an exponential's operand, which it refuses (the base review of 9bc8a9da1, F1)."""
    value = functools.partial(_value, units=units, queries=queries, width=width, raw_scale=raw_scale,
                              out_dtype=out_dtype, out_heads=out_heads, output_mod=output_mod,
                              output_kinds=output_kinds)
    live = [not isinstance(tangent, SymbolicZero) for tangent in tangents]
    if not any(live):
        out = value(*primals)
        return out, tuple(SymbolicZero(jax.typeof(x).to_tangent_aval()) for x in out)
    perturbed, fill = programs.holes(primals, live)
    moving = tuple(tangent for tangent, is_live in zip(tangents, live, strict=True) if is_live)
    return jax.jvp(lambda *active: value(*fill(active)), tuple(perturbed), moving)


_merged = contracted("test_a_padded_merges_derivatives_are_the_laws")(
    declared_tangent(KernelGlue(), _launched, _tangent, nondiff_argnums=(0, 1, 2, 3, 4, 5, 6, 7),
                     symbolic_zeros=True))


def _chunks(slices: int, read: Callable[[jax.Array], tuple]) -> tuple[Callable, int]:
    """A row's `slices` as the law reads them a chunk at a time (`means.merged_state_chunked`), `(load,
    chunks)`: `read(start)` a chunk's outputs `[K, 1, C]`, maxima and denominators `[K, 1]` from slice
    `start`, `K` the law's width of a row of them (`windows.chunk_width`), its slots past the count held by
    none (`valid`)."""
    width = windows.chunk_width(slices)

    def load(c):
        start = mosaic.step_index(c) * width
        valid = start + jax.lax.broadcasted_iota(jnp.int32, (width, 1), 0) < slices
        return (*read(start), valid)

    return load, -(-slices // width)




THREADS = 4 * 32
"""A program's threads: Pallas Triton's default of four warps (`CompilerParams.num_warps`), the merge's."""


def _launch_for_block(units: windows.Units, padded: jax.ShapeDtypeStruct, stacked: jax.ShapeDtypeStruct,
                      queries: int, width: int, block: int, target: generation.GpuTarget,
                      has_initial: bool = False, raw_scale: float | None = None, out_dtype=None,
                      out_heads: int | None = None, family: str | None = None,
                      output_mod: Callable | None = None, output_kinds: tuple[Kind, ...] = (),
                      output_specs: tuple[jax.ShapeDtypeStruct, ...] = ()) -> tuple[cost.Work, int]:
    """Full launch work and grid size at `block` channels for the selected backend (`_launch_mosaic` when
    `family` is set, else Pallas Triton `_launch`) on the common one-window path
    (`means.merged_state_chunked`): one anchor/hull joining pass plus one window's two dependent passes
    (`_placing` and `_summing`), for a span of `3 * chunks` steps. Device traffic charges
    `min(width, read_width)` channels per slice on Mosaic GPU (predicated loads) and `read_width` on Triton
    (physical ref bounds), `width` channels for output writes, initial-state reads, and `Kind.OUTPUT` hook
    reads, scalar hook reads per block, and row-stat stores once per row (`c_block == 0`, omitted when
    `out_dtype` finishes in-kernel)."""
    batch, slices, _, heads, read_width = padded.shape
    total_slices = slices + int(has_initial)
    channels = 1 << (width - 1).bit_length()
    chunks = -(-total_slices // windows.chunk_width(total_slices))
    blocks = channels // block
    rows = batch * queries * heads
    launched = rows * blocks
    if launched == 0:
        return cost.Work(), 0
    extra_s = ((jax.ShapeDtypeStruct((batch, heads, queries), stacked.dtype),
                jax.ShapeDtypeStruct((batch, heads, queries), stacked.dtype),
                jax.ShapeDtypeStruct((batch, heads, queries, width), padded.dtype)) if has_initial else ())
    if family is not None:
        launch = functools.partial(
            _launch_mosaic, units=units, queries=queries, width=width, block=block, raw_scale=raw_scale,
            out_dtype=out_dtype, out_heads=out_heads, output_mod=output_mod, output_kinds=output_kinds,
            family=family)
    else:
        launch = functools.partial(
            _launch, units=units, queries=queries, width=width, block=block, raw_scale=raw_scale,
            out_dtype=out_dtype, out_heads=out_heads, output_mod=output_mod, output_kinds=output_kinds)
    with generation.as_target(target):
        (kernel,) = (equation for equation in programs.equations(
            jax.make_jaxpr(launch)(padded, stacked, *extra_s, *output_specs).jaxpr)
            if programs.opaque(equation))
    body = kernel.params["jaxprs"][0] if "jaxprs" in kernel.params else kernel.params["jaxpr"]
    forms = emission.forms_of(kernel.params.get("compiler_params"))
    block_roles = (emission.emitted(body, min(blocks, 2), "block", device=range(len(body.invars)),
                                    forms=forms).roles
                   if family is not None
                   else emission.emitted(body, 1, None, device=range(len(body.invars)), forms=forms).roles)

    def _role_work(role: emission.Role) -> cost.Work:
        if len(role.walks) == 3:
            joining_walk, mean_walk, _ = role.walks
        elif role.walks:
            joining_walk, mean_walk = cost.Work(), role.walks[0]
        else:
            joining_walk = mean_walk = cost.Work()
        return (role.fixed + emission.repeated(joining_walk, chunks - 1)
                + emission.repeated(mean_walk, 2 * chunks))

    passes = 3 * chunks
    padded_item, stacked_item = padded.dtype.itemsize, stacked.dtype.itemsize
    chunk_size = windows.chunk_width(total_slices)
    init_reads = int(has_initial)
    output_bytes = sum(
        rows * (width if kind is Kind.OUTPUT else blocks) * spec.dtype.itemsize
        for kind, spec in zip(output_kinds, output_specs, strict=True))
    if family is not None:
        row_work = (_role_work(block_roles[0]) + emission.repeated(_role_work(block_roles[1]), blocks - 1)
                    if blocks > 1 else _role_work(block_roles[0]))
        emitted_total = emission.repeated(row_work, rows)
        slice_load = (block_roles[0].walks[0].device_bytes / chunk_size
                      if block_roles[0].walks
                      else (min(width, read_width) / blocks) * padded_item + 2 * stacked_item)
        load_bytes = blocks * 3 * slices * slice_load
        single_writer = blocks == 1 or block_roles[0].fixed.device_bytes > block_roles[1].fixed.device_bytes
        stats_writers = 0 if out_dtype is not None else (1 if single_writer else blocks)
        out_item = padded_item if out_dtype is None else jnp.dtype(out_dtype).itemsize
        total_device_bytes = float(
            rows * (load_bytes + init_reads * (width * padded_item + blocks * 2 * stacked_item)
                    + width * out_item + stats_writers * 2 * stacked_item) + output_bytes)
    else:
        emitted_total = emission.repeated(_role_work(block_roles[0]), launched)
        if out_dtype is None:
            total_device_bytes = float(
                rows * ((3 * slices * read_width + (init_reads + 1) * width) * padded_item
                        + (blocks * (3 * slices + init_reads) + 1) * 2 * stacked_item))
        else:
            out_item = jnp.dtype(out_dtype).itemsize
            total_device_bytes = float(
                rows * (3 * slices * read_width * padded_item + init_reads * width * padded_item
                        + width * out_item + blocks * (3 * slices + init_reads) * 2 * stacked_item)
                + output_bytes)
    total = dataclasses.replace(emitted_total, device_bytes=total_device_bytes, span=float(passes),
                                kernels=1.0)
    return total, launched


def _candidate_blocks(width: int, total_slices: int) -> tuple[int, ...]:
    """Legal power-of-two channel blocks for `width` and `total_slices`, largest first."""
    channels = 1 << (width - 1).bit_length()
    blocks = tuple(block for block in (channels >> shift for shift in range(channels.bit_length()))
                   if block * windows.chunk_width(total_slices) >= THREADS)
    return blocks or (channels,)


def _compute_planned_block(units: windows.Units, padded: jax.ShapeDtypeStruct,
                           stacked: jax.ShapeDtypeStruct, queries: int, width: int,
                           target: generation.GpuTarget, has_initial: bool, raw_scale: float | None,
                           out_dtype, out_heads: int | None, output_mod: Callable | None,
                           output_kinds: tuple[Kind, ...],
                           output_specs: tuple[jax.ShapeDtypeStruct, ...]) -> tuple[int, cost.Work]:
    _, slices, _, _, _ = padded.shape
    total_slices = slices + int(has_initial)
    family = generation.family(target.generation) if _uses_mosaic(target) else None
    blocks = _candidate_blocks(width, total_slices)

    candidates = {block: _launch_for_block(
        units, padded, stacked, queries, width, block, target, has_initial, raw_scale, out_dtype, out_heads,
        family=family, output_mod=output_mod, output_kinds=output_kinds, output_specs=output_specs)
        for block in blocks}

    def seconds(block: int) -> float:
        total, launched = candidates[block]
        return cost.launch_seconds(total, launched, target)

    chosen = min(blocks, key=lambda block: (seconds(block), -block))
    return chosen, candidates[chosen][0]


@functools.cache
def _planned_block_cached(units: windows.Units, padded: jax.ShapeDtypeStruct,
                          stacked: jax.ShapeDtypeStruct, queries: int, width: int,
                          target: generation.GpuTarget, has_initial: bool, raw_scale: float | None,
                          out_dtype, out_heads: int | None, output_mod: Callable | None,
                          output_kinds: tuple[Kind, ...],
                          output_specs: tuple[jax.ShapeDtypeStruct, ...],
                          trace_context) -> tuple[int, cost.Work]:
    del trace_context
    return _compute_planned_block(units, padded, stacked, queries, width, target, has_initial, raw_scale,
                                  out_dtype, out_heads, output_mod, output_kinds, output_specs)

def _holdable_hook(hook: Callable | None) -> bool:
    """Whether `hook` (bare or `tuning.Prepared`) may be cached without retaining closures or arrays."""
    if hook is None:
        return True
    source = hook.source if isinstance(hook, tuning.Prepared) else hook
    return tuning.holdable(source)


def _planned_block(units: windows.Units, padded: jax.ShapeDtypeStruct, stacked: jax.ShapeDtypeStruct,
                   queries: int, width: int, target: generation.GpuTarget, has_initial: bool = False,
                   raw_scale: float | None = None, out_dtype=None, out_heads: int | None = None,
                   output_mod: Callable | None = None, output_kinds: tuple[Kind, ...] = (),
                   output_specs: tuple[jax.ShapeDtypeStruct, ...] = ()) -> tuple[int, cost.Work]:
    """The channels a program of `_kernel` merges and that launch's total device work (`_launch_for_block`),
    cached on `units` and the active trace context (rows 8809, 5385) when `output_mod` is holdable."""
    if not _holdable_hook(output_mod):
        return _compute_planned_block(units, padded, stacked, queries, width, target, has_initial, raw_scale,
                                      out_dtype, out_heads, output_mod, output_kinds, output_specs)
    return _planned_block_cached(units, padded, stacked, queries, width, target, has_initial, raw_scale,
                                 out_dtype, out_heads, output_mod, output_kinds, output_specs,
                                 options.trace_context())


def _channel_block(units: algebra.Forward | windows.Units, padded: jax.ShapeDtypeStruct,
                   stacked: jax.ShapeDtypeStruct, queries: int, width: int, target: generation.GpuTarget,
                   has_initial: bool = False, raw_scale: float | None = None, out_dtype=None,
                   out_heads: int | None = None) -> int:
    """The channels a program of `_kernel` merges for operands `padded` and `stacked` (`_launch`): of the
    powers of two from a row's channels down to the least whose chunk of slices holds an element a thread
    (`THREADS`), the one whose launch the cost model prices least (`_planned_block`). A program a row's whole
    channels left a decode step's few rows on a few multiprocessors (B1: 32 rows of a GH200's 132,
    merged_padded 19.2 us at 4096 keys, row 8439)."""
    return _planned_block(algebra.units_of(units), padded, stacked, queries, width, target, has_initial,
                          raw_scale, out_dtype, out_heads)[0]


def launch_work(units: algebra.Forward | windows.Units, padded: jax.ShapeDtypeStruct,
                stacked: jax.ShapeDtypeStruct, queries: int, width: int, target: generation.GpuTarget,
                has_initial: bool = False, raw_scale: float | None = None, out_dtype=None,
                out_heads: int | None = None, output_mod: Callable | None = None,
                output_kinds: tuple[Kind, ...] = (),
                output_specs: tuple[jax.ShapeDtypeStruct, ...] = ()) -> cost.Work:
    """One native `merged_padded` launch's device work at `_channel_block`'s chosen block (`_planned_block`,
    row 8779): common-path arithmetic across the grid, valid device traffic with row-stat stores counted once
    per row (row 8809), span `3 * chunks`, one kernel."""
    return _planned_block(algebra.units_of(units), padded, stacked, queries, width, target, has_initial,
                          raw_scale, out_dtype, out_heads, output_mod, output_kinds, output_specs)[1]


def _uses_mosaic(target: generation.GpuTarget) -> bool:
    """Whether `target` lowers `merged_padded` on Mosaic GPU (NVIDIA sm80+ instruction families) rather than
    the Pallas Triton fallback kept for pre-Ampere CUDA GPUs (`platform="cuda"`, row 9795)."""
    family = generation.family(target.generation)
    return family is not None and family != "tpu"


@dataclasses.dataclass(frozen=True)
class MergeConfig:
    """One Mosaic GPU merge launch's configuration: the channels a program merges."""

    block: int


@dataclasses.dataclass(frozen=True)
class MergedPadded(tuning.KernelOp):
    """A split walk's Mosaic GPU merge (`_launch_mosaic`, row 9795) as a certified launch: its channel block
    derived by `_planned_block` and certified before its program lowers (`report.certified`)."""

    HOOKS: ClassVar[tuple[str, ...]] = ("output_mod",)

    units: windows.Units = dataclasses.field(default_factory=windows.Units)
    queries: int = 1
    width: int = 1
    raw_scale: float | None = None
    out_dtype: np.dtype | None = None
    out_heads: int | None = None
    output_mod: Callable | None = None
    output_kinds: tuple[Kind, ...] = ()
    output_dtypes: tuple[str, ...] = ()
    family: str = "mma.sync"

    config_cls: ClassVar[type[MergeConfig]] = MergeConfig

    def traced(self, name: str, value: Callable | str | None) -> str | None:
        del name
        if value is None or isinstance(value, str):
            return value
        layouts = {Kind.HEAD: (), Kind.QUERY: (2, 1), Kind.OUTPUT: (2, self.width)}
        params = [jax.ShapeDtypeStruct(layouts[kind], dtype)
                  for kind, dtype in zip(self.output_kinds, self.output_dtypes, strict=True)]
        return tuning.program(value, jax.ShapeDtypeStruct((2, self.width), jnp.float32), *params)

    def _get_heuristics_config(self, ba) -> MergeConfig:
        padded, stacked = ba.arguments["padded"], ba.arguments["stacked"]
        extra = ba.arguments.get("extra", ())
        padded_s = jax.ShapeDtypeStruct(padded.shape, padded.dtype)
        stacked_s = jax.ShapeDtypeStruct(stacked.shape, stacked.dtype)
        init_args, out_args = _split_extra(extra, self.output_kinds)
        out_specs = tuple(jax.ShapeDtypeStruct(arg.shape, arg.dtype) for arg in out_args)
        block, _ = _planned_block(
            self.units, padded_s, stacked_s, self.queries, self.width, generation.gpu_target(),
            bool(init_args), self.raw_scale, self.out_dtype, self.out_heads, self.output_mod,
            self.output_kinds, out_specs)
        return MergeConfig(block)

    def remaining(self, ba) -> MergeConfig | None:
        target = generation.gpu_target()
        derived = self._get_heuristics_config(ba)
        if not report.refused(self, ba, derived, target):
            return derived
        init_args, _ = _split_extra(ba.arguments.get("extra", ()), self.output_kinds)
        total_slices = ba.arguments["padded"].shape[1] + int(bool(init_args))
        for block in _candidate_blocks(self.width, total_slices):
            candidate = MergeConfig(block)
            if not report.refused(self, ba, candidate, target):
                return candidate
        return None

    def pinnable(self, ba, config) -> bool:
        del ba
        if not isinstance(config, MergeConfig) or config.block <= 0:
            return False
        channels = 1 << (self.width - 1).bit_length()
        return (config.block & (config.block - 1)) == 0 and config.block <= channels

    def unpinnable(self, ba, config) -> str:
        del ba
        if not isinstance(config, MergeConfig):
            return f"it is no {self.config_cls.__name__}"
        channels = 1 << (self.width - 1).bit_length()
        return f"block {config.block} is not a positive power of two <= {channels}"

    def _fwd(self, padded: jax.Array, stacked: jax.Array, extra: tuple[jax.Array, ...] = (), *,
             config: MergeConfig, return_residuals: bool = False):
        del return_residuals
        return _launch_mosaic(
            padded, stacked, *extra, units=self.units, queries=self.queries, width=self.width,
            block=config.block, raw_scale=self.raw_scale, out_dtype=self.out_dtype,
            out_heads=self.out_heads, output_mod=self.output_mod, output_kinds=self.output_kinds,
            family=self.family), None


def _kernel(padded, stacked, *extra, units: windows.Units, queries: int, width: int,
            raw_scale: float | None = None, out_dtype=None, out_heads: int | None = None,
            output_mod: Callable | None = None, output_kinds: tuple[Kind, ...] = ()):
    """`_value` as one CUDA launch (`MergedPadded` on sm80+ Mosaic GPU families, `_launch` Triton fallback
    on pre-Ampere CUDA GPUs), a program a row's block of channels (`_channel_block`)."""
    batch, _, _, heads, _ = padded.shape
    if batch == 0 or queries == 0 or heads == 0:
        return _value(padded, stacked, *extra, units=units, queries=queries, width=width,
                      raw_scale=raw_scale, out_dtype=out_dtype, out_heads=out_heads,
                      output_mod=output_mod, output_kinds=output_kinds)
    init_args, out_args = _split_extra(extra, output_kinds)
    target = generation.gpu_target()
    if _uses_mosaic(target):
        dtype = None if out_dtype is None else jnp.dtype(out_dtype)
        family = generation.family(target.generation) or "mma.sync"
        out_dtypes = tuple(kept_dtype(arg.dtype) for arg in out_args)
        return MergedPadded(
            units=units, queries=queries, width=width, raw_scale=raw_scale, out_dtype=dtype,
            out_heads=out_heads, output_mod=output_mod, output_kinds=output_kinds,
            output_dtypes=out_dtypes, family=family)(padded, stacked, tuple(extra))
    padded_s, stacked_s = (jax.ShapeDtypeStruct(operand.shape, operand.dtype)
                           for operand in (padded, stacked))
    out_specs = tuple(jax.ShapeDtypeStruct(arg.shape, arg.dtype) for arg in out_args)
    block, _ = _planned_block(units, padded_s, stacked_s, queries, width, target, bool(init_args),
                              raw_scale, out_dtype, out_heads, output_mod, output_kinds, out_specs)
    return _launch(
        padded, stacked, *extra, units=units, queries=queries, width=width, block=block,
        raw_scale=raw_scale, out_dtype=out_dtype, out_heads=out_heads, output_mod=output_mod,
        output_kinds=output_kinds)


def _tile_layout(tile_block: int):
    """Thread-local slice axis, column-sharded channel axis (`tile_block >= 64` a power of two): zero SMEM
    and zero warp shuffles for every reduction and halving tree along axis 0 (row 9795)."""
    warps = mgpu_fragments.WARPS_IN_WARPGROUP
    lanes = mgpu_fragments.WARP_SIZE
    if tile_block >= THREADS:
        tiling = mgpu_fragments.Tiling(((1, THREADS), (lanes,), (1,)))
        return plgpu.Layout.TILED(tiling, warp_dims=(-3,), lane_dims=(-2,), vector_dim=-1)
    tiling = mgpu_fragments.Tiling(((1, tile_block), (lanes,), (1,)))
    return plgpu.Layout.TILED(
        tiling, warp_dims=(mgpu_fragments.Replicated(warps // (tile_block // lanes)), -3),
        lane_dims=(-2,), vector_dim=-1)

def _tile_1d_layout(tile_layout):
    """1D channel-sharded layout derived from `_tile_layout(block)` by reducing its leading slice axis."""
    reduced = tile_layout.to_mgpu().reduce((0,))
    return plgpu.Layout.TILED(reduced.tiling, warp_dims=reduced.warp_dims,
                              lane_dims=reduced.lane_dims, vector_dim=reduced.vector_dim)

def _gmem_ptr(ref):
    """Global-memory address-space-1 pointer to `ref`'s base."""
    return mgpu_llvm.addrspacecast(mgpu_llvm.PointerType.get(address_space=1), mgpu_utils.memref_ptr(ref))


def _i64_offset(indices: Sequence, strides: Sequence[int]):
    """64-bit element offset `sum(index * stride)` from i32 MLIR indices and static integer strides."""
    i64 = mgpu_ir.IntegerType.get_signless(64)
    total = mgpu_utils.c(0, i64)
    for index, stride in zip(indices, strides, strict=True):
        term = mgpu_arith.muli(mgpu_arith.extsi(i64, index), mgpu_utils.c(stride, i64))
        total = mgpu_arith.addi(total, term)
    return total


def _predicated_load(base, offset, cond, element, zero):
    """Predicated global load returning `zero` when `cond` is false, avoiding out-of-bounds memory access."""
    branch = mgpu_scf.IfOp(cond, [element], has_else=True)
    with mgpu_ir.InsertionPoint(branch.then_block):
        ptr = mgpu_utils.getelementptr(base, [offset], element)
        mgpu_scf.YieldOp([mgpu_llvm.load(element, ptr)])
    assert branch.else_block is not None
    with mgpu_ir.InsertionPoint(branch.else_block):
        mgpu_scf.YieldOp([zero])
    return branch.results[0]


def _thread_col(block: int):
    """Warpgroup thread index `tid` and its unique column offset within `min(block, THREADS)`."""
    i32 = mgpu_ir.IntegerType.get_signless(32)
    unique = min(block, THREADS)
    tid = mgpu_arith.remui(mgpu_utils.thread_idx(), mgpu_utils.c(THREADS, i32))
    col = tid if unique == THREADS else mgpu_arith.remui(tid, mgpu_utils.c(unique, i32))
    return tid, col, unique


def _mgpu_load_chunk(padded_ref, stacked_ref, b, q, h, start, c_start, col_ok, *, chunk: int,
                     tile_block: int, slices: int, read_width: int, blocks: int = 1, tile_layout, tile_1d):
    """Predicated coalesced GMEM load of one `[chunk, tile_block]` output chunk (gated by `col_ok` and
    `sl < slices`) and its replicated `[chunk, tile_block]` maxima and totals (row 9795)."""
    scalar = plgpu.Layout.WG_SPLAT(())
    out_ty = plgpu.ShapeDtypeStruct((chunk, tile_block), padded_ref.dtype, layout=tile_layout)
    stat_ty = plgpu.ShapeDtypeStruct((chunk, tile_block), stacked_ref.dtype, layout=tile_layout)

    @emission.stating(lambda p, s, *_: cost.Work(
        device_bytes=chunk * ((read_width / blocks) * p.dtype.itemsize + 2 * s.dtype.itemsize)))
    def body(ctx, p_ref, s_ref, b_v, q_v, h_v, start_v, c_start_v, col_ok_v):
        del ctx
        pos0 = mgpu_ir.DenseI64ArrayAttr.get([0])
        i32, i64 = mgpu_ir.IntegerType.get_signless(32), mgpu_ir.IntegerType.get_signless(64)
        b_i, q_i, h_i = (v.registers.item() for v in (b_v, q_v, h_v))
        st_i, cs_i = start_v.registers.item(), c_start_v.registers.item()
        _, tid_col, unique = _thread_col(tile_block)
        reps = tile_block // unique
        col_ok_regs = col_ok_v.registers.reshape(reps)

        p_elem = mgpu_ir.MemRefType(p_ref.type).element_type
        p_strides, _ = mgpu_ir.MemRefType(p_ref.type).get_strides_and_offset()
        p_base = _gmem_ptr(p_ref)
        p_row = _i64_offset((b_i, q_i, h_i), (p_strides[0], p_strides[2], p_strides[3]))
        p_zero = mgpu_arith.constant(p_elem, 0.0)
        p_vec = mgpu_ir.VectorType.get((1,), p_elem)
        p_mgpu = tile_layout.to_mgpu()
        p_regs = np.empty(p_mgpu.registers_shape((chunk, tile_block)), dtype=object)

        s_elem = mgpu_ir.MemRefType(s_ref.type).element_type
        s_strides, _ = mgpu_ir.MemRefType(s_ref.type).get_strides_and_offset()
        s_base = _gmem_ptr(s_ref)
        m_row = _i64_offset((b_i, h_i, q_i), (s_strides[1], s_strides[2], s_strides[4]))
        t_row = mgpu_arith.addi(m_row, mgpu_utils.c(s_strides[0], i64))
        s_zero = mgpu_arith.constant(s_elem, 0.0)
        s_vec = mgpu_ir.VectorType.get((1,), s_elem)
        m_regs = np.empty(p_mgpu.registers_shape((chunk, tile_block)), dtype=object)
        t_regs = np.empty(p_mgpu.registers_shape((chunk, tile_block)), dtype=object)
        slices_c = mgpu_utils.c(slices, i32)
        for s_idx in range(chunk):
            sl = mgpu_arith.addi(st_i, mgpu_utils.c(s_idx, i32))
            sl_ok = mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, sl, slices_c)
            sl64 = mgpu_arith.extsi(i64, sl)
            s_off = mgpu_arith.muli(sl64, mgpu_utils.c(s_strides[3], i64))
            m_val = _predicated_load(s_base, mgpu_arith.addi(m_row, s_off), sl_ok, s_elem, s_zero)
            t_val = _predicated_load(s_base, mgpu_arith.addi(t_row, s_off), sl_ok, s_elem, s_zero)
            m_regs[s_idx, ...] = mgpu_vector.from_elements(s_vec, [m_val])
            t_regs[s_idx, ...] = mgpu_vector.from_elements(s_vec, [t_val])
            p_sl_base = mgpu_arith.addi(p_row, mgpu_arith.muli(sl64, mgpu_utils.c(p_strides[1], i64)))
            for rep in range(reps):
                local_col = mgpu_arith.addi(tid_col, mgpu_utils.c(rep * unique, i32))
                col = mgpu_arith.addi(cs_i, local_col)
                col_bit = mgpu_vector.extract(col_ok_regs[rep], dynamic_position=[], static_position=pos0)
                ok = mgpu_arith.andi(sl_ok, col_bit)
                col_off = mgpu_arith.muli(mgpu_arith.extsi(i64, col), mgpu_utils.c(p_strides[4], i64))
                loaded = _predicated_load(p_base, mgpu_arith.addi(p_sl_base, col_off), ok, p_elem, p_zero)
                p_regs[s_idx, rep, ...] = mgpu_vector.from_elements(p_vec, [loaded])

        return (mgpu_fragments.FragmentedArray(_registers=p_regs, _layout=p_mgpu, _is_signed=None),
                mgpu_fragments.FragmentedArray(_registers=m_regs, _layout=p_mgpu, _is_signed=None),
                mgpu_fragments.FragmentedArray(_registers=t_regs, _layout=p_mgpu, _is_signed=None))

    return plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(), plgpu.RefType(), *(scalar,) * 5, tile_1d),
        return_type=(out_ty, stat_ty, stat_ty),
    )(body)(padded_ref, stacked_ref, b, q, h, start, c_start, col_ok)


def _mgpu_load_initial(init_m_ref, init_t_ref, init_acc_ref, b, h, q, c_start, *, block: int,
                       tile_block: int, width: int, blocks: int = 1, tile_1d):
    """Predicated GMEM load of one row's `Initial` state (`[tile_block]` acc, replicated m and total)."""
    scalar = plgpu.Layout.WG_SPLAT(())
    acc_ty = plgpu.ShapeDtypeStruct((tile_block,), init_acc_ref.dtype, layout=tile_1d)
    stat_ty = plgpu.ShapeDtypeStruct((tile_block,), init_m_ref.dtype, layout=tile_1d)

    @emission.stating(lambda m, t, a, *_: cost.Work(
        device_bytes=(width / blocks) * a.dtype.itemsize + m.dtype.itemsize + t.dtype.itemsize))
    def body(ctx, im_ref, it_ref, ia_ref, b_v, h_v, q_v, c_start_v):
        del ctx
        i32, i64 = mgpu_ir.IntegerType.get_signless(32), mgpu_ir.IntegerType.get_signless(64)
        b_i, h_i, q_i, cs_i = (v.registers.item() for v in (b_v, h_v, q_v, c_start_v))
        _, tid_col, unique = _thread_col(tile_block)
        reps = tile_block // unique

        s_elem = mgpu_ir.MemRefType(im_ref.type).element_type
        s_vec = mgpu_ir.VectorType.get((1,), s_elem)
        t1_mgpu = tile_1d.to_mgpu()
        reg_shape = t1_mgpu.registers_shape((tile_block,))
        im_strides, _ = mgpu_ir.MemRefType(im_ref.type).get_strides_and_offset()
        it_strides, _ = mgpu_ir.MemRefType(it_ref.type).get_strides_and_offset()
        m_val = mgpu_llvm.load(s_elem, mgpu_utils.getelementptr(
            _gmem_ptr(im_ref), [_i64_offset((b_i, h_i, q_i), im_strides)], s_elem))
        t_val = mgpu_llvm.load(s_elem, mgpu_utils.getelementptr(
            _gmem_ptr(it_ref), [_i64_offset((b_i, h_i, q_i), it_strides)], s_elem))
        m_regs = np.full(reg_shape, mgpu_vector.from_elements(s_vec, [m_val]), dtype=object)
        t_regs = np.full(reg_shape, mgpu_vector.from_elements(s_vec, [t_val]), dtype=object)

        a_elem = mgpu_ir.MemRefType(ia_ref.type).element_type
        a_strides, _ = mgpu_ir.MemRefType(ia_ref.type).get_strides_and_offset()
        a_base = _gmem_ptr(ia_ref)
        a_row = _i64_offset((b_i, h_i, q_i), a_strides[:3])
        a_zero = mgpu_arith.constant(a_elem, 0.0)
        a_vec = mgpu_ir.VectorType.get((1,), a_elem)
        a_regs = np.empty(reg_shape, dtype=object)
        block_c, width_c = mgpu_utils.c(block, i32), mgpu_utils.c(width, i32)
        for rep in range(reps):
            local_col = mgpu_arith.addi(tid_col, mgpu_utils.c(rep * unique, i32))
            col = mgpu_arith.addi(cs_i, local_col)
            ok = mgpu_arith.andi(
                mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, local_col, block_c),
                mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, col, width_c))
            col_off = mgpu_arith.muli(mgpu_arith.extsi(i64, col), mgpu_utils.c(a_strides[3], i64))
            loaded = _predicated_load(a_base, mgpu_arith.addi(a_row, col_off), ok, a_elem, a_zero)
            a_regs[rep, ...] = mgpu_vector.from_elements(a_vec, [loaded])

        return (mgpu_fragments.FragmentedArray(_registers=a_regs, _layout=t1_mgpu, _is_signed=None),
                mgpu_fragments.FragmentedArray(_registers=m_regs, _layout=t1_mgpu, _is_signed=None),
                mgpu_fragments.FragmentedArray(_registers=t_regs, _layout=t1_mgpu, _is_signed=None))

    return plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(), plgpu.RefType(), plgpu.RefType(), *(scalar,) * 4),
        return_type=(acc_ty, stat_ty, stat_ty),
    )(body)(init_m_ref, init_t_ref, init_acc_ref, b, h, q, c_start)


def _mgpu_store_finished(merged_ref, acc, b, out_t, out_h, c_start, *, block: int, tile_block: int,
                         width: int, blocks: int = 1, acc_layout):
    """Predicated coalesced GMEM store of one `[block]` slice of finished output (`[tile_block]` or
    row 0 of `[2, tile_block]`, row 9795)."""
    scalar = plgpu.Layout.WG_SPLAT(())

    @emission.stating(lambda ref, *_: cost.Work(device_bytes=(width / blocks) * ref.dtype.itemsize))
    def body(ctx, dest_ref, acc_v, b_v, t_v, h_v, c_start_v):
        del ctx
        pos0 = mgpu_ir.DenseI64ArrayAttr.get([0])
        i32, i64 = mgpu_ir.IntegerType.get_signless(32), mgpu_ir.IntegerType.get_signless(64)
        b_i, t_i, h_i, cs_i = (v.registers.item() for v in (b_v, t_v, h_v, c_start_v))
        tid, tid_col, unique = _thread_col(tile_block)
        reps = tile_block // unique
        elem = mgpu_ir.MemRefType(dest_ref.type).element_type
        strides, _ = mgpu_ir.MemRefType(dest_ref.type).get_strides_and_offset()
        base = _gmem_ptr(dest_ref)
        row_base = _i64_offset((b_i, t_i, h_i), strides[:3])
        active_tid = mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, tid, mgpu_utils.c(min(block, unique), i32))
        block_c, width_c = mgpu_utils.c(block, i32), mgpu_utils.c(width, i32)
        acc_regs = (acc_v.registers[0] if len(acc_v.shape) == 2 else acc_v.registers).reshape(reps)
        for rep in range(reps):
            local_col = mgpu_arith.addi(tid_col, mgpu_utils.c(rep * unique, i32))
            col = mgpu_arith.addi(cs_i, local_col)
            in_blk = mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, local_col, block_c)
            ok = mgpu_arith.andi(
                mgpu_arith.andi(active_tid, in_blk),
                mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, col, width_c))
            branch = mgpu_scf.IfOp(ok)
            with mgpu_ir.InsertionPoint(branch.then_block):
                stored = mgpu_vector.extract(acc_regs[rep], dynamic_position=[], static_position=pos0)
                off = mgpu_arith.addi(row_base, mgpu_arith.muli(
                    mgpu_arith.extsi(i64, col), mgpu_utils.c(strides[3], i64)))
                mgpu_llvm.store(stored, mgpu_utils.getelementptr(base, [off], elem))
                mgpu_scf.YieldOp([])

    plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(), acc_layout, *(scalar,) * 4),
    )(body)(merged_ref, acc, b, out_t, out_h, c_start)

def _mgpu_store_stats(max_ref, tot_ref, maximum, total, b, q, h, *, tile_1d):
    """Single-thread GMEM store of one row's scalar maximum and total (thread 0; caller gates `c_block`)."""
    scalar = plgpu.Layout.WG_SPLAT(())

    @emission.stating(lambda m_ref, t_ref, *_: cost.Work(
        device_bytes=m_ref.dtype.itemsize + t_ref.dtype.itemsize))
    def body(ctx, m_ref, t_ref, m_v, t_v, b_v, q_v, h_v):
        del ctx
        pos0 = mgpu_ir.DenseI64ArrayAttr.get([0])
        i32 = mgpu_ir.IntegerType.get_signless(32)
        b_i, q_i, h_i = (v.registers.item() for v in (b_v, q_v, h_v))
        tid = mgpu_arith.remui(mgpu_utils.thread_idx(), mgpu_utils.c(THREADS, i32))
        elected = mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.eq, tid, mgpu_utils.c(0, i32))
        branch = mgpu_scf.IfOp(elected)
        with mgpu_ir.InsertionPoint(branch.then_block):
            m_elem = mgpu_ir.MemRefType(m_ref.type).element_type
            t_elem = mgpu_ir.MemRefType(t_ref.type).element_type
            m_strides, _ = mgpu_ir.MemRefType(m_ref.type).get_strides_and_offset()
            t_strides, _ = mgpu_ir.MemRefType(t_ref.type).get_strides_and_offset()
            m_val = mgpu_vector.extract(m_v.registers.flat[0], dynamic_position=[], static_position=pos0)
            t_val = mgpu_vector.extract(t_v.registers.flat[0], dynamic_position=[], static_position=pos0)
            mgpu_llvm.store(m_val, mgpu_utils.getelementptr(
                _gmem_ptr(m_ref), [_i64_offset((b_i, q_i, h_i), m_strides)], m_elem))
            mgpu_llvm.store(t_val, mgpu_utils.getelementptr(
                _gmem_ptr(t_ref), [_i64_offset((b_i, q_i, h_i), t_strides)], t_elem))
            mgpu_scf.YieldOp([])

    plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(), plgpu.RefType(), tile_1d, tile_1d, *(scalar,) * 3),
    )(body)(max_ref, tot_ref, maximum, total, b, q, h)


def _mgpu_load_output_param(param_ref, b, out_t, out_h, c_start, *, kind: Kind, block: int,
                            tile_block: int, width: int, blocks: int = 1, tile_layout):
    """Predicated coalesced GMEM load of one `output_mod` parameter (`Kind.HEAD` as 0D scalar, `Kind.QUERY`
    as `[2, 1]` splat, `Kind.OUTPUT` as `[2, tile_block]` `tile_layout` fragment), widening 16-bit floats to
    fp32 in registers and preserving signedness for integer and Boolean operands (row 5385)."""
    scalar = plgpu.Layout.WG_SPLAT(())
    is_float = jnp.issubdtype(param_ref.dtype, jnp.floating)
    is_bool = param_ref.dtype == jnp.bool_
    widen = is_float and param_ref.dtype != jnp.float32
    out_dtype = jnp.dtype(jnp.float32) if is_float else param_ref.dtype
    is_signed = mgpu_utils.is_signed(out_dtype)
    is_scalar = kind is not Kind.OUTPUT
    if kind is Kind.HEAD:
        out_ty = plgpu.ShapeDtypeStruct((), out_dtype, layout=scalar)
    elif kind is Kind.QUERY:
        out_ty = plgpu.ShapeDtypeStruct((2, 1), out_dtype, layout=plgpu.Layout.WG_SPLAT((2, 1)))
    else:
        out_ty = plgpu.ShapeDtypeStruct((2, tile_block), out_dtype, layout=tile_layout)
    n_elems = 1.0 if is_scalar else width / blocks

    @emission.stating(lambda p, *_: cost.Work(device_bytes=n_elems * p.dtype.itemsize))
    def body(ctx, p_ref, b_v, t_v, h_v, c_start_v):
        del ctx
        i1, i32, i64 = (mgpu_ir.IntegerType.get_signless(bits) for bits in (1, 32, 64))
        b_i, t_i, h_i, cs_i = (v.registers.item() for v in (b_v, t_v, h_v, c_start_v))
        mem_ty = mgpu_ir.MemRefType(p_ref.type)
        elem = mem_ty.element_type
        strides, _ = mem_ty.get_strides_and_offset()
        shape = mem_ty.shape
        base = _gmem_ptr(p_ref)
        f32 = mgpu_ir.F32Type.get()
        res_elem = i1 if is_bool else (f32 if widen else elem)
        zero_elem = mgpu_arith.constant(elem, 0.0 if is_float else 0)

        def cast_loaded(raw):
            if widen:
                return mgpu_arith.extf(f32, raw)
            if is_bool and elem != i1:
                return mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.ne, raw, zero_elem)
            return raw

        if is_scalar:
            if kind is Kind.HEAD:
                off = _i64_offset((h_i,), strides)
                splat_shape: tuple[int, ...] = ()
            else:
                b_idx = mgpu_utils.c(0, i32) if shape[0] == 1 else b_i
                h_idx = mgpu_utils.c(0, i32) if shape[1] == 1 else h_i
                off = _i64_offset((b_idx, h_idx, t_i), strides)
                splat_shape = (2, 1)
            loaded = cast_loaded(mgpu_llvm.load(elem, mgpu_utils.getelementptr(base, [off], elem)))
            return mgpu_fragments.FragmentedArray.splat(loaded, splat_shape, is_signed=is_signed)
        vec = mgpu_ir.VectorType.get((1,), res_elem)
        t2_mgpu = tile_layout.to_mgpu()
        reg_shape = t2_mgpu.registers_shape((2, tile_block))
        b_idx = mgpu_utils.c(0, i32) if shape[0] == 1 else b_i
        row_off = _i64_offset((b_idx, t_i, h_i), strides[:3])
        _, tid_col, unique = _thread_col(tile_block)
        reps = tile_block // unique
        regs = np.empty(reg_shape, dtype=object)
        block_c, width_c = mgpu_utils.c(block, i32), mgpu_utils.c(width, i32)
        for rep in range(reps):
            local_col = mgpu_arith.addi(tid_col, mgpu_utils.c(rep * unique, i32))
            col = mgpu_arith.addi(cs_i, local_col)
            ok = mgpu_arith.andi(
                mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, local_col, block_c),
                mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, col, width_c))
            col_off = mgpu_arith.muli(mgpu_arith.extsi(i64, col), mgpu_utils.c(strides[3], i64))
            raw = _predicated_load(base, mgpu_arith.addi(row_off, col_off), ok, elem, zero_elem)
            loaded = cast_loaded(raw)
            regs[:, rep, ...] = mgpu_vector.from_elements(vec, [loaded])
        return mgpu_fragments.FragmentedArray(_registers=regs, _layout=t2_mgpu, _is_signed=is_signed)

    return plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(), *(scalar,) * 4),
        return_type=out_ty,
    )(body)(param_ref, b, out_t, out_h, c_start)


def _launch_mosaic(padded, stacked, *extra, units: windows.Units, queries: int, width: int, block: int,
                   raw_scale: float | None = None, out_dtype=None, out_heads: int | None = None,
                   output_mod: Callable | None = None, output_kinds: tuple[Kind, ...] = (),
                   family: str = "mma.sync"):
    """`_value` as one Mosaic GPU launch on NVIDIA sm80+ GPUs (row 9795): zero SMEM, zero warp shuffles,
    predicated coalesced GMEM loads/stores over arbitrary `width`/`slices`, executing the exact same
    `algebra.finished_merged_chunked` body as `_launch`, with optional fused `output_mod` epilogue
    (row 5385)."""
    batch, slices, _, heads, read_width = padded.shape
    init_args, _ = _split_extra(extra, output_kinds)
    has_initial = bool(init_args)
    total_slices = slices + int(has_initial)
    chunk = max(windows.chunk_width(total_slices), 2)
    channels = 1 << (width - 1).bit_length()
    blocks = channels // block
    tile_block = max(block, 64)
    tile_layout = _tile_layout(tile_block)
    tile_1d = _tile_1d_layout(tile_layout)
    rows = (batch, queries, heads)
    n_init = len(init_args)
    n_params = len(output_kinds)
    def entry(padded_ref, stacked_ref, *rest):
        row = lax.axis_index("row")
        h, bq = lax.rem(row, heads), lax.div(row, heads)
        q, b = lax.rem(bq, queries), lax.div(bq, queries)
        c_block = lax.axis_index("block")
        c_start = c_block * block
        local_cols = jax.lax.broadcasted_iota(jnp.int32, (tile_block,), 0)
        read_col_ok = plgpu.layout_cast(
            (local_cols < block) & (c_start + local_cols < min(width, read_width)), tile_1d)
        init_refs, param_refs, out_refs = (
            rest[:n_init], rest[n_init:n_init + n_params], rest[n_init + n_params:])
        if has_initial:
            init_m_ref, init_t_ref, init_acc_ref = init_refs
            raw_acc, raw_m, raw_t = _mgpu_load_initial(
                init_m_ref, init_t_ref, init_acc_ref, b, h, q, c_start, block=block, tile_block=tile_block,
                width=width, blocks=blocks, tile_1d=tile_1d)
            init_state: algebra.Partial | None = algebra.initial_state(
                units, algebra.Partial(raw_acc, raw_m, raw_t), raw_scale=raw_scale)
        else:
            init_state = None

        def read(start):
            out, maximum, total = _mgpu_load_chunk(
                padded_ref, stacked_ref, b, q, h, start, c_start, read_col_ok, chunk=chunk,
                tile_block=tile_block, slices=slices, read_width=min(width, read_width), blocks=blocks,
                tile_layout=tile_layout, tile_1d=tile_1d)
            if init_state is not None:
                slice_pos = plgpu.layout_cast(
                    start + jax.lax.broadcasted_iota(jnp.int32, (chunk, tile_block), 0), tile_layout)
                is_init = slice_pos == slices
                out = jnp.where(is_init, init_state.acc, out)
                maximum = jnp.where(is_init, init_state.m, maximum)
                total = jnp.where(is_init, init_state.total, total)
            return out, maximum, total

        def load(c):
            step = jnp.int32(c) if isinstance(c, int) else mosaic.step_index(c)
            start = step * chunk
            valid = plgpu.layout_cast(
                start + jax.lax.broadcasted_iota(jnp.int32, (chunk, tile_block), 0) < total_slices,
                tile_layout)
            return (*read(start), valid)

        state = algebra.finished_merged_chunked(units, load, -(-total_slices // chunk))
        if out_dtype is not None:
            assert out_heads is not None
            group = out_heads // heads
            out_t, out_h = lax.div(q, group), h * group + lax.rem(q, group)
            (merged_ref,) = out_refs
            acc = plgpu.layout_cast(state.acc, tile_1d)
            if output_mod is not None:
                vals = tuple(
                    _mgpu_load_output_param(ref, b, out_t, out_h, c_start, kind=kind, block=block,
                                            tile_block=tile_block, width=width, blocks=blocks,
                                            tile_layout=tile_layout)
                    for ref, kind in zip(param_refs, output_kinds, strict=True))
                acc = output_mod(
                    plgpu.layout_cast(lax.broadcast_in_dim(acc, (2, tile_block), (1,)), tile_layout), *vals)
                store_layout = tile_layout
            else:
                store_layout = tile_1d
            written = plgpu.layout_cast(acc.astype(out_dtype), store_layout)
            _mgpu_store_finished(merged_ref, written, b, out_t, out_h, c_start, block=block,
                                 tile_block=tile_block, width=width, blocks=blocks, acc_layout=store_layout)
        else:
            merged_ref, maximum_ref, total_ref = out_refs
            written = plgpu.layout_cast(state.acc, tile_1d)
            _mgpu_store_finished(merged_ref, written, b, q, h, c_start, block=block, tile_block=tile_block,
                                 width=width, blocks=blocks, acc_layout=tile_1d)
            pl.when(c_block == 0)(
                lambda: _mgpu_store_stats(maximum_ref, total_ref, plgpu.layout_cast(state.m, tile_1d),
                                          plgpu.layout_cast(state.total, tile_1d), b, q, h, tile_1d=tile_1d))

    if out_dtype is not None:
        assert out_heads is not None
        group = out_heads // heads
        tokens = queries // group
        out_shape: tuple = (jax.ShapeDtypeStruct((batch, tokens, out_heads, width), out_dtype),)
    else:
        out_shape = (jax.ShapeDtypeStruct((*rows, width), padded.dtype),
                     jax.ShapeDtypeStruct(rows, stacked.dtype), jax.ShapeDtypeStruct(rows, stacked.dtype))
    launched = mosaic.kernel(
        entry, out_type=out_shape, grid=(batch * queries * heads, channels // block),
        grid_names=("row", "block"), warpgroups=1, compiled=mosaic.compiler_params(approx_math=False),
        interpret=False, family=family, name="merged_padded")
    merged_out = launched(padded, stacked, *extra)
    return merged_out if isinstance(merged_out, tuple) else (merged_out,)


def _triton_output_specs(out_args: Sequence, output_kinds: tuple[Kind, ...], *, group: int,
                         block: int) -> list[pl.BlockSpec]:
    """Pallas Triton `BlockSpec`s for `output_params` indexed at `(b, out_t, out_h, c)`."""
    specs: list[pl.BlockSpec] = []
    for arg, kind in zip(out_args, output_kinds, strict=True):
        if kind is Kind.HEAD:
            specs.append(pl.BlockSpec((None,), lambda b, q, h, c: (h * group + q % group,)))
        elif kind is Kind.QUERY:
            s0, s1 = arg.shape[0], arg.shape[1]
            specs.append(pl.BlockSpec(
                (None, None, None),
                lambda b, q, h, c, s0=s0, s1=s1: (0 if s0 == 1 else b,
                                                  0 if s1 == 1 else h * group + q % group, q // group)))
        else:
            s0 = arg.shape[0]
            specs.append(pl.BlockSpec(
                (None, None, None, block),
                lambda b, q, h, c, s0=s0: (0 if s0 == 1 else b, q // group, h * group + q % group, c)))
    return specs


def _launch(padded, stacked, *extra, units: windows.Units, queries: int, width: int, block: int,
            raw_scale: float | None = None, out_dtype=None, out_heads: int | None = None,
            output_mod: Callable | None = None, output_kinds: tuple[Kind, ...] = ()):
    """`_value` as one launch, a program a row's `block` of channels, its slices (plus any raw `Initial`
    state normalized in-kernel, row 9804) read a chunk at a time by `finished_merged_chunked`. When
    `out_dtype` and `out_heads` are given (row 7926), finishes, applies optional `output_mod` (row 5385),
    casts and ungroups into `[B, T, out_heads, width]` directly in the store."""
    batch, slices, _, heads, _ = padded.shape
    init_args, out_args = _split_extra(extra, output_kinds)
    has_initial = bool(init_args)
    total_slices = slices + int(has_initial)
    chunk = windows.chunk_width(total_slices)
    tile = 1 << (-(-total_slices // chunk) * chunk - 1).bit_length()
    channels = 1 << (width - 1).bit_length()
    n_init = len(init_args)
    n_params = len(output_kinds)
    def merge(out_ref, maxima_ref, totals_ref, *rest):
        init_refs = rest[:n_init]
        param_refs = rest[n_init:n_init + n_params]
        out_refs = rest[n_init + n_params:]
        if has_initial:
            init_m_ref, init_t_ref, init_acc_ref = init_refs
            raw_t = init_t_ref[...].reshape(1, 1)
            raw_m = init_m_ref[...].reshape(1, 1)
            raw_acc = init_acc_ref[...].reshape(1, 1, block)
            init_state: algebra.Partial | None = algebra.initial_state(
                units, algebra.Partial(raw_acc, raw_m, raw_t), raw_scale=raw_scale)
        else:
            init_state = None

        def read(start):
            at = pl.ds(start, chunk)
            out = out_ref[at]
            maximum = maxima_ref[0, at].reshape(chunk, 1)
            total = totals_ref[0, at].reshape(chunk, 1)
            if init_state is not None:
                slice_pos = start + jax.lax.broadcasted_iota(jnp.int32, (chunk, 1), 0)
                is_init = slice_pos == slices
                out = jnp.where(is_init[..., None], init_state.acc, out)
                maximum = jnp.where(is_init, init_state.m, maximum)
                total = jnp.where(is_init, init_state.total, total)
            return out, maximum, total

        state = algebra.finished_merged_chunked(units, *_chunks(total_slices, read))
        if out_dtype is not None:
            (merged_ref,) = out_refs
            acc = state.acc
            if output_mod is not None:
                vals = tuple(
                    (v.reshape((1, 1)) if kind is Kind.QUERY
                     else v.reshape((1, block)) if kind is Kind.OUTPUT else v)
                    for ref, kind in zip(param_refs, output_kinds, strict=True)
                    for v in (ref[...].astype(jnp.float32)
                              if jnp.issubdtype(ref.dtype, jnp.floating) else ref[...],))
                acc = output_mod(acc, *vals)
            merged_ref[...] = acc.astype(out_dtype)
        else:
            merged_ref, maximum_ref, total_ref = out_refs
            merged_ref[...] = state.acc

            @pl.when(pl.program_id(3) == 0)
            def _():
                maximum_ref[...] = state.m
                total_ref[...] = state.total

    rows = (batch, queries, heads)
    in_specs = [pl.BlockSpec((None, tile, None, 1, block), lambda b, q, h, c: (b, 0, q, h, c)),
                pl.BlockSpec((None, None, 1, tile, None), lambda b, q, h, c: (0, b, h, 0, q)),
                pl.BlockSpec((None, None, 1, tile, None), lambda b, q, h, c: (1, b, h, 0, q))]
    if has_initial:
        init_row = pl.BlockSpec((None, 1, None), lambda b, q, h, c: (b, h, q))
        in_specs.extend([init_row, init_row,
                         pl.BlockSpec((None, 1, None, block), lambda b, q, h, c: (b, h, q, c))])
    if out_dtype is not None:
        assert out_heads is not None
        group = out_heads // heads
        tokens = queries // group
        in_specs.extend(_triton_output_specs(out_args, output_kinds, group=group, block=block))
        out_specs = [pl.BlockSpec((None, None, 1, block),
                                  lambda b, q, h, c: (b, q // group, h * group + q % group, c))]
        out_shape = [jax.ShapeDtypeStruct((batch, tokens, out_heads, width), out_dtype)]
    else:
        row = pl.BlockSpec((None, None, 1), lambda b, q, h, c: (b, q, h))
        out_specs = [pl.BlockSpec((None, None, 1, block), lambda b, q, h, c: (b, q, h, c)), row, row]
        out_shape = [jax.ShapeDtypeStruct((*rows, width), padded.dtype),
                     jax.ShapeDtypeStruct(rows, stacked.dtype), jax.ShapeDtypeStruct(rows, stacked.dtype)]
    return tuple(launches.pallas_call(
        merge, name="merged_padded", grid=(*rows, channels // block),
        in_specs=in_specs, out_specs=out_specs, out_shape=out_shape,
    )(padded, stacked, stacked, *extra))


def _on_cuda(padded, stacked, *extra, units: windows.Units, queries: int, width: int,
             raw_scale: float | None = None, out_dtype=None, out_heads: int | None = None,
             output_mod: Callable | None = None, output_kinds: tuple[Kind, ...] = ()):
    """The merge on a CUDA GPU: one launch (`_kernel`) of fp32 states (with bf16/fp16/fp32/int output hook
    parameters); the law's path (`_value`) of any other's, a float64 truth's among them, whose constants
    Triton's float32 kernel holds none of."""
    init_args, out_args = _split_extra(extra, output_kinds)
    native = (stacked.dtype == jnp.float32 and all(a.dtype == jnp.float32 for a in init_args)
              and all(a.dtype != jnp.float64 for a in out_args))
    run = _kernel if native else _value
    return run(padded, stacked, *extra, units=units, queries=queries, width=width, raw_scale=raw_scale,
               out_dtype=out_dtype, out_heads=out_heads, output_mod=output_mod, output_kinds=output_kinds)


mlir.register_lowering(merged_padded_p, mlir.lower_fun(_on_cuda, multiple_results=True), platform="cuda")
