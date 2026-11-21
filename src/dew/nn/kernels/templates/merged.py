"""A split walk's slices merged where its launch wrote them: one Triton launch on a GPU, the law's own path
elsewhere and under differentiation.

A decode step's walk splits a row's keys across programs (`attention_decode`), each slice's output, maximum
and denominator padded to its launch's tile of rows. The law merges them (`attention_algebra.merged_state`:
`slice_state` of each, then `finished_merged`), which in XLA took the padded arrays' slices, their selects and
the law's total and output as separate launches around it. `merged_padded` takes the launch's padded arrays as
they are; on a GPU it lowers as one Pallas Triton launch that reads them in place and runs the law's own
functions on each row (`slice_state`, `finished_merged`), elsewhere as the law's path on their slices
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
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax._src import api_util, dispatch
from jax._src.interpreters import partial_eval as pe
from jax.custom_derivatives import SymbolicZero
from jax.experimental import pallas as pl
from jax.extend import core as jcore
from jax.interpreters import batching, mlir

from dew.algebra import windows
from dew.nn import programs
from dew.nn.derivatives import KernelGlue, contracted, declared_tangent
from dew.nn.kernels import generation, launches
from dew.nn.kernels.templates import attention_algebra as algebra, cost, emission, mosaic
from dew.nn.kernels.templates.attention import Initial, ungrouped_rows


def _trimmed(padded, stacked, *, queries: int, width: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A split walk's padded outputs `padded` `[B, S, R, H, Dv]` and stacked maxima and denominators `stacked`
    `[2, B, H, S, R]` at their first `queries` rows and `width` columns: the slices' outputs, maxima and
    denominators the law merges (`attention_algebra.merged_state`), the one reading of their layout."""
    return padded[:, :, :queries, :, :width], stacked[0, ..., :queries], stacked[1, ..., :queries]


def _value(padded, stacked, *extra, units: windows.Units, queries: int, width: int,
           raw_scale: float | None = None, out_dtype=None, out_heads: int | None = None):
    """The law's merge of a split walk's padded slices (`_trimmed`) and optional raw `Initial` state
    (`extra` = `(init_m, init_total, init_acc)`), delegated directly to `algebra.merged_state`. When
    `out_dtype` and `out_heads` are given (row 7926), finishes, ungroups and casts to `[B, T, out_heads,
    width]`."""
    initial = Initial(*extra) if extra else None
    state = algebra.merged_state(units, *_trimmed(padded, stacked, queries=queries, width=width),
                                 initial=initial, raw_scale=raw_scale)
    if out_dtype is None:
        return tuple(state[:3])
    assert out_heads is not None
    return (ungrouped_rows(state.acc, out_heads).astype(out_dtype),)
def merged_split(spec: algebra.Forward, padded, stacked, *, queries: int, width: int,
                 initial: Initial | None = None) -> algebra.Partial:
    """A split walk's padded slices merged (`_trimmed`'s layout) with the call's `initial` state where it has
    one, through `merged_padded` (one launch on a GPU, normalizing `initial` inside the launch; row 9804)."""
    return merged_padded(spec, padded, stacked, queries=queries, width=width, initial=initial)


merged_padded_p = jcore.Primitive("merged_padded")
merged_padded_p.multiple_results = True


@merged_padded_p.def_abstract_eval
def _abstract(padded, stacked, *extra, units, queries, width, raw_scale=None, out_dtype=None,
              out_heads=None):
    """The merge's result types, its value's (`_value`) as JAX types it: its shapes and dtypes, and the
    operands' shardings and manual axes (`shard_map`'s varying ones) as the law's own rules carry them.
    Types made fresh dropped them, so a scan of the law's output into this merge's was refused, its manual
    axes unmatched (the base review of 9eb04fc04, F1)."""
    value = functools.partial(_value, units=units, queries=queries, width=width, raw_scale=raw_scale,
                              out_dtype=out_dtype, out_heads=out_heads)
    operands = (padded, stacked, *extra)
    return pe.abstract_eval_fun(value, *operands,
                                debug_info=api_util.debug_info("merged_padded", value, operands, {}))

merged_padded_p.def_impl(functools.partial(dispatch.apply_primitive, merged_padded_p))
mlir.register_lowering(merged_padded_p, mlir.lower_fun(_value, multiple_results=True))


def _batched(args, dims, **params):
    """A batch of splits merged as one launch (`merged_padded_p` again, kept for a GPU's lowering): the
    batch folded into the splits' leading rows, `padded`'s `[N B, ...]`, `stacked`'s `[2, N B, ...]` and any
    initial state's `[N B, ...]`, an operand the batch does not map (JAX's `not_mapped`) broadcast along it;
    each result unfolded. Mapped by `jax.vmap` of the law's path, a batched split merge ran XLA's law, not
    its launch (the merge P0's base review of aa990591f, F9)."""
    padded, stacked, *extra = args
    size = next(arg.shape[dim] for arg, dim in zip(args, dims, strict=True) if dim is not None)
    padded = (jnp.broadcast_to(padded, (size, *padded.shape)) if dims[0] is None
              else jnp.moveaxis(padded, dims[0], 0))
    stacked = (jnp.broadcast_to(stacked, (size, *stacked.shape)) if dims[1] is None
               else jnp.moveaxis(stacked, dims[1], 0))
    # The extents folded and unfolded by name, so a batch of no splits (`size` 0) keeps its rows' count.
    batch = padded.shape[1]
    folded_extra = []
    for arg, dim in zip(extra, dims[2:], strict=True):
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
                                    padded, stacked, *extra))


def merged_finished(spec: algebra.Forward, padded: jax.Array, stacked: jax.Array, *, queries: int,
                    width: int, out_dtype, out_heads: int, initial: Initial | None = None) -> jax.Array:
    """`merged_padded` finished, ungrouped to `[B, queries // (out_heads // KV), out_heads, width]` and cast
    to `out_dtype` inside the same launch (row 7926: eliminates the post-merge finish/transpose/cast XLA
    fusion on a split decode step)."""
    if algebra.finished_form(spec) is not spec:
        raise ValueError("a split walk's merge takes its finished form's states (`finished_form`)")
    extra = () if initial is None else (initial.m, initial.total, initial.acc)
    raw_scale = spec.scale if (initial is not None and spec.raw) else None
    (out,) = _merged(algebra.units_of(spec), queries, width, raw_scale, jnp.dtype(out_dtype), out_heads,
                     padded, stacked, *extra)
    return out


def _launched(units, queries, width, raw_scale, out_dtype, out_heads, *operands):
    """`merged_padded_p`'s value: one launch on a GPU (`_kernel`), the law's path elsewhere."""
    return tuple(merged_padded_p.bind(*operands, units=units, queries=queries, width=width,
                                      raw_scale=raw_scale, out_dtype=out_dtype, out_heads=out_heads))


def _tangent(units, queries, width, raw_scale, out_dtype, out_heads, primals, tangents):
    """The law's path's value and tangent (`jax.jvp` of `_value`) along the operands that move alone, one
    whose tangent is a symbolic zero held as the law's own rule holds it (`means._merged_state_jvp`): a
    differentiated step merges as the law does. An instantiated zero for the outputs moved the law's ratio
    along an exponential's operand, which it refuses (the base review of 9bc8a9da1, F1)."""
    value = functools.partial(_value, units=units, queries=queries, width=width, raw_scale=raw_scale,
                              out_dtype=out_dtype, out_heads=out_heads)
    live = [not isinstance(tangent, SymbolicZero) for tangent in tangents]
    if not any(live):
        out = value(*primals)
        return out, tuple(SymbolicZero(jax.typeof(x).to_tangent_aval()) for x in out)
    perturbed, fill = programs.holes(primals, live)
    moving = tuple(tangent for tangent, is_live in zip(tangents, live, strict=True) if is_live)
    return jax.jvp(lambda *active: value(*fill(active)), tuple(perturbed), moving)

_merged = contracted("test_a_padded_merges_derivatives_are_the_laws")(
    declared_tangent(KernelGlue(), _launched, _tangent, nondiff_argnums=(0, 1, 2, 3, 4, 5),
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
                      queries: int, width: int, block: int, has_initial: bool = False,
                      raw_scale: float | None = None, out_dtype=None,
                      out_heads: int | None = None) -> tuple[cost.Work, cost.Work, int]:
    """One program's on-chip work, the full launch's work, and the grid size of `_launch` at `block` channels
    on the common one-window path (`means.merged_state_chunked`): one anchor/hull joining pass (`joining`:
    chunk 0 in `role.fixed` plus `chunks - 1` loop steps) followed by one window's two dependent passes
    (`_placing` and `_summing`, `2 * chunks` steps), excluding the rare re-landing `lax.cond` branch, for a
    span of `3 * chunks` steps. Device traffic charges `padded.shape[-1]` channels read per slice (matching
    Triton's physical input bounds), `width` channels for output writes and initial-state reads, and row-stat
    stores once per row (`program_id(3) == 0`, omitted when `out_dtype` finishes in-kernel)."""
    batch, slices, _, heads, read_width = padded.shape
    total_slices = slices + int(has_initial)
    channels = 1 << (width - 1).bit_length()
    chunks = -(-total_slices // windows.chunk_width(total_slices))
    blocks = channels // block
    rows = batch * queries * heads
    launched = rows * blocks
    if launched == 0:
        return cost.Work(), cost.Work(), 0
    extra_s = ((jax.ShapeDtypeStruct((batch, heads, queries), stacked.dtype),
                jax.ShapeDtypeStruct((batch, heads, queries), stacked.dtype),
                jax.ShapeDtypeStruct((batch, heads, queries, width), padded.dtype)) if has_initial else ())
    launch = functools.partial(_launch, units=units, queries=queries, width=width, block=block,
                               raw_scale=raw_scale, out_dtype=out_dtype, out_heads=out_heads)
    (kernel,) = (equation for equation in jax.make_jaxpr(launch)(padded, stacked, *extra_s).jaxpr.eqns
                 if programs.opaque(equation))
    body = kernel.params["jaxpr"]
    (role,) = emission.emitted(body, 1, None, device=range(len(body.invars))).roles
    if len(role.walks) == 3:
        joining_walk, mean_walk, _ = role.walks
    else:
        joining_walk, mean_walk = cost.Work(), role.walks[0]
    walk_work = emission.repeated(joining_walk, chunks - 1) + emission.repeated(mean_walk, 2 * chunks)
    passes = 3 * chunks
    padded_item, stacked_item = padded.dtype.itemsize, stacked.dtype.itemsize
    init_reads = int(has_initial)
    if out_dtype is None:
        total_device_bytes = float(rows * ((3 * slices * read_width + (init_reads + 1) * width) * padded_item
                                           + (blocks * (3 * slices + init_reads) + 1) * 2 * stacked_item))
    else:
        out_item = jnp.dtype(out_dtype).itemsize
        total_device_bytes = float(rows * (3 * slices * read_width * padded_item
                                           + init_reads * width * padded_item + width * out_item
                                           + blocks * (3 * slices + init_reads) * 2 * stacked_item))
    per_program = dataclasses.replace(role.fixed + walk_work, device_bytes=total_device_bytes / launched,
                                      span=float(passes))
    total = dataclasses.replace(emission.repeated(per_program, launched), device_bytes=total_device_bytes,
                                span=float(passes), kernels=1.0)
    return per_program, total, launched


@functools.cache
def _planned_block(units: windows.Units, padded: jax.ShapeDtypeStruct, stacked: jax.ShapeDtypeStruct,
                   queries: int, width: int, target: generation.GpuTarget, has_initial: bool = False,
                   raw_scale: float | None = None, out_dtype=None,
                   out_heads: int | None = None) -> tuple[int, cost.Work]:
    """The channels a program of `_kernel` merges and that launch's total device work (`_launch_for_block`),
    cached together on `units` alone (row 8809) so `_channel_block` and `launch_work` trace each candidate
    once."""
    _, slices, _, _, _ = padded.shape
    total_slices = slices + int(has_initial)
    channels = 1 << (width - 1).bit_length()
    blocks = [block for block in (channels >> shift for shift in range(channels.bit_length()))
              if block * windows.chunk_width(total_slices) >= THREADS] or [channels]

    candidates = {block: _launch_for_block(units, padded, stacked, queries, width, block, has_initial,
                                           raw_scale, out_dtype, out_heads) for block in blocks}

    def seconds(block: int) -> float:
        _, total, launched = candidates[block]
        return cost.launch_seconds(total, launched, target)

    chosen = min(blocks, key=lambda block: (seconds(block), -block))
    return chosen, candidates[chosen][1]


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
                out_heads: int | None = None) -> cost.Work:
    """One native `merged_padded` launch's device work at `_channel_block`'s chosen block (`_planned_block`,
    row 8779): common-path arithmetic across the grid, valid device traffic with row-stat stores counted once
    per row (row 8809), span `3 * chunks`, one kernel."""
    return _planned_block(algebra.units_of(units), padded, stacked, queries, width, target, has_initial,
                          raw_scale, out_dtype, out_heads)[1]


def _kernel(padded, stacked, *extra, units: windows.Units, queries: int, width: int,
            raw_scale: float | None = None, out_dtype=None, out_heads: int | None = None):
    """`_value` as one launch (`_launch`), a program a row's block of channels (`_channel_block`)."""
    batch, _, _, heads, _ = padded.shape
    if batch == 0 or queries == 0 or heads == 0:
        return _value(padded, stacked, *extra, units=units, queries=queries, width=width,
                      raw_scale=raw_scale, out_dtype=out_dtype, out_heads=out_heads)
    padded_s, stacked_s = (jax.ShapeDtypeStruct(operand.shape, operand.dtype)
                           for operand in (padded, stacked))
    block = _channel_block(units, padded_s, stacked_s, queries, width, generation.gpu_target(), bool(extra),
                           raw_scale, out_dtype, out_heads)
    return _launch(padded, stacked, *extra, units=units, queries=queries, width=width, block=block,
                   raw_scale=raw_scale, out_dtype=out_dtype, out_heads=out_heads)


def _launch(padded, stacked, *extra, units: windows.Units, queries: int, width: int, block: int,
            raw_scale: float | None = None, out_dtype=None, out_heads: int | None = None):
    """`_value` as one launch, a program a row's `block` of channels, its slices (plus any raw `Initial`
    state normalized in-kernel, row 9804) read a chunk at a time by `finished_merged_chunked`. When
    `out_dtype` and `out_heads` are given (row 7926), finishes, casts and ungroups into `[B, T, out_heads,
    width]` directly in the store."""
    batch, slices, _, heads, _ = padded.shape
    has_initial = bool(extra)
    total_slices = slices + int(has_initial)
    chunk = windows.chunk_width(total_slices)
    tile = 1 << (-(-total_slices // chunk) * chunk - 1).bit_length()
    channels = 1 << (width - 1).bit_length()

    def merge(out_ref, maxima_ref, totals_ref, *rest):
        if has_initial:
            init_m_ref, init_t_ref, init_acc_ref, *out_refs = rest
            raw_t = init_t_ref[...].reshape(1, 1)
            raw_m = init_m_ref[...].reshape(1, 1)
            raw_acc = init_acc_ref[...].reshape(1, 1, block)
            init_state: algebra.Partial | None = algebra.initial_state(
                units, algebra.Partial(raw_acc, raw_m, raw_t), raw_scale=raw_scale)
        else:
            out_refs = rest
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
            merged_ref[...] = state.acc.astype(out_dtype)
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
             raw_scale: float | None = None, out_dtype=None, out_heads: int | None = None):
    """The merge on a CUDA GPU: one launch (`_kernel`) of fp32 states, a launch's; the law's path
    (`_value`) of any other's, a float64 truth's among them, whose constants Triton's float32 kernel holds
    none of."""
    fp32 = stacked.dtype == jnp.float32 and all(a.dtype == jnp.float32 for a in extra)
    run = _kernel if fp32 else _value
    return run(padded, stacked, *extra, units=units, queries=queries, width=width, raw_scale=raw_scale,
               out_dtype=out_dtype, out_heads=out_heads)


mlir.register_lowering(merged_padded_p, mlir.lower_fun(_on_cuda, multiple_results=True), platform="cuda")
