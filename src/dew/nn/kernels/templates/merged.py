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
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax._src import api_util, core as jax_core, dispatch
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas.triton import lowering as triton_lowering
from jax.custom_derivatives import SymbolicZero
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton
from jax.extend import core as jcore
from jax.interpreters import batching, mlir

from dew.algebra import means, scaled
from dew.nn import derivatives, precision, programs
from dew.nn.derivatives import KernelGlue, contracted, declared_tangent
from dew.nn.kernels import generation, launches
from dew.nn.kernels.templates import attention_algebra as algebra, cost, emission, mosaic


def _trimmed(padded, stacked, *, queries: int, width: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A split walk's padded outputs `padded` `[B, S, R, H, Dv]` and stacked maxima and denominators `stacked`
    `[2, B, H, S, R]` at their first `queries` rows and `width` columns: the slices' outputs, maxima and
    denominators the law merges (`attention_algebra.merged_state`), the one reading of their layout."""
    return padded[:, :, :queries, :, :width], stacked[0, ..., :queries], stacked[1, ..., :queries]


def _value(padded, stacked, *, units: means.Units, queries: int, width: int):
    """The law's merge of a split walk's padded slices (`_trimmed`), as `(output, maximum, total)`,
    `[B, Q, H, width]` and `[B, Q, H]`."""
    return tuple(algebra.merged_state(units, *_trimmed(padded, stacked, queries=queries, width=width))[:3])


def merged_split(spec: algebra.Forward, padded, stacked, *, queries: int, width: int,
                 initial=None) -> algebra.Partial:
    """A split walk's padded slices merged (`_trimmed`'s layout) with the call's `initial` state where it has
    one, as the law's own path (`attention_algebra.merged_state`); without, where the launch wrote them
    (`merged_padded`: one launch on a GPU)."""
    if initial is None:
        return merged_padded(spec, padded, stacked, queries=queries, width=width)
    return algebra.merged_state(spec, *_trimmed(padded, stacked, queries=queries, width=width), initial)


merged_padded_p = jcore.Primitive("merged_padded")
merged_padded_p.multiple_results = True


@merged_padded_p.def_abstract_eval
def _abstract(padded, stacked, *, units, queries, width):
    """The merge's result types, its value's (`_value`) as JAX types it: its shapes and dtypes, and the
    operands' shardings and manual axes (`shard_map`'s varying ones) as the law's own rules carry them.
    Types made fresh dropped them, so a scan of the law's output into this merge's was refused, its manual
    axes unmatched (the base review of 9eb04fc04, F1)."""
    value = functools.partial(_value, units=units, queries=queries, width=width)
    return pe.abstract_eval_fun(value, padded, stacked,
                                debug_info=api_util.debug_info("merged_padded", value, (padded, stacked), {}))


merged_padded_p.def_impl(functools.partial(dispatch.apply_primitive, merged_padded_p))
mlir.register_lowering(merged_padded_p, mlir.lower_fun(_value, multiple_results=True))


def _batched(args, dims, **params):
    """A batch of splits merged as one launch (`merged_padded_p` again, kept for a GPU's lowering): the
    batch folded into the splits' leading rows, `padded`'s `[N B, ...]` and `stacked`'s `[2, N B, ...]`, an
    operand the batch does not map (JAX's `not_mapped`) broadcast along it; each result unfolded. Mapped by
    `jax.vmap` of the law's path, a batched split merge ran XLA's law, not its launch (the merge P0's base
    review of aa990591f, F9)."""
    padded, stacked = args
    size = next(arg.shape[dim] for arg, dim in zip(args, dims, strict=True) if dim is not None)
    padded = (jnp.broadcast_to(padded, (size, *padded.shape)) if dims[0] is None
              else jnp.moveaxis(padded, dims[0], 0))
    stacked = (jnp.broadcast_to(stacked, (size, *stacked.shape)) if dims[1] is None
               else jnp.moveaxis(stacked, dims[1], 0))
    # The extents folded and unfolded by name, so a batch of no splits (`size` 0) keeps its rows' count.
    batch = padded.shape[1]
    folded = merged_padded_p.bind(padded.reshape(size * batch, *padded.shape[2:]),
                                  jnp.moveaxis(stacked, 0, 1).reshape(2, size * batch, *stacked.shape[3:]),
                                  **params)
    return tuple(merge.reshape(size, batch, *merge.shape[1:]) for merge in folded), (0, 0, 0)


batching.primitive_batchers[merged_padded_p] = _batched


def merged_padded(spec: algebra.Forward, padded: jax.Array, stacked: jax.Array, *, queries: int,
                  width: int) -> algebra.Partial:
    """A split walk's slices as its launch wrote them, padded (`padded` `[B, S, R, H, Dv]`, its maxima and
    denominators stacked `[2, B, H, S, R]`), merged into one finished state of their first `queries` rows and
    `width` columns, `[B, Q, H]` rows (`attention_algebra.merged_state`'s, no initial state)."""
    if algebra.finished_form(spec) is not spec:
        raise ValueError("a split walk's merge takes its finished form's states (`finished_form`)")
    return algebra.Partial(*_merged(algebra.units_of(spec), queries, width, padded, stacked))


def _launched(units, queries, width, padded, stacked):
    """`merged_padded_p`'s value: one launch on a GPU (`_kernel`), the law's path elsewhere."""
    return tuple(merged_padded_p.bind(padded, stacked, units=units, queries=queries, width=width))


def _tangent(units, queries, width, primals, tangents):
    """The law's path's value and tangent (`jax.jvp` of `_value`) along the operands that move alone, one
    whose tangent is a symbolic zero held as the law's own rule holds it (`means._merged_state_jvp`): a
    differentiated step merges as the law does. An instantiated zero for the outputs moved the law's ratio
    along an exponential's operand, which it refuses (the base review of 9bc8a9da1, F1)."""
    value = functools.partial(_value, units=units, queries=queries, width=width)
    (padded, stacked), (along_padded, along_stacked) = primals, tangents
    if isinstance(along_padded, SymbolicZero):
        return jax.jvp(lambda stacked: value(padded, stacked), (stacked,), (along_stacked,))
    if isinstance(along_stacked, SymbolicZero):
        return jax.jvp(lambda padded: value(padded, stacked), (padded,), (along_padded,))
    return jax.jvp(value, primals, tangents)


_merged = contracted("test_a_padded_merges_derivatives_are_the_laws")(
    declared_tangent(KernelGlue(), _launched, _tangent, nondiff_argnums=(0, 1, 2), symbolic_zeros=True))


def _chunks(slices: int, read: Callable[[jax.Array], tuple]) -> tuple[Callable, int]:
    """A row's `slices` as the law reads them a chunk at a time (`means.merged_state_chunked`), `(load,
    chunks)`: `read(start)` a chunk's outputs `[K, 1, C]`, maxima and denominators `[K, 1]` from slice
    `start`, `K` the law's width of a row of them (`means.chunk_width`), its slots past the count held by
    none (`valid`)."""
    width = means.chunk_width(slices)

    def load(c):
        start = mosaic.step_index(c) * width
        valid = start + jax.lax.broadcasted_iota(jnp.int32, (width, 1), 0) < slices
        return (*read(start), valid)

    return load, -(-slices // width)




THREADS = 4 * 32
"""A program's threads: Pallas Triton's default of four warps (`CompilerParams.num_warps`), the merge's."""


@functools.cache
def merge_work(units: means.Units, padded: jax.ShapeDtypeStruct, stacked: jax.ShapeDtypeStruct,
               queries: int, width: int, target: generation.GpuTarget) -> tuple[int, cost.Work]:
    """The channel block (`_channel_block`) and total emitted work (`cost.Work`, row 8779) of `_launch`
    for operands `padded` and `stacked`: first-block programs write `maximum_ref` and `total_ref` while
    other channel blocks do not (row 8809), and the cache keys on `units` alone rather than the full
    `Forward` spec."""
    batch, slices, _, heads, _ = padded.shape
    channels, chunks = 1 << (width - 1).bit_length(), -(-slices // means.chunk_width(slices))
    blocks = [block for block in (channels >> shift for shift in range(channels.bit_length()))
              if block * means.chunk_width(slices) >= THREADS] or [channels]
    row_meta_bytes = 2 * stacked.dtype.itemsize

    def priced(block: int) -> tuple[float, cost.Work]:
        launch = functools.partial(_launch, units=units, queries=queries, width=width, block=block)
        (kernel,) = (equation for equation in jax.make_jaxpr(launch)(padded, stacked).jaxpr.eqns
                     if programs.opaque(equation))
        body = kernel.params["jaxpr"]
        (role,) = emission.emitted(body, 1, None, device=range(len(body.invars))).roles
        work = role.fixed + emission.repeated(sum(role.walks, cost.Work()), chunks)
        n_blocks = channels // block
        rows_count = batch * queries * heads
        launched = rows_count * n_blocks
        # Only channel block 0 writes maximum_ref and total_ref (`pl.program_id(3) == 0`, row 8809).
        total_bytes = rows_count * (
            work.device_bytes + (n_blocks - 1) * max(0, work.device_bytes - row_meta_bytes))
        waves = -(-launched // (target.multiprocessors or 1))
        traffic = total_bytes / target.memory_rate if target.memory_rate else 0.0
        seconds = max(waves * cost.walked(work, target), traffic)
        return seconds, cost.Work(device_bytes=total_bytes, kernels=1)

    best_block, (_, best_work) = min(((b, priced(b)) for b in blocks), key=lambda pair: (pair[1][0], -pair[0]))
    return best_block, best_work


def _channel_block(units: means.Units, padded: jax.ShapeDtypeStruct, stacked: jax.ShapeDtypeStruct,
                   queries: int, width: int, target: generation.GpuTarget) -> int:
    """The channels a program of `_kernel` merges for operands `padded` and `stacked` (`merge_work`)."""
    return merge_work(units, padded, stacked, queries, width, target)[0]


def _kernel(padded, stacked, *, units: means.Units, queries: int, width: int):
    """`_value` as one launch (`_launch`), a program a row's block of channels (`_channel_block`)."""
    shaped = tuple(jax.ShapeDtypeStruct(operand.shape, operand.dtype) for operand in (padded, stacked))
    block = _channel_block(units, *shaped, queries, width, generation.gpu_target())
    return _launch(padded, stacked, units=units, queries=queries, width=width, block=block)


def _launch(padded, stacked, *, units: means.Units, queries: int, width: int, block: int):
    """`_value` as one launch, a program a row's `block` of channels, its slices read a chunk at a time by
    the law's chunked merge (`attention_algebra.finished_merged_chunked`: the value `_value`'s merge
    computes, its live set a chunk's whatever the slices). Triton's tiles are powers of two: a row's slices
    a block of the least power of two past their chunks, read past their count by none; its channels the
    least power of two past the width in blocks, its loads and stores past the arrays' columns masked
    (Pallas's partial blocks), each channel merged alone, so a lane no column holds touches none that does.
    Each program forms its row's law (its factors, total and maximum, which no channel changes) for its
    block; the first block's writes the row's maximum and total."""
    batch, slices, _, heads, _ = padded.shape
    chunk = means.chunk_width(slices)
    tile = 1 << (-(-slices // chunk) * chunk - 1).bit_length()
    channels = 1 << (width - 1).bit_length()

    def merge(out_ref, maxima_ref, totals_ref, merged_ref, maximum_ref, total_ref):
        def read(start):
            at = pl.ds(start, chunk)
            return out_ref[at], maxima_ref[0, at].reshape(chunk, 1), totals_ref[0, at].reshape(chunk, 1)

        state = algebra.finished_merged_chunked(units, *_chunks(slices, read))
        merged_ref[...] = state.acc

        @pl.when(pl.program_id(3) == 0)
        def _():
            maximum_ref[...] = state.m
            total_ref[...] = state.total

    rows = (batch, queries, heads)
    row = pl.BlockSpec((None, None, 1), lambda b, q, h, c: (b, q, h))
    return tuple(launches.pallas_call(
        merge, name="merged_padded", grid=(*rows, channels // block),
        in_specs=[pl.BlockSpec((None, tile, None, 1, block), lambda b, q, h, c: (b, 0, q, h, c)),
                  pl.BlockSpec((None, None, 1, tile, None), lambda b, q, h, c: (0, b, h, 0, q)),
                  pl.BlockSpec((None, None, 1, tile, None), lambda b, q, h, c: (1, b, h, 0, q))],
        out_specs=[pl.BlockSpec((None, None, 1, block), lambda b, q, h, c: (b, q, h, c)), row, row],
        out_shape=[jax.ShapeDtypeStruct((*rows, width), padded.dtype),
                   jax.ShapeDtypeStruct(rows, stacked.dtype), jax.ShapeDtypeStruct(rows, stacked.dtype)],
    )(padded, stacked, stacked))


def _on_cuda(padded, stacked, *, units: means.Units, queries: int, width: int):
    """The merge on a CUDA GPU: one launch (`_kernel`) of fp32 states, a launch's; the law's path
    (`_value`) of any other's, a float64 truth's among them, whose constants Triton's float32 kernel holds
    none of."""
    run = _kernel if stacked.dtype == jnp.float32 else _value
    return run(padded, stacked, units=units, queries=queries, width=width)


mlir.register_lowering(merged_padded_p, mlir.lower_fun(_on_cuda, multiple_results=True), platform="cuda")


def _divided(numerator, denominator):
    """A correctly rounded fp32 quotient (`precision.correctly_rounded_divide`'s) in a Triton kernel."""
    numerator, denominator = jnp.broadcast_arrays(numerator, denominator)
    [quotient] = pltriton.elementwise_inline_asm(
        "div.rn.f32 $0, $1, $2;", args=[numerator, denominator], constraints="=r,r,r", pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(numerator.shape, numerator.dtype)])
    return quotient


# The law's primitives in a Triton kernel as their own values (the kernel runs the law's functions; every
# one the law makes, `scaled.MADE`), Dew's marked calls as their bodies, and its correctly rounded division
# as PTX's.
for _primitive, (_value_of, _several) in scaled.MADE.items():
    triton_lowering.register_lowering(_primitive)(triton_lowering.lower_fun(_value_of,
                                                                            multiple_results=_several))
triton_lowering.register_lowering(derivatives.marked_call_p)(triton_lowering.lower_fun(
    lambda *args, call_jaxpr, **_: jax_core.eval_jaxpr(call_jaxpr, (), *args), multiple_results=True))
triton_lowering.register_lowering(precision.correctly_rounded_divide_p)(
    triton_lowering.lower_fun(_divided, multiple_results=False))
