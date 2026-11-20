"""The Mamba-2 chunked SSD scan as one Pallas kernel, for TPU.

`dew.nn.mixers.mamba2.chunk_ssd` splits a sequence into chunks of `C` steps
and, per batch element and head, runs

    y_i = sum_{j <= i} (C_i . B_j) exp(segsum_ij) x_j            (intra-chunk)
        + exp(acs_i) C_i S^T                                     (the carry)
    S' = exp(acs_C-1) S + sum_j exp(segsum_C-1,j) x_j B_j^T      (the write)

with `segsum_ij = sum_{j < k <= i} A dt_k`, `acs` its cumulative sum and `x`
carrying its `dt`. XLA's six einsums send every chunk's `[C, C]` segment
sums and Gram matrix and its `[P, N]` state through HBM; the kernel keeps
them in one program and writes only `y`, the final state and, for a
gradient, each chunk's entering state. Mosaic's grid is ordered and a
revisited window survives a step, so `grid=(batch, heads, chunks)` runs the
chunk innermost with the state in the shared output window. It takes the
operands as the XLA scan does, `[B, T, H, ...]` (`_chunk_scan`); the
heads go ahead of the time inside it, the one layout Mosaic blocks.

The backward is the hand-written reverse recurrence, `dS_n = exp(acs_C-1)
dS_n+1 + dy^T (C exp(acs))`, since autodiff would keep every chunk's
intermediates. The XLA path is the oracle and the scan for CPU and refused
geometries; a GPU whose target row names it runs mamba_ssm's own kernels
(`dew.nn.kernels.mamba_ssd`), else the XLA path. tests/test_ssd_kernel.py
holds the kernel to `chunk_ssd` and its `jax.grad`.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.typing import DTypeLike

from dew.nn import derivatives
from dew.nn.aligned import aligned_cotangents
from dew.nn.kernels import launches

from ..precision import exact_float32
from . import generation

MIN_WIDTH = 8
"""The narrowest chunk, head width or state width the kernel is chosen for.
Below it a tile is mostly padding: 8 is Mosaic's sublane count."""

MIN_CHUNK = 64
"""The shortest chunk worth a program. The intra-chunk term is `C` times the
work of the carry, so under 64 a program spends more on its own state
recurrence than on the matmul it exists for and XLA's batched einsum wins."""

TPU_PROGRAM_WORDS = 1 << 20
"""The fp32 words one Mosaic program may hold, 4 MiB. Its operands live in VMEM,
128 MiB of it per core, so what bounds the window here is the double buffering
the pipeline runs around it rather than the window itself."""

EXACT = jax.lax.Precision.HIGHEST
"""The precision of the two matmuls that build and unbuild the segment sums.
Their operands are the chunk's `A dt` against a 0/1 matrix, so a pass in
bf16 or tf32 would round the exponent itself, and `exp` would carry that
rounding into every decay the chunk applies. The matmuls over the operands
take their dtype's precision (`_mm`)."""


def _program_words(chunk_size: int, head_dim: int, state_size: int) -> int:
    """The fp32 words one program holds: a chunk's `[C, C]` segment sums, its
    `[C, P]` input block, its two `[C, N]` blocks and the `[P, N]` state."""
    return (chunk_size * chunk_size + chunk_size * (head_dim + 2 * state_size)
            + head_dim * state_size)


def ssd_kernel_runs(chunk_size: int, head_dim: int, state_size: int, backend: str, *,
                    dtype: DTypeLike) -> bool:
    """Return whether the SSD kernel is chosen for this geometry and the scan's `dtype`.

    The kernel needs a tpu backend, a float32 scan (a wider scan takes the
    XLA path in its own dtype), a chunk long enough to pay for a program,
    three power-of-two widths so that Mosaic's tiling wastes no lanes, and a
    tile inside the per-program budget. This kernel's one-program form ran 6x
    to 12x slower than XLA as Triton on an RTX 4080 and overflowed shared
    memory at chunks of 128 and 256 (docs/performance.md); a GPU runs
    mamba_ssm's decomposition into fourteen kernels (`dew.nn.kernels.mamba_ssd`).
    """
    if backend != 'tpu' or jnp.dtype(dtype) != jnp.float32:
        return False
    widths = (chunk_size, head_dim, state_size)
    if any(width < MIN_WIDTH or width & (width - 1) for width in widths):
        return False
    return chunk_size >= MIN_CHUNK and _program_words(*widths) <= TPU_PROGRAM_WORDS


def ssd_kernel_platform(chunk_size: int, head_dim: int, state_size: int, *,
                        dtype: DTypeLike) -> str | None:
    """Return the backend to build this `dtype` scan's kernel for, or None for the XLA path.

    The scan's route is the selector's to report (`recurrence_chunk.SCAN`, `generation`'s Route
    collection), where the kernel report reads it (row 971)."""
    backend = generation.backend()
    return backend if ssd_kernel_runs(chunk_size, head_dim, state_size, backend, dtype=dtype) else None


def _triangles(size: int):
    """`below[i, k] = i >= k` and `above[k, j] = k > j`, the two triangles a
    chunk's segment sums are a product of."""
    rows = jax.lax.broadcasted_iota(jnp.int32, (size, size), 0)
    cols = jax.lax.broadcasted_iota(jnp.int32, (size, size), 1)
    return rows >= cols, rows > cols


def _segment_sums(a, below, above):
    """`out[i, j] = sum_{j < k <= i} a[k]`, `segment_sum` (the XLA path's
    spelling) as the matmul `(below * a) @ above`.

    Each entry sums its own range, which is what the accuracy here rests on:
    the difference `acs_i - acs_j` of two cumulative sums loses the chunk's
    whole accumulated magnitude to cancellation, and `exp` turns that into a
    relative error on every decay. Measured over tests/fixtures/mamba2, the
    difference form sits 3.3e-05 from the XLA path where this sits 1.9e-06.
    """
    spread = jnp.where(below, a[None, :], 0.0)
    return spread, jax.lax.dot_general(spread, above.astype(spread.dtype),
                                       (((1,), (0,)), ((), ())), precision=EXACT)


def _bottom(size: int):
    """`mask[i, j] = (i == C - 1)`, the bottom row of a `[C, C]` tile. Reading
    and writing that row is a masked reduction here because neither backend
    lowers a slice of a value."""
    return jax.lax.broadcasted_iota(jnp.int32, (size, size), 0) == jnp.int32(size - 1)


def _last_row(matrix):
    """The bottom row of a `[C, C]` matrix: `acs_C-1 - acs_j`, the decay from
    every step of the chunk to the state leaving it."""
    return jnp.sum(jnp.where(_bottom(matrix.shape[0]), matrix, 0.0), axis=0)


def _mm(lhs, rhs):
    """`lhs @ rhs`, the chunk's fp32 operands multiplied in fp32 where nothing states a precision
    (`precision.exact_float32`), where a TPU's Pallas matmul at JAX's default takes one bf16 pass, since a
    chunk's products carry the recurrent state, whose rounding every later chunk reads compounded
    (template numerics review 2026-10-26 follow-up, P1-1: every Mamba-2 layer's scan on a TPU)."""
    return jnp.matmul(lhs, rhs, precision=exact_float32(None))


def _chunk_forward(x, b, c, a, state):
    """One chunk in fp32: `x` `[C, P]` already scaled by `dt`, `b` and `c`
    `[C, N]`, `a` `[C]` the per-step `A dt`, `state` `[P, N]` entering it.
    Returns the chunk's output `[C, P]` and the state leaving it."""
    below, above = _triangles(a.shape[0])
    spread, segments = _segment_sums(a, below, above)
    decay = jnp.where(below, jnp.exp(segments), 0.0)
    entered = jnp.exp(jnp.sum(spread, axis=-1))
    leaving, alpha = jnp.exp(_last_row(segments)), jnp.exp(jnp.sum(a))
    y = _mm(_mm(c, b.T) * decay, x) + entered[:, None] * _mm(c, state.T)
    return y, alpha * state + _mm(x.T, b * leaving[:, None])


def _chunk_backward(x, b, c, a, state, dy, dnext):
    """The reverse of `_chunk_forward` at one chunk. `dy` `[C, P]` and `dnext`
    `[P, N]` are the gradients of the chunk's output and of the state leaving
    it; returns those of `x`, `b`, `c`, `a` and of the state that entered."""
    below, above = _triangles(a.shape[0])
    spread, segments = _segment_sums(a, below, above)
    decay = jnp.where(below, jnp.exp(segments), 0.0)
    entered = jnp.exp(jnp.sum(spread, axis=-1))
    leaving, alpha = jnp.exp(_last_row(segments)), jnp.exp(jnp.sum(a))

    gram, outer = _mm(c, b.T), _mm(dy, x.T)
    dgram, written = outer * decay, _mm(x, dnext)
    dx = _mm((gram * decay).T, dy) + _mm(b * leaving[:, None], dnext.T)
    db = _mm(dgram.T, c) + written * leaving[:, None]
    dc = _mm(dgram, b) + entered[:, None] * _mm(dy, state)
    dstate = _mm((dy * entered[:, None]).T, c) + alpha * dnext

    dleaving = jnp.sum(written * b, axis=1) * leaving
    dsegments = (jnp.where(below, outer * gram * decay, 0.0)
                 + jnp.where(_bottom(below.shape[0]), dleaving[None, :], 0.0))
    dspread = jax.lax.dot_general(dsegments, above.astype(dy.dtype), (((1,), (1,)), ((), ())),
                                  precision=EXACT)
    # Every step of the chunk feeds `acs` and the chunk's total alike, so what
    # reaches those two spreads back over the whole of `a` rather than a step.
    dspread = dspread + (jnp.sum(dy * _mm(c, state.T), axis=1) * entered)[:, None]
    da = jnp.sum(jnp.where(below, dspread, 0.0), axis=0) + jnp.sum(dnext * state) * alpha
    return dx, db, dc, da, dstate


def _tpu_forward(x_ref, b_ref, c_ref, a_ref, state_ref, y_ref, final_ref, *saved):
    """`grid=(batch, heads, chunks)` with the chunk innermost. `final_ref`'s
    window does not move with the chunk, so it carries the state as well as
    returning it; on the chunk that opens a head it holds nothing yet, and the
    state the caller passed is read instead."""
    opening = pl.program_id(2) == 0
    carried = jnp.where(opening, state_ref[...], final_ref[...])
    for entered_ref in saved:
        entered_ref[...] = carried
    y, leaving = _chunk_forward(x_ref[...], b_ref[...], c_ref[...], a_ref[...], carried)
    y_ref[...] = y
    final_ref[...] = leaving


def _tpu_backward(x_ref, b_ref, c_ref, a_ref, entered_ref, dy_ref, dfinal_ref, dx_ref, db_ref, dc_ref, da_ref,
                  dstate_ref):
    """`_tpu_forward` read backwards. The grid still counts up; the index maps
    hand it the chunks in reverse, and `dstate_ref` carries as `final_ref`
    did."""
    opening = pl.program_id(2) == 0
    dnext = jnp.where(opening, dfinal_ref[...], dstate_ref[...])
    dx, db, dc, da, dstate = _chunk_backward(
        x_ref[...], b_ref[...], c_ref[...], a_ref[...], entered_ref[...], dy_ref[...], dnext)
    dx_ref[...], db_ref[...], dc_ref[...], da_ref[...] = dx, db, dc, da
    dstate_ref[...] = dstate


def _marked(loads_ref, t):
    """Whether token `t` of a chunk opens one of its segments: its first, or a document's first (its `load`
    past -1, `recurrence_chunk.Chunk.loads`), read from the chunk's loads in scalar memory."""
    return (t == jnp.int32(0)) | (loads_ref[0, t] >= jnp.int32(0))


def _segment_count(loads_ref, count: int):
    """The segments (documents) opening inside a chunk of `count` tokens (row 9078)."""
    return jax.lax.fori_loop(jnp.int32(0), jnp.int32(count),
                             lambda t, n: n + _marked(loads_ref, t).astype(jnp.int32), jnp.int32(0))


def _segment_after(loads_ref, origin, count: int):
    """The token after the segment opening at `origin`: the next token opening one, else the chunk's end."""
    end = jnp.int32(count)

    def found(t, after):
        return jnp.where((after == end) & _marked(loads_ref, t), t, after)

    return jax.lax.fori_loop(origin + jnp.int32(1), end, found, end)

def _windowed(x, b, c, a, origin, through):
    """A chunk's operands with every token outside `[origin, through]` the identity step: no input, read or
    write, a unit decay (`a` zero), as the portable math's window (`scan.Derived.prepare`'s `window`)."""
    rows = jax.lax.broadcasted_iota(jnp.int32, (x.shape[0], 1), 0)
    steps = jax.lax.broadcasted_iota(jnp.int32, a.shape, a.ndim - 1)
    inside = (rows >= origin) & (rows <= through)
    held = inside.astype(x.dtype)
    return (x * held, b * held, c * held, jnp.where((steps >= origin) & (steps <= through), a, 0.0)), inside


def _slot(given_ref, table, at, zero):
    """The memory at one-based table slot `at`: `given_ref`'s first, then the stored `table`'s; zero at slot
    zero."""
    count = given_ref.shape[0]
    taken = zero
    for slot in range(count):
        taken = jnp.where(at == jnp.int32(slot + 1), given_ref[slot], taken)
    if table is not None:
        for slot in range(table.shape[0]):
            taken = jnp.where(at == jnp.int32(count + slot + 1), table[slot], taken)
    return taken


def _opened(given_ref, table, entry, load, add, document, zero):
    """The memory a document opening at a token enters: its one-based table slot's (`_slot`) and the slot
    it adds (`add`, `Chunk.adds`: zero, none), its entry write added where the call gives one (`entry`: each
    document's keys `[R, N]`, values `[P, R]` and weight `[1, 1]`, a term's three factors at a time,
    `precision.twice_scaled`)."""
    taken = _slot(given_ref, table, load, zero) + _slot(given_ref, table, add, zero)
    if entry is None:
        return taken
    # Imported here: the kernels' package imports this module, and the launch's module reaches the run's
    # stated values (`dew._stated`) through its tuning, which a test imports before the package.
    from dew.nn.kernels.chunked_launch import entry_write

    key, value, weight = (ref[document] for ref in entry)
    # Its terms' outer products summed aligned to the largest, which may leave fp32's range where the
    # sum is in it (`precision.aligned_sum`).
    return taken + entry_write(key, value, weight, values_first=True)


def _tpu_forked_forward(loads_ref, stores_ref, begun_ref, adds_ref, x_ref, b_ref, c_ref, a_ref, state_ref,
                        given_ref, *refs, held: int, entered: bool, keep: bool):
    """A forked call's chunk (`_forked_forward`), its documents walked in place: the chunk's segments in turn,
    a segment a document's tokens within it, each from the memory entering it, the chunk's own where its
    document continued from the chunk before, else its document's, taken up from its table at its first token
    (`_opened`, the stored slots as the segments before left them, so a document stores and a later one of the
    same chunk loads); each its window of the chunk (`_windowed`, `_chunk_forward`), its state left in its
    document's stored slot where the segment ends a storing document, and carried where it ends the chunk.
    Each token's load, store and document are the chunk's in scalar memory, its segments' bounds read from
    them."""
    entry = refs[:3] if entered else None
    y_ref, final_ref, *outputs = refs[3:] if entered else refs
    table = outputs.pop(0) if held else None
    count = x_ref.shape[0]
    opening = pl.program_id(2) == 0
    carried = jnp.where(opening, state_ref[...], final_ref[...])
    if table is not None:
        @pl.when(opening)
        def _():
            table[...] = jnp.zeros(table.shape, table.dtype)
    if keep:
        outputs[0][...] = carried
    x, b, c, a = x_ref[...], b_ref[...], c_ref[...], a_ref[...]
    segments = _segment_count(loads_ref, count)

    def segment(_, state):
        y, leaving, origin = state
        after = _segment_after(loads_ref, origin, count)
        through = after - jnp.int32(1)
        load = loads_ref[0, origin]
        memory = jnp.where(load < jnp.int32(0), carried, _opened(given_ref, table, entry, load,
                                                                 adds_ref[0, origin], begun_ref[0, origin],
                                                                 jnp.zeros_like(carried)))
        operands, inside = _windowed(x, b, c, a, origin, through)
        out, left = _chunk_forward(*operands, memory)
        if table is not None:
            store = stores_ref[0, through]
            for slot in range(held):
                table[slot] = jnp.where(store == jnp.int32(slot + 1), left, table[slot])
        return jnp.where(inside, out, y), jnp.where(through == jnp.int32(count - 1), left, leaving), after

    y, leaving, _ = jax.lax.fori_loop(jnp.int32(0), segments, segment,
                                      (jnp.zeros(y_ref.shape, jnp.float32), carried, jnp.int32(0)))
    y_ref[...] = y
    final_ref[...] = leaving


def _tpu_forked_backward(loads_ref, stores_ref, begun_ref, adds_ref, x_ref, b_ref, c_ref, a_ref, entered_ref,
                         dy_ref, dfinal_ref, given_ref, *refs, held: int, entered: bool):
    """`_tpu_forked_forward` read backwards, the chunks in reverse and each chunk's segments last first, the
    table the forward's last (`table`: each slot stored once, before every document that loads it): a
    segment's gradients from its window (`_chunk_backward`), the state it left's its slot's gradient where it
    stored one (each later document that took it up's) and the next chunk's where it ends the chunk; the
    gradient of the memory it entered the chunk before's where its document continued, else its slot's
    (`dgiven` for a given memory's, `dtable` for a stored one's) and its entry write's operands'."""
    table, *refs = refs if held else (None, *refs)
    entry = refs[:3] if entered else None
    refs = refs[3:] if entered else refs
    dx_ref, db_ref, dc_ref, da_ref, dstate_ref, dgiven, *outputs = refs
    dtable = outputs.pop(0) if held else None
    dentry = outputs if entered else None
    count = x_ref.shape[0]
    opening = pl.program_id(2) == 0

    @pl.when(opening)
    def _():
        for ref in (dgiven, *(() if dtable is None else (dtable,)), *(dentry or ())):
            ref[...] = jnp.zeros(ref.shape, ref.dtype)

    dnext = jnp.where(opening, dfinal_ref[...], dstate_ref[...])
    x, b, c, a, dy, start = x_ref[...], b_ref[...], c_ref[...], a_ref[...], dy_ref[...], entered_ref[...]
    segments = _segment_count(loads_ref, count)

    def origin_of(k):
        def counted(t, state):
            seen, origin = state
            marked = _marked(loads_ref, t)
            return seen + marked.astype(jnp.int32), jnp.where(marked & (seen == k), t, origin)

        return jax.lax.fori_loop(jnp.int32(0), jnp.int32(count), counted, (jnp.int32(0), jnp.int32(0)))[1]

    def segment(i, state):
        dx, db, dc, da, dprev = state
        k = segments - jnp.int32(1) - i
        origin = origin_of(k)
        through = _segment_after(loads_ref, origin, count) - jnp.int32(1)
        load, add = loads_ref[0, origin], adds_ref[0, origin]
        document = begun_ref[0, origin]
        memory = jnp.where(load < jnp.int32(0), start, _opened(given_ref, table, entry, load, add, document,
                                                               jnp.zeros_like(start)))
        dleft = jnp.where(through == jnp.int32(count - 1), dnext, 0.0)
        if dtable is not None:
            store = stores_ref[0, through]
            for slot in range(held):
                stored = store == jnp.int32(slot + 1)
                dleft = dleft + jnp.where(stored, dtable[slot], 0.0)
                dtable[slot] = jnp.where(stored, 0.0, dtable[slot])
        operands, inside = _windowed(x, b, c, a, origin, through)
        steps = jax.lax.broadcasted_iota(jnp.int32, a.shape, a.ndim - 1)
        within = (steps >= origin) & (steps <= through)
        wx, wb, wc, wa, dmemory = _chunk_backward(*operands, memory, jnp.where(inside, dy, 0.0), dleft)
        count_given = given_ref.shape[0]
        # The slot a document loads and the one it adds (zero, none) each take its memory's gradient.
        for at in (load, add):
            for slot in range(count_given):
                dgiven[slot] = dgiven[slot] + jnp.where(at == jnp.int32(slot + 1), dmemory, 0.0)
            if dtable is not None:
                for slot in range(held):
                    dtable[slot] = dtable[slot] + jnp.where(at == jnp.int32(count_given + slot + 1), dmemory,
                                                            0.0)
        if entry is not None and dentry is not None:
            # Formed only where the segment opens its document: a continuation's memory gradient is the
            # chunk's, against which a term's products left fp32's range where the document's are in it,
            # and masked after, inf times zero gave every key gradient NaN (base review 13abb0d6b F1).
            entries, gradients = entry, dentry

            @pl.when(load >= jnp.int32(0))
            def _():
                from dew.nn.kernels.chunked_launch import entry_terms

                key, value, weight = (ref[document] for ref in entries)
                dkey, dvalue, dweight = gradients
                # The entry write's transpose in its own layout (`entry_terms`), each distinct factor's
                # cotangent in order of first appearance (`aligned_cotangents`): the shared weight's one, its
                # terms' products summed aligned before they round (each term's rounded apart, two past
                # fp32's largest that cancel gave inf - inf), then each term's value column's and key row's.
                # A write of rank zero, none: its gradients none, its weight's zero (row 8967).
                if terms := entry_terms(key, value, weight, values_first=True):
                    weighed, *factors = aligned_cotangents(terms, dmemory)
                    dkey[document] = dkey[document] + jnp.concatenate(factors[1::2], axis=0)
                    dvalue[document] = dvalue[document] + jnp.concatenate(factors[0::2], axis=1)
                    dweight[document] = dweight[document] + weighed
        return (jnp.where(inside, dx + wx, dx), jnp.where(inside, db + wb, db),
                jnp.where(inside, dc + wc, dc), jnp.where(within, da + wa, da),
                jnp.where(load < jnp.int32(0), dmemory, dprev))

    zero = jnp.zeros
    dx, db, dc, da, dprev = jax.lax.fori_loop(
        jnp.int32(0), segments, segment,
        (zero(x.shape, jnp.float32), zero(b.shape, jnp.float32), zero(c.shape, jnp.float32),
         zero(a.shape, jnp.float32), zero(start.shape, jnp.float32)))
    dx_ref[...], db_ref[...], dc_ref[...], da_ref[...], dstate_ref[...] = dx, db, dc, da, dprev


def _specs(chunks: int, chunk_size: int, head_dim: int, state_size: int, *,
           reverse: bool = False):
    """The block specs of the arrays a chunk reads or writes, in the kernel's
    own layout: `[B, H, T, P]` for `x` and `y`, `[B, H, T, N]` for `b` and
    `c`, each chunk `chunk_size` steps of the time, `[B, H, NC, 1, C]` for
    `a`, `[B, H, NC, P, N]` for the state each chunk entered, and `[B, H, P, N]`
    for the state crossing the sequence. Returned in that order; a forked
    call's tokens (`_tokens_spec`) and whole tables (`_whole`) have their own.

    Mosaic blocks the last two dimensions or nothing, which is what puts the
    heads ahead of the time here (`_heads_first`), and `a`'s chunk in a
    dimension of its own. A program holds one chunk and the grid counts them,
    backwards when `reverse`.
    """
    def position(grid: tuple[int, ...]) -> tuple[int, int, int]:
        return grid[0], grid[1], chunks - 1 - grid[2] if reverse else grid[2]

    def indices(*values):
        # int32 whatever the process's integer width: under x64 a Python
        # int traces as int64, and Mosaic cannot return one from an index map.
        return tuple(jnp.asarray(value, jnp.int32) for value in values)

    def over_time(*grid):
        batch, head, chunk = position(grid)
        return indices(batch, head, chunk, 0)

    def over_chunk(*grid):
        batch, head, chunk = position(grid)
        return indices(batch, head, chunk, 0, 0)

    def over_carry(*grid):
        batch, head, _ = position(grid)
        return indices(batch, head, 0, 0)

    return (pl.BlockSpec((None, None, chunk_size, head_dim), over_time),
            pl.BlockSpec((None, None, chunk_size, state_size), over_time),
            pl.BlockSpec((None, None, chunk_size, state_size), over_time),
            pl.BlockSpec((None, None, None, None, chunk_size), over_chunk),
            pl.BlockSpec((None, None, None, head_dim, state_size), over_chunk),
            pl.BlockSpec((None, None, head_dim, state_size), over_carry))


def _heads_first(t):
    """`[B, T, H, ...] <-> [B, H, T, ...]`, the layout Mosaic can block."""
    return jnp.swapaxes(t, 1, 2)


def _decay_blocks(a, chunk_size: int):
    """`a` `[B, T, H]` as `[B, H, NC, 1, C]`, a chunk in each of its last two dimensions."""
    batch, total, heads = a.shape
    return _heads_first(a).reshape(batch, heads, total // chunk_size, 1, chunk_size)


def _interpreting(platform: str) -> bool:
    """Whether pallas runs its own interpreter instead of a Mosaic kernel:
    wherever the call's code is chosen for another platform than the kernel's
    (`generation.backend`). That is how CPU reads the kernel's numbers, while a
    step traced for an assumed TPU (`generation.assumed`) lowers the kernel
    itself on any host."""
    return generation.backend() != platform


def _call(body, name: str, platform: str, grid: tuple[int, ...], in_specs, out_specs, out_shape):
    """One Mosaic `pallas_call` named `name`, interpreted where the process holds no TPU."""
    params = pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary"))
    return launches.pallas_call(body, grid=grid, in_specs=list(in_specs), out_specs=list(out_specs),
                                out_shape=out_shape, compiler_params=params,
                                interpret=_interpreting(platform), name=name)


def _forked_blocks(forked, heads: int, chunk_size: int):
    """`_ssd_scan`'s `forked` `(given [B, S', H, P, N], tokens (loads, stores, documents [B, T] int32),
    entry (keys [B, D, R, H, N], values [B, D, R, H, P], weights [B, D, H]) or None)` in the kernels'
    layout: each token's load, store and document a chunk's row `[B, NC, 1, C]` int32 (`_tokens_spec`), the
    memories `[B, H, S', P, N]` and the entry write's operands each head's, `[B, H, D, R, N]` keys,
    `[B, H, D, P, R]` values (a term a column) and `[B, H, D, 1, 1]` weights, float32."""
    given, tokens, entry = forked
    batch, total = tokens[0].shape
    chunked = tuple(jnp.asarray(t, jnp.int32).reshape(batch, total // chunk_size, 1, chunk_size)
                    for t in tokens)
    memories = jnp.swapaxes(given, 1, 2).astype(jnp.float32)
    if entry is None:
        return chunked, memories, ()
    key, value, weight = entry
    return chunked, memories, (jnp.moveaxis(key, 3, 1).astype(jnp.float32),
                               jnp.moveaxis(jnp.swapaxes(value, 2, 4), 3, 1).astype(jnp.float32),
                               jnp.moveaxis(weight, 2, 1).astype(jnp.float32)[..., None, None])


def _tokens_spec(chunks: int, chunk_size: int, *, reverse: bool = False):
    """The block spec of a chunk's tokens' loads, stores or documents (`_forked_blocks`), in scalar memory,
    which the forked kernels read a token at a time: a row of one chunk's tokens `[1, C]`, the array's own
    last two dimensions, as the TPU lowering takes a block in scalar memory (a squeezed second-to-last
    dimension it refuses)."""
    def over_chunk(*grid):
        chunk = chunks - 1 - grid[2] if reverse else grid[2]
        return tuple(jnp.asarray(value, jnp.int32) for value in (grid[0], chunk, 0, 0))

    return pl.BlockSpec((None, None, 1, chunk_size), over_chunk, memory_space=pltpu.SMEM)


def _whole(shape: tuple[int, ...]):
    """The block spec of a head's whole `shape` (an entry write's operands, a table), not moving with the
    chunk."""
    def over_carry(*grid):
        return tuple(jnp.asarray(value, jnp.int32) for value in (grid[0], grid[1], *(0,) * len(shape)))

    return pl.BlockSpec((None, None, *shape), over_carry)


def _forward(x, b, c, a, state, chunk_size: int, platform: str, *, keep_entered: bool, forked=None,
             held: int = 0):
    """The forward kernel. Operands arrive as `_chunk_scan` takes them and
    the output leaves so; `keep_entered` also returns the state every chunk
    entered with, which is the residual the backward pass reads; a forked call's
    documents are walked in place (`forked`, `held` stored, `_tpu_forked_forward`),
    its stored slots returned beside."""
    batch, total, heads, head_dim = x.shape
    chunks, state_size = total // chunk_size, b.shape[-1]
    x_spec, b_spec, c_spec, a_spec, entered_spec, carry_spec = _specs(chunks, chunk_size, head_dim,
                                                                      state_size)
    chunked = jax.ShapeDtypeStruct((batch, heads, chunks, head_dim, state_size), jnp.float32)
    out_shape = [jax.ShapeDtypeStruct((batch, heads, total, head_dim), jnp.float32),
                 jax.ShapeDtypeStruct(state.shape, jnp.float32)]
    out_specs = [x_spec, carry_spec]
    operands = (_heads_first(x), _heads_first(b), _heads_first(c), _decay_blocks(a, chunk_size), state)
    in_specs = (x_spec, b_spec, c_spec, a_spec, carry_spec)
    body = _tpu_forward
    if forked is not None:
        tokens, given, entry = _forked_blocks(forked, heads, chunk_size)
        operands = (*tokens, *operands, given, *entry)
        in_specs = ((_tokens_spec(chunks, chunk_size),) * 4 + in_specs
                    + tuple(_whole(part.shape[2:]) for part in (given, *entry)))
        if held:
            out_shape.append(jax.ShapeDtypeStruct((batch, heads, held, head_dim, state_size), jnp.float32))
            out_specs.append(_whole((held, head_dim, state_size)))
        body = functools.partial(_tpu_forked_forward, held=held, entered=bool(entry), keep=keep_entered)
    if keep_entered:
        out_shape.append(chunked)
        out_specs.append(entered_spec)
    y, final, *rest = _call(body, "ssd_forward", platform, (batch, heads, chunks), in_specs, out_specs,
                            out_shape)(*operands)
    return (_heads_first(y), final, *rest)


def _backward(x, b, c, a, states, dy, dfinal, chunk_size: int, platform: str, forked=None, table=None,
              held: int = 0):
    """The backward kernel, one reverse pass over the same blocks from the states each chunk entered
    (`states`); a forked call's (`forked`, its stored slots `table` the forward's, `held` of them) also gives
    its given memories' gradients `[B, S', H, P, N]` and its entry write's operands' where it gives them
    (`_tpu_forked_backward`)."""
    batch, total, heads, head_dim = x.shape
    chunks, state_size = total // chunk_size, b.shape[-1]
    x_spec, b_spec, c_spec, a_spec, entered_spec, carry_spec = _specs(
        chunks, chunk_size, head_dim, state_size, reverse=True)
    widths = (head_dim, state_size, state_size)
    out_shape = [jax.ShapeDtypeStruct((batch, heads, total, width), jnp.float32) for width in widths]
    out_shape.append(jax.ShapeDtypeStruct((batch, heads, chunks, 1, chunk_size), jnp.float32))
    out_shape.append(jax.ShapeDtypeStruct(dfinal.shape, jnp.float32))
    out_specs = [x_spec, b_spec, c_spec, a_spec, carry_spec]
    operands = (_heads_first(x), _heads_first(b), _heads_first(c), _decay_blocks(a, chunk_size), states,
                _heads_first(dy), dfinal)
    in_specs = (x_spec, b_spec, c_spec, a_spec, entered_spec, x_spec, carry_spec)
    body, entry = _tpu_backward, ()
    if forked is not None:
        tokens, given, entry = _forked_blocks(forked, heads, chunk_size)
        stored = () if table is None else (table,)
        operands = (*tokens, *operands, given, *stored, *entry)
        in_specs = ((_tokens_spec(chunks, chunk_size, reverse=True),) * 4 + in_specs
                    + tuple(_whole(part.shape[2:]) for part in (given, *stored, *entry)))
        for part in (given, *stored, *entry):
            out_shape.append(jax.ShapeDtypeStruct(part.shape, jnp.float32))
            out_specs.append(_whole(part.shape[2:]))
        body = functools.partial(_tpu_forked_backward, held=held, entered=bool(entry))
    dx, db, dc, da, dstate, *dforked = _call(body, "ssd_backward", platform, (batch, heads, chunks), in_specs,
                                             out_specs, out_shape)(*operands)
    if forked is not None:
        # The stored slots' gradients are the walk's own (each joined its storing segment's).
        dforked = [dforked[0], *dforked[1 + bool(held):]]
    return (_heads_first(dx), _heads_first(db), _heads_first(dc),
            _heads_first(da.reshape(batch, heads, total)), dstate, *dforked)


def _ssd_scan(x, b, c, a, state, forked, chunk_size: int, platform: str,
              held: int = 0) -> tuple[jax.Array, ...]:
    """Run the chunked SSD scan on the Pallas kernel for `platform`.

    Operands arrive as `xla_chunk_scan` takes them, all fp32: `x`
    `[B, T, H, P]` already scaled by `dt`, `b` and `c` `[B, T, H, N]` with the
    groups expanded, `a` `[B, T, H]` the per-step `A dt`, the length whole
    chunks of `chunk_size`, `state` `[B, H, P, N]` entering the sequence, and
    `forked` None or a forked call's `(given [B, S', H, P, N] float32, tokens
    (loads, stores, documents [B, T] int32, `chunked_launch.Forked.tokens`),
    entry (keys [B, D, R, H, N], values [B, D, R, H, P], weights [B, D, H]) or
    None)`, its documents walked in place, each taking its memory up at its
    first token with its entry write added (`recurrence_chunk.Chunk.loads`,
    `held` stored slots). Returns the output `[B, T, H, P]` and the final
    state, the same pair `xla_chunk_scan` returns.
    """
    return tuple(_forward(x, b, c, a, state, chunk_size, platform, keep_entered=False, forked=forked,
                          held=held)[:2])


def _ssd_portable_forked(x, b, c, a, state, forked, chunk_size: int, held: int):
    """Pure-JAX forked chunk scan reusing `_chunk_forward`, `_windowed`, `_opened`, and `_segment_count`
    with no cross-layer import (rows 7874, 7875; base review e488dc198 F2, F5)."""
    batch, total, heads, head_dim = x.shape
    chunks, state_size = total // chunk_size, b.shape[-1]
    tokens, given, entry = _forked_blocks(forked, heads, chunk_size)
    xs = jnp.moveaxis(x.reshape(batch, chunks, chunk_size, heads, head_dim), (1, 3), (2, 1))
    bs = jnp.moveaxis(b.reshape(batch, chunks, chunk_size, heads, state_size), (1, 3), (2, 1))
    cs = jnp.moveaxis(c.reshape(batch, chunks, chunk_size, heads, state_size), (1, 3), (2, 1))
    as_ = jnp.moveaxis(a.reshape(batch, chunks, chunk_size, heads), (1, 3), (2, 1))

    def walk_head(x_h, b_h, c_h, a_h, st_h, given_h, entry_h, tok_b):
        loads_b, stores_b, begun_b, adds_b = tok_b
        table0 = jnp.zeros((held, head_dim, state_size), st_h.dtype) if held else None

        def chunk_step(carry, inputs):
            carried, table = carry
            cx, cb, cc, ca, cl, cs_, cbeg, cadd = inputs
            segments = _segment_count(cl, chunk_size)

            def seg_step(_, seg_state):
                y_acc, leaving, origin, tbl = seg_state
                after = _segment_after(cl, origin, chunk_size)
                through = after - jnp.int32(1)
                load = cl[0, origin]
                mem = jnp.where(load < 0, carried,
                                _opened(given_h, tbl, entry_h, load, cadd[0, origin], cbeg[0, origin],
                                        jnp.zeros_like(carried)))
                operands, inside = _windowed(cx, cb, cc, ca, origin, through)
                out, left = _chunk_forward(*operands, mem)
                if tbl is not None:
                    store = cs_[0, through]
                    for slot in range(held):
                        tbl = tbl.at[slot].set(jnp.where(store == slot + 1, left, tbl[slot]))
                return (jnp.where(inside, out, y_acc), jnp.where(through == chunk_size - 1, left, leaving),
                        after, tbl)

            y_c, left_c, _, tbl_out = jax.lax.fori_loop(
                0, segments, seg_step,
                (jnp.zeros_like(cx), carried, jnp.int32(0), table))
            return (left_c, tbl_out), y_c

        (final_h, _), ys_h = jax.lax.scan(
            chunk_step, (st_h, table0), (x_h, b_h, c_h, a_h, loads_b, stores_b, begun_b, adds_b))
        return ys_h, final_h

    def walk_row(x_r, b_r, c_r, a_r, st_r, given_r, entry_r, tok_r):
        return jax.vmap(walk_head, in_axes=(0, 0, 0, 0, 0, 0, 0 if entry_r is not None else None, None))(
            x_r, b_r, c_r, a_r, st_r, given_r, entry_r, tok_r)

    entry_in = entry if entry else None
    ys, final = jax.vmap(walk_row)(xs, bs, cs, as_, state, given, entry_in, tokens)
    return jnp.moveaxis(ys.reshape(batch, heads, total, head_dim), 1, 2), final


def _ssd_portable(x, b, c, a, state, forked, chunk_size: int, held: int) -> tuple[jax.Array, jax.Array]:
    """Pure-JAX chunked SSD scan reusing `_chunk_forward` (plain and forked) so forward mode and higher-order
    derivatives through `_chunk_scan` combine reads and decays before forming any carry with no cross-layer
    import (rows 7874, 7875; base review e488dc198 F2, F5)."""
    if forked is not None:
        return _ssd_portable_forked(x, b, c, a, state, forked, chunk_size, held)
    batch, total, heads, head_dim = x.shape
    chunks, state_size = total // chunk_size, b.shape[-1]
    xs = jnp.moveaxis(x.reshape(batch, chunks, chunk_size, heads, head_dim), (1, 3), (0, 2))
    bs = jnp.moveaxis(b.reshape(batch, chunks, chunk_size, heads, state_size), (1, 3), (0, 2))
    cs = jnp.moveaxis(c.reshape(batch, chunks, chunk_size, heads, state_size), (1, 3), (0, 2))
    as_ = jnp.moveaxis(a.reshape(batch, chunks, chunk_size, heads), (1, 3), (0, 2))

    def step(carry, chunk_args):
        cx, cb, cc, ca = chunk_args
        cy, nxt = jax.vmap(jax.vmap(_chunk_forward))(cx, cb, cc, ca, carry)
        return nxt, cy

    final, ys = jax.lax.scan(step, state, (xs, bs, cs, as_))
    return jnp.moveaxis(ys, (0, 2), (1, 3)).reshape(batch, total, heads, head_dim), final


@functools.cache
def _ssd_rule(chunk_size: int, platform: str, held: int):
    """`_chunk_scan`'s rule via `generation.linear_tangent`: forward and reverse run the TPU Pallas kernels
    (`ssd_forward`, `ssd_backward`), while forward mode and higher-order derivatives differentiate through
    `_ssd_portable` (rows 7874, 7875)."""
    def primal(x, b, c, a, state, forked):
        return _ssd_scan(x, b, c, a, state, forked, chunk_size, platform, held)

    def forward(x, b, c, a, state, forked):
        y, final, *kept = _forward(x, b, c, a, state, chunk_size, platform, keep_entered=True,
                                   forked=forked, held=held)
        table, entered = (kept[0], kept[1]) if forked is not None and held else (None, kept[0])
        return (y, final), (entered, table)

    def backward(saved, operands, cotangent):
        x, b, c, a, _, forked = operands
        entered, table = saved
        dy, dfinal = cotangent
        dx, db, dc, da, dstate, *dforked = _backward(x, b, c, a, entered, dy, dfinal, chunk_size, platform,
                                                     forked=forked, table=table, held=held)
        dtaken = None
        if forked is not None:
            given, tokens, entry = forked
            dentry = None
            if entry is not None:
                dkey, dvalue, dweight = dforked[1:]
                dentry = (jnp.moveaxis(dkey, 1, 3).astype(entry[0].dtype),
                          jnp.swapaxes(jnp.moveaxis(dvalue, 1, 3), 2, 4).astype(entry[1].dtype),
                          jnp.moveaxis(dweight[..., 0, 0], 1, 2).astype(entry[2].dtype))
            no_tokens = tuple(jnp.zeros(t.shape, jax.dtypes.float0) for t in tokens)
            dtaken = (jnp.swapaxes(dforked[0], 1, 2).astype(given.dtype), no_tokens, dentry)
        return dx, db, dc, da, dstate, dtaken

    def portable(x, b, c, a, state, forked):
        return _ssd_portable(x, b, c, a, state, forked, chunk_size, held)

    return derivatives.contracted("test_the_tpu_ssd_scan_holds_its_orders")(
        generation.linear_tangent(forward, backward, portable, kind=derivatives.KernelGlue(),
                                  primal=primal))


def _chunk_scan(x, b, c, a, state, forked, chunk_size: int, platform: str,
                held: int = 0) -> tuple[jax.Array, ...]:
    """Run the chunked SSD scan on the Pallas kernel for `platform`, with every derivative order via
    `generation.linear_tangent` (rows 7874, 7875)."""
    return _ssd_rule(chunk_size, platform, held)(x, b, c, a, state, forked)
