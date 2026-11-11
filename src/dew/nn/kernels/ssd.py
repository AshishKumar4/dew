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
    return jax.lax.broadcasted_iota(jnp.int32, (size, size), 0) == size - 1


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


def _tpu_forward(x_ref, b_ref, c_ref, a_ref, state_ref, *refs, documents: bool = False, held: int = 0,
                 entered: bool = False):
    """`grid=(batch, heads, chunks)` with the chunk innermost. `final_ref`'s
    window does not move with the chunk, so it carries the state as well as
    returning it; on the chunk that opens a head it holds nothing yet, and the
    state the caller passed is read instead. A forked call's (`documents`,
    `_ssd_scan`'s): a chunk whose `load` is past -1 enters that one-based table
    slot's memory (`given`'s first, then the `held` stored; zero a zero memory)
    in place of the carried one, and one whose `store` is past zero leaves its
    memory in that stored slot (`table`, whose window, as `final_ref`'s, does
    not move with the chunk); where `entered`, a chunk taking a slot up adds its
    entry write (`entry_ref`) to the memory it takes."""
    forked = table = entry_ref = None
    if documents:
        given_ref, load_ref, store_ref, *refs = refs
        entry_ref = refs.pop(0) if entered else None
        y_ref, final_ref, *saved = refs
        table = saved.pop(0) if held else None
        forked = (given_ref, load_ref, store_ref)
    else:
        y_ref, final_ref, *saved = refs
    opening = pl.program_id(2) == 0
    carried = jnp.where(opening, state_ref[...], final_ref[...])
    if forked is not None:
        given, load_ref, _ = forked
        if table is not None:
            emptied = table

            @pl.when(opening)
            def _():
                emptied[...] = jnp.zeros(emptied.shape, emptied.dtype)
        load = load_ref[...]
        taken = jnp.zeros_like(carried)
        count = given.shape[0]
        for slot in range(count):
            taken = jnp.where(load == slot + 1, given[slot], taken)
        if table is not None:
            for slot in range(held):
                taken = jnp.where(load == count + slot + 1, table[slot], taken)
        carried = jnp.where(load >= 0, taken if entry_ref is None else taken + entry_ref[...], carried)
    for entered_ref in saved:
        entered_ref[...] = carried
    y, leaving = _chunk_forward(x_ref[...], b_ref[...], c_ref[...], a_ref[...], carried)
    y_ref[...] = y
    final_ref[...] = leaving
    if forked is not None and table is not None:
        store = forked[2][...]
        for slot in range(held):
            table[slot] = jnp.where(store == slot + 1, leaving, table[slot])


def _tpu_backward(x_ref, b_ref, c_ref, a_ref, entered_ref, dy_ref, dfinal_ref, *refs,
                  documents: bool = False, held: int = 0, entered: bool = False):
    """`_tpu_forward` read backwards. The grid still counts up; the index maps
    hand it the chunks in reverse, and `dstate_ref` carries as `final_ref`
    did. A forked call's: at a chunk that stored a slot, the slot's gradient
    (`dtable`, each later chunk that took it up's) joins the state it left; at a
    chunk that took a slot up, the gradient of the state it entered goes to that
    slot's (`dgiven` for a given memory's), and to its entry write's
    (`dentry_ref`, where `entered`), and none carries before it."""
    forked = table = dentry_ref = None
    if documents:
        load_ref, store_ref, dx_ref, db_ref, dc_ref, da_ref, dstate_ref, dgiven, *dtable = refs
        forked = (load_ref, store_ref, dgiven)
        dentry_ref = dtable.pop() if entered else None
        table = dtable[0] if held else None
    else:
        dx_ref, db_ref, dc_ref, da_ref, dstate_ref = refs
    opening = pl.program_id(2) == 0
    dnext = jnp.where(opening, dfinal_ref[...], dstate_ref[...])
    if forked is not None:
        load_ref, store_ref, dgiven = forked
        emptied = [ref for ref in (dgiven, table) if ref is not None]

        @pl.when(opening)
        def _():
            for ref in emptied:
                ref[...] = jnp.zeros(ref.shape, ref.dtype)

        if table is not None:
            store = store_ref[...]
            for slot in range(held):
                stored = store == slot + 1
                dnext = dnext + jnp.where(stored, table[slot], 0.0)
                table[slot] = jnp.where(stored, 0.0, table[slot])
    dx, db, dc, da, dstate = _chunk_backward(
        x_ref[...], b_ref[...], c_ref[...], a_ref[...], entered_ref[...], dy_ref[...], dnext)
    dx_ref[...], db_ref[...], dc_ref[...], da_ref[...] = dx, db, dc, da
    if forked is not None:
        load_ref, _, dgiven = forked
        load = load_ref[...]
        count = dgiven.shape[0]
        for slot in range(count):
            dgiven[slot] = dgiven[slot] + jnp.where(load == slot + 1, dstate, 0.0)
        if table is not None:
            for slot in range(held):
                table[slot] = table[slot] + jnp.where(load == count + slot + 1, dstate, 0.0)
        if dentry_ref is not None:
            dentry_ref[...] = jnp.where(load >= 0, dstate, 0.0)
        dstate = jnp.where(load >= 0, 0.0, dstate)
    dstate_ref[...] = dstate


def _specs(chunks: int, chunk_size: int, head_dim: int, state_size: int, *,
           reverse: bool = False):
    """The block specs of the arrays a chunk reads or writes, in the kernel's
    own layout: `[B, H, T, P]` for `x` and `y`, `[B, H, T, N]` for `b` and
    `c`, each chunk `chunk_size` steps of the time, `[B, H, NC, 1, C]` for
    `a`, `[B, H, NC, P, N]` for the state each chunk entered, `[B, H, P, N]`
    for the state crossing the sequence, and `[B, H, NC, 1, 1]` for the
    slot each chunk takes up or leaves its memory in (`documents`). Returned
    in that order; `_slots` gives a table's.

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
            pl.BlockSpec((None, None, head_dim, state_size), over_carry),
            pl.BlockSpec((None, None, None, 1, 1), over_chunk))


def _slots(slots: int, head_dim: int, state_size: int):
    """The block spec of a forked call's table of `slots` memories `[B, H, slots, P, N]` (`documents`), its
    window a head's whole table, not moving with the chunk."""
    def over_carry(*grid):
        return tuple(jnp.asarray(value, jnp.int32) for value in (grid[0], grid[1], 0, 0, 0))

    return pl.BlockSpec((None, None, slots, head_dim, state_size), over_carry)


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


def _documents_blocks(documents, heads: int):
    """`_ssd_scan`'s forked `documents` `(given [B, S', H, P, N], loads [B, NC], stores [B, NC], entries
    [B, NC, H, P, N] or None)` in the kernel's layout: the memories `[B, H, ., P, N]` float32 and the slots
    `[B, H, NC, 1, 1]` float32, the entries where given."""
    given, loads, stores, entries = documents
    batch, chunks = loads.shape

    def chunked(slots):
        return jnp.broadcast_to(jnp.asarray(slots, jnp.float32)[:, None, :, None, None],
                                (batch, heads, chunks, 1, 1))

    return (jnp.swapaxes(given, 1, 2).astype(jnp.float32), chunked(loads), chunked(stores),
            *(() if entries is None else (jnp.swapaxes(entries, 1, 2).astype(jnp.float32),)))


def _forward(x, b, c, a, state, chunk_size: int, platform: str, *, keep_entered: bool, documents=None,
             held: int = 0):
    """The forward kernel. Operands arrive as `_chunk_scan` takes them and
    the output leaves so; `keep_entered` also returns the state every chunk
    entered with, which is the residual the backward pass reads; a forked call's
    chunks take their table's memories up (`documents`, `held` stored,
    `_tpu_forward`)."""
    batch, total, heads, head_dim = x.shape
    chunks, state_size = total // chunk_size, b.shape[-1]
    x_spec, b_spec, c_spec, a_spec, entered_spec, carry_spec, slot_spec = _specs(
        chunks, chunk_size, head_dim, state_size)
    chunked = jax.ShapeDtypeStruct((batch, heads, chunks, head_dim, state_size), jnp.float32)
    out_shape = [jax.ShapeDtypeStruct((batch, heads, total, head_dim), jnp.float32),
                 jax.ShapeDtypeStruct(state.shape, jnp.float32)]
    out_specs = [x_spec, carry_spec]
    inputs = ()
    entered = documents is not None and documents[3] is not None
    if documents is not None:
        inputs = (_slots(documents[0].shape[1], head_dim, state_size), slot_spec, slot_spec,
                  *((entered_spec,) if entered else ()))
        if held:
            out_shape.append(jax.ShapeDtypeStruct((batch, heads, held, head_dim, state_size), jnp.float32))
            out_specs.append(_slots(held, head_dim, state_size))
    if keep_entered:
        out_shape.append(chunked)
        out_specs.append(entered_spec)
    run = _call(functools.partial(_tpu_forward, documents=documents is not None, held=held, entered=entered),
                "ssd_forward",
                platform, (batch, heads, chunks), (x_spec, b_spec, c_spec, a_spec, carry_spec, *inputs),
                out_specs, out_shape)
    y, final, *rest = run(_heads_first(x), _heads_first(b), _heads_first(c), _decay_blocks(a, chunk_size),
                          state, *(() if documents is None else _documents_blocks(documents, heads)))
    if documents is not None and held:
        rest.pop(0)
    return (_heads_first(y), final, *rest)


def _backward(x, b, c, a, states, dy, dfinal, chunk_size: int, platform: str, documents=None,
              held: int = 0):
    """The backward kernel, one reverse pass over the same blocks from the states each chunk entered
    (`states`); a forked call's (`documents`, `held` stored) also gives its given memories' gradients
    `[B, S', H, P, N]`, and its entry writes' `[B, NC, H, P, N]` where it gives them (`_tpu_backward`)."""
    batch, total, heads, head_dim = x.shape
    chunks, state_size = total // chunk_size, b.shape[-1]
    x_spec, b_spec, c_spec, a_spec, entered_spec, carry_spec, slot_spec = _specs(
        chunks, chunk_size, head_dim, state_size, reverse=True)
    widths = (head_dim, state_size, state_size)
    out_shape = [jax.ShapeDtypeStruct((batch, heads, total, width), jnp.float32) for width in widths]
    out_shape.append(jax.ShapeDtypeStruct((batch, heads, chunks, 1, chunk_size), jnp.float32))
    out_shape.append(jax.ShapeDtypeStruct(dfinal.shape, jnp.float32))
    out_specs = [x_spec, b_spec, c_spec, a_spec, carry_spec]
    blocks, inputs = (), ()
    entered = documents is not None and documents[3] is not None
    if documents is not None:
        given, *blocks = _documents_blocks(documents, heads)
        blocks = blocks[:2]
        inputs = (slot_spec, slot_spec)
        for slots in (given.shape[2], *((held,) if held else ())):
            out_shape.append(jax.ShapeDtypeStruct((batch, heads, slots, head_dim, state_size), jnp.float32))
            out_specs.append(_slots(slots, head_dim, state_size))
        if entered:
            out_shape.append(jax.ShapeDtypeStruct((batch, heads, chunks, head_dim, state_size), jnp.float32))
            out_specs.append(entered_spec)
    run = _call(functools.partial(_tpu_backward, documents=documents is not None, held=held, entered=entered),
                "ssd_backward",
                platform, (batch, heads, chunks),
                (x_spec, b_spec, c_spec, a_spec, entered_spec, x_spec, carry_spec, *inputs), out_specs,
                out_shape)
    dx, db, dc, da, dstate, *dforked = run(
        _heads_first(x), _heads_first(b), _heads_first(c), _decay_blocks(a, chunk_size), states,
        _heads_first(dy), dfinal, *blocks)
    taken = dforked[:1] + (dforked[len(dforked) - 1:] if entered else [])
    return (_heads_first(dx), _heads_first(db), _heads_first(dc),
            _heads_first(da.reshape(batch, heads, total)), dstate,
            *(jnp.swapaxes(gradient, 1, 2) for gradient in taken))


def _ssd_scan(x, b, c, a, state, documents, chunk_size: int, platform: str,
              held: int = 0) -> tuple[jax.Array, ...]:
    """Run the chunked SSD scan on the Pallas kernel for `platform`.

    Operands arrive as `xla_chunk_scan` takes them, all fp32: `x`
    `[B, T, H, P]` already scaled by `dt`, `b` and `c` `[B, T, H, N]` with the
    groups expanded, `a` `[B, T, H]` the per-step `A dt`, the length whole
    chunks of `chunk_size`, `state` `[B, H, P, N]` entering the sequence, and
    `documents` None or a forked call's `(given [B, S', H, P, N] float32,
    loads [B, NC], stores [B, NC] int32, entries [B, NC, H, P, N] float32 or
    None)` for documents each starting a chunk, a chunk's entry write added to
    the memory it takes up (`recurrence_chunk.Chunk.loads`, `held` stored
    slots). Returns the output
    `[B, T, H, P]` and the final state, the same pair `xla_chunk_scan` returns.
    """
    return tuple(_forward(x, b, c, a, state, chunk_size, platform, keep_entered=False, documents=documents,
                          held=held))


def _ssd_fwd(x, b, c, a, state, documents, chunk_size: int, platform: str, held: int = 0):
    y, final, entered = _forward(x, b, c, a, state, chunk_size, platform, keep_entered=True,
                                 documents=documents, held=held)
    return (y, final), (x, b, c, a, entered, documents)


def _ssd_bwd(chunk_size: int, platform: str, held: int, residual, cotangent):
    x, b, c, a, entered, documents = residual
    dy, dfinal = cotangent
    dx, db, dc, da, dstate, *dforked = _backward(x, b, c, a, entered, dy, dfinal, chunk_size, platform,
                                                 documents=documents, held=held)
    ddocuments = None if documents is None else (
        dforked[0].astype(documents[0].dtype), None, None,
        None if documents[3] is None else dforked[1].astype(documents[3].dtype))
    return dx, db, dc, da, dstate, ddocuments


_chunk_scan = derivatives.contracted("test_the_tpu_ssd_scan_holds_its_orders")(
    derivatives.declared_derivative(derivatives.KernelGlue(), _ssd_scan, _ssd_fwd, _ssd_bwd,
                                    nondiff_argnums=(6, 7, 8)))
