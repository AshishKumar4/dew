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
operands as the XLA scan does, `[B, T, H, ...]` (`ssd_chunk_scan`); the
heads go ahead of the time inside it, the one layout Mosaic blocks.

The backward is the hand-written reverse recurrence, `dS_n = exp(acs_C-1)
dS_n+1 + dy^T (C exp(acs))`, since autodiff would keep every chunk's
intermediates. The XLA path is the oracle and the scan for CPU and refused
geometries; a GPU whose target row names it runs mamba_ssm's own kernels
(`dew.nn.kernels.mamba_ssd`), else the XLA path. tests/test_ssd_kernel.py
holds the kernel to `chunk_ssd` and its `jax.grad`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.typing import DTypeLike

from ..precision import float32_products, products_dtype
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
    """`lhs @ rhs` at the precision its operands' dtype states (`precision.float32_products`): the chunk's
    fp32 operands multiplied in fp32, where a TPU's Pallas matmul at its default precision takes one bf16
    pass (template numerics review 2026-10-26 follow-up, P1-1: every Mamba-2 layer's scan on a TPU)."""
    return jnp.matmul(lhs, rhs, precision=float32_products(products_dtype(lhs, rhs)))


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
    carried = jnp.where(pl.program_id(2) == 0, state_ref[...], final_ref[...])
    for entered_ref in saved:
        entered_ref[...] = carried
    y, leaving = _chunk_forward(x_ref[...], b_ref[...], c_ref[...], a_ref[...], carried)
    y_ref[...] = y
    final_ref[...] = leaving


def _tpu_backward(x_ref, b_ref, c_ref, a_ref, entered_ref, dy_ref, dfinal_ref,
                  dx_ref, db_ref, dc_ref, da_ref, dstate_ref):
    """`_tpu_forward` read backwards. The grid still counts up; the index maps
    hand it the chunks in reverse, and `dstate_ref` carries as `final_ref`
    did."""
    dnext = jnp.where(pl.program_id(2) == 0, dfinal_ref[...], dstate_ref[...])
    dx, db, dc, da, dstate = _chunk_backward(
        x_ref[...], b_ref[...], c_ref[...], a_ref[...], entered_ref[...], dy_ref[...], dnext)
    dx_ref[...], db_ref[...], dc_ref[...], da_ref[...] = dx, db, dc, da
    dstate_ref[...] = dstate


def _specs(chunks: int, chunk_size: int, head_dim: int, state_size: int, *,
           reverse: bool = False):
    """The block specs of the arrays a chunk reads or writes, in the kernel's
    own layout: `[B, H, T, P]` for `x` and `y`, `[B, H, T, N]` for `b` and
    `c`, each chunk `chunk_size` steps of the time, `[B, H, NC, 1, C]` for
    `a`, `[B, H, NC, P, N]` for the state each chunk entered, and
    `[B, H, P, N]` for the state crossing the sequence. Returned in that order.

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


def _call(body, platform: str, grid: tuple[int, ...], in_specs, out_specs, out_shape):
    """One Mosaic `pallas_call`, interpreted where the process holds no TPU."""
    params = pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary"))
    return pl.pallas_call(body, grid=grid, in_specs=list(in_specs), out_specs=list(out_specs),
                          out_shape=out_shape, compiler_params=params,
                          interpret=_interpreting(platform))


def _forward(x, b, c, a, state, chunk_size: int, platform: str, *, keep_entered: bool):
    """The forward kernel. Operands arrive as `ssd_chunk_scan` takes them and
    the output leaves so; `keep_entered` also returns the state every chunk
    entered with, which is the residual the backward pass reads."""
    batch, total, heads, head_dim = x.shape
    chunks, state_size = total // chunk_size, b.shape[-1]
    x_spec, b_spec, c_spec, a_spec, entered_spec, carry_spec = _specs(
        chunks, chunk_size, head_dim, state_size)
    out_shape = [jax.ShapeDtypeStruct((batch, heads, total, head_dim), jnp.float32),
                 jax.ShapeDtypeStruct(state.shape, jnp.float32)]
    out_specs = [x_spec, carry_spec]
    if keep_entered:
        out_shape.append(jax.ShapeDtypeStruct((batch, heads, chunks, head_dim, state_size),
                                              jnp.float32))
        out_specs.append(entered_spec)
    run = _call(_tpu_forward, platform, (batch, heads, chunks),
                (x_spec, b_spec, c_spec, a_spec, carry_spec), out_specs, out_shape)
    y, final, *entered = run(_heads_first(x), _heads_first(b), _heads_first(c), _decay_blocks(a, chunk_size),
                             state)
    return (_heads_first(y), final, *entered)


def _backward(x, b, c, a, entered, dy, dfinal, chunk_size: int, platform: str):
    """The backward kernel, one reverse pass over the same blocks."""
    batch, total, heads, head_dim = x.shape
    chunks, state_size = total // chunk_size, b.shape[-1]
    x_spec, b_spec, c_spec, a_spec, entered_spec, carry_spec = _specs(
        chunks, chunk_size, head_dim, state_size, reverse=True)
    widths = (head_dim, state_size, state_size)
    out_shape = [jax.ShapeDtypeStruct((batch, heads, total, width), jnp.float32) for width in widths]
    out_shape.append(jax.ShapeDtypeStruct((batch, heads, chunks, 1, chunk_size), jnp.float32))
    out_shape.append(jax.ShapeDtypeStruct(dfinal.shape, jnp.float32))
    run = _call(_tpu_backward, platform, (batch, heads, chunks),
                (x_spec, b_spec, c_spec, a_spec, entered_spec, x_spec, carry_spec),
                (x_spec, b_spec, c_spec, a_spec, carry_spec), out_shape)
    dx, db, dc, da, dstate = run(_heads_first(x), _heads_first(b), _heads_first(c),
                                 _decay_blocks(a, chunk_size), entered, _heads_first(dy), dfinal)
    return (_heads_first(dx), _heads_first(db), _heads_first(dc),
            _heads_first(da.reshape(batch, heads, total)), dstate)


def _ssd_scan(x, b, c, a, state, chunk_size: int, platform: str) -> tuple[jax.Array, jax.Array]:
    """Run the chunked SSD scan on the Pallas kernel for `platform`.

    Operands arrive as `xla_chunk_scan` takes them, all fp32: `x`
    `[B, T, H, P]` already scaled by `dt`, `b` and `c` `[B, T, H, N]` with the
    groups expanded, `a` `[B, T, H]` the per-step `A dt`, the length whole
    chunks of `chunk_size`, `state` `[B, H, P, N]` entering the sequence.
    Returns the output `[B, T, H, P]` and the final state, the same pair
    `xla_chunk_scan` returns.
    """
    y, final = _forward(x, b, c, a, state, chunk_size, platform, keep_entered=False)
    return y, final


def _ssd_fwd(x, b, c, a, state, chunk_size: int, platform: str):
    y, final, entered = _forward(x, b, c, a, state, chunk_size, platform, keep_entered=True)
    return (y, final), (x, b, c, a, entered)


def _ssd_bwd(chunk_size: int, platform: str, residual, cotangent):
    x, b, c, a, entered = residual
    dy, dfinal = cotangent
    return _backward(x, b, c, a, entered, dy, dfinal, chunk_size, platform)


# `jax.custom_vjp` is generic in its return type, and a `functools.partial`
# decorator loses that binding, so it is built by hand, as chunked_cross_entropy's
# `_bounded_head` is.
ssd_chunk_scan = jax.custom_vjp(_ssd_scan, nondiff_argnums=(5, 6))
ssd_chunk_scan.defvjp(_ssd_fwd, _ssd_bwd)
