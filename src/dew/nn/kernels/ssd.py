"""The Mamba-2 chunked SSD scan as one Pallas kernel, for GPU and for TPU.

`dew.nn.mixers.mamba2.chunk_ssd` splits a sequence into chunks of `C` steps
and, per batch element and head, runs

    y_i = sum_{j <= i} (C_i . B_j) exp(segsum_ij) x_j            (intra-chunk)
        + exp(acs_i) C_i S^T                                     (the carry)
    S' = exp(acs_C-1) S + sum_j exp(segsum_C-1,j) x_j B_j^T      (the write)

with `segsum_ij = sum_{j < k <= i} A dt_k`, `acs` its cumulative sum and `x`
already carrying its `dt`. XLA runs that as six einsums over the whole batch,
so every chunk's `[C, C]` segment sums, its `[C, C]` Gram matrix and its
`[P, N]` state cross HBM. The kernel keeps them inside one program and writes
only `y`, the final state, and (when a gradient needs it) the state each chunk
entered with.

The two backends need different loop structures around the same chunk, which
`_chunk_forward` and `_chunk_backward` hold:

- Triton's grid is a parallel launch, so no program may depend on another's
  state. The GPU kernel runs `grid=(batch, heads)` and carries the state
  through a `fori_loop` over the chunks inside one program.
- Mosaic's grid is ordered and a window it revisits survives a grid step, so
  the TPU kernel runs `grid=(batch, heads, chunks)` with the chunk innermost
  and the state in the output window the chunks share.

The backward pass is the reverse recurrence, `dS_n = exp(acs_C-1) dS_n+1 +
dy^T (C exp(acs))`, with each chunk's own gradients from the matrices the
forward built. It is written by hand rather than left to autodiff, which
would hold every intermediate of every chunk alive to the backward pass.

The XLA path stays the oracle and the fallback: CPU takes it, and so does any
geometry `ssd_kernel_runs` refuses. tests/test_ssd_kernel.py holds the kernel
to `chunk_ssd` and to `jax.grad` of it.
"""

from __future__ import annotations

import functools
import logging

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu, triton as plgpu

from .generation import filter_triton_deprecation, triton_compiles

_log = logging.getLogger(__name__)

MIN_WIDTH = 8
"""The narrowest chunk, head width or state width the kernel is chosen for.
Below it a tile is mostly padding: 8 is Mosaic's sublane count and the
smallest block shape Triton lays out without splitting a warp's lanes."""

MIN_CHUNK = 64
"""The shortest chunk worth a program. The intra-chunk term is `C` times the
work of the carry, so under 64 a program spends more on its own state
recurrence than on the matmul it exists for and XLA's batched einsum wins."""

GPU_PROGRAM_WORDS = 1 << 16
"""The fp32 words one CUDA thread block may hold, 256 KiB. That is the
register file of an SM plus the shared memory a block gets on Ada and Hopper,
and a program's `[C, C]` segment sums, its `[C, P]` and two `[C, N]` operands
and its `[P, N]` state have to fit in it. A wider geometry needs the chunk
sub-tiled into row blocks, which this formulation does not do, so it takes the
XLA path instead."""

TPU_PROGRAM_WORDS = 1 << 20
"""The same budget on TPU, 4 MiB. A Mosaic program's operands live in VMEM,
128 MiB of it per core, so what bounds the window here is the double buffering
the pipeline runs around it rather than the window itself."""

GPU_WARPS = 8
"""256 threads per program. The widest tile is `[C, C]`, so the warps split
its rows; 8 keeps a 128-row chunk at 16 rows per warp, the shape Triton's
`dot` emits without a cross-warp reduction."""

GPU_STAGES = 2
"""A program's chunks carry state through each other, so two of them cannot
run at once; the pipeline has only the next chunk's loads to overlap."""

EXACT = jax.lax.Precision.HIGHEST
"""The precision of the two matmuls that build and unbuild the segment sums.
Their operands are the chunk's `A dt` against a 0/1 matrix, so a pass in
bf16 or tf32 would round the exponent itself, and `exp` would carry that
rounding into every decay the chunk applies. The matmuls over the operands
are left at the default the XLA path's einsums use."""


def _program_words(chunk_size: int, head_dim: int, state_size: int) -> int:
    """The fp32 words one program holds: a chunk's `[C, C]` segment sums, its
    `[C, P]` input block, its two `[C, N]` blocks and the `[P, N]` state."""
    return (chunk_size * chunk_size + chunk_size * (head_dim + 2 * state_size)
            + head_dim * state_size)


def ssd_kernel_runs(chunk_size: int, head_dim: int, state_size: int, backend: str) -> bool:
    """Whether the SSD kernel takes this geometry: a tpu backend or a gpu the
    Triton kernels compile for (`dew.nn.kernels.generation.triton_compiles`), a
    chunk long enough to pay for a program, three widths that are powers of
    two so that neither Triton's block padding nor Mosaic's tiling throws
    lanes away, and a tile inside the backend's per-program budget.

    `chunk_ssd` asks this at trace time and takes the XLA path when it says
    no, the way attention's 'auto' asks `cudnn_runs`.
    """
    if backend not in ('gpu', 'tpu') or (backend == 'gpu' and not triton_compiles()):
        return False
    widths = (chunk_size, head_dim, state_size)
    if any(width < MIN_WIDTH or width & (width - 1) for width in widths):
        return False
    budget = GPU_PROGRAM_WORDS if backend == 'gpu' else TPU_PROGRAM_WORDS
    return chunk_size >= MIN_CHUNK and _program_words(*widths) <= budget


@functools.cache
def _announce(backend: str, chunk_size: int, head_dim: int, state_size: int,
              platform: str | None) -> None:
    """One line per geometry a process traces, from the trace that chose."""
    geometry = f"chunk {chunk_size}, head width {head_dim}, state {state_size}"
    if platform is None:
        _log.info("mamba2 ssd scan on %s: the xla path (%s, %d fp32 words a program)",
                  backend, geometry, _program_words(chunk_size, head_dim, state_size))
    else:
        _log.info("mamba2 ssd scan on %s: the pallas kernel (%s)", platform, geometry)


def ssd_kernel_platform(chunk_size: int, head_dim: int, state_size: int) -> str | None:
    """The backend to build this scan's kernel for, or None for the XLA path."""
    backend = jax.default_backend()
    platform = backend if ssd_kernel_runs(chunk_size, head_dim, state_size, backend) else None
    _announce(backend, chunk_size, head_dim, state_size, platform)
    return platform


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


def _chunk_forward(x, b, c, a, state):
    """One chunk in fp32: `x` `[C, P]` already scaled by `dt`, `b` and `c`
    `[C, N]`, `a` `[C]` the per-step `A dt`, `state` `[P, N]` entering it.
    Returns the chunk's output `[C, P]` and the state leaving it."""
    below, above = _triangles(a.shape[0])
    spread, segments = _segment_sums(a, below, above)
    decay = jnp.where(below, jnp.exp(segments), 0.0)
    entered = jnp.exp(jnp.sum(spread, axis=-1))
    leaving, alpha = jnp.exp(_last_row(segments)), jnp.exp(jnp.sum(a))
    y = (c @ b.T * decay) @ x + entered[:, None] * (c @ state.T)
    return y, alpha * state + x.T @ (b * leaving[:, None])


def _chunk_backward(x, b, c, a, state, dy, dnext):
    """The reverse of `_chunk_forward` at one chunk. `dy` `[C, P]` and `dnext`
    `[P, N]` are the gradients of the chunk's output and of the state leaving
    it; returns those of `x`, `b`, `c`, `a` and of the state that entered."""
    below, above = _triangles(a.shape[0])
    spread, segments = _segment_sums(a, below, above)
    decay = jnp.where(below, jnp.exp(segments), 0.0)
    entered = jnp.exp(jnp.sum(spread, axis=-1))
    leaving, alpha = jnp.exp(_last_row(segments)), jnp.exp(jnp.sum(a))

    gram, outer = c @ b.T, dy @ x.T
    dgram, written = outer * decay, x @ dnext
    dx = (gram * decay).T @ dy + (b * leaving[:, None]) @ dnext.T
    db = dgram.T @ c + written * leaving[:, None]
    dc = dgram @ b + entered[:, None] * (dy @ state)
    dstate = (dy * entered[:, None]).T @ c + alpha * dnext

    dleaving = jnp.sum(written * b, axis=1) * leaving
    dsegments = (jnp.where(below, outer * gram * decay, 0.0)
                 + jnp.where(_bottom(below.shape[0]), dleaving[None, :], 0.0))
    dspread = jax.lax.dot_general(dsegments, above.astype(dy.dtype), (((1,), (1,)), ((), ())),
                                  precision=EXACT)
    # Every step of the chunk feeds `acs` and the chunk's total alike, so what
    # reaches those two spreads back over the whole of `a` rather than a step.
    dspread = dspread + (jnp.sum(dy * (c @ state.T), axis=1) * entered)[:, None]
    da = jnp.sum(jnp.where(below, dspread, 0.0), axis=0) + jnp.sum(dnext * state) * alpha
    return dx, db, dc, da, dstate


def _gpu_forward(x_ref, b_ref, c_ref, a_ref, state_ref, y_ref, final_ref, *saved,
                 chunks: int):
    """`grid=(batch, heads)`: every chunk of one head is this program's, and
    the recurrence over them is a loop inside it."""
    def chunk(index, carried):
        for entered_ref in saved:
            entered_ref[index] = carried
        y, leaving = _chunk_forward(x_ref[index], b_ref[index], c_ref[index],
                                    a_ref[index], carried)
        y_ref[index] = y
        return leaving

    final_ref[...] = jax.lax.fori_loop(0, chunks, chunk, state_ref[...])


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


def _gpu_backward(x_ref, b_ref, c_ref, a_ref, entered_ref, dy_ref, dfinal_ref,
                  dx_ref, db_ref, dc_ref, da_ref, dstate_ref, *, chunks: int):
    """`_gpu_forward` read backwards, the last chunk of a head first."""
    def chunk(step, dnext):
        index = chunks - 1 - step
        dx, db, dc, da, dstate = _chunk_backward(
            x_ref[index], b_ref[index], c_ref[index], a_ref[index],
            entered_ref[index], dy_ref[index], dnext)
        dx_ref[index], db_ref[index], dc_ref[index], da_ref[index] = dx, db, dc, da
        return dstate

    dstate_ref[...] = jax.lax.fori_loop(0, chunks, chunk, dfinal_ref[...])


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
           whole: bool, reverse: bool = False):
    """The block specs of the arrays a chunk reads or writes, in the kernel's
    own layout: `[B, H, NC, C, P]` for `x` and `y`, `[B, H, NC, C, N]` for `b`
    and `c`, `[B, H, NC, 1, C]` for `a`, `[B, H, NC, P, N]` for the state each
    chunk entered, and `[B, H, P, N]` for the state crossing the sequence.
    Returned in that order.

    Mosaic blocks the last two dimensions or nothing, which is what puts the
    heads ahead of the chunks here rather than in the layout `chunk_ssd`
    builds. `whole` gives one program every chunk at once, the grid the GPU
    kernel loops over; otherwise a program holds one chunk and the grid counts
    them, backwards when `reverse`.
    """
    leading = chunks if whole else None

    def position(grid: tuple[int, ...]) -> tuple[int, int, int]:
        if whole:
            return grid[0], grid[1], 0
        return grid[0], grid[1], chunks - 1 - grid[2] if reverse else grid[2]

    def over_chunk(*grid):
        batch, head, chunk = position(grid)
        return batch, head, chunk, 0, 0

    def over_carry(*grid):
        batch, head, _ = position(grid)
        return batch, head, 0, 0

    return (pl.BlockSpec((None, None, leading, chunk_size, head_dim), over_chunk),
            pl.BlockSpec((None, None, leading, chunk_size, state_size), over_chunk),
            pl.BlockSpec((None, None, leading, chunk_size, state_size), over_chunk),
            pl.BlockSpec((None, None, leading, None, chunk_size), over_chunk),
            pl.BlockSpec((None, None, leading, head_dim, state_size), over_chunk),
            pl.BlockSpec((None, None, head_dim, state_size), over_carry))


def _heads_first(blocked):
    """`[NC, B, C, H, W] -> [B, H, NC, C, W]`, the layout Mosaic can block."""
    return jnp.transpose(blocked, (1, 3, 0, 2, 4))


def _chunks_first(blocked):
    """`[B, H, NC, C, W] -> [NC, B, C, H, W]`, back to what `chunk_ssd` holds."""
    return jnp.transpose(blocked, (2, 0, 3, 1, 4))


def _interpreting(platform: str) -> bool:
    """Whether pallas has to run its own interpreter instead of a kernel: it
    compiles for a device, so a process holding none of that platform can only
    interpret. That is how CPU reads the kernel's numbers, and it is what lets
    a test name a backend it has no device for."""
    return all(device.platform != platform for device in jax.devices())


def _call(body, platform: str, grid: tuple[int, ...], in_specs, out_specs, out_shape):
    """One `pallas_call` built for the backend `platform` names."""
    if platform == 'gpu':
        filter_triton_deprecation()
        params = plgpu.CompilerParams(num_warps=GPU_WARPS, num_stages=GPU_STAGES)
    else:
        params = pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"))
    return pl.pallas_call(body, grid=grid, in_specs=list(in_specs), out_specs=list(out_specs),
                          out_shape=out_shape, compiler_params=params,
                          interpret=_interpreting(platform))


def _forward(x_c, b_c, c_c, a_c, state, platform: str, *, keep_entered: bool):
    """The forward kernel. Operands arrive in `chunk_ssd`'s layout and the
    output leaves in it; `keep_entered` also returns the state every chunk
    entered with, which is the residual the backward pass reads."""
    chunks, batch, chunk_size, heads, head_dim = x_c.shape
    state_size = b_c.shape[-1]
    whole = platform == 'gpu'
    x_spec, b_spec, c_spec, a_spec, entered_spec, carry_spec = _specs(
        chunks, chunk_size, head_dim, state_size, whole=whole)
    laid_out = (batch, heads, chunks, chunk_size, head_dim)
    out_shape = [jax.ShapeDtypeStruct(laid_out, jnp.float32),
                 jax.ShapeDtypeStruct(state.shape, jnp.float32)]
    out_specs = [x_spec, carry_spec]
    if keep_entered:
        out_shape.append(jax.ShapeDtypeStruct((batch, heads, chunks, head_dim, state_size),
                                              jnp.float32))
        out_specs.append(entered_spec)
    body = functools.partial(_gpu_forward, chunks=chunks) if whole else _tpu_forward
    run = _call(body, platform, (batch, heads) if whole else (batch, heads, chunks),
                (x_spec, b_spec, c_spec, a_spec, carry_spec), out_specs, out_shape)
    y, final, *entered = run(_heads_first(x_c), _heads_first(b_c), _heads_first(c_c),
                             jnp.transpose(a_c, (1, 2, 0, 3))[:, :, :, None, :], state)
    return (_chunks_first(y), final, *entered)


def _backward(x_c, b_c, c_c, a_c, entered, dy, dfinal, platform: str):
    """The backward kernel, one reverse pass over the same blocks."""
    chunks, batch, chunk_size, heads, head_dim = x_c.shape
    state_size = b_c.shape[-1]
    whole = platform == 'gpu'
    x_spec, b_spec, c_spec, a_spec, entered_spec, carry_spec = _specs(
        chunks, chunk_size, head_dim, state_size, whole=whole, reverse=not whole)
    widths = (head_dim, state_size, state_size)
    out_shape = [jax.ShapeDtypeStruct((batch, heads, chunks, chunk_size, width), jnp.float32)
                 for width in widths]
    out_shape.append(jax.ShapeDtypeStruct((batch, heads, chunks, 1, chunk_size), jnp.float32))
    out_shape.append(jax.ShapeDtypeStruct(dfinal.shape, jnp.float32))
    body = functools.partial(_gpu_backward, chunks=chunks) if whole else _tpu_backward
    run = _call(body, platform, (batch, heads) if whole else (batch, heads, chunks),
                (x_spec, b_spec, c_spec, a_spec, entered_spec, x_spec, carry_spec),
                (x_spec, b_spec, c_spec, a_spec, carry_spec), out_shape)
    dx, db, dc, da, dstate = run(
        _heads_first(x_c), _heads_first(b_c), _heads_first(c_c),
        jnp.transpose(a_c, (1, 2, 0, 3))[:, :, :, None, :], entered, _heads_first(dy), dfinal)
    return (_chunks_first(dx), _chunks_first(db), _chunks_first(dc),
            jnp.transpose(da[:, :, :, 0, :], (2, 0, 1, 3)), dstate)


def _ssd_scan(x_c, b_c, c_c, a_c, state, platform: str) -> tuple[jax.Array, jax.Array]:
    """Run the chunked SSD scan on the Pallas kernel for `platform`.

    Operands arrive in `chunk_ssd`'s layout, all fp32: `x_c`
    `[NC, B, C, H, P]` already scaled by `dt`, `b_c` and `c_c`
    `[NC, B, C, H, N]` with the groups expanded, `a_c` `[NC, B, H, C]` the
    per-step `A dt`, `state` `[B, H, P, N]` entering the sequence. Returns
    the output in `x_c`'s layout and the final state, the same pair
    `xla_chunk_scan` returns.
    """
    y, final = _forward(x_c, b_c, c_c, a_c, state, platform, keep_entered=False)
    return y, final


def _ssd_fwd(x_c, b_c, c_c, a_c, state, platform: str):
    y, final, entered = _forward(x_c, b_c, c_c, a_c, state, platform, keep_entered=True)
    return (y, final), (x_c, b_c, c_c, a_c, entered)


def _ssd_bwd(platform: str, residual, cotangent):
    x_c, b_c, c_c, a_c, entered = residual
    dy, dfinal = cotangent
    return _backward(x_c, b_c, c_c, a_c, entered, dy, dfinal, platform)


# `jax.custom_vjp` is generic in its return type, and a `functools.partial`
# decorator loses that binding, so it is built by hand, as chunked_cross_entropy's
# `_bounded_head` is.
ssd_chunk_scan = jax.custom_vjp(_ssd_scan, nondiff_argnums=(5,))
ssd_chunk_scan.defvjp(_ssd_fwd, _ssd_bwd)
