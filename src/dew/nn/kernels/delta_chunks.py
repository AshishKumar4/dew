"""The chunked delta rule's memory across chunks, as two Pallas kernels for CUDA.

`dew.nn.linear.chunk_gated_delta_rule` resolves each chunk of `C` tokens in
matrix form, every chunk at once, which leaves one sequential step: the
memory `S` entering each chunk,

    v_i     = u_i - w_i S_i                                    (corrected values)
    S_i+1   = exp(g_i,C) S_i + (k_i exp(g_i,C - g_i))^T v_i    (the write)

with `w` the reference's `k_cumdecay`, `u` its corrected `value` and `g` the
log decay cumulated within the chunk. XLA runs it as a `lax.scan` launching
each chunk's small batched products in turn (`xla_chunk_states`). Here one
program, one row-head and one block of value columns, walks its chunks in
order with its `[Dk, BLOCK]` slice of the memory on chip, as fla's
`chunk_gated_delta_rule_fwd_h` does (fla-org/flash-linear-attention v0.5.2,
fla/ops/common/chunk_delta_h.py). No column of the memory reads another, so
the blocks are independent programs.

The backward is the reverse recurrence for the gradient of the memory
leaving each chunk (fla's `chunk_gated_delta_rule_bwd_dhu`),

    dv_i    = dv_i' + (k_i exp(g_i,C - g_i)) dS_i+1
    dS_i    = dS_i' + exp(g_i,C) dS_i+1 - w_i^T dv_i

where `dv_i'` and `dS_i'` are the cotangents the output sends the corrected
values and the entered state. The kernel writes only the dS_i+1; everything
it owes the operands, `dv_i` included, is one batched product per chunk
from them, left to XLA. The kernels' products take the
precision XLA's take, the configured default matmul precision (TF32 on a GPU
at the default, IEEE fp32 at 'highest'), so choosing them changes no
arithmetic class. tests/test_delta_chunks.py holds both to `xla_chunk_states`
and its `jax.vjp`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from ..sharding import seen_whole
from .generation import first_refusal

BLOCK = 32
"""Value columns a program carries."""
WARPS = 4
STAGES = 2
"""Triton's software pipeline depth. Each stage buffers a chunk's fp32
`[C, Dk]` operands in shared memory."""
MIN_WIDTH = 16
"""The narrowest side of a product Triton's `dot` takes."""


def refusal(key: jax.Array, out_vals: jax.Array) -> str | None:
    """Why the kernels do not take a rule over these per-chunk operands
    (`[B, H, NC, C, Dk]` keys, `[B, H, NC, C, Dv]` corrected values), or None
    where they do: an fp32 rule, chunks and keys power-of-two wide from
    `MIN_WIDTH`, values a multiple of `BLOCK` wide, and operands no mesh axis
    splits (`seen_whole`)."""
    chunk, width = key.shape[-2:]
    columns = out_vals.shape[-1]
    return first_refusal(
        (key.dtype == jnp.float32, f"the rule runs in {key.dtype}, and the kernels in float32"),
        (all(side >= MIN_WIDTH and side & (side - 1) == 0 for side in (chunk, width)),
         f"chunks of {chunk} and keys {width} wide; Triton multiplies powers of two from {MIN_WIDTH}"),
        (columns % BLOCK == 0, f"values {columns} wide, not a multiple of the {BLOCK}-column block"),
        (seen_whole(key), "a mesh axis splits the rule's operands, and the kernels see whole arrays"),
    )


def _forward_kernel(key_ref, w_ref, gc_ref, u_ref, state_ref, entered_ref, corrected_ref, final_ref):
    """One program's chunks in order. Refs are its row-head's `[NC, C, Dk]`
    keys and `w`, `[NC, C]` cumulated decays, `[NC, C, BLOCK]` values and its
    `[Dk, BLOCK]` slice of the memory; it writes the slice each chunk enters,
    the corrected values and the slice leaving the last chunk."""
    chunks, size = gc_ref.shape

    def chunk(n, state):
        entered_ref[n] = state
        gc, last = gc_ref[n], gc_ref[n, size - 1]
        corrected = u_ref[n] - w_ref[n] @ state
        corrected_ref[n] = corrected
        written = key_ref[n] * jnp.exp(last - gc)[:, None]
        return state * jnp.exp(last) + written.T @ corrected

    final_ref[...] = jax.lax.fori_loop(0, chunks, chunk, state_ref[...])


def _backward_kernel(key_ref, w_ref, gc_ref, d_entered_ref, d_corrected_ref, d_final_ref,
                     d_leaving_ref, d_state_ref):
    """`_forward_kernel` read backwards: from the cotangent of the slice
    leaving the last chunk, each chunk's gradient of the slice leaving it,
    then the gradient of the slice entering the first. The corrected values'
    gradient it carries is the recurrence's own; the one returned is
    recomputed outside (`_states_bwd`)."""
    chunks, size = gc_ref.shape

    def chunk(i, d_state):
        n = chunks - 1 - i
        d_leaving_ref[n] = d_state
        gc, last = gc_ref[n], gc_ref[n, size - 1]
        written = key_ref[n] * jnp.exp(last - gc)[:, None]
        d_u = d_corrected_ref[n] + written @ d_state
        return d_entered_ref[n] + d_state * jnp.exp(last) - w_ref[n].T @ d_u

    d_state_ref[...] = jax.lax.fori_loop(0, chunks, chunk, d_final_ref[...])


def _call(body, name: str, whole, columned, outputs, interpret: bool):
    """One `pallas_call` of `body` over `grid=(rows * heads, Dv // BLOCK)`.
    Arrays are `[B, H, ...]`, flattened to one leading row-head axis. Every
    block reads the `whole` operands entire (keys, `w`, decays); the
    `columned` operands and the `outputs` are as wide as the values,
    `[B, H, ..., Dv]`, and each block takes its `BLOCK` columns."""
    from jax.experimental.pallas import triton as plgpu

    batch, heads = whole[0].shape[:2]
    rows, columns = batch * heads, outputs[-1].shape[-1]

    def spec(array, *, blocked: bool):
        inner = array.shape[2:-1]
        return pl.BlockSpec((None, *inner, BLOCK if blocked else array.shape[-1]),
                            lambda i, j: (i, *(0,) * len(inner), j if blocked else 0))

    def flat(array):
        return array.reshape(rows, *array.shape[2:])

    blocks = pl.pallas_call(
        body, grid=(rows, columns // BLOCK),
        in_specs=([spec(array, blocked=False) for array in whole]
                  + [spec(array, blocked=True) for array in columned]),
        out_specs=[spec(shape, blocked=True) for shape in outputs],
        out_shape=[jax.ShapeDtypeStruct((rows, *shape.shape[2:]), jnp.float32) for shape in outputs],
        compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=STAGES),
        interpret=interpret, name=name,
    )(*map(flat, whole), *map(flat, columned))
    return [out.reshape(shape.shape) for out, shape in zip(blocks, outputs, strict=True)]


def _states(key, w, u, gc, state, interpret: bool):
    """`xla_chunk_states` on the kernels: operands in its layout, fp32, and
    the same `(entered, corrected, final)` back."""
    batch, heads, chunks = key.shape[:3]
    entered = jax.ShapeDtypeStruct((batch, heads, chunks, *state.shape[2:]), jnp.float32)
    entered, corrected, final = _call(_forward_kernel, "gated_delta_states", (key, w, gc), (u, state),
                                      (entered, u, state), interpret)
    return entered, corrected, final


def _states_fwd(key, w, u, gc, state, interpret: bool):
    entered, corrected, final = _states(key, w, u, gc, state, interpret)
    return (entered, corrected, final), (key, w, gc, entered, corrected)


def _states_bwd(interpret: bool, residual, cotangent):
    key, w, gc, entered, corrected = residual
    d_entered, d_corrected, d_final = cotangent
    d_leaving, d_state = _call(_backward_kernel, "gated_delta_states_backward", (key, w, gc),
                               (d_entered, d_corrected, d_final), (d_entered, d_final), interpret)
    last = gc[..., -1:]
    leave = jnp.exp(last - gc)                                         # [.., C]
    # The corrected values' gradient, from the gradients of the slices each
    # chunk leaves, which the w gradient reads too. Read back from the
    # kernel, it sat 3.07 times as far from float64 as the scan's at IEEE
    # products on an A100, and w's 1.25; as one batched product here, 1.01
    # and 1.00 (c49). Why the kernel's product rounds worse is measured, not
    # explained: both are IEEE fp32 dots (jax 0.11.2 lowers a HIGHEST fp32
    # dot to Triton's IEEE input precision) and both exps libdevice's; the
    # order Triton's dot accumulates in, against cuBLAS's, is an unconfirmed
    # lead.
    d_u = d_corrected + (key * leave[..., None]) @ d_leaving
    d_written = corrected @ jnp.swapaxes(d_leaving, -1, -2)            # [.., C, Dk]
    # The written keys' decay `exp(g_C - g_t)`: what reaches its exponent
    # goes to g_C with one sign and to g_t with the other.
    d_exponent = jnp.sum(d_written * key, axis=-1) * leave
    d_last = (jnp.exp(last[..., 0]) * jnp.sum(entered * d_leaving, axis=(-2, -1))
              + jnp.sum(d_exponent, axis=-1))
    d_gc = (-d_exponent).at[..., -1].add(d_last)
    return (d_written * leave[..., None], -(d_u @ jnp.swapaxes(entered, -1, -2)), d_u, d_gc, d_state)


# `jax.custom_vjp` is generic in its return type, and a `functools.partial`
# decorator loses that binding, so it is built by hand, as ssd's is.
chunk_states = jax.custom_vjp(_states, nondiff_argnums=(5,))
chunk_states.defvjp(_states_fwd, _states_bwd)
