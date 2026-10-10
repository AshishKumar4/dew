"""The chunked gated delta rule's output and its VJP, with local products on chip.

One chunk computes `(q exp(hi)) h + ((q k^T) decay) v`, where the pairwise
decay uses both halves of the compensated log-decay sum. Neither that
`[C, C]` decay nor the paired queries and keys leave the program. The value
columns are independent in the forward and in the cotangents of `h` and
`v`; the other cotangents split the key columns and sum only the small
per-position decay contributions outside the kernel. This is the output
decomposition of fla-org/flash-linear-attention v0.5.2,
fla/ops/common/chunk_o.py:42-152,193-378,552-648, with Dew's compensated
decays and the configured default matmul precision, as delta_chunks uses.
"""

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .generation import first_refusal

BLOCK = 32
"""Value columns per output program and key columns per operand-gradient program."""
WARPS = 4
STAGES = 2


def refusal(key, value) -> str | None:
    """Bounds on top of delta_chunks.refusal: a chunk's square products and
    one head's values must fit on chip in the operand-gradient kernel."""
    return first_refusal(
        (key.shape[-2] <= 64, f"output chunks of {key.shape[-2]}, wider than 64"),
        (key.shape[-1] <= 128, f"output keys {key.shape[-1]} wide, wider than 128"),
        (value.shape[-1] <= 256 and value.shape[-1] & (value.shape[-1] - 1) == 0,
         f"output values {value.shape[-1]} wide, not a power of two through 256"),
    )


def _decay(hi, lo):
    """Inclusive pairwise decay, masked before exp so unused differences
    cannot overflow (dew.nn.linear.chunk_decay)."""
    index = jnp.arange(hi.shape[0])
    lower = index[:, None] >= index[None, :]
    diff = (hi[:, None] - hi[None, :]) + (lo[:, None] - lo[None, :])
    return jnp.where(lower, jnp.exp(jnp.where(lower, diff, 0.0)), 0.0)


def _forward_kernel(q_ref, k_ref, hi_ref, lo_ref, h_ref, v_ref, out_ref):
    q, k, hi, lo = q_ref[...], k_ref[...], hi_ref[...], lo_ref[...]
    paired = (q @ k.T) * _decay(hi, lo)
    out_ref[...] = (q * jnp.exp(hi)[:, None]) @ h_ref[...] + paired @ v_ref[...]


def _backward_values_kernel(q_ref, k_ref, hi_ref, lo_ref, do_ref, dh_ref, dv_ref):
    q, k, hi, lo = q_ref[...], k_ref[...], hi_ref[...], lo_ref[...]
    paired = (q @ k.T) * _decay(hi, lo)
    dh_ref[...] = (q * jnp.exp(hi)[:, None]).T @ do_ref[...]
    dv_ref[...] = paired.T @ do_ref[...]


def _backward_operands_kernel(q_ref, k_ref, hi_ref, lo_ref, h_ref, v_ref, do_ref,
                              dq_ref, dk_ref, dhi_ref, dlo_ref):
    """A block of key columns owes one share of each position's decay
    gradient. Its local `[C, C]` products need no atomic writes or residual."""
    q, k, hi, lo = q_ref[...], k_ref[...], hi_ref[...], lo_ref[...]
    do = do_ref[...]
    d_pair = (do @ v_ref[...].T) * _decay(hi, lo)
    inter = (do @ h_ref[...].T) * jnp.exp(hi)[:, None]
    dq_ref[...] = inter + d_pair @ k
    dk_ref[...] = d_pair.T @ q
    d_diff = d_pair * (q @ k.T)
    d_decay = jnp.sum(d_diff, axis=1) - jnp.sum(d_diff, axis=0)
    dhi_ref[...] = d_decay + jnp.sum(inter * q, axis=1)
    dlo_ref[...] = d_decay


def _whole(array):
    inner = array.shape[3:]
    return pl.BlockSpec((None, *inner), lambda i, j: (i, *(0,) * len(inner)))


def _columns(array, width):
    return pl.BlockSpec((None, array.shape[-2], width), lambda i, j: (i, 0, j))


def _call(body, name, operands, outputs, in_specs, out_specs, blocks, interpret):
    """Flatten only the leading `[B, H, NC]` dimensions: one independent
    row-head-chunk and column block per program, without a layout transpose."""
    from jax.experimental.pallas import triton as plgpu

    rows = math.prod(operands[0].shape[:3])

    def flat(array):
        return array.reshape(rows, *array.shape[3:])

    values = pl.pallas_call(
        body, grid=(rows, blocks), in_specs=in_specs, out_specs=out_specs,
        out_shape=[jax.ShapeDtypeStruct((rows, *out.shape[3:]), jnp.float32) for out in outputs],
        compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=STAGES),
        interpret=interpret, name=name,
    )(*map(flat, operands))
    return tuple(value.reshape(out.shape) for value, out in zip(values, outputs, strict=True))


def _output(q, k, hi, lo, h, v, interpret: bool):
    whole, columned = (q, k, hi, lo), (h, v)
    return _call(_forward_kernel, "gated_delta_output", (*whole, *columned), (v,),
                 [*map(_whole, whole), *(_columns(a, BLOCK) for a in columned)], [_columns(v, BLOCK)],
                 v.shape[-1] // BLOCK, interpret)[0]


def _output_fwd(q, k, hi, lo, h, v, interpret: bool):
    return _output(q, k, hi, lo, h, v, interpret), (q, k, hi, lo, h, v)


def _output_bwd(interpret: bool, residual, do):
    q, k, hi, lo, h, v = residual
    whole = (q, k, hi, lo)
    dh, dv = _call(_backward_values_kernel, "gated_delta_output_values_backward", (*whole, do), (h, v),
                   [*map(_whole, whole), _columns(do, BLOCK)],
                   [_columns(a, BLOCK) for a in (h, v)], v.shape[-1] // BLOCK, interpret)
    width = min(BLOCK, k.shape[-1])
    blocks = k.shape[-1] // width
    partial = jax.ShapeDtypeStruct((*hi.shape[:3], blocks, hi.shape[-1]), jnp.float32)
    parts = pl.BlockSpec((None, None, hi.shape[-1]), lambda i, j: (i, j, 0))
    state_rows = pl.BlockSpec((None, width, h.shape[-1]), lambda i, j: (i, j, 0))
    dq, dk, dhi, dlo = _call(
        _backward_operands_kernel, "gated_delta_output_operands_backward", (*residual, do),
        (q, k, partial, partial),
        [_columns(q, width), _columns(k, width), _whole(hi), _whole(lo), state_rows, _whole(v), _whole(do)],
        [_columns(q, width), _columns(k, width), parts, parts], blocks, interpret)
    return dq, dk, jnp.sum(dhi, axis=-2), jnp.sum(dlo, axis=-2), dh, dv


chunk_output = jax.custom_vjp(_output, nondiff_argnums=(6,))
chunk_output.defvjp(_output_fwd, _output_bwd)
