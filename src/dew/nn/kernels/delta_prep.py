"""The chunked gated delta rule's in-chunk correction, forward and backward.

The strictly lower products and compensated pairwise decays stay on chip.
The inverse follows dew.nn.linear.strictly_lower_inverse's block doubling:
16-wide diagonal blocks use masked forward substitution, then neighbouring
inverses merge with `X2 A21 X1`. Unlike a power series, every intermediate
is bounded by the inverse's entries even when a chunk's keys align. This
is fla-org/flash-linear-attention v0.5.2, fla/ops/utils/solve_tril.py:58-86,
147-169, with the products before and after the solve fused into the same
program (fla/ops/gated_delta_rule/chunk_fwd.py:40-68). The inverse alone is
saved for the VJP, as in fla/ops/gated_delta_rule/wy_fast.py:119-248.

Each program owns one row-head-chunk. It walks the key and value columns
in blocks so its square products fit on chip, without atomics or batched
GEMMs. Products take the configured default matmul precision.
"""

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .delta_output import _decay
from .generation import first_refusal

BLOCK = 32
WARPS = 4
STAGES = 2


def refusal(key, value) -> str | None:
    """Bounds beyond delta_chunks.refusal for the inverse and its products."""
    return first_refusal(
        (key.shape[-2] <= 64, f"prep chunks of {key.shape[-2]}, wider than 64"),
        (key.shape[-1] <= 128, f"prep keys {key.shape[-1]} wide, wider than 128"),
        (value.shape[-1] <= 256, f"prep values {value.shape[-1]} wide, wider than 256"),
    )


def _join(first, second, axis):
    """Join two register tiles with Triton's two-way stack and reshape.
    Its Pallas lowering does not concatenate tiles wider than one entry."""
    shape = list(first.shape)
    shape[axis] *= 2
    return jnp.moveaxis(jnp.stack((first, second), axis=-1), -1, axis).reshape(shape)


def _inverse(a):
    """Diagonal blocks, then their pairwise merges: never powers of A.
    Masked row reductions keep the 16-wide solve below Triton's dot bound."""
    size = a.shape[0]
    if size == 16:
        index = jnp.arange(size)

        def row(i, inverse):
            original = jnp.sum(jnp.where(index[:, None] == i, a, 0.0), axis=0)
            corrected = original + jnp.sum(original[:, None] * inverse, axis=0)
            return jnp.where(index[:, None] == i, corrected, inverse)

        return jax.lax.fori_loop(2, size, row, a) + jnp.eye(size, dtype=jnp.float32)
    top, bottom = jnp.split(a, 2, axis=0)
    a11, _ = jnp.split(top, 2, axis=1)
    a21, a22 = jnp.split(bottom, 2, axis=1)
    first, second = _inverse(a11), _inverse(a22)
    below = (second @ a21) @ first
    return _join(_join(first, jnp.zeros_like(first), 1), _join(below, second, 1), 0)


def _forward_kernel(k_ref, v_ref, beta_ref, hi_ref, lo_ref, w_ref, u_ref, inverse_ref):
    k, beta, hi, lo = k_ref[...], beta_ref[...], hi_ref[...], lo_ref[...]
    size, width = k.shape
    index = jnp.arange(size)
    paired = ((k * beta[:, None]) @ k.T) * _decay(hi, lo)
    inverse = _inverse(jnp.where(index[:, None] > index[None, :], -paired, 0.0))
    inverse_ref[...] = inverse
    keyed = min(BLOCK, width)

    def key_columns(i, carry):
        columns = pl.ds(i * keyed, keyed)
        kb = k_ref[:, columns] * beta[:, None]
        w_ref[:, columns] = inverse @ (kb * jnp.exp(hi)[:, None])
        return carry

    def value_columns(i, carry):
        columns = pl.ds(i * BLOCK, BLOCK)
        u_ref[:, columns] = inverse @ (v_ref[:, columns] * beta[:, None])
        return carry

    jax.lax.fori_loop(0, width // keyed, key_columns, None)
    jax.lax.fori_loop(0, v_ref.shape[-1] // BLOCK, value_columns, None)


def _backward_kernel(k_ref, v_ref, beta_ref, hi_ref, lo_ref, inverse_ref, dw_ref, du_ref,
                     dk_ref, dv_ref, db_ref, dhi_ref, dlo_ref):
    beta, hi, lo, inverse = beta_ref[...], hi_ref[...], lo_ref[...], inverse_ref[...]
    size, width = k_ref.shape
    keyed = min(BLOCK, width)

    def key_products(i, d_inverse):
        columns = pl.ds(i * keyed, keyed)
        kbe = (k_ref[:, columns] * beta[:, None]) * jnp.exp(hi)[:, None]
        return d_inverse + dw_ref[:, columns] @ kbe.T

    def value_products(i, carry):
        d_inverse, d_beta = carry
        columns = pl.ds(i * BLOCK, BLOCK)
        value, du = v_ref[:, columns], du_ref[:, columns]
        d_vb = inverse.T @ du
        dv_ref[:, columns] = d_vb * beta[:, None]
        return (d_inverse + du @ (value * beta[:, None]).T,
                d_beta + jnp.sum(d_vb * value, axis=1))

    d_inverse = jax.lax.fori_loop(0, width // keyed, key_products, jnp.zeros((size, size), jnp.float32))
    d_inverse, d_beta = jax.lax.fori_loop(
        0, v_ref.shape[-1] // BLOCK, value_products, (d_inverse, jnp.zeros(size, jnp.float32)))
    d_a = (inverse.T @ d_inverse) @ inverse.T
    index = jnp.arange(size)
    d_kk = -jnp.where(index[:, None] > index[None, :], d_a, 0.0) * _decay(hi, lo)

    def key_gradients(i, carry):
        d_beta, d_hi, d_diff = carry
        columns = pl.ds(i * keyed, keyed)
        key = k_ref[:, columns]
        kb = key * beta[:, None]
        kbe = kb * jnp.exp(hi)[:, None]
        d_kbe = inverse.T @ dw_ref[:, columns]
        d_kb = d_kk @ key + d_kbe * jnp.exp(hi)[:, None]
        dk_ref[:, columns] = d_kk.T @ kb + d_kb * beta[:, None]
        return (d_beta + jnp.sum(d_kb * key, axis=1),
                d_hi + jnp.sum(d_kbe * kbe, axis=1), d_diff + d_kk * (kb @ key.T))

    d_beta, d_hi, d_diff = jax.lax.fori_loop(
        0, width // keyed, key_gradients,
        (d_beta, jnp.zeros(size, jnp.float32), jnp.zeros((size, size), jnp.float32)))
    d_decay = jnp.sum(d_diff, axis=1) - jnp.sum(d_diff, axis=0)
    db_ref[...] = d_beta
    dhi_ref[...] = d_hi + d_decay
    dlo_ref[...] = d_decay


def _call(body, name, operands, outputs, interpret):
    """One program per `[B, H, NC]` chunk, each ref a whole inner operand.
    Column loops load only the blocks needed for their products."""
    from jax.experimental.pallas import triton as plgpu

    rows = math.prod(operands[0].shape[:3])

    def spec(array):
        inner = array.shape[3:]
        return pl.BlockSpec((None, *inner), lambda i: (i, *(0,) * len(inner)))

    blocks = pl.pallas_call(
        body, grid=(rows,), in_specs=list(map(spec, operands)), out_specs=list(map(spec, outputs)),
        out_shape=[jax.ShapeDtypeStruct((rows, *out.shape[3:]), jnp.float32) for out in outputs],
        compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=STAGES),
        interpret=interpret, name=name,
    )(*(a.reshape(rows, *a.shape[3:]) for a in operands))
    return tuple(block.reshape(out.shape) for block, out in zip(blocks, outputs, strict=True))


def _prep_with_inverse(k, v, beta, hi, lo, interpret: bool):
    inverse = jax.ShapeDtypeStruct((*k.shape[:4], k.shape[-2]), jnp.float32)
    return _call(_forward_kernel, "gated_delta_prep", (k, v, beta, hi, lo), (k, v, inverse), interpret)


def _prep(k, v, beta, hi, lo, interpret: bool):
    w, u, _ = _prep_with_inverse(k, v, beta, hi, lo, interpret)
    return w, u


def _prep_fwd(k, v, beta, hi, lo, interpret: bool):
    w, u, inverse = _prep_with_inverse(k, v, beta, hi, lo, interpret)
    return (w, u), (k, v, beta, hi, lo, inverse)


def _prep_bwd(interpret: bool, residual, cotangent):
    k, v, beta, hi, lo, _ = residual
    return _call(_backward_kernel, "gated_delta_prep_backward", (*residual, *cotangent),
                 (k, v, beta, hi, lo), interpret)


chunk_prep = jax.custom_vjp(_prep, nondiff_argnums=(5,))
chunk_prep.defvjp(_prep_fwd, _prep_bwd)
