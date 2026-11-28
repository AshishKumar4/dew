"""TTT-Linear (row 1448), as Dew's in-context memory computes it (`dew.nn.memory.InContextMemory`, a
`ResidualMemory` whose norm is slow, on the l2 objective): a memory `W` `[D, D]` and its bias `b` `[D]`
learned at test time, a mini-batch of `chunk` tokens at a time, each mini-batch's inner gradients `u_s` taken
at the memory entering it (`titans.inner_gradient`, the LayerNorm reconstruction loss's, `|LN(k W + b) w + c
- (v - k)|^2`), each token stepping the memory by its own gradient at its own learning rate and reading the
memory its step leaves through the residual (`titans.read_out`),

    W_t = W_0 - sum_{s <= t} eta_s k_s^T u_s,    b_t = b_0 - sum_{s <= t} eta_s u_s,
    o_t = q_t + LN(q_t W_t + b_t) w + c,

`q` scaled by `scale`. fla's reference (`chunk_ttt_linear_ref`, fla/ops/ttt/naive.py) rates a token's whole
prefix by the reading token's rate and reads `Z + LN(Z)` around its readout: the TTT paper's residual is
the query's (Sec 2.7), as Dew's memory's. A C-class orchestrator (docs/design/kernel-templates.md: its
orchestration outside the recurrence tile), as Titans' (`titans.titans`): a mini-batch's gradients at once,
then its tokens a step at a time, each rated step one exact product rounded once (`aligned_sum`), the rate
meeting its key and update before they multiply (keys of 2^70 and updates of 2^69 under a rate of 2^-20
write 2^119: train 1470's review), so no query meets a key. A partial last mini-batch's absent tokens write
nothing, their readouts zero before the LayerNorm sees them."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dew.nn.aligned import aligned_sum
from dew.nn.kernels.titans import (
    absent_zeroed,
    mini_batches,
    prepared_operands,
    read_out,
    real_gradients,
    walked_batches,
)


def ttt_linear(query, key, value, weight, bias, eta, state=None, *, chunk: int = 16, scale=None,
               eps: float = 1e-6, valid=None):
    """TTT-Linear a mini-batch of `chunk` tokens at a time: `query`, `key` and `value` `[B, T, H, D]`, the
    inner loss's LayerNorm affine `weight` and `bias` `[H, D]`, the learning rate `eta` `[B, T, H]` and
    `state` the entering `(W, b)` (`[B, H, D, D]`, `[B, H, 1, D]`; zeros where None); the output
    `[B, T, H, D]` in the operands' dtype and the final `(W, b)` in `at_least_fp32` of it, which every
    step computes in. A token `valid` (`[B, T]`) marks absent is none: its operands zero
    (`titans.absent_zeroed`), the state as it was."""
    dtype = jnp.result_type(query, key, value)
    # TTT's memory `[D, Dv]` and bias `[1, Dv]`; its queries scaled.
    shapes = ((key.shape[-1], value.shape[-1]), (1, value.shape[-1]))
    scale = key.shape[-1] ** -0.5 if scale is None else scale
    q, k, v, (weight, bias), (eta,), states = prepared_operands(
        query, key, value, weight, bias, (eta,), state, shapes, scale)
    q, k, v, eta = absent_zeroed(valid, q, k, v, eta)

    def mini_batch(entering, at):
        q, k, v, eta, real = at
        # Each real token's gradient at the memory entering the mini-batch.
        update = real_gradients(k, v, k @ entering[0] + entering[1], real, weight, bias, eps)

        def token(state, along):
            q, k, update, eta, real = along
            # Its step one exact product of its rate, key and update rounded once, the bias's its rate's.
            step = (aligned_sum([[eta[..., None, None], k[..., :, None], update[..., None, :]]]),
                    (eta[..., None] * update)[:, :, None])
            # An absent token leaves the state as it was.
            state = tuple(jnp.where(real[..., None, None], held - stepped, held)
                          for held, stepped in zip(state, step, strict=True))
            return state, jnp.einsum('bhk,bhkv->bhv', q, state[0]) + state[1][:, :, 0]

        tokens = tuple(jnp.moveaxis(x, 2, 0) for x in (q, k, update, eta, real))
        return jax.lax.scan(token, entering, tokens)

    keys, *rest = mini_batches(chunk, k, q, v, eta, valid=valid)
    final, out = walked_batches(mini_batch, states, (rest[0], keys, *rest[1:]))
    return read_out(out, q, weight, bias, eps, dtype), final
