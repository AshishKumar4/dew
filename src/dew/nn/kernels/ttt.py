"""TTT-Linear (row 1448), as Dew's in-context memory computes it (`dew.nn.memory.InContextMemory`, a
`ResidualMemory` whose norm is slow, on the l2 objective): a memory `W` `[D, D]` and its bias `b` `[D]`
learned at test time, a mini-batch of `chunk` tokens at a time, each mini-batch's inner gradients `u_s` taken
at the memory entering it (`titans.inner_gradient`, the LayerNorm reconstruction loss's, `|k + (LN(k W + b) w
+ c) - v|^2`), each token stepping the memory by its own gradient at its own learning rate and reading the
memory its step leaves through the residual (`titans.read_out`),

    W_t = W_0 - sum_{s <= t} eta_s k_s^T u_s,    b_t = b_0 - sum_{s <= t} eta_s u_s,
    o_t = q_t + (LN(q_t W_t + b_t) w + c),

`q` scaled by `scale`. fla's reference (`chunk_ttt_linear_ref`, fla/ops/ttt/naive.py) rates a token's whole
prefix by the reading token's rate and reads `Z + LN(Z)` around its readout: the TTT paper's residual is
the query's (Sec 2.7), as Dew's memory's. A C-class orchestrator (docs/design/kernel-templates.md: its
orchestration outside the recurrence tile), as Titans' (`titans.titans`): a mini-batch's gradients at once,
then its steps, each rated step one exact product rounded once (`aligned_sum`), the rate meeting its key and
update before they multiply (keys of 2^70 and updates of 2^69 under a rate of 2^-20 write 2^119: train
1470's review), their prefix sums at once, grouped as the in-context memory's general path groups a chunk's
(one associative scan), each token reading the entering state plus its prefix, so no query meets a key. A
partial last mini-batch's absent tokens write nothing, their readouts zero before the LayerNorm sees them."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dew.nn.aligned import aligned_sum
from dew.nn.kernels.titans import mini_batches, prepared_operands, read_out, real_gradients
from dew.nn.residuals import memory_chunks


def ttt_linear(query, key, value, weight, bias, eta, state=None, *, chunk: int = 16, scale=None,
               eps: float = 1e-6, valid=None):
    """TTT-Linear a mini-batch of `chunk` tokens at a time: `query`, `key` and `value` `[B, T, H, D]`, the
    inner loss's LayerNorm affine `weight` and `bias` `[H, D]`, the learning rate `eta` `[B, T, H]` and
    `state` the entering `(W, b)` (`[B, H, D, D]`, `[B, H, 1, D]`; zeros where None); the output
    `[B, T, H, D]` in the operands' dtype and the final `(W, b)` in `at_least_fp32` of it, which every
    step computes in. A token `valid` (`[B, T]`) marks absent is none: its operands zero
    (`titans.prepared_operands`), its readout zero, the state as it was."""
    dtype = jnp.result_type(query, key, value)
    # TTT's memory `[D, Dv]` and bias `[1, Dv]`; its queries scaled.
    shapes = ((key.shape[-1], value.shape[-1]), (1, value.shape[-1]))
    scale = key.shape[-1] ** -0.5 if scale is None else scale
    q, k, v, (weight, bias), (eta,), states = prepared_operands(
        query, key, value, weight, bias, (eta,), state, shapes, scale, valid=valid)

    def mini_batch(entering, at):
        q, k, v, eta, real = at
        # Each real token's gradient at the memory entering the mini-batch.
        update = real_gradients(k, v, k @ entering[0] + entering[1], real, weight, bias, eps)

        # Each real token's step one exact product of its rate, key and update rounded once, the bias's its
        # rate's; an absent token steps nothing.
        live = real[..., None, None]
        rated = aligned_sum([[eta[..., None, None], k[..., :, None], update[..., None, :]]])
        steps = (jnp.where(live, -rated, 0), jnp.where(live, -(eta[..., None] * update)[..., None, :], 0))
        # The mini-batch's steps summed apart from the entering state, which each read and the outgoing state
        # alone add, grouped as the memory's general path groups a chunk's (`dew.nn.memory`'s chunk, one
        # associative scan over its tokens): a step below the entering state's last place is kept rather than
        # rounded away token by token, and steps that cancel are summed as that path sums them (79g's review:
        # four steps of 0.98 into a memory of 2^25 left it as it was; bias steps of 2^24, 0, 1 and -2^24
        # summed left to right lost the 1).
        prefixes = tuple(jax.lax.associative_scan(jnp.add, summed, axis=2) for summed in steps)
        states = tuple(held[:, :, None] + prefix for held, prefix in zip(entering, prefixes, strict=True))
        # An absent token's readout zero, not its kept bias, before the LayerNorm reads it (a bias of 2^127
        # overflowed its variance, NaN through the absent output's zero cotangent).
        readout = jnp.einsum('bhck,bhckv->bhcv', q, states[0]) + states[1][:, :, :, 0]
        reads = jnp.where(real[..., None], readout, jnp.zeros((), readout.dtype))
        return tuple(state[:, :, -1] for state in states), jnp.moveaxis(reads, 2, 0)

    keys, *rest = mini_batches(chunk, k, q, v, eta, valid=valid)
    final, out = memory_chunks(mini_batch, states, (rest[0], keys, *rest[1:]))
    return read_out(out, q, weight, bias, eps, dtype), final
