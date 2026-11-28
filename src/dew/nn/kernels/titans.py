"""Titans' linear memory (row 1449; fla's `chunk_titans_linear_ref`, fla/ops/titans/naive.py, which has no
kernel): a memory `M` learned at test time by gradient descent with momentum `S`, each head's `[D, D]`,

    S_t = eta_t S_{t-1} - 2 theta_t k_t^T u_t,    M_t = (1 - alpha_t) M_{t-1} + S_t,
    o_t = q_t + LN(q_t M_t) w + b,

the memory's residual around its query, as Dew's in-context memory reads it (`dew.nn.memory`'s
`ResidualMemory`; fla's reference reads `q M + LN(q M)`), `2 k_t^T u_t` the gradient at `M` of the inner
loss `|LN(k_t M) w + b - (v_t - k_t)|^2`, `u_t` taken
through the LayerNorm's backward (`inner_gradient`) at the memory entering the token's mini-batch: `chunk`
tokens from the sequence's start share one memory their gradients are taken at, as fla's reference anchors
them (its `M_prev_nabla`). The LayerNorm's backward is TTT's (fla/ops/ttt/naive.py), `(D g - sum g -
x sum(g x)) / (D sigma)`, `sigma = sqrt(var + eps)`; fla's Titans reference writes it `D g - sum g /
(D sigma) - x sum(g x) / (D sigma)`, its first term undivided, which no loss's gradient is.

`titans` takes a mini-batch at a time (docs/design/kernel-templates.md's C class: its orchestration outside
the recurrence tile): its tokens' gradients at once at the memory entering it, then its tokens a step at a
time, each writing its own and reading the pair it leaves, so no query meets a key (a live score of 2^70 by
2^70 overflowed beside a zero write) and no state is weighed after a read (a memory of 2^70 forgotten whole
read `0 * inf`). Documents are not packed: a mini-batch is the sequence's `chunk` tokens, which a document
starting within one would split (fla's reference takes none either)."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from jax.ad_checkpoint import checkpoint_name

from dew.nn import sharding
from dew.nn.aligned import aligned_sum
from dew.nn.inputs import sequence_blocks
from dew.nn.precision import at_least_fp32
from dew.nn.residuals import MEMORY_RESIDUALS, RECOMPUTED

CARRY, CHUNK = MEMORY_RESIDUALS
"""The in-context memory's remat names (`dew.nn.memory.CARRY`, `CHUNK`): a mini-batch's entering state, and
its outputs with the state it leaves."""


def layer_normed(x, eps: float):
    """`x`'s LayerNorm over its last axis without an affine, `(x - mean x) / sqrt(var + eps)`: the plain
    formula, fla's reference's and the token recurrence's, its derivatives autodiff's, so every order is
    finite wherever the plain formula's is (docs/design/testing.md, the derivative contract). A quotient by
    the root, not a product by `rsqrt`, whose derivative `rsqrt(v) / v` passes the range first (2^150 at
    `eps` 2^-100, its product with a zero NaN)."""
    centered = x - jnp.mean(x, axis=-1, keepdims=True)
    return centered / jnp.sqrt(jnp.mean(centered * centered, axis=-1, keepdims=True) + eps)


def inner_gradient(k, v, km, weight, bias, eps: float):
    """Half the inner loss's gradient at the memory's readout `km` (`k M`), `[..., D]`: the LayerNorm's
    output `LN(km) w + b` against the target `v - k`, its error times `w`, through the LayerNorm's pullback
    (autodiff's, `layer_normed`)."""
    normed, pullback = jax.vjp(functools.partial(layer_normed, eps=eps), km)
    (gradient,) = pullback((normed * weight + bias - (v - k)) * weight)
    return gradient


def mini_batches(chunk: int, keys, *operands, valid=None):
    """`keys` and each of `operands` `[B, H, T, ...]` as `[N, B, H, chunk, ...]`, their tokens past the last
    whole chunk zero (`inputs.sequence_blocks`), and which tokens are real, `[N, B, H, chunk]`, placed and
    varying as the keys, which write the states (an Explicit mesh's rows; a checked `shard_map`'s axes: a
    mask of the queries' varied the states where only the queries did, train 1455's review), a token
    `valid` (`[B, T]`) marks absent none."""
    def batched(x):
        return jnp.moveaxis(sequence_blocks(jnp.moveaxis(x, 2, 1), chunk, leading=True), 3, 2)

    real = jnp.ones_like(keys[..., 0], dtype=bool)
    if valid is not None:
        real = real & jnp.asarray(valid, bool)[:, None]
    return (*map(batched, (keys, *operands)), batched(real))


def absent_zeroed(valid, *operands) -> tuple:
    """`operands` (`[B, H, T, ...]`) zero at the tokens `valid` (`[B, T]`) marks absent, as they are where it
    is None, so an absent token's operands meet no product: a caller's padding (an in-context memory's
    compacted rows' tails, `dew.nn.memory`) holds any value, whose overflow met its zero cotangent as
    NaN."""
    if valid is None:
        return operands
    present = jnp.asarray(valid, bool)[:, None]
    return tuple(jnp.where(present.reshape(*present.shape, *(1,) * (x.ndim - 3)), x, jnp.zeros((), x.dtype))
                 for x in operands)


def real_gradients(k, v, readout, real, weight, bias, eps: float):
    """Each real token's inner gradient at its `readout` (`inner_gradient`), an absent one's zero, its
    readout zero before the LayerNorm sees it (an entering bias of 2^127 overflowed its variance, NaN)."""
    live, zeros = real[..., None], jnp.zeros_like(readout)
    return jnp.where(live, inner_gradient(k, v, jnp.where(live, readout, zeros), weight, bias, eps), zeros)


def walked_batches(mini_batch, states, batches):
    """`mini_batch(entering, at)` over `batches` in turn (`lax.scan`) from `states`, with the in-context
    memory's remat seams (`dew.nn.memory`'s chunks'): each mini-batch's entering state named `CARRY`, its
    leaving state and outputs `CHUNK`, and where the run's remat recomputes the chunks (`RECOMPUTED`) each
    mini-batch recomputed from the state it entered with."""
    def named(entering, at):
        return checkpoint_name(mini_batch(checkpoint_name(entering, CARRY), at), CHUNK)

    if CHUNK in RECOMPUTED.get():
        named = jax.checkpoint(named, prevent_cse=False)
    return jax.lax.scan(named, states, batches)


def read_out(walked, queries, weight, bias, eps: float, dtype):
    """A walk's readouts `[N, C, B, H, D]` as the call's output `[B, T, H, D]` in `dtype`, the padding
    dropped: each query (`[B, H, T, D]`, as the walk read it) plus its readout's LayerNorm under the inner
    loss's affine, the memory's residual around its input (Dew's `ResidualMemory`, the TTT paper's Sec
    2.7)."""
    out = jnp.moveaxis(walked.reshape(-1, *walked.shape[2:]), 0, 2)[:, :, :queries.shape[2]]
    return jnp.moveaxis(queries + layer_normed(out, eps) * weight + bias, 1, 2).astype(dtype)


def placed_alike(affine, *operands):
    """`operands` `[B, H, ...]` and the inner loss's `affine` `[H, 1, D]` each placed alike where an Explicit
    axis splits one: the operands with the rows and heads of the first operand split there
    (`sharding.explicit_spec`), or the heads of the affine where it alone is split, and the affine with
    those heads, the rest whole, so the walk's products and selects meet one placement and its scans' seeds
    are placed as its steps leave them: a replicated query, or a zero seed, beside batch-split keys, values
    and rates was refused (train 1409's review and a head review), as were replicated states beside writes
    a head-split affine placed (train 1430's)."""
    placed = next(filter(None, map(sharding.explicit_spec, operands)), None)
    if placed is not None:
        mesh, rows, heads = placed.mesh, *placed.spec[:2]
    else:
        split = next(filter(None, map(sharding.explicit_spec, affine)), None)
        if split is None:
            return affine, operands
        mesh, rows, heads = split.mesh, None, split.spec[0]

    def placed_as(x, *axes):
        spec = jax.sharding.PartitionSpec(*axes, *(None,) * (x.ndim - len(axes)))
        return jax.sharding.reshard(x, jax.sharding.NamedSharding(mesh, spec))
    return tuple(placed_as(x, heads) for x in affine), tuple(placed_as(x, rows, heads) for x in operands)


def prepared_operands(query, key, value, weight, bias, gates, state, shapes, scale=None):
    """A mini-batched memory's operands (`titans`, `ttt.ttt_linear`) in `at_least_fp32` of their dtype, heads
    before tokens (`[B, H, T, D]`; `gates` each `[B, H, T]`), the inner loss's affine `[H, 1, D]` and the
    entering states (zeros `[B, H, *shape]` of each of `shapes` where `state` is None), placed alike, the
    state-writing keys' rows and heads first (an Explicit mesh's; `placed_alike`), and the states typed as
    every operand writing them varies (a checked `shard_map`'s manual axes), as every step's states are, the
    queries, which read them alone, apart (a state left as it entered, replicated beside queries split, was
    typed split: train 1455's review); `(q, k, v, affine, gates, states)`, the queries times `scale` where
    given. One preparation for both memories (train 1439's review)."""
    work = at_least_fp32(jnp.result_type(query, key, value))

    def heads(x):
        return jnp.moveaxis(jnp.asarray(x, work), 2, 1)

    q, k, v = heads(query), heads(key), heads(value)
    q = q if scale is None else q * scale
    affine = tuple(jnp.asarray(x, work)[:, None] for x in (weight, bias))
    gates = tuple(heads(x) for x in gates)
    states = (tuple(jnp.zeros((*k.shape[:2], *shape), work) for shape in shapes) if state is None
              else tuple(jnp.asarray(x, work) for x in state))
    affine, (k, q, v, *rest) = placed_alike(affine, k, q, v, *gates, *states)
    gates, states = tuple(rest[:len(gates)]), tuple(rest[len(gates):])
    states = tuple(sharding.varying_like(x, k, v, *gates, *affine, *states) for x in states)
    return q, k, v, affine, gates, states


def titans(query, key, value, weight, bias, theta, alpha, eta, state=None, *, chunk: int = 16,
           eps: float = 1e-6, valid=None, coefficient: float = 2.0):
    """Titans' linear memory a mini-batch of `chunk` tokens at a time: `query`, `key` and `value`
    `[B, T, H, D]`, the inner loss's LayerNorm affine `weight` and `bias` `[H, D]`, the learning rate
    `theta`, forgetting `alpha` and momentum `eta` `[B, T, H]`, and `state` the entering `(M, S)`
    `[B, H, D, D]` each (zeros where None, fla's momentum always so); the output `[B, T, H, D]` in the
    operands' dtype and the final `(M, S)` in `at_least_fp32` of it, which every step computes in. Each
    mini-batch's gradients at once at the memory entering it, a padded token's none (its readout zero before
    the LayerNorm sees it), then its tokens a step at a time. A partial last mini-batch leaves the state its
    last real token does, as the token recurrence: its padding changes nothing (a mini-batch boundary
    elsewhere moves the memory later inner gradients are taken at, padding none). An intentional divergence
    from fla's reference, which pads it and gives its last padded slot the last real token's gates
    (fla/ops/titans/naive.py:332-342 at 413ad2c), decaying the state again (one real token at `alpha` 1/2 in
    a mini-batch of two keeps a quarter of the memory where the recurrence keeps half). A token `valid`
    (`[B, T]`) marks absent is none: its operands zero (`absent_zeroed`), its readout what the walk reads,
    the state as it was. `coefficient` is the write's factor before its rate, `coefficient theta k u` (2,
    the squared loss's gradient; 1 where the caller's loss is half of it, as the in-context memory's), one
    factor of the exact product, so no rate is halved or doubled first."""
    dtype = jnp.result_type(query, key, value)
    # Titans' memory and momentum each `[D, Dv]`.
    shapes = ((key.shape[-1], value.shape[-1]),) * 2
    q, k, v, (weight, bias), (theta, alpha, eta), states = prepared_operands(
        query, key, value, weight, bias, (theta, alpha, eta), state, shapes)
    q, k, v, theta, alpha, eta = absent_zeroed(valid, q, k, v, theta, alpha, eta)

    def mini_batch(pair, at):
        q, k, v, theta, alpha, eta, real = at
        gradient = real_gradients(k, v, k @ pair[0], real, weight, bias, eps)

        def token(pair, along):
            q, k, gradient, theta, alpha, eta, real = along
            # The write `2 theta k u` one exact product of the 2, the rate, the key and the gradient, rounded
            # once (`aligned_sum`), its transpose's alike: no partial product leaves the range its value does
            # not, in either direction (weighed onto the gradient first, one of `3 2^126` overflowed, train
            # 1430's review; doubled onto a key of 2^127 first, a zero gradient wrote NaN, train 1455's;
            # doubled onto a cotangent of 2^127 first, the transpose overflowed, a head review; the rate of
            # 2^127 doubled first overflowed, train 1461's).
            write = aligned_sum([[jnp.asarray(coefficient, theta.dtype), theta[..., None, None],
                                  k[..., :, None], gradient[..., None, :]]])
            momentum = eta[..., None, None] * pair[1] - write
            memory = (1.0 - alpha)[..., None, None] * pair[0] + momentum
            # A padded token leaves the pair as it was.
            pair = tuple(jnp.where(real[..., None, None], new, old)
                         for new, old in zip((memory, momentum), pair, strict=True))
            return pair, jnp.einsum('bhk,bhkv->bhv', q, pair[0])

        tokens = tuple(jnp.moveaxis(x, 2, 0) for x in (q, k, gradient, theta, alpha, eta, real))
        return jax.lax.scan(token, pair, tokens)

    keys, *rest = mini_batches(chunk, k, q, v, theta, alpha, eta, valid=valid)
    final, out = walked_batches(mini_batch, states, (rest[0], keys, *rest[1:]))
    return read_out(out, q, weight, bias, eps, dtype), final
