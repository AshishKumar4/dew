"""TransDreamer's transformer state-space model (Chen et al., arXiv 2202.09481): a Dreamer world model's
dynamics as a causal transformer over the latents and actions before each step.

At step t the posterior reads the observation's tokens alone, q(z_t | x_t), so a window's latents are
drawn at once; the transformer reads each step's previous latent and action, (z_{t-1}, a_{t-1}), and
answers the deterministic state h_t, from which the prior predicts z_t, p(z_t | h_t). h_t therefore
holds nothing of x_t, and the prior is a prediction. Attention stays within an episode: a step that
starts one reads a learned start embedding and attends to nothing before it. Training runs the window in
parallel; acting and imagination run the same attention a step at a time over a cache of keys and
values (`Cache`), so the policy and the imagined rollouts read the function the model was trained as.

The sizes default to FlyDreamer's (deter 512, 32 categoricals of 32 classes, 4 layers of 8 heads, rotary
positions, RMS-normed queries and keys, a gated feed-forward four widths wide before its 2/3).
"""

from __future__ import annotations

import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct

from dew.nn.rope import rotary_freqs, rotate


@struct.dataclass
class Cache:
    """The keys and values a step attends to, `[B, layers, C, heads, head]` each, with each slot's
    `position` within its episode and whether it is `valid` `[B, C]`, the slot the next step writes
    (`index`, wrapping round the C slots) and that step's `position` within its episode `[B]`."""

    keys: jax.Array
    values: jax.Array
    valid: jax.Array
    index: jax.Array
    position: jax.Array


class _Block(nn.Module):
    """A pre-norm transformer block: rotary, RMS-normed multi-head attention, then a SiLU-gated
    feed-forward layer, each added to its input."""

    width: int
    heads: int
    ffup: int
    dtype: jnp.dtype

    def setup(self) -> None:
        head = self.width // self.heads
        dense = functools.partial(nn.DenseGeneral, dtype=self.dtype, param_dtype=self.dtype)
        norm = functools.partial(nn.RMSNorm, epsilon=1e-6, dtype=self.dtype, param_dtype=self.dtype)
        self.attention_norm, self.feed_norm = norm(), norm()
        self.query, self.key, self.value = (dense((self.heads, head)) for _ in range(3))
        self.query_norm, self.key_norm = norm(), norm()
        self.out = dense(self.width, axis=(-2, -1))
        hidden = max(self.width, int(self.width * self.ffup * 2 / 3) // 32 * 32)
        self.gate, self.up = dense(hidden), dense(hidden)
        self.down = dense(self.width)

    def projected(self, x: jax.Array, positions: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """The rotated queries and keys, and the values, of rows `[B, S, width]` at `positions` `[B, S]`."""
        h = self.attention_norm(x)
        cos, sin = rotary_freqs(positions, self.width // self.heads, 10000.0, dtype=jnp.float32)
        query = rotate(self.query_norm(self.query(h)), cos, sin)
        key = rotate(self.key_norm(self.key(h)), cos, sin)
        return query, key, self.value(h)

    def finished(self, x: jax.Array, query: jax.Array, keys: jax.Array, values: jax.Array,
                 mask: jax.Array) -> jax.Array:
        """The block's output for rows `x` whose queries attend to `keys` and `values` under `mask`
        `[B, S, K]`."""
        attended = jax.nn.dot_product_attention(query, keys, values, mask=mask[:, None])
        x = x + self.out(attended)
        h = self.feed_norm(x)
        return x + self.down(nn.silu(self.gate(h)) * self.up(h))


class TSSM(nn.Module):
    """TransDreamer's dynamics (module docstring) over `tokens` `[..., E]` and actions of `action_size`."""

    action_size: int
    stoch: int = 32
    classes: int = 32
    width: int = 512
    layers: int = 4
    heads: int = 8
    ffup: int = 4
    unimix: float = 0.01
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        dense = functools.partial(nn.Dense, dtype=self.dtype, param_dtype=self.dtype)
        norm = functools.partial(nn.RMSNorm, epsilon=1e-6, dtype=self.dtype, param_dtype=self.dtype)
        self.embed = dense(self.width)
        self.start = self.param("start", nn.initializers.normal(0.02), (self.width,), self.dtype)
        self.blocks = [_Block(self.width, self.heads, self.ffup, self.dtype) for _ in range(self.layers)]
        self.out_norm = norm()
        self.prior_layers = [(dense(self.width), norm()) for _ in range(2)]
        self.prior_out = dense(self.stoch * self.classes)
        self.post_hidden, self.post_norm = dense(self.width), norm()
        self.post_out = dense(self.stoch * self.classes)

    def __call__(self, tokens: jax.Array, previous: jax.Array, first: jax.Array) -> jax.Array:
        """Initializes every parameter: the posterior over `tokens`, then a window's states over its
        drawn latents."""
        stoch = jax.nn.one_hot(jnp.argmax(self.posterior(tokens), -1), self.classes, dtype=self.dtype)
        return self.prior(self.states(self.inputs(stoch, previous, first), first))

    def _logits(self, x: jax.Array) -> jax.Array:
        """Log-probabilities `[..., stoch, classes]`, mixed with the uniform by `unimix`."""
        probs = jax.nn.softmax(x.reshape(*x.shape[:-1], self.stoch, self.classes).astype(jnp.float32), -1)
        return jnp.log((1 - self.unimix) * probs + self.unimix / self.classes)

    def posterior(self, tokens: jax.Array) -> jax.Array:
        """q(z_t | x_t): the latent's log-probabilities from an observation's tokens alone."""
        return self._logits(self.post_out(nn.silu(self.post_norm(self.post_hidden(tokens)))))

    def prior(self, deter: jax.Array) -> jax.Array:
        """p(z_t | h_t): the latent's log-probabilities predicted from the state."""
        x = deter
        for linear, norm in self.prior_layers:
            x = nn.silu(norm(linear(x)))
        return self._logits(self.prior_out(x))

    def inputs(self, stoch: jax.Array, previous: jax.Array, first: jax.Array) -> jax.Array:
        """Each step's input `[..., width]`: its previous latent `stoch` `[..., S, K]` (each step's own,
        shifted by one along the step axis by the caller) and action, embedded, or the start embedding
        where the step `first` starts an episode."""
        flat = stoch.reshape(*stoch.shape[:-2], -1).astype(self.dtype)
        embedded = self.embed(jnp.concatenate([flat, previous.astype(self.dtype)], -1))
        return jnp.where(first[..., None], self.start, embedded)

    def states(self, x: jax.Array, first: jax.Array) -> jax.Array:
        """The states h `[B, T, width]` of a window's inputs `x` `[B, T, width]`, each step attending to
        the steps of its episode up to itself; the window's first step starts its own segment."""
        segment, positions = _segments(first)
        mask = (segment[:, :, None] == segment[:, None, :]) & jnp.tril(jnp.ones((x.shape[1],) * 2, bool))
        for block in self.blocks:
            query, keys, values = block.projected(x, positions)
            x = block.finished(x, query, keys, values, mask)
        return self.out_norm(x)

    def window_cache(self, x: jax.Array, first: jax.Array, horizon: int) -> Cache:
        """For each step k of a window `[B, T]`, the cache imagination from it attends to: the window's
        keys and values at the steps of k's episode up to k, `B * T` rows of `T + horizon` slots, the
        next written at slot T, at the position after k's."""
        rows, length = first.shape
        segment, positions = _segments(first)
        # [B, k, t]: the window's step t is in the cache of start k.
        valid = (segment[:, :, None] == segment[:, None, :]) & jnp.tril(jnp.ones((length,) * 2, bool))
        keys, values = [], []
        for block in self.blocks:
            query, key, value = block.projected(x, positions)
            keys.append(key)
            values.append(value)
            x = block.finished(x, query, key, value, valid)
        stacked = [jnp.stack(held, 1) for held in (keys, values)]  # [B, layers, T, heads, head]

        def spread(held):
            tiled = jnp.broadcast_to(held[:, None], (rows, length, *held.shape[1:]))
            padded = jnp.pad(tiled, [(0, 0)] * 3 + [(0, horizon)] + [(0, 0)] * 2)
            return padded.reshape(rows * length, *padded.shape[2:])

        valid = jnp.pad(valid, ((0, 0), (0, 0), (0, horizon))).reshape(rows * length, length + horizon)
        return Cache(*(spread(held) for held in stacked), valid,
                     jnp.full(rows * length, length, jnp.int32), (positions + 1).reshape(-1))

    def empty_cache(self, rows: int, size: int) -> Cache:
        """A cache of `size` slots holding nothing, for acting from fresh episodes."""
        head = self.width // self.heads
        zeros = jnp.zeros((rows, self.layers, size, self.heads, head), self.dtype)
        return Cache(zeros, zeros, jnp.zeros((rows, size), bool), jnp.zeros(rows, jnp.int32),
                     jnp.zeros(rows, jnp.int32))

    def step(self, x: jax.Array, first: jax.Array, cache: Cache) -> tuple[jax.Array, Cache]:
        """One step's state `[B, width]` from its input `x` `[B, width]`, attending to the cache and to
        itself; a step that starts an episode `first` attends to nothing cached. The step's keys and
        values are written at the cache's next slot, the oldest where it is full."""
        size = cache.valid.shape[1]
        first = jnp.asarray(first, bool)
        valid = cache.valid & ~first[:, None]
        position = jnp.where(first, 0, cache.position)
        slot = cache.index % size
        written = jax.nn.one_hot(slot, size, dtype=bool)
        keys, values = cache.keys, cache.values
        h = x[:, None]
        for layer, block in enumerate(self.blocks):
            query, key, value = block.projected(h, position[:, None])
            keys = keys.at[jnp.arange(x.shape[0]), layer, slot].set(key[:, 0])
            values = values.at[jnp.arange(x.shape[0]), layer, slot].set(value[:, 0])
            mask = (valid | written)[:, None, :]
            h = block.finished(h, query, keys[:, layer], values[:, layer], mask)
        return self.out_norm(h[:, 0]), Cache(keys, values, valid | written, cache.index + 1, position + 1)


def _segments(first: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Each step's episode within a window `[B, T]` (the window's first step starting one) and its
    position from that episode's first step in the window."""
    starts = jnp.asarray(first, bool).at[:, 0].set(True)
    segment = jnp.cumsum(starts, axis=1)
    index = jnp.arange(first.shape[1])
    began = jax.lax.cummax(jnp.where(starts, index, 0), axis=1)
    return segment, index - began


__all__ = ["TSSM", "Cache"]
