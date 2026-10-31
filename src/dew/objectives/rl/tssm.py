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
from flax.typing import Dtype

from dew.nn.attention import reference_attention
from dew.nn.precision import at_least_fp32, float32_einsum
from dew.nn.rope import rotary_freqs, rotate
from dew.rl.distributions import unimix_log_probs


@struct.dataclass
class Window:
    """A window's keys and values `[B, layers, T, heads, head]` that imagination from each of its `B * T`
    steps attends to, held once for all of them, and which of the window's steps each start reads `[B * T, T]`
    (those of its episode up to itself)."""

    keys: jax.Array
    values: jax.Array
    valid: jax.Array


@struct.dataclass
class Cache:
    """The keys and values a step attends to, `[B, layers, C, heads, head]` each, with whether each slot is
    `valid` `[B, C]`, the slot the next step writes (`index`, wrapping round the C slots) and that step's
    `position` within its episode `[B]`; for imagination, beside them the `window` its starts share."""

    keys: jax.Array
    values: jax.Array
    valid: jax.Array
    index: jax.Array
    position: jax.Array
    window: Window | None = None


class _Block(nn.Module):
    """A pre-norm transformer block: rotary, RMS-normed multi-head attention (its logits and softmax in at
    least float32, `reference_attention`), then a SiLU-gated feed-forward layer, each added to its
    input."""

    width: int
    heads: int
    ffup: int
    dtype: Dtype

    def setup(self) -> None:
        head = self.width // self.heads
        dense = functools.partial(nn.DenseGeneral, dtype=self.dtype, param_dtype=at_least_fp32(self.dtype))
        norm = functools.partial(nn.RMSNorm, epsilon=1e-6, dtype=self.dtype,
                                 param_dtype=at_least_fp32(self.dtype))
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
        cos, sin = rotary_freqs(positions, self.width // self.heads, 10000.0, dtype=at_least_fp32(self.dtype))
        query = rotate(self.query_norm(self.query(h)), cos, sin)
        key = rotate(self.key_norm(self.key(h)), cos, sin)
        return query, key, self.value(h)

    def finished(self, x: jax.Array, query: jax.Array, keys: jax.Array, values: jax.Array,
                 mask: jax.Array) -> jax.Array:
        """The block's output for rows `x` whose queries attend to `keys` and `values` under `mask`
        `[B, S, K]`."""
        return self.after(x, reference_attention(query, keys, values, mask=mask[:, None], wide_logits=True))

    def after(self, x: jax.Array, attended: jax.Array) -> jax.Array:
        """The block's output for rows `x` from what their queries `attended` `[B, S, heads, head]`: its
        projection added, then the feed-forward layer's."""
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
    context: int = 64
    """The steps an acting cache holds (`empty_cache`), the windows' length it trained on."""
    dtype: Dtype = jnp.float32
    """The computation's dtype, bfloat16 for speed; parameters stay at least float32 (`at_least_fp32`),
    so a bfloat16 computation updates float32 weights (DreamerV3's COMPUTE_DTYPE beside its float32
    parameters), and the latents' distributions are at least float32."""

    def setup(self) -> None:
        dense = functools.partial(nn.Dense, dtype=self.dtype, param_dtype=at_least_fp32(self.dtype))
        norm = functools.partial(nn.RMSNorm, epsilon=1e-6, dtype=self.dtype,
                                 param_dtype=at_least_fp32(self.dtype))
        self.embed = dense(self.width)
        self.start = self.param("start", nn.initializers.normal(0.02), (self.width,),
                                at_least_fp32(self.dtype))
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
        return unimix_log_probs(x.reshape(*x.shape[:-1], self.stoch, self.classes), self.unimix)

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
        """For each step k of a window `[B, T]`, the cache imagination from it attends to: the window's keys
        and values (`Window`, held once), of which k reads its episode's steps up to k, and `horizon` slots
        of its own for the steps it imagines, the next at the position after k's."""
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
        window = Window(jnp.stack(keys, 1), jnp.stack(values, 1), valid.reshape(rows * length, length))
        own = (rows * length, self.layers, horizon, self.heads, self.width // self.heads)
        return Cache(jnp.zeros(own, self.dtype), jnp.zeros(own, self.dtype),
                     jnp.zeros((rows * length, horizon), bool), jnp.zeros(rows * length, jnp.int32),
                     (positions + 1).reshape(-1), window)

    @nn.nowrap
    def empty_cache(self, rows: int, size: int) -> Cache:
        """A cache of `size` slots holding nothing, for acting from fresh episodes (`context` slots).
        Within an episode's first `size` steps acting reads what training's window does; past them the
        cache holds the last `size` steps' keys, each computed over the context before it."""
        shape = (rows, self.layers, size, self.heads, self.width // self.heads)
        # Keys and values each their own buffer: a rollout donates the cache it steps, each buffer once.
        return Cache(jnp.zeros(shape, self.dtype), jnp.zeros(shape, self.dtype),
                     jnp.zeros((rows, size), bool), jnp.zeros(rows, jnp.int32), jnp.zeros(rows, jnp.int32))

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
        # A start whose imagined episode restarts reads its window no more, at this step or after.
        window = cache.window
        if window is not None:
            window = Window(window.keys, window.values, window.valid & ~first[:, None])
        h = x[:, None]
        for layer, block in enumerate(self.blocks):
            query, key, value = block.projected(h, position[:, None])
            keys = keys.at[jnp.arange(x.shape[0]), layer, slot].set(key[:, 0])
            values = values.at[jnp.arange(x.shape[0]), layer, slot].set(value[:, 0])
            mask = (valid | written)[:, None, :]
            if window is None:
                h = block.finished(h, query, keys[:, layer], values[:, layer], mask)
            else:
                attended = _windowed(query, window, layer, keys[:, layer], values[:, layer], mask[:, 0])
                h = block.after(h, attended)
        following = Cache(keys, values, valid | written, cache.index + 1, position + 1, window)
        return self.out_norm(h[:, 0]), following


def _windowed(query: jax.Array, window: Window, layer: int, keys: jax.Array, values: jax.Array,
              mask: jax.Array) -> jax.Array:
    """`reference_attention` (wide logits) of each start's query `[B * T, 1, heads, head]` over the keys and
    values of `window`'s `layer` its start reads and its own `keys` and `values` `[B * T, C, heads, head]`
    under `mask` `[B * T, C]`, one softmax over both: the window's part one product for all of its starts,
    so no start holds a copy of its window."""
    windows, _, length, heads, head = window.keys.shape
    wide = at_least_fp32(query.dtype)
    asked = query[:, 0].astype(wide)
    shared = float32_einsum("bshd,bthd->bsht", asked.reshape(windows, length, heads, head),
                            window.keys[:, layer].astype(wide)).reshape(-1, heads, length)
    own = float32_einsum("rhd,rchd->rhc", asked, keys.astype(wide))
    logits = jnp.concatenate([shared, own], -1) * head ** -0.5
    seen = jnp.concatenate([window.valid, mask], -1)[:, None]
    weights = jax.nn.softmax(jnp.where(seen, logits, jnp.finfo(query.dtype).min)).astype(values.dtype)
    spread = weights[..., :length].reshape(windows, length, heads, length)
    from_window = float32_einsum("bsht,bthd->bshd", spread, window.values[:, layer],
                                 preferred_element_type=wide)
    from_own = float32_einsum("rhc,rchd->rhd", weights[..., length:], values, preferred_element_type=wide)
    return (from_window.reshape(-1, heads, head) + from_own).astype(values.dtype)[:, None]


def _segments(first: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Each step's episode within a window `[B, T]` (the window's first step starting one) and its
    position from that episode's first step in the window."""
    starts = jnp.asarray(first, bool).at[:, 0].set(True)
    segment = jnp.cumsum(starts, axis=1)
    index = jnp.arange(first.shape[1])
    began = jax.lax.cummax(jnp.where(starts, index, 0), axis=1)
    return segment, index - began


__all__ = ["TSSM", "Cache", "Window"]
