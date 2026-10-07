"""Models a user writes as plain Flax modules, Linen or NNX, for the capability matrix.

None of them is registered, derives from a Dew class or carries Dew's
metadata. Each has the capabilities `dew.nn.protocols` names by defining
them, as a user's own model would: `hidden_states` and `logits` for a token
model, `causal` to say which positions its states read, `mask_token_id` for
one trained by masked diffusion, `interval` for a denoiser of an interval.
An NNX model reaches Dew through Flax's own bridge (`flax.nnx.bridge.ToLinen`),
which runs only its `__call__` unless a Linen method names another
(`nnx_method`), so `NNXTokens` names the two a token model's readers call.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, nnx
from flax.nnx import bridge

from dew.diffusion.process import DenoisingCondition
from dew.inputs.encoders import CharTable, ConditionEncoder
from dew.nn.dit import masked_mean


class TokenModel(nn.Module):
    """One attention block over embedded tokens and a two-layer head.

    The head is not one matrix: a gelu sits between its two layers, so its
    logits come only from the model's own `logits`, never from a table.
    `causal` lets each position read itself and those before it, and False
    lets it read the whole row. It reads one document a row and refuses
    packed rows, as a user's model that never learned segment ids would.
    """

    vocab_size: int
    features: int = 16
    causal: bool = True
    mask_token_id: int | None = None

    def setup(self):
        self.embed = nn.Embed(self.vocab_size, self.features)
        self.attention = nn.SelfAttention(num_heads=2, qkv_features=self.features)
        self.norm = nn.LayerNorm()
        self.widen = nn.Dense(2 * self.features)
        self.out = nn.Dense(self.vocab_size)

    def hidden_states(self, tokens, *, train=False, segment_ids=None, positions=None, **fields):
        if segment_ids is not None or positions is not None:
            raise ValueError("this model reads one document a row; it takes no packed segment_ids "
                             "or positions")
        x = self.embed(tokens)
        mask = nn.make_causal_mask(tokens) if self.causal else None
        return self.norm(x + self.attention(x, mask=mask, deterministic=not train))

    def logits(self, tokens, *, train=False, **fields):
        return self.logits_from_hidden(self.hidden_states(tokens, train=train, **fields))

    def logits_from_hidden(self, hidden):
        return self.out(nn.gelu(self.widen(hidden))).astype(jnp.float32)

    def __call__(self, tokens, *, train=False, **fields):
        return self.logits(tokens, train=train, **fields)


def time_features(time: jax.Array, features: int) -> jax.Array:
    """Sinusoidal features of a `[B]` time, `[B, features]`."""
    half = features // 2
    frequencies = jnp.exp(-jnp.log(1e4) * jnp.arange(half) / half)
    angles = time.astype(jnp.float32)[:, None] * frequencies
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class Denoiser(nn.Module):
    """Two convolutions around the noise level, the mean of the prompt's
    states and, set `interval`, the duration of the interval it denoises."""

    features: int = 8
    interval: bool = False

    @nn.compact
    def __call__(self, sample, time, textcontext=None, *, duration=None, train=False):
        if duration is not None and not self.interval:
            raise ValueError("this denoiser reads no interval; set interval=True")
        level = time_features(time, self.features)
        if self.interval:
            level = level + nn.Dense(self.features)(
                time_features(jnp.zeros_like(time) if duration is None else duration, self.features))
        if textcontext is not None:
            level = level + nn.Dense(self.features)(masked_mean(textcontext.hidden, textcontext.mask))
        x = nn.Conv(self.features, (3, 3))(sample) + nn.Dense(self.features)(level)[:, None, None, :]
        return nn.Conv(sample.shape[-1], (3, 3))(nn.silu(x)).astype(sample.dtype)


@dataclass(frozen=True)
class FamilyText(ConditionEncoder[str, DenoisingCondition]):
    """A character table's states as a published family's `DenoisingCondition`:
    states `width` wide, a pooled vector `pooled` wide for the families that
    read one, the guidance scale a guidance-embedded family reads, and the
    token mask where the family reads it."""

    table: CharTable
    params: dict
    width: int
    pooled: int | None = None
    guidance: float | None = None
    masked: bool = False

    @classmethod
    def from_pretrained(cls, checkpoint: str = "char_table", *, params=None, width: int = 16,
                        pooled: int | None = None, guidance: float | None = None, masked: bool = False):
        table = CharTable.from_pretrained(features=width, params=None if params is None else params["table"])
        if params is None:
            projection = np.random.RandomState(1).normal(size=(width, pooled or 1)) / np.sqrt(width)
            params = {"table": table.params, "pooled": jnp.asarray(projection, jnp.float32)}
        return cls(table, params, width, pooled, guidance, masked)

    def tokenize(self, texts: Sequence[str]):
        return self.table.tokenize(texts)

    def encode(self, params, tokens) -> DenoisingCondition:
        text = self.table.encode(params["table"], tokens)
        pooled = None if self.pooled is None else masked_mean(text.hidden, text.mask) @ params["pooled"]
        guidance = None if self.guidance is None else jnp.full(text.hidden.shape[:1], self.guidance)
        return DenoisingCondition(text.hidden, pooled, guidance=guidance,
                                  mask=jnp.asarray(text.mask, bool) if self.masked else None)

    def captions(self, tokens) -> tuple[str, ...]:
        return self.table.captions(tokens)

    def to_json(self) -> dict:
        return {"checkpoint": "char_table", "width": self.width, "pooled": self.pooled,
                "guidance": self.guidance, "masked": self.masked}


class NNXLanguageModel(nnx.Module):
    """A causal token model written in NNX: an embedding, a running mean of
    its gated states, and a head."""

    def __init__(self, vocab_size: int, features: int = 16, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab_size, features, rngs=rngs)
        self.mix = nnx.Linear(features, features, rngs=rngs)
        self.head = nnx.Linear(features, vocab_size, rngs=rngs)

    def hidden_states(self, tokens, *, train=False, segment_ids=None, positions=None, **fields):
        if segment_ids is not None or positions is not None:
            raise ValueError("this model reads one document a row; it takes no packed segment_ids "
                             "or positions")
        x = jax.nn.gelu(self.mix(self.embed(tokens)))
        return jnp.cumsum(x, axis=1) / jnp.arange(1, tokens.shape[1] + 1)[None, :, None]

    def logits(self, tokens, *, train=False, **fields):
        return self.head(self.hidden_states(tokens, train=train, **fields)).astype(jnp.float32)

    def __call__(self, tokens, *, train=False, **fields):
        return self.logits(tokens, train=train)


class NNXTokens(bridge.ToLinen):
    """An NNX token model as Linen, naming the methods its readers call; its
    states read only the past, and it names no mask token."""

    causal: bool = True
    mask_token_id: int | None = None

    def hidden_states(self, tokens, *, train=False, **fields):
        return self(tokens, nnx_method="hidden_states", train=train, **fields)

    def logits(self, tokens, *, train=False, **fields):
        return self(tokens, nnx_method="logits", train=train, **fields)


class NNXDenoiser(nnx.Module):
    """A denoiser written in NNX: a convolution around the noise level and the prompt's mean state."""

    def __init__(self, channels: int, features: int = 8, *, rngs: nnx.Rngs):
        self.features = features
        self.inner = nnx.Conv(channels, features, (3, 3), rngs=rngs)
        self.level = nnx.Linear(features, features, rngs=rngs)
        self.text = nnx.Linear(16, features, rngs=rngs)
        self.outer = nnx.Conv(features, channels, (3, 3), rngs=rngs)

    def __call__(self, sample, time, textcontext=None, *, train=False):
        level = time_features(time, self.features)
        if textcontext is not None:
            level = level + self.text(masked_mean(textcontext.hidden, textcontext.mask))
        x = self.inner(sample) + self.level(level)[:, None, None, :]
        return self.outer(nnx.silu(x)).astype(sample.dtype)
