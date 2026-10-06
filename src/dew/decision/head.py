"""The decision head: a score per option, read off a backbone's final states."""

import functools

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.decision.questions import Choice, Noul, Question, Score
from dew.nn.activations import gelu_exact_torch
from dew.nn.backbones.decoder_block import BlockWiring, DecoderBlock, GatedMLP, decoder_norm
from dew.nn.inputs import AttentionMetadata
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers import AttentionMixer

KINDS: tuple[type[Question], ...] = (Choice, Score, Noul)
"""The question types in the order the head's type embedding holds them,
Laya's (laya/common.py QTYPES: choice 0, score 1, noul 2)."""


def kind_of(question: Question) -> int:
    """Return the row of the head's type embedding that `question`'s type uses."""
    return KINDS.index(type(question))


class DecisionHead(nn.Module):
    """Scores each option of a question at its marker in the backbone's states.

    This is Laya's head (laya/common.py `DecisionModel` at NandhaKishorM/laya
    a4a8921). The question type's embedding is added to every state, then
    `layers` pre-norm transformer blocks run over the row, as
    `nn.TransformerEncoderLayer(norm_first=True)` builds them: LayerNorm with
    bias at 1e-5, biased attention with `features // 64` heads and no position
    encoding, a biased ReLU feed-forward four times as wide, and padding keys
    masked. Then a LayerNorm, Dense, erf GELU, Dense scorer reads the state at
    each marker. The blocks are Dew's `DecoderBlock`. Laya's action head is left
    out, because its own model card reports that it carries no signal.

    `__call__` takes the backbone's `[B, L, features]` states, `valid` `[B, L]`,
    `markers` `[B, K]` and `kinds` `[B]`, and returns the `[B, K]` fp32 option
    logits, unmasked; the caller drops the slots past each row's options.
    """

    features: int
    layers: int = 2
    dropout_rate: float = 0.0
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"

    def setup(self):
        heads = max(1, self.features // 64)
        # The head reads the whole row in one call and never caches keys,
        # so the decode cache's capacity is never allocated.
        context = MixerContext(
            emb_features=self.features, num_heads=heads, num_kv_heads=heads,
            head_dim=self.features // heads, max_seq_len=1, causal=False, qk_norm=False,
            attention_bias=True, attention_dropout_rate=self.dropout_rate, dtype=self.dtype,
            precision=self.precision, attention_impl=self.attention_impl)
        feedforward = functools.partial(
            GatedMLP, hidden_features=4 * self.features, out_features=self.features, activation="relu",
            use_bias=True, dtype=self.dtype, precision=self.precision)
        self.type_embedding = nn.Embed(len(KINDS), self.features, dtype=self.dtype, name="type_embedding")
        self.blocks = [DecoderBlock(
            mixer=AttentionMixer(nope=True).build(context), feedforward=feedforward,
            emb_features=self.features, wiring=BlockWiring(), norm_type="layer", norm_bias=True,
            norm_eps=1e-5, dropout_rate=self.dropout_rate, dtype=self.dtype, precision=self.precision,
            name=f"layers_{index}") for index in range(self.layers)]
        norm = decoder_norm("layer", epsilon=1e-5, bias=True, scale_offset=False, scale_after_cast=False,
                            dtype=self.dtype)
        self.scorer_norm = norm(name="scorer_norm")
        self.scorer_hidden = nn.Dense(self.features, dtype=self.dtype, precision=self.precision,
                                      name="scorer_hidden")
        self.scorer_out = nn.Dense(1, dtype=self.dtype, precision=self.precision, name="scorer_out")

    def __call__(self, states: jax.Array, valid: jax.Array, markers: jax.Array, kinds: jax.Array,
                 train: bool = False) -> jax.Array:
        x = states + self.type_embedding(kinds)[:, None, :].astype(states.dtype)
        metadata = AttentionMetadata(valid=valid)
        for block in self.blocks:
            x = block(x, train=train, attention_metadata=metadata)
            if not isinstance(x, jax.Array):
                raise TypeError("the head's blocks carry one plain residual stream")
        marked = jnp.take_along_axis(x, markers[..., None], axis=1)
        hidden = gelu_exact_torch(self.scorer_hidden(self.scorer_norm(marked)))
        return self.scorer_out(hidden)[..., 0].astype(jnp.float32)
