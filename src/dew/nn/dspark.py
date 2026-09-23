"""DSpark: DeepSeek-V4.1's block drafter for speculative decoding.

arXiv 2609.19969 section 2.4.3; the reference is `DSparkBlock`,
`DSparkAttention`, `DSparkMarkovHead`, `DSparkConfidenceHead` and
`Transformer.forward_spec` of the release's inference/model.py (v41:1020-1156,
:1274-1282, at the revision tools/deepseek_v41_reference.py pins).

The target model records, at each of `target_layers`, the mean over its
residual streams of that layer's input; the drafter's first stage projects
their concatenation to the model width and norms it (`main_proj`,
`main_norm`). That context is what every stage's sliding attention reads
beside its draft block: the block is the token the target just drew followed
by `block_size - 1` noise tokens, at the positions after the context's last.
Each stage is a decoder block of its own (mHC streams under Single-Pass
mixing, one sliding attention whose block queries see the whole block, and
a routed MoE with the drafter's own expert count), chained as the trunk's
layers are. The last stage collapses the streams, norms them and scores
them with the trunk's head; the Markov head then adds a low-rank bigram
bias from each drafted token to the next position's logits as it drafts
left to right, and the confidence head reads each position's collapsed
state beside that bigram embedding.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from .attention import RMSNorm
from .deepseek_v4 import DRAFT_CONTEXT
from .hyper_connections import Carried, collapse_by, expand_streams, first_stream
from .sharding import logical_axes


@dataclasses.dataclass(frozen=True)
class DSpark:
    """The drafter, by the release's config names: `stages`
    (num_nextn_predict_layers) blocks drafting `block_size` tokens past the
    one drawn, `noise_token_id` filling the block, `target_layers` the trunk
    layers whose inputs form the context, a Markov head of `markov_rank`,
    and `experts` routed experts with `top_k` per token. `layer_type` is the
    layer kind whose V4 attention every stage attends with, a sliding one
    in the release (v41:1034)."""

    stages: int
    block_size: int
    noise_token_id: int
    target_layers: tuple[int, ...]
    markov_rank: int
    experts: int
    top_k: int
    layer_type: str

    def __post_init__(self):
        object.__setattr__(self, "target_layers", tuple(int(layer) for layer in self.target_layers))
        if self.stages < 1 or self.block_size < 1 or not self.target_layers:
            raise ValueError("DSpark drafts a block of one token or more through one stage or "
                             "more from one target layer or more")
        if list(self.target_layers) != sorted(set(self.target_layers)):
            raise ValueError("DSpark's target layers are distinct and in order, as the context "
                             "concatenates them")


@logical_axes({
    ("main_proj",): (None, "embed"),
    ("confidence",): (None, None),
}, heuristic=(("markov_embed",), ("markov_head",)))
class DSparkStage(nn.Module):
    """One drafter stage: its decoder block, the context projection on the
    first, and the norm, Markov head and confidence head on the last."""

    block: Callable[..., nn.Module]
    emb_features: int
    targets: int
    vocab_size: int
    markov_rank: int
    first: bool
    last: bool
    norm_eps: float
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        self.layer = self.block(name='block')
        if self.first:
            self.main_proj = nn.Dense(self.emb_features, use_bias=False, dtype=self.dtype,
                                      precision=self.precision, name='main_proj')
            self.main_norm = RMSNorm(epsilon=self.norm_eps, scale_after_cast=True,
                                     dtype=self.dtype, name='main_norm')
        if self.last:
            self.norm = RMSNorm(epsilon=self.norm_eps, scale_after_cast=True,
                                dtype=self.dtype, name='norm')
            self.markov_embed = self.param('markov_embed', nn.initializers.normal(1.0),
                                           (self.vocab_size, self.markov_rank), jnp.float32)
            self.markov_head = self.param('markov_head', nn.initializers.normal(self.markov_rank ** -0.5),
                                          (self.vocab_size, self.markov_rank), jnp.float32)
            self.confidence = nn.Dense(1, use_bias=False, dtype=jnp.float32,
                                       precision=self.precision, name='confidence')

    def context(self, hidden):
        """The drafter's context off the concatenated target states (v41:1130)."""
        return self.main_norm(self.main_proj(hidden))

    def __call__(self, carry, context, decode: bool):
        return self.layer(carry, decode=decode, kv_store={DRAFT_CONTEXT: context})

    def seed(self, context):
        """Append the context to this stage's window cache, drafting nothing
        (the release's prefill, v41:1044-1052, :1123-1125)."""
        empty = jnp.zeros((context.shape[0], 0, context.shape[-1]), context.dtype)
        self.layer.self_attn(empty, decode=True, kv_store={DRAFT_CONTEXT: context})

    def head(self, carry: Carried):
        """The collapsed pre-norm state and the normed one the head scores."""
        hidden = collapse_by(carry.pre, carry.streams)
        return hidden, self.norm(hidden)

    def markov(self, tokens):
        """The bigram bias `[B, vocab]` a drafted token adds to the next
        position, and the token's embedding (v41:1077-1086)."""
        embedded = jnp.take(self.markov_embed, tokens, axis=0)
        return jnp.einsum('br,vr->bv', embedded, self.markov_head, precision=self.precision), embedded

    def confident(self, hidden, embedded):
        """Each position's acceptance score (v41:1089-1097)."""
        return self.confidence(jnp.concatenate(
            [hidden.astype(jnp.float32), embedded.astype(jnp.float32)], -1))[..., 0]


def draft(stages, spec: DSpark, embed: Callable, logits_of: Callable, hc_mult: int,
          states, tokens, *, decode: bool, key=None, temperature: float = 0.0):
    """One drafting pass (Transformer.forward_spec, v41:1274-1282).

    `states` `[B, M, targets * D]` is the target's recorded context up to
    the position the block drafts after, `tokens` `[B]` the token drawn
    there. Returns the drafted ids `[B, block_size + 1]` (the drawn token
    first), their logits `[B, block_size, vocab]` with the Markov bias
    added, and each position's confidence `[B, block_size]`. Draws are
    greedy unless a `key` and a positive `temperature` are given. `tokens`
    None only seeds the cached windows.
    """
    context = stages[0].context(states)
    if tokens is None:
        for stage in stages:
            stage.seed(context)
        return None
    block = jnp.full((tokens.shape[0], spec.block_size), spec.noise_token_id, jnp.int32)
    block = block.at[:, 0].set(tokens)
    streams = expand_streams(embed(block), hc_mult)
    carry = Carried(streams, first_stream(streams))
    for stage in stages:
        carry = stage(carry, context, decode)
    last = stages[-1]
    hidden_state, normed = last.head(carry)
    logits = logits_of(normed)
    drafted, embeds, scored = [tokens], [], []
    for index in range(spec.block_size):
        bias, embedded = last.markov(drafted[-1])
        step = logits[:, index] + bias
        if key is None or temperature <= 0:
            chosen = jnp.argmax(step, axis=-1)
        else:
            key, sub = jax.random.split(key)
            chosen = jax.random.categorical(sub, step / temperature, axis=-1)
        drafted.append(chosen.astype(jnp.int32))
        embeds.append(embedded)
        scored.append(step)
    confidence = last.confident(hidden_state, jnp.stack(embeds, 1))
    return jnp.stack(drafted, 1), jnp.stack(scored, 1), confidence
