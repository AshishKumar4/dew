"""DSpark: DeepSeek's block drafter for speculative decoding.

arXiv 2609.19969 section 2.4.3; the reference is the release's
inference/model.py `DSparkBlock`, `DSparkAttention`, `DSparkMarkovHead`,
`DSparkConfidenceHead` and `Transformer.forward_spec`: V4.1's (v41:1020-1156,
:1274-1282, at the revision tools/deepseek_v41_reference.py pins) and
V4-Flash-0731's (0731:743-874, :928-936, at the revision
tools/deepseek_v4_dspark_reference.py pins). The target records the stream
mean of each `target_layers` input (V4.1) or output (0731); the first
stage projects and norms their concatenation (`main_proj`, `main_norm`),
the context every stage's sliding attention reads beside its draft block,
the drawn token then `block_size - 1` noise tokens. Each stage is a decoder
block (the trunk's mHC, sliding attention whose block queries see the
whole block, a routed MoE with the drafter's expert count), chained as the
trunk's layers are: under V4.1's Single-Pass schedule from the first
stream, under V4's plain one from the copied embeddings. The last stage
collapses (by the carried pre, or through a learned head of its own under
V4's schedule), norms and scores with the trunk's head; the Markov head
adds a low-rank bigram bias from each drafted token to the next position's
logits, left to right, and the confidence head reads each collapsed state
beside that bigram embedding.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from .attention import RMSNorm
from .deepseek_v4 import DRAFT_CONTEXT, DRAFT_VALID
from .hyper_connections import Carried, HyperConnections, HyperHead, collapse_by, expand_streams, first_stream
from .precision import at_least_fp32
from .protocols import ProjectionGroup
from .sharding import logical_axes


@dataclasses.dataclass(frozen=True)
class DSpark:
    """The drafter, by the release's config names: `stages` (n_mtp_layers,
    one trailing compress ratio each) blocks drafting `block_size` tokens
    past the one drawn, `noise_token_id` filling the block, `target_layers` the trunk
    layers whose streams form the context, a Markov head of `markov_rank`,
    and `experts` routed experts with `top_k` per token. `layer_type` is the
    layer kind whose V4 attention every stage attends with, a sliding one
    in the release (v41:1034). `reads` is where a target layer's streams
    are averaged: its attention's input, after its engram (V4.1,
    v41:1264-1266), or the layer's output (V4-Flash-0731, 0731:918-921)."""

    stages: int
    block_size: int
    noise_token_id: int
    target_layers: tuple[int, ...]
    markov_rank: int
    experts: int
    top_k: int
    layer_type: str
    reads: Literal['input', 'output'] = 'input'

    def __post_init__(self):
        object.__setattr__(self, "target_layers", tuple(int(layer) for layer in self.target_layers))
        if self.stages < 1 or self.block_size < 1 or not self.target_layers:
            raise ValueError("DSpark drafts a block of one token or more through one stage or "
                             "more from one target layer or more")
        if list(self.target_layers) != sorted(set(self.target_layers)):
            raise ValueError("DSpark's target layers are distinct and in order, as the context "
                             "concatenates them")
        if self.reads not in ('input', 'output'):
            raise ValueError(f"DSpark reads a target layer's streams at its input or its output, "
                             f"got {self.reads!r}")


@logical_axes({
    ("main_proj",): (None, "embed"),
    ("confidence",): (None, None),
})
class DSparkStage(nn.Module):
    """One drafter stage: its decoder block, the context projection on the
    first, and the norm, Markov head and confidence head on the last.
    `weighted_head` is the trunk's streams when the last stage collapses
    them through a learned head of its own (`hc_head`, 0731:838-841, :862),
    None when it collapses by the pre Single-Pass carries (v41:1144)."""

    block: Callable[..., nn.Module]
    emb_features: int
    vocab_size: int
    markov_rank: int
    first: bool
    last: bool
    norm_eps: float
    weighted_head: HyperConnections | None = None
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
            self.confidence = nn.Dense(1, use_bias=False, dtype=at_least_fp32(self.dtype),
                                       precision=self.precision, name='confidence')
            if self.weighted_head is not None:
                self.hc_head = HyperHead(spec=self.weighted_head, emb_features=self.emb_features,
                                         norm_eps=self.norm_eps, name='hc_head')

    def projection_groups(self) -> tuple[ProjectionGroup, ...]:
        """The packed groups its block declares (`ProjectionSites`)."""
        return self.layer.projection_groups()

    def context(self, hidden):
        """The drafter's context off the concatenated target states (v41:1130)."""
        return self.main_norm(self.main_proj(hidden))

    def __call__(self, carry, store, decode: bool):
        return self.layer(carry, decode=decode, kv_store=store)

    def seed(self, store):
        """Append the context to this stage's window cache, drafting nothing
        (the release's prefill, v41:1044-1052, :1123-1125)."""
        context = store[DRAFT_CONTEXT]
        empty = jnp.zeros((context.shape[0], 0, context.shape[-1]), context.dtype)
        self.layer.self_attn(empty, decode=True, kv_store=store)

    def head(self, carry):
        """The collapsed pre-norm state and the normed one the head scores:
        `carry` is `Carried` under Single-Pass, else the streams."""
        hidden = collapse_by(carry.pre, carry.streams) if isinstance(carry, Carried) else self.hc_head(carry)
        return hidden, self.norm(hidden)

    def markov(self, tokens):
        """The bigram bias `[B, vocab]` a drafted token adds to the next
        position, and the token's embedding (v41:1077-1086)."""
        embedded = jnp.take(self.markov_embed, tokens, axis=0)
        return jnp.einsum('br,vr->bv', embedded, self.markov_head, precision=self.precision), embedded

    def confident(self, hidden, embedded):
        """Each position's acceptance score (v41:1089-1097)."""
        wide = at_least_fp32(hidden.dtype)
        return self.confidence(jnp.concatenate(
            [hidden.astype(wide), embedded.astype(wide)], -1))[..., 0]


def draft(stages, spec: DSpark, embed: Callable, logits_of: Callable, hc: HyperConnections,
          states, tokens, *, decode: bool, valid=None, choose: Callable | None = None):
    """One drafting pass (Transformer.forward_spec, v41:1274-1282,
    0731:928-936), with `CausalTransformer.draft`'s arguments and returns;
    `states` is the context `draft` reads, `hc` the trunk's streams, and the
    logits carry the Markov bias.
    """
    store = {}
    if states is not None:
        store[DRAFT_CONTEXT] = stages[0].context(states)
        if valid is not None:
            store[DRAFT_VALID] = valid
    if tokens is None:
        for stage in stages:
            stage.seed(store)
        return None
    block = jnp.full((tokens.shape[0], spec.block_size), spec.noise_token_id, jnp.int32)
    block = block.at[:, 0].set(tokens.astype(block.dtype))
    streams = expand_streams(embed(block), hc.hc_mult)
    carry = Carried(streams, first_stream(streams)) if hc.single_pass else streams
    for stage in stages:
        carry = stage(carry, store, decode)
    last = stages[-1]
    hidden_state, normed = last.head(carry)
    logits = logits_of(normed)
    drafted, embeds, scored = [tokens], [], []
    for index in range(spec.block_size):
        bias, embedded = last.markov(drafted[-1])
        step = logits[:, index] + bias
        chosen = jnp.argmax(step, axis=-1) if choose is None else choose(index, step)
        drafted.append(chosen.astype(jnp.int32))
        embeds.append(embedded)
        scored.append(step)
    confidence = last.confident(hidden_state, jnp.stack(embeds, 1))
    return jnp.stack(drafted, 1), jnp.stack(scored, 1), confidence
