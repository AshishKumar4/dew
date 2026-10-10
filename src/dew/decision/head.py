"""Decision heads: a score per option, read off a backbone's final states.

Every head takes the backbone's `[B, L, D]` final states and the laid-out
`DecisionInputs`, and returns `[B, Q, K]` fp32 option logits, unmasked: the
caller drops the slots past each question's options. `DecisionHead` is
Laya's, which reads one marker token per option; `JointSchemaHead` is
Clef's, which reads every question of a row together and pools the tokens
of each option. A head that reads the backbone's output table says so with
`reads_table`, and is given it as `[V, D]` rows.
"""

import dataclasses
import functools
import math
from collections.abc import Mapping
from typing import ClassVar, Self

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew import records
from dew.decision.layout import DecisionInputs
from dew.decision.questions import KINDS, Choice, Noul, Score
from dew.nn.activations import gelu_exact_torch
from dew.nn.attention import NormalAttention, l2_normalized
from dew.nn.backbones.decoder_block import BlockWiring, DecoderBlock, GatedMLP, decoder_norm
from dew.nn.inputs import AttentionMetadata
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers import AttentionMixer
from dew.records import JSON


def _layer_norm(dtype: Dtype | None, name: str) -> nn.Module:
    """torch's `nn.LayerNorm`: a scale and a bias at 1e-5."""
    return decoder_norm("layer", epsilon=1e-5, bias=True, scale_offset=False, scale_after_cast=False,
                        dtype=dtype)(name=name)


class Head(nn.Module):
    """What every decision head offers: its call, whether it reads the output
    table, and the record a run keeps of it.

    A run records a head's own fields but not its width, which is the
    backbone's, nor its numerics, which the run's model config gives.
    """

    reads_table: ClassVar[bool] = False
    width_field: ClassVar[str] = "features"
    """The field holding the width of the states the head reads."""

    def record(self) -> JSON:
        """Return the head's name and the fields a run records."""
        kept: dict[str, JSON] = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)
                                 if field.name not in (*_NUMERICS, self.width_field)}
        return {"name": type(self).__name__, "fields": kept}

    @staticmethod
    def from_record(record: JSON, width: int, *, dtype: Dtype | None = None,
                    attention_impl: str = "auto") -> "Head":
        """Rebuild the head a run recorded, reading states `width` wide."""
        held = records.record(record, "head")
        name = records.text(held.get("name", "DecisionHead"), "head.name")
        if name not in HEADS:
            raise ValueError(f"a run's head is one of {sorted(HEADS)}, not {name!r}")
        fields = records.record(held.get("fields", held), "head.fields")
        return HEADS[name].rebuilt(width, fields, dtype=dtype, attention_impl=attention_impl)

    @classmethod
    def rebuilt(cls, width: int, fields: Mapping[str, object], *, dtype: Dtype | None,
                attention_impl: str) -> Self:
        """Return the head of these recorded `fields`, reading states `width` wide."""
        raise NotImplementedError


_NUMERICS = ("parent", "name", "dtype", "precision", "attention_impl")


class DecisionHead(Head):
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

    It reads one question per row (`QuestionLayout`), the first token of each
    option slot being its marker.
    """

    features: int
    layers: int = 2
    dropout_rate: float = 0.0
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"

    @classmethod
    def rebuilt(cls, width: int, fields: Mapping[str, object], *, dtype: Dtype | None,
                attention_impl: str) -> Self:
        return cls(features=width, layers=records.integer(fields.get("layers", 2), "head.layers"),
                   dropout_rate=records.number(fields.get("dropout_rate", 0.0), "head.dropout_rate"),
                   dtype=dtype, attention_impl=attention_impl)

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
        self.scorer_norm = _layer_norm(self.dtype, "scorer_norm")
        self.scorer_hidden = nn.Dense(self.features, dtype=self.dtype, precision=self.precision,
                                      name="scorer_hidden")
        self.scorer_out = nn.Dense(1, dtype=self.dtype, precision=self.precision, name="scorer_out")

    def __call__(self, states: jax.Array, inputs: DecisionInputs, table: jax.Array | None = None,
                 train: bool = False) -> jax.Array:
        if inputs.kinds.shape[1] != 1:
            raise ValueError("Laya's head reads one question per row; lay the rows out with a QuestionLayout")
        x = states + self.type_embedding(inputs.kinds[:, 0])[:, None, :].astype(states.dtype)
        metadata = AttentionMetadata(valid=inputs.valid)
        for block in self.blocks:
            x = block(x, train=train, attention_metadata=metadata)
            if not isinstance(x, jax.Array):
                raise TypeError("the head's blocks carry one plain residual stream")
        marked = jnp.take_along_axis(x, inputs.markers[:, 0, :, None], axis=1)
        hidden = gelu_exact_torch(self.scorer_hidden(self.scorer_norm(marked)))
        return self.scorer_out(hidden)[:, None, :, 0].astype(jnp.float32)


# Clef numbers its type embedding noul 0, choice 1, score 2
# (joint_schema_model.py QUESTION_TYPES), and the rows number them as KINDS does.
_CLEF_ROWS = tuple({Noul: 0, Choice: 1, Score: 2}[kind] for kind in KINDS)


class _RoutingLayer(nn.Module):
    """Clef's `EvidenceRoutingLayer`: each option attends to the row's memory, then a feed-forward."""

    width: int
    heads: int
    feedforward: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"

    @nn.compact
    def __call__(self, queries: jax.Array, memory: jax.Array, tokens: jax.Array) -> jax.Array:
        attention = NormalAttention(self.width, heads=self.heads, dim_head=self.width // self.heads,
                                    dtype=self.dtype, precision=self.precision,
                                    attention_impl=self.attention_impl, name="attention")
        memory = _layer_norm(self.dtype, "memory_norm")(memory)
        normed = _layer_norm(self.dtype, "query_norm")(queries)
        queries = queries + attention(normed, memory, key_value_seq_lengths=tokens)
        mlp = GatedMLP(hidden_features=self.feedforward, out_features=self.width, activation="gelu_exact",
                       use_bias=True, dtype=self.dtype, precision=self.precision, name="feedforward")
        return queries + mlp(_layer_norm(self.dtype, "feedforward_norm")(queries))


class _FieldLayer(nn.Module):
    """`nn.TransformerDecoderLayer(norm_first=True, activation="gelu")`: the fields
    attend to each other, then to the row's memory, then a feed-forward."""

    width: int
    heads: int
    feedforward: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"

    @nn.compact
    def __call__(self, fields: jax.Array, memory: jax.Array, questions: jax.Array,
                 tokens: jax.Array) -> jax.Array:
        def attention(name: str) -> NormalAttention:
            return NormalAttention(self.width, heads=self.heads, dim_head=self.width // self.heads,
                                   dtype=self.dtype, precision=self.precision,
                                   attention_impl=self.attention_impl, name=name)

        normed = _layer_norm(self.dtype, "norm1")(fields)
        fields = fields + attention("self_attn")(normed, key_value_seq_lengths=questions)
        normed = _layer_norm(self.dtype, "norm2")(fields)
        fields = fields + attention("multihead_attn")(normed, memory, key_value_seq_lengths=tokens)
        mlp = GatedMLP(hidden_features=self.feedforward, out_features=self.width, activation="gelu_exact",
                       use_bias=True, dtype=self.dtype, precision=self.precision, name="mlp")
        return fields + mlp(_layer_norm(self.dtype, "norm3")(fields))


def _span_means(values: jax.Array, spans: jax.Array) -> jax.Array:
    """The mean of `values` `[B, L, D]` over each `[..., 2]` (start, end) span of its row.

    `spans` is `[B, ...]`; an empty span, a padding slot's, gives zeros.
    """
    positions = jnp.arange(values.shape[1])
    inside = (positions >= spans[..., :1]) & (positions < spans[..., 1:])
    counts = jnp.maximum(jnp.sum(inside, axis=-1, keepdims=True), 1)
    flat = inside.reshape(inside.shape[0], -1, inside.shape[-1]).astype(values.dtype)
    summed = jnp.einsum("bnl,bld->bnd", flat, values).reshape(*inside.shape[:-1], values.shape[-1])
    return summed / counts.astype(values.dtype)


class JointSchemaHead(Head):
    """Clef's joint schema head: decides every question of a row together.

    This is `JointSchemaHead` of joint_schema_model.py at Cloudflare/clef
    2f3de3dd. Over the backbone's normed states it pools each question's
    instructions and each option's tokens, and each option's output-table rows
    (the lexical vector). The options attend to the row in `routing_layers`
    routing layers; each question becomes a field from its own pooled tokens,
    an attention-weighted summary of its options, the row's last token and its
    type; the fields attend to each other and to the row in `layers` decoder
    layers. An option's logit is a lexical prior (the cosine of its lexical
    vector with its question and the last token) plus a gated joint score (the
    cosine of field and option, and a residual MLP over the pair). The three
    scales are learned in log space and held to 100 at most.

    It reads every question of a row (`JointLayout`), and the backbone's output
    table.
    """

    hidden_size: int
    width: int = 1024
    routing_layers: int = 2
    layers: int = 4
    heads: int = 16
    feedforward: int = 4096
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"
    reads_table: ClassVar[bool] = True
    width_field: ClassVar[str] = "hidden_size"

    @classmethod
    def rebuilt(cls, width: int, fields: Mapping[str, object], *, dtype: Dtype | None,
                attention_impl: str) -> Self:
        def size(name: str) -> int:
            return records.integer(fields[name], f"head.{name}")

        return cls(hidden_size=width, width=size("width"), routing_layers=size("routing_layers"),
                   layers=size("layers"), heads=size("heads"), feedforward=size("feedforward"), dtype=dtype,
                   attention_impl=attention_impl)

    def setup(self):
        def dense(features: int, name: str) -> nn.Dense:
            return nn.Dense(features, use_bias=False, dtype=self.dtype, precision=self.precision, name=name)

        self.hidden_norm = _layer_norm(self.dtype, "hidden_norm")
        self.memory_projection = dense(self.width, "memory_projection")
        self.question_projection = dense(self.width, "question_projection")
        self.option_question_projection = dense(self.width, "option_question_projection")
        self.global_projection = dense(self.width, "global_projection")
        self.option_context_projection = dense(self.width, "option_context_projection")
        self.option_lexical_projection = dense(self.width, "option_lexical_projection")
        self.type_embedding = nn.Embed(len(KINDS), self.width, dtype=self.dtype, name="type_embedding")
        options = {"width": self.width, "heads": self.heads, "feedforward": self.feedforward,
                   "dtype": self.dtype, "precision": self.precision, "attention_impl": self.attention_impl}
        self.evidence_layers = [_RoutingLayer(**options, name=f"evidence_layers_{index}")
                                for index in range(self.routing_layers)]
        self.option_summary_norm = _layer_norm(self.dtype, "option_summary_norm")
        self.field_layers = [_FieldLayer(**options, name=f"layers_{index}") for index in range(self.layers)]
        self.field_norm = _layer_norm(self.dtype, "field_norm")
        self.option_norm = _layer_norm(self.dtype, "option_norm")
        self.scorer_hidden = nn.Dense(self.width, dtype=self.dtype, precision=self.precision,
                                      name="scorer_hidden")
        self.scorer_out = nn.Dense(1, dtype=self.dtype, precision=self.precision, name="scorer_out")
        self.prior_logit_scale = self.param("prior_logit_scale", nn.initializers.zeros_init(), ())
        self.joint_logit_scale = self.param("joint_logit_scale", nn.initializers.zeros_init(), ())
        self.residual_gate = self.param("residual_gate", nn.initializers.zeros_init(), ())

    def __call__(self, states: jax.Array, inputs: DecisionInputs, table: jax.Array | None = None,
                 train: bool = False) -> jax.Array:
        if table is None:
            raise ValueError("Clef's head reads the backbone's output table, its lexical prior")
        hidden = self.hidden_norm(states)
        batch = inputs.valid.shape[0]
        memory = self.memory_projection(hidden)
        last = jnp.maximum(jnp.sum(inputs.valid, axis=1) - 1, 0)
        last_state = hidden[jnp.arange(batch), last]
        questions = _span_means(hidden, inputs.spans)
        contexts = _span_means(hidden, inputs.option_spans)
        rows_of_tokens = jnp.take(table, inputs.tokens, axis=0).astype(hidden.dtype)
        lexical = _span_means(rows_of_tokens, inputs.option_spans)
        count, width = inputs.options.shape[1:]

        queries = (self.option_context_projection(contexts) + self.option_lexical_projection(lexical)
                   + self.option_question_projection(questions)[:, :, None])
        # A row's tokens and questions are right-padded, so each attention is told where its keys end
        # (at least one, for a padding row) rather than given a dense mask, which a fused kernel reads
        # as a bias per head, [B, H, Q*K, T]: 0.24 GiB of the step's temporaries at B=16 with 2,800
        # option queries over 3,072 tokens (c47).
        tokens = jnp.maximum(jnp.sum(inputs.valid, axis=1, dtype=jnp.int32), 1)
        asked = jnp.maximum(jnp.sum(inputs.questions, axis=1, dtype=jnp.int32), 1)
        routed = queries.reshape(batch, count * width, self.width)
        for layer in self.evidence_layers:
            routed = layer(routed, memory, tokens)
        routed = routed.reshape(batch, count, width, self.width)

        fields = self.question_projection(questions)
        affinity = jnp.einsum("bqkw,bqw->bqk", routed, fields) / math.sqrt(self.width)
        weights = jax.nn.softmax(jnp.where(inputs.options, affinity, -jnp.inf), axis=-1)
        summaries = jnp.einsum("bqk,bqkw->bqw", jnp.where(inputs.options, weights, 0.0), routed)
        rows = jnp.asarray(_CLEF_ROWS)[inputs.kinds]
        fields = (fields + self.option_summary_norm(summaries) + self.global_projection(last_state)[:, None]
                  + self.type_embedding(rows))
        for layer in self.field_layers:
            fields = layer(fields, memory, asked, tokens)
        fields = self.field_norm(fields)

        anchor = l2_normalized(questions + last_state[:, None], 1e-12)
        prior = (jnp.exp(jnp.minimum(self.prior_logit_scale, math.log(100.0)))
                 * jnp.einsum("bqkd,bqd->bqk", l2_normalized(lexical, 1e-12), anchor))
        options = self.option_norm(routed)
        repeated = jnp.broadcast_to(fields[:, :, None], options.shape)
        cosine = jnp.sum(l2_normalized(repeated, 1e-8) * l2_normalized(options, 1e-8), axis=-1)
        features = jnp.concatenate([repeated, options, repeated * options, jnp.abs(repeated - options)], -1)
        residual = self.scorer_out(gelu_exact_torch(self.scorer_hidden(features)))[..., 0]
        joint = jnp.exp(jnp.minimum(self.joint_logit_scale, math.log(100.0))) * cosine + residual
        return (prior + jax.nn.sigmoid(self.residual_gate) * joint).astype(jnp.float32)


HEADS: Mapping[str, type[Head]] = {head.__name__: head for head in (DecisionHead, JointSchemaHead)}
"""The heads a run may record, by name."""
