"""Decision heads: a score per option, read off a backbone's final states.

Every head takes the backbone's `[B, L, D]` final states and the laid-out
`DecisionInputs`, and returns `[B, Q, K]` fp32 option logits, unmasked: the
caller drops the slots past each question's options. `DecisionHead` is
Laya's, which reads one marker token per option; `JointSchemaHead` reads
every question of a row together and pools the tokens of each option. A head
that reads the backbone's output table says so with `reads_table`, and is
given it as `[V, D]` rows.
"""

import functools
import math
from collections.abc import Sequence
from typing import ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew import records
from dew.decision.layout import DecisionInputs, Encoded
from dew.decision.questions import KINDS, Choice, Noul, Score
from dew.nn.activations import gelu_exact_torch
from dew.nn.attention import NormalAttention, l2_normalized
from dew.nn.backbones.decoder_block import BlockWiring, DecoderBlock, decoder_norm
from dew.nn.blocks import TokenEmbedding
from dew.nn.dense import Dense
from dew.nn.flops import attention_flops, projection_flops, real_lengths
from dew.nn.inputs import AttentionMetadata
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers import AttentionMixer
from dew.nn.mlp import GatedMLP
from dew.nn.precision import accumulated, at_least_fp32, capped_logits, scaled_logits
from dew.nn.scatter import DROPPED
from dew.records import JSON
from dew.registry import Record, class_record, from_record, import_path, imported, record_fields


def _layer_norm(dtype: Dtype | None, name: str) -> nn.Module:
    """torch's `nn.LayerNorm`: a scale and a bias at 1e-5."""
    return decoder_norm("layer", epsilon=1e-5, bias=True, scale_offset=False, scale_after_cast=False,
                        dtype=dtype)(name=name)


class Head(nn.Module):
    """What every decision head offers: its call, whether it reads the output
    table, and the record a run keeps of it.

    A run records a head's fields, its numerics among them, but not its
    width, which is the backbone's.
    """

    reads_table: ClassVar[bool] = False
    reads_labels: ClassVar[bool] = False
    """Whether the head reads the answer state of each question at its row's answer cue
    (`DecisionInputs.readout`) against labels (`Labels`), the backbone's output table its `table`
    (`OutputTable`, None where no matrix alone is the backbone's head)."""
    draws_from_rows: ClassVar[bool] = False
    """Whether the head draws its parameters from the backbone's vocabulary rows (`VocabularyRows`),
    its `table` when it is drawn, and reads no table after."""
    one_question: ClassVar[bool] = False
    """Whether the head reads one question a row, refusing a joint row's several
    questions at the host boundary (`validate_rows`)."""
    width_field: ClassVar[str] = "features"
    """The field holding the width of the states the head reads."""

    @nn.nowrap
    def tied_labels(self) -> tuple[tuple[int, ...], ...] | None:
        """The label ids, `[K][V]`, at which the head reads the backbone's own logits (`tied`), or
        None where it reads none."""
        return None

    @nn.nowrap
    def validate_rows(self, rows: Sequence[Encoded]) -> None:
        """Refuse encoded host rows that ask more questions than this head reads."""
        if self.one_question and any(len(row.questions) > 1 for row in rows):
            raise ValueError("this head reads one question a row; lay the rows out with a "
                             "QuestionLayout or use a joint head")

    def record(self) -> Record:
        """Return the head as a run records it: its class by import path (`dew.registry`) and its
        fields but the width, which the backbone it reads gives, so any head reloads, one of a
        package the run trusts too."""
        fields = record_fields(self, type(self))
        del fields[self.width_field]
        return {"class": import_path(type(self)), "fields": fields}

    @staticmethod
    def from_record(record: JSON, width: int) -> "Head":
        """Rebuild the head a run recorded (`dew.registry.class_record`), reading states `width`
        wide, the one field its record leaves to the backbone, in the numerics it recorded."""
        held = class_record(records.record(record, "head"))
        if held is None:
            raise ValueError("a run's head is a class record, {\"class\": ..., \"fields\": {...}}")
        path, fields = held
        head = imported(path)
        if not (isinstance(head, type) and issubclass(head, Head)):
            raise ValueError(f"a run's head is a decision head (`Head`), not {path!r}")
        return from_record(head, {**records.record(fields, "head.fields"), head.width_field: width})


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
    option slot being its marker, and a window's rows apart from each other.
    """

    one_question: ClassVar[bool] = True

    features: int
    layers: int = 2
    dropout_rate: float = 0.0
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"

    @nn.nowrap
    def decision_flops(self, lengths, questions, options, *, pooled=None) -> float:
        """Full-row encoder layers and the scorer at real option markers; it pools no spans."""
        if pooled is not None:
            raise TypeError("Laya's head scores options at their markers in the row, and pools no context")
        decision_geometry(lengths, questions, options)
        per_layer = (12 * projection_flops(lengths, self.features, self.features)
                     + attention_flops(lengths, 1, self.features, self.features, causal=False))
        return (self.layers * per_layer + projection_flops(options, self.features, self.features)
                + projection_flops(options, self.features, 1))

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
        self.type_embedding = TokenEmbedding(len(KINDS), self.features, dtype=self.dtype,
                                             name="type_embedding")
        self.blocks = [DecoderBlock(
            mixer=AttentionMixer(nope=True).build(context), feedforward=feedforward,
            emb_features=self.features, wiring=BlockWiring(), norm_type="layer", norm_bias=True,
            norm_eps=1e-5, dropout_rate=self.dropout_rate, dtype=self.dtype, precision=self.precision,
            name=f"layers_{index}") for index in range(self.layers)]
        self.scorer_norm = _layer_norm(self.dtype, "scorer_norm")
        self.scorer_hidden = Dense(self.features, dtype=self.dtype, precision=self.precision,
                                   name="scorer_hidden")
        self.scorer_out = Dense(1, dtype=self.dtype, precision=self.precision, name="scorer_out")

    def __call__(self, states: jax.Array, inputs: DecisionInputs, table: jax.Array | None = None,
                 train: bool = False) -> jax.Array:
        if inputs.spans_in_context:
            raise TypeError("Laya's head scores each option at its marker in the row, so it reads no "
                            "schema read as the row's context; use a JointSchemaHead")
        # Every token takes its row's question type: a QuestionLayout row asks one question.
        owned = inputs.question_segments[:, None, :] == inputs.segments[:, :, None]
        kinds = jnp.sum(jnp.where(owned, inputs.kinds[:, None, :], 0), axis=-1)
        x = states + self.type_embedding(kinds).astype(states.dtype)
        metadata = AttentionMetadata(coordinates=inputs.coordinates())
        for block in self.blocks:
            x = block(x, train=train, attention_metadata=metadata)
            if not isinstance(x, jax.Array):
                raise TypeError("the head's blocks carry one plain residual stream")
        batch, count, width = inputs.markers.shape
        marked = _gathered(x, inputs.markers.reshape(batch, count * width))
        hidden = gelu_exact_torch(self.scorer_hidden(self.scorer_norm(marked)))
        scores = self.scorer_out(hidden)[..., 0].reshape(batch, count, width)
        return scores.astype(at_least_fp32(scores.dtype))


# The type embedding's rows: noul 0, choice 1, score 2, read through KINDS' order,
# which a saved joint head's table keeps.
_TYPE_ROWS = tuple({Noul: 0, Choice: 1, Score: 2}[kind] for kind in KINDS)


class _Reading(NamedTuple):
    """Which of the window's tokens each attention query of the joint head reads: those of its own
    row, by documents (`dew.nn.documents.CrossDocuments`). `tokens` `[B, L]` are the window's
    rows, `questions` `[B, Q]` each question's and `slots` `[B, P]` each option slot's, 0 for
    padding, each in window order with its padding after, as packed sequences need them."""

    tokens: jax.Array
    questions: jax.Array
    slots: jax.Array
    max_segments: int


class _RoutingLayer(nn.Module):
    """An evidence routing layer: each option attends to the row's memory, then a feed-forward."""

    width: int
    heads: int
    feedforward: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"

    @nn.compact
    def __call__(self, queries: jax.Array, memory: jax.Array, read: _Reading) -> jax.Array:
        attention = NormalAttention(self.width, heads=self.heads, dim_head=self.width // self.heads,
                                    dtype=self.dtype, precision=self.precision,
                                    attention_impl=self.attention_impl, name="attention")
        memory = _layer_norm(self.dtype, "memory_norm")(memory)
        normed = _layer_norm(self.dtype, "query_norm")(queries)
        queries = queries + attention(normed, memory, segment_ids=read.slots, key_segment_ids=read.tokens,
                                      max_segments=read.max_segments)
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
    def __call__(self, fields: jax.Array, memory: jax.Array, read: _Reading) -> jax.Array:
        def attention(name: str) -> NormalAttention:
            return NormalAttention(self.width, heads=self.heads, dim_head=self.width // self.heads,
                                   dtype=self.dtype, precision=self.precision,
                                   attention_impl=self.attention_impl, name=name)

        normed = _layer_norm(self.dtype, "norm1")(fields)
        fields = fields + attention("self_attn")(normed, segment_ids=read.questions,
                                                 max_segments=read.max_segments)
        normed = _layer_norm(self.dtype, "norm2")(fields)
        fields = fields + attention("multihead_attn")(
            normed, memory, segment_ids=read.questions, key_segment_ids=read.tokens,
            max_segments=read.max_segments)
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
    summed = accumulated(functools.partial(jnp.einsum, "bnl,bld->bnd"), flat, values)
    summed = summed.reshape(*inside.shape[:-1], values.shape[-1])
    return summed / counts.astype(values.dtype)


def _gathered(values: jax.Array, indices: jax.Array) -> jax.Array:
    """`values` `[B, N, ...]` at each row's `indices` `[B, P]`: `[B, P, ...]`."""
    return jnp.take_along_axis(values, indices.reshape(*indices.shape, *(1,) * (values.ndim - 2)), axis=1)


def _per_slot(values: jax.Array, owners: jax.Array, inputs: DecisionInputs) -> jax.Array:
    """Each question's `values` `[B, Q, F]` at its option slots, `[B, P, F]`: on the grid a question's
    `width` slots repeat it, a broadcast whose transpose sums them, and compacted slots gather by their
    `owners`. A gather's transpose is a scatter, which XLA's deterministic GPU scatter sorts: the
    joint head's three [8, 75, 1024] reads at [8, 7500, 1024] slots took 3.99 ms forward and backward
    on an A100 40GB as gathers, 0.18 as broadcasts."""
    if inputs.option_slots is not None:
        return _gathered(values, owners)
    batch, count, width = inputs.options.shape
    return jnp.broadcast_to(values[:, :, None], (batch, count, width, *values.shape[2:])).reshape(
        batch, count * width, *values.shape[2:])


def _laid_back(values: jax.Array, inputs: DecisionInputs) -> jax.Array:
    """Per-slot `values` `[B, P, ...]` back on the `[B, Q, K, ...]` grid, zero
    where no real option is: the grid itself where the inputs compact nothing."""
    batch, count, width = inputs.options.shape
    if inputs.option_slots is None:
        grid = values.reshape(batch, count, width, *values.shape[2:])
        return jnp.where(inputs.options.reshape(*inputs.options.shape, *(1,) * (grid.ndim - 3)), grid, 0)
    assert inputs.slotted is not None
    # A padding slot indexes slot 0 (`compacted`) and so computes slot 0's own
    # values, and the scatter's gradient masks a duplicate index; dropping it
    # still keeps a padding slot from writing anywhere on the grid, a row of
    # no real options' slot 0 included. `DROPPED`, not the grid's size, which
    # XLA's deterministic GPU scatter would land on the next row's slot 0.
    target = jnp.where(inputs.slotted, inputs.option_slots, DROPPED)
    laid = jnp.zeros((batch, count * width, *values.shape[2:]), values.dtype)
    laid = laid.at[jnp.arange(batch)[:, None], target].set(values, mode="drop")
    return laid.reshape(batch, count, width, *values.shape[2:])


class JointSchemaHead(Head):
    """The joint schema head: decides every question of a row together.

    Over the backbone's normed states it pools each question's
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
    table. In a window of packed rows each question and option reads its own
    row alone, and the row's last token is its own. Each option is computed on
    its own but for its weight in its question's summary, so where the inputs
    compact each window's real options (`DecisionInputs.option_slots`) it
    computes those alone, and lays their
    routed states (for the summaries) and their logits back on the grid.
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

    @nn.nowrap
    def decision_flops(self, lengths, questions, options, *, pooled=None) -> float:
        """Memory/option projections, evidence O-by-L and field Q-by-L attention, and the span
        pooling over each row, or over `pooled` tokens a row where the spans are a context's.

        Each attention has query/output maps on its queries and key/value
        maps on memory. The field layers also have Q-by-Q self-attention.
        The residual scorer reads four width-sized features per option.
        Span pooling is a masked matrix product and is counted. Lookup
        embeddings and pointwise normalization/gates are excluded.
        """
        length, question, option = decision_geometry(lengths, questions, options)
        width, hidden = self.width, self.hidden_size
        projected = np.sum(length + 2 * question + 2 * option) + np.count_nonzero(length)
        count = 2 * hidden * width * projected
        evidence = (4 * width * width * np.sum(length + option)
                    + 4 * width * np.sum(length * option)
                    + 4 * width * self.feedforward * np.sum(option))
        fields = (12 * width * width * np.sum(question) + 4 * width * width * np.sum(length)
                  + 4 * width * np.sum(question * question + length * question)
                  + 4 * width * self.feedforward * np.sum(question))
        scorer = (8 * width * width + 2 * width) * np.sum(option)
        summaries = (6 * width + 2 * hidden) * np.sum(option)
        spanned = length if pooled is None else real_lengths(pooled)
        pooling = 2 * hidden * np.sum(spanned * (question + 2 * option))
        return float(count + pooling + self.routing_layers * evidence
                     + self.layers * fields + scorer + summaries)

    def setup(self):
        def dense(features: int, name: str) -> Dense:
            return Dense(features, use_bias=False, dtype=self.dtype, precision=self.precision, name=name)

        self.hidden_norm = _layer_norm(self.dtype, "hidden_norm")
        self.memory_projection = dense(self.width, "memory_projection")
        self.question_projection = dense(self.width, "question_projection")
        self.option_question_projection = dense(self.width, "option_question_projection")
        self.global_projection = dense(self.width, "global_projection")
        self.option_context_projection = dense(self.width, "option_context_projection")
        self.option_lexical_projection = dense(self.width, "option_lexical_projection")
        self.type_embedding = TokenEmbedding(len(KINDS), self.width, dtype=self.dtype, name="type_embedding")
        options = {"width": self.width, "heads": self.heads, "feedforward": self.feedforward,
                   "dtype": self.dtype, "precision": self.precision, "attention_impl": self.attention_impl}
        self.evidence_layers = [_RoutingLayer(**options, name=f"evidence_layers_{index}")
                                for index in range(self.routing_layers)]
        self.option_summary_norm = _layer_norm(self.dtype, "option_summary_norm")
        self.field_layers = [_FieldLayer(**options, name=f"layers_{index}") for index in range(self.layers)]
        self.field_norm = _layer_norm(self.dtype, "field_norm")
        self.option_norm = _layer_norm(self.dtype, "option_norm")
        self.scorer_hidden = Dense(self.width, dtype=self.dtype, precision=self.precision,
                                   name="scorer_hidden")
        self.scorer_out = Dense(1, dtype=self.dtype, precision=self.precision, name="scorer_out")
        self.prior_logit_scale = self.param("prior_logit_scale", nn.initializers.zeros_init(), ())
        self.joint_logit_scale = self.param("joint_logit_scale", nn.initializers.zeros_init(), ())
        self.residual_gate = self.param("residual_gate", nn.initializers.zeros_init(), ())

    def __call__(self, states: jax.Array, inputs: DecisionInputs, table: jax.Array | None = None,
                 train: bool = False, spanned: jax.Array | None = None) -> jax.Array:
        """The `[B, Q, K]` logits of each question's options over the row's final `states`. Where the
        questions' spans are in the row's context (`DecisionInputs.spans_in_context`), `spanned` are
        the backbone's final states over the context (`dew.nn.cross_attention.encode_context`), which the
        questions and options are pooled from, their tokens the context's; the memory every option
        and field reads, and the readout, are the row's either way."""
        if table is None:
            raise ValueError("the joint schema head reads the backbone's output table, its lexical prior")
        if inputs.spans_in_context != (spanned is not None):
            raise ValueError("a schema read as the row's context comes with the context's states, and "
                             "only then")
        hidden = self.hidden_norm(states)
        pooled, pooled_tokens = hidden, inputs.tokens
        if spanned is not None:
            assert inputs.context is not None, "the context's states come with its tokens"
            pooled, pooled_tokens = self.hidden_norm(spanned), inputs.context
        batch = inputs.tokens.shape[0]
        memory = self.memory_projection(hidden)
        # Each question reads its own row of the window: that row's last token, and its tokens alone.
        rows = inputs.question_segments
        own = (rows[:, :, None] == inputs.segments[:, None, :]) | (rows[:, :, None] == 0)
        last = jnp.max(jnp.where(own, jnp.arange(inputs.tokens.shape[1]), 0), axis=-1)
        last_state = _gathered(hidden, last)
        questions = _span_means(pooled, inputs.spans)
        count, width = inputs.options.shape[1:]
        # The option slots computed, `[B, P]` flat indices into the grid, and each one's question.
        slots = (jnp.broadcast_to(jnp.arange(count * width, dtype=jnp.int32), (batch, count * width))
                 if inputs.option_slots is None else inputs.option_slots)
        owners = slots // width
        option_spans = _gathered(inputs.option_spans.reshape(batch, count * width, 2), slots)
        contexts = _span_means(pooled, option_spans)
        rows_of_tokens = jnp.take(table, pooled_tokens, axis=0).astype(hidden.dtype)
        lexical = _span_means(rows_of_tokens, option_spans)

        queries = (self.option_context_projection(contexts) + self.option_lexical_projection(lexical)
                   + _per_slot(self.option_question_projection(questions), owners, inputs))
        # A slot reads its own row's tokens and questions, by the rows' documents; a padding slot
        # or question reads nothing, and its attention comes back zero.
        slot_rows = _per_slot(rows, owners, inputs)
        read = _Reading(inputs.segments, rows, slot_rows if inputs.slotted is None
                        else jnp.where(inputs.slotted, slot_rows, 0), inputs.max_segments)
        routed = queries
        for layer in self.evidence_layers:
            routed = layer(routed, memory, read)
        grid = _laid_back(routed, inputs)

        fields = self.question_projection(questions)
        affinity = (accumulated(functools.partial(jnp.einsum, "bqkw,bqw->bqk"), grid, fields)
                    / math.sqrt(self.width))
        weights = jax.nn.softmax(jnp.where(inputs.options, affinity, -jnp.inf), axis=-1)
        summaries = accumulated(functools.partial(jnp.einsum, "bqk,bqkw->bqw"),
                                jnp.where(inputs.options, weights, 0.0), grid)
        types = jnp.asarray(_TYPE_ROWS)[inputs.kinds]
        fields = (fields + self.option_summary_norm(summaries) + self.global_projection(last_state)
                  + self.type_embedding(types))
        for layer in self.field_layers:
            fields = layer(fields, memory, read)
        fields = self.field_norm(fields)

        anchor = _per_slot(l2_normalized(questions + last_state, 1e-12), owners, inputs)
        prior = (jnp.exp(jnp.minimum(self.prior_logit_scale, math.log(100.0)))
                 * accumulated(functools.partial(jnp.einsum, "bpd,bpd->bp"),
                               l2_normalized(lexical, 1e-12), anchor))
        options = self.option_norm(routed)
        repeated = _per_slot(fields, owners, inputs)
        cosine = jnp.sum(l2_normalized(repeated, 1e-8) * l2_normalized(options, 1e-8), axis=-1)
        features = jnp.concatenate([repeated, options, repeated * options, jnp.abs(repeated - options)], -1)
        residual = self.scorer_out(gelu_exact_torch(self.scorer_hidden(features)))[..., 0]
        joint = jnp.exp(jnp.minimum(self.joint_logit_scale, math.log(100.0))) * cosine + residual
        scores = prior + jax.nn.sigmoid(self.residual_gate) * joint
        return _laid_back(scores.astype(at_least_fp32(scores.dtype)), inputs)


def decision_geometry(lengths, questions, options):
    """One nonnegative integer geometry per row, widened before products."""
    geometry = tuple(real_lengths(column) for column in (lengths, questions, options))
    if any(column.shape != geometry[0].shape for column in geometry[1:]):
        raise ValueError("decision FLOPs need lengths, questions and options of the same [rows] shape")
    return geometry


def _answers(states: jax.Array, inputs: DecisionInputs) -> jax.Array:
    """Each question's answer state `[B, Q, D]`, at its row's answer cue (`DecisionInputs.readout`)."""
    if inputs.readout is None:
        raise ValueError("a readout of labels reads rows whose options stand under labels (Labels)")
    return _gathered(states, inputs.readout)


class LabelReadout(Head):
    """Dew's readout of labels: each option slot's logit is the backbone's own logit for its label at
    the question's answer cue (`dew.decision.layout.Labels`), through the output table's input QDQ,
    scale, bias and softcap, computed in fp32 (`DecisionModel.label_logits`, `OutputTable.columns`).
    A label read as several token variants (`label_ids[k]`, one id a variant) is their grouped logit,
    the logsumexp of theirs. Untrained, the head is the backbone's own readout.

    `untied` gives each slot's every variant a trainable row and bias of its own, drawn from its
    variant's output row (scale folded in) and bias, read through the same input QDQ and softcap and
    grouped alike, so at its draw it is the tied readout at every state; it needs the backbone's
    output table to draw from (`table`). Its rows and their product compute in at least float32, its
    table's or `dtype` where wider.

    A row's labels are the head's, its first as many as the question has options (`validate_rows`);
    option slots past the labels, a batch's padding, read -inf."""

    features: int
    label_ids: tuple[tuple[int, ...], ...]
    untied: bool = False
    dtype: Dtype | None = None
    reads_labels: ClassVar[bool] = True

    @nn.nowrap
    def tied_labels(self) -> tuple[tuple[int, ...], ...] | None:
        return None if self.untied else self.label_ids

    @nn.nowrap
    def validate_rows(self, rows: Sequence[Encoded]) -> None:
        """Refuse rows whose labels are not the head's (`_labels_of`)."""
        super().validate_rows(rows)
        _refuse_other_labels(rows, self.label_ids)

    @nn.nowrap
    def decision_flops(self, lengths, questions, options) -> float:
        """The labels' product over each answer state: its slots' variants, a row each."""
        decision_geometry(lengths, questions, options)
        variants = len(self.label_ids[0]) if self.label_ids else 1
        return projection_flops(questions, self.features, len(self.label_ids) * variants)

    @nn.compact
    def __call__(self, states: jax.Array, inputs: DecisionInputs, table=None, train: bool = False, *,
                 tied: jax.Array | None = None) -> jax.Array:
        width = inputs.options.shape[-1]
        ids = jnp.asarray(self.label_ids, jnp.int32)
        if not self.untied:
            assert tied is not None, "a tied readout reads the backbone's own label logits"
            return _past_labels(jax.nn.logsumexp(tied[..., :width, :], axis=-1), width)
        if table is None:
            raise TypeError("an untied readout draws its rows from the backbone's output table, which a head "
                            "past a transform, an adapter or a code has not; read the labels tied instead")

        def drawn(values: jax.Array) -> jax.Array:
            return values.astype(_wide(values.dtype, self.dtype))

        rows = self.param("rows", lambda key: drawn(jnp.take(table.rows(), ids.reshape(-1), axis=0)).reshape(
            *ids.shape, -1))
        bias = self.param("bias", lambda key: (
            jnp.zeros(ids.shape, _wide(table.matrix.dtype, self.dtype)) if table.bias is None
            else drawn(jnp.take(table.bias, ids.reshape(-1)).reshape(ids.shape))))
        answers = table.states(_answers(states, inputs))
        answers = answers.astype(_wide(jnp.result_type(answers, rows), self.dtype))
        logits = jnp.einsum("bqd,kvd->bqkv", answers, rows[:width], precision=jax.lax.Precision.HIGHEST)
        capped = capped_logits(scaled_logits(logits, None, bias[:width]), table.softcap)
        return _past_labels(jax.nn.logsumexp(capped, axis=-1), width)


class SlotReadout(Head):
    """A published slot release's head, its law as the release's runtime computes it: slot i's score
    is `<slot_i, h> + <q(h), k(m_i)> / sqrt(head_dim)`, `h` the question's answer state
    (`DecisionInputs.readout`), `m_i` the mean of the final states over option i's whole rendered line
    (its span), `q` and `k` each a `[head_dim, features]` projection, with no slot bias and no softcap on
    the sum. Its slot rows are drawn from the backbone's output rows of the slots' label ids, `q` from a
    normal of std `features ** -0.5` and `k` at zero, so at its draw it scores the labels' rows alone;
    a release's reader loads all three (`readout_head.pt`).

    `slot_rows` is the rows the head holds, the labels' count where unset: a release may hold more
    rows than its tokenizer spells labels for, the rows past the labels kept, drawn at zero, and read by
    no option. Drawn fresh, its slot rows are the backbone's vocabulary rows at its labels
    (`VocabularyRows`, its `table` then), or a normal of std `features ** -0.5` over a backbone that
    gives none. It computes in at least float32, its states' or `dtype` where wider; option slots past
    the labels, a batch's padding, read -inf."""

    features: int
    label_ids: tuple[int, ...]
    head_dim: int = 64
    slot_rows: int | None = None
    dtype: Dtype | None = None
    reads_labels: ClassVar[bool] = True
    draws_from_rows: ClassVar[bool] = True

    @nn.nowrap
    def validate_rows(self, rows: Sequence[Encoded]) -> None:
        """Refuse rows whose labels are not the head's (`_labels_of`)."""
        super().validate_rows(rows)
        _refuse_other_labels(rows, tuple((label,) for label in self.label_ids))

    @nn.nowrap
    def decision_flops(self, lengths, questions, options) -> float:
        """Each real option's slot product and key, its affinity with its question's query, and its
        line's pooling over the row; each question's query."""
        widths, _, counted = decision_geometry(lengths, questions, options)
        return (projection_flops(questions, self.features, self.head_dim)
                + projection_flops(options, self.features, 1 + self.head_dim)
                + 2 * np.sum(counted) * self.head_dim + 2 * np.sum(counted * widths) * self.features)

    @nn.compact
    def __call__(self, states: jax.Array, inputs: DecisionInputs, table=None, train: bool = False, *,
                 tied: jax.Array | None = None) -> jax.Array:
        width = inputs.options.shape[-1]
        labels = len(self.label_ids)
        held = labels if self.slot_rows is None else self.slot_rows
        if held < labels:
            raise ValueError(f"{held} slot rows hold fewer than the {labels} labels")
        real = min(width, labels)
        ids = np.asarray(self.label_ids, np.int32)
        drawn_dtype = _wide(jnp.float32 if table is None else table.dtype, self.dtype)

        def drawn(key):
            if table is None:
                rows = jax.random.normal(key, (labels, self.features), drawn_dtype) * self.features ** -0.5
            else:
                rows = table.take(ids, axis=0).astype(drawn_dtype)
            return jnp.pad(rows, ((0, held - labels), (0, 0)))

        slots = self.param("slots", drawn)
        q = self.param("q", nn.initializers.normal(self.features ** -0.5), (self.head_dim, self.features),
                       drawn_dtype)
        k = self.param("k", nn.initializers.zeros, (self.head_dim, self.features), drawn_dtype)
        states = states.astype(_wide(jnp.result_type(states, slots), self.dtype))
        answers = _answers(states, inputs)
        pooled = _span_means(states, inputs.option_spans[:, :, :real])
        highest = jax.lax.Precision.HIGHEST
        score = jnp.einsum("bqd,kd->bqk", answers, slots[:real], precision=highest)
        query = jnp.einsum("bqd,hd->bqh", answers, q, precision=highest)
        key = jnp.einsum("bqkd,hd->bqkh", pooled, k, precision=highest)
        scores = score + jnp.einsum("bqh,bqkh->bqk", query, key, precision=highest) / math.sqrt(self.head_dim)
        return _past_labels(scores, width)


def _wide(dtype, floor: Dtype | None) -> jnp.dtype:
    """`dtype` at least float32, and at least `floor` where one is given."""
    return at_least_fp32(dtype if floor is None else jnp.promote_types(dtype, floor))


def _past_labels(scores: jax.Array, width: int) -> jax.Array:
    """`scores` `[B, Q, K]` of a readout's labels widened to `width` option slots, -inf past them."""
    return jnp.pad(scores, ((0, 0), (0, 0), (0, width - scores.shape[-1])), constant_values=-jnp.inf)


def _refuse_other_labels(rows: Sequence[Encoded], labels: tuple[tuple[int, ...], ...]) -> None:
    """Refuse a row whose question stands under other labels than a readout's `labels`, or under none,
    or has more options than the readout reads: each question's labels are the readout's first."""
    for row in rows:
        for laid in row.questions:
            if laid.label_ids is None:
                raise ValueError(f"question {laid.name!r} stands under no labels, which a readout reads; lay "
                                 "its rows out with labels (QuestionLayout.labels)")
            if tuple(laid.label_ids) != labels[:len(laid.label_ids)] or len(laid.label_ids) > len(labels):
                raise ValueError(f"question {laid.name!r} stands under {len(laid.label_ids)} labels other "
                                 f"than the readout's first, of {len(labels)}; lay the rows out under the "
                                 "readout's labels")

