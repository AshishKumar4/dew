"""A backbone and a decision head, applied as two modules over one variables tree."""

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import ClassVar

import jax
import jax.numpy as jnp

from dew.batch import COORDINATES_FIELD, Variables, part, prefix_sum, sows_appended, thaw
from dew.decision.head import Head, answer_states
from dew.decision.inputs import DecisionInputs
from dew.decision.layout import HOWS, ContextReader
from dew.inputs.condition import BackboneCall, BackboneEncoder, ConditionEncoder
from dew.inputs.spec import Condition, encode_conditions, token_widths
from dew.nn.cross_attention import ContextEncoder, encode_context
from dew.nn.inputs import CONTEXT, CONTEXT_LENGTHS, CONTEXT_SEGMENTS, ModelInputs
from dew.nn.inspection import INSPECTED, QK, component, in_parts
from dew.nn.progress import END, Progress, ReadsProgress, progressed
from dew.nn.protocols import EXIT_PASSES, EXITS, CountedCondition, Looped, TokenModel, VocabularyRows
from dew.nn.visibility import block_ends, block_starts
from dew.nn.wrapping import HeadedModel
from dew.objectives.base import joined


@dataclass(frozen=True)
class ContextTokens:
    """A context stream's tokens as a window packs them: `tokens` `[B, C]`, each row's first `lengths` real,
    and where a row packs several documents' contexts, each token's document (`segments`), at most
    `max_segments` a row (`dew.nn.cross_attention.encode_context`)."""

    tokens: jax.Array
    lengths: jax.Array
    segments: jax.Array | None = None
    max_segments: int = 1


@dataclass(frozen=True)
class BackboneContext(BackboneEncoder[Sequence[int]]):
    """A stream of the model's own tokens its cross-attention reads (decision-inputs.md section 2), encoded
    by the model itself (`dew.nn.cross_attention.encode_context`): `table`, each token's row of its own
    token table, or `backbone`, its own final states over the stream in a pass of its own, a forward alone
    where the model trains nothing but its cross-attention blocks. Its tokens are a `ContextTokens`."""

    encoder: ContextEncoder = "table"
    params: Variables = field(default_factory=dict, compare=False)
    """None of its own: the model it conditions holds its weights."""
    keyword: ClassVar[str] = "conditioning"

    def encode_backbone(self, backbone: BackboneCall, tokens: ContextTokens) -> Mapping[str, jax.Array]:
        return encode_context(backbone.module, backbone.variables, tokens.tokens, tokens.lengths,
                              encoder=self.encoder, train=backbone.train, rngs=backbone.rngs,
                              segments=tokens.segments, max_segments=tokens.max_segments,
                              progress=backbone.progress)

    def recorded_fields(self) -> Mapping[str, object]:
        return {"encoder": self.encoder}


@dataclass(frozen=True)
class DecisionModel(HeadedModel):
    """Any backbone with final states, read by a decision head (`HeadedModel`).

    The rows a window packs reach the backbone as its documents (`Coordinates`).
    In the parallel layout an option attends to no other option's tokens, and the
    backbone's attention windows measure distance in positions
    (`attention_pairwise_mask`, `attention_key_positions`).
    """

    head: Head
    context_encoder: ContextEncoder = "table"
    """How the backbone encodes a row's context for its cross-attention (`BackboneContext.encoder`): `table`,
    its tokens through the backbone's own embedding table, or `backbone`, its own final states over it in a
    pass of its own, which a run records; read where no other encoder is configured (`context`)."""
    context: Condition | None = None
    """The context's condition where a run configures its encoder (any `ConditionEncoder`, a frozen text
    encoder or a character table, as a diffusion run configures its conditions), its tokens the batch's
    `context` field, which the encoder tokenizes, its weights the model's `context` module; None, the
    backbone's own (`context_encoder`)."""

    @functools.cached_property
    def context_reader(self) -> "ContextReader | None":
        """The tokenization a row's context is read by where a run configures its encoder (`context`):
        built once, the one every reader of the context takes (`dew.decision.layout.context_tokens`); None,
        the decision's tokenizer."""
        return None if self.context is None else ContextReader(self.context.encoder)

    def context_condition(self, spans_in_context: bool) -> Condition:
        """The condition a row's context is (decision-inputs.md section 2): a schema read as the row's
        context is the backbone's own pass over it, whose states its spans are pooled from; else the
        configured one (`context`), or the backbone's by `context_encoder`."""
        if spans_in_context:
            return Condition(BackboneContext("backbone"), field="context")
        return self.context or Condition(BackboneContext(self.context_encoder), field="context")

    def own_pass(self, spans_in_context: bool) -> bool:
        """Whether a row's context (`context_condition`) is the backbone's own pass over it, which a step
        counts the work of (`dew.decision.DecisionObjective`), rather than its token table or another
        encoder."""
        match self.context_condition(spans_in_context).encoder:
            case BackboneContext(encoder="backbone"):
                return True
            case _:
                return False

    def _conditioned(self, condition: Condition, variables: Variables, inputs: DecisionInputs,
                     live: BackboneCall) -> Mapping[str, jax.Array]:
        """The conditioning the backbone's cross-attention reads of `inputs`' context under `condition`
        (`encode_conditions`, the one encoding of a condition): the backbone's own encoding, handed the live
        model (`BackboneContext`), or a configured encoder's over its tokens of the context under the
        model's `context` weights, its states and lengths the context's."""
        assert inputs.context is not None and inputs.context_lengths is not None
        match condition.encoder:
            case BackboneEncoder() as backbone:
                tokens = ContextTokens(inputs.context, inputs.context_lengths, inputs.context_segments,
                                       inputs.max_segments)
                return encode_conditions((("context", backbone),), {}, {"context": tokens},
                                         path="DecisionModel", backbone=live)["context"]
            case encoder:
                weights = {"context": thaw(part(variables, "context"))["params"]}

                def encoded(tokens: jax.Array, lengths: jax.Array) -> jax.Array:
                    mask = (jnp.arange(tokens.shape[1])[None, :] < lengths[:, None]).astype(jnp.int32)
                    held = {"context": {"input_ids": tokens, "attention_mask": mask}}
                    return encode_conditions((("context", encoder),), weights, held,
                                             path="DecisionModel")["context"].hidden

                # Each row's context a document of its own, a window's rows' contexts end to end, each
                # encoded alone at the encoder's own width (`_apart`), as the backbone's pass reads each
                # document alone.
                segments, documents = context_documents(inputs.context, inputs.context_lengths,
                                                        inputs.context_segments, inputs.max_segments)
                states = _apart(encoded, inputs.context, segments, documents,
                                token_widths(encoder)["input_ids"][0])
                return {CONTEXT: states, CONTEXT_LENGTHS: inputs.context_lengths,
                        **({} if inputs.context_segments is None else {CONTEXT_SEGMENTS: segments})}

    def reads_progress(self) -> bool:
        """Whether its backbone's computation changes over a run (`dew.nn.progress.ReadsProgress`)."""
        return isinstance(self.backbone, ReadsProgress) and self.backbone.reads_progress()

    def initialize_head(self, key: jax.Array, backbone: Variables, tokens: jax.Array) -> Variables:
        """Draw the head over a held backbone, at `tokens`' shape, without running the backbone.

        Readouts draw from its output table or vocabulary rows as their own contract declares;
        a tied readout has no local parameters. The token values are only initialization geometry.
        """
        head = self.head
        states = jnp.zeros((*tokens.shape, getattr(head, head.width_field)), jnp.float32)
        one = jnp.ones((tokens.shape[0], 1), bool)
        inputs = DecisionInputs(
            tokens=tokens,
            segments=jnp.ones_like(tokens),
            kinds=jnp.zeros((tokens.shape[0], 1), jnp.int32),
            questions=one,
            spans=jnp.broadcast_to(jnp.asarray([[[0, 1]]]), (*one.shape, 2)),
            option_spans=jnp.broadcast_to(jnp.asarray([[[[0, 1]]]]), (*one.shape, 1, 2)),
            options=one[..., None],
        )
        variables = joined({"backbone": backbone})
        if head.reads_labels:
            inputs = replace(inputs, readout=jnp.zeros(one.shape, jnp.int32))
            gives_rows = isinstance(self.backbone, VocabularyRows)
            table = (self.output_table(variables) if not head.draws_from_rows
                     else self.table(variables) if gives_rows else None)
            labels = head.tied_labels()
            tied = None if labels is None else jnp.zeros((*one.shape, 1, len(labels[0])), jnp.float32)
            return head.init(key, states, inputs, table, tied=tied)
        table = self.table(variables) if head.reads_table else None
        return head.init(key, states, inputs, table)

    def states(self, variables: Variables, inputs: DecisionInputs, *, media: ModelInputs | None = None,
               train: bool = False, rngs: dict[str, jax.Array] | None = None,
               progress: Progress = END) -> jax.Array:
        """Return the backbone's `[B, L, features]` final states, each packed
        row read apart from the others in its window.

        `media` are the backbone's own inputs for images a request's one row
        holds (`Images.inputs`), with which it reads them. Rows that carry a
        context (`DecisionInputs.context`) hand the backbone's cross-attention
        each row's context, embedded by the backbone's own table (`stream_embeddings`),
        or, where the context holds the schema (`DecisionInputs.spans_in_context`),
        the backbone's own final states over it (`encode_context`). The backbone reads the run's
        `progress` where it reads it (`dew.nn.progress`), the run's end by default.
        """
        return self._read(variables, inputs, media=media, train=train, rngs=rngs, progress=progress)[0]

    def _read(self, variables: Variables, inputs: DecisionInputs, *, media: ModelInputs | None,
              train: bool, rngs: dict[str, jax.Array] | None, progress: Progress,
              opened: Sequence[str] = ()) -> tuple[jax.Array, jax.Array | None, Variables]:
        """The row's final states (`states`), the context's where the questions' spans are in it, and
        what the row's pass sowed into the `opened` collections."""
        backbone = thaw(part(variables, "backbone"))
        # The run's progress, the declared input of a backbone that reads it (`dew.nn.progress`).
        stepped = progressed(self.backbone, progress)

        def applied(*args, **kwargs) -> tuple[jax.Array, Variables]:
            with component("backbone"):
                out = self.backbone.apply(backbone, *args, mutable=list(opened) or False, **kwargs)
            return out if opened else (out, {})
        # The backbone's final states: its `hidden_states`, or a native encoder's own forward.
        states_method = self.states_method
        laid = inputs.coordinates()
        # A row in blocks is read by the backbone's attention whole within each block, on the kinds
        # that read blocks (`AttentionMetadata.block_groups`); a row in order passes none.
        blocked: dict[str, object] = {} if inputs.blocks is None else {"block_groups": inputs.blocks}
        if inputs.whole_blocks:
            blocked["whole_blocks"] = True
        spanned = conditioning = None
        if inputs.context is not None:
            if self.context_length is None:
                raise TypeError(f"rows carry a context, which a {type(self.backbone).__name__} without "
                                "cross-attention layers does not read")
            assert inputs.context_lengths is not None, "a row's context comes with its length"
            # The row's context condition (`context_condition`), the backbone handed to it where it is its
            # encoder; a window's rows each read their own context (`DecisionInputs.context_segments`).
            live = BackboneCall(self.backbone, part(variables, "backbone"), train, rngs, progress)
            condition = self.context_condition(inputs.spans_in_context)
            conditioning = self._conditioned(condition, variables, inputs, live)
            spanned = conditioning[CONTEXT] if inputs.spans_in_context else None
        if media is not None:
            if inputs.slots is not None or inputs.streams is not None:
                raise ValueError("a row of option positions, or of streams, holds no images")
            def read_media(module, rows, given):
                # The images' own conditioning beside the context's, one mapping the backbone reads;
                # the rows' layout is the decision's coordinates alone, the processor's left out.
                kwargs = {name: value for name, value in given.kwargs().items() if name != COORDINATES_FIELD}
                if conditioning is not None:
                    held = kwargs.get("conditioning", {})
                    assert isinstance(held, Mapping)
                    kwargs["conditioning"] = {**held, **conditioning}
                return getattr(module, states_method)(rows.tokens, train=train, coordinates=laid, **kwargs,
                                                      **blocked, **stepped)
            states, sown = applied(inputs, media, rngs=rngs, method=read_media)
            return _array(states), spanned, sown
        if inputs.streams is not None:
            if blocked or inputs.slots is not None or inputs.reads is None:
                raise ValueError("a row in streams reads as its layout's reads say, which blocks and "
                                 "parallel options say otherwise")
            # Each token reads the streams its record says, as runs over its document's (`stream_runs`).
            runs = stream_runs(inputs.streams, inputs.segments, inputs.reads, inputs.fragments)
            states, sown = applied(inputs.tokens, train=train, coordinates=laid, attention_key_runs=runs,
                                   attention_key_positions=inputs.positions, method=states_method,
                                   conditioning=conditioning, rngs=rngs, **stepped)
            return _array(states), spanned, sown
        if inputs.slots is None:
            def read(module, tokens):
                return getattr(module, states_method)(tokens, train=train, coordinates=laid,
                                                      conditioning=conditioning, **blocked, **stepped)
            states, sown = applied(inputs.tokens, rngs=rngs, method=read)
            return _array(states), spanned, sown
        if blocked:
            raise ValueError("a row of option positions sets what each token reads, which blocks set too")
        # A token reads its own document's keys but another option's, as runs (`key_runs`): no [B, L, L] mask;
        # a causal backbone's runs end at each token.
        causal = isinstance(self.backbone, TokenModel) and self.backbone.causal
        runs = key_runs(inputs.slots, inputs.segments, inputs.options.shape[-1], causal=causal)
        states, sown = applied(inputs.tokens, train=train, coordinates=laid, attention_key_runs=runs,
                               attention_key_positions=inputs.positions, method=states_method,
                               conditioning=conditioning, rngs=rngs, **stepped)
        return _array(states), spanned, sown

    def logits(self, variables: Variables, inputs: DecisionInputs, *, media: ModelInputs | None = None,
               train: bool = False, rngs: dict[str, jax.Array] | None = None,
               progress: Progress = END) -> jax.Array:
        """Return the `[B, Q, K]` fp32 option logits, with -inf past each question's options, the
        backbone at the run's `progress` (`states`)."""
        return self.observed_logits(variables, inputs, media=media, train=train, rngs=rngs,
                                    progress=progress)[0]

    def observed_logits(self, variables: Variables, inputs: DecisionInputs, *,
                        media: ModelInputs | None = None, train: bool = False,
                        rngs: dict[str, jax.Array] | None = None, progress: Progress = END,
                        opened: Sequence[str] = ()) -> tuple[jax.Array, Variables]:
        """The option logits (`logits`), and what the row's pass sowed into the `opened` collections
        (`dew.objectives.base.Step.sown`): the backbone's and the head's, each of which it applies apart
        (`beside_head`)."""
        states, spanned, sown = self._read(variables, inputs, media=media, train=train, rngs=rngs,
                                           progress=progress, opened=opened)
        logits, head_sown = self.head_logits(variables, states, inputs, spanned, train=train, rngs=rngs,
                                             opened=opened)
        return logits, beside_head(sown, head_sown, opened)

    def exit_logits(self, variables: Variables, inputs: DecisionInputs, *,
                    media: ModelInputs | None = None) -> tuple[jax.Array, jax.Array]:
        """The option logits (`logits`) and each question's 1-based exit pass `[B, Q]` int32, read at its
        answer cue (`DecisionInputs.readout`): the pass whose state the backbone's exit threshold handed on
        for that token (`EXIT_PASSES`, `Loop.exit_threshold`), the forward the logits are read from. For a
        readout of labels over a looped backbone under an exit threshold (`reports_exits`)."""
        logits, sown = self.observed_logits(variables, inputs, media=media, opened=(EXIT_PASSES,))
        (chosen,) = jax.tree.leaves(sown[EXIT_PASSES])
        return logits, answer_states(chosen[..., None], inputs)[..., 0]

    @property
    def reports_exits(self) -> bool:
        """Whether its forward chooses each token's exit pass and reads its answers at a cue alone
        (`exit_logits`): a looped backbone under an exit threshold (`Loop.exit_threshold`) beneath a
        readout of labels that pools no spans (a slot head's answers read its options' lines too)."""
        backbone = self.backbone
        return (self.head.reads_labels and not self.head.pools_spans and isinstance(backbone, Looped)
                and backbone.loop is not None and backbone.loop.exit_threshold is not None)

    def pass_logits(self, variables: Variables, inputs: DecisionInputs, *,
                    media: ModelInputs | None = None, train: bool = False,
                    rngs: dict[str, jax.Array] | None = None, progress: Progress = END,
                    opened: Sequence[str] = ()
                    ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...], Variables]:
        """Each pass's option logits of a looped backbone whose every pass exits (`Loop.exiting`), in
        order, the last pass's `observed_logits`': each earlier pass's exit states as the backbone's forward
        gathered them (`Looped.exit_passes`) read by the head as the last pass's states are; every pass's
        exit-gate logits `[B, L, 1]`, none without a gate, which Ouro's exit objective weighs
        (`DecisionObjective.exit_entropy`); and what the rows' pass sowed into the `opened` collections, as
        `observed_logits` gives it."""
        states, spanned, sown = self._read(variables, inputs, media=media, train=train, rngs=rngs,
                                           progress=progress, opened=[*opened, EXITS])
        backbone = self.backbone
        assert isinstance(backbone, Looped), "a looped backbone's exits"
        exited = backbone.exit_passes(sown)
        earlier, gates = exited.states, exited.logits
        # Every exit through the head, each sowing what the step opened: a QK-Clip bounds the head by the
        # largest logit any scored exit reads, each exit's call kept after the last's (`sows_appended`);
        # a watched module keeps its latest call's summary, the last pass's, as a module run more than once
        # does (`dew.nn.inspection.watching`).
        read = [self.head_logits(variables, state, inputs, spanned, train=train, rngs=rngs, opened=opened)
                for state in (*earlier, states)]
        sown = {name: value for name, value in sown.items() if name != EXITS}
        head = dict(read[-1][1])
        if QK in opened:
            head[QK] = functools.reduce(sows_appended, (held.get(QK, {}) for _, held in read), {})
        return tuple(logits for logits, _ in read), gates, beside_head(sown, head, opened)

    def head_logits(self, variables: Variables, states: jax.Array, inputs: DecisionInputs,
                    spanned: jax.Array | None = None, *, train: bool = False,
                    rngs: dict[str, jax.Array] | None = None,
                    opened: Sequence[str] = ()) -> tuple[jax.Array, Variables]:
        """The head's option logits `[B, Q, K]` over the backbone's `states`, -inf past each question's
        options, and what it sowed into the `opened` collections: the head's part of `observed_logits`,
        which reads its questions' spans from `spanned`, the pass that holds them, where the head pools
        them."""
        head = thaw(part(variables, "head"))
        head_sown: dict = {}

        def applied(*args, **kwargs) -> jax.Array:
            if not opened:
                return _array(self.head.apply(head, *args, **kwargs))
            with component("head"):
                out, held = self.head.apply(head, *args, mutable=list(opened), **kwargs)
            head_sown.update(held)
            return _array(out)
        # The questions' spans are pooled from the pass that holds them: the context's, where it holds
        # them. A head that pools none reads a question in its context through the cross-attention alone.
        spans = spanned if self.head.pools_spans else None
        if self.head.reads_labels:
            # A readout of labels reads its answer cue in the row.
            labels = self.head.tied_labels()
            tied = None if labels is None else self.readout_logits(variables, states, inputs, labels)
            # A head drawn from the vocabulary rows reads no table after its draw.
            table = None if self.head.draws_from_rows else self.output_table(variables)
            logits = _array(
                applied(states, inputs, table, tied=tied, train=train, rngs=rngs)
                if spans is None else
                applied(states, inputs, table, tied=tied, spanned=spans, train=train, rngs=rngs))
        else:
            table = self.table(variables) if self.head.reads_table else None
            logits = _array(
                applied(states, inputs, table, train=train, rngs=rngs) if spans is None
                else applied(states, inputs, table, spanned=spans, train=train, rngs=rngs))
        return jnp.where(inputs.options, logits, -jnp.inf), head_sown

    def readout_logits(self, variables: Variables, states: jax.Array, inputs: DecisionInputs,
                       labels: tuple[tuple[int, ...], ...]) -> jax.Array:
        """The backbone's own logits `[B, Q, K, V]` at each label's ids of each question's answer
        state (`DecisionInputs.readout`), its first `K` labels as many as its option slots
        (`HeadedModel.label_logits`)."""
        answers = answer_states(states, inputs)
        return self.label_logits(variables, answers, labels)[:, :, :inputs.options.shape[-1]]


def beside_head(sown: Variables, head: Variables, opened: Sequence[str]) -> Variables:
    """What a row's backbone sowed (`sown`) with what its head sowed (`head`) beside it, in each collection
    the step `opened`: the attention logit maxima (`qk`) of each under its own parameters' name
    (`backbone`, `head`), so a QK-Clip finds every layer's maxima where it finds its weights (row 1608);
    the watched activations (`inspected`) each under its part of the model's variables
    (`dew.nn.inspection.in_parts`), which the log names them by (`backbone/...`, `head/...`); every other
    collection the head sowed into, alike."""
    merged = dict(sown)
    for name in opened:
        if name == QK:
            merged[QK] = {"backbone": sown.get(QK, {}), "head": head.get(QK, {})}
        elif name == INSPECTED or head.get(name):
            merged[name] = in_parts({"backbone": sown.get(name, {}), "head": head.get(name, {})})
    return merged


def key_runs(slots: jax.Array, segments: jax.Array, options: int, *, causal: bool = False) -> jax.Array:
    """Each token's key runs `[B, L, options + 2, 2]` in the parallel layout, the visibility `_read`
    gives its backbone as runs (`AttentionMetadata.key_runs`): a token reads its own document's keys but
    another option's, over `slots` `[B, L]` (0 shared, k for option k's tokens) and `segments` `[B, L]`
    (`DecisionInputs`), and under `causal` none after it. The runs are the document's fragments: its
    prefix, each of its options, its suffix, each a run of its own, in which the layout's positions never
    decrease (`layout._parallel` restarts them only between options). A document's options lie back to
    back, as `layout._parallel` lays them, its shared tokens only before the first and after the last: a
    shared token between two options would lie in no fragment. A shared token reads every fragment;
    an option's token the prefix, its own option and the suffix; padding (segment 0), trailing, its own
    span. An unread fragment is `(0, 0)`. `options` is the layout's option capacity (the last axis of
    `DecisionInputs.options`), so the runs' count is static."""
    length = slots.shape[1]
    index = jnp.arange(length, dtype=jnp.int32)
    first, last = block_starts(segments), block_ends(segments)
    # Each document's number in its row, which the per-document reductions key by.
    documents = prefix_sum(index[None, :] == first, dtype=jnp.int32) - 1

    def spans(slot_row, document_row):
        # Each option's span in each token's document, `[options, L]` twice, empty past the keys.
        def span(option):
            hit = slot_row == option
            lo = jax.ops.segment_min(jnp.where(hit, index, length), document_row, num_segments=length)
            hi = jax.ops.segment_max(jnp.where(hit, index + 1, 0), document_row, num_segments=length)
            return lo[document_row], hi[document_row]
        return jax.vmap(span)(jnp.arange(1, options + 1))

    option_lo, option_hi = jax.vmap(spans)(slots, documents)
    option_lo, option_hi = (jnp.moveaxis(bound, 1, 2) for bound in (option_lo, option_hi))
    held = option_hi > option_lo
    any_option = jnp.any(held, axis=-1)
    prefix_hi = jnp.where(any_option, jnp.min(jnp.where(held, option_lo, length), axis=-1), last)
    suffix_lo = jnp.where(any_option, jnp.max(jnp.where(held, option_hi, 0), axis=-1), last)
    lo = jnp.concatenate([first[..., None], option_lo, suffix_lo[..., None]], axis=-1)
    hi = jnp.concatenate([prefix_hi[..., None], option_hi, last[..., None]], axis=-1)
    # An option's token reads no other option.
    fragment = jnp.arange(options + 2)
    read = ((slots[..., None] == 0) | (fragment == 0) | (fragment == options + 1)
            | (fragment == slots[..., None]))
    if causal:
        hi = jnp.minimum(hi, index[None, :, None] + 1)
    kept = read & (hi > lo)
    return jnp.stack([jnp.where(kept, lo, 0), jnp.where(kept, hi, 0)], axis=-1).astype(jnp.int32)


def stream_runs(streams: jax.Array, segments: jax.Array, reads: Sequence[Sequence[int]],
                fragments: int) -> jax.Array:
    """Each token's key runs `[B, L, fragments, 2]` where its row's tokens stand in streams
    (`DecisionInputs.streams`, `[B, L]`, -1 on the padding), the visibility `_read` gives its backbone as
    runs (`AttentionMetadata.key_runs`): a token reads the runs of its own document's `fragments` (each
    a run of one stream, `fragment_count`) its stream reads by `reads` (`Reads.lowered`: per stream, how
    it reads itself, then each stream; 1 whole, 2 up to the token, 0 none), padding (segment 0) its own
    span alone. An unread run is `(0, 0)`."""
    batch, length = streams.shape
    index = jnp.arange(length, dtype=jnp.int32)
    first = block_starts(segments)
    # A run opens where its document does or its stream turns; each token's run is numbered in its
    # document, and every run keyed by its document's first token.
    turned = jnp.concatenate([jnp.ones((batch, 1), bool), streams[:, 1:] != streams[:, :-1]], axis=1)
    count = prefix_sum(turned | (index[None, :] == first), dtype=jnp.int32)
    fragment = count - jnp.take_along_axis(count, first, axis=1)
    keyed = first * fragments + fragment

    def tables(keys, stream_row):
        slots = length * fragments
        return (jax.ops.segment_min(index, keys, num_segments=slots),
                jax.ops.segment_max(index + 1, keys, num_segments=slots),
                jax.ops.segment_max(stream_row, keys, num_segments=slots))

    lo, hi, kind = jax.vmap(tables)(keyed, streams)
    own = (first[..., None] * fragments + jnp.arange(fragments)).reshape(batch, -1)
    lo, hi, kind = (jnp.take_along_axis(table, own, axis=1).reshape(batch, length, fragments)
                    for table in (lo, hi, kind))
    # The padding reads itself whole and nothing else: a stream of its own past the layout's.
    count_streams = len(reads)
    padding = [1, *[0] * (count_streams + 1)]
    table = jnp.asarray([*([own_how, *others, 0] for own_how, *others in reads), padding], jnp.int32)
    reader = jnp.where(streams < 0, count_streams, streams)
    read = jnp.where(kind < 0, count_streams, kind)
    how = jnp.where(read == reader[..., None], table[reader, 0][..., None],
                    table[reader[..., None], 1 + read])
    # A causal read (`HOWS`) reads up to the token itself.
    hi = jnp.where(how == HOWS["causal"], jnp.minimum(hi, index[None, :, None] + 1), hi)
    kept = (how > 0) & (hi > lo)
    return jnp.stack([jnp.where(kept, lo, 0), jnp.where(kept, hi, 0)], axis=-1).astype(jnp.int32)


def _array(applied: object) -> jax.Array:
    """A module's output when the call names no mutable collection."""
    if not isinstance(applied, jax.Array):
        raise TypeError(f"expected an array from the module, got {type(applied).__name__}")
    return applied


def context_documents(tokens: jax.Array, lengths: jax.Array, segments: jax.Array | None,
                      documents: int) -> tuple[jax.Array, int]:
    """Each context token's document (1 to `documents`, 0 none) and how many a row holds at most: a packed
    window's own (`DecisionInputs.context_segments`), or each row's context one document of its
    `lengths`."""
    if segments is not None:
        return segments, documents
    return (jnp.arange(tokens.shape[1])[None, :] < lengths[:, None]).astype(jnp.int32), 1


def document_lengths(segments: jax.Array, documents: int) -> jax.Array:
    """Each row's documents' token counts `[B, documents]`, by `segments` (`context_documents`)."""
    return jnp.sum(segments[:, :, None] == jnp.arange(1, documents + 1), axis=1)


def state_width(encoder: ConditionEncoder) -> int:
    """The width of the states a configured context encoder hands the cross-attention, as it declares its
    work (`CountedCondition.conditioning_flops`, `ConditionWork.features`)."""
    if not isinstance(encoder, CountedCondition):
        raise ValueError(f"{type(encoder).__name__} declares no conditioning_flops, whose features are the "
                         "width its states reach the context layers at")
    return encoder.conditioning_flops(encoder.tokenize([""])).features


def _apart(encoded: Callable[[jax.Array, jax.Array], jax.Array], tokens: jax.Array, segments: jax.Array,
           documents: int, width: int) -> jax.Array:
    """`encoded` (tokens `[N, width]` and lengths `[N]` to states `[N, width, F]`) over each document of a
    window's context alone: the tokens of each document `segments` `[B, C]` names (1 to `documents`, 0
    none) moved to a row of their own `width` long, encoded, and their states moved back to where the
    tokens stood, zeros where none does; so a configured encoder keeps a window's rows' contexts apart, as
    the backbone's pass does, each at its own width (a document is at most that long: the encoder's
    tokenization kept no more of it, `dew.decision.layout.context_tokens`)."""
    batch, columns = tokens.shape
    owned = segments[:, :, None] == jnp.arange(1, documents + 1)
    place = jnp.sum((jnp.cumsum(owned, axis=1) - 1) * owned, axis=-1)
    document = jnp.clip(segments - 1, 0, documents - 1)
    rows = jnp.broadcast_to(jnp.arange(batch)[:, None], (batch, columns))
    # A token of no document is dropped (an index past the documents').
    target = jnp.where(segments > 0, document, documents)
    apart = jnp.zeros((batch, documents, width), tokens.dtype)
    apart = apart.at[rows, target, place].set(tokens, mode="drop")
    lengths = document_lengths(segments, documents).reshape(-1)
    states = encoded(apart.reshape(batch * documents, width), lengths)
    states = states.reshape(batch, documents, width, -1)[rows, document, jnp.minimum(place, width - 1)]
    return jnp.where((segments > 0)[:, :, None], states, 0).astype(states.dtype)
