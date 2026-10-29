"""A backbone and a decision head, applied as two modules over one variables tree."""

from collections.abc import Mapping
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from dew.batch import Variables, part, prefix_sum, thaw
from dew.decision.head import Head
from dew.decision.layout import DecisionInputs
from dew.nn.cross_attention import ContextEncoder, encode_context
from dew.nn.inputs import CONTEXT, ModelInputs
from dew.nn.protocols import TokenModel, VocabularyRows
from dew.nn.scatter import gather_positions
from dew.nn.visibility import block_ends, block_starts
from dew.nn.wrapping import HeadedModel
from dew.objectives.base import joined


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
    """How a row's context becomes the states its cross-attention reads (`encode_context`): `table`, its
    tokens through the backbone's own embedding table, or `backbone`, the backbone's final states over
    it in a pass of its own, as a schema read in context always is."""

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
               train: bool = False, rngs: dict[str, jax.Array] | None = None) -> jax.Array:
        """Return the backbone's `[B, L, features]` final states, each packed
        row read apart from the others in its window.

        `media` are the backbone's own inputs for images a request's one row
        holds (`Images.inputs`), with which it reads them. Rows that carry a
        context (`DecisionInputs.context`) hand the backbone's cross-attention
        each row's context, embedded by the backbone's own table (`token_embeddings`),
        or, where the context holds the schema (`DecisionInputs.spans_in_context`),
        the backbone's own final states over it (`encode_context`).
        """
        return self._read(variables, inputs, media=media, train=train, rngs=rngs)[0]

    def _read(self, variables: Variables, inputs: DecisionInputs, *, media: ModelInputs | None,
              train: bool, rngs: dict[str, jax.Array] | None) -> tuple[jax.Array, jax.Array | None]:
        """The row's final states (`states`), and the context's where the questions' spans are in it."""
        backbone = thaw(part(variables, "backbone"))
        laid = inputs.coordinates()
        # A row in blocks is read by the backbone's attention whole within each block, on the kinds
        # that read blocks (`AttentionMetadata.block_groups`); a row in order passes none.
        blocked: dict[str, jax.Array] = {} if inputs.blocks is None else {"block_groups": inputs.blocks}
        spanned = conditioning = None
        if inputs.context is not None:
            if self.context_length is None:
                raise TypeError(f"rows carry a context, which a {type(self.backbone).__name__} without "
                                "cross-attention layers does not read")
            assert inputs.context_lengths is not None, "a row's context comes with its length"
            # The context's own pass where it holds the schema, whose spans the head pools from that
            # same pass, else as the model encodes it.
            # A window's rows each read their own context (`DecisionInputs.context_segments`).
            conditioning = encode_context(
                self.backbone, part(variables, "backbone"), inputs.context, inputs.context_lengths,
                encoder="backbone" if inputs.spans_in_context else self.context_encoder, train=train,
                rngs=rngs, segments=inputs.context_segments, max_segments=inputs.max_segments)
            spanned = conditioning[CONTEXT] if inputs.spans_in_context else None
        if media is not None:
            if inputs.slots is not None:
                raise ValueError("a row of option positions holds no images")
            def read_media(module, rows, given):
                # The images' own conditioning beside the context's, one mapping the backbone reads;
                # the rows' layout is the decision's coordinates alone, the processor's flat fields
                # left out (`token_coordinates`).
                kwargs = {name: value for name, value in given.kwargs().items()
                          if name not in ("positions", "attention_mask", "segment_ids")}
                if conditioning is not None:
                    held = kwargs.get("conditioning", {})
                    assert isinstance(held, Mapping)
                    kwargs["conditioning"] = {**held, **conditioning}
                return module.hidden_states(rows.tokens, train=train, coordinates=laid, **kwargs, **blocked)
            return _array(self.backbone.apply(backbone, inputs, media, rngs=rngs, method=read_media)), spanned
        if inputs.slots is None:
            def read(module, tokens):
                return module.hidden_states(tokens, train=train, coordinates=laid, conditioning=conditioning,
                                            **blocked)
            return _array(self.backbone.apply(backbone, inputs.tokens, rngs=rngs, method=read)), spanned
        if blocked:
            raise ValueError("a row of option positions sets what each token reads, which blocks set too")
        # A token reads its own document's keys but another option's, as runs (`key_runs`): no [B, L, L] mask;
        # a causal backbone's runs end at each token.
        causal = isinstance(self.backbone, TokenModel) and self.backbone.causal
        runs = key_runs(inputs.slots, inputs.segments, inputs.options.shape[-1], causal=causal)
        return _array(self.backbone.apply(backbone, inputs.tokens, train=train, coordinates=laid,
                                          attention_key_runs=runs,
                                          attention_key_positions=inputs.positions, method="hidden_states",
                                          conditioning=conditioning, rngs=rngs)), spanned

    def logits(self, variables: Variables, inputs: DecisionInputs, *, media: ModelInputs | None = None,
               train: bool = False, rngs: dict[str, jax.Array] | None = None) -> jax.Array:
        """Return the `[B, Q, K]` fp32 option logits, with -inf past each question's options."""
        states, spanned = self._read(variables, inputs, media=media, train=train, rngs=rngs)
        head = thaw(part(variables, "head"))
        if self.head.reads_labels:
            if spanned is not None:
                raise TypeError("a readout of labels reads its questions in the row, not in its context")
            labels = self.head.tied_labels()
            tied = None if labels is None else self.readout_logits(variables, states, inputs, labels)
            # A head drawn from the vocabulary rows reads no table after its draw.
            table = None if self.head.draws_from_rows else self.output_table(variables)
            logits = _array(self.head.apply(head, states, inputs, table, train=train, tied=tied, rngs=rngs))
        else:
            table = self.table(variables) if self.head.reads_table else None
            # The questions' spans are pooled from the pass that holds them: the context's, where it
            # holds the schema.
            logits = _array(self.head.apply(head, states, inputs, table, train=train, rngs=rngs)
                            if spanned is None else
                            self.head.apply(head, states, inputs, table, train, spanned, rngs=rngs))
        return jnp.where(inputs.options, logits, -jnp.inf)

    def readout_logits(self, variables: Variables, states: jax.Array, inputs: DecisionInputs,
                       labels: tuple[tuple[int, ...], ...]) -> jax.Array:
        """The backbone's own logits `[B, Q, K, V]` at each label's ids of each question's answer
        state (`DecisionInputs.readout`), its first `K` labels as many as its option slots
        (`HeadedModel.label_logits`)."""
        if inputs.readout is None:
            raise ValueError("a readout of labels reads rows whose options stand under labels (Labels)")
        answers = gather_positions(states, inputs.readout)
        return self.label_logits(variables, answers, labels)[:, :, :inputs.options.shape[-1]]


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


def _array(applied: object) -> jax.Array:
    """A module's output when the call names no mutable collection."""
    if not isinstance(applied, jax.Array):
        raise TypeError(f"expected an array from the module, got {type(applied).__name__}")
    return applied
