"""A backbone and a decision head, applied as two modules over one variables tree."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype

from dew.decision.head import Head
from dew.decision.layout import DecisionInputs
from dew.nn.inputs import CONTEXT, CONTEXT_LENGTHS, ModelInputs
from dew.nn.protocols import CrossAttending, HiddenStates, VocabularyRows
from dew.nn.visibility import block_ends, block_starts
from dew.objectives.base import Variables, part, thaw


@dataclass(frozen=True)
class DecisionModel:
    """Any backbone with final states (`HiddenStates`), read by a decision head.

    The variables hold the two under `backbone` and `head` in every collection
    (`dew.objectives.base.joined`). Each is applied as its own module over its
    part, so a backbone that `Pretrained.adapt` gave LoRA factors runs with its
    own adapter, with its base weights under `frozen`.

    The rows a window packs reach the backbone as its documents (`Coordinates`).
    In the parallel layout an option attends to no other option's tokens, and the
    backbone's attention windows measure distance in positions
    (`attention_pairwise_mask`, `attention_key_positions`).
    """

    backbone: nn.Module
    head: Head

    def __post_init__(self):
        if not isinstance(self.backbone, HiddenStates):
            raise TypeError(f"a decision head reads its backbone's final states (`HiddenStates`), "
                            f"which a {type(self.backbone).__name__} does not give")

    def states(self, variables: Variables, inputs: DecisionInputs, *, media: ModelInputs | None = None,
               train: bool = False, rngs: dict[str, jax.Array] | None = None) -> jax.Array:
        """Return the backbone's `[B, L, features]` final states, each packed
        row read apart from the others in its window.

        `media` are the backbone's own inputs for images a request's one row
        holds (`Images.inputs`), with which it reads them. Rows that carry a
        context (`DecisionInputs.context`) hand the backbone's cross-attention
        each row's context tokens embedded by the backbone's own table
        (`token_embeddings`).
        """
        backbone = thaw(part(variables, "backbone"))
        laid = inputs.coordinates()
        conditioning = None if inputs.context is None else self.conditioning(backbone, inputs)
        if media is not None:
            if inputs.slots is not None:
                raise ValueError("a row of option positions holds no images")
            def read_media(module, rows, given):
                # The images' own conditioning beside the context's, one mapping the backbone reads.
                kwargs = dict(given.kwargs())
                if conditioning is not None:
                    held = kwargs.get("conditioning", {})
                    assert isinstance(held, Mapping)
                    kwargs["conditioning"] = {**held, **conditioning}
                return module.hidden_states(rows.tokens, train=train, coordinates=laid, **kwargs)
            return _array(self.backbone.apply(backbone, inputs, media, rngs=rngs, method=read_media))
        if inputs.slots is None:
            return _array(self.backbone.apply(backbone, inputs.tokens, train=train, coordinates=laid,
                                              conditioning=conditioning, method="hidden_states", rngs=rngs))
        slots, segments = inputs.slots, inputs.segments
        other = (slots[:, :, None] > 0) & (slots[:, None, :] > 0) & (slots[:, :, None] != slots[:, None, :])
        # A token reads its own row's, and the padding its own; an option reads no other option.
        visible = (segments[:, :, None] == segments[:, None, :]) & ~other
        return _array(self.backbone.apply(backbone, inputs.tokens, train=train, coordinates=laid,
                                          attention_pairwise_mask=visible,
                                          attention_key_positions=inputs.positions, method="hidden_states",
                                          conditioning=conditioning, rngs=rngs))

    @property
    def context_length(self) -> int | None:
        """The most context tokens a row carries: the backbone's cross-attention's (`CrossAttending`),
        or None for a backbone that reads no context."""
        if not isinstance(self.backbone, CrossAttending) or self.backbone.cross_attention is None:
            return None
        return self.backbone.cross_attention.context_length

    def conditioning(self, backbone: Variables, inputs: DecisionInputs) -> dict[str, jax.Array]:
        """The conditioning a backbone's cross-attention reads for rows that carry a context: each
        row's context tokens embedded by the backbone's own table (`token_embeddings`), at the
        width and dtype its blocks take them in (`CrossAttending`), and how many are real."""
        if self.context_length is None:
            raise TypeError(f"rows carry a context, which a {type(self.backbone).__name__} without "
                            "cross-attention layers does not read")
        assert isinstance(self.backbone, CrossAttending)
        assert inputs.context_lengths is not None, "a row's context comes with its length"
        context = self.backbone.apply(backbone, inputs.context, method="token_embeddings")
        return {CONTEXT: jnp.asarray(context, self.backbone.context_dtype),
                CONTEXT_LENGTHS: inputs.context_lengths}

    def table(self, variables: Variables) -> jax.Array:
        """Return the backbone's head as `[V, D]` rows (`VocabularyRows`), which a head may read."""
        if not isinstance(self.backbone, VocabularyRows):
            raise TypeError(f"the head reads its backbone's vocabulary rows (`VocabularyRows`), which a "
                            f"{type(self.backbone).__name__} does not give")
        return jnp.asarray(self.backbone.apply(thaw(part(variables, "backbone")), method="vocabulary_rows"))

    @staticmethod
    def head_size(backbone: nn.Module) -> tuple[int, Dtype | None]:
        """Return the width of `backbone`'s final states, and the dtype a head over them computes in.

        The backbone is traced, not run, so this costs no compute. A backbone that
        computes in reduced precision has its head compute in it too; over
        full-precision states a head computes at its parameters' precision.
        """
        tokens = jnp.zeros((1, 2), jnp.int32)

        def states(key: jax.Array) -> jax.Array:
            return backbone.init_with_output(key, tokens, method="hidden_states")[0]

        shape = jax.eval_shape(states, jax.random.key(0))
        reduced = jnp.finfo(shape.dtype).bits < 32
        return shape.shape[-1], shape.dtype if reduced else None

    def logits(self, variables: Variables, inputs: DecisionInputs, *, media: ModelInputs | None = None,
               train: bool = False, rngs: dict[str, jax.Array] | None = None) -> jax.Array:
        """Return the `[B, Q, K]` fp32 option logits, with -inf past each question's options."""
        states = self.states(variables, inputs, media=media, train=train, rngs=rngs)
        table = self.table(variables) if self.head.reads_table else None
        logits = _array(self.head.apply(thaw(part(variables, "head")), states, inputs, table, train=train,
                                        rngs=rngs))
        return jnp.where(inputs.options, logits, -jnp.inf)


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
    documents = jnp.cumsum(index[None, :] == first, axis=1) - 1

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
