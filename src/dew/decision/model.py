"""A backbone and a decision head, applied as two modules over one variables tree."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype

from dew.batch import Variables, coordinates
from dew.decision.head import Head
from dew.decision.layout import DecisionInputs
from dew.nn.cross_attention import crosses_alone
from dew.nn.inputs import CONTEXT, CONTEXT_LENGTHS, ModelInputs
from dew.nn.protocols import CrossAttending, HiddenStates, VocabularyRows
from dew.objectives.base import part, thaw


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
        each row's context, embedded by the backbone's own table (`token_embeddings`),
        or, where the context holds the schema (`DecisionInputs.spans_in_context`),
        the backbone's own final states over it (`context_states`).
        """
        return self._read(variables, inputs, media=media, train=train, rngs=rngs)[0]

    def _read(self, variables: Variables, inputs: DecisionInputs, *, media: ModelInputs | None,
              train: bool, rngs: dict[str, jax.Array] | None) -> tuple[jax.Array, jax.Array | None]:
        """The row's final states (`states`), and the context's where the questions' spans are in it."""
        backbone = thaw(part(variables, "backbone"))
        laid = inputs.coordinates()
        spanned = conditioning = None
        if inputs.context is not None:
            # The context's own pass where it holds the schema, else the backbone's token table.
            spanned = (self.context_states(variables, inputs, train=train, rngs=rngs)
                       if inputs.spans_in_context else None)
            conditioning = self.conditioning(backbone, inputs, spanned)
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
            return _array(self.backbone.apply(backbone, inputs, media, rngs=rngs, method=read_media)), spanned
        if inputs.slots is None:
            return _array(self.backbone.apply(backbone, inputs.tokens, train=train, coordinates=laid,
                                              conditioning=conditioning, method="hidden_states",
                                              rngs=rngs)), spanned
        slots, segments = inputs.slots, inputs.segments
        other = (slots[:, :, None] > 0) & (slots[:, None, :] > 0) & (slots[:, :, None] != slots[:, None, :])
        # A token reads its own row's, and the padding its own; an option reads no other option.
        visible = (segments[:, :, None] == segments[:, None, :]) & ~other
        return _array(self.backbone.apply(backbone, inputs.tokens, train=train, coordinates=laid,
                                          attention_pairwise_mask=visible,
                                          attention_key_positions=inputs.positions, method="hidden_states",
                                          conditioning=conditioning, rngs=rngs)), spanned

    def context_states(self, variables: Variables, inputs: DecisionInputs, *, train: bool = False,
                       rngs: dict[str, jax.Array] | None = None) -> jax.Array:
        """The backbone's `[B, C, features]` final states over each row's context, its real tokens a
        document of their own: the same backbone and weights as the row's pass, an adapter's
        included, read without a context of its own. Where the backbone trains nothing but its
        cross-attention blocks, which a pass without a context runs none of, the pass trains
        nothing, and its gradient is not taken."""
        backbone = thaw(part(variables, "backbone"))
        assert inputs.context is not None and inputs.context_lengths is not None
        real = jnp.arange(inputs.context.shape[1])[None, :] < inputs.context_lengths[:, None]
        laid = coordinates(segments=real.astype(jnp.int32), max_segments=1, right_padded=True)
        states = _array(self.backbone.apply(backbone, inputs.context, train=train, coordinates=laid,
                                            method="hidden_states", rngs=rngs))
        trained = variables.get("params", {}).get("backbone", {})
        return jax.lax.stop_gradient(states) if crosses_alone(trained) else states



    @property
    def context_length(self) -> int | None:
        """The most context tokens a row carries: the backbone's cross-attention's (`CrossAttending`),
        or None for a backbone that reads no context."""
        if not isinstance(self.backbone, CrossAttending) or self.backbone.cross_attention is None:
            return None
        return self.backbone.cross_attention.context_length

    def conditioning(self, backbone: Variables, inputs: DecisionInputs,
                     states: jax.Array | None = None) -> dict[str, jax.Array]:
        """The conditioning a backbone's cross-attention reads for rows that carry a context: each
        row's context as `states`, its own pass's where given (`context_states`), else its tokens
        embedded by the backbone's own table (`token_embeddings`), at the width and dtype its blocks
        take them in (`CrossAttending`), and how many are real."""
        if self.context_length is None:
            raise TypeError(f"rows carry a context, which a {type(self.backbone).__name__} without "
                            "cross-attention layers does not read")
        assert isinstance(self.backbone, CrossAttending)
        assert inputs.context_lengths is not None, "a row's context comes with its length"
        context = (self.backbone.apply(backbone, inputs.context, method="token_embeddings") if states is None
                   else states)
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
        states, spanned = self._read(variables, inputs, media=media, train=train, rngs=rngs)
        table = self.table(variables) if self.head.reads_table else None
        # The questions' spans are pooled from the pass that holds them: the context's, where it holds
        # the schema.
        head = thaw(part(variables, "head"))
        logits = _array(self.head.apply(head, states, inputs, table, train=train, rngs=rngs)
                        if spanned is None else
                        self.head.apply(head, states, inputs, table, train, spanned, rngs=rngs))
        return jnp.where(inputs.options, logits, -jnp.inf)


def _array(applied: object) -> jax.Array:
    """A module's output when the call names no mutable collection."""
    if not isinstance(applied, jax.Array):
        raise TypeError(f"expected an array from the module, got {type(applied).__name__}")
    return applied
