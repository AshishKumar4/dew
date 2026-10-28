"""Models around a backbone: a module around one causal decoder, answering the decoder's training, cache
and serving hooks (`dew.nn.protocols`) as the decoder answers them, its decoder's parameters under the
field that holds it (a multimodal model's language model, DiffusionGemma's text decoder); and a
backbone with a head over its final states, two modules over one variables tree (`HeadedModel`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype

from dew.batch import Variables, part, thaw
from dew.nn.backbones.decoder_stack import DecoderBank
from dew.nn.protocols import (
    CrossAttending,
    HiddenStates,
    OutputTable,
    VocabularyRows,
    block_chunks_of,
    head_table,
    label_logits,
)

if TYPE_CHECKING:
    from dew._records import JSON
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.decoder_block import RematPolicy
    from dew.nn.kv_cache import KVCache


class DecoderWrapper(nn.Module):
    """The hooks a module around one decoder answers as the decoder does: its geometry, its
    banks and cache under the wrapper's field, a clone over the decoder each hook returns, and the
    decoder's projection of its own parameters. A subclass names the field (`wraps`) and keeps
    its own modules beside it as they are."""

    wraps: ClassVar[str]
    """The field that holds the decoder, the name its parameters sit under in the wrapper's tree."""

    @property
    def _decoder(self) -> CausalTransformer:
        return getattr(self, self.wraps)

    @nn.nowrap
    def _over(self, decoder) -> Self:
        """The wrapper over `decoder` in place of its own, its other modules as they are."""
        return self.clone(**{self.wraps: decoder})

    @nn.nowrap
    def replicated_flops(self, stages: int, lengths) -> float:
        return self._decoder.replicated_flops(stages, lengths)

    @property
    def vocab_size(self) -> int:
        return self._decoder.vocab_size

    @property
    def max_seq_len(self) -> int:
        return self._decoder.max_seq_len

    @property
    def position_capacity(self) -> int | None:
        return self._decoder.position_capacity

    @property
    def emb_features(self) -> int:
        return self._decoder.emb_features

    @property
    def final_logit_softcap(self) -> float | None:
        return self._decoder.final_logit_softcap

    @property
    def causal(self) -> bool:
        return self._decoder.causal

    @property
    def mask_token_id(self) -> int | None:
        return self._decoder.mask_token_id

    @property
    def shifted_logits(self) -> bool:
        return self._decoder.shifted_logits

    def bank_sites(self, store: Variables) -> tuple[DecoderBank, ...]:
        decoder = {collection: tree[self.wraps] for collection, tree in store.items() if self.wraps in tree}
        return tuple(DecoderBank((self.wraps, *site.namespace), site.view, site.scanned)
                     for site in self._decoder.bank_sites(decoder))

    @property
    def kv_cache(self) -> KVCache:
        return self._decoder.kv_cache

    @property
    def keeps_triton_gemm(self) -> bool:
        return self._decoder.keeps_triton_gemm

    @nn.nowrap
    def with_cache_capacity(self, capacity: int) -> Self:
        return self._over(self._decoder.with_cache_capacity(capacity))

    @nn.nowrap
    def with_kv_cache(self, layout: KVCache) -> Self:
        return self._over(self._decoder.with_kv_cache(layout))

    @nn.nowrap
    def projected(self, params, *, updated: bool):
        """The decoder's projection of its own parameters (`ProjectsParameters`); the wrapper's
        other parameters as they are."""
        if self.wraps not in params:
            return params
        return {**params, self.wraps: self._decoder.projected(params[self.wraps], updated=updated)}

    @nn.nowrap
    def generation_refusal(self) -> str | None:
        return self._decoder.generation_refusal()

    @nn.nowrap
    def block_chunks(self) -> tuple[int | None, ...]:
        """The decoder's trunk layers that read blocks, by their attention chunk (`BlockReading`); the
        wrapper's own modules read none."""
        return block_chunks_of(self._decoder)

    @nn.nowrap
    def with_remat(self, policy: RematPolicy | None) -> Self:
        """The decoder's layer policy; the wrapper's other modules as they were."""
        return self._over(self._decoder.with_remat(policy))

    @nn.nowrap
    def recompute_record(self) -> JSON:
        return self._decoder.recompute_record()

    @nn.nowrap
    def recompute_more(self) -> Self | None:
        decoder = self._decoder.recompute_more()
        return None if decoder is None else self._over(decoder)

    @nn.nowrap
    def recompute_half(self) -> Self | None:
        decoder = self._decoder.recompute_half()
        return None if decoder is None else self._over(decoder)

    @nn.nowrap
    def restore_recompute(self, record: JSON) -> Self:
        return self._over(self._decoder.restore_recompute(record))


@dataclass(frozen=True)
class HeadedModel:
    """Any backbone with final states (`HiddenStates`) and a head over them, applied as two modules
    over one variables tree: the two under `backbone` and `head` in every collection
    (`dew.objectives.base.joined`). Each is applied as its own module over its part, so a backbone
    that `Pretrained.adapt` gave LoRA factors runs with its own adapter, with its base weights under
    `frozen`. A task's model reads its rows through it (`dew.decision.DecisionModel`); a head reads
    the backbone's final states, its vocabulary rows or its output table, and its own logits at a
    label's tokens (`label_logits`)."""

    backbone: nn.Module
    head: nn.Module

    def __post_init__(self):
        if not isinstance(self.backbone, HiddenStates):
            raise TypeError(f"a head reads its backbone's final states (`HiddenStates`), which a "
                            f"{type(self.backbone).__name__} does not give")

    @property
    def context_length(self) -> int | None:
        """The most context tokens a row carries: the backbone's cross-attention's (`CrossAttending`),
        or None for a backbone that reads no context."""
        if not isinstance(self.backbone, CrossAttending) or self.backbone.cross_attention is None:
            return None
        return self.backbone.cross_attention.context_length

    def table(self, variables: Variables) -> jax.Array | np.ndarray:
        """Return the backbone's head as `[V, D]` rows (`VocabularyRows`), which a head may read."""
        if not isinstance(self.backbone, VocabularyRows):
            raise TypeError(f"the head reads its backbone's vocabulary rows (`VocabularyRows`), which a "
                            f"{type(self.backbone).__name__} does not give")
        rows = self.backbone.apply(thaw(part(variables, "backbone")), method="vocabulary_rows")
        if not isinstance(rows, (jax.Array, np.ndarray)):
            raise TypeError(f"vocabulary_rows must return an array, got {type(rows).__name__}")
        return rows

    def output_table(self, variables: Variables) -> OutputTable | None:
        """The backbone's head as one matrix (`AffineHead`), or None where no matrix alone is it."""
        return head_table(self.backbone, thaw(part(variables, "backbone")))

    def label_logits(self, variables: Variables, answers: jax.Array,
                     ids: tuple[tuple[int, ...], ...]) -> jax.Array:
        """The backbone's own logits `[B, N, K, V]` at each label's `ids` of each answer state of
        `answers` `[B, N, features]` (`dew.nn.protocols.label_logits`)."""
        return label_logits(self.backbone, thaw(part(variables, "backbone")), answers, ids)

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
