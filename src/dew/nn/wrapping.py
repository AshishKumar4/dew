"""A module around one causal decoder, answering the decoder's training, cache and serving hooks
(`dew.nn.protocols`) as the decoder answers them, its decoder's parameters under the field that holds
it: a multimodal model's language model, DiffusionGemma's text decoder."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

from flax import linen as nn

from dew.nn.backbones.decoder_stack import DecoderBank
from dew.nn.protocols import block_chunks_of

if TYPE_CHECKING:
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.decoder_block import RematPolicy
    from dew.nn.kv_cache import KVCache
    from dew.records import JSON


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

    @property
    def vocab_size(self) -> int:
        return self._decoder.vocab_size

    @property
    def max_seq_len(self) -> int:
        return self._decoder.max_seq_len

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

    @property
    def bank_sites(self) -> tuple[DecoderBank, ...]:
        return tuple(DecoderBank((self.wraps, *site.namespace), site.view, site.scanned)
                     for site in self._decoder.bank_sites)

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
