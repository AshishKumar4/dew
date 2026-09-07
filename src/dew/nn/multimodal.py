"""Native vision conditioning around the shared causal decoder."""

from __future__ import annotations

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from flax.typing import Dtype, PrecisionLike

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.vision import ProjectorBase, TowerBase
from dew.registry import models


@struct.dataclass
class Fusion:
    """Decoder token identities and their scaled, conditioned embeddings."""

    tokens: jax.Array
    embeddings: jax.Array


class VisionConditioner(nn.Module):
    """An image encoder and projector with row-aligned media inputs.

    Pixels have shape [batch, images, channels, height, width]. All images
    in a numeric batch share their processed resolution; the processor keeps
    per-image lengths and does not substitute preprocessing inside the model.
    """

    family: str
    vision: TowerBase
    projection: ProjectorBase
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        self.tower = self.vision.build().clone(dtype=self.dtype, precision=self.precision)
        self.projector = self.projection.build().clone(dtype=self.dtype, precision=self.precision)

    def __call__(self, conditioning: Mapping[str, jax.Array], train: bool = False) -> jax.Array:
        pixels = conditioning["pixel_values"]
        if pixels.ndim != 5:
            raise ValueError("pixel_values must be [B, images, C, H, W]")
        batch, images = pixels.shape[:2]
        flat = pixels.reshape(batch * images, *pixels.shape[2:])
        features = self.tower(flat)
        projected = self.projector(features)
        return projected.reshape(batch, images * projected.shape[1], projected.shape[-1])

    def fuse(self, tokens: jax.Array, embeddings: jax.Array,
             image_indices: jax.Array, conditioning: Mapping[str, jax.Array],
             train: bool = False) -> Fusion:
        """Replace marked text slots with the corresponding soft feature."""
        if image_indices.shape != tokens.shape:
            raise ValueError("image_indices must align with the token rows")
        features = self(conditioning, train=train)
        picked = features[jnp.arange(tokens.shape[0])[:, None], jnp.maximum(image_indices, 0)]
        embeddings = jnp.where((image_indices >= 0)[..., None], picked.astype(embeddings.dtype), embeddings)
        return Fusion(tokens, embeddings)


@models("multimodal_transformer")
class MultimodalTransformer(nn.Module):
    """A vision-conditioned decoder using the ordinary decoder cache and head.

    ``image_indices`` identifies the soft feature for each text slot, or -1
    for text. Media are evaluated during conditioned forward/prefill calls;
    subsequent decoding reads the cached language states without rerunning
    the vision encoder. Parameters retain the existing language_model, tower
    and projector names.
    """

    language_model: CausalTransformer
    vision: TowerBase
    projection: ProjectorBase
    family: str
    image_token_id: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str | None = None

    def setup(self):
        self.conditioner = VisionConditioner(
            self.family, self.vision, self.projection,
            dtype=self.dtype, precision=self.precision)
        nn.share_scope(self.conditioner, self)

    @property
    def vocab_size(self) -> int:
        return self.language_model.vocab_size

    @property
    def max_seq_len(self) -> int:
        return self.language_model.max_seq_len

    @property
    def emb_features(self) -> int:
        return self.language_model.emb_features

    @property
    def final_logit_softcap(self) -> float | None:
        return self.language_model.final_logit_softcap

    @property
    def causal(self) -> bool:
        return self.language_model.causal

    def hidden_states(self, tokens, train: bool = False, decode: bool = False,
                      positions=None, segment_ids=None, image_indices=None,
                      conditioning: Mapping[str, jax.Array] | None = None):
        if conditioning is None:
            if image_indices is not None:
                raise ValueError("image_indices require conditioning payloads")
            return self.language_model.hidden_states(
                tokens, train=train, decode=decode, positions=positions, segment_ids=segment_ids)
        if image_indices is None:
            raise ValueError("conditioned inputs require image_indices")
        safe_tokens = jnp.where(image_indices >= 0, 0, tokens)
        embeddings = self.language_model.embed_tokens(safe_tokens)
        if self.language_model.embedding_scale:
            embeddings = (embeddings * jnp.asarray(
                math.sqrt(self.emb_features), self.language_model.embed_tokens.embedding.dtype)).astype(embeddings.dtype)
        fused = self.conditioner.fuse(safe_tokens, embeddings, image_indices, conditioning, train=train)
        slots = jnp.broadcast_to(jnp.arange(tokens.shape[1]), tokens.shape)
        return self.language_model.hidden_states(
            fused.tokens, train=train, decode=decode, positions=positions, segment_ids=segment_ids,
            input_embeddings=fused.embeddings, embedding_positions=slots)

    def __call__(self, tokens, train: bool = False, decode: bool = False,
                 positions=None, segment_ids=None, image_indices=None,
                 conditioning: Mapping[str, jax.Array] | None = None):
        hidden = self.hidden_states(tokens, train=train, decode=decode, positions=positions,
                                    segment_ids=segment_ids, image_indices=image_indices,
                                    conditioning=conditioning)
        return self.language_model._logits(hidden)

    def head_weight(self, params):
        """The decoder's shared fp32 head matrix for chunked objective scoring."""
        return self.language_model.head_weight(params["language_model"])

    def init_cache(self, batch_size: int):
        """Allocate the nested language cache without evaluating media."""
        self.language_model.init_cache(batch_size)
