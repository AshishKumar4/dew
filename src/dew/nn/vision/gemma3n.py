"""Gemma 3n's MobileNet-v5 vision tower and its hard/soft vision embedder."""

from __future__ import annotations

import dataclasses
import functools

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.blocks import TokenEmbedding
from dew.nn.dense import Dense
from dew.nn.norms import RMSNorm
from dew.nn.precision import float32_dot_general

from ..mobilenet import MobileNetV5, MobileNetV5Encoder
from .common import ProjectorBase, TowerBase, TowerGeometry


@dataclasses.dataclass(frozen=True)
class Gemma3nVision(MobileNetV5, TowerBase):
    """Gemma 3n's vision tower: MobileNet-v5's encoder, by its construction
    fields (`MobileNetV5`)."""

    def build(self) -> nn.Module:
        return MobileNetV5Encoder(MobileNetV5(**dataclasses.asdict(self)))

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(channels=self.in_chans)


class Gemma3nProjector(ProjectorBase):
    """Gemma3nMultimodalEmbedder's vision hard and soft token paths."""

    vision_width: int
    text_width: int
    vocab_size: int = 128
    vocab_offset: int = 262144
    norm_eps: float = 1e-6
    scales_soft: bool = True
    """Whether soft features are multiplied by sqrt(`vision_width`) before the embedder reads them:
    Gemma 3n's image features are (modeling_gemma3n.py:1983 at transformers 5.17.0), and its audio
    embedder reads its encoder's features as they are (:2192), which its norm's epsilon tells apart."""
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        if min(self.vision_width, self.text_width, self.vocab_size) < 1 or self.vocab_offset < 0:
            raise ValueError(
                "vision/text widths and vocab_size must be positive; vocab_offset is nonnegative"
            )
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        self.embedding = TokenEmbedding(self.vocab_size, self.vision_width, dtype=self.dtype,
                                  name="embedding")
        norm = functools.partial(RMSNorm, epsilon=self.norm_eps, dtype=self.dtype)
        self.hard_embedding_norm = norm(name="hard_embedding_norm")
        self.soft_embedding_norm = norm(name="soft_embedding_norm")
        self.embedding_projection = Dense(self.text_width, use_bias=False,
                                              dtype=self.dtype, precision=self.precision,
                                              name="embedding_projection", dot_general=float32_dot_general)
        self.embedding_post_projection_norm = norm(with_scale=False,
                                                   name="embedding_post_projection_norm")

    def __call__(self, features):
        features = jnp.asarray(features)
        if self.scales_soft:
            features = features * jnp.asarray(self.vision_width ** 0.5, features.dtype)
        return self.soft_embeddings(features)

    def soft_embeddings(self, features):
        """The reference embedder over soft features, without `scales_soft`'s scaling."""
        features = jnp.asarray(features)
        if features.ndim != 3 or features.shape[-1] != self.vision_width:
            raise ValueError(f"soft features must be [B, N, {self.vision_width}], got {features.shape}")
        if not jnp.issubdtype(features.dtype, jnp.floating):
            raise ValueError("soft features must be floating point")
        if self.is_initializing():
            # Both paths belong to one checkpoint even when init starts with
            # image features. Linen creates an embedding only when called.
            self.hard_embedding_norm(self.embedding(jnp.zeros((1, 1), jnp.int32)))
        return self.embedding_post_projection_norm(
            self.embedding_projection(self.soft_embedding_norm(features)))

    def embed_hard(self, ids):
        """Embed vocabulary ids in [vocab_offset, vocab_offset + vocab_size).

        Pure: the host processor validates ids before device work.
        """
        embedded = self.embedding(ids - self.vocab_offset)
        if self.is_initializing():
            self.soft_embedding_norm(jnp.zeros_like(embedded))
        return self.embedding_post_projection_norm(
            self.embedding_projection(self.hard_embedding_norm(embedded)))

    def merge_hard_embeddings(self, token_embeddings, ids):
        """Replace this vocabulary range's slots with hard embeddings, through the
        reference's dummy id for every other slot."""
        mask = (ids >= self.vocab_offset) & (ids < self.vocab_offset + self.vocab_size)
        chosen = jnp.where(mask, ids, self.vocab_offset + self.vocab_size - 1)
        hard = self.embed_hard(chosen).astype(token_embeddings.dtype)
        return jnp.where(mask[..., None], hard, token_embeddings)


