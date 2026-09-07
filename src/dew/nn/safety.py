"""CLIP vision projection and published concept-threshold image classification."""
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype

from dew.nn.text_encoders import CLIPTowerOutput, CLIPVisionTransformer


class CLIPSafetyHead(nn.Module):
    vision_model: CLIPVisionTransformer
    projection_dim: int
    concepts: int = 17
    special_concepts: int = 3
    dtype: Dtype = jnp.float32

    def setup(self):
        self.visual_projection = nn.Dense(self.projection_dim, use_bias=False, dtype=self.dtype)
        self.concept_embeds = self.param("concept_embeds", nn.initializers.ones, (self.concepts, self.projection_dim))
        self.special_care_embeds = self.param("special_care_embeds", nn.initializers.ones, (self.special_concepts, self.projection_dim))
        self.concept_embeds_weights = self.param("concept_embeds_weights", nn.initializers.ones, (self.concepts,))
        self.special_care_embeds_weights = self.param("special_care_embeds_weights", nn.initializers.ones, (self.special_concepts,))

    def features(self, pixels):
        output = self.vision_model(pixels)
        assert isinstance(output, CLIPTowerOutput)
        return self.visual_projection(output.pooler_output)

    def __call__(self, pixels):
        images = self.features(pixels)
        images = images / jnp.maximum(jnp.linalg.norm(images, axis=-1, keepdims=True), 1e-12)
        def scores(concepts, threshold):
            concepts = concepts / jnp.maximum(jnp.linalg.norm(concepts, axis=-1, keepdims=True), 1e-12)
            return images @ concepts.T - threshold
        special = jnp.round(scores(self.special_care_embeds, self.special_care_embeds_weights), 3)
        adjustment = jnp.any(special > 0, axis=-1, keepdims=True) * 0.01
        concepts = jnp.round(scores(self.concept_embeds, self.concept_embeds_weights) + adjustment, 3)
        return jnp.any(concepts > 0, axis=-1)
