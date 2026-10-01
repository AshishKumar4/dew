"""Representation alignment: REPA and iREPA.

REPA (Yu et al. 2025, "Representation Alignment for Generation: Training
Diffusion Transformers Is Easier Than You Think") adds to the denoising loss
the negative cosine similarity between a frozen encoder's patch features of
the clean image and a projection of the denoiser's hidden tokens at one
layer, averaged over tokens and examples. iREPA (Singh et al. 2026, "What
matters for Representation Alignment: Global Information or Spatial
Structure?") projects with one convolution over the token grid instead of an
MLP and z-scores the encoder's features over space. The official code is
sihyun-yu/REPA's `loss.py` and `models/sit.py`, and End2End-Diffusion/iREPA's
`projectors.py` and `spatial_zscore`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.nn.blocks import torch_bicubic_resize
from dew.objectives.base import Variables

ALIGNMENT = "alignment_projector"
"""Where the projector lives in `params`, beside the model's own modules."""

REPRESENTATION = "representation"
"""The collection the frozen encoder's weights ride under, as `encoders` do."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Projector(nn.Module):
    """REPA's `build_mlp` (two hidden SiLU layers of `width`) or iREPA's one
    same-padded convolution over the token grid, into `features`."""

    kind: Literal["mlp", "conv"]
    features: int
    width: int = 2048
    kernel_size: int = 3

    @nn.compact
    def __call__(self, tokens: jax.Array) -> jax.Array:
        if self.kind == "mlp":
            hidden = nn.silu(nn.Dense(self.width)(tokens))
            hidden = nn.silu(nn.Dense(self.width)(hidden))
            return nn.Dense(self.features)(hidden)
        batch, count, width = tokens.shape
        side = math.isqrt(count)
        if side * side != count:
            raise ValueError(f"a convolution projector needs a square token grid, not {count} tokens")
        grid = nn.Conv(self.features, (self.kernel_size, self.kernel_size), padding="SAME")(
            tokens.reshape(batch, side, side, width))
        return grid.reshape(batch, count, self.features)


def spatial_zscore(features: jax.Array, gamma: float) -> jax.Array:
    """iREPA's spatial normalization of `[B, N, D]` features: gamma times the
    mean over the tokens subtracted, then divided by their standard
    deviation (torch's unbiased `Tensor.std`, the reference's) plus 1e-6."""
    mean = jnp.mean(features, axis=1, keepdims=True)
    std = jnp.std(features, axis=1, keepdims=True, ddof=1)
    return (features - gamma * mean) / (std + 1e-6)


@dataclass(frozen=True)
class Alignment:
    """Align the model's hidden tokens at `layer` with a frozen encoder's.

    `encoder` is a Flax module mapping pixels, `[B, S, S, 3]` normalized by
    `mean` and `std` after resizing to `resolution` (None keeps the data's),
    to patch features `[B, N, D]` or `[B, h, w, D]`; `variables` are its
    weights, held frozen under `REPRESENTATION`. `layer` names the
    submodule of the model whose output is aligned (`dit_block_7` for
    REPA's depth 8 on `simple_dit`), a `[B, N, width]` token sequence in
    raster order over the same grid as the encoder's patches.

    `weight` is REPA's `proj_coeff`. `projector` "mlp" of `width` is REPA's;
    "conv" of `kernel_size` with `spatial_norm` gamma (0.6 in iREPA's
    training script) is iREPA's, which z-scores each feature over the
    tokens after subtracting gamma times its spatial mean. The pixel
    statistics default to ImageNet's, DINOv2's; `resolution` None feeds the
    encoder the data's size, and REPA's DINOv2 reads 224 pixels of a 256
    image (`resolution=224`). The loss adds `weight / 2` times the
    alignment, since Dew's L2 halves the denoising error REPA adds it to.
    """

    encoder: nn.Module
    variables: Variables
    layer: str
    weight: float = 0.5
    projector: Literal["mlp", "conv"] = "mlp"
    width: int = 2048
    kernel_size: int = 3
    spatial_norm: float | None = None
    resolution: int | None = None
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def __post_init__(self):
        if self.projector not in ("mlp", "conv"):
            raise ValueError(f"projector is mlp or conv, not {self.projector!r}")

    def targets(self, variables: Variables, images: jax.Array) -> jax.Array:
        """The encoder's `[B, N, D]` features of `images` in [-1, 1], spatially
        normalized under iREPA, with no gradient."""
        pixels = (images.astype(jnp.float32) + 1) / 2
        pixels = (pixels - jnp.asarray(self.mean)) / jnp.asarray(self.std)
        if self.resolution is not None and self.resolution != pixels.shape[1]:
            pixels = torch_bicubic_resize(pixels, self.resolution, self.resolution)
        features = self.encoder.apply(variables, pixels)
        if not isinstance(features, jax.Array):
            raise TypeError("a representation encoder must return one array of patch features")
        features = features.reshape(features.shape[0], -1, features.shape[-1]).astype(jnp.float32)
        if self.spatial_norm is not None:
            features = spatial_zscore(features, self.spatial_norm)
        return jax.lax.stop_gradient(features)

    def module(self, features: int) -> Projector:
        return Projector(self.projector, features, self.width, self.kernel_size)

    def init(self, key, width: int, count: int, features: int) -> Variables:
        """The projector's variables for `count` hidden tokens of `width`."""
        return self.module(features).init(key, jnp.zeros((1, count, width)))

    def loss(self, projector: Variables, hidden: jax.Array, targets: jax.Array) -> jax.Array:
        """REPA's projection loss: -cos(target, projection) per token, the mean
        over tokens and examples."""
        if hidden.shape[:2] != targets.shape[:2]:
            raise ValueError(f"the model's {hidden.shape[1]} tokens at {self.layer!r} do not "
                             f"match the encoder's {targets.shape[1]} patches")
        projected = self.module(targets.shape[-1]).apply(projector, hidden)
        assert isinstance(projected, jax.Array)
        projected = projected.astype(jnp.float32)

        def unit(value):
            return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-12)

        return -jnp.mean(jnp.sum(unit(projected) * unit(targets), axis=-1))

    def captures(self, module: nn.Module, method: str) -> bool:
        """The `capture_intermediates` filter that keeps `layer`'s output."""
        return method == "__call__" and module.name == self.layer


__all__ = ["ALIGNMENT", "REPRESENTATION", "Alignment", "Projector", "spatial_zscore"]
