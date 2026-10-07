"""Representation alignment: REPA and iREPA.

REPA (Yu et al. 2025, "Representation Alignment for Generation: Training
Diffusion Transformers Is Easier Than You Think") adds a term to the
denoising loss: the negative cosine similarity between a frozen encoder's
patch features of the clean image and a projection of the denoiser's hidden
tokens at one layer, averaged over tokens and examples. iREPA (Singh et al.
2026, "What matters for Representation Alignment: Global Information or
Spatial Structure?") projects with one convolution over the token grid where
REPA uses an MLP, and z-scores the encoder's features over space. The official code is
sihyun-yu/REPA's `loss.py` and `models/sit.py`, and End2End-Diffusion/iREPA's
`projectors.py` and `spatial_zscore`.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.nn.blocks import torch_bicubic_resize
from dew.objectives.base import Batch, Objective, Variables

ALIGNMENT = "alignment_projector"
"""The projector's key in `params`, beside the model's own modules."""

REPRESENTATION = "representation"
"""The variables collection that holds the frozen encoder's weights, as `encoders` hold theirs."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Projector(nn.Module):
    """Projects hidden tokens to `features` with REPA's `build_mlp` or iREPA's convolution.

    `build_mlp` has two hidden SiLU layers of `width`. iREPA's projector is
    one same-padded convolution over the token grid, which must be square.
    """

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
    """Return iREPA's spatial normalization of `[B, N, D]` features.

    Gamma times each feature's mean over the tokens is subtracted, and the
    result is divided by the feature's standard deviation over the tokens
    plus 1e-6. The standard deviation is torch's unbiased `Tensor.std`, as
    in the reference.
    """
    mean = jnp.mean(features, axis=1, keepdims=True)
    std = jnp.std(features, axis=1, keepdims=True, ddof=1)
    return (features - gamma * mean) / (std + 1e-6)


@dataclass(frozen=True)
class Alignment:
    """Aligns the model's hidden tokens at `layer` with a frozen encoder's features.

    `encoder` is a Flax module that maps pixels `[B, S, S, 3]` to patch
    features `[B, N, D]` or `[B, h, w, D]`, any module a record names, as a
    model is named. Its weights are the starting tree's `REPRESENTATION`
    when it holds them, else `source`'s: a transformers `Dinov2Model`
    checkpoint (`repo`, `repo@revision` or a directory), by default REPA's
    DINOv2-B/14, which a run's record pins to a commit. With `encoder` None
    the encoder is the source's own, read at `resolution` pixels. The pixels
    are resized to `resolution` (None keeps the data's size) and normalized
    by `mean` and `std`. `layer` names the submodule of the model whose
    output is aligned (`dit_block_7`, REPA's depth 8 on `simple_dit`). That
    output must be a `[B, N, width]` token sequence in raster order over the
    same grid as the encoder's patches.

    `weight` is REPA's `proj_coeff`. `projector="mlp"` with `width` is REPA's
    projector. `projector="conv"` with `kernel_size` and a `spatial_norm`
    gamma (0.6 in iREPA's training script) is iREPA's, which z-scores each
    feature over the tokens after subtracting gamma times its spatial mean.
    The pixel statistics default to ImageNet's, which DINOv2 uses; REPA's
    DINOv2 reads 224 pixels of a 256 image. The loss adds `weight / 2` times
    the alignment term, because Dew's L2 halves the denoising error that
    REPA adds the term to.
    """

    layer: str = "dit_block_7"
    encoder: nn.Module | None = None
    source: str | None = "facebook/dinov2-base"
    weight: float = 0.5
    projector: Literal["mlp", "conv"] = "mlp"
    width: int = 2048
    kernel_size: int = 3
    spatial_norm: float | None = None
    resolution: int | None = 224
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def __post_init__(self):
        if self.projector not in ("mlp", "conv"):
            raise ValueError(f"projector is mlp or conv, not {self.projector!r}")

    def network(self, held: Variables | None) -> tuple[Alignment, Variables]:
        """This alignment with its encoder, and the encoder's weights: `held`, a
        starting tree's `REPRESENTATION`, when given, else `source`'s."""
        if self.encoder is not None and held is not None:
            return self, held
        if self.source is None:
            raise ValueError(f"the alignment's encoder takes its weights from the starting tree's "
                             f"{REPRESENTATION!r}, which holds none, or from a source; name one")
        from dew.interop.pretrained import split_revision
        from dew.nn.autoencoders.rae import load_dinov2

        name, revision = split_revision(self.source)
        module, params, _ = load_dinov2(name, revision=revision,
                                        params=None if held is None else held["params"])
        if self.encoder is not None:
            module = self.encoder
        elif self.resolution is not None:
            module = module.clone(input_size=self.resolution)
        return dataclasses.replace(self, encoder=module), {"params": params}

    def targets(self, variables: Variables, images: jax.Array) -> jax.Array:
        """Return the encoder's `[B, N, D]` features of `images` in [-1, 1], with no gradient.

        Under iREPA (`spatial_norm` set) the features are spatially normalized.
        """
        dtype = jnp.promote_types(images.dtype, jnp.float32)
        pixels = (images.astype(dtype) + 1) / 2
        pixels = (pixels - jnp.asarray(self.mean)) / jnp.asarray(self.std)
        if self.resolution is not None and self.resolution != pixels.shape[1]:
            pixels = torch_bicubic_resize(pixels, self.resolution, self.resolution)
        assert self.encoder is not None, "`network` gives the alignment its encoder"
        features = self.encoder.apply(variables, pixels)
        if not isinstance(features, jax.Array):
            raise TypeError("a representation encoder must return one array of patch features")
        features = features.reshape(features.shape[0], -1, features.shape[-1]).astype(
            jnp.promote_types(features.dtype, jnp.float32))
        if self.spatial_norm is not None:
            features = spatial_zscore(features, self.spatial_norm)
        return jax.lax.stop_gradient(features)

    def module(self, features: int) -> Projector:
        return Projector(self.projector, features, self.width, self.kernel_size)

    def init(self, key, width: int, count: int, features: int) -> Variables:
        """Return the projector's variables for `count` hidden tokens of `width`."""
        return self.module(features).init(key, jnp.zeros((1, count, width)))

    def loss(self, projector: Variables, hidden: jax.Array, targets: jax.Array, batch: Batch) -> jax.Array:
        """Return REPA's projection loss, the mean of -cos(target, projection) over tokens and examples.

        The examples are `batch`'s rows (`Objective.row_mean`).

        Raises `ValueError` when the model's token count differs from the
        encoder's patch count.
        """
        if hidden.shape[:2] != targets.shape[:2]:
            raise ValueError(f"the model's {hidden.shape[1]} tokens at {self.layer!r} do not "
                             f"match the encoder's {targets.shape[1]} patches")
        projected = self.module(targets.shape[-1]).apply(projector, hidden)
        assert isinstance(projected, jax.Array)
        projected = projected.astype(jnp.promote_types(projected.dtype, jnp.float32))

        def unit(value):
            return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-12)

        return -Objective.row_mean(jnp.sum(unit(projected) * unit(targets), axis=-1), batch).mean()[0]

    def captures(self, module: nn.Module, method: str) -> bool:
        """Return whether `capture_intermediates` should keep this call: True for `layer`'s output."""
        return method == "__call__" and module.name == self.layer


__all__ = ["ALIGNMENT", "REPRESENTATION", "Alignment", "Projector", "spatial_zscore"]
