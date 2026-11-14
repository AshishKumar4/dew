"""Image preprocessing and image conditions for latent diffusion: a CLIP image transform, a safety check's
inputs, and an inpainting run's masked latents; the text conditioning that binds a released pipeline's
encoders is dew.interop's (`dew.interop.conditioners`)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from dew.artifacts import pixel_levels
from dew.nn.blocks import torch_nearest_resize
from dew.nn.safety import CLIPSafetyHead


def latent_image_conditions(autoencoder, params, pixels, mask, key, noise=None):
    """Turns normalized pixels and a binary pixel mask into native UNet inputs.

    Returns mask [B,h,w,1] and masked_image [B,h,w,C] under the model keyword
    names. Spatial inputs stay present in both guidance branches. The mask
    shrinks to the latent grid by torch's nearest rule, as diffusers'
    inpainting pipelines shrink it (`prepare_mask_latents`).
    """
    mask = jnp.asarray(mask, jnp.float32)
    if mask.shape != (*pixels.shape[:-1], 1):
        raise ValueError("Image and mask geometry must match, with one mask channel")
    mask = (mask >= 0.5).astype(jnp.float32)
    latent = autoencoder.encode(params, pixels * (mask < 0.5), key, noise=noise)
    mask = torch_nearest_resize(mask, *latent.shape[1:3])
    return {"mask": mask, "masked_image": latent}





@lru_cache(maxsize=32)
def _cubic_weights(source: int, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Compact Pillow bicubic taps, with its 22-bit integer coefficients."""
    scale = source / target
    support = 2.0 * max(scale, 1.0)
    taps = int(np.ceil(2 * support)) + 1
    indices = np.zeros((target, taps), np.int32)
    weights = np.zeros((target, taps), np.int32)
    for row in range(target):
        center = (row + 0.5) * scale
        left = max(int(center - support + 0.5), 0)
        right = min(int(center + support + 0.5), source)
        positions = (np.arange(left, right) - center + 0.5) / max(scale, 1.0)
        x = np.abs(positions)
        kernel = np.where(x < 1, (1.5 * x - 2.5) * x * x + 1,
                          np.where(x < 2, ((-0.5 * x + 2.5) * x - 4) * x + 2, 0.0))
        normalized = kernel / kernel.sum() * (1 << 22)
        indices[row, :right - left] = np.arange(left, right)
        weights[row, : right - left] = np.trunc(normalized + np.where(normalized >= 0, 0.5, -0.5)).astype(
            np.int32
        )
    return indices, weights


@dataclass(frozen=True)
class CLIPImageTransform:
    """Runs the published CLIP preprocessing as JAX arithmetic over uint8 NHWC
    pixels, integer levels in any dtype.

    The reference is the PIL image processor: `CLIPImageProcessor` as
    transformers 4 ran it, which the safety fixtures were generated with
    (tools/diffusers_pipeline_reference.py), and which transformers 5 ships
    as `CLIPImageProcessorPil`. Transformers 5's `CLIPImageProcessor`
    resamples through torchvision instead and lands up to one uint8 level
    away.
    """
    size: int | tuple[int, int]
    crop: tuple[int, int]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    rescale: float = 1 / 255
    resize: bool = True
    center_crop: bool = True
    normalize: bool = True

    @classmethod
    def from_config(cls, config):
        """The transform a `preprocessor_config.json` states, each field it leaves out the PIL processor's
        default: OpenAI's mean and std as transformers states them (`transformers.utils.constants`, a
        package the pipeline's tokenizers already load); a 224 shortest edge and center crop, bicubic
        resampling (3) and a 1/255 rescale as `CLIPImageProcessorPil` declares them (transformers 5.17,
        models/clip/image_processing_pil_clip.py and `BaseImageProcessor.rescale_factor`), whose module
        this path does not import (it would load transformers' image processing for four constants)."""
        from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

        if config.get("resample", 3) != 3:
            raise ValueError("Only bicubic CLIP preprocessing is implemented")
        size = config.get("size", {"shortest_edge": 224})
        if isinstance(size, dict):
            size = size["shortest_edge"] if "shortest_edge" in size else (size["height"], size["width"])
        crop = config.get("crop_size", {"height": 224, "width": 224})
        crop = (crop, crop) if isinstance(crop, int) else (crop["height"], crop["width"])
        return cls(
            size,
            crop,
            tuple(config.get("image_mean", OPENAI_CLIP_MEAN)),
            tuple(config.get("image_std", OPENAI_CLIP_STD)),
            config.get("rescale_factor", cls.rescale) if config.get("do_rescale", True) else 1.0,
            config.get("do_resize", True),
            config.get("do_center_crop", True),
            config.get("do_normalize", True),
        )

    def __call__(self, pixels):
        pixels = jnp.asarray(pixels, jnp.float32)
        _, height, width, _ = pixels.shape
        if self.resize:
            if isinstance(self.size, int):
                # transformers' get_resize_output_image_size: the shorter side
                # takes the size exactly, the longer int(size * long / short).
                if width <= height:
                    target = (int(self.size * height / width), self.size)
                else:
                    target = (self.size, int(self.size * width / height))
            else:
                target = self.size
            row_indices, rows = (jnp.asarray(value) for value in _cubic_weights(height, target[0]))
            column_indices, columns = (jnp.asarray(value) for value in _cubic_weights(width, target[1]))
            # Each integer pass rounds and clamps before the next pass, as
            # Pillow does. Sparse taps avoid a dense HxW resampling matrix.
            pixels = pixels.astype(jnp.int32)
            horizontal = jnp.sum(
                jnp.take(pixels, column_indices, axis=2) * columns[None, None, :, :, None], axis=3
            )
            pixels = jnp.clip((horizontal + (1 << 21)) >> 22, 0, 255)
            vertical = jnp.sum(jnp.take(pixels, row_indices, axis=1) * rows[None, :, :, None, None], axis=2)
            pixels = jnp.clip((vertical + (1 << 21)) >> 22, 0, 255).astype(jnp.float32)
        if self.center_crop:
            crop_h, crop_w = self.crop
            pad_h, pad_w = max(0, crop_h - pixels.shape[1]), max(0, crop_w - pixels.shape[2])
            if pad_h or pad_w:
                # transformers' center_crop puts the odd pad row on top and the
                # odd pad column on the left, ceil((crop - side) / 2).
                pixels = jnp.pad(pixels, ((0, 0), (pad_h - pad_h // 2, pad_h // 2),
                                          (pad_w - pad_w // 2, pad_w // 2), (0, 0)))
            top, left = (pixels.shape[1] - crop_h) // 2, (pixels.shape[2] - crop_w) // 2
            pixels = pixels[:, top:top + crop_h, left:left + crop_w]
        # transformers rescales in float64 and rounds once to float32 (`image_transforms.rescale`); over
        # integer pixels that is a table of the 256 levels, rounded on the host. A float32 product with
        # the rounded factor rounds twice, half again as far from the exact levels.
        levels = jnp.asarray(np.arange(256, dtype=np.float64) * self.rescale, jnp.float32)
        pixels = jnp.take(levels, pixels.astype(jnp.int32))
        if self.normalize:
            pixels = (pixels - jnp.asarray(self.mean, jnp.float32)) / jnp.asarray(self.std, jnp.float32)
        return pixels.transpose(0, 3, 1, 2)


@dataclass(frozen=True, eq=False)
class ImageSafety:
    """The checkpoint's frozen safety head over decoded images, under jit."""
    model: CLIPSafetyHead
    transform: CLIPImageTransform

    def __call__(self, variables, images):
        # The levels its preprocessing reads, as every decoded image is converted (`pixel_levels`).
        pixels = pixel_levels(images, jnp)
        flagged = self.model.apply({"params": variables["encoders"]["safety"]}, self.transform(pixels))
        assert isinstance(flagged, jax.Array)
        return jnp.where(flagged[:, None, None, None], -jnp.ones_like(images), images)
