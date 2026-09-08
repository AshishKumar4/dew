"""Native CLIP conditioning and image preprocessing for latent diffusion."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from collections.abc import Mapping
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from transformers import CLIPTokenizer

from dew.inputs.encoders import ConditionEncoder
from dew.nn.backbones.unet_condition import DenoisingCondition
from dew.nn.text_encoders import CLIPTextTransformer
from dew.nn.safety import CLIPSafetyHead
from dew.objectives.base import Variables
from dew.registry import dtype_name, encoders

def latent_image_conditions(autoencoder, params, pixels, mask, key):
    """Normalized image pixels and a binary pixel mask to native UNet inputs.

    Returns mask [B,h,w,1] and masked_image [B,h,w,C] under the model
    keyword names. Spatial inputs stay present in both guidance branches.
    """
    if autoencoder is None:
        raise ValueError("Masked-image conditioning requires an autoencoder")
    mask = jnp.asarray(mask, jnp.float32)
    if mask.shape != (*pixels.shape[:-1], 1):
        raise ValueError("Image and mask geometry must match, with one mask channel")
    mask = (mask >= 0.5).astype(jnp.float32)
    latent = autoencoder.encode(params, pixels * (mask < 0.5), key)
    mask = jax.image.resize(mask, (*latent.shape[:-1], 1), method="nearest")
    return {"mask": mask, "masked_image": latent}



class _TextFeatures(NamedTuple):
    last: jax.Array
    penultimate: jax.Array
    pooled: jax.Array


def _text_features(tower, ids):
    hidden = tower.token_embedding(ids) + tower.position_embedding(jnp.arange(ids.shape[1]))
    penultimate = hidden
    for layer in tower.layers:
        penultimate = hidden
        hidden = layer(hidden)
    hidden = tower.final_layer_norm(hidden)
    index = jnp.argmax(ids if tower.eos_token_id == 2 else ids == tower.eos_token_id, axis=-1)
    return _TextFeatures(hidden, penultimate, hidden[jnp.arange(ids.shape[0]), index])


@encoders("clip_diffusion")
@dataclass(eq=False)
class CLIPConditioner(ConditionEncoder[str | Mapping[str, object]]):
    towers: tuple[CLIPTextTransformer, ...]
    tokenizers: tuple[CLIPTokenizer, ...]
    names: tuple[str, ...]
    params: Variables
    checkpoint: str
    height: int
    width: int
    pooled: bool = False
    aesthetics: bool = False

    @classmethod
    def from_pretrained(cls, checkpoint: str, **kwargs):
        from dew.interop.pretrained import load_pretrained
        source = load_pretrained(checkpoint, **kwargs)
        if source.inputs is None:
            raise TypeError("This checkpoint has no image conditioning specification")
        return source.inputs.conditions["conditioning"].encoder

    def tokenize(self, data: Sequence[str | Mapping[str, object]], second=None):
        rows, secondary, zero, negative = [], [], [], []
        for item in data:
            record = {"text": item} if isinstance(item, str) else item
            text = record.get("text", "")
            if not isinstance(text, str):
                raise ValueError("A text-conditioning record needs string text")
            rows.append(text)
            secondary.append(record.get("second", text))
            zero.append(bool(record.get("zero", False)))
            negative.append(bool(record.get("negative", False)))
        ids = []
        for index, tokenizer in enumerate(self.tokenizers):
            text = (secondary if second is None else second) if len(self.towers) == 2 and index == 1 else rows
            ids.append(tokenizer(list(text), padding="max_length", max_length=tokenizer.model_max_length,
                                 truncation=True, return_tensors="np").input_ids)
        return {"input_ids": np.stack(ids, axis=1) if self.pooled else ids[0],
                "zero_condition": np.asarray(zero, bool), "negative": np.asarray(negative, bool)}

    def time_ids(self, count, dtype, *, original_size=None, crops_coords_top_left=(0, 0),
                 target_size=None, aesthetic_score=6.0):
        size = (self.height, self.width)
        values = (*(original_size or size), *crops_coords_top_left,
                  *((aesthetic_score,) if self.aesthetics else (target_size or size)))
        return jnp.broadcast_to(jnp.asarray(values, dtype), (count, len(values)))

    def encode(self, params, tokens) -> DenoisingCondition:
        ids = tokens["input_ids"]
        outputs = []
        for index, (name, tower) in enumerate(zip(self.names, self.towers)):
            output = tower.apply({"params": params[name]["text_model"]},
                                 ids[:, index] if self.pooled else ids, method=_text_features)
            assert isinstance(output, _TextFeatures)
            outputs.append(output)
        hidden = jnp.concatenate([value.penultimate for value in outputs], axis=-1) if self.pooled else outputs[0].last
        zero = jnp.asarray(tokens["zero_condition"])
        hidden = jnp.where(zero[:, None, None], jnp.zeros_like(hidden), hidden)
        if not self.pooled:
            return DenoisingCondition(hidden)
        pooled = outputs[-1].pooled @ params[self.names[-1]]["text_projection"]["kernel"]
        pooled = jnp.where(zero[:, None], jnp.zeros_like(pooled), pooled)
        time_ids = tokens.get("time_ids")
        if time_ids is None:
            time_ids = self.time_ids(ids.shape[0], hidden.dtype)
            if self.aesthetics:
                time_ids = time_ids.at[:, -1].set(jnp.where(tokens["negative"], 2.5, 6.0))
        return DenoisingCondition(hidden, pooled, time_ids)

    def captions(self, tokens):
        ids = tokens["input_ids"][:, 0] if self.pooled else tokens["input_ids"]
        return tuple(self.tokenizers[0].batch_decode(np.asarray(ids), skip_special_tokens=True))

    def to_json(self):
        return {"checkpoint": self.checkpoint, "dtype": dtype_name(self.towers[0].dtype)}


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
        weights[row, :right - left] = np.trunc(normalized + np.where(normalized >= 0, 0.5, -0.5)).astype(np.int32)
    return indices, weights


@dataclass(frozen=True)
class CLIPImageTransform:
    """Published CLIP preprocessing as JAX arithmetic over uint8 NHWC pixels."""
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
        if config.get("resample", 3) != 3:
            raise ValueError("Only bicubic CLIP preprocessing is implemented")
        size = config.get("size", {"shortest_edge": 224})
        if isinstance(size, dict):
            size = size["shortest_edge"] if "shortest_edge" in size else (size["height"], size["width"])
        crop = config.get("crop_size", {"height": 224, "width": 224})
        crop = (crop, crop) if isinstance(crop, int) else (crop["height"], crop["width"])
        return cls(size, crop, tuple(config.get("image_mean", (0.48145466, 0.4578275, 0.40821073))),
                   tuple(config.get("image_std", (0.26862954, 0.26130258, 0.27577711))),
                   config.get("rescale_factor", 1 / 255) if config.get("do_rescale", True) else 1.0,
                   config.get("do_resize", True), config.get("do_center_crop", True), config.get("do_normalize", True))

    def __call__(self, pixels):
        pixels = jnp.asarray(pixels, jnp.float32)
        batch, height, width, _ = pixels.shape
        if self.resize:
            if isinstance(self.size, int):
                scale = self.size / min(height, width)
                target = (int(height * scale), int(width * scale))
            else:
                target = self.size
            row_indices, rows = (jnp.asarray(value) for value in _cubic_weights(height, target[0]))
            column_indices, columns = (jnp.asarray(value) for value in _cubic_weights(width, target[1]))
            # Each integer pass rounds and clamps before the next pass, as
            # Pillow does. Sparse taps avoid a dense HxW resampling matrix.
            pixels = pixels.astype(jnp.int32)
            horizontal = jnp.sum(jnp.take(pixels, column_indices, axis=2) * columns[None, None, :, :, None], axis=3)
            pixels = jnp.clip((horizontal + (1 << 21)) >> 22, 0, 255)
            vertical = jnp.sum(jnp.take(pixels, row_indices, axis=1) * rows[None, :, :, None, None], axis=2)
            pixels = jnp.clip((vertical + (1 << 21)) >> 22, 0, 255).astype(jnp.float32)
        if self.center_crop:
            crop_h, crop_w = self.crop
            pad_h, pad_w = max(0, crop_h - pixels.shape[1]), max(0, crop_w - pixels.shape[2])
            if pad_h or pad_w:
                pixels = jnp.pad(pixels, ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)))
            top, left = (pixels.shape[1] - crop_h) // 2, (pixels.shape[2] - crop_w) // 2
            pixels = pixels[:, top:top + crop_h, left:left + crop_w]
        pixels = pixels * self.rescale
        if self.normalize:
            pixels = (pixels - jnp.asarray(self.mean, jnp.float32)) / jnp.asarray(self.std, jnp.float32)
        return pixels.transpose(0, 3, 1, 2)


@dataclass(frozen=True, eq=False)
class ImageSafety:
    """The checkpoint's frozen safety head over decoded images, under jit."""
    model: CLIPSafetyHead
    transform: CLIPImageTransform

    def __call__(self, variables, images):
        pixels = jnp.clip(jnp.round((images + 1) * 127.5), 0, 255)
        flagged = self.model.apply({"params": variables["encoders"]["safety"]}, self.transform(pixels))
        assert isinstance(flagged, jax.Array)
        return jnp.where(flagged[:, None, None, None], -jnp.ones_like(images), images)
