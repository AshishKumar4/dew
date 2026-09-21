"""Native CLIP conditioning and image preprocessing for latent diffusion."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from transformers import CLIPTokenizer, PreTrainedTokenizerBase

from dew.diffusion.process import DenoisingCondition
from dew.inputs.encoders import ConditionEncoder
from dew.nn.safety import CLIPSafetyHead
from dew.nn.text_encoders import CLIPTextTransformer, T5EncoderTransformer
from dew.objectives.base import Variables
from dew.registry import dtype_name, encoders


def _prompt(record: Mapping[str, object], key: str, default: str) -> str:
    """One text slot of a conditioning record."""
    text = record.get(key, default)
    if not isinstance(text, str):
        raise ValueError(f"A text-conditioning record's {key} must be a string")
    return text


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


Composition = Literal["clip", "clip_pooled", "sd3", "flux"]
"""How a checkpoint family composes its text towers into one conditioning.

- `clip`: one CLIP tower's last hidden states, which is Stable Diffusion 1
  and 2.
- `clip_pooled`: two towers' penultimate states concatenated along the width,
  the second tower's projected pooled vector and the size/crop time ids,
  which is SDXL and its refiner.
- `sd3`: the same width concatenation padded out to the T5 width, with the T5
  tower's final states concatenated along the sequence, and BOTH towers'
  projected pooled vectors concatenated as the pooled one.
- `flux`: the T5 tower's final states alone, with the first tower's pooled
  vector unprojected and no CLIP tokens in the sequence.
"""


@dataclass(frozen=True, eq=False)
class T5Segment:
    """The T5 tower a family reads beside its CLIP ones: the tower, its
    tokenizer, the component name its parameters and tokenizer live under -
    an SD3 directory's third text encoder, a Flux directory's second - and
    the sequence budget its pipeline pads to."""

    tower: T5EncoderTransformer
    tokenizer: PreTrainedTokenizerBase
    name: str
    tokens: int


@encoders("diffusion_text")
@dataclass(eq=False)
class DiffusionConditioner(ConditionEncoder[str | Mapping[str, object]]):
    """The text conditioning of a published latent diffusion checkpoint.

    One encoder owns every family's composition: which towers run, which of
    their states the model reads, and how the pooled vector is built. The
    towers themselves are the native CLIP and T5 towers, called the way their
    own source pipelines call them - the SD3 and Flux pipelines pass their T5
    ids with no attention mask, which is what `T5EncoderTransformer` does
    with none, and no generic T5 default changes for it.
    """

    towers: tuple[CLIPTextTransformer, ...]
    tokenizers: tuple[CLIPTokenizer, ...]
    names: tuple[str, ...]
    params: Variables
    checkpoint: str
    height: int
    width: int
    context_width: int
    """The width of the token sequence the denoiser reads: the UNet's
    `cross_attention_dim`, the joint transformers' `joint_attention_dim`. The
    SD3 composition pads its CLIP states out to it and writes the zero segment
    its pipeline substitutes for an absent third encoder at it."""
    composition: Composition = "clip"
    aesthetics: bool = False
    t5: T5Segment | None = None
    guidance: float | None = None
    param_dtype: str = "float32"

    @classmethod
    def from_pretrained(cls, checkpoint: str, *, dtype: str | None = "bfloat16",
                        param_dtype: str = "float32", revision: str | None = None,
                        attention_impl: str = "auto", params: Variables | None = None):
        from dew.interop.pretrained import load_diffusion_conditioner

        return load_diffusion_conditioner(checkpoint, dtype=dtype, param_dtype=param_dtype,
                                          revision=revision, attention_impl=attention_impl, params=params)

    @property
    def stacked(self) -> bool:
        """Whether the CLIP ids ride one array with a tower axis, which every
        family but plain Stable Diffusion does: the XL refiner carries a
        single tower that way too, since its pipeline still writes a tower's
        row rather than a bare batch."""
        return self.composition != "clip"

    def __post_init__(self):
        towers, t5 = len(self.towers), self.t5 is not None
        if towers != len(self.tokenizers) or towers != len(self.names):
            raise ValueError("Every text tower needs its own tokenizer and its own name")
        expected = {"clip": (1, 1), "clip_pooled": (1, 2), "sd3": (2, 2), "flux": (1, 1)}[
            self.composition]
        if not expected[0] <= towers <= expected[1]:
            raise ValueError(f"{self.composition} conditioning runs "
                             f"{'-'.join(str(bound) for bound in sorted(set(expected)))} "
                             f"CLIP towers, not {towers}")
        if self.guidance is not None and self.composition != "flux":
            raise ValueError(f"{self.composition} conditioning carries no guidance input")
        if self.composition not in ("sd3", "flux"):
            if t5:
                raise ValueError(f"{self.composition} conditioning has no T5 segment")
        elif self.composition == "flux" and not t5:
            # SD3's pipeline writes a zero segment for an absent third encoder;
            # Flux's `_get_t5_prompt_embeds` has no such path.
            raise ValueError("Flux conditioning needs its T5 encoder")

    def tokenize(self, data: Sequence[str | Mapping[str, object]]):
        """One row per item, with each text slot routed to the tower whose
        source pipeline reads it: `text` to the first CLIP tower, `second` to
        the second one, and the T5 tower's own slot, which is `third` where a
        family has two CLIP towers beside it and `second` where it has one.
        """
        rows, second, third, zero, negative, guidance = [], [], [], [], [], []
        for item in data:
            record: Mapping[str, object] = {"text": item} if isinstance(item, str) else item
            text = _prompt(record, "text", "")
            rows.append(text)
            second.append(_prompt(record, "second", text))
            third.append(_prompt(record, "third", text))
            zero.append(bool(record.get("zero", False)))
            negative.append(bool(record.get("negative", False)))
            guidance.append(self._guidance(record))
        ids = [tokenizer(second if index == 1 else rows, padding="max_length",
                         max_length=tokenizer.model_max_length, truncation=True,
                         return_tensors="np").input_ids
               for index, tokenizer in enumerate(self.tokenizers)]
        tokens = {"input_ids": np.stack(ids, axis=1) if self.stacked else ids[0],
                  "zero_condition": np.asarray(zero, bool), "negative": np.asarray(negative, bool)}
        if self.guidance is not None:
            tokens["guidance"] = np.asarray(guidance, np.float32)
        if self.t5 is not None:
            tokens["t5_input_ids"] = self.t5.tokenizer(
                third if self.composition == "sd3" else second, padding="max_length",
                max_length=self.t5.tokens, truncation=True, add_special_tokens=True,
                return_tensors="np").input_ids
        return tokens

    def _guidance(self, record: Mapping[str, object]) -> float:
        """The guidance a row is walked at: its own where it names one, and
        this checkpoint's pipeline default otherwise. A composition whose
        model reads no guidance refuses a record that names one."""
        value = record.get("guidance")
        if value is None:
            return 0.0 if self.guidance is None else self.guidance
        if self.guidance is None:
            raise ValueError("This checkpoint's model reads no guidance value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError("A record's guidance must be a finite number")
        return float(value)

    def time_ids(self, count, dtype, *, original_size=None, crops_coords_top_left=(0, 0),
                 target_size=None, aesthetic_score=6.0):
        size = (self.height, self.width)
        values = (*(original_size or size), *crops_coords_top_left,
                  *((aesthetic_score,) if self.aesthetics else (target_size or size)))
        return jnp.broadcast_to(jnp.asarray(values, dtype), (count, len(values)))

    def _clip(self, params, ids) -> list[_TextFeatures]:
        outputs = []
        for index, (name, tower) in enumerate(zip(self.names, self.towers)):
            output = tower.apply({"params": params[name]["text_model"]},
                                 ids[:, index] if self.stacked else ids, method=_text_features)
            assert isinstance(output, _TextFeatures)
            outputs.append(output)
        return outputs

    def _projected(self, params, name: str, pooled):
        return pooled @ params[name]["text_projection"]["kernel"]

    def _t5_states(self, params, tokens, rows: int, dtype) -> jax.Array:
        """The T5 segment, or the zero segment the SD3 pipeline writes when its
        third encoder is absent, which is `tokenizer_max_length` long - the
        CLIP tokenizer's window, not the T5 sequence the call asked for."""
        if self.t5 is None:
            return jnp.zeros((rows, self.tokenizers[0].model_max_length, self.context_width), dtype)
        # The SD3 and Flux pipelines call their T5 encoder with ids only.
        states = self.t5.tower.apply({"params": params[self.t5.name]},
                                     jnp.asarray(tokens["t5_input_ids"]))
        assert isinstance(states, jax.Array)
        return states

    def _zeroed(self, value, zero):
        """A dropped row's conditioning is zeros, in the shape it arrives."""
        return jnp.where(zero.reshape(zero.shape + (1,) * (value.ndim - 1)),
                         jnp.zeros_like(value), value)

    def encode(self, params, tokens) -> DenoisingCondition:
        ids = tokens["input_ids"]
        rows, zero = ids.shape[0], jnp.asarray(tokens["zero_condition"])
        outputs = self._clip(params, ids)
        if self.composition == "clip":
            return DenoisingCondition(self._zeroed(outputs[0].last, zero))
        if self.composition == "clip_pooled":
            hidden = jnp.concatenate([value.penultimate for value in outputs], axis=-1)
            pooled = self._projected(params, self.names[-1], outputs[-1].pooled)
            time_ids = tokens.get("time_ids")
            if time_ids is None:
                time_ids = self.time_ids(rows, hidden.dtype)
                if self.aesthetics:
                    time_ids = time_ids.at[:, -1].set(jnp.where(tokens["negative"], 2.5, 6.0))
            return DenoisingCondition(self._zeroed(hidden, zero),
                                      self._zeroed(pooled, zero), time_ids)
        if self.composition == "flux":
            hidden = self._t5_states(params, tokens, rows, outputs[0].pooled.dtype)
            # Flux reads the CLIP pooled vector unprojected.
            pooled = outputs[0].pooled
            guidance = (None if self.guidance is None
                        else jnp.asarray(tokens["guidance"], hidden.dtype))
            return DenoisingCondition(self._zeroed(hidden, zero),
                                      self._zeroed(pooled, zero), guidance=guidance)
        # sd3: the CLIP states padded out to the joint width, the T5 segment
        # after them along the sequence, and both projected pooled vectors.
        clip = jnp.concatenate([value.penultimate for value in outputs], axis=-1)
        padded = jnp.pad(clip, ((0, 0), (0, 0), (0, self.context_width - clip.shape[-1])))
        hidden = jnp.concatenate(
            [padded, self._t5_states(params, tokens, rows, clip.dtype)], axis=1)
        pooled = jnp.concatenate(
            [self._projected(params, name, value.pooled)
             for name, value in zip(self.names, outputs)], axis=-1)
        return DenoisingCondition(self._zeroed(hidden, zero), self._zeroed(pooled, zero))

    def captions(self, tokens):
        ids = tokens["input_ids"][:, 0] if self.stacked else tokens["input_ids"]
        return tuple(self.tokenizers[0].batch_decode(np.asarray(ids), skip_special_tokens=True))

    def to_json(self):
        return {"checkpoint": self.checkpoint, "dtype": dtype_name(self.towers[0].dtype),
                "param_dtype": self.param_dtype}


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
        _, height, width, _ = pixels.shape
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
