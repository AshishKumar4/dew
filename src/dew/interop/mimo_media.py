"""MiMo-V2.6's host processing of raw images, which its release states in config.json's `processor_config`
and ships no code for.

The reference is vLLM's MiMo processor (vllm/transformers_utils/processors/mimo_v2_omni.py at vLLM
d254f6a9, ported from SGLang's): an image is resized to sides that are multiples of a merged patch and
hold between the processor's least and most pixels (`smart_resize`), bilinearly with half-pixel centers
and no antialiasing (torch's `F.interpolate(mode="bilinear", align_corners=False)`), standardized by the
release's pixel mean and deviation on its 0-255 scale, repeated over its temporal patch and cut into the
tower's patch rows in its merge order; each placeholder of the prompt (`vision_start`, image tokens,
`vision_end`) becomes the image's merge blocks' tokens between its markers, the prompt's text between
them tokenized part by part. tools/mimo_reference.py --record-images holds this module to it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Unpack

import jax
import jax.numpy as jnp
import numpy as np

from dew import _records
from dew._records import JSON
from dew.nn.inputs import Media, pad_token_rows

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from dew.interop.processors import ProcessorCall

PIXEL_MEAN = (123.675, 116.28, 103.53)
"""The release's pixel mean, per channel, on the 0-255 scale (vLLM's `_PIXEL_MEAN`)."""
PIXEL_STD = (58.395, 57.12, 57.375)
"""The release's pixel deviation, per channel, on the 0-255 scale (vLLM's `_PIXEL_STD`)."""


def smart_resize(height: int, width: int, factor: int, least: int, most: int) -> tuple[int, int]:
    """The sides an image of `height` x `width` is resized to: each a multiple of `factor` (a merged
    patch), nearest its own, the image's aspect kept as their product is brought within `least` and
    `most` pixels; a side below `factor` scaled up to it first, and an aspect over 200 refused (vLLM's
    `_smart_resize`)."""
    if min(height, width) < factor:
        if height < width:
            height, width = factor, int(width * factor / height)
        else:
            width, height = factor, int(height * factor / width)
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"an image's aspect may be at most 200, got {height}x{width}")
    rows, columns = round(height / factor) * factor, round(width / factor) * factor
    if rows * columns > most:
        beta = math.sqrt(height * width / most)
        rows = math.floor(height / beta / factor) * factor
        columns = math.floor(width / beta / factor) * factor
    elif rows * columns < least:
        beta = math.sqrt(least / (height * width))
        rows, columns = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return int(rows), int(columns)


def image_patches(pixels: jax.Array, size: tuple[int, int], *, patch: int, merge: int,
                  temporal: int) -> jax.Array:
    """`pixels`, an `[height, width, 3]` image on the 0-255 scale in the dtype it is computed in, resized
    to `size` bilinearly with half-pixel centers and no antialiasing, standardized, repeated over
    `temporal` frames and cut into `[patches, 3 * temporal * patch * patch]` rows, the patches of a merge
    block adjacent (vLLM's `_transform_single` and `_flatten_visual`)."""
    channels = pixels.shape[-1]
    resized = jax.image.resize(jnp.moveaxis(pixels, -1, 0), (channels, *size), method="bilinear",
                               antialias=False)
    mean, std = (jnp.asarray(values, pixels.dtype)[:, None, None] for values in (PIXEL_MEAN, PIXEL_STD))
    frames = jnp.broadcast_to((resized - mean) / std, (temporal, channels, *size))
    rows, columns = size[0] // patch, size[1] // patch
    blocks = frames.reshape(1, temporal, channels, rows // merge, merge, patch, columns // merge, merge,
                            patch)
    return blocks.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(rows * columns,
                                                               channels * temporal * patch ** 2)


class MiMoProcessor:
    """MiMo-V2.6's processor of text and raw images (`dew.interop.processors.HostProcessor`), from its
    tokenizer and its config's `processor_config` and `vision_config`, as vLLM's MiMo processor builds
    one (`from_hf_config`): the least and most pixels an image keeps default to 4 and 4096 merged
    patches. Videos and audio stay the release's processed fields' (`Processor.from_hf`)."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: Mapping[str, object]) -> None:
        self.tokenizer = tokenizer
        held = _records.record(config.get("processor_config") or {}, "processor_config")
        vision = _records.record(config.get("vision_config") or {}, "vision_config")
        self.patch = _records.integer(vision.get("patch_size", 14), "patch_size")
        self.merge = _records.integer(vision.get("spatial_merge_size", 2), "spatial_merge_size")
        self.temporal = _records.integer(vision.get("temporal_patch_size", 2), "temporal_patch_size")
        unit = self.patch * self.merge
        self.least = _records.integer(held.get("image_min_pixels") or 4 * unit * unit, "image_min_pixels")
        self.most = _records.integer(held.get("image_max_pixels") or 4096 * unit * unit, "image_max_pixels")
        self.ids = {name: _records.integer(held.get(f"{name}_token_id", config.get(f"{name}_token_id")),
                                           f"{name}_token_id")
                    for name in ("vision_start", "vision_end", "image")}

    def sizes(self, image: np.ndarray | jax.Array) -> tuple[int, int]:
        """The sides `image` is resized to (`smart_resize`)."""
        return smart_resize(image.shape[0], image.shape[1], self.patch * self.merge, self.least, self.most)

    def patch_rows(self, image: jax.Array) -> jax.Array:
        """`image`'s patch rows (`image_patches`), computed in its own floating dtype."""
        return image_patches(image, self.sizes(image), patch=self.patch, merge=self.merge,
                             temporal=self.temporal)

    def patches(self, image: np.ndarray) -> np.ndarray:
        """`image`'s patch rows, computed in float32 (`patch_rows`)."""
        return np.asarray(self.patch_rows(jnp.asarray(image, jnp.float32)))

    def token(self, name: str) -> str:
        """The tokenizer's text of the processor's `name` token id (`vision_start`, `vision_end`, `image`)."""
        held = self.tokenizer.convert_ids_to_tokens(self.ids[name])
        if not isinstance(held, str):
            raise TypeError(f"the tokenizer reads {name}_token_id {self.ids[name]} as {held!r}, not one "
                            "token")
        return held

    def row(self, text: str, images: Sequence[np.ndarray]) -> tuple[list[int], list[np.ndarray],
                                                                     list[tuple[int, int, int]]]:
        """`text`'s tokens with each placeholder an image of `images` in order, the images' patch rows
        and grids: the text between placeholders tokenized part by part, as the reference encodes it."""
        start, end, image = (self.token(name) for name in ("vision_start", "vision_end", "image"))
        tokens: list[int] = []
        patches, grids = [], []
        remaining = iter(images)
        parts = text.split(start)
        for index, part in enumerate(parts):
            if index:
                inner, _, part = part.partition(end)
                if inner.replace(image, "") or not inner:
                    raise ValueError(f"a placeholder holds {image} alone between {start} and {end}")
                held = next(remaining, None)
                if held is None:
                    raise ValueError("the prompt holds more image placeholders than images")
                rows, columns = self.sizes(held)
                grid = (1, rows // self.patch, columns // self.patch)
                patches.append(self.patches(held))
                grids.append(grid)
                blocks = math.prod(grid) // self.merge ** 2
                tokens += [self.ids["vision_start"], *[self.ids["image"]] * blocks, self.ids["vision_end"]]
            if part:
                tokens += list(self.tokenizer.encode(part))
        return tokens, patches, grids

    def __call__(self, *, images: Media | None = None,
                 **kwargs: Unpack[ProcessorCall]) -> dict[str, np.ndarray]:
        """The fields of the call's `text` rows and `images`, each row's placeholders taking the next
        images in order: `input_ids` and `attention_mask` padded as the tokenizer pads
        (`pad_token_rows`), `pixel_values` and `image_grid_thw` where any image is given. Videos and
        audio are refused: the model takes the release's processed fields of them (`Processor.from_hf`)."""
        text = kwargs.get("text")
        if text is None or {"audio", "videos", "video_metadata"} & kwargs.keys() \
                or kwargs.get("return_tensors", "np") != "np":
            raise ValueError("MiMo-V2.6's processor here reads text and raw images into NumPy fields; "
                             "hand the model its processed video and audio fields (`from_hf`)")
        given = [_rgb(image) for image in ([] if images is None else
                                          images if isinstance(images, Sequence) else [images])]
        rows, patches, grids = [], [], []
        taken = 0
        for text_row in [text] if isinstance(text, str) else list(text):
            count = text_row.count(self.token("vision_start"))
            tokens, row_patches, row_grids = self.row(text_row, given[taken:taken + count])
            taken += count
            rows.append(tokens)
            patches += row_patches
            grids += row_grids
        if taken != len(given):
            raise ValueError(f"the prompts hold {taken} image placeholders for {len(given)} images")
        side = "left" if self.tokenizer.padding_side == "left" else "right"
        pad = _records.integer(self.tokenizer.pad_token_id or 0, "pad_token_id")
        padded, valid = pad_token_rows(rows, pad_id=pad, padding_side=side)
        fields = {"input_ids": padded, **valid}
        if patches:
            fields.update(pixel_values=np.concatenate(patches).astype(np.float32),
                          image_grid_thw=np.asarray(grids, np.int64))
        return fields

    def apply_chat_template(self, conversation: Sequence[Mapping[str, object]],
                            **kwargs: JSON) -> str | Sequence[int] | Mapping[str, object]:
        from dew.interop.processors import hosts

        tokenizer = self.tokenizer
        if not hosts(tokenizer):
            raise TypeError("MiMo-V2.6's tokenizer lacks a host processor operation")
        return tokenizer.apply_chat_template(conversation, **kwargs)

    def batch_decode(self, sequences: list[list[int]], *, skip_special_tokens: bool) -> list[str]:
        return self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)

    def save_pretrained(self, save_directory: str) -> None:
        # The release ships no processor: its tokenizer's files, and config.json's processor_config.
        self.tokenizer.save_pretrained(save_directory)


def _rgb(image: object) -> np.ndarray:
    """An image as `[height, width, 3]` uint8, as the reference converts one (`Image.convert("RGB")`)."""
    from PIL import Image

    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    held = np.asarray(image)
    if held.ndim != 3 or held.shape[-1] != 3 or held.dtype != np.uint8:
        raise ValueError(f"an image is [height, width, 3] uint8 or a PIL image, got {held.dtype} "
                         f"{held.shape}")
    return held
