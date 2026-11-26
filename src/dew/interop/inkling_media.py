"""Inkling 2's host processing of raw images, which its release names (processor_config.json's
`Inkling2Processor`) but no transformers Dew supports ships.

The reference is transformers' own (models/inkling2/image_processing_inkling2.py and processing_inkling2.py at
huggingface/transformers 3a96649): an image is never resized; an RGBA one is laid over an 8-pixel
checkerboard of 204 and 153 in integer arithmetic, any other read as RGB; it is padded at its bottom and
right to whole merged tokens (`native_patch_size * adapter_merge_size` pixels a side) with -1 on the 0-255
scale, standardized as `(x - 127.5) / 127.5` in float32, and cut into one `[side, side, 3]` block a merged
token in raster order, its grid `[1, rows, columns]` counted in merged tokens; each image placeholder of the
prompt becomes as many placeholders as the image's merged tokens, the prompt then tokenized whole.
tools/inkling2_fixture.py --processor records the reference's fields this module is held to.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Unpack

import numpy as np

from dew import _records
from dew._records import JSON
from dew.nn.inputs import pad_token_rows

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from dew.interop.processors import HostMedia, ProcessorCall

CHECKER = 8
"""The side of the checkerboard an RGBA image is laid over, in pixels."""
CHECKER_SHADES = (204, 153)
"""The checkerboard's shades on the 0-255 scale, its first square's first."""
PAD = -1.0
"""The value an image is padded with to whole merged tokens, on the 0-255 scale, before its standardizing."""
CENTER = np.float32(127.5)
"""The standardizing's center and scale on the 0-255 scale: the release's mean and deviation, 0.5, over its
rescale factor 1 / 255, fused as the reference fuses them."""


def rgb(image: object) -> np.ndarray:
    """An image as `[height, width, 3]` uint8, as the reference reads one: a path, a URL, a base64 string
    or data URI, or encoded bytes decoded in its own mode (`opened`); a PIL image turned upright by its EXIF
    orientation, as transformers' `load_image` turns one, and converted to RGBA where it holds transparency
    and else to RGB; an RGBA array laid over the checkerboard (`composited`); an RGB array as it is. Its
    alpha rule and its orientation hold however the image is given: transformers' `load_image` converts a
    path's or a URL's to RGB, its transparent pixels black, before any processor reads it."""
    from PIL import Image, ImageOps

    if isinstance(image, str | Path | bytes | bytearray):
        image = opened(image)
    if isinstance(image, Image.Image):
        upright = ImageOps.exif_transpose(image)
        image = np.asarray(upright.convert("RGBA" if upright.has_transparency_data else "RGB"))
    held = np.asarray(image)
    if held.ndim != 3 or held.shape[-1] not in (3, 4) or held.dtype != np.uint8:
        raise ValueError(f"an image is [height, width, 3 or 4] uint8 or a PIL image, got {held.dtype} "
                         f"{held.shape}")
    return composited(held) if held.shape[-1] == 4 else held


URL_BYTES = 64 * 2**20
"""The most bytes an image a URL names is read to (`opened`), as Dew's url fetch reads one (`Fetch`)."""
URL_SECONDS = 60.0
"""The time an image a URL names is read within, from connecting to its last byte (`bounded_read`)."""


def opened(image: str | Path | bytes | bytearray):
    """An encoded image as PIL opens it, in its own mode, from each form transformers' `load_image` reads:
    a URL's bytes fetched within a deadline and a size (`URL_SECONDS`, `URL_BYTES`, `bounded_read`), a
    file's path, a base64 string or data URI
    (`data:image/...;base64,...`) decoded, and bytes as given."""
    import base64
    import binascii
    import io
    import os

    from PIL import Image

    if isinstance(image, str) and image.startswith(("http://", "https://")):
        from dew.data.sources.urls import bounded_read

        status, payload = bounded_read(image, URL_BYTES + 1, URL_SECONDS)
        if not 200 <= status < 300:
            raise ValueError(f"the image at {image} answered HTTP {status}")
        if len(payload) > URL_BYTES:
            raise ValueError(f"the image at {image} is past {URL_BYTES} bytes")
        image = payload
    elif isinstance(image, str) and (image.startswith("data:image/") or not os.path.isfile(image)):
        # Inline before any path probe (a long data URI is no path, `os.path.isfile` False where
        # `Path.is_file` raises), decoded as transformers' `base64.decodebytes` decodes one: line-wrapped
        # base64 too, any other character refused.
        encoded = "".join((image.split(",", 1)[1] if image.startswith("data:image/") else image).split())
        try:
            image = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError(f"an image string is a URL, a file's path, base64 or a data URI: {image[:64]!r}"
                             ) from error
    return Image.open(io.BytesIO(image) if isinstance(image, bytes | bytearray) else Path(image))


def composited(rgba: np.ndarray) -> np.ndarray:
    """An `[height, width, 4]` uint8 image laid over the checkerboard, each channel `(c a + b (255 - a) +
    127) // 255` in integers, its background shade `b` alternating every `CHECKER` pixels."""
    height, width, _ = rgba.shape
    squares = (np.arange(height)[:, None] // CHECKER + np.arange(width)[None, :] // CHECKER) % 2
    background = np.where(squares == 0, *CHECKER_SHADES)[..., None]
    held = rgba.astype(np.int32)
    alpha = held[..., 3:]
    return ((held[..., :3] * alpha + background * (255 - alpha) + 127) // 255).astype(np.uint8)


def merged_blocks(image: np.ndarray, side: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    """`image` `[height, width, 3]` uint8 as its merged tokens' pixel blocks `[rows * columns, side, side,
    3]` float32 in raster order and its grid `(1, rows, columns)`: padded at its bottom and right to whole
    blocks with `PAD` and standardized about `CENTER`."""
    height, width, channels = image.shape
    rows, columns = math.ceil(height / side), math.ceil(width / side)
    padded = np.full((rows * side, columns * side, channels), PAD, np.float32)
    padded[:height, :width] = image
    standardized = (padded - CENTER) / CENTER
    blocks = standardized.reshape(rows, side, columns, side, channels).transpose(0, 2, 1, 3, 4)
    return blocks.reshape(rows * columns, side, side, channels), (1, rows, columns)


class InklingProcessor:
    """Inkling 2's processor of text and raw images (`dew.interop.processors.HostProcessor`), from its
    tokenizer and its config's `vision_config` (a merged token's side, `native_patch_size *
    adapter_merge_size`) and `image_token_id`, its placeholder. Its fields are the reference's: the model
    reads its merged blocks as packed patches (`Processor._unmerged`)."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: Mapping[str, object]) -> None:
        self.tokenizer = tokenizer
        vision = _records.record(config.get("vision_config") or {}, "vision_config")
        patch = _records.integer(vision.get("native_patch_size", vision.get("patch_size", 20)), "patch_size")
        merge = _records.integer(vision.get("adapter_merge_size", vision.get("spatial_merge_size", 2)),
                                 "adapter_merge_size")
        self.side = patch * merge
        self.image_token_id = _records.integer(config.get("image_token_id", 200054), "image_token_id")
        held = tokenizer.convert_ids_to_tokens(self.image_token_id)
        if not isinstance(held, str):
            raise TypeError(f"the tokenizer reads image_token_id {self.image_token_id} as {held!r}, not one "
                            "token")
        self.image_token = held

    def __call__(self, *, images: HostMedia | None = None,
                 **kwargs: Unpack[ProcessorCall]) -> dict[str, np.ndarray]:
        """The fields of the call's `text` rows and `images`, each row's placeholders taking the next images
        in order, each placeholder made as many as its image's merged tokens: `input_ids` and
        `attention_mask` padded as the tokenizer pads (`pad_token_rows`), `pixel_values` and
        `image_grid_thw` where any image is given. Videos and audio are refused."""
        text = kwargs.get("text")
        if text is None or {"audio", "videos", "video_metadata"} & kwargs.keys() \
                or kwargs.get("return_tensors", "np") != "np":
            raise ValueError("Inkling 2's processor here reads text and raw images into NumPy fields")
        from dew.interop.processors import is_image_collection

        # A path, URL or byte buffer is one image, not a sequence of its characters (the host processor's
        # one rule, `is_image_collection`).
        given = [rgb(image) for image in ([] if images is None else images if is_image_collection(images)
                                          else [images])]
        processed = [merged_blocks(image, self.side) for image in given]
        remaining = iter(processed)
        rows = []
        for row in [text] if isinstance(text, str) else list(text):
            parts = row.split(self.image_token)
            expanded = parts[0]
            for part in parts[1:]:
                held = next(remaining, None)
                if held is None:
                    raise ValueError("the prompts hold more image placeholders than images")
                expanded += self.image_token * math.prod(held[1]) + part
            rows.append(list(self.tokenizer.encode(expanded)))
        if next(remaining, None) is not None:
            raise ValueError(f"the prompts hold fewer image placeholders than their {len(given)} images")
        from dew.interop.processors import row_padding

        pad, side = row_padding(self)
        padded, valid = pad_token_rows(rows, pad_id=pad, padding_side=side)
        fields = {"input_ids": padded, **valid}
        if processed:
            fields.update(pixel_values=np.concatenate([blocks for blocks, _ in processed]),
                          image_grid_thw=np.asarray([grid for _, grid in processed], np.int64))
        return fields

    def apply_chat_template(self, conversation: Sequence[Mapping[str, object]],
                            **kwargs: JSON) -> str | Sequence[int] | Mapping[str, object]:
        """The tokenizer's template over `conversation`, its images through this processor's own call
        (`dew.interop.processors.chat_call`)."""
        from dew.interop.processors import chat_call

        return chat_call(self, self.tokenizer, conversation, "Inkling 2", **kwargs)

    def batch_decode(self, sequences: list[list[int]], *, skip_special_tokens: bool) -> list[str]:
        return self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)

    def save_pretrained(self, save_directory: str) -> None:
        # The release's processor files beside the tokenizer's are its own; the tokenizer's here.
        self.tokenizer.save_pretrained(save_directory)
