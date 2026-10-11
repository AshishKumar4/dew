"""Image datasets read from TFDS, the Hugging Face hub and arrayrecord shards.

Every image dataset resizes, augments and captions its records with the same
transform. The datasets differ in where the records come from and how one
record is read, which a subclass defines in its `source` and `record`
methods. Each record comes out as
`{"image": uint8 [size, size, 3], "caption": str}`, plus `"label"` when the
source has a class index. The run's condition reads the captions through
`load(tokenize=)`, so the dataset keeps only the text and the encoder
decides which tokens it becomes. cv2, tensorflow_datasets and HF datasets
are imported only when used, so `import dew.data` loads none of them.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import os
import struct
from typing import Literal

import numpy as np

Augmentation = Literal["none", "flip_only", "flip_jitter"]


def import_opencv() -> None:
    """Import OpenCV in the thread that opens a loader, before its readers start.

    The reader threads reach their first decode together, so each would
    otherwise make the first import of cv2 at once. An import that fails in
    one of them leaves the others the half-built module, which surfaces as
    `module 'cv2' has no attribute 'INTER_AREA'` instead of the failure
    itself. Imported here, a broken install raises its own error when the
    loader opens. The import stays out of the module's top so that reading
    text never loads OpenCV.
    """
    importlib.import_module("cv2")


def unpack_dict_of_byte_arrays(packed_data: bytes) -> dict[str, bytes]:
    """The `str -> bytes` entries of one packed arrayrecord record.

    Each entry is a uint32 key length, the utf-8 key, a uint32 value length
    and the value, in that order. `pack_dict_of_byte_arrays` writes it.
    """
    unpacked_dict = {}
    offset = 0
    while offset < len(packed_data):
        key_length = struct.unpack_from('I', packed_data, offset)[0]
        offset += struct.calcsize('I')
        key = packed_data[offset:offset+key_length].decode('utf-8')
        offset += key_length
        byte_array_length = struct.unpack_from('I', packed_data, offset)[0]
        offset += struct.calcsize('I')
        byte_array = packed_data[offset:offset+byte_array_length]
        offset += byte_array_length
        unpacked_dict[key] = byte_array
    return unpacked_dict

def pack_dict_of_byte_arrays(unpacked: dict) -> bytes:
    """Pack `unpacked`'s entries in dict order, in the layout
    `unpack_dict_of_byte_arrays` reads.

    Each entry is a uint32 key length, the utf-8 key, a uint32 value length
    and the value.
    """
    packed = bytearray()
    for key, byte_array in unpacked.items():
        encoded = key.encode('utf-8')
        packed += struct.pack('I', len(encoded))
        packed += encoded
        packed += struct.pack('I', len(byte_array))
        packed += byte_array
    return bytes(packed)


def decode_image(encoded: bytes, *, at_least: int | None = None) -> np.ndarray:
    """An encoded image as RGB uint8 (`as_rgb`), in the orientation its
    pixels are stored. EXIF orientation is ignored, as PIL's `Image.open`
    ignores it, on the reduced decodes too, which would otherwise apply it.

    With `at_least`, an opaque image is decoded at the largest 1/2, 1/4 or
    1/8 reduction that keeps both sides >= `at_least` (the DCT scale of a
    JPEG), so the resize after it still only shrinks and most of the decode
    is skipped.

    Every failure is a ValueError. PIL reads the header first, since it
    refuses a decompression bomb from the header alone where cv2 would
    decode up to 2**30 pixels, and cv2 hands back None for a half-written
    file.
    """
    import cv2
    height, width, transparent = _header(encoded)
    buffer = np.frombuffer(encoded, dtype=np.uint8)
    # IMREAD_UNCHANGED keeps the alpha the colour flags drop, and ignores EXIF.
    flags = cv2.IMREAD_UNCHANGED if transparent else cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    if at_least is not None and not transparent:
        for factor, reduced in ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4),
                                (2, cv2.IMREAD_REDUCED_COLOR_2)):
            if min(height, width) // factor >= at_least:
                flags = reduced | cv2.IMREAD_IGNORE_ORIENTATION
                break
    try:
        image = cv2.imdecode(buffer, flags)
    except cv2.error as error:
        raise ValueError(f"cv2 refused {len(encoded)} bytes of image") from error
    if image is None:
        raise ValueError(f"cv2 could not decode {len(encoded)} bytes of image")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
    return as_rgb(image)


def as_rgb(pixels: np.ndarray) -> np.ndarray:
    """Grey, grey and alpha, RGB or RGBA pixels, 8 or 16 bits, as RGB uint8.

    Grey is replicated and a 16-bit sample kept to its high byte. An image
    with transparency is composited onto white, as img2dataset does for the
    url shards the online loader streams.
    """
    if pixels.dtype == np.uint16:
        pixels = np.right_shift(pixels, 8).astype(np.uint8)
    pixels = pixels.reshape(*pixels.shape[:2], -1)
    if pixels.shape[-1] in (2, 4):
        # c * a / 255 + 255 - a, rounded to nearest in integers; the exact
        # quotient is k / 255, which is never a tie.
        alpha = pixels[..., -1:].astype(np.uint32)
        pixels = ((pixels[..., :-1] * alpha + 255 * (255 - alpha) + 127) // 255).astype(np.uint8)
    return np.ascontiguousarray(np.broadcast_to(pixels, (*pixels.shape[:2], 3)))


def _header(encoded: bytes) -> tuple[int, int, bool]:
    """(height, width, whether it carries transparency) from the header alone;
    PIL reads no pixels for this."""
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(encoded)) as header:
            width, height = header.size
            transparent = "A" in header.getbands() or "transparency" in header.info
    except Exception as error:
        raise ValueError(f"could not read the header of {len(encoded)} bytes of image") from error
    return height, width, transparent


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """`image` at `size` square; area interpolation down, cubic up."""
    import cv2
    if image.shape[:2] == (size, size):
        return image
    interpolation = cv2.INTER_AREA if max(image.shape[:2]) > size else cv2.INTER_CUBIC
    return cv2.resize(image, (size, size), interpolation=interpolation)


@dataclasses.dataclass(frozen=True)
class Augment:
    """The augmentations an augmentation mode applies.

    'flip_only' sets `flip`, 'flip_jitter' sets both `flip` and `jitter`, and
    'none' gives no Augment at all.
    """

    flip: bool
    jitter: bool


def image_augmentations(mode: Augmentation) -> Augment | None:
    """The augmentations `mode` names: flip_only (DiT style), flip_jitter,
    or none (deterministic evaluation and debugging)."""
    if mode == 'none':
        return None
    if mode == 'flip_only':
        return Augment(flip=True, jitter=False)
    if mode == 'flip_jitter':
        return Augment(flip=True, jitter=True)
    raise ValueError(f"augmentation {mode!r} is not one of none, flip_only, flip_jitter")


# ColorJitter(brightness=0.2, contrast=0.05, saturation=0.2, hue=0): the three
# factors' uniform ranges, applied in a random order per record.
_JITTER_RANGES = ((0.8, 1.2), (0.95, 1.05), (0.8, 1.2))


def augment_image(augment: Augment | None, image: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """Flips and colour-jitters `image`, seeded by the record's own rng.

    Every draw comes from grain's per-record rng, a Philox keyed by the
    record index, so a record's augmentation is the same however many
    workers, threads or processes produced its batch. The jitter is
    torchvision's float ColorJitter (`jitter_host`); uint8 pixels go through
    float32 and are rounded once at the end.
    """
    if augment is None:
        return image
    if augment.flip and rng.random() < 0.5:
        image = image[:, ::-1]
    if not augment.jitter:
        return np.ascontiguousarray(image)

    from .image_augmentation import jitter_host

    factors = np.asarray([rng.uniform(low, high) for low, high in _JITTER_RANGES])
    pixels = jitter_host(image.astype(np.float32), factors, rng.permutation(3))
    return np.rint(pixels).astype(np.uint8)


@functools.cache
def class_names(path: str) -> tuple[str, ...]:
    """Read the class names of a labels file, one per line.

    The result is cached, so each file is read once per process.
    """
    with open(os.path.expanduser(path)) as handle:
        return tuple(line.strip() for line in handle)


def record_caption(element, columns: tuple[str, ...] = ("caption", "text")) -> str:
    """The caption a record already carries, for datasets that ship their text:
    the first of `columns` it holds."""
    for key in columns:
        if key in element:
            return element[key]
    raise KeyError(
        f"an image record needs one of the columns {list(columns)}, this one has "
        f"{sorted(element)}")


