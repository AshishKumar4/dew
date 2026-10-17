"""Bounded RGB image decoding, shared by every source and image step."""

from __future__ import annotations

import io

import numpy as np


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
    from PIL import Image

    try:
        with Image.open(io.BytesIO(encoded)) as header:
            width, height = header.size
            transparent = "A" in header.getbands() or "transparency" in header.info
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ValueError(f"could not read the header of {len(encoded)} bytes of image") from error
    return height, width, transparent


def decode_image(encoded: bytes, *, at_least: int | None = None) -> np.ndarray:
    """Decode RGB uint8 pixels without EXIF rotation, optionally using a reduced JPEG decode."""
    import cv2

    height, width, transparent = _header(encoded)
    buffer = np.frombuffer(encoded, dtype=np.uint8)
    flags = cv2.IMREAD_UNCHANGED if transparent else cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    if at_least is not None and not transparent:
        for factor, reduced in (
            (8, cv2.IMREAD_REDUCED_COLOR_8),
            (4, cv2.IMREAD_REDUCED_COLOR_4),
            (2, cv2.IMREAD_REDUCED_COLOR_2),
        ):
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
