"""Bounded RGB image decoding, shared by every source and image step."""

from __future__ import annotations

import io

import numpy as np

from dew.data.images import as_rgb


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
