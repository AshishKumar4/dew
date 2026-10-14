"""The LPIPS fixtures' drawn weights, NumPy alone: `tools/lpips_reference.py`
and `tools/repae_reference.py` write fixtures from them, and
tests/test_lpips.py draws them to check those fixtures' SHA-256."""

from __future__ import annotations

import numpy as np

CONVOLUTIONS = (0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28)
"""VGG16's `features` indices that hold a convolution."""
WIDTHS = (64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512)
"""Each of those convolutions' output channels."""
CHANNELS = (64, 128, 256, 512, 512)
"""The five LPIPS stages' channels, one linear head each."""


def drawn_weights(seed: int = 0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """VGG16's `features` and the five linear heads, drawn in torchvision's
    order from `np.random.default_rng(seed)`'s float32 uniforms: each kernel
    He uniform over its fan-in, each bias uniform in [-0.01, 0.01), each
    head's weights uniform in [0, 0.2). A uniform is integer bits times
    2^-24 and the rest is float32 arithmetic, so every machine draws the
    same bits. (numpy's float32 normals take their tail through libm's
    log1pf, whose last bit differs between glibc 2.39 and 2.43.)"""
    rng = np.random.default_rng(seed)

    def uniform(shape, bound: float) -> np.ndarray:
        return (rng.random(shape, dtype=np.float32) * np.float32(2) - np.float32(1)) * np.float32(bound)

    vgg, inputs = {}, 3
    for index, width in zip(CONVOLUTIONS, WIDTHS, strict=True):
        vgg[f"features.{index}.weight"] = uniform((width, inputs, 3, 3), np.sqrt(6 / (9 * inputs)))
        vgg[f"features.{index}.bias"] = uniform(width, 0.01)
        inputs = width
    linear = {f"lin{stage}.model.1.weight": rng.random((1, width, 1, 1), dtype=np.float32) * np.float32(0.2)
              for stage, width in enumerate(CHANNELS)}
    return vgg, linear
