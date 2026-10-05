#!/usr/bin/env python3
"""Write tests/fixtures/colour_jitter/torchvision.npz: torchvision's float
ColorJitter ops on fixed images, factors and every order of the three.

`torchvision.transforms.v2.functional`'s `adjust_brightness`,
`adjust_contrast` and `adjust_saturation` (torchvision 0.29), the three
ColorJitter applies for brightness, contrast and saturation with hue off,
run on float images in [0, 1], each case once in float64 and once in
float32. The factors sit at the ends of Dew's ranges and the images hold
bright pixels, so a brightened pixel leaves the range and the per-op clamp
decides it. Pixels are stored in [0, 255], channels last.

    python tools/colour_jitter_reference.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.transforms.v2 import functional

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "colour_jitter" / "torchvision.npz"
OPS = (functional.adjust_brightness, functional.adjust_contrast, functional.adjust_saturation)


def jittered(image: np.ndarray, factors: np.ndarray, order, dtype) -> np.ndarray:
    pixels = torch.as_tensor(image, dtype=dtype).permute(2, 0, 1) / 255
    for index in order:
        pixels = OPS[index](pixels, float(factors[index]))
    return (pixels.permute(1, 2, 0) * 255).numpy()


def main() -> None:
    if torchvision.__version__.split("+")[0] != "0.29.0":
        raise SystemExit(f"the fixture pins torchvision 0.29.0, got {torchvision.__version__}")
    rng = np.random.default_rng(11)
    arrays = {}
    for case, order in enumerate(itertools.permutations(range(3))):
        image = rng.integers(0, 256, (16, 16, 3)).astype(np.uint8)
        image[: 6] = rng.integers(200, 256, (6, 16, 3))
        factors = np.asarray([rng.choice([0.8, 1.2]), rng.uniform(0.95, 1.05), rng.choice([0.8, 1.2])])
        arrays[f"{case}/image"] = image
        arrays[f"{case}/factors"] = factors
        arrays[f"{case}/order"] = np.asarray(order)
        arrays[f"{case}/jittered_f64"] = jittered(image, factors, order, torch.float64)
        arrays[f"{case}/jittered"] = jittered(image, factors, order, torch.float32)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, **arrays)
    print(f"{FIXTURE}: six orders")


if __name__ == "__main__":
    main()
