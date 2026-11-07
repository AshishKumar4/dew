"""REPA's and iREPA's DINOv2 preprocessing by their official code run on drawn images, in float32 and
float64: the reference arm of tools/venv_reference.py's `repa-preprocessing` site (its protocol
tools/reference_worker.py), run in an environment holding torch and torchvision, on the CPU.

    python tools/repa_worker.py repa-preprocessing    `IN OUT` lines on stdin, each `done`

IN holds each draw's `<draw>.pixels`, uint8 images `[B, 256, 256, 3]`. OUT gets `<draw>.reference.<leaf>`
and `<draw>.truth.<leaf>` in float32 and float64: `dinov2`, the images through sihyun-yu/REPA's
`preprocess_raw_image` (train.py) "dinov2" branch (/255, ImageNet's normalization, bicubic to 224) read as
`[B, N, 3]` features, which End2End-Diffusion/iREPA's `DINOv2Encoder.preprocess` computes alike (each run
checks), and `zscore`, iREPA's `spatial_zscore` (ldm/utils.py, gamma GAMMA) of them; each definition
extracted from its file at a pinned commit and run as published.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import torch
from reference_source import definitions
from reference_worker import serve

REPA = "https://raw.githubusercontent.com/sihyun-yu/REPA/67f714503e3892f993844aab088ffc5791c92613/"
IREPA = "https://raw.githubusercontent.com/End2End-Diffusion/iREPA/99ad4ac234efe8de52ce157120f72856e836d09f/ldm/"
GAMMA = 0.6
TIMM = {"IMAGENET_DEFAULT_MEAN": (0.485, 0.456, 0.406), "IMAGENET_DEFAULT_STD": (0.229, 0.224, 0.225)}
"""timm.data's ImageNet statistics, which REPA's train.py and iREPA's encoders import."""


def run(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's preprocessed features and their spatial z-scores, in float32 and float64."""
    from torchvision.transforms import Normalize

    repa = definitions(REPA + "train.py", ("preprocess_raw_image",),
                       {"torch": torch, "Normalize": Normalize, **TIMM})["preprocess_raw_image"]
    irepa = definitions(IREPA + "vision_encoder.py", ("DINOv2Encoder",),
                        {"torch": torch, "Normalize": Normalize, "VisionEncoder": object, "Dict": dict,
                         "Optional": object, **TIMM})["DINOv2Encoder"]
    zscore = definitions(IREPA + "utils.py", ("spatial_zscore",), {"torch": torch})["spatial_zscore"]
    out = {}
    for name in sorted({key.split(".", 1)[0] for key in cases}):
        pixels = cases[f"{name}.pixels"]
        for kind, dtype in (("reference", torch.float32), ("truth", torch.float64)):
            raw = torch.as_tensor(pixels, dtype=dtype).permute(0, 3, 1, 2)
            dinov2 = repa(raw, "dinov2")
            if not torch.equal(irepa.preprocess(types.SimpleNamespace(resolution=256), raw), dinov2):
                raise ValueError("REPA's and iREPA's DINOv2 preprocessing differ")
            features = dinov2.permute(0, 2, 3, 1).reshape(len(pixels), -1, 3)
            out[f"{name}.{kind}.dinov2"] = features.numpy()
            out[f"{name}.{kind}.zscore"] = zscore(features, alpha=GAMMA).numpy()
    return out


if __name__ == "__main__":
    if sys.argv[1:] != ["repa-preprocessing"]:
        raise SystemExit(__doc__)
    torch.set_num_threads(1)
    serve(run)
