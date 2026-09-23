#!/usr/bin/env python3
"""Write pytorch-fid's own pool3 features and FID for tests/test_metrics.py.

pytorch-fid is the reference implementation FID numbers are reported with,
and its published weights are the ones the jax-fid checkpoint Dew converts
was ported from. This runs pytorch-fid's `InceptionV3([3])` on those
weights, with its own input path (uint8 / 255, bilinear to 299x299 without
antialiasing, then 2x - 1), over two fixed image sets, and records the
features and `calculate_frechet_distance` between the two sets. Set "a" is
64x64 and is upsampled on the way in; set "b" is stored at 50x50 and scored
at 400x400 (every pixel repeated 8 times), so the downsampling path is
compared too.

It imports nothing from Dew and runs in its own environment, CPU torch.
pytorch-fid 0.3.0 calls `scipy.linalg.sqrtm(..., disp=False)`, which scipy
1.18 no longer takes, so scipy is held below 1.16:

    uv venv ~/.cache/dew/reference-venvs/eval-fix --python 3.12
    uv pip install --python ~/.cache/dew/reference-venvs/eval-fix/bin/python \\
        --torch-backend cpu torch torchvision pytorch-fid==0.3.0 numpy "scipy<1.16"
    ~/.cache/dew/reference-venvs/eval-fix/bin/python tools/pytorch_fid_reference.py
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pytorch_fid
import torch
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import FID_WEIGHTS_URL, InceptionV3

FIXTURE = (Path(__file__).resolve().parents[1]
           / "tests/fixtures/inception/pytorch_fid_reference.npz")
COUNT = 8
UPSCALE = 8
"""Set "b" is scored at its stored size times this, past 299, so it is downsampled."""


def image_sets(seed: int = 20260922) -> tuple[np.ndarray, np.ndarray]:
    """Two sets of smooth colour fields with noise on top, not blank frames.

    Each image is a low-resolution random field upsampled bilinearly plus
    per-pixel noise, so the extractor sees edges and texture at more than
    one scale.
    """
    rng = np.random.default_rng(seed)

    def draw(size: int) -> np.ndarray:
        coarse = torch.from_numpy(rng.uniform(0, 255, (COUNT, 3, 6, 6)).astype(np.float32))
        field = torch.nn.functional.interpolate(coarse, size=(size, size), mode="bilinear",
                                                align_corners=False).numpy()
        noisy = field + rng.normal(0, 24, field.shape)
        return np.clip(np.rint(noisy), 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)

    return draw(64), draw(50)


def scored(images: np.ndarray) -> np.ndarray:
    """What `upscaled` does to set "b", written once for both sides."""
    return np.repeat(np.repeat(images, UPSCALE, axis=1), UPSCALE, axis=2)


def features(model: InceptionV3, images: np.ndarray) -> np.ndarray:
    """pytorch-fid's pool3 features: the [0, 1] tensor its data loader hands it."""
    batch = torch.from_numpy(images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
    with torch.no_grad():
        (pool,) = model(batch)
    return pool[:, :, 0, 0].numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURE)
    args = parser.parse_args()
    torch.manual_seed(0)
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval()
    weights = Path(torch.hub.get_dir()) / "checkpoints" / FID_WEIGHTS_URL.rsplit("/", 1)[-1]
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    a, b = image_sets()
    features_a, features_b = features(model, a), features(model, scored(b))
    stats = [(f.astype(np.float64).mean(axis=0), np.cov(f.astype(np.float64), rowvar=False))
             for f in (features_a, features_b)]
    distance = calculate_frechet_distance(*stats[0], *stats[1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, images_a=a, images_b=b, upscale_b=UPSCALE,
                        features_a=features_a, features_b=features_b, fid=np.float64(distance),
                        pytorch_fid=pytorch_fid.__version__, torch=torch.__version__,
                        weights=FID_WEIGHTS_URL, weights_sha256=digest)
    print(f"wrote {args.out}: fid {distance!r}, weights sha256 {digest}")


if __name__ == "__main__":
    main()
