"""Score image metrics: `FID`, `CLIPScore`, `PSNR`, `SSIM` and `CLIPDistance`,
each a `Metric` the trainer scores an `ImageGrid` with, and each registered
in `dew.registry.metrics` under the name a run record gives it.

`FID().score(generated, reference)` and `CLIPScore().score(images, prompts)`
are the same numbers over image sets already in hand, with no trainer and no
batch."""

from .common import ImageMetric, Mean
from .fid import FID, frechet_distance
from .images import CLIPDistance, CLIPScore
from .psnr import PSNR, peak_signal_noise_ratio
from .ssim import SSIM, structural_similarity

__all__ = [
    "FID",
    "PSNR",
    "SSIM",
    "CLIPDistance",
    "CLIPScore",
    "ImageMetric",
    "Mean",
    "frechet_distance",
    "peak_signal_noise_ratio",
    "structural_similarity",
]
