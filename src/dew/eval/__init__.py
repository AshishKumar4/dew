"""Score image metrics: `FID`, `CLIPScore`, `PSNR`, `SSIM` and `CLIPDistance`,
each a `Metric` the trainer scores an `ImageGrid` with, and each registered
in `dew.registry.metrics` under the name a run record gives it.

`fid(generated, reference)` and `clip_score(images, prompts)` are the same
numbers over image sets already in hand, with no trainer and no batch."""

from .common import ImageMetric, Mean, frames
from .fid import FID, fid, frechet_distance
from .images import CLIPDistance, CLIPScore, clip_image_text_cosine, clip_score
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
    "clip_image_text_cosine",
    "clip_score",
    "fid",
    "frames",
    "frechet_distance",
    "peak_signal_noise_ratio",
    "structural_similarity",
]
