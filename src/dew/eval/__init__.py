"""Image metrics behind `dew.registry.metrics`: `metrics.fid()`,
`metrics.clip_score()`, `metrics.psnr()`, `metrics.ssim()`, `metrics.clip()`,
each a factory returning a `Metric` the trainer scores an `ImageGrid` with."""

from .common import ImageMetric, frames
from .images import clip, clip_score
from .fid import fid, frechet_distance
from .psnr import peak_signal_noise_ratio, psnr
from .ssim import structural_similarity, ssim

__all__ = [
    "ImageMetric",
    "frames",
    # CLIP-based
    "clip",
    "clip_score",
    # FID
    "fid",
    "frechet_distance",
    # Pixel-level
    "peak_signal_noise_ratio",
    "psnr",
    "structural_similarity",
    "ssim",
]
