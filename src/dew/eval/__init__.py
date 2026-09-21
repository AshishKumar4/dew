"""Image metrics behind `dew.registry.metrics`: `metrics.fid()`,
`metrics.clip_score()`, `metrics.psnr()`, `metrics.ssim()`, `metrics.clip()`,
each a factory returning a `Metric` the trainer scores an `ImageGrid` with."""

from .common import ImageMetric, frames
from .fid import fid, frechet_distance
from .images import clip, clip_score
from .psnr import peak_signal_noise_ratio, psnr
from .ssim import ssim, structural_similarity

__all__ = ["ImageMetric", "clip", "clip_score", "fid", "frames", "frechet_distance",
           "peak_signal_noise_ratio", "psnr", "ssim", "structural_similarity"]
