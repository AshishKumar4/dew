"""Score image metrics behind `dew.registry.metrics`: `metrics["fid"]()`,
`metrics["clip_score"]()`, `metrics.psnr()`, `metrics.ssim()`, `metrics.clip()`,
each a factory returning a `Metric` the trainer scores an `ImageGrid` with.

`fid(generated, reference)` and `clip_score(images, prompts)` are the same
numbers over image sets already in hand, with no trainer and no batch."""

from .common import ImageMetric, frames
from .fid import FID, fid, frechet_distance
from .images import clip, clip_image_text_cosine, clip_score, clip_score_metric
from .psnr import peak_signal_noise_ratio, psnr
from .ssim import ssim, structural_similarity

__all__ = ["FID", "ImageMetric", "clip", "clip_image_text_cosine", "clip_score",
           "clip_score_metric", "fid", "frames", "frechet_distance",
           "peak_signal_noise_ratio", "psnr", "ssim", "structural_similarity"]
