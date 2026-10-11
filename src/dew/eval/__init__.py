"""Image metrics: `FID`, `CLIPScore`, `PSNR`, `SSIM`, `LPIPS` and `CLIPDistance`.

Each is a `Metric` that the trainer scores an `ImageGrid` with, with an alias
in `dew.registry.metrics` a run config may name it by.

`FID().score(generated, reference)` and `CLIPScore().score(images, prompts)`
compute the same numbers over image sets you already have, without a trainer
or a batch; `CLIPScore().scores(...)` gives each image's, and
`clip_image_features` the CLIP image embedding a head can read.
"""

from .common import ImageMetric, Mean
from .fid import FID, frechet_distance
from .images import CLIPDistance, CLIPScore, clip_image_features
from .lpips import LPIPS, LPIPSNetwork
from .psnr import PSNR, peak_signal_noise_ratio
from .ssim import SSIM, structural_similarity

__all__ = [
    "FID",
    "LPIPS",
    "PSNR",
    "SSIM",
    "CLIPDistance",
    "CLIPScore",
    "ImageMetric",
    "LPIPSNetwork",
    "Mean",
    "clip_image_features",
    "frechet_distance",
    "peak_signal_noise_ratio",
    "structural_similarity",
]
