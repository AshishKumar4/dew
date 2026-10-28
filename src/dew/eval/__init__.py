"""Metrics: the image metrics `FID`, `CLIPScore`, `PSNR`, `SSIM`, `LPIPS` and `CLIPDistance`, and the
calibration of a classifier's forecasts.

Each image metric is a `Metric` that the trainer scores an `ImageGrid` with, with an alias
in `dew.registry.metrics` a run config may name it by.

`FID().score(generated, reference)` and `CLIPScore().score(images, prompts)`
compute the same numbers over image sets you already have, without a trainer
or a batch; `CLIPScore().scores(...)` gives each image's, and
`clip_image_features` the CLIP image embedding a head can read.

A classifier's forecasts over its options calibrate by `dew.eval.calibration`: a fitted temperature
(`fit_temperature`), a histogram map, an abstention threshold, and the calibration error and
risk-coverage area they are scored by.
"""

from .calibration import (
    abstention_threshold,
    expected_calibration_error,
    fit_temperature,
    histogram_map,
    reliability_bins,
    risk_coverage_area,
)
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
    "abstention_threshold",
    "clip_image_features",
    "expected_calibration_error",
    "fit_temperature",
    "frechet_distance",
    "histogram_map",
    "peak_signal_noise_ratio",
    "reliability_bins",
    "risk_coverage_area",
    "structural_similarity",
]
