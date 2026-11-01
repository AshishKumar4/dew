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

Each name is imported from its own module when first read, so a reader of calibration or of `Mean` reads
no image metric.
"""

from typing import TYPE_CHECKING

from dew._exports import lookup as _lookup

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.eval.calibration import (
        abstention_threshold,
        expected_calibration_error,
        fit_temperature,
        histogram_map,
        reliability_bins,
        risk_coverage_area,
    )
    from dew.eval.common import ImageMetric, Mean
    from dew.eval.fid import FID, frechet_distance
    from dew.eval.images import CLIPDistance, CLIPScore, clip_image_features
    from dew.eval.lpips import LPIPS, LPIPSNetwork
    from dew.eval.psnr import PSNR, peak_signal_noise_ratio
    from dew.eval.ssim import SSIM, structural_similarity

_EXPORTS = {
    "abstention_threshold": "dew.eval.calibration",
    "expected_calibration_error": "dew.eval.calibration",
    "fit_temperature": "dew.eval.calibration",
    "histogram_map": "dew.eval.calibration",
    "reliability_bins": "dew.eval.calibration",
    "risk_coverage_area": "dew.eval.calibration",
    "ImageMetric": "dew.eval.common",
    "Mean": "dew.eval.common",
    "FID": "dew.eval.fid",
    "frechet_distance": "dew.eval.fid",
    "CLIPDistance": "dew.eval.images",
    "CLIPScore": "dew.eval.images",
    "clip_image_features": "dew.eval.images",
    "LPIPS": "dew.eval.lpips",
    "LPIPSNetwork": "dew.eval.lpips",
    "PSNR": "dew.eval.psnr",
    "peak_signal_noise_ratio": "dew.eval.psnr",
    "SSIM": "dew.eval.ssim",
    "structural_similarity": "dew.eval.ssim",
}

__getattr__ = _lookup(__name__, _EXPORTS)

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
