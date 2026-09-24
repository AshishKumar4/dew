"""SSIM in pure jax (Wang et al. 2004), batched over images and video.

Standard parameters, 11x11 gaussian window, sigma 1.5, means taken over
channels after per-channel SSIM. No scipy/skimage dependency;
`tests/test_metrics.py` states the tolerance against the filtered equations
and the difference observed.
"""

import jax
import jax.numpy as jnp

from dew.artifacts import ImageGrid
from dew.registry import metrics

from .common import ImageMetric, paired
from .psnr import frame_batch

# Standard SSIM constants for the 11x11/sigma-1.5 gaussian window
_K1 = 0.01
_K2 = 0.03
_WINDOW_SIZE = 11
_SIGMA = 1.5


def _gaussian_window_1d(size: int, sigma: float) -> jnp.ndarray:
    coords = jnp.arange(size, dtype=jnp.float32) - (size - 1) / 2.0
    kernel = jnp.exp(-(coords**2) / (2.0 * sigma**2))
    return kernel / jnp.sum(kernel)


def _ssim_single_channel(
    x: jnp.ndarray, y: jnp.ndarray, data_range: float
) -> jnp.ndarray:
    """SSIM map between two single-channel images of shape (H, W).

    The window is applied as a 2D separable convolution with VALID padding,
    exactly the reference implementation's filtering; the score is the mean
    of the SSIM map.
    """
    window_1d = _gaussian_window_1d(_WINDOW_SIZE, _SIGMA)
    window_2d = jnp.outer(window_1d, window_1d)
    window_2d = window_2d[None, None]  # (1, 1, K, K) for conv_general_dilated

    def filt(img: jnp.ndarray) -> jnp.ndarray:
        img = img[None, None]  # (1, 1, H, W)
        # HIGHEST whatever the process default: a TPU's DEFAULT is one bf16
        # pass, which moves the variances below off the reference's fp64.
        out = jax.lax.conv_general_dilated(
            img, window_2d,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            precision=jax.lax.Precision.HIGHEST,
        )
        return out[0, 0]  # (H', W')

    # Variance and covariance do not move under a shift, so each plane is
    # centred on its own mean first. E[x^2] - E[x]^2 then subtracts small
    # numbers instead of two near-equal large ones. Over the raw planes a
    # flat region at value c cancels to c^2 (S - S^2), S being the window's
    # sum as a backend rounds it, which 1 / c2 below turns into 1e-4 of
    # SSIM. Centred, the residue is (c - mean)^2 (S - S^2): exactly zero
    # for a constant plane, and smaller for any region nearer the plane's
    # mean than zero. The shift is exact because the window sums to one, so
    # filtering a centred plane and adding the mean back is the plane's own
    # local mean, which the luminance term reads.
    x_mean, y_mean = jnp.mean(x), jnp.mean(y)
    x, y = x - x_mean, y - y_mean
    nu_x, nu_y = filt(x), filt(y)
    sigma_x2 = filt(x**2) - nu_x**2
    sigma_y2 = filt(y**2) - nu_y**2
    sigma_xy = filt(x * y) - nu_x * nu_y
    mu_x, mu_y = nu_x + x_mean, nu_y + y_mean
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y

    c1 = (_K1 * data_range) ** 2
    c2 = (_K2 * data_range) ** 2

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / denominator
    # A one-pass fp32 mean of near-equal values rounds every partial sum the
    # same way (7.8e-6 off at 64x64 on CPU XLA). The second pass sums what
    # the first left over, which is small, and lands within 1e-7 of the
    # map's float64 mean.
    first = jnp.mean(ssim_map)
    return first + jnp.mean(ssim_map - first)


def structural_similarity(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    data_range: float,
    per_example: bool = False,
) -> jnp.ndarray:
    """SSIM (Wang et al. 2004) per frame, an 11x11 gaussian window of sigma
    1.5 on each channel and the channels averaged.

    `data_range` is the dynamic range of the signal, 2.0 for [-1, 1] inputs.
    The mean over frames comes back unless `per_example` asks for the (N,)
    per-frame scores. Identical inputs give 1.0.
    """
    pred, targ = frame_batch(predictions), frame_batch(targets)
    # vmap over frames then channels. Each (H, W) plane is scored independently,
    # giving (N, C), which collapses to one score per frame.
    per_channel = jax.vmap(
        jax.vmap(_ssim_single_channel, in_axes=(2, 2, None)),
        in_axes=(0, 0, None),
    )(pred, targ, data_range)
    scores = jnp.mean(per_channel, axis=-1)
    return scores if per_example else jnp.mean(scores)


@metrics("ssim")
def ssim(data_range: float = 2.0, field: str = "image", reads: type = ImageGrid) -> ImageMetric:
    """Mean SSIM between the sampled frames and the batch's, higher is
    better, on the same [-1, 1] scale as `psnr`. `reads` names the artifact
    type the trainer hands this metric; a video run passes `VideoGrid`."""
    def measure(artifact, batch):
        samples, targets = paired(artifact, batch, field)
        return structural_similarity(samples, targets, data_range, per_example=True)

    return ImageMetric(name="ssim", measure=measure, reads=reads)
