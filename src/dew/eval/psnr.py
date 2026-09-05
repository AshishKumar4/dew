"""PSNR in pure jax, batched over images and video.

Both functions accept (B, H, W, C) or (B, T, H, W, C): video is flattened
to (B*T, H, W, C) and every frame scores independently.
"""

import jax.numpy as jnp

from dew.artifacts import ImageGrid
from dew.registry import metrics
from .common import ImageMetric, paired


def frame_batch(images: jnp.ndarray) -> jnp.ndarray:
    """`images` as (N, H, W, C) frames, a video's clips laid end to end."""
    images = jnp.asarray(images)
    if images.ndim == 5:
        batch, frames, height, width, channels = images.shape
        return images.reshape((batch * frames, height, width, channels))
    if images.ndim != 4:
        raise ValueError(
            f"expected (B, H, W, C) images or (B, T, H, W, C) video, got shape {images.shape}")
    return images


def peak_signal_noise_ratio(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    data_range: float,
    per_example: bool = False,
) -> jnp.ndarray:
    """PSNR = 10 log10(data_range^2 / MSE), per frame, as skimage defines it.

    `data_range` is the dynamic range of the signal, 2.0 for [-1, 1] inputs
    and 255 for uint8. The mean over frames comes back unless `per_example`
    asks for the (N,) per-frame scores. Identical inputs give +inf.
    """
    pred, targ = frame_batch(predictions), frame_batch(targets)
    mse = jnp.mean((pred - targ) ** 2, axis=(1, 2, 3))
    scores = 10.0 * jnp.log10(data_range**2 / mse)
    return scores if per_example else jnp.mean(scores)


@metrics("psnr")
def psnr(data_range: float = 2.0, field: str = "image", reads: type = ImageGrid) -> ImageMetric:
    """Mean PSNR in dB between the sampled frames and the batch's, higher is
    better.

    The artifact is in [-1, 1] and the batch holds uint8 pixels, which are
    put on the objective's scale, so both sides span the range that the
    default data_range of 2.0 describes. `reads` names the artifact type the
    trainer hands this metric; a video run passes `VideoGrid`.
    """
    def measure(artifact, batch):
        samples, targets = paired(artifact, batch, field)
        return peak_signal_noise_ratio(samples, targets, data_range)

    return ImageMetric(name="psnr", measure=measure, reads=reads)
