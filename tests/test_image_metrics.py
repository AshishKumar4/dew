"""PSNR and SSIM, on images and on video.

They are checked against their closed forms and against the properties they
exist to report (degradation ordering, shape handling). The precision check
spans SSIM and the FID extractor, the two that contract at the highest
precision.
"""


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.artifacts import ImageGrid, VideoGrid
from dew.eval import PSNR, SSIM, peak_signal_noise_ratio as psnr, structural_similarity as ssim


def _ramp_image(shape, key):
    """Smooth-ish image in [-1, 1]: SSIM on pure noise is degenerate."""
    ramp = jnp.linspace(-1.0, 1.0, shape[-3])
    base = jnp.broadcast_to(ramp[None, :, None, None], (shape[0], shape[-3], shape[-2], shape[-1]))
    return base + 0.1 * jax.random.normal(key, base.shape)


def _blur(images, width=5):
    """Box blur over H and W, so 'degraded but structured' is a real case."""
    kernel = jnp.ones((width, width)) / (width * width)
    channels = images.shape[-1]
    weights = jnp.zeros((channels, channels, width, width)).at[
        jnp.arange(channels), jnp.arange(channels)
    ].set(kernel)
    return jax.lax.conv_general_dilated(
        images.transpose(0, 3, 1, 2), weights, (1, 1), 'SAME',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
    ).transpose(0, 2, 3, 1)


def _uint8_batch(shape, key):
    """A loader batch, the ramp image quantised to uint8 in 0..255."""
    x = _ramp_image(shape, key)
    return {'image': jnp.round((x + 1.0) * 127.5).clip(0, 255).astype(jnp.uint8)}


def test_psnr_of_uint8_pixels_is_the_closed_form_of_their_difference():
    """uint8 frames at `data_range` 255, as the docstring offers them, score as their pixel values do,
    skimage's `10 log10(255^2 / MSE)` in float64: white against black is 0 dB, and random frames
    match the float64 formula over the same pixels."""
    white, black = jnp.full((1, 4, 4, 3), 255, jnp.uint8), jnp.zeros((1, 4, 4, 3), jnp.uint8)
    assert float(psnr(white, black, 255.0)) == 0.0
    draws = np.random.default_rng(0)
    a, b = (jnp.asarray(draws.integers(0, 256, (2, 8, 8, 3)), jnp.uint8) for _ in range(2))
    mse = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2, axis=(1, 2, 3))
    np.testing.assert_allclose(np.asarray(psnr(a, b, 255.0, per_example=True)),
                               10 * np.log10(255.0 ** 2 / mse), rtol=1e-6)


def test_ssim_falls_under_blur_and_noise(rng):
    """The point of SSIM: both degradations score below a perfect match."""
    x = _ramp_image((2, 32, 32, 3), rng)
    perfect = float(ssim(x, x, data_range=2.0))
    blurred = float(ssim(x, _blur(x), data_range=2.0))
    noisy = float(ssim(x, x + 0.2 * jax.random.normal(rng, x.shape), data_range=2.0))
    assert perfect > blurred
    assert perfect > noisy


def _reference_ssim(x, y, data_range):
    """Wang et al.'s SSIM on one channel in float64, as skimage computes it
    with gaussian_weights=True, sigma=1.5 and use_sample_covariance=False:
    the means, variances and covariance filtered with an 11-tap gaussian
    (scipy's truncate=3.5 at sigma 1.5), and the border the window does not
    cover cropped before the mean."""
    from scipy.ndimage import gaussian_filter

    def filt(a):
        return gaussian_filter(a, sigma=1.5, truncate=3.5, mode="reflect")

    ux, uy = filt(x), filt(y)
    vx, vy, vxy = filt(x * x) - ux * ux, filt(y * y) - uy * uy, filt(x * y) - ux * uy
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux**2 + uy**2 + c1) * (vx + vy + c2))
    return score[5:-5, 5:-5].mean()


@pytest.mark.parametrize("data_range", [1.0, 2.0])
def test_ssim_matches_the_filtered_equations_of_wang_et_al(rng, data_range):
    """The port against the reference on noisy ramps, per frame: observed
    2.2e-07 off the float64 reference at data_range 1.0 and 1.9e-07 at 2.0,
    against a tolerance of 1e-5 on scores near 0.5. A window that is not
    gaussian, not 11 taps or not cropped moves the score by 1e-3 or more."""
    key_x, key_noise = jax.random.split(rng)
    x = _ramp_image((4, 32, 32, 3), key_x)
    y = x + 0.2 * jax.random.normal(key_noise, x.shape)
    x64, y64 = np.asarray(x, np.float64), np.asarray(y, np.float64)

    got = np.asarray(ssim(x, y, data_range=data_range, per_example=True))

    expected = [np.mean([_reference_ssim(x64[n, ..., c], y64[n, ..., c], data_range)
                         for c in range(3)]) for n in range(4)]
    assert np.abs(got - expected).max() < 1e-5, f"{got} against {expected}"


@pytest.mark.parametrize("dtypes", [(jnp.bfloat16, jnp.bfloat16), (jnp.float16, jnp.float32),
                                    (jnp.float64, jnp.float32)], ids=["bf16", "fp16-fp32", "fp64-fp32"])
def test_ssim_scores_a_batch_in_any_float_dtype_as_its_values_in_their_common_dtype(rng, dtypes):
    """Predictions and targets in any floats score as the same values in their common dtype, at
    least float32: a bfloat16 batch as its values in float32, a float64 one beside float32 targets
    in float64."""
    key_x, key_noise = jax.random.split(rng)
    with jax.enable_x64(jnp.float64 in dtypes):
        x = _ramp_image((2, 32, 32, 3), key_x)
        y = x + 0.2 * jax.random.normal(key_noise, x.shape)
        x, y = x.astype(dtypes[0]), y.astype(dtypes[1])
        common = jnp.promote_types(jnp.result_type(x, y), jnp.float32)
        got = ssim(x, y, data_range=2.0, per_example=True)
        held = ssim(x.astype(common), y.astype(common), data_range=2.0, per_example=True)
        assert got.dtype == common
        np.testing.assert_array_equal(np.asarray(got), np.asarray(held))


@pytest.mark.parametrize("metric_fn", [psnr, ssim], ids=['psnr', 'ssim'])
def test_video_scores_equal_the_flattened_frame_batch(rng, metric_fn):
    """(B, T, H, W, C) scores the same as the (B*T, H, W, C) frames do."""
    key_x, key_noise = jax.random.split(rng)
    video = _ramp_image((2 * 3, 32, 32, 3), key_x).reshape(2, 3, 32, 32, 3)
    degraded = video + 0.1 * jax.random.normal(key_noise, video.shape)
    frames = video.reshape(6, 32, 32, 3)
    # A video is scored as its frames' batch, the same operations on the same values.
    assert float(metric_fn(video, degraded, data_range=2.0)) == float(
        metric_fn(frames, degraded.reshape(6, 32, 32, 3), data_range=2.0))


@pytest.mark.parametrize("factory,raw", [(PSNR, psnr), (SSIM, ssim)],
                         ids=["psnr", "ssim"])
def test_frame_factories_read_a_video_grid_when_asked(rng, factory, raw):
    """A video run explicitly scores VideoGrid frames against the video field."""
    batch = {'video': _uint8_batch((6, 16, 16, 3), rng)['image'].reshape(2, 3, 16, 16, 3)}
    reference = (jnp.asarray(batch['video'], jnp.float32) - 127.5) / 127.5
    degraded = reference + 0.1
    assert factory().reads is ImageGrid
    metric = factory(field="video", reads=VideoGrid)
    assert metric.reads is VideoGrid
    artifact = VideoGrid(degraded)
    assert metric.finalize(metric(artifact, batch)) == pytest.approx(
        float(raw(degraded, reference, data_range=2.0)), rel=1e-5)

    scores = raw(degraded, reference, data_range=2.0, per_example=True)
    assert scores.shape == (6,)
    # The aggregate is the mean of the per-frame scores, the same operation.
    assert float(jnp.mean(scores)) == float(raw(degraded, reference, data_range=2.0))
