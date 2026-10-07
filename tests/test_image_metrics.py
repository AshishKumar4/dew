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
from reference_error import equations

from dew.artifacts import ImageGrid, VideoGrid
from dew.eval import PSNR, SSIM, peak_signal_noise_ratio as psnr, structural_similarity as ssim
from dew.eval.inception import InceptionV3


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


def _normalised(batch):
    """The batch on the objective's [-1, 1] scale, where the sampler's output lives."""
    return (jnp.asarray(batch['image'], jnp.float32) - 127.5) / 127.5


def test_psnr_of_identical_images_is_infinite(rng):
    x = _ramp_image((2, 32, 32, 3), rng)
    assert jnp.isinf(psnr(x, x, data_range=2.0))


def test_ssim_of_identical_images_is_one(rng):
    x = _ramp_image((2, 32, 32, 3), rng)
    assert float(ssim(x, x, data_range=2.0)) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("offset,data_range", [(0.1, 1.0), (0.1, 2.0), (0.5, 2.0)])
def test_psnr_matches_the_closed_form_for_a_known_error(offset, data_range):
    """A constant offset makes MSE exact, so PSNR is exact too."""
    x = jnp.zeros((2, 8, 8, 3))
    got = psnr(x, x + offset, data_range=data_range)
    expected = 10.0 * np.log10(data_range**2 / offset**2)
    assert float(got) == pytest.approx(expected, rel=1e-5)


def test_psnr_falls_as_noise_grows(rng):
    x = _ramp_image((2, 32, 32, 3), rng)
    key_a, key_b = jax.random.split(rng)
    small = psnr(x, x + 0.05 * jax.random.normal(key_a, x.shape), data_range=2.0)
    large = psnr(x, x + 0.20 * jax.random.normal(key_b, x.shape), data_range=2.0)
    assert float(small) > float(large)


def test_ssim_falls_under_blur_and_noise(rng):
    """The point of SSIM: both degradations score below a perfect match."""
    x = _ramp_image((2, 32, 32, 3), rng)
    perfect = float(ssim(x, x, data_range=2.0))
    blurred = float(ssim(x, _blur(x), data_range=2.0))
    noisy = float(ssim(x, x + 0.2 * jax.random.normal(rng, x.shape), data_range=2.0))
    assert perfect > blurred
    assert perfect > noisy


def test_ssim_falls_as_noise_grows(rng):
    x = _ramp_image((2, 32, 32, 3), rng)
    key_a, key_b = jax.random.split(rng)
    small = ssim(x, x + 0.05 * jax.random.normal(key_a, x.shape), data_range=2.0)
    large = ssim(x, x + 0.20 * jax.random.normal(key_b, x.shape), data_range=2.0)
    assert float(small) > float(large)


@pytest.mark.parametrize("size", [16, 64])
def test_ssim_matches_the_closed_form_on_constant_images(size):
    """Constant images have zero local variance, which collapses SSIM to
    (2 mu_x mu_y + C1) / (mu_x^2 + mu_y^2 + C1), and the variance has to come
    out zero in fp32 whatever order a backend sums the window in. Taken as
    E[x^2] - E[x]^2 over the raw planes it cancels to a few ulp, which 1 / C2
    turns into 1e-4 of SSIM: 6e-5 on CPU, 1.25e-4 on a TPU at HIGHEST.
    The score is then the mean of a map of near-equal values, where a
    one-pass fp32 sum rounds every partial sum the same way: 7.8e-6 off at
    64x64 on CPU XLA. The closed form itself, evaluated in fp32, rounds
    eight times (the plane mean and filter, 2 mu_x mu_y + C1, mu_y^2, the
    sum, + C1, the two products by C2, the division), and one float32 step
    at each moves the score up to 6.25e-7, so the bound is 6.3e-7: the
    one-pass mean fails it by 13x and the raw-plane cancellation by 100x."""
    data_range = 1.0
    x = jnp.full((1, size, size, 1), 0.5)
    y = jnp.full((1, size, size, 1), 0.7)
    c1 = (0.01 * data_range) ** 2
    expected = (2 * 0.5 * 0.7 + c1) / (0.5**2 + 0.7**2 + c1)
    assert float(ssim(x, y, data_range=data_range)) == pytest.approx(expected, rel=6.3e-7)


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


def test_every_contraction_in_ssim_and_the_extractor_asks_for_the_highest_precision():
    """A contraction that names no precision runs at the process default, and
    a TPU's DEFAULT is one bf16 pass, which moves SSIM's variances and the
    extractor's features off their fp32 references. The suite itself runs
    at 'highest' (conftest), so this traces both where the process default
    is DEFAULT and finds every convolution and dot asking for HIGHEST; a
    CPU computes the same values at either."""
    extractor = InceptionV3(channel_divisor=16)
    pixels = jax.ShapeDtypeStruct((1, 299, 299, 3), jnp.float32)
    frames = jax.ShapeDtypeStruct((2, 16, 16, 3), jnp.float32)
    variables = jax.eval_shape(extractor.init, jax.random.key(0), pixels)
    with jax.default_matmul_precision("default"):
        graphs = {"ssim": jax.make_jaxpr(lambda x, y: ssim(x, y, 2.0))(frames, frames),
                  "extractor": jax.make_jaxpr(extractor.apply)(variables, pixels)}
    highest = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    for name, graph in graphs.items():
        precisions = [equation.params["precision"] for equation in equations(graph)
                      if equation.primitive.name in ("conv_general_dilated", "dot_general")]
        assert precisions and all(p == highest for p in precisions), (name, set(precisions))


@pytest.mark.parametrize("metric_fn", [psnr, ssim], ids=['psnr', 'ssim'])
def test_video_scores_equal_the_flattened_frame_batch(rng, metric_fn):
    """(B, T, H, W, C) scores the same as the (B*T, H, W, C) frames do."""
    key_x, key_noise = jax.random.split(rng)
    video = _ramp_image((2 * 3, 32, 32, 3), key_x).reshape(2, 3, 32, 32, 3)
    degraded = video + 0.1 * jax.random.normal(key_noise, video.shape)
    frames = video.reshape(6, 32, 32, 3)
    assert float(metric_fn(video, degraded, data_range=2.0)) == pytest.approx(
        float(metric_fn(frames, degraded.reshape(6, 32, 32, 3), data_range=2.0)), rel=1e-5
    )


@pytest.mark.parametrize("metric_fn", [psnr, ssim], ids=['psnr', 'ssim'])
@pytest.mark.parametrize("shape", [(2, 32, 32, 3), (2, 3, 32, 32, 3), (2, 32, 32, 1)])
def test_per_example_scores_have_one_entry_per_frame(rng, metric_fn, shape):
    key_x, key_noise = jax.random.split(rng)
    x = _ramp_image((int(np.prod(shape[:-3])), *shape[-3:]), key_x).reshape(shape)
    y = x + 0.1 * jax.random.normal(key_noise, x.shape)
    scores = metric_fn(x, y, data_range=2.0, per_example=True)
    assert scores.shape == (int(np.prod(shape[:-3])),)
    assert float(jnp.mean(scores)) == pytest.approx(
        float(metric_fn(x, y, data_range=2.0)), rel=1e-5
    )


@pytest.mark.parametrize("metric_fn", [psnr, ssim], ids=['psnr', 'ssim'])
def test_metrics_reject_unbatched_inputs(metric_fn):
    with pytest.raises(ValueError, match=r"\(B, H, W, C\)"):
        metric_fn(jnp.zeros((32, 32, 3)), jnp.zeros((32, 32, 3)), data_range=2.0)


def test_psnr_metric_scores_a_perfect_reconstruction_as_infinite(rng):
    """The trainer hands the metric the objective's [-1, 1] artifact and the
    loader's uint8 batch, and the same image on both sides has zero error."""
    batch = _uint8_batch((2, 32, 32, 3), rng)
    metric = PSNR()
    assert np.isinf(metric.finalize(metric(ImageGrid(_normalised(batch)), batch)))


def test_ssim_metric_scores_a_perfect_reconstruction_as_one(rng):
    batch = _uint8_batch((2, 32, 32, 3), rng)
    metric = SSIM()
    assert metric.finalize(metric(ImageGrid(_normalised(batch)), batch)) == pytest.approx(1.0, abs=1e-4)


def test_psnr_metric_matches_the_closed_form_for_a_grey_level_error():
    """A constant error of 51 grey levels is 0.4 on the [-1, 1] scale, and the
    default data_range of 2.0 makes the score 10 log10(4 / 0.16). The name is
    the wandb key the score logs under."""
    batch = {'image': jnp.full((2, 8, 8, 3), 100, dtype=jnp.uint8)}
    generated = jnp.full((2, 8, 8, 3), (151 - 127.5) / 127.5)
    expected = 10.0 * np.log10(2.0**2 / 0.4**2)
    metric = PSNR()
    assert metric.name == "psnr"
    assert metric.finalize(metric(ImageGrid(generated), batch)) == pytest.approx(expected, rel=1e-5)


def test_ssim_metric_matches_the_closed_form_on_constant_images():
    """Constant images collapse SSIM to (2 mu_x mu_y + C1) / (mu_x^2 + mu_y^2 + C1).
    A zero sample against grey level 128 keeps both means near zero, so the score
    depends on C1 = (0.01 * data_range)^2 and pins the default data_range of 2.0
    together with the batch scale."""
    batch = {'image': jnp.full((1, 16, 16, 1), 128, dtype=jnp.uint8)}
    mu_y = (128 - 127.5) / 127.5
    c1 = (0.01 * 2.0) ** 2
    expected = c1 / (mu_y**2 + c1)
    metric = SSIM()
    assert metric.name == "ssim"
    assert metric.finalize(metric(ImageGrid(jnp.zeros((1, 16, 16, 1))), batch)) == pytest.approx(
        expected, rel=1e-4
    )


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
