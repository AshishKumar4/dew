"""Evaluation metric tests.

PSNR and SSIM are checked against their closed forms and against the
properties they exist to report (degradation ordering, shape handling).
The Frechet distance itself is checked against closed forms that need no
weights; the end-to-end InceptionV3 path downloads the FID checkpoint and is
network-marked. The CLIP metrics build on the tiny checkpoint under
tests/fixtures/clip and score against the cosines of the reference's own
embeddings.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dew.eval as metrics
from dew.artifacts import ImageGrid, VideoGrid
from dew.eval import (
    ImageMetric,
    clip,
    clip_score,
    fid,
    peak_signal_noise_ratio as psnr,
    psnr as psnr_metric,
    ssim as ssim_metric,
    structural_similarity as ssim,
)
from dew.eval.fid import frechet_distance
from dew.registry import metrics as registry

CLIP_TINY = Path(__file__).resolve().parent / "fixtures" / "clip" / "tiny"


def test_frechet_distance_of_a_distribution_with_itself_is_zero(rng):
    features = np.asarray(jax.random.normal(rng, (256, 16)))
    mu, sigma = features.mean(axis=0), np.cov(features, rowvar=False)
    assert frechet_distance(mu, sigma, mu, sigma) == pytest.approx(0.0, abs=1e-6)


def test_frechet_distance_of_shifted_gaussians_is_the_squared_mean_gap():
    sigma = np.eye(8)
    mu_a = np.zeros(8)
    mu_b = np.full(8, 0.5)
    assert frechet_distance(mu_a, sigma, mu_b, sigma) == pytest.approx(8 * 0.25, abs=1e-6)


def test_frechet_distance_grows_with_covariance_mismatch():
    mu = np.zeros(8)
    identity = np.eye(8)
    near = frechet_distance(mu, identity, mu, identity * 1.5)
    far = frechet_distance(mu, identity, mu, identity * 4.0)
    assert 0 < near < far


@pytest.mark.network
def test_fid_metric_scores_real_images_better_than_noise(rng):
    metric = fid()
    assert isinstance(metric, ImageMetric)
    assert metric.name == 'fid' and metric.reads is ImageGrid

    key_real, key_noise = jax.random.split(rng)
    real = jax.random.randint(key_real, (8, 64, 64, 3), 0, 256, dtype=jnp.int32).astype(jnp.uint8)
    batch = {'image': real}

    # Generated samples live in [-1, 1]; the same images should score far
    # closer to the batch than unrelated noise does
    matching = metric(ImageGrid((jnp.asarray(real, jnp.float32) - 127.5) / 127.5), batch)
    unrelated = metric(ImageGrid(jax.random.normal(key_noise, (8, 64, 64, 3))), batch)
    assert np.isfinite(matching) and np.isfinite(unrelated)
    assert matching < unrelated


############################################################################################################
# PSNR / SSIM
############################################################################################################


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


def test_ssim_matches_the_closed_form_on_constant_images():
    """Constant images have zero local variance, which collapses SSIM to
    (2 mu_x mu_y + C1) / (mu_x^2 + mu_y^2 + C1)."""
    data_range = 1.0
    x = jnp.full((1, 16, 16, 1), 0.5)
    y = jnp.full((1, 16, 16, 1), 0.7)
    c1 = (0.01 * data_range) ** 2
    expected = (2 * 0.5 * 0.7 + c1) / (0.5**2 + 0.7**2 + c1)
    assert float(ssim(x, y, data_range=data_range)) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("metric_fn", [psnr, ssim], ids=['psnr', 'ssim'])
def test_video_scores_equal_the_flattened_frame_batch(rng, metric_fn):
    """(B, T, H, W, C) must score exactly as the (B*T, H, W, C) frames do."""
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
    x = _ramp_image((int(np.prod(shape[:-3])),) + shape[-3:], key_x).reshape(shape)
    y = x + 0.1 * jax.random.normal(key_noise, x.shape)
    scores = metric_fn(x, y, data_range=2.0, per_example=True)
    assert scores.shape == (int(np.prod(shape[:-3])),)
    assert float(jnp.mean(scores)) == pytest.approx(
        float(metric_fn(x, y, data_range=2.0)), rel=1e-5
    )


@pytest.mark.parametrize("metric_fn", [psnr, ssim], ids=['psnr', 'ssim'])
def test_metrics_reject_unbatched_inputs(metric_fn):
    with pytest.raises(AssertionError):
        metric_fn(jnp.zeros((32, 32, 3)), jnp.zeros((32, 32, 3)), data_range=2.0)


def test_psnr_metric_factory_wires_up_the_trainer_contract(rng):
    metric = psnr_metric()
    assert isinstance(metric, ImageMetric)
    assert metric.name == 'psnr' and metric.reads is ImageGrid

    batch = _uint8_batch((2, 32, 32, 3), rng)
    degraded = _normalised(batch) + 0.1
    assert metric(ImageGrid(degraded), batch) == pytest.approx(
        float(psnr(degraded, _normalised(batch), data_range=2.0)), rel=1e-5
    )
    assert metric.reduce([1.0, 3.0]) == 2.0


def test_ssim_metric_factory_wires_up_the_trainer_contract(rng):
    metric = ssim_metric()
    assert isinstance(metric, ImageMetric)
    assert metric.name == 'ssim' and metric.reads is ImageGrid

    batch = _uint8_batch((2, 32, 32, 3), rng)
    degraded = _blur(_normalised(batch))
    assert metric(ImageGrid(degraded), batch) == pytest.approx(
        float(ssim(degraded, _normalised(batch), data_range=2.0)), rel=1e-5
    )


def test_psnr_metric_scores_a_perfect_reconstruction_as_infinite(rng):
    """The trainer hands the metric the objective's [-1, 1] artifact and the
    loader's uint8 batch, and the same image on both sides has zero error."""
    batch = _uint8_batch((2, 32, 32, 3), rng)
    assert np.isinf(psnr_metric()(ImageGrid(_normalised(batch)), batch))


def test_ssim_metric_scores_a_perfect_reconstruction_as_one(rng):
    batch = _uint8_batch((2, 32, 32, 3), rng)
    assert ssim_metric()(ImageGrid(_normalised(batch)), batch) == pytest.approx(1.0, abs=1e-4)


def test_psnr_metric_matches_the_closed_form_for_a_grey_level_error():
    """A constant error of 51 grey levels is 0.4 on the [-1, 1] scale, and the
    default data_range of 2.0 makes the score 10 log10(4 / 0.16)."""
    batch = {'image': jnp.full((2, 8, 8, 3), 100, dtype=jnp.uint8)}
    generated = jnp.full((2, 8, 8, 3), (151 - 127.5) / 127.5)
    expected = 10.0 * np.log10(2.0**2 / 0.4**2)
    assert psnr_metric()(ImageGrid(generated), batch) == pytest.approx(expected, rel=1e-5)


def test_ssim_metric_matches_the_closed_form_on_constant_images():
    """Constant images collapse SSIM to (2 mu_x mu_y + C1) / (mu_x^2 + mu_y^2 + C1).
    A zero sample against grey level 128 keeps both means near zero, so the score
    depends on C1 = (0.01 * data_range)^2 and pins the default data_range of 2.0
    together with the batch scale."""
    batch = {'image': jnp.full((1, 16, 16, 1), 128, dtype=jnp.uint8)}
    mu_y = (128 - 127.5) / 127.5
    c1 = (0.01 * 2.0) ** 2
    expected = c1 / (mu_y**2 + c1)
    assert ssim_metric()(ImageGrid(jnp.zeros((1, 16, 16, 1))), batch) == pytest.approx(
        expected, rel=1e-4
    )


def test_a_video_metric_scores_the_clips(rng):
    """A metric built to read VideoGrid scores the clips as the flattened
    frames, against the batch's video field."""
    batch = {'video': _uint8_batch((6, 16, 16, 3), rng)['image'].reshape(2, 3, 16, 16, 3)}
    reference = (jnp.asarray(batch['video'], jnp.float32) - 127.5) / 127.5
    degraded = reference + 0.1
    metric = ImageMetric(name='psnr', reads=VideoGrid, measure=psnr_metric(field='video').measure)
    assert metric(VideoGrid(degraded), batch) == pytest.approx(
        float(psnr(degraded, reference, data_range=2.0)), rel=1e-5)


@pytest.mark.parametrize("factory,raw", [(psnr_metric, psnr), (ssim_metric, ssim)],
                         ids=["psnr", "ssim"])
def test_frame_factories_read_a_video_grid_when_asked(rng, factory, raw):
    """A video run's metric reads `VideoGrid`: the factory default stays
    `ImageGrid`, and the trainer's `_pick` finds the artifact the objective
    produces."""
    from dew.training.trainer import _pick
    batch = {'video': _uint8_batch((6, 16, 16, 3), rng)['image'].reshape(2, 3, 16, 16, 3)}
    reference = (jnp.asarray(batch['video'], jnp.float32) - 127.5) / 127.5
    degraded = reference + 0.1
    assert factory().reads is ImageGrid
    metric = factory(field="video", reads=VideoGrid)
    assert metric.reads is VideoGrid
    artifact = _pick((VideoGrid(degraded),), metric.reads)
    assert metric(artifact, batch) == pytest.approx(
        float(raw(degraded, reference, data_range=2.0)), rel=1e-5)


def test_metrics_package_exports_resolve():
    """The package surface the trainer configures metrics from, and the
    registry the design names them in."""
    assert set(metrics.__all__) == {
        'ImageMetric',
        'frames',
        'clip',
        'clip_score',
        'fid',
        'frechet_distance',
        'peak_signal_noise_ratio',
        'psnr',
        'structural_similarity',
        'ssim',
    }
    for name in metrics.__all__:
        assert getattr(metrics, name) is not None
    assert registry['psnr'] is psnr_metric and registry.psnr is psnr_metric
    assert {'fid', 'clip', 'clip_score', 'psnr', 'ssim'} <= set(registry)


############################################################################################################
# CLIP
############################################################################################################


def clip_fixture():
    """The reference's inputs and embeddings, and the sampler-shaped batch
    that carries them: images in [-1, 1] and the tokenized captions."""
    reference = np.load(CLIP_TINY / "reference.npz")
    recipe = json.loads((CLIP_TINY / "prompts.json").read_text())["images"]
    images = np.random.RandomState(recipe["seed"]).randint(
        0, 256, tuple(recipe["shape"]), dtype=np.uint8)
    generated = jnp.asarray(images, jnp.float32) / 127.5 - 1.0
    batch = {"text": {"input_ids": reference["input_ids"],
                      "attention_mask": reference["attention_mask"]}}
    image = reference["image_embeds"].astype(np.float64)
    text = reference["text_embeds"].astype(np.float64)
    cosine = ((image * text).sum(-1)
              / np.linalg.norm(image, axis=-1) / np.linalg.norm(text, axis=-1))
    return generated, batch, cosine


# One fp32 cosine differs by ~1e-7 across devices; CLIPScore is a hundred
# cosines, so its bound is the same relative one.
CLIP_TOLERANCE = 1e-5
CLIP_SCORE_TOLERANCE = 1e-3


def test_clip_metric_scores_the_reference_cosine():
    """The factory builds on the vendored towers, the images go through the
    checkpoint's own processor, and the score is the reference's
    mean(1 - cos): observed 3.2e-08 off it against a tolerance of 1e-5. Before
    the towers were vendored, the factory raised ImportError on
    `FlaxCLIPModel`, which transformers 5 removed."""
    metric = clip(modelname=str(CLIP_TINY))
    assert isinstance(metric, ImageMetric)
    assert metric.name == 'clip_similarity' and metric.reads is ImageGrid
    generated, batch, cosine = clip_fixture()

    score = metric(ImageGrid(generated), batch)

    expected = np.mean(1.0 - cosine)
    assert abs(score - expected) < CLIP_TOLERANCE, f"{score} against {expected}"


def test_clip_score_metric_clamps_the_reference_cosine():
    """CLIPScore is 100 * max(cos, 0) averaged; the fixture holds one negative
    cosine (-0.072) among three positive ones, so the clamp does work here.
    Observed 6.1e-06 off the reference on CPU and 1.0e-05 on an RTX 4080,
    against a tolerance of 1e-3 on a score of order 15."""
    metric = clip_score(modelname=str(CLIP_TINY))
    assert metric.name == 'clip_score'
    generated, batch, cosine = clip_fixture()
    assert (cosine < 0).any() and (cosine > 0).any()

    score = metric(ImageGrid(generated), batch)

    expected = np.mean(100.0 * np.maximum(cosine, 0.0))
    assert abs(score - expected) < CLIP_SCORE_TOLERANCE, f"{score} against {expected}"
    assert score != pytest.approx(np.mean(100.0 * cosine), abs=1e-3)


def test_a_sample_outside_the_pixel_range_is_clipped_not_wrapped():
    """A sampler does not promise [-1, 1]. Casting 1.2 straight to uint8 wraps
    it to a dark pixel, which the old metric did; the score of an overshooting
    white image has to be the score of a white one."""
    metric = clip_score(modelname=str(CLIP_TINY))
    _, batch, _ = clip_fixture()
    white = jnp.ones((4, 16, 12, 3), jnp.float32)

    assert metric(ImageGrid(1.2 * white), batch) == metric(ImageGrid(white), batch)
    assert metric(ImageGrid(-1.2 * white), batch) == metric(ImageGrid(-white), batch)
    assert metric(ImageGrid(white), batch) != metric(ImageGrid(-white), batch)


def test_a_metric_pairs_its_samples_with_the_records_they_came_from():
    """An objective samples a fixed few rows of the batch, so psnr and ssim
    score four samples against a batch of eight rather than failing to
    broadcast, which is what a diffusion run with --val-metrics psnr did."""
    key = jax.random.key(0)
    samples = jax.random.uniform(key, (4, 32, 32, 3), minval=-1.0, maxval=1.0)
    batch = {"image": jax.random.randint(key, (8, 32, 32, 3), 0, 256, jnp.uint8)}
    for name in ("psnr", "ssim"):
        score = registry[name]()(ImageGrid(samples), batch)
        assert np.isfinite(score)


def test_a_metric_refuses_a_batch_with_fewer_records_than_samples():
    with pytest.raises(ValueError, match="at least as many"):
        registry["psnr"]()(ImageGrid(jnp.zeros((8, 32, 32, 3))),
                           {"image": jnp.zeros((4, 32, 32, 3), jnp.uint8)})
