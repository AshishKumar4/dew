"""Streaming evaluation statistics against complete-population references."""

import numpy as np
import pytest

from dew.artifacts import ImageGrid, TokenScores, VideoGrid
from dew.eval import psnr, ssim
from dew.eval.fid import FIDStats, GaussianStats, fid, frechet_distance
from dew.objectives.lm.objective import perplexity
from dew.objectives.rl.preference import DPOObjective
from dew.objectives.rl.grpo import GRPOObjective


def test_fid_pools_unequal_batches_and_singletons():
    rng = np.random.default_rng(93)
    generated = rng.normal(size=(19, 7)) + np.arange(19)[:, None] / 5
    real = rng.normal(size=(23, 7)) @ np.diag(np.arange(1, 8))
    metric = fid()
    accumulated = None
    for gen, ref in zip(np.split(generated, [1, 5, 12]), np.split(real, [0, 8, 22])):
        contribution = FIDStats(GaussianStats.from_features(gen, population="generated"),
                                GaussianStats.from_features(ref, population="real"))
        accumulated = contribution if accumulated is None else metric.merge(accumulated, contribution)
    expected = frechet_distance(generated.mean(0), np.cov(generated, rowvar=False),
                                real.mean(0), np.cov(real, rowvar=False))
    assert (accumulated.generated.count, accumulated.real.count) == (19, 23)
    np.testing.assert_allclose(accumulated.generated.covariance(population="generated"),
                               np.cov(generated, rowvar=False), rtol=1e-13, atol=1e-13)
    assert metric.finalize(accumulated) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_fid_refuses_insufficient_or_nonfinite_populations():
    one = GaussianStats.from_features(np.ones((1, 3)), population="generated")
    real = GaussianStats.from_features(np.eye(3), population="real")
    with pytest.raises(ValueError, match="generated.*at least two"):
        fid().finalize(FIDStats(one, real))
    with pytest.raises(ValueError, match="real.*finite"):
        GaussianStats.from_features([[1, np.nan]], population="real")


def test_image_mean_weights_each_image_once_across_unequal_batches():
    metric = psnr()
    targets = np.zeros((5, 4, 4, 1), dtype=np.float32)
    errors = np.array([.1, .2, .4, .8, 1.0], dtype=np.float32)
    generated = targets - 1.0 + errors[:, None, None, None]
    first = metric(ImageGrid(generated[:4]), {"image": targets[:4]})
    last = metric(ImageGrid(generated[4:]), {"image": targets[4:]})
    expected = np.mean(10 * np.log10(4 / errors.astype(np.float64) ** 2))
    assert metric.finalize(metric.merge(first, last)) == pytest.approx(expected, rel=1e-6)
    with pytest.raises(ValueError, match="equal counts"):
        metric(ImageGrid(generated[:4]), {"image": targets})


def test_perplexity_streams_weighted_targets_and_empty_contributions():
    metric = perplexity()
    first = metric(TokenScores(np.array([[1., 4.]]), np.array([[1., .5]])), {})
    empty = metric(TokenScores(np.array([[9.]]), np.array([[0.]])), {})
    last = metric(TokenScores(np.array([[2.]]), np.array([[3.]])), {})
    pooled = metric.merge(metric.merge(first, empty), last)
    assert metric.finalize(pooled) == pytest.approx(np.exp(9 / 4.5))
    with pytest.raises(ValueError, match="no counted target"):
        metric.finalize(empty)


@pytest.mark.parametrize("objective_type", [DPOObjective, GRPOObjective])
def test_policy_preview_uses_policy_weights_instead_of_frozen_reference(objective_type):
    import jax
    import jax.numpy as jnp
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.base import Step
    from dew.objectives.lm import Samples
    from dew.sampling.text import Sampling, generate

    model = CausalTransformer(vocab_size=8, emb_features=16, num_layers=1, num_heads=2,
                              mlp_features=32, max_seq_len=16, tie_embeddings=False,
                              dtype="float32", attention_impl="xla")
    objective = objective_type(model, seq_len=8,
                               samples=Samples(prompt=[1, 2], max_new_tokens=3, sampling=Sampling(temperature=0)))
    policy = objective.init(jax.random.key(0))
    reference = objective.init(jax.random.key(1))
    key = jax.random.key(9)
    step = Step(step=jnp.asarray(0), key=key, ema=reference)
    preview = objective.preview(policy, {}, step)
    expected = generate(model, policy, jnp.asarray([[1, 2]]), 3, key=key, sampling=Sampling(temperature=0)).tokens
    frozen = generate(model, reference, jnp.asarray([[1, 2]]), 3, key=key, sampling=Sampling(temperature=0)).tokens
    assert not np.array_equal(expected, frozen)
    np.testing.assert_array_equal(preview.tokens, expected)


@pytest.mark.parametrize("factory", [psnr, ssim])
@pytest.mark.parametrize("reference_shape", [(1, 1, 16, 16, 3), (1, 2, 1, 16, 3), (1, 2, 16, 16, 1)])
def test_paired_video_metrics_refuse_broadcastable_missing_pixels(factory, reference_shape):
    metric = factory(field="video", reads=VideoGrid)
    generated = VideoGrid(np.zeros((1, 2, 16, 16, 3), np.float32))
    with pytest.raises(ValueError, match="pixel shapes"):
        metric(generated, {"video": np.zeros(reference_shape, np.uint8)})
