"""Streaming evaluation statistics against complete-population references."""

import numpy as np
import pytest

from dew.artifacts import ImageGrid, TokenScores
from dew.eval import PSNR
from dew.eval.fid import FID, FIDStats, GaussianStats, frechet_distance
from dew.objectives.lm import Perplexity
from dew.objectives.rl.grpo import GRPOObjective
from dew.objectives.rl.preference import DPOObjective


def test_fid_pools_unequal_batches_and_singletons():
    rng = np.random.default_rng(93)
    generated = rng.normal(size=(19, 7)) + np.arange(19)[:, None] / 5
    real = rng.normal(size=(23, 7)) @ np.diag(np.arange(1, 8))
    metric = FID()
    accumulated = None
    for gen, ref in zip(np.split(generated, [1, 5, 12]), np.split(real, [0, 8, 22]), strict=True):
        contribution = FIDStats(GaussianStats.from_features(gen, population="generated"),
                                GaussianStats.from_features(ref, population="real"))
        accumulated = contribution if accumulated is None else metric.merge(accumulated, contribution)
    expected = frechet_distance(generated.mean(0), np.cov(generated, rowvar=False),
                                real.mean(0), np.cov(real, rowvar=False))
    assert (accumulated.generated.count, accumulated.real.count) == (19, 23)
    np.testing.assert_allclose(accumulated.generated.covariance(population="generated"),
                               np.cov(generated, rowvar=False), rtol=1e-13, atol=1e-13)
    assert metric.finalize(accumulated) == pytest.approx(expected, rel=1e-12, abs=1e-12)

    mu = np.zeros(8)
    identity = np.eye(8)
    assert frechet_distance(mu, identity, mu, identity * 1.5) == pytest.approx(
        8 * (1 - np.sqrt(1.5)) ** 2, abs=1e-6)
    assert frechet_distance(mu, identity, mu, identity * 4.0) == pytest.approx(
        8 * (1 - 2.0) ** 2, abs=1e-6)


def test_image_mean_weights_each_image_once_across_unequal_batches():
    metric = PSNR()
    targets = np.zeros((5, 4, 4, 1), dtype=np.float32)
    errors = np.array([.1, .2, .4, .8, 1.0], dtype=np.float32)
    generated = targets - 1.0 + errors[:, None, None, None]
    first = metric(ImageGrid(generated[:4]), {"image": targets[:4]})
    last = metric(ImageGrid(generated[4:]), {"image": targets[4:]})
    expected = np.mean(10 * np.log10(4 / errors.astype(np.float64) ** 2))
    assert metric.finalize(metric.merge(first, last)) == pytest.approx(expected, rel=1e-6)
    with pytest.raises(ValueError):
        metric(ImageGrid(generated[:4]), {"image": targets})


def test_perplexity_streams_weighted_targets_and_empty_contributions():
    metric = Perplexity()
    first = metric(
        TokenScores(
            np.array([[1.0, 4.0]]),
            np.array([[1.0, 0.5]]),
            correct=np.zeros_like(np.array([[1.0, 4.0]]), dtype=bool),
        ),
        {},
    )
    empty = metric(
        TokenScores(
            np.array([[9.0]]), np.array([[0.0]]), correct=np.zeros_like(np.array([[9.0]]), dtype=bool)
        ),
        {},
    )
    last = metric(
        TokenScores(
            np.array([[2.0]]), np.array([[3.0]]), correct=np.zeros_like(np.array([[2.0]]), dtype=bool)
        ),
        {},
    )
    pooled = metric.merge(metric.merge(first, empty), last)
    assert metric.finalize(pooled) == pytest.approx(np.exp(9 / 4.5))
    with pytest.raises(ValueError):
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
    objective = objective_type(
        model, seq_len=8, samples=Samples(prompt=[1, 2], max_new_tokens=3, sampling=Sampling(temperature=0))
    )
    policy = objective.init(jax.random.key(0))
    reference = objective.init(jax.random.key(1))
    key = jax.random.key(9)
    step = Step(step=jnp.asarray(0), key=key, ema=reference)
    preview = objective.preview(policy, {}, step)
    expected = generate(
        model, policy, jnp.asarray([[1, 2]]), 3, key=key, sampling=Sampling(temperature=0)
    ).tokens
    frozen = generate(
        model, reference, jnp.asarray([[1, 2]]), 3, key=key, sampling=Sampling(temperature=0)
    ).tokens
    assert not np.array_equal(expected, frozen)
    np.testing.assert_array_equal(preview.tokens, expected)
