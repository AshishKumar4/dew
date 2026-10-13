"""Advantage estimators against hand computations and against two references.

Every function in `dew.rl.advantage` is checked twice: once against arithmetic
small enough to do on paper, written out in the test that does it, and once
against fixed tensors run through verl and Tunix by tools/parity_rl.py.

Tolerances and the largest differences observed, fp32 on CPU:

| quantity | tolerance | verl | Tunix |
| --- | --- | --- | --- |
| masked mean, per row | 2e-7 | 0.0 | 0.0 |
| masked whiten, kept positions | 5e-7 | 2.4e-07 | 0.0 |
| group advantage, normalised | 2e-7 | 1.2e-07 | 1.2e-07 |
| group advantage, unnormalised | 2e-7 | 0.0 | 0.0 |
| RLOO advantage | 2e-7 | 1.2e-07 | 6.0e-08 |
| GAE advantage | 5e-7 | 2.4e-07 | 3.0e-08 |
| GAE return | 5e-7 | 1.2e-07 | 0.0 |

Whitening also checks every output against scalar float64 arithmetic. Its
padding values whiten to about +/-20, where one fp32 ULP is 1.9e-6; the
loss-consumed entries retain the absolute reference bound above. Padding
instead uses the equation's forward-error bound, not a machine's snapshot.
"""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.rl.advantage import gae, group_advantage, masked_mean, masked_whiten, rloo_advantage

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rl" / "advantage.npz"
ADVANTAGE_TOLERANCE = 2e-7
GAE_TOLERANCE = 5e-7


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(FIXTURE))


def largest(computed, expected):
    return float(np.max(np.abs(np.asarray(computed) - np.asarray(expected))))


def test_masked_mean_matches_the_references_per_row(reference):
    rows = masked_mean(jnp.asarray(reference["values"]),
                       jnp.asarray(reference["response_mask"]), axis=-1)

    for name in ("verl_masked_mean_rows", "tunix_masked_mean_rows"):
        difference = largest(rows, reference[name])
        assert difference < ADVANTAGE_TOLERANCE, f"{name}: {difference:.3e}"


@pytest.mark.parametrize("compiled", [False, True])
def test_masked_whiten_matches_the_references(reference, compiled):
    x = reference["values"].astype(np.float64)
    keep = reference["response_mask"].astype(bool)
    whiten = jax.jit(masked_whiten) if compiled else masked_whiten
    whitened = np.asarray(whiten(jnp.asarray(reference["values"]),
                                jnp.asarray(reference["response_mask"])))

    count = int(keep.sum())
    mean = math.fsum(x[keep]) / (count + 1e-8)
    variance = math.fsum((value - mean) ** 2 for value in x[keep])
    variance = variance / (count + 1e-8) * count / (count - 1)
    deviation = math.sqrt(variance + 1e-8)
    expected = (x - mean) / deviation
    # Conservative gamma_(n+8) envelope for this fixed, nondegenerate
    # binary-mask fixture, not a universal whitening error theorem. It
    # covers reductions, centring, Bessel correction and rsqrt/multiply;
    # the mean's absolute input scale protects near-zero centered outputs.
    unit = np.finfo(np.float32).eps / 2
    gamma = (count + 8) * unit / (1 - (count + 8) * unit)
    bound = gamma * (np.abs(x - mean) + math.fsum(abs(x[keep])) / count) / deviation
    for actual in (whitened, reference["verl_masked_whiten"],
                   reference["tunix_masked_whiten"]):
        assert np.all(np.abs(actual.astype(np.float64) - expected) <= bound)
    for name in ("verl_masked_whiten", "tunix_masked_whiten"):
        difference = largest(whitened[keep], reference[name][keep])
        assert difference < GAE_TOLERANCE, f"{name}: {difference:.3e}"

    values = jnp.array([[1.0, 3.0, 100.0], [5.0, 7.0, -100.0]])
    mask = jnp.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    assert np.allclose(whiten(values, mask),
                        np.array([[-3, -1, 96], [1, 3, -104]]) / math.sqrt(20 / 3), atol=1e-6)


@pytest.mark.parametrize("normalise", [True, False])
def test_group_advantage_matches_the_references(reference, normalise):
    advantages = group_advantage(jnp.asarray(reference["rewards"]),
                                 int(reference["group"]),
                                 normalise_by_std=normalise)

    suffix = "" if normalise else "_unnormalised"
    for source in ("verl", "tunix"):
        name = f"{source}_group_advantage{suffix}"
        difference = largest(advantages, reference[name])
        assert difference < ADVANTAGE_TOLERANCE, f"{name}: {difference:.3e}"

    if normalise:
        hand = group_advantage(jnp.array([1.0, 0.0, 0.0, 0.0]), 4)
        assert hand.tolist() == pytest.approx([0.75 / 0.500001] + [-0.25 / 0.500001] * 3, abs=1e-6)
    else:
        hand = group_advantage(jnp.array([2.0, -1.0, 0.25, 1.0]), 4, normalise_by_std=False)
        assert hand.tolist() == pytest.approx([1.4375, -1.5625, -0.3125, 0.4375], abs=1e-7)


def test_rloo_advantage_matches_the_references(reference):
    advantages = rloo_advantage(jnp.asarray(reference["rewards"]),
                                int(reference["group"]))

    for name in ("verl_rloo_advantage", "tunix_rloo_advantage"):
        difference = largest(advantages, reference[name])
        assert difference < ADVANTAGE_TOLERANCE, f"{name}: {difference:.3e}"

    hand = rloo_advantage(jnp.array([1.0, 0.0, 0.0, 0.0]), 4)
    assert hand.tolist() == pytest.approx([1.0, -1 / 3, -1 / 3, -1 / 3], abs=1e-7)


def test_gae_reads_nothing_behind_the_mask(reference):
    """Replace every value and every reward the mask drops, and no advantage
    the loss will read moves. A recursion that discounts through padding fails
    this and passes every test that only feeds it rectangular batches."""
    mask = jnp.asarray(reference["response_mask"])
    rewards = jnp.asarray(reference["token_rewards"])
    values = jnp.asarray(reference["values"])
    keep = mask > 0

    advantages, returns = gae(rewards, values, mask, gamma=0.99, lam=0.95)
    poisoned, poisoned_returns = gae(jnp.where(keep, rewards, -13.0),
                                     jnp.where(keep, values, 41.0),
                                     mask, gamma=0.99, lam=0.95)

    assert np.array_equal(np.asarray(advantages)[keep], np.asarray(poisoned)[keep])
    assert np.array_equal(np.asarray(returns)[keep],
                          np.asarray(poisoned_returns)[keep])


def test_gae_matches_the_references(reference):
    advantages, returns = gae(jnp.asarray(reference["token_rewards"]),
                              jnp.asarray(reference["values"]),
                              jnp.asarray(reference["response_mask"]),
                              gamma=float(reference["gamma"]),
                              lam=float(reference["lam"]))

    for source in ("verl", "tunix"):
        for computed, name in ((advantages, f"{source}_gae_advantage"),
                               (returns, f"{source}_gae_return")):
            difference = largest(computed, reference[name])
            assert difference < GAE_TOLERANCE, f"{name}: {difference:.3e}"

    _, hand_returns = gae(jnp.array([[0.0, 0.0, 1.0]]), jnp.array([[1.0, 2.0, 3.0]]),
                          jnp.ones((1, 3)), gamma=0.5, lam=0.5)
    assert hand_returns[0].tolist() == pytest.approx([0.75, 1.0, 1.0], abs=1e-6)
    hand_advantages, _ = gae(jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 5.0]]), jnp.zeros((2, 3)),
                              jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]), gamma=1.0, lam=1.0)
    assert hand_advantages[0].tolist() == pytest.approx([0.4 / math.sqrt(0.3)] * 3, abs=1e-6)
    assert hand_advantages[1].tolist() == pytest.approx([-0.6 / math.sqrt(0.3)] * 3, abs=1e-6)
