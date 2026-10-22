"""Advantage estimators against hand computations and against two references.

Every function in `dew.rl.advantage` is checked twice: once against arithmetic
small enough to do on paper, exactly, and once on par with verl and Tunix from
verl's float64 run over drawn rewards, values and token rewards (check 1,
tests/reference_error.py), recorded by tools/parity_rl.py, the masks and
groups the fixed fixtures' (tools/parity_rl.py `advantage_inputs`).
"""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording

from dew.rl.advantage import gae, group_advantage, rloo_advantage
from tools.parity_rl import advantage_inputs, dew_estimators

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rl" / "advantage.npz"
RECORDED = Path(__file__).resolve().parent / "fixtures" / "rl"


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(FIXTURE))


estimators = jax.jit(dew_estimators)
"""Every estimator Dew has on a recorded draw, by the recording's leaf names."""


@pytest.mark.parametrize("source", ["verl", "tunix"])
def test_the_estimators_are_as_exact_as_the_references(source):
    """Masked mean and whitening, group and leave-one-out advantages and GAE's advantages and returns,
    on par with the reference's from verl's float64 run over its recorded seeds (check 1), padding
    positions included; the truth Dew's own estimators in float64, held at every seed to the
    recording's sketch of verl's."""
    path = RECORDED / f"recorded_advantage_{source}.npz"
    seeds = declared_seeds(path)[source]
    draw, recorded, truth = indexed_recording(recording(path), source, seeds, inputs_of=advantage_inputs,
                                              truth_fn=estimators)
    assert_as_exact(lambda index, inputs: estimators(inputs), recorded, draw, f"{source} estimators",
                    seeds=seeds, truth_fn=truth)


def test_the_estimators_on_paper():
    """Arithmetic small enough to do on paper, exact in float32: a group's advantages centred and,
    unnormalised, its mean taken off; the leave-one-out baseline; and returns discounted by halves."""
    hand = group_advantage(jnp.array([2.0, -1.0, 0.25, 1.0], jnp.float32), 4, normalise_by_std=False)
    np.testing.assert_array_equal(hand, [1.4375, -1.5625, -0.3125, 0.4375])
    hand = rloo_advantage(jnp.array([1.0, 0.0, 0.0, 0.0], jnp.float32), 4)
    np.testing.assert_array_equal(hand, [1.0, -np.float32(1 / 3), -np.float32(1 / 3), -np.float32(1 / 3)])
    rewards, values = jnp.array([[0.0, 0.0, 1.0]], jnp.float32), jnp.array([[1.0, 2.0, 3.0]], jnp.float32)
    _, hand_returns = gae(rewards, values, jnp.ones((1, 3), jnp.float32), gamma=0.5, lam=0.5)
    np.testing.assert_array_equal(hand_returns[0], [0.75, 1.0, 1.0])


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


@pytest.mark.parametrize("padding", [math.nan, math.inf], ids=["nan", "inf"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_gae_reads_no_nonfinite_padding(reference, padding, compiled):
    """Padding the mask drops holds NaN or infinity, as an uninitialised
    buffer can: every advantage and return the loss reads is the one
    finite padding gives, eager and compiled."""
    mask = jnp.asarray(reference["response_mask"])
    rewards = jnp.asarray(reference["token_rewards"])
    values = jnp.asarray(reference["values"])
    keep = mask > 0
    assert not bool(jnp.all(keep)), "the reference batch has no padding to fill"
    run = jax.jit(gae, static_argnames=("gamma", "lam")) if compiled else gae

    advantages, returns = run(rewards, values, mask, gamma=0.99, lam=0.95)
    padded, padded_returns = run(jnp.where(keep, rewards, padding), jnp.where(keep, values, padding),
                                 mask, gamma=0.99, lam=0.95)

    assert np.array_equal(np.asarray(advantages)[keep], np.asarray(padded)[keep])
    assert np.array_equal(np.asarray(returns)[keep], np.asarray(padded_returns)[keep])
