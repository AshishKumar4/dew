"""Advantage estimators against hand computations and against two references.

Every function in `dew.rl.advantage` is checked twice: once against arithmetic
small enough to do on paper, written out in the test that does it, and once
against fixed tensors run through verl and Tunix by tools/parity_rl.py.

Tolerances and the largest differences observed, fp32 on CPU:

| quantity | tolerance | verl | Tunix |
| --- | --- | --- | --- |
| masked mean, per row | 2e-7 | 0.0 | 0.0 |
| masked whiten | 5e-7 | 2.4e-07 | 0.0 |
| group advantage, normalised | 2e-7 | 1.2e-07 | 1.2e-07 |
| group advantage, unnormalised | 2e-7 | 0.0 | 0.0 |
| RLOO advantage | 2e-7 | 1.2e-07 | 6.0e-08 |
| GAE advantage | 5e-7 | 2.4e-07 | 3.0e-08 |
| GAE return | 5e-7 | 1.2e-07 | 0.0 |

The residues are summation order in fp32: verl reduces in torch, Tunix in
numpy and JAX, Dew in JAX. Against Tunix, which reduces in JAX as Dew does,
three of the seven agree bit for bit. The two references disagree with each
other by 2.4e-07 on these inputs, which is the floor any port can reach.
"""

import importlib.util
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.rl.advantage import (
    gae, group_advantage, masked_mean, masked_whiten, rloo_advantage,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rl" / "advantage.npz"
GENERATOR = Path(__file__).resolve().parents[1] / "tools" / "parity_rl.py"
ADVANTAGE_TOLERANCE = 2e-7
GAE_TOLERANCE = 5e-7


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(FIXTURE))


@pytest.fixture(scope="module")
def generator():
    """tools/ holds scripts, not an importable package, and the module scope
    of the generator is constants and stubs, so loading it runs no reference."""
    spec = importlib.util.spec_from_file_location("parity_rl", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def largest(computed, expected):
    return float(np.max(np.abs(np.asarray(computed) - np.asarray(expected))))


def test_masked_mean_ignores_every_position_the_mask_drops():
    """A padded position holds whatever the allocator left there, so the mean
    of [1, 2, nan] under mask [1, 1, 0] is 1.5 and not a nan. Multiplying by a
    zero mask instead of replacing would return one here and poison the loss."""
    x = jnp.array([[1.0, 2.0, jnp.nan], [4.0, jnp.inf, -3.0]])
    mask = jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    assert masked_mean(x, mask, axis=-1).tolist() == [1.5, 4.0]
    assert float(masked_mean(x, mask)) == pytest.approx(7 / 3, abs=1e-6)


def test_masked_mean_of_nothing_is_zero_and_not_a_nan():
    """A fully masked row is a row nobody generated. Both references add 1e-8
    to the denominator so it reads zero, because a nan here spreads to every
    metric in the batch."""
    assert float(masked_mean(jnp.array([[5.0, 5.0]]), jnp.zeros((1, 2)))) == 0.0


def test_masked_mean_matches_the_references_per_row(reference):
    rows = masked_mean(jnp.asarray(reference["values"]),
                       jnp.asarray(reference["response_mask"]), axis=-1)

    for name in ("verl_masked_mean_rows", "tunix_masked_mean_rows"):
        difference = largest(rows, reference[name])
        assert difference < ADVANTAGE_TOLERANCE, f"{name}: {difference:.3e}"


def test_masked_whiten_centres_and_scales_by_the_unbiased_deviation():
    """Four values survive the mask, 1, 3, 5 and 7. Their mean is 4, their mean
    square deviation 5 and the Bessel correction 4/3, so the deviation is
    sqrt(20/3). The two positions outside the mask are scaled by the same
    numbers rather than zeroed, which is what both references return and what
    the loss masks again."""
    x = jnp.array([[1.0, 3.0, 100.0], [5.0, 7.0, -100.0]])
    mask = jnp.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    whitened = masked_whiten(x, mask)

    deviation = math.sqrt(20 / 3)
    assert np.allclose(whitened, np.array([[-3, -1, 96], [1, 3, -104]]) / deviation,
                       atol=1e-6)


def test_masked_whiten_matches_the_references(reference):
    whitened = masked_whiten(jnp.asarray(reference["values"]),
                             jnp.asarray(reference["response_mask"]))

    for name in ("verl_masked_whiten", "tunix_masked_whiten"):
        difference = largest(whitened, reference[name])
        assert difference < GAE_TOLERANCE, f"{name}: {difference:.3e}"


def test_group_advantage_is_the_group_mean_over_its_deviation():
    """One completion of four solves the task, so the mean is 0.25 and the
    unbiased deviation 0.5. The winner scores 0.75/0.500001 and the rest
    -0.25/0.500001. The 1e-6 is the reference's, and it is why these are not
    exactly 1.5 and -0.5."""
    advantages = group_advantage(jnp.array([1.0, 0.0, 0.0, 0.0]), 4)

    assert advantages.tolist() == pytest.approx(
        [0.75 / 0.500001] + [-0.25 / 0.500001] * 3, abs=1e-6)


def test_group_advantage_of_a_group_that_agrees_is_exactly_zero():
    """Every completion earning the same reward carries no signal. The eps
    under a zero deviation has to leave a zero numerator at zero, or a group
    the policy has mastered would push gradients around."""
    advantages = group_advantage(jnp.array([0.5, 0.5, 0.5, 0.5]), 4)

    assert advantages.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_group_advantage_unnormalised_only_subtracts_the_mean():
    """Dr.GRPO: mean 0.5625 subtracted, nothing divided."""
    advantages = group_advantage(jnp.array([2.0, -1.0, 0.25, 1.0]), 4,
                                 normalise_by_std=False)

    assert advantages.tolist() == pytest.approx([1.4375, -1.5625, -0.3125, 0.4375],
                                                abs=1e-7)


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


@pytest.mark.parametrize("estimator", [group_advantage, rloo_advantage])
def test_a_group_smaller_than_two_is_refused_by_name(estimator):
    """A group of one has no baseline. verl answers it with the raw reward,
    Tunix with a nan from ddof 1 or with zeros, and TRL with zeros. Three
    answers to one question means the question is wrong, and the question here
    is a group size, so this refuses and names it."""
    with pytest.raises(ValueError, match="group >= 2"):
        estimator(jnp.array([1.0, 0.0]), 1)


@pytest.mark.parametrize("estimator", [group_advantage, rloo_advantage])
def test_rewards_that_do_not_divide_into_groups_are_refused(estimator):
    with pytest.raises(ValueError, match="do not divide into groups of 4"):
        estimator(jnp.zeros(6), 4)


@pytest.mark.parametrize("estimator", [group_advantage, rloo_advantage])
def test_token_level_rewards_are_refused(estimator):
    """The batch carries one scalar per completion. A `[B, T]` column would
    reshape without complaint and mix tokens from different prompts into one
    group."""
    with pytest.raises(ValueError, match=r"\[B\]"):
        estimator(jnp.zeros((8, 4)), 4)


def test_rloo_advantage_is_the_reward_minus_the_rest_of_the_group():
    """Leave one out means the winner's baseline is the three zeros around it,
    so it scores 1, and each loser's baseline is 1/3."""
    advantages = rloo_advantage(jnp.array([1.0, 0.0, 0.0, 0.0]), 4)

    assert advantages.tolist() == pytest.approx([1.0, -1 / 3, -1 / 3, -1 / 3],
                                                abs=1e-7)


def test_rloo_advantage_matches_the_references(reference):
    advantages = rloo_advantage(jnp.asarray(reference["rewards"]),
                                int(reference["group"]))

    for name in ("verl_rloo_advantage", "tunix_rloo_advantage"):
        difference = largest(advantages, reference[name])
        assert difference < ADVANTAGE_TOLERANCE, f"{name}: {difference:.3e}"


def test_gae_follows_the_recursion_backwards_from_the_last_step():
    """gamma 0.5, lam 0.5, values [1, 2, 3], one reward of 1 at the end.

    delta_2 = 1 + 0 - 3 = -2, and A_2 = -2.
    delta_1 = 0 + 0.5*3 - 2 = -0.5, and A_1 = -0.5 + 0.25*(-2) = -1.
    delta_0 = 0 + 0.5*2 - 1 = 0, and A_0 = 0 + 0.25*(-1) = -0.25.

    The returns are those advantages plus the values, before any whitening,
    which is the order both references compute them in.
    """
    _, returns = gae(jnp.array([[0.0, 0.0, 1.0]]), jnp.array([[1.0, 2.0, 3.0]]),
                     jnp.ones((1, 3)), gamma=0.5, lam=0.5)

    assert returns[0].tolist() == pytest.approx([0.75, 1.0, 1.0], abs=1e-6)


def test_gae_whitens_the_advantages_over_the_mask():
    """Two rows at gamma and lam 1 with no critic. The first row's reward of 1
    walks back to every step and the second row's reward of 5 sits behind the
    mask, so the raw advantages are [1, 1, 1] and [0, 0, 0]. Over the five
    unmasked positions the mean is 0.6 and the corrected deviation sqrt(0.3),
    so the whitened values are 0.4/sqrt(0.3) and -0.6/sqrt(0.3)."""
    advantages, _ = gae(jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 5.0]]),
                        jnp.zeros((2, 3)),
                        jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
                        gamma=1.0, lam=1.0)

    deviation = math.sqrt(0.3)
    assert advantages[0].tolist() == pytest.approx([0.4 / deviation] * 3, abs=1e-6)
    assert advantages[1].tolist() == pytest.approx([-0.6 / deviation] * 3, abs=1e-6)


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


def test_the_estimators_are_jittable():
    """The rollout computes advantages on device, inside the same jit as the
    reward columns it just wrote, so a python branch on a traced value here
    would only show up there."""
    grouped = jax.jit(lambda rewards: group_advantage(rewards, 4))
    discounted = jax.jit(lambda r, v, m: gae(r, v, m, 0.99, 0.95))

    assert grouped(jnp.array([1.0, 0.0, 0.0, 0.0])).shape == (4,)
    assert discounted(jnp.zeros((2, 3)), jnp.zeros((2, 3)), jnp.ones((2, 3)))[0].shape == (2, 3)


def test_the_fixture_holds_the_inputs_the_generator_names(generator):
    """A fixture regenerates only if the inputs beside it are the inputs it was
    made from. Editing a reward in tools/parity_rl.py and forgetting to run it
    leaves this file comparing Dew against the reference's answer to a
    different question."""
    fixture = dict(np.load(FIXTURE))

    assert int(fixture["group"]) == generator.GROUP
    for name, expected in (("rewards", generator.REWARDS),
                           ("uids", generator.UIDS),
                           ("token_rewards", generator.TOKEN_REWARDS),
                           ("values", generator.VALUES),
                           ("response_mask", generator.RESPONSE_MASK)):
        assert np.array_equal(fixture[name], expected), name
    assert fixture["gamma"] == np.float32(generator.GAMMA)
    assert fixture["lam"] == np.float32(generator.LAM)
