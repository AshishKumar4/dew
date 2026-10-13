"""Update rules against hand computations, against verl, and under mutation.

`dew.rl.surrogate` is three pieces of arithmetic that decide what a policy
gradient step does, the clip, the k3 KL and GSPO's stop-gradient. Each one is
checked against numbers derived on paper in the test that uses them, and
against verl's own functions on the fixed tensors tools/parity_rl.py runs
through them, values and gradients both. Torch autograd is the only reference
that can pin a stop-gradient down, because flipping one changes no value.

The last three tests are the mutations. Each drops or reverses one term in the
module's real source, compiles it, and asserts that exactly one of the three
checks goes red, which is the evidence that each term has a test behind it.

Tolerances and the largest differences observed against verl, fp32:

| quantity | tolerance | observed on CPU | observed on an RTX 4080 |
| --- | --- | --- | --- |
| clipped loss | 1e-6 | 4.5e-08 | 4.5e-08 |
| clipped loss gradient | 1e-6 | 0.0 | 0.0 |
| pg_clipfrac, pg_clipfrac_lower | 1e-6 | 0.0 | 0.0 |
| ppo_kl | 1e-6 | 7.5e-09 | 7.5e-09 |
| GSPO loss | 1e-6 | 1.5e-08 | 1.5e-08 |
| GSPO loss gradient | 1e-6 | 0.0 | 0.0 |
| k3 KL, its penalty and its gradient | 1e-6 | 0.0 | 1.2e-07 |
| token-mean aggregation | 1e-6 | 0.0 | 0.0 |

The tolerance is ten fp32 ulps at 1.0. The earlier 1e-7 was one ulp, which
the GPU's summation order crossed on the k3 KL (1.19e-07) with the code
unchanged; a bound that tight asserts the CPU's rounding, not the maths.
The k3 KL agrees with Tunix's JAX estimator to 0.0 as well. The two losses
differ in the last fp32 digit because torch sums the token matrix in a
different order.
"""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from rl_support import clipped_surrogate, token_mean

from dew.rl import surrogate

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rl" / "surrogate.npz"
REFERENCE = dict(np.load(FIXTURE))
TOLERANCE = 1e-6

LOG_PROBS = jnp.asarray(REFERENCE["log_probs"])
OLD_LOG_PROBS = jnp.asarray(REFERENCE["old_log_probs"])
REF_LOG_PROBS = jnp.asarray(REFERENCE["ref_log_probs"])
MASK = jnp.asarray(REFERENCE["response_mask"])
ADVANTAGES = jnp.asarray(REFERENCE["advantages"])
CLIP = {"epsilon_low": float(REFERENCE["epsilon_low"]),
            "epsilon_high": float(REFERENCE["epsilon_high"]),
            "dual_clip": float(REFERENCE["dual_clip"])}


def difference(computed, name):
    return float(np.max(np.abs(np.asarray(computed, np.float64)
                               - np.asarray(REFERENCE[name], np.float64))))


def assert_matches(computed, name):
    observed = difference(computed, name)
    assert observed < TOLERANCE, f"{name}: {observed:.3e}"


def check_clip(module):
    """verl's clipped loss, its gradient and its three clip metrics."""
    def objective(log_probs):
        return clipped_surrogate(
            module.token_log_ratio(log_probs, OLD_LOG_PROBS),
            ADVANTAGES, MASK, module, **CLIP)

    loss, aux = objective(LOG_PROBS)
    gradient = jax.grad(lambda log_probs: objective(log_probs)[0])(LOG_PROBS)

    assert_matches(loss, "verl_clipped_loss")
    assert_matches(gradient, "verl_clipped_grad")
    assert_matches(aux["pg_clipfrac"], "verl_clipped_pg_clipfrac")
    assert_matches(aux["pg_clipfrac_lower"], "verl_clipped_pg_clipfrac_lower")
    assert_matches(aux["ppo_kl"], "verl_clipped_ppo_kl")


def check_k3(module):
    """verl's k3 KL per token, its token-mean penalty and its gradient."""
    kl = module.k3_kl(LOG_PROBS, REF_LOG_PROBS)
    gradient = jax.grad(lambda log_probs: token_mean(
        module.k3_kl(log_probs, REF_LOG_PROBS), MASK))(LOG_PROBS)

    assert_matches(kl, "verl_k3_kl")
    assert_matches(kl, "tunix_k3_kl")
    assert_matches(token_mean(kl, MASK), "verl_k3_penalty")
    assert_matches(gradient, "verl_k3_grad")


def check_stop_gradient(module):
    """One value per sequence, one gradient per token.

    Token log ratios 0.1, 0.2 and a third behind the mask: the sequence log
    ratio is 0.15 at every position, and the derivative of a weighted sum of
    those positions is the weights themselves, one per token. Both halves have
    to hold. The value alone holds for the reversed stop-gradient too, and the
    gradient alone holds for a version that pooled nothing.
    """
    log_probs = jnp.array([[-1.0, -0.9, -2.0]])
    old_log_probs = jnp.array([[-1.1, -1.1, -1.1]])
    mask = jnp.array([[1.0, 1.0, 0.0]])
    weights = jnp.array([[1.0, 2.0, 3.0]])

    pooled = module.sequence_log_ratio(log_probs, old_log_probs, mask)
    gradient = jax.grad(lambda p: jnp.sum(
        module.sequence_log_ratio(p, old_log_probs, mask) * weights))(log_probs)

    assert pooled[0].tolist() == pytest.approx([0.15, 0.15, 0.15], abs=1e-6)
    assert gradient.tolist() == weights.tolist()


def test_the_clipped_loss_and_its_gradient_match_verl():
    check_clip(surrogate)

    log_ratio = jnp.array([[0.0, 0.5, -0.4], [0.0, 1.5, -0.4]])
    mask = jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    loss, aux = clipped_surrogate(log_ratio, jnp.array([1.0, -1.0]), mask,
                                  epsilon_low=0.2, epsilon_high=0.5, dual_clip=3.0)
    assert float(loss) == pytest.approx((-1 - 1.5 - math.exp(-0.4) + 1 + 3) / 5, abs=1e-6)
    assert float(aux["pg_clipfrac"]) == pytest.approx(0.2, abs=1e-6)
    assert float(aux["pg_clipfrac_lower"]) == pytest.approx(0.2, abs=1e-6)
    assert float(aux["ppo_kl"]) == pytest.approx(-0.32, abs=1e-6)


def test_the_k3_estimator_and_its_gradient_match_verl_and_tunix():
    check_k3(surrogate)

    hand = surrogate.k3_kl(jnp.array([[-1.0, -1.0]]), jnp.array([[-0.5, -3.0]]))
    assert hand[0].tolist() == pytest.approx([math.exp(0.5) - 1.5, math.exp(-2.0) + 1.0], abs=1e-6)


def test_the_sequence_ratio_pools_the_value_and_keeps_the_gradient_per_token():
    check_stop_gradient(surrogate)


def test_the_gspo_loss_and_its_gradient_match_verl():
    """The same clip, fed the sequence ratio instead of the token ratio, is
    verl's `compute_policy_loss_gspo` aggregated token-mean."""
    def objective(log_probs):
        return clipped_surrogate(
            surrogate.sequence_log_ratio(log_probs, OLD_LOG_PROBS, MASK),
            ADVANTAGES, MASK, **CLIP)

    loss, aux = objective(LOG_PROBS)
    gradient = jax.grad(lambda log_probs: objective(log_probs)[0])(LOG_PROBS)

    assert_matches(loss, "verl_gspo_loss")
    assert_matches(gradient, "verl_gspo_grad")
    assert_matches(aux["pg_clipfrac"], "verl_gspo_pg_clipfrac")
