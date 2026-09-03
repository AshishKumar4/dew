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

Tolerances and the largest differences observed against verl, fp32 on CPU:

| quantity | tolerance | observed |
| --- | --- | --- |
| clipped loss | 1e-7 | 4.5e-08 |
| clipped loss gradient | 1e-7 | 0.0 |
| pg_clipfrac, pg_clipfrac_lower | 1e-7 | 0.0 |
| ppo_kl | 1e-7 | 7.5e-09 |
| GSPO loss | 1e-7 | 1.5e-08 |
| GSPO loss gradient | 1e-7 | 0.0 |
| k3 KL, its penalty and its gradient | 1e-7 | 0.0 |
| token-mean aggregation | 1e-7 | 0.0 |

The k3 KL agrees with Tunix's JAX estimator to 0.0 as well. The two losses
differ in the last fp32 digit because torch sums the token matrix in a
different order.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.rl import surrogate

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rl" / "surrogate.npz"
REFERENCE = dict(np.load(FIXTURE))
TOLERANCE = 1e-7

LOG_PROBS = jnp.asarray(REFERENCE["log_probs"])
OLD_LOG_PROBS = jnp.asarray(REFERENCE["old_log_probs"])
REF_LOG_PROBS = jnp.asarray(REFERENCE["ref_log_probs"])
MASK = jnp.asarray(REFERENCE["response_mask"])
ADVANTAGES = jnp.asarray(REFERENCE["advantages"])
CLIP = dict(epsilon_low=float(REFERENCE["epsilon_low"]),
            epsilon_high=float(REFERENCE["epsilon_high"]),
            dual_clip=float(REFERENCE["dual_clip"]))


def difference(computed, name):
    return float(np.max(np.abs(np.asarray(computed, np.float64)
                               - np.asarray(REFERENCE[name], np.float64))))


def assert_matches(computed, name):
    observed = difference(computed, name)
    assert observed < TOLERANCE, f"{name}: {observed:.3e}"


def check_clip(module):
    """verl's clipped loss, its gradient and its three clip metrics."""
    def objective(log_probs):
        return module.clipped_surrogate(
            module.token_log_ratio(log_probs, OLD_LOG_PROBS),
            ADVANTAGES, MASK, **CLIP)

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
    gradient = jax.grad(lambda log_probs: module.token_mean(
        module.k3_kl(log_probs, REF_LOG_PROBS), MASK))(LOG_PROBS)

    assert_matches(kl, "verl_k3_kl")
    assert_matches(kl, "tunix_k3_kl")
    assert_matches(module.token_mean(kl, MASK), "verl_k3_penalty")
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


CHECKS = {"clip": check_clip, "k3": check_k3, "stop_gradient": check_stop_gradient}


def mutate(original, replacement):
    """The real module with one term rewritten, compiled under a new name."""
    source = Path(surrogate.__file__).read_text()
    assert source.count(original) == 1, (
        f"{original!r} appears {source.count(original)} times, so the mutation "
        f"no longer says what it used to")
    namespace = {"__name__": "dew.rl.surrogate_mutant"}
    exec(compile(source.replace(original, replacement), "<mutant>", "exec"), namespace)
    return SimpleNamespace(**namespace)


def assert_only_this_check_fails(mutant, broken):
    with pytest.raises(AssertionError):
        CHECKS[broken](mutant)
    for name, check in CHECKS.items():
        if name != broken:
            check(mutant)


def test_token_mean_divides_by_the_number_of_tokens_it_kept():
    """Three unmasked tokens summing to 8, and a nan behind the mask that a
    multiply by zero would turn into a nan loss."""
    x = jnp.array([[1.0, jnp.nan], [3.0, 4.0]])
    mask = jnp.array([[1.0, 0.0], [1.0, 1.0]])

    assert float(surrogate.token_mean(x, mask)) == pytest.approx(8 / 3, abs=1e-6)


def test_token_mean_matches_verls_aggregation():
    assert_matches(surrogate.token_mean(LOG_PROBS, MASK), "verl_token_mean_log_probs")


def test_token_log_ratio_clamps_what_it_is_about_to_exponentiate():
    """A token the policy now gives 30 nats more probability than the rollout
    did would exponentiate to 1e13 and take the step with it."""
    ratio = surrogate.token_log_ratio(jnp.array([[-1.0, -0.5, 30.0]]),
                                      jnp.array([[-1.5, -0.5, 0.0]]))

    assert ratio[0].tolist() == pytest.approx([0.5, 0.0, 20.0], abs=1e-6)


def test_the_clipped_loss_and_its_metrics_match_a_hand_computation():
    """Log ratios 0, 0.5, -0.4 against advantage +1, and 0, 1.5, -0.4 against
    advantage -1, with the last token of the second row masked. The band is
    [0.8, 1.5] and the dual clip is 3.

    The first row's ratios are 1, 1.6487, 0.6703. The middle one leaves the
    band, so the clipped term -1.5 is the larger loss and the clip bites; the
    last one leaves it downwards, where for a positive advantage the unclipped
    term is still the larger, so it does not.

    The second row's advantage is negative, so its ratios raise the loss. The
    middle ratio 4.4817 gives 4.4817, above the dual clip's 3, which caps it.

    That is (-1 - 1.5 - 0.6703 + 1 + 3) / 5 over the five unmasked tokens, one
    clipped token in five, one dual-clipped token in five, and a ppo_kl of
    -(0 + 0.5 - 0.4 + 0 + 1.5) / 5.
    """
    log_ratio = jnp.array([[0.0, 0.5, -0.4], [0.0, 1.5, -0.4]])
    mask = jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    loss, aux = surrogate.clipped_surrogate(
        log_ratio, jnp.array([1.0, -1.0]), mask,
        epsilon_low=0.2, epsilon_high=0.5, dual_clip=3.0)

    assert float(loss) == pytest.approx(
        (-1 - 1.5 - math.exp(-0.4) + 1 + 3) / 5, abs=1e-6)
    assert float(aux["pg_clipfrac"]) == pytest.approx(0.2, abs=1e-6)
    assert float(aux["pg_clipfrac_lower"]) == pytest.approx(0.2, abs=1e-6)
    assert float(aux["ppo_kl"]) == pytest.approx(-0.32, abs=1e-6)


def test_the_clipped_loss_and_its_gradient_match_verl():
    check_clip(surrogate)


def test_a_dual_clip_that_cannot_bind_is_refused():
    """verl asserts the same bound. Below 1 the cap would sit inside the
    clipping band and replace the surrogate with a constant."""
    with pytest.raises(ValueError, match="dual_clip > 1"):
        surrogate.clipped_surrogate(jnp.zeros((1, 2)), jnp.zeros(1),
                                    jnp.ones((1, 2)), dual_clip=1.0)


def test_the_k3_estimator_matches_a_hand_computation():
    """exp(d) - d - 1 for d = log pi_ref - log pi: 0.5 gives 0.1487 and -2
    gives 1.1353. Both are positive, which the plain log ratio is not."""
    kl = surrogate.k3_kl(jnp.array([[-1.0, -1.0]]), jnp.array([[-0.5, -3.0]]))

    assert kl[0].tolist() == pytest.approx(
        [math.exp(0.5) - 1.5, math.exp(-2.0) + 1.0], abs=1e-6)


def test_the_k3_estimator_and_its_gradient_match_verl_and_tunix():
    check_k3(surrogate)


def test_the_k3_estimator_bounds_a_token_that_drifted():
    """Differences of 30, -5 and 2.5 nats. verl clamps the exponent at 20 and
    the estimate at 10, so the first reads 10 instead of 1e13 and the third,
    at 8.68, is left alone."""
    kl = surrogate.k3_kl(jnp.asarray(REFERENCE["extreme_log_probs"]),
                         jnp.asarray(REFERENCE["extreme_ref_log_probs"]))

    assert_matches(kl, "verl_k3_kl_extreme")
    assert kl[0].tolist() == pytest.approx(
        [10.0, math.exp(-5.0) + 4.0, math.exp(2.5) - 3.5], abs=1e-5)


def test_the_sequence_ratio_pools_the_value_and_keeps_the_gradient_per_token():
    check_stop_gradient(surrogate)


def test_the_gspo_loss_and_its_gradient_match_verl():
    """The same clip, fed the sequence ratio instead of the token ratio, is
    verl's `compute_policy_loss_gspo` aggregated token-mean."""
    def objective(log_probs):
        return surrogate.clipped_surrogate(
            surrogate.sequence_log_ratio(log_probs, OLD_LOG_PROBS, MASK),
            ADVANTAGES, MASK, **CLIP)

    loss, aux = objective(LOG_PROBS)
    gradient = jax.grad(lambda log_probs: objective(log_probs)[0])(LOG_PROBS)

    assert_matches(loss, "verl_gspo_loss")
    assert_matches(gradient, "verl_gspo_grad")
    assert_matches(aux["pg_clipfrac"], "verl_gspo_pg_clipfrac")


def test_dropping_the_clip_fails_only_the_clip_check():
    """Without the maximum the surrogate is the unclipped policy gradient. It
    still runs, still trains, and lets one token's ratio move the policy as far
    as it likes."""
    assert_only_this_check_fails(
        mutate("worse = jnp.maximum(unclipped, clipped)", "worse = unclipped"),
        "clip")


def test_dropping_the_k3_correction_fails_only_the_k3_check():
    """`-d` is the same KL in expectation and a different number on every
    batch, with higher variance and a negative value wherever the policy moved
    towards the reference."""
    assert_only_this_check_fails(
        mutate("jnp.exp(diff) - diff - 1", "-diff"), "k3")


def test_flipping_the_stop_gradient_fails_only_the_gradient_check():
    """The reversed subtraction is worth a test of its own because it is
    invisible in the loss. Every value the mutant produces is the value the
    real module produces, and every gradient has the wrong sign."""
    mutant = mutate("log_probs - jax.lax.stop_gradient(log_probs)",
                    "jax.lax.stop_gradient(log_probs) - log_probs")

    pooled = mutant.sequence_log_ratio(LOG_PROBS, OLD_LOG_PROBS, MASK)
    assert np.array_equal(
        np.asarray(pooled),
        np.asarray(surrogate.sequence_log_ratio(LOG_PROBS, OLD_LOG_PROBS, MASK)))

    assert_only_this_check_fails(mutant, "stop_gradient")
