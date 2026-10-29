"""Each named RL method (dew.objectives.rl.methods) against its paper's objective, written apart from Dew
over each session's tokens: the policy term's value and its gradient in each token's log-probability,
over drawn sessions packed two to a row, Dew run in float64 within float64 rounding of the paper's form
(`assert_computes_the_oracle`), where any difference in the function moves it by orders of magnitude;
the global estimator and DAPO's overlong punishment by their equations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_computes_the_oracle
from rl_support import TinyHead

from dew.objectives.rl.grpo import _Terms
from dew.objectives.rl.methods import METHODS, method
from dew.objectives.rl.sessions import Call, Session, Status, advantages, overlong_penalty

LENGTHS = ((3, 5), (4, 2))
"""Each row's two sessions' trained token counts."""
WIDTH = 12
DRAWS = 8
"""The drawn batches each method is compared over."""
ROUNDINGS = 4 * WIDTH
"""The longest chain of float64 roundings a term goes through: a session's tokens summed, its ratio
exponentiated and clipped, the sessions summed, each a few roundings."""


def draw(seed):
    """A row grid of two sessions a row, their current and old log-probabilities, each session's
    advantage on its tokens and its session weight (one over its token count)."""
    rng = np.random.default_rng([858, seed])
    rows = len(LENGTHS)
    segments = np.zeros((rows, WIDTH), np.int32)
    mask = np.zeros((rows, WIDTH), np.float32)
    advantage = np.zeros((rows, WIDTH), np.float32)
    weights = np.zeros((rows, WIDTH), np.float32)
    for row, lengths in enumerate(LENGTHS):
        start = 0
        for session, length in enumerate(lengths, start=1):
            span = slice(start, start + length)
            segments[row, span], mask[row, span] = session, 1
            advantage[row, span] = rng.normal()
            weights[row, span] = 1 / length
            start += length
    old = -rng.uniform(0.5, 3.0, (rows, WIDTH)).astype(np.float32)
    policy = (old + rng.normal(0, 0.25, (rows, WIDTH))).astype(np.float32)
    return policy, old, advantage, mask, segments, weights


def sessions_of(mask, segments):
    """Each session's token positions, `(row, columns)`."""
    return [(row, np.flatnonzero((segments[row] == session) & (mask[row] > 0)))
            for row in range(mask.shape[0]) for session in np.unique(segments[row]) if session]


def dew_term(name, **settings):
    """The method's policy term over a draw, through `GRPOObjective.policy_ratio`, and its gradient in
    the current log-probabilities, in float64."""
    objective, _ = method(name).build(TinyHead(vocab_size=8), 7, max_new_tokens=8, **settings)

    def term(policy, old, advantage, mask, segments, weights):
        wide = {name: jnp.asarray(value, jnp.float64) for name, value in
                (("old", old), ("advantage", advantage), ("mask", mask), ("weights", weights))}

        def value(current):
            terms = _Terms(current, wide["old"], wide["old"], wide["advantage"], wide["mask"],
                           jnp.asarray(segments), wide["weights"], proximal=False, qk_stats=None,
                           routing=None)
            ratio, *_ = objective.policy_ratio(terms, wide["mask"], jnp.ones_like(wide["mask"]))
            return ratio.total / ratio.mass
        current = jnp.asarray(policy, jnp.float64)
        return jax.device_get((value(current), jax.grad(value)(current)))
    return term


def paper(objective):
    """A paper's objective `objective(per-session arrays) -> loss` and its gradient in float64, by
    autodiff of the formula written over each session's tokens alone."""
    def term(policy, old, advantage, mask, segments, weights):
        spans = sessions_of(np.asarray(mask), np.asarray(segments))

        def value(current):
            return objective([(current[row, columns], jnp.asarray(old, jnp.float64)[row, columns],
                               jnp.asarray(np.asarray(advantage)[row, columns[0]], jnp.float64))
                              for row, columns in spans])
        current = jnp.asarray(policy, jnp.float64)
        return jax.device_get((value(current), jax.grad(value)(current)))
    return term


def clipped(ratio, advantage, low, high):
    return jnp.minimum(ratio * advantage, jnp.clip(ratio, 1 - low, 1 + high) * advantage)


def grpo(sessions, low=0.2, high=0.2):
    # arXiv:2402.03300 equation 3, the penalty apart: (1/G) sum_i (1/|o_i|) sum_t min(r A, clip(r) A).
    return -sum(jnp.mean(clipped(jnp.exp(p - o), a, low, high)) for p, o, a in sessions) / len(sessions)


def dapo(sessions):
    # arXiv:2503.14476 equation 10: one over every token of the group, clip-higher (0.2, 0.28).
    total = sum(jnp.sum(clipped(jnp.exp(p - o), a, 0.2, 0.28)) for p, o, a in sessions)
    return -total / sum(len(p) for p, _, _ in sessions)


def dr_grpo(sessions, budget=8):
    # arXiv:2503.20783 section 3.2: GRPO with |o_i| replaced by the generation budget.
    total = sum(jnp.sum(clipped(jnp.exp(p - o), a, 0.2, 0.2)) for p, o, a in sessions)
    return -total / budget / len(sessions)


def gspo(sessions, low=0.2, high=0.2):
    # arXiv:2507.18071 equation 5: s_i = (pi / pi_old)^(1/|y_i|), clipped at the sequence.
    return -sum(clipped(jnp.exp(jnp.mean(p - o)), a, low, high) for p, o, a in sessions) / len(sessions)


def cispo(sessions, high=0.2):
    # arXiv:2506.13585 equation 4: sg(clip(r, 0, 1 + eps_high)) A log pi, one over every token.
    total = sum(jnp.sum(jax.lax.stop_gradient(jnp.clip(jnp.exp(p - o), 0, 1 + high)) * a * p)
                for p, o, a in sessions)
    return -total / sum(len(p) for p, _, _ in sessions)


def gmpo(sessions):
    # arXiv:2507.20673 equation 4: (prod_t min(r A, clip(r, e^-0.4, e^0.4) A))^(1/|o|) sgn(A).
    def one(p, o, a):
        ratio = jnp.exp(p - o)
        kept = jnp.minimum(ratio * a, jnp.clip(ratio, np.exp(-0.4), np.exp(0.4)) * a)
        return jnp.exp(jnp.mean(jnp.log(jnp.abs(kept)))) * np.sign(a)
    return -sum(one(*session) for session in sessions) / len(sessions)


def rloo(sessions):
    # arXiv:2402.14740 equation 6: (1/k) sum_i A_i grad log pi(y_i), a sequence's log-likelihood its
    # tokens' sum.
    return -sum(a * jnp.sum(p) for p, _, a in sessions) / len(sessions)


@pytest.mark.parametrize(("name", "formula", "settings"), [
    ("grpo", grpo, {}), ("dapo", dapo, {}), ("dr_grpo", dr_grpo, {}), ("gspo", gspo, {}),
    ("cispo", cispo, {"epsilon_high": 0.2}), ("gmpo", gmpo, {}), ("reinforce_pp", grpo, {}),
    ("rloo", rloo, {})], ids=lambda value: value if isinstance(value, str) else "")
def test_each_method_is_its_papers_objective(name, formula, settings):
    """The method's policy term and its gradient in each token's log-probability are its paper's
    objective's, within float64 rounding of it over every draw."""
    with jax.enable_x64(new_val=True):
        for seed in range(DRAWS):
            drawn = draw(seed)
            found, expected = dew_term(name, **settings)(*drawn), paper(formula)(*drawn)
            for leaf, (ours, theirs) in enumerate(zip(found, expected, strict=True)):
                assert_computes_the_oracle(np.asarray(ours), np.asarray(theirs),
                                           f"{name}[{leaf}] draw {seed}", roundings=ROUNDINGS)


def test_a_method_fixes_its_composition_and_refuses_what_it_lacks():
    """A method refuses an override of a part it fixes, a reference penalty where it has none, an
    unknown name, and a method that reads the response budget without one."""
    model = TinyHead(vocab_size=8)
    with pytest.raises(ValueError):
        METHODS["dr_grpo"].build(model, 8, max_new_tokens=8, aggregation="token-mean")
    with pytest.raises(ValueError):
        METHODS["gspo"].build(model, 8, beta=0.1)
    with pytest.raises(ValueError):
        METHODS["dapo"].build(model, 8)
    with pytest.raises(ValueError):
        method("ppo-lite")


def test_the_global_estimator_centres_each_group_then_whitens_the_batch():
    """REINFORCE++ with a baseline (arXiv:2501.03262 equations 6-7): each member's reward less its group's
    mean, the batch of those whitened, with the paper's epsilon."""
    rewards = {("a", "g"): [1.0, 0.0, 0.5], ("b", "g"): [2.0, 2.0, -1.0]}
    call = Call((1, 2), (3,), (-0.1,), "stop", 0)
    sessions = [Session(task, group, sample, 0, (call,), Status.COMPLETED, reward)
                for (task, group), values in rewards.items() for sample, reward in enumerate(values)]
    found = np.asarray([values[0] for values in advantages(sessions, "global")])
    centred = np.concatenate([np.asarray(values) - np.mean(values) for values in rewards.values()])
    expected = (centred - centred.mean()) / (centred.std() + 1e-8)
    # Each group's mean is exact in float32 on these rewards, so the whitening alone rounds, in float64.
    np.testing.assert_array_equal(found, expected.astype(np.float32))


@pytest.mark.parametrize(("length", "penalty"), [(4, 0.0), (6, 0.0), (7, -0.5), (8, -1.0), (9, -1.0)])
def test_dapos_overlong_punishment_is_its_equation(length, penalty):
    """DAPO's soft overlong punishment (arXiv:2503.14476 equation 13) at a budget of 8 with a cache of 2:
    none to 6 tokens, -(length - 6)/2 to 8, -1 past it."""
    assert overlong_penalty(length, (8, 2)) == penalty
