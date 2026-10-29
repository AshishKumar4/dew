"""Each named RL method (dew.objectives.rl.methods) against its paper's objective, written apart from Dew
in float64: each method's loss term, its reference penalty in the loss included, and its gradient in each
token's log-probability, over drawn sessions packed two to a row, on par with the paper's form by check 1
(tests/reference_error.py). Through a scheduler's call over scripted sessions, the returns the estimators
compare: DAPO's overlong punishment and dynamic sampling, RLOO's reference penalty in the reward, and the
global estimator's whitening over a batch split into data shares."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from reference_error import assert_as_exact
from rl_support import TinyHead

from dew.objectives.rl import RolloutScheduler, Synchronous
from dew.objectives.rl.grpo import _Terms
from dew.objectives.rl.methods import METHODS, method
from dew.objectives.rl.scheduler import scored_rows
from dew.objectives.rl.sessions import (
    ADVANTAGES_KEY,
    IDS_KEY,
    OLD_LOG_PROBS_KEY,
    RESPONSE_MASK_KEY,
    Call,
    Session,
    Status,
    advantages,
    centred_moments,
    pooled,
)
from dew.rl import group_advantage, rloo_advantage
from dew.training import Trainer
from dew.training.transaction import with_ema

LENGTHS = ((3, 5), (4, 2))
"""Each row's two sessions' trained token counts."""
WIDTH = 12
SEEDS = dict.fromkeys(("grpo", "dapo", "dr_grpo", "gspo", "cispo", "gmpo", "reinforce_pp", "rloo"), 16)


def draw(seed):
    """A row grid of two sessions a row, their current, old and reference log-probabilities, each
    session's advantage on its tokens and its session weight (one over its token count)."""
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
    reference = (old + rng.normal(0, 0.25, (rows, WIDTH))).astype(np.float32)
    return policy, old, reference, advantage, mask, segments, weights


SPANS = [(row, slice(start, start + length)) for row, lengths in enumerate(LENGTHS)
         for start, length in zip(np.cumsum((0, *lengths[:-1])).tolist(), lengths, strict=True)]
"""Each session's token positions, `(row, columns)`, the same in every draw."""


def dew_term(name, **settings):
    """The method's loss term over a draw, its policy term (`GRPOObjective.policy_ratio`) and its
    reference penalty in the loss (`reference_penalty`, weighted by `beta`) as the loss adds them, and
    its gradient in the current log-probabilities."""
    objective, _ = method(name).build(TinyHead(vocab_size=8), 7, max_new_tokens=8, **settings)

    def term(policy, old, reference, advantage, mask, segments, weights):
        def value(current):
            terms = _Terms(current, jnp.asarray(old), jnp.asarray(old), jnp.asarray(advantage),
                           jnp.asarray(mask), jnp.asarray(segments), jnp.asarray(weights), proximal=False,
                           qk_stats=None, routing=None)
            pg, weighting, mass, _ = objective.policy_ratio(terms, jnp.asarray(mask),
                                                            jnp.ones_like(jnp.asarray(mask)))
            total = pg.total
            if objective.beta > 0:
                penalty = objective.reference_penalty(current, jnp.asarray(reference), weighting, mass)
                total = total + objective.beta * penalty.total
            return total / mass
        current = jnp.asarray(policy)
        return value(current), jax.grad(value)(current)
    return term


def paper(objective):
    """A paper's objective `objective(per-session arrays) -> loss` and its gradient, by autodiff of the
    formula written over each session's tokens alone (`SPANS`), in its inputs' precision: float32 the
    reference, float64 the truth."""
    def term(policy, old, reference, advantage, mask, segments, weights):
        def value(current):
            return objective([(current[row, columns], old[row, columns], reference[row, columns],
                               advantage[row, columns.start]) for row, columns in SPANS])
        return value(policy), jax.grad(value)(policy)
    return term


def clipped(ratio, advantage, low, high):
    return jnp.minimum(ratio * advantage, jnp.clip(ratio, 1 - low, 1 + high) * advantage)


def grpo(sessions, low=0.2, high=0.2, beta=0.05):
    # arXiv:2402.03300 equation 3: (1/G) sum_i (1/|o_i|) sum_t [min(r A, clip(r) A) - beta D_KL], with
    # equation 4's estimator D_KL = pi_ref/pi - log(pi_ref/pi) - 1.
    def one(p, o, ref, a):
        divergence = jnp.exp(ref - p) - (ref - p) - 1
        return jnp.mean(clipped(jnp.exp(p - o), a, low, high) - beta * divergence)
    return -sum(one(*session) for session in sessions) / len(sessions)


def dapo(sessions):
    # arXiv:2503.14476 equation 10: one over every token of the group, clip-higher (0.2, 0.28).
    total = sum(jnp.sum(clipped(jnp.exp(p - o), a, 0.2, 0.28)) for p, o, _, a in sessions)
    return -total / sum(len(p) for p, _, _, _ in sessions)


def dr_grpo(sessions, budget=8):
    # arXiv:2503.20783 section 3.2: GRPO with |o_i| replaced by the generation budget.
    total = sum(jnp.sum(clipped(jnp.exp(p - o), a, 0.2, 0.2)) for p, o, _, a in sessions)
    return -total / budget / len(sessions)


def gspo(sessions, low=0.2, high=0.2):
    # arXiv:2507.18071 equation 5: s_i = (pi / pi_old)^(1/|y_i|), clipped at the sequence.
    return -sum(clipped(jnp.exp(jnp.mean(p - o)), a, low, high) for p, o, _, a in sessions) / len(sessions)


def cispo(sessions, high=0.2):
    # arXiv:2506.13585 equation 4: sg(clip(r, 0, 1 + eps_high)) A log pi, one over every token.
    total = sum(jnp.sum(jax.lax.stop_gradient(jnp.clip(jnp.exp(p - o), 0, 1 + high)) * a * p)
                for p, o, _, a in sessions)
    return -total / sum(len(p) for p, _, _, _ in sessions)


def gmpo(sessions):
    # arXiv:2507.20673 equation 4: (prod_t min(r A, clip(r, e^-0.4, e^0.4) A))^(1/|o|) sgn(A).
    def one(p, o, _, a):
        ratio = jnp.exp(p - o)
        kept = jnp.minimum(ratio * a, jnp.clip(ratio, np.exp(-0.4), np.exp(0.4)) * a)
        return jnp.exp(jnp.mean(jnp.log(jnp.abs(kept)))) * jnp.sign(a)
    return -sum(one(*session) for session in sessions) / len(sessions)


def reinforce_pp(sessions, weight=0.1):
    # arXiv:2501.03262 equation 8: the PPO surrogate less lambda times the k2 loss
    # E[(1/2) (log pi/pi_ref)^2] (appendix B.1), each session's token mean averaged over the sessions.
    def one(p, o, ref, a):
        return jnp.mean(-clipped(jnp.exp(p - o), a, 0.2, 0.2) + weight * 0.5 * jnp.square(p - ref))
    return sum(one(*session) for session in sessions) / len(sessions)


def rloo(sessions):
    # arXiv:2402.14740 equation 6: (1/k) sum_i A_i grad log pi(y_i), a sequence's log-likelihood its
    # tokens' sum.
    return -sum(a * jnp.sum(p) for p, _, _, a in sessions) / len(sessions)


@pytest.mark.parametrize(("name", "formula", "settings"), [
    ("grpo", grpo, {"beta": 0.05}), ("dapo", dapo, {}), ("dr_grpo", dr_grpo, {}), ("gspo", gspo, {}),
    ("cispo", cispo, {"epsilon_high": 0.2}), ("gmpo", gmpo, {}),
    ("reinforce_pp", reinforce_pp, {"beta": 0.1}), ("rloo", rloo, {})],
    ids=lambda value: value if isinstance(value, str) else "")
def test_each_method_is_its_papers_objective(name, formula, settings):
    """The method's loss term, its reference penalty in the loss included, and its gradient in each
    token's log-probability are its paper's objective's (check 1, the paper's form in float64 the truth)."""
    assert_as_exact(dew_term(name, **settings), paper(formula), draw, name, seeds=SEEDS[name])


def test_a_method_fixes_its_composition_and_refuses_what_it_lacks():
    """A method refuses an override of a part it fixes, a reference penalty where it has none (in the
    loss or in the reward), and an unknown name."""
    model = TinyHead(vocab_size=8)
    for name, override in (("dr_grpo", {"aggregation": "token-mean"}), ("gspo", {"beta": 0.1}),
                           ("dapo", {"reward_kl": 0.1}), ("rloo", {"beta": 0.1})):
        with pytest.raises(ValueError):
            METHODS[name].build(model, 8, max_new_tokens=8, **override)
    with pytest.raises(ValueError):
        method("ppo-lite")


@pytest.mark.parametrize("shares", [1, 2])
def test_the_global_estimator_whitens_the_whole_batch_over_its_data_shares(shares):
    """REINFORCE++ with a baseline (arXiv:2501.03262 equations 6-7): each member's reward less its group's
    mean, the batch of those whitened, with the paper's epsilon. Split into data shares, each share's
    moments pooled (`pooled`), every share's members are whitened by the whole batch's mean and
    deviation, as one share's are."""
    rewards = {0: [1.0, 0.0, 0.5], 1: [2.0, 2.0, -1.0], 2: [4.0, -3.0]}
    sessions = {task: [session(task, sample, reward, (1,)) for sample, reward in enumerate(values)]
                for task, values in rewards.items()}
    parts = [[*sessions[0], *sessions[1], *sessions[2]]] if shares == 1 else [
        [*sessions[0], *sessions[1]], sessions[2]]
    whitening = pooled([centred_moments(part) for part in parts])
    found = np.asarray([values[0] for part in parts
                        for values in advantages(part, "global", whitening=whitening)])
    centred = np.concatenate([np.asarray(values) - np.mean(values) for values in rewards.values()])
    expected = (centred - centred.mean()) / (centred.std() + 1e-8)
    # Each group's mean is exact in float32 on these rewards, so the whitening alone rounds, in float64.
    np.testing.assert_array_equal(found, expected.astype(np.float32))


VOCAB = 24
PROMPT = (1, 2)
TOOL = 23


class Scripted:
    """A collector that answers each task with its scripted sessions, whatever the policy."""

    width = 16

    def __init__(self, sessions):
        self.sessions = sessions

    def rows(self, tasks, samples):
        return tasks * samples

    def collect(self, snapshot, tasks, samples):
        return [self.sessions[task.id] for task in tasks]


def session(task, sample, reward, lengths, *, paid=(), status=Status.COMPLETED, raw=-0.5):
    """Task `task`'s sample `sample`: a call of each of `lengths` sampled ids, the later ones asked after
    a tool's answer (`TOOL`), each id the session's own (`token`), the calls paid `paid` and recording
    `raw` as each id's raw likelihood (an array of them, or one for every id)."""
    raw = np.broadcast_to(np.asarray(raw, np.float64), (sum(lengths),))
    calls, prompt, start = [], PROMPT, 0
    for index, length in enumerate(lengths):
        tokens = (token(task, sample),) * length
        calls.append(Call(prompt, tokens, (-0.25,) * length, "stop", 0,
                          raw_log_probs=tuple(raw[start:start + length].tolist()),
                          reward=paid[index] if index < len(paid) else None))
        prompt, start = (*prompt, *tokens, TOOL), start + length
    return Session(str(task), "", sample, 0, tuple(calls), status, reward)


def token(task, sample):
    """The id session `sample` of task `task` samples, apart from every other session's."""
    return 3 + 4 * task + sample


def scheduled(sessions, tasks, objective, **settings):
    """One synchronous call over `sessions` of `tasks` (`Scripted`), at the starting state of
    `objective`'s trainer, its packed batch, the scheduler and that state."""
    trainer = Trainer(objective, optax.sgd(0.1), key=jax.random.key(3))
    state = trainer.initial_state()
    scheduler = RolloutScheduler(Synchronous(Scripted(sessions)), **settings)
    return scheduler(state, {"task_id": np.asarray(tasks, np.int32)}, jax.random.key(0)), scheduler, state


def per_session(packed, values):
    """Each sampled id's value from its session's (`values` by the id it samples) on the packed grid."""
    ids, mask = np.asarray(packed[IDS_KEY]), np.asarray(packed[RESPONSE_MASK_KEY]) > 0
    return np.where(mask, np.vectorize(lambda id_: values.get(int(id_), 0.0))(ids), 0).astype(np.float32)


def test_dapo_admits_the_groups_whose_returns_vary_with_their_overlong_punishment():
    """DAPO's rollout (`METHODS["dapo"].rollout`) over a batch of four tasks admitting two: a group whose
    returns all agree is left out and the next task's group takes its place, the returns being what
    the estimator compares, the verifier's reward with the step rewards a session's calls earned and the
    soft overlong punishment (arXiv:2503.14476 equation 13) over its sampled ids alone, across its calls
    and past its prompt's and the tool's ids, a truncated one trained on its reward. Equal terminal
    rewards with unequal step rewards vary. The admitted ids hold their group's advantage of those
    returns, and the left-out and the surplus groups' ids are not packed."""
    budget, cache = 8, 4
    sessions = {
        # One return each: left out.
        "0": [session(0, 0, 1.0, (3,)), session(0, 1, 1.0, (2, 2))],
        # One terminal reward, but one session's first call was paid 0.5.
        "1": [session(1, 0, 1.0, (2, 1), paid=(0.5,)), session(1, 1, 1.0, (3,), paid=(0.0,))],
        # One terminal reward over 4 and 6 sampled ids, the second truncated and punished -0.5.
        "2": [session(2, 0, 1.0, (4,)), session(2, 1, 1.0, (3, 3), status=Status.TRUNCATED)],
        # Varied, and past the two groups admitted.
        "3": [session(3, 0, 0.0, (2,)), session(3, 1, 1.0, (2,))],
    }
    objective, rollout = METHODS["dapo"].build(TinyHead(vocab_size=VOCAB), 15, max_new_tokens=budget,
                                               overlong_cache=cache)
    packed, scheduler, _ = scheduled(sessions, [0, 1, 2, 3], objective, groups=2, admit=2,
                                     truncation="score", **rollout)

    def punished(length):
        # Equation 13, written apart: none to L_max - L_cache, -(L - (L_max - L_cache)) / L_cache to L_max.
        if length <= budget - cache:
            return 0.0
        return (budget - cache - length) / cache if length <= budget else -1.0

    expected = {}
    for task in ("1", "2"):
        returns = [member.reward + sum(call.reward or 0.0 for call in member.calls)
                   + punished(sum(len(call.tokens) for call in member.calls)) for member in sessions[task]]
        found = np.asarray(group_advantage(jnp.asarray(returns, jnp.float32), 2))
        expected.update({token(int(task), sample): float(value) for sample, value in enumerate(found)})
    mask = np.asarray(packed[RESPONSE_MASK_KEY]) > 0
    np.testing.assert_array_equal(np.asarray(packed[ADVANTAGES_KEY]) * mask, per_session(packed, expected))
    assert set(np.asarray(packed[IDS_KEY])[mask].tolist()) == set(expected)
    assert (scheduler.metrics["groups"], scheduler.metrics["uniform_groups"]) == (2.0, 1.0)


def test_dapo_refuses_a_batch_none_of_whose_groups_vary():
    """A batch whose every group earned one return has nothing DAPO trains on: the call raises."""
    sessions = {"0": [session(0, 0, 1.0, (2,)), session(0, 1, 1.0, (2,))]}
    objective, rollout = METHODS["dapo"].build(TinyHead(vocab_size=VOCAB), 15, max_new_tokens=8)
    with pytest.raises(RuntimeError):
        scheduled(sessions, [0], objective, groups=2, **rollout)


def test_rloo_pays_its_reference_penalty_in_the_reward_before_the_baseline():
    """RLOO (arXiv:2402.14740 equations 3 and 6): each sequence's reward less reward_kl times its summed
    log-ratio of the policy that drew it (its recorded raw likelihoods, the proximal policy) to the
    reference, the trainer's frozen starting policy, its calls' ids together, and each member's
    advantage that return less the mean of the others' in its group."""
    weight, rng = 0.1, np.random.default_rng(2402)
    raws = {(task, sample): -rng.uniform(0.5, 3.0, 5) for task in range(2) for sample in range(3)}
    rewards = {(0, 0): 1.0, (0, 1): 0.0, (0, 2): 0.5, (1, 0): 0.25, (1, 1): 0.25, (1, 2): 1.0}
    sessions = {str(task): [session(task, sample, rewards[task, sample], (3, 2), raw=raws[task, sample])
                            for sample in range(3)] for task in range(2)}
    objective, rollout = METHODS["rloo"].build(TinyHead(vocab_size=VOCAB), 15, max_new_tokens=8,
                                               reward_kl=weight)
    packed, _, state = scheduled(sessions, [0, 1], objective, groups=3, reference=objective, **rollout)
    # The reference's likelihood of each sampled id as the step scores the packed rows, and the
    # proximal policy's as they hold it (the calls' raw likelihoods).
    reference = np.asarray(scored_rows(objective, with_ema(state.variables, state.ema), packed), np.float64)
    proximal = np.asarray(packed[OLD_LOG_PROBS_KEY], np.float64)
    ids, mask = np.asarray(packed[IDS_KEY]), np.asarray(packed[RESPONSE_MASK_KEY]) > 0
    expected = {}
    for task in range(2):
        returns = []
        for sample in range(3):
            mine = mask & (ids == token(task, sample))
            assert mine.sum() == len(raws[task, sample])
            returns.append(rewards[task, sample] - weight * float(np.sum(proximal[mine] - reference[mine])))
        found = np.asarray(rloo_advantage(jnp.asarray(returns, jnp.float32), 3))
        expected.update({token(task, sample): float(value) for sample, value in enumerate(found)})
    np.testing.assert_array_equal(np.asarray(packed[ADVANTAGES_KEY]) * mask, per_session(packed, expected))
