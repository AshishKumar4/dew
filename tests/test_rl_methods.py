"""Each named RL method (dew.objectives.rl.methods) against its paper's objective, written apart from Dew:
each method's loss term, its reference penalty in the loss included, and its gradient in each token's
log-probability, over drawn sessions packed two to a row, run in float64 computing the paper's form in
float64 (`assert_computes_the_oracle`), and its gradient in float32 on par with the paper's form's by
check 1 (tests/reference_error.py). Through a scheduler's call over scripted sessions, the returns the
estimators compare: DAPO's overlong punishment and dynamic sampling, within a batch and across the
batches it draws further, RLOO's reference penalty in the reward, and the global estimator's whitening
over a batch split into data shares."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from reference_error import assert_as_exact, assert_computes_the_oracle, chain_roundings
from rl_support import TinyHead

from dew.config import ObjectiveConfig
from dew.config.lm import PromptRollout
from dew.nn.progress import Progress
from dew.objectives.rl import RolloutScheduler, Synchronous
from dew.objectives.rl.grpo import _Terms
from dew.objectives.rl.methods import method
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

LENGTHS = ((3, 5), (4, 2), (6, 4), (2, 7)) * 8
"""Each row's two sessions' trained token counts: enough sessions that a loss's rounding is a sum of many
terms', which check 1 resolves over its seeds."""
WIDTH = 12
SEEDS = 16
"""The draws check 1 holds each method's float32 gradient over."""
ORACLE_DRAWS = 4
"""The draws each method's float64 term is held to its paper's form on."""


def draw(seed, dtype=np.float32):
    """A row grid of two sessions a row, their current, old and reference log-probabilities, each
    session's advantage on its tokens and its session weight (one over its token count), in `dtype`:
    a float64 batch's weights are one over the count exactly, as a float64 pack would write them."""
    rng = np.random.default_rng([858, seed])
    rows = len(LENGTHS)
    segments = np.zeros((rows, WIDTH), np.int32)
    mask = np.zeros((rows, WIDTH), dtype)
    advantage = np.zeros((rows, WIDTH), np.float32)
    weights = np.zeros((rows, WIDTH), dtype)
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
    return (*(value.astype(dtype) for value in (policy, old, reference, advantage)), mask, segments,
            weights)


SPANS = [(row, slice(start, start + length)) for row, lengths in enumerate(LENGTHS)
         for start, length in zip(np.cumsum((0, *lengths[:-1])).tolist(), lengths, strict=True)]
"""Each session's token positions, `(row, columns)`, the same in every draw."""


def composed(name, model, seq_len, *, max_new_tokens=8, overlong_cache=None, **fields):
    """The method's objective over `model` and its scheduler's settings as a run composes them: a rollout
    naming it (`PromptRollout.method`), whose settings the run derives into the objective's
    (`ObjectiveConfig.build`, refusing `fields` the method fixes) and its scheduler's
    (`PromptRollout.scheduler_settings`)."""
    rollout = PromptRollout(reward="rl_support:completion_length", method=name, max_new_tokens=max_new_tokens,
                            overlong_cache=overlong_cache)
    objective = ObjectiveConfig("grpo", fields).build(model=model, seq_len=seq_len,
                                                       **rollout.objective_settings())
    return objective, rollout.scheduler_settings()


def dew_term(name, **settings):
    """The method's loss term over a draw, its policy term (`GRPOObjective.policy_ratio`) and its
    reference penalty in the loss (`reference_penalty`, weighted by `beta`) as the loss adds them, and
    its gradient in the current log-probabilities."""
    objective, _ = composed(name, TinyHead(vocab_size=8), 7, **settings)

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
    token's log-probability compute its paper's objective: run in float64, within the float64 rounding of
    their own arithmetic of the paper's form in float64. In float32 the gradient it trains on is on par
    with the paper's form's (check 1, the paper's form in float64 the truth); a loss is one value a draw,
    whose float32 rounding check 1 cannot resolve over the draws a site runs."""
    term, written = dew_term(name, **settings), paper(formula)
    for seed in range(ORACLE_DRAWS):
        with jax.enable_x64(new_val=True):
            inputs = draw(seed, np.float64)
            found, expected = term(*inputs), written(*inputs)
            roundings = chain_roundings(jax.make_jaxpr(term)(*inputs))
        for kind, mine, theirs in zip(("loss", "gradient"), found, expected, strict=True):
            assert_computes_the_oracle(np.asarray(mine), np.asarray(theirs), f"{name} {kind} at draw {seed}",
                                       roundings=roundings)
    assert_as_exact(lambda *inputs: term(*inputs)[1], lambda *inputs: written(*inputs)[1], draw,
                    f"{name} gradient", seeds=SEEDS)


def test_reinforce_pps_penalty_is_representable_where_its_value_is():
    """REINFORCE++'s k2 penalty, half a log-ratio's square, at a log-ratio of -2**64: 2**127, which float32
    holds, though the square alone would not."""
    objective, _ = composed("reinforce_pp", TinyHead(vocab_size=8), 7, beta=0.1)
    one = jnp.ones((1, 1), jnp.float32)
    penalty = objective.reference_penalty(jnp.full((1, 1), -2.0 ** 64, jnp.float32), 0 * one, one, one.sum())
    assert float(penalty.total) == 2.0 ** 127


def test_a_method_fixes_its_composition_and_refuses_what_it_lacks():
    """A run's method refuses an objective argument of a part it fixes, a reference penalty where it has
    none (in the loss or in the reward), an overlong window where it punishes no overlong response or
    one past its budget, and an unknown name."""
    model = TinyHead(vocab_size=8)
    for name, override in (("dr_grpo", {"aggregation": "token-mean"}), ("gspo", {"beta": 0.1}),
                           ("dapo", {"reward_kl": 0.1}), ("rloo", {"beta": 0.1}),
                           ("grpo", {"overlong_cache": 2}), ("dapo", {"overlong_cache": 9})):
        with pytest.raises(ValueError):
            composed(name, model, 8, **override)
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
    """A collector that answers each task with its scripted sessions, whatever the policy, each call
    stamped with the version it was drawn at, recording each collection's tasks and the key it was
    drawn under."""

    width = 16

    def __init__(self, sessions):
        self.sessions = sessions
        self.collected = []

    def rows(self, tasks, samples):
        return tasks * samples

    def collect(self, snapshot, tasks, samples):
        self.collected.append(([int(task.id) for task in tasks],
                               np.asarray(jax.random.key_data(snapshot.key)).tolist()))
        return [[dataclasses.replace(session, calls=tuple(dataclasses.replace(call, version=snapshot.updates)
                                                          for call in session.calls))
                 for session in self.sessions[task.id]] for task in tasks]


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


def scheduled(sessions, tasks, objective, further=None, **settings):
    """One synchronous call over `sessions` of `tasks` (`Scripted`), at the starting state of
    `objective`'s trainer, handed `further` where it draws further batches, its packed batch, the
    scheduler and that state."""
    trainer = Trainer(objective, optax.sgd(0.1), key=jax.random.key(3))
    state = trainer.initial_state()
    scheduler = RolloutScheduler(Synchronous(Scripted(sessions)), **settings)
    drawing = {} if further is None else {"further": further}
    return (scheduler(state, {"task_id": np.asarray(tasks, np.int32)}, jax.random.key(0), **drawing),
            scheduler, state)


def per_session(packed, values):
    """Each sampled id's value from its session's (`values` by the id it samples) on the packed grid."""
    ids, mask = np.asarray(packed[IDS_KEY]), np.asarray(packed[RESPONSE_MASK_KEY]) > 0
    return np.where(mask, np.vectorize(lambda id_: values.get(int(id_), 0.0))(ids), 0).astype(np.float32)


def test_dapo_admits_the_groups_whose_returns_vary_with_their_overlong_punishment():
    """DAPO's rollout as a run composes it (`PromptRollout(method="dapo")`) over a batch of four tasks
    admitting two: a group whose returns all agree is left out and the next task's group takes its
    place, the returns being what the estimator compares, the verifier's reward with the step rewards
    a session's calls earned and the soft overlong punishment (arXiv:2503.14476 equation 13) over its
    sampled ids alone, across its calls and past its prompt's and the tool's ids, over the run's
    overlong window (`overlong_cache`), a truncated one trained on its reward. Equal terminal
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
    objective, rollout = composed("dapo", TinyHead(vocab_size=VOCAB), 15, max_new_tokens=budget,
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
    objective, rollout = composed("dapo", TinyHead(vocab_size=VOCAB), 15)
    with pytest.raises(RuntimeError):
        scheduled(sessions, [0], objective, groups=2, **rollout)


def agreeing(task):
    """Task `task`'s two sessions, one return between them: a group dynamic sampling leaves out."""
    return [session(task, 0, 1.0, (2,)), session(task, 1, 1.0, (2,))]


def varying(task):
    """Task `task`'s two sessions, returns 0 and 1: a group dynamic sampling admits."""
    return [session(task, 0, 0.0, (2,)), session(task, 1, 1.0, (2,))]


def test_dapo_draws_further_batches_until_its_step_holds_its_groups():
    """DAPO's dynamic sampling across data batches (arXiv:2503.14476 Algorithm 1; verl's
    `max_num_gen_batches`): a step short of its `admit` groups once its own batch's agreeing groups are
    left out draws the stream's next batch (`further`) and admits from it, its groups packed into the
    step's rows with their own advantages, each further batch sampled under the step's key folded by
    its place, and draws none once the step is full or `generation_batches` are drawn."""
    sessions = {str(task): (agreeing if task in (0, 2, 4) else varying)(task) for task in range(5)}
    objective, rollout = composed("dapo", TinyHead(vocab_size=VOCAB), 15)
    ahead = iter([[2, 3], [4, 1]])
    drawn = []

    def further():
        batch = {"task_id": np.asarray(next(ahead), np.int32)}
        drawn.append(batch["task_id"].tolist())
        return batch

    packed, scheduler, _ = scheduled(sessions, [0, 1], objective, further, groups=2, admit=2,
                                     generation_batches=3, **rollout)
    assert drawn == [[2, 3]]
    expected = {}
    for task in (1, 3):
        found = np.asarray(group_advantage(jnp.asarray([0.0, 1.0], jnp.float32), 2))
        expected.update({token(task, sample): float(value) for sample, value in enumerate(found)})
    mask = np.asarray(packed[RESPONSE_MASK_KEY]) > 0
    np.testing.assert_array_equal(np.asarray(packed[ADVANTAGES_KEY]) * mask, per_session(packed, expected))
    assert set(np.asarray(packed[IDS_KEY])[mask].tolist()) == set(expected)
    collected = scheduler.acquisition.collector.collected
    key = jax.random.key(0)
    assert collected == [([0, 1], np.asarray(jax.random.key_data(key)).tolist()),
                         ([2, 3], np.asarray(jax.random.key_data(jax.random.fold_in(key, 1))).tolist())]
    assert (scheduler.metrics["groups"], scheduler.metrics["generation_batches"]) == (2.0, 2.0)
    # Two batches at most: the second's agreeing group leaves the step one group short.
    ahead = iter([[2, 4], [3, 1]])
    drawn.clear()
    _, capped, _ = scheduled(sessions, [0, 1], objective, further, groups=2, admit=2, generation_batches=2,
                             **rollout)
    assert drawn == [[2, 4]] and capped.metrics["groups"] == 1.0


def test_a_run_drawing_further_batches_resumes_after_the_last_it_drew(tmp_path):
    """A run whose step draws the stream's next batch to fill its groups (`generation_batches`) reads it
    through the trainer's stream (`FitRun.further`), whose position its checkpoint saves: its next
    step reads the batch after the one drawn, and a run resumed from that checkpoint collects what the
    run that went on collected, the same tasks under the same keys."""
    from dew.checkpoints import Checkpoints
    from dew.data.dataset import Loader, replayed
    from dew.training import Layout

    sessions = {str(task): (agreeing if task == 0 else varying)(task) for task in range(5)}
    # Eight tasks a batch, one a device: the step's batch admits two groups of the four it takes, and
    # the batch it draws the other two.
    batches = [[0, 0, 0, 0, 1, 1, 0, 0], [2, 3, 2, 3, 0, 0, 0, 0], [4, 1, 4, 1, 4, 1, 4, 1],
               [2, 3, 2, 3, 2, 3, 2, 3]]
    objective, rollout = composed("dapo", TinyHead(vocab_size=VOCAB), 15)

    def fit(steps, checkpoints=None):
        collector = Scripted(sessions)
        scheduler = RolloutScheduler(Synchronous(collector), groups=2, admit=4, generation_batches=3,
                                     **rollout)
        stream = Loader(train=replayed(lambda partition: ({"task_id": np.asarray(batch, np.int32)}
                                                          for batch in batches)),
                        val=None, records=None, batch=8)
        trainer = Trainer(objective, optax.sgd(0.1), key=jax.random.key(3), rollout=scheduler,
                          layout=Layout(min_shard=1, tolerance=1.0), checkpoints=checkpoints)
        trainer.fit(stream, steps=steps, log_every=steps, checkpoint_every=None if checkpoints is None else 1)
        return collector.collected

    whole = fit(2)
    assert [tasks for tasks, _ in whole] == batches[:3]
    checkpoints = str(tmp_path / "run")
    assert fit(1, Checkpoints(checkpoints)) + fit(2, Checkpoints(checkpoints)) == whole


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
    objective, rollout = composed("rloo", TinyHead(vocab_size=VOCAB), 15, reward_kl=weight)
    packed, _, state = scheduled(sessions, [0, 1], objective, groups=3, reference=objective, **rollout)
    # The reference's likelihood of each sampled id as the step scores the packed rows, and the
    # proximal policy's as they hold it (the calls' raw likelihoods).
    reference = np.asarray(scored_rows(objective, with_ema(state.variables, state.ema), packed,
                                       Progress(state.updates)), np.float64)
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
