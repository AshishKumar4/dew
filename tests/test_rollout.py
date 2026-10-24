"""The rollout capability: sampling between the data stream and the step.

The trainer calls a `Rollout` with the state, the prefetched batch and a key
folded twice off the run key, then reshards the result and compiles the step
once. With `rollout=None` the prefetched batch trains untouched. A
`RolloutScheduler.sampled` fills that contract for online RL: G completions
per prompt from the objective's policy, scored by a reward call, advantaged
with the group or RLOO family from `dew.rl`, with the old log-probabilities
the sampler recorded under the same weights.
"""

import jax
import numpy as np
import pytest
from affine_run import dataset, frozen_trainer, raw_leaf
from recording import RecordingTracker
from rl_support import TinyHead

from dew.data.prompts import _utf8
from dew.objectives.lm import LMObjective
from dew.objectives.rl import RolloutScheduler
from dew.rl import group_advantage, rloo_advantage
from dew.sampling import Sampling

VOCAB = 8
PROMPT_WIDTH = 8
NEW_TOKENS = 4
GROUPS = 2


class Recorder:
    """A rollout that returns the batch untouched and keeps every key."""

    def __init__(self):
        self.keys = []
        self.steps = []

    def __call__(self, state, batch, key):
        self.keys.append(np.asarray(jax.random.key_data(key)).tobytes())
        self.steps.append(int(state.step))
        return batch


# --- the loop with and without a rollout -------------------------------------

def test_an_identity_rollout_leaves_the_loop_byte_identical():
    """The same two steps with `rollout=None` and with a rollout that hands
    the batch back: the same final state, every parameter, optimizer moment,
    average and counter and the key, and identical logged losses. The
    rollout lives outside the compiled step, so None adds no ops to the
    loop; the compiled step is unreachable for a direct jaxpr comparison
    because it runs under `set_mesh`, which refuses to trace."""
    assert frozen_trainer().rollout is None
    plain, ident = RecordingTracker(), RecordingTracker()
    alone = frozen_trainer(tracker=plain).fit(dataset(), steps=2, log_every=1)
    rolled = frozen_trainer(rollout=lambda state, batch, key: batch,
                            tracker=ident).fit(dataset(), steps=2, log_every=1)

    for left, right in zip(jax.tree.leaves(alone), jax.tree.leaves(rolled), strict=True):
        np.testing.assert_array_equal(np.asarray(raw_leaf(left)), np.asarray(raw_leaf(right)))
    assert [logged["train/loss"] for _, logged in plain.scalars if "train/loss" in logged] == [
        logged["train/loss"] for _, logged in ident.scalars if "train/loss" in logged] != []


def test_rollout_keys_fold_forward_and_reproduce():
    """Three steps draw three distinct keys, and a second identical run draws
    the same three: the rollout stream is checkpointed randomness, not noise."""
    first, second = Recorder(), Recorder()
    frozen_trainer(rollout=first).fit(dataset(), steps=3, log_every=3)
    frozen_trainer(rollout=second).fit(dataset(), steps=3, log_every=3)
    assert first.steps == [0, 1, 2]
    assert len(set(first.keys)) == 3
    assert first.keys == second.keys

def test_a_resumed_run_does_not_replay_rollouts(tmp_path):
    """Two steps, then the resumed run samples steps 2 and 3, never 0 and 1
    again."""
    rollout = Recorder()
    frozen_trainer(tmp_path, rollout=rollout).fit(dataset(), steps=2, log_every=2)
    frozen_trainer(tmp_path, rollout=rollout).fit(dataset(), steps=4, log_every=4)

    assert rollout.steps == [0, 1, 2, 3]


def test_rollout_seconds_logs_only_with_a_rollout():
    """The log tick carries `train/rollout_seconds` when a rollout is set,
    and no such key without one. The run's final goodput line is not a step
    tick, so only ticks with a loss count."""
    tracking = RecordingTracker()
    frozen_trainer(rollout=Recorder(), tracker=tracking).fit(dataset(), steps=2, log_every=1)
    ticks = [logged for _, logged in tracking.scalars if "train/loss" in logged]
    assert len(ticks) == 2
    assert all(tick["train/rollout_seconds"] >= 0.0 for tick in ticks)

    tracking = RecordingTracker()
    frozen_trainer(tracker=tracking).fit(dataset(), steps=2, log_every=1)
    ticks = [logged for _, logged in tracking.scalars if "train/loss" in logged]
    assert len(ticks) == 2
    assert all("train/rollout_seconds" not in tick for tick in ticks)


def test_a_rollouts_metrics_log_as_of_its_latest_call():
    """Each tick carries the rollout's `metrics` under `rollout/`, as they
    stood after the call that made the tick's last batch."""
    class Counting(Recorder):
        metrics: dict[str, float]

        def __call__(self, state, batch, key):
            self.metrics = {"reward/mean": float(state.step) + 0.5}
            return super().__call__(state, batch, key)

    tracking = RecordingTracker()
    frozen_trainer(rollout=Counting(), tracker=tracking).fit(dataset(), steps=4, log_every=2)
    ticks = [logged for _, logged in tracking.scalars if "train/loss" in logged]
    assert [tick["rollout/reward/mean"] for tick in ticks] == [1.5, 3.5]


# --- the synchronous sampled rollout -------------------------------------------


class Calls:
    """Records each call's arguments; scores how many of the completion's tokens are the reference."""

    def __init__(self):
        self.seen = []

    def __call__(self, completion, record):
        self.seen.append((completion, record))
        return float(completion.split().count(record.ground_truth))


def tiny_objective(seq=PROMPT_WIDTH + NEW_TOKENS - 1):
    return LMObjective(TinyHead(vocab_size=VOCAB), seq)


def prompt_batch(rows=2):
    """Two fixed-width prompt rows with distinct reward contexts, one containing multibyte UTF-8."""
    width = PROMPT_WIDTH
    prompts = np.zeros((rows, width), np.int32)
    prompts[0] = [1, 2, 3, 4, 5, 6, 7, 1]
    prompts[1] = [2, 3, 4, 5, 6, 7, 1, 2]
    info = max(len("rule"), len("other"), 1)
    def pad(text):
        return _utf8(text, info)
    return {
        "prompt": prompts,
        "prompt_length": np.full(rows, width, np.int32),
        "data_source": np.stack([pad("rulé"), pad("other")]),
        "ground_truth": np.stack([pad("1"), pad("2")]),
        "extra_info": np.stack([pad(""), pad("")]),
    }


class FakeState:
    def __init__(self, variables, updates=0):
        self.variables = variables
        self.step = self.updates = updates


def chains(out):
    """Each completion's chain, sampled mask and columns, by rollout index."""
    found = {}
    for index in np.unique(out["session_index"][out["session_index"] >= 0]):
        where = out["session_index"] == index
        found[int(index)] = {name: np.asarray(out[name])[where]
                             for name in ("input_ids", "response_mask", "old_log_probs",
                                          "behavior_log_probs", "advantages", "versions")}
    return found


def test_a_rollout_packs_into_the_objectives_rows_and_refuses_a_draw_too_long_for_them():
    """An objective wider than prompt plus budget packs its draws into its own
    rows, each chain where a narrow objective's sits and padding after it; one
    too narrow for a prompt and its draw is refused by the packer."""
    def drawn(seq):
        rollout = RolloutScheduler.sampled(tiny_objective(seq=seq), Calls(), groups=GROUPS,
                                           max_new_tokens=NEW_TOKENS, sampling=Sampling(temperature=0.0))
        return rollout(FakeState(params), prompt_batch(), jax.random.key(1))

    params = tiny_objective().init(jax.random.key(0))
    width = PROMPT_WIDTH + NEW_TOKENS
    out, fitted = drawn(width + 3), drawn(width - 1)
    assert out["input_ids"].shape == (4, width + 4)
    for name in ("input_ids", "response_mask", "old_log_probs", "behavior_log_probs", "advantages"):
        np.testing.assert_array_equal(np.asarray(out[name])[:, :width], fitted[name], name)
    assert not np.asarray(out["text_segment_ids"])[:, width:].any()
    with pytest.raises(ValueError):
        drawn(width - 2)


def test_a_sampled_rollout_packs_its_completions():
    """G completions per prompt from the real sampler, greedy and
    deterministic, packed as one-call rollouts: a fixed [prompts * G, prompt
    width + budget] batch in the packed layout, each chain its prompt then its
    sampled ids, groups contiguous in rollout order."""
    objective = tiny_objective()
    params = objective.init(jax.random.key(0))
    rollout = RolloutScheduler.sampled(objective, Calls(), groups=GROUPS,
                                       max_new_tokens=NEW_TOKENS, sampling=Sampling(temperature=0.0))
    batch = prompt_batch()

    out = rollout(FakeState(params, updates=3), batch, jax.random.key(1))

    for name in ("input_ids", "text_segment_ids", "text_positions", "response_mask",
                 "old_log_probs", "behavior_log_probs", "advantages"):
        assert out[name].shape == (4, PROMPT_WIDTH + NEW_TOKENS), name
    found = chains(out)
    assert sorted(found) == [0, 1, 2, 3]
    for index, chain in found.items():
        np.testing.assert_array_equal(chain["input_ids"][:PROMPT_WIDTH], batch["prompt"][index // GROUPS])
        np.testing.assert_array_equal(chain["response_mask"], [0] * PROMPT_WIDTH + [1] * NEW_TOKENS)
        assert (chain["versions"][PROMPT_WIDTH:] == 3).all()
        rescored = np.asarray(objective.per_token_log_probs(params, chain["input_ids"][None]))[0]
        sampled = chain["response_mask"] != 0
        np.testing.assert_allclose(chain["old_log_probs"][sampled], rescored[sampled[1:]], rtol=1e-5)
    again = rollout(FakeState(params, updates=3), batch, jax.random.key(1))
    for key in out:
        np.testing.assert_array_equal(np.asarray(out[key]), np.asarray(again[key]))


def test_a_prompts_group_is_the_policys_continuations_of_one_request():
    """A rollout draws each prompt's group as the policy's `groups`
    continuations of one request under its key: one prefill, and the chains
    the task's own call draws, in its row order."""
    objective = tiny_objective()
    params = objective.init(jax.random.key(0))
    sampling = Sampling(temperature=1.3, top_k=5)
    rollout = RolloutScheduler.sampled(objective, Calls(), groups=GROUPS, max_new_tokens=NEW_TOKENS,
                                       sampling=sampling)
    batch = prompt_batch()
    out = rollout(FakeState(params), batch, jax.random.key(4))
    drawn = objective.policy(params, sampling)(batch["prompt"], NEW_TOKENS, key=jax.random.key(4),
                                               n=GROUPS).host()
    found = chains(out)
    for index, chain in found.items():
        np.testing.assert_array_equal(chain["input_ids"], drawn.tokens[index])
        sampled = chain["response_mask"] != 0
        np.testing.assert_array_equal(chain["behavior_log_probs"][sampled], drawn.behavior_log_probs[index])


def test_the_reward_sees_each_rows_own_context():
    """Each call reads its own row's data source and reference."""
    objective = tiny_objective()
    params = objective.init(jax.random.key(0))
    reward = Calls()
    rollout = RolloutScheduler.sampled(objective, reward, groups=GROUPS,
                                       max_new_tokens=NEW_TOKENS, sampling=Sampling(temperature=0.0))

    rollout(FakeState(params), prompt_batch(), jax.random.key(1))

    assert [record.data_source for _, record in reward.seen] == ["rulé", "rulé", "other", "other"]
    assert [record.ground_truth for _, record in reward.seen] == ["1", "1", "2", "2"]


@pytest.mark.parametrize("estimator", ["group", "mean", "rloo"])
def test_each_estimator_turns_the_calls_rewards_into_the_packed_advantages(estimator):
    """Four completions a prompt under a sampling policy, scored by how many
    of their tokens are the prompt's reference: the rewards differ within
    every group, so the advantages are not all zero, and the packed column
    holds the named family, as `pack` names it, over exactly those rewards
    (Dr.GRPO's centred rewards for "mean")."""
    groups = 4
    objective = tiny_objective()
    params = objective.init(jax.random.key(0))
    reward = Calls()
    rollout = RolloutScheduler.sampled(objective, reward, groups=groups, max_new_tokens=NEW_TOKENS,
                                       sampling=Sampling(temperature=1.3, top_k=5), estimator=estimator)

    out = rollout(FakeState(params), prompt_batch(), jax.random.key(1))

    assert len(reward.seen) == 2 * groups
    scores = np.asarray([reward(*seen) for seen in reward.seen[:2 * groups]], np.float32)
    assert all(len(set(group)) > 1 for group in scores.reshape(-1, groups))
    expected = np.asarray({"group": group_advantage(scores, groups),
                           "mean": group_advantage(scores, groups, normalise_by_std=False),
                           "rloo": rloo_advantage(scores, groups)}[estimator], np.float32)
    assert all(np.count_nonzero(group) >= 2 for group in expected.reshape(-1, groups))
    for index, chain in chains(out).items():
        np.testing.assert_allclose(chain["advantages"], expected[index], rtol=1e-5)


def test_a_draw_is_scored_by_any_finite_reward_float_takes():
    """A reward may be a verifier's bool or a NumPy or JAX scalar, as `float` takes it; a non-finite
    one is refused, and a reward whose external boundary failed leaves an INFRA_ERROR session."""
    import jax.numpy as jnp

    from dew.data.prompts import PromptRecord
    from dew.objectives.rl.rollout import scored_draw
    from dew.objectives.rl.sessions import Call, Status

    call = Call((1, 2), (3, 4), (-0.5, -0.25), "stop", 0)
    record = PromptRecord("rule", "1", "")
    for reward, expected in ((True, 1.0), (np.float32(0.5), 0.5), (jnp.asarray(2.0), 2.0)):
        session = scored_draw("task", 0, call, "text", record, lambda text, record, reward=reward: reward)
        assert session.status is Status.COMPLETED and session.reward == expected
    with pytest.raises(ValueError):
        scored_draw("task", 0, call, "text", record, lambda text, record: float("nan"))

    def unreachable(text, record):
        raise OSError("the verifier service is down")

    assert scored_draw("task", 0, call, "text", record, unreachable).status is Status.INFRA_ERROR
