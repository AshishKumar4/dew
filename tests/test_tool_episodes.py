"""Real sampled tool actions, verifier rewards and action-only GRPO updates."""

from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import replace
import json

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.data import Dataset
from dew.inference import TextGeneration
from dew.objectives.base import Step
from dew.objectives.rl import GRPOObjective
from dew.objectives.rl.episodes import (
    EpisodeCancelled, EpisodeFailure, EpisodeRollout, EpisodeStatus, Observation,
)
from dew.sampling import Sampling
from dew.training import Checkpoints, Trainer

# Compact tool vocabulary: a call, a tool result, a final answer, and EOS.
PAD, START, CALL_THREE, EOS, NINE, ANSWER_NINE, WRONG, CALL_FOUR, SIXTEEN, ANSWER_SIXTEEN = range(10)
VOCAB = 12
PROMPT, RESPONSE, TURNS, GROUPS = 8, 3, 3, 4
SAMPLING = Sampling(temperature=.7, top_k=2, eos_id=EOS, pad_id=PAD)


def transition_logits():
    logits = np.full((VOCAB, VOCAB), -1000., np.float32)
    logits[:, EOS] = 1000.
    logits[START, :] = -1000.
    logits[START, CALL_THREE] = .2
    logits[START, CALL_FOUR] = 0.
    for observation, answer in ((NINE, ANSWER_NINE), (SIXTEEN, ANSWER_SIXTEEN)):
        logits[observation, :] = -1000.
        logits[observation, answer] = .25
        logits[observation, WRONG] = 0.
    return logits


class ToolPolicy(nn.Module):
    """A trainable bigram policy with a finite tool-call vocabulary.

    The shared sampler draws calls and final answers from logits; the test
    environment executes square(3) or square(4). No generated code executes.
    """

    vocab_size: int = VOCAB
    max_seq_len: int = PROMPT + RESPONSE
    final_logit_softcap = None
    precision = None

    def setup(self):
        self.table = self.param("table", lambda key: jnp.asarray(transition_logits()))

    def hidden_states(self, tokens, train=False, attention_mask=None, positions=None, segment_ids=None):
        return jax.nn.one_hot(tokens, self.vocab_size)

    def head_weight(self, params):
        return params["table"]

    @nn.compact
    def init_cache(self, batch_size):
        self.variable("cache", "seen", lambda: jnp.zeros(batch_size, jnp.int32))

    def __call__(self, tokens, train=False, decode=False, attention_mask=None, positions=None, segment_ids=None):
        if decode:
            seen = self.get_variable("cache", "seen")
            valid = jnp.ones_like(tokens, bool) if attention_mask is None else attention_mask
            self.put_variable("cache", "seen", seen + valid.sum(-1))
        return self.hidden_states(tokens) @ self.table


class SquareSession:
    def __init__(self, identity, harness):
        self.identity, self.harness = identity, harness
        self.answer = None

    def reset(self):
        self.harness.opened.append(self.identity)
        if self.harness.failure == "reset":
            raise OSError("environment unavailable")
        return Observation((START,))

    def step(self, action):
        assert action.tokens[-1] == EOS
        command = action.tokens[:-1]
        if self.harness.failure == "tool":
            raise OSError("tool transport failed")
        if self.harness.failure == "cancel":
            raise CancelledError("owner cancelled")
        if self.harness.failure == "cancel_status":
            return Observation((), EpisodeStatus.CANCELLED, "owner cancelled")
        if self.answer is None:
            assert len(command) == 1
            value = {CALL_THREE: 3, CALL_FOUR: 4}[command[0]]
            self.answer = value * value
            self.harness.calls.append((self.identity, value, self.answer))
            observation = {9: NINE, 16: SIXTEEN}[self.answer]
            return Observation((*action.context, *action.tokens, observation),
                               detail=f"square({value}) = {self.answer}")
        answer = {ANSWER_NINE: 9, ANSWER_SIXTEEN: 16, WRONG: 8}[command[0]]
        return Observation((), EpisodeStatus.COMPLETED, json.dumps({"answer": answer, "expected": self.answer}))


class Harness:
    def __init__(self, failure=None):
        self.failure = failure
        self.opened, self.closed, self.calls = [], [], []

    @contextmanager
    def __call__(self, identity):
        try:
            yield SquareSession(identity, self)
        finally:
            self.closed.append(identity)
            if self.failure == "close":
                raise OSError("environment release failed")


def verify(episode):
    if episode.status == EpisodeStatus.TRUNCATED:
        return -1.
    result = json.loads(episode.detail)
    return float(result["answer"] == result["expected"])


def build(harness=None, *, record=None, accumulation=1, **changes):
    model = ToolPolicy()
    objective = GRPOObjective(model, PROMPT + RESPONSE - 1, beta=.05)
    trainer = Trainer(objective, optax.sgd(.05), key=jax.random.key(19), accumulation=accumulation)
    rollout = EpisodeRollout(TextGeneration(model, objective.init(jax.random.key(0))), harness or Harness(), verify,
                             PROMPT, RESPONSE, TURNS, groups=GROUPS,
                             sampling=SAMPLING, record=record)
    return trainer, replace(rollout, **changes)


def collect(rollout, state, key=23):
    return rollout.collect(state, {"task_id": np.array([31], np.int32)}, jax.random.key(key))


def test_multiturn_actions_keep_cached_likelihoods_and_observations_out_of_targets():
    harness = Harness()
    trainer, rollout = build(harness)
    state = trainer.initial_state()

    episodes = collect(rollout, state)
    batch = rollout.project(episodes)

    assert harness.opened == harness.closed
    assert len(harness.calls) == GROUPS
    assert {episode.reward for episode in episodes} == {0., 1.}
    table = transition_logits()
    expected = np.zeros_like(batch["old_log_probs"])
    for index, episode in enumerate(episodes):
        assert episode.status == EpisodeStatus.COMPLETED
        assert len(episode.transitions) == 2
        for turn, transition in enumerate(episode.transitions):
            action = transition.action
            row = index * TURNS + turn
            assert action.terminated and len(action.tokens) == 2
            assert batch["prompt_length"][row] == len(action.context)
            np.testing.assert_array_equal(batch["input_ids"][row, PROMPT - len(action.context):PROMPT], action.context)
            context = list(action.context)
            for position, token in enumerate(action.tokens):
                logits = table[context[-1]]
                raw = jax.nn.log_softmax(jnp.asarray(logits))[token]
                cutoff = np.sort(logits)[-2]
                scores = np.where(logits >= cutoff, logits / SAMPLING.temperature, -np.inf)
                behavior = jax.nn.log_softmax(jnp.asarray(scores))[token]
                expected[row, position] = raw
                np.testing.assert_allclose(action.raw_log_probs[position], raw, atol=2e-6)
                np.testing.assert_allclose(action.behavior_log_probs[position], behavior, atol=2e-6)
                context.append(token)
        np.testing.assert_array_equal(batch["response_mask"][index * TURNS:(index + 1) * TURNS],
                                      [[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    np.testing.assert_allclose(batch["old_log_probs"], expected, atol=2e-6)
    assert np.max(np.abs(batch["old_log_probs"] - batch["behavior_log_probs"])) > .01
    targets = batch["input_ids"][:, PROMPT:][batch["response_mask"].astype(bool)]
    assert NINE not in targets and SIXTEEN not in targets

    # Ordinary GRPO rescoring sees the exact contexts that produced the actions.
    assert isinstance(trainer.objective, GRPOObjective)
    raw = trainer.objective.per_token_log_probs(
        state.params, jnp.asarray(batch["input_ids"]),
        left_padding=jnp.asarray(PROMPT - batch["prompt_length"]))[:, PROMPT - 1:]
    np.testing.assert_allclose(np.asarray(raw)[batch["response_mask"].astype(bool)],
                               expected[batch["response_mask"].astype(bool)], atol=2e-6)


def test_grpo_gradient_matches_action_only_categorical_reference():
    """Explicit categorical derivative; zero observation-target gradient.

    The numerical reference sums -A log pi(action) at the on-policy point.
    Observation and padded-turn targets would change its support and gradient.
    Largest observed difference from the categorical reference: 6.22e-09.
    """
    trainer, rollout = build()
    state = trainer.initial_state()
    episodes = collect(rollout, state)
    batch = {name: jnp.asarray(value) for name, value in rollout.project(episodes).items()}
    info = Step(state.microstep, jax.random.key(1), state.ema)

    def loss(table):
        stats, _ = trainer.objective.loss({"params": {"table": table}}, batch, info)
        return trainer.objective.reduce_loss(stats)[0]

    gradient = np.asarray(jax.grad(loss)(state.params["params"]["table"]))
    expected = np.zeros((VOCAB, VOCAB), np.float64)
    rewards = np.array([episode.reward for episode in episodes], np.float64)
    advantages = (rewards - rewards.mean()) / (rewards.std(ddof=1) + 1e-6)
    total = sum(len(turn.action.tokens) for episode in episodes for turn in episode.transitions)
    table = transition_logits().astype(np.float64)
    for episode, advantage in zip(episodes, advantages, strict=True):
        for turn in episode.transitions:
            previous = turn.action.context[-1]
            for token in turn.action.tokens:
                probabilities = np.exp(table[previous] - table[previous].max())
                probabilities /= probabilities.sum()
                probabilities[token] -= 1
                expected[previous] += advantage * probabilities / total
                previous = token
    difference = float(np.max(np.abs(gradient - expected)))
    assert difference < 2e-6, f"categorical action-gradient maximum difference: {difference}"
    np.testing.assert_array_equal(gradient[EOS], 0)
    assert np.max(np.abs(gradient[[NINE, SIXTEEN]])) > 1e-4


@pytest.mark.parametrize("failure,status", [
    ("reset", EpisodeStatus.ERROR), ("tool", EpisodeStatus.ERROR),
    ("close", EpisodeStatus.ERROR), ("cancel", EpisodeStatus.CANCELLED),
    ("cancel_status", EpisodeStatus.CANCELLED),
])
def test_environment_failures_abort_collection_and_release_sessions(failure, status):
    harness = Harness(failure)
    records = []
    trainer, rollout = build(harness, record=records.append)
    before = trainer.initial_state()
    expected = EpisodeCancelled if status == EpisodeStatus.CANCELLED else EpisodeFailure
    with pytest.raises(expected) as caught:
        collect(rollout, before)
    assert harness.opened == harness.closed
    assert len(harness.closed) == len(records) == 1
    assert caught.value.episode is records[0]
    assert records[0].status == status and records[0].reward is None
    if failure != "reset":
        assert records[0].transitions[0].action.tokens[-1] == EOS
        assert records[0].transitions[0].action.raw_log_probs


@pytest.mark.parametrize("mode", ["raise", "nan"])
def test_verifier_failure_is_not_a_zero_reward(mode):
    harness = Harness()
    def verifier(episode):
        if mode == "raise":
            raise OSError("verifier unavailable")
        return float("nan")
    trainer, rollout = build(harness, verifier=verifier)
    with pytest.raises(EpisodeFailure) as caught:
        collect(rollout, trainer.initial_state())
    assert caught.value.episode.reward is None
    assert caught.value.episode.status == EpisodeStatus.ERROR
    assert len(caught.value.episode.transitions) == 2
    assert harness.opened == harness.closed


@pytest.mark.parametrize("limits,turns,tool_calls", [
    ({"max_new_tokens": 1}, 1, 0),
    ({"max_turns": 1}, 1, GROUPS),
    ({"max_prompt_tokens": 2}, 1, GROUPS),
])
def test_truncation_preserves_actions_but_does_not_execute_partial_calls(limits, turns, tool_calls):
    harness = Harness()
    trainer, rollout = build(harness, **limits)
    episodes = collect(rollout, trainer.initial_state())
    assert len(harness.calls) == tool_calls
    assert harness.opened == harness.closed
    assert all(episode.status == EpisodeStatus.TRUNCATED and episode.reward == -1. for episode in episodes)
    assert all(len(episode.transitions) == turns for episode in episodes)
    if limits.get("max_new_tokens") == 1:
        assert all(not episode.transitions[0].action.terminated for episode in episodes)


def test_projection_rejects_cross_policy_episodes():
    trainer, rollout = build()
    episodes = collect(rollout, trainer.initial_state())
    crossed = replace(episodes[0], policy_step=episodes[0].policy_step + 1)
    with pytest.raises(ValueError, match="policy snapshot"):
        rollout.project((crossed, *episodes[1:]))
    changed = replace(episodes[0].transitions[0].action, policy_step=7)
    crossed = replace(episodes[0], transitions=(replace(episodes[0].transitions[0], action=changed),
                                               *episodes[0].transitions[1:]))
    with pytest.raises(ValueError, match="policy snapshot"):
        rollout.project((crossed, *episodes[1:]))


class Tasks:
    def __init__(self):
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        ids = np.arange(jax.device_count(), dtype=np.int32) + self.index * 100
        self.index += 1
        return {"task_id": ids}

    def get_state(self):
        return json.dumps(self.index).encode()

    def set_state(self, state):
        self.index = json.loads(state)


def run(directory, steps, records, accumulation=1):
    trainer, rollout = build(record=records.append, accumulation=accumulation)
    trainer.rollout = rollout
    trainer.checkpoints = Checkpoints(str(directory))
    data = Dataset(train=Tasks, val=None, records=None, batch=jax.device_count())
    return trainer.fit(data, steps=steps, log_every=1)


@pytest.mark.parametrize("accumulation", [1, 2])
def test_trainer_update_and_checkpoint_continue_without_replaying_committed_episodes(tmp_path, accumulation):
    complete, resumed = [], []
    continuous = run(tmp_path / "continuous", 2, complete, accumulation)
    first = run(tmp_path / "interrupted", 1, resumed, accumulation)
    assert int(first.step) == 1 and int(first.updates) == 1 // accumulation
    second = run(tmp_path / "interrupted", 2, resumed, accumulation)
    assert int(second.step) == 2 and int(second.updates) == 2 // accumulation
    logits = np.asarray(ToolPolicy().apply(second.params, jnp.array([[NINE], [SIXTEEN]])))[:, 0]
    probabilities = jax.nn.softmax(logits, axis=-1)
    initial = jax.nn.softmax(jnp.asarray(transition_logits()[[NINE, SIXTEEN]]), axis=-1)
    assert float(probabilities[0, ANSWER_NINE] + probabilities[1, ANSWER_SIXTEEN]) > float(
        initial[0, ANSWER_NINE] + initial[1, ANSWER_SIXTEEN]) + 1e-5
    for expected, actual in zip(jax.tree.leaves(continuous), jax.tree.leaves(second), strict=True):
        if jnp.issubdtype(expected.dtype, jax.dtypes.prng_key):
            expected, actual = jax.random.key_data(expected), jax.random.key_data(actual)
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
    assert [episode.identity for episode in resumed] == [episode.identity for episode in complete]
    assert len({episode.identity for episode in resumed}) == len(resumed)
    assert {episode.policy_step for episode in resumed} == set(range(2 // accumulation))
    for uninterrupted, restored in zip(complete, resumed, strict=True):
        assert uninterrupted == restored


def test_collection_does_not_follow_live_mapping_replacements():
    trainer, rollout = build()
    state = trainer.initial_state()
    live = {"params": dict(state.params["params"])}
    state = replace(state, params=live)
    original = np.asarray(live["params"]["table"]).copy()

    def changing_verifier(episode):
        # Replacing caller-owned weights between episodes must not change the
        # snapshot already supplied to collection.
        live["params"]["table"] = jnp.zeros_like(live["params"]["table"])
        return verify(episode)

    episodes = collect(replace(rollout, verifier=changing_verifier), state)

    for episode in episodes:
        for turn in episode.transitions:
            previous = turn.action.context[-1]
            for token, recorded in zip(turn.action.tokens, turn.action.raw_log_probs, strict=True):
                np.testing.assert_allclose(recorded,
                    jax.nn.log_softmax(jnp.asarray(original[previous]))[token], atol=2e-6)
                previous = token


@pytest.mark.parametrize("failure", ["tool", "cancel"])
def test_aborted_episode_leaves_the_previous_trainer_checkpoint_intact(tmp_path, failure):
    directory = tmp_path / "run"
    first = run(directory, 1, [])
    expected = np.asarray(first.params["params"]["table"]).copy()
    records = []
    trainer, rollout = build(Harness(failure), record=records.append)
    trainer.rollout = rollout
    trainer.checkpoints = Checkpoints(str(directory))
    data = Dataset(train=Tasks, val=None, records=None, batch=jax.device_count())
    exception = EpisodeFailure if failure == "tool" else EpisodeCancelled

    with pytest.raises(exception):
        trainer.fit(data, steps=2)

    restored, _, _ = trainer.place()
    assert int(restored.step) == int(restored.updates) == 1
    np.testing.assert_array_equal(restored.params["params"]["table"], expected)
    assert len(records) == 1 and records[0].identity.attempt == 1
    assert records[0].reward is None


@pytest.mark.parametrize("corruption", ["context", "termination"])
def test_inference_provenance_is_checked_before_tool_execution(corruption):
    harness = Harness()
    trainer, rollout = build(harness)
    class CorruptPolicy:
        def __init__(self, inner):
            self.inner = inner

        def bind(self, variables):
            return CorruptPolicy(self.inner.bind(variables))

        def __call__(self, inputs, max_new_tokens, *, key, sampling):
            result = self.inner(inputs, max_new_tokens, key=key, sampling=sampling)
            if corruption == "context":
                return replace(result, tokens=result.tokens.at[0, 0].set(PAD))
            return replace(result, terminated=~result.terminated)

    with pytest.raises(EpisodeFailure, match="inference") as caught:
        collect(replace(rollout, policy=CorruptPolicy(rollout.policy)), trainer.initial_state())
    assert harness.calls == [] and harness.opened == harness.closed
    assert caught.value.episode.reward is None


def test_projection_refuses_replayed_samples_and_cross_attempt_groups():
    trainer, rollout = build()
    episodes = collect(rollout, trainer.initial_state())
    with pytest.raises(ValueError, match="distinct identity"):
        rollout.project((episodes[0], episodes[0], *episodes[2:]))
    changed = replace(episodes[0], identity=replace(episodes[0].identity, attempt=7))
    with pytest.raises(ValueError, match="policy snapshot and attempt"):
        rollout.project((changed, *episodes[1:]))


def test_verifier_runs_before_environment_resources_are_released():
    harness = Harness()

    def live_verifier(episode):
        assert episode.identity in harness.opened
        assert episode.identity not in harness.closed, "verifier lost its environment resources"
        return verify(episode)

    trainer, rollout = build(harness, verifier=live_verifier)
    episodes = collect(rollout, trainer.initial_state())
    assert {episode.reward for episode in episodes} == {0., 1.}
    assert harness.opened == harness.closed
