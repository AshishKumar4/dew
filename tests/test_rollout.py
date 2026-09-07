"""The rollout capability: sampling between the data stream and the step.

The trainer calls a `Rollout` with the state, the prefetched batch and a key
folded twice off the run key, then reshards the result and compiles the step
once. With `rollout=None` the prefetched batch trains untouched. A
`SampledRollout` fills that contract for online RL: G completions per prompt
from `dew.sampling.generate`, scored by a reward call, advantaged with the
group or RLOO family from `dew.rl`, with old log-probabilities rescored
through the objective's own head.
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn

from dew.objectives.base import Aux, EMASpec, Objective
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling
from dew.objectives.rl import SampledRollout
from dew.rl import group_advantage, rloo_advantage
from dew.training import Checkpoints, Layout, Trainer

FEATURES = 3
VOCAB = 8
PROMPT_WIDTH = 8
NEW_TOKENS = 4
GROUPS = 2


class Affine(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(2)(x)


class Regression(Objective):
    """Squared error of an affine map, for the loop-level tests."""

    def __init__(self):
        self.model = Affine()
        self.ema = EMASpec(decay=optax.constant_schedule(1.0))

    def init(self, key):
        return self.model.init(key, jnp.zeros((1, FEATURES)))

    def loss(self, params, batch, step):
        prediction = self.model.apply(params, batch["x"])
        return jnp.mean((prediction - batch["y"]) ** 2), Aux({"probe": jnp.asarray(1.0)})


class Counting:
    """A deterministic, checkpointable stream of regression batches."""

    def __init__(self, batch=8):
        self.index = 0
        self.batch = batch

    def __iter__(self):
        return self

    def __next__(self):
        rng = np.random.default_rng(self.index)
        self.index += 1
        x = rng.normal(size=(self.batch, FEATURES)).astype(np.float32)
        return {"x": x, "y": 2 * x[:, :2]}

    def get_state(self):
        return json.dumps({"index": self.index}).encode()

    def set_state(self, state):
        self.index = json.loads(state)["index"]


class Data:
    """The `Dataset` contract the trainer reads: train, val, batch, records."""

    def __init__(self, train=Counting, val=None, batch=8, records=None):
        self._train, self._val = train, val
        self.batch, self.records = batch, records

    def train(self):
        return self._train()

    @property
    def val(self):
        return self._val

    @property
    def steps_per_epoch(self):
        return None if self.records is None else self.records // self.batch


def leaves(state):
    return [np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(state.params)]


def make_trainer(tmp_path=None, **kwargs):
    checkpoints = None if tmp_path is None else Checkpoints(str(tmp_path / "run"))
    return Trainer(
        Regression(),
        optax.sgd(0.1),
        key=jax.random.key(0),
        layout=Layout(min_shard=1, tolerance=1.0),
        checkpoints=checkpoints,
        **kwargs,
    )


class Recorder:
    """A rollout that returns the batch untouched and keeps every key."""

    def __init__(self):
        self.keys = []
        self.steps = []

    def __call__(self, state, batch, key):
        self.keys.append(np.asarray(jax.random.key_data(key)).tobytes())
        self.steps.append(int(state.step))
        return batch


class Logging:
    """A tracker that keeps every logged scalars mapping."""

    def __init__(self):
        self.scalars = []

    def log(self, scalars, step):
        self.scalars.append(dict(scalars))

    def artifact(self, value, step):
        pass


# --- the loop with and without a rollout -------------------------------------

def test_an_identity_rollout_leaves_the_loop_byte_identical():
    """The same two steps with `rollout=None` and with a rollout that hands
    the batch back: identical parameters and identical logged losses. The
    rollout lives outside the compiled step, so None adds no ops to the
    loop; the compiled step is unreachable for a direct jaxpr comparison
    because it runs under `set_mesh`, which refuses to trace."""
    assert make_trainer().rollout is None
    plain, ident = Logging(), Logging()
    make_trainer(tracker=plain).fit(Data(), steps=2, log_every=1)
    make_trainer(rollout=lambda state, batch, key: batch,
                 tracker=ident).fit(Data(), steps=2, log_every=1)

    assert [logged["train/loss"] for logged in plain.scalars if "train/loss" in logged] == [
        logged["train/loss"] for logged in ident.scalars if "train/loss" in logged] != []


def test_changing_batches_compile_once():
    """A rollout that rewrites the batch content every call, shapes fixed:
    the trainer compiles the step on the first rolled batch and never again."""
    calls = []

    def shifting(state, batch, key):
        calls.append(1)
        return {"x": batch["x"] + len(calls), "y": batch["y"]}

    trainer = make_trainer(rollout=shifting)
    compiles = [trainer.compile]
    trainer.compile = lambda *args: (compiles.append(1), compiles[0](*args))[1]

    trainer.fit(Data(), steps=4, log_every=4)

    assert len(calls) == 4
    assert len(compiles) == 2, "the step compiled more than once"


def test_rollout_keys_fold_forward_and_reproduce():
    """Three steps draw three distinct keys, and a second identical run draws
    the same three: the rollout stream is checkpointed randomness, not noise."""
    first, second = Recorder(), Recorder()
    make_trainer(rollout=first).fit(Data(), steps=3, log_every=3)
    make_trainer(rollout=second).fit(Data(), steps=3, log_every=3)
    assert first.steps == [0, 1, 2]
    assert len(set(first.keys)) == 3
    assert first.keys == second.keys

def test_a_resumed_run_does_not_replay_rollouts(tmp_path):
    """Two steps, then the resumed run samples steps 2 and 3, never 0 and 1
    again."""
    rollout = Recorder()
    make_trainer(tmp_path, rollout=rollout).fit(Data(), steps=2, log_every=2)
    make_trainer(tmp_path, rollout=rollout).fit(Data(), steps=4, log_every=4)

    assert rollout.steps == [0, 1, 2, 3]


def test_rollout_seconds_logs_only_with_a_rollout():
    """The log tick carries `train/rollout_seconds` when a rollout is set,
    and no such key without one. The run's final goodput line is not a step
    tick, so only ticks with a loss count."""
    tracking = Logging()
    make_trainer(rollout=Recorder(), tracker=tracking).fit(Data(), steps=2, log_every=1)
    ticks = [logged for logged in tracking.scalars if "train/loss" in logged]
    assert len(ticks) == 2
    assert all(tick["train/rollout_seconds"] >= 0.0 for tick in ticks)

    tracking = Logging()
    make_trainer(tracker=tracking).fit(Data(), steps=2, log_every=1)
    ticks = [logged for logged in tracking.scalars if "train/loss" in logged]
    assert len(ticks) == 2
    assert all("train/rollout_seconds" not in tick for tick in ticks)

class TinyHead(nn.Module):
    """A position-wise map with the backbone's scoring contract, standing in
    for the causal stack: int32 ids in, float32 logits out, the head split
    off behind `hidden_states` and `head_weight`."""

    vocab_size: int
    final_logit_softcap = None
    precision = None

    def setup(self):
        self.lm_head = nn.Dense(self.vocab_size, use_bias=False)

    @nn.compact
    def hidden_states(self, tokens, train: bool = False):
        x = nn.Embed(self.vocab_size, 8)(tokens)
        h = nn.LayerNorm()(x)
        return nn.LayerNorm()(x + nn.Dense(8)(nn.gelu(nn.Dense(16)(h))))

    @nn.compact
    def init_cache(self, batch_size):
        """A placeholder cache: the trunk mixes nothing across positions, so
        incremental decoding keeps no state, but `generate` threads one."""
        self.variable("cache", "index", lambda: jnp.zeros((batch_size,), jnp.int32))

    def __call__(self, tokens, train: bool = False, decode: bool = False):
        return self.lm_head(
            self.hidden_states(tokens, train=train)).astype(jnp.float32)

    def head_weight(self, params):
        return params["lm_head"]["kernel"].astype(jnp.float32)
# --- SampledRollout ------------------------------------------------------------


class Calls:
    """Records each call's arguments; scores 1 when the completion's first token matches the reference."""

    def __init__(self):
        self.seen = []

    def __call__(self, data_source, completion, ground_truth, extra_info):
        self.seen.append((data_source, completion, ground_truth, extra_info))
        return 1.0 if completion.split()[:1] == [ground_truth] else 0.0


def tiny_objective(seq=PROMPT_WIDTH + NEW_TOKENS - 1):
    return LMObjective(TinyHead(vocab_size=VOCAB), seq)


def prompt_batch(rows=2):
    """Two fixed-width prompt rows with distinct reward contexts."""
    width = PROMPT_WIDTH
    prompts = np.zeros((rows, width), np.int32)
    prompts[0] = [1, 2, 3, 4, 5, 6, 7, 1]
    prompts[1] = [2, 3, 4, 5, 6, 7, 1, 2]
    info = max(len("rule"), len("other"), 1)
    pad = lambda text: np.pad(
        np.frombuffer(text.encode(), np.uint8).astype(np.int32), (0, info - len(text)))
    return {
        "prompt": prompts,
        "prompt_length": np.full(rows, width, np.int32),
        "data_source": np.stack([pad("rule"), pad("other")]),
        "ground_truth": np.stack([pad("1"), pad("2")]),
        "extra_info": np.stack([pad(""), pad("")]),
    }


class FakeState:
    def __init__(self, params):
        self.params = params


def test_a_sampled_rollout_packs_fixed_shapes():
    """G completions per prompt from the real sampler, greedy and
    deterministic: full-bleed rectangles with the prompt order kept and
    groups contiguous inside each row."""
    objective = tiny_objective()
    params = objective.init(jax.random.key(0))
    reward = Calls()
    rollout = SampledRollout(objective, reward, groups=GROUPS,
                             max_new_tokens=NEW_TOKENS, sampling=Sampling(temperature=0.0))
    batch = prompt_batch()

    out = rollout(FakeState(params), batch, jax.random.key(1))

    assert out["input_ids"].shape == (4, PROMPT_WIDTH + NEW_TOKENS)
    assert out["response_mask"].shape == (4, NEW_TOKENS)
    assert out["old_log_probs"].shape == (4, NEW_TOKENS)
    assert out["advantages"].shape == (4, NEW_TOKENS)
    assert out["rewards"].shape == (4,)
    assert out["prompt_length"].shape == (4,)
    np.testing.assert_array_equal(out["input_ids"][:, :PROMPT_WIDTH],
                                  np.repeat(batch["prompt"], GROUPS, axis=0))
    np.testing.assert_array_equal(out["response_mask"], np.ones((4, NEW_TOKENS)))
    again = rollout(FakeState(params), batch, jax.random.key(1))
    for key in out:
        np.testing.assert_array_equal(np.asarray(out[key]), np.asarray(again[key]))


def test_rewards_and_advantages_follow_the_calls():
    """The reward sees each row's own context, and the advantages are the
    group family over exactly the rewards the calls returned."""
    objective = tiny_objective()
    params = objective.init(jax.random.key(0))
    reward = Calls()
    rollout = SampledRollout(objective, reward, groups=GROUPS,
                             max_new_tokens=NEW_TOKENS, sampling=Sampling(temperature=0.0))

    out = rollout(FakeState(params), prompt_batch(), jax.random.key(1))

    assert len(reward.seen) == 4
    assert [seen[0] for seen in reward.seen] == ["rule", "rule", "other", "other"]
    assert [seen[2] for seen in reward.seen] == ["1", "1", "2", "2"]
    assert all(isinstance(seen[1], str) for seen in reward.seen)
    expected = np.asarray(group_advantage(
        np.asarray(out["rewards"], np.float32).reshape(-1), GROUPS), np.float32)
    np.testing.assert_allclose(
        np.asarray(out["advantages"]).reshape(2, GROUPS, NEW_TOKENS),
        np.broadcast_to(expected.reshape(2, GROUPS)[..., None], (2, GROUPS, NEW_TOKENS)),
        rtol=1e-5)


def test_old_log_probs_come_from_the_sampling_head():
    """The stored log-probabilities are the objective's own head over the
    concatenation, at the response slice: position p predicts token p + 1, so
    the response starts one before the prompt width."""
    objective = tiny_objective()
    params = objective.init(jax.random.key(0))
    rollout = SampledRollout(objective, Calls(), groups=GROUPS,
                             max_new_tokens=NEW_TOKENS, sampling=Sampling(temperature=0.0))

    out = rollout(FakeState(params), prompt_batch(), jax.random.key(1))

    rescored = np.asarray(objective.per_token_log_probs(params, out["input_ids"]))
    np.testing.assert_allclose(
        np.asarray(out["old_log_probs"]),
        rescored[:, PROMPT_WIDTH - 1:PROMPT_WIDTH - 1 + NEW_TOKENS], rtol=1e-5)


def test_the_mask_stops_after_the_first_stop_token():
    """With no stop token every response counts whole; when the first
    response token is the stop token, only it counts."""
    objective = tiny_objective()
    params = objective.init(jax.random.key(0))
    batch = prompt_batch()
    first = SampledRollout(objective, Calls(), groups=GROUPS,
                           max_new_tokens=NEW_TOKENS, sampling=Sampling(temperature=0.0))(
                               FakeState(params), batch, jax.random.key(1))
    stop = int(np.asarray(first["input_ids"])[0, PROMPT_WIDTH])

    stopped = SampledRollout(objective, Calls(), groups=GROUPS,
                             max_new_tokens=NEW_TOKENS, sampling=Sampling(temperature=0.0, eos_id=stop))(FakeState(params), batch, jax.random.key(1))

    np.testing.assert_array_equal(np.asarray(stopped["response_mask"])[0], [1, 0, 0, 0])


def test_rloo_advantages_keep_fixed_shapes():
    """The leave-one-out family flows through the same rectangles."""
    objective = tiny_objective()
    params = objective.init(jax.random.key(0))
    rollout = SampledRollout(objective, Calls(), groups=GROUPS,
                             max_new_tokens=NEW_TOKENS, sampling=Sampling(temperature=0.0), sample="rloo")

    out = rollout(FakeState(params), prompt_batch(), jax.random.key(1))

    expected = np.asarray(rloo_advantage(
        np.asarray(out["rewards"], np.float32).reshape(-1), GROUPS), np.float32)
    np.testing.assert_allclose(
        np.asarray(out["advantages"])[:, 0], expected.reshape(-1), rtol=1e-5)


def test_a_misconfigured_rollout_is_refused():
    objective = tiny_objective()
    with pytest.raises(ValueError, match="at least two"):
        SampledRollout(objective, Calls(), groups=1)
    with pytest.raises(ValueError, match="at least one token"):
        SampledRollout(objective, Calls(), max_new_tokens=0)
    with pytest.raises(ValueError, match="advantage families"):
        SampledRollout(objective, Calls(), sample="best_of_n")


def test_a_misaligned_objective_is_refused():
    """The concatenation must fit the objective's window: seq_len one below
    the prompt width plus new tokens, else the rescoring reads the wrong
    slice."""
    rollout = SampledRollout(tiny_objective(seq=4), Calls(), groups=GROUPS,
                             max_new_tokens=NEW_TOKENS, sampling=Sampling(temperature=0.0))
    params = tiny_objective(seq=4).init(jax.random.key(0))
    with pytest.raises(ValueError, match="one below"):
        rollout(FakeState(params), prompt_batch(), jax.random.key(1))
