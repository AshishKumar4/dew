"""Asynchronous rollouts: draws run ahead of the update, and no batch trains staler than max_lag.

`AsyncRollout` submits the next batches while the trainer updates on the
current one. The checks: a batch drawn one update ago trains with its policy
version on every row and its old likelihoods rescored under the current
weights; a batch that fell further behind than `max_lag` is drawn again under
freshly pushed weights; configurations that could exceed the bound are
refused; and a real `Trainer` run over Dew's own continuous-batching server
keeps every consumed batch within the bound while the server's weights follow
the updates.
"""

import itertools
from concurrent.futures import Future

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.data import Dataset
from dew.inference import Draw, NativeRolloutServer, TextGeneration
from dew.inference.serving import Server
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.rl import POLICY_VERSION_KEY, AsyncRollout, GRPOObjective
from dew.sampling import Sampling
from dew.training import Layout, Trainer

VOCAB = 13
EOS = 12
WIDTH = 6
BUDGET = 4
GROUPS = 2
SAMPLING = Sampling(temperature=1.0, eos_id=EOS)


def model():
    return CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=1, num_heads=2,
                             head_dim=8, mlp_features=32, max_seq_len=64, dtype="float32")


def objective():
    return GRPOObjective(model(), seq_len=WIDTH + BUDGET - 1, behavior_importance_cap=2.0)


def prompt_batch(first, rows=2):
    """`rows` prompts, alternately four tokens left-padded and the full width."""
    prompts = np.zeros((rows, WIDTH), np.int32)
    lengths = np.zeros(rows, np.int32)
    for row in range(rows):
        lengths[row] = 4 if row % 2 == 0 else WIDTH
        prompts[row, WIDTH - lengths[row]:] = (first + row + np.arange(lengths[row])) % (VOCAB - 1)
    empty = np.zeros((rows, 1), np.int32)
    return {"prompt": prompts, "prompt_length": lengths,
            "data_source": empty, "ground_truth": empty, "extra_info": empty}


def batches(rows=2):
    for first in itertools.cycle(range(1, 11)):
        yield prompt_batch(first, rows)


class State:
    def __init__(self, params, updates):
        self.params, self.updates = params, updates


class Recording:
    """A rollout server whose draws are fixed ids with known likelihoods.

    `load` records every push and moves the version, as a real server does.
    """

    def __init__(self):
        self.version = 0
        self.loads = []
        self.submitted = []

    @property
    def sampling(self):
        return SAMPLING

    def submit(self, prompt, max_new_tokens, *, seed):
        self.submitted.append((tuple(prompt), self.version))
        future = Future()
        tokens = (1 + seed % 5, 3, EOS)
        future.set_result(Draw(tuple(prompt), tokens, (-.5, -.25, -.125), (-.4, -.2, -.1), True, self.version))
        return future

    def load(self, variables, version):
        self.loads.append(version)
        self.version = version

    def close(self):
        pass


def first_token_reward(source, completion, truth, info):
    words = completion.split()
    return float(words[0]) if words else 0.0


def decode(ids):
    return " ".join(str(token) for token in ids)


def rollout_over(server, **options):
    target = objective()
    params = target.init(jax.random.key(0))
    records = []
    rollout = AsyncRollout(target, server, first_token_reward, decode=decode, groups=GROUPS,
                           max_new_tokens=BUDGET, log=records.append, **options)
    data = rollout.prompts(Dataset(train=batches, val=None, records=None, batch=2))
    return rollout, iter(data.train()), params, records


def test_a_batch_one_update_old_trains_with_its_version_and_rescored_old_likelihoods():
    server = Recording()
    rollout, stream, params, records = rollout_over(server, max_lag=1, ahead=1)
    first, second = next(stream), next(stream)

    fresh = rollout(State(params, 0), first, jax.random.key(1))
    # The second batch was submitted during the first call, under version 0.
    assert [version for _, version in server.submitted] == [0] * 8
    np.testing.assert_array_equal(fresh[POLICY_VERSION_KEY], np.zeros(4, np.int32))
    # Lag zero with the server's raw likelihoods: those are the old policy.
    np.testing.assert_allclose(fresh["old_log_probs"][:, :3], np.tile([-.4, -.2, -.1], (4, 1)))
    np.testing.assert_allclose(fresh["behavior_log_probs"][:, :3], np.tile([-.5, -.25, -.125], (4, 1)))

    stale = rollout(State(params, 1), second, jax.random.key(2))
    assert server.loads == [1]
    np.testing.assert_array_equal(stale[POLICY_VERSION_KEY], np.zeros(4, np.int32))
    assert records[-1].lag == 1 and records[-1].redrawn == 0
    # One update behind, the old policy is the trainer's own, rescored.
    rescored = rollout.objective.per_token_log_probs(
        params, stale["input_ids"], left_padding=WIDTH - stale["prompt_length"])[:, WIDTH - 1:WIDTH - 1 + BUDGET]
    np.testing.assert_allclose(stale["old_log_probs"], np.asarray(rescored) * stale["response_mask"], rtol=1e-6)
    assert not np.allclose(stale["old_log_probs"][:, :3], np.tile([-.4, -.2, -.1], (4, 1)))
    np.testing.assert_allclose(stale["behavior_log_probs"][:, :3], np.tile([-.5, -.25, -.125], (4, 1)))


def test_a_batch_past_max_lag_is_drawn_again_under_pushed_weights():
    server = Recording()
    rollout, stream, params, records = rollout_over(server, max_lag=1, ahead=1)
    first, second = next(stream), next(stream)
    rollout(State(params, 0), first, jax.random.key(1))

    # Three updates land before the second batch is consumed, as after a
    # resume or several accumulated commits: its version-0 draws are 3 behind.
    out = rollout(State(params, 3), second, jax.random.key(2))
    assert records[-1].version == 3 and records[-1].lag == 0 and records[-1].redrawn == 1
    np.testing.assert_array_equal(out[POLICY_VERSION_KEY], np.full(4, 3, np.int32))


def test_a_stalled_push_is_caught_at_consumption():
    class Stalling(Recording):
        """Pushes that silently fail to move the served weights until asked twice."""

        def load(self, variables, version):
            self.loads.append(version)
            if len(self.loads) > 1:
                self.version = version

    server = Stalling()
    rollout, stream, params, records = rollout_over(server, max_lag=1, ahead=1)
    first, second = next(stream), next(stream)
    rollout(State(params, 0), first, jax.random.key(1))
    out = rollout(State(params, 2), second, jax.random.key(2))
    assert records[-1].redrawn == 1 and records[-1].lag == 0
    assert np.all(2 - out[POLICY_VERSION_KEY] <= 1)


@pytest.mark.parametrize("options, message", [
    ({"max_lag": 1, "ahead": 2}, "past max_lag"),
    ({"max_lag": 1, "ahead": 1, "sync_every": 2}, "past max_lag"),
    ({"max_lag": 0, "ahead": 1}, "past max_lag"),
])
def test_a_schedule_that_could_exceed_the_bound_is_refused(options, message):
    with pytest.raises(ValueError, match=message):
        AsyncRollout(objective(), Recording(), first_token_reward, decode=decode, **options)


def test_stale_rollouts_without_the_importance_correction_are_refused():
    with pytest.raises(ValueError, match="behavior_importance_cap"):
        AsyncRollout(GRPOObjective(model(), seq_len=WIDTH + BUDGET - 1), Recording(), first_token_reward,
                     decode=decode, max_lag=1)


def test_a_batch_that_did_not_come_through_the_stream_is_refused():
    rollout, stream, params, _ = rollout_over(Recording(), max_lag=1)
    next(stream)
    with pytest.raises(ValueError, match="next registered"):
        rollout(State(params, 0), prompt_batch(7), jax.random.key(1))


def test_a_trainer_run_on_the_native_server_stays_within_the_bound():
    target = objective()
    params = target.init(jax.random.key(0))
    server = NativeRolloutServer(Server.from_task(TextGeneration(target.model, params, None, sampling=SAMPLING),
                                                  slots=8, capacity=64))
    records = []
    rollout = AsyncRollout(target, server, first_token_reward, decode=decode, groups=GROUPS,
                           max_new_tokens=BUDGET, max_lag=2, ahead=1, sync_every=2, log=records.append)
    try:
        trainer = Trainer(target, optax.adam(1e-2), key=jax.random.key(3), rollout=rollout,
                          layout=Layout(min_shard=1, tolerance=1.0))
        # One prompt per device of the test's eight-device host platform.
        stream = rollout.prompts(Dataset(train=lambda: batches(jax.device_count()), val=None, records=None,
                                         batch=jax.device_count()))
        state = trainer.fit(stream,
                            steps=6, log_every=6)
    finally:
        server.close()
        rollout.close()
    assert int(state.updates) == 6
    assert [record.updates for record in records] == list(range(6))
    assert all(0 <= record.lag <= 2 for record in records)
    assert any(record.lag > 0 for record in records), "no batch was ever drawn ahead of its update"
    # Pushed every second update: the served tree is the one after update 4.
    assert server.version == 4
    assert not all(jnp.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(state.params)))


def test_the_native_server_serves_loaded_weights_at_its_own_precision():
    target = objective()
    params = target.init(jax.random.key(0))
    served = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), params)
    backend = Server.from_task(TextGeneration(target.model, served, None, sampling=SAMPLING), slots=2, capacity=64)
    server = NativeRolloutServer(backend)
    try:
        before = server.submit([1, 2, 3], BUDGET, seed=5).result()
        trained = jax.tree.map(lambda leaf: leaf * 1.5, params)
        server.load(trained, 7)
        after = server.submit([1, 2, 3], BUDGET, seed=5).result()
    finally:
        server.close()
    assert (before.version, after.version) == (0, 7)
    assert before.behavior_log_probs != after.behavior_log_probs
    for leaf, expected in zip(jax.tree.leaves(backend.variables), jax.tree.leaves(trained), strict=True):
        assert leaf.dtype == jnp.bfloat16
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(expected.astype(jnp.bfloat16)))


def test_a_refused_request_raises_at_submit_and_the_native_server_keeps_serving():
    target = objective()
    params = target.init(jax.random.key(0))
    server = NativeRolloutServer(Server.from_task(TextGeneration(target.model, params, None, sampling=SAMPLING),
                                                  slots=2, capacity=64))
    try:
        running = server.submit([1, 2, 3], 40, seed=1)
        with pytest.raises(ValueError, match="max_seq_len"):
            server.submit(list(range(1, 11)) * 6, 10, seed=2)
        assert len(running.result(timeout=120).tokens) >= 1
        assert server.submit([4, 5], 3, seed=3).result(timeout=120).prompt == (4, 5)
    finally:
        server.close()
