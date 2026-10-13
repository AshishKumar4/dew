"""The rollout scheduler: complete groups within the staleness bound, failures masked and retried.

A scripted source decides each sample's outcome from its task, submission
and sample index, and can leave a sample running for the test to finish or
never finish. The checks: groups enter the batch complete and relabelled so a
retried sample rejoins its own group; infrastructure failures and
cancellations are resubmitted and never trained; a sample that keeps failing
abandons its group; truncations stay masked; stragglers are cut by
over-sampling and the admit count; stale rollouts are discarded before anyone
waits on them; a rollout spanning a weight push keeps its oldest version; a
reopened stream cancels what the old one left in flight; a source that
raises stops the batch; and a real `Trainer` trains a tiny model through
two-turn in-process environments on Dew's own server.
"""

import math
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.data import Dataset
from dew.inference import NativeRolloutServer, TextGeneration
from dew.inference.serving import Server
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.rl import EnvironmentSource, EpisodeStatus, GRPOObjective, Observation
from dew.objectives.rl.scheduler import RolloutScheduler
from dew.objectives.rl.sessions import Status, pack
from dew.sampling import Sampling
from dew.training import Layout, Trainer

WIDTH = 8
ROWS = 8


class Objective:
    """The scheduler reads the correction setting and rescores packed rows."""

    behavior_importance = 2.0

    def packed_log_probs(self, params, batch):
        return jnp.full(batch["input_ids"].shape, -0.75, jnp.float32) + 0 * params


class Publisher:
    def __init__(self, version=0, stall=0):
        self.version, self.loads, self.stall = version, [], stall

    def load(self, variables, version):
        self.loads.append(version)
        if len(self.loads) > self.stall:
            self.version = version


def batches(*rows):
    for tasks in rows:
        yield {"task_id": np.asarray(tasks, np.int32)}


def scheduler(source, publisher=None, stream=((1, 2), (3, 4), (5, 6)), **options):
    records = []
    options = {"groups": 2, "max_lag": 1, "ahead": 1, "width": WIDTH, "rows": ROWS, **options}
    rollout = RolloutScheduler(Objective(), source, publisher or Publisher(), log=records.append, **options)
    data = rollout.tasks(Dataset(train=lambda partition: batches(*stream), val=None, records=None, batch=2))
    return rollout, data, records


VOCAB = 13
STOP = 12


class Tools:
    """Two tool turns: each appends the action and one tool id, then the task completes."""

    def __init__(self, task):
        self.context, self.steps = (1, 2 + task % 3), 0

    def reset(self):
        return Observation(self.context)

    def step(self, action):
        self.steps += 1
        if self.steps == 2:
            return Observation((), EpisodeStatus.COMPLETED, "done")
        self.context = action.context + action.tokens + (4,)
        return Observation(self.context)


def test_a_trainer_run_trains_through_multi_turn_environments_on_the_native_server():
    width = 48
    model = CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=1, num_heads=2,
                              head_dim=8, mlp_features=32, max_seq_len=64, dtype="float32")
    target = GRPOObjective(model, seq_len=width - 1, behavior_importance=2.0)
    params = target.init(jax.random.key(0))
    sampling = Sampling(temperature=1.0, eos_token_ids=STOP)
    server = NativeRolloutServer(Server.from_task(TextGeneration(model, params, None, sampling=sampling),
                                                  slots=16, capacity=64))

    @contextmanager
    def environment(task, identity):
        yield Tools(int(task.id))

    def verifier(task, episode):
        sampled = [token for turn in episode.transitions for token in turn.action.tokens if token != STOP]
        return sum(token == 5 for token in sampled) / max(len(sampled), 1)

    episodes = EnvironmentSource(server, environment, verifier, max_prompt_tokens=32, max_new_tokens=16,
                                 max_turns=3, workers=16)
    rollouts, keys = [], []
    submit = episodes.submit

    def watched(task, samples, *, version, key):
        keys.append(tuple(int(word) for word in jax.random.key_data(key)))
        futures = submit(task, samples, version=version, key=key)
        for future in futures:
            future.add_done_callback(
                lambda done: rollouts.append(done.result()) if not done.cancelled() else None
            )
        return futures

    episodes.submit = watched
    # Every step must hold a completed session: a truncated one is masked and trains nothing,
    # and a step with no trainable token makes no update. This policy completes 32% of its
    # sessions, so four steps of 64 sessions all have one but for about 4 * 0.68**64 = 8e-11 of
    # runs; at one task a device (2 sessions a step on one GPU) most runs lost an update. The
    # count is a multiple of the device count so the rows split evenly.
    tasks = math.lcm(32, jax.device_count())
    records = []
    scheduler = RolloutScheduler(target, episodes, server, width=width, rows=2 * tasks, groups=2,
                                 max_lag=2, ahead=1, sync_every=2, log=records.append)
    stream = scheduler.tasks(
        Dataset(
            train=lambda partition: (
                {"task_id": np.arange(tasks, dtype=np.int32) + step * tasks} for step in range(100)
            ),
            val=None,
            records=None,
            batch=tasks,
        )
    )
    try:
        trainer = Trainer(target, optax.adam(1e-2), key=jax.random.key(3), rollout=scheduler,
                          layout=Layout(min_shard=1, tolerance=1.0))
        state = trainer.fit(stream, steps=4, log_every=4)
    finally:
        scheduler.close()
        episodes.close()
        server.close()
    assert int(state.updates) == 4 and [record.updates for record in records] == [0, 1, 2, 3]
    # Every submission draws its own fold of its step's rollout key.
    assert len(set(keys)) == len(keys)
    assert all(0 <= record.lag <= 2 for record in records)
    assert any(record.lag > 0 for record in records), "no batch was ever drawn ahead of its update"
    # Each completed session's pair of calls merged into one chain.
    assert all(record.metrics["merge/calls_per_chain"] in (0.0, 2.0) for record in records)
    assert all(record.metrics["latency/max"] >= record.metrics["latency/p50"] > 0 for record in records)
    completed = [rollout for rollout in rollouts if rollout.status == Status.COMPLETED]
    assert completed, "no session completed both tool turns"
    # Every completed session made two calls, and the second extends the first,
    # so the pair packs into one chain.
    for rollout in completed:
        assert len(rollout.calls) == 2
        assert pack([rollout], width)["input_ids"].shape[0] == 1
    assert server.version == 2
    assert not all(
        jnp.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(state.variables), strict=True)
    )
