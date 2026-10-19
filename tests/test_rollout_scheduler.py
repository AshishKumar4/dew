"""The rollout scheduler driving a real `Trainer`: a tiny model trains
through two-turn in-process environments on Dew's own server. A sandboxed
environment that runs past its wall clock on a first attempt is an
infrastructure failure the scheduler retries, and a verifier that returns a
non-finite reward stops the run instead of being retried. One run goes
through every way a rollout leaves its group, spare samples and early
admission included, and resumes from its checkpoint."""

import math
import threading
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from episode_support import environment as sandbox

from dew.checkpoints import Checkpoints
from dew.data.dataset import Loader, replayed
from dew.inference import NativeRolloutServer, TextGeneration
from dew.inference.serving import Server
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.rl import EnvironmentSource, EpisodeStatus, GRPOObjective, Observation
from dew.objectives.rl.scheduler import RolloutScheduler
from dew.objectives.rl.sessions import Status, pack
from dew.sampling import Sampling
from dew.training import Layout, Trainer

VOCAB = 13
STOP = 12
WIDTH = 48


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


@contextmanager
def tools(task, identity):
    yield Tools(int(task.id))


def fives(task, episode):
    sampled = [token for turn in episode.transitions for token in turn.action.tokens if token != STOP]
    return sum(token == 5 for token in sampled) / max(len(sampled), 1)


class Watched:
    """An `EnvironmentSource` that records every submission's key, task and rollouts, each rollout
    it finishes, and every rollout the scheduler cancels."""

    def __init__(self, source):
        self.source, self.keys, self.rollouts = source, [], []
        self.submitted, self.cancelled = [], set()

    def submit(self, task, samples, *, version, key):
        self.keys.append(tuple(int(word) for word in jax.random.key_data(key)))
        futures = self.source.submit(task, samples, version=version, key=key)
        self.submitted.append((task.id, futures))
        for future in futures:
            future.add_done_callback(self._finished)
        return futures

    def cancel(self, futures):
        self.cancelled.update(futures)
        self.source.cancel(futures)

    def _finished(self, done):
        if not done.cancelled() and done.exception() is None:
            self.rollouts.append(done.result())


@contextmanager
def rig(environment, verifier, *, tasks, log, **options):
    """A scheduler over `tasks` tasks a batch, two sessions a task, on Dew's own server: yields
    the objective, its initial variables, a fit of `steps` updates (`fit(steps, checkpoints=)`),
    the watched source and the server, and closes them after."""
    model = CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=1, num_heads=2,
                              head_dim=8, mlp_features=32, max_seq_len=64, dtype="float32")
    target = GRPOObjective(model, seq_len=WIDTH - 1, behavior_importance=2.0)
    params = target.init(jax.random.key(0))
    sampling = Sampling(temperature=1.0, eos_token_ids=STOP)
    server = NativeRolloutServer(Server.from_task(TextGeneration(model, params, None, sampling=sampling),
                                                  slots=16, capacity=64))
    episodes = EnvironmentSource(server, environment, verifier, max_prompt_tokens=32, max_new_tokens=16,
                                 max_turns=3, workers=16)
    watched = Watched(episodes)
    scheduler = RolloutScheduler(target, watched, server, width=WIDTH, rows=2 * tasks, groups=2,
                                 log=log, **options)
    stream = scheduler.tasks(
        Loader(
            train=replayed(lambda partition: (
                {"task_id": np.arange(tasks, dtype=np.int32) + step * tasks} for step in range(100)
            )),
            val=None,
            records=None,
            batch=tasks,
        )
    )

    def fit(steps, checkpoints=None):
        trainer = Trainer(target, optax.adam(1e-2), key=jax.random.key(3), rollout=scheduler,
                          layout=Layout(min_shard=1, tolerance=1.0), checkpoints=checkpoints)
        return trainer.fit(stream, steps=steps, log_every=steps,
                           checkpoint_every=None if checkpoints is None else 1)

    try:
        yield params, fit, watched, server
    finally:
        scheduler.close()
        episodes.close()
        server.close()


def train(environment, verifier, *, tasks, steps, **options):
    """Fit `steps` updates on `tasks` tasks a batch, two sessions a task; return the initial
    variables, the final state, the scheduler's records, the watched source and the served version."""
    records = []
    with rig(environment, verifier, tasks=tasks, log=records.append, **options) as (
            params, fit, watched, server):
        state = fit(steps)
    return params, state, records, watched, server.version


def test_a_trainer_run_trains_through_multi_turn_environments_on_the_native_server():
    # Every step must hold a completed session: a truncated one is masked and trains nothing,
    # and a step with no trainable token makes no update. This policy completes 32% of its
    # sessions, so four steps of 64 sessions all have one but for about 4 * 0.68**64 = 8e-11 of
    # runs; at one task a device (2 sessions a step on one GPU) most runs lost an update. The
    # count is a multiple of the device count so the rows split evenly.
    params, state, records, watched, version = train(
        tools, fives, tasks=math.lcm(32, jax.device_count()), steps=4, max_lag=2, ahead=1, sync_every=2)
    assert int(state.updates) == 4 and [record.updates for record in records] == [0, 1, 2, 3]
    # Every submission draws its own fold of its step's rollout key.
    assert len(set(watched.keys)) == len(watched.keys)
    assert all(0 <= record.lag <= 2 for record in records)
    assert any(record.lag > 0 for record in records), "no batch was ever drawn ahead of its update"
    # Each completed session's pair of calls merged into one chain.
    assert all(record.metrics["merge/calls_per_chain"] in (0.0, 2.0) for record in records)
    assert all(record.metrics["latency/max"] >= record.metrics["latency/p50"] > 0 for record in records)
    completed = [rollout for rollout in watched.rollouts if rollout.status == Status.COMPLETED]
    assert completed, "no session completed both tool turns"
    # Every completed session made two calls, and the second extends the first,
    # so the pair packs into one chain.
    for rollout in completed:
        assert len(rollout.calls) == 2
        assert pack([rollout], WIDTH)["input_ids"].shape[0] == 1
    assert version == 2
    assert not all(
        jnp.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(state.variables), strict=True)
    )


def test_an_environment_timeout_is_retried_and_its_group_still_trains():
    """Each sample's first attempt runs in a sandboxed worker that hangs at reset. The sandbox's
    own wall clock, not a scheduler clock, ends it; the scheduler resubmits each sample it failed,
    and every group trains."""
    attempted = set()
    hanging = sandbox("hang", wall_seconds=1)

    @contextmanager
    def flaky(task, identity):
        # A resubmission is a one-sample submit, so it is sample 0 of its task again.
        first = (task.id, identity.sample) not in attempted
        attempted.add((task.id, identity.sample))
        if first:
            with hanging(identity) as hung:
                yield hung
        else:
            yield Tools(int(task.id))

    tasks = math.lcm(2, jax.device_count())
    _, state, [record], watched, _ = train(flaky, fives, tasks=tasks, steps=1, max_lag=0, ahead=0)
    assert int(state.step) == 1
    assert record.groups == tasks and record.abandoned == 0
    assert record.resubmitted == {"infra_error": 2 * tasks}
    failed = [rollout for rollout in watched.rollouts if rollout.status == Status.INFRA_ERROR]
    assert len(failed) == 2 * tasks
    assert all("TimeoutError" in rollout.detail and not rollout.calls for rollout in failed)


def test_a_non_finite_reward_stops_the_run_instead_of_being_retried():
    """A verifier that breaks its contract is not an infrastructure failure: the
    run raises it rather than abandoning the group and training on the others."""
    def broken(task, episode):
        return float("nan") if task.id == "0" else fives(task, episode)

    with pytest.raises(ValueError):
        train(tools, broken, tasks=math.lcm(2, jax.device_count()), steps=1, max_lag=0, ahead=0)


class Admissions:
    """The scheduler's records, which straggling sessions wait on until their batch is admitted."""

    def __init__(self):
        self.records, self._logged, self._released = [], threading.Condition(), False

    def log(self, record):
        with self._logged:
            self.records.append(record)
            self._logged.notify_all()

    def wait(self, updates):
        with self._logged:
            self._logged.wait_for(lambda: self._released or any(
                record.updates >= updates for record in self.records))

    def release(self):
        with self._logged:
            self._released = True
            self._logged.notify_all()


def test_stragglers_failures_and_a_resume_leave_each_batch_its_first_complete_groups(tmp_path):
    """Of every five tasks, task 0's first attempts hang in a sandbox past its wall clock and are
    retried, task 1's container never starts and its group is abandoned at its second failure of
    one sample, tasks 2 and 3 run a spare sample beside their two, and task 4's sessions straggle
    until their batch has been admitted. The scheduler admits the first three complete groups of
    each batch, 0, 2 and 3, and cancels the rest. A second fit resumes from the first's checkpoint:
    reopening the stream cancels the batch the first fit drew ahead, and the batch trains again
    from fresh submissions."""
    tasks = math.lcm(5, jax.device_count())
    hanging = sandbox("hang", wall_seconds=1)
    admissions, attempted = Admissions(), set()

    @contextmanager
    def faulty(task, identity):
        role = int(task.id) % 5
        # A resubmission is a one-sample submit, so it is sample 0 of its task again.
        first = (task.id, identity.sample) not in attempted
        attempted.add((task.id, identity.sample))
        if role == 0 and first:
            with hanging(identity) as hung:
                yield hung
            return
        if role == 1:
            raise OSError("the container runtime started no container")
        if role == 4:
            admissions.wait(int(task.id) // tasks)
        yield Tools(int(task.id))

    options = {"oversample": 1, "admit": 3 * tasks // 5, "max_attempts": 2, "max_lag": 1, "ahead": 1}
    checkpoints = str(tmp_path / "run")
    with rig(faulty, fives, tasks=tasks, log=admissions.log, **options) as (_, fit, watched, _):
        # The stragglers of batches drawn ahead are released before the source closes, which waits
        # for its running sessions.
        try:
            first = fit(1, Checkpoints(checkpoints))
            drawn_ahead = [future for task, futures in watched.submitted
                           if int(task) >= tasks for future in futures]
            resumed = fit(2, Checkpoints(checkpoints))
        finally:
            admissions.release()
    assert int(first.updates) == 1 and int(resumed.updates) == 2
    batches = tasks // 5
    for record in admissions.records:
        assert record.groups == 3 * batches and record.abandoned == batches and record.cut == 0
        # Two spares and three stragglers a group of five at least; an abandoned group's
        # samples in flight when it is given up are cancelled too.
        assert record.cancelled >= 5 * batches
        assert set(record.resubmitted) <= {"infra_error"}
    [before, after] = admissions.records
    assert (before.updates, after.updates) == (0, 1)
    # Task 0 resubmits two of its three timed-out samples, a spare stands in for the first;
    # task 1 one or two before its second failure of one sample abandons it.
    assert 3 * batches <= before.resubmitted["infra_error"] <= 4 * batches
    # The resumed batch's task 0 ran its first attempts before the resume, so its fresh
    # submissions run; only task 1 fails.
    assert batches <= after.resubmitted["infra_error"] <= 2 * batches
    # Reopening the stream cancelled every rollout the first fit drew ahead.
    assert drawn_ahead and set(drawn_ahead) <= watched.cancelled
    rollouts = {}
    for rollout in watched.rollouts:
        rollouts.setdefault(int(rollout.task) % 5, []).append(rollout)
    timed_out = [rollout for rollout in rollouts[0] if rollout.status == Status.INFRA_ERROR]
    assert len(timed_out) >= 3 * batches and all("TimeoutError" in rollout.detail for rollout in timed_out)
    assert all(rollout.status == Status.INFRA_ERROR and not rollout.calls for rollout in rollouts[1])
    # A straggler's session sees its cancellation at its first draw, after its batch was admitted.
    assert all(rollout.status == Status.CANCELLED and not rollout.calls for rollout in rollouts[4])
