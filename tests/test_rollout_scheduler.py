"""The rollout scheduler driving a real `Trainer`: a tiny model trains
through two-turn in-process environments on Dew's own server. A sandboxed
environment that runs past its wall clock on a first attempt is an
infrastructure failure the scheduler retries, and a verifier that returns a
non-finite reward stops the run instead of being retried. One run goes
through every way a rollout leaves its group, spare samples and early
admission included, and resumes from its checkpoint."""

import dataclasses
import functools
import math
import os
import threading
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from episode_support import environment as sandbox

from dew.checkpoints import Checkpoints
from dew.data.dataset import Loader, replayed, tapped
from dew.inference import NativeRolloutServer, TextGeneration
from dew.inference.serving import Server
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.rl import EnvironmentSource, EpisodeStatus, GRPOObjective, Observation
from dew.objectives.rl.scheduler import Asynchronous, RolloutScheduler
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
        self.context = action.prompt + action.tokens + (4,)
        return Observation(self.context)


@contextmanager
def tools(task, identity):
    yield Tools(int(task.id))


def fives(task, episode):
    sampled = [token for turn in episode.transitions for token in turn.action.tokens if token != STOP]
    return sum(token == 5 for token in sampled) / max(len(sampled), 1)


class Watched:
    """An `EnvironmentSource` that records every submission's key, and per submission (numbered as
    the source numbers them, `EpisodeId.task`) its task and the scenario round it was made in, each
    rollout it finishes, and which samples the scheduler cancelled. `changed` is notified on each."""

    def __init__(self, source):
        self.source, self.keys, self.rollouts = source, [], []
        self.submitted, self.round, self.released = [], 0, False
        self.results, self.cancelled = {}, set()
        self.changed = threading.Condition()
        self._samples = {}

    def submit(self, task, samples, *, version, key):
        self.keys.append(tuple(int(word) for word in jax.random.key_data(key)))
        with self.changed:
            serial = len(self.submitted)
            futures = self.source.submit(task, samples, version=version, key=key)
            self.submitted.append((task.id, self.round, futures))
            self._samples.update({future: (serial, sample) for sample, future in enumerate(futures)})
            self.changed.notify_all()
        for future in futures:
            future.add_done_callback(self._finished)
        return futures

    def cancel(self, futures):
        # The source marks them cancelled first, so a session woken by the notice sees its cancel.
        self.source.cancel(futures)
        with self.changed:
            self.cancelled.update(self._samples[future] for future in futures)
            self.changed.notify_all()

    def _finished(self, done):
        if not done.cancelled() and done.exception() is None:
            with self.changed:
                self.rollouts.append(done.result())
                self.results[self._samples[done]] = done.result()
                self.changed.notify_all()

    def wait(self, reached):
        """Wait until `reached()`, a step of a scenario, holds, or the rig is closing."""
        with self.changed:
            self.changed.wait_for(lambda: self.released or reached())

    def release(self):
        """Let every wait end: the rig is closing, after its scheduler cancelled what it holds."""
        with self.changed:
            self.released = True
            self.changed.notify_all()


@contextmanager
def rig(environment, verifier, *, tasks, log, workers=16, scheduler=RolloutScheduler,
        registered=lambda batch: None, **options):
    """A `scheduler` over `tasks` tasks a batch, two sessions a task, on Dew's own server, its
    sessions on `workers` threads; `registered(batch)` sees each batch once the scheduler has
    registered it. Yields the initial variables, a fit of `steps` updates (`fit(steps,
    checkpoints=)`), the scheduler, the watched source and the server, and closes them after: the
    scheduler cancels what it holds, every wait of the watched source ends, and the source waits
    for its sessions."""
    model = CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=1, num_heads=2,
                              head_dim=8, mlp_features=32, max_seq_len=64, dtype="float32")
    target = GRPOObjective(model, seq_len=WIDTH - 1, behavior_importance=2.0)
    params = target.init(jax.random.key(0))
    sampling = Sampling(temperature=1.0, eos_token_ids=STOP)
    server = NativeRolloutServer(Server.from_task(TextGeneration(model, params, None, sampling=sampling),
                                                  slots=16, capacity=64))
    episodes = EnvironmentSource(server, environment, verifier, max_prompt_tokens=32, max_new_tokens=16,
                                 max_turns=3, workers=workers)
    watched = Watched(episodes)
    acquired = {name: options.pop(name) for name in ("ahead", "max_lag", "sync_every", "max_attempts")
                if name in options}
    rollout = scheduler(Asynchronous(watched, server, target, width=WIDTH, rows=2 * tasks, **acquired),
                        groups=2, log=log, **options)
    stream = rollout.tasks(
        Loader(
            train=replayed(lambda partition: (
                {"task_id": np.arange(tasks, dtype=np.int32) + step * tasks} for step in range(100)
            )),
            val=None,
            records=None,
            batch=tasks,
        )
    )
    stream = dataclasses.replace(stream, train=tapped(stream.train, registered))

    def fit(steps, checkpoints=None):
        trainer = Trainer(target, optax.adam(1e-2), key=jax.random.key(3), rollout=rollout,
                          layout=Layout(min_shard=1, tolerance=1.0), checkpoints=checkpoints)
        return trainer.fit(stream, steps=steps, log_every=steps,
                           checkpoint_every=None if checkpoints is None else 1)

    try:
        yield params, fit, rollout, watched, server
    finally:
        rollout.close()
        watched.release()
        episodes.close()
        server.close()


def train(environment, verifier, *, tasks, steps, **options):
    """Fit `steps` updates on `tasks` tasks a batch, two sessions a task; return the initial
    variables, the final state, the scheduler's records, the watched source and the served version."""
    records = []
    with rig(environment, verifier, tasks=tasks, log=records.append, **options) as (
            params, fit, _, watched, server):
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


class Recording(RolloutScheduler):
    """The scheduler, keeping each call's state, task batch and the batch it returns to the step.
    Its first call runs once `before_first()` returns."""

    def __init__(self, *arguments, before_first, **options):
        super().__init__(*arguments, **options)
        self.before_first, self.calls = before_first, []

    def __call__(self, state, batch, key):
        if not self.calls:
            self.before_first()
        packed = super().__call__(state, batch, key)
        self.calls.append((leaves(state), np.asarray(batch["task_id"]), packed))
        return packed


def leaves(state):
    """Every leaf of a training state on the host, a key as its words."""
    return [np.asarray(jax.random.key_data(leaf) if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key) else leaf)
            for leaf in jax.tree.leaves(state)]


ESCAPE_SECONDS = 300
"""A straggler's sandbox's wall clock: it runs until the scheduler cancels it, and this ends it
where a scheduler never does, failing the scenario rather than hanging it."""


@contextmanager
def straggling(directory, watched, identity, sample):
    """A sandbox whose reset reads a pipe under `directory` that is written once the scheduler
    cancels `sample` (or the rig closes)."""
    gate = directory / f"gate-{identity.task}-{identity.sample}"
    os.mkfifo(gate)
    # Held open for reading and writing, so the pipe keeps a release written before the sandbox
    # opens it.
    held = os.open(gate, os.O_RDWR)
    ended = threading.Event()

    def release():
        watched.wait(lambda: sample in watched.cancelled or ended.is_set())
        if not ended.is_set():
            os.write(held, b"\n")

    releaser = threading.Thread(target=release, daemon=True)
    try:
        releaser.start()
        with sandbox("wait", os.devnull, str(gate), wall_seconds=ESCAPE_SECONDS)(identity) as waiting:
            yield waiting
    finally:
        # The releaser is done with the pipe before it closes: the sandbox may have ended at its
        # wall clock, or failed to start, before the scheduler cancelled it.
        with watched.changed:
            ended.set()
            watched.changed.notify_all()
        if releaser.ident is not None:
            releaser.join()
        os.close(held)


def assert_tallies(records, batches):
    """Each batch admits groups 0, 2 and 3 of every five and abandons group 1. Cancelled: group 1's
    one live sample at its abandonment, the spares of 2 and 3, the three stragglers, and in the
    resumed batch group 0's spare too (its first attempts ran before the resume). Group 0 resubmits
    two of its three timed-out samples, a spare standing in for the first, and group 1 one or two
    before a second failure of one sample gives it up, as its first retry fails before or after
    its last first attempt is read."""
    [before, after] = records
    assert (before.updates, before.version, before.lag) == (0, 0, 0)
    assert (after.updates, after.version, after.lag) == (1, 1, 0)
    for record, cancelled in ((before, 6 * batches), (after, 7 * batches)):
        assert (record.groups, record.abandoned, record.cut) == (3 * batches, batches, 0)
        assert record.cancelled == cancelled and set(record.resubmitted) == {"infra_error"}
    assert 3 * batches <= before.resubmitted["infra_error"] <= 4 * batches
    assert batches <= after.resubmitted["infra_error"] <= 2 * batches


def test_stragglers_failures_and_a_resume_leave_each_batch_its_first_complete_groups(tmp_path):
    """Of every five tasks, task 0's first attempts hang in a sandbox past its wall clock and are
    retried, task 1's container never starts and its group is abandoned at its second failure of one
    sample, tasks 2 and 3 run a spare sample beside their two, and task 4's sessions straggle in a
    sandbox until the scheduler cancels them. The scheduler admits the first three complete groups
    of each batch, 0, 2 and 3, and cancels the rest. The winners start once their batch's failing
    groups are given up, so every tally below is the scheduler's alone. A second fit resumes from
    the first's checkpoint: reopening the stream cancels the batch the first fit drew ahead, and the
    batch trains again from fresh submissions under the restored state."""
    tasks = math.lcm(5, jax.device_count())
    batches = tasks // 5
    hanging = sandbox("hang", wall_seconds=1)
    attempted, firsts, stragglers, registered, records = set(), {}, set(), set(), []
    progress = threading.Condition()

    def seen(batch):
        with progress:
            registered.add(int(np.asarray(batch["task_id"])[0]) // tasks)
            progress.notify_all()

    def drawn_ahead():
        # The first call starts once the stream has registered the batch after it, so it draws
        # that batch ahead of its update.
        with progress:
            progress.wait_for(lambda: 1 in registered)

    def given_up(round, batch):
        """Whether every failing group of `batch` in `round` was given up: one of its samples
        was cancelled while its batch was being admitted."""
        failing = {serial for serial, (task, made, _) in enumerate(watched.submitted)
                   if made == round and int(task) % 5 == 1 and int(task) // tasks == batch}
        given = {watched.submitted[serial][0] for serial, _ in watched.cancelled if serial in failing}
        return len(given) == batches

    @contextmanager
    def faulty(task, identity):
        index, sample = int(task.id), (identity.task, identity.sample)
        role, batch = index % 5, index // tasks
        # The submission holds the lock until it is recorded, so its sessions read it whole.
        with watched.changed:
            _, round, _ = watched.submitted[identity.task]
        # A resubmission is a one-sample submit, so it is sample 0 of its task again.
        first = (task.id, identity.sample) not in attempted
        attempted.add((task.id, identity.sample))
        if role == 0 and first:
            with watched.changed:
                firsts[sample] = batch
                watched.changed.notify_all()
            with hanging(identity) as hung:
                yield hung
            return
        if role == 1:
            raise OSError("the container runtime started no container")
        if role == 4:
            with watched.changed:
                stragglers.add(sample)
            with straggling(tmp_path, watched, identity, sample) as waiting:
                yield waiting
            return
        if role in (2, 3):
            watched.wait(lambda: given_up(round, batch) or sample in watched.cancelled)
        yield Tools(index)

    options = {"oversample": 1, "admit": 3 * tasks // 5, "max_attempts": 2, "max_lag": 1, "ahead": 1}
    # A straggler holds its worker until it is cancelled, and up to three batches' sessions run at
    # once (the first fit's batch drawn ahead, and the resumed fit's two).
    checkpoints = str(tmp_path / "run")
    recording = functools.partial(Recording, before_first=drawn_ahead)
    with rig(faulty, fives, tasks=tasks, log=records.append, workers=9 * tasks, scheduler=recording,
             registered=seen, **options) as (_, fit, recording, watched, _):
        first = fit(1, Checkpoints(checkpoints))
        # The batch the first fit drew ahead has started its first attempts before the resume.
        watched.wait(lambda: sum(batch == 1 for batch in firsts.values()) == 3 * batches)
        ahead_samples = [(serial, sample) for serial, (task, _, futures) in enumerate(watched.submitted)
                         if int(task) // tasks == 1 for sample in range(len(futures))]
        watched.round = 1
        resumed = fit(2, Checkpoints(checkpoints))
        resubmitted = [int(task) for task, made, _ in watched.submitted if made == 1]
    assert int(first.updates) == 1 and int(resumed.updates) == 2
    [(_, trained, packed), (restored, again, repacked)] = recording.calls
    # The resumed call starts from the first fit's whole state, key and counters included, and
    # trains the second batch.
    for got, want in zip(restored, leaves(first), strict=True):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(trained, np.arange(tasks))
    np.testing.assert_array_equal(again, np.arange(tasks, 2 * tasks))
    # Every id a batch trains was drawn under the version its call published: the resumed batch
    # from fresh submissions alone. (A truncated session trains nothing, so its ids carry none.)
    for batch, version in ((packed, 0), (repacked, 1)):
        assert set(np.unique(batch["versions"][batch["versions"] >= 0])) == {version}
    assert_tallies(records, batches)
    # Reopening the stream cancelled every sample the first fit drew ahead; the resumed fit
    # submitted only its own batches, the second batch again first.
    assert ahead_samples and set(ahead_samples) <= watched.cancelled
    assert min(resubmitted) == tasks and all(task >= tasks for task in resubmitted)
    # A first attempt of group 0 ran into its sandbox's wall clock: infra, with no call. Every
    # straggler was cancelled by the scheduler and ended at its next draw, with no call.
    for sample in firsts:
        assert watched.results[sample].status is Status.INFRA_ERROR and not watched.results[sample].calls
    assert stragglers <= watched.cancelled
    for sample in stragglers:
        assert watched.results[sample].status is Status.CANCELLED and not watched.results[sample].calls
    assert all(rollout.status is Status.INFRA_ERROR and not rollout.calls
               for rollout in watched.rollouts if int(rollout.task) % 5 == 1)
