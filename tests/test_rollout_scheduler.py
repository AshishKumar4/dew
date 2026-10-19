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
from dew.objectives.rl.scheduler import RolloutScheduler, task_ids
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
    """An `EnvironmentSource` that records every submission's key, and per submission (numbered as
    the source numbers them, `EpisodeId.task`) its task and the scenario round it was made in, each
    rollout it finishes, and which samples the scheduler cancelled. `changed` is notified on each."""

    def __init__(self, source):
        self.source, self.keys, self.rollouts = source, [], []
        self.submitted, self.round = [], 0
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
        with self.changed:
            self.cancelled.update(self._samples[future] for future in futures)
            self.changed.notify_all()
        self.source.cancel(futures)

    def _finished(self, done):
        if not done.cancelled() and done.exception() is None:
            with self.changed:
                self.rollouts.append(done.result())
                self.results[self._samples[done]] = done.result()
                self.changed.notify_all()

    def wait(self, reached, what):
        """Wait until `reached()`, a step of a scenario, holds; fail naming `what` if the run does
        not reach it within two minutes, rather than hang."""
        with self.changed:
            assert self.changed.wait_for(reached, timeout=120), f"the run never reached {what}"


@contextmanager
def rig(environment, verifier, *, tasks, log, workers=16, weights=lambda server, watched: server,
        read=task_ids, **options):
    """A scheduler over `tasks` tasks a batch, two sessions a task, on Dew's own server, its
    sessions on `workers` threads, its weights published through `weights(server, watched)` and its
    batches read into tasks by `read`: yields the initial variables, a fit of `steps` updates
    (`fit(steps, checkpoints=)`), the state a trainer restores from `checkpoints`
    (`restored(checkpoints)`), the watched source and the server, and closes them after."""
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
    scheduler = RolloutScheduler(target, watched, weights(server, watched), width=WIDTH, rows=2 * tasks,
                                 tasks=read, groups=2, log=log, **options)
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

    def trainer(checkpoints):
        return Trainer(target, optax.adam(1e-2), key=jax.random.key(3), rollout=scheduler,
                       layout=Layout(min_shard=1, tolerance=1.0), checkpoints=checkpoints)

    def fit(steps, checkpoints=None):
        return trainer(checkpoints).fit(stream, steps=steps, log_every=steps,
                                        checkpoint_every=None if checkpoints is None else 1)

    def restored(checkpoints):
        return trainer(checkpoints).place()[0]

    try:
        yield params, fit, restored, watched, server
    finally:
        scheduler.close()
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


class Ordered:
    """The served weights, whose version the scheduler reads first in each call: its first call waits
    until the stream has registered the batch after the one it trains, so it draws that batch ahead."""

    def __init__(self, server, watched, registered):
        self.server, self.watched, self.registered, self.waited = server, watched, registered, False

    @property
    def version(self):
        if not self.waited:
            self.watched.wait(lambda: 1 in self.registered, "the second batch registered")
            self.waited = True
        return self.server.version

    def load(self, variables, version):
        self.server.load(variables, version)


def test_stragglers_failures_and_a_resume_leave_each_batch_its_first_complete_groups(tmp_path):
    """Of every five tasks, task 0's first attempts hang in a sandbox past its wall clock and are
    retried, task 1's container never starts and its group is abandoned at its second failure of one
    sample, tasks 2 and 3 run a spare sample beside their two, and task 4's sessions straggle. The
    scheduler admits the first three complete groups of each batch, 0, 2 and 3, and cancels the
    rest. The winners start once their batch's failing groups are given up, and a straggler runs
    until the scheduler cancels it, so every tally below is the scheduler's alone. A second fit
    resumes from the first's checkpoint: reopening the stream cancels the batch the first fit drew
    ahead, and the batch trains again from fresh submissions under the restored weights."""
    tasks = math.lcm(5, jax.device_count())
    batches = tasks // 5
    hanging = sandbox("hang", wall_seconds=1)
    attempted, firsts, registered, records = set(), {}, set(), []

    def numbered(batch):
        found = task_ids(batch)
        with watched.changed:
            registered.add(int(found[0].id) // tasks)
            watched.changed.notify_all()
        return found

    def given_up(round, batch):
        """Whether every failing group of `batch` in `round` was given up: one of its samples
        was cancelled before its batch was admitted."""
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
        if role in (2, 3):
            watched.wait(lambda: given_up(round, batch) or sample in watched.cancelled,
                         f"batch {batch}'s failing groups given up")
        if role == 4:
            watched.wait(lambda: sample in watched.cancelled, f"straggler {sample} cancelled")
        yield Tools(index)

    options = {"oversample": 1, "admit": 3 * tasks // 5, "max_attempts": 2, "max_lag": 1, "ahead": 1,
               "read": numbered}
    # A straggler holds its worker until it is cancelled, and up to three batches' sessions run at
    # once (the first fit's batch drawn ahead, and the resumed fit's two).
    checkpoints = str(tmp_path / "run")
    with rig(faulty, fives, tasks=tasks, log=records.append, workers=9 * tasks,
             weights=lambda server, watched: Ordered(server, watched, registered), **options) as (
            _, fit, restored, watched, _):
        first = fit(1, Checkpoints(checkpoints))
        # The batch the first fit drew ahead has started its first attempts before the resume.
        watched.wait(lambda: sum(batch == 1 for batch in firsts.values()) == 3 * batches,
                     "the second batch's first attempts started")
        ahead = [(serial, sample) for serial, (task, _, futures) in enumerate(watched.submitted)
                 if int(task) // tasks == 1 for sample in range(len(futures))]
        again = restored(Checkpoints(checkpoints))
        watched.round = 1
        resumed = fit(2, Checkpoints(checkpoints))
        resubmitted = [int(task) for task, made, _ in watched.submitted if made == 1]
    assert int(first.updates) == 1 and int(resumed.updates) == 2
    for got, want in zip(jax.tree.leaves((again.variables, again.opt_state)),
                         jax.tree.leaves((first.variables, first.opt_state)), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    [before, after] = records
    # Each batch admits groups 0, 2 and 3 of every five and abandons group 1. Cancelled: group 1's
    # one live sample at its abandonment, the spares of 2 and 3, the three stragglers, and in the
    # resumed batch group 0's spare too (its first attempts ran before the resume).
    assert (before.updates, before.version, before.lag) == (0, 0, 0)
    assert (after.updates, after.version, after.lag) == (1, 1, 0)
    for record, cancelled in ((before, 6 * batches), (after, 7 * batches)):
        assert (record.groups, record.abandoned, record.cut) == (3 * batches, batches, 0)
        assert record.cancelled == cancelled and set(record.resubmitted) == {"infra_error"}
    # Group 0 resubmits two of its three timed-out samples, a spare standing in for the first, and
    # group 1 one or two before a second failure of one sample gives it up, as its first retry fails
    # before or after its last first attempt is read.
    assert 3 * batches <= before.resubmitted["infra_error"] <= 4 * batches
    assert batches <= after.resubmitted["infra_error"] <= 2 * batches
    # Reopening the stream cancelled every sample the first fit drew ahead; the resumed fit
    # submitted only its own batches, the second batch again first.
    assert ahead and set(ahead) <= watched.cancelled
    assert min(resubmitted) == tasks and all(task >= tasks for task in resubmitted)
    # A first attempt of group 0 hung past the sandbox's wall clock: infra, with no call.
    assert all(watched.results[sample].status is Status.INFRA_ERROR and not watched.results[sample].calls
               for sample in firsts if sample in watched.results)
    roles = {}
    for rollout in watched.rollouts:
        roles.setdefault(int(rollout.task) % 5, []).append(rollout)
    assert all(rollout.status is Status.INFRA_ERROR and not rollout.calls for rollout in roles[1])
    # A straggler sees its cancellation at its first draw.
    assert all(rollout.status is Status.CANCELLED and not rollout.calls for rollout in roles[4])
