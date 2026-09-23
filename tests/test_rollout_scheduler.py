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

import threading
from concurrent.futures import Future
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.data import Dataset
from dew.inference import NativeRolloutServer, TextGeneration
from dew.inference.serving import Server
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.rl import (
    EnvironmentSource,
    EpisodeStatus,
    GRPOObjective,
    Observation,
    Score,
    sessions as sessions_module,
)
from dew.objectives.rl.scheduler import RolloutScheduler
from dew.objectives.rl.sessions import Call, Session, Status, pack
from dew.sampling import Sampling
from dew.training import Layout, Trainer

EOS = 9
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


class State:
    def __init__(self, updates):
        self.params, self.updates = jnp.zeros(()), updates


def finished(reward=1.0, *versions, status=Status.COMPLETED, components=None):
    """A rollout of one call per version, each call extending the last."""
    calls, prompt = [], (1, 2)
    for version in versions or (0,):
        calls.append(Call(prompt, (3, EOS), (-.5, -.25), "stop", version))
        prompt = (*prompt, 3, EOS, 4)
    scored = status.trainable or status == Status.TRUNCATED
    return Session("source-task", "source-group", 7, 7, tuple(calls), status,
                   reward if scored else None, components or {}, "")


class Scripted:
    """A source whose `outcome(task, submission, sample, version)` returns a rollout or None to keep it running."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.submitted, self.cancelled, self.running = [], [], []

    def submit(self, task, samples, *, version):
        submission = len(self.submitted)
        self.submitted.append((task.id, samples, version))
        futures = []
        for sample in range(samples):
            future = Future()
            value = self.outcome(task.id, submission, sample, version)
            if value is None:
                self.running.append(future)
            elif isinstance(value, BaseException):
                future.set_exception(value)
            else:
                future.set_result(value)
            futures.append(future)
        return futures

    def cancel(self, futures):
        for future in futures:
            self.cancelled.append(future)
            future.cancel()


def batches(*rows):
    for tasks in rows:
        yield {"task_id": np.asarray(tasks, np.int32)}


def scheduler(source, publisher=None, stream=((1, 2), (3, 4), (5, 6)), **options):
    records = []
    options = {"groups": 2, "max_lag": 1, "ahead": 1, "width": WIDTH, "rows": ROWS, **options}
    rollout = RolloutScheduler(Objective(), source, publisher or Publisher(), log=records.append, **options)
    data = rollout.tasks(Dataset(train=lambda: batches(*stream), val=None, records=None, batch=2))
    return rollout, data, records


def trained(batch):
    """The rollout indices that carry loss mass, and their summed weights."""
    mask = batch["response_mask"] > 0
    return sorted(set(batch["session_index"][mask].tolist())), float(batch["session_weights"].sum())


def test_complete_groups_are_packed_and_the_next_batch_is_submitted_ahead():
    source = Scripted(lambda task, submission, sample, version: finished(float(sample), version))
    rollout, data, records = scheduler(source, estimator="mean")
    stream = iter(data.train())
    first, _ = next(stream), next(stream)
    batch = rollout(State(0), first, None)
    # Both tasks of batch 0 and of batch 1, two samples each, under version 0.
    assert source.submitted == [("1", 2, 0), ("2", 2, 0), ("3", 2, 0), ("4", 2, 0)]
    indices, weight = trained(batch)
    assert indices == [0, 1, 2, 3] and weight == pytest.approx(4.0)
    # Sample rewards 0 and 1 in each group, centred per group.
    by_rollout = {int(i): float(a) for i, a in zip(batch["session_index"][batch["response_mask"] > 0],
                                                   batch["advantages"][batch["response_mask"] > 0], strict=True)}
    assert by_rollout == pytest.approx({0: -.5, 1: .5, 2: -.5, 3: .5})
    np.testing.assert_allclose(batch["old_log_probs"], -0.75 * batch["response_mask"])
    record = records[-1]
    assert (record.groups, record.lag) == (2, 0)
    assert record.metrics["reward/mean"] == 0.5 and record.metrics["status/completed"] == 1.0


def test_an_infra_failure_is_resubmitted_into_its_own_group_and_never_trained():
    def outcome(task, submission, sample, version):
        if task == "1" and submission == 0 and sample == 0:
            return finished(0.0, version, status=Status.INFRA_ERROR)
        return finished(float(sample), version)

    source = Scripted(outcome)
    rollout, data, records = scheduler(source, ahead=0)
    batch = rollout(State(0), next(iter(data.train())), None)
    # The failed sample went back once, alone, under the served version.
    assert source.submitted == [("1", 2, 0), ("2", 2, 0), ("1", 1, 0)]
    assert records[-1].resubmitted == {"infra_error": 1}
    assert records[-1].metrics["status/completed"] == 1.0
    indices, _ = trained(batch)
    assert indices == [0, 1, 2, 3]
    # The retry joined task 1's group as sample 0, attempt 1: its reward 0
    # against sibling 1 is a group of two, so no member's advantage is zero.
    assert records[-1].groups == 2
    assert np.all(batch["advantages"][batch["response_mask"] > 0] != 0)


def test_a_sample_that_keeps_failing_abandons_its_group_and_a_batch_with_none_left_raises():
    source = Scripted(lambda task, submission, sample, version: finished(
        0.0, version, status=Status.INFRA_ERROR) if task == "1" else finished(float(sample), version))
    rollout, data, records = scheduler(source, ahead=0, max_attempts=2)
    stream = iter(data.train())
    batch = rollout(State(0), next(stream), None)
    assert records[-1].abandoned == 1 and records[-1].groups == 1
    assert trained(batch)[0] == [0, 1]
    assert [entry for entry in source.submitted if entry[0] == "1"] == [("1", 2, 0), ("1", 1, 0), ("1", 1, 0)]

    dead = Scripted(lambda task, submission, sample, version: finished(0.0, version, status=Status.CANCELLED))
    rollout, data, _ = scheduler(dead, ahead=0, max_attempts=1)
    with pytest.raises(RuntimeError, match="no group"):
        rollout(State(0), next(iter(data.train())), None)


def test_a_truncated_member_completes_its_group_but_carries_no_loss():
    source = Scripted(lambda task, submission, sample, version: finished(
        5.0, version, status=Status.TRUNCATED) if sample == 1 else finished(1.0, version))
    rollout, data, records = scheduler(source, ahead=0, groups=3)
    batch = rollout(State(0), next(iter(data.train())), None)
    metrics = records[-1].metrics
    assert metrics["status/completed"] == pytest.approx(4 / 6) and metrics["status/truncated"] == pytest.approx(2 / 6)
    assert source.submitted == [("1", 3, 0), ("2", 3, 0)]
    indices, _ = trained(batch)
    assert indices == [0, 2, 3, 5]
    # The truncation's reward 5 enters no baseline: the two scored members tie.
    np.testing.assert_array_equal(batch["advantages"], 0)


def test_a_scored_truncation_trains_on_its_reward_when_the_scheduler_says_so():
    source = Scripted(lambda task, submission, sample, version: finished(
        5.0, version, status=Status.TRUNCATED) if sample == 1 else finished(1.0, version))
    rollout, data, records = scheduler(source, ahead=0, groups=2, truncation="score", estimator="mean")
    batch = rollout(State(0), next(iter(data.train())), None)
    assert trained(batch)[0] == [0, 1, 2, 3]
    assert records[-1].metrics["reward/mean"] == 3.0 and "masked/truncated" not in records[-1].metrics


def split(reward, version):
    """Two calls whose second prompt rewrites the first id, so each packs as its own chain."""
    calls = (Call((1, 2), (3, EOS), (-.5, -.25), "stop", version),
             Call((6, 2, 3, EOS, 4), (3, EOS), (-.5, -.25), "stop", version))
    return Session("source-task", "source-group", 7, 7, calls, Status.COMPLETED, reward, {}, "")


def test_a_group_whose_chains_overflow_the_rows_is_cut_instead_of_failing_the_step():
    # A group of two split sessions needs three 8-id rows; two groups need six, above rows=4.
    source = Scripted(lambda task, submission, sample, version: split(float(sample), version))
    rollout, data, records = scheduler(source, ahead=0, rows=4)
    batch = rollout(State(0), next(iter(data.train())), None)
    assert batch["input_ids"].shape == (4, WIDTH)
    assert (records[-1].groups, records[-1].cut) == (1, 1)
    assert trained(batch)[0] == [0, 1]

    rollout, data, _ = scheduler(source, ahead=0, rows=2)
    with pytest.raises(ValueError, match="rows"):
        rollout(State(0), next(iter(data.train())), None)


def test_a_group_that_cannot_fit_the_rows_alone_is_refused_whichever_group_completes_first():
    # Task 1's group needs one row and completes first; task 2's needs three alone.
    source = Scripted(lambda task, submission, sample, version: split(float(sample), version) if task == "2"
                      else finished(float(sample), version))
    rollout, data, _ = scheduler(source, ahead=0, rows=2)
    with pytest.raises(ValueError, match="group of task 2 needs 3 rows"):
        rollout(State(0), next(iter(data.train())), None)


def test_an_unscored_truncation_under_score_is_retried_rather_than_failing_the_step():
    # A harness that hit the context limit before its verifier ran reports a
    # truncation with no reward; `score` has nothing to train it on.
    def outcome(task, submission, sample, version):
        if (task, submission, sample) == ("1", 0, 0):
            return Session("t", "g", 0, 0, finished(0.0, version).calls, Status.TRUNCATED, None)
        return finished(float(sample), version, status=Status.TRUNCATED)

    source = Scripted(outcome)
    rollout, data, records = scheduler(source, ahead=0, truncation="score")
    batch = rollout(State(0), next(iter(data.train())), None)
    assert records[-1].resubmitted == {"unscored": 1} and records[-1].groups == 2
    assert trained(batch)[0] == [0, 1, 2, 3]


def test_admission_builds_each_sessions_chains_once_however_many_groups_complete(monkeypatch):
    # Fitting every completing group against those before it must not rebuild
    # their chains: once when a group completes, once more in pack.
    built = []
    original = sessions_module._chains
    monkeypatch.setattr(sessions_module, "_chains", lambda *args: built.append(args) or original(*args))
    tasks = tuple(range(16))
    source = Scripted(lambda task, submission, sample, version: finished(float(sample), version))
    rollout, data, records = scheduler(source, ahead=0, rows=32, stream=(tasks,))
    rollout(State(0), next(iter(data.train())), None)
    assert records[-1].groups == 16 and len(built) == 2 * 16 * 2


def test_oversampled_stragglers_are_cancelled_once_the_group_is_full():
    source = Scripted(lambda task, submission, sample, version: None if sample == 0 else finished(
        float(sample), version))
    rollout, data, records = scheduler(source, ahead=0, oversample=1)
    rollout(State(0), next(iter(data.train())), None)
    assert source.submitted == [("1", 3, 0), ("2", 3, 0)]
    assert len(source.cancelled) == 2 and all(future in source.running for future in source.cancelled)
    assert records[-1].groups == 2 and records[-1].cancelled == 2


def test_a_spare_that_fails_after_its_group_filled_neither_abandons_nor_retries_it():
    source = Scripted(lambda task, submission, sample, version: finished(
        0.0, version, status=Status.INFRA_ERROR) if sample == 2 else finished(float(sample), version))
    rollout, data, records = scheduler(source, ahead=0, oversample=1, max_attempts=1)
    batch = rollout(State(0), next(iter(data.train())), None)
    assert (records[-1].groups, records[-1].abandoned, records[-1].resubmitted) == (2, 0, {})
    assert trained(batch)[0] == [0, 1, 2, 3]
    assert source.submitted == [("1", 3, 0), ("2", 3, 0)]


def test_a_straggler_is_waited_for_and_admitted_when_it_finishes():
    source = Scripted(lambda task, submission, sample, version: None if (task, sample) == ("2", 1) else finished(
        float(sample), version))
    rollout, data, records = scheduler(source, ahead=0)
    stream = iter(data.train())
    late = threading.Timer(0.05, lambda: source.running[0].set_result(finished(3.0, 0)))
    late.start()
    batch = rollout(State(0), next(stream), None)
    late.join()
    assert trained(batch)[0] == [0, 1, 2, 3] and records[-1].waited >= 0.04
    assert not source.cancelled


def bounded(rollout, state, batch, source, seconds=20.0):
    """`rollout(state, batch)` on a thread, failed rather than hung when it outlives `seconds`.

    A deadline regression leaves the scheduler waiting on futures that never
    finish; failing them afterwards releases the thread.
    """
    outcome = {}

    def run():
        try:
            outcome["batch"] = rollout(state, batch, None)
        except BaseException as failure:
            outcome["error"] = failure

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        for future in source.running:
            if not future.done():
                future.set_exception(TimeoutError("released by the test"))
        thread.join(seconds)
        pytest.fail(f"the scheduler still waited after {seconds} s on rollouts past their deadline")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["batch"]


def test_a_rollout_past_its_deadline_is_cancelled_and_retried_and_counts_as_a_failed_attempt():
    source = Scripted(lambda task, submission, sample, version: None if (task, submission, sample) == ("1", 0, 0)
                      else finished(float(sample), version))
    rollout, data, records = scheduler(source, ahead=0, timeout=0.05)
    batch = bounded(rollout, State(0), next(iter(data.train())), source)
    assert records[-1].resubmitted == {"timeout": 1} and records[-1].cancelled == 1
    assert source.cancelled == source.running and trained(batch)[0] == [0, 1, 2, 3]

    hung = Scripted(lambda task, submission, sample, version: None if task == "1" else finished(
        float(sample), version))
    rollout, data, records = scheduler(hung, ahead=0, timeout=0.05, max_attempts=2)
    bounded(rollout, State(0), next(iter(data.train())), hung)
    assert (records[-1].groups, records[-1].abandoned) == (1, 1)
    assert all(future.cancelled() for future in hung.running)


def test_admit_takes_the_first_complete_groups_and_cancels_the_rest():
    source = Scripted(lambda task, submission, sample, version: None if task == "2" else finished(
        float(sample), version))
    rollout, data, records = scheduler(source, ahead=0, admit=2, stream=((1, 2, 3),))
    batch = rollout(State(0), next(iter(data.train())), None)
    assert records[-1].groups == 2 and trained(batch)[0] == [0, 1, 2, 3]
    assert len(source.cancelled) == 2


def test_stale_rollouts_are_discarded_and_redrawn_under_pushed_weights():
    # Batch 1 was submitted at version 0; three updates land before it is
    # consumed. Task 3's rollouts finished under version 0 and are dropped;
    # task 4's still run and are cancelled without being waited on.
    publisher = Publisher()
    source = Scripted(lambda task, submission, sample, version: None if (task == "4" and version == 0)
                      else finished(float(sample), version))
    rollout, data, records = scheduler(source, publisher)
    stream = iter(data.train())
    first, second = next(stream), next(stream)
    rollout(State(0), first, None)
    batch = rollout(State(3), second, None)
    assert publisher.loads == [3]
    assert records[-1].resubmitted == {"stale": 4} and records[-1].lag == 0
    assert set(batch["versions"][batch["response_mask"] > 0].tolist()) == {3}
    assert len(source.cancelled) == 2


def test_a_rollout_spanning_a_push_keeps_its_oldest_version():
    source = Scripted(lambda task, submission, sample, version: finished(float(sample), 0, 1))
    rollout, data, records = scheduler(source)
    stream = iter(data.train())
    rollout(State(0), next(stream), None)
    batch = rollout(State(1), next(stream), None)
    assert records[-1].version == 0 and records[-1].lag == 1
    assert set(batch["versions"][batch["response_mask"] > 0].tolist()) == {0, 1}


def test_a_stalled_push_is_pushed_again_and_one_that_never_takes_raises():
    source = Scripted(lambda task, submission, sample, version: finished(float(sample), version))
    publisher = Publisher(stall=1)
    rollout, data, _ = scheduler(source, publisher, ahead=0)
    rollout(State(2), next(iter(data.train())), None)
    assert publisher.loads == [2, 2] and publisher.version == 2

    rollout, data, _ = scheduler(source, Publisher(stall=5), ahead=0)
    with pytest.raises(RuntimeError, match="did not take"):
        rollout(State(2), next(iter(data.train())), None)


def test_a_resumed_stream_cancels_the_old_in_flight_rollouts_and_resubmits_under_restored_weights():
    source = Scripted(lambda task, submission, sample, version: None if version == 0 and task in "34"
                      else finished(float(sample), version))
    publisher = Publisher()
    rollout, data, _ = scheduler(source, publisher)
    stream = iter(data.train())
    first, _ = next(stream), next(stream)
    rollout(State(0), first, None)
    in_flight = list(source.running)
    assert len(in_flight) == 4

    # A resume reopens the stream with the restored update clock; the engines
    # serve what they served before. The scripted stream restarts at batch 0.
    resumed = iter(data.train())
    assert all(future.cancelled() for future in in_flight)
    first, _ = next(resumed), next(resumed)
    batch = rollout(State(5), first, None)
    assert publisher.loads == [5]
    assert source.submitted[-4:] == [("1", 2, 5), ("2", 2, 5), ("3", 2, 5), ("4", 2, 5)]
    assert set(batch["versions"][batch["response_mask"] > 0].tolist()) == {5}


def test_engines_serving_weights_newer_than_the_restored_clock_are_pushed_back():
    # A fit re-run in the same process restores update 5 while the engines
    # still serve the version pushed at update 10.
    publisher = Publisher(version=10)
    source = Scripted(lambda task, submission, sample, version: finished(float(sample), version))
    rollout, data, records = scheduler(source, publisher, ahead=0)
    batch = rollout(State(5), next(iter(data.train())), None)
    assert publisher.loads == [5] and source.submitted == [("1", 2, 5), ("2", 2, 5)]
    assert records[-1].lag == 0
    assert set(batch["versions"][batch["response_mask"] > 0].tolist()) == {5}


def test_a_source_that_raises_stops_the_batch_and_cancels_its_work():
    source = Scripted(lambda task, submission, sample, version: ValueError("broken source") if task == "1"
                      else None)
    rollout, data, _ = scheduler(source, ahead=0)
    with pytest.raises(ValueError, match="broken source"):
        rollout(State(0), next(iter(data.train())), None)
    assert len(source.running) == 2 and all(future.cancelled() for future in source.running)


def test_reward_components_from_the_source_are_averaged_into_the_record():
    source = Scripted(lambda task, submission, sample, version: finished(
        float(sample), version, components={"tests": float(sample), "format": 1.0}))
    rollout, data, records = scheduler(source, ahead=0)
    rollout(State(0), next(iter(data.train())), None)
    metrics = records[-1].metrics
    assert (metrics["reward/component/format"], metrics["reward/component/tests"]) == (1.0, 0.5)


def test_a_batch_that_did_not_come_through_the_stream_is_refused():
    rollout, data, _ = scheduler(Scripted(lambda *_: None))
    next(iter(data.train()))
    with pytest.raises(ValueError, match="next registered"):
        rollout(State(0), {"task_id": np.asarray([8, 9], np.int32)}, None)


@pytest.mark.parametrize("options, message", [
    ({"max_lag": 1, "ahead": 2}, "past max_lag"),
    ({"max_lag": 1, "ahead": 1, "sync_every": 2}, "past max_lag"),
    ({"max_lag": 0, "ahead": 1}, "past max_lag"),
])
def test_a_schedule_that_could_exceed_the_bound_is_refused(options, message):
    with pytest.raises(ValueError, match=message):
        RolloutScheduler(Objective(), Scripted(lambda *_: None), Publisher(), width=WIDTH, rows=ROWS, **options)


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
    sampling = Sampling(temperature=1.0, eos_id=STOP)
    server = NativeRolloutServer(Server.from_task(TextGeneration(model, params, None, sampling=sampling),
                                                  slots=16, capacity=64))

    @contextmanager
    def environment(identity):
        yield Tools(identity.task)

    def verifier(episode):
        sampled = [token for turn in episode.transitions for token in turn.action.tokens if token != STOP]
        share = sum(token == 5 for token in sampled) / max(len(sampled), 1)
        return Score(share, {"turns": float(len(episode.transitions))})

    episodes = EnvironmentSource(server, environment, verifier, max_prompt_tokens=32, max_new_tokens=16,
                                 max_turns=3, workers=16)
    rollouts = []
    submit = episodes.submit

    def watched(task, samples, *, version):
        futures = submit(task, samples, version=version)
        for future in futures:
            future.add_done_callback(lambda done: rollouts.append(done.result()) if not done.cancelled() else None)
        return futures

    episodes.submit = watched
    tasks = jax.device_count()
    records = []
    scheduler = RolloutScheduler(target, episodes, server, width=width, rows=2 * tasks, groups=2,
                                 max_lag=2, ahead=1, sync_every=2, log=records.append)
    stream = scheduler.tasks(Dataset(train=lambda: ({"task_id": np.arange(tasks, dtype=np.int32) + step * tasks}
                                                    for step in range(100)), val=None, records=None, batch=tasks))
    try:
        trainer = Trainer(target, optax.adam(1e-2), key=jax.random.key(3), rollout=scheduler,
                          layout=Layout(min_shard=1, tolerance=1.0))
        state = trainer.fit(stream, steps=4, log_every=4)
    finally:
        scheduler.close()
        episodes.close()
        server.close()
    assert int(state.updates) == 4 and [record.updates for record in records] == [0, 1, 2, 3]
    assert all(0 <= record.lag <= 2 for record in records)
    assert any(record.lag > 0 for record in records), "no batch was ever drawn ahead of its update"
    # Every scored session took both turns, and each pair of calls merged into one chain.
    assert all(record.metrics.get("reward/component/turns", 2.0) == 2.0 for record in records)
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
    assert not all(jnp.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(state.params), strict=True))
