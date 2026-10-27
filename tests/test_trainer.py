"""The general trainer on a small objective: the loop, checkpoints, resume,
validation, the rejected mixed-precision step and the custom step.

Nothing here knows a modality. The objective is a two-output affine map, the
data is synthetic, and what is asserted is what the trainer owns: the step
count, what lands on disk and when, what a resume restores, what reaches the
tracker, and what a failure does to the run.

Local resume combines Dew's run and data-position policies; Orbax oracles the stored state.
"""

import collections
import contextlib
import dataclasses
import itertools
import json
import os
import subprocess
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest
from affine_run import BATCH, FEATURES, Counting, Features, Regression, Spread, dataset, raw_leaf, val_batches
from flax.core import FrozenDict, freeze as frozen_dict
from recording import Preempting, RecordingTracker
from steady_state import steady_state
from token_support import write_token_corpus

from dew import position
from dew.artifacts import Representations
from dew.batch import merge, under
from dew.checkpoints import Ranking
from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loader, replayed
from dew.objectives.base import Aux, EMASpec, Objective
from dew.optim import Cosine, OptimConfig, Power
from dew.state import CHECKPOINTED, TrainState
from dew.training import Checkpoints, Layout, MeshSpec, Trainer, ema_update
from dew.training.evaluation import WARM_BATCHES
from dew.training.model_flops import _TokenCount
from dew.training.runtime import Preempted


def endless():
    """A stream without get_state."""
    source = Counting()
    while True:
        yield next(source)


def make_trainer(tmp_path=None, objective=None, optimizer=None, keep=3, **kwargs):
    checkpoints = None if tmp_path is None else Checkpoints(str(tmp_path / "run"), keep=keep)
    return Trainer(
        Regression() if objective is None else objective,
        optax.sgd(0.1) if optimizer is None else optimizer,
        key=jax.random.key(0),
        layout=Layout(min_shard=1, tolerance=1.0),
        checkpoints=checkpoints,
        **kwargs,
    )


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

class TokenRegression(Regression):
    counts_tokens = True

    def scored_tokens(self, batch):
        return batch["index"] + 1


@pytest.mark.parametrize("log_every", [1, 4])
@pytest.mark.parametrize(("accumulation", "steps", "logged"), [(1, 3, [8, 16, 24]), (2, 4, [0, 24, 0, 56])])
def test_a_real_token_budget_counts_each_microbatch_without_padding(log_every, accumulation, steps, logged):
    """Eight rows of lengths 1, 2, 3, 4 a step, without padding: a budget of 30 tokens counts the
    updates committed, so it ends at the first commit past 30, step 3 (48) one microbatch an update
    and step 4 (80) two, whose first update commits 24."""
    tracker = RecordingTracker()
    trainer = make_trainer(objective=TokenRegression(), tracker=tracker, accumulation=accumulation)
    state = trainer.fit(dataset(), tokens=30, log_every=log_every)
    assert int(state.step) == steps
    if log_every == 1:
        assert [scalars["train/tokens"] for _, scalars in tracker.scalars
                if "train/tokens" in scalars] == logged


def test_a_token_budget_and_step_cap_stop_at_the_first_reached():
    state = make_trainer(objective=TokenRegression()).fit(dataset(), steps=2, tokens=30)
    assert int(state.step) == 2


@pytest.mark.parametrize("schedule", [None, Power(peak_value=.1, warmup_steps=2, a=.1, b=0),
                                       Cosine(peak_value=.1, warmup_steps=1, decay_steps=8)])
def test_a_token_only_run_uses_the_schedules_own_horizon(schedule):
    config = OptimConfig() if schedule is None else OptimConfig(learning_rate=schedule)
    state = make_trainer(objective=TokenRegression(), optimizer=config).fit(dataset(), tokens=30)
    assert int(state.step) == 3


def test_a_token_budget_resumes_its_count_from_the_checkpoint(tmp_path):
    first = make_trainer(tmp_path, objective=TokenRegression())
    assert int(first.fit(dataset(), steps=2, tokens=30, log_every=1).step) == 2
    assert first.checkpoints.control(2)["committed_tokens"] == 24
    resumed = make_trainer(tmp_path, objective=TokenRegression())
    assert int(resumed.fit(dataset(), tokens=30, log_every=1).step) == 3
    assert resumed.checkpoints.control(3)["committed_tokens"] == 48
    complete = make_trainer(tmp_path, objective=TokenRegression())
    assert int(complete.fit(dataset(), tokens=30).step) == 3


def test_a_token_budget_counts_the_tokens_of_committed_updates(tmp_path):
    """Under an accumulation of two, a run's tokens are those of the updates it committed: a run
    stopped inside a window counts none of its batch, and its resume counts that batch at the
    window's commit, once (spec training §3)."""
    first = make_trainer(tmp_path, objective=TokenRegression(), accumulation=2)
    stopped = first.fit(dataset(), steps=3, log_every=1)
    assert int(stopped.step) == 3
    assert first.checkpoints.control(3)["committed_tokens"] == 24
    assert _TokenCount.host(jax.device_get(stopped.accumulation.tokens)) == 24
    resumed = make_trainer(tmp_path, objective=TokenRegression(), accumulation=2)
    assert int(resumed.fit(dataset(), steps=4, log_every=1).step) == 4
    assert resumed.checkpoints.control(4)["committed_tokens"] == 80


def test_a_fit_places_its_device_prefetch_and_plans_its_step_beside_them():
    """`fit(device_prefetch=n)` queues n placed batches beside the step, and the step's memory plan holds
    room for them and the one being placed: n + 1 batches, so each batch deeper grows the held bytes by
    one placed batch's bytes, the same each time, and the run trains the same."""
    def run(depth: int):
        trainer = make_trainer()
        state = trainer.fit(dataset(), steps=2, log_every=1, device_prefetch=depth)
        return trainer.memory["held"], jax.tree.map(np.asarray, state.variables)

    (one, first), (two, _), (three, third) = run(1), run(2), run(3)
    assert two - one == three - two > 0
    for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(third), strict=True):
        np.testing.assert_array_equal(left, right)


def test_a_checkpoint_that_counted_every_steps_tokens_is_refused_by_name(tmp_path):
    """A checkpoint whose control counts every step's tokens under `tokens`, the field runs wrote before
    they counted commits, holds no count this Dew reads: a run resuming from it is refused, by steps
    or by tokens, rather than counting from zero."""
    import json
    import pathlib

    first = make_trainer(tmp_path, objective=TokenRegression())
    first.fit(dataset(), steps=2, log_every=1, checkpoint_every=1)
    first.checkpoints.wait()
    # The step's metadata as a run before commits counted wrote it: the old field, every step's count.
    (written,) = pathlib.Path(first.checkpoints.directory, "2").glob("_CHECKPOINT_METADATA")
    metadata = json.loads(written.read_text())
    control = metadata["custom_metadata"]["control"]
    control["tokens"] = control.pop("committed_tokens")
    written.write_text(json.dumps(metadata))
    for length in ({"steps": 3}, {"tokens": 60}):
        with pytest.raises(ValueError):
            make_trainer(tmp_path, objective=TokenRegression()).fit(dataset(), log_every=1, **length)


def test_an_explicit_state_starts_a_token_count_with_its_own_reader(tmp_path):
    make_trainer(tmp_path, objective=TokenRegression()).fit(dataset(), steps=2, tokens=30)
    trainer = make_trainer(tmp_path, objective=TokenRegression())
    state = trainer.initial_state()
    assert int(trainer.fit(dataset(), tokens=30, state=state).step) == 3


def test_a_step_only_prefix_keeps_lifetime_real_tokens(tmp_path):
    first = make_trainer(tmp_path, objective=TokenRegression())
    first.fit(dataset(), steps=2, log_every=1)
    assert first.checkpoints.control(2)["committed_tokens"] == 24
    resumed = make_trainer(tmp_path, objective=TokenRegression())
    assert int(resumed.fit(dataset(), tokens=30, log_every=1).step) == 3


class LargeTokenRegression(Regression):
    counts_tokens = True

    def scored_tokens(self, batch):
        return batch["real_length"]


class LargeTokenRows(Counting):
    def __next__(self):
        batch = super().__next__()
        lengths = np.full(BATCH, 32768, np.int64)
        lengths[-1] -= 1
        return {**batch, "real_length": lengths}


@pytest.mark.parametrize("cadence", [1, 100])
def test_a_token_budget_stays_exact_above_float32_integer_resolution(cadence):
    trainer = make_trainer(objective=LargeTokenRegression())
    state = trainer.fit(dataset(LargeTokenRows), tokens=65 * 262143 + 1, log_every=cadence)
    assert int(state.step) == 66


@pytest.mark.parametrize("cadence", [1, 100])
def test_resumed_tokens_stay_exact_above_float32_integer_resolution(tmp_path, cadence):
    first = make_trainer(tmp_path, objective=LargeTokenRegression())
    first.fit(dataset(LargeTokenRows), steps=65, log_every=cadence)
    assert first.checkpoints.control(65)["committed_tokens"] == 65 * 262143
    resumed = make_trainer(tmp_path, objective=LargeTokenRegression())
    state = resumed.fit(dataset(LargeTokenRows), tokens=65 * 262143 + 1, log_every=cadence)
    assert int(state.step) == 66


@pytest.mark.parametrize("variant", ["ema", "accumulation", "schedule", "dynamic_scale", "checkpoints"])
def test_steps_after_the_first_logs_neither_compile_nor_move_data_unasked(variant, tmp_path):
    """Past its first two logging intervals and checkpoints the loop reruns
    the programs it compiled, and the only data that crosses is the batches
    it places and what it reads, by name, at the logging and checkpoint
    cadences (`steady_state`). A float() of the loss in the loop, a fresh
    counter built on the host, or a counter on another device than the
    loss's would each fail it."""
    window = contextlib.ExitStack()

    class Steady(RecordingTracker):
        def log(self, scalars, step):
            super().log(scalars, step)
            if step == 8:
                window.enter_context(steady_state())
            elif step == 24:
                window.close()

    options = {"ema": {}, "accumulation": {"accumulation": 2}, "dynamic_scale": {"dynamic_scale": True},
               "schedule": {"optimizer": optax.inject_hyperparams(optax.adam)(
                   learning_rate=optax.cosine_decay_schedule(1e-2, 24))},
               "checkpoints": {"tmp_path": tmp_path}}[variant]
    tracker = Steady()
    try:
        make_trainer(tracker=tracker, **options).fit(
            dataset(), steps=24, log_every=4, checkpoint_every=4 if variant == "checkpoints" else None)
    finally:
        window.close()
    assert [step for step, scalars in tracker.scalars if "train/loss" in scalars] == [4, 8, 12, 16, 20, 24]


def test_integer_root_key_matches_a_typed_key_bit_exactly():
    integer = Trainer(Regression(), optax.adam(1e-3), key=0)
    typed = Trainer(Regression(), optax.adam(1e-3), key=jax.random.key(0))
    left = integer.fit(dataset(), steps=3, log_every=1)
    right = typed.fit(dataset(), steps=3, log_every=1)
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for actual, expected in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(np.asarray(raw_leaf(actual)), np.asarray(raw_leaf(expected)))


def test_typed_root_key_round_trips_and_resumes_bit_exactly(tmp_path):
    def build(path=None):
        return Trainer(Regression(), optax.adam(1e-3), key=0,
                       checkpoints=None if path is None else Checkpoints(str(path)))

    baseline = build().fit(dataset(), steps=4, log_every=1)
    split = build(tmp_path / "typed")
    prefix = split.fit(dataset(), steps=2, checkpoint_every=1, log_every=1)
    assert jnp.issubdtype(prefix.key.dtype, jax.dtypes.prng_key)
    fresh = build(tmp_path / "typed")
    restored, _, position = fresh.place()
    assert restored.key.dtype == prefix.key.dtype
    assert int(prefix.step) == int(restored.step) == 2
    for field in ("variables", "opt_state", "ema"):
        for before, after in zip(jax.tree.leaves(getattr(prefix, field)),
                                 jax.tree.leaves(getattr(restored, field)), strict=True):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after), err_msg=field)
    assert json.loads(position)["index"] == 2
    np.testing.assert_array_equal(jax.random.key_data(restored.key), jax.random.key_data(prefix.key))
    resumed = fresh.fit(dataset(), steps=4, checkpoint_every=1, log_every=1)
    assert int(resumed.step) == 4 and fresh.checkpoints.latest == 4
    assert (tmp_path / "typed" / "4").is_dir()
    assert not (tmp_path / "typed" / "0").exists()
    for left, right in zip(jax.tree.leaves(resumed), jax.tree.leaves(baseline), strict=True):
        np.testing.assert_array_equal(np.asarray(raw_leaf(left)), np.asarray(raw_leaf(right)))


# The threads Orbax writes checkpoints on, by the names orbax-checkpoint 0.12 gives them: its
# asynchronous save, its metadata stores' executors, its deleter and the event loops it runs.
WRITERS = ("async_save", "metadata_store", "DeleterThread", "ScanThread", "_run_loop", "_run_event_loop")


def checkpoint_writers() -> set[threading.Thread]:
    """The live threads of this process that write checkpoints."""
    return {thread for thread in threading.enumerate() if any(name in thread.name for name in WRITERS)}


def test_closed_checkpoints_leave_no_writer_running_across_trials(tmp_path):
    """Trials in one process each close their checkpoints, as RunConfig.train does, so Orbax's
    managers and metadata stores leave no writer behind: after the first, which starts what Orbax
    keeps for the process, every checkpoint writer alive after each trial was alive before it,
    whatever threads the rest of the process starts or ends, and a closed Checkpoints reopens to
    read what it wrote."""
    def trial(index):
        with Checkpoints(str(tmp_path / str(index))) as checkpoints:
            Trainer(Regression(), optax.adam(1e-3), key=0, checkpoints=checkpoints).fit(
                dataset(), steps=2, checkpoint_every=1, log_every=1)
        return checkpoints

    trial(0)
    before = checkpoint_writers()
    for index in (1, 2):
        reopened = trial(index)
        assert checkpoint_writers() <= before
    assert reopened.latest == 2


def failed_write(monkeypatch) -> OSError:
    """The failure Orbax reports for an asynchronous write that failed, as a full disk fails one:
    once, from the next wait on its manager, which then holds no error. Stood in for at that
    boundary, since a test cannot fill a disk."""
    failure, reported = OSError("async checkpoint write failed"), []
    finished = ocp.CheckpointManager.wait_until_finished

    def wait(manager):
        finished(manager)
        if not reported:
            reported.append(manager)
            raise failure

    monkeypatch.setattr(ocp.CheckpointManager, "wait_until_finished", wait)
    return failure


@pytest.mark.parametrize("body_fails", [False, True])
def test_checkpoints_whose_write_failed_close_and_keep_the_error_their_run_raised(tmp_path, monkeypatch,
                                                                                  body_fails):
    """A run whose last checkpoint write failed still closes its checkpoints' managers, each waited
    on and closed after the failure, so no checkpoint writer outlives the scope. A run that failed
    first raises its own error with the write's failure noted on it; one that did not raises the
    write's."""
    def trained(index):
        checkpoints = Checkpoints(str(tmp_path / str(index)))
        Trainer(Regression(), optax.adam(1e-3), key=0, checkpoints=checkpoints).fit(
            dataset(), steps=2, checkpoint_every=1, log_every=1)
        return checkpoints

    with trained(0):
        pass
    before = checkpoint_writers()
    primary = ValueError("training failed first")
    with pytest.raises((ValueError, OSError)) as raised, trained(1):
        failure = failed_write(monkeypatch)
        if body_fails:
            raise primary
    assert raised.value is (primary if body_fails else failure)
    assert bool(getattr(primary, "__notes__", ())) == body_fails
    assert checkpoint_writers() <= before


def test_a_run_that_fails_after_a_failed_write_raises_its_own_error_and_closes_its_checkpoints(
        tmp_path, monkeypatch):
    """RunConfig.train closes the run's tracker and checkpoints however it ends: a run whose step
    after training fails, once its last checkpoint write failed, raises its own error with the
    write's failure noted on it, and leaves no checkpoint writer behind."""
    from dew.config import RunConfig, TrainerConfig

    run = RunConfig(trainer=TrainerConfig(steps=2, checkpoint_dir=str(tmp_path),
                                          eval_every=None, checkpoint_every=1))
    run.train(Regression(), dataset(), name="settled")
    before = checkpoint_writers()
    primary = ValueError("calibration failed")

    def after(state, directory):
        failed_write(monkeypatch)
        raise primary

    with pytest.raises(ValueError) as raised:
        run.train(Regression(), dataset(), name="failed", after=after)
    assert raised.value is primary and primary.__notes__
    assert checkpoint_writers() <= before


class Keyed(Regression):
    """Loss scaled by the step key, so the key stream is observable in the parameters.

    Scaling multiplies the gradients, which a constant offset would not, so two
    runs that draw different keys land in different places.
    """

    def loss(self, variables, batch, step):
        base, aux = super().loss(variables, batch, step)
        factor = 1.0 + 0.01 * jax.random.normal(step.key, ())
        return base * factor, aux


def test_a_resumed_run_continues_the_key_stream(tmp_path):
    """A resume continues fold_in(run_key, step) where it stopped.

    The data position already decides which batches the resumed run sees, so an
    unkeyed loss cannot tell a resumed run from a restarted key stream. Scaling
    the loss by the step key makes the keys observable: a resume that restarted
    its keys, or restored the step but not the run key, lands where the
    uninterrupted run does not.
    """
    make_trainer(tmp_path, objective=Keyed()).fit(dataset(), steps=2, log_every=1)
    resumed = make_trainer(tmp_path, objective=Keyed()).fit(dataset(), steps=4, log_every=1)
    whole = make_trainer(objective=Keyed()).fit(dataset(), steps=4, log_every=1)

    # The restore is exact and the resumed steps are the uninterrupted run's program on the same
    # values, so a resume that kept its key stream lands on the same bits.
    for expected, actual in zip(jax.tree.leaves(whole.variables),
                                jax.tree.leaves(resumed.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def drawn(seed=0, opened=None, diverge=None):
    """A plain generator, without a position of its own: each batch drawn
    from `seed`'s stream, or from fresh entropy with `seed=None`. With
    `opened` (a list counting the generator's openings) and `diverge`, the
    batches past index `diverge` differ on each opening."""
    if opened is not None:
        opened.append(None)
    rng = np.random.default_rng(seed)
    for index in itertools.count():
        if diverge is not None and index == diverge + 1:
            rng = np.random.default_rng([seed, len(opened)])
        x = rng.normal(size=(BATCH, FEATURES)).astype(np.float32)
        yield {"x": x, "y": 2 * x[:, :2]}


def stream(reader):
    return Loader(reader, None, batch=BATCH, records=None)


def preempted(tmp_path, data, at=3, steps=6):
    """`data` trained toward `steps` and preempted when step `at` is logged, its checkpoint written."""
    with pytest.raises(Preempted):
        make_trainer(tmp_path, tracker=Preempting(at)).fit(data, steps=steps, log_every=1)
    assert Checkpoints(str(tmp_path / "run")).latest is not None


def preempted_and_resumed(tmp_path, data, at=3, steps=6):
    """`data` trained to `steps`, preempted when step `at` is logged, and resumed in a fresh trainer."""
    preempted(tmp_path, data, at, steps)
    return make_trainer(tmp_path).fit(data, steps=steps, log_every=1)


def same_variables(left, right) -> bool:
    return all(np.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(jax.device_get(left.variables)), jax.tree.leaves(jax.device_get(right.variables)),
        strict=True))


def test_a_generator_declared_to_replay_resumes_on_the_batches_the_whole_run_reads(tmp_path):
    """A generator declared to read the same batches each time it opens is
    positioned by replay: preempted at step 3 and resumed, it reads its first
    three batches again and trains on the rest, landing on the uninterrupted
    run bit for bit."""
    data = stream(replayed(lambda partition: drawn()))
    resumed = preempted_and_resumed(tmp_path, data)
    assert int(resumed.step) == 6
    assert same_variables(resumed, make_trainer().fit(data, steps=6, log_every=1))


def test_an_undeclared_generator_trains_and_its_resume_is_refused(tmp_path):
    """A plain generator trains and checkpoints, but its checkpoint holds no
    position, so a resume, which would read its data again from the start,
    is refused."""
    data = stream(lambda partition: drawn())
    preempted(tmp_path, data)
    with pytest.raises(ValueError):
        make_trainer(tmp_path).fit(data, steps=6, log_every=1)


def test_a_final_checkpoint_resumes_only_a_stream_that_keeps_or_replays_its_position(tmp_path):
    """A run that writes only its final checkpoint, as by default, holds no data position for a plain
    generator: a later run toward more steps, which would read its data again from the start, is
    refused by name. A generator declared to replay resumes from it onto the uninterrupted run."""
    plain = stream(lambda partition: drawn())
    make_trainer(tmp_path / "plain").fit(plain, steps=3, log_every=1)
    assert Checkpoints(str(tmp_path / "plain" / "run")).latest == 3
    with pytest.raises(ValueError):
        make_trainer(tmp_path / "plain").fit(plain, steps=6, log_every=1)
    declared = stream(replayed(lambda partition: drawn()))
    make_trainer(tmp_path / "declared").fit(declared, steps=3, log_every=1)
    resumed = make_trainer(tmp_path / "declared").fit(declared, steps=6, log_every=1)
    assert int(resumed.step) == 6
    assert same_variables(resumed, make_trainer().fit(declared, steps=6, log_every=1))


def test_a_checkpoint_saved_without_a_position_resumes_no_stream_past_step_zero(tmp_path):
    """A checkpoint saved by hand without the stream's position (Checkpoints.save's `saved`) past
    step 0 resumes no stream, not one that keeps positions either: with none to restore it would
    read its data again from the start, apart from the uninterrupted run, so the run is refused. At
    step 0 nothing was read, and the run trains on."""
    trainer = make_trainer(tmp_path)
    state = trainer.fit(dataset(), steps=3, log_every=1, checkpoint_final=False)
    trainer.checkpoints.save(3, state)
    trainer.checkpoints.wait()
    with pytest.raises(ValueError):
        make_trainer(tmp_path).fit(dataset(), steps=6, log_every=1)
    fresh = make_trainer(tmp_path / "fresh")
    fresh.checkpoints.save(0, fresh.initial_state())
    fresh.checkpoints.wait()
    assert int(make_trainer(tmp_path / "fresh").fit(dataset(), steps=2, log_every=1).step) == 2


def test_a_declared_stream_whose_batch_at_the_position_differs_is_refused(tmp_path):
    """The replay guard: a stream declared to replay that reads back another
    batch at its position, one over fresh entropy, is refused."""
    data = stream(replayed(lambda partition: drawn(None)))
    preempted(tmp_path, data)
    with pytest.raises(ValueError):
        make_trainer(tmp_path).fit(data, steps=6, log_every=1)


def test_the_replay_guard_reads_only_the_batch_at_the_position(tmp_path):
    """The guard's limit: a declared stream that agrees up to its position,
    the four batches before the preemption, and differs after it resumes
    without complaint, on other data than the uninterrupted run read, which
    is why replay is declared, not assumed."""
    opened = []
    data = stream(replayed(lambda partition: drawn(0, opened, diverge=3)))
    resumed = preempted_and_resumed(tmp_path, data)
    assert int(resumed.step) == 6
    assert not same_variables(resumed, make_trainer().fit(data, steps=6, log_every=1))


# --------------------------------------------------------------------------
# Checkpoints: what lands, when, and what a resume gets back
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cadence, steps, expected", [(None, 5, set()), (2, 5, {2, 4}), (2, 6, {2, 4, 6})])
def test_final_checkpoint_policy_is_independent_of_the_cadence(tmp_path, cadence, steps, expected):
    trainer = make_trainer(tmp_path, keep=4)
    trainer.fit(dataset(), steps=steps, checkpoint_every=cadence, checkpoint_final=False)
    assert set(trainer.checkpoints._open().all_steps()) == expected


def held_lm_trainer(container=dict, **settings):
    """A one-layer LM trainer over held weights, which the objective holds
    in `container` (Flax's FrozenDict, say)."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective

    model = CausalTransformer(vocab_size=32, emb_features=8, num_layers=1, num_heads=1,
                              mlp_features=16, max_seq_len=8)
    weights = jax.jit(LMObjective(model, seq_len=4).init)(jax.random.key(0))
    objective = LMObjective(model, seq_len=4, variables=container(weights), ema_decay=0.999)
    return Trainer(objective, optax.adam(1e-3), key=jax.random.key(0),
                   layout=Layout(min_shard=1, tolerance=1.0), **settings), objective, weights


def test_native_frozen_variables_train_save_and_resume_as_plain_ones(tmp_path):
    """A FrozenDict of variables, Flax's own container, trains, saves and
    resumes as the same tree of dicts does, to the same numbers, and a
    supplied state holding one fits: the trainer holds every tree as dicts."""
    from flax.core import FrozenDict

    batch = {"text": jnp.tile(jnp.arange(1, 6, dtype=jnp.int32)[None], (jax.device_count(), 1))}

    def trained(container, directory, steps):
        trainer, _, _ = held_lm_trainer(container, checkpoints=Checkpoints(str(directory)))
        state, _, _ = trainer.place()
        step = trainer.compile(state, batch)
        for _ in range(steps):
            state, *_ = step(state, batch)
        trainer.checkpoints.save(int(state.step), state)
        trainer.checkpoints.wait()
        return trainer, state

    _, frozen = trained(FrozenDict, tmp_path / "frozen", 2)
    assert type(frozen.variables) is dict and type(frozen.variables["params"]) is dict
    _, resumed = trained(FrozenDict, tmp_path / "frozen", 2)
    _, plain = trained(dict, tmp_path / "plain", 4)
    assert int(resumed.step) == 4
    for got, want in zip(jax.tree.leaves((resumed.variables, resumed.opt_state)),
                         jax.tree.leaves((plain.variables, plain.opt_state)), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    regression = make_trainer(objective=Regression())
    supplied = regression.initial_state()
    assert int(regression.fit(dataset(), steps=1, state=supplied.replace(
        variables=FrozenDict(supplied.variables))).step) == 1


def test_a_runs_checkpoint_opens_with_orbax_alone(tmp_path):
    """A step is one Orbax PyTree item: Orbax's own manager reads the state's
    fields by name, the PRNG key as a key, and an EMA leaf the metadata
    lists as byte planes decodes to the run's average bit for bit, as the
    checkpoint guide says."""
    import orbax.checkpoint as ocp

    trainer, _, _ = held_lm_trainer(checkpoints=Checkpoints(str(tmp_path)))
    state, _, _ = trainer.place()
    batch = {"text": jnp.tile(jnp.arange(1, 6, dtype=jnp.int32)[None], (jax.device_count(), 1))}
    step = trainer.compile(state, batch)
    for _ in range(2):
        state, *_ = step(state, batch)
    trainer.checkpoints.save(int(state.step), state)
    trainer.checkpoints.wait()

    manager = ocp.CheckpointManager(str(tmp_path), item_handlers=ocp.PyTreeCheckpointHandler())
    tree = manager.restore(manager.latest_step(), args=ocp.args.PyTreeRestore())
    deltas = manager.metadata(manager.latest_step()).custom_metadata["ema_deltas"]
    assert int(tree["step"]) == int(state.step) == 2
    assert jnp.issubdtype(tree["key"].dtype, jax.dtypes.prng_key)
    np.testing.assert_array_equal(jax.random.key_data(tree["key"]), jax.random.key_data(state.key))
    for got, want in zip(jax.tree.leaves(tree["variables"]), jax.tree.leaves(state.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    weights = {jax.tree_util.keystr(path): np.asarray(leaf)
               for path, leaf in jax.tree_util.tree_flatten_with_path(tree["variables"])[0]}
    averages = dict(jax.tree_util.tree_flatten_with_path(state.ema)[0])
    decoded = 0
    for path, stored in jax.tree_util.tree_flatten_with_path(tree["ema"])[0]:
        name, want = jax.tree_util.keystr(path), np.asarray(averages[path])
        if name in deltas:
            live = weights[name]
            unsigned = np.dtype(f"uint{8 * live.dtype.itemsize}")
            planes = np.asarray(stored)
            bits = sum(planes[k].astype(unsigned) << unsigned.type(8 * k) for k in range(len(planes)))
            stored, decoded = (live.view(unsigned) ^ bits).view(live.dtype), decoded + 1
        np.testing.assert_array_equal(np.asarray(stored), want)
    assert decoded == len(deltas) > 0


def test_overlapping_token_windows_resume_in_a_fresh_trainer_bit_exactly(tmp_path):
    from dew.data import Loading, presets
    from dew.data.dataset import Forwarding
    from dew.data.sources.tokens import TokenFiles

    corpus = write_token_corpus(tmp_path / "tokens", np.arange(29))

    class WindowsRegression(Regression):
        def loss(self, variables, batch, step):
            x = batch["text"][:, :FEATURES].astype(jnp.float32) / 32
            return super().loss(variables, {"x": x, "y": 2 * x[:, :2]}, step)

    class Observed(Forwarding):
        def __init__(self, source, seen):
            self._source, self.seen = source, seen

        def __iter__(self):
            return self

        def __next__(self):
            batch = next(self._source)
            self.seen.append(batch["text"].copy())
            return batch

        def get_state(self):
            return self._source.get_state()

        def set_state(self, state):
            self._source.set_state(state)

    def dataset(seen):
        windows = TokenFiles(corpus.train).pipe(presets.token_windows, seq_len=4, batch=BATCH, stride=1)
        windows = windows.read("memory").loading(Loading(workers=0, threads=1, read_buffer=1))
        data = compile(windows, seed=7)
        return dataclasses.replace(data, train=lambda partition: Observed(data.train(partition), seen))

    def trainer(directory):
        return make_trainer(directory, objective=WindowsRegression(), optimizer=optax.adam(1e-3))

    whole_seen, prefix_seen, resumed_seen = [], [], []
    whole = trainer(tmp_path / "whole").fit(dataset(whole_seen), steps=4, checkpoint_every=1)
    prefix = trainer(tmp_path / "split").fit(dataset(prefix_seen), steps=2, checkpoint_every=1)
    fresh = trainer(tmp_path / "split")
    restored, _, place = fresh.place()
    assert position.read(place).records == 2 * BATCH
    assert jax.tree.structure(restored) == jax.tree.structure(prefix)
    for left, right in zip(jax.tree.leaves(restored), jax.tree.leaves(prefix), strict=True):
        actual, expected = np.asarray(raw_leaf(left)), np.asarray(raw_leaf(right))
        assert (actual.dtype, actual.shape, actual.tobytes()) == (
            expected.dtype,
            expected.shape,
            expected.tobytes(),
        )
    resumed = fresh.fit(dataset(resumed_seen), steps=4, checkpoint_every=1)
    assert resumed_seen[0].tobytes() == whole_seen[2].tobytes(), "the first batch after resume"
    assert jax.tree.structure(resumed) == jax.tree.structure(whole)
    for left, right in zip(jax.tree.leaves(resumed), jax.tree.leaves(whole), strict=True):
        actual, expected = np.asarray(raw_leaf(left)), np.asarray(raw_leaf(right))
        assert (actual.dtype, actual.shape, actual.tobytes()) == (
            expected.dtype,
            expected.shape,
            expected.tobytes(),
        )
    _, resumed_place = Checkpoints(str(tmp_path / "split/run")).restore(share=DataPartition())
    _, whole_place = Checkpoints(str(tmp_path / "whole/run")).restore(share=DataPartition())
    assert resumed_place == whole_place
    assert position.read(resumed_place).records == 4 * BATCH


@pytest.mark.gpu(reason="CUDA embedding-gradient reductions")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_a_cuda_lm_repeats_and_resumes_bit_exactly_with_deterministic_ops(tmp_path, dtype):
    """Repeated token IDs share embedding-gradient updates. CUDA's default
    scatter-add order is not repeatable; conftest enables deterministic ops
    before the backend opens. Check every state leaf, not just parameters.
    bf16 at the default precision runs the head that rounds its logits and
    their gradient to bf16, and resumes bit-exactly too."""
    from dew.data import Loading, presets
    from dew.data.sources.tokens import TokenFiles
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective

    corpus = write_token_corpus(tmp_path / "tokens", np.tile(np.arange(8), 256), tokenizer="symbols",
                                vocab_size=8)
    data = compile(TokenFiles(corpus.train).pipe(presets.token_windows, seq_len=16, batch=8).read(
        "memory").loading(Loading(workers=0)))
    model = CausalTransformer(vocab_size=8, emb_features=16, num_layers=1,
                              num_heads=2, mlp_features=32, max_seq_len=32, dtype=dtype)
    objective = LMObjective(model, seq_len=16, ema_decay=None)

    def trainer(checkpoints=None):
        return Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0),
                       checkpoints=checkpoints)

    baseline = trainer().fit(data, steps=3)
    repeated = trainer().fit(data, steps=3)
    split = trainer(Checkpoints(str(tmp_path / "checkpoints")))
    prefix = split.fit(data, steps=2, checkpoint_every=1)
    restarted = trainer(Checkpoints(str(tmp_path / "checkpoints")))
    restored, _, position = restarted.place()
    assert position is not None
    resumed = restarted.fit(data, steps=3, checkpoint_every=1)
    for actual, expected in ((restored, prefix), (repeated, baseline), (resumed, baseline)):
        assert jax.tree.structure(actual) == jax.tree.structure(expected)
        for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(np.asarray(raw_leaf(left)), np.asarray(raw_leaf(right)))


REPEATED_CONV_STEPS = """
import sys

import jax
import numpy as np
import optax

from dew.cache import enable_compilation_cache
from dew.diffusion import presets
from dew.inputs import Field, InputSpec
from dew.objectives.diffusion import DiffusionObjective
from dew.nn.backbones import SimpleDiT
from dew.sampling import Euler
from dew.training import Trainer

enable_compilation_cache(sys.argv[2])
model = SimpleDiT(patch_size=2, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2)
objective = DiffusionObjective(model, presets.Flow()(), InputSpec(Field("image", (16, 16, 3))), guidance=None,
                               solver=Euler(), steps=2, ema_decay=None)
trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0))
state = trainer.initial_state()
batch = {"image": np.random.default_rng(0).integers(0, 256, (32, 16, 16, 3)).astype(np.uint8)}
step = trainer.compile(state, batch)
for _ in range(2):
    state, *_ = step(state, batch)
leaves = jax.tree.leaves(jax.tree.map(lambda leaf: jax.random.key_data(leaf)
                                      if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
                                      else leaf, state))
np.savez(sys.argv[1], *[np.asarray(leaf) for leaf in leaves])
"""


@pytest.mark.gpu(reason="cuDNN and cuBLAS kernel choice")
def test_a_conv_model_trains_to_the_bit_again_after_another_run_measured_kernels_into_its_cache(tmp_path):
    """XLA chooses a convolution's cuDNN algorithm and a GEMM's cuBLAS one per
    process, so only a second process shows whether it agrees. Two fresh
    processes train a DiT (a convolution embeds its patches) two Adam steps
    under the cuda lane's flags, with a run between them that measures its
    kernels, as XLA does by default, into their cache; every state leaf of
    the second agrees with the first's. JAX's executable cache is off, so
    each process compiles."""
    from lane_environment import REPEATABLE_GPU_FLAGS

    root = Path(__file__).resolve().parents[1]
    # The lane's flags and the picks settings conftest and gpu-run named stay out of the three runs.
    named = (*(flag.split("=")[0] for flag in REPEATABLE_GPU_FLAGS),
             "--xla_gpu_per_fusion_autotune_cache_dir", "--xla_gpu_experimental_autotune_cache_mode")
    flags = [flag for flag in os.environ.get("XLA_FLAGS", "").split() if not flag.startswith(named)]
    environment = {**os.environ, "JAX_PLATFORMS": "cuda", "PYTHONPATH": str(root / "src"),
                   "JAX_ENABLE_COMPILATION_CACHE": "false"}
    lane = {**environment, "XLA_FLAGS": " ".join([*flags, *REPEATABLE_GPU_FLAGS])}
    measuring = {**environment, "XLA_FLAGS": " ".join(flags)}
    cache = tmp_path / "cache"

    def trained(name: str, env: dict[str, str]):
        out = tmp_path / f"{name}.npz"
        done = subprocess.run([sys.executable, "-c", REPEATED_CONV_STEPS, str(out), str(cache)], env=env,
                              capture_output=True, text=True, timeout=600)
        assert done.returncode == 0, done.stderr
        return np.load(out)

    first = trained("first", lane)
    trained("measuring", measuring)
    assert any((cache / "autotune").rglob("*.pb"))
    again = trained("again", lane)
    assert first.files == again.files
    for name in first.files:
        np.testing.assert_array_equal(first[name], again[name])


RATE_RESUMES = """
import json
import shutil
import sys

import jax
import numpy as np
from affine_run import Regression, dataset
from recording import RecordingTracker

from dew.cache import enable_compilation_cache
from dew.optim import OptimConfig
from dew.telemetry.instrumentation import compilation_cache_counts
from dew.training import Checkpoints, Layout, Trainer
from dew.optim import Cosine, ParamGroup

root, grouped = sys.argv[1], sys.argv[2] == "grouped"
enable_compilation_cache(root + "/cache")


def fit(peak, steps, path):
    groups = (ParamGroup("kernel", ("*kernel",), learning_rate_multiplier=0.5), ParamGroup("rest", ("*",)))
    config = OptimConfig(learning_rate=Cosine(peak_value=peak, warmup_steps=1, decay_steps=8),
                         param_groups=groups if grouped else ())
    sink = RecordingTracker()
    trainer = Trainer(Regression(), config, key=jax.random.key(0), layout=Layout(min_shard=1, tolerance=1.0),
                      checkpoints=Checkpoints(path), tracker=sink)
    before = compilation_cache_counts()
    state = trainer.fit(dataset(), steps=steps, checkpoint_every=2, log_every=1)
    after = compilation_cache_counts()
    return {"params": [np.asarray(leaf).tolist() for leaf in jax.tree.leaves(state.variables["params"])],
            "rates": {step: row["train/learning_rate"] for step, row in sink.scalars
                      if "train/learning_rate" in row},
            **{name: after.get(f"train/cache_{name}", 0) - before.get(f"train/cache_{name}", 0)
               for name in ("hits", "misses")}}


runs = {"first": fit(1e-3, 2, root + "/run")}
for name, peak in (("slow", 4e-3), ("fast", 8e-3)):
    shutil.copytree(root + "/run", f"{root}/{name}")
    runs[name] = fit(peak, 3, f"{root}/{name}")
print(json.dumps(runs))
"""


@pytest.mark.parametrize("grouped", [False, True], ids=["one", "groups"])
def test_a_run_resumed_at_another_rate_steps_at_it_on_the_compiled_step(tmp_path, grouped):
    """A run trained at one peak rate, resumed from its checkpoint at a
    second and then a third: each resumed step moves the parameters by its
    configured rate, not the checkpoint's, and the third compiles nothing:
    the rate is an operand of the optimizer's state, so the step it runs is
    the executable the persistent cache holds. The second resume pays for the
    restore's own programs, which the first run never compiled. In a fresh
    process, with a cache of its own."""
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "tests")])}
    done = subprocess.run(
        [sys.executable, "-c", RATE_RESUMES, str(tmp_path), "grouped" if grouped else "one"],
        env=environment, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    runs = json.loads(done.stdout.splitlines()[-1])
    assert runs["fast"]["misses"] == 0 and runs["fast"]["hits"] >= 1
    for start, slow, fast in zip(*(runs[name]["params"] for name in ("first", "slow", "fast")), strict=True):
        start, slow, fast = (np.asarray(leaf, np.float32) for leaf in (start, slow, fast))
        assert np.all(slow != start)
        # Each step's parameters round to float32, so the moves differ by up to their spacing.
        np.testing.assert_allclose(fast - start, 2 * (slow - start), rtol=1e-5,
                                   atol=2 * np.spacing(np.abs(start).max()))
    if not grouped:
        for name, peak in (("slow", 4e-3), ("fast", 8e-3)):
            schedule = optax.warmup_cosine_decay_schedule(0.0, peak, 1, 8)
            assert runs[name]["rates"]["3"] == pytest.approx(float(schedule(2)))


def test_the_compiled_step_consumes_the_state_it_is_given():
    """`new = step(old, batch)` runs the update in place: the old state's
    buffers move into the new one, so peak memory holds one copy of the
    parameters and optimizer state. Reading the old state afterwards is
    the caller's error, and JAX names it."""
    trainer = make_trainer()
    state, _, _ = trainer.place()
    batch = next(Counting())
    step = trainer.compile(state, batch)
    stale = jax.tree.leaves(state.variables)[0]
    advanced, _loss, _, finite, _, _ = step(state, batch)
    assert bool(finite) and int(advanced.step) == 1
    assert stale.is_deleted()
    again, _, _, _, _, _ = step(advanced, batch)
    assert int(again.step) == 2


def test_restore_preserves_the_optimizer_state_the_ema_and_the_key(tmp_path):
    trainer = make_trainer(tmp_path, optimizer=optax.adam(1e-3))
    trained = trainer.fit(dataset(), steps=3, log_every=1)

    resumed = make_trainer(tmp_path, optimizer=optax.adam(1e-3))
    state, _, position = resumed.place()

    assert int(state.step) == 3, "the step counter was reset"
    for field in ("variables", "opt_state", "ema"):
        for before, after in zip(jax.tree.leaves(getattr(trained, field)),
                                 jax.tree.leaves(getattr(state, field)), strict=True):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after), err_msg=field)
    assert jnp.array_equal(jax.random.key_data(trained.key), jax.random.key_data(state.key))
    assert json.loads(position)["index"] == 3


def test_a_checkpoint_of_the_retired_rate_scale_and_shape_is_refused(tmp_path):
    """Before 1.0 a configured rate was held as a scale times a schedule's shape, under
    `learning_rate_scale` and `learning_rate_shape` in the optimizer's state, where Dew now holds the
    schedule's own fields (`dew.optim.RateState`). Resuming such a checkpoint is refused (ValueError),
    even by a run whose state has the same tree, rather than stepping at a rate it no longer forms."""
    def factory(learning_rate_scale, learning_rate_shape):
        return optax.sgd(learning_rate_scale * learning_rate_shape)

    retired = optax.inject_hyperparams(factory)(learning_rate_scale=0.1, learning_rate_shape=1.0)
    make_trainer(tmp_path, optimizer=retired).fit(dataset(), steps=1, log_every=1)
    with pytest.raises(ValueError):
        make_trainer(tmp_path, optimizer=retired).place()


def test_a_param_group_named_as_the_retired_rate_resumes(tmp_path):
    """The retired rate is refused by its structure, a scale and a shape side by side as an injected
    state's scalars: a run of today's optimizer whose param group takes one of those names writes a
    checkpoint its resume reads back."""
    from dew.optim import ParamGroup

    config = OptimConfig(optimizer="adam", learning_rate=0.1, clip_grads=0,
                         param_groups=(ParamGroup("learning_rate_scale", ("*",)),))
    make_trainer(tmp_path, optimizer=config).fit(dataset(), steps=1, log_every=1)
    # An OptimConfig is built over the run's length, which the resumed fit gives.
    resumed = make_trainer(tmp_path, optimizer=config).fit(dataset(), steps=2, log_every=1)
    assert int(resumed.step) == 2


def test_a_model_whose_parameters_are_named_as_the_retired_rate_resumes(tmp_path):
    """A model may name two scalar parameters as the retired rate's scale and shape: Adam's moments then
    hold the pair side by side, which is not an injected state's hyperparameters, so the checkpoint its
    run writes resumes."""
    from dew.checkpoints import RETIRED_RATE as RETIRED_NAMES

    class Named(Regression):
        def fresh_variables(self, key, held):
            return {"params": {name: jnp.zeros(()) for name in RETIRED_NAMES}}

        def loss(self, variables, batch, step):
            weights = variables["params"]
            return jnp.mean((batch["x"][:, 0] * weights["learning_rate_scale"]
                             + weights["learning_rate_shape"] - batch["y"][:, 0]) ** 2), Aux({})

    def run(steps):
        return make_trainer(tmp_path, objective=Named(), optimizer=optax.adam(0.1)).fit(
            dataset(), steps=steps, log_every=1)

    run(1)
    resumed = run(2)
    assert int(resumed.step) == 2


def _rewrite_position(trainer, step, rows, shares):
    """Save `step` again with `rows` as the checkpoint's position table, each
    row read for the share `shares` names."""
    restored, _ = trainer.checkpoints.restore()
    written = [np.frombuffer(row, np.uint8) for row in rows]
    table = {"rows": np.stack(written),
             "lengths": np.array([len(row) for row in written], np.int64),
             "shares": np.array(shares, np.int64)}
    manager = trainer.checkpoints._open()
    manager.save(step, args=ocp.args.PyTreeSave({**restored, "position": table}), force=True)
    manager.wait_until_finished()


def test_a_global_position_is_read_by_any_process_count(tmp_path):
    """A global position is a record count over an order every process count
    reads the same way, so the table two processes wrote is this one
    process's position as well. Only a shard offset ties a resume to the
    count that wrote it."""
    trainer = make_trainer(tmp_path)
    trainer.fit(dataset(), steps=1, log_every=1)
    global_position = position.encode(position.Global(records=16, order="Counting"))
    _rewrite_position(trainer, 2, [global_position, global_position], shares=[[0, 2], [1, 2]])

    assert Checkpoints(str(tmp_path / "run")).restore(step=2, share=DataPartition())[1] == global_position


def held_state(params, ema):
    """A train state that holds `params` and `ema` and scalars beside them."""
    count = jnp.zeros((), jnp.int32)
    return TrainState(step=count, microstep=count, updates=count, variables=params,
                      opt_state={"count": count}, ema=ema,
                      key=jax.random.key_data(jax.random.key(0)), scale=None,
                      window_size=jnp.ones((), jnp.int32), accumulation=None)


def bit_patterns(shape, dtype, seed):
    """Values drawn bit by bit, so NaN payloads and subnormals turn up, with
    both zeros, a NaN of every payload bit set and the smallest subnormal first."""
    kind = np.dtype(f"u{np.dtype(dtype).itemsize}")
    bits = np.random.default_rng(seed).integers(0, np.iinfo(kind).max, size=shape,
                                                dtype=kind, endpoint=True)
    bits.flat[:4] = [1 << (8 * kind.itemsize - 1), 0, np.iinfo(kind).max, 1]
    return bits.view(dtype)


def same_bits(expected, actual):
    expected, actual = np.asarray(expected), np.asarray(actual)
    return (expected.dtype == actual.dtype and expected.shape == actual.shape
            and np.array_equal(expected.view(np.uint8), actual.view(np.uint8)))


def awkward_state():
    """Weights in fp32 and bf16, and an EMA of each dtype, one of them
    narrower than the weight it follows."""
    params = {"params": {"kernel": bit_patterns((16, 8), np.float32, 0),
                         "scale": bit_patterns((8,), jnp.bfloat16, 1),
                         "bias": bit_patterns((8,), np.float32, 2)}}
    ema = {"params": {"kernel": bit_patterns((16, 8), np.float32, 3),
                      "scale": bit_patterns((8,), jnp.bfloat16, 4),
                      "bias": bit_patterns((8,), jnp.bfloat16, 5)}}
    return held_state(jax.tree.map(jnp.asarray, params), jax.tree.map(jnp.asarray, ema))


def test_the_ema_comes_back_bit_for_bit_however_it_is_read(tmp_path):
    """Every NaN payload, signed zero and subnormal of the EMA survives a save,
    read whole, typed, without the weights beside it, into pinned host memory
    and as the shapes the checkpoint reports."""
    state = awkward_state()
    checkpoints = Checkpoints(str(tmp_path / "run"))
    checkpoints.save(0, state, None)
    checkpoints.wait()
    checkpoints = Checkpoints(str(tmp_path / "run"))

    def typed(tree, memory_kind="device"):
        where = jax.sharding.SingleDeviceSharding(jax.devices()[0], memory_kind=memory_kind)
        return jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=where),
                            tree)

    whole, _ = checkpoints.restore()
    ema_alone, _ = checkpoints.restore({"ema": typed(state.ema)})
    pinned, _ = checkpoints.restore({"variables": typed(state.variables),
                                     "ema": typed(state.ema, "pinned_host")})
    resumed, _ = checkpoints.restore(held_state(typed(state.variables), typed(state.ema)))
    for restored in (whole["ema"], ema_alone["ema"], pinned["ema"], resumed.ema):
        assert jax.tree.all(jax.tree.map(same_bits, state.ema, restored))
    assert jax.tree.all(jax.tree.map(same_bits, state.variables, resumed.variables))
    assert all(leaf.sharding.memory_kind == "pinned_host" for leaf in jax.tree.leaves(pinned["ema"]))
    assert jax.tree.map(lambda leaf: (leaf.shape, leaf.dtype), checkpoints.stored()["ema"]) == \
        jax.tree.map(lambda leaf: (leaf.shape, leaf.dtype), state.ema)


def test_an_ema_held_in_lists_comes_back_bit_for_bit(tmp_path):
    """Variables may nest sequences; the EMA's leaves inside them are found by
    their key paths, read whole, alone, and beside weights asked for in
    another dtype, which sends the weights they undo to a read of their own."""
    kernels = [bit_patterns((4, 3), np.float32, seed) for seed in (0, 1)]
    params = {"params": {"layers": [{"w": jnp.asarray(kernel)} for kernel in kernels]}}
    ema = {"params": {"layers": [{"w": jnp.asarray(bit_patterns((4, 3), np.float32, seed))}
                                 for seed in (2, 3)]}}
    checkpoints = Checkpoints(str(tmp_path / "run"))
    checkpoints.save(0, held_state(params, ema), None)
    checkpoints.wait()
    checkpoints = Checkpoints(str(tmp_path / "run"))
    where = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def typed(tree, dtype=None):
        return jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, dtype or leaf.dtype,
                                                              sharding=where), tree)

    whole, _ = checkpoints.restore()
    alone, _ = checkpoints.restore({"ema": typed(ema)})
    beside, _ = checkpoints.restore({"variables": typed(params, jnp.bfloat16), "ema": typed(ema)})
    for restored in (whole["ema"], alone["ema"], beside["ema"]):
        assert jax.tree.all(jax.tree.map(same_bits, ema, restored))


def test_a_checkpoint_that_stores_the_ema_as_itself_still_restores(tmp_path):
    """Checkpoints written before the EMA was stored as its difference from
    the weights hold it as plain arrays, and read back as they were written."""
    state = awkward_state()
    written = ocp.CheckpointManager(str(tmp_path / "run"), item_handlers=ocp.PyTreeCheckpointHandler())
    written.save(0, args=ocp.args.PyTreeSave({name: getattr(state, name) for name in CHECKPOINTED}))
    written.wait_until_finished()

    checkpoints = Checkpoints(str(tmp_path / "run"))
    template = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype,
                                                              sharding=leaf.sharding), state.ema)
    for restored in (checkpoints.restore()[0]["ema"], checkpoints.restore({"ema": template})[0]["ema"]):
        assert jax.tree.all(jax.tree.map(same_bits, state.ema, restored))


def test_the_best_step_is_the_lowest_loss(tmp_path):
    """The loss rides along with the save, and the lowest one is what orbax
    keeps and reports however the run wanders; a metric-less save is newer
    than the best and must not displace it."""
    checkpoints = Checkpoints(str(tmp_path / "best"), keep=1)
    state = make_trainer().initial_state()
    for step, loss in ((1, 0.9), (2, 0.3), (3, 0.7)):
        checkpoints.save(step, state.replace(step=jnp.asarray(step)), None,
                         ranking=Ranking("train/loss", loss))
    checkpoints.wait()

    assert checkpoints.best == 2
    assert set(checkpoints._open().all_steps()) == {2, 3}

    checkpoints.save(4, state.replace(step=jnp.asarray(4)), None)
    checkpoints.wait()
    assert checkpoints.best == 2
    assert set(checkpoints._open().all_steps()) == {2, 4}
    assert Checkpoints(str(tmp_path / "best")).best == 2, "the metric did not survive a reopen"


# --------------------------------------------------------------------------
# Local checkpoints
# --------------------------------------------------------------------------

def local_trainer(tmp_path, **kwargs):
    return Trainer(
        Regression(), optax.sgd(0.1), key=jax.random.key(0),
        layout=Layout(min_shard=1, tolerance=1.0),
        checkpoints=Checkpoints(str(tmp_path / "run"), keep=4,
                                local_directory=str(tmp_path / "local"), local_every=2),
        **kwargs)


def test_terminal_local_cadence_does_not_need_a_final_persistent_save(tmp_path):
    trainer = local_trainer(tmp_path)
    trainer.fit(dataset(), steps=2, checkpoint_final=False)
    restored, _, _ = local_trainer(tmp_path).place()
    assert int(restored.step) == 2
    assert not os.path.exists(tmp_path / "run" / "2")


def test_the_local_checkpoint_is_written_on_its_own_cadence_and_keeps_the_newest(tmp_path):
    """Local every two, persistent every three: after eight steps the
    persistent directory holds 3, 6 and the final 8 with their losses, the
    local one holds 8 alone (2, 4 and 6 replaced), and the persistent directory holds nothing of the
    local cadence."""
    trainer = local_trainer(tmp_path)
    trainer.fit(dataset(), steps=8, log_every=1, checkpoint_every=3)
    checkpoints = trainer.checkpoints

    assert sorted(checkpoints._open().all_steps()) == [3, 6, 8]
    assert sorted(checkpoints._open_local().all_steps()) == [8]
    assert checkpoints.best in (3, 6, 8)
    assert checkpoints.local_path == str(tmp_path / "local" / "process0")
    assert (tmp_path / "local" / "process0" / "8" / "commit_success.txt").exists()
    assert not (tmp_path / "local" / "process0" / "4").exists()
    assert not (tmp_path / "run" / "4").exists()


class ExplodingManager:
    """Orbax when the filesystem refuses the write.

    Stubbed, not provoked with a read-only directory: a real failed async
    orbax write leaves a background thread that never joins, which hangs
    interpreter exit and with it the whole test session.
    """

    def latest_step(self):
        return None

    def all_steps(self):
        return []

    def save(self, *args, **kwargs):
        raise OSError("No space left on device")

    def wait_until_finished(self):
        pass


def test_a_checkpoint_that_does_not_land_fails_the_run(tmp_path):
    """A checkpoint that did not get written is data loss, not a log line."""
    trainer = make_trainer(tmp_path)
    trainer.checkpoints._manager = ExplodingManager()
    with pytest.raises(OSError):
        trainer.fit(dataset(), steps=1, log_every=1)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def test_eval_every_scores_the_validation_split_and_logs_the_artifacts():
    seen = []
    tracker = RecordingTracker()
    objective = Features()
    trainer = make_trainer(objective=objective, tracker=tracker)
    state = trainer.fit(dataset(val=val_batches(3)), steps=4, log_every=2, eval_every=2,
                        metrics=(Spread(seen),), preview=True)

    # The pass that warms the evaluations before the first step scores its first batches, and two
    # passes the whole split: at step 2, and at the end of the run.
    assert seen == [((BATCH, 2), (BATCH, FEATURES))] * (WARM_BATCHES + 6)
    scored = [(step, s) for step, s in tracker.scalars if "val/spread" in s]
    assert [step for step, _ in scored] == [2, 4]
    # The pass at the end scores the averaged weights over the whole split, so
    # its spread is the mean of the three batches' own.
    expected = np.mean([float(jnp.std(objective.model.apply(state.averaged, batch["x"])))
                        for batch in val_batches(3)()])
    assert scored[-1][1]["val/spread"] == pytest.approx(expected, rel=1e-6)
    # The first batch's artifact of each pass reaches the tracker.
    assert [step for step, value in tracker.artifacts if isinstance(value, Representations)] == [2, 4]


def test_each_suite_is_scored_on_its_own_cadence_with_its_own_metrics():
    """Two named suites: one every 2 steps with fit's metric, one every 3 with
    its own; each at its multiples and at the end, under its own name."""
    from dew.training import EvalSuite

    tracker = RecordingTracker()
    trainer = make_trainer(objective=Features(), tracker=tracker)
    trainer.fit(dataset(val=None), steps=6, log_every=2, eval_every=2, metrics=(Spread([]),),
                validation={"often": dataset(val=val_batches(1)).val,
                            "rare": EvalSuite(dataset(val=val_batches(2)).val, (Spread([]),), 3)})
    steps = {name: [step for step, scores in tracker.scalars if f"{name}/spread" in scores]
             for name in ("often", "rare")}
    assert steps == {"often": [2, 4, 6], "rare": [3, 6]}


@pytest.mark.parametrize(("written", "expected"), [(("25%", "50%", 2), ([2, 4, 6, 8], [4, 8])),
                                                   ((2, "50%", "25%"), ([2, 4, 6, 8], [4, 8]))])
def test_a_cadence_may_be_a_share_of_the_run(tmp_path, written, expected):
    """A log, validation or suite cadence written as a share of the run's planned steps is the whole
    steps nearest it, as the steps it resolves to are: 25% and 50% of 8 steps log every 2 and score
    every 4, and the run's start records what each resolved to."""
    from dew.training import EvalSuite

    log_every, eval_every, suite_every = written
    tracker = RecordingTracker()
    trainer = make_trainer(tmp_path, objective=Features(), tracker=tracker)
    trainer.fit(dataset(val=None), steps=8, log_every=log_every, eval_every=eval_every,
                metrics=(Spread([]),),
                validation={"often": dataset(val=val_batches(1)).val,
                            "own": EvalSuite(dataset(val=val_batches(1)).val, (Spread([]),), suite_every)})
    logged = sorted({step for step, scalars in tracker.scalars if "train/loss" in scalars})
    scored = sorted({step for step, scalars in tracker.scalars if "often/spread" in scalars})
    assert (logged, scored) == expected
    assert sorted({step for step, scalars in tracker.scalars if "own/spread" in scalars}) == [2, 4, 6, 8]


def test_the_run_start_records_the_cadences_and_warmups_it_resolved(tmp_path):
    """The run's start record holds each cadence and schedule warmup as the run resolved it, a share
    of the run among them: 25% of 8 steps logs every 2, and a warmup of 50% is 4 updates."""
    from dew.training import LocalTracker

    with LocalTracker(tmp_path / "tracking") as tracker:
        trainer = make_trainer(tmp_path, tracker=tracker,
                               optimizer=OptimConfig(learning_rate=Cosine(1e-2, warmup_steps="50%")))
        trainer.fit(dataset(), steps=8, log_every="25%", checkpoint_every="50%")
    written = (tmp_path / "tracking" / "records.jsonl").read_text().splitlines()
    started = next(record for record in map(json.loads, written) if record["type"] == "FitStarted")
    assert started["value"]["cadences"] == {"log_every": 2, "eval_every": None, "checkpoint_every": 4,
                                            "warmup_steps.learning_rate": 4}


def test_a_local_checkpoint_cadence_may_be_a_share_of_the_run(tmp_path):
    """`Checkpoints.local_every` may be a share of the run, resolved as `fit`'s other cadences are: 25%
    of 8 steps writes the local checkpoint every 2, and the run's start records it."""
    from dew.telemetry.records import CheckpointRequested, FitStarted

    tracker = RecordingTracker()
    checkpoints = Checkpoints(str(tmp_path / "run"), local_directory=str(tmp_path / "local"),
                              local_every="25%")
    trainer = Trainer(Regression(), optax.sgd(0.1), key=jax.random.key(0), tracker=tracker,
                      layout=Layout(min_shard=1, tolerance=1.0), checkpoints=checkpoints)
    trainer.fit(dataset(), steps=8, log_every=4)
    local = [step for step, value in tracker.artifacts
             if isinstance(value, CheckpointRequested) and value.local]
    assert local == [2, 4, 6, 8]
    started = next(value for _, value in tracker.artifacts if isinstance(value, FitStarted))
    assert started.cadences["local_every"] == 2
    with pytest.raises(ValueError):
        Checkpoints(str(tmp_path / "other"), local_directory=str(tmp_path / "local"), local_every="25")


def test_a_share_of_a_token_budget_falls_where_the_committed_tokens_pass_its_multiples():
    """A cadence written as a share of a token budget alone falls at each step whose commit passes a
    multiple of that share of the budget's tokens: 25% of 96 logs at the steps that bring the
    committed tokens past 24, 48, 72 and 96, as a run logging every step counts them."""
    every = RecordingTracker()
    make_trainer(objective=TokenRegression(), tracker=every).fit(dataset(), tokens=96, log_every=1)
    # A log every step reports the tokens that step committed; the run's are their running sum.
    each = sorted((step, scalars["train/tokens"]) for step, scalars in every.scalars
                  if "train/tokens" in scalars)
    committed = dict(zip((step for step, _ in each), itertools.accumulate(tokens for _, tokens in each),
                         strict=True))
    passing = sorted(step for step, tokens in committed.items()
                     if tokens // 24 > committed.get(step - 1, 0) // 24)
    shared = RecordingTracker()
    make_trainer(objective=TokenRegression(), tracker=shared).fit(dataset(), tokens=96, log_every="25%")
    logged = sorted({step for step, scalars in shared.scalars if "train/tokens" in scalars})
    assert passing and logged == passing


@pytest.mark.parametrize("written", ["5", "0%", "150%", "five%"])
def test_a_cadence_that_is_neither_steps_nor_a_share_is_refused(written):
    with pytest.raises(ValueError):
        make_trainer().fit(dataset(), steps=4, log_every=written)


def test_a_share_of_a_run_with_no_planned_length_is_refused():
    """A run bounded by nothing but its data's end has no planned length to take a share of."""
    with pytest.raises(ValueError):
        make_trainer().fit(stream(lambda partition: endless()), eval_every="5%")


def test_metrics_without_a_cadence_are_scored_once_at_the_end():
    """A metric costly enough to want once is scored by the pass every suite gets at the run's end
    alone, where neither fit's eval_every nor the suite's every schedules another. Metrics with no
    validation data to score are refused before the run, which would otherwise never score them."""
    from dew.training import EvalSuite

    held = dataset(val=val_batches(2)).val
    for options in ({"metrics": (Spread([]),)}, {"validation": {"val": EvalSuite(held, (Spread([]),))}}):
        tracker = RecordingTracker()
        make_trainer(objective=Features(), tracker=tracker).fit(
            dataset(val=val_batches(2)), steps=4, log_every=2, **options)
        assert [step for step, scores in tracker.scalars if "val/spread" in scores] == [4]
    with pytest.raises(ValueError):
        make_trainer(objective=Features()).fit(dataset(val=None), steps=4, log_every=2, metrics=(Spread([]),))


def test_the_validation_split_is_the_one_suite_val():
    """`dataset.val` alone and a one-suite mapping of it are the same run."""
    from dew.training import EvalSuite

    runs = []
    held = dataset(val=val_batches(3)).val
    for validation in (None, {"val": held}, {"val": EvalSuite(held, (Spread([]),), 3)}):
        tracker = RecordingTracker()
        make_trainer(objective=Features(), tracker=tracker).fit(
            dataset(val=val_batches(3)), steps=6, log_every=2, eval_every=3, metrics=(Spread([]),),
            validation=validation)
        runs.append([(step, scores["val/spread"]) for step, scores in tracker.scalars
                     if "val/spread" in scores])
    assert runs[0] == runs[1] == runs[2] and [step for step, _ in runs[0]] == [3, 6]


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def test_the_log_tick_carries_the_loss_the_objective_metrics_and_the_throughput():
    tracker = RecordingTracker()
    make_trainer(tracker=tracker).fit(dataset(endless), steps=4, log_every=2)

    ticks = [(step, scalars) for step, scalars in tracker.scalars if "train/loss" in scalars]
    assert [step for step, _ in ticks] == [2, 4]
    for _, scalars in ticks:
        assert scalars["train/probe"] == 1.0
        assert np.isfinite(scalars["train/loss"])
        assert scalars["train/step_time_ms"] > 0
        assert scalars["train/samples_per_sec"] > 0


# --------------------------------------------------------------------------
# Divergence
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Rejected mixed-precision steps
# --------------------------------------------------------------------------

class ScaledObjective(Objective):
    """loss = scale * sum(w^2), with the scale carried by the batch so that
    one batch overflows the scaled float32 loss while the params stay sane."""

    def __init__(self):
        self.ema = EMASpec(decay=lambda step: 0.5)

    def fresh_variables(self, key, held):
        return {"params": {"w": jnp.ones((2,))}}

    def loss(self, variables, batch, step):
        return jnp.sum(variables["params"]["w"] ** 2) * batch["scale"][0], Aux({})


def host(state):
    return np.array(state.variables["params"]["w"]), np.array(state.ema["params"]["w"])


def unscaled(accum: int, microsteps: int):
    """The weights and their average after `microsteps` good microbatches of the same trainer with no
    dynamic scale: what a scaled run's accepted steps must equal bit for bit, its scale a power of two
    that multiplies the loss and divides the gradients exactly."""
    trainer = Trainer(ScaledObjective(), optax.sgd(.1), key=jax.random.key(0), accumulation=accum)
    state = trainer.initial_state()
    good = {"scale": jnp.ones((jax.device_count(),), jnp.float32)}
    step = trainer.compile(state, good)
    for _ in range(microsteps):
        state, *_ = step(state, good)
    return host(state)


@pytest.mark.parametrize("accum", [1, 2])
def test_a_rejected_dynamic_scale_step_leaves_no_trace(accum):
    """A dynamic scale's accepted steps are the unscaled run's, bit for bit, and a rejected one leaves
    no trace: after it, the run steps on as the unscaled run of its accepted microbatches alone."""
    trainer = Trainer(ScaledObjective(), optax.sgd(.1), key=jax.random.key(0),
                      accumulation=accum, dynamic_scale=True)
    state = trainer.initial_state()
    good = {"scale": jnp.ones((jax.device_count(),), jnp.float32)}
    bad = {"scale": jnp.full((jax.device_count(),), 1e35, jnp.float32)}
    step = trainer.compile(state, good)
    for _ in range(2 * accum - 1):
        state, *_ = step(state, good)
    w, ema = host(state)
    assert np.all(w != 1.0)
    for got, want in zip((w, ema), unscaled(accum, 2 * accum - 1), strict=True):
        np.testing.assert_array_equal(got, want)
    # the step consumes the state
    counted, advanced, scaled = int(state.step), int(state.microstep), float(state.scale.scale)
    state, _, _, finite, accepted, _ = step(state, bad)
    assert bool(finite) and not bool(accepted)
    assert int(state.step) == counted + 1
    # The rejection drops its window, the accepted microbatch before it with it (`dropped_window`).
    assert int(state.microstep) == advanced - advanced % accum
    assert float(state.scale.scale) == scaled / 2
    np.testing.assert_array_equal(state.variables["params"]["w"], w)
    np.testing.assert_array_equal(state.ema["params"]["w"], ema)
    for _ in range(accum):
        state, *_ = step(state, good)
    w, ema = host(state)
    # Two accepted updates: the rejection dropped its window, the pending microbatch with it.
    for got, want in zip((w, ema), unscaled(accum, 2 * accum), strict=True):
        np.testing.assert_array_equal(got, want)
    assert int(state.microstep) == 2 * accum


# --------------------------------------------------------------------------
# The collections an objective writes back
# --------------------------------------------------------------------------

class Counted(Objective):
    """A `stats` collection the loss advances by one every step."""

    ema = None

    def fresh_variables(self, key, held):
        return {"params": {"w": jnp.ones((2,))}, "stats": {"seen": jnp.zeros(())}}

    def loss(self, variables, batch, step):
        loss = jnp.sum(variables["params"]["w"] ** 2)
        return loss, Aux({}, variables={"stats": {"seen": variables["stats"]["seen"] + 1}})


class FrozenCounted(Counted):
    """`Counted`, its collection written back in a Flax `FrozenDict`, as Flax's `apply` writes one
    under `flax_return_frozendict`."""

    def loss(self, variables, batch, step):
        loss = jnp.sum(variables["params"]["w"] ** 2)
        return loss, Aux({}, variables=frozen_dict({"stats": {"seen": variables["stats"]["seen"] + 1}}))


class ListedCounted(Counted):
    """`Counted`, its count held in a list, which the loss writes back as a Flax `FrozenDict`."""

    def fresh_variables(self, key, held):
        return {"params": {"w": jnp.ones((2,))}, "stats": {"seen": [{"count": jnp.zeros(())}]}}

    def loss(self, variables, batch, step):
        loss = jnp.sum(variables["params"]["w"] ** 2)
        count = variables["stats"]["seen"][0]["count"] + 1
        return loss, Aux({}, variables={"stats": {"seen": [frozen_dict({"count": count})]}})


class FrozenListedCounted(Counted):
    """`ListedCounted`, its count starting in a Flax `FrozenDict` in its list, which the loss writes back
    as a dict."""

    def fresh_variables(self, key, held):
        return {"params": {"w": jnp.ones((2,))}, "stats": {"seen": [frozen_dict({"count": jnp.zeros(())})]}}

    def loss(self, variables, batch, step):
        loss = jnp.sum(variables["params"]["w"] ** 2)
        count = variables["stats"]["seen"][0]["count"] + 1
        return loss, Aux({}, variables={"stats": {"seen": [{"count": count}]}})


def seen(stats) -> float:
    """How many steps a `Counted` collection counted, held as `Counted` or `ListedCounted` holds it."""
    held = stats["seen"]
    return float(held[0]["count"] if isinstance(held, list) else held)


@pytest.mark.parametrize("objective", [Counted, FrozenCounted, ListedCounted, FrozenListedCounted],
                         ids=["dict", "frozen", "listed", "frozen-listed"])
def test_aux_variables_are_written_back_into_the_state_and_checkpointed(tmp_path, objective):
    """A collection the loss writes back replaces the state's, in the state's dicts whichever Flax
    container it comes in, under a mapping or a list."""
    trainer = make_trainer(tmp_path, objective=objective(), optimizer=optax.adamw(1e-2, weight_decay=1.0))
    state = trainer.fit(dataset(), steps=3, log_every=1)
    nodes = jax.tree.leaves(state.variables, is_leaf=lambda node: isinstance(node, FrozenDict))
    assert not any(isinstance(node, FrozenDict) for node in nodes)
    assert seen(state.variables["stats"]) == 3.0
    assert set(state.opt_state[0].mu) == {"w"}

    restored, _ = trainer.checkpoints.restore()
    assert seen(restored["variables"]["stats"]) == 3.0


class ListedParameters(Objective):
    """Parameters held in a list, as a stack of layers can be."""

    ema = None

    def fresh_variables(self, key, held):
        return {"params": {"layers": [{"w": jnp.ones((2,))}]}}

    def loss(self, variables, batch, step):
        return jnp.sum(variables["params"]["layers"][0]["w"] ** 2), Aux({})


MAPPINGS = {"dict": dict, "frozen": frozen_dict, "ordered": collections.OrderedDict}
"""The kinds of mapping a caller's tree may hold its collections in."""


def held_count(kind: str, container: str, count) -> dict:
    """A `stats` collection of `count`, `{"count": count}` in a mapping of `kind`, as `container` holds
    it: the mapping itself, or in a list or a tuple."""
    node = MAPPINGS[kind]({"count": count})
    return {"seen": node if container == "mapping" else [node] if container == "list" else (node,)}


def counted(stats):
    """The count `held_count` holds."""
    seen = stats["seen"]
    return (seen if isinstance(seen, Mapping) else seen[0])["count"]


class Kinded(Counted):
    """`Counted`, its count starting in mappings of `started`'s kind and written back in `written`'s,
    each held as `container` holds it (`held_count`)."""

    def __init__(self, started: str, written: str, container: str):
        self.started, self.written, self.container = started, written, container

    def fresh_variables(self, key, held):
        stats = held_count(self.started, self.container, jnp.zeros(()))
        return {"params": {"w": jnp.ones((2,))}, "stats": stats}

    def loss(self, variables, batch, step):
        loss = jnp.sum(variables["params"]["w"] ** 2)
        stats = held_count(self.written, self.container, counted(variables["stats"]) + 1)
        return loss, Aux({}, variables={"stats": stats})


@pytest.mark.parametrize("container", ["mapping", "list", "tuple"])
@pytest.mark.parametrize("kind", list(MAPPINGS))
@pytest.mark.parametrize("entry", ["init", "supplied", "write-back"])
def test_a_tree_of_any_kind_of_mapping_is_held_as_dicts_where_it_enters(tmp_path, entry, kind, container):
    """A tree a run is handed holds its collections in any kind of mapping (a dict, a Flax
    `FrozenDict`, an `OrderedDict`), under a list or a tuple too: where the objective's init returns
    it, where a supplied state holds it and where the loss writes it back, the state holds it as
    dicts, so every step's tree has one structure and the run counts every step."""
    objective = Kinded(kind if entry == "init" else "dict", kind if entry == "write-back" else "dict",
                       container)
    trainer = make_trainer(tmp_path, objective=objective, optimizer=optax.sgd(1e-2))
    supplied = None
    if entry == "supplied":
        initial = trainer.initial_state()
        stats = held_count(kind, container, counted(initial.variables["stats"]))
        supplied = dataclasses.replace(initial, variables={**initial.variables, "stats": stats})
    state = trainer.fit(dataset(), steps=3, log_every=1, state=supplied)
    others = [node for node in jax.tree.leaves(
        state, is_leaf=lambda node: isinstance(node, Mapping) and type(node) is not dict)
        if isinstance(node, Mapping)]
    assert not others and float(counted(state.variables["stats"])) == 3.0


@jax.tree_util.register_pytree_node_class
class MomentumRecord(Mapping):
    """An optimizer's state that is a mapping of its own kind, read by attribute: its momentum."""

    def __init__(self, momentum):
        self.momentum = momentum

    def __getitem__(self, key):
        return {"momentum": self.momentum}[key]

    def __iter__(self):
        return iter(("momentum",))

    def __len__(self):
        return 1

    def tree_flatten(self):
        return (self.momentum,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def recorded_momentum() -> optax.GradientTransformation:
    """Momentum kept in a `MomentumRecord`."""
    def init(params):
        return MomentumRecord(jax.tree.map(jnp.zeros_like, params))

    def update(updates, state, params=None):
        momentum = jax.tree.map(lambda held, gradient: 0.9 * held + gradient, state.momentum, updates)
        return jax.tree.map(lambda held: -1e-2 * held, momentum), MomentumRecord(momentum)

    return optax.GradientTransformation(init, update)


def test_a_supplied_state_keeps_an_optimizer_record_that_is_a_mapping_of_its_own_kind(tmp_path):
    """Only container mappings become dicts where a state enters: an optimizer's record that is a
    mapping of its own kind stays one, its fields read as its update reads them."""
    trainer = make_trainer(tmp_path, objective=Counted(), optimizer=recorded_momentum())
    state = trainer.fit(dataset(), steps=2, log_every=1, state=trainer.initial_state())
    assert int(state.step) == 2 and isinstance(state.opt_state, MomentumRecord)


def test_a_supplied_state_whose_parameters_and_moments_hold_a_frozen_dict_in_a_list_trains(tmp_path):
    """A state handed to fit is held whole as dicts of its own: its variables under any mapping, and a
    Flax `FrozenDict` in a list both where a parameter sits and where the optimizer's moments of it do,
    so the step's gradients and moments share one structure."""
    trainer = make_trainer(tmp_path, objective=ListedParameters(), optimizer=optax.adam(1e-2))
    initial = trainer.initial_state()
    adam = initial.opt_state[0]
    frozen = {"layers": [frozen_dict(initial.variables["params"]["layers"][0])]}
    moments = adam._replace(mu={"layers": [frozen_dict(adam.mu["layers"][0])]},
                            nu={"layers": [frozen_dict(adam.nu["layers"][0])]})
    supplied = dataclasses.replace(initial, variables=collections.OrderedDict(params=frozen),
                                   opt_state=(moments, *initial.opt_state[1:]))
    state = trainer.fit(dataset(), steps=2, log_every=1, state=supplied)
    nodes = jax.tree.leaves(state, is_leaf=lambda node: isinstance(node, FrozenDict))
    assert int(state.step) == 2 and not any(isinstance(node, FrozenDict) for node in nodes)


def test_a_supplied_state_holding_a_frozen_dict_in_a_list_trains_as_its_dicts_do(tmp_path):
    """A state handed to fit holds its trees as dicts of its own, a Flax `FrozenDict` in a list
    included, so the collection its loss writes back keeps the state's structure."""
    trainer = make_trainer(tmp_path, objective=ListedCounted(), optimizer=optax.adamw(1e-2, weight_decay=1.0))
    initial = trainer.initial_state()
    stats = {"seen": [frozen_dict({"count": initial.variables["stats"]["seen"][0]["count"]})]}
    state = trainer.fit(dataset(), steps=3, log_every=1,
                        state=dataclasses.replace(initial, variables={**initial.variables, "stats": stats}))
    assert seen(state.variables["stats"]) == 3.0


# --------------------------------------------------------------------------
# The EMA over a subtree
# --------------------------------------------------------------------------

class TwoTrees(Objective):
    """Two independent parameter subtrees, so EMA scoping is observable."""

    def __init__(self, ema):
        self.ema = ema

    def fresh_variables(self, key, held):
        return {"params": {"tracked": {"w": jnp.ones((2,))},
                           "untracked": {"w": jnp.ones((2,))}}}

    def loss(self, variables, batch, step):
        total = sum(jnp.sum(leaf ** 2) for leaf in jax.tree.leaves(variables))
        return total, Aux({})


def test_the_ema_stores_only_the_selected_subtree_and_merges_over_the_rest():
    objective = TwoTrees(EMASpec(decay=optax.constant_schedule(0.5),
                                 select=under("params", "tracked")))
    state = make_trainer(objective=objective).fit(dataset(endless), steps=2, log_every=1)

    assert set(state.ema["params"]) == {"tracked"}
    live = state.variables["params"]
    assert np.any(np.asarray(state.ema["params"]["tracked"]["w"]) != np.asarray(live["tracked"]["w"]))
    merged = merge(state.variables, state.ema)
    assert merged["params"]["untracked"] is live["untracked"]
    np.testing.assert_array_equal(merged["params"]["tracked"]["w"], state.ema["params"]["tracked"]["w"])
    params = {"params": {"tracked": {"w": jnp.full((2,), 3.0)}, "untracked": {"w": jnp.zeros((2,))}}}
    average = {"params": {"tracked": {"w": jnp.ones((2,))}}}
    updated = ema_update(average, params, 0.5)
    assert set(updated["params"]) == {"tracked"}
    np.testing.assert_array_equal(updated["params"]["tracked"]["w"], 2.0)


# --------------------------------------------------------------------------
# The custom step
# --------------------------------------------------------------------------

class TwoPlayers(Objective):
    """A generator that moves a scalar toward a target and a discriminator
    that tracks the generator: two losses, two parameter groups."""

    ema = None

    def fresh_variables(self, key, held):
        return {"params": {"gen": {"g": jnp.zeros(())}, "disc": {"d": jnp.zeros(())}}}

    def loss(self, variables, batch, step):
        raise AssertionError("the custom step never calls the single loss")

    def generator_loss(self, params, batch):
        return (params["gen"]["g"] - 1.0) ** 2

    def discriminator_loss(self, params, batch):
        return (params["disc"]["d"] - jax.lax.stop_gradient(params["gen"]["g"])) ** 2


def two_optimizers(gen, disc):
    """Both optimizers' states under one init; the step updates one at a time."""
    def init(params):
        return {"gen": gen.init(params["gen"]), "disc": disc.init(params["disc"])}

    def update(*_):
        raise AssertionError("the alternating step drives the two optimizers itself")

    return optax.GradientTransformation(init, update)


def alternating(gen, disc):
    def make_step(objective, optimizer):
        def step(state, batch):
            trainable = state.variables["params"]

            def player(name, optimizer, loss_of):
                def turn(operand):
                    # One player's step: its loss's gradient through its own optimizer, the other held.
                    loss, grads = jax.value_and_grad(loss_of)(trainable, batch)
                    updates, moved = optimizer.update(grads[name], state.opt_state[name], trainable[name])
                    params = {**trainable, name: optax.apply_updates(trainable[name], updates)}
                    return params, {**state.opt_state, name: moved}, loss
                return turn

            params, opt_state, loss = jax.lax.cond(
                state.microstep % 2 == 0, player("gen", gen, objective.generator_loss),
                player("disc", disc, objective.discriminator_loss), None)
            new_state = state.replace(
                microstep=state.microstep + 1,
                updates=state.updates + 1,
                opt_state=opt_state,
                variables={**state.variables, "params": params},
            )
            return new_state, loss, Aux({"player": (state.microstep % 2).astype(jnp.float32)})
        return step
    return make_step


def test_a_custom_step_alternates_two_optimizers_on_the_same_checkpoints_and_tracker(tmp_path):
    """The escape hatch: a GAN-style step with two optimizers, checkpointed
    and logged by the same trainer, resumed from the same directory."""
    gen, disc = optax.sgd(0.25), optax.sgd(0.25)
    tracker = RecordingTracker()

    def trainer():
        return Trainer(TwoPlayers(), two_optimizers(gen, disc), key=jax.random.key(0),
                       layout=Layout(min_shard=1, tolerance=1.0),
                       checkpoints=Checkpoints(str(tmp_path / "gan"), keep=2),
                       tracker=tracker, step=alternating(gen, disc))

    state = trainer().fit(dataset(), steps=2, log_every=1)
    # Step 0 moved the generator half way to 1 (lr 0.25 on a gradient of 2(g - 1));
    # step 1 moved the discriminator half way to the generator. Every value is a few bits' dyadic
    # fraction, so each product and sum is exact.
    assert float(state.variables["params"]["gen"]["g"]) == 0.5
    assert float(state.variables["params"]["disc"]["d"]) == 0.25
    assert [s["train/player"] for _, s in tracker.scalars if "train/player" in s] == [0.0, 1.0]

    resumed = trainer().fit(dataset(), steps=4, log_every=1)
    assert float(resumed.variables["params"]["gen"]["g"]) == 0.75
    assert float(resumed.variables["params"]["disc"]["d"]) == 0.5
    assert Checkpoints(str(tmp_path / "gan")).latest == 4


@pytest.mark.mesh
def test_a_fresh_state_is_built_in_the_buffers_its_held_checkpoint_arrives_in(monkeypatch):
    """A held checkpoint reaches the state's JIT as a copy placed where the
    state keeps each variable, sharded as it is, and the JIT takes those
    buffers over. Handed over as they are, the arrays were placed below the
    state and freed after it was built, a hole as large as the checkpoint:
    on an A100 Qwen3-1.7B's 'minimal' rung then found no block for its
    temporaries. The objective's own arrays stay as they were."""
    from jax.tree_util import Partial

    held = {}
    put = jax.device_put

    def recorded(x, *args, **kwargs):
        out = put(x, *args, **kwargs)
        if isinstance(out, Partial):
            held.update(jax.tree_util.tree_leaves_with_path(out.keywords["variables"]))
        return out

    monkeypatch.setattr(jax, "device_put", recorded)
    trainer, _objective, weights = held_lm_trainer(mesh=MeshSpec(fsdp=jax.device_count()))
    state, shardings, _ = trainer.place()
    params = dict(jax.tree_util.tree_leaves_with_path(shardings.variables))
    sharded = 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(weights):
        copy = held[path]
        assert copy.is_deleted(), jax.tree_util.keystr(path)
        assert copy.sharding == params[path], jax.tree_util.keystr(path)
        sharded += not copy.sharding.is_fully_replicated
        assert not leaf.is_deleted()
    assert sharded
    jax.tree.map(np.testing.assert_array_equal, state.variables, weights)



@pytest.mark.mesh
def test_only_the_held_arrays_an_output_takes_over_are_donated():
    """`placed_held` splits a held tree into the copies a state takes over,
    each placed as its variable, and the rest, replicated: an array beside
    the variables and one whose shape is not its variable's. `taking_over`
    donates the first alone, so JAX finds every donation usable and warns of
    none, the state takes those buffers over, and the rest stay alive until
    their holder drops them, as donated they were freed only as the call
    returned."""
    import warnings

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from jax.tree_util import Partial

    from dew.training.distributed import placed_held, taking_over

    mesh = Mesh(np.array(jax.devices()), ("fsdp",))
    rows = 8 * jax.device_count()
    sharded, replicated = NamedSharding(mesh, P("fsdp")), NamedSharding(mesh, P())

    def initialize(key, *, variables, table):
        params = variables["params"]
        return {"params": {"kernel": params["kernel"] * 2, "bias": jnp.pad(params["bias"], (0, 4))},
                "count": table.sum()}

    kernel, bias, table = jnp.arange(rows * 4.0).reshape(rows, 4), jnp.ones(4), jnp.arange(5.0)
    held = Partial(initialize, variables={"params": {"kernel": kernel, "bias": bias}}, table=table)
    variables = {"params": {"kernel": jax.ShapeDtypeStruct((rows, 4), jnp.float32),
                            "bias": jax.ShapeDtypeStruct((8,), jnp.float32)}}
    shardings = {"params": {"kernel": sharded, "bias": replicated}, "count": replicated}
    taken, kept = placed_held(held, variables, {"params": shardings["params"]}, mesh)
    copy = taken.keywords["variables"]["params"]["kernel"]
    assert copy.sharding == sharded
    assert taken.keywords["variables"]["params"]["bias"] is None and taken.keywords["table"] is None
    assert kept.keywords["variables"]["params"]["kernel"] is None
    with warnings.catch_warnings():
        warnings.filterwarnings("error", "Some donated buffers were not usable")
        state = taking_over(lambda initializer, key: initializer(key), taken, kept, shardings)(
            jax.random.key(0))
    assert copy.is_deleted()
    assert not any(leaf.is_deleted() for leaf in jax.tree.leaves(kept))
    assert not any(leaf.is_deleted() for leaf in (kernel, bias, table))
    np.testing.assert_array_equal(state["params"]["kernel"], kernel * 2)
    np.testing.assert_array_equal(state["params"]["bias"], [1, 1, 1, 1, 0, 0, 0, 0])
    assert float(state["count"]) == 10.0

def test_a_step_compiles_from_its_arrays_shapes_before_they_are_placed():
    """Compiling reads the parameters', optimizer state's and key's shapes
    and shardings, not their values, so a run compiles its step ahead of
    time, into the persistent cache, before it holds them on devices; the
    trainer's own settings (the window, the scaler) stay values. The loss's
    shape once folded the step into the key eagerly, which a key known only
    by its shape cannot do (74761deb)."""
    trainer, _, _ = held_lm_trainer()
    state, shardings, _ = trainer.place()

    def shape(leaf, sharding):
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding)

    abstract = dataclasses.replace(
        state, variables=jax.tree.map(shape, state.variables, shardings.variables),
        opt_state=jax.tree.map(shape, state.opt_state, shardings.opt_state),
        key=shape(state.key, shardings.key))

    compiled = trainer.compile(abstract, {"text": jax.ShapeDtypeStruct((8, 5), jnp.int32)})
    batch = {"text": jnp.zeros((8, 5), jnp.int32)}
    # The same step compiled from placed arrays of the same values: one program, so the step compiled
    # from shapes alone computes its loss and its update bit for bit as it does.
    placed, _, _ = trainer.place()
    concrete = trainer.compile(placed, batch)
    advanced, loss, _, finite, _, _ = jax.block_until_ready(compiled(state, batch))
    again, their_loss, _, _, _, _ = jax.block_until_ready(concrete(placed, batch))
    assert np.array_equal(np.asarray(loss), np.asarray(their_loss))
    assert all(np.array_equal(np.asarray(mine), np.asarray(theirs)) for mine, theirs in
               zip(jax.tree.leaves(advanced.variables), jax.tree.leaves(again.variables), strict=True))
    assert int(advanced.step) == 1 and bool(finite)


def test_a_field_no_device_holds_stays_on_the_host_for_the_code_that_reads_it():
    """A batch may carry a field no device holds, a caption: the step and the
    objective's evaluation take the rest, and a metric reads the caption on
    the host, for the rows it scores."""

    def captioned(source):
        for batch in source:
            yield {**batch, "caption": np.array([f"row {index}" for index in batch["index"]])}

    captions = []

    class Captions(Spread):
        def __call__(self, artifact, batch):
            captions.extend(batch["caption"].tolist())
            return super().__call__(artifact, batch)

    data = dataset(lambda: captioned(Counting()), val=lambda: captioned(val_batches(1)()))
    state = Trainer(Features(), optax.sgd(.1), key=jax.random.key(0)).fit(
        data, steps=2, eval_every=2, metrics=(Captions([]),), checkpoint_final=False)
    scored = next(val_batches(1)())["index"]
    assert int(state.step) == 2 and captions == [f"row {index}" for index in scored]


def test_a_batch_of_other_dtypes_at_the_same_shapes_runs_a_step_compiled_for_them():
    """The compiled step is held under the batch's whole abstract signature, its fields and
    each leaf's dtype as well as its shape: a stream whose inputs turn half precision at the
    same shapes runs a step compiled for them, which an executable for float32 refuses."""

    class Narrowing(Counting):
        def __next__(self):
            batch = super().__next__()
            return batch if self.index % 2 else {**batch, "x": batch["x"].astype(np.float16)}

    tracker = RecordingTracker()
    Trainer(Regression(), optax.sgd(.1), key=jax.random.key(0), tracker=tracker).fit(
        dataset(Narrowing), steps=4, log_every=1, checkpoint_final=False)
    compiles = [scalars["train/compiles"] for _, scalars in tracker.scalars if "train/compiles" in scalars]
    assert compiles[-1] == 2


def test_a_failing_metric_fails_the_validation_pass():
    class Broken:
        name = "broken"
        reads = Representations

        def __call__(self, artifact, batch):
            raise ZeroDivisionError("metric over an empty batch")

        def merge(self, accumulated, contribution):
            return accumulated + contribution

        def finalize(self, accumulated):
            return accumulated

    with pytest.raises(ZeroDivisionError):
        make_trainer(objective=Features()).fit(dataset(val=val_batches(1)), steps=1,
                                               log_every=1, eval_every=1, metrics=(Broken(),))


def test_a_mesh_without_stages_takes_its_one_microbatch_asked_for():
    """A mesh with no stage axis feeds one microbatch a step through it, which a run may ask for by
    name: it trains as the run that leaves it unset does, where asking once refused the run."""
    def fit(**asked):
        trainer = Trainer(Regression(), optax.sgd(0.1), key=jax.random.key(0), **asked)
        return trainer.fit(dataset(), steps=2, log_every=1)

    named, planned = fit(microbatches=1), fit()
    for left, right in zip(jax.tree.leaves(named.variables), jax.tree.leaves(planned.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_a_state_built_from_a_held_tree_steps_without_freeing_the_tree():
    """A state the trainer builds eagerly from the objective's held tree (`Trainer.initial_state`),
    stepped on one device, whose step takes its arguments' shardings and donates them: the objective's
    held arrays stay alive, since the state holds buffers of its own, as a placed state does
    (`distributed.placed_held`)."""
    trainer, objective, _ = held_lm_trainer()
    trainer.device_mesh = MeshSpec().build(jax.devices()[:1])
    state = trainer.initial_state()
    batch = {"text": jnp.arange(1, 6, dtype=jnp.int32)[None]}
    stepped, *_ = trainer.compile(state, batch)(state, batch)
    jax.block_until_ready(stepped)
    assert not any(leaf.is_deleted() for leaf in jax.tree.leaves(objective.variables))
