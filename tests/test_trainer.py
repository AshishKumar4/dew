"""The general trainer on a small objective: the loop, checkpoints, resume,
validation, the rejected mixed-precision step and the custom step.

Nothing here knows a modality. The objective is a two-output affine map, the
data is synthetic, and what is asserted is what the trainer owns: the step
count, what lands on disk and when, what a resume restores, what reaches the
tracker, and what a failure does to the run.
"""

import contextlib
import dataclasses
import gc
import io
import json
import logging
import math
import os
import re
import subprocess
import sys
import weakref
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest
from affine_run import BATCH, FEATURES, Counting, Data, Features, Regression, Spread, raw_leaf, val_batches
from flax import linen as nn
from flax.errors import ScopeParamShapeError
from recording import RecordingTracker
from rich.console import Console
from steady_state import steady_state

from dew import position
from dew.artifacts import Representations
from dew.checkpoints import STATE_LEAVES, Ranking
from dew.config import OptimConfig
from dew.data import DataPartition
from dew.objectives.base import Aux, EMASpec, Objective, freeze, merge, select, under
from dew.training import (
    Checkpoints,
    Layout,
    MeshSpec,
    Trainer,
    TrainState,
    display,
    ema_update,
    memory,
    trainer as trainer_module,
)
from dew.training.optim import Cosine
from dew.training.transaction import write_back


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

def test_fit_lets_go_of_each_state_its_step_consumed():
    """A step donates the state it is handed, so once the next state exists
    nothing in fit may still hold the old one: its arrays have lost their
    buffers, and a drain of every live array (`dew.Profiler`) waits on them."""
    trainer = make_trainer()
    placed, held, alive = trainer.place, [], []

    def place():
        state, shardings, position = placed()
        held.append(weakref.ref(state))
        return state, shardings, position

    class Watching(RecordingTracker):
        def log(self, scalars, step):
            if "train/loss" in scalars:
                gc.collect()
                alive.append(held[0]() is not None)

    trainer.place = place
    trainer.tracker = Watching()
    trainer.fit(Data(), steps=2, log_every=1)
    assert alive == [False, False]


def test_train_state_exposes_the_whole_variables_tree():
    objective = Regression()
    objective.model = AffineWithOffset()
    trainer = Trainer(objective, optax.sgd(.1), key=0)
    state = trainer.initial_state()
    assert "params" in state.variables
    np.testing.assert_array_equal(state.variables["constants"]["offset"], jnp.zeros((2,)))
    shifted = state.replace(variables={**state.variables, "constants": {"offset": jnp.ones((2,))}})
    prediction = objective.model.apply(shifted.variables, jnp.zeros((1, FEATURES)))
    np.testing.assert_array_equal(prediction, jnp.ones((1, 2)))
    assert not hasattr(state, "params")


def test_fit_trains_to_the_step_it_was_asked_for():
    state = make_trainer().fit(Data(endless), steps=4, log_every=2)
    assert int(state.step) == 4


def test_an_optim_config_is_built_over_the_updates_fit_makes():
    """Four steps of two microbatches are two updates: the run is the config built over two, bit
    for bit, not over four. Before fit there is no length to build it over."""
    config = OptimConfig(schedule=Cosine(peak=0.1, warmup_steps=0), weight_decay=0.1, b2=0.99)
    with pytest.raises(ValueError, match="fit"):
        make_trainer(optimizer=config).initial_state()
    built, over_two, over_four = (make_trainer(optimizer=given, accumulation=2).fit(Data(), steps=4).variables
                                  for given in (config, config.build(2), config.build(4)))
    jax.tree.map(np.testing.assert_array_equal, built, over_two)
    assert not all(jax.tree.leaves(jax.tree.map(np.array_equal, built, over_four)))


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
            Data(), steps=24, log_every=4, checkpoint_every=4 if variant == "checkpoints" else None)
    finally:
        window.close()
    assert [step for step, scalars in tracker.scalars if "train/loss" in scalars] == [4, 8, 12, 16, 20, 24]


def test_integer_root_key_matches_a_typed_key_bit_exactly():
    integer = Trainer(Regression(), optax.adam(1e-3), key=0)
    typed = Trainer(Regression(), optax.adam(1e-3), key=jax.random.key(0))
    left = integer.fit(Data(), steps=3, log_every=1)
    right = typed.fit(Data(), steps=3, log_every=1)
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for actual, expected in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(np.asarray(raw_leaf(actual)), np.asarray(raw_leaf(expected)))


def test_integer_root_seed_is_recorded_at_fit_start():
    from dew.telemetry.records import FitStarted

    tracker = RecordingTracker()
    Trainer(Regression(), optax.sgd(.1), key=23, tracker=tracker).fit(Data(), steps=1)
    started = next(value for _, value in tracker.artifacts if isinstance(value, FitStarted))
    assert started.seed == 23


@pytest.mark.mesh
def test_the_banner_says_how_much_of_the_parameters_the_mesh_splits(capsys):
    """MeshSpec(fsdp=2, tensor=2) over a model whose every parameter sits
    below the layout's min_shard splits nothing, and the run said only
    "mesh data 2 x fsdp 2 x tensor 2". The record and the banner carry the
    share of the parameters' bytes a parameter axis splits: none of the
    affine map's, then all of them once the floor is one element."""
    from dew.telemetry.records import FitStarted

    for min_shard, share in ((2**16, 0.0), (1, 1.0)):
        tracker = RecordingTracker()
        Trainer(Regression(), optax.sgd(.1), key=0, mesh=MeshSpec(fsdp=2, tensor=2), tracker=tracker,
                layout=Layout(min_shard=min_shard, tolerance=1.0)).fit(Data(), steps=1)
        started = next(value for _, value in tracker.artifacts if isinstance(value, FitStarted))
        assert started.sharded == share
        assert f"{share:.0%} of the parameters' bytes split" in capsys.readouterr().out


def test_typed_root_key_round_trips_and_resumes_bit_exactly(tmp_path):
    def build(path=None):
        return Trainer(Regression(), optax.adam(1e-3), key=0,
                       checkpoints=None if path is None else Checkpoints(str(path)))

    baseline = build().fit(Data(), steps=4, log_every=1)
    split = build(tmp_path / "typed")
    prefix = split.fit(Data(), steps=2, checkpoint_every=1, log_every=1)
    assert jnp.issubdtype(prefix.key.dtype, jax.dtypes.prng_key)
    fresh = build(tmp_path / "typed")
    restored, _, _ = fresh.place()
    assert restored.key.dtype == prefix.key.dtype
    np.testing.assert_array_equal(jax.random.key_data(restored.key), jax.random.key_data(prefix.key))
    resumed = fresh.fit(Data(), steps=4, checkpoint_every=1, log_every=1)
    for left, right in zip(jax.tree.leaves(resumed), jax.tree.leaves(baseline), strict=True):
        np.testing.assert_array_equal(np.asarray(raw_leaf(left)), np.asarray(raw_leaf(right)))


@pytest.mark.mesh(devices=2)
def test_root_key_stays_replicated_on_a_multi_device_mesh():
    trainer = Trainer(Regression(), optax.sgd(.1), key=0, mesh=MeshSpec(fsdp=2),
                      layout=Layout(min_shard=1, tolerance=1.0))
    state, shardings, _ = trainer.place()
    assert shardings.key.spec == jax.sharding.PartitionSpec()
    assert state.key.sharding.mesh == shardings.key.mesh
    for shard in state.key.addressable_shards:
        np.testing.assert_array_equal(jax.random.key_data(shard.data), jax.random.key_data(state.key))


def test_a_held_fit_error_releases_the_prefetch_iterator(monkeypatch):
    refs = []

    class ObservedPrefetch(trainer_module.DevicePrefetchIterator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            refs.append(weakref.ref(self))

    class WrongWidth(Counting):
        def __next__(self):
            batch = super().__next__()
            if self.index == 4:
                batch = {**batch, "x": np.concatenate([batch["x"], batch["x"][:, :1]], axis=1)}
            return batch

    monkeypatch.setattr(trainer_module, "DevicePrefetchIterator", ObservedPrefetch)
    with pytest.raises(ScopeParamShapeError) as failure:
        make_trainer().fit(Data(train=WrongWidth), steps=10, log_every=100)
    # A caller may retain the error for reporting; its traceback must not
    # retain the closed worker and its queue after fit has relinquished it.
    assert failure.value.__traceback__ is not None
    gc.collect()
    assert len(refs) == 1
    assert refs[0]() is None


def test_off_a_terminal_fit_prints_one_line_per_logging_interval(capsys):
    """Piped to a log, the run's progress is plain lines at the logging
    cadence, not a live display's redraws."""
    make_trainer().fit(Data(endless), steps=7, log_every=2)
    output = capsys.readouterr().out

    assert [int(step) for step in re.findall(r"^step\s+(\d+)/7\b", output, re.M)] == [2, 4, 6], output
    assert "\x1b" not in output, "terminal control codes went to a pipe"


def test_the_summary_gives_the_last_steps_loss_and_only_this_fits_steps(tmp_path, capsys):
    """Seven steps logged every two end on a step no interval read, and a
    fit asked again for the step the run is at trains nothing."""
    tracker = RecordingTracker()
    make_trainer(tracker=tracker).fit(Data(), steps=7, log_every=1)
    losses = {step: scalars["train/loss"] for step, scalars in tracker.scalars if "train/loss" in scalars}
    capsys.readouterr()

    trainer = make_trainer(tmp_path)
    trainer.fit(Data(), steps=7, log_every=2)
    summary = re.search(r"Trained (\d+) steps.*final loss (\S+)", capsys.readouterr().out, re.S)
    assert summary is not None
    assert int(summary[1]) == 7
    # Printed to four significant digits: within half a unit of the last.
    assert abs(float(summary[2]) - losses[7]) <= 5e-4 * abs(losses[7])
    assert abs(float(summary[2]) - losses[6]) > 5e-4 * abs(losses[7])

    trainer.fit(Data(), steps=7, log_every=2)
    assert re.search(r"Trained \d+ steps", capsys.readouterr().out) is None


def test_a_second_fit_continues_from_the_state_on_disk(tmp_path):
    trainer = make_trainer(tmp_path)
    first = trainer.fit(Data(), steps=3, log_every=1)
    resumed = make_trainer(tmp_path).fit(Data(), steps=5, log_every=1)

    assert int(first.step) == 3 and int(resumed.step) == 5
    assert Checkpoints(str(tmp_path / "run")).latest == 5


def test_constructing_a_trainer_opens_nothing(tmp_path):
    make_trainer(tmp_path)
    assert not (tmp_path / "run").exists(), "the checkpoint directory was created"


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
    make_trainer(tmp_path, objective=Keyed()).fit(Data(), steps=2, log_every=1)
    resumed = make_trainer(tmp_path, objective=Keyed()).fit(Data(), steps=4, log_every=1)
    whole = make_trainer(objective=Keyed()).fit(Data(), steps=4, log_every=1)

    for expected, actual in zip(jax.tree.leaves(whole.variables),
                                jax.tree.leaves(resumed.variables), strict=True):
        np.testing.assert_allclose(np.asarray(expected), np.asarray(actual), rtol=1e-6)


def test_the_ema_lags_the_parameters_at_the_configured_decay():
    trainer = make_trainer()
    state = trainer.initial_state()
    batch = next(Counting())
    starts = [np.asarray(leaf) for leaf in jax.tree.leaves(state.variables)]  # the step consumes `state`
    new_state, *_ = trainer.compile(state, batch)(state, batch)
    for start, end, ema in zip(starts, jax.tree.leaves(new_state.variables),
                               jax.tree.leaves(new_state.ema), strict=True):
        np.testing.assert_allclose(ema, .5 * start + .5 * np.asarray(end), rtol=1e-6)
    assert int(new_state.step) == int(new_state.updates) == 1


# --------------------------------------------------------------------------
# Checkpoints: what lands, when, and what a resume gets back
# --------------------------------------------------------------------------

def test_fit_checkpoints_the_final_step(tmp_path):
    """The last save of a run carries the step the run ended on, not step 0."""
    trainer = make_trainer(tmp_path)
    trainer.fit(Data(), steps=4, log_every=1)

    assert trainer.checkpoints.latest == 4
    assert os.path.isdir(tmp_path / "run" / "4")
    assert not os.path.exists(tmp_path / "run" / "0")


def test_checkpoint_every_saves_on_its_own_cadence(tmp_path):
    """A cadence that does not divide log_every still fires, and the end of
    the run does not write the step the loop already wrote."""
    trainer = make_trainer(tmp_path, keep=4)
    saved = []
    real_save = trainer.checkpoints.save

    def spy(step, state, position, metrics=None, *, share=None, **metadata):
        saved.append((step, None if metrics is None else sorted(metrics)))
        return real_save(step, state, position, metrics, share=share, **metadata)

    trainer.checkpoints.save = spy
    trainer.fit(Data(), steps=6, log_every=4, checkpoint_every=2)

    assert saved == [(2, ["train/loss"]), (4, ["train/loss"]), (6, ["train/loss"])]
    assert set(trainer.checkpoints._open().all_steps()) == {2, 4, 6}


def test_checkpoint_every_needs_a_stream_that_reports_its_position(tmp_path):
    with pytest.raises(ValueError, match="get_state"):
        make_trainer(tmp_path).fit(Data(endless), steps=2, checkpoint_every=1)


def test_checkpoint_every_needs_a_stream_that_can_also_be_put_back(tmp_path):
    """A stream that reports its position but cannot be set back to it would
    write position=None into every checkpoint and fail only on resume, so it
    is refused where a stream with no position is."""
    class HalfCheckpointable:
        def __init__(self):
            self.inner = Counting()

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.inner)

        def get_state(self):
            return self.inner.get_state()

    with pytest.raises(ValueError, match="get_state"):
        make_trainer(tmp_path).fit(Data(HalfCheckpointable), steps=2, checkpoint_every=1)


def test_checkpoint_every_without_a_checkpointer_is_refused():
    """Asking for checkpoints from a trainer that has nowhere to write them
    raises a ValueError before the first step, not at the first save."""
    with pytest.raises(ValueError, match="no checkpointer"):
        make_trainer().fit(Data(), steps=2, checkpoint_every=1)


def test_fit_that_never_trains_checkpoints_step_zero(tmp_path):
    """A run that ends at step 0 writes a step-0 checkpoint."""
    trainer = make_trainer(tmp_path)
    trainer.fit(Data(), steps=0)
    assert trainer.checkpoints.latest == 0


def test_a_run_past_its_target_is_refused(tmp_path):
    make_trainer(tmp_path).fit(Data(), steps=3, log_every=1)
    with pytest.raises(ValueError, match="past"):
        make_trainer(tmp_path).fit(Data(), steps=2)


def held_lm_trainer(**settings):
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective

    model = CausalTransformer(vocab_size=32, emb_features=8, num_layers=1, num_heads=1,
                              mlp_features=16, max_seq_len=8)
    weights = jax.jit(LMObjective(model, seq_len=4).init)(jax.random.key(0))
    objective = LMObjective(model, seq_len=4, variables=weights, ema_decay=0.999)
    return Trainer(objective, optax.adam(1e-3), key=jax.random.key(0),
                   layout=Layout(min_shard=1, tolerance=1.0), **settings), objective, weights


def test_overlapping_token_windows_resume_in_a_fresh_trainer_bit_exactly(tmp_path):
    from dew.data import Loading, TokenWindows
    from dew.data.dataset import Forwarding

    tokens = np.arange(29, dtype=np.uint16)
    tokens.tofile(tmp_path / "train.bin")
    tokens.tofile(tmp_path / "val.bin")

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
        data = TokenWindows(path=str(tmp_path), seq_len=4, stride=1, seed=7,
                            loading=Loading(workers=0, threads=1, read_buffer=1)).load(batch=BATCH)
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


def ladder_lm_trainer(directory, fits):
    """An LM trainer on a decoder whose fit check answers `fits(rung)` for
    the rung each compile is at: (whether the head is tiled, the remat's
    record). It stands in for the free memory a process finds."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective

    model = CausalTransformer(vocab_size=32, emb_features=8, num_layers=2, num_heads=1,
                              mlp_features=16, max_seq_len=8)
    objective = LMObjective(model, seq_len=4)
    trainer = make_trainer(directory, objective=objective, optimizer=optax.adam(1e-2))

    def headroom(executable, devices, held=0):
        rung = (objective.head_tile is not None, memory.recompute_record(objective))
        return 0 if fits(rung) else -1
    return trainer, objective, headroom


def lm_windows(tmp_path):
    from dew.data import Loading, TokenWindows

    tokens = (np.arange(97, dtype=np.uint16) * 7) % 32
    tokens.tofile(tmp_path / "train.bin")
    tokens.tofile(tmp_path / "val.bin")
    return TokenWindows(path=str(tmp_path), seq_len=4, stride=1, seed=7,
                        loading=Loading(workers=0, threads=1, read_buffer=1)).load(batch=8)


def test_a_resumed_run_compiles_the_rung_its_checkpoint_trained_on(tmp_path, monkeypatch):
    """The rung a run settles on is part of what it resumes: a process that
    restores the state finds other free memory than the one that built it
    (on an A100 a Qwen3-1.7B run took 'full', and its resumed process,
    without the fresh state's hole, 'minimal', and parted from step 70). The
    resumed run compiles the checkpoint's rung where the step fits more
    lightly too, and trains bit for bit as the uninterrupted one."""
    def tiled_and_minimal(rung):
        return rung in ((True, 'minimal'), (True, 'full'))

    whole, _, headroom = ladder_lm_trainer(tmp_path / "whole", tiled_and_minimal)
    monkeypatch.setattr(memory, 'step_headroom', headroom)
    expected = whole.fit(lm_windows(tmp_path), steps=4, checkpoint_every=2)
    split, _, headroom = ladder_lm_trainer(tmp_path / "split", tiled_and_minimal)
    monkeypatch.setattr(memory, 'step_headroom', headroom)
    split.fit(lm_windows(tmp_path), steps=2, checkpoint_every=2)

    resumed, objective, headroom = ladder_lm_trainer(tmp_path / "split", lambda rung: True)
    monkeypatch.setattr(memory, 'step_headroom', headroom)
    actual = resumed.fit(lm_windows(tmp_path), steps=4, checkpoint_every=2)
    assert (objective.head_tile is not None, memory.recompute_record(objective)) == (
        True, 'minimal')
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert np.asarray(raw_leaf(left)).tobytes() == np.asarray(raw_leaf(right)).tobytes()


def test_a_resumed_run_that_cannot_fit_its_checkpoints_rung_climbs_and_says_so(tmp_path, monkeypatch, caplog):
    """A process that cannot fit the rung its checkpoint trained on moves up
    the ladder, never down, and says the run now computes otherwise."""
    split, _, headroom = ladder_lm_trainer(tmp_path / "split", lambda rung: rung[0])
    monkeypatch.setattr(memory, 'step_headroom', headroom)
    split.fit(lm_windows(tmp_path), steps=2, checkpoint_every=2)
    caplog.clear()

    resumed, objective, headroom = ladder_lm_trainer(tmp_path / "split", lambda rung: rung == (True, 'full'))
    monkeypatch.setattr(memory, 'step_headroom', headroom)
    resumed.fit(lm_windows(tmp_path), steps=3, checkpoint_every=2)
    assert memory.recompute_record(objective) == 'full'
    assert "the rung its checkpoint trained on" in caplog.text


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="CUDA embedding-gradient reductions")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_a_cuda_lm_repeats_and_resumes_bit_exactly_with_deterministic_ops(tmp_path, dtype):
    """Repeated token IDs share embedding-gradient updates. CUDA's default
    scatter-add order is not repeatable; conftest enables deterministic ops
    before the backend opens. Check every state leaf, not just parameters.
    bf16 at the default precision runs the head that rounds its logits and
    their gradient to bf16, and resumes bit-exactly too."""
    from dew.data import Loading, TokenWindows
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective

    tokens = np.tile(np.arange(8, dtype=np.uint8), 256)
    tokens.tofile(tmp_path / "train.bin")
    tokens[:128].tofile(tmp_path / "val.bin")
    (tmp_path / "meta.json").write_text(json.dumps({
        "tokenizer": "symbols", "vocab_size": 8, "dtype": "uint8",
        "train_tokens": len(tokens), "val_tokens": 128, "eos_id": None,
    }))
    data = TokenWindows(path=str(tmp_path), seq_len=16,
                        loading=Loading(workers=0)).load(batch=8)
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

from dew.diffusion import presets
from dew.inputs import Field, InputSpec
from dew.objectives.diffusion import DiffusionObjective
from dew.nn.backbones import SimpleDiT
from dew.sampling import Euler
from dew.training import Trainer

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


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="cuDNN convolution autotuning")
def test_two_processes_train_a_conv_model_to_the_bit_under_the_repeatable_flags(tmp_path):
    """Autotuning picks a convolution's cuDNN algorithm per process, so only
    a second process shows whether it agrees. Two fresh processes, without a
    compilation cache, train a DiT (a convolution embeds its patches) two
    Adam steps under the cuda lane's flags; every state leaf agrees."""
    from lane_environment import REPEATABLE_GPU_FLAGS

    root = Path(__file__).resolve().parents[1]
    flags = [flag for flag in os.environ.get("XLA_FLAGS", "").split()
             if not flag.startswith(("--xla_gpu_deterministic_ops", "--xla_gpu_autotune_level"))]
    environment = {**os.environ, "JAX_PLATFORMS": "cuda", "PYTHONPATH": str(root / "src"),
                   "XLA_FLAGS": " ".join([*flags, *REPEATABLE_GPU_FLAGS]),
                   "JAX_ENABLE_COMPILATION_CACHE": "false"}
    runs = []
    for index in range(2):
        out = tmp_path / f"state-{index}.npz"
        done = subprocess.run([sys.executable, "-c", REPEATED_CONV_STEPS, str(out)], env=environment,
                              capture_output=True, text=True, timeout=600)
        assert done.returncode == 0, done.stderr
        runs.append(np.load(out))
    assert runs[0].files == runs[1].files
    for name in runs[0].files:
        np.testing.assert_array_equal(runs[0][name], runs[1][name])


def test_the_state_is_built_from_the_initializer_and_the_key_alone():
    """What `place` compiles takes the objective's held variables and the run
    key as arguments, so a loaded checkpoint reaches the device as data.

    Resolved from the objective instead, the same construction captures the
    whole tree as a constant: 2.2 GiB inside the executable for a 0.6B
    checkpoint, which is past the compilation cache's 2 GiB entry limit. Both
    calls build the same state, so this is where the arrays travel, not what
    they are.
    """
    trainer, objective, weights = held_lm_trainer()
    shardings = trainer.shardings(jax.eval_shape(trainer.initial_state))

    passed = jax.make_jaxpr(trainer.initial_state)(objective.initializer, trainer.key).consts
    resolved = jax.make_jaxpr(trainer.initial_state)().consts
    explicit = jax.jit(trainer.initial_state, out_shardings=shardings)(
        objective.initializer, trainer.key)
    default = jax.jit(trainer.initial_state, out_shardings=shardings)()

    assert passed == [], "the state JIT still captures arrays it was handed as data"
    assert sum(int(np.asarray(raw_leaf(value)).nbytes) for value in resolved) >= sum(
        int(np.asarray(leaf).nbytes) for leaf in jax.tree.leaves(weights)), (
            "resolving the input no longer captures the tree, so this proves nothing")
    for before, after in zip(jax.tree.leaves(default), jax.tree.leaves(explicit), strict=True):
        np.testing.assert_array_equal(np.asarray(raw_leaf(before)), np.asarray(raw_leaf(after)))


def test_place_builds_the_state_through_the_overridable_method():
    """`place` compiles the same method a subclass overrides, so a trainer
    that adjusts the state it starts from still decides what a run begins
    with. Compiling a private construction instead skipped the override."""
    class Marked(Trainer):
        def initial_state(self, initializer=None, key=None):
            state = super().initial_state(initializer, key)
            return dataclasses.replace(state, updates=jnp.asarray(7, jnp.int32))

    _, objective, _ = held_lm_trainer()
    trainer = Marked(objective, optax.adam(1e-3), key=jax.random.key(0),
                     layout=Layout(min_shard=1, tolerance=1.0))

    state, _, position = trainer.place()

    assert int(state.updates) == 7, "place bypassed the overridden state construction"
    assert int(trainer.initial_state().updates) == 7 and position is None


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
    advanced, _loss, _, finite, _ = step(state, batch)
    assert bool(finite) and int(advanced.step) == 1
    assert stale.is_deleted()
    again, _, _, _, _ = step(advanced, batch)
    assert int(again.step) == 2


def test_a_placed_state_holds_every_buffer_once():
    """Donation moves each buffer of the state exactly once, so no two leaves
    may share one; the EMA in particular starts equal to the parameters but
    not as them."""
    _, objective, _ = held_lm_trainer()
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0),
                      layout=Layout(min_shard=1, tolerance=1.0))
    # Both ways a state comes into being: built eagerly, and placed on the
    # mesh through one compiled initializer.
    for state in (trainer.initial_state(), trainer.place()[0]):
        assert state.ema is not None
        pointers: dict[int, list[str]] = {}
        for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
            for shard in leaf.addressable_shards:
                pointers.setdefault(shard.data.unsafe_buffer_pointer(), []).append(jax.tree_util.keystr(path))
        shared = [paths for paths in pointers.values() if len(paths) > 1]
        assert not shared, shared


def test_place_asks_the_objective_for_its_held_variables_once():
    """Shapes and values come from one resolution of the same inputs, so an
    objective is not asked to produce its held tree twice per placement."""
    asked = []

    class Counted(Objective):
        ema = None

        def held_variables(self):
            asked.append(1)
            return {"params": {"w": jnp.ones((2,))}}

        def init(self, key, variables=None):
            return self.held_variables() if variables is None else variables

        def loss(self, variables, batch, step):
            return jnp.sum(variables["params"]["w"]), Aux({})

    trainer = Trainer(Counted(), optax.sgd(0.1), key=jax.random.key(0),
                      layout=Layout(min_shard=1, tolerance=1.0))
    asked.clear()

    trainer.place()

    assert len(asked) == 1, f"held_variables was evaluated {len(asked)} times in one place()"


def test_restore_preserves_the_optimizer_state_the_ema_and_the_key(tmp_path):
    trainer = make_trainer(tmp_path, optimizer=optax.adam(1e-3))
    trained = trainer.fit(Data(), steps=3, log_every=1)

    resumed = make_trainer(tmp_path, optimizer=optax.adam(1e-3))
    state, _, position = resumed.place()

    assert int(state.step) == 3, "the step counter was reset"
    for field in ("variables", "opt_state", "ema"):
        for before, after in zip(jax.tree.leaves(getattr(trained, field)),
                                 jax.tree.leaves(getattr(state, field)), strict=True):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after), err_msg=field)
    assert jnp.array_equal(jax.random.key_data(trained.key), jax.random.key_data(state.key))
    assert json.loads(position)["index"] == 3


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


def test_an_ema_that_follows_its_weights_is_stored_in_fewer_bytes(tmp_path):
    """An average agrees with the weights it follows in its leading bits, and
    the checkpoint stores it in fewer bytes than an EMA that does not."""
    rng = np.random.default_rng(0)
    kernel = rng.normal(size=(512, 512)).astype(np.float32)
    near = kernel * (1 + 1e-4 * rng.normal(size=kernel.shape)).astype(np.float32)
    far = rng.normal(size=kernel.shape).astype(np.float32)

    def stored_bytes(ema, name):
        checkpoints = Checkpoints(str(tmp_path / name))
        checkpoints.save(0, held_state({"params": {"kernel": jnp.asarray(kernel)}},
                                       {"params": {"kernel": jnp.asarray(ema)}}), None)
        checkpoints.wait()
        return sum(path.stat().st_size for path in (tmp_path / name).rglob("*") if path.is_file())

    assert stored_bytes(near, "near") < stored_bytes(far, "far") - 0.3 * kernel.nbytes


def test_a_checkpoint_that_stores_the_ema_as_itself_still_restores(tmp_path):
    """Checkpoints written before the EMA was stored as its difference from
    the weights hold it as plain arrays, and read back as they were written."""
    state = awkward_state()
    written = ocp.CheckpointManager(str(tmp_path / "run"), item_handlers=ocp.PyTreeCheckpointHandler())
    written.save(0, args=ocp.args.PyTreeSave({name: getattr(state, name) for name in STATE_LEAVES}))
    written.wait_until_finished()

    checkpoints = Checkpoints(str(tmp_path / "run"))
    template = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype,
                                                              sharding=leaf.sharding), state.ema)
    for restored in (checkpoints.restore()[0]["ema"], checkpoints.restore({"ema": template})[0]["ema"]):
        assert jax.tree.all(jax.tree.map(same_bits, state.ema, restored))


class AffineWithOffset(nn.Module):
    """`Affine` once it has gained a variable its earlier checkpoints lack."""

    @nn.compact
    def __call__(self, x):
        return nn.Dense(2)(x) + self.variable("constants", "offset", jnp.zeros, (2,)).value


def test_a_checkpoint_without_a_leaf_the_model_now_has_is_refused_by_name(tmp_path):
    """A resume cannot hand back a state with a leaf the checkpoint never
    stored, so it refuses and names the leaf."""
    make_trainer(tmp_path).fit(Data(), steps=2, log_every=1)
    grown = Regression()
    grown.model = AffineWithOffset()
    with pytest.raises(ValueError, match=r"holds no .*\['variables'\]\['constants'\]\['offset'\]"):
        make_trainer(tmp_path, objective=grown).place()


def test_a_resumed_run_continues_the_data_where_it_stopped(tmp_path):
    """A run killed at step 2 and resumed to 4 lands where an uninterrupted
    four-step run lands: the batches after the checkpoint are neither
    replayed nor skipped, so the parameters agree."""
    make_trainer(tmp_path).fit(Data(), steps=2, log_every=1)
    resumed = make_trainer(tmp_path).fit(Data(), steps=4, log_every=1)
    whole = make_trainer().fit(Data(), steps=4, log_every=1)

    for expected, actual in zip(jax.tree.leaves(whole.variables),
                                jax.tree.leaves(resumed.variables), strict=True):
        np.testing.assert_allclose(np.asarray(expected), np.asarray(actual), rtol=1e-6)
    # The position written at the end names the batch a resume would read next.
    _, position = Checkpoints(str(tmp_path / "run")).restore(share=DataPartition())
    assert json.loads(position)["index"] == 4


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


def test_a_bucket_uri_reaches_orbax_verbatim(tmp_path, monkeypatch):
    """A URI has no local form: abspath would make it <cwd>/gs:/bucket."""
    monkeypatch.chdir(tmp_path)
    assert Checkpoints("gs://bucket/checkpoints/run").directory == "gs://bucket/checkpoints/run"
    assert Checkpoints("./relative").directory == str(tmp_path / "relative")
    assert not (tmp_path / "gs:").exists()


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


def stop_at_local_step(trainer, stop: int):
    """Kill the run right after its local save of `stop` has landed."""
    class Stop(Exception):
        pass

    save_local = trainer.checkpoints.save_local

    def save_then_stop(step, state, position, **options):
        save_local(step, state, position, **options)
        trainer.checkpoints.wait()
        if step == stop:
            raise Stop()

    trainer.checkpoints.save_local = save_then_stop
    with pytest.raises(Stop):
        trainer.fit(Data(), steps=100, log_every=1, checkpoint_every=3)


def test_the_local_checkpoint_is_written_on_its_own_cadence_and_keeps_the_newest(tmp_path):
    """Local every two, persistent every three: after eight steps the
    persistent directory holds 3, 6 and the final 8 with their losses, the
    local one holds 6 alone (2 and 4 replaced, the end of the run being the
    persistent save's), and the persistent directory holds nothing of the
    local cadence."""
    trainer = local_trainer(tmp_path)
    trainer.fit(Data(), steps=8, log_every=1, checkpoint_every=3)
    checkpoints = trainer.checkpoints

    assert sorted(checkpoints._open().all_steps()) == [3, 6, 8]
    assert sorted(checkpoints._open_local().all_steps()) == [6]
    assert checkpoints.best in (3, 6, 8)
    assert checkpoints.local_path == str(tmp_path / "local" / "process0")
    assert (tmp_path / "local" / "process0" / "6" / "commit_success.txt").exists()
    assert not (tmp_path / "local" / "process0" / "4").exists()
    assert not (tmp_path / "run" / "4").exists()


def test_a_newer_local_checkpoint_wins_the_resume_and_leaves_the_persistent_one(tmp_path):
    """Killed after local step 8 with persistent step 6 the newest on disk:
    the resume opens at 8, reads it from the local directory, and lands
    where an unkilled run lands, while the persistent files at 3 and 6 are
    byte for byte what they were."""
    stop_at_local_step(local_trainer(tmp_path), 8)
    persistent = tmp_path / "run"
    assert sorted(int(p.name) for p in persistent.iterdir() if p.name.isdigit()) == [3, 6]
    before = {path: path.read_bytes() for path in persistent.rglob("*") if path.is_file()}
    checkpoints = Checkpoints(str(persistent), local_directory=str(tmp_path / "local"),
                              local_every=2)
    assert checkpoints.latest == 8
    assert checkpoints.source(8) == str(tmp_path / "local" / "process0")
    assert checkpoints.source(6) == str(persistent)

    resumed = local_trainer(tmp_path).fit(Data(), steps=9, log_every=1, checkpoint_every=3)
    whole = make_trainer(tmp_path / "whole").fit(Data(), steps=9, log_every=1, checkpoint_every=3)

    assert int(resumed.step) == 9
    assert jax.tree.map(lambda a, b: bool(np.array_equal(a, b)),
                        resumed.variables, whole.variables) == jax.tree.map(lambda _: True, whole.variables)
    assert all(path.read_bytes() == data for path, data in before.items())
    assert sorted(int(p.name) for p in persistent.iterdir() if p.name.isdigit()) == [3, 6, 9]


@pytest.mark.mesh
def test_a_local_checkpoint_refuses_another_placement_and_names_the_way_out(tmp_path):
    """The local copy holds this process's shards for the mesh it was
    written on; a resume on another mesh is refused with the leaf that
    moved and the persistent step to fall back to."""
    stop_at_local_step(local_trainer(tmp_path), 8)

    with pytest.raises(ValueError) as error:
        local_trainer(tmp_path, mesh=MeshSpec(fsdp=2)).fit(Data(), steps=9, log_every=1)
    message = str(error.value)
    assert "holds step 8 written with" in message and "places it as" in message
    assert f"delete {tmp_path / 'local'}" in message
    assert "persistent checkpoint at step 6" in message


def test_local_checkpoints_take_both_the_directory_and_the_cadence(tmp_path):
    with pytest.raises(ValueError, match="both local_directory and local_every"):
        Checkpoints(str(tmp_path / "run"), local_directory=str(tmp_path / "local"))
    with pytest.raises(ValueError, match="both local_directory and local_every"):
        Checkpoints(str(tmp_path / "run"), local_every=2)
    with pytest.raises(ValueError, match="local_every must be at least 1"):
        Checkpoints(str(tmp_path / "run"), local_directory=str(tmp_path / "local"), local_every=0)


def test_a_local_cadence_needs_a_stream_that_reports_its_position(tmp_path):
    with pytest.raises(ValueError, match="get_state"):
        local_trainer(tmp_path).fit(Data(endless), steps=2)


class ExplodingManager:
    """Orbax when the filesystem refuses the write.

    Stubbed, not provoked with a read-only directory: a real failed async
    orbax write leaves a background thread that never joins, which hangs
    interpreter exit and with it the whole test session.
    """

    def latest_step(self):
        return None

    def save(self, *args, **kwargs):
        raise OSError("No space left on device")

    def wait_until_finished(self):
        pass


def test_a_checkpoint_that_does_not_land_fails_the_run(tmp_path):
    """A checkpoint that did not get written is data loss, not a log line."""
    trainer = make_trainer(tmp_path)
    trainer.checkpoints._manager = ExplodingManager()
    with pytest.raises(OSError):
        trainer.fit(Data(), steps=1, log_every=1)


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


def test_a_position_written_by_another_process_count_is_refused(tmp_path):
    """`Counting` reports where its own stream stopped; a table two processes
    wrote, each over its own share, has no row the single share of one
    process can take over, and says so."""
    trainer = make_trainer(tmp_path)
    trainer.fit(Data(), steps=1, log_every=1)
    _, saved = trainer.checkpoints.restore(share=DataPartition())
    _rewrite_position(trainer, 2, [saved, saved], shares=[[0, 2], [1, 2]])

    with pytest.raises(ValueError, match=r"shares \[\(0, 2\), \(1, 2\)\] \(index, count\), and this "
                                         r"reader reads share 0 of 1"):
        make_trainer(tmp_path).fit(Data(), steps=3)


def test_a_share_offset_resumes_on_whichever_processes_read_that_share(tmp_path):
    """Two processes that read one share, as a sequence split across them
    does, wrote one offset twice; a single process reading that share
    resumes from it, and a table whose readers of one share disagree is
    refused rather than resumed from either."""
    trainer = make_trainer(tmp_path)
    trainer.fit(Data(), steps=1, log_every=1)
    _, saved = trainer.checkpoints.restore(share=DataPartition())
    _rewrite_position(trainer, 2, [saved, saved], shares=[[0, 1], [0, 1]])

    assert Checkpoints(str(tmp_path / "run")).restore(step=2, share=DataPartition())[1] == saved

    other = json.dumps({"index": 7}).encode()
    _rewrite_position(trainer, 3, [saved, other], shares=[[0, 1], [0, 1]])
    with pytest.raises(ValueError, match="share 0 of 1 that differ between the processes"):
        Checkpoints(str(tmp_path / "run")).restore(step=3, share=DataPartition())


def test_a_global_position_is_read_by_any_process_count(tmp_path):
    """A global position is a record count over an order every process count
    reads the same way, so the table two processes wrote is this one
    process's position as well. Only a shard offset ties a resume to the
    count that wrote it."""
    trainer = make_trainer(tmp_path)
    trainer.fit(Data(), steps=1, log_every=1)
    global_position = position.encode(position.Global(records=16, order="Counting"))
    _rewrite_position(trainer, 2, [global_position, global_position], shares=[[0, 2], [1, 2]])

    assert Checkpoints(str(tmp_path / "run")).restore(step=2, share=DataPartition())[1] == global_position


def test_a_global_position_round_trips_and_a_partial_one_is_refused():
    """Dew writes every field of its global position, the phases a run
    completed among them, and reads back only a position with all of them:
    one without its completed phases is damaged, not a shorter format."""
    for place in (position.Global(records=16, order="Counting"),
                  position.Global(records=20, order="B", completed=(("A", 12),))):
        assert position.decode(position.encode(place)) == place
    with pytest.raises(ValueError, match=r"missing \['completed'\]"):
        position.decode(json.dumps({position.ENVELOPE: {"records": 16, "order": "Counting"}}).encode())


def test_global_positions_that_disagree_between_processes_are_refused(tmp_path):
    """Every process reports the same global position, so two that differ are
    two orders, and no one of them is this run's place in its own."""
    trainer = make_trainer(tmp_path)
    trainer.fit(Data(), steps=1, log_every=1)
    _rewrite_position(trainer, 2, [
        position.encode(position.Global(records=records, order="Counting"))
        for records in (16, 32)], shares=[[0, 2], [1, 2]])

    with pytest.raises(ValueError, match="global data position that differs between the 2 processes"):
        Checkpoints(str(tmp_path / "run")).restore(step=2, share=DataPartition())


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def test_eval_every_scores_the_validation_split_and_logs_the_artifacts():
    seen = []
    tracker = RecordingTracker()
    objective = Features()
    trainer = make_trainer(objective=objective, tracker=tracker)
    state = trainer.fit(Data(val=val_batches(3)), steps=4, log_every=2, eval_every=2,
                        metrics=(Spread(seen),), preview=True)

    # Two passes: at step 2, and at the end of the run.
    assert seen == [((BATCH, 2), (BATCH, FEATURES))] * 6
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
    trainer.fit(Data(val=None), steps=6, log_every=2, eval_every=2, metrics=(Spread([]),),
                validation={"often": Data(val=val_batches(1)).val,
                            "rare": EvalSuite(Data(val=val_batches(2)).val, (Spread([]),), 3)})
    steps = {name: [step for step, scores in tracker.scalars if f"{name}/spread" in scores]
             for name in ("often", "rare")}
    assert steps == {"often": [2, 4, 6], "rare": [3, 6]}


def test_the_validation_split_is_the_one_suite_val():
    """`dataset.val` alone and a one-suite mapping of it are the same run."""
    from dew.training import EvalSuite

    runs = []
    held = Data(val=val_batches(3)).val
    for validation in (None, {"val": held}, {"val": EvalSuite(held, (Spread([]),), 3)}):
        tracker = RecordingTracker()
        make_trainer(objective=Features(), tracker=tracker).fit(
            Data(val=val_batches(3)), steps=6, log_every=2, eval_every=3, metrics=(Spread([]),),
            validation=validation)
        runs.append([(step, scores["val/spread"]) for step, scores in tracker.scalars
                     if "val/spread" in scores])
    assert runs[0] == runs[1] == runs[2] and [step for step, _ in runs[0]] == [3, 6]


@pytest.mark.parametrize("width", [30, 120])
def test_a_fit_on_a_terminal_of_any_width_shows_every_metric(width, monkeypatch):
    """The live panel lays itself out for the terminal it has: one too
    narrow for the sparklines still shows each metric's name and value,
    through the evaluations, to the last frame."""
    screen = io.StringIO()
    evaluation_budgets = []
    render_metrics = display.TrainingDisplay.metrics
    advance_step = display.TrainingDisplay.step

    def metrics(panel, rows, inner, lines, *, evaluation=False):
        if evaluation:
            evaluation_budgets.append(lines)
        return render_metrics(panel, rows, inner, lines, evaluation=evaluation)

    def step(panel, number):
        advance_step(panel, number)
        if number == 2:
            logging.getLogger("dew.training.test").warning("diagnostic above the live panel")

    monkeypatch.setattr(display.TrainingDisplay, "metrics", metrics)
    monkeypatch.setattr(display.TrainingDisplay, "step", step)
    monkeypatch.setattr(display, "terminal", lambda console: True)
    monkeypatch.setattr(display, "Console", lambda: Console(file=screen, width=width, height=40,
                                                            force_terminal=True, color_system=None))
    make_trainer(objective=Features()).fit(Data(val=val_batches(3)), steps=6, log_every=2, eval_every=3,
                                           metrics=(Spread([]),))

    output = screen.getvalue()
    assert "diagnostic above the live panel" in output
    assert "eval val at step" not in output
    assert evaluation_budgets and max(evaluation_budgets) < 40
    last = output.rpartition("dew · ")[2]
    for name in ("loss", "step_time_ms", "spread"):
        assert re.search(rf" {name} +\S", last), last
    assert "val" in last and "step 6" in last, last
    summary = output.rpartition("✓ ")[2]
    assert "val (ema)" in summary and "at 6" in summary and "spread" in summary, summary
    if width == 120:
        assert re.search(r"spread +\S+ +[▁▂▃▄▅▆▇█]{2}", last), last
        assert re.search(r"[▁▂▃▄▅▆▇█]{2}", summary), summary


@pytest.mark.parametrize(("averaged", "label"), [(True, r"val \(ema\)"), (False, "val")])
def test_off_a_terminal_evaluation_keeps_its_plain_line(averaged, label, capsys):
    """An evaluation of the averaged weights says so beside its split."""
    objective = Features()
    if not averaged:
        objective.ema = None
    trainer = make_trainer(objective=objective)
    trainer.fit(Data(val=val_batches(3)), steps=6, log_every=2, eval_every=3, metrics=(Spread([]),))
    output = capsys.readouterr().out
    assert re.search(rf"eval {label} at step 3: spread \S+ \(24 records in \S+ s\)", output), output
    assert re.search(rf"eval {label} at step 6: spread \S+ .+ \(24 records in \S+ s\)", output), output


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
        make_trainer(objective=Features()).fit(Data(val=val_batches(1)), steps=1,
                                               log_every=1, eval_every=1, metrics=(Broken(),))


def test_a_scheduled_validation_pass_with_no_consumer_is_refused():
    """eval_every with neither metrics nor a tracked preview would open
    nothing and report nothing; the contradiction is named before training."""
    with pytest.raises(ValueError, match="nothing consumes"):
        make_trainer().fit(Data(val=val_batches(1)), steps=1, log_every=1, eval_every=1)
    with pytest.raises(ValueError, match="needs a tracker"):
        make_trainer().fit(Data(val=val_batches(1)), steps=1, log_every=1, eval_every=1, preview=True)


def test_a_failing_validation_loader_fails_the_pass():
    class UnreadableSplit:
        def __iter__(self):
            return self

        def __next__(self):
            raise OSError("val.bin: Input/output error")

    with pytest.raises(OSError, match=r"val.bin"):
        make_trainer(objective=Features()).fit(Data(val=UnreadableSplit), steps=1,
                                               log_every=1, eval_every=1, metrics=(Spread([]),))


def test_a_metric_that_reads_a_type_the_objective_does_not_produce_is_an_error():
    class WantsText:
        name = "text"
        reads = str

        def __call__(self, artifact, batch):
            return 0.0

        def merge(self, accumulated, contribution):
            return accumulated + contribution

        def finalize(self, accumulated):
            return accumulated

    with pytest.raises(ValueError, match="reads str"):
        make_trainer(objective=Features()).fit(Data(val=val_batches(1)), steps=1,
                                               log_every=1, eval_every=1, metrics=(WantsText(),))


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def test_the_log_tick_carries_the_loss_the_objective_metrics_and_the_throughput():
    tracker = RecordingTracker()
    make_trainer(tracker=tracker).fit(Data(endless), steps=4, log_every=2)

    ticks = [(step, scalars) for step, scalars in tracker.scalars if "train/loss" in scalars]
    assert [step for step, _ in ticks] == [2, 4]
    for _, scalars in ticks:
        assert scalars["train/probe"] == 1.0
        assert np.isfinite(scalars["train/loss"])
        assert scalars["train/step_time_ms"] > 0
        assert scalars["train/samples_per_sec"] > 0


class ManualClock:
    """Both clocks the trainer reads, advanced by hand."""

    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def perf_counter(self):
        return self.now


def test_the_first_log_tick_measures_steps_not_the_compile(monkeypatch):
    """Every interval, the first one included, reports the time its steps
    took; placement and compile are outside every window and never land in
    train/step_time_ms, and the goodput numbers at the end count them as the
    time to the first step."""
    clock = ManualClock()
    monkeypatch.setattr(trainer_module, "time", clock)
    tracker = RecordingTracker()
    trainer = make_trainer(tracker=tracker)
    place, compile_step = trainer.place, trainer.compile

    def slow_place():
        clock.now += 20.0
        return place()

    def compile_then_time_each_step(*args):
        executable = compile_step(*args)
        clock.now += 100.0

        def timed(*step_args):
            outputs = executable(*step_args)
            clock.now += 1.0
            return outputs
        return timed

    monkeypatch.setattr(trainer, "place", slow_place)
    monkeypatch.setattr(trainer, "compile", compile_then_time_each_step)
    trainer.fit(Data(endless), steps=3, log_every=1)

    ticks = [s for _, s in tracker.scalars if "train/step_time_ms" in s]
    assert [s["train/step_time_ms"] for s in ticks] == pytest.approx([1000.0] * 3)
    # Placement (20), the compile (100) and the first step (1) make the time
    # to the first step; the two steps after it are the 2 of 123 seconds in steps.
    goodput = [(step, s) for step, s in tracker.scalars if "goodput/step_fraction" in s]
    assert [step for step, _ in goodput] == [3]
    assert goodput[0][1]["goodput/time_to_first_step_s"] == pytest.approx(121.0)
    assert goodput[0][1]["goodput/step_fraction"] == pytest.approx(2 / 123)


def test_goodput_counts_evaluations_and_checkpoints_as_time_outside_steps(monkeypatch, tmp_path):
    """Four steps of one second each, a compile of ten, an evaluation of
    five at step two and at the end, a checkpoint write of two at step two
    and at the end: the first step lands at 11, the other three steps are
    the 3 seconds in steps of the 28 the fit took."""
    clock = ManualClock()
    monkeypatch.setattr(trainer_module, "time", clock)
    tracker = RecordingTracker()
    trainer = make_trainer(tmp_path, objective=Features(), tracker=tracker)
    compile_step, evaluate, save = trainer.compile, trainer_module.Evaluation.run, trainer.checkpoints.save

    def compile_then_time_each_step(*args):
        executable = compile_step(*args)
        clock.now += 10.0

        def timed(*step_args):
            outputs = executable(*step_args)
            clock.now += 1.0
            return outputs
        return timed

    def slow_evaluate(*args, **kwargs):
        clock.now += 5.0
        return evaluate(*args, **kwargs)

    def slow_save(*args, **keywords):
        clock.now += 2.0
        return save(*args, **keywords)

    monkeypatch.setattr(trainer, "compile", compile_then_time_each_step)
    monkeypatch.setattr(trainer_module.Evaluation, "run", slow_evaluate)
    monkeypatch.setattr(trainer.checkpoints, "save", slow_save)
    trainer.fit(Data(val=val_batches()), steps=4, log_every=1, eval_every=2, checkpoint_every=2,
                metrics=(Spread([]),))

    goodput = [s for _, s in tracker.scalars if "goodput/step_fraction" in s]
    assert len(goodput) == 1
    assert goodput[0]["goodput/time_to_first_step_s"] == pytest.approx(11.0)
    assert goodput[0]["goodput/step_fraction"] == pytest.approx(3 / 28)


def test_goodput_arithmetic():
    """The fraction is what is left of the wall time after the first step
    and the time outside steps; a fit that ran no step has no first step and
    no time in steps."""
    assert trainer_module.goodput(10.0, 2.0, 3.0) == {
        "goodput/time_to_first_step_s": 2.0, "goodput/step_fraction": 0.5}
    assert trainer_module.goodput(10.0, None, 4.0) == {"goodput/step_fraction": 0.0}
    assert trainer_module.goodput(0.0, None, 0.0) == {"goodput/step_fraction": 0.0}


# --------------------------------------------------------------------------
# Divergence
# --------------------------------------------------------------------------

class Diverging(Regression):
    def loss(self, variables, batch, step):
        loss, aux = super().loss(variables, batch, step)
        return loss * jnp.nan, aux


def test_sustained_non_finite_loss_stops_the_run():
    with pytest.raises(RuntimeError, match="non-finite"):
        make_trainer(objective=Diverging()).fit(Data(endless), steps=8, log_every=1)


def test_a_healthy_run_does_not_trip_the_detector():
    state = make_trainer().fit(Data(endless), steps=6, log_every=1)
    assert int(state.step) == 6


# --------------------------------------------------------------------------
# Rejected mixed-precision steps
# --------------------------------------------------------------------------

class ScaledObjective(Objective):
    """loss = scale * sum(w^2), with the scale carried by the batch so that
    one batch overflows the scaled float32 loss while the params stay sane."""

    def __init__(self):
        self.ema = EMASpec(decay=lambda step: 0.5)

    def init(self, key, variables=None):
        return {"params": {"w": jnp.ones((2,))}}

    def loss(self, variables, batch, step):
        return jnp.sum(variables["params"]["w"] ** 2) * batch["scale"][0], Aux({})


def host(state):
    return np.array(state.variables["params"]["w"]), np.array(state.ema["params"]["w"])


@pytest.mark.parametrize("accum", [1, 2])
def test_a_rejected_dynamic_scale_step_leaves_no_trace(accum):
    trainer = Trainer(ScaledObjective(), optax.sgd(.1), key=jax.random.key(0),
                      accumulation=accum, dynamic_scale=True)
    state = trainer.initial_state()
    good = {"scale": jnp.ones((jax.device_count(),), jnp.float32)}
    bad = {"scale": jnp.full((jax.device_count(),), 1e35, jnp.float32)}
    step = trainer.compile(state, good)
    for _ in range(2 * accum - 1):
        state, *_ = step(state, good)
    w, ema = host(state)
    np.testing.assert_allclose(w, .8, rtol=1e-6)
    np.testing.assert_allclose(ema, .9, rtol=1e-6)
    # the step consumes the state
    counted, advanced, scaled = int(state.step), int(state.microstep), float(state.scale.scale)
    state, _, _, finite, accepted = step(state, bad)
    assert bool(finite) and not bool(accepted)
    assert int(state.step) == counted + 1
    assert int(state.microstep) == advanced
    assert float(state.scale.scale) == scaled / 2
    np.testing.assert_array_equal(state.variables["params"]["w"], w)
    np.testing.assert_array_equal(state.ema["params"]["w"], ema)
    state, *_ = step(state, good)
    w, ema = host(state)
    np.testing.assert_allclose(w, .64, rtol=1e-6)
    np.testing.assert_allclose(ema, .77, rtol=1e-6)
    assert int(state.microstep) == 2 * accum


@pytest.mark.parametrize("dynamic_scale", [False, True])
def test_a_one_microbatch_step_commits_without_a_conditional(dynamic_scale):
    """A GPU conditional whose branch holds a collective, as fsdp's global
    gradient norm is, reads its predicate on the host: the host waited in
    every step until the device reached the commit, and the device idled
    through the next step's host work. The Flowers DiT stepped in 32.5 ms on
    four RTX 3090s under fsdp=4, 15.9 of them host idle, against flaxdiff's
    22.4. A window of one microbatch commits by selects, so its compiled
    step holds no conditional, whatever the device."""
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(.1))
    trainer = Trainer(ScaledObjective(), optimizer, key=jax.random.key(0), dynamic_scale=dynamic_scale)
    state = trainer.initial_state()
    trainer.compile(state, {"scale": jnp.ones((jax.device_count(),), jnp.float32)})
    assert trainer.executable is not None
    # The instruction, not the word: an executable compiled here carries its
    # stack frames, this test's name among them.
    assert not re.search(r"\sconditional\(", trainer.executable.as_text())


def test_mixed_precision_trains_through_fit():
    """Every step of a healthy run is accepted, so the loop counts three
    updates and not three rejections."""
    state = make_trainer(dynamic_scale=True).fit(Data(endless), steps=3, log_every=1)
    assert int(state.step) == 3 and int(state.updates) == 3


# --------------------------------------------------------------------------
# The collections an objective writes back
# --------------------------------------------------------------------------

class Counted(Objective):
    """A `stats` collection the loss advances by one every step."""

    ema = None

    def init(self, key, variables=None):
        return {"params": {"w": jnp.ones((2,))}, "stats": {"seen": jnp.zeros(())}}

    def loss(self, variables, batch, step):
        loss = jnp.sum(variables["params"]["w"] ** 2)
        return loss, Aux({}, variables={"stats": {"seen": variables["stats"]["seen"] + 1}})


def test_aux_variables_are_written_back_into_the_state_and_checkpointed(tmp_path):
    trainer = make_trainer(tmp_path, objective=Counted())
    state = trainer.fit(Data(), steps=3, log_every=1)
    assert float(state.variables["stats"]["seen"]) == 3.0

    restored, _ = trainer.checkpoints.restore()
    assert float(restored["variables"]["stats"]["seen"]) == 3.0


def test_the_optimizer_never_touches_a_non_parameter_collection():
    state = make_trainer(objective=Counted(), optimizer=optax.adamw(1e-2, weight_decay=1.0)).fit(
        Data(endless), steps=2, log_every=1)
    assert float(state.variables["stats"]["seen"]) == 2.0
    assert set(state.opt_state[0].mu) == {"w"}


def test_write_back_refuses_the_params_collection_and_unknown_ones():
    params = {"params": {"w": jnp.ones(())}, "stats": {"seen": jnp.zeros(())}}
    with pytest.raises(ValueError, match="params collection"):
        write_back(params, {"params": {"w": jnp.zeros(())}})
    with pytest.raises(ValueError, match="not a collection"):
        write_back(params, {"cache": {}})
    with pytest.raises(ValueError, match="structure"):
        write_back(params, {"stats": {"other": jnp.zeros(())}})
    assert float(write_back(params, {"stats": {"seen": jnp.ones(())}})["stats"]["seen"]) == 1.0


# --------------------------------------------------------------------------
# The EMA over a subtree
# --------------------------------------------------------------------------

class TwoTrees(Objective):
    """Two independent parameter subtrees, so EMA scoping is observable."""

    def __init__(self, ema):
        self.ema = ema

    def init(self, key, variables=None):
        return {"params": {"tracked": {"w": jnp.ones((2,))},
                           "untracked": {"w": jnp.ones((2,))}}}

    def loss(self, variables, batch, step):
        total = sum(jnp.sum(leaf ** 2) for leaf in jax.tree.leaves(variables))
        return total, Aux({})


def test_the_ema_stores_only_the_selected_subtree_and_merges_over_the_rest():
    objective = TwoTrees(EMASpec(decay=optax.constant_schedule(0.5),
                                 select=under("params", "tracked")))
    state = make_trainer(objective=objective).fit(Data(endless), steps=2, log_every=1)

    assert set(state.ema["params"]) == {"tracked"}
    live = state.variables["params"]
    assert not np.allclose(state.ema["params"]["tracked"]["w"], live["tracked"]["w"])
    merged = merge(state.variables, state.ema)
    assert merged["params"]["untracked"] is live["untracked"]
    np.testing.assert_array_equal(merged["params"]["tracked"]["w"], state.ema["params"]["tracked"]["w"])


def test_select_drops_branches_that_keep_nothing():
    tree = {"params": {"a": 1}, "encoders": {"text": {"table": 2}}}
    assert select(tree, under("params")) == {"params": {"a": 1}}
    with pytest.raises(ValueError, match="no leaf"):
        select(tree, under("nothing"))


def test_ema_update_moves_only_the_leaves_the_average_holds():
    params = {"params": {"tracked": {"w": jnp.full((2,), 3.0)}, "untracked": {"w": jnp.zeros((2,))}}}
    ema = {"params": {"tracked": {"w": jnp.ones((2,))}}}
    updated = ema_update(ema, params, 0.5)
    assert set(updated["params"]) == {"tracked"}
    np.testing.assert_allclose(updated["params"]["tracked"]["w"], 2.0)


# --------------------------------------------------------------------------
# The custom step
# --------------------------------------------------------------------------

class TwoPlayers(Objective):
    """A generator that moves a scalar toward a target and a discriminator
    that tracks the generator: two losses, two parameter groups."""

    ema = None

    def init(self, key, variables=None):
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

            def generator(operand):
                loss, grads = jax.value_and_grad(objective.generator_loss)(trainable, batch)
                updates, gen_state = gen.update(grads["gen"], state.opt_state["gen"], trainable["gen"])
                params = {**trainable, "gen": optax.apply_updates(trainable["gen"], updates)}
                return params, {**state.opt_state, "gen": gen_state}, loss

            def discriminator(operand):
                loss, grads = jax.value_and_grad(objective.discriminator_loss)(trainable, batch)
                updates, disc_state = disc.update(grads["disc"], state.opt_state["disc"], trainable["disc"])
                params = {**trainable, "disc": optax.apply_updates(trainable["disc"], updates)}
                return params, {**state.opt_state, "disc": disc_state}, loss

            params, opt_state, loss = jax.lax.cond(state.microstep % 2 == 0, generator, discriminator, None)
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

    state = trainer().fit(Data(), steps=2, log_every=1)
    # Step 0 moved the generator half way to 1 (lr 0.25 on a gradient of 2(g - 1));
    # step 1 moved the discriminator half way to the generator.
    assert float(state.variables["params"]["gen"]["g"]) == pytest.approx(0.5)
    assert float(state.variables["params"]["disc"]["d"]) == pytest.approx(0.25)
    assert [s["train/player"] for _, s in tracker.scalars if "train/player" in s] == [0.0, 1.0]

    resumed = trainer().fit(Data(), steps=4, log_every=1)
    assert float(resumed.variables["params"]["gen"]["g"]) == pytest.approx(0.75)
    assert float(resumed.variables["params"]["disc"]["d"]) == pytest.approx(0.5)
    assert Checkpoints(str(tmp_path / "gan")).latest == 4


class Alternating(TwoPlayers):
    """`TwoPlayers` under the built-in transaction: even updates train the
    generator and odd ones the discriminator, each with its own copy of the
    optimizer (`Objective.optimizer`), and the EMA follows the generator's
    updates alone (`Objective.averages`)."""

    ema = EMASpec(decay=optax.constant_schedule(0.5), select=under("params", "gen"))

    def loss(self, variables, batch, step):
        return jax.lax.cond(self.generating(step.step), self.generator_loss, self.discriminator_loss,
                            variables["params"], batch)

    @staticmethod
    def generating(update):
        return update % 2 == 0

    def optimizer(self, tx, *, accumulation):
        if accumulation > 1:
            raise ValueError("the players alternate update by update; train with accumulation=1")
        return optax.multi_transform(
            {"gen": optax.conditionally_mask(tx, self.generating),
             "disc": optax.conditionally_mask(tx, lambda update: ~self.generating(update))},
            {"gen": "gen", "disc": "disc"})

    def averages(self, update):
        return self.generating(update)


def test_an_objective_alternates_two_networks_each_with_its_own_optimizer(tmp_path):
    """The same alternation without a custom step. Each network steps only
    on its own updates, from its own Adam state: after four updates the
    generator has taken Adam's steps one and two on its two gradients and the
    discriminator its own, as two separate Adams would, though each
    network's gradient is zero on the other's updates and Adam's momentum
    would move it there. The EMA averages on generator updates only, and a
    resumed run continues both optimizers where they were."""
    tx = optax.adam(0.1)

    def trainer():
        return Trainer(Alternating(), tx, key=jax.random.key(0),
                       layout=Layout(min_shard=1, tolerance=1.0),
                       checkpoints=Checkpoints(str(tmp_path / "players"), keep=2))

    halfway = trainer().fit(Data(), steps=2, log_every=10)
    state = trainer().fit(Data(), steps=4, log_every=10)
    objective = TwoPlayers()

    def own(network, loss, others):
        """`network` stepped alone by its own Adam, the other at `others`."""
        params = {"gen": {"g": jnp.zeros(())}, "disc": {"d": jnp.zeros(())}}
        opt_state = tx.init(params[network])
        history = []
        for other in others:
            grads = jax.grad(loss)({**params, **other}, None)[network]
            update, opt_state = tx.update(grads, opt_state, params[network])
            params = {**params, network: optax.apply_updates(params[network], update)}
            history.append(params[network])
        return history

    generator = own("gen", objective.generator_loss, [{}, {}])
    # The compiled step and the eager loop round once differently: an ulp.
    def near(value):
        return pytest.approx(float(value), rel=1e-6)

    assert float(halfway.variables["params"]["gen"]["g"]) == near(generator[0]["g"])
    assert float(state.variables["params"]["gen"]["g"]) == near(generator[1]["g"])
    first, second = (float(value["g"]) for value in generator)
    discriminator = own("disc", objective.discriminator_loss, [{"gen": {"g": jnp.asarray(first)}},
                                                               {"gen": {"g": jnp.asarray(second)}}])
    assert float(halfway.variables["params"]["disc"]["d"]) == near(discriminator[0]["d"])
    assert float(state.variables["params"]["disc"]["d"]) == near(discriminator[1]["d"])
    # Averaged on updates 0 and 2: half the generator after each, from zero.
    assert float(state.ema["params"]["gen"]["g"]) == near(0.5 * (0.5 * first) + 0.5 * second)
    with pytest.raises(ValueError, match="accumulation=1"):
        Trainer(Alternating(), tx, key=jax.random.key(0), accumulation=2)


GiB = 2**30


def planned_step(temporaries, outputs=0):
    """A compiled step as the fit check reads it: XLA's memory analysis."""
    from types import SimpleNamespace

    return SimpleNamespace(memory_analysis=lambda: SimpleNamespace(
        output_size_in_bytes=outputs, alias_size_in_bytes=0, temp_size_in_bytes=int(temporaries)))


def device_memory(limit, in_use, largest=0, pool=None, platform="gpu"):
    """A device as the fit check reads it: its platform and its allocator's
    memory_stats. Only XLA's GPU pool (BFC) reports pool_bytes."""
    from types import SimpleNamespace

    stats = {"bytes_limit": int(limit), "bytes_in_use": int(in_use), "largest_free_block_bytes": int(largest)}
    if pool is not None:
        stats["pool_bytes"] = int(pool)
    return SimpleNamespace(platform=platform, local_hardware_id=0, memory_stats=lambda: stats)


def test_a_step_fits_where_one_free_block_holds_its_temporaries(monkeypatch):
    """XLA's GPU step takes its temporaries as one allocation, so it fits
    where one free block holds them, whatever share of the limit is left.

    The A100 numbers: the 176M DiT at batch 128 planned 28.3 GiB of a 29.6
    GiB pool, 4.4% of it to spare, and ran 274.6 ms a step without remat; an
    8% reserve sent it to 'dots' at 314.9 ms. Qwen3-1.7B's 'minimal' rung
    needed 11.68 GiB of temporaries beside 13.3 GiB of state, and the 16.3
    GiB free were 5.9 GiB below the state and 10.4 GiB above it: neither
    block held them, and the run ran out of memory on its first step."""
    from dew.training.memory import step_headroom

    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_allocator_spatial_partitioning=false")
    limit = 29.6 * GiB
    whole = device_memory(limit, 2.1 * GiB, largest=limit - 2.1 * GiB, pool=limit)
    assert step_headroom(planned_step(26.2 * GiB), [whole]) > 0
    split = device_memory(limit, 13.3 * GiB, largest=10.4 * GiB, pool=limit)
    assert step_headroom(planned_step(11.68 * GiB), [split]) < 0
    assert step_headroom(planned_step(10 * GiB), [split]) > 0


def test_a_partitioned_pool_needs_room_for_the_temporaries_twice(monkeypatch):
    """XLA partitions a preallocated BFC pool unless the run turns it off.
    There a batch prefetched beside a step's temporaries leaves them no
    block to return to, and the next step needs a second block as large: the
    RTX 4080's 4096-token step, 6.5 GiB of temporaries with 10.45 GiB free in
    one block, failed so in 5 of 16 runs. With the partitioning off it fits."""
    from dew.training import memory as module
    from dew.training.memory import step_headroom

    monkeypatch.setattr(module, "gpu_free_bytes", lambda ordinal: None)

    limit = 13.24 * GiB
    pool = device_memory(limit, 2.79 * GiB, largest=10.45 * GiB, pool=limit)
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
    assert step_headroom(planned_step(6.5 * GiB), [pool]) < 0
    assert step_headroom(planned_step(5 * GiB), [pool]) > 0
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_allocator_spatial_partitioning=false")
    assert step_headroom(planned_step(6.5 * GiB), [pool]) > 0
    growing = device_memory(limit, 2.79 * GiB, largest=GiB, pool=4 * GiB)
    monkeypatch.setenv("XLA_FLAGS", "")
    assert step_headroom(planned_step(6.5 * GiB), [growing]) > 0


def test_a_growing_pool_places_temporaries_in_a_region_it_has_yet_to_take(monkeypatch):
    """A pool that grows (XLA_PYTHON_CLIENT_PREALLOCATE=false) takes a new
    region for an allocation its free blocks cannot hold, up to its limit."""
    from dew.training import memory as module
    from dew.training.memory import step_headroom

    monkeypatch.setattr(module, "gpu_free_bytes", lambda ordinal: None)

    monkeypatch.setenv("XLA_FLAGS", "")
    growing = device_memory(16 * GiB, 3 * GiB, largest=GiB, pool=4 * GiB)
    assert step_headroom(planned_step(10 * GiB), [growing]) > 0
    assert step_headroom(planned_step(13.5 * GiB), [growing]) < 0


def test_a_growing_pool_takes_no_more_of_its_limit_than_the_gpu_has_free(monkeypatch):
    """A pool that grows takes a new region from the GPU, which another
    process may already hold: past what the driver reports free, its limit
    is a number, not memory."""
    from dew.training import memory as module
    from dew.training.memory import step_headroom

    monkeypatch.setenv("XLA_FLAGS", "")
    growing = device_memory(16 * GiB, 3 * GiB, largest=GiB, pool=4 * GiB)
    monkeypatch.setattr(module, "gpu_free_bytes", lambda ordinal: 6 * GiB)
    assert step_headroom(planned_step(5 * GiB), [growing]) > 0
    assert step_headroom(planned_step(10 * GiB), [growing]) < 0
    monkeypatch.setattr(module, "gpu_free_bytes", lambda ordinal: None)
    assert step_headroom(planned_step(10 * GiB), [growing]) > 0


def test_an_allocator_without_a_pool_is_read_by_its_free_bytes(monkeypatch):
    """A TPU's allocator reports no pool, so its free bytes are all the
    check reads."""
    from dew.training.memory import step_headroom

    monkeypatch.setenv("XLA_FLAGS", "")
    tpu = device_memory(16 * GiB, 3 * GiB, platform="tpu")
    assert step_headroom(planned_step(12.9 * GiB), [tpu]) > 0
    assert step_headroom(planned_step(13.1 * GiB), [tpu]) < 0


def test_cuda_async_needs_room_for_the_temporaries_twice(monkeypatch):
    """cuda_async reports no pool and no free block, and its pool can hold
    a step's freed temporaries where the next step cannot reuse them: the
    RTX 4080's 8192-token step, 10.3 GiB of them with 10.45 GiB free, failed
    in 1 of 8 runs at a 0.85 pool. So the step needs room for them twice."""
    from dew.training.memory import step_headroom

    monkeypatch.setenv("XLA_FLAGS", "")
    unpooled = device_memory(13.24 * GiB, 2.79 * GiB)
    assert step_headroom(planned_step(10.31 * GiB), [unpooled]) < 0
    assert step_headroom(planned_step(5 * GiB), [unpooled]) > 0


def test_a_step_fits_beside_the_bytes_the_loop_holds_outside_it(monkeypatch):
    """The batches the loop prefetches beside the step's own are placed
    while it runs, so the step fits only with room for them too."""
    from dew.training.memory import step_headroom

    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_allocator_spatial_partitioning=false")
    pool = device_memory(16 * GiB, 3 * GiB, largest=13 * GiB, pool=16 * GiB)
    assert step_headroom(planned_step(12 * GiB), [pool]) > 0
    assert step_headroom(planned_step(12 * GiB), [pool], held=2 * GiB) < 0


def test_the_fit_check_holds_room_for_the_batches_fit_prefetches(monkeypatch):
    """`fit` queues PREFETCH_DEPTH batches and places one more while a step
    runs, so the check holds room for that many more of the batch the step
    compiles for, as each device holds its share."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.training.distributed import PREFETCH_DEPTH, batch_shardings

    seen = []

    def headroom(executable, devices, held=0):
        seen.append(held)
        return 0

    monkeypatch.setattr(memory, 'step_headroom', headroom)
    model = CausalTransformer(vocab_size=32, emb_features=8, num_layers=1, num_heads=1,
                              mlp_features=16, max_seq_len=8)
    trainer = Trainer(LMObjective(model, seq_len=4), optax.sgd(1e-3), key=jax.random.key(0))
    state, _, _ = trainer.place()
    batch = {'text': jnp.zeros((8, 5), jnp.int32)}
    trainer.compile(state, batch)
    placed = jax.device_put(batch, batch_shardings(trainer.device_mesh, batch))
    assert seen == [(PREFETCH_DEPTH + 1) * placed['text'].addressable_shards[0].data.nbytes]


def test_accumulation_must_be_positive():
    with pytest.raises(ValueError, match="accumulation"):
        make_trainer(accumulation=0)


@pytest.mark.parametrize("tokens", [4096, 8192, 16384])
def test_sm89_step_matches_the_measured_head_without_a_latency_cliff(tmp_path, tokens):
    """Compare real Trainer steps with the measured recipe: unfused whole
    logits at 4096 tokens, fused whole logits at 8192, and a 4096-row tile
    at 16384. At 8192 the fused whole logits hold 10.3 GiB of temporaries
    beside 2.8 GiB of state in the 13.24 GiB pool, and the step has to keep
    them. Fresh processes keep conftest's deterministic XLA flags
    out of the measurement; those flags change which whole-logits step fits.
    The ABBA order and warmed step medians allow 4% noise, below the old 8%,
    18%, and 3x regressions. Children need room for a preallocated pool,
    Dew's BFC allocator as `prepare_process` sets it up.
    """
    devices = jax.devices()
    if len(devices) != 1 or "RTX 4080" not in devices[0].device_kind:
        pytest.skip("measured on one 16 GiB RTX 4080")
    if "xla_gpu_enable_triton_gemm" in os.environ.get("XLA_FLAGS", ""):
        pytest.skip("an explicit Triton option overrides the measured default")
    # Freed arrays can still occupy the parent's growable BFC pool. Check
    # physical free VRAM, not just JAX's live arrays or its preallocation flag.
    fraction = 0.85
    memory = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10).stdout.splitlines()
    if len(memory) != 1:
        pytest.skip("the isolated benchmark needs one physical GPU")
    total, free = (int(value) for value in memory[0].split(","))
    if free < fraction * total:
        pytest.skip(f"{free} MiB free VRAM; the child pool needs {fraction * total:.0f} MiB")
    root = Path(__file__).resolve().parents[1]
    case = {"architecture": "causal_transformer", "config": {
        "vocab_size": 151936, "emb_features": 1024, "num_layers": 2, "num_heads": 16,
        "num_kv_heads": 8, "head_dim": 128, "mlp_features": 3072, "max_seq_len": 1024,
        "tie_embeddings": True}, "dtype": "bfloat16", "batch_size": tokens // 1024,
        "seq_len": 1024}
    flags = " ".join(flag for flag in os.environ.get("XLA_FLAGS", "").split()
                     if not flag.startswith("--xla_gpu_deterministic_ops"))
    environment = {**os.environ, "JAX_PLATFORMS": "cuda", "PYTHONPATH": str(root / "src"),
                   "JAX_DEFAULT_MATMUL_PRECISION": "default", "XLA_FLAGS": flags,
                   "XLA_PYTHON_CLIENT_MEM_FRACTION": str(fraction), "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}
    environment.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    samples = ([], [])
    for index, reference in enumerate((False, True, True, False)):
        objective = {"head_tile": [4096, 8192] if tokens == 16384 else "whole"} if reference else {}
        options = (
            f" --xla_gpu_enable_triton_gemm={'true' if tokens == 8192 else 'false'}" if reference else ""
        )
        record = tmp_path / f"step-{index}.json"
        done = subprocess.run(
            [sys.executable, "tools/benchmark_step.py", "--cases",
             json.dumps([{**case, "objective": objective}]), "--warmup", "6", "--steps", "20",
             "--json-out", str(record)], cwd=root,
            env={**environment, "XLA_FLAGS": flags + options}, capture_output=True, text=True, timeout=180)
        assert done.returncode == 0, done.stdout + done.stderr
        if not reference:
            assert ("with the whole logits kept" in done.stderr) == (tokens == 16384), done.stderr
        row, = json.loads(record.read_text())
        assert row["finite"], row
        samples[int(reference)].append(row["p50_ms"])
    measured, baseline = (float(np.median(values)) for values in samples)
    assert measured < baseline * 1.04, (tokens, measured, baseline)


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

    from dew.objectives.base import Step

    compiled = trainer.compile(abstract, {"text": jax.ShapeDtypeStruct((8, 5), jnp.int32)})
    batch = {"text": jnp.zeros((8, 5), jnp.int32)}
    expected, _ = trainer.objective.scalar_loss(state.variables, batch,
                              Step(state.microstep, jax.random.fold_in(state.key, state.step), None))
    advanced, loss, _, finite, _ = jax.block_until_ready(compiled(state, batch))
    assert loss == pytest.approx(float(expected), rel=1e-6)
    assert int(advanced.step) == 1 and bool(finite)


APART = {"xla_gpu_dot_merger_threshold_mb": 0}


@pytest.mark.parametrize("generation, flags, tokens, frozen, expected", [
    ("sm89", "", 4, False, {**APART, "xla_gpu_enable_triton_gemm": False}),
    ("sm86", "", 4, False, APART),
    ("sm86", "", 128, True, None),
    ("sm89", "", 128, True, {"xla_gpu_enable_triton_gemm": False}),
    ("sm86", "", 132, True, APART),
    ("sm86", "", 1024, False, APART),
    ("sm86", "", math.inf, True, APART),
    ("sm89", "--xla_gpu_dot_merger_threshold_mb=64 --xla_gpu_enable_triton_gemm=true", 4, False, None),
    ("v6e", "", 4, False, None),
    ("cpu", "", 4, True, None),
])
def test_a_gpu_training_step_compiles_its_dots_apart(monkeypatch, generation, flags, tokens, frozen,
                                                       expected):
    """A GPU training step runs dots that share an input apart, where XLA's
    merger would concatenate their weights every step, except a step beside
    frozen weights on 128 tokens or fewer a device (32 rows of 4 at most), which
    keeps the merger as decoding does; the Triton GEMM fusions go off on the
    generations measured faster without them; a flag the run named stands.
    The options are the step's own, so a process that also serves keeps the
    merger there."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    monkeypatch.setattr(memory, 'device_generation', lambda: generation)
    monkeypatch.setenv("XLA_FLAGS", flags)
    model = CausalTransformer(vocab_size=32, emb_features=8, num_layers=1, num_heads=1,
                              mlp_features=16, max_seq_len=8)
    objective = LMObjective(model, seq_len=4)
    assert memory.step_compiler_options(objective, tokens, frozen) == expected


def test_a_frozen_step_tells_the_options_its_tokens_and_split(monkeypatch):
    """The trainer hands the options the tokens one device steps and whether
    the state holds frozen weights, as a LoRA objective's does."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.training import trainer as trainer_module

    seen = []
    monkeypatch.setattr(trainer_module, 'step_compiler_options',
                        lambda objective, tokens, frozen: seen.append((tokens, frozen)))
    model = CausalTransformer(vocab_size=32, emb_features=8, num_layers=1, num_heads=1,
                              mlp_features=16, max_seq_len=8)
    fresh = LMObjective(model, seq_len=4).init(jax.random.key(0))
    queries = freeze(fresh, lambda path: path[-2:] == ("q_proj", "kernel"))
    for variables, frozen in ((fresh, False), (queries, True)):
        trainer = Trainer(LMObjective(model, seq_len=4, variables=variables), optax.sgd(1e-3),
                          key=jax.random.key(0), checkpoints=None, tracker=None)
        state, _, _ = trainer.place()
        trainer.compile(state, {"text": jax.ShapeDtypeStruct((8, 5), jnp.int32)})
        assert seen[-1] == (8 // len(jax.devices()) * 4, frozen)


def test_a_step_without_tokens_compiles_whatever_its_batch_holds():
    """Only an objective that names its row tokens has rows counted for the
    options; another's batch may hold no rows at all, a scalar per field."""
    assert memory.device_tokens(object(), {"weight": jnp.asarray(.5)}, 1) == math.inf


def test_the_step_runs_the_program_it_compiled(monkeypatch, tmp_path):
    """The first execution requests no compilation after the public compile
    call, including a second program retrieved from the persistent cache.
    """
    from jax import monitoring

    from dew.training import trainer as trainer_module

    monkeypatch.setattr(trainer_module, 'step_compiler_options',
                        lambda objective, tokens, frozen: {'xla_embed_ir_in_executable': False})
    trainer, _, _ = held_lm_trainer()
    state, _, _ = trainer.place()
    batch = {"text": jnp.zeros((8, 5), jnp.int32)}
    events = []

    def record(event, **metadata):
        if event == "/jax/compilation_cache/compile_requests_use_cache":
            events.append(event)

    previous_dir = jax.config.jax_compilation_cache_dir
    previous_enabled = jax.config.jax_enable_compilation_cache
    monitoring.register_event_listener(record)
    try:
        jax.config.update("jax_compilation_cache_dir", str(tmp_path))
        jax.config.update("jax_enable_compilation_cache", val=True)
        step = trainer.compile(state, batch)
        assert events, "the public compile must reach the compilation event listener"
        # Input placement is separate from executing the compiled transaction.
        assert trainer.executable is not None
        state, batch = jax.device_put((state, batch), trainer.executable.input_shardings[0])
        jax.block_until_ready((state, batch))
        events.clear()
        jax.block_until_ready(step(state, batch))
        assert not events, events
    finally:
        monitoring.unregister_event_listener(record)
        jax.config.update("jax_compilation_cache_dir", previous_dir)
        jax.config.update("jax_enable_compilation_cache", previous_enabled)

