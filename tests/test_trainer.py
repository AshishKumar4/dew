"""The general trainer on a small objective: the loop, checkpoints, resume,
validation, the rejected mixed-precision step and the custom step.

Nothing here knows a modality. The objective is a two-output affine map, the
data is synthetic, and what is asserted is what the trainer owns: the step
count, what lands on disk and when, what a resume restores, what reaches the
tracker, and what a failure does to the run.
"""

import json
import os
import sys

from flax import linen as nn
from flax.training import dynamic_scale as dynamic_scale_lib
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest

from dew.artifacts import Representations
from dew.objectives.base import Aux, EMASpec, Objective, merge, select, under
from dew.training import Checkpoints, Layout, MeshSpec, Trainer, TrainState
from dew.training import trainer as trainer_module
from dew.training.trainer import ema_update, write_back

BATCH = 8
FEATURES = 3


class Affine(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(2)(x)


class Regression(Objective):
    """Squared error of an affine map against `2 * x[:, :2]`."""

    def __init__(self, ema_decay=0.5):
        self.model = Affine()
        self.ema = EMASpec(decay=optax.constant_schedule(ema_decay))

    def init(self, key):
        return self.model.init(key, jnp.zeros((1, FEATURES)))

    def loss(self, params, batch, step):
        prediction = self.model.apply(params, batch["x"])
        return jnp.mean((prediction - batch["y"]) ** 2), Aux({"probe": jnp.asarray(1.0)})


class Counting:
    """An endless, checkpointable stream whose batches say which they are."""

    def __init__(self, batch=BATCH):
        self.index = 0
        self.batch = batch

    def __iter__(self):
        return self

    def __next__(self):
        rng = np.random.default_rng(self.index)
        self.index += 1
        x = rng.normal(size=(self.batch, FEATURES)).astype(np.float32)
        return {"x": x, "y": 2 * x[:, :2], "index": np.full((self.batch,), self.index - 1)}

    def get_state(self):
        return json.dumps({"index": self.index}).encode()

    def set_state(self, state):
        self.index = json.loads(state)["index"]


class Data:
    """The `Dataset` contract the trainer reads: train, val, batch, records."""

    def __init__(self, train=Counting, val=None, batch=BATCH, records=None):
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


def endless():
    """A stream without get_state."""
    source = Counting()
    while True:
        yield next(source)


def val_batches(count=3):
    def stream():
        source = Counting()
        for _ in range(count):
            yield next(source)
    return stream


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


class RecordingTracker:
    def __init__(self):
        self.scalars = []
        self.artifacts = []

    def log(self, scalars, step):
        self.scalars.append((step, dict(scalars)))

    def artifact(self, value, step):
        self.artifacts.append((step, value))


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

def test_fit_trains_to_the_step_it_was_asked_for():
    state = make_trainer().fit(Data(endless), steps=4, log_every=2)
    assert int(state.step) == 4


def test_a_second_fit_continues_from_the_state_on_disk(tmp_path):
    trainer = make_trainer(tmp_path)
    first = trainer.fit(Data(), steps=3, log_every=1)
    resumed = make_trainer(tmp_path).fit(Data(), steps=5, log_every=1)

    assert int(first.step) == 3 and int(resumed.step) == 5
    assert Checkpoints(str(tmp_path / "run")).latest == 5


def test_constructing_a_trainer_opens_nothing(tmp_path):
    trainer = make_trainer(tmp_path)
    assert not (tmp_path / "run").exists(), "the checkpoint directory was created"
    assert "wandb" not in sys.modules
    assert trainer.mesh == MeshSpec()


def test_the_run_key_and_the_step_derive_every_step_key():
    """Two runs from one key see the same keys at the same steps, so a
    resumed run continues the stream the uninterrupted one draws from."""
    seen = []

    class Recording(Regression):
        def loss(self, params, batch, step):
            seen.append(step.key)
            return super().loss(params, batch, step)

    state = make_trainer(objective=Recording()).fit(Data(endless), steps=2, log_every=1)
    expected = [jax.random.fold_in(state.key, index) for index in range(2)]
    # The step traces once; the keys it saw are traced values of fold_in(key, step).
    assert len(seen) == 1
    assert jnp.array_equal(jax.random.key_data(expected[0]),
                           jax.random.key_data(jax.random.fold_in(state.key, 0)))


def test_the_ema_lags_the_parameters_at_the_configured_decay():
    """One SGD step then one EMA step at decay 0.5: the average sits half way."""
    trainer = make_trainer()
    state = trainer.initial_state()
    step = trainer._default_step()
    batch = next(Counting())

    new_state, _, _, _ = step(state, None, batch)

    before = jax.tree.leaves(state.params)
    after = jax.tree.leaves(new_state.params)
    averaged = jax.tree.leaves(new_state.ema)
    for start, end, ema in zip(before, after, averaged, strict=True):
        np.testing.assert_allclose(np.asarray(ema), 0.5 * np.asarray(start) + 0.5 * np.asarray(end),
                                   rtol=1e-6)
    assert int(new_state.step) == 1


def test_an_objective_without_an_ema_carries_none():
    class Plain(Regression):
        ema = None

        def __init__(self):
            self.model = Affine()

    state = make_trainer(objective=Plain()).fit(Data(endless), steps=1, log_every=1)
    assert state.ema is None


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

    def spy(step, state, position, metrics=None):
        saved.append((step, None if metrics is None else sorted(metrics)))
        return real_save(step, state, position, metrics)

    trainer.checkpoints.save = spy
    trainer.fit(Data(), steps=6, log_every=4, checkpoint_every=2)

    assert saved == [(2, ["loss"]), (4, ["loss"]), (6, ["loss"])]
    assert set(trainer.checkpoints._open().all_steps()) == {2, 4, 6}


def test_checkpoint_every_needs_a_stream_that_reports_its_position():
    with pytest.raises(ValueError, match="get_state"):
        make_trainer().fit(Data(endless), steps=2, checkpoint_every=1)


def test_fit_that_never_trains_checkpoints_step_zero(tmp_path):
    """A run that really ends at step 0 is the one case where step 0 is honest."""
    trainer = make_trainer(tmp_path)
    trainer.fit(Data(), steps=0)
    assert trainer.checkpoints.latest == 0


def test_a_run_past_its_target_is_refused(tmp_path):
    make_trainer(tmp_path).fit(Data(), steps=3, log_every=1)
    with pytest.raises(ValueError, match="past"):
        make_trainer(tmp_path).fit(Data(), steps=2)


def test_the_checkpoint_holds_exactly_the_state_and_the_position(tmp_path):
    """The leaves a checkpoint holds are the state's five and the data
    position; metrics, loss scales and epochs are not among them."""
    trainer = make_trainer(tmp_path)
    trainer.fit(Data(), steps=2, log_every=1)

    stored = trainer.checkpoints._open().item_metadata(2)
    assert set(stored.keys()) == {"step", "params", "opt_state", "ema", "key", "position"}


def test_restore_preserves_the_optimizer_state_the_ema_and_the_key(tmp_path):
    trainer = make_trainer(tmp_path, optimizer=optax.adam(1e-3))
    trained = trainer.fit(Data(), steps=3, log_every=1)

    resumed = make_trainer(tmp_path, optimizer=optax.adam(1e-3))
    state, shardings, position = resumed.place()

    assert int(state.step) == 3, "the step counter was reset"
    for field in ("params", "opt_state", "ema"):
        for before, after in zip(jax.tree.leaves(getattr(trained, field)),
                                 jax.tree.leaves(getattr(state, field)), strict=True):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after), err_msg=field)
    assert jnp.array_equal(jax.random.key_data(trained.key), jax.random.key_data(state.key))
    assert json.loads(position)["index"] == 3


def test_a_resumed_run_continues_the_data_where_it_stopped(tmp_path):
    """A run killed at step 2 and resumed to 4 lands where an uninterrupted
    four-step run lands: the batches after the checkpoint are neither
    replayed nor skipped, so the parameters agree."""
    make_trainer(tmp_path).fit(Data(), steps=2, log_every=1)
    resumed = make_trainer(tmp_path).fit(Data(), steps=4, log_every=1)
    whole = make_trainer().fit(Data(), steps=4, log_every=1)

    for expected, actual in zip(jax.tree.leaves(whole.params),
                                jax.tree.leaves(resumed.params), strict=True):
        np.testing.assert_allclose(np.asarray(expected), np.asarray(actual), rtol=1e-6)
    # The position written at the end names the batch a resume would read next.
    _, position = Checkpoints(str(tmp_path / "run")).restore()
    assert json.loads(position)["index"] == 4


def test_the_best_step_is_the_lowest_loss(tmp_path):
    """The loss rides along with the save, and the lowest one is what orbax
    keeps and reports however the run wanders; a metric-less save is newer
    than the best and must not displace it."""
    checkpoints = Checkpoints(str(tmp_path / "best"), keep=1)
    state = make_trainer().initial_state()
    for step, loss in ((1, 0.9), (2, 0.3), (3, 0.7)):
        checkpoints.save(step, state.replace(step=jnp.asarray(step)), None, {"loss": loss})
    checkpoints.wait()

    assert checkpoints.best == 2
    assert set(checkpoints._open().all_steps()) == {2, 3}

    checkpoints.save(4, state.replace(step=jnp.asarray(4)), None)
    checkpoints.wait()
    assert checkpoints.best == 2
    assert set(checkpoints._open().all_steps()) == {2, 4}
    assert Checkpoints(str(tmp_path / "best")).best == 2, "the metric did not survive a reopen"


def test_a_checkpoint_the_run_cannot_hold_is_refused_with_what_to_do(tmp_path):
    """Swapping the solver or the accumulation has to be a message, not a
    crash from inside orbax's tree walk."""
    make_trainer(tmp_path, optimizer=optax.adam(1e-3)).fit(Data(), steps=1, log_every=1)

    for other in (dict(optimizer=optax.contrib.muon(1e-3)),
                  dict(optimizer=optax.adam(1e-3), accumulation=2)):
        with pytest.raises(ValueError) as error:
            make_trainer(tmp_path, **other).fit(Data(), steps=2)
        message = str(error.value)
        assert "does not fit this run's train state" in message
        assert "opt_state" in message and "gradient accumulation" in message
        assert str(tmp_path / "run") in message


def test_a_bucket_uri_reaches_orbax_verbatim(tmp_path, monkeypatch):
    """A URI has no local form: abspath would make it <cwd>/gs:/bucket."""
    monkeypatch.chdir(tmp_path)
    assert Checkpoints("gs://bucket/checkpoints/run").directory == "gs://bucket/checkpoints/run"
    assert Checkpoints("./relative").directory == str(tmp_path / "relative")
    assert not (tmp_path / "gs:").exists()


class ExplodingManager:
    """Orbax when the filesystem refuses the write.

    Stubbed rather than provoked with a read-only directory: a genuinely
    failed async orbax write leaves a background thread that never joins,
    which hangs interpreter exit and with it the whole test session.
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


def test_a_position_written_by_another_process_count_is_refused(tmp_path):
    """Each position is where one process's shard stopped; a table with two
    rows has no row this single process can take over, and says so."""
    trainer = make_trainer(tmp_path)
    trainer.fit(Data(), steps=1, log_every=1)
    restored, position = trainer.checkpoints.restore()
    row = np.frombuffer(position, np.uint8)
    doubled = {"rows": np.stack([row, row]), "lengths": np.array([len(row)] * 2, np.int64)}
    manager = trainer.checkpoints._open()
    manager.save(2, args=ocp.args.PyTreeSave({**restored, "position": doubled}), force=True)
    manager.wait_until_finished()

    with pytest.raises(ValueError, match="position for each of 2 processes and this run has 1 process"):
        make_trainer(tmp_path).fit(Data(), steps=3)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

class Features(Regression):
    """An objective whose evaluation returns its predictions as representations."""

    artifact = Representations

    def evaluate(self, params, batch, step):
        params = params if step.ema is None else step.ema
        return Representations(features=self.model.apply(params, batch["x"]),
                               labels=batch["index"])


class Spread:
    name = "spread"
    reads = Representations

    def __init__(self, seen):
        self.seen = seen

    def __call__(self, artifact, batch):
        self.seen.append((np.asarray(artifact.features).shape, np.asarray(batch["x"]).shape))
        return float(jnp.std(artifact.features))

    def reduce(self, values):
        return float(np.mean(values))


def test_eval_every_scores_the_validation_split_and_logs_the_artifacts(tmp_path):
    seen = []
    tracker = RecordingTracker()
    trainer = make_trainer(objective=Features(), tracker=tracker)
    trainer.fit(Data(val=val_batches(3)), steps=4, log_every=2, eval_every=2,
                metrics=(Spread(seen),))

    # Two passes: at step 2, and at the end of the run.
    assert seen == [((BATCH, 2), (BATCH, FEATURES))] * 6
    scored = [(step, s) for step, s in tracker.scalars if "val/spread" in s]
    assert [step for step, _ in scored] == [2, 4]
    assert all(np.isfinite(s["val/spread"]) for _, s in scored)
    # The first batch's artifact of each pass reaches the tracker.
    assert [step for step, _ in tracker.artifacts] == [2, 4]
    assert all(isinstance(value, Representations) for _, value in tracker.artifacts)


def test_a_failing_metric_fails_the_validation_pass():
    class Broken:
        name = "broken"
        reads = Representations

        def __call__(self, artifact, batch):
            raise ZeroDivisionError("metric over an empty batch")

        def reduce(self, values):
            return 0.0

    with pytest.raises(ZeroDivisionError):
        make_trainer(objective=Features()).fit(Data(val=val_batches(1)), steps=1,
                                               log_every=1, eval_every=1, metrics=(Broken(),))


def test_a_failing_validation_loader_fails_the_pass():
    class UnreadableSplit:
        def __iter__(self):
            return self

        def __next__(self):
            raise OSError("val.bin: Input/output error")

    with pytest.raises(OSError, match="val.bin"):
        make_trainer(objective=Features()).fit(Data(val=UnreadableSplit), steps=1,
                                               log_every=1, eval_every=1)


def test_a_metric_that_reads_a_type_the_objective_does_not_produce_is_an_error():
    class WantsText:
        name = "text"
        reads = str

        def __call__(self, artifact, batch):
            return 0.0

        def reduce(self, values):
            return 0.0

    with pytest.raises(ValueError, match="reads str"):
        make_trainer(objective=Features()).fit(Data(val=val_batches(1)), steps=1,
                                               log_every=1, eval_every=1, metrics=(WantsText(),))


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def test_the_log_tick_carries_the_loss_the_objective_metrics_and_the_throughput():
    tracker = RecordingTracker()
    make_trainer(tracker=tracker).fit(Data(endless), steps=4, log_every=2)

    steps = [step for step, _ in tracker.scalars]
    assert steps == [2, 4]
    for _, scalars in tracker.scalars:
        assert scalars["train/probe"] == 1.0
        assert np.isfinite(scalars["train/loss"])
        assert scalars["train/step_time_ms"] > 0
        assert scalars["train/samples_per_sec"] > 0


class ManualClock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now


def test_the_first_log_tick_measures_steps_not_the_compile(monkeypatch):
    """Every interval, the first one included, reports the time its steps
    took, so the compile never lands in train/step_time_ms."""
    clock = ManualClock()
    monkeypatch.setattr(trainer_module, "time", clock)
    tracker = RecordingTracker()
    trainer = make_trainer(tracker=tracker)
    compile_step = trainer.compile

    def compile_then_time_each_step(*args):
        executable = compile_step(*args)
        clock.now += 100.0

        def timed(*step_args):
            outputs = executable(*step_args)
            clock.now += 1.0
            return outputs
        return timed

    monkeypatch.setattr(trainer, "compile", compile_then_time_each_step)
    trainer.fit(Data(endless), steps=3, log_every=1)

    assert [s["train/step_time_ms"] for _, s in tracker.scalars] == pytest.approx([1000.0] * 3)


def test_only_process_zero_logs_and_every_process_validates(monkeypatch):
    """A tracker on another process receives nothing, while the validation
    pass runs everywhere because its collectives need every process."""
    monkeypatch.setattr(jax, "process_index", lambda: 1)
    seen = []
    tracker = RecordingTracker()
    make_trainer(objective=Features(), tracker=tracker).fit(
        Data(val=val_batches(1)), steps=2, log_every=1, eval_every=1, metrics=(Spread(seen),))
    assert tracker.scalars == [] and tracker.artifacts == []
    assert len(seen) == 2


# --------------------------------------------------------------------------
# Divergence
# --------------------------------------------------------------------------

class Diverging(Regression):
    def loss(self, params, batch, step):
        loss, aux = super().loss(params, batch, step)
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

    def init(self, key):
        return {"params": {"w": jnp.ones((2,))}}

    def loss(self, params, batch, step):
        return jnp.sum(params["params"]["w"] ** 2) * batch["scale"][0], Aux({})


def host(state):
    return (np.array(state.params["params"]["w"]),
            np.array(state.ema["params"]["w"]),
            int(state.step))


@pytest.mark.parametrize("accum", [1, 2])
def test_a_rejected_dynamic_scale_step_leaves_no_trace(accum):
    """A step whose scaled gradients overflowed is skipped, and skipped means
    all of it. The params and the optimizer state are held; the step counter
    and the EMA have to be held with them, or a rejected step ages every
    schedule and averages in params that were never updated."""
    trainer = Trainer(ScaledObjective(), optax.sgd(0.1), key=jax.random.key(0),
                      accumulation=accum, dynamic_scale=True)
    step = trainer._default_step()
    state, scale = trainer.initial_state(), dynamic_scale_lib.DynamicScale()
    good = {"scale": jnp.ones((1,), jnp.float32)}
    # 1e35 * sum(w^2) * the 65536 loss scale is past float32's max.
    bad = {"scale": jnp.full((1,), 1e35, jnp.float32)}

    # One landed update (w = 1 - 0.1 * 2, ema = 0.5 + 0.5 * 0.8), then the
    # micro-steps leading up to the next one, so the rejected step is the one
    # whose update would have landed.
    for _ in range(2 * accum - 1):
        state, scale, _, _ = step(state, scale, good)
    w, ema, count = host(state)
    np.testing.assert_allclose(w, 0.8, rtol=1e-6)
    np.testing.assert_allclose(ema, 0.9, rtol=1e-6)
    assert count == 2 * accum - 1

    state, scale, loss, _ = step(state, scale, bad)
    assert not bool(jnp.isfinite(loss))
    held_w, held_ema, held_count = host(state)
    np.testing.assert_array_equal(held_w, w)
    np.testing.assert_array_equal(held_ema, ema)
    assert held_count == count

    state, scale, _, _ = step(state, scale, good)
    w, ema, count = host(state)
    np.testing.assert_allclose(w, 0.64, rtol=1e-6)    # 0.8 - 0.1 * 2 * 0.8
    np.testing.assert_allclose(ema, 0.77, rtol=1e-6)  # 0.5 * 0.9 + 0.5 * 0.64
    assert count == 2 * accum


def test_mixed_precision_trains_through_fit():
    state = make_trainer(dynamic_scale=True).fit(Data(endless), steps=3, log_every=1)
    assert int(state.step) == 3


# --------------------------------------------------------------------------
# The collections an objective writes back
# --------------------------------------------------------------------------

class Counted(Objective):
    """A `stats` collection the loss advances by one every step."""

    ema = None

    def init(self, key):
        return {"params": {"w": jnp.ones((2,))}, "stats": {"seen": jnp.zeros(())}}

    def loss(self, params, batch, step):
        loss = jnp.sum(params["params"]["w"] ** 2)
        return loss, Aux({}, variables={"stats": {"seen": params["stats"]["seen"] + 1}})


def test_aux_variables_are_written_back_into_the_state_and_checkpointed(tmp_path):
    trainer = make_trainer(tmp_path, objective=Counted())
    state = trainer.fit(Data(), steps=3, log_every=1)
    assert float(state.params["stats"]["seen"]) == 3.0

    restored, _ = trainer.checkpoints.restore()
    assert float(restored["params"]["stats"]["seen"]) == 3.0


def test_the_optimizer_never_touches_a_non_parameter_collection():
    state = make_trainer(objective=Counted(), optimizer=optax.adamw(1e-2, weight_decay=1.0)).fit(
        Data(endless), steps=2, log_every=1)
    assert float(state.params["stats"]["seen"]) == 2.0
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

    def init(self, key):
        return {"params": {"tracked": {"w": jnp.ones((2,))},
                           "untracked": {"w": jnp.ones((2,))}}}

    def loss(self, params, batch, step):
        total = sum(jnp.sum(leaf ** 2) for leaf in jax.tree.leaves(params))
        return total, Aux({})


def test_the_ema_stores_only_the_selected_subtree_and_merges_over_the_rest():
    objective = TwoTrees(EMASpec(decay=optax.constant_schedule(0.5),
                                 select=under("params", "tracked")))
    state = make_trainer(objective=objective).fit(Data(endless), steps=2, log_every=1)

    assert set(state.ema["params"]) == {"tracked"}
    live = state.params["params"]
    assert not np.allclose(state.ema["params"]["tracked"]["w"], live["tracked"]["w"])
    merged = merge(state.params, state.ema)
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


def test_ema_decay_follows_the_update_schedule():
    """I-JEPA's momentum ramp: the trainer reads decay at the update count."""
    ema = EMASpec(decay=optax.linear_schedule(0.996, 1.0, transition_steps=100))
    assert float(ema.decay(0)) == pytest.approx(0.996)
    assert float(ema.decay(100)) == pytest.approx(1.0)
    assert float(ema.decay(50)) == pytest.approx(0.998)


# --------------------------------------------------------------------------
# The custom step
# --------------------------------------------------------------------------

class TwoPlayers(Objective):
    """A generator that moves a scalar toward a target and a discriminator
    that tracks the generator: two losses, two parameter groups."""

    ema = None

    def init(self, key):
        return {"params": {"gen": {"g": jnp.zeros(())}, "disc": {"d": jnp.zeros(())}}}

    def loss(self, params, batch, step):
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
            trainable = state.params["params"]

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

            params, opt_state, loss = jax.lax.cond(state.step % 2 == 0, generator, discriminator, None)
            new_state = state.replace(step=state.step + 1, opt_state=opt_state,
                                      params={**state.params, "params": params})
            return new_state, loss, Aux({"player": (state.step % 2).astype(jnp.float32)})
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
    assert float(state.params["params"]["gen"]["g"]) == pytest.approx(0.5)
    assert float(state.params["params"]["disc"]["d"]) == pytest.approx(0.25)
    assert [s["train/player"] for _, s in tracker.scalars] == [0.0, 1.0]

    resumed = trainer().fit(Data(), steps=4, log_every=1)
    assert float(resumed.params["params"]["gen"]["g"]) == pytest.approx(0.75)
    assert float(resumed.params["params"]["disc"]["d"]) == pytest.approx(0.5)
    assert Checkpoints(str(tmp_path / "gan")).latest == 4


def test_accumulation_must_be_positive():
    with pytest.raises(ValueError, match="accumulation"):
        make_trainer(accumulation=0)
