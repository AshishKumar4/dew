"""Trainer smoke tests: training without wandb, and checkpoint save/restore.

These run the real ObjectiveTrainer on CPU with a tiny DiT and an
unconditional synthetic dataset - the same code path a real run takes, minus
wandb and conditioning encoders.
"""

import os

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest

from dew.eval.common import EvaluationMetric
from dew.inputs import DiffusionInputConfig
from dew.nn.backbones.dit import SimpleDiT
from dew.diffusion.transforms import get_diffusion_preset
from dew.objectives.base import EMASpec, Objective
from dew.training import ObjectiveTrainer, SimpleTrainer
from dew.training import objective_trainer as gdt
from dew.training import trainer as trainer_module
from dew.training.distributed import DevicePrefetchIterator
from dew.checkpoints.utils import get_latest_checkpoint

RES = 8


def make_trainer(tmp_path, name="smoke", load_from_checkpoint=None, **kwargs):
    train_schedule, _, transform = get_diffusion_preset("edm")
    return ObjectiveTrainer(
        model=SimpleDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1),
        optimizer=optax.adam(1e-3),
        noise_schedule=train_schedule,
        model_output_transform=transform,
        input_config=DiffusionInputConfig(
            sample_data_key="image",
            sample_data_shape=(RES, RES, 3),
            conditions=[],
        ),
        rngs=jax.random.PRNGKey(0),
        name=name,
        wandb_config=None,
        distributed_training=False,
        checkpoint_base_path=str(tmp_path),
        load_from_checkpoint=load_from_checkpoint,
        **kwargs,
    )


def batch_iterator():
    # uint8-range images, as the data pipeline provides them
    images = np.tile(np.linspace(0, 255, RES, dtype=np.float32)[None, :, None, None], (8, 1, RES, 3))
    while True:
        yield {"image": jnp.asarray(images)}


def test_fit_without_wandb(tmp_path):
    trainer = make_trainer(tmp_path)
    data = {"train": batch_iterator, "train_len": 32, "local_batch_size": 8}
    state = trainer.fit(data, training_steps_per_epoch=4, epochs=1, val_steps_per_epoch=0)
    assert state is not None
    assert int(state.step) == 4


def test_save_writes_checkpoint(tmp_path):
    """The checkpoint has to land under the step it was asked for."""
    trainer = make_trainer(tmp_path)
    trainer.save(epoch=0, step=1)
    # Saving is async to keep it off the training loop; this is the barrier
    trainer.wait_for_checkpoints()
    assert trainer.checkpointer.latest_step() == 1


def test_restore_preserves_optimizer_state(tmp_path):
    trainer = make_trainer(tmp_path)

    # One real update so the adam moments, step counter and EMA are non-trivial
    grads = jax.tree.map(jnp.ones_like, trainer.state.params)
    trainer.state = trainer.state.apply_gradients(grads=grads).apply_ema(0.99)
    trainer.save(epoch=0, step=1)
    trainer.wait_for_checkpoints()

    restored = make_trainer(tmp_path, load_from_checkpoint=trainer.checkpoint_path())
    assert int(restored.state.step) == 1, "optimizer step counter was reset"

    old_opt = jax.tree.leaves(trainer.state.opt_state)
    new_opt = jax.tree.leaves(restored.state.opt_state)
    assert all(np.allclose(a, b) for a, b in zip(old_opt, new_opt)), "adam moments were reset"

    old_ema = jax.tree.leaves(trainer.state.ema_params)
    new_ema = jax.tree.leaves(restored.state.ema_params)
    assert all(np.allclose(a, b) for a, b in zip(old_ema, new_ema)), "ema params were reset"


def test_fit_video_model(tmp_path):
    """Video end to end: 5D batches through the real trainer and the video DiT."""
    from dew.nn.backbones.video_dit import VideoDiT

    train_schedule, _, transform = get_diffusion_preset("edm")
    trainer = ObjectiveTrainer(
        model=VideoDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1),
        optimizer=optax.adam(1e-3),
        noise_schedule=train_schedule,
        model_output_transform=transform,
        input_config=DiffusionInputConfig(
            sample_data_key="video",
            sample_data_shape=(3, RES, RES, 3),
            conditions=[],
        ),
        rngs=jax.random.PRNGKey(0),
        name="video-smoke",
        wandb_config=None,
        distributed_training=False,
        checkpoint_base_path=str(tmp_path),
    )

    def video_batches():
        frames = np.tile(np.linspace(0, 255, RES, dtype=np.float32)[None, None, :, None, None],
                         (4, 3, 1, RES, 3))
        while True:
            yield {"video": jnp.asarray(frames)}

    data = {"train": video_batches, "train_len": 16, "local_batch_size": 4}
    state = trainer.fit(data, training_steps_per_epoch=2, epochs=1, val_steps_per_epoch=0)
    assert int(state.step) == 2


# --------------------------------------------------------------------------
# The end-of-run checkpoint
# --------------------------------------------------------------------------

def test_fit_checkpoints_the_final_step(tmp_path):
    """The last save of a run must carry the step the run ended on.

    The in-loop save only fires on an epoch that improved the best loss, so
    pinning the best loss out of reach leaves the end-of-run save as the only
    one, which is the case where the final weights would otherwise land under
    step 0, behind whatever older checkpoint a resume would then pick.
    """
    trainer = make_trainer(tmp_path)
    trainer.best_loss = -1.0
    data = {"train": batch_iterator, "train_len": 32, "local_batch_size": 8}
    trainer.fit(data, training_steps_per_epoch=2, epochs=2, val_steps_per_epoch=0)

    assert trainer.checkpointer.latest_step() == 4
    assert set(trainer.checkpointer.all_steps()) == {4}
    assert not os.path.exists(os.path.join(trainer.checkpoint_path(), "0"))


def test_fit_skips_the_final_save_when_the_loop_already_wrote_that_step(tmp_path):
    trainer = make_trainer(tmp_path, name="skip")
    written = []
    real_save = trainer.save

    def spy(epoch=0, step=0, state=None, rngstate=None, metrics=None):
        written.append(step)
        return real_save(epoch=epoch, step=step, state=state, rngstate=rngstate,
                         metrics=metrics)

    trainer.save = spy
    data = {"train": batch_iterator, "train_len": 32, "local_batch_size": 8}
    trainer.fit(data, training_steps_per_epoch=4, epochs=1, val_steps_per_epoch=0)

    # One save, from the epoch's best-loss branch; the tail must not add a second
    assert written == [4]


def test_checkpoint_every_steps_saves_on_its_own_cadence(tmp_path):
    """Step-based checkpointing was implemented in the loop but never reachable
    from fit(), and it only fired on steps that were also logging ticks."""
    trainer = make_trainer(tmp_path, name="cadence", max_checkpoints_to_keep=4)
    # The epoch's best-loss branch would otherwise add a save of its own
    trainer.best_loss = -1.0
    written = []
    real_save = trainer.save

    def spy(epoch=0, step=0, state=None, rngstate=None, metrics=None):
        written.append((step, metrics))
        return real_save(epoch=epoch, step=step, state=state, rngstate=rngstate,
                         metrics=metrics)

    trainer.save = spy
    data = {"train": batch_iterator, "train_len": 32, "local_batch_size": 8}
    trainer.fit(data, training_steps_per_epoch=6, epochs=1, val_steps_per_epoch=0,
                checkpoint_every_steps=2)
    trainer.wait_for_checkpoints()

    # 2 and 4 mid-epoch, then 6 from the loop rather than a duplicate final save
    assert written == [(2, None), (4, None), (6, None)]
    assert trainer.last_saved_step == 6
    assert set(trainer.checkpointer.all_steps()) == {2, 4, 6}
    assert trainer.checkpointer.best_step() is None, "a cadence save became the best"
def test_checkpoint_at_epoch_boundary_records_the_epoch_loss(tmp_path):
    trainer = make_trainer(tmp_path, name="boundary", max_checkpoints_to_keep=4)
    data = {"train": batch_iterator, "train_len": 32, "local_batch_size": 8}

    trainer.fit(data, training_steps_per_epoch=2, epochs=1,
                val_steps_per_epoch=0, checkpoint_every_steps=2)

    assert trainer.checkpointer.all_steps() == [2]
    assert trainer.checkpointer.best_step() == 2


# --------------------------------------------------------------------------
# Which checkpoint is the best one
# --------------------------------------------------------------------------

def test_best_step_is_the_lowest_epoch_loss(tmp_path):
    """The epoch loss rides along with the save, and the lowest one is what
    orbax keeps and reports however the run wanders."""
    trainer = make_trainer(tmp_path, name="best", max_checkpoints_to_keep=1)
    for step, loss in ((1, 0.9), (2, 0.3), (3, 0.7)):
        trainer.save(epoch=step, step=step, metrics={"loss": loss})
    trainer.wait_for_checkpoints()

    assert trainer.checkpointer.best_step() == 2
    # The newest step is kept for resuming, the lowest loss for publishing
    assert set(trainer.checkpointer.all_steps()) == {2, 3}

    # A metric-less save is newer than the best and must not displace it
    trainer.save(epoch=4, step=4)
    trainer.wait_for_checkpoints()
    assert trainer.checkpointer.best_step() == 2
    assert set(trainer.checkpointer.all_steps()) == {2, 4}

    reopened = make_trainer(tmp_path, name="best",
                            load_from_checkpoint=str(tmp_path / "best"))
    assert reopened.checkpointer.best_step() == 2, "the metric did not survive a reopen"


def test_a_populated_checkpoint_directory_is_not_written_over(tmp_path):
    """A second run into the same directory stops before it trains, because orbax
    refuses to overwrite a step. Resuming and starting fresh are both fine;
    neither is guessed."""
    trainer = make_trainer(tmp_path, name="taken")
    trainer.save(epoch=0, step=2)
    trainer.wait_for_checkpoints()

    with pytest.raises(ValueError, match="already holds checkpoints up to step 2"):
        make_trainer(tmp_path, name="taken")
    with pytest.raises(ValueError, match="already holds checkpoints up to step 2"):
        make_trainer(tmp_path, name="taken", checkpoint_step=2)

    resumed = make_trainer(tmp_path, name="taken",
                           load_from_checkpoint=trainer.checkpoint_path())
    assert resumed.latest_step == 2
    # A directory of its own is the other way out, and an empty one is fine
    assert make_trainer(tmp_path, name="untaken").checkpointer.latest_step() is None


def test_resume_from_a_checkpoint_that_stored_its_own_best_state(tmp_path):
    """Checkpoints holding a second train state under 'best_state' still
    resume; the copy is skipped rather than restored."""
    trainer = make_trainer(tmp_path, name="old-layout")
    trainer.state = trainer.state.apply_gradients(
        grads=jax.tree.map(jnp.ones_like, trainer.state.params)).apply_ema(0.99)

    written = str(tmp_path / "old-layout-checkpoint")
    manager = ocp.CheckpointManager(
        written, options=ocp.CheckpointManagerOptions(create=True))
    manager.save(2, args=ocp.args.PyTreeSave({
        'rngs': trainer.get_rngstate(),
        'state': trainer.get_state(),
        'best_state': trainer.get_state(),
        'best_loss': np.array(0.25),
        'epoch': 1,
    }), force=True)
    manager.wait_until_finished()
    manager.close()

    restored = make_trainer(tmp_path, name="resumed", load_from_checkpoint=written)
    assert restored.latest_step == 2
    assert int(restored.state.step) == 1, "the train state was not restored"
    assert restored.best_loss == 0.25
    assert not hasattr(restored, "best_state")
    for before, after in zip(jax.tree.leaves(trainer.state.params),
                             jax.tree.leaves(restored.state.params)):
        np.testing.assert_allclose(np.asarray(before), np.asarray(after))


def test_fit_that_never_trains_checkpoints_step_zero(tmp_path):
    """A run that really ends at step 0 is the one case where step 0 is honest."""
    trainer = make_trainer(tmp_path, name="nosteps")
    data = {"train": batch_iterator, "train_len": 32, "local_batch_size": 8}
    trainer.fit(data, training_steps_per_epoch=2, epochs=0, val_steps_per_epoch=0)

    assert trainer.checkpointer.latest_step() == 0


def test_get_latest_checkpoint_ignores_non_step_entries(tmp_path):
    """Orbax leaves locks, metadata and interrupted tmp directories behind, and
    a step directory can exist while still being empty."""
    for name in ("1", "10", "2"):
        os.makedirs(tmp_path / name)
        (tmp_path / name / "manifest.ocdbt").write_bytes(b"x")
    os.makedirs(tmp_path / "11")                       # written, then interrupted
    os.makedirs(tmp_path / "12.orbax-checkpoint-tmp")
    (tmp_path / "_CHECKPOINT_METADATA").write_text("{}")
    (tmp_path / "descriptor").write_text("")

    assert get_latest_checkpoint(str(tmp_path)) == str(tmp_path / "10")


def test_get_latest_checkpoint_reports_an_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_latest_checkpoint(str(tmp_path))


# --------------------------------------------------------------------------
# Where the checkpoints go
# --------------------------------------------------------------------------

class RecordingManager:
    """Orbax as far as the constructor reads it, remembering where it was pointed.

    A bucket URI is the one directory the trainer must hand over untouched,
    and a test that let orbax open it would need a bucket.
    """

    def __init__(self, directory, **kwargs):
        self.directory = directory

    def latest_step(self):
        return None


def test_a_bucket_uri_reaches_orbax_verbatim(tmp_path, monkeypatch):
    """`--trainer.checkpoint-fs gcs` prefixes the directory with gs://, and
    checkpoint_path used to run it through abspath and makedirs: orbax got
    <cwd>/gs:/bucket/... and a local directory named `gs:` appeared under the
    working directory. A URI has no local form and goes through as it is."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(trainer_module.ocp, "CheckpointManager", RecordingManager)
    trainer = make_trainer("gs://bucket/checkpoints", name="Bucket Run")

    assert trainer.checkpoint_path() == "gs://bucket/checkpoints/bucket_run"
    assert trainer.checkpointer.directory == "gs://bucket/checkpoints/bucket_run"
    assert not (tmp_path / "gs:").exists(), "a local directory named gs: was created"


def test_a_relative_checkpoint_path_resumes(tmp_path, monkeypatch):
    """`--trainer.load-from-checkpoint ./checkpoints/<run>` is what the README
    shows. checkpoint_path resolves its own directory; load handed the path
    straight to orbax, which refuses a relative one."""
    trainer = make_trainer(tmp_path, name="relative")
    trainer.save(epoch=0, step=2)
    trainer.wait_for_checkpoints()

    monkeypatch.chdir(tmp_path)
    resumed = make_trainer(tmp_path, name="relative-resumed", load_from_checkpoint="./relative")
    assert resumed.latest_step == 2
    assert resumed.loaded_checkpoint_path == str(tmp_path / "relative" / "2")



class ExplodingCheckpointer:
    """Orbax when the filesystem refuses the write.

    Stubbed rather than provoked with a read-only directory: a genuinely
    failed *async* orbax write leaves a background thread that never joins,
    which hangs interpreter exit and with it the whole test session.
    """

    def save(self, *args, **kwargs):
        raise OSError("No space left on device")

    def wait_until_finished(self):
        pass


def test_save_propagates_persistence_failures(tmp_path):
    """A checkpoint that did not get written is data loss, not a log line."""
    trainer = make_trainer(tmp_path, name="unwritable")
    trainer.checkpointer = ExplodingCheckpointer()
    with pytest.raises(OSError):
        trainer.save(epoch=0, step=1)
    assert trainer.last_saved_step is None


def test_registry_failure_leaves_the_checkpoint_alone(tmp_path, monkeypatch):
    """Publishing to wandb is reporting, not persistence: it may fail, but it
    must neither abort the save nor delete the local checkpoint."""
    trainer = make_trainer(tmp_path, name="registry")
    trainer.wandb = object()  # truthy: enough to enter the publishing branch

    def unavailable(_path):
        raise FileNotFoundError("checkpoint already handed to the registry")

    monkeypatch.setattr(gdt, "get_latest_checkpoint", unavailable)
    trainer.save(epoch=0, step=3)
    trainer.wait_for_checkpoints()

    assert trainer.checkpointer.latest_step() == 3
    assert os.path.exists(os.path.join(trainer.checkpoint_path(), "3"))


# --------------------------------------------------------------------------
# Validation failures
# --------------------------------------------------------------------------

class Affine(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(2)(x)


def test_a_failing_metric_fails_the_validation_pass(tmp_path):
    """A metric that raises has to take the pass down with it. Printed and
    swallowed, the pass ends with no scores and the run carries on as if it
    had been evaluated."""
    def divide_by_zero(artifacts, batch):
        raise ZeroDivisionError("metric over an empty batch")

    trainer = make_trainer(tmp_path, name="failing-metric", eval_metrics=[
        EvaluationMetric(function=divide_by_zero, name="broken")])
    with pytest.raises(ZeroDivisionError):
        trainer.validation_loop(trainer.state, trainer._define_validation_step(),
                                batch_iterator, 1, 0)


class UnreadableSplit:
    """A validation split whose first record cannot be read."""

    def __iter__(self):
        return self

    def __next__(self):
        raise OSError("val.bin: Input/output error")


def test_a_failing_validation_loader_fails_the_pass(tmp_path):
    """A split that cannot be read has the same duty as a step that cannot
    run. The train side already stops on a record it cannot read; the
    validation side printed the error and scored nothing."""
    trainer = make_trainer(tmp_path, name="failing-loader")
    with pytest.raises(OSError, match="val.bin"):
        trainer.validation_loop(trainer.state, trainer._define_validation_step(),
                                UnreadableSplit, 1, 0)


def test_a_failing_validation_step_fails_the_base_pass(tmp_path):
    """SimpleTrainer runs its own loop, with the same duty."""
    trainer = SimpleTrainer(
        model=Affine(), input_shapes={"x": (3,)}, optimizer=optax.sgd(0.1),
        rngs=jax.random.PRNGKey(0), name="base", distributed_training=False,
        checkpoint_base_path=str(tmp_path))

    def divide_by_zero(state, batch):
        raise ZeroDivisionError("validation over an empty batch")

    with pytest.raises(ZeroDivisionError):
        trainer.validation_loop(trainer.state, divide_by_zero, None, 1, 0)


# --------------------------------------------------------------------------
# Rejected mixed-precision steps
# --------------------------------------------------------------------------

class ScaledObjective(Objective):
    """loss = scale * sum(w^2), with the scale carried by the batch so that
    one batch overflows the scaled float32 loss while the params stay sane."""

    input_shapes = {"scale": ()}

    def __init__(self):
        self.ema = EMASpec(decay=lambda step: 0.5)

    def init_params(self, rng):
        return {"params": {"w": jnp.ones((2,))}}

    def loss(self, params, ema_params, batch, rng, step):
        return jnp.sum(params["params"]["w"] ** 2) * batch["scale"][0], {}

    def make_validation_step(self, **kwargs):
        return lambda val_state, batch: None


def host(state):
    # Copies, because the step donates the state it is handed.
    return (np.array(state.params["params"]["w"]),
            np.array(state.ema_params["params"]["w"]),
            int(state.step))


@pytest.mark.parametrize("accum", [1, 2])
def test_a_rejected_dynamic_scale_step_leaves_no_trace(tmp_path, accum):
    """A step whose scaled gradients overflowed is skipped, and skipped means
    all of it. The params and the optimizer state are held; the step counter
    and the EMA have to be held with them, or a rejected step ages every
    schedule and averages in params that were never updated."""
    optimizer = optax.sgd(0.1)
    if accum > 1:
        optimizer = optax.MultiSteps(optimizer, every_k_schedule=accum)
    trainer = ObjectiveTrainer(
        model=Affine(), optimizer=optimizer, rngs=jax.random.PRNGKey(0),
        objective=ScaledObjective(), grad_accum_steps=accum, name="rejected",
        checkpoint_base_path=str(tmp_path), distributed_training=False,
        use_dynamic_scale=True)
    train_step = trainer._define_train_step(batch_size=1)
    good = {"scale": jnp.ones((1,), jnp.float32)}
    # 1e35 * sum(w^2) * the 65536 loss scale is past float32's max.
    bad = {"scale": jnp.full((1,), 1e35, jnp.float32)}

    state, rng = trainer.state, trainer.rngstate
    # One landed update (w = 1 - 0.1 * 2, ema = 0.5 + 0.5 * 0.8), then the
    # micro-steps leading up to the next one, so the rejected step is the one
    # whose update would have landed.
    for _ in range(2 * accum - 1):
        state, _, _, rng, _ = train_step(state, rng, good)
    w, ema, step = host(state)
    np.testing.assert_allclose(w, 0.8, rtol=1e-6)
    np.testing.assert_allclose(ema, 0.9, rtol=1e-6)
    assert step == 2 * accum - 1

    state, _, _, rng, is_finite = train_step(state, rng, bad)
    assert not bool(is_finite)
    held_w, held_ema, held_step = host(state)
    np.testing.assert_array_equal(held_w, w)
    np.testing.assert_array_equal(held_ema, ema)
    assert held_step == step

    state, *_ = train_step(state, rng, good)
    w, ema, step = host(state)
    np.testing.assert_allclose(w, 0.64, rtol=1e-6)    # 0.8 - 0.1 * 2 * 0.8
    np.testing.assert_allclose(ema, 0.77, rtol=1e-6)  # 0.5 * 0.9 + 0.5 * 0.64
    assert step == 2 * accum


# --------------------------------------------------------------------------
# Throughput logging
# --------------------------------------------------------------------------

class ManualClock:
    """Stands in for the time module inside the trainer; the test moves it."""

    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now


class RecordingRun:
    """The slice of a wandb run the training loop logs to."""

    def __init__(self):
        self.logged = []

    def log(self, metrics, step=None):
        self.logged.append(dict(metrics))


def test_the_first_log_tick_measures_steps_not_the_compile(tmp_path, monkeypatch):
    """Every interval, the first one included, reports the time its steps
    took. The first tick used to start its clock before the compile, so the
    first train/step_time_ms and train/mfu of every run were the compile."""
    trainer = make_trainer(tmp_path, name="tick", log_every=1)
    clock = ManualClock()
    monkeypatch.setattr(trainer_module, "time", clock)
    compile_step = trainer._compiled_step

    def compile_then_time_each_step(*args):
        executable = compile_step(*args)
        clock.now += 100.0

        def timed(*step_args):
            outputs = executable(*step_args)
            clock.now += 1.0
            return outputs
        return timed

    monkeypatch.setattr(trainer, "_compiled_step", compile_then_time_each_step)
    trainer.wandb = RecordingRun()
    source = DevicePrefetchIterator(batch_iterator(), trainer.batch_sharding)
    trainer.train_loop(trainer.state, trainer._define_train_step(batch_size=8),
                       source, 3, 0, trainer.rngstate)

    step_times = [m["train/step_time_ms"] for m in trainer.wandb.logged
                  if "train/step_time_ms" in m]
    assert step_times == pytest.approx([1000.0, 1000.0, 1000.0])
