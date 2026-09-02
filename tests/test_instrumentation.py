"""Throughput accounting, divergence detection and the profiler hook.

None of the performance work is evaluable without these numbers, so they get
the same treatment as the training maths.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.inputs import DiffusionInputConfig
from dew.nn.backbones.dit import SimpleDiT
from dew.diffusion.transforms import get_diffusion_preset
from dew.training import ObjectiveTrainer
from dew.training.distributed import DevicePrefetchIterator
from dew.telemetry.instrumentation import (
    compiled_flops, enable_compilation_cache, model_flops_utilization, step_flops,
)

RES = 8
BATCH = 8


def make_trainer(tmp_path, **kwargs):
    train_schedule, _, transform = get_diffusion_preset("edm")
    return ObjectiveTrainer(
        model=SimpleDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1),
        optimizer=optax.adam(1e-3),
        noise_schedule=train_schedule,
        model_output_transform=transform,
        input_config=DiffusionInputConfig(
            sample_data_key="image", sample_data_shape=(RES, RES, 3), conditions=[]),
        rngs=jax.random.PRNGKey(0),
        name="instr",
        wandb_config=None,
        distributed_training=False,
        checkpoint_base_path=str(tmp_path),
        **kwargs,
    )


def batches():
    images = np.tile(np.linspace(0, 255, RES, dtype=np.float32)[None, :, None, None],
                     (BATCH, 1, RES, 3))
    while True:
        yield {"image": images}


def data_dict():
    return {"train": batches, "train_len": BATCH * 8,
            "local_batch_size": BATCH, "global_batch_size": BATCH}


def test_step_flops_reports_a_positive_count(tmp_path):
    trainer = make_trainer(tmp_path)
    step = trainer._define_train_step(batch_size=BATCH)
    source = DevicePrefetchIterator(batches(), trainer.batch_sharding)
    flops = step_flops(step, trainer.state, trainer.rngstate, next(source))
    assert flops is not None and flops > 0


def test_throughput_metrics_are_consistent(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.global_batch_size = 64
    metrics = trainer._throughput_metrics(elapsed=2.0, steps=10)
    assert metrics["train/step_time_ms"] == pytest.approx(200.0)
    assert metrics["train/samples_per_sec"] == pytest.approx(320.0)


def test_throughput_metrics_ignore_a_zero_interval(tmp_path):
    assert make_trainer(tmp_path)._throughput_metrics(elapsed=0.0, steps=0) == {}


def test_mfu_is_skipped_on_unknown_hardware():
    # CPU is deliberately absent from the peak-FLOPs table
    assert model_flops_utilization(1e12, 1.0) is None


def test_mfu_uses_the_per_device_flop_count(monkeypatch):
    from dew.telemetry import instrumentation
    monkeypatch.setitem(instrumentation.PEAK_FLOPS_PER_DEVICE,
                        jax.devices()[0].device_kind, 100.0)
    assert instrumentation.model_flops_utilization(50.0, 1.0) == pytest.approx(0.5)
    assert instrumentation.model_flops_utilization(50.0, 2.0) == pytest.approx(0.25)


def test_compiled_flops_is_per_device_under_spmd():
    devices = jax.devices()
    mesh = jax.make_mesh((len(devices),), ("data",), devices=devices)
    split = NamedSharding(mesh, P("data"))
    replicated = NamedSharding(mesh, P())
    batch, width = len(devices), 32
    x = jax.device_put(jnp.ones((batch, width)), split)
    weight = jax.device_put(jnp.ones((width, width)), replicated)
    executable = jax.jit(
        lambda values, kernel: values @ kernel,
        in_shardings=(split, replicated), out_shardings=split,
    ).lower(x, weight).compile()

    whole_batch_flops = 2 * batch * width * width
    assert compiled_flops(executable) == pytest.approx(
        whole_batch_flops / len(devices))


def test_epoch_loss_accumulates_bfloat16_losses_in_float32(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, log_every=1000)

    def step(state, rng_state, batch):
        del batch
        loss = jnp.array(1.5, jnp.bfloat16)
        return state, loss, {}, rng_state, jnp.array(True)

    monkeypatch.setattr(trainer, "_compiled_step", lambda *_: step)
    steps = 400
    epoch_loss, *_ = trainer.train_loop(
        trainer.state, object(), iter([None] * steps), steps, 0, trainer.rngstate)

    assert epoch_loss.dtype == jnp.float32
    assert float(epoch_loss / steps) == pytest.approx(1.5)


def test_fit_reports_throughput_to_wandb(tmp_path):
    """The logging tick must actually carry the numbers, not just the loss."""
    logged = []

    class FakeWandb:
        def log(self, payload, step=None):
            logged.append(payload)

        def define_metric(self, *args, **kwargs):
            pass

    trainer = make_trainer(tmp_path, log_every=1)
    trainer.wandb = FakeWandb()
    trainer.fit(data_dict(), training_steps_per_epoch=3, epochs=1, val_steps_per_epoch=0)

    ticks = [p for p in logged if "train/samples_per_sec" in p]
    assert ticks, "no throughput was logged"
    assert all(p["train/step_time_ms"] > 0 for p in ticks)
    assert all(p["train/samples_per_sec"] > 0 for p in ticks)


def test_compilation_cache_directory_is_configured(tmp_path):
    path = str(tmp_path / "xla-cache")
    enable_compilation_cache(path)
    assert os.path.isdir(path)
    assert jax.config.jax_compilation_cache_dir == path


def test_profiler_writes_a_trace(tmp_path, monkeypatch):
    """The window has to open after the warmup: a trace that starts at step 0
    is mostly compilation, and reports its occupancy instead of the loop's."""
    trainer = make_trainer(tmp_path, profile_steps=2, profile_warmup_steps=2)
    started_at = []
    real_start = jax.profiler.start_trace

    def spy(*args, **kwargs):
        # The train state is replaced after every step, so its counter is the
        # number of steps that have run by the time the trace opens.
        started_at.append(int(trainer.state.step))
        return real_start(*args, **kwargs)

    monkeypatch.setattr(jax.profiler, "start_trace", spy)
    trainer.fit(data_dict(), training_steps_per_epoch=5, epochs=1, val_steps_per_epoch=0)

    assert started_at == [2], "the trace did not open at the configured step"
    assert os.path.isdir(trainer.profile_path())
    assert any(files for _, _, files in os.walk(trainer.profile_path()))


def test_an_unfinished_profile_window_is_still_closed(tmp_path):
    """A window wider than the epoch has to close anyway: a trace left running
    takes the next one down with it."""
    trainer = make_trainer(tmp_path / "long", profile_steps=8, profile_warmup_steps=1)
    trainer.fit(data_dict(), training_steps_per_epoch=3, epochs=1, val_steps_per_epoch=0)
    assert any(files for _, _, files in os.walk(trainer.profile_path()))

    second = make_trainer(tmp_path / "short", profile_steps=1, profile_warmup_steps=0)
    second.fit(data_dict(), training_steps_per_epoch=2, epochs=1, val_steps_per_epoch=0)
    assert any(files for _, _, files in os.walk(second.profile_path()))
def test_profiler_runs_only_once_across_epochs(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, profile_steps=1, profile_warmup_steps=0)
    starts = []
    stops = []
    monkeypatch.setattr(jax.profiler, "start_trace", lambda *a, **k: starts.append(1))
    monkeypatch.setattr(jax.profiler, "stop_trace", lambda: stops.append(1))

    trainer.fit(data_dict(), training_steps_per_epoch=1, epochs=3,
                val_steps_per_epoch=0)

    assert len(starts) == 1
    assert len(stops) == 1


def test_profiler_warmup_can_cross_an_epoch_boundary(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, profile_steps=1, profile_warmup_steps=2)
    started_at = []
    monkeypatch.setattr(
        jax.profiler, "start_trace",
        lambda *a, **k: started_at.append(int(trainer.state.step)))
    monkeypatch.setattr(jax.profiler, "stop_trace", lambda: None)

    trainer.fit(data_dict(), training_steps_per_epoch=1, epochs=3,
                val_steps_per_epoch=0)

    assert started_at == [2]


def test_the_training_step_is_compiled_once_per_run(tmp_path, monkeypatch):
    """Reading the cost analysis used to compile the step a second time, which
    doubled the startup cost of every fit()."""
    trainer = make_trainer(tmp_path)
    compiles = []
    real_compile = jax.stages.Lowered.compile

    def counting_compile(lowered, *args, **kwargs):
        compiles.append(lowered)
        return real_compile(lowered, *args, **kwargs)

    monkeypatch.setattr(jax.stages.Lowered, "compile", counting_compile)

    jitted = []
    real_define = trainer._define_train_step

    def capture(**kwargs):
        step = real_define(**kwargs)
        jitted.append(step)
        return step

    monkeypatch.setattr(trainer, "_define_train_step", capture)
    trainer.fit(data_dict(), training_steps_per_epoch=3, epochs=2, val_steps_per_epoch=0)

    assert len(compiles) == 1, "the training step was compiled more than once"
    # Both epochs ran on that one executable; a jit call would have compiled
    # its own and left it in the jit cache.
    assert jitted[0]._cache_size() == 0, "the loop went through the jit path too"
    assert trainer.flops_per_step and trainer.flops_per_step > 0


# --------------------------------------------------------------------------
# Divergence
# --------------------------------------------------------------------------

def test_sustained_non_finite_loss_stops_the_run(tmp_path):
    trainer = make_trainer(
        tmp_path, log_every=1, max_bad_loss_steps=3,
        loss_fn=lambda pred, target: jnp.full_like(pred, jnp.nan))
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.fit(data_dict(), training_steps_per_epoch=8, epochs=1, val_steps_per_epoch=0)


def test_healthy_run_does_not_trip_the_detector(tmp_path):
    trainer = make_trainer(tmp_path, log_every=1, max_bad_loss_steps=3)
    trainer.fit(data_dict(), training_steps_per_epoch=6, epochs=1, val_steps_per_epoch=0)
    assert int(trainer.state.step) == 6
