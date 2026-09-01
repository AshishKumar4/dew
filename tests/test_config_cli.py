"""The typed run config: serialization, CLI parsing, and what reaches the trainer."""

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from dew.config import DataConfig, ModelConfig, OptimConfig, RunConfig, TrainerConfig
from dew.data.dataloaders import load_data
from dew.registry import build_model
from dew.training import build_optimizer

RECIPES = Path(__file__).resolve().parents[1] / "recipes"
RES = 32
PATCH = 4


def load_recipe(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_train_recipe", RECIPES / name / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def populated_config(cls=RunConfig, **objective_knobs):
    """A config with nothing left at its default, so a round-trip has work to do."""
    return cls(
        model=ModelConfig("simple_dit+hilbert", {
            "patch_size": PATCH, "emb_features": 32, "num_layers": 2, "num_heads": 2,
        }),
        data=DataConfig(dataset="oxford_flowers102", batch_size=8, image_size=64,
                        loader="grain", augmentation_mode="flip_only", worker_count=2),
        optim=OptimConfig(optimizer="lamb", optimizer_opts={"b1": 0.95},
                          learning_rate=1e-3, learning_rate_schedule="cosine",
                          weight_decay=0.01, clip_grads=1.0, grad_accum_steps=3,
                          use_dynamic_scale=True),
        trainer=TrainerConfig(name="run-{dataset}", epochs=3, steps_per_epoch=7,
                              checkpoint_every_steps=11,
                              checkpoint_fs="gcs", fsdp_size=2, wandb_offline=True),
        **objective_knobs,
    )


def fake_dataset(*args, **kwargs):
    def batches():
        rs = np.random.RandomState(0)
        while True:
            yield {"image": jnp.asarray(rs.uniform(0, 255, (4, RES, RES, 3))),
                   "label": jnp.asarray(rs.randint(0, 4, 4))}
    return {"train": batches, "val": batches, "train_len": 16, "local_batch_size": 4}


def test_run_config_round_trips_through_json():
    config = populated_config()
    assert RunConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


def test_recipe_configs_round_trip_too():
    diffusion = load_recipe("diffusion")
    config = populated_config(diffusion.DiffusionRunConfig, noise_schedule="flow",
                              flow_shift=3.0, min_snr_gamma=5.0,
                              val_metrics=["clip", "fid"], autoencoder="stable_diffusion")
    assert diffusion.DiffusionRunConfig.from_dict(config.to_dict()) == config


def test_tyro_parses_the_flags_into_the_same_config():
    parsed = tyro.cli(RunConfig, args=[
        "--model.architecture", "simple_dit+hilbert",
        "--model.config",
        f'{{"patch_size": {PATCH}, "emb_features": 32, "num_layers": 2, "num_heads": 2}}',
        "--data.dataset", "oxford_flowers102", "--data.batch-size", "8",
        "--data.image-size", "64", "--data.loader", "grain",
        "--data.augmentation-mode", "flip_only", "--data.worker-count", "2",
        "--optim.optimizer", "lamb", "--optim.optimizer-opts", '{"b1": 0.95}',
        "--optim.learning-rate", "1e-3", "--optim.learning-rate-schedule", "cosine",
        "--optim.weight-decay", "0.01", "--optim.clip-grads", "1.0",
        "--optim.grad-accum-steps", "3", "--optim.use-dynamic-scale",
        "--trainer.name", "run-{dataset}", "--trainer.epochs", "3",
        "--trainer.steps-per-epoch", "7", "--trainer.checkpoint-every-steps", "11",
        "--trainer.checkpoint-fs", "gcs",
        "--trainer.fsdp-size", "2", "--trainer.wandb-offline",
    ])
    assert parsed == populated_config()


def test_model_config_passes_through_to_the_registry():
    diffusion = load_recipe("diffusion")
    config = populated_config(diffusion.DiffusionRunConfig,
                              autoencoder="stable_diffusion")
    architecture, kwargs = diffusion.model_kwargs(config, channels=4, sample_size=8)

    assert architecture == 'simple_dit'
    # The suffix and the latent channels are the recipe's to add, the rest is
    # whatever --model.config said
    assert kwargs['use_hilbert'] is True
    assert kwargs['output_channels'] == 4
    assert kwargs['emb_features'] == 32

    model = build_model(architecture, kwargs)
    assert model.patch_size == PATCH
    assert model.emb_features == 32
    assert model.output_channels == 4
    assert model.scan_order == 'hilbert'


def test_grad_accum_steps_wraps_the_optimizer_and_reaches_the_trainer(tmp_path, monkeypatch):
    diffusion = load_recipe("diffusion")
    assert not isinstance(diffusion.build_optimizer(OptimConfig(), 10), optax.MultiSteps)
    assert isinstance(
        diffusion.build_optimizer(OptimConfig(grad_accum_steps=3), 10), optax.MultiSteps)
    jepa = load_recipe("jepa")
    monkeypatch.setattr(jepa, "load_data", fake_dataset)
    trainer = jepa.main(jepa.JepaRunConfig(
        model=ModelConfig("jepa_encoder", {
            "patch_size": PATCH, "emb_features": 16, "num_layers": 1, "num_heads": 2,
            "mlp_ratio": 2,
        }),
        data=DataConfig(image_size=RES, batch_size=4, val_steps_per_epoch=1),
        optim=OptimConfig(learning_rate=1e-3, grad_accum_steps=2),
        trainer=TrainerConfig(epochs=1, steps_per_epoch=2, distributed_training=False,
                              checkpoint_dir=str(tmp_path)),
        predictor={"predictor_features": 8, "num_layers": 1, "num_heads": 2},
    ))

    assert trainer.grad_accum_steps == 2
    assert isinstance(trainer.state.tx, optax.MultiSteps)


def reference_optimizer(config: OptimConfig, steps_per_epoch: int):
    """The recipes' old inline construction, verbatim, for equivalence."""
    learning_rate = config.learning_rate
    if config.learning_rate_schedule == 'cosine':
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate, peak_value=config.learning_rate_peak,
            warmup_steps=config.learning_rate_warmup_steps,
            decay_steps=steps_per_epoch * config.learning_rate_decay_epochs,
            end_value=config.learning_rate_end,
        )
    opts = dict(config.optimizer_opts)
    if config.weight_decay is not None:
        opts['weight_decay'] = config.weight_decay
    solver = {'adam': optax.adam, 'adamw': optax.adamw, 'lamb': optax.lamb}[
        config.optimizer](learning_rate, **opts)
    if config.clip_grads > 0:
        solver = optax.chain(optax.clip_by_global_norm(config.clip_grads), solver)
    if config.grad_accum_steps > 1:
        solver = optax.MultiSteps(solver, every_k_schedule=config.grad_accum_steps)
    return solver


def run_steps(solver, steps=9):
    """The updates a solver emits on a fixed gradient stream."""
    params = {"w": jnp.asarray([0.5, -0.25])}
    state = solver.init(params)
    out = []
    rng = np.random.RandomState(0)
    for _ in range(steps):
        grads = {"w": jnp.asarray(rng.randn(2))}
        updates, state = solver.update(grads, state, params)
        out.append(np.asarray(updates["w"]))
    return out


def old_inline_schedule(config: OptimConfig, steps_per_epoch: int):
    """The warmup-cosine schedule the recipes used to build inline."""
    return optax.warmup_cosine_decay_schedule(
        init_value=config.learning_rate, peak_value=config.learning_rate_peak,
        warmup_steps=config.learning_rate_warmup_steps,
        decay_steps=steps_per_epoch * config.learning_rate_decay_epochs,
        end_value=config.learning_rate_end,
    )


def test_library_build_optimizer_runs_the_old_inline_cosine_schedule():
    config = OptimConfig(
        optimizer="adamw", optimizer_opts={"b1": 0.0, "b2": 0.0, "eps": 0.0},
        learning_rate=1e-3, learning_rate_schedule="cosine",
        learning_rate_peak=3e-3, learning_rate_end=1e-5,
        learning_rate_warmup_steps=3, learning_rate_decay_epochs=2,
    )
    steps_per_epoch = 5  # decay_steps = 5 * 2

    # b1=b2=eps=0 makes adamw emit -schedule(step) * sign(g), so the update
    # magnitudes are the learning rates the solver actually runs.
    grads = {"w": jnp.asarray([1.0, -1.0])}
    solver = build_optimizer(config, steps_per_epoch)
    state = solver.init(grads)
    for step in range(13):  # the warmup, the peak, the decay, the flat tail
        updates, state = solver.update(grads, state, grads)
        assert np.isclose(-float(updates["w"][0]),
                          float(old_inline_schedule(config, steps_per_epoch)(step)),
                          rtol=1e-4)


def test_library_build_optimizer_clips_the_global_norm():
    """Clip caps the gradient tree before the solver sees it. Adam's own
    normalization hides pure rescaling, so a large eps makes the cap show."""
    solver = build_optimizer(
        OptimConfig(learning_rate=1e-3, clip_grads=0.5, optimizer_opts={"eps": 1.0}), 10)
    grads = {"w": jnp.asarray([3.0, 4.0])}  # global norm 5, over the 0.5 cap
    a, _ = solver.update(grads, solver.init(grads), grads)

    # The same first stage optax would build by hand.
    reference = optax.chain(optax.clip_by_global_norm(0.5), optax.adamw(1e-3, eps=1.0))
    r, _ = reference.update(grads, reference.init(grads), grads)
    assert jnp.allclose(a["w"], r["w"])

    # A 10x twin caps to the same tree, so the same update.
    b, _ = solver.update(jax.tree.map(lambda g: g * 10, grads),
                         solver.init(grads), grads)
    assert jnp.allclose(a["w"], b["w"])

    # Without the cap the two differ, so it is the clip that made them equal.
    unclipped = build_optimizer(OptimConfig(learning_rate=1e-3,
                                            optimizer_opts={"eps": 1.0}), 10)
    c, _ = unclipped.update(grads, unclipped.init(grads), grads)
    d, _ = unclipped.update(jax.tree.map(lambda g: g * 10, grads),
                            unclipped.init(grads), grads)
    assert not jnp.allclose(c["w"], d["w"])


def test_load_data_dispatches_over_the_registries(monkeypatch):
    """The loader picks the factory from registry membership, not name spelling."""
    from dew.data import dataloaders

    calls = []
    sentinel = object()

    def record(name):
        def factory(*args, **kwargs):
            calls.append((name, args, kwargs))
            return sentinel
        return factory

    monkeypatch.setattr(dataloaders, "get_dataset_grain", record("grain"))
    monkeypatch.setattr(dataloaders, "get_media_dataset_grain", record("media"))
    monkeypatch.setattr(dataloaders, "get_dataset_online", record("online"))

    # A datasetMap name with loader='grain' goes to the legacy image factory.
    config = DataConfig(dataset="oxford_flowers102", loader="grain")
    assert load_data(config) is sentinel
    assert calls[0][0] == "grain"
    assert calls[0][2]["dataset_source"] == config.dataset_path

    # A mediaDatasetMap-only name (no datasetMap entry) goes to the media factory.
    calls.clear()
    load_data(DataConfig(dataset="voxceleb2", loader="grain"))
    assert calls[0][0] == "media"

    # An onlineDatasetMap name, forced online: the read thread/buffer scaling.
    calls.clear()
    config = DataConfig(dataset="combined_online", loader="online",
                        read_thread_count=10, worker_buffer_size=20)
    load_data(config)
    assert calls[0][0] == "online"
    assert calls[0][2]["read_thread_count"] == 40
    assert calls[0][2]["worker_buffer_size"] == 100

    # Auto mode streams only when the name is registered solely online.
    calls.clear()
    load_data(DataConfig(dataset="combined_online", loader="auto"))
    assert calls[0][0] == "online"

    calls.clear()
    load_data(DataConfig(dataset="oxford_flowers102", loader="auto"))
    assert calls[0][0] == "grain"

    calls.clear()
    load_data(DataConfig(dataset="voxceleb2", loader="auto"))
    assert calls[0][0] == "media"
