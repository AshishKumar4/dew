"""The typed run config: serialization, CLI parsing, and what reaches the trainer."""

import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
import tyro

from dew.config import DataConfig, ModelConfig, OptimConfig, RunConfig, TrainerConfig
from dew.registry import build_model

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
        "--trainer.steps-per-epoch", "7", "--trainer.checkpoint-fs", "gcs",
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
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(jepa, "get_dataset_grain", fake_dataset)
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
