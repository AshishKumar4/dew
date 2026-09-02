"""Train a JEPA encoder (I-JEPA over images, V-JEPA over video).

A sibling of the diffusion recipe rather than a flag on it: the two share the
data pipeline, the registry and the trainer, but nothing else. A JEPA run has
no noise schedule, no sampler, no text conditioning and no VAE, and folding an
--objective switch into one recipe would mean threading None through all of
them. The Objective seam is what makes the same trainer serve both.

    python recipes/jepa/train.py --data.dataset oxford_flowers102 \
        --data.image-size 128 --trainer.epochs 100 --probe-classes 102 \
        --model.config '{"patch_size": 16, "emb_features": 384, "num_layers": 12}'
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import jax
import tyro

from dew.config import DataConfig, JsonDict, ModelConfig, OptimConfig, RunConfig
from dew.data.dataloaders import load_data
from dew.inputs import DiffusionInputConfig
from dew.objectives.jepa import (
    JepaObjective, multi_block_mask, get_linear_probe_metric, get_knn_probe_metric,
)
from dew.registry import apply_precision_policy, build_model, canonicalize_architecture
from dew.training import ObjectiveTrainer, build_optimizer, prepare_process
from dew.training.distributed import DEFAULT_MIN_SHARD_SIZE

os.environ['TOKENIZERS_PARALLELISM'] = "false"

DEFAULT_ENCODER_CONFIG = {"precision": "default"}

# What the predictor takes from the encoder unless --predictor overrides it.
# Its depth and width are its own: a predictor as wide as the encoder makes the
# objective too easy.
SHARED_MODEL_KEYS = ("emb_features", "num_heads", "mlp_ratio", "ssm_attention_ratio",
                     "ssm_state_dim", "dropout_rate", "attention_impl", "dtype", "precision")


@dataclass(frozen=True)
class JepaRunConfig(RunConfig):
    """A run, plus the JEPA objective's own knobs."""

    model: ModelConfig = field(
        default_factory=lambda: ModelConfig("jepa_encoder", dict(DEFAULT_ENCODER_CONFIG)))
    data: DataConfig = field(default_factory=lambda: DataConfig(batch_size=64))
    optim: OptimConfig = field(
        default_factory=lambda: OptimConfig(
            learning_rate=1e-3, learning_rate_peak=1.5e-3, learning_rate_end=1e-6))
    predictor: JsonDict = field(default_factory=dict)
    """Predictor kwargs, over the encoder's shared ones."""
    frames_per_sample: Optional[int] = None
    """Set for video (V-JEPA), leave unset for images (I-JEPA)."""
    num_target_blocks: int = 4
    target_scale: list[float] = field(default_factory=lambda: [0.15, 0.2])
    target_aspect: list[float] = field(default_factory=lambda: [0.75, 1.5])
    momentum: list[float] = field(default_factory=lambda: [0.996, 1.0])
    """Target-encoder EMA momentum, ramped over momentum_steps."""
    momentum_steps: Optional[int] = None
    """Defaults to the full training run."""
    probe_classes: Optional[int] = None
    """Number of classes for the frozen-encoder probes."""
    probe_label_key: str = 'label'
    knn_k: int = 20


def build_encoder(config: JepaRunConfig):
    """The encoder, and the config the registry built it from."""
    architecture, suffix_flags = canonicalize_architecture(config.model.architecture)
    encoder_config = apply_precision_policy(
        architecture, {**config.model.config, **suffix_flags},
        dtype=config.model.dtype, attention_impl=config.model.attention_impl)
    if encoder_config.get('use_hilbert') and encoder_config.get('use_zigzag'):
        raise ValueError("use_hilbert and use_zigzag are mutually exclusive")
    return build_model(architecture, encoder_config), encoder_config


def build_predictor(config: JepaRunConfig, encoder_config: dict, encoder, grid,
                    is_video: bool):
    """The predictor that reads the encoder's embeddings, and its config."""
    predictor_config = {
        **{k: v for k, v in encoder_config.items() if k in SHARED_MODEL_KEYS},
        **config.predictor,
        "grid": grid,
        "factorized": is_video,
        "scan_order": encoder.scan_order,
    }
    return build_model('jepa_predictor', predictor_config), predictor_config


def run_summary(config: JepaRunConfig, encoder_config: dict) -> dict:
    """Flat view of the run, for the wandb config."""
    return {
        **encoder_config,
        "architecture": config.model.architecture,
        "dataset": config.data.dataset,
        "image_size": config.data.image_size,
        "batch_size": config.data.batch_size,
        "learning_rate": config.optim.learning_rate,
        "epochs": config.trainer.epochs,
    }


def main(config: JepaRunConfig) -> ObjectiveTrainer:
    prepare_process(config.data.augmentation_mode, config.trainer.wandb_offline,
                    config.trainer.multi_host)

    checkpoint_dir = config.trainer.checkpoint_dir
    if config.trainer.checkpoint_fs == 'gcs':
        checkpoint_dir = f"gs://{checkpoint_dir}"

    data = load_data(config.data)
    steps_per_epoch = (config.trainer.steps_per_epoch
                       or data['train_len'] // config.data.batch_size)
    total_steps = steps_per_epoch * config.trainer.epochs

    is_video = config.frames_per_sample is not None
    architecture, _ = canonicalize_architecture(config.model.architecture)
    if is_video != (architecture == 'jepa_video_encoder'):
        raise ValueError(
            "--frames-per-sample and --model.architecture jepa_video_encoder go together")

    encoder, encoder_config = build_encoder(config)
    grid = (config.data.image_size // encoder.patch_size,) * 2
    predictor, predictor_config = build_predictor(
        config, encoder_config, encoder, grid, is_video)

    mask = multi_block_mask(
        grid,
        num_targets=config.num_target_blocks,
        scale=tuple(config.target_scale),
        aspect=tuple(config.target_aspect),
    )
    print(f"Mask geometry: {mask.block_area} tokens per target block "
          f"({mask.block_shapes}), {mask.num_context} context tokens of {mask.num_patches}")

    sample_data_shape = ((config.frames_per_sample, config.data.image_size,
                          config.data.image_size, 3)
                         if is_video else
                         (config.data.image_size, config.data.image_size, 3))
    input_config = DiffusionInputConfig(
        sample_data_key='video' if is_video else 'image',
        sample_data_shape=sample_data_shape,
        conditions=[],
    )
    objective = JepaObjective(
        encoder=encoder,
        predictor=predictor,
        mask=mask,
        sample_data_key=input_config.sample_data_key,
        sample_data_shape=sample_data_shape,
        momentum=tuple(config.momentum),
        momentum_steps=config.momentum_steps or total_steps,
    )

    eval_metrics = []
    if config.probe_classes:
        eval_metrics = [
            get_linear_probe_metric(config.probe_classes, label_key=config.probe_label_key),
            get_knn_probe_metric(config.probe_classes, label_key=config.probe_label_key,
                                 k=config.knn_k),
        ]

    name = config.trainer.name or (
        f"jepa-{config.data.dataset}/res-{config.data.image_size}/patch-{encoder.patch_size}/"
        f"mixer-{encoder.ssm_attention_ratio}/emb-{encoder.emb_features}/"
        f"lr-{config.optim.learning_rate}/date-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
    print("Experiment_Name:", name)

    wandb_config: Optional[dict[str, Any]] = None
    if config.trainer.wandb_project is not None:
        wandb_config = {
            "project": config.trainer.wandb_project,
            "entity": config.trainer.wandb_entity,
            "name": name,
            "config": {
                "encoder": encoder_config,
                "predictor": predictor_config,
                "architecture": architecture,
                "mask": {"grid": grid, "block_shapes": mask.block_shapes,
                         "block_area": mask.block_area, "num_context": mask.num_context},
                "dataset": {"name": config.data.dataset, "length": data['train_len']},
                "arguments": run_summary(config, encoder_config),
                "run_config": config.to_dict(),
            },
        }
        if config.trainer.resume_last_run is not None:
            wandb_config['id'] = config.trainer.resume_last_run

    trainer = ObjectiveTrainer(
        model=encoder,
        optimizer=build_optimizer(config.optim, steps_per_epoch),
        input_config=input_config,
        rngs=jax.random.PRNGKey(4),
        objective=objective,
        name=name,
        wandb_config=wandb_config,
        distributed_training=config.trainer.distributed_training,
        checkpoint_base_path=checkpoint_dir,
        checkpoint_step=config.trainer.checkpoint_step,
        load_from_checkpoint=config.trainer.load_from_checkpoint,
        max_checkpoints_to_keep=config.trainer.max_checkpoints_to_keep,
        eval_metrics=eval_metrics,
        best_tracker_metric=(
            config.trainer.best_tracker_metric
            or ("val/knn_probe_accuracy" if eval_metrics else "train/best_loss")),
        grad_accum_steps=config.optim.grad_accum_steps,
        use_dynamic_scale=config.optim.use_dynamic_scale,
        fsdp_size=config.trainer.fsdp_size,
        fsdp_min_param_size=config.trainer.fsdp_min_param_size or DEFAULT_MIN_SHARD_SIZE,
        logical_axis_rules=config.trainer.logical_axis_rules,
        sharding_tolerance=config.trainer.sharding_tolerance,
        compilation_cache_dir=config.trainer.compilation_cache_dir,
        profile_steps=config.trainer.profile_steps,
        log_every=config.trainer.log_every,
    )

    start = time.time()
    trainer.fit(data, training_steps_per_epoch=steps_per_epoch,
                epochs=config.trainer.epochs,
                val_steps_per_epoch=config.data.val_steps_per_epoch,
                checkpoint_every_steps=config.trainer.checkpoint_every_steps)
    print(f"Training finished in {time.time() - start:.0f}s")
    return trainer


if __name__ == '__main__':
    main(tyro.cli(JepaRunConfig))
