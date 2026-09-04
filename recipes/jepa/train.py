"""Train a JEPA encoder (I-JEPA over images, V-JEPA over video).

    python recipes/jepa/train.py --data.image-size 224 --trainer.batch-size 64 \
        --trainer.epochs 300 --model.config '{"patch_size": 16, "emb_features": 384, \
        "num_layers": 12, "num_heads": 6}' --probe-classes 102

The encoder is --model, the predictor takes the encoder's width and heads plus
--predictor, and the probes score the frozen encoder at every validation.
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import jax
import tyro

import dew.io
from dew.config import JsonDict, ModelConfig, OptimConfig, RunConfig
from dew.data import ImageDataset, VideoDataset
from dew.inputs import Field
from dew.objectives.jepa import JepaObjective, multi_block_mask
from dew.registry import datasets, metrics, models
from dew.training import (Checkpoints, Trainer, TrainState, WandbTracker,
                          build_optimizer, prepare_process, run_timestamp)

# HF tokenizers fork a thread pool; grain's workers fork the process.
os.environ['TOKENIZERS_PARALLELISM'] = "false"

DEFAULT_ENCODER_CONFIG = {"precision": "default"}

# What the predictor takes from the encoder unless --predictor overrides it.
# Its depth and width are its own: a predictor as wide as the encoder makes the
# objective too easy. The compute dtype and the attention kernel are not here:
# they belong to the run's precision policy, which writes them into both models.
SHARED_MODEL_KEYS = ("emb_features", "num_heads", "mlp_ratio", "ssm_attention_ratio",
                     "ssm_state_dim", "dropout_rate", "precision")


@dataclass(frozen=True)
class JepaRunConfig(RunConfig):
    """A run, plus the JEPA objective's own knobs."""

    model: ModelConfig = field(
        default_factory=lambda: ModelConfig("jepa_encoder", dict(DEFAULT_ENCODER_CONFIG)))
    optim: OptimConfig = field(
        default_factory=lambda: OptimConfig(
            learning_rate=1e-3, learning_rate_peak=1.5e-3, learning_rate_end=1e-6))
    predictor: JsonDict = field(default_factory=dict)
    """Predictor kwargs, over the encoder's shared ones."""
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


def sample_field(config: JepaRunConfig) -> Field:
    """The batch field the encoder reads, at the resolution the data comes in."""
    spec = config.data
    if isinstance(spec, ImageDataset):
        return Field("image", (spec.image_size, spec.image_size, 3))
    if isinstance(spec, VideoDataset):
        return Field("video", (spec.frames, spec.frame_size, spec.frame_size, 3))
    raise ValueError(
        f"the JEPA recipe trains on image or video datasets, not {datasets.name_of(type(spec))}")


def build_encoder(config: JepaRunConfig):
    """The encoder, and the fields the registry built it from."""
    fields = config.model.fields()
    return models.build(config.model.architecture, **fields), fields


def build_predictor(config: JepaRunConfig, encoder_fields: dict, encoder, grid,
                    is_video: bool):
    """The predictor that reads the encoder's embeddings, and its fields."""
    fields = {
        **{k: v for k, v in encoder_fields.items() if k in SHARED_MODEL_KEYS},
        **config.predictor,
        "grid": grid,
        "factorized": is_video,
        "scan_order": encoder.scan_order,
    }
    fields = ModelConfig('jepa_predictor', fields, dtype=config.model.dtype,
                         attention_impl=config.model.attention_impl).fields()
    return models.build('jepa_predictor', **fields), fields


def run_summary(config: JepaRunConfig, encoder_fields: dict) -> dict:
    """Flat view of the run, for the tracker."""
    return {
        **encoder_fields,
        "architecture": config.model.architecture,
        "dataset": datasets.name_of(type(config.data)),
        "image_size": sample_field(config).shape[-2],
        "batch_size": config.trainer.batch_size,
        "learning_rate": config.optim.learning_rate,
    }


def main(config: JepaRunConfig) -> TrainState:
    prepare_process(config.trainer.wandb, config.trainer.multi_host,
                    config.trainer.xla_flags, config.trainer.compilation_cache_dir)

    data = config.data.load(batch=config.trainer.batch_size)
    steps = config.trainer.total_steps(data)

    sample = sample_field(config)
    is_video = sample.key == "video"
    if is_video != (config.model.architecture == 'jepa_video_encoder'):
        raise ValueError(
            "a video dataset and --model.architecture jepa_video_encoder go together")

    encoder, encoder_fields = build_encoder(config)
    grid = (sample.shape[-2] // encoder.patch_size,) * 2
    predictor, predictor_fields = build_predictor(
        config, encoder_fields, encoder, grid, is_video)

    mask = multi_block_mask(
        grid,
        num_targets=config.num_target_blocks,
        scale=tuple(config.target_scale),
        aspect=tuple(config.target_aspect),
    )
    print(f"Mask geometry: {mask.block_area} tokens per target block "
          f"({mask.block_shapes}), {mask.num_context} context tokens of {mask.num_patches}")

    objective = JepaObjective(
        encoder=encoder,
        predictor=predictor,
        mask=mask,
        sample=sample,
        momentum=tuple(config.momentum),
        momentum_steps=config.momentum_steps or steps,
        label_key=config.probe_label_key,
    )

    probes = ()
    if config.probe_classes:
        probes = (metrics.linear_probe(config.probe_classes),
                  metrics.knn_probe(config.probe_classes, k=config.knn_k))

    name = config.trainer.name or (
        f"jepa-{datasets.name_of(type(config.data))}/res-{sample.shape[-2]}/"
        f"patch-{encoder.patch_size}/mixer-{encoder.ssm_attention_ratio}/"
        f"emb-{encoder.emb_features}/lr-{config.optim.learning_rate}/date-{run_timestamp()}")
    print("Experiment_Name:", name)
    directory = os.path.join(config.trainer.checkpoint_dir, name)

    run_config = config.to_dict()
    tracker = None
    if config.trainer.wandb is not None:
        tracker = WandbTracker(
            config.trainer.wandb.project, name, entity=config.trainer.wandb.entity,
            offline=config.trainer.wandb.offline,
            config={"run_config": run_config, "encoder": encoder_fields,
                    "predictor": predictor_fields,
                    "mask": {"grid": grid, "block_shapes": mask.block_shapes,
                             "block_area": mask.block_area, "num_context": mask.num_context},
                    "arguments": run_summary(config, encoder_fields),
                    "dataset": {"name": datasets.name_of(type(config.data)),
                                "records": data.records},
                    "steps": steps})

    checkpoints = Checkpoints(directory, keep=config.trainer.keep)
    config.save(checkpoints.directory)
    trainer = Trainer(
        objective, build_optimizer(config.optim, steps),
        key=jax.random.key(config.trainer.seed),
        mesh=config.trainer.mesh,
        layout=config.trainer.layout,
        accumulation=config.trainer.accumulation,
        dynamic_scale=config.trainer.dynamic_scale,
        checkpoints=checkpoints,
        tracker=tracker,
        profile=config.trainer.profile,
    )

    start = time.time()
    state = trainer.fit(
        data, steps=steps,
        log_every=config.trainer.log_every,
        eval_every=config.trainer.eval_every or data.steps_per_epoch,
        checkpoint_every=config.trainer.checkpoint_every or data.steps_per_epoch,
        metrics=probes,
    )
    print(f"Training finished in {time.time() - start:.0f}s")
    if tracker is not None:
        dew.io.publish(checkpoints.path(checkpoints.latest), re.sub(r"[^\w.-]", "-", name),
                       tracker=tracker)
    return state


if __name__ == '__main__':
    main(tyro.cli(tyro.conf.CascadeSubcommandArgs[JepaRunConfig]))
