"""Train a diffusion model over images or latents.

    python recipes/diffusion/train.py --data.image-size 128 --trainer.batch-size 32 \
        --trainer.epochs 2000 --model.architecture simple_dit \
        --model.config '{"patch_size": 4, "emb_features": 512, "num_layers": 12, "num_heads": 8}'

The dataset is a subcommand over the registry (`data:cc12m --data.path /mnt/gcs`),
and so are the preset (`preset:flow --preset.shift 3.0`) and the sampler.
Architecture kwargs go through --model.config as one JSON object, straight to
the registry, so the manifest is exactly what built the model.
"""

import dataclasses
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

import jax
import tyro

import dew.diffusion.presets
import dew.io
import dew.sampling
from dew.config import JsonDict, ModelConfig, RunConfig
from dew.data import ImageDataset, OnlineImages, VideoDataset
from dew.inputs import CLIPText, Condition, Field, InputSpec
from dew.interop.manifest import Manifest
from dew.objectives.diffusion import DiffusionObjective
from dew.registry import datasets, metrics, models, presets, samplers
from dew.sampling import CFG
from dew.training import (Checkpoints, Profile, Trainer, TrainState, WandbTracker,
                          build_optimizer, prepare_process, run_timestamp)

# HF tokenizers fork a thread pool; grain's workers fork the process.
os.environ['TOKENIZERS_PARALLELISM'] = "false"

ATTENTION = {
    "heads": 8, "use_projection": False,
    "use_self_and_cross": True, "only_pure_attention": True,
}

# The default unet: attention everywhere but the full-resolution stage, where
# it costs the most. Every other architecture takes its own kwargs as JSON.
DEFAULT_MODEL_CONFIG = {
    "attention_configs": [None, ATTENTION, ATTENTION, ATTENTION],
    "precision": "default",
}

DEFAULT_EXPERIMENT_NAME = ("dataset-{dataset}/image_size-{image_size}/batch-{batch_size}/"
                           "schd-{preset}/arch-{architecture}/lr-{learning_rate}")


@dataclass(frozen=True)
class DiffusionRunConfig(RunConfig):
    """A run, plus the diffusion objective's own knobs."""

    model: ModelConfig = field(
        default_factory=lambda: ModelConfig("unet", dict(DEFAULT_MODEL_CONFIG)))
    preset: presets.union = field(default_factory=presets.EDM)
    """The convention the model is trained and sampled with."""
    sampler: samplers.union = field(default_factory=samplers.EulerAncestral)
    """The solver validation samples with."""
    guidance: float = 3.0
    """Classifier-free guidance scale for validation samples; 0 samples the
    conditional prediction alone."""
    sampling_steps: int = 200
    unconditional_prob: float = 0.12
    """Fraction of training examples whose condition is dropped."""
    ema_decay: float = 0.999
    text_encoder: str = "openai/clip-vit-large-patch14"
    autoencoder: Optional[Literal['stable_diffusion']] = None
    autoencoder_opts: JsonDict = field(
        default_factory=lambda: {"modelname": "pcuenq/sd-vae-ft-mse-flax"})
    val_metrics: list[Literal['clip', 'clip_score', 'fid']] = field(
        default_factory=lambda: ['clip'])


def sample_field(config: DiffusionRunConfig) -> Field:
    """The batch field the model generates, at the resolution the data comes in."""
    spec = config.data
    if isinstance(spec, (ImageDataset, OnlineImages)):
        return Field("image", (spec.image_size, spec.image_size, 3))
    if isinstance(spec, VideoDataset):
        return Field("video", (spec.frames, spec.frame_size, spec.frame_size, 3))
    raise ValueError(
        f"the diffusion recipe trains on image or video datasets, not "
        f"{datasets.name_of(type(spec))}")


def load_autoencoder(config: DiffusionRunConfig):
    """The VAE for latent diffusion, or None."""
    if config.autoencoder is None:
        return None
    from dew.nn.autoencoders.sd_vae import StableDiffusionVAE
    return StableDiffusionVAE(**config.autoencoder_opts)


def model_fields(config: DiffusionRunConfig, sample: Field, autoencoder) -> dict:
    """The fields the registry builds the model from: the run's precision
    settings and the channels the model denoises, over --model.config."""
    fields = config.model.fields()
    if autoencoder is None:
        channels, size = sample.shape[-1], sample.shape[-2]
    else:
        channels = autoencoder.latent_channels
        size = sample.shape[-2] // autoencoder.downscale_factor
    if config.model.architecture == 'diffusers_unet_simple':
        fields.update(sample_size=size, in_channels=channels, out_channels=channels)
    else:
        fields['output_channels'] = channels
    return fields


def build_inputs(config: DiffusionRunConfig) -> InputSpec:
    """Images conditioned on the batch's pretokenized text through CLIP."""
    return InputSpec(
        sample=sample_field(config),
        conditions={"textcontext": Condition(CLIPText.from_pretrained(config.text_encoder))})


def build_eval_metrics(names: list[str]) -> list:
    """Validation metrics; each pulls its own weights on construction."""
    return [metrics[name]() for name in names]


def run_summary(config: DiffusionRunConfig, fields: dict, arguments_hash: str) -> dict:
    """Flat view of the run, for the experiment name."""
    sample = sample_field(config)
    return {
        **fields,
        "architecture": config.model.architecture,
        "dataset": datasets.name_of(type(config.data)),
        "image_size": sample.shape[-2],
        "batch_size": config.trainer.batch_size,
        "preset": presets.name_of(type(config.preset)),
        "learning_rate": config.optim.learning_rate,
        "arguments_hash": arguments_hash,
        "date": run_timestamp(),
    }


def experiment_name(config: DiffusionRunConfig, summary: dict) -> str:
    """The configured name, or one built from the fields that shape the run."""
    name = config.trainer.name or DEFAULT_EXPERIMENT_NAME
    if not re.search(r"\{.+?\}", name):
        return name

    name = name + "/arguments_hash-{arguments_hash}/date-{date}"
    if config.autoencoder is not None:
        name = f"LDM-{name}"
    if 'hybrid_dit' in config.model.architecture:
        name = f"SSM-{name}"
    if summary.get('scan_order', 'raster') != 'raster':
        name = f"{summary['scan_order'].capitalize()}-{name}"
    return name.format(**summary)


def main(config: DiffusionRunConfig) -> TrainState:
    prepare_process(config.trainer.wandb_offline, config.trainer.multi_host,
                    config.trainer.xla_flags, config.trainer.compilation_cache_dir)
    print(f"Local devices: {jax.local_devices()}")

    data = config.data.load(batch=config.trainer.batch_size)
    steps = config.trainer.total_steps(data)

    autoencoder = load_autoencoder(config)
    inputs = build_inputs(config)
    fields = model_fields(config, inputs.sample, autoencoder)
    model = models.build(config.model.architecture, **fields)
    process = config.preset()
    objective = DiffusionObjective(
        model, process, inputs,
        autoencoder=autoencoder,
        unconditional_prob=config.unconditional_prob,
        ema_decay=config.ema_decay,
        sampler=config.sampler,
        guidance=CFG(config.guidance) if config.guidance else None,
        steps=config.sampling_steps,
    )

    run_config = config.to_dict()
    # hash() is randomized per process; identical configs must map to the same
    # experiment
    arguments_hash = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode()).hexdigest()[:16]
    summary = run_summary(config, fields, arguments_hash)
    name = experiment_name(config, summary)
    print("Experiment_Name:", name)
    directory = os.path.join(config.trainer.checkpoint_dir, name)

    tracker = None
    if config.trainer.wandb_project is not None:
        tracker = WandbTracker(
            config.trainer.wandb_project, name, entity=config.trainer.wandb_entity,
            offline=config.trainer.wandb_offline,
            config={"run_config": run_config, "model": fields, "arguments": summary,
                    "dataset": {"name": summary["dataset"], "records": data.records,
                                "steps_per_epoch": data.steps_per_epoch},
                    "steps": steps})

    Manifest(
        config=run_config,
        model={"name": config.model.architecture, "fields": fields},
        inputs=inputs.to_json(),
        preset={"name": summary["preset"], "fields": dataclasses.asdict(config.preset)},
        autoencoder=(None if config.autoencoder is None
                     else {"name": config.autoencoder, "fields": dict(config.autoencoder_opts)}),
    ).write(directory)

    checkpoints = Checkpoints(directory, keep=config.trainer.keep)
    trainer = Trainer(
        objective, build_optimizer(config.optim, steps),
        key=jax.random.key(config.trainer.seed),
        mesh=config.trainer.mesh,
        layout=config.trainer.layout,
        accumulation=config.trainer.accumulation,
        dynamic_scale=config.trainer.dynamic_scale,
        checkpoints=checkpoints,
        tracker=tracker,
        profile=(Profile(os.path.join(directory, "profile"), config.trainer.profile_steps)
                 if config.trainer.profile_steps else None),
    )
    print(f"Training on {summary['dataset']} for {steps} steps "
          f"({data.steps_per_epoch} steps per epoch)")
    state = trainer.fit(
        data, steps=steps,
        log_every=config.trainer.log_every,
        eval_every=config.trainer.eval_every or data.steps_per_epoch,
        checkpoint_every=config.trainer.checkpoint_every or data.steps_per_epoch,
        metrics=build_eval_metrics(config.val_metrics),
    )
    if tracker is not None:
        dew.io.publish(checkpoints.path(checkpoints.latest), artifact_name(name), tracker=tracker)
    return state


def artifact_name(name: str) -> str:
    """The run name as a W&B artifact name, which allows no slashes."""
    return re.sub(r"[^\w.-]", "-", name)


if __name__ == '__main__':
    main(tyro.cli(tyro.conf.CascadeSubcommandArgs[DiffusionRunConfig]))
