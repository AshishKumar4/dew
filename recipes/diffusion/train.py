"""Train a diffusion model over images or latents.

    python recipes/diffusion/train.py --data.image-size 128 --trainer.batch-size 32 \
        --trainer.epochs 2000 --model.architecture simple_dit \
        --model.config '{"patch_size": 4, "emb_features": 512, "num_layers": 12, "num_heads": 8}'

The dataset is a subcommand over the registry (`data:cc12m --data.path /mnt/gcs`),
and so are the preset (`preset:flow --preset.shift 3.0`), the sampler, the text
condition (`text:None` for an unconditional run) and the autoencoder
(`autoencoder:stable-diffusion-autoencoder`). Architecture kwargs go through
--model.config as one JSON object, straight to the registry. The run spec is
`dew.objectives.diffusion.DiffusionRunConfig`, saved as run.json next to the
checkpoints, and `config.build()` is the one construction training and
inference share.
"""

import hashlib
import json
import os
import re

import jax
import tyro

import dew.io
from dew.objectives.diffusion import DiffusionRunConfig
from dew.registry import datasets, presets
from dew.training import (Checkpoints, Profile, Trainer, TrainState, WandbTracker,
                          build_optimizer, prepare_process, run_timestamp)

# HF tokenizers fork a thread pool; grain's workers fork the process.
os.environ['TOKENIZERS_PARALLELISM'] = "false"

DEFAULT_EXPERIMENT_NAME = ("dataset-{dataset}/image_size-{image_size}/batch-{batch_size}/"
                           "schd-{preset}/arch-{architecture}/lr-{learning_rate}")


def run_summary(config: DiffusionRunConfig, fields: dict, arguments_hash: str) -> dict:
    """Flat view of the run, for the experiment name."""
    sample = config.sample_field()
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

    # The objective first: its conditions are what read the dataset's
    # captions, so the encoder the run names decides the tokens.
    objective = config.build()
    data = config.data.load(batch=config.trainer.batch_size,
                            tokenize=objective.inputs.tokenize)
    steps = config.trainer.total_steps(data)
    fields = config.model_fields(objective.autoencoder)

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
        metrics=config.build_eval_metrics(),
    )
    if tracker is not None:
        dew.io.publish(checkpoints.path(checkpoints.latest), artifact_name(name), tracker=tracker)
    return state


def artifact_name(name: str) -> str:
    """The run name as a W&B artifact name, which allows no slashes."""
    return re.sub(r"[^\w.-]", "-", name)


if __name__ == '__main__':
    main(tyro.cli(tyro.conf.CascadeSubcommandArgs[DiffusionRunConfig]))
