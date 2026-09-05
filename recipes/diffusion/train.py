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
import re

import jax
import tyro

from dew.objectives.diffusion import DiffusionRunConfig
from dew.registry import datasets, presets
from dew.training import TrainState, prepare_process, run_timestamp

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
    prepare_process(config.trainer.wandb, config.trainer.multi_host,
                    config.trainer.xla_flags, config.trainer.compilation_cache_dir)
    print(f"Local devices: {jax.local_devices()}")

    # The objective first: its conditions are what read the dataset's
    # captions, so the encoder the run names decides the tokens.
    objective = config.build()
    data = config.data.load(batch=config.trainer.batch_size,
                            tokenize=objective.inputs.tokenize)
    fields = config.model_fields(objective.autoencoder)

    # hash() is randomized per process; identical configs must map to the same
    # experiment
    arguments_hash = hashlib.sha256(
        json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()[:16]
    summary = run_summary(config, fields, arguments_hash)
    return config.train(
        objective, data, name=experiment_name(config, summary),
        metrics=config.build_eval_metrics(),
        summary={"model": fields, "arguments": summary,
                 "dataset": {"name": summary["dataset"], "records": data.records,
                             "steps_per_epoch": data.steps_per_epoch}})


if __name__ == '__main__':
    main(tyro.cli(tyro.conf.CascadeSubcommandArgs[DiffusionRunConfig]))
