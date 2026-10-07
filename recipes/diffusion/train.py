"""Train a diffusion model over images or latents.

    python recipes/diffusion/train.py --data.path ~/.cache/dew/datasets/oxford_flowers102/2.1.1 \\
        --data.image-size 128 --trainer.batch-size 32 --trainer.epochs 2000 \\
        --model simple_dit --model.patch-size 4 --model.emb-features 512 \\
        --model.num-layers 12 --model.num-heads 8

The dataset is a subcommand over the registry (`data:cc12m --data.path /mnt/gcs`),
and so are the preset (`preset:flow --preset.shift 3.0`), the solver, the text
condition (`text:None` for an unconditional run) and the autoencoder
(`autoencoder:stable-diffusion-autoencoder`). The corpora this recipe names
(`oxford-flowers102`, the default, `cc12m`, the LAION sets) are `CORPORA` below,
values of the specs `dew.data` reads. `--model` picks the model's class and
`--model.<field>` sets each of its fields. The run spec is
`dew.objectives.diffusion.DiffusionRunConfig`, saved as run.json next to the
checkpoints, and training and inference both build from `config.build()`.

`--pretrained stabilityai/stable-diffusion-3.5-medium preset:none` fine-tunes a
published pipeline (SD3, Flux, Qwen-Image or a UNet) on its own conditioning,
autoencoder and convention; `mode:flow-grpo --mode.reward clip_score` trains the
model with Flow-GRPO on that reward instead of the denoising loss.
"""

import dataclasses
import functools
import hashlib
import json
import operator
import re
from typing import Annotated

import jax
import tyro

from dew.data import ArrayRecordImages, DatasetSpec, OnlineImages, TFDSImages
from dew.objectives.diffusion import DiffusionRunConfig
from dew.registry import datasets, models, presets
from dew.training import TrainState, prepare_process, run_timestamp

# The corpora this recipe trains on, each a value of a core spec: where the
# data lives and how it is captioned. On the command line one is a subcommand
# (`data:cc12m --data.path /mnt/gcs`); in a run's record it is the core spec
# with these fields, so the record reads back without this file.
FLOWER_CAPTIONS = ("a photo of a {}", "a photo of a {} flower", "This is a photo of a {}",
                   "This is a photo of a {} flower", "A photo of a {} flower")

# The msml612 shards live in gs://msml612-diffusion-data, read through a gcs
# fuse mount handed over as `path`; the regional url tables are fetched as read.
REGIONAL = "gs://dew-datasets-regional/datasets/"
CORPORA: dict[str, DatasetSpec] = {
    "oxford-flowers102": TFDSImages(name="oxford_flowers102", caption_templates=FLOWER_CAPTIONS),
    # laion-aesthetics-12M (score >= 6) plus MS-COCO 2017: 228 shards, 236 GiB, about 15M samples.
    "laion12m-coco": ArrayRecordImages(shards=("arrayrecord2/laion12m_coco",)),
    # laion-2B-en aesthetic >= 4.2 subset: 569 shards, 550 GiB, larger but noisier.
    "laion2b-aesthetic": ArrayRecordImages(shards=("arrayrecord2/laion2B-en-aesthetic",)),
    # diffusiondb (SD synthetic images and prompts): 31 shards, 60 GiB, 1.97M samples.
    "diffusiondb": ArrayRecordImages(shards=("arrayrecord2/diffusiondb",)),
    # Conceptual Captions 3M: 50 shards, 37 GiB, about 3.3M samples (shard 00039 missing).
    "cc3m": ArrayRecordImages(shards=("arrayrecord2/cc3m",)),
    # The four msml612 datasets together, about 883 GiB and 20M samples.
    "combined-msml612": ArrayRecordImages(shards=(
        "arrayrecord2/laion12m_coco", "arrayrecord2/laion2B-en-aesthetic",
        "arrayrecord2/diffusiondb", "arrayrecord2/cc3m")),
    "cc12m": ArrayRecordImages(shards=("arrayrecord2/cc12m",)),
    # Four arrayrecord2 shard sets of the msml612 bucket, about 30M samples.
    "combined-30m": ArrayRecordImages(shards=(
        "arrayrecord2/laion-aesthetics-12m+mscoco-2017", "arrayrecord2/cc12m",
        "arrayrecord2/aestheticCoyo_0.26_clip_5.5aesthetic_256plus",
        "arrayrecord2/playground+leonardo_x4+cc3m.parquet")),
    # Every url table in the regional bucket; the liked sets are listed several
    # times over, which weights them up.
    "combined-online": OnlineImages(sources=tuple(REGIONAL + name for name in (
        "laion-aesthetics-12m+mscoco-2017", "coyo700m-aesthetic-5.4_25M",
        "leonardo-liked-1.8m", "leonardo-liked-1.8m", "leonardo-liked-1.8m", "cc12m",
        "playground-liked", "leonardo-liked-1.8m", "leonardo-liked-1.8m", "cc3m", "cc3m",
        "laion2B-en-aesthetic-4.2_37M"))),
}
Corpus = functools.reduce(operator.or_, (Annotated[type(spec), tyro.conf.subcommand(name, default=spec)]
                                           for name, spec in CORPORA.items()))
# Every registered spec stays its own subcommand beside the corpora, as in the
# core config; the union holds what has registered by now.
Registered = datasets.union


def corpus_name(spec: DatasetSpec) -> str:
    """The corpus `spec` is, by the fields its `CORPORA` entry sets, or its
    spec's registry name: what the experiment name says it trained on."""
    for name, corpus in CORPORA.items():
        chosen = [field.name for field in dataclasses.fields(corpus)
                  if getattr(corpus, field.name) != getattr(type(corpus)(), field.name)]
        if type(spec) is type(corpus) and all(
                getattr(spec, name) == getattr(corpus, name) for name in chosen):
            return name
    return datasets.alias_of(type(spec))


@dataclasses.dataclass(frozen=True)
class DiffusionRecipeConfig(DiffusionRunConfig):
    """`DiffusionRunConfig` with this recipe's corpora on the command line,
    flowers by default."""

    data: Corpus | Registered = dataclasses.field(
        default_factory=lambda: CORPORA["oxford-flowers102"])


DEFAULT_EXPERIMENT_NAME = ("dataset-{dataset}/image_size-{image_size}/batch-{batch_size}/"
                           "schd-{preset}/arch-{architecture}/lr-{learning_rate}")


def run_summary(config: DiffusionRunConfig, fields: dict, arguments_hash: str) -> dict:
    """Flat view of the run, for the experiment name."""
    sample = config.sample_field()
    return {
        **fields,
        "architecture": config.pretrained or models.label(config.model.name),
        "dataset": corpus_name(config.data),
        "image_size": sample.shape[-2],
        "batch_size": config.trainer.batch_size,
        "preset": "source" if config.preset is None else presets.alias_of(type(config.preset)),
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
    if models.label(config.model.name) == 'hybrid_dit':
        name = f"SSM-{name}"
    if summary.get('scan_order', 'raster') != 'raster':
        name = f"{summary['scan_order'].capitalize()}-{name}"
    return name.format(**summary)


def main(config: DiffusionRunConfig) -> TrainState:
    prepare_process(config.trainer.wandb, config.trainer.multi_host,
                    config.trainer.xla_flags, config.trainer.compilation_cache_dir,
                    layout=config.trainer.layout)
    print(f"Local devices: {jax.local_devices()}")

    config = config.pinned()
    # The objective first: its conditions are what read the dataset's
    # captions, so the encoder the run names decides the tokens.
    objective = config.build()
    data = config.data.load(batch=config.trainer.batch_size,
                            tokenize=objective.inputs.tokenize)
    # A pretrained pipeline's fields are its checkpoint's, which it records.
    fields = {} if config.pretrained is not None else config.model_fields(objective.autoencoder)

    # hash() is randomized per process; identical configs must map to the same
    # experiment
    arguments_hash = hashlib.sha256(
        json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()[:16]
    summary = run_summary(config, fields, arguments_hash)
    return config.train(
        objective, data, name=experiment_name(config, summary),
        metrics=config.build_eval_metrics(), rollout=config.rollout(objective),
        summary={"model": fields, "arguments": summary,
                 "dataset": {"name": summary["dataset"], "records": data.records,
                             "steps_per_epoch": data.steps_per_epoch}})


if __name__ == '__main__':
    main(DiffusionRecipeConfig.cli())
