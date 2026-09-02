"""Train a diffusion model over images or latents.

    python recipes/diffusion/train.py --data.dataset oxford_flowers102 \
        --data.image-size 128 --data.batch-size 32 --trainer.epochs 2000 \
        --model.architecture simple_dit \
        --model.config '{"patch_size": 4, "emb_features": 512, "num_layers": 12, "num_heads": 8}'

Architecture kwargs go through --model.config as one JSON object, straight to
the registry, so the wandb config is exactly what built the model.
"""

import hashlib
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

import jax
import tqdm
import tyro

from dew.config import JsonDict, ModelConfig, OptimConfig, RunConfig
from dew.data.dataloaders import load_data
from dew.inputs import ConditionalInputConfig, DiffusionInputConfig
from dew.inputs.processors import defaultTextEncodeModel
from dew.diffusion.transforms import get_diffusion_preset
from dew.registry import apply_precision_policy, build_model, canonicalize_architecture
from dew.sampling.euler import EulerAncestralSampler
from dew.training import ObjectiveTrainer, build_optimizer, prepare_process
from dew.training.distributed import DEFAULT_MIN_SHARD_SIZE

warnings.filterwarnings("ignore")
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
                           "schd-{noise_schedule}/arch-{architecture}/lr-{learning_rate}")

DEFAULT_BEST_TRACKER_METRIC = "val/clip_similarity"


@dataclass(frozen=True)
class DiffusionRunConfig(RunConfig):
    """A run, plus the diffusion objective's own knobs."""

    model: ModelConfig = field(
        default_factory=lambda: ModelConfig("unet", dict(DEFAULT_MODEL_CONFIG)))
    noise_schedule: Literal['cosine', 'karras', 'edm', 'flow', 'flow_matching'] = 'edm'
    min_snr_gamma: Optional[float] = None
    """min-SNR-gamma loss weighting (Hang et al. 2023); 5.0 is the paper value,
    unset keeps the schedule's own weighting."""
    flow_shift: float = 1.0
    """Resolution shift for the flow matching schedule, see
    dew.diffusion.schedules.flow.compute_resolution_shift."""
    autoencoder: Optional[Literal['stable_diffusion']] = None
    autoencoder_opts: JsonDict = field(
        default_factory=lambda: {"modelname": "pcuenq/sd-vae-ft-mse-flax"})
    val_metrics: list[Literal['clip', 'clip_score', 'fid']] = field(
        default_factory=lambda: ['clip'])
    validation_prompts: Optional[str] = None
    """Text file of captions, one per line, sampled from at validation instead
    of the dataset's own val split."""
    dataset_test: bool = False
    """Pull 2000 batches through the pipeline before training, for benchmarking."""


def load_autoencoder(config: DiffusionRunConfig):
    """The VAE for latent diffusion, with the shape it hands the model."""
    if config.autoencoder is None:
        return None, 3, config.data.image_size
    print("Using Stable Diffusion Autoencoder for Latent Diffusion Modeling")
    from dew.nn.autoencoders.sd_vae import StableDiffusionVAE
    autoencoder = StableDiffusionVAE(**config.autoencoder_opts)
    return (autoencoder, autoencoder.latent_channels,
            config.data.image_size // autoencoder.downscale_factor)


def model_kwargs(config: DiffusionRunConfig, channels: int, sample_size: int):
    """Canonical architecture name and the kwargs the registry builds it from."""
    architecture, suffix_flags = canonicalize_architecture(config.model.architecture)
    kwargs = apply_precision_policy(
        architecture, {**config.model.config, **suffix_flags},
        dtype=config.model.dtype, attention_impl=config.model.attention_impl)
    if kwargs.get('use_hilbert') and kwargs.get('use_zigzag'):
        raise ValueError("use_hilbert and use_zigzag are mutually exclusive")
    if architecture == 'diffusers_unet_simple':
        kwargs.update(sample_size=sample_size, in_channels=channels, out_channels=channels)
    else:
        kwargs['output_channels'] = channels
    return architecture, kwargs


def build_input_config(config: DiffusionRunConfig) -> DiffusionInputConfig:
    """Images conditioned on pretokenized text, encoded by the default CLIP."""
    return DiffusionInputConfig(
        sample_data_key='image',
        sample_data_shape=(config.data.image_size, config.data.image_size, 3),
        conditions=[
            ConditionalInputConfig(
                encoder=defaultTextEncodeModel(),
                conditioning_data_key='text',
                pretokenized=True,
                unconditional_input="",
                model_key_override="textcontext",
            )
        ],
    )


def build_eval_metrics(names: list[str]) -> list:
    """Validation metrics, imported lazily since each pulls its own weights."""
    metrics = []
    if 'clip' in names:
        from dew.eval.images import get_clip_metric
        print("Using legacy CLIP distance metric (val/clip_similarity) for validation")
        metrics.append(get_clip_metric())
    if 'clip_score' in names:
        from dew.eval.images import get_clip_score_metric
        print("Using CLIPScore (val/clip_score, higher is better) for validation")
        metrics.append(get_clip_score_metric())
    if 'fid' in names:
        from dew.eval.fid import get_fid_metric
        print("Using per-batch FID (val/fid) for validation")
        metrics.append(get_fid_metric())
    return metrics


def validation_prompt_batches(path: str, encoder, batch_size: int, steps: int):
    """Validation batches of captions read from a file, and how many there are.

    A fixed prompt list keeps the sampled grids comparable across runs, which a
    shuffled val split does not, and it is the only validation a caption-less
    dataset can offer a text-conditioned model.
    """
    with open(path) as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts in {path}")

    def get_val_dataset():
        for step in range(steps):
            start = step * batch_size
            batch = [prompts[(start + i) % len(prompts)] for i in range(batch_size)]
            # dict(): the tokenizer's own mapping is one pytree leaf, so the
            # batch would never reach the devices field by field
            yield {"text": dict(encoder.tokenize(batch))}

    return get_val_dataset, len(prompts)


def run_summary(config: DiffusionRunConfig, model_config: dict, arguments_hash: str) -> dict:
    """Flat view of the run, for the wandb config and the experiment name."""
    return {
        **model_config,
        "architecture": config.model.architecture,
        "dataset": config.data.dataset,
        "image_size": config.data.image_size,
        "batch_size": config.data.batch_size,
        "noise_schedule": config.noise_schedule,
        "learning_rate": config.optim.learning_rate,
        "epochs": config.trainer.epochs,
        "arguments_hash": arguments_hash,
        "date": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
    }


def experiment_name(config: DiffusionRunConfig, summary: dict, latent: bool) -> str:
    """The configured name, or one built from the fields that shape the run."""
    name = config.trainer.name or DEFAULT_EXPERIMENT_NAME
    if not re.search(r"\{.+?\}", name):
        return name

    name = name + "/arguments_hash-{arguments_hash}/date-{date}"
    if latent:
        name = f"LDM-{name}"
    if 'hybrid_dit' in config.model.architecture:
        name = f"SSM-{name}"
    if summary.get('use_hilbert'):
        name = f"Hilbert-{name}"
    return name.format(**summary)


def main(config: DiffusionRunConfig) -> ObjectiveTrainer:
    prepare_process(config.data.augmentation_mode, config.trainer.wandb_offline,
                    config.trainer.multi_host)
    print(f"Local devices: {jax.local_devices()}")

    data = load_data(config.data)

    if config.dataset_test:
        dataset = iter(data['train']())
        for _ in tqdm.tqdm(range(2000)):
            next(dataset)

    datalen = data['train_len']
    batches = datalen // config.data.batch_size
    steps_per_epoch = config.trainer.steps_per_epoch or batches

    autoencoder, channels, sample_size = load_autoencoder(config)
    architecture, model_config = model_kwargs(config, channels, sample_size)
    model = build_model(architecture, model_config)

    input_config = build_input_config(config)
    eval_metrics = build_eval_metrics(config.val_metrics)
    train_schedule, sampling_schedule, prediction_transform = get_diffusion_preset(
        config.noise_schedule, shift=config.flow_shift, min_snr_gamma=config.min_snr_gamma,
    )

    run_config = config.to_dict()
    # hash() is randomized per process; identical configs must map to the same
    # experiment
    arguments_hash = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode()).hexdigest()[:16]
    summary = run_summary(config, model_config, arguments_hash)
    name = experiment_name(config, summary, latent=autoencoder is not None)
    print("Experiment_Name:", name)

    checkpoint_dir = config.trainer.checkpoint_dir
    if config.trainer.checkpoint_fs == 'gcs':
        checkpoint_dir = f"gs://{checkpoint_dir}"

    wandb_config: Optional[dict[str, Any]] = None
    if config.trainer.wandb_project is not None:
        wandb_config = {
            "project": config.trainer.wandb_project,
            "entity": config.trainer.wandb_entity,
            "name": name,
            "config": {
                "model": model_config,
                "architecture": architecture,
                "dataset": {
                    "name": config.data.dataset,
                    "length": datalen,
                    "batches": batches,
                },
                "learning_rate": config.optim.learning_rate,
                "batch_size": config.data.batch_size,
                "epochs": config.trainer.epochs,
                "input_shapes": input_config.get_input_shapes(autoencoder=autoencoder),
                "input_config": input_config.serialize(),
                "arguments": summary,
                "run_config": run_config,
                "autoencoder": config.autoencoder,
                "autoencoder_opts": json.dumps(config.autoencoder_opts),
                "arguments_hash": arguments_hash,
            },
        }
        if config.trainer.resume_last_run is not None:
            wandb_config['id'] = config.trainer.resume_last_run

    trainer = ObjectiveTrainer(
        model,
        optimizer=build_optimizer(config.optim, steps_per_epoch),
        input_config=input_config,
        noise_schedule=train_schedule,
        rngs=jax.random.PRNGKey(4),
        name=name,
        model_output_transform=prediction_transform,
        load_from_checkpoint=config.trainer.load_from_checkpoint,
        checkpoint_step=config.trainer.checkpoint_step,
        wandb_config=wandb_config,
        distributed_training=config.trainer.distributed_training,
        checkpoint_base_path=checkpoint_dir,
        autoencoder=autoencoder,
        use_dynamic_scale=config.optim.use_dynamic_scale,
        native_resolution=config.data.image_size,
        max_checkpoints_to_keep=config.trainer.max_checkpoints_to_keep,
        eval_metrics=eval_metrics,
        best_tracker_metric=config.trainer.best_tracker_metric or DEFAULT_BEST_TRACKER_METRIC,
        ema_decay=config.trainer.ema_decay,
        grad_accum_steps=config.optim.grad_accum_steps,
        fsdp_size=config.trainer.fsdp_size,
        fsdp_min_param_size=config.trainer.fsdp_min_param_size or DEFAULT_MIN_SHARD_SIZE,
        logical_axis_rules=config.trainer.logical_axis_rules,
        sharding_tolerance=config.trainer.sharding_tolerance,
        compilation_cache_dir=config.trainer.compilation_cache_dir,
        profile_steps=config.trainer.profile_steps,
        log_every=config.trainer.log_every,
    )

    if trainer.distributed_training:
        print("Distributed Training enabled")
    print(f"Training on {config.data.dataset} dataset with {steps_per_epoch} steps per epoch")

    if config.validation_prompts is not None:
        data['val'], data['val_len'] = validation_prompt_batches(
            config.validation_prompts,
            input_config.conditions[0].encoder,
            data['local_batch_size'],
            config.data.val_steps_per_epoch,
        )

    trainer.fit(
        data,
        training_steps_per_epoch=steps_per_epoch,
        epochs=config.trainer.epochs,
        sampler_class=EulerAncestralSampler,
        sampling_noise_schedule=sampling_schedule,
        val_steps_per_epoch=config.data.val_steps_per_epoch,
        checkpoint_every_steps=config.trainer.checkpoint_every_steps,
    )
    return trainer


if __name__ == '__main__':
    main(tyro.cli(DiffusionRunConfig))
