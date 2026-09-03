"""Train an autoregressive language model over a directory of token files.

A sibling of the diffusion and JEPA recipes: same trainer, same sharding, same
checkpoints, and a different objective. The data is not images but the
`train.bin` / `val.bin` / `meta.json` a tokenizer run wrote, so the recipe
takes the vocabulary from the data rather than the command line.

    python tools/tokenize_text.py --input data/shakespeare.txt \
        --out data/shakespeare-byte --tokenizer byte
    python recipes/lm/train.py --data.dataset data/shakespeare-byte \
        --sequence-length 256 --data.batch-size 32 --trainer.epochs 10 \
        --model.config '{"emb_features": 384, "num_layers": 6, "num_heads": 6}'
"""

import json
import os
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import jax
import tyro

from dew.config import DataConfig, ModelConfig, OptimConfig, RunConfig
from dew.data.dataloaders import load_data
from dew.eval import get_perplexity_metric
from dew.objectives.lm import LMObjective
from dew.registry import apply_precision_policy, build_model, canonicalize_architecture
from dew.training import ObjectiveTrainer, build_optimizer, prepare_process
from dew.training.distributed import DEFAULT_MIN_SHARD_SIZE

os.environ['TOKENIZERS_PARALLELISM'] = "false"

DEFAULT_MODEL_CONFIG = {"emb_features": 512, "num_layers": 8, "num_heads": 8}


@dataclass(frozen=True)
class LmRunConfig(RunConfig):
    """A run, plus the language model's own knobs."""

    model: ModelConfig = field(
        default_factory=lambda: ModelConfig("causal_transformer", dict(DEFAULT_MODEL_CONFIG)))
    data: DataConfig = field(default_factory=lambda: DataConfig(batch_size=32))
    optim: OptimConfig = field(
        default_factory=lambda: OptimConfig(
            learning_rate=6e-4, learning_rate_peak=6e-4, learning_rate_end=6e-5,
            weight_decay=0.1, clip_grads=1.0))
    sequence_length: int = 256
    """Context the model is trained on; a batch row carries one token more."""
    tokenizer: str = "byte"
    """What the token files were written with: 'byte', or an HF tokenizer name."""
    sample_prompt: str = ""
    """Prompt the validation samples continue; empty continues a newline."""
    sample_tokens: int = 128
    """Tokens generated per validation sample; 0 logs no text."""


def read_meta(dataset: str) -> dict:
    """The tokenizer and vocabulary the token files were written with."""
    meta_path = Path(dataset) / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"{meta_path} is missing: --data.dataset is the token directory that "
            "tools/tokenize_text.py wrote, not a dataset name")
    return json.loads(meta_path.read_text())


def token_data_config(config: LmRunConfig) -> DataConfig:
    """Where the run's sequence length and tokenizer reach the token loader."""
    return replace(config.data, sequence_length=config.sequence_length,
                   tokenizer=config.tokenizer)


def context_length(config: LmRunConfig, samples: Optional[dict]) -> int:
    """How far the position table and the KV cache have to reach.

    Generation decodes into a cache sized once at build time, so a sampling
    budget longer than the training context is what decides the model's
    max_seq_len rather than the sequence length being trained on.
    """
    if samples is None:
        return config.sequence_length
    return max(config.sequence_length,
               len(samples["prompt"]) + samples["max_new_tokens"])


def build_lm(config: LmRunConfig, vocab_size: int, max_seq_len: int):
    """The model, and the config the registry built it from."""
    architecture, suffix_flags = canonicalize_architecture(config.model.architecture)
    model_config = apply_precision_policy(
        architecture,
        # Data decides the vocabulary. Training and sampling decide the context.
        {**config.model.config, **suffix_flags, "max_seq_len": max_seq_len,
         "vocab_size": vocab_size},
        dtype=config.model.dtype, attention_impl=config.model.attention_impl)
    return build_model(architecture, model_config), model_config


def build_tokenizer(name: str):
    """The tokenizer that decodes generated ids back into text."""
    # Deferred: only a run that logs samples needs a tokenizer at all, and an
    # HF one reads its files off disk.
    from dew.data.text import ByteTokenizer, HFTokenizer
    return ByteTokenizer() if name == "byte" else HFTokenizer(name)


def build_samples(config: LmRunConfig) -> Optional[dict]:
    """What the objective generates and decodes at every validation."""
    if config.sample_tokens <= 0:
        return None
    tokenizer = build_tokenizer(config.tokenizer)
    return {
        "prompt": tokenizer.encode(config.sample_prompt or "\n"),
        "max_new_tokens": config.sample_tokens,
        "decode": tokenizer.decode,
    }


def run_summary(config: LmRunConfig, model_config: dict) -> dict:
    """Flat view of the run, for the wandb config."""
    return {
        **model_config,
        "architecture": config.model.architecture,
        "dataset": config.data.dataset,
        "sequence_length": config.sequence_length,
        "tokenizer": config.tokenizer,
        "batch_size": config.data.batch_size,
        "learning_rate": config.optim.learning_rate,
        "epochs": config.trainer.epochs,
    }


def main(config: LmRunConfig) -> ObjectiveTrainer:
    prepare_process(config.data.augmentation_mode, config.trainer.wandb_offline,
                    config.trainer.multi_host, config.trainer.xla_flags)

    checkpoint_dir = config.trainer.checkpoint_dir
    if config.trainer.checkpoint_fs == 'gcs':
        checkpoint_dir = f"gs://{checkpoint_dir}"

    meta = read_meta(config.data.dataset)
    if config.tokenizer != meta['tokenizer']:
        # Decoding with a different tokenizer than the ids were written with
        # produces text that says nothing about the model.
        raise ValueError(
            f"--tokenizer {config.tokenizer} does not match the token files, which "
            f"were written with {meta['tokenizer']}")
    vocab_size = int(meta['vocab_size'])

    data = load_data(token_data_config(config))
    steps_per_epoch = (config.trainer.steps_per_epoch
                       or data['train_len'] // config.data.batch_size)

    samples = build_samples(config)
    model, model_config = build_lm(config, vocab_size,
                                   context_length(config, samples))
    objective = LMObjective(
        model,
        config.sequence_length,
        vocab_size=vocab_size,
        ema_decay=config.trainer.ema_decay,
        samples=samples,
    )

    name = config.trainer.name or (
        f"lm-{Path(config.data.dataset).name}/seq-{config.sequence_length}/"
        f"emb-{model.emb_features}/layers-{model.num_layers}/"
        f"lr-{config.optim.learning_rate}/"
        f"date-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
    print("Experiment_Name:", name)

    wandb_config: Optional[dict[str, Any]] = None
    if config.trainer.wandb_project is not None:
        wandb_config = {
            "project": config.trainer.wandb_project,
            "entity": config.trainer.wandb_entity,
            "name": name,
            "config": {
                "model_config": model_config,
                "architecture": config.model.architecture,
                "dataset": {"name": config.data.dataset, "length": data['train_len'],
                            "tokens": meta.get('train_tokens')},
                "arguments": run_summary(config, model_config),
                "run_config": config.to_dict(),
            },
        }
        if config.trainer.resume_last_run is not None:
            wandb_config['id'] = config.trainer.resume_last_run

    trainer = ObjectiveTrainer(
        model=model,
        optimizer=build_optimizer(config.optim, steps_per_epoch),
        rngs=jax.random.PRNGKey(4),
        input_config=None,
        objective=objective,
        name=name,
        wandb_config=wandb_config,
        distributed_training=config.trainer.distributed_training,
        checkpoint_base_path=checkpoint_dir,
        checkpoint_step=config.trainer.checkpoint_step,
        load_from_checkpoint=config.trainer.load_from_checkpoint,
        max_checkpoints_to_keep=config.trainer.max_checkpoints_to_keep,
        eval_metrics=[get_perplexity_metric()],
        best_tracker_metric=config.trainer.best_tracker_metric or "val/perplexity",
        grad_accum_steps=config.optim.grad_accum_steps,
        use_dynamic_scale=config.optim.use_dynamic_scale,
        fsdp_size=config.trainer.fsdp_size,
        fsdp_min_param_size=config.trainer.fsdp_min_param_size or DEFAULT_MIN_SHARD_SIZE,
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
    main(tyro.cli(LmRunConfig))
