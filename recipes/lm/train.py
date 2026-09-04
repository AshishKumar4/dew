"""Train an autoregressive language model over a directory of token files.

A sibling of the diffusion and JEPA recipes: same trainer, same sharding, same
checkpoints, and a different objective. The data is not images but the
`train.bin` / `val.bin` / `meta.json` a tokenizer run wrote, so the recipe
takes the vocabulary from the data rather than the command line.

    python tools/tokenize_text.py --input data/shakespeare.txt \
        --out data/shakespeare-byte --tokenizer byte
    python recipes/lm/train.py --data.path data/shakespeare-byte \
        --data.seq-len 256 --trainer.batch-size 32 --trainer.epochs 10 \
        --model.config '{"emb_features": 384, "num_layers": 6, "num_heads": 6}'

`data:packed-tokens` packs whole documents into the windows instead.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jax
import tyro

import dew.io
from dew.config import ModelConfig, OptimConfig, RunConfig
from dew.data import ByteTokenizer, HFTokenizer, PackedTokens, TokenWindows
from dew.objectives.lm import LMObjective, Samples
from dew.registry import datasets, metrics, models
from dew.training import (Checkpoints, Trainer, TrainState, WandbTracker,
                          build_optimizer, prepare_process, run_timestamp)

# HF tokenizers fork a thread pool; grain's workers fork the process.
os.environ['TOKENIZERS_PARALLELISM'] = "false"

DEFAULT_MODEL_CONFIG = {"emb_features": 512, "num_layers": 8, "num_heads": 8}


@dataclass(frozen=True)
class LmRunConfig(RunConfig):
    """A run, plus the language model's own knobs."""

    model: ModelConfig = field(
        default_factory=lambda: ModelConfig("causal_transformer", dict(DEFAULT_MODEL_CONFIG)))
    data: datasets.union = field(default_factory=TokenWindows)
    optim: OptimConfig = field(
        default_factory=lambda: OptimConfig(
            learning_rate=6e-4, learning_rate_peak=6e-4, learning_rate_end=6e-5,
            weight_decay=0.1, clip_grads=1.0))
    tokenizer: str = "byte"
    """What the token files were written with: 'byte', or an HF tokenizer name."""
    ema_decay: float = 0.999
    sample_prompt: str = ""
    """Prompt the validation samples continue; empty continues a newline."""
    sample_tokens: int = 128
    """Tokens generated per validation sample; 0 logs no text."""
    pretrained: Optional[str] = None
    """Hugging Face decoder to continue training: a hub repo id or a local
    directory in that layout. The checkpoint decides the architecture, so
    --model.config may then carry max_seq_len and nothing else."""
    balance_rate: Optional[float] = None
    """How far a sparse run moves each router's balancing bias against its
    load every step (DeepSeek's aux-loss-free balancing). Needs a mixture
    with bias=True; unset leaves the bias where it is."""

    def __post_init__(self):
        if not isinstance(self.data, (TokenWindows, PackedTokens)):
            raise ValueError(
                "the language model recipe trains on token files: "
                "data:token-windows or data:packed-tokens")


def read_meta(path: Optional[str]) -> dict:
    """The tokenizer and vocabulary the token files were written with."""
    if not path:
        raise ValueError("--data.path is the token directory tools/tokenize_text.py wrote")
    meta_path = Path(path) / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"{meta_path} is missing: --data.path is the token directory that "
            "tools/tokenize_text.py wrote, not a dataset name")
    return json.loads(meta_path.read_text())


def context_length(config: LmRunConfig, samples: Optional[Samples]) -> int:
    """How far the position table and the KV cache have to reach.

    Generation decodes into a cache sized once at build time, so a sampling
    budget longer than the training context is what decides the model's
    max_seq_len rather than the sequence length being trained on.
    """
    if samples is None:
        return config.data.seq_len
    return max(config.data.seq_len, len(samples.prompt) + samples.max_new_tokens)


def model_fields(config: LmRunConfig, vocab_size: int, max_seq_len: int) -> dict:
    """The fields the registry builds the model from."""
    # Data decides the vocabulary. Training and sampling decide the context.
    return {**config.model.fields(), "max_seq_len": max_seq_len, "vocab_size": vocab_size}


def load_pretrained(config: LmRunConfig, vocab_size: int, max_seq_len: int,
                    meta: dict):
    """The decoder a --pretrained run continues, its variables and the fields
    it was built from.

    The checkpoint decides every architecture field, so the only thing
    --model.config may still say is how far the KV cache reaches. The fields
    that come back are dew's, not the checkpoint's, so a pretrained run logs
    the same vocabulary a fresh one does, compute dtype and kernel included.
    The tokenizer of the token files has to be the one the checkpoint was
    trained with: continuing pretraining on ids from another vocabulary trains
    the embedding table against noise.
    """
    from dew.interop.hf_decoders import load_pretrained_decoder

    overridden = sorted(set(config.model.config) - {"max_seq_len"})
    if overridden:
        raise ValueError(
            f"--model.config carries {overridden}, which the checkpoint at "
            f"{config.pretrained} decides. Only max_seq_len is still a choice.")

    model, variables, fields = load_pretrained_decoder(
        config.pretrained,
        dtype=config.model.dtype, attention_impl=config.model.attention_impl,
        max_seq_len=config.model.config.get("max_seq_len", max_seq_len))
    expected = checkpoint_tokenizer(config.pretrained)
    if meta["tokenizer"] != expected:
        raise ValueError(
            f"the token files were written with {meta['tokenizer']}, and "
            f"{config.pretrained} expects {expected}. Retokenize with "
            f"--tokenizer {expected}.")
    # A decoder's embedding table is usually padded past the tokenizer's ids
    # (Qwen3 stores 151936 rows for 151669 tokens), so covering them is the
    # requirement, not matching the count.
    if model.vocab_size < vocab_size:
        raise ValueError(
            f"{config.pretrained} has room for {model.vocab_size} ids and the "
            f"token files use {vocab_size}")
    return model, variables, fields


def checkpoint_tokenizer(pretrained: str) -> str:
    """The tokenizer name a checkpoint expects its ids to come from.

    A hub repo is its own tokenizer's name. A directory written by
    save_pretrained_decoder records the name it was exported with, since the
    path it happens to sit at says nothing.
    """
    generation_config = Path(pretrained) / "generation_config.json"
    if generation_config.is_file():
        recorded = json.loads(generation_config.read_text()).get("tokenizer_name")
        if recorded:
            return recorded
    return pretrained


def build_tokenizer(name: str):
    """The tokenizer that decodes generated ids back into text."""
    return ByteTokenizer() if name == "byte" else HFTokenizer(name)


def build_samples(config: LmRunConfig) -> Optional[Samples]:
    """What the objective generates and decodes at every validation."""
    if config.sample_tokens <= 0:
        return None
    tokenizer = build_tokenizer(config.tokenizer)
    return Samples(
        prompt=tokenizer.encode(config.sample_prompt or "\n"),
        max_new_tokens=config.sample_tokens,
        temperature=0.8,
        top_k=40,
        decode=tokenizer.decode,
    )


def run_summary(config: LmRunConfig, fields: dict) -> dict:
    """Flat view of the run, for the tracker."""
    return {
        **fields,
        "architecture": config.model.architecture,
        "dataset": config.data.path,
        "sequence_length": config.data.seq_len,
        "tokenizer": config.tokenizer,
        "batch_size": config.trainer.batch_size,
        "learning_rate": config.optim.learning_rate,
    }


def main(config: LmRunConfig) -> TrainState:
    prepare_process(config.trainer.wandb, config.trainer.multi_host,
                    config.trainer.xla_flags, config.trainer.compilation_cache_dir)

    meta = read_meta(config.data.path)
    if config.tokenizer != meta['tokenizer']:
        # Decoding with a different tokenizer than the ids were written with
        # produces text that says nothing about the model.
        raise ValueError(
            f"--tokenizer {config.tokenizer} does not match the token files, which "
            f"were written with {meta['tokenizer']}")
    vocab_size = int(meta['vocab_size'])

    data = config.data.load(batch=config.trainer.batch_size)
    if data.steps_per_epoch < 1:
        raise ValueError(
            f"{data.records} training windows do not fill one batch of "
            f"{config.trainer.batch_size}, so an epoch is no steps at all: read "
            "more data or lower --trainer.batch-size")
    steps = config.trainer.total_steps(data)

    samples = build_samples(config)
    context = context_length(config, samples)
    pretrained = None
    if config.pretrained is None:
        fields = model_fields(config, vocab_size, context)
        model = models.build(config.model.architecture, **fields)
    else:
        model, pretrained, fields = load_pretrained(config, vocab_size, context, meta)
    objective = LMObjective(
        model,
        config.data.seq_len,
        ema_decay=config.ema_decay,
        samples=samples,
        pretrained=pretrained,
        balance_rate=config.balance_rate,
    )

    name = config.trainer.name or (
        f"lm-{Path(config.data.path).name}/seq-{config.data.seq_len}/"
        f"emb-{model.emb_features}/layers-{model.num_layers}/"
        f"lr-{config.optim.learning_rate}/"
        f"date-{run_timestamp()}")
    print("Experiment_Name:", name)
    directory = os.path.join(config.trainer.checkpoint_dir, name)

    run_config = config.to_dict()
    tracker = None
    if config.trainer.wandb is not None:
        tracker = WandbTracker(
            config.trainer.wandb.project, name, entity=config.trainer.wandb.entity,
            offline=config.trainer.wandb.offline,
            config={"run_config": run_config, "model": fields,
                    "arguments": run_summary(config, fields),
                    "dataset": {"path": config.data.path, "records": data.records,
                                "tokens": meta.get('train_tokens')},
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
        metrics=(metrics.perplexity(),),
    )
    print(f"Training finished in {time.time() - start:.0f}s")
    if tracker is not None:
        dew.io.publish(checkpoints.path(checkpoints.latest), re.sub(r"[^\w.-]", "-", name),
                       tracker=tracker)
    return state


if __name__ == '__main__':
    main(tyro.cli(tyro.conf.CascadeSubcommandArgs[LmRunConfig]))
