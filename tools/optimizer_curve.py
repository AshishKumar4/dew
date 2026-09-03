#!/usr/bin/env python3
"""One optimizer's loss curve at a fixed token budget, on a token dataset.

Every arm of a comparison sees the same model, the same seed and the same
batches in the same order, so the curves differ only by the solver. The step
is the trainer's own compiled step and the model and the data come from
`recipes/lm/train.py`, so nothing here is a second wiring of a run. What is
local is the loop, which records every step's loss instead of logging one per
interval.

`--optimizer` takes an `OPTIMIZER_MAP` name, or `muon-unsplit` for
`optax.contrib.muon` with its own ndim == 2 rule, which is what the 'muon'
entry did before the parameter groups landed.

Usage:
    PYTHONPATH=src python tools/optimizer_curve.py \
        --dataset data/shakespeare-byte --optimizer muon \
        --learning-rate 3e-3 --out /tmp/muon-3e-3.json
"""

import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import optax
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipes" / "lm"))

from dew.config import DataConfig, ModelConfig, OptimConfig, TrainerConfig  # noqa: E402
from dew.data.dataloaders import load_data  # noqa: E402
from dew.objectives.lm import LMObjective  # noqa: E402
from dew.training import ObjectiveTrainer  # noqa: E402
from dew.training.distributed import DevicePrefetchIterator  # noqa: E402
from dew.training.optim import build_optimizer  # noqa: E402

import train as lm_recipe  # noqa: E402

UNSPLIT = "muon-unsplit"


def build_solver(config: "Comparison", optim: OptimConfig):
    """The solver this arm runs, from the library or from optax directly."""
    if config.optimizer != UNSPLIT:
        return build_optimizer(optim, steps_per_epoch=config.steps)
    return optax.contrib.muon(config.learning_rate,
                              weight_decay=config.weight_decay,
                              adam_weight_decay=config.weight_decay)


@dataclass(frozen=True)
class Comparison:
    """One run of one solver."""

    dataset: str
    """A directory written by tools/tokenize_text.py."""
    out: str
    """Where the per-step losses are written, as JSON."""
    optimizer: str = "muon"
    """An OPTIMIZER_MAP name, or muon-unsplit."""
    learning_rate: float = 3e-3
    weight_decay: float = 0.1
    steps: int = 2000
    batch_size: int = 16
    sequence_length: int = 128
    emb_features: int = 256
    num_layers: int = 4
    num_heads: int = 4
    seed: int = 0


def run(config: Comparison) -> dict:
    meta = lm_recipe.read_meta(config.dataset)
    vocab_size = int(meta["vocab_size"])
    run_config = lm_recipe.LmRunConfig(
        model=ModelConfig("causal_transformer", {
            "emb_features": config.emb_features, "num_layers": config.num_layers,
            "num_heads": config.num_heads}),
        data=DataConfig(dataset=config.dataset, batch_size=config.batch_size,
                        # A corpus this size is read from page cache, so a
                        # worker pool costs more than it saves.
                        worker_count=0, read_thread_count=1, read_buffer_size=1,
                        worker_buffer_size=1),
        optim=OptimConfig(
            optimizer='adamw' if config.optimizer == UNSPLIT else config.optimizer,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay),
        trainer=TrainerConfig(distributed_training=False, multi_host=False,
                              compilation_cache_dir=None, wandb_project=None),
        sequence_length=config.sequence_length,
        tokenizer=meta["tokenizer"],
    )
    data = load_data(lm_recipe.token_data_config(run_config))
    model, model_config = lm_recipe.build_lm(
        run_config, vocab_size, config.sequence_length)
    objective = LMObjective(model, config.sequence_length, vocab_size=vocab_size,
                            ema_decay=1.0)

    trainer = ObjectiveTrainer(
        model=model,
        optimizer=build_solver(config, run_config.optim),
        rngs=jax.random.PRNGKey(config.seed),
        input_config=None,
        objective=objective,
        name=f"curve-{config.optimizer}-{config.learning_rate}",
        wandb_config=None,
        distributed_training=False,
        checkpoint_base_path="/tmp/dew-optimizer-curve",
    )
    parameters = sum(x.size for x in jax.tree.leaves(trainer.state.params))

    train_step = trainer._define_train_step(batch_size=config.batch_size)
    source = DevicePrefetchIterator(data["train"](), trainer.batch_sharding)
    state, rng = trainer.state, trainer.rngstate
    losses = []
    start = time.time()
    for step in range(config.steps):
        state, loss, _, rng, is_finite = train_step(state, rng, next(source))
        losses.append(float(loss))
        if not bool(is_finite):
            raise RuntimeError(f"loss went non-finite at step {step}")
    seconds = time.time() - start

    return {
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "steps": config.steps,
        "tokens_per_step": config.batch_size * config.sequence_length,
        "tokens": config.steps * config.batch_size * config.sequence_length,
        "corpus_tokens": meta.get("train_tokens"),
        "parameters": parameters,
        "model": model_config,
        "seed": config.seed,
        "seconds": seconds,
        "device": jax.devices()[0].device_kind,
        "losses": losses,
    }


def main(config: Comparison) -> None:
    result = run(config)
    Path(config.out).write_text(json.dumps(result))
    tail = result["losses"][-50:]
    print(f"{result['optimizer']} lr={result['learning_rate']} "
          f"tokens={result['tokens']} "
          f"final_loss={sum(tail) / len(tail):.4f} "
          f"seconds={result['seconds']:.0f}")


if __name__ == "__main__":
    main(tyro.cli(Comparison))
