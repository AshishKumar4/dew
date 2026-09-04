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

from dew.config import OptimConfig  # noqa: E402
from dew.data import TokenWindows  # noqa: E402
from dew.objectives.lm import LMObjective  # noqa: E402
from dew import models  # noqa: E402  naming a registry fills it
from dew.registry import with_precision  # noqa: E402
from dew.training import MeshSpec, Trainer  # noqa: E402
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
    vocab_size = meta["vocab_size"]
    optim = OptimConfig(
        optimizer='adamw' if config.optimizer == UNSPLIT else config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay)
    # One worker and one read thread: the arms must see identical batches, and
    # a curve is not a throughput measurement.
    data = TokenWindows(path=config.dataset, seq_len=config.sequence_length, seed=config.seed,
                        worker_count=0, read_threads=1, read_buffer=1,
                        worker_buffer=1).load(batch=config.batch_size)
    fields = with_precision(
        "causal_transformer",
        dict(vocab_size=vocab_size, emb_features=config.emb_features,
             num_layers=config.num_layers, num_heads=config.num_heads,
             max_seq_len=config.sequence_length),
        dtype="bfloat16", attention_impl="auto")
    model = models.build("causal_transformer", **fields)
    objective = LMObjective(model, config.sequence_length, ema_decay=1.0)

    trainer = Trainer(objective, build_solver(config, optim),
                      key=jax.random.key(config.seed), mesh=MeshSpec(fsdp=1),
                      checkpoints=None, tracker=None)
    abstract = jax.eval_shape(trainer.initial_state)
    state = jax.jit(trainer.initial_state, out_shardings=trainer.shardings(abstract))()
    parameters = sum(x.size for x in jax.tree.leaves(state.params))

    source = DevicePrefetchIterator(data.train(), trainer.batch_sharding)
    train_step = trainer.compile(state, next(source))
    scale = None
    losses = []
    start = time.time()
    for step in range(config.steps):
        state, scale, loss, _, is_finite = train_step(state, scale, next(source))
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
        "model": fields,
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
