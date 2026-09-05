#!/usr/bin/env python3
"""Record one optimizer's loss curve at a fixed token budget on a token dataset.

Every arm of a comparison sees the same model, the same seed and the same
batches in the same order, so the curves differ only by the solver. The step
is the trainer's own compiled step over the model and the data a language
model recipe trains, so nothing here is a second wiring of a run. What is
local is the loop, which records every step's loss instead of logging one per
interval.

`--optimizer` takes one of the library's solver names, or `muon-unsplit`,
which runs `optax.contrib.muon` with its own rule (every rank-2 parameter
through Muon) as the comparison arm for the parameter groups the `muon`
entry declares.

Usage:
    PYTHONPATH=src python tools/optimizer_curve.py \\
        --dataset data/shakespeare-byte --optimizer muon \\
        --learning-rate 3e-3 --out /tmp/muon-3e-3.json
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import jax
import optax
import tyro

from dew.config import OptimConfig
from dew.data import Loading, TokenWindows
from dew.objectives.lm import LMObjective
from dew import models  # naming a registry fills it
from dew.registry import with_precision
from dew.training import MeshSpec, Trainer
from dew.training.distributed import DevicePrefetchIterator
from dew.training.optim import build_optimizer

Solver = Literal["adam", "adamw", "lamb", "muon", "muon-unsplit"]
"""The OPTIMIZER_MAP names, plus the unsplit Muon arm."""


@dataclass(frozen=True)
class Comparison:
    """One run of one solver."""

    dataset: str
    """A directory written by tools/tokenize_text.py."""
    out: str
    """Where the per-step losses are written, as JSON."""
    optimizer: Solver = "muon"
    learning_rate: float = 3e-3
    weight_decay: float = 0.1
    steps: int = 2000
    batch_size: int = 16
    sequence_length: int = 128
    emb_features: int = 256
    num_layers: int = 4
    num_heads: int = 4
    seed: int = 0


def build_solver(config: Comparison) -> optax.GradientTransformation:
    """The solver this arm runs, from the library or from optax directly."""
    if config.optimizer == "muon-unsplit":
        return optax.contrib.muon(config.learning_rate,
                                  weight_decay=config.weight_decay,
                                  adam_weight_decay=config.weight_decay)
    optim = OptimConfig(optimizer=config.optimizer, learning_rate=config.learning_rate,
                        weight_decay=config.weight_decay)
    return build_optimizer(optim, steps=config.steps)


@dataclass(frozen=True)
class Curve:
    """One arm's record, written to --out as JSON."""

    optimizer: str
    learning_rate: float
    weight_decay: float
    steps: int
    tokens_per_step: int
    tokens: int
    corpus_tokens: int
    parameters: int
    model: dict[str, object]
    seed: int
    seconds: float
    device: str
    losses: list[float]


def run(config: Comparison) -> Curve:
    # The tokenizer run wrote the vocabulary beside the token files, which is
    # where the recipe reads it from too.
    meta = json.loads((Path(config.dataset) / "meta.json").read_text())
    vocab_size = int(meta["vocab_size"])
    # One worker and one read thread: the arms must see identical batches, and
    # a curve is not a throughput measurement.
    data = TokenWindows(path=config.dataset, seq_len=config.sequence_length, seed=config.seed,
                        loading=Loading(workers=0, threads=1, read_buffer=1,
                                        worker_buffer=1)).load(batch=config.batch_size)
    fields = with_precision(
        "causal_transformer",
        dict(vocab_size=vocab_size, emb_features=config.emb_features,
             num_layers=config.num_layers, num_heads=config.num_heads,
             max_seq_len=config.sequence_length),
        dtype="bfloat16", attention_impl="auto")
    model = models.build("causal_transformer", **fields)
    objective = LMObjective(model, config.sequence_length, ema_decay=1.0)

    trainer = Trainer(objective, build_solver(config),
                      key=jax.random.key(config.seed), mesh=MeshSpec(fsdp=1),
                      checkpoints=None, tracker=None)
    abstract = jax.eval_shape(trainer.initial_state)
    state = jax.jit(trainer.initial_state, out_shardings=trainer.shardings(abstract))()
    parameters = sum(x.size for x in jax.tree.leaves(state.params))

    source = DevicePrefetchIterator(data.train(), trainer.device_mesh)
    train_step = trainer.compile(state, next(source))
    scale = None
    losses: list[float] = []
    start = time.perf_counter()
    for step in range(config.steps):
        state, scale, loss, _, is_finite = train_step(state, scale, next(source))
        losses.append(float(loss))
        if not bool(is_finite):
            raise RuntimeError(f"loss went non-finite at step {step}")
    seconds = time.perf_counter() - start

    return Curve(
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        steps=config.steps,
        tokens_per_step=config.batch_size * config.sequence_length,
        tokens=config.steps * config.batch_size * config.sequence_length,
        corpus_tokens=int(meta["train_tokens"]),
        parameters=parameters,
        model=fields,
        seed=config.seed,
        seconds=seconds,
        device=jax.devices()[0].device_kind,
        losses=losses,
    )


def main(config: Comparison) -> None:
    curve = run(config)
    Path(config.out).write_text(json.dumps(asdict(curve)))
    tail = curve.losses[-50:]
    print(f"{curve.optimizer} lr={curve.learning_rate} tokens={curve.tokens} "
          f"final_loss={sum(tail) / len(tail):.4f} seconds={curve.seconds:.0f}")


if __name__ == "__main__":
    main(tyro.cli(Comparison))
