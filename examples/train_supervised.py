"""A classifier written as a Python experiment and trained with `dew train`.

    dew train examples/train_supervised.py --set trainer.steps=600 --set model.hidden=64

This file builds the run in plain Python: its own Flax model and loss, two
interleaved spirals it writes once as JSON lines, and `Supervised`, which
trains any model on a loss over its outputs, with no subclass of
`Objective`. `run` is the `RunConfig`; each `--set` changes one field of it,
read as that field's own type reads it. `dew train` imports the file as the
module `train_supervised`, so `run.json` names the model and the loss by
import path (`train_supervised:MLP`), and a new process rebuilds the same
run from that record alone:

    PYTHONPATH=examples dew train checkpoints/train_supervised/run.json \\
        --trust train_supervised --set trainer.checkpoint_dir=again

Without `--set`, the record continues the run from its own checkpoints.
"""

import json
from pathlib import Path

import numpy as np
import optax
from flax import linen as nn

from dew.cache import dew_cache_dir
from dew.config import ModelConfig, ObjectiveConfig, OptimConfig, RunConfig, TrainerConfig
from dew.data import HFOptions, HubDataset
from dew.inputs import Field, InputSpec
from dew.objectives.supervised import Accuracy


class MLP(nn.Module):
    """Two hidden layers, then a logit per class."""

    hidden: int = 32
    classes: int = 2

    @nn.compact
    def __call__(self, x):
        for _ in range(2):
            x = nn.gelu(nn.Dense(self.hidden)(x))
        return nn.Dense(self.classes)(x)


def cross_entropy(outputs, batch):
    """Each example's softmax cross entropy against its label."""
    return optax.softmax_cross_entropy_with_integer_labels(outputs, batch["label"])


def spirals(path: Path, count: int, seed: int) -> str:
    """Write `count` points of two interleaved spirals to `path`, once, and return it."""
    if not path.is_file():
        rng = np.random.default_rng(seed)
        label = rng.integers(0, 2, count)
        turn = rng.uniform(0.25, 1.0, count) * 3 * np.pi
        points = np.stack([turn * np.cos(turn + np.pi * label), turn * np.sin(turn + np.pi * label)], -1)
        points = points / (3 * np.pi) + rng.normal(0, 0.02, points.shape)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps({"x": point.tolist(), "label": int(cls)}) + "\n"
                                for point, cls in zip(points, label, strict=True)))
    return str(path)


def run() -> RunConfig:
    data = Path(dew_cache_dir()) / "examples" / "spirals"
    files = {"train": spirals(data / "train.jsonl", 2048, 0),
             "validation": spirals(data / "validation.jsonl", 256, 1)}
    return RunConfig(
        model=ModelConfig.from_model(MLP()),
        data=HubDataset(name="json", split="train", val_split="validation",
                        options=HFOptions(data_files=files)),
        objective=ObjectiveConfig("supervised", {
            "loss": cross_entropy, "metrics": (Accuracy(),), "inputs": InputSpec(Field("x", (2,)))}),
        optim=OptimConfig(learning_rate=3e-3),
        trainer=TrainerConfig(steps=300, batch_size=64, log_every=50, eval_every=100,
                              checkpoint_every=100))
