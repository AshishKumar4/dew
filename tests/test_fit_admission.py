"""What a run admits, at one boundary (row 1872, audit TH-07): a configured model is built from its
effective record, and a trainer makes its key only after the process is ready."""

import json
import os
import subprocess
import sys
from pathlib import Path

import flax.linen as nn
import numpy as np
import optax
import pytest
from affine_run import Replaying
from objective_cases import IMAGES

from dew.config import TrainerConfig
from dew.config.jepa import JepaRunConfig
from dew.data import TFDS, Loading, presets
from dew.data.dataset import Loader
from dew.inputs import Field, InputSpec
from dew.objectives.supervised import Supervised
from dew.registry import ModelConfig
from dew.training import Trainer

ROOT = Path(__file__).resolve().parents[1]


class Built(JepaRunConfig):
    """A JEPA run whose training hands back the objective it built, untrained."""

    def train(self, objective, dataset, **_):
        return objective


def test_a_jepa_predictor_takes_the_width_its_encoder_was_built_at(tmp_path):
    """An encoder whose width its record pins as a default (a run recorded at another class default)
    is built at that width, and the predictor it feeds takes the same width: before, the predictor
    read the stated fields alone and took its class's 384 (AT-06)."""
    rows = TFDS("dew_images", path=str(IMAGES), split="all").pipe(presets.images, 16, batch=8)
    encoder = ModelConfig("jepa_encoder", {"patch_size": 2, "num_layers": 1, "num_heads": 2, "mlp_ratio": 2,
                                           "dtype": "float32", "attention_impl": "xla"},
                          defaults={"emb_features": 16})
    run = Built(model=encoder, predictor={"num_layers": 1, "predictor_features": 8}, num_target_blocks=2,
                data=rows.read("memory").loading(Loading(workers=0)),
                trainer=TrainerConfig(checkpoint_dir=str(tmp_path), steps=1))
    objective = run.prepare().train("built")
    assert (objective.encoder.emb_features, objective.predictor.emb_features) == (16, 16)


KEYED = """
import json
import optax
from dew.devices import backend_open
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Trainer

model = models.build("causal_transformer", vocab_size=16, emb_features=8, num_layers=1, num_heads=2,
                     mlp_features=16, max_seq_len=8)
trainer = Trainer(LMObjective(model, 8), optax.sgd(0.1), key=3)
built = backend_open()
key = trainer.key
print(json.dumps({"built": built, "read": backend_open()}))
"""
"""A fresh process that builds a trainer from a seed, then reads its key."""


def test_a_trainer_opens_no_backend_until_its_key_is_read():
    """XLA reads its flags and compilation cache as JAX opens its backend, which `fit` readies first:
    a trainer built from a seed leaves the backend closed, and its key is made when first read
    (AT-09). Before, the constructor made the key and opened the backend."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT / "src"),
                                                                     os.environ.get("PYTHONPATH"))))}
    done = subprocess.run([sys.executable, "-c", KEYED], capture_output=True, text=True, env=env, cwd=ROOT,
                          check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {"built": False, "read": True}


class Pooled(nn.Module):
    """A map that reads its input's mean, so it takes any width its batch has."""

    @nn.compact
    def __call__(self, x):
        return nn.Dense(2)(x.mean(axis=-1, keepdims=True))


def test_a_batch_of_a_shape_the_run_has_not_stepped_is_checked_against_its_declaration():
    """An objective declares its input three wide; a stream whose second batch is five wide, as a
    later stage of a ramp or a schedule may make it, is refused at that batch, before it compiles a
    step for it. Before, only the first batch was checked, and a model that reads any width trained
    on the undeclared one (AT-03)."""
    def batch_at(step: int) -> dict:
        x = np.random.default_rng(step).normal(size=(8, 3 if step == 0 else 5)).astype(np.float32)
        return {"x": x, "y": x[:, :2]}

    objective = Supervised(Pooled(), lambda outputs, batch: (outputs - batch["y"]) ** 2,
                           inputs=InputSpec(Field("x", (3,))))
    trainer = Trainer(objective, optax.sgd(0.1), key=0)
    data = Loader(train=lambda partition: Replaying(batch_at), val=None, batch=8, records=None)
    with pytest.raises(ValueError):
        trainer.fit(data, steps=2, log_every=1)


@pytest.mark.parametrize("eval_every", [None, 5], ids=["at-the-end", "midway"])
def test_a_validation_split_of_undeclared_inputs_is_refused_before_the_run_trains(eval_every):
    """A model that reads any width trains on the three-wide inputs its objective declares, and its
    validation split is five wide: the split is read before the run updates, whether it is scored
    every five steps or only at the end, and the run is refused there, its training stream read no
    further than its first batches. Before, only training batches were checked, and an end-only split
    was first read after training (AT-03)."""
    def batch(width: int, seed: int) -> dict:
        x = np.random.default_rng(seed).normal(size=(8, width)).astype(np.float32)
        return {"x": x, "y": x[:, :2]}

    stream = Replaying(lambda step: batch(3, step))
    objective = Supervised(Pooled(), lambda outputs, batch: (outputs - batch["y"]) ** 2,
                           inputs=InputSpec(Field("x", (3,))))
    data = Loader(train=lambda partition: stream, val=lambda partition: [batch(5, 0), batch(5, 1)], batch=8,
                  records=None)
    with pytest.raises(ValueError):
        Trainer(objective, optax.sgd(0.1), key=0).fit(data, steps=40, log_every=1, eval_every=eval_every)
    assert stream.position < 10
