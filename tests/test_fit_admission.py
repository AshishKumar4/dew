"""What a run admits, at one boundary (row 1872, audit TH-07): a configured model is built from its
effective record, and a trainer makes its key only after the process is ready."""

import json
import os
import subprocess
import sys
from pathlib import Path

from objective_cases import IMAGES

from dew.config import TrainerConfig
from dew.config.jepa import JepaRunConfig
from dew.data import TFDS, Loading, presets
from dew.registry import ModelConfig

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
