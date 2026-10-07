"""A Python experiment whose model returns more than one output: a Gaussian
autoencoder's reconstruction and latent, trained by `Supervised` on the
reconstruction's squared error plus the latent's KL divergence from a unit
Gaussian. `dew train` imports it as `autoencoder_experiment`."""

import json
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.config import ModelConfig, ObjectiveConfig, OptimConfig, RunConfig, TrainerConfig
from dew.data import HFOptions, HubDataset
from dew.inputs import Field, InputSpec

FEATURES = 8


class Autoencoder(nn.Module):
    """Encodes a vector to a Gaussian latent and decodes the latent's mean."""

    latent: int = 2
    hidden: int = 16

    @nn.compact
    def __call__(self, x):
        hidden = nn.gelu(nn.Dense(self.hidden)(x.astype(jnp.float32)))
        mean, log_variance = nn.Dense(self.latent)(hidden), nn.Dense(self.latent)(hidden)
        reconstruction = nn.Dense(FEATURES)(nn.gelu(nn.Dense(self.hidden)(mean)))
        return reconstruction, (mean, log_variance)


def evidence_bound(outputs, batch):
    """Each example's squared reconstruction error plus its latent's KL divergence from a unit Gaussian."""
    reconstruction, (mean, log_variance) = outputs
    error = jnp.sum(jnp.square(reconstruction - batch["x"]), axis=-1)
    divergence = 0.5 * jnp.sum(jnp.square(mean) + jnp.exp(log_variance) - 1.0 - log_variance, axis=-1)
    return error + divergence


def reconstruction_error(outputs, batch):
    """Each example's mean squared reconstruction error."""
    return jnp.mean(jnp.square(outputs[0] - batch["x"]), axis=-1)


def points(path: Path, count: int, seed: int) -> str:
    """Write `count` points of a plane in `FEATURES` dimensions to `path`, once, and return it."""
    if not path.is_file():
        rng = np.random.default_rng(seed)
        plane = np.random.default_rng(0).normal(size=(2, FEATURES)) / np.sqrt(2)
        x = rng.normal(size=(count, 2)) @ plane + rng.normal(0, 0.05, (count, FEATURES))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps({"x": row.tolist()}) + "\n" for row in x))
    return str(path)


def run() -> RunConfig:
    data = Path(tempfile.gettempdir()) / "dew-tests" / "autoencoder"
    files = {"train": points(data / "train.jsonl", 1024, 1),
             "validation": points(data / "validation.jsonl", 64, 2)}
    return RunConfig(
        model=ModelConfig.from_model(Autoencoder()),
        data=HubDataset(name="json", split="train", val_split="validation",
                        options=HFOptions(data_files=files)),
        objective=ObjectiveConfig("supervised", {"loss": evidence_bound, "metrics": (reconstruction_error,),
                                                 "inputs": InputSpec(Field("x", (FEATURES,)))}),
        optim=OptimConfig(learning_rate=1e-2),
        trainer=TrainerConfig(steps=40, batch_size=32, log_every=1, eval_every=None, checkpoint_every=40))
