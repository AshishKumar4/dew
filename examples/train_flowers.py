"""Train a pixel-space diffusion model on prepared Oxford Flowers ArrayRecords.

CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda python examples/train_flowers.py \
    --data ~/.cache/dew/datasets/oxford_flowers102/2.1.1 --steps 1000
"""
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from PIL import Image

from dew import Checkpoints, Field, InputSpec, Trainer, sample
from dew.data import Loading, OxfordFlowers
from dew.diffusion.presets import EDM
from dew.objectives.diffusion import DiffusionObjective
from dew.nn.backbones import SimpleDiT
from dew.sampling import Heun


@dataclass
class Config:
    data: Path
    output: Path = Path("runs/flowers64")
    steps: int = 1000
    batch: int = 16


def main(config: Config):
    data = OxfordFlowers(
        path=str(config.data.expanduser()),
        split="train",
        image_size=64,
        val_batches=0,
        loading=Loading(workers=2, threads=2, read_buffer=16, worker_buffer=2),
    ).load(batch=config.batch)
    model = SimpleDiT(
        patch_size=4,
        emb_features=128,
        num_layers=4,
        num_heads=4,
        dtype=jnp.bfloat16,
        attention_impl="auto",
    )
    process = EDM()()
    objective = DiffusionObjective(
        model,
        process,
        InputSpec(Field("image", (64, 64, 3))),
    )
    trainer = Trainer(
        objective,
        optax.adamw(2e-4),
        key=jax.random.key(0),
        checkpoints=Checkpoints(str(config.output / "checkpoints")),
    )
    state = trainer.fit(
        data,
        steps=config.steps,
        log_every=20,
        checkpoint_every=200,
    )

    denoise = process.denoiser(model, state.averaged, conditions={})
    noise = process.noise(jax.random.key(1), (8, 64, 64, 3))
    images = sample(
        denoise,
        noise,
        steps=40,
        solver=Heun(),
        key=jax.random.key(2),
    )
    pixels = np.clip(np.rint((np.asarray(images) + 1) * 127.5), 0, 255).astype(np.uint8)
    grid = np.concatenate(tuple(pixels), axis=1)
    config.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(config.output / "samples.png")
    print(f"Saved {config.output / 'samples.png'} after {int(state.updates)} updates")
    return state


if __name__ == "__main__":
    main(tyro.cli(Config))
