"""Train a pixel-space diffusion model on prepared Oxford Flowers ArrayRecords.

CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda python examples/train_flowers.py \\
    --data ~/.cache/dew/datasets/oxford_flowers102/2.1.1 --steps 1000

It trains on every record: no split is held out, and nothing is scored. With `--profile N` it is a
profiling run, which traces N steps into `<output>/profile` after two warm-up steps and draws no
samples:

CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda python examples/train_flowers.py \\
    --data ~/.cache/dew/datasets/oxford_flowers102/2.1.1 --steps 50 --profile 20
"""
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from PIL import Image

from dew import Checkpoints, Field, InputSpec, Trainer, sample
from dew.artifacts import uint8_pixels
from dew.data import TFDS, Dataset, presets
from dew.data.dataset import Loader
from dew.diffusion.presets import EDM
from dew.nn.backbones import SimpleDiT
from dew.objectives.diffusion import DiffusionObjective
from dew.optim import OptimConfig
from dew.sampling import Heun
from dew.training import ProfileWindow


@dataclass
class Config:
    data: Path
    output: Path = Path("runs/flowers64")
    steps: int = 1000
    batch: int = 16
    image_size: int = 64
    profile: int = 0  # steps to trace; above 0 the run only profiles and draws no samples


def flowers(config: Config) -> Dataset:
    """Every training record of the prepared split, in a new order each epoch; none is held out."""
    return (TFDS("oxford_flowers102", split="train", path=str(config.data.expanduser()))
            .shuffle(seed=0).repeat()
            .pipe(presets.images, config.image_size, batch=config.batch, augment=True)
            .workers(2))


def main(config: Config, data: Dataset | Loader | None = None):
    if data is None:
        data = flowers(config)
    model = SimpleDiT(
        patch_size=4,
        emb_features=128,
        num_layers=4,
        num_heads=4,
        dtype=jnp.bfloat16,
        attention_impl="auto",
    )
    objective = DiffusionObjective(
        model,
        EDM(regime="pixel"),
        InputSpec(Field("image", (config.image_size, config.image_size, 3))),
    )
    trainer = Trainer(
        objective,
        OptimConfig(learning_rate=2e-4),
        key=jax.random.key(0),
        checkpoints=Checkpoints(str(config.output / "checkpoints")),
        profile=(ProfileWindow(str(config.output / "profile"), steps=config.profile)
                 if config.profile else None),
    )
    state = trainer.fit(
        data,
        steps=config.steps,
        log_every=20,
        checkpoint_every=200,
    )
    if config.profile:
        print(f"Wrote a {config.profile}-step trace to {config.output / 'profile'}")
        return state

    process = objective.process
    denoise = process.denoiser(model, state.averaged, conditions={})
    noise = process.noise(jax.random.key(1), (8, config.image_size, config.image_size, 3))
    images = sample(
        denoise,
        noise,
        steps=40,
        solver=Heun(),
        key=jax.random.key(2),
    )
    pixels = uint8_pixels(images)
    grid = np.concatenate(tuple(pixels), axis=1)
    config.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(config.output / "samples.png")
    print(f"Saved {config.output / 'samples.png'} after {int(state.updates)} updates")
    return state


if __name__ == "__main__":
    main(tyro.cli(Config))
