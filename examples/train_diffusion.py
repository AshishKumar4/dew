"""Train a text-to-image diffusion model on Oxford Flowers, sample from it, export the weights.

    python examples/train_diffusion.py --data-path /data/oxford_flowers102/2.1.1 --epochs 200
    python examples/train_diffusion.py --data-path /data/oxford_flowers102/2.1.1 --steps 20 --image-size 32
"""
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from PIL import Image

from dew.artifacts import uint8_pixels
from dew.data import TFDS, Dataset, images, presets
from dew.diffusion.presets import EDM
from dew.inputs import CLIPText, Condition, Field, InputSpec
from dew.interop import save_hf_layout
from dew.nn.backbones import SimpleDiT
from dew.objectives.diffusion import DiffusionObjective
from dew.optim import OptimConfig
from dew.sampling import CFG, Heun
from dew.training import Checkpoints, MeshSpec, Trainer


@dataclass
class Config:
    data_path: Path | None = None
    image_size: int = 128
    batch_size: int = 32
    epochs: int = 200
    steps: int | None = None
    """Run length in steps; unset trains for `epochs` passes over the data."""
    learning_rate: float = 2e-4
    fsdp: int = 1
    model: dict = field(default_factory=lambda: {
        "patch_size": 4, "emb_features": 512, "num_layers": 12, "num_heads": 8})
    prompts: tuple[str, ...] = ("a water lily", "a sunflower", "a red rose", "a purple orchid")
    out: Path = Path("runs/flowers")


def text_conditioned_inputs(image_size: int) -> InputSpec:
    """Images conditioned on CLIP text under the model's `textcontext` keyword."""
    return InputSpec(
        sample=Field("image", (image_size, image_size, 3)),
        conditions={"textcontext": Condition(CLIPText.from_pretrained("openai/clip-vit-large-patch14"))})


def flowers(config: Config) -> tuple[Dataset, Dataset]:
    """Every Flowers image, captioned with its class, less four batches held out for validation."""
    source = TFDS("oxford_flowers102", split="all",
                  path=None if config.data_path is None else str(config.data_path.expanduser()))
    train, validation = source.holdout(records=4 * config.batch_size)
    # rejects=False: every record decodes or the run stops, so an epoch's length is known up front.
    sized = {"batch": config.batch_size, "caption": images.caption("a photo of a {label}"), "rejects": False}
    return (train.shuffle(seed=0).repeat().pipe(presets.images, config.image_size, augment=True, **sized),
            validation.pipe(presets.images, config.image_size, **sized))


def main(config: Config, data=None, inputs=None):
    inputs = inputs or text_conditioned_inputs(config.image_size)
    train, validation = (data, None) if data is not None else flowers(config)
    model = SimpleDiT(**config.model, output_channels=3, dtype=jnp.bfloat16)
    # The objective encodes each batch's captions with its text condition as the trainer reads them.
    objective = DiffusionObjective(model, EDM(regime="pixel"), inputs,
                                   solver=Heun(), guidance=CFG(3.0), steps=40)

    trainer = Trainer(objective, OptimConfig(learning_rate=config.learning_rate), key=jax.random.key(0),
                      mesh=MeshSpec(fsdp=config.fsdp),
                      checkpoints=Checkpoints(str(config.out / "checkpoints")))
    state = trainer.fit(train, validation=validation, steps=config.steps,
                        epochs=None if config.steps else config.epochs, log_every=50)

    # The averaged weights stay on
    # the trainer's mesh, prompts split over it, and host() reads the rows back.
    pipe = objective.pipeline(state)
    images = pipe(list(config.prompts), steps=50, guidance=3.0, solver=Heun(), key=1).host().images
    pixels = uint8_pixels(images)
    grid = np.concatenate(list(pixels), axis=1)
    config.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(config.out / "samples.png")

    save_hf_layout(state.averaged["params"],
                   {"architecture": "simple_dit", **config.model, "output_channels": 3, "dtype": "bfloat16"},
                   config.out / "export")
    return state


if __name__ == "__main__":
    main(tyro.cli(Config))
