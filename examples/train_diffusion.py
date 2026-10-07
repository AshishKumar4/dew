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
from dew.config import OptimConfig
from dew.data import TFDSImages
from dew.diffusion import presets
from dew.inputs import CLIPText, Condition, Field, InputSpec
from dew.interop import save_hf_layout
from dew.nn.backbones import SimpleDiT
from dew.objectives.diffusion import DiffusionObjective
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


def main(config: Config, data=None, inputs=None):
    inputs = inputs or text_conditioned_inputs(config.image_size)
    data = data or TFDSImages(
        path=None if config.data_path is None else str(config.data_path.expanduser()),
        name="oxford_flowers102",
        image_size=config.image_size,
    ).load(batch=config.batch_size, tokenize=inputs.tokenize)
    steps = config.steps or data.epoch_steps(config.epochs)
    model = SimpleDiT(**config.model, output_channels=3, dtype=jnp.bfloat16)
    objective = DiffusionObjective(model, presets.EDM(regime="pixel"), inputs,
                                   solver=Heun(), guidance=CFG(3.0), steps=40)

    trainer = Trainer(objective, OptimConfig(learning_rate=config.learning_rate), key=jax.random.key(0),
                      mesh=MeshSpec(fsdp=config.fsdp),
                      checkpoints=Checkpoints(str(config.out / "checkpoints")))
    state = trainer.fit(data, steps=steps, log_every=50)

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
