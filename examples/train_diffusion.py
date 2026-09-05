"""Train a text-to-image diffusion model on Oxford Flowers, sample from it, export the weights.

    python examples/train_diffusion.py --epochs 200 --image-size 128
    python examples/train_diffusion.py --steps 20 --image-size 32   # smoke run
"""
from dataclasses import dataclass, field
from pathlib import Path

import jax
import numpy as np
import optax
import tyro
from PIL import Image

from dew.data import OxfordFlowers
from dew.diffusion import presets
from dew.inputs import CLIPText, Condition, Field, InputSpec
from dew.interop import save_hf_layout
from dew.objectives.diffusion import DiffusionObjective
import dew.nn.backbones  # registers the models
from dew.registry import models
from dew.sampling import CFG, Heun, TextToImage
from dew.training import Checkpoints, MeshSpec, Trainer


@dataclass
class Config:
    image_size: int = 128
    batch_size: int = 32
    epochs: int = 200
    steps: int | None = None
    """Run length in steps; unset trains for `epochs` passes over the data."""
    learning_rate: float = 2e-4
    fsdp: int = 1
    model: dict = field(default_factory=lambda: dict(patch_size=4, emb_features=512, num_layers=12, num_heads=8))
    prompts: tuple[str, ...] = ("a water lily", "a sunflower", "a red rose", "a purple orchid")
    out: Path = Path("runs/flowers")


def text_conditioned_inputs(image_size: int) -> InputSpec:
    """Images conditioned on CLIP text under the model's `textcontext` keyword."""
    return InputSpec(
        sample=Field("image", (image_size, image_size, 3)),
        conditions={"textcontext": Condition(CLIPText.from_pretrained("openai/clip-vit-large-patch14"))})


def main(config: Config, data=None, inputs=None):
    inputs = inputs or text_conditioned_inputs(config.image_size)
    data = data or OxfordFlowers(image_size=config.image_size).load(
        batch=config.batch_size, tokenize=inputs.tokenize)
    steps = config.steps or data.epoch_steps(config.epochs)
    fields = dict(config.model, output_channels=3, dtype="bfloat16")
    model = models.build("simple_dit", **fields)
    process = presets.EDM()()
    objective = DiffusionObjective(model, process, inputs, sampler=Heun(), guidance=CFG(3.0), steps=40)

    trainer = Trainer(objective, optax.adamw(config.learning_rate), key=jax.random.key(0),
                      mesh=MeshSpec(fsdp=config.fsdp),
                      checkpoints=Checkpoints(str(config.out / "checkpoints")))
    state = trainer.fit(data, steps=steps, log_every=50)

    pipe = TextToImage(model=model, process=process, inputs=inputs, params=state.averaged)
    images = pipe(list(config.prompts), steps=50, guidance=3.0, sampler=Heun(), key=jax.random.key(1))
    pixels = np.clip(np.round((np.asarray(images) + 1.0) * 127.5), 0, 255).astype(np.uint8)
    grid = np.concatenate(list(pixels), axis=1)
    config.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(config.out / "samples.png")

    save_hf_layout(state.averaged["params"], {"architecture": "simple_dit", **fields}, config.out / "export")
    return state


if __name__ == "__main__":
    main(tyro.cli(Config))
