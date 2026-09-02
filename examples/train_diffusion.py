"""Train a text-to-image diffusion model on Oxford Flowers, sample from it, export the weights.

    python examples/train_diffusion.py --epochs 200 --image-size 128
    python examples/train_diffusion.py --epochs 1 --steps-per-epoch 20 --image-size 32   # smoke run
"""
from dataclasses import dataclass, field
from pathlib import Path

import jax
import numpy as np
import optax
import tyro
from PIL import Image

from dew.data.dataloaders import get_dataset_grain
from dew.diffusion.transforms import get_diffusion_preset
from dew.image_ops import denormalize_images
from dew.inputs import ConditionalInputConfig, DiffusionInputConfig
from dew.inputs.encoders import CLIPTextEncoder
from dew.interop import save_hf_layout
from dew.registry import apply_precision_policy, build_model
from dew.sampling import EulerAncestralSampler
from dew.training import ObjectiveTrainer


@dataclass
class Config:
    dataset: str = "oxford_flowers102"
    image_size: int = 128
    batch_size: int = 32
    epochs: int = 200
    steps_per_epoch: int | None = None
    learning_rate: float = 2e-4
    fsdp_size: int = 1
    model: dict = field(default_factory=lambda: dict(patch_size=4, emb_features=512, num_layers=12, num_heads=8))
    prompts: tuple[str, ...] = ("a water lily", "a sunflower", "a red rose", "a purple orchid")
    out: Path = Path("runs/flowers")


def text_conditioned_inputs(image_size: int) -> DiffusionInputConfig:
    """The sample and its conditions: an image, conditioned on a CLIP text embedding."""
    text_encoder = CLIPTextEncoder.from_modelname("openai/clip-vit-large-patch14")
    return DiffusionInputConfig(
        sample_data_key="image",
        sample_data_shape=(image_size, image_size, 3),
        conditions=[ConditionalInputConfig(encoder=text_encoder)],
    )


def main(config: Config, data=None, inputs=None):
    data = data or get_dataset_grain(config.dataset, batch_size=config.batch_size, image_scale=config.image_size)
    inputs = inputs or text_conditioned_inputs(config.image_size)

    # A preset is a training schedule, a sampling schedule and a prediction transform that belong together.
    train_schedule, sample_schedule, transform = get_diffusion_preset("edm")
    model_config = apply_precision_policy("simple_dit", config.model, dtype="bfloat16", attention_impl="auto")
    model = build_model("simple_dit", model_config)

    trainer = ObjectiveTrainer(
        model, optax.adamw(config.learning_rate), input_config=inputs,
        noise_schedule=train_schedule, model_output_transform=transform,
        rngs=jax.random.PRNGKey(0), name=config.out.name, checkpoint_base_path=str(config.out / "checkpoints"),
        fsdp_size=config.fsdp_size,
    )
    steps = config.steps_per_epoch or data["train_len"] // config.batch_size
    state = trainer.fit(data, training_steps_per_epoch=steps, epochs=config.epochs,
                        sampler_class=EulerAncestralSampler, sampling_noise_schedule=sample_schedule)

    sampler = EulerAncestralSampler(model, sample_schedule, transform, inputs, guidance_scale=3.0)
    images = sampler.generate_samples(params=state.ema_params, num_samples=len(config.prompts),
                                      resolution=config.image_size, diffusion_steps=50,
                                      conditioning=list(config.prompts))
    grid = np.concatenate(np.asarray(denormalize_images(images)), axis=1)
    Image.fromarray(grid).save(config.out / "samples.png")

    save_hf_layout(state.ema_params, config={"architecture": "simple_dit", **model_config}, directory=config.out / "export")
    return state


if __name__ == "__main__":
    main(tyro.cli(Config))
