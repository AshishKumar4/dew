# dew

Dew is a library of tools (schedulers, samplers, models, data loaders, trainers) for training generative models in JAX and Flax. The focus is on understandability and readability over performance. It covers image and video diffusion, flow matching, latent diffusion with a VAE, I-JEPA/V-JEPA self-supervised training and autoregressive language models, all through one trainer with data-parallel and FSDP sharding.

Dew is the successor to FlaxDiff, restructured once the trainer stopped being about diffusion alone: what to optimize is now an objective you plug in.

## Install

```bash
pip install "dew-ml @ git+https://github.com/AshishKumar4/dew"
```

There is no release on PyPI yet. The package will ship as `dew-ml` and imports as `dew`. Extras pull in the heavier dependencies only when you need them: `av` for video and audio readers, `metrics` for FID, `streaming` for the URL-streaming loader, `tfds` for TFDS datasets, `interop` for safetensors.

## A training run

```python
import jax, optax

from dew.data.dataloaders import get_dataset_grain
from dew.inputs import DiffusionInputConfig, ConditionalInputConfig
from dew.inputs.encoders import CLIPTextEncoder
from dew.diffusion.transforms import get_diffusion_preset
from dew.registry import build_model
from dew.training import ObjectiveTrainer

BATCH_SIZE, IMAGE_SIZE = 16, 128

data = get_dataset_grain("oxford_flowers102", batch_size=BATCH_SIZE, image_scale=IMAGE_SIZE)
train_schedule, sample_schedule, transform = get_diffusion_preset("edm")

trainer = ObjectiveTrainer(
    model=build_model("simple_dit", dict(emb_features=512, num_layers=8, num_heads=8, patch_size=8)),
    optimizer=optax.adamw(2e-4),
    input_config=DiffusionInputConfig(
        sample_data_key="image",
        sample_data_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        conditions=[ConditionalInputConfig(
            encoder=CLIPTextEncoder.from_modelname("openai/clip-vit-large-patch14"))],
    ),
    noise_schedule=train_schedule,
    model_output_transform=transform,
    rngs=jax.random.PRNGKey(4),
    name="flowers-edm",
)

trainer.fit(data, training_steps_per_epoch=data["train_len"] // BATCH_SIZE, epochs=100)
```

For a run described by flags instead of by code, see [Recipes](recipes.md).

## Where things are

- [The objectives seam](concepts/objectives.md): what the trainer owns, what the objective owns, and how to add one.
- [Distributed training](concepts/distributed.md): the mesh, the sharding rules, prefetch, checkpoints.
- [The data pipeline](concepts/data.md): sources, augmenters, grain loaders, resume.
- [API](api.md): the public modules.

## Tests

```bash
pip install -e .[test]
JAX_PLATFORMS=cpu pytest -m "not network" -q
```

`tests/conftest.py` asks XLA for 8 host devices, so the sharding tests run a real 4x2 data/fsdp mesh without an accelerator. The kernel lane runs the model and training files again on a GPU, with no `JAX_PLATFORMS` override. Tests marked `network` download pretrained weights and are excluded by default.
