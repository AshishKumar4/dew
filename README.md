<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
    <img src="docs/assets/banner-light.svg" alt="dew" width="640">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/AshishKumar4/dew/actions/workflows/ci.yml"><img src="https://github.com/AshishKumar4/dew/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-2aa7a1" alt="MIT">
</p>

<p align="center">
  <a href="#overview">Overview</a> |
  <a href="#quick-install">Quick install</a> |
  <a href="#what-does-dew-look-like">What does Dew look like?</a> |
  <a href="#documentation">Documentation</a>
</p>

## Overview

Dew is a general framework for training machine learning models in JAX and Flax. It trains diffusion models, flow matching models, JEPA encoders, and your own architectures and objectives, sharded across devices and hosts. One set of primitives does all of this: a trainer, data pipelines, objectives, samplers and evaluation. Language models come next.

Dew includes:

* **Objectives** (`dew.objectives`): `DiffusionObjective` for pixel and latent diffusion, and `JepaObjective` for I-JEPA and V-JEPA. An objective defines the parameters, the loss and the validation step. The trainer does the rest.
* **Models** (`dew.nn`): UNet, UNet3D, UViT, DiT, MMDiT, HierarchicalMMDiT, SSM-DiT, VideoDiT and JEPA encoders. A Stable Diffusion VAE. One attention module that runs on the reference, XLA, cuDNN or TPU kernel with the same parameters.
* **Diffusion maths** (`dew.diffusion`): linear, cosine, exp, sqrt, Karras VE, EDM and flow matching schedules. Epsilon, x0, v, flow and Karras prediction transforms. Presets that pair a schedule with a transform.
* **Samplers** (`dew.sampling`): DDPM, DDIM, Euler, Euler ancestral, Heun, RK4 and multistep DPM, with interval-limited classifier-free guidance.
* **Trainer** (`dew.training`): data parallel and FSDP on one mesh, gradient accumulation, EMA, bf16 compute with fp32 parameters, async Orbax checkpoints that resume mid-epoch, Weights & Biases logging, profiling and MFU.
* **Data** (`dew.data`): Grain pipelines over TFDS, GCS ArrayRecord, local video, VoxCeleb2 and URL streams, with deterministic augmentation.
* **Evaluation and export** (`dew.eval`, `dew.interop`): FID, CLIP score, PSNR and SSIM. Safetensors export in the Hugging Face layout, for transformers, vLLM and verl.

Dew is the successor to [FlaxDiff](https://github.com/AshishKumar4/FlaxDiff). It is a personal research project. Expect sharp edges.

## Quick install

Dew needs Python 3.11 or later.

```bash
pip install dew-ml
```

The base install comes with a CPU-only JAX. For a GPU or a TPU, install the matching [JAX build](https://docs.jax.dev/en/latest/installation.html):

```bash
pip install "jax[cuda12]"   # NVIDIA GPUs
pip install "jax[tpu]"      # Cloud TPUs
```

Optional extras: `[tfds]` for TFDS datasets, `[av]` for video and audio, `[streaming]` for URL streaming, `[metrics]` for FID, `[interop]` for safetensors.

`dew` on PyPI is an unused placeholder, so the package is `dew-ml` for now.

## What does Dew look like?

Train a text-to-image diffusion model on Oxford Flowers:

```python
import jax, optax
from dew.data.dataloaders import get_dataset_grain
from dew.diffusion.transforms import get_diffusion_preset
from dew.inputs import DiffusionInputConfig, ConditionalInputConfig
from dew.inputs.encoders import CLIPTextEncoder
from dew.registry import build_model
from dew.sampling.euler import EulerAncestralSampler
from dew.training import ObjectiveTrainer

data = get_dataset_grain("oxford_flowers102", batch_size=16, image_scale=128)
text = CLIPTextEncoder.from_modelname("openai/clip-vit-large-patch14")
inputs = DiffusionInputConfig(
    sample_data_key="image", sample_data_shape=(128, 128, 3),
    conditions=[ConditionalInputConfig(encoder=text)],
)
train_schedule, sample_schedule, transform = get_diffusion_preset("edm")
model = build_model("simple_dit", dict(emb_features=512, num_layers=8, num_heads=8, patch_size=8))

trainer = ObjectiveTrainer(
    model=model, optimizer=optax.adamw(2e-4), input_config=inputs,
    noise_schedule=train_schedule, model_output_transform=transform,
    rngs=jax.random.PRNGKey(4), name="flowers-edm", checkpoint_base_path="./checkpoints",
)
state = trainer.fit(data, training_steps_per_epoch=data["train_len"] // 16, epochs=100,
                    sampler_class=EulerAncestralSampler, sampling_noise_schedule=sample_schedule)
```

Sample from it:

```python
sampler = EulerAncestralSampler(model, sample_schedule, transform, inputs, guidance_scale=3.0)
images = sampler.generate_samples(params=state.ema_params, num_samples=2, resolution=128,
                                  diffusion_steps=50, conditioning=["a water lily", "a rose"])
```

Run the same training from the command line. Every config field is a flag:

```bash
python recipes/diffusion/train.py --data.dataset oxford_flowers102 --model.architecture simple_dit \
    --model.config '{"patch_size": 4, "emb_features": 512, "num_layers": 12, "num_heads": 8}'
python recipes/jepa/train.py --data.dataset oxford_flowers102 --probe-classes 102
```

Add an objective. The trainer accepts any class with these methods:

```python
class LMObjective(Objective):
    tag = "lm"

    def __init__(self, model, seq_len):
        self.model, self.seq_len = model, seq_len
        self.ema = EMASpec(decay=lambda step: 0.999)

    def init_params(self, rng):
        return self.model.init(rng, jnp.zeros((1, self.seq_len), jnp.int32))["params"]

    def loss(self, params, ema_params, batch, rng, step):
        logits = self.model.apply({"params": params}, batch["text"][:, :-1])
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, batch["text"][:, 1:]).mean()
        return ce, {"ce": ce}

    def make_validation_step(self, **kwargs):
        return lambda val_state, batch: self.loss(val_state.params, None, batch, None, 0)[0]
```

## Architecture

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.svg">
    <img src="docs/assets/architecture-light.svg" alt="Dew modules by layer" width="100%">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pipeline-dark.svg">
    <img src="docs/assets/pipeline-light.svg" alt="A training step" width="100%">
  </picture>
</p>

## Documentation

* Concepts: [objectives](docs/concepts/objectives.md), [distributed training](docs/concepts/distributed.md), [the data pipeline](docs/concepts/data.md)
* [API reference](docs/api.md) and [recipes](docs/recipes.md)
* [Diffusion explained](https://nbviewer.org/github/AshishKumar4/dew/blob/main/tutorials/simple%20diffusion%20flax.ipynb), a notebook that builds diffusion from scratch without the library
* [Gallery](docs/gallery.md), [references](docs/references.md), [migrating from FlaxDiff](docs/from-flaxdiff.md)

## Roadmap

* Autoregressive and diffusion language model objectives
* Audio conditioned video models
* Multi-host validation on a TPU pod
* FID-50k

## Development

```bash
git clone --recurse-submodules git@github.com:AshishKumar4/dew.git
cd dew && pip install -e ".[test]"
JAX_PLATFORMS=cpu pytest -m "not network" -q
```

The tests simulate 8 devices on CPU, so the sharding tests run on any machine.

## Acknowledgements

**This project is partially supported by [Google TPU Research Cloud](https://sites.research.google/trc/about/). I would like to thank the Google Cloud TPU team for providing me with the resources to train the bigger text-conditional models in multi-host distributed settings.**

Dew builds on [JAX](https://github.com/jax-ml/jax), [Flax](https://github.com/google/flax), [Optax](https://github.com/google-deepmind/optax), [Orbax](https://github.com/google/orbax), [Grain](https://github.com/google/grain), [tyro](https://github.com/brentyi/tyro), [albumentations](https://github.com/albumentations-team/albumentations) and [Weights & Biases](https://github.com/wandb/wandb). The VAE and parts of the attention code come from [diffusers](https://github.com/huggingface/diffusers). The InceptionV3 comes from [jax-fid](https://github.com/matthias-wright/jax-fid). The Karras samplers follow [k-diffusion](https://github.com/crowsonkb/k-diffusion) and the [EDM](https://github.com/NVlabs/edm) reference code.

## License

[MIT](LICENSE)
