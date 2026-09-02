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

Dew is a framework for training models from scratch in JAX and Flax. The trainer, data pipeline, sharding and checkpointing are shared; the training objective is the plug-in. Diffusion and JEPA ship today, language models are on the roadmap.

Dew includes:

* **Objectives** (`dew.objectives`): `DiffusionObjective` for pixel and latent diffusion with min-SNR weighting, and `JepaObjective` for I-JEPA and V-JEPA with collapse telemetry and probes. An objective defines `init_params`, `loss` and a validation step.
* **Models** (`dew.nn`): UNet, UNet3D, UViT, DiT, MMDiT, HierarchicalMMDiT, SSM-DiT, VideoDiT and JEPA encoders, a vendored Stable Diffusion VAE, Hilbert and zigzag patch orders. One attention path over the reference, XLA, cuDNN and TPU kernels with an identical parameter tree.
* **Diffusion maths** (`dew.diffusion`): linear, cosine, exp, sqrt, Karras VE, EDM and flow matching schedules; epsilon, x0, v, flow and Karras prediction transforms; presets that pair them.
* **Samplers** (`dew.sampling`): DDPM, DDIM, Euler, Euler ancestral, Heun, RK4 and multistep DPM, with interval-limited classifier-free guidance.
* **Trainer** (`dew.training`): data parallel and FSDP on a `(data, fsdp)` mesh, gradient accumulation, EMA, bf16 compute over fp32 parameters, async Orbax checkpoints with mid-epoch resume, Weights & Biases logging, profiling and MFU.
* **Data** (`dew.data`): Grain pipelines over TFDS, GCS ArrayRecord, local video, VoxCeleb2 and URL streams, with augmentation seeded per record.
* **Evaluation and interop**: FID, CLIP score, PSNR and SSIM; safetensors export in the Hugging Face layout that transformers, vLLM and verl read.

Dew is the successor to [FlaxDiff](https://github.com/AshishKumar4/FlaxDiff) and carries its history. It is a personal research project, not a product. Expect sharp edges, and please open an issue when you find one.

## Quick install

```bash
pip install dew-ml             # imports as dew
pip install "jax[cuda12]"      # or "jax[tpu]"; the base install is CPU only
```

Extras: `[tfds]` datasets, `[av]` video and audio, `[streaming]` URL streaming, `[metrics]` FID, `[interop]` safetensors.

The `dew` name on PyPI is a fileless placeholder from 2015. A PEP 541 request is pending; the package is `dew-ml` until it resolves.

## What does Dew look like?

Text conditioned diffusion on Oxford Flowers:

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
input_config = DiffusionInputConfig(
    sample_data_key="image", sample_data_shape=(128, 128, 3),
    conditions=[ConditionalInputConfig(encoder=text)],
)
train_schedule, sample_schedule, transform = get_diffusion_preset("edm")
model = build_model("simple_dit", dict(emb_features=512, num_layers=8, num_heads=8, patch_size=8))

trainer = ObjectiveTrainer(
    model=model, optimizer=optax.adamw(2e-4), input_config=input_config,
    noise_schedule=train_schedule, model_output_transform=transform,
    rngs=jax.random.PRNGKey(4), name="flowers-edm", checkpoint_base_path="./checkpoints",
)
state = trainer.fit(data, training_steps_per_epoch=data["train_len"] // 16, epochs=100,
                    sampler_class=EulerAncestralSampler, sampling_noise_schedule=sample_schedule)

sampler = EulerAncestralSampler(model, sample_schedule, transform, input_config, guidance_scale=3.0)
images = sampler.generate_samples(params=state.ema_params, num_samples=2, resolution=128,
                                  diffusion_steps=50, conditioning=["a water lily", "a rose"])
```

The recipes expose the same trainer as a typed command line. Every config field is a flag; `--help` prints the tree.

```bash
python recipes/diffusion/train.py --data.dataset oxford_flowers102 --model.architecture simple_dit \
    --model.config '{"patch_size": 4, "emb_features": 512, "num_layers": 12, "num_heads": 8}'
python recipes/jepa/train.py --data.dataset oxford_flowers102 --probe-classes 102
```

A new objective is a class with a loss. This one trains a byte-level language model through the unmodified trainer:

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

* [The objectives seam](docs/concepts/objectives.md), [distributed training](docs/concepts/distributed.md), [the data pipeline](docs/concepts/data.md)
* [API reference](docs/api.md) and [recipes](docs/recipes.md)
* [Diffusion explained](https://nbviewer.org/github/AshishKumar4/dew/blob/main/tutorials/simple%20diffusion%20flax.ipynb), a from-scratch notebook independent of the library
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

The test suite requests 8 XLA host devices, so the FSDP, data parallel and resume tests run on any machine. The model, sampler and trainer tests also run on a GPU without the override.

## Acknowledgements

**This project is partially supported by [Google TPU Research Cloud](https://sites.research.google/trc/about/). I would like to thank the Google Cloud TPU team for providing me with the resources to train the bigger text-conditional models in multi-host distributed settings.**

Dew builds on [JAX](https://github.com/jax-ml/jax), [Flax](https://github.com/google/flax), [Optax](https://github.com/google-deepmind/optax), [Orbax](https://github.com/google/orbax), [Grain](https://github.com/google/grain), [tyro](https://github.com/brentyi/tyro), [albumentations](https://github.com/albumentations-team/albumentations) and [Weights & Biases](https://github.com/wandb/wandb). The VAE and parts of the attention code come from [diffusers](https://github.com/huggingface/diffusers), the InceptionV3 from [jax-fid](https://github.com/matthias-wright/jax-fid), and the Karras samplers follow [k-diffusion](https://github.com/crowsonkb/k-diffusion) and the [EDM](https://github.com/NVlabs/edm) reference code.

## License

[MIT](LICENSE)
