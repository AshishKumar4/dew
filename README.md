<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
    <img src="docs/assets/banner-light.svg" alt="dew" width="720">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/AshishKumar4/dew/actions/workflows/ci.yml"><img src="https://github.com/AshishKumar4/dew/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-2aa7a1" alt="MIT">
</p>

Dew trains models from scratch in JAX and Flax. It grew out of [FlaxDiff](https://github.com/AshishKumar4/FlaxDiff), my diffusion library, and keeps all of it. What you optimize is now a plug-in, so one trainer, one data pipeline and one sharding and checkpointing setup serve diffusion models, JEPA encoders and, next, language models.

**This project is partially supported by [Google TPU Research Cloud](https://sites.research.google/trc/about/). I would like to thank the Google Cloud TPU team for providing me with the resources to train the bigger text-conditional models in multi-host distributed settings.**

## How it fits together

An objective owns the parameters, the loss and the validation step. The trainer owns the mesh, the jitted step, EMA, gradient accumulation, checkpoints and logging, and never looks inside the objective.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.svg">
    <img src="docs/assets/architecture-light.svg" alt="Dew modules" width="100%">
  </picture>
</p>

Every host runs the same pipeline on its own shard of the data. The mesh joins them.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pipeline-dark.svg">
    <img src="docs/assets/pipeline-light.svg" alt="One training run" width="100%">
  </picture>
</p>

## Install

```bash
pip install dew-ml             # imports as dew
pip install "jax[cuda12]"      # or "jax[tpu]"; the base install is CPU only
```

Extras: `[tfds]` datasets, `[av]` video and audio, `[streaming]` URL streaming, `[metrics]` FID, `[interop]` safetensors. To work on it: `git clone --recurse-submodules` and `pip install -e ".[test]"`.

The `dew` name on PyPI is an empty placeholder from 2015. I have asked for it. Until then the package is `dew-ml`.

## Quickstart

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
input_config = DiffusionInputConfig(
    sample_data_key="image", sample_data_shape=(128, 128, 3),
    conditions=[ConditionalInputConfig(encoder=CLIPTextEncoder.from_modelname("openai/clip-vit-large-patch14"))],
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

The recipes wrap the same trainer in a typed config. Every field is a flag, and `--help` prints the tree.

```bash
python recipes/diffusion/train.py --data.dataset oxford_flowers102 --model.architecture simple_dit \
    --model.config '{"patch_size": 4, "emb_features": 512, "num_layers": 12, "num_heads": 8}'
python recipes/jepa/train.py --data.dataset oxford_flowers102 --probe-classes 102
```

## Objectives

Anything with a loss trains. This byte-level language model ran through the trainer as it is:

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

`ObjectiveTrainer(model, optimizer, objective=LMObjective(model, seq_len), ...)`, then `fit`. The seam is described in [docs/concepts/objectives.md](docs/concepts/objectives.md).

## What's inside

| | |
|---|---|
| `dew.nn` | UNet, UNet3D, UViT, DiT, MMDiT, HierarchicalMMDiT, SSM-DiT, VideoDiT, JEPA encoders. Hilbert and zigzag scan orders. SD VAE and a small autoencoder. One attention path (reference, XLA, cuDNN, TPU) with one parameter tree. |
| `dew.diffusion` | Schedules: linear, cosine, exp, sqrt, Karras VE, EDM, flow matching. Transforms: epsilon, x0, v, flow, Karras. `get_diffusion_preset` pairs them. |
| `dew.objectives` | `DiffusionObjective` (pixel or latent, min-SNR weighting). `JepaObjective` (I-JEPA, V-JEPA, collapse telemetry, probes). |
| `dew.sampling` | DDPM, DDIM, Euler, Euler ancestral, Heun, RK4, multistep DPM. Interval-limited CFG. Pipelines that rebuild a model from its run record. |
| `dew.training` | Data parallel and FSDP on a `(data, fsdp)` mesh. One compile per run. EMA on the update clock. Async orbax checkpoints, latest and best, mid-epoch resume. wandb, profiler, MFU. bf16 compute over fp32 params by default. |
| `dew.data` | grain pipelines over TFDS, GCS ArrayRecord, local video, VoxCeleb2 and URL streams. Augmentation seeded per record. CLIP text and HF audio conditioning. |
| `dew.eval` | FID, CLIP score, PSNR, SSIM. |
| `dew.interop` | Flax params to and from safetensors. `save_hf_layout` writes what transformers, vLLM and verl read. |

Full tables in [docs/api.md](docs/api.md).

## Testing

```bash
JAX_PLATFORMS=cpu pytest -m "not network" -q
```

`conftest.py` gives XLA 8 host devices, so the FSDP, data parallel, checkpoint and resume tests run on any machine. The model, sampler and trainer files also run on a real GPU without the override.

## Docs

- [Concepts](docs/concepts/): the objectives seam, distributed training, the data pipeline
- [Recipes](docs/recipes.md) and the [API](docs/api.md)
- [Diffusion explained](https://nbviewer.org/github/AshishKumar4/dew/blob/main/tutorials/simple%20diffusion%20flax.ipynb), a notebook written from scratch, independent of the library
- [Gallery](docs/gallery.md), [references](docs/references.md), [coming from FlaxDiff](docs/from-flaxdiff.md)

## Roadmap

- Autoregressive and diffusion language model objectives
- Audio conditioned video models end to end
- Multi-host validation on a TPU pod
- FID-50k

## Gallery

![Text to image](images/medium_epoch5.png)

Text to image at 128px on Laion-Aesthetics 12M, CC12M, MS COCO and a COYO-700M subset, trained on a TPU-v4-32. More in [docs/gallery.md](docs/gallery.md).

## Acknowledgements

Built on JAX, Flax, Optax, Orbax, Grain, tyro, albumentations and Weights & Biases. The VAE and parts of the attention code come from diffusers, the InceptionV3 from jax-fid, and the Karras samplers follow k-diffusion and the EDM code. Papers and blogs are in [docs/references.md](docs/references.md).

## About

I started this as a hobby project to learn Flax and the maths behind diffusion models. My day job is Golang systems engineering with some applied ML, so the code reflects a learning journey. Please open an issue if you find mistakes. Pull requests are welcome.

## License

MIT.
