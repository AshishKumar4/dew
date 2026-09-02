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
  <a href="#diffusion"><b>Diffusion</b></a> |
  <a href="#jepa"><b>JEPA</b></a> |
  <a href="#objectives"><b>Objectives</b></a> |
  <a href="#scaling"><b>Scaling</b></a> |
  <a href="#installation"><b>Installation</b></a> |
  <a href="#documentation"><b>Documentation</b></a>
</p>

## What is Dew?

Dew is a general framework for training machine learning models in JAX and Flax. It trains diffusion models, flow matching models, JEPA encoders, and your own architectures and objectives, sharded across devices and hosts. One trainer, one data pipeline and one set of primitives serve all of them.

Dew separates what you train from how you train it. An objective defines the parameters, the loss and the validation step. The trainer defines the device mesh, the compiled step, gradient accumulation, EMA, checkpoints and logging. Diffusion and JEPA are the objectives that ship with Dew. A language model is [a few more lines](#objectives).

The models, schedules, samplers and metrics are plain Flax and JAX. Each can be used on its own.

This is a personal research project, not a product. Expect sharp edges.

```python
import jax, optax
from dew.data.dataloaders import get_dataset_grain
from dew.diffusion.transforms import get_diffusion_preset
from dew.inputs import DiffusionInputConfig
from dew.registry import build_model
from dew.training import ObjectiveTrainer

data = get_dataset_grain("oxford_flowers102", batch_size=32, image_scale=64)
train_schedule, sample_schedule, transform = get_diffusion_preset("edm")
model = build_model("simple_dit", dict(emb_features=256, num_layers=6, num_heads=4, patch_size=4))

trainer = ObjectiveTrainer(
    model, optax.adamw(3e-4),
    input_config=DiffusionInputConfig(sample_data_key="image", sample_data_shape=(64, 64, 3), conditions=[]),
    noise_schedule=train_schedule, model_output_transform=transform,
    rngs=jax.random.PRNGKey(0), name="flowers", checkpoint_base_path="./checkpoints",
)
state = trainer.fit(data, training_steps_per_epoch=data["train_len"] // 32, epochs=50)
```

### Contents

* [Diffusion](#diffusion)
* [JEPA](#jepa)
* [Objectives](#objectives)
* [Scaling](#scaling)
* [Data](#data)
* [Recipes](#recipes)
* [Evaluation and export](#evaluation-and-export)
* [Installation](#installation)
* [Documentation](#documentation)
* [Roadmap](#roadmap)

## Diffusion

### Schedules and prediction transforms

Training a diffusion model needs three choices: the noise schedule for training, the noise schedule for sampling, and the prediction transform, which says what the network outputs (the noise, the clean sample, a velocity). `get_diffusion_preset` returns a matched set:

```python
from dew.diffusion.transforms import get_diffusion_preset

train_schedule, sample_schedule, transform = get_diffusion_preset("edm")     # EDM training, Karras sampling
train_schedule, sample_schedule, transform = get_diffusion_preset("flow")    # rectified flow, as in Stable Diffusion 3
train_schedule, sample_schedule, transform = get_diffusion_preset("cosine")  # discrete cosine schedule, epsilon prediction
```

`min_snr_gamma=5.0` adds min-SNR loss weighting to any preset. The parts are also available on their own: `dew.diffusion.schedules` has the linear, cosine, exp, sqrt, Karras VE, EDM and flow matching schedules, and `dew.diffusion.transforms` has the epsilon, x0, v, flow and Karras transforms.

### Conditioning

`DiffusionInputConfig` describes the sample and its conditions. A condition pairs an encoder with the batch key it reads. Dew ships a CLIP text encoder and a Hugging Face audio encoder:

```python
from dew.inputs import DiffusionInputConfig, ConditionalInputConfig
from dew.inputs.encoders import CLIPTextEncoder

text = CLIPTextEncoder.from_modelname("openai/clip-vit-large-patch14")
inputs = DiffusionInputConfig(
    sample_data_key="image", sample_data_shape=(128, 128, 3),
    conditions=[ConditionalInputConfig(encoder=text)],
)
```

The trainer drops the conditions on a fraction of the batch (`unconditional_prob`, 0.12 by default), so the model also learns the unconditional distribution that classifier-free guidance needs.

### Latent diffusion

Pass an autoencoder and the objective trains in its latent space. `StableDiffusionVAE` loads the Stable Diffusion VAE from the Hugging Face hub; `SimpleAutoEncoder` is a small convolutional one you train yourself.

```python
from dew.nn.autoencoders import StableDiffusionVAE

trainer = ObjectiveTrainer(model, optimizer, input_config=inputs, autoencoder=StableDiffusionVAE(), ...)
```

### Sampling

A sampler takes the model, the sampling schedule, the transform and the input config. Guidance is a property of the sampler. `guidance_start` and `guidance_stop` limit it to an interval of the trajectory, which improves sample quality at high guidance scales.

```python
from dew.sampling import EulerAncestralSampler

sampler = EulerAncestralSampler(model, sample_schedule, transform, inputs,
                                guidance_scale=4.0, guidance_start=0.1, guidance_stop=0.9)
images = sampler.generate_samples(params=state.ema_params, num_samples=4, resolution=128,
                                  diffusion_steps=50, conditioning=["a water lily", "a rose"])
```

The samplers are `DDPMSampler`, `DDIMSampler`, `EulerSampler`, `EulerAncestralSampler`, `HeunSampler`, `RK4Sampler` and `MultiStepDPM`. All of them take the same arguments.

### Models

`build_model` constructs any registered architecture from its name and a dict of keyword arguments:

```python
from dew.registry import build_model

unet = build_model("unet", dict(emb_features=256, feature_depths=[64, 128, 256, 512]))
dit = build_model("simple_dit+hilbert", dict(emb_features=512, num_layers=12, num_heads=8, patch_size=2))
mmdit = build_model("simple_mmdit", dict(emb_features=512, num_layers=12, num_heads=8, patch_size=2))
```

The registry has `unet`, `unet_3d`, `uvit`, `simple_udit`, `simple_dit`, `simple_mmdit`, `hierarchical_mmdit`, `hybrid_dit` (S5 state space blocks interleaved with attention), `video_dit` and the JEPA encoders. The `+hilbert` and `+zigzag` suffixes change the order in which patches enter the transformer. Every model takes `dtype` and `attention_impl`; the parameter tree does not depend on either, so a checkpoint trained with cuDNN attention on a GPU loads on a TPU.

## JEPA

`JepaObjective` trains an I-JEPA or V-JEPA encoder. The predictor reads the encoder's embeddings of the visible patches and predicts the embeddings of masked target blocks. The targets come from a target encoder, an EMA of the context encoder. The objective logs the representation standard deviation and off-diagonal covariance on every step, which is how a collapsing run shows itself.

```python
from dew.objectives.jepa import JepaObjective, multi_block_mask
from dew.objectives.jepa.probes import get_linear_probe_metric, get_knn_probe_metric
from dew.registry import build_model

encoder = build_model("jepa_encoder", dict(patch_size=16, emb_features=384, num_layers=12, num_heads=6))
predictor = build_model("jepa_predictor", dict(grid=(14, 14), emb_features=384, predictor_features=192,
                                                num_layers=6, num_heads=6))
objective = JepaObjective(encoder, predictor, mask=multi_block_mask((14, 14)),
                          sample_data_key="image", sample_data_shape=(224, 224, 3))

trainer = ObjectiveTrainer(
    encoder, optax.adamw(1e-3), objective=objective,
    input_config=DiffusionInputConfig(sample_data_key="image", sample_data_shape=(224, 224, 3), conditions=[]),
    eval_metrics=[get_linear_probe_metric(102), get_knn_probe_metric(102)],
    rngs=jax.random.PRNGKey(0), name="ijepa-flowers",
)
```

The probes fit a linear classifier and a kNN classifier on the frozen embeddings of each validation batch and report their accuracy. `jepa_video_encoder` and a `factorized=True` predictor do the same for video.

## Objectives

An objective is a class with four methods. `init_params` builds the parameter tree, which can hold several modules. `loss` returns a scalar and a dict of metrics. `make_validation_step` returns the function that runs at the end of each epoch. `log_validation_artifacts` sends its output to Weights & Biases. `ema` says which part of the tree gets an exponential moving average, and how fast.

This objective trains a byte-level language model:

```python
import jax.numpy as jnp, optax
from dew.objectives import Objective, EMASpec

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

trainer = ObjectiveTrainer(model, optax.adamw(1e-3), objective=LMObjective(model, seq_len=256), ...)
```

The trainer compiles `loss` into a sharded training step, applies the optimizer, moves the EMA on every optimizer update, and checkpoints the parameters, the EMA and the optimizer state. `DiffusionObjective` and `JepaObjective` are written the same way.

## Scaling

The trainer places everything on a two-dimensional mesh named `(data, fsdp)`. The batch is split across all devices. With `fsdp_size=1` the parameters are replicated, which is data parallelism. With `fsdp_size=N` every parameter and optimizer moment larger than `fsdp_min_param_size` is split across `N` devices along its largest divisible axis, and the rest stay replicated. One compiled step serves both, with input and output shardings declared to XLA and the state buffers donated.

```python
trainer = ObjectiveTrainer(model, optimizer, ..., fsdp_size=4, grad_accum_steps=2)
```

On a TPU pod or any multi-process run, every host runs the same script. `--trainer.multi-host` joins the processes into one JAX runtime before the model is built. The data pipeline shards records by process, so each host reads its own part of the dataset.

Checkpoints are written asynchronously with Orbax. The latest checkpoint and the best checkpoint by validation loss are both kept. The position of the data iterator is saved with them, so a resumed run continues from the next unseen batch. `checkpoint_every_steps` adds a fixed cadence between epoch boundaries.

Models compute in bf16 with fp32 parameters by default in the recipes, and attention runs on the fused kernel for the current hardware (`attention_impl="auto"` picks cuDNN on GPUs). On an RTX 4080, a 142M parameter DiT trains 2.3x faster this way than in fp32 with reference attention, with a third of the activation memory.

| Setting | Recipe flag |
|---|---|
| FSDP degree | `--trainer.fsdp-size 4` |
| Gradient accumulation | `--optim.grad-accum-steps 2` |
| Multi-host | `--trainer.multi-host` |
| Compute dtype | `--model.dtype bfloat16` |
| Attention kernel | `--model.attention-impl auto` |
| Checkpoint cadence | `--trainer.checkpoint-every-steps 2000` |
| Compilation cache | `--trainer.compilation-cache-dir ~/.cache/dew/xla` |

## Data

The data pipeline is built on [Grain](https://github.com/google/grain). A dataset is a random-access source plus a transform. The sources cover TFDS datasets, ArrayRecord shards on a GCS mount, local video directories, VoxCeleb2, and URLs streamed while training. Decoding, resizing and augmentation run as Grain transforms that draw randomness from the record's own generator. A record gets the same augmentation and the same caption on any number of workers and hosts.

```python
from dew.config import DataConfig
from dew.data.dataloaders import load_data

data = load_data(DataConfig(dataset="oxford_flowers102", batch_size=32, image_size=128, worker_count=8))
batch = next(data["train"]())          # {"image": (32, 128, 128, 3) uint8, "text": {"input_ids", "attention_mask"}}
```

`load_data` picks the loader from the dataset registry. Validation records are held out of the training set. The registry names are in `dew.data.registry`.

## Recipes

`recipes/diffusion/train.py` and `recipes/jepa/train.py` are the complete training programs. A run is a `RunConfig` with `model`, `data`, `optim` and `trainer` parts, and [tyro](https://github.com/brentyi/tyro) turns it into a command line: `--trainer.fsdp-size 4` sets `config.trainer.fsdp_size`. Architecture arguments go to `--model.config` as JSON and straight to `build_model`.

```bash
python recipes/diffusion/train.py --data.dataset oxford_flowers102 --data.image-size 128 \
    --model.architecture simple_dit \
    --model.config '{"patch_size": 4, "emb_features": 512, "num_layers": 12, "num_heads": 8}' \
    --trainer.epochs 2000 --trainer.wandb-project my-project

python recipes/jepa/train.py --data.dataset oxford_flowers102 --probe-classes 102 \
    --model.config '{"patch_size": 16, "emb_features": 384}'

python recipes/diffusion/train.py --help
```

The config that ran is logged with the run, and `DiffusionInferencePipeline` rebuilds the model from that record.

## Evaluation and export

Validation runs at the end of every epoch. For diffusion it generates samples with the sampler you pass to `fit`; for JEPA it embeds the validation batch. `EvaluationMetric` objects score the result:

```python
from dew.eval import get_clip_metric, get_fid_metric

trainer = ObjectiveTrainer(model, optimizer, ..., eval_metrics=[get_clip_metric(), get_fid_metric()])
trainer.fit(data, ..., sampler_class=EulerAncestralSampler, sampling_noise_schedule=sample_schedule)
```

`dew.eval` has FID (with a vendored InceptionV3), CLIP score, PSNR and SSIM. Per-batch FID tracks a run over time; it is not comparable with published FID-50k numbers.

A trained model loads back from its checkpoint directory or from the Weights & Biases model registry:

```python
from dew.sampling.loading import load_from_checkpoint
from dew.sampling.pipelines import DiffusionInferencePipeline

state = load_from_checkpoint("./checkpoints/flowers", step="best")
pipeline = DiffusionInferencePipeline.from_wandb_registry(modelname="flowers", project="my-project")
```

`dew.interop` writes the parameters to safetensors, one tensor per leaf, and `save_hf_layout` writes the `model.safetensors` and `config.json` pair that transformers, vLLM and verl load:

```python
from dew.interop import save_hf_layout

save_hf_layout(state.ema_params, config=model_config, directory="export/flowers")
```

## Installation

Dew needs Python 3.11 or later. The base install comes with a CPU-only JAX; install the [JAX build](https://docs.jax.dev/en/latest/installation.html) for your accelerator as well.

| Hardware | Command |
|---|---|
| CPU | `pip install dew-ml` |
| NVIDIA GPU | `pip install dew-ml "jax[cuda12]"` |
| Cloud TPU | `pip install dew-ml "jax[tpu]"` |

Optional extras: `[tfds]` for TFDS datasets, `[av]` for video and audio, `[streaming]` for URL streaming, `[metrics]` for FID, `[interop]` for safetensors.

To work on Dew itself:

```bash
git clone --recurse-submodules git@github.com:AshishKumar4/dew.git
cd dew && pip install -e ".[test]"
JAX_PLATFORMS=cpu pytest -m "not network" -q
```

The tests simulate 8 devices on CPU, so the sharding tests run on any machine.

`dew` on PyPI is an unused placeholder, so the package is `dew-ml` for now.

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

## Acknowledgements

**This project is partially supported by [Google TPU Research Cloud](https://sites.research.google/trc/about/). I would like to thank the Google Cloud TPU team for providing me with the resources to train the bigger text-conditional models in multi-host distributed settings.**

Dew is the successor to [FlaxDiff](https://github.com/AshishKumar4/FlaxDiff). It builds on [JAX](https://github.com/jax-ml/jax), [Flax](https://github.com/google/flax), [Optax](https://github.com/google-deepmind/optax), [Orbax](https://github.com/google/orbax), [Grain](https://github.com/google/grain), [tyro](https://github.com/brentyi/tyro), [albumentations](https://github.com/albumentations-team/albumentations) and [Weights & Biases](https://github.com/wandb/wandb). The VAE and parts of the attention code come from [diffusers](https://github.com/huggingface/diffusers). The InceptionV3 comes from [jax-fid](https://github.com/matthias-wright/jax-fid). The Karras samplers follow [k-diffusion](https://github.com/crowsonkb/k-diffusion) and the [EDM](https://github.com/NVlabs/edm) reference code.

## License

[MIT](LICENSE)
