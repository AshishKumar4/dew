<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
  <img src="docs/assets/banner-light.svg" alt="dew" width="360">
</picture>

<h1>Dew: a general training framework for JAX and Flax</h1>

<a href="https://github.com/AshishKumar4/dew/actions/workflows/ci.yml"><img src="https://github.com/AshishKumar4/dew/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB" alt="Python 3.11+"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2aa7a1" alt="MIT"></a>

<p>
<a href="#diffusion"><b>Diffusion</b></a>
| <a href="#jepa"><b>JEPA</b></a>
| <a href="#language-models"><b>Language models</b></a>
| <a href="#objectives"><b>Objectives</b></a>
| <a href="#scaling"><b>Scaling</b></a>
| <a href="#installation"><b>Install guide</b></a>
| <a href="docs/index.md"><b>Documentation</b></a>
</p>
</div>

## What is Dew?

Dew trains machine learning models from scratch in JAX and Flax: diffusion and flow matching models, JEPA encoders, autoregressive language models, and your own architectures and objectives, on one set of primitives.

Dew separates what you train from how you train it. An [objective](#objectives) defines the parameters, the loss and the validation step. The [trainer](#scaling) defines the device mesh, the compiled step, gradient accumulation, EMA, checkpoints and logging, and treats every objective the same way. Diffusion, JEPA and language modelling ship as objectives; a new one is a class with a loss.

The [models](#models), the [diffusion maths](#schedules-and-prediction-transforms), the [samplers](#sampling) and the [metrics](#evaluation-and-export) are plain Flax and JAX, and each can be used on its own. The [data pipeline](#data) is built on Grain and gives the same batches on any number of workers and hosts.

Dew is the successor to [FlaxDiff](https://github.com/AshishKumar4/FlaxDiff). It is a personal research project, not a product. Expect sharp edges, and please [report](https://github.com/AshishKumar4/dew/issues) the ones you find.

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
    rngs=jax.random.PRNGKey(0), name="flowers",
)
state = trainer.fit(data, training_steps_per_epoch=data["train_len"] // 32, epochs=50)
# state.params, state.ema_params and state.opt_state, sharded over every device; checkpoints under ./checkpoints/flowers
```

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.svg">
  <img src="docs/assets/architecture-light.svg" alt="Dew modules by layer" width="100%">
</picture>
</div>

### Contents
* [Diffusion](#diffusion)
* [JEPA](#jepa)
* [Language models](#language-models)
* [Objectives](#objectives)
* [Scaling](#scaling)
* [Data](#data)
* [Recipes and examples](#recipes-and-examples)
* [Evaluation and export](#evaluation-and-export)
* [Installation](#installation)
* [Citing Dew](#citing-dew)
* [Reference documentation](#reference-documentation)

## Diffusion

### Schedules and prediction transforms

A diffusion model is defined by three choices: the noise schedule for training, the noise schedule for sampling, and the prediction transform, which says what the network outputs. Use `get_diffusion_preset` to get a matched set:

```python
from dew.diffusion.transforms import get_diffusion_preset

train_schedule, sample_schedule, transform = get_diffusion_preset("edm")    # EDM training, Karras sampling
train_schedule, sample_schedule, transform = get_diffusion_preset("flow")   # rectified flow, as in Stable Diffusion 3
train_schedule, sample_schedule, transform = get_diffusion_preset("cosine", min_snr_gamma=5.0)
```

The parts are available on their own. `dew.diffusion.schedules` has the linear, cosine, exp, sqrt, Karras VE, EDM and flow matching schedules. `dew.diffusion.transforms` has the epsilon, x0, v, flow and Karras transforms. See [docs/api.md](docs/api.md#diffusion) for the list.

### Conditioning

`DiffusionInputConfig` describes the sample and its conditions. A condition pairs an encoder with the batch key it reads:

```python
from dew.inputs import DiffusionInputConfig, ConditionalInputConfig
from dew.inputs.encoders import CLIPTextEncoder

text = CLIPTextEncoder.from_modelname("openai/clip-vit-large-patch14")
inputs = DiffusionInputConfig(
    sample_data_key="image", sample_data_shape=(128, 128, 3),
    conditions=[ConditionalInputConfig(encoder=text)],
)
```

The trainer drops the conditions on 12% of each batch (`unconditional_prob`), so the model also learns the unconditional distribution that classifier-free guidance needs. Pass `autoencoder=StableDiffusionVAE()` to the trainer to train in the latent space of the Stable Diffusion VAE instead of pixels.

### Sampling

A sampler takes the model, the sampling schedule, the transform and the input config. Guidance belongs to the sampler; `guidance_start` and `guidance_stop` limit it to an interval of the trajectory:

```python
from dew.sampling import EulerAncestralSampler

sampler = EulerAncestralSampler(model, sample_schedule, transform, inputs,
                                guidance_scale=4.0, guidance_start=0.1, guidance_stop=0.9)
images = sampler.generate_samples(params=state.ema_params, num_samples=4, resolution=128,
                                  diffusion_steps=50, conditioning=["a water lily", "a rose"])
# images.shape == (4, 128, 128, 3), values in [-1, 1]
```

`DDPMSampler`, `DDIMSampler`, `EulerSampler`, `EulerAncestralSampler`, `HeunSampler`, `RK4Sampler` and `MultiStepDPM` all take the same arguments.

### Models

`build_model` constructs any registered architecture from its name and a dict of keyword arguments:

```python
from dew.registry import build_model

unet = build_model("unet", dict(emb_features=256, feature_depths=[64, 128, 256, 512]))
dit = build_model("simple_dit+hilbert", dict(emb_features=512, num_layers=12, num_heads=8, patch_size=2))
mmdit = build_model("simple_mmdit", dict(emb_features=512, num_layers=12, num_heads=8, patch_size=2,
                                         dtype="bfloat16", attention_impl="auto"))
```

The registry has `unet`, `unet_3d`, `uvit`, `simple_udit`, `simple_dit`, `simple_mmdit`, `hierarchical_mmdit`, `hybrid_dit` (S5 state space blocks between attention blocks), `video_dit`, `causal_transformer` and the JEPA encoders. The `+hilbert` and `+zigzag` suffixes change the order in which patches enter the transformer. Every model takes `dtype` and `attention_impl`, and the parameter tree does not depend on either, so a checkpoint trained with cuDNN attention on a GPU loads on a TPU. See [docs/benchmarks.md](docs/benchmarks.md) for what each one costs per step.

## JEPA

`JepaObjective` trains an I-JEPA or V-JEPA encoder. The predictor reads the encoder's embeddings of the visible patches and predicts the embeddings of masked target blocks; the targets come from a target encoder that is an EMA of the context encoder. The objective logs the representation standard deviation and off-diagonal covariance on every step, which is how a collapsing run shows itself.

```python
from dew.objectives.jepa import JepaObjective, multi_block_mask
from dew.objectives.jepa.probes import get_linear_probe_metric, get_knn_probe_metric

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
# validation logs val/linear_probe_accuracy and val/knn_probe_accuracy, from classifiers fit on the frozen embeddings
```

`jepa_video_encoder` and a `factorized=True` predictor do the same for video. See [docs/concepts/objectives.md](docs/concepts/objectives.md) for more.

## Language models

`CausalTransformer` is a decoder with the parts current open models use: RMSNorm, grouped-query attention, rotary positions, a gated MLP, q/k normalisation, and optional sliding-window layers, embedding scaling and logit softcapping. Its parameter tree follows the Hugging Face layout, so Qwen and Gemma checkpoints map onto it by renaming keys. `LMObjective` is next-token cross entropy in fp32; at validation it reports perplexity and generates text.

```python
from dew.data.dataloaders import get_token_dataset_grain
from dew.objectives.lm import LMObjective
from dew.sampling.text import generate

data = get_token_dataset_grain("data/shakespeare/train.bin", "data/shakespeare/val.bin",
                               batch_size=64, seq_len=256)
model = build_model("causal_transformer", dict(vocab_size=256, emb_features=384, num_layers=6, num_heads=6,
                                              max_seq_len=512, dtype="bfloat16", attention_impl="auto"))
trainer = ObjectiveTrainer(model, optax.adamw(1e-3), objective=LMObjective(model, seq_len=256, vocab_size=256),
                           input_config=None, rngs=jax.random.PRNGKey(0), name="shakespeare")
state = trainer.fit(data, training_steps_per_epoch=300, epochs=4)
# val/perplexity 391 -> 4.6 over the four epochs, on an RTX 4080

tokens = generate(model, state.ema_params, prompt, max_new_tokens=300,
                  rng=jax.random.PRNGKey(0), temperature=0.8, top_k=40)
# prompt is int32 [1, 6] for b"ROMEO:"; tokens.shape == (1, 306)
```

`generate` prefills the KV cache on the prompt and decodes in one `lax.scan`: 0.9 ms per token for a 12-layer, 512-wide model on an RTX 4080. `tools/tokenize_text.py` writes the token files with a byte-level or any Hugging Face tokenizer, and `TokenFileSource` reads them by memory map. See [docs/concepts/language_models.md](docs/concepts/language_models.md) for more.

## Objectives

An objective has four methods. `init_params` builds the parameter tree, which can hold several modules. `loss` returns a scalar and a dict of metrics. `make_validation_step` returns the function that runs at the end of each epoch. `log_validation_artifacts` sends its output to Weights & Biases. `ema` says which part of the tree gets an exponential moving average, and `input_shapes` tells the trainer what a batch looks like.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/seam-dark.svg">
  <img src="docs/assets/seam-light.svg" alt="The four calls the trainer makes into an objective" width="100%">
</picture>
</div>

`LMObjective`, as it ships, is the whole pattern:

```python
import jax.numpy as jnp, optax
from dew.objectives import Objective, EMASpec

class LMObjective(Objective):
    tag = "lm"

    def __init__(self, model, seq_len, *, vocab_size, ema_decay=0.999):
        self.model, self.seq_len, self.vocab_size = model, seq_len, vocab_size
        self.ema = EMASpec(decay=lambda step: ema_decay)

    @property
    def input_shapes(self):
        return {"tokens": ((self.seq_len,), jnp.int32)}

    def init_params(self, rng):
        return self.model.init(rng, jnp.zeros((1, self.seq_len), jnp.int32))

    def loss(self, params, ema_params, batch, rng, step):
        tokens = batch["text"]
        logits = self.model.apply(params, tokens[:, :-1]).astype(jnp.float32)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, tokens[:, 1:]).mean()
        return ce, {"ce": ce, "perplexity": jnp.exp(ce)}

    def make_validation_step(self, **kwargs):
        return lambda val_state, batch: {"ce": self.loss(val_state.params, None, batch, None, 0)[0]}
```

The trainer compiles `loss` into one sharded step with the optimizer and the EMA update, and checkpoints the parameters, the EMA and the optimizer state. `DiffusionObjective` and `JepaObjective` are written the same way. See [docs/concepts/objectives.md](docs/concepts/objectives.md) for more.

## Scaling

The trainer places the run on a two-dimensional mesh named `(data, fsdp)`. The batch is split across all devices. With `fsdp_size=1` the parameters are replicated, which is data parallelism. With `fsdp_size=N` every parameter and optimizer moment above `fsdp_min_param_size` is split across `N` devices along its largest divisible axis. One compiled step serves both, with its shardings declared to XLA and its buffers donated.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/mesh-dark.svg">
  <img src="docs/assets/mesh-light.svg" alt="The (data, fsdp) mesh on two hosts" width="100%">
</picture>
</div>

```python
trainer = ObjectiveTrainer(model, optimizer, ..., fsdp_size=4, grad_accum_steps=2)
# 8 devices: mesh (data=2, fsdp=4); every large parameter split four ways, the EMA and Adam moments with it
```

On a TPU pod every host runs the same script. The recipes join the hosts into one JAX runtime from the cluster environment before the model is built, and stop with an error if they cannot. The data pipeline shards records by process, so each host reads its own part of the dataset.

Checkpoints are written asynchronously with Orbax. The latest checkpoint and the best checkpoint by validation loss are both kept, and the position of the data iterator is saved with them, so a resumed run continues from the next unseen batch.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/training-loop-dark.svg">
  <img src="docs/assets/training-loop-light.svg" alt="The training loop" width="100%">
</picture>
</div>

Models compute in bf16 with fp32 parameters by default, and attention runs on the fused kernel for the current hardware. On an RTX 4080 a 142M parameter DiT trains 2.3x faster this way than in fp32 with reference attention, with a third of the activation memory.

| | Trainer argument | Recipe flag |
|---|---|---|
| FSDP degree | `fsdp_size=4` | `--trainer.fsdp-size 4` |
| Gradient accumulation | `grad_accum_steps=2` | `--optim.grad-accum-steps 2` |
| Checkpoint cadence | `fit(..., checkpoint_every_steps=2000)` | `--trainer.checkpoint-every-steps 2000` |
| Compute dtype | `build_model(..., dtype="bfloat16")` | `--model.dtype bfloat16` |
| Attention kernel | `build_model(..., attention_impl="auto")` | `--model.attention-impl auto` |
| Process pool | `prepare_process(multi_host=True)` | `--trainer.multi-host` |

See [docs/concepts/distributed.md](docs/concepts/distributed.md) for more.

## Data

The data pipeline is built on [Grain](https://github.com/google/grain). A dataset is a random-access source and a transform. Decoding, resizing and augmentation draw their randomness from the record's own generator, so a record gets the same augmentation and the same caption on any number of workers and hosts.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pipeline-dark.svg">
  <img src="docs/assets/pipeline-light.svg" alt="From a data source to the mesh" width="100%">
</picture>
</div>

```python
from dew.config import DataConfig
from dew.data.dataloaders import load_data

data = load_data(DataConfig(dataset="oxford_flowers102", batch_size=32, image_size=128, worker_count=8))
batch = next(data["train"]())
# batch["image"].shape == (32, 128, 128, 3), uint8; batch["text"] holds the tokenized captions
```

The sources cover TFDS datasets, ArrayRecord shards on a GCS mount, local video directories, VoxCeleb2, tokenized text, and URLs streamed while training. `load_data` picks the loader from the dataset registry and holds validation records out of the training set. See [docs/concepts/data.md](docs/concepts/data.md) for more.

## Recipes and examples

`recipes/diffusion/train.py`, `recipes/jepa/train.py` and `recipes/lm/train.py` are complete training programs. A run is a `RunConfig` with `model`, `data`, `optim` and `trainer` parts, and [tyro](https://github.com/brentyi/tyro) turns it into a command line, so `--trainer.fsdp-size 4` sets `config.trainer.fsdp_size`:

```bash
python recipes/diffusion/train.py --data.dataset oxford_flowers102 --data.image-size 128 \
    --model.architecture simple_dit \
    --model.config '{"patch_size": 4, "emb_features": 512, "num_layers": 12, "num_heads": 8}' \
    --trainer.epochs 2000 --trainer.wandb-project my-project

python tools/tokenize_text.py --input shakespeare.txt --out data/shakespeare --tokenizer byte --val-fraction 0.02
python recipes/lm/train.py --data.dataset data/shakespeare --sequence-length 256 \
    --model.config '{"emb_features": 384, "num_layers": 6, "num_heads": 6}' --sample-prompt "ROMEO:"
```

The config that ran is logged with the run, and the inference pipeline rebuilds the model from that record.

The scripts in [`examples/`](examples/) go from a dataset to a trained model, samples on disk and exported weights, and each one runs in the test suite at a tiny size:

* [`train_diffusion.py`](examples/train_diffusion.py): a text-to-image DiT on Oxford Flowers, four prompts sampled to `samples.png`, the EMA weights exported as safetensors.
* [`train_jepa.py`](examples/train_jepa.py): an I-JEPA encoder with linear and kNN probes, the encoder saved on its own.
* [`train_lm.py`](examples/train_lm.py): a byte-level language model on a tokenized corpus, a sample written after training.

See [docs/recipes.md](docs/recipes.md) for the full config tree.

## Evaluation and export

Validation runs at the end of every epoch: the objective produces artifacts (samples, embeddings or text) and `EvaluationMetric` objects score them.

```python
from dew.eval import get_clip_metric, get_fid_metric

trainer = ObjectiveTrainer(model, optimizer, ..., eval_metrics=[get_clip_metric(), get_fid_metric()])
trainer.fit(data, ..., sampler_class=EulerAncestralSampler, sampling_noise_schedule=sample_schedule)
# logs val/clip_similarity and val/fid per epoch; per-batch FID tracks a run over time and is not FID-50k
```

A trained model loads back from its checkpoint directory or from the Weights & Biases model registry, and exports to the safetensors layout that transformers, vLLM and verl read:

```python
from dew.sampling.loading import load_from_checkpoint
from dew.interop import save_hf_layout

state = load_from_checkpoint("./checkpoints/flowers", step="best")
save_hf_layout(state["ema_params"], config=model_config, directory="export/flowers")
# export/flowers/model.safetensors and export/flowers/config.json
```

`dew.eval` has FID (with a vendored InceptionV3), CLIP score, PSNR, SSIM and perplexity. See [docs/api.md](docs/api.md#evaluation-and-interop) for more.

## Installation

### Supported platforms

|            | Linux x86_64 | Cloud TPU VM |
|------------|--------------|--------------|
| CPU        | yes          | yes          |
| NVIDIA GPU | yes          | n/a          |
| Google TPU | n/a          | yes          |

CI runs the test suite on CPU on every push, and the GPU lane runs on an RTX 4080 before each merge. TPU pods trained the models in the [gallery](docs/gallery.md); the current trainer's multi-host path is on the [roadmap](#roadmap) for revalidation.

### Instructions

Dew needs Python 3.11 or later. The base install comes with a CPU-only JAX; install the [JAX build](https://docs.jax.dev/en/latest/installation.html) for your accelerator as well.

| Platform   | Instructions                             |
|------------|------------------------------------------|
| CPU        | `pip install dew-ml`                     |
| NVIDIA GPU | `pip install dew-ml "jax[cuda12]"`       |
| Google TPU | `pip install dew-ml "jax[tpu]"`          |

Optional extras: `dew-ml[tfds]` for TFDS datasets, `[av]` for video and audio, `[streaming]` for URL streaming, `[metrics]` for FID, `[interop]` for safetensors. The package imports as `dew`; the bare `dew` name on PyPI is an unused placeholder, so the distribution is `dew-ml` for now.

To work on Dew itself:

```bash
git clone --recurse-submodules https://github.com/AshishKumar4/dew.git
cd dew && pip install -e ".[test]"
JAX_PLATFORMS=cpu pytest -m "not network" -q
```

The tests simulate 8 devices on CPU, so the sharding and resume tests run on any machine.

## Roadmap

* Loading Qwen3 and Gemma checkpoints into `CausalTransformer`, with logit parity tests against the reference implementation
* GPU kernels measured against the current ones: Pallas flash attention, fused norms, FP8 matmuls
* Diffusion language models
* Audio conditioned video models
* Multi-host validation on a TPU pod
* FID-50k

## Citing Dew

```
@software{dew2026github,
  author = {Ashish Kumar Singh},
  title = {Dew: a general training framework for JAX and Flax},
  url = {https://github.com/AshishKumar4/dew},
  version = {0.1.0},
  year = {2026},
}
```

## Acknowledgements

**This project is partially supported by [Google TPU Research Cloud](https://sites.research.google/trc/about/). I would like to thank the Google Cloud TPU team for providing me with the resources to train the bigger text-conditional models in multi-host distributed settings.**

Dew builds on [JAX](https://github.com/jax-ml/jax), [Flax](https://github.com/google/flax), [Optax](https://github.com/google-deepmind/optax), [Orbax](https://github.com/google/orbax), [Grain](https://github.com/google/grain), [tyro](https://github.com/brentyi/tyro), [albumentations](https://github.com/albumentations-team/albumentations) and [Weights & Biases](https://github.com/wandb/wandb). The VAE and parts of the attention code come from [diffusers](https://github.com/huggingface/diffusers), the InceptionV3 from [jax-fid](https://github.com/matthias-wright/jax-fid), and the Karras samplers follow [k-diffusion](https://github.com/crowsonkb/k-diffusion) and the [EDM](https://github.com/NVlabs/edm) reference code. The papers are listed in [docs/references.md](docs/references.md).

## Reference documentation

* [Concepts](docs/concepts/): [objectives](docs/concepts/objectives.md), [distributed training](docs/concepts/distributed.md), [the data pipeline](docs/concepts/data.md), [language models](docs/concepts/language_models.md)
* [API reference](docs/api.md), [recipes](docs/recipes.md), [benchmarks](docs/benchmarks.md)
* [Diffusion explained](https://nbviewer.org/github/AshishKumar4/dew/blob/main/tutorials/simple%20diffusion%20flax.ipynb), a notebook that builds diffusion from scratch without the library
* [Gallery](docs/gallery.md) and [migrating from FlaxDiff](docs/from-flaxdiff.md)

Dew is released under the [MIT license](LICENSE).
