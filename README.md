<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
  <img src="docs/assets/banner-light.svg" alt="dew" width="360">
</picture>

<h1>Dew: building and training modern architectures at scale, in JAX and Flax</h1>

<a href="https://github.com/AshishKumar4/dew/actions/workflows/ci.yml"><img src="https://github.com/AshishKumar4/dew/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB" alt="Python 3.11+"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2aa7a1" alt="MIT"></a>
</div>

Dew is a framework for building and training modern machine learning architectures in JAX and Flax. It trains image and video diffusion, flow matching, latent diffusion, I-JEPA and V-JEPA encoders, and autoregressive and masked-diffusion language models, through one trainer that runs on one CPU, one GPU, a TPU pod slice, or many hosts.

It grew out of [FlaxDiff](https://github.com/AshishKumar4/FlaxDiff), my diffusion library. I wanted to train more than diffusion, so I pulled the trainer, the data pipeline, the sharding and the checkpointing out of it, and made the thing being learned a plug-in instead of baked into a diffusion trainer. What you train is one small class; the machinery around it is shared.

```python
# runs elsewhere: downloads Oxford Flowers and trains for real
import jax, optax
from dew import Checkpoints, Field, InputSpec, MeshSpec, Trainer, models, presets
from dew.data import OxfordFlowers
from dew.objectives.diffusion import DiffusionObjective

data = OxfordFlowers(image_size=64).load(batch=32)
model = models.SimpleDiT(emb_features=256, num_layers=6, num_heads=4, patch_size=4, dtype="bfloat16")
objective = DiffusionObjective(model, presets.EDM()(), InputSpec(Field("image", (64, 64, 3))))

trainer = Trainer(objective, optax.adamw(3e-4), key=jax.random.key(0),
                  mesh=MeshSpec(fsdp=1), checkpoints=Checkpoints("runs/flowers"))
state = trainer.fit(data, steps=50 * data.steps_per_epoch)
# checkpoints under runs/flowers, with run.json beside them
```

## Contents

* [What it trains](#what-it-trains)
* [The parts](#the-parts)
* [Diffusion](#diffusion)
* [Language models](#language-models)
* [JEPA](#jepa)
* [Scaling](#scaling)
* [Data](#data)
* [Configuration](#configuration)
* [Checkpoints and resume](#checkpoints-and-resume)
* [Extending Dew](#extending-dew)
* [Recipes and examples](#recipes-and-examples)
* [Evaluation and export](#evaluation-and-export)
* [Installation](#installation)
* [Roadmap](#roadmap)
* [Citing](#citing)

## What it trains

| Model | Kind |
|---|---|
| `unet`, `unet_3d` | Convolutional UNets for images and video; the 3D one inflates 2D checkpoints |
| `uvit`, `simple_udit` | U-shaped transformers |
| `simple_dit`, `simple_mmdit`, `hierarchical_mmdit` | DiT, the SD3-style dual-stream MMDiT, and a multi-resolution MMDiT |
| `hybrid_dit` | S5 state space blocks between attention blocks |
| `video_dit` | Factorized spatial and temporal attention for video |
| `jepa_encoder`, `jepa_video_encoder`, `jepa_predictor` | The ViTs the JEPA objective trains |
| `causal_transformer` | A decoder for language models in the Hugging Face layout, with grouped-query attention or a gated delta net per layer, and a mixture of experts per layer |

Each is a Flax module, registered under a name so a config file can build it: `models.build("simple_dit", patch_size=4, ...)`. Every model takes `dtype` and `attention_impl`, and the parameter tree does not depend on either, so a checkpoint trained with cuDNN attention on a GPU loads unchanged on a TPU.

Language model checkpoints load from Hugging Face and match the reference logits: Llama 2, 3 and 3.1, Mistral, Mixtral, Qwen 2, Qwen 3 and Qwen3-MoE, Qwen 3.5 (the gated delta net and full attention hybrid), Gemma 1, 2, 3 and the Gemma 4 text decoder, OLMo 3, and DeepSeek V3 and V3.2 (multi-head latent attention, the sparse indexer, shared experts and the balancing bias).

## The parts

Everything in dew is one of a small number of pieces, and each is usable on its own.

* An **objective** is what is being learned: the parameters, the loss, and what validation produces. `DiffusionObjective`, `JepaObjective` and `LMObjective` ship; a new one is a class with three methods.
* A **trainer** is the machinery around it: the device mesh, the compiled sharded step, EMA, checkpoints, logging. It knows nothing about the modality.
* A **dataset** turns a source into batches through Grain, with the same augmentation and caption for a record on any number of workers and hosts.
* **Schedules, transforms, samplers and guidance** are the diffusion maths, as plain values.
* **Metrics and export** measure a run and write a Hugging Face-style checkpoint.

## Diffusion

A diffusion run is a model, a noise process, and what conditions it. A `Process` holds the training schedule, the sampling schedule and the prediction transform; a preset is a frozen record that builds one.

```python
from dew import presets

process = presets.EDM()()                       # EDM training, Karras sampling
process = presets.Flow(shift=3.0)()             # rectified flow, as in Stable Diffusion 3
process = presets.build("edm", sigma_data=0.5)()  # by name, from a config
```

`InputSpec` describes the sample and its conditions, pairing an encoder with the batch field it reads. The objective drops conditions on a fraction of each batch, so classifier-free guidance has an unconditional model to steer against.

```python
# runs elsewhere: downloads the CLIP text tower
from dew import Condition, Field, InputSpec
from dew.inputs import CLIPText

text = CLIPText.from_pretrained("openai/clip-vit-large-patch14")
inputs = InputSpec(sample=Field("image", (128, 128, 3)),
                   conditions={"textcontext": Condition(text, field="text", unconditional="")})
```

Sampling is one function. `process.denoiser` turns the model into a denoiser, a solver takes one step, `sample` runs the steps in one `lax.scan`, and guidance wraps the denoiser.

```python
import jax
from dew import CFG, sample
from dew.sampling import Heun

weights = state.averaged
encode = lambda prompts: {"textcontext": text.encode(weights["encoders"]["textcontext"], text.tokenize(prompts))}
denoise = process.denoiser(model, weights, encode(["a water lily", "a rose"]), unconditional=encode(["", ""]))
x_T = process.noise(jax.random.key(1), (2, *inputs.sample.shape))
images = sample(denoise, x_T, steps=8, solver=Heun(), guidance=CFG(4.0), key=jax.random.key(2))
```

`TextToImage` does the encoding and decoding for you, from a run directory: `TextToImage.from_run("runs/flowers")`. The solvers `ddpm`, `ddim`, `euler`, `euler_ancestral`, `heun`, `rk4` and `multistep_dpm` are registered in `samplers`.

## Language models

`CausalTransformer` is a decoder with the parts current open models use: RMSNorm, grouped-query attention, rotary positions, a gated MLP, q/k normalisation, and optional sliding-window layers, embedding scaling and logit softcapping. `LMObjective` is next-token cross entropy; at validation it reports perplexity and generates text.

```python
# runs elsewhere: reads a tokenized corpus from disk and trains for real
import jax, optax
from dew import Checkpoints, Trainer, metrics, models
from dew.data import TokenWindows
from dew.objectives.lm import LMObjective, Samples

data = TokenWindows(path="data/shakespeare", seq_len=256).load(batch=64)
model = models.CausalTransformer(vocab_size=256, emb_features=384, num_layers=6, num_heads=6,
                                 max_seq_len=512, dtype="bfloat16")
objective = LMObjective(model, seq_len=256, samples=Samples(prompt=list(b"ROMEO:"), max_new_tokens=300))
trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0), checkpoints=Checkpoints("runs/shakespeare"))
state = trainer.fit(data, steps=1200, metrics=(metrics.perplexity(),))
```

`generate` prefills the KV cache and decodes in one scan, and `tools/tokenize_text.py` writes the token files a corpus trains from. A masked-diffusion objective on the same decoder trains diffusion language models on the sqrt schedule.

## JEPA

`JepaObjective` trains an I-JEPA or V-JEPA encoder. The predictor reads the encoder's embeddings of the visible patches and predicts the embeddings of masked target blocks; the targets come from a target encoder that is an EMA of the context encoder.

```python
import jax, optax
from dew import Checkpoints, Field, Trainer, metrics, models
from dew.objectives.jepa import JepaObjective, multi_block_mask

encoder = models.JepaEncoder(patch_size=4, emb_features=32, num_layers=2, num_heads=2)
predictor = models.JepaPredictor(grid=(4, 4), emb_features=32, predictor_features=16, num_layers=1, num_heads=2)
objective = JepaObjective(encoder, predictor, mask=multi_block_mask((4, 4), num_targets=1, scale=(0.2, 0.3)), sample=Field("image", (16, 16, 3)))
trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0), checkpoints=Checkpoints("runs/ijepa"))
state = trainer.fit(data, steps=steps, metrics=(metrics.linear_probe(5), metrics.knn_probe(5)))
```

Linear and kNN probes score the frozen embeddings at validation. The objective also logs the representation standard deviation each step, which is how a collapsing run shows itself.

## Scaling

The trainer places the run on a mesh named `(data, expert, fsdp)`. `MeshSpec(fsdp=, expert=)` sets two of the sizes and the data axis fills the rest. With `fsdp=1` the parameters are replicated, which is data parallelism; with `fsdp=N` every large parameter and optimizer moment is split across `N` devices along the axis its module declared with `@logical_axes`.

```python
from dew import MeshSpec, Trainer
trainer = Trainer(objective, optimizer, key=key, mesh=MeshSpec(fsdp=1), accumulation=2)
# 8 devices: mesh (data=2, expert=1, fsdp=4); each large parameter split four ways
```

On a TPU pod every host runs the same script, the data pipeline shards records by process, and a `gs://` checkpoint directory writes the shards to a bucket, which is the shared storage a resume across hosts needs. `docs/tpu.md` is the walkthrough, and `dew-tpu` creates a slice and starts a recipe on it.

Models compute in bf16 with fp32 parameters by default, and attention runs on the fused kernel for the hardware (`attention_impl="auto"`). The mesh has five axes: data, FSDP and expert sharding today, a tensor axis a run's rules can redirect a width onto, and a sequence axis that splits token rows; attention that computes across the sequence axis, and pipeline stages, are on the roadmap.

## Data

The data pipeline is built on Grain. A dataset is a random-access source and a transform; decoding, resizing and augmentation draw their randomness from the record's own generator, so a record gets the same treatment on any number of workers and hosts.

| Source | Records | Specs |
|---|---|---|
| TFDS datasets | images with labels or captions | `OxfordFlowers`, `HFImages` |
| ArrayRecord shards on a GCS mount | images with captions, at LAION and COYO scale | `ArrayRecordImages` |
| Local video directories, VoxCeleb2 | video clips with audio | `LocalVideos`, `VoxCeleb2` |
| Tokenized text (`train.bin`, `val.bin`) | token windows or packed documents | `TokenWindows`, `PackedTokens` |
| URL streams (LAION-style tables) | images fetched while training | `OnlineImages` |

`load` returns a `Dataset`: a `train` and a `val` iterator, the record count and the global batch. Augmentation is a field of the spec, not an environment variable.

## Configuration

A run is a `RunConfig`, and tyro turns it into a command line, so `--optim.learning-rate 1e-4` sets `config.optim.learning_rate`. The recipes subclass it to add their objective's knobs.

```python
from dew.config import RunConfig, ModelConfig, OptimConfig, TrainerConfig, Wandb
from dew.data import Loading, OxfordFlowers

config = RunConfig(
    model=ModelConfig("simple_dit", dict(patch_size=4, emb_features=512, num_layers=12, num_heads=8),
                      dtype="bfloat16", attention_impl="auto"),
    data=OxfordFlowers(image_size=128, loading=Loading(workers=16)),
    optim=OptimConfig(optimizer="adamw", learning_rate=2e-4, weight_decay=0.01, clip_grads=1.0),
    trainer=TrainerConfig(name="flowers-dit", batch_size=64, epochs=500, wandb=Wandb(project="dew")),
)
config.save("runs/flowers-dit")   # run.json, next to the checkpoints
```

`--help` on any recipe prints the whole tree with its defaults.

## Checkpoints and resume

A checkpoint holds the step, the parameters, the optimizer state, the EMA, the run key and every process's place in its data stream, written asynchronously by Orbax. A run whose directory holds a checkpoint resumes from the latest one and continues from the next unseen batch. For inference, the run directory is the whole record:

```python
# runs elsewhere: needs a run directory a recipe wrote
from dew.sampling import TextToImage

pipe = TextToImage.from_run("runs/flowers")
images = pipe(["a water lily", "a sunflower"], steps=40, guidance=4.0, key=jax.random.key(0))
```

## Extending Dew

Each piece is added by writing a class and registering it under a name, after which it is available to configs and recipes.

**A model** is a Flax module plus a sharding declaration, so the layout knows which dimensions mean what:

```python
import flax.linen as nn
from dew import models
from dew.nn.sharding import logical_axes

@models("my_dit")
@logical_axes({("mixer_in",): ("embed", "mlp"), ("mixer_out",): ("mlp", "embed")})
class MyDiT(nn.Module):
    ...
# models.build("my_dit", ...) now works, recipes included
```

**An objective** is a class with `init`, `loss` and `evaluate` and the `inputs` and `ema` attributes, registered with `@objectives(name)`. **A dataset** is a frozen dataclass whose `load(batch=)` returns a `Dataset`, registered with `@datasets(name)`. **A metric** is a function from an artifact and its batch to a value, behind `@metrics(name)`. **A sampler** is a `Solver` behind `@samplers(name)`, **a conditioning encoder** a `ConditionEncoder` behind `@encoders(name)`, and **a preset** a dataclass callable to a `Process` behind `@presets(name)`.

## Recipes and examples

`recipes/diffusion/train.py`, `recipes/jepa/train.py` and `recipes/lm/train.py` are complete training programs over the config tree.

```bash
python recipes/diffusion/train.py data:oxford-flowers --data.image-size 128 \
    --model.architecture simple_dit \
    --model.config '{"patch_size": 4, "emb_features": 512, "num_layers": 12, "num_heads": 8}' \
    --trainer.epochs 2000 preset:edm
```

The scripts in [`examples/`](examples/) go from a dataset to a trained model and exported weights. In [`tutorials/`](tutorials/), [01](tutorials/01-diffusion-from-scratch.ipynb) builds diffusion from scratch without the library. The other notebooks (02 to 08) are **WORK IN PROGRESS**: they were written against an earlier API and are being rewritten.

## Evaluation and export

Validation runs every `eval_every` steps: the objective produces artifacts and the metrics score them. `dew.eval` has FID (a vendored InceptionV3), CLIP score, PSNR, SSIM and perplexity. A trained model exports to the `model.safetensors` and `config.json` pair a Hugging Face-style loader reads.

```python
from dew.interop import save_hf_layout

save_hf_layout(state.averaged["params"], config={"architecture": "simple_dit", **fields}, directory="export/flowers")
```

`push_to_hub(directory, repo_id)` uploads that directory, and `pull_from_hub(repo_id)` downloads a repo snapshot.

## Installation

Dew needs Python 3.11 or later. There is no release on PyPI yet; install from the repository. The base install comes with a CPU-only JAX; install the [JAX build](https://docs.jax.dev/en/latest/installation.html) for your accelerator as well.

| Platform   | Instructions                                                                 |
|------------|------------------------------------------------------------------------------|
| CPU        | `pip install "dew-ml @ git+https://github.com/AshishKumar4/dew"`               |
| NVIDIA GPU | `pip install "dew-ml @ git+https://github.com/AshishKumar4/dew" "jax[cuda12]"` |
| Google TPU | `pip install "dew-ml @ git+https://github.com/AshishKumar4/dew" "jax[tpu]"`    |

Optional extras: `[tfds]` for TFDS datasets, `[av]` for video and audio, `[streaming]` for URL streaming, `[metrics]` for FID, `[interop]` for safetensors. The package imports as `dew`.

```bash
git clone https://github.com/AshishKumar4/dew.git
cd dew && pip install -e ".[test]"
JAX_PLATFORMS=cpu pytest -m "not network" -q
```

To work on dew itself, read [CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

The goal is to train the way the large labs train and to run what they release, on the same trainer.

**Architecture parity.** Everything MaxText trains: DeepSeek V2 and Kimi K2 on the latent attention that landed, GLM 4.5 and 5, gpt-oss with its attention sinks and MXFP4 weights, Llama 4, the Gemma 4 mixture-of-experts sizes and Gemma 3n; then the vision towers of Gemma 4, Llama 4 and Qwen 3.5; diffusion language models at the open-weight scale. Each family lands when its logits match the reference implementation on a real checkpoint.

**Systems.** Attention that shards the sequence over the sequence axis, and pipeline stages over the stage axis; int8 and FP8 training with fine-grained scaling, and MXFP4 and FP8 weight loading; the MuonClip optimizer; emergency checkpointing and goodput measurement; scan over layers for compile time at depth.

**Post-training.** A clean story for SFT and reinforcement learning that fits the same objective-and-trainer seam, rather than a second framework beside it.

## Acknowledgements

**This project is partially supported by [Google TPU Research Cloud](https://sites.research.google/trc/about/). I would like to thank the Google Cloud TPU team for providing me with the resources to train the bigger text-conditional models in multi-host distributed settings.**

Dew builds on [JAX](https://github.com/jax-ml/jax), [Flax](https://github.com/google/flax), [Optax](https://github.com/google-deepmind/optax), [Orbax](https://github.com/google/orbax), [Grain](https://github.com/google/grain), [tyro](https://github.com/brentyi/tyro), [albumentations](https://github.com/albumentations-team/albumentations) and [Weights & Biases](https://github.com/wandb/wandb). The VAE and parts of the attention code come from [diffusers](https://github.com/huggingface/diffusers), the InceptionV3 from [jax-fid](https://github.com/matthias-wright/jax-fid), and the Karras samplers follow [k-diffusion](https://github.com/crowsonkb/k-diffusion) and the [EDM](https://github.com/NVlabs/edm) reference code. The papers are listed in [docs/references.md](docs/references.md).

## Reference documentation

* [Concepts](docs/concepts/): [objectives](docs/concepts/objectives.md), [distributed training](docs/concepts/distributed.md), [the data pipeline](docs/concepts/data.md), [language models](docs/concepts/language_models.md), [mixture of experts](docs/concepts/moe.md)
* [API reference](docs/api.md), [recipes](docs/recipes.md), [benchmarks](docs/benchmarks.md), [TPUs](docs/tpu.md)
* [Diffusion explained](https://nbviewer.org/github/AshishKumar4/dew/blob/main/tutorials/01-diffusion-from-scratch.ipynb), a notebook that builds diffusion from scratch without the library
* [Gallery](docs/gallery.md) and [migrating from FlaxDiff](docs/from-flaxdiff.md)

Dew is released under the [MIT license](LICENSE).
