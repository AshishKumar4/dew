<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
  <img src="docs/assets/banner-light.svg" alt="Dew" width="960">
</picture>

# Dew

Model training with JAX and Flax

[![CI](https://github.com/AshishKumar4/dew/actions/workflows/ci.yml/badge.svg)](https://github.com/AshishKumar4/dew/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-2aa7a1)](LICENSE)

[Documentation](docs/index.md) · [Installation](#installation) · [Examples](examples/) · [API reference](docs/reference/core-api.md)
</div>

Dew is a framework for training language models, image and video diffusion models, and JEPA encoders in JAX. It provides Flax Linen models and training objectives, with a shared trainer for optimization, device sharding, evaluation, and checkpoints.

Use the supplied architectures, load a supported Hugging Face checkpoint, or train your own Flax model. Model variables and training state remain JAX PyTrees; optimizers are Optax transformations, data loading uses Grain, and checkpoints use Orbax.

Dew is under active development. APIs and training checkpoint formats can change before 1.0. See [model support](docs/reference/model-families.md) and [current limitations](docs/reference/support.md) for the supported configurations.

## Contents

- [Getting started](#getting-started)
- [Features](#features)
- [Models](#models)
- [Training](#training)
- [Diffusion and sampling](#diffusion-and-sampling)
- [Distributed training](#distributed-training)
- [Data and configuration](#data-and-configuration)
- [Installation](#installation)
- [Documentation and examples](#documentation-and-examples)
- [Contributing and acknowledgements](#contributing-and-acknowledgements)

## Getting started

This example trains a small flow-matching DiT and generates four 64×64 RGB samples. It creates random images so you can run it without a dataset download. Replace `images` with your training data for a real experiment.

```python
import itertools

import jax
import numpy as np
import optax

from dew import Checkpoints, Dataset, Field, InputSpec, Trainer, models, presets, sample
from dew.objectives.diffusion import DiffusionObjective
from dew.sampling import Euler

images = np.random.default_rng(0).integers(0, 256, (8, 64, 64, 3), dtype=np.uint8)
data = Dataset(train=lambda: itertools.repeat({"image": images}),
               val=None, records=8, batch=8)
model = models.build("simple_dit", emb_features=64, num_layers=2,
                     num_heads=4, patch_size=8, attention_impl="xla")
process = presets.Flow()()
objective = DiffusionObjective(model, process, InputSpec(Field("image", (64, 64, 3))))
trainer = Trainer(objective, optax.adamw(3e-4), key=jax.random.key(0),
                  checkpoints=Checkpoints("runs/readme-flow"))
state = trainer.fit(data, steps=20, log_every=5)

denoise = process.denoiser(model, state.averaged, conditions={})
noise = process.noise(jax.random.key(1), (4, 64, 64, 3))
samples = sample(denoise, noise, steps=16, solver=Euler(), key=jax.random.key(2))
np.save("samples.npy", np.asarray(samples))
print("Generated:", samples.shape)
```

The program logs the training loss, writes a checkpoint under `runs/readme-flow`, and saves `samples.npy` with shape `(4, 64, 64, 3)`. It runs on CPU; use a CUDA JAX installation to run it on a GPU. Twenty updates on random images demonstrate the training and sampling APIs, not image generation quality.

`DiffusionObjective` constructs noisy inputs and prediction targets. `Trainer` differentiates the loss and applies the optimizer. `state.averaged` supplies EMA weights for sampling. To train on images from disk, follow the [diffusion guide](docs/guides/diffusion.md) and [dataset recipes](docs/recipes.md).

For a longer example, run [`examples/readme_demo.py`](examples/readme_demo.py). It trains a language model, resumes its checkpoint, applies DPO, and trains a separate flow model. The script creates its own data and saves both checkpoints and image previews:

```bash
JAX_PLATFORMS=cpu python examples/readme_demo.py --out runs/demo
```

## Features

- **Language modeling:** autoregressive pretraining, packed documents, assistant-only SFT, DPO, and GRPO with callable rewards.
- **Diffusion:** image and video denoising, rectified flow, latent diffusion, masked-token diffusion, classifier-free guidance, and interchangeable schedules and solvers.
- **Representation learning:** I-JEPA and V-JEPA with context/target encoders, predictors, block masking, and linear/kNN probes.
- **Training systems:** data/FSDP/expert/tensor/sequence parallelism, layer scans, gradient accumulation, mixed precision, asynchronous checkpoints, and configurable EMA.
- **Interoperability:** supported Hugging Face configuration and weight translation, safetensors export, CLIP/T5 conditioning, and VAE components.
- **Evaluation:** perplexity, FID, CLIP score, PSNR, SSIM, representation diagnostics, generated previews, and Weights & Biases tracking.

The supplied DPO and GRPO objectives use the same trainer as pretraining. PPO-related estimators and loss functions are available separately in `dew.rl`; they are not a complete PPO training system. Full Transformers/Diffusers/MaxText coverage, serving, and multi-turn agent training remain development goals.

## Models

### Language models

`CausalTransformer` combines attention, recurrent mixers, feed-forward layers, and family-specific residual/embedding operations. Hugging Face loaders translate supported configurations and tensor layouts into this model.

| Family | Implemented components |
|---|---|
| Llama 2, 3, 3.1 | Grouped-query attention and supported RoPE scaling variants |
| Llama 4 | Text decoder, interleaved attention, routed experts, vision tower and projector |
| Mistral, Mixtral | Sliding-window attention and Mixtral experts |
| Qwen 2, Qwen 3, Qwen3-MoE | Attention/projection conventions, Q/K normalization, and routed experts |
| Qwen 3.5 | Gated delta-net/attention hybrid text decoder and vision components |
| Gemma 1, 2, 3 | Embedding scaling, norm/softcap conventions, local/global attention; Gemma 3 vision |
| Gemma 3n | AltUp, LAuReL, per-layer inputs, sparse activations, shared KV layers, and MobileNet-v5 image encoder |
| Gemma 4 | Text configurations including routed experts; vision tower and projector |
| OLMo 3 | Decoder normalization and attention conventions |
| DeepSeek V2, V2-Lite, V3, V3.2 | MLA, routing/shared experts, balancing, and V3.2 sparse-indexer components |
| Kimi K2 | DeepSeek-derived decoder and checkpoint translation |
| GLM 4.5/5 configurations using `glm4_moe` | Attention, routing, and supported multi-token prediction depths |
| gpt-oss | Attention sinks, biased/clamped experts, and MXFP4 weight unpacking |
| LLaDA, Dream | Bidirectional masked-diffusion decoders |
| Diffusion Gemma | Text decoder and block-diffusion denoising components |

The [family reference](docs/reference/model-families.md) lists accepted model types, configuration restrictions, and numerical checks. Support for a family does not cover every later model with the same brand name. Released-weight coverage and export support vary by family.

### Diffusion and representation models

| Registry name | Architecture |
|---|---|
| `unet`, `unet_3d` | Image and video UNets |
| `uvit`, `simple_udit` | U-shaped transformers |
| `simple_dit` | Patch-based diffusion transformer |
| `simple_mmdit` | Dual-stream text/image transformer |
| `hierarchical_mmdit` | Multi-resolution dual-stream transformer |
| `hybrid_dit` | Transformer with S5 state-space blocks |
| `video_dit` | Factorized spatial and temporal attention |
| `jepa_encoder`, `jepa_video_encoder` | Image and video context encoders |
| `jepa_predictor` | Masked-representation predictor |

CLIP, T5, SigLIP, and model-specific vision/projector components provide conditioning and multimodal inputs. Autoencoders support latent-space training. These components do not yet cover all pretrained pipelines available in Diffusers.

## Training

A training run combines a model, an objective, a dataset, and an optimizer:

| Object | Responsibility |
|---|---|
| Flax model | Forward computation and variable structure |
| `Objective` | Initialization, loss, evaluation outputs, and optional previews |
| `Dataset` | Training and validation iterator factories |
| `Trainer` | Differentiation, optimizer updates, placement, checkpoints, and logging |
| `TrainState` | Parameters, optimizer state, random key, progress, and EMA variables |

### Language modeling

This decoder learns a repeating token sequence. Each row has 17 IDs: the model receives the first 16 and predicts the following 16. A real corpus uses the same layout, with IDs produced by its tokenizer.

```python
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew import Dataset, Trainer, metrics, models
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling, generate

row = np.resize(np.array([1, 2, 3, 4], dtype=np.int32), 17)
batch = {"text": np.tile(row, (8, 1))}
data = Dataset(train=lambda: itertools.repeat(batch),
               val=lambda: iter([batch]), records=8, batch=8)
model = models.build("causal_transformer", vocab_size=8, emb_features=32,
                     num_layers=1, num_heads=2, mlp_features=64, max_seq_len=32,
                     dtype=jnp.float32, attention_impl="xla")
objective = LMObjective(model, seq_len=16)
lm_state = Trainer(objective, optax.adam(0.01), key=jax.random.key(0)).fit(
    data, steps=40, log_every=10, eval_every=20, metrics=(metrics.perplexity(),))
continuation = generate(model, lm_state.params, jnp.array([[1, 2]], jnp.int32),
                        max_new_tokens=8, key=jax.random.key(1),
                        sampling=Sampling(temperature=0.0))
print(np.asarray(continuation.tokens))
```

The continuation includes the prompt and follows the learned pattern: `[[1, 2, 3, 4, 1, 2, 3, 4, 1, 2]]`. `temperature=0` selects the highest-probability token. Validation uses EMA weights, which can lag the live parameters during a short run.

For corpus training, `TokenWindows` reads tokenized binary files and `PackedTokens` packs documents with segment IDs and positions. `ChatMessages` renders chat templates and tracks token roles. Set `LMObjective(loss_role=Role.ASSISTANT)` to train only on assistant targets. See [language models](docs/concepts/language_models.md) for checkpoint loading and text tokenization.

### Preference optimization

Continue from `model` and `lm_state` above with a chosen and rejected response. Both start with the prompt `[1, 2]`; the masks restrict the loss to the two response tokens.

```python
import json

from dew.data import Loading, PreferencePairs
from dew.objectives.rl import DPOObjective

pair = {"chosen": [1, 2, 3, 4], "rejected": [1, 2, 6, 4],
        "chosen_mask": [0, 0, 1, 1], "rejected_mask": [0, 0, 1, 1]}
pairs = PreferencePairs(records=(json.dumps(pair),) * 8, seq_len=4,
                        loading=Loading(workers=0, threads=1, read_buffer=2)).load(batch=8)
dpo = DPOObjective(model, seq_len=3, beta=0.1, pretrained=lm_state.params)
dpo_state = Trainer(dpo, optax.adam(0.001), key=jax.random.key(2)).fit(
    pairs, steps=10, log_every=5)
```

`DPOObjective` keeps the starting policy as a frozen reference and optimizes the relative likelihood of the chosen response. `PreferencePairs.seq_len` is the full ID-row width; the objective scores one fewer position because of the next-token shift.

`FlowGRPOObjective` applies group-relative rewards to stochastic flow trajectories. `FlowRollout` samples image groups, evaluates rewards, and records the transition densities used by the clipped policy objective. See [FlowGRPO](docs/concepts/post_training.md) for a complete image-reward example.

For online reinforcement learning, `SampledRollout` generates groups of responses and calls a reward function. `GRPOObjective` trains on their advantages, old log probabilities, and response masks. [`recipes/chain.py`](recipes/chain.py) connects SFT, DPO, and GRPO stages. The [post-training guide](docs/concepts/post_training.md) covers reward callbacks and rollout settings.

### Custom Flax models

An objective can train an ordinary Linen module. This example learns `y = 2x + 1` with a single dense layer.

```python
import itertools

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew import Aux, Dataset, Field, InputSpec, Objective, Trainer

class Regression(Objective):
    model = nn.Dense(1)
    inputs = InputSpec(Field("x", (1,)))

    def init(self, key):
        return self.model.init(key, jnp.ones((1, 1)))

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        loss = jnp.mean((prediction - batch["y"]) ** 2)
        return loss, Aux(metrics={"mse": loss})

x = np.linspace(-1, 1, 32, dtype=np.float32).reshape(32, 1)
batch = {"x": x, "y": 2 * x + 1}
data = Dataset(train=lambda: itertools.repeat(batch),
               val=None, records=32, batch=32)
objective = Regression()
state = Trainer(objective, optax.sgd(0.1), key=jax.random.key(0)).fit(
    data, steps=100, log_every=25)
print(objective.model.apply(state.params, jnp.array([[0.0], [1.0]])))
```

The predictions approach `1` and `3`. `init` creates the variables, `loss` computes a differentiable scalar, and `Aux` supplies metrics and optional mutable-variable updates. You can pass a custom objective directly to `Trainer`; registration is only needed to construct it by name from a configuration.

The [objective guide](docs/concepts/objectives.md) also covers BatchNorm state and EMA selection.

### Evaluation and checkpoints

Pass `eval_every` and metrics to `fit` to score validation data. A tracker can also receive generated previews. Perplexity reduces token losses over the validation pass; FID accumulates population statistics before computing the final distance. The [evaluation guide](docs/guides/evaluation.md) explains metric selection and distributed behavior.

`Checkpoints` saves numerical state and data position through Orbax. Rebuild the run with the same checkpoint directory to continue it. `fit(steps=1200)` sets a total target: restoring step 1000 runs toward 1200, not 2200. Recipe configuration is separate; `RunConfig.save` writes `run.json`.

See [checkpointing and resume](docs/guides/checkpoints.md) for restore requirements, local checkpoints, and current recovery limitations.

## Diffusion and sampling

A `Process` combines a noise schedule, a prediction transform, and loss weighting. Presets build common combinations:

| Component | Options |
|---|---|
| Presets | EDM, Karras, Cosine, Flow, Sqrt; MDLM for masked-token diffusion |
| Prediction transforms | Noise, clean-sample, velocity, flow, Karras preconditioning |
| Weighting | Schedule weighting, P2-related weighting, Min-SNR |
| Solvers | DDPM, DDIM, Euler, Euler ancestral, Heun, RK4, MultiStepDPM |
| Guidance | Classifier-free guidance with optional time interval |
| Conditions | `InputSpec`/`Condition`, CLIP, T5, labels or custom encoders |

Training and inference can use different schedules, as in EDM's log-normal training distribution and Karras sampling grid. The sampler uses `jax.lax.scan`; changing the solver does not require another training loop. `MultiStepDPM` is Dew's sigma-space multistep integrator, not a DPM-Solver++ implementation.

`TextToImage` combines text encoding, denoising, and optional latent decoding. `TextToImage.from_run` reconstructs compatible runs from their saved recipe and checkpoint. See [diffusion](docs/guides/diffusion.md) for text conditioning, latent models, and sampling.

## Distributed training

`MeshSpec` describes the device topology; `Layout` maps model dimensions onto it. For example, `MeshSpec(fsdp=4)` splits eligible parameters and optimizer state over four devices. The remaining devices form the data-parallel axis. Use the same `Trainer` interface on one device or a mesh.

The mesh also supports expert, tensor, sequence, and stage axes. Sequence-parallel attention exchanges the keys/values needed by local queries. The experimental GPipe stage path partitions computation; parameters and optimizer state currently remain replicated across stages.

Models use configurable compute dtypes and hardware-dependent attention kernels: cuDNN on compatible NVIDIA GPU shapes, a Pallas TPU path, and XLA implementations for other configurations. Qwix supplies optional int8/fp8 computation, and MuonClip adds per-head QK clipping to Muon. Quantized weight loading is separate from quantized training.

Set `remat=True` on a decoder to recompute block activations during the backward pass. This reduces retained activation memory at the cost of additional computation; it composes with layer scanning.

Start with [distributed training](docs/concepts/distributed.md) and the [TPU guide](docs/tpu.md). [Benchmarks](docs/benchmarks.md) and [performance notes](docs/performance.md) record workload sizes, hardware, memory, and timing. Physical multi-host GPU/TPU qualification is still limited.

## Data and configuration

Dew's data specifications prepare batches through Grain. Token loaders support fixed windows and packed documents; chat, preference, and prompt loaders provide post-training data. Image/video sources include local files, Hugging Face datasets, TFDS, ArrayRecord shards, and URL streams.

`Loading` controls workers, read threads, and buffering. `Dataset` also accepts your own iterator factories, as in the opening example. See [data loading](docs/concepts/data.md) for transforms, deterministic randomness, batching, and iterator ownership.

The recipes expose dataclass configurations through tyro. `ModelConfig`, `OptimConfig`, and `TrainerConfig` collect model, optimizer, and run settings; task-specific configurations add diffusion or language-model options. `--help` shows the available command-line arguments:

```bash
python recipes/lm/train.py --help
python recipes/diffusion/train.py --help
python recipes/jepa/train.py --help
```

The [recipe guide](docs/recipes.md) includes a complete text-corpus preparation and training workflow.


## Installation

Install from the repository with [uv](https://docs.astral.sh/uv/getting-started/installation/):

Python 3.14 is recommended; Python 3.12 remains supported. TFDS training reads prepared ArrayRecords without TensorFlow. Builders that need TensorFlow run in a separate preparation environment, described in the installation guide.

```bash
git clone https://github.com/AshishKumar4/dew.git
cd dew
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e .
```

Install the JAX build for your hardware:

| Hardware | JAX package |
|---|---|
| CPU | Included in the base installation |
| NVIDIA GPU | `uv pip install -U "jax[cuda13]"` |
| Google TPU | `uv pip install -U "jax[tpu]"` |

Consult the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for driver requirements and other platforms. Optional Dew extras include `tfds`, `av`, `streaming`, `metrics`, `interop`, and `test`. The [installation guide](docs/installation.md) covers development dependencies and dataset preparation.

## Documentation and examples

- [Getting started](docs/getting-started.md): training a custom model.
- [Custom objectives](docs/concepts/objectives.md): loss functions and mutable model state.
- [Language models](docs/concepts/language_models.md): tokens, checkpoints, and generation.
- [Post-training](docs/concepts/post_training.md): SFT, DPO, GRPO, and rewards.
- [Diffusion](docs/guides/diffusion.md): image, video, conditioning, and latents.
- [Representation learning](docs/guides/representation-learning.md): JEPA encoders and predictors.
- [API reference](docs/reference/core-api.md): constructors, arguments, and state contracts.
- [Examples](examples/) and [recipes](docs/recipes.md): complete programs to adapt.

## Contributing and acknowledgements

Questions, bug reports, and contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes. For numerical issues, include the model configuration, dependency versions, dtype, hardware, and a small reproduction.

Dew grew out of [FlaxDiff](https://github.com/AshishKumar4/FlaxDiff). This project is partially supported by [Google TPU Research Cloud](https://sites.research.google/trc/about/). I would like to thank the Google Cloud TPU team for providing resources for the larger text-conditional experiments.

Dew builds on JAX, Flax, Optax, Grain, Orbax, tyro, and Weights & Biases. [References and attribution](docs/references.md) lists the papers and upstream implementations used by its models and algorithms.

Dew is licensed under [MIT](LICENSE). Adapted components, model weights, and datasets retain their applicable notices and licenses. If you use Dew in research, cite the repository and the papers for the models and methods you use.
