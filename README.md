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

Train a diffusion transformer on Oxford Flowers at 64×64, then generate a sample grid on an NVIDIA GPU.

Install Dew and CUDA JAX in a virtual environment:

```bash
git clone https://github.com/AshishKumar4/dew.git
cd dew
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e ".[tfds]" "jax[cuda12]"
```

Prepare the dataset once in a separate environment. TensorFlow is needed for this TFDS builder, but not for reading the prepared data during training.

```bash
uv venv --python 3.13 .venv-data
uv pip install --python .venv-data/bin/python \
    "tensorflow-datasets==4.9.10" "tensorflow==2.21.0" scipy
CUDA_VISIBLE_DEVICES="" .venv-data/bin/python - <<'PY'
from pathlib import Path
import tensorflow_datasets as tfds

builder = tfds.builder(
    "oxford_flowers102",
    data_dir=Path.home() / ".cache" / "dew" / "datasets",
)
builder.download_and_prepare(file_format="array_record")
print(builder.data_dir)
PY
```

An abridged [`examples/train_flowers.py`](examples/train_flowers.py); the file adds a command-line configuration and the sampling step:

```python
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from dew import Checkpoints, Field, InputSpec, Trainer
from dew.data import Loading, OxfordFlowers
from dew.diffusion.presets import EDM
from dew.objectives.diffusion import DiffusionObjective
from dew.nn.backbones import SimpleDiT


def train():
    data_path = Path.home() / ".cache/dew/datasets/oxford_flowers102/2.1.1"
    data = OxfordFlowers(
        path=str(data_path),
        split="train",
        image_size=64,
        val_batches=0,
        loading=Loading(workers=2, threads=2, read_buffer=16, worker_buffer=2),
    ).load(batch=16)

    model = SimpleDiT(
        patch_size=4,
        emb_features=128,
        num_layers=4,
        num_heads=4,
        dtype=jnp.bfloat16,
        attention_impl="auto",
    )
    objective = DiffusionObjective(
        model,
        EDM()(),
        InputSpec(Field("image", (64, 64, 3))),
    )
    trainer = Trainer(
        objective,
        optax.adamw(2e-4),
        key=jax.random.key(0),
        checkpoints=Checkpoints("runs/flowers64/checkpoints"),
    )
    return trainer.fit(data, steps=1000, log_every=20, checkpoint_every=200)


if __name__ == "__main__":
    state = train()
```

The `__main__` guard allows Grain to start its data-loading workers. `Field` describes one image; the dataset supplies batches of 16. The objective adds noise and constructs the denoising targets. The trainer runs optimization and checkpoints; `state.averaged` contains the EMA weights used for sampling.

Run the full script, which also saves `samples.png`:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda python examples/train_flowers.py \
    --data "$HOME/.cache/dew/datasets/oxford_flowers102/2.1.1" \
    --steps 1000
```

Use `--steps 20` for a short run, or increase `--steps` to train longer.

[`examples/train_diffusion.py`](examples/train_diffusion.py) adds pretrained CLIP text conditioning. For an offline run without a dataset download, [`examples/readme_demo.py`](examples/readme_demo.py) demonstrates language modeling, checkpoint continuation, DPO, and flow matching.

### Change the training setup

Pass an Optax optimizer to `Trainer`. To use momentum SGD in the Flowers script, replace its optimizer argument:

```python
optimizer = optax.sgd(learning_rate=1e-2, momentum=0.9)
trainer = Trainer(
    objective,
    optimizer,
    key=jax.random.key(0),
    checkpoints=Checkpoints("runs/flowers-sgd/checkpoints"),
)
```

Set `ema_decay` when you construct the objective. A value closer to 1 averages weights over more updates:

```python
process = EDM()()
objective = DiffusionObjective(
    model,
    process,
    InputSpec(Field("image", (64, 64, 3))),
    ema_decay=0.999,
)
```

After training, sample from the averaged weights with `state.averaged`. Use `state.params` instead to sample from the latest weights:

```python
from dew import sample
from dew.sampling import Heun

denoise = process.denoiser(model, state.averaged, conditions={})
images = sample(
    denoise,
    process.noise(jax.random.key(1), (8, 64, 64, 3)),
    solver=Heun(),
    steps=40,
    key=jax.random.key(2),
)
```

Set `ema_decay=None` to train without an averaged copy, then sample with `state.params`.

Choose the Flowers model's computation precision with `SimpleDiT(dtype=jnp.bfloat16)` or `dtype=jnp.float32`. Its master weights and optimizer state stay fp32.

For int8 quantization-aware training, install Qwix with `uv pip install qwix` and wrap the model **before** constructing the objective:

```python
from dew.training.quantization import Quantization, apply_quantization

model = apply_quantization(model, Quantization(dtype="int8"))
```

Qwix quantizes the trunk matmuls while retaining fp32 master weights. This changes the training arithmetic; it is a separate choice from bf16 computation.

Import model classes directly when writing Python: `from dew.nn.backbones import SimpleDiT, CausalTransformer`. The examples below also use `models.build(name, **fields)` for models selected by configuration; both construct the same Flax classes.

## Features

- **Language modeling:** autoregressive pretraining, packed documents, assistant-only SFT, DPO, and GRPO with callable rewards.
- **Diffusion:** image and video denoising, rectified flow, latent diffusion, masked-token diffusion, classifier-free guidance, and interchangeable schedules and solvers.
- **Representation learning:** I-JEPA and V-JEPA with context/target encoders, predictors, block masking, and linear/kNN probes.
- **Training systems:** data/FSDP/expert/tensor/sequence parallelism, layer scans, gradient accumulation, mixed precision, asynchronous checkpoints, and configurable EMA.
- **Interoperability:** supported Hugging Face configuration and weight translation, safetensors export, CLIP/T5 conditioning, and VAE components.
- **Evaluation:** perplexity, FID, CLIP score, PSNR, SSIM, representation diagnostics, generated previews, and Weights & Biases tracking.

The supplied DPO and GRPO objectives use the same trainer as pretraining. PPO-related estimators and loss functions are available separately in `dew.rl`; they are not a complete PPO training system. Full Transformers/Diffusers/MaxText coverage, serving, and multi-turn agent training remain development goals.

## Models

### Supported text models

These model configurations have checkpoint loading, training, and text generation paths in Dew. Text checkpoints are listed separately from multimodal models.

| Model or family | Supported |
|---|---|
| Llama 2, Llama 3, Llama 3.1 | Yes |
| Mistral, Mixtral | Yes |
| Qwen 2, Qwen 3, Qwen3-MoE | Yes |
| Gemma 1, Gemma 2, Gemma 3 text | Yes |
| OLMo 3 | Yes |
| DeepSeek V2 and V2-Lite (`deepseek_v2`) | Yes |
| Kimi K2 text | Yes |
| GLM checkpoints using `glm4_moe` | Yes |
| gpt-oss | Yes |
| LLaDA, Dream | Yes, with masked-diffusion training and sampling |

### Not yet supported as complete models

| Model or family | Supported | Missing work |
|---|---|---|
| Gemma 3 multimodal | No | Complete processor, loading, training, and image-conditioned generation workflow |
| Gemma 3n, Gemma 4 multimodal | No | Complete image/audio model workflows |
| Llama 4 multimodal | No | Complete image-conditioned training and generation workflow |
| Qwen 3.5 full model | No | Multimodal workflows and complete MTP handling |
| DeepSeek V3 and V3.2 full training | No | Complete MTP handling and V3.2 indexer-training objective |
| Diffusion Gemma | No | Complete pretrained canvas-generation and training workflows |
| DeepSeek V4, Qwen 3.8, GLM 5.3, Kimi K3, Muse Spark | No | Native model integrations |
| Complete SDXL, SD3, Flux, and other Diffusers pretrained pipelines | No | Pipeline-specific model loading and task workflows |

The [model reference](docs/reference/model-families.md) records exact checkpoint types and configuration requirements. Work on the unsupported models continues; tested layers alone do not put a model in the supported list.

### Diffusion and representation models

These Dew architectures can be trained from scratch.

| Model | Registry name | Training |
|---|---|---|
| Image/video UNet | `unet`, `unet_3d` | Diffusion and flow matching |
| U-shaped transformers | `uvit`, `simple_udit` | Diffusion and flow matching |
| DiT | `simple_dit` | Diffusion and flow matching |
| Dual-stream MMDiT | `simple_mmdit`, `hierarchical_mmdit` | Text-conditioned diffusion |
| S5/transformer hybrid | `hybrid_dit` | Diffusion |
| Video DiT | `video_dit` | Video diffusion |
| I-JEPA / V-JEPA | `jepa_encoder`, `jepa_video_encoder`, `jepa_predictor` | Masked representation prediction |

CLIP and T5 text encoders and VAE interfaces provide conditioning and latent-space training.

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

### Supervised fine-tuning

Fine-tune the trained decoder above by marking which targets belong to the assistant. Here the first four tokens provide the prompt; the remaining tokens are the response. `ChatMessages` produces this role column from chat templates when reading conversation data.

```python
from dew.data.chat import Role

roles = np.full(batch["text"].shape, Role.USER, dtype=np.int8)
roles[:, 4:] = Role.ASSISTANT
sft_batch = {**batch, "text_roles": roles}
sft_data = Dataset(
    train=lambda: itertools.repeat(sft_batch),
    val=None,
    records=8,
    batch=8,
)
sft_objective = LMObjective(
    model,
    seq_len=16,
    pretrained=lm_state.params,
    loss_role=Role.ASSISTANT,
    ema_decay=None,
)
sft_state = Trainer(
    sft_objective,
    optax.adamw(1e-3),
    key=jax.random.key(2),
).fit(sft_data, steps=20, log_every=10)
```

The loss counts assistant targets after the next-token shift. Prompt tokens still provide context. For conversation files, `ChatMessages` also preserves tool calls, tool responses, and tool schemas.

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

### Reinforcement learning with a reward function

This example continues the same decoder with a reward for choosing the expected next token. A task verifier can replace the reward function. The prompt batch uses the same numeric layout as `Prompts`, including UTF-8 reward metadata.

```python
from dew.objectives.rl import GRPOObjective, SampledRollout

prompt_batch = {
    "prompt": np.tile(np.array([1, 2], dtype=np.int32), (8, 1)),
    "prompt_length": np.full(8, 2, dtype=np.int32),
    "data_source": np.tile(
        np.frombuffer(b"pattern", dtype=np.uint8).astype(np.int32),
        (8, 1),
    ),
    "ground_truth": np.full((8, 1), ord("3"), dtype=np.int32),
    "extra_info": np.zeros((8, 0), dtype=np.int32),
}


def reward(data_source, completion, ground_truth, extra_info):
    return float(completion.split()[:1] == [ground_truth])


rl_data = Dataset(
    train=lambda: itertools.repeat(prompt_batch),
    val=None,
    records=8,
    batch=8,
)
rl_objective = GRPOObjective(
    model,
    seq_len=5,
    beta=0.01,
    pretrained=lm_state.params,
)
rollout = SampledRollout(
    rl_objective,
    reward=reward,
    groups=4,
    max_new_tokens=4,
    sampling=Sampling(temperature=2.0, top_k=4),
)
rl_state = Trainer(
    rl_objective,
    optax.adamw(1e-4),
    key=jax.random.key(3),
    rollout=rollout,
).fit(rl_data, steps=20, log_every=10)
```

Each prompt produces four responses. Their relative rewards determine the advantages. GRPO uses a clipped policy objective and an optional reference KL term; `beta` sets its coefficient. For language tasks, supply a tokenizer's decoding function through `SampledRollout.decode`.

### Diffusion language models

Masked diffusion trains a bidirectional decoder to recover corrupted tokens. Unlike autoregressive training, each input row contains exactly `seq_len` tokens; there is no next-token shift. Reserve a vocabulary ID for the mask.

```python
from dew.diffusion.discrete import MDLM
from dew.objectives import Step
from dew.objectives.diffusion import MaskedDiffusionObjective

masked_batch = {"text": batch["text"][:, :16]}
masked_data = Dataset(
    train=lambda: itertools.repeat(masked_batch),
    val=None,
    records=8,
    batch=8,
)
masked_model = models.build(
    "causal_transformer",
    vocab_size=8,
    emb_features=32,
    num_layers=1,
    num_heads=2,
    mlp_features=64,
    max_seq_len=16,
    causal=False,
)
masked_objective = MaskedDiffusionObjective(
    masked_model,
    MDLM(mask_id=7)(),
    seq_len=16,
    ema_decay=0.9,
    samples=4,
    steps=16,
)
masked_state = Trainer(
    masked_objective,
    optax.adamw(3e-3),
    key=jax.random.key(4),
).fit(masked_data, steps=100, log_every=50)
generated = masked_objective.preview(
    masked_state.params,
    masked_batch,
    Step(
        step=masked_state.microstep,
        key=jax.random.key(5),
        ema=masked_state.averaged,
    ),
)
print(np.asarray(generated.tokens))
```

LLaDA and Dream use this masked-token path. Diffusion Gemma uses a different canvas/self-conditioning process.

### JEPA representation learning

Train an image encoder by predicting masked patch representations. This uses the Flowers data prepared in the quickstart, with 8×8 patches and a smaller predictor.

```python
from pathlib import Path

import jax
import optax

from dew import Field, Trainer, metrics, models
from dew.data import Loading, OxfordFlowers
from dew.objectives.jepa import JepaObjective, multi_block_mask


def train_jepa():
    data = OxfordFlowers(
        path=str(Path.home() / ".cache/dew/datasets/oxford_flowers102/2.1.1"),
        split="train",
        image_size=64,
        val_batches=2,
        loading=Loading(workers=0, threads=1, read_buffer=2),
    ).load(batch=16)
    encoder = models.build(
        "jepa_encoder",
        patch_size=8,
        emb_features=64,
        num_layers=2,
        num_heads=4,
    )
    predictor = models.build(
        "jepa_predictor",
        grid=(8, 8),
        emb_features=64,
        predictor_features=32,
        num_layers=1,
        num_heads=4,
    )
    objective = JepaObjective(
        encoder,
        predictor,
        mask=multi_block_mask((8, 8), num_targets=1, scale=(0.25, 0.25)),
        sample=Field("image", (64, 64, 3)),
        momentum_steps=20,
    )
    return Trainer(
        objective,
        optax.adamw(1e-3),
        key=jax.random.key(0),
    ).fit(
        data,
        steps=20,
        log_every=10,
        eval_every=20,
        metrics=(metrics.knn_probe(102),),
    )


if __name__ == "__main__":
    jepa_state = train_jepa()
```

The target encoder follows an EMA of the context encoder. The loss compares predicted and target representations, rather than reconstructing pixels. The kNN metric is a half-batch diagnostic; use a larger labeled evaluation for representation quality. `jepa_video_encoder` extends the same objective to video clips.

### Loading pretrained weights

Load a supported checkpoint from a Hub repository or local directory. This example downloads Qwen3-0.6B and its tokenizer on first use; the released-weight download has not been run for this README.

```python
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from dew.interop import load_pretrained_decoder
from dew.sampling import Sampling, generate

checkpoint = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
pretrained_model, variables, config = load_pretrained_decoder(
    checkpoint,
    dtype="bfloat16",
    max_seq_len=512,
)
prompt = jnp.asarray(
    [tokenizer.encode("Explain gradient accumulation in one paragraph.")],
    dtype=jnp.int32,
)
result = generate(
    pretrained_model,
    variables,
    prompt,
    max_new_tokens=128,
    key=jax.random.key(0),
    sampling=Sampling(temperature=0.8, top_k=40, eos_id=tokenizer.eos_token_id),
)
print(tokenizer.decode(result.tokens[0], skip_special_tokens=True))
```

Pass `pretrained=variables` to `LMObjective` or a post-training objective to continue from those weights. Tokenize training data with the checkpoint's own tokenizer. For pretrained diffusion runs, `TextToImage.from_run` reads the saved recipe and checkpoint; CLIP/T5 conditioning and a configured VAE are restored with the run.

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

Pass `eval_every` and metrics to `fit` to score validation data. Set `preview=True` when you also want generated previews. Perplexity reduces token losses over the validation pass; FID accumulates population statistics before computing the final distance.

`Checkpoints` saves numerical state and data position through Orbax. Rebuild the run with the same checkpoint directory to continue it. `fit(steps=1200)` sets a total target: restoring step 1000 runs toward 1200, not 2200. Recipe configuration is separate; `RunConfig.save` writes `run.json`.

See [checkpointing and resume](docs/guides/checkpoints.md) for restore requirements, local checkpoints, and current recovery limitations.

### Standalone evaluation and local reports

`evaluate` scores trained variables without an optimizer. It returns metric values and optional previews. `LocalTracker` writes scalar history, artifacts, and plots; it needs no W&B account or installation. Install `dew-ml[plots]` for Matplotlib output.

```python
import itertools

import jax
import numpy as np
import optax

from dew import Dataset, LocalTracker, Trainer, evaluate, metrics, models
from dew.objectives.lm import LMObjective, Samples
from dew.sampling import Sampling

row = np.resize(np.array([1, 2, 3, 4], dtype=np.int32), 17)
batch = {"text": np.tile(row, (8, 1))}
data = Dataset(
    train=lambda: itertools.repeat(batch),
    val=lambda: iter([batch]),
    records=8,
    batch=8,
)
model = models.build(
    "causal_transformer",
    vocab_size=8,
    emb_features=32,
    num_layers=1,
    num_heads=2,
    mlp_features=64,
    max_seq_len=32,
)
objective = LMObjective(
    model,
    seq_len=16,
    samples=Samples([1, 2], 8, sampling=Sampling(temperature=0)),
)
with LocalTracker("runs/lm-report", plots=True) as tracker:
    state = Trainer(
        objective,
        optax.adam(0.01),
        key=jax.random.key(0),
        tracker=tracker,
    ).fit(data, steps=40, log_every=10)
    result = evaluate(
        objective,
        state.params,
        data.val,
        metrics=(metrics.perplexity(),),
        key=jax.random.key(1),
        step=int(state.step),
        preview=True,
    )
    tracker.log(result.scalars, step=result.step)
    for preview in result.previews:
        tracker.artifact(preview, step=result.step)
    print(result.scores)
```

This run reports perplexity around 1.002 and saves the training-loss curve, scalar journal, and generated text under `runs/lm-report`. Plots render when the tracker closes, rather than on every training step. Use `plots=False` for scalar/artifact recording only, or call `tracker.plot()` explicitly.

For W&B, install `dew-ml[wandb]` and pass a `WandbTracker`. `Trackers` sends the same reports to multiple backends:

```python
from dew import LocalTracker, Trackers, WandbTracker

tracker = Trackers(
    LocalTracker("runs/experiment/tracking"),
    WandbTracker(project="dew-experiments", offline=True),
)
```

Use this tracker in the same `with` block and `Trainer` call above. A custom backend implements `log`, `artifact`, and `close`. Run configuration, progress, checkpoint requests, profiler windows, and failures use typed reporting records. `Profile` enables detailed device traces.

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

### Multiple hosts

Use the same script on each host. The example below uses two hosts with two visible GPUs each and shards model state over all four devices. Both hosts must read the same token files and checkpoint directory.

Prepare byte-token data from your corpus and place it on shared storage:

```bash
python tools/tokenize_text.py \
    --input corpus.txt \
    --out /shared/tokens \
    --tokenizer byte \
    --val-fraction 0.01
```

Save as `train_multihost.py`. Initialize the process group before constructing device arrays.

```python
import os

import jax
import optax


def main():
    jax.distributed.initialize(
        coordinator_address=os.environ["DEW_COORDINATOR"],
        num_processes=int(os.environ["DEW_WORLD_SIZE"]),
        process_id=int(os.environ["DEW_RANK"]),
    )
    try:
        from dew import Checkpoints, MeshSpec, Trainer, models
        from dew.data import Loading, TokenWindows
        from dew.objectives.lm import LMObjective

        data = TokenWindows(
            path=os.environ["DEW_TOKEN_DIR"],
            seq_len=128,
            loading=Loading(workers=0, threads=1, read_buffer=2),
        ).load(batch=16)
        model = models.build(
            "causal_transformer",
            vocab_size=256,
            emb_features=128,
            num_layers=2,
            num_heads=4,
            mlp_features=256,
            max_seq_len=128,
            dtype="bfloat16",
        )
        trainer = Trainer(
            LMObjective(model, seq_len=128),
            optax.adamw(3e-4),
            key=jax.random.key(0),
            mesh=MeshSpec(fsdp=4),
            checkpoints=Checkpoints(os.environ["DEW_CHECKPOINT_DIR"]),
        )
        state = trainer.fit(
            data,
            steps=int(os.environ.get("DEW_STEPS", "1000")),
            log_every=20,
            checkpoint_every=200,
        )
        print(f"Process {jax.process_index()}: {int(state.updates)} updates")
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
```

Set these variables on both hosts, changing `DEW_RANK` to `1` on the second host and using the first host's reachable address for the coordinator:

```bash
export DEW_COORDINATOR="10.0.0.1:43217"
export DEW_WORLD_SIZE=2
export DEW_RANK=0
export DEW_TOKEN_DIR=/shared/tokens
export DEW_CHECKPOINT_DIR=/shared/runs/lm
CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda python train_multihost.py
```

`batch=16` is global, so each process reads eight rows. The program completed with two local JAX processes and four simulated CPU devices; the two-host GPU launch has not been exercised.

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
| NVIDIA GPU | `uv pip install -U "jax[cuda12]"` (or `cuda13` for CUDA 13 drivers) |
| Google TPU | `uv pip install -U "jax[tpu]"` |

Consult the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for driver requirements and other platforms. Optional Dew extras include `tfds`, `av`, `streaming`, `metrics`, `interop`, `plots`, `wandb`, and `test`. The [installation guide](docs/installation.md) covers development dependencies and dataset preparation.

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
