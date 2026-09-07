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

APIs and checkpoint formats can change before 1.0. [Models](#models) lists the configurations whose whole workflow runs.

## Contents

- [Getting started](#getting-started)
- [Features](#features)
- [Models](#models)
- [Training](#training)
- [Diffusion and sampling](#diffusion-and-sampling)
- [Generating and serving](#generating-and-serving)
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

`SimpleDiT(dtype=jnp.bfloat16)` sets the computation dtype; master weights and
optimizer state stay fp32, so the optimizer still accumulates in full
precision.

For int8 quantization-aware training, install Qwix with `uv pip install qwix`
and wrap the model **before** constructing the objective:

```python
from dew.training.quantization import Quantization, apply_quantization

model = apply_quantization(model, Quantization(dtype="int8"))
```

Qwix rewrites the model's matmuls to quantize their operands and keeps fp32
master weights, so the training arithmetic changes and needs its own accuracy
check. On a GPU this works for the decoders, whose trunk is all matmuls. The
Flowers DiT also has a patch-embedding convolution, and cuDNN has no int8
lowering for it, so quantizing that model raises `Can't lower one or more
integer convolutions to idioms supported by CuDNN`; it trains quantized on CPU.

Import model classes directly when writing Python: `from dew.nn.backbones import SimpleDiT, CausalTransformer`. The examples below also use `models.build(name, **fields)` for models selected by configuration; both construct the same Flax classes.

## Features

- **Language modeling:** autoregressive pretraining, packed documents, assistant-only SFT, DPO, and GRPO with callable rewards.
- **Diffusion:** image and video denoising, rectified flow, latent diffusion, masked-token diffusion, classifier-free guidance, and interchangeable schedules and solvers.
- **Representation learning:** I-JEPA and V-JEPA with context/target encoders, predictors, block masking, and linear/kNN probes.
- **Training systems:** data/FSDP/expert/tensor/sequence parallelism, layer scans, gradient accumulation, mixed precision, asynchronous checkpoints, and configurable EMA.
- **Interoperability:** supported Hugging Face configuration and weight translation, safetensors export, CLIP/T5 conditioning, and VAE components.
- **Evaluation:** perplexity, FID, CLIP score, PSNR, SSIM, representation diagnostics, generated previews, and Weights & Biases tracking.

The DPO and GRPO objectives run on the same trainer as pretraining. `dew.rl` holds PPO's advantage estimators and loss terms as separate functions to build a policy loop from.

## Models

A configuration is listed as supported when one checkpoint of it runs the whole
workflow: `load_pretrained` builds the model, its processor prepares the media
the model accepts, `Trainer` commits an optimizer update, generation decodes
from the trained weights, and the export reloads into the same model. A
translated config or a parity-checked layer does not put a model here.

### Text decoders

`load_pretrained` reads the checkpoint's `config.json` and builds a
`CausalTransformer`. Training uses `LMObjective`, generation uses
`dew.sampling.generate` or `Pretrained.text_generation()`, and
`Pretrained.save` writes `config.json`, `model.safetensors` and
`generation_config.json` back in the Hugging Face layout.

| Model or family | `model_type` |
|---|---|
| Llama 2, Llama 3, Llama 3.1 | `llama` |
| Mistral | `mistral` |
| Qwen 2, Qwen 3 | `qwen2`, `qwen3` |
| Qwen 3.5 text | `qwen3_5_text` |
| Gemma 1, Gemma 2, Gemma 3 text | `gemma`, `gemma2`, `gemma3_text` |
| Gemma 3n text, Gemma 4 text | `gemma3n_text`, `gemma4_text` |
| OLMo 3 | `olmo3` |
| gpt-oss | `gpt_oss` |

### Native multimodal models

`load_pretrained` returns the model, the checkpoint's own processor, and the
weights. The processor turns text and raw media into `ModelInputs`, which
`LMObjective`, `Trainer` and cached generation take unchanged. The export
carries the processor and tokenizer files beside the weights.

| Model or family | `model_type` | Media |
|---|---|---|
| Gemma 3 | `gemma3` | Images |
| Gemma 4 | `gemma4` | Images; waveforms |
| Qwen 3.5 | `qwen3_5` | Images, with M-RoPE positions |
| Llama 4 | `llama4` | Tiled images |

Gemma 4's images and waveforms live in separate checkpoints: a vision checkpoint
carries no audio tower, and asking its processor for waveforms raises `this
source has no audio tower`. The audio towers run on the reference kernel in
float32; the GPU audio path is waiting on its own gate.

### Block-diffusion decoders

| Model | `model_type` | Workflow |
|---|---|---|
| Diffusion Gemma | `diffusion_gemma` | Canvas generation and the published Google SFT recipe |

`BlockDiffusionObjective` trains the canvas loss from the loaded weights and
`Pretrained.block_generation()` decodes canvases. The objective makes
`layer_scalar` trainable, so the export goes through the objective's model:
`replace(loaded, model=objective.model).save(directory, variables=state.params)`.

### Incomplete configurations

These load and generate; the named piece is what keeps them off the list above.

| Model or family | `model_type` | Missing piece |
|---|---|---|
| Mixtral, Qwen3-MoE | `mixtral`, `qwen3_moe` | No routed-expert export writer |
| GLM 4.5/5 | `glm4_moe` | No partial-rotary export writer |
| DeepSeek V2, V3, V3.2 | `deepseek_v2`, `deepseek_v3`, `deepseek_v32` | No MLA export writer |
| Llama 4 text | `llama4_text` | No export writer |
| LLaDA | `llada` | No masked-objective pretrained seam |
| Dream | `dream`, `Dream` | Export writer, training seam |
| Kimi K2 | `kimi_k2` | Config translation only, no weights |
| Gemma 3n multimodal | `gemma3n` | Vision tower sharding axes undeclared |

DeepSeek's MLA layers need `attention_impl="reference"`; `auto` rejects their
query and value head widths. Gemma 3n's MobileNet-v5 tower leaves 132 of its
707 parameter leaves without declared sharding axes, so `Trainer` refuses the
model even though its gradients are finite. `MaskedDiffusionObjective` takes no
`pretrained` argument, so LLaDA and Dream train from a fresh init rather than
from their released weights.

No processor takes video: `Processor.__call__` accepts `text`, `images` and
`audio`. Diffusers pipelines such as SDXL, SD3 and Flux have no loader.

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

Load a supported checkpoint from a Hub repository or local directory. This block needs a 1.2 GB download of Qwen3-0.6B and its tokenizer, so it is the one block below that was not run for this README.

```python
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from dew.interop import load_pretrained
from dew.sampling import Sampling, generate

checkpoint = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
pretrained = load_pretrained(
    checkpoint,
    dtype="bfloat16",
    max_seq_len=512,
)
prompt = jnp.asarray(
    [tokenizer.encode("Explain gradient accumulation in one paragraph.")],
    dtype=jnp.int32,
)
result = generate(
    pretrained.model,
    pretrained.variables,
    prompt,
    max_new_tokens=128,
    key=jax.random.key(0),
    sampling=Sampling(temperature=0.8, top_k=40, eos_id=tokenizer.eos_token_id),
)
print(tokenizer.decode(result.tokens[0], skip_special_tokens=True))
```

Pass `pretrained=pretrained.variables` to `LMObjective` or a post-training objective to continue from those weights, and tokenize the training data with the checkpoint's own tokenizer. [Generating and serving](#generating-and-serving) draws from the result and exports it.

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

Training and inference can use different schedules, as in EDM's log-normal training distribution and Karras sampling grid. The sampler uses `jax.lax.scan`, so changing the solver reuses the same trained weights. `MultiStepDPM` integrates in sigma space and keeps the previous denoiser outputs to raise the order of each step.

`TextToImage` combines text encoding, denoising, and optional latent decoding. See [diffusion](docs/guides/diffusion.md) for text conditioning, latent models, and sampling.

## Generating and serving

`dew.inference` binds weights to a generation task. `TextGeneration` decodes
tokens, `BlockGeneration` decodes Diffusion Gemma canvases, and `TextToImage`
denoises images. Each one is frozen around the weights it was built with, and
`bind` returns a new task over another snapshot.

### Drawing from a trained language model

`LMObjective.policy` hands back the `TextGeneration` bound to the parameters
you pass it, which is the same task the GRPO rollout samples with. This
continues the decoder trained in [language modeling](#language-modeling):

```python
from dew.inference import TextGeneration

task = objective.policy(lm_state.params, Sampling(temperature=0.0))
drawn = task([[1, 2]], 8, key=jax.random.key(1))
print(np.asarray(drawn.tokens), np.asarray(drawn.lengths))

same = TextGeneration(model, lm_state.params, sampling=Sampling(temperature=0.0))
print(np.array_equal(np.asarray(same([[1, 2]], 8, key=jax.random.key(1)).tokens),
                     np.asarray(drawn.tokens)))
```

This prints `[[1 2 3 4 1 2 3 4 1 2]] [8]` and `True`: the constructor and
`policy` build the same task. `lengths` counts the 8 response actions, not the
2 prompt tokens the row also carries. `Generation` also returns `terminated`,
and the `behavior_log_probs` and `raw_log_probs` a policy ratio needs.

A task built by `Pretrained.text_generation()` carries the checkpoint's
processor, so it accepts strings and `decode` returns text. Without a
processor the task takes token rows or `ModelInputs`.

### Drawing from a trained diffusion run

`TextToImage.from_run` reads a run directory: the `run.json` a
`DiffusionRunConfig` wrote and the weights of its latest checkpoint, with the
EMA copy merged over the live parameters. Nothing about the model has to be
restated at generation time.

```python
from pathlib import Path

import jax
import numpy as np
import optax

from dew import Checkpoints, Trainer
from dew.config import ModelConfig, TrainerConfig
from dew.data import Loading, OxfordFlowers
from dew.diffusion import presets
from dew.inference import TextToImage
from dew.objectives.diffusion import DiffusionRunConfig
from dew.sampling import Heun

run = Path("runs/flowers-run")
config = DiffusionRunConfig(
    model=ModelConfig("simple_dit", {"patch_size": 4, "emb_features": 128,
                                     "num_layers": 4, "num_heads": 4}),
    data=OxfordFlowers(
        path=str(Path.home() / ".cache/dew/datasets/oxford_flowers102/2.1.1"),
        image_size=64,
        val_batches=0,
        loading=Loading(workers=0, threads=1, read_buffer=2),
    ),
    trainer=TrainerConfig(checkpoint_dir=str(run), batch_size=16, steps=20, keep=1),
    preset=presets.EDM(),
    text=None,
)


def main():
    objective = config.build()
    checkpoints = Checkpoints(str(run), keep=1)
    state = Trainer(objective, optax.adamw(2e-4), key=jax.random.key(0),
                    checkpoints=checkpoints).fit(
        config.data.load(batch=16, tokenize=objective.inputs.tokenize),
        steps=20, log_every=20, checkpoint_every=20)
    checkpoints.wait()
    config.save(str(run))
    print(sorted(path.name for path in run.iterdir()))

    task = TextToImage.from_run(str(run))
    images = task(["a flower", "another flower"], steps=20, sampler=Heun(),
                  key=jax.random.key(1))
    print(np.asarray(images).shape, int(state.updates))


if __name__ == "__main__":
    main()
```

The run directory then holds `['20', 'run.json']` and the task draws
`(2, 64, 64, 3)` images clipped to `[-1, 1]`. `text=None` trains
unconditionally, so the prompt list only decides how many images to draw; a
`TextCondition` run encodes the prompts through the encoder it names.
`config.data.load` takes `tokenize=objective.inputs.tokenize` because the
objective's conditions are what read the dataset's captions, and
`Loading(workers=0)` keeps that caption reader in the training process, where
its shutdown is part of the run.

### Exporting a decoder and serving it

`save_pretrained_decoder` writes the Hugging Face layout, and hands the
tokenizer the run trained with the job of saving its own files. The result is a
directory another runtime can read without anything being copied into it.

Tokenize a corpus with the tokenizer the export will carry, so the token ids
and the exported vocabulary are the same one. `tiny-tools` is the small
byte-level BPE tokenizer committed for the tests, which keeps this runnable in
a fresh checkout:

```bash
python tools/tokenize_text.py \
    --input corpus.txt \
    --out runs/tokens \
    --tokenizer tests/fixtures/tokenizers/tiny-tools
```

```python
import json
from pathlib import Path

import jax
import numpy as np
import optax

from dew import Trainer, models
from dew.data import HFTokenizer, Loading, TokenWindows
from dew.interop import load_pretrained, save_pretrained_decoder
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling

tokens = Path("runs/tokens")
export = Path("runs/dew-decoder")
meta = json.loads((tokens / "meta.json").read_text())
tokenizer = HFTokenizer(meta["tokenizer"])

data = TokenWindows(path=str(tokens), seq_len=128,
                    loading=Loading(workers=0, threads=1, read_buffer=2)
                    ).load(batch=16)
model = models.build("causal_transformer", vocab_size=meta["vocab_size"],
                     emb_features=128, num_layers=4, num_heads=4, num_kv_heads=2,
                     mlp_features=256, max_seq_len=128, dtype="float32",
                     qk_norm=False, tie_embeddings=False)
state = Trainer(LMObjective(model, seq_len=128, ema_decay=None),
                optax.adamw(3e-3), key=jax.random.key(0)).fit(
    data, steps=400, log_every=200)

save_pretrained_decoder(model, state.params, str(export), tokenizer=tokenizer)
print(sorted(path.name for path in export.iterdir()))

task = load_pretrained(str(export), dtype="float32").text_generation()
drawn = task("The trainer", 12, key=jax.random.key(1),
             sampling=Sampling(temperature=0.0))
print(task.decode(drawn))
```

The export directory holds the weights, the config both runtimes read, and the
tokenizer's own files:

```
runs/dew-decoder/
├── chat_template.jinja
├── config.json
├── generation_config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

`qk_norm=False` and `tie_embeddings=False` make the exporter write a `llama`
config, which is the architecture llama.cpp converts; the default decoder
exports as `qwen3`, and Ollama 0.32.9 answers `unsupported architecture
"Qwen3ForCausalLM"`. `ollama create` reads the export through the running
daemon, so start `ollama serve` first, then convert the directory with a
Modelfile:

```bash
cat > Modelfile <<'EOF'
FROM runs/dew-decoder
TEMPLATE "{{ .Prompt }}"
PARAMETER num_gpu 0
PARAMETER num_ctx 128
EOF
ollama create dew-decoder -f Modelfile
```

`num_gpu 0` keeps the runner on the CPU, and `TEMPLATE "{{ .Prompt }}"` passes
the prompt through unchanged, which is what makes the served draw comparable to
the local one.

`OllamaCompletion` and `OpenAICompletion` wrap the vendors' own SDKs, which
the caller constructs and owns. Passing `Sampling` is how you ask for the
checkpoint's own policy: the client turns it into backend options and cancels
the daemon's repetition penalty and truncations on the way.

```python
import ollama

from dew.inference import OllamaCompletion
from dew.sampling import Sampling

client = OllamaCompletion("dew-decoder", ollama.Client(host="http://127.0.0.1:11434"))
served = client("The trainer", 12, sampling=Sampling(temperature=0.0), seed=0, raw=True)
print(served.texts, served.finish_reasons)
```

Asked for its own argmax, the daemon reproduces Dew's greedy draw token for
token, so `served.texts[0] == task.decode(drawn)[0]`. `Completion` also carries
`token_counts`, a `usage` record, and the SDK's own responses under
`responses`. For vLLM, serve the same directory and name the provider, which
unlocks the sampling controls vLLM accepts beyond the OpenAI schema:

```bash
vllm serve runs/dew-decoder --served-model-name dew-decoder
```

```python
import openai

from dew.inference import OpenAICompletion

client = OpenAICompletion(
    "dew-decoder",
    openai.OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none"),
    provider="vllm",
)
```

Without `provider="vllm"` the client refuses `top_k`, `min_p` and `eos_id`
rather than dropping them, because generic OpenAI endpoints do not take them.

## Distributed training

`MeshSpec` describes the device topology; `Layout` maps model dimensions onto it. For example, `MeshSpec(fsdp=4)` splits eligible parameters and optimizer state over four devices. The remaining devices form the data-parallel axis. Use the same `Trainer` interface on one device or a mesh.

The mesh also supports expert, tensor, sequence, and stage axes. Sequence-parallel attention exchanges the keys and values that local queries need. The GPipe stage axis partitions the execution view; parameters and optimizer state stay replicated across stages, so a stage split saves activation memory rather than parameter memory.

Models use configurable compute dtypes and hardware-dependent attention kernels: cuDNN on compatible NVIDIA GPU shapes, a Pallas TPU path, and XLA implementations for other configurations. Qwix supplies optional int8/fp8 computation, and MuonClip adds per-head QK clipping to Muon. Quantized weight loading is separate from quantized training.

Set `remat` on a decoder to a policy name such as `"full"`, `"minimal"`, or `"save_qkv_proj"` to recompute block activations during the backward pass while keeping the named projections. Offloaded policies such as `"minimal_offloaded"` keep those residuals in pinned host memory, and `Layout(host=("opt_state", "ema"))` places optimizer state there between steps. Both reduce device memory at the cost of additional computation or transfers; they compose with layer scanning.

Start with [distributed training](docs/concepts/distributed.md) and the [TPU guide](docs/tpu.md). [Benchmarks](docs/benchmarks.md) and [performance notes](docs/performance.md) record workload sizes, hardware, memory, and timing.

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

`batch=16` is global, so each process reads eight rows. To rehearse the launch on one machine, point the coordinator at `127.0.0.1`, set `XLA_FLAGS=--xla_force_host_platform_device_count=2` and `JAX_PLATFORMS=cpu`, and start rank 0 and rank 1 side by side: two processes of two simulated devices fill the same `MeshSpec(fsdp=4)`, and each prints `20 updates` for `DEW_STEPS=20`.

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

Python 3.14 is recommended and Python 3.12 is supported. Install from the repository with [uv](https://docs.astral.sh/uv/getting-started/installation/):

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

Consult the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for driver requirements and other platforms. The optional extras are `av`, `inference-clients`, `interop`, `metrics`, `plots`, `streaming`, `test`, `tfds`, `vision`, and `wandb`: `interop` reads and writes safetensors, `vision` supplies the host image processors the multimodal checkpoints call, and `inference-clients` brings the Ollama and OpenAI SDKs the serving section uses. The [installation guide](docs/installation.md) covers development dependencies and dataset preparation.

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
