<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AshishKumar4/dew/main/docs/assets/banner-dark.svg">
  <img src="https://raw.githubusercontent.com/AshishKumar4/dew/main/docs/assets/banner-light.svg" alt="Dew" width="960">
</picture>

# Dew

Model training with JAX and Flax

[![CI](https://github.com/AshishKumar4/dew/actions/workflows/ci.yml/badge.svg)](https://github.com/AshishKumar4/dew/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-2aa7a1)](https://github.com/AshishKumar4/dew/blob/main/LICENSE)

[Documentation](https://dewml.dev) · [Installation](#installation) · [Examples](https://github.com/AshishKumar4/dew/tree/main/examples) · [API reference](https://github.com/AshishKumar4/dew/blob/main/docs/reference/core-api.md)
</div>

Dew is a JAX framework for training language models, image and video diffusion models, and JEPA encoders. It comes with common model architectures and training objectives, and one trainer handles optimization, sharding across devices, evaluation and checkpoints for all of them.

Dew started as a fork of [FlaxDiff](https://github.com/AshishKumar4/FlaxDiff), the diffusion-only framework I have been building and refining for years. With FlaxDiff I trained Stable Diffusion-style text-to-image models from scratch on more than 400M text-image pairs, on 128 TPU v4 chips.

You can use the built-in architectures, load a supported Hugging Face checkpoint, or train your own Flax model. Variables and training state are ordinary JAX PyTrees, optimizers are Optax transformations, data loading uses Grain, and checkpoints use Orbax.

Dew is not at 1.0 yet, so APIs and checkpoint formats can still change. [Models](#models) lists what is supported and the known limits.

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

This section trains a diffusion transformer on Oxford Flowers at 64×64 on an NVIDIA GPU, then samples a grid of images.

Install Dew and CUDA JAX in a virtual environment:

```bash
git clone https://github.com/AshishKumar4/dew.git
cd dew
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e ".[tfds,cuda12]"
```

Prepare the dataset once, in a separate environment. The TFDS builder needs TensorFlow, but training reads the prepared files without it. TFDS 4.9.10 imports `importlib_resources` while it prepares a dataset but only declares that dependency for Python before 3.9, so the command installs it explicitly.

```bash
uv venv --python 3.13 .venv-data
uv pip install --python .venv-data/bin/python \
    "tensorflow-datasets==4.9.10" "tensorflow==2.21.0" scipy importlib_resources
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

Here is an abridged [`examples/train_flowers.py`](https://github.com/AshishKumar4/dew/blob/main/examples/train_flowers.py). The full file adds command-line options and the sampling step:

```python
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from dew import Checkpoints, Field, InputSpec, Trainer
from dew.data import Loading, TFDSImages
from dew.diffusion.presets import EDM
from dew.objectives.diffusion import DiffusionObjective
from dew.nn.backbones import SimpleDiT


def train():
    data_path = Path.home() / ".cache/dew/datasets/oxford_flowers102/2.1.1"
    data = TFDSImages(
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
        EDM(regime="pixel"),
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

The `__main__` guard lets Grain start its data-loading worker processes. `Field` describes one image, and the dataset yields batches of 16. The objective adds noise and builds the denoising targets. The trainer runs the optimizer and writes checkpoints, and `state.averaged` holds the EMA weights you sample from.

Run the full script. It also saves a sample grid to `runs/flowers64/samples.png`:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda python examples/train_flowers.py \
    --data "$HOME/.cache/dew/datasets/oxford_flowers102/2.1.1" \
    --steps 1000
```

Use `--steps 20` for a short run, or increase `--steps` to train longer.

[`examples/train_diffusion.py`](https://github.com/AshishKumar4/dew/blob/main/examples/train_diffusion.py) adds pretrained CLIP text conditioning, and [`examples/train_flowers_tpu.py`](https://github.com/AshishKumar4/dew/blob/main/examples/train_flowers_tpu.py) runs the same job across a TPU slice and scores the result. For an offline run with no dataset download, [`examples/readme_demo.py`](https://github.com/AshishKumar4/dew/blob/main/examples/readme_demo.py) covers language modeling, resuming from a checkpoint, DPO and flow matching.

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
objective = DiffusionObjective(
    model,
    EDM(regime="pixel"),
    InputSpec(Field("image", (64, 64, 3))),
    ema_decay=0.999,
)
```

After training, sample from the averaged weights with `state.averaged`. Use `state.variables` instead to sample from the latest weights:

```python
from dew import sample
from dew.sampling import Heun

process = objective.process
denoise = process.denoiser(model, state.averaged, conditions={})
images = sample(
    denoise,
    process.noise(jax.random.key(1), (8, 64, 64, 3)),
    solver=Heun(),
    steps=40,
    key=jax.random.key(2),
)
```

Set `ema_decay=None` to train without an averaged copy, then sample with `state.variables`.

`SimpleDiT(dtype=jnp.bfloat16)` sets the dtype the model computes in. The
master weights and optimizer state stay in fp32, so updates still accumulate in
full precision.

For int8 quantization-aware training, install the `quantization` extra
(`uv pip install "dewml[quantization]"`, which installs Qwix) and wrap the model before you construct the objective:

```python
from dew.training import Quantization

model = Quantization(dtype="int8", patterns=(".*dit_block_.*",)).apply(model)
```

The pattern selects the DiT's transformer blocks for int8 and leaves the
patch-embedding convolution in its floating-point dtype. Master weights stay in
fp32. Change `patterns` to quantize other modules.

In code, you build a model from its class (`from dew.nn.backbones import SimpleDiT, CausalTransformer`). A saved run records the class by its import path, which a recipe imports to rebuild the run, and `dew.registry.models` holds short aliases such as `simple_dit` for the command line.

## Features

| Area | What Dew provides |
|---|---|
| Language modeling | Autoregressive pretraining, packed documents, assistant-only SFT, DPO, and GRPO with callable rewards |
| Diffusion | Image and video denoising, rectified flow, latent diffusion, masked-token diffusion, classifier-free guidance, and interchangeable schedules and solvers |
| Representation learning | I-JEPA and V-JEPA with context and target encoders, predictors, block masking, and linear and kNN probes |
| Training systems | Data, FSDP, expert, tensor and sequence parallelism, layer scans, gradient accumulation, mixed precision, asynchronous checkpoints, and configurable EMA |
| Interoperability | Hugging Face configuration and weight translation for the supported families, safetensors export, CLIP and T5 conditioning, and VAE components |
| Evaluation | Perplexity, FID, CLIP score, PSNR, SSIM, representation diagnostics, generated previews, and Weights & Biases tracking |

DPO and GRPO train with the same `Trainer` as pretraining. `dew.rl` has PPO's advantage estimators and loss terms as separate functions, so you can build your own policy loop from them.

`attention_impl` selects the attention kernel: `"reference"`, `"xla"`, `"cudnn"`, `"tpu"` (Pallas splash attention) or `"auto"`. With `"auto"`, Dew picks a kernel each time it traces an attention call, in this order:

1. The reference path, if the call asks for arithmetic that no fused kernel does (a matmul precision above default, a softmax outside fp32, or a compute dtype different from the inputs'), has float64 inputs, or runs bf16 on a GPU older than sm80.
2. cuDNN, on a supported GPU, if the call has no attention sinks.
3. Splash attention, on a TPU, if the call qualifies: lengths of at least 512 tokens that it can tile, a mask it can describe, no additive bias, and whole sequences at the kernel.
4. XLA otherwise.

The kernel never changes the parameter tree, so a checkpoint trained with one kernel loads with any other.

## Models

[Supported models](https://dewml.dev/reference/models/) lists every checkpoint
family `Pretrained.load` can read, by the `model_type` in its `config.json`,
and every architecture Dew can train from scratch. The site builds that page
from Dew's registries. The notes below cover training, inference and export
for each group.

### Text decoders

`Pretrained.load` reads the checkpoint's `config.json` and builds a
`CausalTransformer`. Training uses `LMObjective`, generation uses
`dew.sampling.generate` or `PretrainedDecoder.text_generation()`, and
`Pretrained.save` writes `config.json`, `model.safetensors` and
`generation_config.json` back in the Hugging Face layout.

`dtype` sets the compute dtype and `param_dtype` sets the dtype parameters
are stored in. By default `Pretrained.load` keeps parameters in FP32. Pass
`param_dtype="bfloat16"` to halve weight memory without changing the compute
dtype. State that is not a parameter keeps its declared precision.

When exported, Kimi K2 keeps its own model type, vocabulary, RoPE settings and
routing widths. Small fixtures test loading, a `Trainer` update, export, and
reloading in the reference implementation; their
[source record](https://github.com/AshishKumar4/dew/blob/main/tests/fixtures/hf/kimi-k2-tiny/source.json) pins the released
configuration. Kimi K2.5 wraps the same decoder in a vision model. Dew loads and
trains the text decoder and writes the vision tower and projector tensors back
byte for byte, but it does not run the vision part.

For Qwen3-Next, GLM-5.3 and GLM-5.3-Flash, tiny fixtures built from the released
configurations match their transformers classes, including the prediction
layer, and export back in the source layout after a `Trainer` update. Dew reads
Mamba 2 from both the Hugging Face port (`Mamba2ForCausalLM`) and the original
`mamba_ssm` checkpoints such as `state-spaces/mamba2-130m`, and saves in the
port's layout. A decoder built on Llama's block that mixes sliding-window and
full attention layers exports as `ministral`, which transformers can load.

For Kimi K3, Dew loads the text decoder out of the vision wrapper. That covers
the KDA and NoPE MLA layers, Attention Residuals over blocks of layers, latent
routed experts with SiTU, and the routed experts' compressed-tensors MXFP4
weights. Export writes the experts back as the same packed pairs, re-encoding
trained experts with the library's own rule
(`dew.interop.codecs.quantize_packed_mxfp4`). The vision tower tensors are
written back unchanged. A tiny fixture built from the released remote code
tests parity, an update, export and greedy decoding, and the
[source record](https://github.com/AshishKumar4/dew/blob/main/tests/fixtures/hf/kimi-k3-source/source.json) lists the shape
of every released tensor.

Kimi Linear (KDA and NoPE MLA layers with DeepSeek's routed experts) computes
what its released remote code computes, with one exception. The released gate
adds the balancing bias to its scores in place, so the routing weights include
the bias. Dew weights the chosen experts by the unbiased scores, as vLLM and
K3's revision of the same file do. A tiny fixture from the released code, with
that line patched, tests parity, an update, export and greedy decoding; the
[source record](https://github.com/AshishKumar4/dew/blob/main/tests/fixtures/hf/kimi-linear-source/source.json) lists the
shape of every released tensor.

### Native multimodal models

For a multimodal checkpoint, `Pretrained.load` returns the model, the
checkpoint's own processor and the weights. The processor turns text and raw
media into `ModelInputs`, which `LMObjective`, `Trainer` and cached generation
accept as they are. Export writes the processor and tokenizer files next to
the weights.

The checkpoint's modality configuration decides which image, video and audio
inputs a model accepts. `Processor.__call__` takes `text`, `images`, `audio`,
`videos` and `video_metadata`. `Processor.chat` applies the checkpoint's own
chat template, so template controls such as `reasoning_effort` and
`preserve_thinking` behave as the checkpoint defines them. The checkpoint's
processor also does the image, video and waveform preprocessing, and Dew
arranges its outputs row by row. DeepSeek-V4.1 ships without a processor, so
you build its pixel values and image positions yourself
([language models](https://github.com/AshishKumar4/dew/blob/main/docs/concepts/language_models.md#multimodal-checkpoints)).

Qwen 3.8 uses the Qwen 3.5 model types. `Qwen/Qwen3.8-27B` loads as a
`qwen3_5` conditional model with a dense hybrid decoder and image and video
inputs. The text-only `Qwen/Qwen3.8-2.4T-A95B` loads as `qwen3_5_moe_text`,
with normalized top-k routing and a sigmoid-gated shared expert.
`tests/fixtures/hf/qwen38-source/source.json` pins both revisions, and every
tensor name in their indexes maps to a Dew parameter, including the
multi-token prediction (MTP) layer they ship. The MTP layer shares the target
embedding and head, trains as an auxiliary loss, and decodes candidate steps
from its own cache. Dew refuses a checkpoint with more than one prediction
layer.

I did not download the released Qwen 3.8 weights. The tests run tiny fixtures
with the released shapes, on CPU in float32, so I have not checked full-size
memory use, bf16 parity, accelerator throughput, multi-host placement or a
speculative accept/reject scheduler.

### Block-diffusion decoders

Diffusion Gemma (`diffusion_gemma`) generates canvases and trains with text
and image-conditioned SFT.

`BlockDiffusionObjective` trains the canvas loss starting from the loaded
weights, and `PretrainedBlockDecoder.block_generation()` decodes canvases. Text
SFT follows Google's published recipe. Image-conditioned SFT goes through the
same `ModelInputs`, `Dataset` and `Trainer`. The images condition the clean
encoder, and their placeholder slots are not text targets. To decode, pass the
matching `images=` to `BlockGeneration`. The objective makes `layer_scalar`
trainable, so export has to use the objective's model:
`replace(loaded, model=objective.model).save(directory, variables=state.variables)`.

### Masked-diffusion decoders

LLaDA (`llada`) and Dream (`dream`, `Dream`) train with the MDLM loss,
starting from their released weights.

`MaskedDiffusionObjective(loaded, MDLM(mask_id=...)(), seq_len)` trains on the
MDLM negative ELBO starting from a loaded checkpoint.
`loaded.save(directory, variables=state.variables)` writes
the trained weights back with the source's own tensor names (OLMo-style names
for LLaDA, the Qwen 2 layout for Dream), next to the config they came with.
The transformers library has no class for either release, so the export tests compare
against the transformers model each one is built from, run with an all-visible
attention mask: `LlamaForCausalLM` for LLaDA and `Qwen2ForCausalLM` for Dream.

### Pretrained diffusion and quantized checkpoints

`Pretrained.load` reads SD, SDXL, SD3, Flux and Qwen-Image 2.1 pipeline
directories. For SD and SDXL that includes img2img, inpainting and the SDXL
refiner.

The loader reads these quantized storage formats:

- DeepSeek's FP8 blocks (`weight_scale_inv`);
- DeepSeek-V4 and V4.1's `.scale` storage (FP8 layers, FP4 routed experts,
  engram tables);
- GPT-OSS's MXFP4;
- compressed-tensors' `mxfp4-pack-quantized`, `nvfp4-pack-quantized` (weights
  only), `pack-quantized` (int codes of 1 to 8 bits), `float-quantized` (FP8 by
  tensor, channel or block), `int-quantized` and `naive-quantized`;
- AutoAWQ's 4-bit gemm packing;
- GPTQ at 2, 4 and 8 bits, act-order included.

It decodes each format to the same values as the release's own
dequantization. `Pretrained.save` writes trained weights back in the source's
format and scale dtype, with the encoding rule of the tool that wrote the
source.

AWQ and GPTQ weights are saved against the scales and zeros the source
shipped. A change smaller than half a grid step is lost, so a lightly trained
model mostly saves back as its source. Dew refuses to save a trained value
outside the grid, because AutoAWQ's packing would spill it into the
neighbouring codes and gptqmodel's would clamp it. Save a substantially trained
model dense instead; the error message names the call. A compressed-tensors
weight is saved the way the library's own compressor writes it: from the weight
in the dtype Dew holds, against the source's scales, clamping a value past the
code range as the library does. Activations that a checkpoint quantizes
dynamically run in the model's dtype, and Dew refuses static activation scales.

The model-family tests use small fixtures with the source's shapes. I have not
validated full-size checkpoints, accelerator performance or physical multi-host
runs. Video inputs are tested on Gemma 4 and Qwen 3.5.

### Diffusion and representation models

Dew trains its own UNets, DiTs, MMDiTs, a video DiT and the I-JEPA and V-JEPA
encoders from scratch, for diffusion, flow matching and masked representation
prediction; [Supported models](https://dewml.dev/reference/models/#architectures-you-can-train-from-scratch)
lists them by alias.

CLIP and T5 text encoders supply conditioning, and the VAE interfaces let you train in latent space.

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

This decoder trains on TinyStories, a corpus of short stories in simple English, with the GPT-2 tokenizer. Download the 22 MB validation file of TinyStories V2 and tokenize it into the `train.bin`, `val.bin` and `meta.json` files that `TokenWindows` reads. `dew tokenize` holds out the first 1% of the tokens for validation.

```bash
hf download roneneldan/TinyStories TinyStoriesV2-GPT4-valid.txt \
    --repo-type dataset --local-dir data
dew tokenize --input data/TinyStoriesV2-GPT4-valid.txt \
    --out data/tinystories --tokenizer gpt2
```

Each row holds 257 token IDs: the model receives the first 256 and predicts the following 256.

```python
import jax
import jax.numpy as jnp
import optax

from dew import Trainer
from dew.data import HFTokenizer, Loading, TokenWindows
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective, Perplexity
from dew.sampling import Sampling, generate

tokenizer = HFTokenizer("gpt2")
data = TokenWindows(path="data/tinystories", seq_len=256,
                    loading=Loading(workers=0)).load(batch=32)
model = CausalTransformer(vocab_size=tokenizer.vocab_size, emb_features=256, num_layers=4,
                          num_heads=4, max_seq_len=256, dtype=jnp.bfloat16)
objective = LMObjective(model, seq_len=256)
lm_state = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0)).fit(
    data, steps=2000, log_every=500, eval_every=1000, metrics=(Perplexity(),))
continuation = generate(model, lm_state.variables, [tokenizer.encode("Once upon a time")],
                        max_new_tokens=40, key=jax.random.key(1),
                        sampling=Sampling(temperature=0.0))
print(tokenizer.decode(continuation.tokens[0]))
```

On one Colab L4 GPU the run takes about three minutes. The training loss falls from 2.75 at step 500 to 1.95 at step 2,000, and validation perplexity reaches 8.7. One run's greedy continuation reads:

> Once upon a time, there was a little girl named Lily. She had a big, red ball. Lily loved to play with her ball. One day, she saw a big box. She wanted to open it.

`temperature=0` selects the highest-probability token. GPU reductions are not bitwise repeatable by default, so a second run can continue differently after the first sentence. Validation uses EMA weights, which lag the live parameters during a short run: at step 1,000 their perplexity is 29.3.

To pack whole documents into the windows instead, set `TokenWindows(pack=True)`. It adds segment IDs and positions, and splits the stream at the EOS ID that `dew tokenize --pack` records. `ChatMessages` reads conversations from a parquet file, a JSONL file or a Hub dataset ID, renders them with the tokenizer's chat template, and records each token's role. Set `LMObjective(loss_role=Role.ASSISTANT)` to train only on assistant targets. See [language models](https://github.com/AshishKumar4/dew/blob/main/docs/concepts/language_models.md) for checkpoint loading and text tokenization.

### Supervised fine-tuning

Next, fine-tune the decoder above on one response to a prompt. Each token has a role: the prompt's tokens are `Role.USER` and the response's are `Role.ASSISTANT`. When you read conversation data, `ChatMessages` builds this role column from the chat template.

```python
import itertools

import numpy as np

from dew import Dataset
from dew.data.chat import Role

prompt = tokenizer.encode("Tom had a red ball.")
response = tokenizer.encode(" He kicked it to his dog.")
row = np.array(prompt + response, dtype=np.int32)
roles = np.array([Role.USER] * len(prompt) + [Role.ASSISTANT] * len(response), dtype=np.int8)
sft_batch = {"text": np.tile(row, (8, 1)), "text_roles": np.tile(roles, (8, 1))}
sft_data = Dataset(
    train=lambda partition: itertools.repeat(sft_batch),
    val=None,
    records=8,
    batch=8,
)
sft_objective = LMObjective(
    model,
    seq_len=len(row) - 1,
    variables=lm_state.variables,
    loss_role=Role.ASSISTANT,
)
sft_state = Trainer(
    sft_objective,
    optax.adamw(1e-3),
    key=jax.random.key(2),
).fit(sft_data, steps=20, log_every=10)
```

The loss counts only the assistant targets, after the next-token shift; the prompt tokens are still there as context. For conversation files, `ChatMessages` also keeps tool calls, tool responses and tool schemas.

### Preference optimization

This continues from `model` and `lm_state` above, with a chosen and a rejected response to the same prompt. The masks limit the loss to the response tokens.

```python
import json

from dew.data import PreferencePairs
from dew.objectives.rl import DPOObjective

rejected = tokenizer.encode(" He kicked kicked kicked it.")
pair = {"chosen": prompt + response, "rejected": prompt + rejected,
        "chosen_mask": [0] * len(prompt) + [1] * len(response),
        "rejected_mask": [0] * len(prompt) + [1] * len(rejected)}
pairs = PreferencePairs(records=(json.dumps(pair),) * 8, seq_len=16,
                        loading=Loading(workers=0, threads=1, read_buffer=2)).load(batch=8)
dpo = DPOObjective(model, seq_len=15, beta=0.1, variables=lm_state.variables)
dpo_state = Trainer(dpo, optax.adam(0.001), key=jax.random.key(2)).fit(
    pairs, steps=10, log_every=5)
```

`DPOObjective` keeps the starting policy as a frozen reference and raises the likelihood of the chosen response relative to the rejected one. `PreferencePairs.seq_len` is the width of the whole ID row, and shorter pairs are padded to it. The objective scores one position fewer because of the next-token shift.

`FlowGRPOObjective` applies group-relative rewards to stochastic flow trajectories. `FlowRollout` samples groups of images, computes their rewards, and records the transition densities that the clipped policy objective uses. [FlowGRPO](https://github.com/AshishKumar4/dew/blob/main/docs/concepts/post_training.md) has a complete example with an image reward.

For online reinforcement learning, `SampledRollout` generates groups of responses and calls a reward function on them. `GRPOObjective` then trains on their advantages, old log probabilities and response masks. [`recipes/chain.py`](https://github.com/AshishKumar4/dew/blob/main/recipes/chain.py) connects SFT, DPO, and GRPO stages. The [post-training guide](https://github.com/AshishKumar4/dew/blob/main/docs/concepts/post_training.md) covers reward callbacks and rollout settings.

### Reinforcement learning with a reward function

This example keeps training the same decoder and rewards stories that mention a dog. A task verifier could replace the reward function. The prompt batch has the same numeric layout that `Prompts` produces, including the UTF-8 reward metadata.

```python
from dew.objectives.rl import GRPOObjective, SampledRollout

prompt = tokenizer.encode("Once upon a time, there was a little")
prompt_batch = {
    "prompt": np.tile(np.array(prompt, dtype=np.int32), (8, 1)),
    "prompt_length": np.full(8, len(prompt), dtype=np.int32),
    "data_source": np.tile(
        np.frombuffer(b"tinystories", dtype=np.uint8).astype(np.int32),
        (8, 1),
    ),
    "ground_truth": np.tile(
        np.frombuffer(b"dog", dtype=np.uint8).astype(np.int32),
        (8, 1),
    ),
    "extra_info": np.zeros((8, 0), dtype=np.int32),
}


def reward(data_source, completion, ground_truth, extra_info):
    return float(ground_truth in completion)


rl_data = Dataset(
    train=lambda partition: itertools.repeat(prompt_batch),
    val=None,
    records=8,
    batch=8,
)
rl_objective = GRPOObjective(
    model,
    seq_len=len(prompt) + 7,
    beta=0.01,
    variables=lm_state.variables,
)
rollout = SampledRollout(
    rl_objective,
    reward=reward,
    groups=4,
    max_new_tokens=8,
    sampling=Sampling(temperature=1.0, top_k=40),
    decode=tokenizer.decode,
)
rl_state = Trainer(
    rl_objective,
    optax.adamw(1e-4),
    key=jax.random.key(3),
    rollout=rollout,
).fit(rl_data, steps=20, log_every=10)
```

Each prompt gets four responses, and `SampledRollout.decode` turns each one into the text the reward reads. The advantages come from how each response's reward compares with the others in its group. GRPO uses a clipped policy objective and an optional KL term against the reference; `beta` sets the KL coefficient. `seq_len` covers the prompt and the 8 response tokens, minus one for the next-token shift.

In the Colab run, 3 of 128 responses sampled from the policy before these 20 steps mention a dog, and all 128 sampled after them do.

### Diffusion language models

Masked diffusion trains a bidirectional decoder to recover masked tokens. There is no next-token shift, so each input row holds exactly `seq_len` tokens. Windows of `seq_len=127` hold 128 IDs each, which is what the 128-token objective needs. The mask token takes the first ID after GPT-2's vocabulary.

```python
from dew.diffusion.discrete import MDLM
from dew.objectives.diffusion import MaskedDiffusionObjective

masked_data = TokenWindows(path="data/tinystories", seq_len=127,
                           loading=Loading(workers=0)).load(batch=64)
mask_id = tokenizer.vocab_size
process = MDLM(mask_id=mask_id)()
masked_model = CausalTransformer(
    vocab_size=mask_id + 1,
    emb_features=256,
    num_layers=4,
    num_heads=4,
    max_seq_len=128,
    causal=False,
    dtype=jnp.bfloat16,
)
masked_objective = MaskedDiffusionObjective(masked_model, process, seq_len=128)
masked_state = Trainer(
    masked_objective,
    optax.adamw(1e-3),
    key=jax.random.key(4),
).fit(masked_data, steps=4000, log_every=1000, eval_every=2000,
      metrics=(Perplexity(),))
drawn = process.generate(masked_model, masked_state.averaged,
                         [tokenizer.encode("Once upon a time")], 48, key=jax.random.key(5))
print(tokenizer.decode(drawn.tokens[0]))
```

On the same L4 the 4,000 steps take about five minutes. The loss, MDLM's negative ELBO per token, falls from 3.48 at step 1,000 to 2.78 at step 4,000, and validation perplexity reaches 15.0. That perplexity is exp of the ELBO, an upper bound on the model's own, so it does not compare directly with the 8.7 above. `process.generate` unmasks the 48 tokens after the prompt in 64 reverse steps. One run's sample reads:

> Once upon a time, in a small town, there lived a little girl named Lily. Mia loved whistle. She had an key telling to try to sleep. One day, she always to see her favorite mom, dad unate to have lots of.

LLaDA and Dream train the same way. Diffusion Gemma uses a different process, with canvases and self-conditioning.

### JEPA representation learning

A JEPA encoder learns by predicting the representations of masked image patches. This example uses the Flowers data prepared in [Getting started](#getting-started), with 8×8 patches and a smaller predictor.

```python
from pathlib import Path

import jax
import optax

from dew import Field, Trainer
from dew.data import Loading, TFDSImages
from dew.objectives.jepa import JepaEncoder, JepaObjective, JepaPredictor, KnnProbe, MultiBlockMask


def train_jepa():
    data = TFDSImages(
        path=str(Path.home() / ".cache/dew/datasets/oxford_flowers102/2.1.1"),
        split="train",
        image_size=64,
        val_batches=2,
        loading=Loading(workers=0, threads=1, read_buffer=2),
    ).load(batch=16)
    encoder = JepaEncoder(
        patch_size=8,
        emb_features=64,
        num_layers=2,
        num_heads=4,
    )
    predictor = JepaPredictor(
        grid=(8, 8),
        emb_features=64,
        predictor_features=32,
        num_layers=1,
        num_heads=4,
    )
    objective = JepaObjective(
        encoder,
        predictor,
        mask=MultiBlockMask.for_grid((8, 8), num_targets=1, scale=(0.25, 0.25)),
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
        metrics=(KnnProbe(102),),
    )


if __name__ == "__main__":
    jepa_state = train_jepa()
```

The target encoder is an EMA of the context encoder. The loss compares predicted and target representations and never reconstructs pixels. The kNN metric is a quick check on half a batch; judge representation quality with a larger labeled evaluation. `jepa_video_encoder` applies the same objective to video clips.

### Loading pretrained weights

You can load a supported checkpoint from a Hub repository or a local directory. This example downloads Qwen3-0.6B and its tokenizer (about 1.5 GB).

```python
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from dew.interop import PretrainedDecoder
from dew.sampling import Sampling, generate

checkpoint = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
pretrained = PretrainedDecoder.load(
    checkpoint,
    dtype=jnp.bfloat16,
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
    sampling=Sampling(temperature=0.8, top_k=40, eos_token_ids=tokenizer.eos_token_id),
)
print(tokenizer.decode(result.tokens[0], skip_special_tokens=True))
```

To train from those weights, pass the bundle in place of the model (`LMObjective(pretrained, seq_len=512)`, or a post-training objective), and tokenize the training data with the checkpoint's own tokenizer. `pretrained.adapt(LoRA(rank=8, modules=("q_proj", "v_proj")), key=0)` adds a low-rank adapter first, so the objective trains only the adapter's factors ([language models](https://github.com/AshishKumar4/dew/blob/main/docs/concepts/language_models.md#pretrained-checkpoints)). [Generating and serving](#generating-and-serving) shows how to sample from and export the result.

### Composing a native decoder

`CausalTransformer` takes its layer pattern as configuration. This decoder
has three sliding-window attention layers followed by one global layer. It
trains on a Grain stream, then generates from the trained state:

```python
import grain.python as grain

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew import Checkpoints, Dataset, Trainer
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling

model = CausalTransformer(
    vocab_size=8, emb_features=64, num_layers=4,
    num_heads=4, num_kv_heads=2, mlp_features=128, max_seq_len=32,
    layer_types=("sliding_attention",) * 3 + ("full_attention",),
    kinds={"sliding_attention": LayerKind(window=8)},
    dtype=jnp.float32,
)
row = np.resize(np.array([1, 2, 3, 4], np.int32), 17)
batch = {"text": np.tile(row, (4, 1))}
stream = grain.MapDataset.source([batch]).repeat().to_iter_dataset()
data = Dataset(train=lambda partition: iter(stream), val=None, records=4, batch=4)
objective = LMObjective(model, seq_len=16)
checkpoints = Checkpoints("runs/custom-decoder")
state = Trainer(objective, optax.adamw(0.003), key=jax.random.key(0),
                checkpoints=checkpoints).fit(data, steps=40, log_every=20,
                                             checkpoint_every=40)
checkpoints.wait()
task = objective.pipeline(state)
result = task([[1, 2]], 8, key=jax.random.key(1), sampling=Sampling(temperature=0))
print(np.asarray(result.tokens))
```

`layer_types` names each layer's kind, and `kinds` says what that kind does:
`sliding_attention` attends to 8 keys including its own, and `full_attention`
attends to all of them. A Grain `to_iter_dataset()` iterator has
`get_state` and `set_state`, which `checkpoint_every` needs to record the data
position. A plain generator has neither, so `Trainer` raises an error if you
use one with `checkpoint_every`. `objective.pipeline` returns a
`TextGeneration` task that uses the state's weights.

### Custom Flax models

An objective can train any Linen module. This example fits `y = 2x + 1` with a single dense layer.

```python
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew import Aux, Dataset, Field, InputSpec, Objective, Trainer

class Regression(Objective):
    model = nn.Dense(1)
    inputs = InputSpec(Field("x", (1,)))

    def init(self, key, variables=None):
        return self.model.init(key, jnp.ones((1, 1)))

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        loss = jnp.mean((prediction - batch["y"]) ** 2)
        return loss, Aux(metrics={"mse": loss})

x = np.linspace(-1, 1, 32, dtype=np.float32).reshape(32, 1)
data = Dataset.from_records({"x": x, "y": 2 * x + 1}, batch=32)
objective = Regression()
state = Trainer(objective, optax.sgd(0.1), key=jax.random.key(0)).fit(
    data, steps=100, log_every=25)
print(objective.model.apply(state.variables, jnp.array([[0.0], [1.0]])))
```

The predictions approach `1` and `3`. `init` creates the variables, `loss` returns a differentiable scalar, and `Aux` holds metrics and any updates to mutable variables. You can pass a custom objective straight to `Trainer`, and a configuration names it by its import path.

The [objective guide](https://github.com/AshishKumar4/dew/blob/main/docs/concepts/objectives.md) also covers BatchNorm state and EMA selection.

### Evaluation and checkpoints

Pass `eval_every` and metrics to `fit` to score the validation data, and set `preview=True` if you also want generated previews. Perplexity is computed from the token losses of the whole validation pass, and FID accumulates statistics over the pass before it computes the distance.

`Checkpoints` saves the training state and the data position with Orbax. To continue a run, rebuild it with the same checkpoint directory. `fit(steps=1200)` is a total: a run restored at step 1000 trains to step 1200, not 2200. The recipe configuration is saved separately; `RunConfig.save` writes it to `run.json`.

See [checkpointing and resume](https://github.com/AshishKumar4/dew/blob/main/docs/guides/checkpoints.md) for restore requirements, local checkpoints, and current recovery limitations.

Which call continues a run depends on the artifact you kept:

| What you have | What it continues into | Call |
|---|---|---|
| A native checkpoint directory | The same run: optimizer state, the root key, and the data position | `Trainer(..., checkpoints=Checkpoints(directory))`, then `fit` |
| A `TrainState` in memory | Generation from the weights you just trained | `objective.pipeline(state)` |
| A saved run: `run.json` beside its checkpoints | Generation, with the model rebuilt from the record | `dew.pipeline(run_directory)` |
| A source checkpoint directory or Hub repository | Generation, or training from those weights | `dew.pipeline(source)`, or `Pretrained.load(source)` for the variables |
| Trained variables another runtime has to read | The source format, without Dew | `Pretrained.save(directory, variables=state.variables)` |
| A saved run another runtime has to read | The same layout, from the run alone | `PretrainedDecoder.from_run(run_directory).save(destination)`, or `dew export <run> <dest>` |

`Pretrained.save` writes the weights, the config it derives, and the
tokenizer or processor files. It does not include optimizer state or data
position, so keep the native checkpoint if you want to resume training.
[`examples/sft_gemma4.py`](https://github.com/AshishKumar4/dew/blob/main/examples/sft_gemma4.py) trains a run and exports
it with `PretrainedDecoder.from_run(...).save(...)`, the last row of the table.

To reproduce a run bit for bit on CUDA you need deterministic GPU reductions.
The flag `--xla_gpu_deterministic_ops=true` turns them on, and
`TrainerConfig.xla_flags` appends it to `XLA_FLAGS`. Check the flag with your
attention backend: under JAX 0.11.1, repeated cuDNN backward calls fail with it
set, while the XLA attention path passed the recorded bitwise checks.
`tests/test_training_qualification.py` resumes a killed fine-tune on the XLA path.

### Standalone evaluation and local reports

`Evaluation.run` scores trained variables without an optimizer and returns metric values and optional previews. `LocalTracker` writes the scalar history, artifacts and plots to disk, with no W&B account or install. Install `dewml[plots]` for Matplotlib output.

```python
import itertools

import jax
import numpy as np
import optax

from dew import Dataset, Evaluation, LocalTracker, Trainer
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective, Perplexity, Samples
from dew.sampling import Sampling

row = np.resize(np.array([1, 2, 3, 4], dtype=np.int32), 17)
batch = {"text": np.tile(row, (8, 1))}
data = Dataset(
    train=lambda partition: itertools.repeat(batch),
    val=lambda partition: iter([batch]),
    records=8,
    batch=8,
)
model = CausalTransformer(
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
    result = Evaluation.run(
        objective,
        state.variables,
        data.val,
        metrics=(Perplexity(),),
        key=jax.random.key(1),
        step=int(state.step),
        preview=True,
    )
    tracker.log(result.scalars, step=result.step)
    for preview in result.previews:
        tracker.artifact(preview, step=result.step)
    print(result.scores)
```

This run reports a perplexity of about 1.002 and saves the training-loss curve, the scalar journal and the generated text under `runs/lm-report`. The tracker draws the plots once, when it closes. Use `plots=False` to record only scalars and artifacts, or call `tracker.plot()` yourself. [`examples/evaluate_and_serve.py`](https://github.com/AshishKumar4/dew/blob/main/examples/evaluate_and_serve.py) scores a finished run the same way and adds an lm-eval-harness suite, image metrics, and a served-model comparison.

`Trackers` sends the same reports to several backends. To switch backends, change the constructor. Install `dewml[wandb]`, `dewml[mlflow]` or `dewml[tensorboard]` for the backend you want:

```python
from dew import LocalTracker, TensorBoardTracker, Trackers

tracker = Trackers(
    LocalTracker("runs/experiment/tracking"),
    TensorBoardTracker("runs/experiment/events"),
)
```

Use it in the same `with` block and `Trainer` call as above. `WandbTracker(project="dew-experiments", offline=True)` and `MLflowTracker("dew-experiments", uri="sqlite:///runs/mlflow.db")` go in the same place. A custom backend implements `log`, `artifact` and `close`. Run configuration, progress, checkpoint requests, profiler windows, sweep trials and failures reach the tracker as typed records.

### Profiling training and inference

Install `dewml[profile]`, or run `uv pip install -e '.[profile]'` in this checkout. Then wrap the work you want to profile in `dew.Profiler`, either as a context manager or with `start` and `stop`. Both forms record the same JAX/XProf capture:

```python
import dew

with dew.Profiler("profiles/run"):
    state = trainer.fit(data, steps=1000)
```

```python
prof = dew.Profiler("profiles/run")
prof.start()
try:
    state = trainer.fit(data, steps=1000)
finally:
    prof.stop()
```

Each capture goes in a new directory, so restarting a profiler keeps the earlier results. Without a path, the first start creates a temporary directory that is not deleted, available as `prof.directory`. A capture keeps the native XPlane traces, the HLO files that exist, and XProf's overview, input, kernel, memory and other supported reports. Its manifest records the backend, package versions, capture options and which reports are available. A counter the backend does not provide is recorded as missing, not as zero. The `profile` extra installs XProf's viewer, and each manifest stores the command that opens its capture under `view_command`, for example `xprof --logdir=profiles/run/capture-<id>`.

A capture leaves JAX's Python tracer off. That tracer records every Python and C call and slows Python-heavy host work several times over, so the host time in its traces is not time the run would spend without it. To trace differently, pass `Profiler` an `options=` value. To start a trace yourself with the same settings, use `jax.profiler.start_trace(directory, profiler_options=capture_options())`, with `capture_options` from `dew.telemetry.profile`.

To trace a window of training, pass `Trainer` a `ProfileWindow` with the trace `directory`, the number of `steps` to trace and the `warmup` steps to run first. The loop starts tracing after the warm-up, stops after the requested steps, and reports the window to the tracker as a `ProfileWindow` record. Use this schedule or an outer `dew.Profiler`, but not both.

### Sweeping a hyperparameter

`RunConfig.sweep` trains one trial for each point of a search space with the ordinary `RunConfig.train`. It keeps a JSON ledger so the sweep can resume, and reports each trial to the tracker you pass it:

```python
from dew import Evaluation, LocalTracker
from dew.config import ModelConfig, OptimConfig, RunConfig, TrainerConfig
from dew.config.sweep import GridSearch
from dew.data import TokenWindows

config = RunConfig(
    # The run records the model it trains, and the batches above stand in
    # for the dataset this names.
    model=ModelConfig.from_model(objective.model),
    data=TokenWindows(seq_len=16),
    optim=OptimConfig(optimizer="adam"),
    trainer=TrainerConfig(name="lm-rate", checkpoint_dir="runs/sweep", steps=40, batch_size=8,
                          eval_every=None, checkpoint_every=None),
)


def trial(run: RunConfig) -> float:
    """Train one point and score it: the perplexity its own run ends on."""
    state = run.train(objective, data, name=run.trainer.name or "lm-rate")
    return float(Evaluation.run(objective, state.variables, data.val, metrics=(Perplexity(),),
                                key=jax.random.key(1), step=int(state.step)).scores["val/perplexity"])


with LocalTracker("runs/sweep/tracking") as tracker:
    trials = config.sweep({"optim.learning_rate": [0.01, 0.003]}, train=trial, trials=2,
                          ledger="runs/sweep/ledger.json", tracker=tracker, search=GridSearch())
best = min(trials, key=lambda trial: trial.value)
print(best.overrides, round(best.value, 4))
```

This prints `{'optim.learning_rate': 0.01} 1.0024`; the slower rate reaches 1.015. Each trial is a real run under `runs/sweep/lm-rate/trial-<index>`, with its own `run.json`, checkpoints and tracking journal. A finished trial is written to the ledger before it is reported, so calling `sweep` again continues an interrupted sweep without retraining the finished trials. `RandomSearch` and `GridSearch` are built in; `OptunaSearch` needs `dewml[hpo]`.

## Diffusion and sampling

A `Process` combines a noise schedule, a prediction transform and a loss weighting. Presets build the common combinations:

| Component | Options |
|---|---|
| Presets | EDM, Karras, Cosine, Flow, Sqrt; MDLM for masked-token diffusion |
| Prediction transforms | Noise, clean-sample, velocity, flow, Karras preconditioning |
| Weighting | Schedule weighting, P2-related weighting, Min-SNR |
| Solvers | DDPM/DDIM, Euler/Heun/RK4, DPM-Solver and DPM-Solver++, DEIS, UniPC, PNDM, LMS, KDPM2, EDM-DPM, LCM, TCD, DPM-Solver SDE |
| Guidance | Classifier-free guidance with an optional interval and rescaling |
| Conditions | `InputSpec`/`Condition`, CLIP, T5, labels or custom encoders |

Training and sampling can use different schedules; EDM, for example, trains on a log-normal distribution of noise levels and samples on a Karras grid. `sample` runs the solver under `jax.lax.scan`, and you can change the solver without retraining. `MultiStepDPM` integrates in sigma space and keeps the previous denoiser outputs to raise the order of each step.

`TextToImage` runs text encoding, denoising and, for latent models, decoding. See [diffusion](https://github.com/AshishKumar4/dew/blob/main/docs/guides/diffusion.md) for text conditioning, latent models, and sampling.

For SD and SDXL checkpoints, `dew.pipeline(source)` rebuilds the source's own
scheduler rather than choosing a solver by name. It supports the source's
clipping, thresholding and timestep spacing, and the Karras, exponential and
beta grids where the matching scheduler has them. An unsupported combination
raises an error; Dew does not fall back to different defaults.

For a text-conditioned image task, pass
`guidance=CFG(scale=7.5, interval=(0.0, 1.0), rescale=0.7)`
after importing `CFG` from `dew.sampling`. Rescaling mixes the guided output
with a version matched to the conditional output's standard deviation.
`rescale=0.0` leaves it unchanged, and `guidance=None` turns classifier-free guidance off.
The tests that compare these schedulers with the source use tiny synthetic
trajectories; they are not quality benchmarks of the released models.

## Generating and serving

A task in `dew.inference` pairs a model with its weights for generation.
`TextGeneration` decodes tokens, `BlockGeneration` decodes Diffusion Gemma
canvases, `MaskedGeneration` samples a whole response from a masked-diffusion
decoder with Dew's MDLM sampler, and `TextToImage` denoises images.
`MaskedGeneration` does not implement LLaDA's or Dream's own remasking recipes.
A task is immutable and keeps the weights it was built with; `bind` returns a
new task with other weights.

### Drawing from a trained language model

`LMObjective.policy` returns a `TextGeneration` over the parameters you pass
it. The GRPO rollout samples with the same task. This continues the decoder
trained in [language modeling](#language-modeling):

```python
from dew.inference import TextGeneration

prompt = tokenizer.encode("Once upon a time")
task = objective.policy(lm_state.variables, Sampling(temperature=0.0))
drawn = task([prompt], 8, key=jax.random.key(1))
print(tokenizer.decode(drawn.tokens[0]), np.asarray(drawn.lengths))

same = TextGeneration(model, lm_state.variables, sampling=Sampling(temperature=0.0))
print(np.array_equal(np.asarray(same([prompt], 8, key=jax.random.key(1)).tokens),
                     np.asarray(drawn.tokens)))
```

This prints `Once upon a time, there was a little girl named Lily [8]` and
`True`, because the constructor and `policy` build the same task. `lengths`
counts the 8 generated tokens and leaves out the 4 prompt tokens in the same
row. `Generation` also returns `terminated`, plus the `behavior_log_probs` and
`raw_log_probs` that a policy ratio needs.

A task built by `PretrainedDecoder.text_generation()` includes the checkpoint's
processor, so it accepts strings and `decode` returns text. Without a
processor the task takes token rows or `ModelInputs`.

### Drawing from a trained diffusion run

`TextToImage.from_run` loads a run directory. It reads the `run.json` that a
`DiffusionRunConfig` wrote and the weights of the latest checkpoint, with the
EMA copy merged over the live parameters, so you don't repeat the model
configuration at generation time.

```python
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from dew import Checkpoints, Trainer
from dew.config import ModelConfig, TrainerConfig
from dew.data import Loading, TFDSImages
from dew.diffusion import presets
from dew.inference import TextToImage
from dew.nn.backbones import SimpleDiT
from dew.objectives.diffusion import DiffusionRunConfig
from dew.sampling import Heun

run = Path("runs/flowers-run")
config = DiffusionRunConfig(
    model=ModelConfig.from_model(SimpleDiT(patch_size=4, emb_features=128, num_layers=4,
                                           num_heads=4, dtype=jnp.bfloat16)),
    data=TFDSImages(
        path=str(Path.home() / ".cache/dew/datasets/oxford_flowers102/2.1.1"),
        image_size=64,
        val_batches=0,
        loading=Loading(workers=0, threads=1, read_buffer=2),
    ),
    trainer=TrainerConfig(checkpoint_dir=str(run), batch_size=16, steps=20, keep=1),
    preset=presets.EDM(regime="pixel"),
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
    images = task(["a flower", "another flower"], steps=20, solver=Heun(),
                  key=jax.random.key(1))
    print(images.host().images.shape, int(state.updates))


if __name__ == "__main__":
    main()
```

The run directory then holds `['20', 'run.json']` and the task draws
`(2, 64, 64, 3)` images clipped to `[-1, 1]`. `text=None` trains an
unconditional model, so the prompt list only sets how many images to draw; a
run with a `TextCondition` encodes the prompts with the encoder it names.
`config.data.load` takes `tokenize=objective.inputs.tokenize` because the
objective's conditions read the dataset's captions. `Loading(workers=0)` keeps
that caption reader in the training process, so it shuts down with the run.

### Exporting a decoder and serving it

`PretrainedDecoder.from_model(model, variables, tokenizer=...)` wraps a model
you trained in Dew so you can export it. `save(directory)` writes the weights
and config in the Hugging Face layout and saves the run's tokenizer files next
to them. Another runtime can load that directory directly.

Tokenize the corpus with the tokenizer you will export, so the token IDs match
the exported vocabulary. Here that is `tiny-tools`, a small byte-level BPE
tokenizer committed for the tests, and the corpus is the TinyStories file from
[Language modeling](#language-modeling):

```bash
dew tokenize \
    --input data/TinyStoriesV2-GPT4-valid.txt \
    --out runs/tokens \
    --tokenizer tests/fixtures/tokenizers/tiny-tools
```

```python
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew import Trainer
from dew.data import HFTokenizer, Loading, TokenCorpus, TokenWindows
from dew.interop import PretrainedDecoder
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling

tokens = Path("runs/tokens")
export = Path("runs/dew-decoder")
corpus = TokenCorpus.read(tokens)
tokenizer = HFTokenizer(corpus.tokenizer)

data = TokenWindows(path=str(tokens), seq_len=128,
                    loading=Loading(workers=0, threads=1, read_buffer=2)
                    ).load(batch=16)
model = CausalTransformer(vocab_size=corpus.vocab_size,
                          emb_features=128, num_layers=4, num_heads=4, num_kv_heads=2,
                          mlp_features=256, max_seq_len=128, dtype=jnp.float32,
                          qk_norm=False, tie_embeddings=False)
state = Trainer(LMObjective(model, seq_len=128),
                optax.adamw(3e-3), key=jax.random.key(0)).fit(
    data, steps=400, log_every=200)

PretrainedDecoder.from_model(model, state.variables, tokenizer=tokenizer).save(export)
print(sorted(path.name for path in export.iterdir()))

task = PretrainedDecoder.load(str(export), dtype=jnp.float32).text_generation()
drawn = task("The trainer", 12, key=jax.random.key(1),
             sampling=Sampling(temperature=0.0))
print(task.decode(drawn))
```

The export directory holds the weights, the config that both runtimes read,
and the tokenizer's files:

```
runs/dew-decoder/
├── chat_template.jinja
├── config.json
├── generation_config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

With `qk_norm=False` and `tie_embeddings=False` the exporter writes a `llama`
config, an architecture llama.cpp can convert. The default decoder exports as
`qwen3`, which Ollama 0.32.9 rejects with `unsupported architecture
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

`num_gpu 0` keeps the runner on the CPU. `TEMPLATE "{{ .Prompt }}"` passes
the prompt through unchanged, so you can compare the served output with the
local one.
These steps ran on Ollama 0.32.9 on Linux x86-64. On the same machine, Ollama
0.34.3's `ollama create` stops at `MLX runtime is not available`.

`OllamaCompletion` and `OpenAICompletion` wrap the vendors' SDK clients,
which you create yourself. Pass a `Sampling` to ask for the same sampling
settings that local generation uses. The client translates it into backend
options and turns off the daemon's own repetition penalties and other
truncations.

```python
import ollama

from dew.inference import OllamaCompletion
from dew.sampling import Sampling

client = OllamaCompletion("dew-decoder", ollama.Client(host="http://127.0.0.1:11434"))
served = client("The trainer", 12, sampling=Sampling(temperature=0.0), key=0, raw=True)
print(served.texts, served.finish_reasons)
```

With `temperature=0.0`, the daemon reproduces Dew's greedy output token for
token, so `served.texts[0] == task.decode(drawn)[0]`. `Completion` also has
`token_counts`, a `usage` record, and the raw SDK responses under
`responses`. For vLLM, serve the same directory and pass `provider="vllm"`,
which enables the sampling controls vLLM accepts beyond the OpenAI schema.
`provider="sglang"` does the same for SGLang:

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

Without `provider="vllm"` or `provider="sglang"`, the client raises an error for `top_k`, `min_p` and
`eos_id` instead of dropping them, because generic OpenAI endpoints do not
accept them.

## Distributed training

`MeshSpec` describes how the devices are arranged, and `Layout` maps model dimensions onto that mesh. For example, `MeshSpec(fsdp=4)` splits eligible parameters and optimizer state over four devices, and the remaining devices form the data-parallel axis. `Trainer` works the same way on one device or a mesh.

The mesh also has expert, tensor, sequence and stage axes. Sequence-parallel attention chooses how to exchange data on each call. It uses Ulysses all-to-alls, which trade sequence rows for heads so that no device holds a whole key or value, when the heads and lengths divide evenly and the all-to-all moves fewer bytes, as it does for causal attention. Otherwise it gathers keys and values. The GPipe stage axis splits execution into stages, but parameters and optimizer state stay replicated across stages, so stages save activation memory and not parameter memory. `MeshSpec(fsdp=8, replicas=2)` is hybrid sharding over two nodes: FSDP inside each node and replicas across them.

Compute dtypes are configurable, and the attention kernel depends on the hardware: cuDNN for supported shapes on NVIDIA GPUs, a Pallas kernel on TPU, and XLA otherwise. Qwix provides optional int8 and fp8 compute, and MuonClip adds per-head QK clipping to Muon. Loading quantized weights is separate from quantized training.

Set `remat` on a decoder to a policy name such as `"full"`, `"minimal"` or `"save_qkv_proj"` to recompute block activations in the backward pass. A policy such as `"save_qkv_proj"` keeps the projections it names. Offloaded policies such as `"minimal_offloaded"` keep those residuals in pinned host memory, and `Layout(host=("opt_state", "ema"))` keeps the optimizer state and the EMA copy there between steps. Both save device memory at the cost of extra compute or transfers, and both work with layer scanning.

Start with [distributed training](https://github.com/AshishKumar4/dew/blob/main/docs/concepts/distributed.md) and the [TPU guide](https://github.com/AshishKumar4/dew/blob/main/docs/tpu.md). [Benchmarks](https://github.com/AshishKumar4/dew/blob/main/docs/benchmarks.md) and [performance notes](https://github.com/AshishKumar4/dew/blob/main/docs/performance.md) record workload sizes, hardware, memory, and timing.

### Multiple hosts

Every host runs the same script. The example below uses two hosts with two GPUs each and shards the model state over all four devices. Both hosts must see the same token files and checkpoint directory. [Training on several nodes](https://github.com/AshishKumar4/dew/blob/main/docs/guides/multi-node.md) covers Slurm, hybrid sharding and long sequences.

Prepare byte-token data from your corpus and place it on shared storage:

```bash
dew tokenize \
    --input corpus.txt \
    --out /shared/tokens \
    --tokenizer byte \
    --val-fraction 0.01
```

Save this as `train_multihost.py`. `prepare_process` joins the process pool, so call it before you create any device array.

```python
import os

import jax
import optax


def main():
    from dew.training.runtime import prepare_process

    prepare_process(multi_host=True)
    try:
        import jax.numpy as jnp

        from dew import Checkpoints, MeshSpec, Trainer
        from dew.data import Loading, TokenWindows
        from dew.nn.backbones import CausalTransformer
        from dew.objectives.lm import LMObjective

        data = TokenWindows(
            path=os.environ["DEW_TOKEN_DIR"],
            seq_len=128,
            loading=Loading(workers=0, threads=1, read_buffer=2),
        ).load(batch=16)
        model = CausalTransformer(
            vocab_size=256,
            emb_features=128,
            num_layers=2,
            num_heads=4,
            mlp_features=256,
            max_seq_len=128,
            dtype=jnp.bfloat16,
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

Launch it from the first host. `dew launch` uses ssh to start one process per GPU on each host (four in all) and gives each process its GPU, the coordinator address, the process count and its rank. The remote shell does not read a login profile, so give the interpreter's absolute path:

```bash
dew launch --hosts 10.0.0.1,10.0.0.2 \
    --env DEW_TOKEN_DIR=/shared/tokens --env DEW_CHECKPOINT_DIR=/shared/runs/lm \
    -- /opt/dew/.venv/bin/python train_multihost.py
```

`batch=16` is the global batch, so each process reads four rows. Without `--hosts`, the same command runs on this machine's GPUs. Inside a Slurm allocation it starts `srun`, and with `--tpu NAME` it runs on every worker of a Cloud TPU. [Training on several nodes](https://github.com/AshishKumar4/dew/blob/main/docs/guides/multi-node.md) covers each case. To rehearse the launch on a machine without GPUs, run `dew launch --processes-per-host 2 --env JAX_PLATFORMS=cpu --env XLA_FLAGS=--xla_force_host_platform_device_count=2 --env DEW_STEPS=20 ...` with local directories. Two processes with two simulated devices each make up the same `MeshSpec(fsdp=4)`, and each process prints `20 updates`.

### Generating text with Gemma 4 on one GPU

`dew.pipeline` loads a published checkpoint and returns a task you can call. Set `JAX_PLATFORMS=cuda` before starting Python. I have not run this released Gemma 4 checkpoint on the 4080, so check that it fits in memory before you try it.

```python
import jax.numpy as jnp

import dew

chat = dew.pipeline("google/gemma-4-E2B-it", dtype=jnp.bfloat16)
result = chat(
    ["Explain gradient accumulation in one paragraph.",
     "Name three uses of a JEPA encoder."],
    128,
    key=0,
)
for text in result.text:
    print(text)
```

The task includes the checkpoint's processor, so it accepts strings and `result.text` holds the decoded continuations. The second positional argument is the token budget; if the checkpoint's `generation_config.json` declares one, that is the default. `n=4` draws four continuations per prompt. The prompts above go straight to the tokenizer. To apply the checkpoint's chat template, call `task.processor.chat(messages)` and pass the `ModelInputs` it returns to the same call.

E2B wraps a multimodal model, and the task also accepts `images=` when the checkpoint declares a vision tower. The `dtype` here sets the compute dtype, not how weights are stored: the Gemma 4 loader keeps FP32 parameters. Loading also needs host buffers, device temporaries and the KV cache, so this example does not show that E2B fits on a 16 GB GPU.

### Generating text with a large decoder on a TPU slice

Every process runs the same script with its own prompts, and `result.host()` returns only that process's real rows. I have not run this on a real TPU slice. The pipeline loads the checkpoint before it shards it, so having enough device memory in total does not mean loading will succeed.

Save this as `generate_tpu.py`:

```python
import os

import jax
import jax.numpy as jnp


def main():
    jax.distributed.initialize()  # Cloud TPU workers discover the coordinator
    try:
        import dew
        from dew.training import Layout, MeshSpec

        task = dew.pipeline(
            os.environ.get("DEW_MODEL", "google/gemma-4-31B-it"),
            mesh=MeshSpec(fsdp=jax.device_count()),
            layout=Layout(min_shard=2**16),
            dtype=jnp.bfloat16,
        )
        rank = jax.process_index()
        result = task([f"Process {rank}: write one sentence about tensors."], 64, key=0)
        for text in result.host().text:
            print(rank, text)
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
```

`MeshSpec(fsdp=jax.device_count())` shards eligible weights over the slice; small leaves, and leaves that don't divide evenly, may stay replicated. On Cloud TPU, `jax.distributed.initialize()` can find the coordinator by itself. On a cluster you manage yourself, pass the coordinator address, process count and process ID, as in the training example. Authenticate each worker through its environment or credential store, and keep tokens out of launch arguments. Save the script on every worker, then preview the launch on every worker (see [Cloud TPUs](https://github.com/AshishKumar4/dew/blob/main/docs/tpu.md)):

```bash
dew launch --tpu dew-16 --zone us-central2-b --dry-run \
    --env DEW_MODEL=google/gemma-4-31B-it -- python generate_tpu.py
```

Each worker needs access to the checkpoint and tokenizer files. A shared download cache still leaves every process with its own loading buffers. When you estimate memory, count the stored weight dtype, replicated leaves, loading peaks and the KV cache; checkpoint bytes divided by the device count is too low. Row counts, tokenized shapes and execution settings must match across processes. Use the same seed on every rank, because Dew derives each global row's key from it. Rehearse with a small checkpoint on a CPU process pool before a real TPU run.

## Data and configuration

Dew's dataset specifications build batches with Grain. The token loaders read fixed windows or packed documents, and the chat, preference and prompt loaders read post-training data. Image and video data can come from local files, Hugging Face datasets, TFDS, ArrayRecord shards or URL streams.

`Loading` sets the number of workers, read threads and buffers. `Dataset` also accepts your own iterator factories, as in the fine-tuning examples above. [Data loading](https://github.com/AshishKumar4/dew/blob/main/docs/concepts/data.md) covers transforms, deterministic randomness, batching and who owns the iterators.

The recipes turn dataclass configurations into command-line options with tyro. `ModelConfig`, `OptimConfig` and `TrainerConfig` hold the model, optimizer and run settings, and task-specific configurations add diffusion or language-model options. `--help` lists the options:

```bash
python recipes/lm/train.py --help
python recipes/diffusion/train.py --help
python recipes/jepa/train.py --help
```

The [recipe guide](https://github.com/AshishKumar4/dew/blob/main/docs/recipes.md) walks through preparing a text corpus and training on it.

## Installation

```bash
curl -LsSf https://dewml.dev/install.sh | sh
```

The script finds the hardware and the build it runs (an NVIDIA GPU and the CUDA version its driver supports, a TPU VM, or the CPU), asks where to install, installs [uv](https://docs.astral.sh/uv/) if it is missing, and installs Dew there. Dew requires Python 3.12 or later; I recommend 3.14, and CI tests both. To install it yourself, choose the extra for your hardware; each installs the accelerator build of the JAX version Dew requires:

| Hardware | uv | pip |
|---|---|---|
| CPU | `uv pip install dewml` | `pip install dewml` |
| NVIDIA GPU, driver 580+ | `uv pip install "dewml[cuda13]"` | `pip install "dewml[cuda13]"` |
| NVIDIA GPU, driver 525+ | `uv pip install "dewml[cuda12]"` | `pip install "dewml[cuda12]"` |
| Google TPU VM | `uv pip install "dewml[tpu]"` | `pip install "dewml[tpu]"` |

CUDA 13 also needs a GPU of compute capability 7.5 or later ([which CUDA build](https://github.com/AshishKumar4/dew/blob/main/docs/installation.md#which-cuda-build)).

To work on Dew itself, install it from a clone instead:

```bash
git clone https://github.com/AshishKumar4/dew.git
cd dew
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e ".[cuda12]"
```

To install the main branch without cloning it, run `uv pip install "dewml[cuda13] @ git+https://github.com/AshishKumar4/dew"`. The extras install the accelerator build of jax 0.11.2, the version Dew requires. A later `-U "jax[...]"` would replace it with a release Dew isn't tested on. A process pool across GPUs keeps its compilation cache only with a patched jax 0.11.2 ([jax-ml/jax#40940](https://github.com/jax-ml/jax/issues/40940)); the [installation guide](https://github.com/AshishKumar4/dew/blob/main/docs/installation.md#process-pools-across-gpus) explains how to install it.

The optional extras are `av`, `cuda12`, `cuda13`, `diffusers`, `eval-harness`, `gguf`, `guided`, `hpo`, `inference-clients`, `interop`, `metrics`, `mlflow`, `mutation`, `plots`, `profile`, `quantization`, `serve`, `streaming`, `tensorboard`, `test`, `tfds`, `torch`, `torchax`, `tpu`, `vision`, `wan` and `wandb`. `interop` reads and writes safetensors, `vision` provides the host image processors that the multimodal checkpoints call, and `inference-clients` installs the Ollama and OpenAI SDKs used in the serving section. The sections above name the extra each feature needs. The [installation guide](https://github.com/AshishKumar4/dew/blob/main/docs/installation.md) covers development dependencies and dataset preparation.

## Documentation and examples

- [Getting started](https://github.com/AshishKumar4/dew/blob/main/docs/getting-started.md): training a custom model.
- [Custom objectives](https://github.com/AshishKumar4/dew/blob/main/docs/concepts/objectives.md): loss functions and mutable model state.
- [Language models](https://github.com/AshishKumar4/dew/blob/main/docs/concepts/language_models.md): tokens, checkpoints, and generation.
- [Post-training](https://github.com/AshishKumar4/dew/blob/main/docs/concepts/post_training.md): SFT, DPO, GRPO, and rewards.
- [Diffusion](https://github.com/AshishKumar4/dew/blob/main/docs/guides/diffusion.md): image, video, conditioning, and latents.
- [Representation learning](https://github.com/AshishKumar4/dew/blob/main/docs/guides/representation-learning.md): JEPA encoders and predictors.
- [API reference](https://github.com/AshishKumar4/dew/blob/main/docs/reference/core-api.md): constructors, arguments, and state contracts.
- [Examples](https://github.com/AshishKumar4/dew/tree/main/examples) and [recipes](https://github.com/AshishKumar4/dew/blob/main/docs/recipes.md): complete programs to adapt.

Each of the five scripts below runs a whole job, from data to scored weights. By default they use settings for real hardware; `--smoke` swaps those for the repository's tiny fixtures, a few steps and one CPU device. [End-to-end examples](https://github.com/AshishKumar4/dew/blob/main/docs/guides/end-to-end.md) gives both command lines for each.

- [`examples/train_flowers_tpu.py`](https://github.com/AshishKumar4/dew/blob/main/examples/train_flowers_tpu.py): a text-to-image DiT trained on Oxford Flowers across a TPU slice, then sampled and scored with FID and CLIPScore.
- [`examples/sft_diffusion_gemma.py`](https://github.com/AshishKumar4/dew/blob/main/examples/sft_diffusion_gemma.py): LoRA SFT of DiffusionGemma with the base weights held in host memory, publishing a PEFT adapter directory.
- [`examples/sft_gemma4.py`](https://github.com/AshishKumar4/dew/blob/main/examples/sft_gemma4.py): full-weight SFT of a Gemma 4 decoder on a Hub chat dataset, exported to the Hugging Face layout.
- [`examples/train_rlvr.py`](https://github.com/AshishKumar4/dew/blob/main/examples/train_rlvr.py): GRPO with verifiable rewards, where each completion is a program run against hidden tests in a sandbox fleet, and rollouts come from Dew's own server or a vLLM or SGLang server one update ahead of training.
- [`examples/evaluate_and_serve.py`](https://github.com/AshishKumar4/dew/blob/main/examples/evaluate_and_serve.py): perplexity, an lm-eval-harness suite, image metrics, and a served-model comparison over one finished run.

## Contributing and acknowledgements

Questions, bug reports, and contributions are welcome. Read [CONTRIBUTING.md](https://github.com/AshishKumar4/dew/blob/main/CONTRIBUTING.md) before submitting changes. For numerical issues, include the model configuration, dependency versions, dtype, hardware, and a small reproduction.

Dew grew out of [FlaxDiff](https://github.com/AshishKumar4/FlaxDiff). This project is partially supported by [Google TPU Research Cloud](https://sites.research.google/trc/about/). I would like to thank the Google Cloud TPU team for providing resources for the larger text-conditional experiments.

Dew builds on JAX, Flax, Optax, Grain, Orbax, tyro, and Weights & Biases. [References and attribution](https://github.com/AshishKumar4/dew/blob/main/docs/references.md) lists the papers and upstream implementations used by its models and algorithms.

Dew is licensed under [MIT](https://github.com/AshishKumar4/dew/blob/main/LICENSE). Adapted components, model weights, and datasets retain their applicable notices and licenses. If you use Dew in research, cite the repository and the papers for the models and methods you use.
