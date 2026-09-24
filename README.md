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

Dew is a framework for training language models, image and video diffusion models, and JEPA encoders in JAX. It provides builtin definitions for common models and training objectives, with a shared trainer for optimization, device sharding, evaluation, and checkpoints.

This framework was built as a fork of my previous pure diffusion focused framework I have been building and refining throughout the years, with which I was able to train stable diffusion like text-to-image models from scratch on 400M+ text-image pairs on 128 TPU v4 chips [Flaxdiff](https://github.com/AshishKumar4/FlaxDiff)

Use the supplied architectures, load a supported Hugging Face checkpoint, or train your own Flax model. Model variables and training state remain JAX PyTrees; optimizers are Optax transformations, data loading uses Grain, and checkpoints use Orbax.

APIs and checkpoint formats can change before 1.0. [Models](#models) lists supported configurations and workflow limits.

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
uv pip install -e ".[tfds,cuda12]"
```

Prepare the dataset once in a separate environment. TensorFlow is needed for this TFDS builder, but not for reading the prepared data during training. TFDS 4.9.10 imports `importlib_resources` while it prepares a dataset but declares it only for Python before 3.9, so the install names it.

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

Run the full script. It also saves a sample grid to `runs/flowers64/samples.png`:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda python examples/train_flowers.py \
    --data "$HOME/.cache/dew/datasets/oxford_flowers102/2.1.1" \
    --steps 1000
```

Use `--steps 20` for a short run, or increase `--steps` to train longer.

[`examples/train_diffusion.py`](examples/train_diffusion.py) adds pretrained CLIP text conditioning, and [`examples/train_flowers_tpu.py`](examples/train_flowers_tpu.py) runs the same job across a TPU slice and scores what it trained. For an offline run without a dataset download, [`examples/readme_demo.py`](examples/readme_demo.py) demonstrates language modeling, checkpoint continuation, DPO, and flow matching.

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
and wrap the model before you construct the objective:

```python
from dew.training.quantization import Quantization, apply_quantization

model = apply_quantization(
    model,
    Quantization(dtype="int8", patterns=(".*dit_block_.*",)),
)
```

This selects the DiT transformer blocks for int8 quantization and leaves the
patch-embedding convolution in its configured floating-point dtype. Master
weights remain fp32. Change `patterns` to select other module paths.

Import model classes directly when writing Python: `from dew.nn.backbones import SimpleDiT, CausalTransformer`. The examples below also use `models.build(name, **fields)` for models selected by configuration; both construct the same Flax classes.

## Features

| Area | What Dew provides |
|---|---|
| Language modeling | Autoregressive pretraining, packed documents, assistant-only SFT, DPO, and GRPO with callable rewards |
| Diffusion | Image and video denoising, rectified flow, latent diffusion, masked-token diffusion, classifier-free guidance, and interchangeable schedules and solvers |
| Representation learning | I-JEPA and V-JEPA with context and target encoders, predictors, block masking, and linear and kNN probes |
| Training systems | Data, FSDP, expert, tensor and sequence parallelism, layer scans, gradient accumulation, mixed precision, asynchronous checkpoints, and configurable EMA |
| Interoperability | Hugging Face configuration and weight translation for the supported families, safetensors export, CLIP and T5 conditioning, and VAE components |
| Evaluation | Perplexity, FID, CLIP score, PSNR, SSIM, representation diagnostics, generated previews, and Weights & Biases tracking |

The DPO and GRPO objectives run on the same trainer as pretraining. `dew.rl` holds PPO's advantage estimators and loss terms as separate functions, so you can build your own policy loop from them.

`attention_impl` selects the attention kernel: `"reference"`, `"xla"`, `"cudnn"`, `"tpu"` (Pallas splash attention), or `"auto"`. With `"auto"`, each trace picks cuDNN on a supported GPU, splash attention on a TPU when the kernel can tile the shapes and the sequence is at least 512 tokens, and XLA everywhere else. The choice never changes the parameter tree, so a checkpoint trained with one kernel loads with any other.

## Models

[Supported models](https://dewml.dev/reference/models/) lists every checkpoint
family `load_pretrained` reads, by the `model_type` in its `config.json`, and
every architecture `models.build` trains from scratch. The site generates the
page from Dew's registries when it builds. The notes below cover their
training, inference and export workflows.

### Text decoders

`load_pretrained` reads the checkpoint's `config.json` and builds a
`CausalTransformer`. Training uses `LMObjective`, generation uses
`dew.sampling.generate` or `Pretrained.text_generation()`, and
`Pretrained.save` writes `config.json`, `model.safetensors` and
`generation_config.json` back in the Hugging Face layout.

`dtype` selects computation; `param_dtype` independently selects parameter
storage. `load_pretrained` keeps FP32 parameters by default. Pass
`param_dtype="bfloat16"` to reduce weight storage without changing the
compute dtype. Non-parameter state retains its declared precision.

Kimi K2 keeps its own model type, vocabulary, RoPE settings, and routing widths
when exported. Small fixtures cover loading, a `Trainer` update, export, and
reference reload. Their [source record](tests/fixtures/hf/kimi-k2-tiny/source.json)
pins the released configuration. Kimi K2.5 nests the same decoder in a vision
wrapper: Dew loads and trains the text half, and keeps the tower and projector
tensors to write them back byte for byte. It runs no vision computation.

Qwen3-Next, GLM-5.3 and GLM-5.3-Flash load from tiny fixtures of the released
configurations with parity against their transformers classes, including the
prediction layer, and a `Trainer` update exports back in the source layout.
Mamba 2 reads the Hugging Face port (`Mamba2ForCausalLM`) and the original
`mamba_ssm` checkpoints such as `state-spaces/mamba2-130m`, and saves in the
port's layout. Exporting a decoder whose layers mix sliding and full attention
on Llama's block writes `ministral`, which transformers reads.

Kimi K3 loads its text decoder from the vision wrapper: KDA and NoPE MLA
layers, Attention Residuals over blocks of layers, latent routed experts with
SiTU, and the routed experts' compressed-tensors MXFP4, which export writes
back as the same packed pairs, re-encoding trained experts by the library's
own rule (`dew.interop.codecs.quantize_packed_mxfp4`). The tower tensors are
kept and written back unchanged.
A tiny fixture from the released remote code covers parity, an update, export
and greedy decoding; the [source record](tests/fixtures/hf/kimi-k3-source/source.json)
lists every released tensor's shape.

Kimi Linear loads as its released remote code computes it, KDA and NoPE MLA
layers with DeepSeek's routed experts, with one exception. The released gate
adds the balancing bias to its scores in place, so its routing weights carry
the bias; Dew weighs the chosen experts by the unbiased scores, as K3's
revision of the same file and vLLM do. A tiny fixture from the released code,
with that line patched, covers parity, an update, export and greedy decoding;
the [source record](tests/fixtures/hf/kimi-linear-source/source.json) lists
every released tensor's shape.

### Native multimodal models

`load_pretrained` returns the model, the checkpoint's own processor, and the
weights. The processor turns text and raw media into `ModelInputs`, which
`LMObjective`, `Trainer` and cached generation take unchanged. The export
carries the processor and tokenizer files beside the weights.

The image, video and audio inputs a model accepts follow the checkpoint's
modality configuration. `Processor.__call__` takes `text`, `images`, `audio`,
`videos` and `video_metadata`. `Processor.chat` runs the checkpoint's own chat
template, so the checkpoint interprets template controls such as
`reasoning_effort` and `preserve_thinking`. The checkpoint's own processor also
does the raw image, video and waveform preprocessing, and Dew arranges its
outputs row by row. DeepSeek-V4.1 ships no processor, so its caller builds
the pixel values and image positions
([language models](docs/concepts/language_models.md#multimodal-checkpoints)).

Qwen 3.8 ships under the Qwen 3.5 model types. `Qwen/Qwen3.8-27B` loads as a
`qwen3_5` conditional model with a dense hybrid decoder, images and videos.
The text-only `Qwen/Qwen3.8-2.4T-A95B` loads as `qwen3_5_moe_text`, with
normalized top-k routing and a sigmoid-gated shared expert.
`tests/fixtures/hf/qwen38-source/source.json` pins both revisions, and every
tensor name in their indexes maps to a Dew parameter. That includes the shipped
multi-token prediction (MTP) layer. The MTP layer shares the target embedding
and head, trains as an auxiliary loss, and decodes candidate steps from its own
cache. I did not download the released weights: the qualification runs tiny
source-shaped fixtures on CPU in float32. I make no claim about full-size memory
use, bf16 parity, accelerator throughput, multi-host placement or a speculative
accept/reject scheduler, and Dew refuses a checkpoint with more than one
prediction layer.

### Block-diffusion decoders

Diffusion Gemma (`diffusion_gemma`) generates canvases and trains with text
and image-conditioned SFT.

`BlockDiffusionObjective` trains the canvas loss from the loaded weights and
`Pretrained.block_generation()` decodes canvases. Text SFT follows Google's
published recipe. Image-conditioned SFT uses the same `ModelInputs`, `Dataset`
and `Trainer` path. Images condition the clean encoder; their placeholder slots
are not text targets. `BlockGeneration` takes the matching `images=` when it
decodes. The objective makes `layer_scalar` trainable, so the export goes
through the objective's model:
`replace(loaded, model=objective.model).save(directory, variables=state.params)`.

### Masked-diffusion decoders

LLaDA (`llada`) and Dream (`dream`, `Dream`) train on the MDLM loss from their
released weights.

`MaskedDiffusionObjective(model, MDLM(mask_id=...)(), seq_len,
pretrained=loaded.variables)` trains the MDLM negative ELBO from a loaded
checkpoint. `loaded.save(directory, variables=state.params)` writes the trained
weights back under the source's own tensor names, beside the config they came
with: LLaDA's OLMo-style names and Dream's Qwen 2 layout.
Transformers has no class for either release, so the export tests compare
against the transformers block each model is built from, run with an
all-visible attention mask: `LlamaForCausalLM` for LLaDA and
`Qwen2ForCausalLM` for Dream.

### Pretrained diffusion and quantized checkpoints

`load_pretrained` reads SD, SDXL, SD3, Flux, and Qwen-Image 2.1 pipeline directories.
SD and SDXL include img2img, inpainting, and the SDXL refiner.

The loader reads these quantized storage formats:

- DeepSeek's FP8 blocks (`weight_scale_inv`);
- DeepSeek-V4 and V4.1's `.scale` storage (FP8 layers, FP4 routed experts,
  engram tables);
- GPT-OSS's MXFP4;
- compressed-tensors' `mxfp4-pack-quantized`, `pack-quantized` (int codes of 1
  to 8 bits), `float-quantized` (FP8 by tensor, channel or block) and
  `int-quantized`;
- AutoAWQ's 4-bit gemm packing;
- GPTQ at 2, 4 and 8 bits, act-order included.

An AWQ or GPTQ weight is saved back against the scales and zeros the source
shipped, so a change smaller than half a grid step is lost and a lightly
trained model saves back mostly as its source. A trained value outside that
grid is refused: AutoAWQ's packing would spill it into the neighbouring
codes and gptqmodel's would clamp it. A substantially trained model saves
dense instead (the error names the call). A compressed-tensors weight is saved as the library's own compressor
writes it against the source's scales, clamping a value past the code range
as the library does. Activations that a checkpoint quantizes dynamically run
in the model's dtype; static activation scales are refused.

It decodes them to the same values each release's own dequantization gives.
`Pretrained.save` writes trained weights back in the source's format and scale
dtype, using the encoding rule of the tool that wrote the source.

Model-family tests use small source-shaped fixtures. I have not validated
full-size checkpoint execution, accelerator performance, or physical multi-host
runs. Video inputs are tested on Gemma 4 and Qwen 3.5.

### Diffusion and representation models

Dew trains its own UNets, DiTs, MMDiTs, a video DiT and the I-JEPA and V-JEPA
encoders from scratch, for diffusion, flow matching and masked representation
prediction; [Supported models](https://dewml.dev/reference/models/#architectures-you-can-train-from-scratch)
lists them by registry name.

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

This decoder trains on TinyStories, a corpus of short stories in simple English, with the GPT-2 tokenizer. Download the 22 MB validation file of TinyStories V2 and tokenize it into the `train.bin`, `val.bin` and `meta.json` that `TokenWindows` reads. The tool holds out the first 1% of the tokens for validation.

```bash
hf download roneneldan/TinyStories TinyStoriesV2-GPT4-valid.txt \
    --repo-type dataset --local-dir data
python tools/tokenize_text.py --input data/TinyStoriesV2-GPT4-valid.txt \
    --out data/tinystories --tokenizer gpt2
```

Each row holds 257 token IDs: the model receives the first 256 and predicts the following 256.

```python
import jax
import jax.numpy as jnp
import optax

from dew import Trainer, metrics, models
from dew.data import HFTokenizer, Loading, TokenWindows
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling, generate

tokenizer = HFTokenizer("gpt2")
data = TokenWindows(path="data/tinystories", seq_len=256,
                    loading=Loading(workers=0)).load(batch=32)
model = models.build("causal_transformer", vocab_size=tokenizer.vocab_size,
                     emb_features=256, num_layers=4, num_heads=4, max_seq_len=256,
                     dtype=jnp.bfloat16)
objective = LMObjective(model, seq_len=256)
lm_state = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0)).fit(
    data, steps=2000, log_every=500, eval_every=1000, metrics=(metrics.perplexity(),))
continuation = generate(model, lm_state.params, [tokenizer.encode("Once upon a time")],
                        max_new_tokens=40, key=jax.random.key(1),
                        sampling=Sampling(temperature=0.0))
print(tokenizer.decode(continuation.tokens[0]))
```

On one Colab L4 GPU the run takes about three minutes. The training loss falls from 2.75 at step 500 to 1.95 at step 2,000, and validation perplexity reaches 8.7. One run's greedy continuation reads:

> Once upon a time, there was a little girl named Lily. She had a big, red ball. Lily loved to play with her ball. One day, she saw a big box. She wanted to open it.

`temperature=0` selects the highest-probability token. GPU reductions are not bitwise repeatable by default, so a second run can continue differently after the first sentence. Validation uses EMA weights, which lag the live parameters during a short run: at step 1,000 their perplexity is 29.3.

`PackedTokens` packs whole documents into the windows instead, with segment IDs and positions; it splits the stream at the EOS ID that `tools/tokenize_text.py --pack` records. `ChatMessages` reads conversations from a parquet file, a JSONL file or a Hub dataset id, renders them with the tokenizer's chat template and tracks token roles. Set `LMObjective(loss_role=Role.ASSISTANT)` to train only on assistant targets. See [language models](docs/concepts/language_models.md) for checkpoint loading and text tokenization.

### Supervised fine-tuning

Fine-tune the decoder above on a response to a prompt. Each token carries a role: the prompt's tokens are `Role.USER` and the response's are `Role.ASSISTANT`. `ChatMessages` produces this role column from chat templates when reading conversation data.

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

Continue from `model` and `lm_state` above with a chosen and a rejected response to the same prompt. The masks restrict the loss to the response tokens.

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
dpo = DPOObjective(model, seq_len=15, beta=0.1, pretrained=lm_state.params)
dpo_state = Trainer(dpo, optax.adam(0.001), key=jax.random.key(2)).fit(
    pairs, steps=10, log_every=5)
```

`DPOObjective` keeps the starting policy as a frozen reference and optimizes the relative likelihood of the chosen response. `PreferencePairs.seq_len` is the full ID-row width, and shorter pairs are padded to it; the objective scores one fewer position because of the next-token shift.

`FlowGRPOObjective` applies group-relative rewards to stochastic flow trajectories. `FlowRollout` samples image groups, evaluates rewards, and records the transition densities used by the clipped policy objective. See [FlowGRPO](docs/concepts/post_training.md) for a complete image-reward example.

For online reinforcement learning, `SampledRollout` generates groups of responses and calls a reward function. `GRPOObjective` trains on their advantages, old log probabilities, and response masks. [`recipes/chain.py`](recipes/chain.py) connects SFT, DPO, and GRPO stages. The [post-training guide](docs/concepts/post_training.md) covers reward callbacks and rollout settings.

### Reinforcement learning with a reward function

This example continues the same decoder with a reward for stories about a dog. A task verifier can replace the reward function. The prompt batch uses the same numeric layout as `Prompts`, including UTF-8 reward metadata.

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
    pretrained=lm_state.params,
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

Each prompt produces four responses, and `SampledRollout.decode` turns each one into the text the reward reads. Their relative rewards determine the advantages. GRPO uses a clipped policy objective and an optional reference KL term; `beta` sets its coefficient. `seq_len` covers the prompt and the 8 response tokens, less one for the next-token shift.

In the Colab run, 3 of 128 responses sampled from the policy before these 20 steps mention a dog, and all 128 sampled after them do.

### Diffusion language models

Masked diffusion trains a bidirectional decoder to recover corrupted tokens. Unlike autoregressive training, each input row contains exactly `seq_len` tokens; there is no next-token shift, so windows of `seq_len=127`, which hold 128 IDs each, feed a 128-token objective. The mask takes the ID after the last GPT-2 token.

```python
from dew.diffusion.discrete import MDLM
from dew.objectives.diffusion import MaskedDiffusionObjective

masked_data = TokenWindows(path="data/tinystories", seq_len=127,
                           loading=Loading(workers=0)).load(batch=64)
mask_id = tokenizer.vocab_size
process = MDLM(mask_id=mask_id)()
masked_model = models.build(
    "causal_transformer",
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
      metrics=(metrics.perplexity(),))
drawn = process.generate(masked_model, masked_state.averaged,
                         [tokenizer.encode("Once upon a time")], 48, key=jax.random.key(5))
print(tokenizer.decode(drawn.tokens[0]))
```

On the same L4 the 4,000 steps take about five minutes. The loss, MDLM's negative ELBO per token, falls from 3.48 at step 1,000 to 2.78 at step 4,000, and validation perplexity reaches 15.0. That perplexity is exp of the ELBO, an upper bound on the model's own, so it does not compare directly with the 8.7 above. `process.generate` unmasks the 48 tokens after the prompt in 64 reverse steps; one run's draw reads:

> Once upon a time, in a small town, there lived a little girl named Lily. Mia loved whistle. She had an key telling to try to sleep. One day, she always to see her favorite mom, dad unate to have lots of.

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

The target encoder follows an EMA of the context encoder. The loss compares predicted and target representations; it does not reconstruct pixels. The kNN metric is a half-batch diagnostic, so use a larger labeled evaluation to judge representation quality. `jepa_video_encoder` extends the same objective to video clips.

### Loading pretrained weights

Load a supported checkpoint from a Hub repository or local directory. This example downloads Qwen3-0.6B and its tokenizer (about 1.5 GB).

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

### Composing a native decoder

`CausalTransformer` takes its layer pattern as configuration. This decoder
uses a local/global attention pattern of three windowed layers to one global
layer, trains on a Grain stream, then generates from the trained state:

```python
import grain.python as grain

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew import Checkpoints, Dataset, Trainer
from dew.nn.backbones.causal_transformer import CausalTransformer, LayerKind
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
objective = LMObjective(model, seq_len=16, ema_decay=None)
checkpoints = Checkpoints("runs/custom-decoder")
state = Trainer(objective, optax.adamw(0.003), key=jax.random.key(0),
                checkpoints=checkpoints).fit(data, steps=40, log_every=20,
                                             checkpoint_every=40)
checkpoints.wait()
task = objective.pipeline(state, ema=False)
result = task([[1, 2]], 8, key=jax.random.key(1), sampling=Sampling(temperature=0))
print(np.asarray(result.tokens))
```

`layer_types` names each layer's kind, and `kinds` says what that kind does:
`sliding_attention` attends to 8 keys including its own, and `full_attention`
attends to all of them. A Grain `to_iter_dataset()` iterator has
`get_state` and `set_state`, which `checkpoint_every` needs to record the data
position. A plain generator has neither, and `Trainer` raises an error if you
combine one with `checkpoint_every`. `objective.pipeline` returns the
`TextGeneration` task over the state's weights.

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

    def init(self, key, variables=None):
        return self.model.init(key, jnp.ones((1, 1)))

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        loss = jnp.mean((prediction - batch["y"]) ** 2)
        return loss, Aux(metrics={"mse": loss})

x = np.linspace(-1, 1, 32, dtype=np.float32).reshape(32, 1)
batch = {"x": x, "y": 2 * x + 1}
data = Dataset(train=lambda partition: itertools.repeat(batch),
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

Which call continues a run depends on the artifact you kept:

| What you have | What it continues into | Call |
|---|---|---|
| A native checkpoint directory | The same run: optimizer state, the root key, and the data position | `Trainer(..., checkpoints=Checkpoints(directory))`, then `fit` |
| A `TrainState` in memory | Generation from the weights you just trained | `objective.pipeline(state)` |
| A saved run: `run.json` beside its checkpoints | Generation, with the model rebuilt from the record | `dew.pipeline(run_directory)` |
| A source checkpoint directory or Hub repository | Generation, or training from those weights | `dew.pipeline(source)`, or `load_pretrained(source)` for the variables |
| Trained variables another runtime has to read | The source format, without Dew | `Pretrained.save(directory, variables=state.params)` |
| A saved run another runtime has to read | The same layout, from the run alone | `dew.interop.export_run(run_directory, destination)`, or `dew export <run> <dest>` |

`Pretrained.save` writes the weights, the config it derives, and the
tokenizer or processor files. It does not include optimizer state or data
position, so keep the native checkpoint if you want to resume training.
[`examples/sft_gemma4.py`](examples/sft_gemma4.py) trains a run and exports
it with `export_run`, the last row of the table.

Reproducing a run bitwise on CUDA needs deterministic GPU reductions. The flag
`--xla_gpu_deterministic_ops=true` requests them, and `TrainerConfig.xla_flags`
appends it to `XLA_FLAGS`. Check the flag against your attention backend. Under
JAX 0.11.1, repeated cuDNN backward calls fail with the flag set, while the XLA
attention path passed the recorded bitwise checks. `tools/qualify_training.py`
uses that path with `--attention-impl xla`.

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
    train=lambda partition: itertools.repeat(batch),
    val=lambda partition: iter([batch]),
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

This run reports perplexity around 1.002 and saves the training-loss curve, scalar journal, and generated text under `runs/lm-report`. The tracker renders plots once, when it closes. Use `plots=False` to record only scalars and artifacts, or call `tracker.plot()` yourself. [`examples/evaluate_and_serve.py`](examples/evaluate_and_serve.py) scores a finished run the same way and adds an lm-eval-harness suite, image metrics, and a served-model comparison.

`Trackers` sends the same reports to several backends, and you switch a backend by changing its constructor. Install `dew-ml[wandb]`, `dew-ml[mlflow]` or `dew-ml[tensorboard]` for the backend you want:

```python
from dew import LocalTracker, TensorBoardTracker, Trackers

tracker = Trackers(
    LocalTracker("runs/experiment/tracking"),
    TensorBoardTracker("runs/experiment/events"),
)
```

Use this tracker in the same `with` block and `Trainer` call above. `WandbTracker(project="dew-experiments", offline=True)` and `MLflowTracker("dew-experiments", uri="sqlite:///runs/mlflow.db")` take the same place. A custom backend implements `log`, `artifact`, and `close`. Run configuration, progress, checkpoint requests, profiler windows, sweep trials, and failures use typed reporting records.

### Profiling training and inference

Install `dew-ml[profile]`, or use `uv pip install -e '.[profile]'` from this checkout. Both forms run the same JAX/XProf capture:

```python
import dew

with dew.profile("profiles/run"):
    state = trainer.fit(data, steps=1000)
```

```python
prof = dew.profile("profiles/run")
prof.start()
try:
    state = trainer.fit(data, steps=1000)
finally:
    prof.stop()
```

Each capture gets a new directory, so restarting a profiler keeps earlier results. Without a path, the first start creates a persistent temporary directory, available as `prof.directory`. A capture keeps the native XPlane traces, the available HLO files, and XProf's overview, input, kernel, memory and other supported reports. Its manifest records the backend, package versions, capture options and which reports are available. A counter the backend does not provide is recorded as missing, not as zero. The `profile` extra installs XProf's own viewer, and each manifest stores the command that opens its capture under `view_command`, for example `xprof --logdir=profiles/run/capture-<id>`.

A capture leaves JAX's Python tracer off. The tracer records every Python and C call and slows Python-heavy host work several times over, so the host time in its traces is time the run doesn't spend. To trace differently, pass `profile` an `options=` value. To start a trace yourself the way a capture does, use `jax.profiler.start_trace(directory, profiler_options=capture_options())`, with `capture_options` from `dew.telemetry.profile`.

To trace a chosen window of training, pass `Trainer` a `ProfileWindow` with the trace `directory`, the number of `steps` to trace, and the `warmup` steps to run first. The loop starts tracing after the warm-up, stops after the requested steps, and reports the window to the tracker as a `ProfileWindow` record. Use either this schedule or an outer `dew.profile`, not both.

### Sweeping a hyperparameter

`sweep` trains one trial per point of a search space through the ordinary `RunConfig.train`, keeps a resumable JSON ledger, and reports each trial through the tracker you pass it:

```python
from dew import LocalTracker, evaluate, metrics
from dew.config import ModelConfig, OptimConfig, RunConfig, TrainerConfig
from dew.config.sweep import grid_search, sweep
from dew.registry import datasets

config = RunConfig(
    model=ModelConfig("causal_transformer", {"vocab_size": 8, "emb_features": 32,
                                             "num_layers": 1, "num_heads": 2,
                                             "mlp_features": 64, "max_seq_len": 32}),
    # The synthetic batches above stand in for the dataset this names.
    data=datasets["token_windows"](seq_len=16),
    optim=OptimConfig(optimizer="adam"),
    trainer=TrainerConfig(name="lm-rate", checkpoint_dir="runs/sweep", steps=40, batch_size=8,
                          eval_every=None, checkpoint_every=None),
)


def trial(run: RunConfig) -> float:
    """Train one point and score it: the perplexity its own run ends on."""
    state = run.train(objective, data, name=run.trainer.name or "lm-rate")
    return float(evaluate(objective, state.params, data.val, metrics=(metrics.perplexity(),),
                          key=jax.random.key(1), step=int(state.step)).scores["val/perplexity"])


with LocalTracker("runs/sweep/tracking") as tracker:
    trials = sweep(config, {"optim.learning_rate": [0.01, 0.003]}, train=trial, trials=2,
                   ledger="runs/sweep/ledger.json", tracker=tracker, search=grid_search)
best = min(trials, key=lambda trial: trial.value)
print(best.overrides, round(best.value, 4))
```

This prints `{'optim.learning_rate': 0.01} 1.0024` against 1.015 for the slower rate. Each trial is a real run under `runs/sweep/lm-rate/trial-<index>` with its own `run.json`, checkpoints and tracking journal, and every finished trial is written to the ledger before it is reported, so rerunning the call continues an interrupted sweep instead of retraining. `random_search` and `grid_search` are built in; `optuna_search` needs `dew-ml[hpo]`.

## Diffusion and sampling

A `Process` combines a noise schedule, a prediction transform, and loss weighting. Presets build common combinations:

| Component | Options |
|---|---|
| Presets | EDM, Karras, Cosine, Flow, Sqrt; MDLM for masked-token diffusion |
| Prediction transforms | Noise, clean-sample, velocity, flow, Karras preconditioning |
| Weighting | Schedule weighting, P2-related weighting, Min-SNR |
| Solvers | DDPM/DDIM, Euler/Heun/RK4, DPM-Solver and DPM-Solver++, DEIS, UniPC, PNDM, LMS, KDPM2, EDM-DPM, LCM, TCD, DPM-Solver SDE |
| Guidance | Classifier-free guidance with an optional interval and rescaling |
| Conditions | `InputSpec`/`Condition`, CLIP, T5, labels or custom encoders |

Training and inference can use different schedules, as in EDM's log-normal training distribution and Karras sampling grid. The sampler uses `jax.lax.scan`, so changing the solver reuses the same trained weights. `MultiStepDPM` integrates in sigma space and keeps the previous denoiser outputs to raise the order of each step.

`TextToImage` combines text encoding, denoising, and optional latent decoding. See [diffusion](docs/guides/diffusion.md) for text conditioning, latent models, and sampling.

For SD and SDXL checkpoints, `dew.pipeline(source)` rebuilds the source's own
scheduler instead of picking a solver by name. It supports the source's
clipping, thresholding and timestep spacing, and Karras, exponential and beta
grids where the corresponding scheduler has them. An unsupported combination
raises an error instead of falling back to different defaults.

For a text-conditioned image task, pass
`guidance=CFG(scale=7.5, interval=(0.0, 1.0), rescale=0.7)`
after importing `CFG` from `dew.sampling`. Rescaling mixes the guided output
with a version matched to the conditional output's standard deviation.
`rescale=0.0` leaves it unchanged; `guidance=None` disables classifier-free guidance.
The source-scheduler oracles use tiny synthetic trajectories, not released-model quality benchmarks.

## Generating and serving

`dew.inference` binds weights to a generation task. `TextGeneration` decodes
tokens, `BlockGeneration` decodes Diffusion Gemma canvases, `MaskedGeneration`
samples a whole response from a masked-diffusion decoder with native MDLM, and
`TextToImage` denoises images. `MaskedGeneration` does not implement LLaDA's or
Dream's own remasking recipes. Each task is frozen around the weights it was
built with, and `bind` returns a new task over another set of weights.

### Drawing from a trained language model

`LMObjective.policy` hands back the `TextGeneration` bound to the parameters
you pass it, which is the same task the GRPO rollout samples with. This
continues the decoder trained in [language modeling](#language-modeling):

```python
from dew.inference import TextGeneration

prompt = tokenizer.encode("Once upon a time")
task = objective.policy(lm_state.params, Sampling(temperature=0.0))
drawn = task([prompt], 8, key=jax.random.key(1))
print(tokenizer.decode(drawn.tokens[0]), np.asarray(drawn.lengths))

same = TextGeneration(model, lm_state.params, sampling=Sampling(temperature=0.0))
print(np.array_equal(np.asarray(same([prompt], 8, key=jax.random.key(1)).tokens),
                     np.asarray(drawn.tokens)))
```

This prints `Once upon a time, there was a little girl named Lily [8]` and `True`: the constructor and
`policy` build the same task. `lengths` counts the 8 response actions, not the
4 prompt tokens the row also carries. `Generation` also returns `terminated`,
and the `behavior_log_probs` and `raw_log_probs` a policy ratio needs.

A task built by `Pretrained.text_generation()` carries the checkpoint's
processor, so it accepts strings and `decode` returns text. Without a
processor the task takes token rows or `ModelInputs`.

### Drawing from a trained diffusion run

`TextToImage.from_run` reads a run directory: the `run.json` that a
`DiffusionRunConfig` wrote, and the weights of its latest checkpoint with the
EMA copy merged over the live parameters. You do not restate the model
configuration at generation time.

```python
from pathlib import Path

import jax
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
    print(images.host().images.shape, int(state.updates))


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

`save_pretrained_decoder` writes the Hugging Face layout and asks the tokenizer
the run trained with to save its own files into the same directory. Another
runtime can read that directory as it is.

Tokenize a corpus with the tokenizer the export will carry, so the token ids
and the exported vocabulary are the same one. `tiny-tools` is the small
byte-level BPE tokenizer committed for the tests, and the corpus is the
TinyStories file that [Language modeling](#language-modeling) downloads:

```bash
python tools/tokenize_text.py \
    --input data/TinyStoriesV2-GPT4-valid.txt \
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

`num_gpu 0` keeps the runner on the CPU. `TEMPLATE "{{ .Prompt }}"` passes
the prompt through unchanged, so the served draw is comparable to the local one.
These steps ran on Ollama 0.32.9 on Linux x86-64. On the same machine, Ollama
0.34.3's `ollama create` stops at `MLX runtime is not available`.

`OllamaCompletion` and `OpenAICompletion` wrap the vendors' own SDK clients,
which you construct and own. Pass a `Sampling` to request the same sampling
policy that local generation uses. The client turns it into backend options
and switches off the daemon's own repetition penalties and other truncations.

```python
import ollama

from dew.inference import OllamaCompletion
from dew.sampling import Sampling

client = OllamaCompletion("dew-decoder", ollama.Client(host="http://127.0.0.1:11434"))
served = client("The trainer", 12, sampling=Sampling(temperature=0.0), seed=0, raw=True)
print(served.texts, served.finish_reasons)
```

With `temperature=0.0`, the daemon reproduces Dew's greedy draw token for
token, so `served.texts[0] == task.decode(drawn)[0]`. `Completion` also carries
`token_counts`, a `usage` record, and the SDK's own responses under
`responses`. For vLLM, serve the same directory and pass `provider="vllm"`,
which enables the sampling controls vLLM accepts beyond the OpenAI schema;
SGLang accepts the same controls with `provider="sglang"`:

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

`MeshSpec` describes the device topology; `Layout` maps model dimensions onto it. For example, `MeshSpec(fsdp=4)` splits eligible parameters and optimizer state over four devices. The remaining devices form the data-parallel axis. Use the same `Trainer` interface on one device or a mesh.

The mesh also supports expert, tensor, sequence, and stage axes. Sequence-parallel attention picks its exchange per call: Ulysses all-to-alls, which trade sequence rows for heads so no device holds a whole key or value, where the heads and lengths divide and they move fewer bytes, as causal attention does; a key/value gather everywhere else. The GPipe stage axis partitions the execution view; parameters and optimizer state stay replicated across stages, so a stage split saves activation memory rather than parameter memory. `MeshSpec(fsdp=8, replicas=2)` is hybrid sharding over two nodes: fsdp inside each node, replicas across them.

Models use configurable compute dtypes and hardware-dependent attention kernels: cuDNN on compatible NVIDIA GPU shapes, a Pallas TPU path, and XLA implementations for other configurations. Qwix supplies optional int8/fp8 computation, and MuonClip adds per-head QK clipping to Muon. Quantized weight loading is separate from quantized training.

Set `remat` on a decoder to a policy name such as `"full"`, `"minimal"`, or `"save_qkv_proj"` to recompute block activations during the backward pass while keeping the named projections. Offloaded policies such as `"minimal_offloaded"` keep those residuals in pinned host memory, and `Layout(host=("opt_state", "ema"))` keeps the optimizer state and the EMA copy there between steps. Both reduce device memory at the cost of extra computation or transfers, and both work with layer scanning.

Start with [distributed training](docs/concepts/distributed.md) and the [TPU guide](docs/tpu.md). [Benchmarks](docs/benchmarks.md) and [performance notes](docs/performance.md) record workload sizes, hardware, memory, and timing.

### Multiple hosts

Use the same script on each host. The example below uses two hosts with two visible GPUs each and shards model state over all four devices. Both hosts must read the same token files and checkpoint directory. [Training on several nodes](docs/guides/multi-node.md) covers Slurm, hybrid sharding and long sequences.

Prepare byte-token data from your corpus and place it on shared storage:

```bash
python tools/tokenize_text.py \
    --input corpus.txt \
    --out /shared/tokens \
    --tokenizer byte \
    --val-fraction 0.01
```

Save as `train_multihost.py`. `prepare_process` joins the process pool, so it comes before any device array.

```python
import os

import jax
import optax


def main():
    from dew.training.runtime import prepare_process

    prepare_process(multi_host=True)
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

Launch it from the first host. `dew launch` starts one process per GPU on each host over ssh, four here, and gives each its GPU, the coordinator, the process count and its rank. The remote shell reads no login profile, so name the interpreter by its absolute path:

```bash
dew launch --hosts 10.0.0.1,10.0.0.2 \
    --env DEW_TOKEN_DIR=/shared/tokens --env DEW_CHECKPOINT_DIR=/shared/runs/lm \
    -- /opt/dew/.venv/bin/python train_multihost.py
```

`batch=16` is global, so each process reads four rows. The same command without `--hosts` runs on the GPUs of this machine; inside a Slurm allocation it starts `srun`, and `--tpu NAME` runs on every worker of a Cloud TPU. [Training on several nodes](docs/guides/multi-node.md) covers each. To rehearse the launch on a machine without GPUs, run `dew launch --processes-per-host 2 --env JAX_PLATFORMS=cpu --env XLA_FLAGS=--xla_force_host_platform_device_count=2 --env DEW_STEPS=20 ...` with local directories: two processes of two simulated devices fill the same `MeshSpec(fsdp=4)`, and each prints `20 updates`.

### Generating text with Gemma 4 on one GPU

`dew.pipeline` loads a published checkpoint and returns a callable task. Set `JAX_PLATFORMS=cuda` before starting Python. I have not run this released Gemma 4 checkpoint on the 4080; check loading memory before attempting it.

```python
import dew

chat = dew.pipeline("google/gemma-4-E2B-it", dtype="bfloat16")
result = chat(
    ["Explain gradient accumulation in one paragraph.",
     "Name three uses of a JEPA encoder."],
    128,
    seed=0,
)
for text in result.text:
    print(text)
```

The task carries the checkpoint's processor, so it accepts strings and `result.text` returns decoded continuations. The second positional argument is the token budget; a checkpoint's `generation_config.json` supplies a default when it declares one. `n=4` draws four continuations per prompt. The prompts above go through the tokenizer directly; use `task.processor.chat(messages)` to apply the checkpoint's chat template and pass the returned `ModelInputs` to the same call.

E2B is a multimodal wrapper; the task also accepts `images=` when the checkpoint declares a vision tower. Here, `dtype="bfloat16"` selects computation, not weight storage. The Gemma 4 loader retains FP32 parameters, and loading also needs host buffers, device temporaries, and the KV cache. This example does not establish that E2B fits a 16 GB GPU.

### Generating text with a large decoder on a TPU slice

Every process runs the same script and supplies its own prompts. `result.host()` returns only that process's real rows. I have not verified this deployment on a real TPU slice. The current pipeline loads the checkpoint before sharding it, so aggregate device memory alone does not establish that loading succeeds.

Save as `generate_tpu.py`:

```python
import os

import jax


def main():
    jax.distributed.initialize()  # Cloud TPU workers discover the coordinator
    try:
        import dew
        from dew.training import Layout, MeshSpec

        task = dew.pipeline(
            os.environ.get("DEW_MODEL", "google/gemma-4-31B-it"),
            mesh=MeshSpec(fsdp=jax.device_count()),
            layout=Layout(min_shard=2**16),
            dtype="bfloat16",
        )
        rank = jax.process_index()
        result = task([f"Process {rank}: write one sentence about tensors."], 64, seed=0)
        for text in result.host().text:
            print(rank, text)
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
```

`MeshSpec(fsdp=jax.device_count())` distributes eligible weights over the slice; small or indivisible leaves may remain replicated. Cloud TPU environments can supply coordinator discovery for `jax.distributed.initialize()`. For manual clusters, pass the coordinator address, process count, and process ID as in the training example. Authenticate on each worker through its environment or credential store; do not put tokens in launch arguments. Save the script on every worker and preview the launch on every worker (see [Cloud TPUs](docs/tpu.md)):

```bash
dew launch --tpu dew-16 --zone us-central2-b --dry-run \
    --env DEW_MODEL=google/gemma-4-31B-it -- python generate_tpu.py
```

Each worker needs access to the checkpoint and tokenizer files. A shared download cache does not eliminate per-process loading buffers. Budget for the stored weight dtype, replicated leaves, loading peaks, and KV cache; dividing checkpoint bytes by device count is insufficient. Row counts, tokenized shapes, and execution controls must agree across processes. Use the same seed on every rank; Dew derives global row keys. Rehearse on a small checkpoint and CPU process pool before an authorized TPU run.

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

I recommend Python 3.14. Dew requires Python 3.12 or later, and CI tests both versions. Install from the repository with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/AshishKumar4/dew.git
cd dew
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e .
```

Add the extra for your hardware; its accelerator build of JAX matches the JAX Dew pins:

| Hardware | Install |
|---|---|
| CPU | `uv pip install -e .` |
| NVIDIA GPU | `uv pip install -e ".[cuda12]"` (or `cuda13` for CUDA 13 drivers) |
| Google TPU | `uv pip install -e ".[tpu]"` |

Installing from the repository without a checkout works the same way: `uv pip install "dew-ml[cuda13] @ git+https://github.com/AshishKumar4/dew"`. Take the accelerator build from these extras, not from `jax[cuda12]`, `jax[cuda13]` or `jax[tpu]`. Dew pins jax to a build with a multi-process cache-key fix ([jax-ml/jax#40940](https://github.com/jax-ml/jax/issues/40940)). pip can't resolve PyPI's jax extras beside that pin in one install, and a later `-U "jax[...]"` would replace the pin with any newer PyPI jax. See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for driver requirements.

The optional extras are `av`, `cuda12`, `cuda13`, `eval-harness`, `gguf`, `guided`, `hpo`, `inference-clients`, `interop`, `metrics`, `mlflow`, `plots`, `profile`, `streaming`, `tensorboard`, `test`, `tfds`, `torch`, `torchax`, `tpu`, `vision` and `wandb`. `interop` reads and writes safetensors, `vision` supplies the host image processors that the multimodal checkpoints call, and `inference-clients` installs the Ollama and OpenAI SDKs that the serving section uses. The sections above name the extra each feature needs. The [installation guide](docs/installation.md) covers development dependencies and dataset preparation.

## Documentation and examples

- [Getting started](docs/getting-started.md): training a custom model.
- [Custom objectives](docs/concepts/objectives.md): loss functions and mutable model state.
- [Language models](docs/concepts/language_models.md): tokens, checkpoints, and generation.
- [Post-training](docs/concepts/post_training.md): SFT, DPO, GRPO, and rewards.
- [Diffusion](docs/guides/diffusion.md): image, video, conditioning, and latents.
- [Representation learning](docs/guides/representation-learning.md): JEPA encoders and predictors.
- [API reference](docs/reference/core-api.md): constructors, arguments, and state contracts.
- [Examples](examples/) and [recipes](docs/recipes.md): complete programs to adapt.

The five scripts below run a whole job, from data to scored weights. Each takes real-hardware settings by default and a `--smoke` flag that trades them for the repository's tiny fixtures, a few steps, and one CPU device. [End-to-end examples](docs/guides/end-to-end.md) gives both command lines for each.

- [`examples/train_flowers_tpu.py`](examples/train_flowers_tpu.py): a text-to-image DiT trained on Oxford Flowers across a TPU slice, then sampled and scored with FID and CLIPScore.
- [`examples/sft_diffusion_gemma.py`](examples/sft_diffusion_gemma.py): LoRA SFT of DiffusionGemma with the base weights held in host memory, publishing a PEFT adapter directory.
- [`examples/sft_gemma4.py`](examples/sft_gemma4.py): full-weight SFT of a Gemma 4 decoder on a Hub chat dataset, exported to the Hugging Face layout.
- [`examples/train_rlvr.py`](examples/train_rlvr.py): GRPO with verifiable rewards, where each completion is a program run against hidden tests in a sandbox fleet, and rollouts come from Dew's own server or a vLLM or SGLang server one update ahead of training.
- [`examples/evaluate_and_serve.py`](examples/evaluate_and_serve.py): perplexity, an lm-eval-harness suite, image metrics, and a served-model comparison over one finished run.

## Contributing and acknowledgements

Questions, bug reports, and contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes. For numerical issues, include the model configuration, dependency versions, dtype, hardware, and a small reproduction.

Dew grew out of [FlaxDiff](https://github.com/AshishKumar4/FlaxDiff). This project is partially supported by [Google TPU Research Cloud](https://sites.research.google/trc/about/). I would like to thank the Google Cloud TPU team for providing resources for the larger text-conditional experiments.

Dew builds on JAX, Flax, Optax, Grain, Orbax, tyro, and Weights & Biases. [References and attribution](docs/references.md) lists the papers and upstream implementations used by its models and algorithms.

Dew is licensed under [MIT](LICENSE). Adapted components, model weights, and datasets retain their applicable notices and licenses. If you use Dew in research, cite the repository and the papers for the models and methods you use.
