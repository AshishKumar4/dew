<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
  <img src="docs/assets/banner-light.svg" alt="Dew: model training with JAX and Flax. Language, diffusion, and representation learning." width="960">
</picture>

<h1>Dew</h1>

<a href="https://github.com/AshishKumar4/dew/actions/workflows/ci.yml"><img src="https://github.com/AshishKumar4/dew/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB" alt="Python 3.11+"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2aa7a1" alt="MIT license"></a>

<p><a href="docs/index.md">User guide</a> · <a href="docs/installation.md">Installation</a> · <a href="docs/reference/core-api.md">Core API</a> · <a href="docs/reference/support.md">Capabilities and limits</a></p>
</div>

Dew is a Python framework for building and training machine-learning models with **JAX and Flax Linen**. It includes language-model, image/video diffusion, and JEPA implementations, together with data loaders, optimization, evaluation, and checkpoints. You can use the supplied models and objectives or define your own.

An objective defines model initialization, the loss, and optional evaluation outputs. `Trainer` computes gradients with JAX, applies an Optax optimizer, places variables and batches across devices, updates selected EMA weights, and coordinates evaluation and checkpoint writes. Language modeling, diffusion, and representation learning use those same training interfaces with different data and losses.

Dew is useful when you want to inspect and change a training algorithm in ordinary Python while retaining JAX's array transformations and Flax's explicit variables. It is **pre-1.0 research software**. Implemented sharding and checkpoint translation do not amount to a qualification of every model, checkpoint, accelerator, or cluster. Current scope and restrictions appear beside the tables below. If your main goal is to serve an existing model behind an API, start with the [ecosystem comparison](#how-dew-compares-with-other-tools): Dew is not an inference server.

## Try several capabilities offline

The complete [`examples/readme_demo.py`](examples/readme_demo.py) program creates all its inputs. It trains a small language model, resumes its checkpoint, starts DPO from the learned policy with a frozen reference, and trains a separate flow-matching image model. There is no dataset download, pretrained model, account, or accelerator requirement.

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, these commands create a checkout and environment. Installation downloads packages; running the demo afterward does not download data or weights.

```bash
git clone https://github.com/AshishKumar4/dew.git
cd dew
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
JAX_PLATFORMS=cpu python examples/readme_demo.py --out runs/readme-demo
```

Use a new output directory for each invocation. The program refuses an existing directory rather than mixing a new experiment with an old one. Its LM stage intentionally restores the checkpoint it just created; the DPO and image stages start separate optimizer states.

The core task construction looks like this in the program:

| Stage | Data it defines | Model and objective | Observable result |
|---|---|---|---|
| Language modeling | Repeating IDs `1, 2, 3, 4` in generated train/validation token files | A one-layer decoder and `LMObjective` | Validation perplexity, an on-disk checkpoint, and a generated continuation |
| Resume | The same files and checkpoint directory | A new `Trainer`, targeting step 24 after step 20 | A reported restore followed by four additional updates |
| DPO | Explicit chosen/rejected ID lists and completion masks | The trained decoder with `DPOObjective` | Preference loss and the reference tree's maximum change |
| Image flow matching | Eight in-memory 8×8 RGB stripe images | A small `SimpleDiT` with `DiffusionObjective` and a `Flow` process | Four generated images in NumPy and PPM formats |

On one CPU, the demonstrated run printed:

```text
LM resumed: 20 -> 24; tokens: [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
DPO: 4 updates; reference max change: 0.0
Flow: 3 updates; preview (4, 8, 8, 3); finite=True
```

The final LM validation perplexity was about 1.096. The DPO loss decreased from 0.6931 to 0.5977. Those results show that the program exercised optimization, continuation, and reference handling; they are not measurements of language ability or preference quality. Three image updates produce a sampling preview, not a trained image generator.

The output directory contains `tokens/`, `lm-checkpoints/`, `summary.json`, `flow-preview.npy`, and `flow-preview.ppm`. NumPy previews use `[-1, 1]`; the PPM file is an RGB image grid. This program does **not** write a `run.json` recipe configuration: `Checkpoints` saves numerical state and data position, not the code or configuration that built the model. [Recipes](#configure-and-run-recipes) add that separate record.

<details>
<summary><strong>Complete program, with all imports and generated inputs</strong></summary>

The same script is included here so you can read the workflow without leaving the guide. Save it as `readme_demo.py` to run it outside the checkout after installing Dew.

```python
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from dew import Checkpoints, Field, InputSpec, Trainer, metrics, models
from dew.data import Dataset, Loading, PreferencePairs, TokenWindows
from dew.diffusion.presets import Flow
from dew.objectives.base import Step
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.lm import LMObjective
from dew.objectives.rl import DPOObjective
from dew.sampling import Euler, generate


@dataclass
class Config:
    out: Path = Path("runs/readme-demo")


def language_model(out):
    tokens = out / "tokens"
    tokens.mkdir()
    for split, repeats in (("train", 256), ("val", 32)):
        np.tile(np.array([1, 2, 3, 4], dtype=np.uint8), repeats).tofile(
            tokens / f"{split}.bin")
    (tokens / "meta.json").write_text(json.dumps({
        "tokenizer": "demo-symbols", "vocab_size": 8, "dtype": "uint8",
        "train_tokens": 1024, "val_tokens": 128, "eos_id": None,
    }))
    data = TokenWindows(
        path=str(tokens), seq_len=8, val_batches=1,
        loading=Loading(workers=0, threads=1, read_buffer=2),
    ).load(batch=8)
    model = models.build(
        "causal_transformer", vocab_size=8, emb_features=16, num_layers=1,
        num_heads=2, mlp_features=32, max_seq_len=16,
        dtype=jnp.float32, attention_impl="xla",
    )
    objective = LMObjective(model, seq_len=8, ema_decay=0.9)
    optimizer = optax.adam(0.01)
    checkpoints = Checkpoints(str(out / "lm-checkpoints"))
    trainer = Trainer(objective, optimizer, key=jax.random.key(0),
                      checkpoints=checkpoints)
    first = trainer.fit(data, steps=20, log_every=10, eval_every=20,
                        checkpoint_every=20, metrics=(metrics.perplexity(),))
    first_step = int(first.step)
    resumed = Trainer(objective, optimizer, key=jax.random.key(0),
                      checkpoints=checkpoints)
    state = resumed.fit(data, steps=24, log_every=4, eval_every=4,
                        checkpoint_every=4, metrics=(metrics.perplexity(),))
    continuation = np.asarray(generate(
        model, state.averaged, jnp.array([[1, 2]], dtype=jnp.int32),
        max_new_tokens=8, key=jax.random.key(1), temperature=0.0,
    ))[0].tolist()
    print(f"LM resumed: {first_step} -> {int(state.step)}; tokens: {continuation}")
    return model, state, {"first_step": first_step, "resumed_step": int(state.step),
                          "generated_ids": continuation}


def preferences(model, pretrained):
    row = {"chosen": [1, 2, 3, 4], "rejected": [1, 2, 6, 4],
           "chosen_mask": [0, 0, 1, 1], "rejected_mask": [0, 0, 1, 1]}
    data = PreferencePairs(
        records=(json.dumps(row),) * 8, seq_len=4,
        loading=Loading(workers=0, threads=1, read_buffer=2),
    ).load(batch=8)
    objective = DPOObjective(model, seq_len=3, beta=0.1, pretrained=pretrained)
    reference = jax.tree.map(lambda x: np.array(x, copy=True), pretrained)
    state = Trainer(objective, optax.adam(0.001), key=jax.random.key(2)).fit(
        data, steps=4, log_every=1)
    delta = max(float(np.max(np.abs(np.asarray(after) - before)))
                for before, after in zip(jax.tree.leaves(reference),
                                         jax.tree.leaves(state.ema), strict=True))
    print(f"DPO: {int(state.step)} updates; reference max change: {delta:.1f}")
    return {"updates": int(state.step), "reference_max_change": delta}


def flow_images(out):
    images = np.zeros((8, 8, 8, 3), dtype=np.uint8)
    images[:, :, ::2, :] = 255
    batch = {"image": images}
    data = Dataset(train=lambda: itertools.repeat(batch), val=None,
                   records=8, batch=8)
    model = models.build(
        "simple_dit", patch_size=4, emb_features=16, num_layers=1,
        num_heads=2, mlp_ratio=2, dtype=jnp.float32, attention_impl="xla",
    )
    objective = DiffusionObjective(
        model, Flow()(), InputSpec(Field("image", (8, 8, 3))),
        sampler=Euler(), guidance=None, steps=4,
    )
    state = Trainer(objective, optax.adam(0.001), key=jax.random.key(3)).fit(
        data, steps=3, log_every=1)
    preview = objective.preview(
        state.params, batch,
        Step(step=state.step, key=jax.random.key(4), ema=state.averaged),
    )
    generated = np.asarray(preview.images)
    np.save(out / "flow-preview.npy", generated)
    pixels = np.round((generated + 1) * 127.5).clip(0, 255).astype(np.uint8)
    grid = np.concatenate(list(pixels), axis=1)
    header = f"P6\n{grid.shape[1]} {grid.shape[0]}\n255\n".encode("ascii")
    (out / "flow-preview.ppm").write_bytes(header + grid.tobytes())
    finite = bool(np.isfinite(generated).all())
    print(f"Flow: {int(state.step)} updates; preview {generated.shape}; finite={finite}")
    return {"updates": int(state.step), "preview_shape": list(generated.shape),
            "finite": finite, "range": [float(generated.min()), float(generated.max())]}


def main(config):
    out = config.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    print("Devices:", jax.devices())
    print("Output:", out)
    model, state, lm = language_model(out)
    summary = {"language_model": lm, "dpo": preferences(model, state.params),
               "flow": flow_images(out)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Saved summary.json, LM checkpoints, and flow-preview.npy/.ppm.")


if __name__ == "__main__":
    main(tyro.cli(Config))
```

</details>

## Contents

1. [Choose a workflow and read its maturity](#choose-a-workflow-and-read-its-maturity)
2. [Understand the framework](#understand-the-framework)
3. [Train language models](#train-language-models)
4. [SFT, preference learning, and reinforcement learning](#sft-preference-learning-and-reinforcement-learning)
5. [Diffusion, flow matching, and latent models](#diffusion-flow-matching-and-latent-models)
6. [Representation learning with JEPA](#representation-learning-with-jepa)
7. [Prepare data and batches](#prepare-data-and-batches)
8. [Optimization, precision, and performance](#optimization-precision-and-performance)
9. [Distribute a training run](#distribute-a-training-run)
10. [Evaluate, checkpoint, resume, and export](#evaluate-checkpoint-resume-and-export)
11. [Configure and run recipes](#configure-and-run-recipes)
12. [Write your own objective](#write-your-own-objective)
13. [How Dew compares with other tools](#how-dew-compares-with-other-tools)
14. [Installation options and learning paths](#installation-options-and-learning-paths)
15. [Contributing, attribution, and citation](#contributing-attribution-and-citation)

## Choose a workflow and read its maturity

A method can exist at several levels: array-level mathematics, a model component, a full training objective, a data-backed recipe, or a qualified deployment. Dew does not use those levels interchangeably.

| Workflow or method | What exists in Dew | Scope to keep in mind |
|---|---|---|
| Custom Flax training | `Objective`, Optax optimizers, and `Trainer` | You define the model's loss and data contract; an arbitrary module is not interchangeable with every supplied objective. |
| Autoregressive pretraining | `LMObjective`, compatible decoders, token windows and document packing | Predicts the next token; vocabulary, context, and tokenizer must agree. |
| Supervised fine-tuning (SFT) | Role-aware chat rendering and `LMObjective(loss_role=Role.ASSISTANT)` | Template boundaries determine the assistant mask. Plain strings do not establish role boundaries. |
| Direct preference optimization (DPO) | `DPOObjective`, chosen/rejected pairs, frozen reference | A reference parameter copy and its forward passes still cost memory and compute. |
| GRPO and RLOO-style advantages | `GRPOObjective`, `SampledRollout`, a Python reward function, group or leave-one-out advantages | Local fixed-width generation; not a distributed serving engine or general agent runtime. |
| PPO-related loss components | Clipped surrogate, GAE, KL estimators, token/sequence log-ratios, masked reductions | These are **primitives**. They do not by themselves provide a PPO trainer, value model, rollout buffer, or environment. |
| Image/video diffusion and rectified flow | `DiffusionObjective`, processes, solvers, conditions, image/video backbones | External checkpoint ecosystems and task-specific pipelines remain narrower than Diffusers. |
| Latent diffusion | A configured autoencoder before/after the denoising model | A VAE requires compatible latent shape/scaling; loading a VAE does not load an SD3/Flux/Wan transformer. |
| Masked discrete diffusion | `MaskedDiffusionObjective`, MDLM/log-linear masking, `Unmask`, LLaDA and Dream decoder paths | Different objective and sampling semantics from autoregressive text or Gaussian image noise. |
| Diffusion Gemma | Block-diffusion denoiser and sampling components with small reference comparisons | Separate from generic MDLM; not evidence of a full released checkpoint running locally. |
| I-JEPA and V-JEPA | Context/target encoders, predictors, masks, representation artifacts, probes | Representation learning, not a complete world-model agent or planner. |
| Mixed precision and quantized computation | Float32/bfloat16 model computation; optional Qwix int8/fp8 training | Experimental numerical/performance choices; checkpoint dequantization is a different capability. |
| Multi-device execution | Data, FSDP, expert, tensor, sequence, and stage mesh axes | Local simulations and numerical checks are not physical multi-host GPU/TPU qualification. |

**Integration work is not released functionality.** FlowGRPO, Gemma 3n vision, rematerialization policies, and revised overflow/accumulation transactions are being implemented or reviewed on development branches. General multi-turn agentic RL, sandbox/tool execution, and a serving system are not complete user workflows in this baseline. GSPO/SAPO, RLVR at scale, newer frontier architectures, and parity with every MaxText/Transformers/Diffusers model are research or implementation targets, not implied by a related primitive. Consult [capabilities and limits](docs/reference/support.md) for the integrated revision you install.

Recent development-branch evidence is narrower than a released workflow:

| Work under integration | Evidence reported for that branch | What it does not establish |
|---|---|---|
| Gemma 3n vision | A 33×29, 2.14-million-parameter, 84-block GPU comparison: float32 with highest matmul precision, tower maximum difference 2.98e-6 and wrapper difference 4.95e-7 | Default-precision/bfloat16 strict parity, real released weights, or full-size deployment; those precision modes recorded larger errors. |
| FlowGRPO | A 64-pixel GPU training step and a two-process CPU proof | A merged API, learned reward quality, actor-fleet operation, or multi-host accelerator qualification. |
| Rematerialization | Reviewed implementation with measured 72–78% lower compiler-reported temporary memory in the measured cases | A 72–78% reduction in total GPU memory, or the same saving/time trade-off for another architecture. |
| Training transactions | Explicit attempted/accepted/committed counters, loss statistics, scaler state, and partial-window checkpoint work | A claim that the older integrated checkpoint contract already has those fields or guarantees. |

These results justify further integration and qualification. Use the support page for the commit you run, rather than treating a branch measurement as a release promise.

### Model and checkpoint families

A model's **architecture** describes its computation. A **checkpoint** supplies a particular configuration and learned tensors. A translator accepting the configuration is only the first step toward using that checkpoint.

The table lists the current decoder translation entries, not every product name that shares part of their ancestry. “Small reference checks” means known small inputs/configurations/weights, not a complete size range or production run. Imports and exports can have different coverage.

| Family | Accepted model type | Computation represented; principal boundary |
|---|---|---|
| Llama 2, 3, 3.1 | `llama` | Grouped-query attention and supported rotary forms; inspect the checkpoint's scaling fields. |
| Mistral | `mistral` | Sliding-window decoder attention. |
| Mixtral | `mixtral` | Routed experts with the family router and weight layout. |
| Qwen 2 | `qwen2` | Q/K/V biases with a bias-free output projection. |
| Qwen 3 | `qwen3` | Q/K normalization and supported layer settings. |
| Qwen3-MoE | `qwen3_moe` | Routed expert layers; sparse-model memory and communication still matter. |
| Qwen 3.5 text | `qwen3_5_text` | Gated delta-net/full-attention hybrid and model-specific layer geometry; not blanket Qwen3-Next/3.6/3.8 coverage. |
| Gemma 1, 2 | `gemma`, `gemma2` | Embedding scaling, activation/norm conventions, local layers, and softcapping where specified. |
| Gemma 3 text | `gemma3_text` | Sandwich norms, attention geometry, and local/global behavior; unsupported rotary configurations raise. |
| Gemma 3n text | `gemma3n_text` | AltUp, LAuReL, per-layer inputs, activation sparsity, and KV sharing; vision/audio are separate paths. |
| Gemma 4 text | `gemma4_text` | Per-layer inputs, KV sharing, global attention geometry, and routed variants in the implemented configuration range. |
| OLMo 3 | `olmo3` | Post-normalization/QK-normalization path; unsupported rotary forms raise. |
| DeepSeek V2 and V2-Lite | `deepseek_v2` | Multi-head latent attention (MLA), family routing, and balance-loss conventions. |
| DeepSeek V3 and V3.2 | `deepseek_v3`, `deepseek_v32` | MLA, grouped routing/shared experts, supported YaRN, and the V3.2 sparse indexer; not DeepSeek V4. |
| Kimi K2 | `kimi_k2` | DeepSeek-derived computation and checkpoint metadata; not blanket Kimi Linear/K2.5/K3 wrapper support. |
| GLM 4.5/5 configurations using this decoder type | `glm4_moe` | Attention biases, partial rotary, routing, and supported prediction depths; later `glm_moe_dsa`/`glm5_next` types are not this entry. |
| gpt-oss | `gpt_oss` | Attention sinks, clamped biased experts, and MXFP4 unpacking on load. |
| Llama 4 text | `llama4_text` | Interleaved rotary/local-attention behavior and experts; multimodal preprocessing is separate. |
| LLaDA | `llada` | Bidirectional masked-token decoder. |
| Dream | `dream`, `Dream` | Bidirectional Qwen-derived masked-token decoder. |
| Diffusion Gemma text | `diffusion_gemma_text` | Text computation used in block-diffusion denoising. |

Vision wrappers exist for **Gemma 3, Llama 4, Gemma 4, and Qwen 3.5**, with small tower/projector/wrapper comparisons. Their image resolution, patch ordering, token placement, and padding rules differ. Some paths are restricted to fixed-resolution still images. A tower-forward comparison does not automatically verify processor behavior, generation caches, backward updates, video, ragged images, or a distributed multimodal run. Gemma 3n's MobileNet-v5 vision integration is branch work in this baseline; audio towers are not complete.

Research notes also cover DeepSeek V4, Qwen3-Next and later Qwen families, newer GLM variants, Kimi Linear/K3, MiniMax, and Nemotron hybrids. A surveyed architecture is not a registered Dew model. Read the [family reference](docs/reference/model-families.md) for restrictions, and record five separate facts when reporting support: configuration translation, small reference parity, backward/generation behavior, actual checkpoint loading, and target-hardware deployment. None of the earlier facts guarantees the later ones.

## Understand the framework

Dew uses a small set of objects across tasks. Understanding them makes a recipe easier to modify than learning a separate trainer for each model family.

| Piece | Responsibility | What it does not decide |
|---|---|---|
| Flax model | Forward computation and variable structure | Which dataset examples count, the optimizer, or the validation schedule |
| `Objective` | Initialization, task loss, optional scoring and previews | Where the training state is placed or when checkpoints are written |
| `Dataset` | Factories for training and finite validation iterators | The model's loss or optimizer |
| Optax optimizer | Transform gradients into parameter updates | Token masks, reward construction, or model architecture |
| `Trainer` | Differentiation, updates, placement, EMA, lifecycle, evaluation events, and checkpoints | The mathematical meaning of a new task |
| `TrainState` | Numerical values that continue the run | Dataset contents, source code, package environment, or a complete recipe description |
| Recipe configuration | Record how a run is constructed | Proof that its dataset/checkpoint is available or its deployment is qualified |

### Arrays, shapes, and dtypes

A batch is a mapping from field names to arrays. Let `B` be batch size, `S` token length, `H/W` image height/width, `T` frame count, and `C` channels:

- Token windows use int32 `text` with shape `[B, S + 1]`; the extra ID provides the final next-token target.
- Images conventionally use `[B, H, W, C]`, and videos `[B, T, H, W, C]`. Built-in image/video loaders provide uint8 values in `[0, 255]`; the objectives normalize them.
- DPO pairs use `[B, 2, S]` IDs and masks, with chosen at index 0 and rejected at index 1.
- `Field("image", (8, 8, 3))` describes **one sample**, excluding the batch dimension. `InputSpec` combines that sample with any condition encoders.

Shapes are part of the computation, not display metadata. A different sequence length, batch shape, layer geometry, or static option can require another JAX compilation. Pad or bucket variable-sized examples with the matching masks; do not silently replace them with a dense batch mean. A tokenizer's IDs must fit the embedding vocabulary, and its special tokens and chat template must agree with the model.

Dew generally keeps learned parameters in float32 while choosing float32 or bfloat16 for model computation. An integer token ID, uint8 source pixel, bfloat16 activation, float32 reduction, and quantized stored weight serve different purposes. Changing a compute dtype does not shrink every allocation in a training run.

### Variables, randomness, and state

A Flax variables tree is a nested mapping of collections. The `params` collection contains learned weights; collections such as `batch_stats` or `moe` can hold mutable state. **`state.params` is the whole variables mapping**, so a normal forward call is `model.apply(state.params, inputs)`, not a blind extra wrapping in another `params` key.

JAX random keys are explicit values. A trainer key seeds initialization and subsequent training draws; an objective receives its per-step context in `Step`. Split a key for independent random operations rather than relying on global random state. Sampling with a new key changes the draw; using a fixed key deliberately makes a comparison reproducible within the same numerical setup. Backend/compiler changes can still affect low floating-point bits and sampled decisions.

The state contains live variables, optimizer state, the run's random key, counters, and selected EMA variables when the objective requests them. EMA means *exponential moving average*: update selected weights toward the live weights after optimizer updates. `state.averaged` overlays those weights onto the complete live variables tree. It raises if the objective does not maintain an EMA; use `state.params` for that case. DPO/GRPO use unit decay to keep a frozen reference in the same field. A reused field is not a free parameter copy.

The attempted-step, accepted-microbatch, committed-update, dynamic-scaler, and partial-accumulation contracts are undergoing a checkpoint-layout change. Until that repair is integrated and verified for your revision, do not equate every attempted step with an accepted update or claim exact overflow continuation. The offline program uses `accumulation=1`, no dynamic loss scaling, and finite losses; its counters coincide on that exercised path. The [core API reference](docs/reference/core-api.md) describes the installed state and lower-level compilation interfaces.

## Train language models

`LMObjective` computes next-token cross entropy for a compatible decoder. For a row `[a, b, c, d]`, the model sees `[a, b, c]` and predicts `[b, c, d]`. Document boundaries, padding, and optional role masks determine which targets contribute. A useful language-model run therefore requires both the right decoder and the right token preparation.

`CausalTransformer` includes configurable RMSNorm, rotary positions, grouped-query attention, gated MLPs, local/global attention, logit softcapping, embedding scaling, and family-specific components. Some layer configurations substitute a gated delta-net or MLA mixer. Mixtures of experts add routers, routed/shared expert computation, and balancing state. These are model components with contracts; combining arbitrary fields does not recreate a named reference architecture.

A chunked vocabulary head computes loss without materializing all token-by-vocabulary logits at once. `head_chunks` trades extra passes for lower peak memory. It does not remove attention activations, optimizer state, or the vocabulary matrix itself. Optional multi-token prediction (MTP) adds future-token prediction depths to a compatible model; setting its loss weight requires those modules to exist.

### Dense layers, experts, and hybrid token mixers

A dense feed-forward layer applies the same learned network to every token. A mixture-of-experts (MoE) layer has several networks and a router that selects `top_k` experts per token. More total experts can increase parameter capacity without running every expert on every token, but the weights, optimizer state, and reference/EMA storage still exist. Tokens must also be dispatched to their selected experts.

Dew represents softmax/sigmoid routing, selected-weight normalization, group-limited selection, shared experts, output scales, and selection biases where a family calls for them. Auxiliary balance losses and auxiliary-loss-free bias updates are distinct algorithms: the former adds a differentiable loss term, while the latter changes non-parameter routing state from observed loads. A configuration with matching tensor shapes can still use the wrong router convention. Expert computation groups tokens for `jax.lax.ragged_dot`; an optional tokamax kernel path requires its own package/backend and performance checks.

Attention also has family-specific alternatives. MLA projects keys/values through latent bottlenecks, while gated delta-net layers maintain recurrent state alongside attention layers in a hybrid decoder. These implementations do not imply support for every recurrent family, KDA variant, Mamba architecture, or later sparse indexer. The [MoE guide](docs/concepts/moe.md) explains routing and placement; the family table identifies which checkpoint translations actually exist.

### Pretrain, continue, then generate

For pretraining, start with random variables and a token dataset. For continuation, load a supported checkpoint and its matching tokenizer, then pass the complete variables mapping through the objective's `pretrained` argument. A file with the correct tensor names but wrong tokenizer or rotary convention can load and still represent the wrong computation.

`dew.sampling.generate` takes the model, variables, int32 prompt `[B, P]`, a maximum number of new tokens, and a random key. It returns `[B, P + new_tokens]`, including the prompt. It performs prefill and KV-cache decoding; temperature 0 selects greedy decoding, and positive temperature/top-k configure sampling. The prompt plus continuation must fit the configured context/cache. The fixed-length interface does not provide arbitrary stopping rules, streaming requests, beam search, or a production serving scheduler.

Masked-diffusion language models instead corrupt token positions and learn to reconstruct them with bidirectional context. Dew supplies an MDLM-style discrete process, a log-linear masking schedule, and an unmasking solver, as well as LLaDA/Dream translation. Diffusion Gemma's block-denoising mechanism is a separate path; it is not a synonym for applying the image flow objective to integer IDs.

Continue with [language models](docs/concepts/language_models.md) for complete training/import/generation examples and [model families](docs/reference/model-families.md) for supported settings.

## SFT, preference learning, and reinforcement learning

**SFT** learns from example assistant responses. `ChatMessages` reads role/content conversations from Parquet, renders the tokenizer's chat template, and packs IDs with roles, document IDs, and positions. With `loss_role=Role.ASSISTANT`, `LMObjective` counts assistant **targets**, not every token in the prompt. Dew checks incremental prefix rendering; a template that changes an earlier tokenization boundary can be rejected. A string called “assistant” is not itself evidence of a correct mask.

**DPO** learns from chosen/rejected responses to the same prompt. `PreferencePairs` takes explicitly tokenized rows and masks. Unlike an LM prediction length, its `seq_len` is the **full ID row width**: the demo uses width 4 and a DPO objective prediction length of 3. Dew freezes the starting policy as the reference, then compares policy/reference completion log-probabilities. Supply masks yourself; omitted masks mean every token counts as completion. DPO validation's chosen-response perplexity is not a preference win-rate evaluation.

**GRPO** generates groups of responses, scores them, and updates the policy from relative rewards. `Prompts` carries the prompt and reward metadata. `SampledRollout` runs outside the compiled optimizer step, generates `groups` completions, calls `reward(data_source, completion, ground_truth, extra_info)`, and produces IDs, old log-probabilities, advantages, and response masks. `GRPOObjective` combines the clipped policy surrogate with an optional k3 KL penalty against the frozen reference. `sample="rloo"` selects leave-one-out advantages rather than the group-normalized form.

A verifiable reward can make a particular task an RLVR experiment, but a reward callback is not an end-to-end RLVR platform. The default decoder returns token-ID text unless you supply a decoding function. Prompt padding, EOS handling, response lengths, and reward normalization need attention before real text use. The local rollout does not provide sandbox isolation, multi-turn tools, a value model, asynchronous actor fleets, or a serving/weight-update system.

The array-level `dew.rl` module supplies `gae`, `group_advantage`, `rloo_advantage`, `masked_mean`, `masked_whiten`, `clipped_surrogate`, `k3_kl`, `preference_logsigmoid`, `sequence_log_ratio`, `token_log_ratio`, and `token_mean`. Their availability is not a claim of complete PPO, GSPO, or SAPO training recipes. FlowGRPO integration is separate branch work; a Gaussian flow model plus a scalar reward is not enough to implement it.

`recipes.chain.Recipe` sequences SFT/DPO/GRPO stages sharing a decoder. Dataset types select objectives, each stage starts a fresh optimizer, and the preceding policy initializes the next stage/reference. Its stage name labels output; it does not infer masks from strings. The chain exposes fewer rollout controls than constructing `SampledRollout` yourself. See [post-training](docs/concepts/post_training.md) for layouts, memory costs, reward semantics, and the narrow TRL/verl comparison evidence.

## Diffusion, flow matching, and latent models

A diffusion run needs a denoising model and a **process** describing how clean data becomes noisy and what the model should predict. The process connects training-time targets with sampling-time conversion. A preset is a configuration value that builds that process; `Flow()()` means “construct the Flow preset, then build its process.”

| Component | Supplied choices | How to use the distinction |
|---|---|---|
| Continuous presets | EDM, Karras, Cosine, Flow, Sqrt | Select a compatible noise/time convention, target/preconditioning, and weighting. |
| Prediction/weighting components | Noise, clean-sample, velocity/flow conventions; P2 and Min-SNR-related weighting where configured | Changing a target or weight changes the trained objective, not just sampling speed. |
| Solvers | DDPM, DDIM, Euler, Euler ancestral, Heun, RK4, MultiStepDPM | Different numerical methods and process requirements; the name does not guarantee every schedule/variant from another library. |
| Guidance | `CFG(scale, interval)` | Combine conditional/unconditional predictions, optionally over only part of the sampling path. |
| Conditions | `InputSpec`, `Condition`, CLIP text, T5 text, CharTable | Connect a model keyword to an encoder and a tokenized batch field. Pretrained towers need their assets. |
| Autoencoders | Stable Diffusion Flax VAE and autoencoder interfaces/modules | Train in latents when desired; match spatial downsampling, channels, and scale/shift. |

The lower-level schedule classes include discrete linear, cosine, and exponential beta schedules; continuous/generalized cosine and square-root schedules; sigma-parameterized EDM/Karras variance-exploding schedules; and the flow-matching time/resolution-shift convention. Prediction transforms cover epsilon (noise), direct clean-sample prediction, velocity, flow matching, and Karras preconditioning. `ScheduleWeighting` and `MinSNR` compose weighting with a compatible process. These are available mathematical components, not an instruction to mix an arbitrary schedule, transform, and solver.

The registered image/video backbones are:

| Registry name | Structure |
|---|---|
| `unet` | Convolutional UNet with configurable per-stage attention |
| `unet_3d` | Video UNet with 3D/spatiotemporal computation |
| `uvit`, `simple_udit` | U-shaped transformer designs |
| `simple_dit` | Patch-based diffusion transformer |
| `simple_mmdit` | Dual-stream text/image transformer |
| `hierarchical_mmdit` | Multi-resolution dual-stream transformer |
| `hybrid_dit` | Transformer blocks combined with S5 state-space computation |
| `video_dit` | Factorized spatial and temporal attention |

A “DiT” or “SD3-style MMDiT” component is not a claim that an arbitrary pretrained DiT/SD3 checkpoint translates into it. For example, Dew's `simple_dit` uses EDM-style random Fourier time features rather than copying DiT's original sinusoidal time embedding.

During conditional training, the objective can drop conditions on some examples so the model learns the unconditional branch used by classifier-free guidance. During sampling, `process.denoiser` combines model prediction with the chosen process, `sample` advances a solver over its time grid, and guidance changes the denoiser's prediction. Sampling steps and optimizer updates are separate counts: the demo trains three updates and samples with four solver steps.

`TextToImage` packages conditioning and decoding, and `TextToImage.from_run` rebuilds compatible diffusion recipes from their recorded configuration and checkpoint. A directory produced by a bare `Checkpoints` call lacks that recipe record. Frozen text towers and autoencoders remain state with real memory costs. The [diffusion guide](docs/guides/diffusion.md), [recipes](docs/recipes.md), and [historical gallery](docs/gallery.md) cover current construction, dataset-backed runs, and older FlaxDiff results respectively.

### Train a 64×64 diffusion model on a GPU

This is a dataset-backed starting configuration for unconditional Oxford Flowers generation, beyond the 8×8 mechanics demo. It needs a CUDA-capable GPU, a compatible CUDA JAX installation, disk space for the prepared dataset/checkpoints, and the `tfds` extra. **The preparation and GPU training below were not executed for this guide.** No throughput, memory-fit, convergence, or image-quality result is claimed for this configuration.

From the installed checkout, prepare Oxford Flowers once. These commands install optional packages and can download the dataset; inspect its access conditions first. Use a new TFDS directory if you already have a TFRecord preparation: Dew's random-access loader needs ArrayRecord rather than that format. The label file is created explicitly because `OxfordFlowers` reads label names when processing records, even in an unconditional run.

```bash
uv pip install -e ".[tfds]"
export TFDS_DATA_DIR="$HOME/dew-data/tfds-arrayrecord"
python - <<'PY'
import os
from pathlib import Path
import tensorflow_datasets as tfds

data_dir = Path(os.environ["TFDS_DATA_DIR"]).expanduser()
builder = tfds.builder("oxford_flowers102", data_dir=str(data_dir))
builder.download_and_prepare(file_format="array_record")
labels = data_dir / "flowers102-labels.txt"
labels.write_text("\n".join(builder.info.features["label"].names) + "\n")
print("Prepared", builder.info.full_name, "and", labels)
PY
```

Save this complete script as `train_flowers64.py`. It trains in pixels, without a CLIP/T5 tower or VAE, so the training script requires no pretrained model weights. `DiffusionRunConfig` records the actual model, process, data, optimizer, and trainer choices in `run.json`.

```python
import os
from pathlib import Path

import jax
import wandb

from dew.config import ModelConfig, OptimConfig, TrainerConfig, Wandb
from dew.data import Loading, OxfordFlowers
from dew.diffusion.presets import EDM
from dew.objectives.diffusion import DiffusionRunConfig
from dew.sampling import Heun
from dew.training.runtime import prepare_process


def main():
    data_dir = Path(os.environ["TFDS_DATA_DIR"]).expanduser()
    config = DiffusionRunConfig(
        model=ModelConfig(
            "simple_dit",
            {"patch_size": 4, "emb_features": 256, "num_layers": 6,
             "num_heads": 4, "mlp_ratio": 4},
            dtype="bfloat16", attention_impl="auto",
        ),
        data=OxfordFlowers(
            image_size=64, augmentation="none", val_batches=4,
            labels=str(data_dir / "flowers102-labels.txt"),
            loading=Loading(workers=4, threads=4, read_buffer=16, worker_buffer=2),
        ),
        preset=EDM(), sampler=Heun(), sampling_steps=32,
        text=None, autoencoder=None, guidance=None, val_metrics=[],
        optim=OptimConfig(learning_rate=2e-4, weight_decay=0.01, clip_grads=1.0),
        trainer=TrainerConfig(
            name="flowers64", checkpoint_dir="runs", batch_size=32, steps=10000,
            log_every=50, eval_every=500, checkpoint_every=500, multi_host=False,
            wandb=Wandb(project="dew-flowers", offline=True),
        ),
    )
    prepare_process(
        config.trainer.wandb, config.trainer.multi_host,
        config.trainer.xla_flags, config.trainer.compilation_cache_dir,
    )
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("This configuration expects a GPU-backed JAX installation")
    objective = config.build()
    data = config.data.load(batch=config.trainer.batch_size,
                            tokenize=objective.inputs.tokenize)
    try:
        state = config.train(objective, data, name="flowers64",
                             metrics=config.build_eval_metrics())
        print("Completed training at step", int(state.step))
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
```

With your CUDA JAX environment activated and `TFDS_DATA_DIR` still set, launch it with:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda python train_flowers64.py
```

The global batch is 32. Four validation batches reserve 128 records from the source before training; this adapter's holdout is not the official Oxford Flowers benchmark split. `augmentation="none"` keeps both training and validation preprocessing deterministic here, since this dataset specification shares its augmentation setting across the two streams. A production data recipe can make a different split/augmentation choice deliberately.

`eval_every=500` schedules events, and the offline W&B tracker provides a preview consumer. The configured solver generates display samples at those events and at normal completion. `val_metrics=[]` is intentional: this example does not load a FID/CLIP feature extractor or present a preview as a population-level quality score. W&B runs offline and needs no online tracking account for this configuration, but its local files still consume disk.

Periodic checkpoints and the final state go under `runs/flowers64`, alongside the separately written `run.json`; the Grain-backed loader contributes the data position. Repeating the same compatible script continues toward total step 10000 rather than requesting 10000 additional updates. Choose a new run name for a new experiment. Before committing to the full duration, change `steps` to a short run, inspect actual GPU memory and losses, and confirm checkpoint access. Increase the total target to continue only after that check.

This is unconditional generation, so captions do not guide its samples. Adding text conditioning requires an explicit tokenizer/encoder and its weights, compatible model inputs, and an unconditional branch for CFG. A lower loss or attractive preview alone does not establish held-out image quality; add a separately specified evaluation population and metric when measuring it.

## Representation learning with JEPA

A Joint-Embedding Predictive Architecture learns representations rather than generating pixels. The **context encoder** processes visible patches, the **predictor** estimates representations at hidden positions, and the **target encoder** supplies targets with gradients stopped. The target encoder follows an EMA of selected context-encoder variables.

Dew registers `jepa_encoder`, `jepa_video_encoder`, and `jepa_predictor`. `JepaObjective` connects them with a mask function and a sample field. Multi-block masks operate on the patch grid, so patch size, grid dimensions, predictor inputs, and target-area/aspect settings must agree. The video path also needs compatible temporal geometry.

Training loss measures prediction error in representation space. Similar representations for all images can yield misleadingly low error, so inspect representation statistics and evaluate a downstream task. Dew supplies linear and k-nearest-neighbor probes over representation artifacts. A probe needs appropriate labeled validation data and an evaluation schedule; it is not automatically enabled by training the encoder.

An encoder trained with JEPA is not yet a policy, planner, or tool-using agent. Read [representation learning](docs/guides/representation-learning.md) for an offline example and guidance on evaluating embeddings.

## Prepare data and batches

`Dataset(train, val, records, batch)` is the value a trainer consumes. `train` is a factory that opens an iterator, `val` opens one finite validation pass or is `None`, `records` is a known record count or `None`, and `batch` is global across JAX processes. A factory should return a fresh iterator owned by that run, not share one iterator across independent trainers.

| Data specification | Input and batch behavior |
|---|---|
| `TokenWindows` | Reads token binaries and `meta.json`; returns overlapping next-token windows under `text`. |
| `PackedTokens` | Packs document chunks with segment IDs and reset positions, preventing cross-document targets/attention. |
| `ChatMessages` | Renders Parquet conversations with a tokenizer's chat template and packs role-aligned tokens. |
| `PreferencePairs` | Reads Parquet or in-memory JSON preference rows and preserves the pair dimension. |
| `Prompts` | Reads IDs, strings, or messages plus reward metadata for online sampling. |
| `OxfordFlowers`, `HFImages` | Public-dataset adapters with image/caption processing; preparing them may download data. |
| ArrayRecord image specifications | CC3M/CC12M, LAION/COCO/COYO-related subsets and combinations, DiffusionDB, CombinedMsml612, CombinedAesthetic, Combined30M | 
| `OnlineImages`, `CombinedOnline` | URL-backed image streaming; network access and source availability remain operational requirements. |
| `LocalVideos`, `VoxCeleb2` | Video files with configured frame extraction and caption/identity conventions. |

The named large-image specifications describe particular shard layouts and defaults. They do not grant access to the original author's buckets or redistribute the datasets. Prepare your own paths, credentials, and licenses. `Loading` controls host workers, read threads, and buffering; those settings affect memory and throughput, not the number of model devices. Its defaults suit larger data workloads, so the demo uses zero workers and one read thread.

A host-side prefetch iterator can overlap reading and device transfer with computation. The trainer owns and finalizes the iterators it opens, including failure paths. Blocking third-party readers must cooperate with shutdown; closing a Python iterator cannot forcibly stop an arbitrary blocked network read. Grain's upstream lifecycle also has limits. See [data](docs/concepts/data.md) before tuning queues or interpreting a memory plateau as a model allocation.

For real text, `tools/tokenize_text.py` writes train/validation binaries and metadata from UTF-8 files. Its byte tokenizer downloads nothing. A Hugging Face tokenizer can load assets from a local directory or the Hub. Keep tokenizer identity, vocabulary, special tokens, packing rules, and held-out split decisions with the experiment. The repeating-symbol demo deliberately shares a pattern across splits; it is a mechanics example, not a generalization benchmark.

## Optimization, precision, and performance

You can pass an Optax gradient transformation directly to `Trainer`, as the demo does. Recipe `OptimConfig` exposes Adam, AdamW, LAMB, Muon, and MuonClip plus learning-rate/schedule, weight-decay, and gradient-clipping settings. These choices are not interchangeable defaults for all architectures.

Muon splits matrices suited to its orthogonalized updates from embeddings, heads, and normalization parameters that receive AdamW-style updates. MuonClip adds QK-Clip using attention statistics supplied by a compatible objective/model. Naming the optimizer does not cause an arbitrary custom model to emit those statistics. MoE balancing and auxiliary losses likewise require compatible routers and state updates; importing a family configuration does not reproduce its original lab's entire optimizer/data curriculum.

Use bfloat16 computation where the backend and numerical path support it, while accounting for float32 master parameters and reductions. Qwix int8/fp8 training is optional and experimental. It wraps selected model computation; it is not the same as keeping every tensor quantized or loading a quantized checkpoint. FP8 block and MXFP4 checkpoint loaders dequantize their stored values, so RAM/device-memory use can be much larger than checkpoint file size. Quantized KV-cache serving is not supplied by that loader.

For gradient accumulation, the denominator matters. Averaging two microbatch means is not a global token mean when one batch contains more valid targets. The revised objective-statistics/transaction contract is being integrated to handle normalization, accepted work, state effects, and resumable partial windows. Do not claim large-batch equivalence from a scalar loss alone; use the documented contract for your revision. Rematerialization, which trades recomputation for activation memory, is also integration work rather than a performance guarantee in this baseline.

### Measure a representative run

A JAX computation has compilation and execution costs. The first update includes tracing/compilation and setup; later updates may reuse an executable or persistent compilation cache. Different shapes or static settings can compile again. Time synchronized results when measuring execution, and distinguish the compiled model step from input loading, validation, checkpointing, and end-to-end goodput.

For context, the recorded **2026-09-02** `small` benchmark at commit `6b0f119` used one RTX 4080 16 GiB, JAX/jaxlib 0.11.1, Flax 0.12.9, Optax 0.2.8, bfloat16 models, Adam, two warmups, and 100 measured steps:

| Model | Example shape / batch | Recorded step time | Recorded compile time |
|---|---|---:|---:|
| `simple_dit` | 64×64×3 / 16 | 7.9 ms | 9.9 s |
| `unet` | 64×64×3 / 16 | 16.4 ms | 36.4 s |
| `causal_transformer` | 512 tokens / 16 | 83.0 ms | 10.1 s |

These are historical measurements for those model sizes, not current-release promises or comparisons with another framework. The [benchmark record](docs/benchmarks.md) includes parameter counts, memory, methodology, later reruns, and regressions. [Performance notes](docs/performance.md) explain measured trade-offs such as vocabulary tiling and quantization. Neither page establishes large-model TPU throughput from a small GPU result.

## Distribute a training run

A JAX **device mesh** gives names to groups of devices. Dew separates `MeshSpec`, which chooses the mesh factors, from `Layout`, which maps model logical axes onto those factors. Unallocated devices form the data-parallel axis.

| Mesh axis | Work it partitions | Practical consideration |
|---|---|---|
| `data` | Different examples across replicas | The global batch must divide across the participating process/device layout. |
| `fsdp` | Sharded model/optimizer storage | Sharding saves some memory but requires communication for computation. |
| `expert` | Routed experts | Expert dispatch, imbalance, and collective cost can dominate. |
| `tensor` | Compatible tensor dimensions within model layers | Width/head geometry and model axis declarations must permit the split. |
| `sequence` | Sequence attention work | Long-context masks, packing, and supported attention paths matter. |
| `stage` | Groups of model layers in a pipeline | GPipe-style execution needs compatible microbatches and does not imply master parameters are persistently stage-sharded. |

FSDP means *fully sharded data parallelism*: replicas can share model-state storage by partitioning it. Tensor parallelism divides operations within layers; pipeline parallelism divides layers over stages. They address different bottlenecks and carry different communication/memory costs. A legal mesh is not automatically an efficient one.

`Layout` uses logical names such as embedding, heads, vocabulary, and experts. A dimension must be divisible by the mesh axis assigned to it. Small tensors may remain replicated; `min_shard` and `tolerance` control layout decisions/checks. Custom modules may need logical-axis declarations. There is no promise that an arbitrary Flax tree acquires an optimal layout without annotation.

Before using multiple hosts, initialize the JAX process pool before creating device arrays, launch the same program on every worker, make data/checkpoint paths available to all processes, and check the device counts and process indices. The recipes provide process setup; library code is responsible for its own environment. Verify a short collective training run and a restore on the actual cluster before a long allocation. CPU device simulations are valuable numerical checks, but they do not test inter-host networks or accelerator kernels.

Dew has XLA attention, a cuDNN path for supported GPU inputs, and a TPU Pallas attention path. Kernel shape/dtype restrictions still apply. [Distributed training](docs/concepts/distributed.md) explains supported combinations and placement details. [The TPU guide](docs/tpu.md) covers CLI previews, permissions, cost, setup, and verification boundaries; `dew-tpu --dry-run` is not a hardware test.

## Evaluate, checkpoint, resume, and export

### Evaluation needs data, a schedule, and a consumer

`Objective.evaluate` supplies scoring artifacts for each coordinated batch; `Objective.preview` produces a separate display sample when needed. Artifacts include `TokenScores`, `ImageGrid`, `VideoGrid`, `Representations`, and `TextSamples`. Metrics consume the artifact type they declare rather than assuming every task returns a scalar or image.

A call to `fit` must supply validation data and an explicit `eval_every` cadence to schedule evaluation. Passing `metrics=(metrics.perplexity(),)` alone does not enable it. A scoring consumer or tracker preview determines whether there is useful evaluation work to perform. The demo sets validation data, cadence, and a perplexity metric together.

Metrics aggregate sufficient statistics over the consumed validation population and finalize once. Perplexity comes from weighted token loss, not an unweighted average of per-batch perplexities. Dew also supplies FID, CLIP distance, CLIPScore, PSNR, SSIM, and representation probes. The legacy `clip` metric is a distance convention; use the documented `clip_score` definition when you mean standard CLIPScore. FID and text/image metrics can require separate pretrained feature-extractor assets.

A few generated previews are not a full generative-quality evaluation. PSNR/SSIM require meaningful paired comparisons; arbitrary generated images and unrelated real images do not form such a task. Uneven multi-process validation shards can leave an unscored tail when execution uses a coordinated prefix. Report the actual examples/population consumed, not merely the intended dataset size. The [evaluation guide](docs/guides/evaluation.md) explains event randomness, scoring/previews, metric reduction, and failure coordination.

### A checkpoint and a run configuration are different records

`Checkpoints` uses Orbax to save numerical training state, along with a data position when the iterator provides it. Periodic checkpointing requires a restorable iterator with `get_state`/`set_state`. The token loader in the demo provides that contract. An endlessly repeated Python batch does not automatically become a resumable data stream.

A checkpointer can write a final state on normal completion even if no periodic checkpoint interval is set. `checkpoint_every=None` therefore does not mean “this trainer cannot write files” when you configured a checkpointer. The trainer waits for its pending writes; a failed write is not a successful saved run.

For ordinary continuation, rebuild the same model/objective/optimizer and point a new trainer at the checkpoint directory. `fit(..., steps=24)` targets total step 24; after restoring step 20, it does not request 24 additional steps. Changing optimizer structure, accumulation, variable layout, or dataset contents is a different compatibility question. Pre-1.0 state layouts can change, and overflow/partial-window recovery requires the repaired contract rather than assumptions from the finite single-update demo.

`run.json` comes from `RunConfig.save` or a recipe, not from `Checkpoints` itself. Keep both the numerical checkpoint and its construction context: model/objective fields, tokenizer or encoder assets, dataset revision, source commit, dependency versions, and device/precision settings. A saved JSON path does not archive the data at that path. See [checkpoint and resume guidance](docs/guides/checkpoints.md) before moving between machines or versions.

### Export for a specific consumer

`save_params` writes a tensor tree. `save_hf_layout` writes `model.safetensors` plus `config.json`, without translating arbitrary tensor names or architectural configuration. The family-aware `save_pretrained_decoder` performs the translation for decoder cases it accepts. Keep the matching tokenizer/processor assets with the export; a tensor file alone is not a complete chat model.

Loading that directory in Transformers, vLLM, or another consumer is a separate verification step. Check configuration recognition, tensor coverage, numerical outputs, preprocessing, and generation behavior in that consumer. Dew's import coverage, export coverage, and a serving engine's model support are three different lists.

## Configure and run recipes

Use a recipe when you want a saved run description and command-line configuration rather than assembling each object by hand. `recipes/lm/train.py`, `recipes/diffusion/train.py`, and `recipes/jepa/train.py` build their task's objects and call the common training path. The Python-only post-training chain is separate.

| Shared configuration | Meaning | Example of the CLI naming convention |
|---|---|---|
| `ModelConfig` | Architecture, constructor fields, compute dtype, attention selection | `--model.architecture causal_transformer` |
| Dataset specification | Source/path, sample geometry, loading/packing settings | `data:token-windows --data.seq-len 128` |
| `OptimConfig` | Optimizer, learning rate/schedule, weight decay, gradient clipping | `--optim.learning-rate 0.0001` |
| `TrainerConfig` | Run length, global batch, cadence, mesh/layout, tracker and process settings | `--trainer.steps 1000` |

The CLI is generated by tyro from dataclasses. Dotted flags select nested fields; subcommands select registered value types. Architecture fields go in one JSON object, with Python-style JSON keys such as `num_layers`. Use `--help` for the actual installed parser instead of translating flags from a different trainer.

From a checkout, these commands inspect the entry points without training or downloading a dataset:

```bash
python recipes/lm/train.py --help
python recipes/diffusion/train.py --help
python recipes/jepa/train.py --help
```

Specify either `trainer.steps` or `trainer.epochs`, not both. Epoch-based schedules require a known dataset size. The LM recipe trains token windows/packed tokens and accepts `lm` or `masked_diffusion`; it does not turn a preference file into DPO by changing a string. A diffusion recipe adds preset, solver, guidance, conditions/autoencoder, and validation choices. JEPA adds predictor geometry, masks, target momentum, and probes.

Unset W&B tracking logs to the terminal. A configured W&B project enables tracking; offline tracking only changes the tracker connection, not any dataset/model downloads elsewhere. Profiling and persistent compilation cache settings are separate. The recipe writes its configuration next to checkpoints, but your environment and data remain external records.

The [recipe walkthrough](docs/recipes.md) sets up a corpus file, token files, and a complete small CLI run. Start there before substituting a Hub checkpoint or a large dataset. The existing `train_diffusion.py`, `train_lm.py`, and `train_jepa.py` examples are additional workflows with their own input requirements; unlike `readme_demo.py`, some need external data or pretrained towers.

## Write your own objective

You do not need to register a custom objective just to pass it to `Trainer`. Its minimum job is to initialize a Flax variables mapping and describe a differentiable loss. For a simple scalar task, return the scalar and `Aux(metrics=...)`. Optional evaluation and state updates remain explicit.

Here is a complete small objective and dataset, using the same scalar API as ordinary Flax training. It learns `y = 2x + 1`; it is independent of the capability-demo variables:

```python
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew import Trainer
from dew.data import Dataset
from dew.objectives.base import Aux, Objective


class Regression(Objective):
    def __init__(self):
        self.model = nn.Dense(features=1)

    def init(self, key):
        return self.model.init(key, jnp.zeros((1, 1), dtype=jnp.float32))

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        mse = jnp.mean((prediction - batch["y"]) ** 2)
        return mse, Aux(metrics={"mse": mse})


x = np.linspace(-1, 1, 32, dtype=np.float32).reshape(32, 1)
batch = {"x": x, "y": 2 * x + 1}
data = Dataset(train=lambda: itertools.repeat(batch), val=None,
               records=32, batch=32)
objective = Regression()
state = Trainer(objective, optax.sgd(0.1), key=jax.random.key(0)).fit(
    data, steps=100, log_every=50)
prediction = objective.model.apply(state.params, x)
print("Training MSE:", float(jnp.mean((prediction - batch["y"]) ** 2)))
```

The optimizer updates the inner `params` collection. If a Linen call returns mutable collections such as batch statistics, put their replacements in `Aux.variables`; do not use that field to replace optimizer-owned parameters. A frozen encoder can live outside the learned collection, and an `EMASpec` can select the subtree to average.

For masked losses or several differently normalized terms, preserve the statistics needed to reduce the objective correctly. A plain scalar cannot tell a trainer how many target tokens contributed to it. The incoming `Mean`/composite-statistics contract keeps simple scalar authoring while making that normalization explicit; follow the [objective guide](docs/concepts/objectives.md) for the integrated API rather than averaging already-normalized means by habit.

A supplied objective can also require a model protocol beyond `__call__`. `LMObjective` reads hidden states and a vocabulary head; JEPA connects encoders/predictors with patch-index geometry; diffusion passes noisy samples, noise levels, and conditioning keywords. Direct Flax constructors give more static type information when the class is known. Registries such as `models.build` are useful when a configuration chooses the class dynamically, and validate the fields that class declares.

For advanced changes, a rollout hook transforms a host batch before the compiled update, whereas a custom step replaces optimization behavior itself. Those are different responsibilities. Replacing the step means owning the numerical/state contract, not merely choosing another loss. The [core API reference](docs/reference/core-api.md) describes the available interfaces and buffer-donation rules.

## How Dew compares with other tools

These tools work at different layers. The comparisons below concern their documented workflows, not measured speed, quality, or feature parity. None of the external workflows was executed to produce the offline Dew example's numbers.

| Tool | Its usual role | Dew's corresponding boundary | A reason to choose that tool instead |
|---|---|---|---|
| [PyTorch](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) | Tensor/autograd framework, modules, optimizers, and user-controlled training loops | Dew sits **above JAX/Flax/Optax**, with a task objective and shared trainer | You want the PyTorch ecosystem, its kernels/integrations, or direct control of an existing PyTorch model. |
| [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/core-api/lightning_module) | Organizes PyTorch training/validation/optimizer hooks and automates loop/device work | An `Objective` separates computation from Dew's trainer and explicit variables/state | You want Lightning's strategies, callbacks, logging integrations, and established PyTorch workflow. |
| [Transformers](https://huggingface.co/docs/transformers/main_classes/trainer) | Broad pretrained-model, tokenizer/processor, generation, and Trainer ecosystem | Selected HF-family translation plus Dew objectives and local generation | Your priority is a released model's supported processor/generation/task API or its existing Trainer recipe. |
| [Diffusers](https://huggingface.co/docs/diffusers/training/overview) | Diffusion pipelines, pretrained components, schedulers, and task-specific training examples | Composable JAX diffusion processes/objectives and backbones | You need a particular released image/video pipeline, LoRA/DreamBooth/ControlNet workflow, or its pretrained-component ecosystem. |
| [MaxText](https://github.com/AI-Hypercomputer/maxtext) | JAX LLM training/reference implementation with model- and hardware-specific scaling work | A smaller common training framework spanning LM, diffusion, and JEPA | You need its documented large-scale LLM training/post-training paths and model/hardware qualification. |
| [vLLM](https://github.com/vllm-project/vllm) | Inference and serving: paged attention memory, continuous batching, distributed inference, request APIs | `generate` is a local fixed-batch sampling function, not a server | You need throughput/latency-oriented serving, streaming requests, supported model dispatch, or structured-output/tool-call serving features. |
| [Ollama](https://docs.ollama.com/) | Running and distributing models through local/cloud applications and APIs | Dew changes learned parameters through training and can export some models | You want to run an existing model locally or integrate its API without constructing a training system. |

### The same training problem, organized differently

For a supervised regression task, **plain PyTorch** usually puts the model call and loss inside a user loop, then calls backward, an optimizer step, and gradient reset. **Lightning** moves task computation into `training_step`, validation into `validation_step`, and optimizer construction into `configure_optimizers`; its trainer handles the surrounding loop. **Dew** puts Flax initialization and loss in an objective, passes the optimizer separately, and returns an explicit updated state. All three can express the same mathematical problem. Fewer visible loop lines do not establish faster computation or less complexity overall.

For language-model fine-tuning, **Transformers Trainer** couples training arguments, a model with the expected forward/loss interface, a data collator, and tokenizer/processor conventions. **Dew** uses compatible translated/built variables, a token/chat dataset, an LM objective, and an Optax optimizer. You must still preserve chat-template, label-shift, padding, and special-token semantics. Dew's smaller API does not replace Transformers' model catalog or mean a checkpoint needs no adaptation.

For diffusion, **Diffusers** documents self-contained, task-specific training scripts whose preprocessing and training loops users can adapt. **Dew** puts process/target construction into `DiffusionObjective` and shares trainer behavior with other modalities. Dew still needs task-specific preprocessing and compatible model/condition/autoencoder choices. Neither a similar solver name nor a matching-looking module is proof of the same complete pipeline.

**MaxText** is the closer JAX training comparison, and it now documents Flax NNX, broad model work, scalable pretraining, and post-training including SFT/GRPO/GSPO. Dew uses Linen and exposes a common objective boundary across several modalities. It does not inherit MaxText's engineering or deployment evidence by implementing similar mesh axes, optimizers, or model components. MaxText itself distinguishes supported releases from its evolving main branch; compare specific revisions and workloads.

**vLLM and Ollama** generally begin where this training discussion ends: running a learned model for users. A Dew export still needs a supported architecture, valid tensor/metadata translation, tokenizer/processor assets, and consumer-side verification. Do not choose Dew's `generate` function as a substitute for request scheduling, model packaging, or a server API that it does not implement.

## Installation options and learning paths

The distribution name is `dew-ml`; imports use `dew`. The project declares Python 3.11 or newer. The demo above was exercised with the project's Python 3.12 environment. Use a virtual environment and record resolved dependency versions for work you need to reproduce.

For package-only use rather than editing a checkout:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install "dew-ml @ git+https://github.com/AshishKumar4/dew"
```

Recipes, tools, and example scripts live in the repository; installing the package alone does not place those files in your current directory. A copied standalone Python example can import the installed package. CUDA and TPU JAX installations require the appropriate platform packages/drivers; follow [JAX's installation instructions](https://docs.jax.dev/en/latest/installation.html) and [Dew installation](docs/installation.md), rather than adding an old CUDA wheel command from a different machine.

Optional extras include `interop` for safetensors conversion, `tfds` for TensorFlow Datasets-backed sources, `streaming` for HF dataset streaming, `av` for video readers, and `metrics` for additional metric dependencies. Installing a metric's Python packages does not necessarily cache its pretrained feature extractor. Qwix and specialized kernels have their own optional runtime requirements. No offline-tracker flag prevents unrelated model or dataset downloads.

Choose a learning path:

- **New to JAX/Flax:** [your first training run](docs/getting-started.md), then [objectives](docs/concepts/objectives.md), [data](docs/concepts/data.md), and [core APIs](docs/reference/core-api.md).
- **Language-model work:** [language modeling](docs/concepts/language_models.md), [post-training](docs/concepts/post_training.md), [MoE](docs/concepts/moe.md), and [family limits](docs/reference/model-families.md).
- **Images/video:** [diffusion](docs/guides/diffusion.md), [representation learning](docs/guides/representation-learning.md), [recipes](docs/recipes.md), and [evaluation](docs/guides/evaluation.md).
- **Longer or distributed runs:** [checkpoints](docs/guides/checkpoints.md), [distributed training](docs/concepts/distributed.md), [performance](docs/performance.md), and [TPUs](docs/tpu.md).

The [documentation index](docs/index.md) collects these guides, while [the module index](docs/api.md) locates public packages. Older notebooks under `tutorials/` are historical/revision-dependent material; the current guides and offline script are the starting point. [Coming from FlaxDiff](docs/from-flaxdiff.md) explains moved responsibilities and why old checkpoints are not automatically compatible.

## Contributing, attribution, and citation

Read [CONTRIBUTING.md](CONTRIBUTING.md) for development and reporting expectations. For a model or numerical issue, include the Dew commit, package versions, model/configuration, relevant shapes/dtypes, backend, and a bounded reproduction. For a deployment claim, include the actual hardware/process topology and operations exercised. A passing tiny fixture should remain described as a tiny fixture.

Dew uses the [MIT license](LICENSE), with attribution and applicable notices for adapted upstream components. The RL math includes Apache-2.0-derived Tunix/verl code; the Flax VAE and attention ancestry includes Diffusers, and the FID implementation draws from jax-fid. Model weights and datasets retain their own licenses and access conditions. [References and attribution](docs/references.md) records the papers, projects, and earlier tutorials behind the implementation.

For research use, cite the underlying model/method and identify the Dew revision and configuration that produced the result. Dew builds on JAX, Flax, Optax, Orbax, and Grain. The earlier FlaxDiff experiments received support from Google TPU Research Cloud; that historical support is not current multi-host qualification or a promise of available cloud resources.

The project began as [FlaxDiff](https://github.com/AshishKumar4/FlaxDiff), focused on diffusion experiments. Dew separated optimizer updates, data loading, sharding, and checkpoint handling from diffusion-specific computation so language-model and representation-learning objectives could use them too. The [FlaxDiff history guide](docs/from-flaxdiff.md) records the moved responsibilities and checkpoint boundaries.
