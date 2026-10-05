# Language models

`LMObjective` trains a decoder to predict the next token in a row of token IDs. It feeds all but the last token to the model and scores predictions against the following tokens with cross entropy. The decoder is usually `CausalTransformer` from `dew.nn.backbones`, and `dew.sampling.generate` generates text from the trained weights. `Pretrained.load` loads published checkpoints; `MaskedDiffusionObjective` and `BlockDiffusionObjective` train non-autoregressive language models.

## Example

The example writes a small corpus of token files, trains a two-layer decoder on it and generates a continuation. It uses `ByteTokenizer`, which maps each UTF-8 byte to one of 256 IDs, so it downloads nothing.

```python
import json
from pathlib import Path

import numpy as np

from dew.data import ByteTokenizer

tokenizer = ByteTokenizer()
stories = [
    "Once upon a time, there was a little girl named Lily. She had a red ball.",
    "One day, Lily saw a big dog in the park. The dog wanted to play.",
    "Lily threw the ball and the dog ran after it. They played all day.",
    "At night, Lily went home. She was happy and tired.",
]
ids = np.array(tokenizer.encode(" ".join(stories * 200)), dtype=np.uint8)
held_out = len(ids) // 20
corpus = Path("data/stories-byte")
corpus.mkdir(parents=True, exist_ok=True)
ids[:held_out].tofile(corpus / "val.bin")
ids[held_out:].tofile(corpus / "train.bin")
(corpus / "meta.json").write_text(json.dumps({
    "tokenizer": "byte", "vocab_size": tokenizer.vocab_size, "dtype": "uint8",
    "train_tokens": len(ids) - held_out, "val_tokens": held_out, "eos_id": None}))
print(len(ids) - held_out, "training tokens,", held_out, "validation tokens")
```

```text
48830 training tokens, 2569 validation tokens
```

This is the layout `dew tokenize` writes: `train.bin` and `val.bin` hold the token IDs back to back, and `meta.json` records the tokenizer, vocabulary size, dtype and counts. The validation split is the head of the stream.

```python
import jax
import optax

from dew import Trainer
from dew.data import Loading, TokenWindows
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective, Perplexity
from dew.sampling import Sampling, generate

data = TokenWindows(path="data/stories-byte", seq_len=64,
                    loading=Loading(workers=0)).load(batch=16)
model = CausalTransformer(vocab_size=tokenizer.vocab_size,
                          emb_features=64, num_layers=2, num_heads=2, mlp_features=256,
                          max_seq_len=128)
objective = LMObjective(model, seq_len=64)
lm_state = Trainer(objective, optax.adamw(3e-3), key=jax.random.key(0)).fit(
    data, steps=300, log_every=100, eval_every=300, metrics=(Perplexity(),))
continuation = generate(model, lm_state.variables, [tokenizer.encode("One day, Lily")],
                        max_new_tokens=40, key=jax.random.key(1),
                        sampling=Sampling(temperature=0.0))
print(tokenizer.decode(continuation.tokens[0]))
```

```text
Training CausalTransformer from step 0 to 300: 147,904 parameters, on 1 × cpu, batch 16, float32
step 100/300  loss 0.09353  ce 0.09353  perplexity 1.098  token_accuracy 97.2%  step_time_ms 29.97  samples_per_sec 533.9  accepted 100.0%  0:00:05 left
step 200/300  loss 0.06881  ce 0.06881  perplexity 1.071  token_accuracy 97.7%  step_time_ms 34.53  samples_per_sec 463.4  accepted 100.0%  0:00:03 left
step 300/300  loss 0.06941  ce 0.06941  perplexity 1.072  token_accuracy 97.8%  step_time_ms 36.10  samples_per_sec 443.3  accepted 100.0%
eval val at step 300: perplexity 1.065 (32 records in 0.93 s)
Trained 300 steps in 0:00:14: first step after 2.76 s, then 30.0 step/s
72.9% of the wall time in steps, final loss 0.06941
One day, Lily saw a big dog in the park. The dog want
```

`TokenWindows(seq_len=64)` yields rows of 65 IDs under the key `text`. `LMObjective(model, seq_len=64)` feeds the first 64 to the model and scores the predictions against the next 64. Token IDs must be integers inside the model's vocabulary. `temperature=0.0` picks the most likely token at every step, and the returned row holds the prompt followed by the continuation.

The four sentences repeat, so training loss is near zero and validation on the held-out start of the same stream scores equally low. Validation and generation both use the trained `lm_state.variables`, since the objective keeps no moving average unless `ema_decay` is set.

## TinyStories

The same code trains a decoder on TinyStories, a corpus of short stories in simple English, with the GPT-2 tokenizer. Download the 22 MB validation file of TinyStories V2 and tokenize it; the tool holds out the first 1% of the tokens for validation:

```bash
hf download roneneldan/TinyStories TinyStoriesV2-GPT4-valid.txt \
    --repo-type dataset --local-dir data
dew tokenize --input data/TinyStoriesV2-GPT4-valid.txt \
    --out data/tinystories --tokenizer gpt2
```

The measured run changes these settings from the example:

| Setting | Value |
|---|---|
| Tokenizer | `HFTokenizer("gpt2")` |
| Data | `TokenWindows(path="data/tinystories", seq_len=256).load(batch=32)` |
| Model | `emb_features=256, num_layers=4, num_heads=4, max_seq_len=256, dtype=jnp.bfloat16` |
| Optimizer | `optax.adamw(1e-3)` |
| Objective | `LMObjective(model, seq_len=256, ema_decay=0.999)` |
| Run | `steps=2000, log_every=500, eval_every=1000` |
| Prompt | `"Once upon a time"`, 40 new tokens, `temperature=0.0` |

On one Colab L4 GPU the run takes about three minutes. The training loss falls from 2.75 at step 500 to 1.95 at step 2,000, and validation perplexity reaches 8.7. Validation uses the EMA weights, whose perplexity is 29.3 at step 1,000. One run's greedy continuation reads:

> Once upon a time, there was a little girl named Lily. She had a big, red ball. Lily loved to play with her ball. One day, she saw a big box. She wanted to open it.

GPU reductions are not bitwise repeatable by default, so a second run can continue differently after the first sentence. [Post-training](post_training.md) continues a decoder with SFT, DPO and GRPO, and [Inference](inference.md) covers generation tasks.

## Tokenizers and token files

Use one tokenizer everywhere: data preparation, model construction, decoding and checkpoint export. `ByteTokenizer` has a vocabulary of 256 and uses ID 255 as its EOS. `HFTokenizer(name)` wraps a Hugging Face tokenizer with that model's vocabulary and chat template, and downloads its files on first use.

`ByteTokenizer` and `HFTokenizer` have `encode` and `decode`. A source loaded with `Pretrained.load` includes the checkpoint's own processor, which has no `encode` method. Call the processor on text: `bundle.processor(["The capital of France is", "Hello"])` returns `ModelInputs`. Its `tokens` are ID rows padded by the tokenizer, and `kwargs()` provides the attention mask and positions. To decode IDs, use `bundle.processor.decode(rows)`; to apply the checkpoint's chat template, when it has one, use `bundle.processor.chat(messages)`.

`RunProcessor(tokenizer)` in `dew.inference` wraps an `encode`/`decode` tokenizer in the same callable interface that a task accepts as `processor=`. A text-to-image run's condition encoder tokenizes the captions, for example `CLIPText.tokenize` in `dew.inputs`.

`dew tokenize`, or `TokenCorpus.write` from `dew.data` in Python, tokenizes a file, or every `.txt` file under a directory, into `train.bin`, `val.bin` and `meta.json`:

```bash
dew tokenize --input data/corpus.txt --out data/corpus-byte \
    --tokenizer byte --val-fraction 0.1
```

| Flag | Default | Meaning |
|---|---|---|
| `--input` | required | A file, or a directory read as every `*.txt` inside it. |
| `--out` | required | Directory for `train.bin`, `val.bin` and `meta.json`. |
| `--tokenizer` | `byte` | `byte`, or a Hugging Face tokenizer name. |
| `--val-fraction` | `0.01` | Share of the token stream held out for validation, taken from its head. |
| `--pack` | off | Write the tokenizer's EOS ID after every input file, so `PackedTokens` can split the stream back into documents. Refused for a tokenizer without an EOS ID. |

The binary files use the smallest unsigned dtype that holds the vocabulary. Two dataset types read them:

- `TokenWindows(path, seq_len).load(batch=B)` cuts fixed windows of `seq_len + 1` IDs, `{"text": int32 [B, seq_len + 1]}`. Each window starts `seq_len` IDs after the last, so every transition is scored once.
- `PackedTokens` packs whole documents into rows and adds `text_segment_ids` and `text_positions`. The objective skips padded targets and the transitions between documents. Packing needs a segment mask in attention, which can change which attention implementations run.

## LMObjective

`LMObjective(model, seq_len, **options)` reads `batch["text"]` of shape `[B, seq_len + 1]`. `model` may be a loaded decoder bundle in place of the model (see [Pretrained checkpoints](#pretrained-checkpoints)). Every auxiliary term is off until its argument is set.

| Argument | Default | Meaning |
|---|---|---|
| `ema_decay` | `None` | Decay of an EMA copy, which evaluation and previews then read. `None` keeps no copy. |
| `pad_id` | `None` | Token ID whose targets carry no loss. |
| `head_chunks` | `4` | Vocabulary slices the tiled head scores in. `1` is the full pass. |
| `head_tile` | `None` | Backward tile of the head, or `'whole'` / `'tiled'`. `None` keeps the whole logits where they fit. |
| `samples` | `None` | `Samples` configuration for generated previews at evaluation. |
| `variables` | `None` | The tree to start from instead of a fresh init, whole or split by `freeze` or an adapter; a split is kept, so its `params` train and its `frozen` stays put. |
| `loss_role` | `None` | Count only targets whose `text_roles` entry equals this `Role` (SFT). |
| `balance_rate` | `None` | Aux-loss-free routing-bias update rate for mixture layers. |
| `aux_loss_alpha`, `seq_aux` | `None`, `True` | DeepSeek V2 expert balance loss and its per-sequence form. |
| `router_z_loss` | `0.0` | ST-MoE router z-loss coefficient. |
| `mtp_weight` | `None` | Weight of DeepSeek V3's multi-token prediction loss. |
| `z_loss` | `0.0` | PaLM's squared log-partition auxiliary. |
| `qk_stats` | `False` | Report per-head attention logit maxima, the key-head count and the head width for `muonclip`, which clips per query group as Megatron Core does and leaves an output gate's weights alone. |
| `token_accuracy` | `True` | Report argmax accuracy. |

The step reports `ce`, `perplexity` and `token_accuracy`. `LMObjective` refuses a model built with `causal=False`.

### Loss and precision

The vocabulary loss runs in float32. The head's matrix product uses the compute dtype, as torch autocast and MaxText do. With bf16 compute, the hidden states and head multiply in bf16 and sum in float32, including the backward pass. This is 1.5x faster on an L4 and 1.6x faster on an RTX 4080 than using fp32 operands. At the default `matmul_precision` (unset or `"default"`), the logits are then rounded to bf16, and their gradient is rounded to bf16 once, matching torch autocast and MaxText. Softmax and the loss use these values in float32. An fp32 model multiplies in fp32 ([Performance measurements](../performance.md)).

That rounding has no measurable training cost. Over 2000 steps of wikitext-103 on an RTX 4080, validation loss changed by at most 4.5e-4 for a 3-layer decoder trained from scratch and 1.6e-3 for a Qwen3-0.6B fine-tune, compared with float32 logits and gradients at the same seed. Two seeds of either rounding differ by 1.3e-2 and 2.9e-3 on average. Rounding makes the 3-layer decoder's step 13% faster and Qwen3-0.6B's 3% faster at 1 x 1024 tokens.

Rounding does affect layout parity. With the logits' gradient rounded once, a bf16 run on 4 RTX 3090s exceeded its layout-parity bound by factors of 1.75 at a dense model's final norm and 5758 at an MoE's expert gate_proj. Keeping the gradient in float32 gave 0.47 and 0.41 times the bound. If you compare layouts in bf16, or resume onto a different mesh and expect the same numbers, set `matmul_precision="highest"` to keep the head in float32. `tools/layout_parity.py` sets it for bf16 decoders. An LM run recorded before this rounding change resumes with the new rounding, which is within the rerun spread above but is not bitwise identical to the earlier run.

`token_accuracy=False` (`--no-token-accuracy` on the recipe) skips the argmax over every logit, which costs 0.77 ms of the head's 8.0 ms on a TPU v6e. Chunking the head with `head_chunks` can lower peak memory but adds work, and the backend compiler may rewrite it; the saving depends on vocabulary size, sequence length, batch size and the compiled executable.

With bf16 compute, Dew keeps the residual stream and every norm and sublayer output in bf16, rounding where transformers does, as do MaxText and Megatron (with `fp32_residual_connection=False`). `torch.autocast` keeps the residual stream and norms in float32 and rounds at each matmul's input. To reproduce Dew's rounding points, the torch run uses autocast with the embedding output and every RMSNorm output cast to bf16, while leaving the rotary table in float32 (`tools/reference_runs/torch_lm.py --precision autocast-bf16-residual`). At those rounding points, the first step's loss depends on the attention kernel as much as on the framework. These measurements use the first batch of a Qwen3-0.6B fine-tune (4 x 1024 tokens, one A100, `tools/reference_runs/step0_attention.py`):

| Run | Attention kernel | First-step loss minus float32 | Final hidden states, relative distance from torch's FlashAttention 2 run |
|---|---|---|---|
| torch | FlashAttention 2 | -1.60e-4 | 0 |
| torch | cuDNN | +8.17e-4 | 1.42e-2 |
| torch | math | -1.40e-4 | 1.42e-2 |
| Dew | cuDNN | +1.09e-3 | 1.38e-2 |
| Dew | XLA | +1.10e-3 | 1.40e-2 |

Plain `torch.autocast` with FlashAttention 2 gives +4.9e-4. Over 256 steps of the same fine-tune, every 32-step window of Dew's loss and gradient norm stays within twice the larger of the torch run's distance from float32 and its run-to-run spread, using Dew's rounding points for that torch run.

### EMA weights

`LMObjective` keeps no moving average unless `ema_decay` is set. Without one, `state.ema` is `None`, `state.averaged` raises `ValueError`, and previews and evaluation read the live variables. With `ema_decay=0.999`, `state.averaged` holds the EMA copy merged into the live variables, evaluation and previews read it, and the console labels the split `val (ema)`. The copy costs one more set of trained parameters. A checkpoint written with one setting does not restore into the other.

To measure validation perplexity, give the dataset a validation split and set `eval_every`, as in the example ([Evaluation and tracking](../guides/evaluation.md)). With a tracker and `fit(preview=True)`, `Samples` adds one generated preview per evaluation event, separate from the teacher-forced scoring of the batch.

### Gradient accumulation

With gradient accumulation, the cross-entropy and multi-token-prediction (MTP) losses are weighted by the mass of supported main targets, including the target-role masks at each MTP depth. Sequence-level router losses normalize by rows. Global router losses first add up the selected-slot counts and the score sums, then take their product. The routing bias stays fixed during an accumulation window and is updated from the window's total counts. A window with zero cross-entropy can still update when a router auxiliary loss is active. Accumulated and combined batches match only when their random draws are the same and mutable state follows its declared rules.

## Pretrained checkpoints

`Pretrained.load(name_or_dir)` reads a local Hugging Face directory or a Hub identifier into a `Pretrained` bundle: the native Flax model, its variables, the checkpoint's processor or tokenizer, the source config and the generation defaults. The bundle is the kind of source it read, each with the methods that work for it: `PretrainedDecoder` (text generation), `PretrainedMaskedDecoder` (LLaDA, Dream), `PretrainedBlockDecoder` (DiffusionGemma), `PretrainedPipeline` (latent diffusion, `text_to_image`) and `PretrainedFallback` (`fallback="torchax"`, training only). Every kind takes an adapter (`adapt`) and stands in for the model in the objective that trains it. Calling `load` on a kind, as `PretrainedDecoder.load(...)`, refuses a source of another kind by name. `dew.pipeline(source)` wraps the same loader and returns a `TextGeneration` (a `BlockGeneration` for DiffusionGemma, a `MaskedGeneration` for LLaDA and Dream) with the weights placed on the current devices and the sampling policy and budget taken from the checkpoint.

`LMObjective(bundle, seq_len, **options)` trains a decoder bundle from its loaded weights. With token files prepared using the checkpoint's tokenizer (`dew tokenize --tokenizer Qwen/Qwen3-0.6B`), fine-tuning uses the same trainer as training from scratch:

```python
from dew.interop import PretrainedDecoder

data = TokenWindows(path="data/qwen3-tokens", seq_len=512).load(batch=4)
bundle = PretrainedDecoder.load("Qwen/Qwen3-0.6B", max_seq_len=512)
objective = LMObjective(bundle, seq_len=512)
state = Trainer(objective, optax.adamw(1e-5), key=jax.random.key(0)).fit(data, steps=100)
```

This downloads the Hub weights and needs memory for the model, gradients and optimizer. The bundle supplies the model, the initial variables and its processor, so `objective.pipeline(state)` takes text prompts; `variables=` or `processor=` beside it overrides that part.

`dew.lora.LoRA` describes a low-rank adapter by PEFT's own fields (`rank`, `modules` as PEFT's `target_modules`, `alpha`, `rslora`, `dropout`), and `bundle.adapt(lora, key=)` returns the same kind of bundle with it bound: the adapted model, the variables with the factors under `params` and every base weight under `frozen`, and the bound `adapter`. Any objective over it trains the factors and leaves every other weight as loaded. `tuned.adapter.save` writes PEFT's adapter directory, and `tuned.save` writes the source's layout with the factors merged into the kernels. Both read the trainer's `state.variables` as it comes back:

```python
from dew.lora import LoRA

tuned = bundle.adapt(LoRA(rank=8, modules=("q_proj", "v_proj")), key=0)
objective = LMObjective(tuned, seq_len=512)
state = Trainer(objective, optax.adamw(1e-4), key=0).fit(data, steps=100)
tuned.adapter.save(state.variables, "qwen3-adapter")
tuned.save("qwen3-merged", variables=state.variables)
print(objective.pipeline(state)("The capital of France is", 8, key=0).text[0])
```

`bundle` itself is unchanged; `adapt` returns a new value, and adapting an adapted bundle is refused. `LoRA(...).apply(model, variables, key=)` binds the same adapter to any model and variables, a model built from its class included, and `LoRA.load(model, variables, path, layouts=)` reads one PEFT or Diffusers wrote; both return the bound `Adapter`, whose `model` and `variables` go to an objective. A run on the command line takes the same spec as `lora:lora --lora.rank 8 --lora.modules q_proj v_proj` beside `--pretrained`, and `Pretrained.from_run(run).adapter.save(...)` writes its factors under the source's names from the run alone. To train part of a model without an adapter, split its starting variables with `dew.objectives.base.freeze(variables, filter)`: the leaves the filter keeps train and the rest stay frozen.

`PretrainedDecoder.from_model` wraps a trained `CausalTransformer` for export, and `save` writes it in the Hugging Face layout. This exports the decoder from the example and loads it back:

```python
import jax.numpy as jnp

import dew
from dew.interop import PretrainedDecoder

trained = PretrainedDecoder.from_model(model, lm_state.variables, tokenizer="byte",
                                       generation_config={"do_sample": False, "max_new_tokens": 24})
trained.save("stories-decoder")
task = dew.pipeline("stories-decoder", dtype=jnp.float32)
print(task.sampling)
result = task([tokenizer.encode("At night")], key=0).host()
print(tokenizer.decode(result.tokens[0]), result.lengths)
```

```text
Sampling(temperature=0.0, top_k=None, eos_id=None, pad_id=0, top_p=1.0, min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0, no_repeat_ngram_size=0, min_new_tokens=0, typical_p=1.0, stop=())
At night, Lily went home. She wa [24]
```

The directory holds `config.json`, `generation_config.json` and `model.safetensors`. `generation_config.json` sets the task's defaults: `do_sample: false` becomes `temperature=0.0`, and `max_new_tokens` sets the budget, so the call passes neither. The byte vocabulary has no Hugging Face tokenizer files, so the export records only `tokenizer_name: "byte"` and the loaded task takes token rows.

For an `HFTokenizer`, export includes the tokenizer's own files (`save_pretrained`) without recording its name. A local tokenizer's name would refer to a path on the exporting machine; saving the files lets another machine load the directory. Its task accepts strings and returns `result.text`. If the exported config cannot reproduce a model's computation, `from_model` refuses the model and names the field responsible.

A Hub name such as `"Qwen/Qwen3-0.6B"` loads the same way; the published checkpoint needs enough host and device memory for its weights and cache. `LMObjective.policy(params)` returns the same kind of task using the training variables, and `SampledRollout` samples with it. To serve the weights with Ollama or vLLM, export them and point the runtime at the directory, as shown in the [README](https://github.com/AshishKumar4/dew/blob/main/README.md#exporting-a-decoder-and-serving-it).

### Loading from the Hub

From the Hub, the loader fetches configs and indexes first, then only the weight files listed in the index's `weight_map`, or `model.safetensors` when there is no index. It does not download other copies in the repository, such as Mistral's `consolidated.safetensors`, a pipeline's fp16 variants or root single-file checkpoints. A tensor stored in two shards is refused. `Pretrained.revision` records the commit resolved by the Hub, so you can identify what you loaded even if the branch moves later; it is `None` for a local directory.

`param_dtype` defaults to float32 master weights. With `param_dtype="auto"`, parameters use the checkpoint's own dtype: `dtype` in `config.json`, or the first floating tensor's dtype if the config has none, matching transformers' `dtype="auto"`. `Pretrained.load(..., mesh=MeshSpec(...), layout=Layout(...))`, also called by `dew.pipeline`, places each weight directly from the memory-mapped checkpoint, one device shard at a time. The host therefore never holds the whole translated model ([Distributed training](distributed.md) has the measurements).

If Dew cannot express a config field, it refuses the field by name. The exceptions are fields in `_INERT_FIELDS` (`dew/interop/hf_decoders.py`) that transformers 5.16.1 neither declares nor reads, such as SmolLM2's `transformers.js_config`, Qwen2.5's `use_mrope: false` and the Mamba2 ports' `rms_norm`. Each family reads only fields declared by its transformers config class. A Llama config that names a sliding window is refused because transformers' Llama attends to every key. Dew reads the window only for model types that apply it, such as Mistral or Ministral. MLX quantization is refused by name.

The real prompt plus the continuation must fit in `model.max_seq_len`; input padding takes no room. `Sampling` holds temperature, top-k, top-p, min-p, EOS and the output padding ID. For a loaded checkpoint, the source's `generation_config.json` fills it in, and `text_generation(sampling=...)` overrides it. Generation prepares the inputs on the host, then runs prefill and decode in one compiled call with one padded input shape and a cache cursor per row, so inputs with different valid lengths but the same padded shape reuse the executable.

The base checkpoints used below are not instruction-tuned chat models. With an instruction-tuned model, use the chat template its checkpoint documents. The [Supported models](../models.md) page lists which decoder families load, train, generate and export.

### Other weight formats

- `Pretrained.load(repo, gguf_file="Model-Q4_K_M.gguf")` reads one GGUF llama.cpp file (`pip install 'dewml[gguf]'`). Its metadata becomes the config, and every tensor is dequantized to float32 on the host. The model then trains and generates like a safetensors load, with `param_dtype` setting parameter storage. If the repository has no tokenizer, Dew uses the one in the file. The llama, qwen2 and qwen3 architectures load; anything else is refused. When you load a repository containing only GGUF files without `gguf_file`, Dew lists the files and names the argument to use.
- A repository containing only PyTorch pickles, `pytorch_model.bin` or its index, also loads. If SFconvertbot has opened a safetensors pull request for that commit, Dew loads its `refs/pr/N` revision, logs which one and records the commit in `Pretrained.revision`. Otherwise, with torch installed (`pip install 'dewml[torch]'`), Dew unpickles the files once with `torch.load(weights_only=True)`, which runs no code from the file. It saves a safetensors copy under `~/.cache/dew/converted/` (`$XDG_CACHE_HOME/dew/converted/` when that is set), and later loads use that copy without importing torch. If neither torch nor a pull request is available, the load is refused with a message naming the extra to install and the https://huggingface.co/spaces/safetensors/convert space. Local directories work the same way.
- `state-spaces/mamba2-*` and other checkpoints in mamba_ssm's own format have a config.json with `d_model`, `n_layer` and `ssm_cfg` but no `model_type`. Dew loads them as the Mamba2 port produced by transformers' conversion script, and `save` writes that port.

### Model types without a Dew family

Loading works in three tiers.

1. Native family: the model type is registered, and a test compares its numbers with transformers. It uses Dew's attention kernels, sharding rules, cached generation and export.
2. Verified mapping: some unregistered model types compute a registered family's model under their own name. CWM is the Llama block, Seed-OSS is Qwen2's and GLM-4.7-Flash is DeepSeek V3's. Before downloading weights, `Pretrained.load` builds a much smaller model from transformers' own class for that type, using the same config and random weights. It loads that model through each registered family whose mapping reads the config, Llama's first, and compares logits in float32. The real checkpoint loads as a native Dew model through the first family that matches, with one `VerifiedMappingWarning` naming tier 2, the family, the transformers version and the measured error. The model saves under the type's own config and tensor names. The check needs torch (`pip install 'dewml[torch]'`); without it, the refusal names that extra. A type that computes something else is refused with each family's reason, for example SmolLM3 (layers without rotary positions), Granite (multipliers) or Phi-3 (fused projections).
3. Generic, opt-in: `Pretrained.load(repo, fallback="torchax")` builds transformers' PyTorch model on the host and runs its forward through torchax. It compiles, differentiates and trains through `Trainer` and `LMObjective`. Variables keep the source's torch names: the optimizer updates `params` and leaves `buffers` untouched, while `TorchLayout` places them on a mesh by name. This tier uses no Dew kernels, reads only float weights and has no cached generation (`text_generation` refuses). It pins the torch version (`pip install 'dewml[torchax]'`), and loading also keeps the torch model in host memory, so plan for at least twice the checkpoint size. The load warns and names the tier.

### Multimodal checkpoints

A multimodal checkpoint loads as a `MultimodalTransformer`, with its text decoder, vision tower and projector under the variable names `language_model`, `tower` and `projector`. Gemma 3n and Gemma 4 also include an audio tower and embedder under `audio_tower` and `audio_projector`. The checkpoint's processor resizes and normalizes media, creates patches and expands placeholders; Dew's `Processor` runs it and arranges its outputs row by row.

The next examples use the tiny checkpoints under `tests/fixtures/hf` in a Dew repository checkout. They run on CPU to show how the calls work, and are too small to judge language quality. This one loads a tiny Gemma 3 and gives it images:

```python
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from dew.interop import PretrainedDecoder

fixtures = Path(dew.__file__).parents[2] / "tests" / "fixtures" / "hf"
source = fixtures / "gemma3-native-tiny"
bundle = PretrainedDecoder.load(source, dtype=jnp.float32, max_seq_len=64)
images = np.load(source / "raw_images.npy")          # uint8 [3, 32, 32, 3]
inputs = bundle.processor(["token7 <start_of_image> token9",
                           "token5 <start_of_image> token8 <start_of_image> token6"],
                          images=[[images[0]], [images[1], images[2]]])
logits = bundle.model.apply(bundle.variables, inputs.tokens, **inputs.kwargs())
task = bundle.text_generation(sampling=Sampling(temperature=0))
for text in task(inputs, 3, key=1).text:
    print(text)
```

```text
token9token9token9
token6token6token6
```

`ModelInputs` has three parts:

| Part | Contents |
|---|---|
| `tokens` | `[B, S]` token IDs. |
| `token_fields` | `attention_mask`, `positions`, `image_indices` and `image_groups` (the soft feature and the image behind each slot, -1 for text), `audio_indices` for audio slots, and for Qwen 3.5 the three-axis `rotary_positions`. |
| `conditioning` | The media, padded to the row with the most images or clips. `pixel_values` is `[B, images, ...]` in the processor's own layout, with `image_position_ids` (Gemma 4) or `image_grid_thw` (Qwen 3.5) beside it. `input_features` and `input_features_mask` are `[B, clips, frames, mel]`. |

The two rows above have different numbers of images, so the processor left-pads the shorter one. `inputs.token_fields["attention_mask"]` is the only record of which slots are valid, and padded slots take no cache. If a batch needs no padding, it has no `attention_mask`; the missing field means every slot is valid. Without a mask, the model passes causality to the kernel as a flag and uses fused attention. An all-true mask would prevent that ([Performance measurements](../performance.md)). To handle both cases, read the field with `token_fields.get("attention_mask")`.

Each process pads its own rows, so a process pool must agree on whether a mask is present before assembling the inputs. For generation, if any process has the field, all processes create it; if none has it, all leave it out. A training batch placed by `shard_batch` has the field on every process. For text-only prompts, leave out `images`. Gemma 3n and Gemma 4 take `audio=[waveform, ...]`, one waveform per audio placeholder in reading order.

The processor checks token IDs, placeholder counts and media shapes on the host; the compiled model does no checking. Pass the same `ModelInputs` to `model.apply`, the objective and `generate`. The model evaluates media during prefill and uses the cache during decode.

To keep training, build the bundle's objective and feed the trainer `{"text": inputs}` batches:

```python
import optax
from dew.data.dataset import Dataset
from dew.training import Layout, MeshSpec

mm_objective = LMObjective(bundle, inputs.tokens.shape[1] - 1, pad_id=0)
rows = 2 * jax.device_count()
batch = inputs.take_rows(jax.numpy.arange(rows) % 2)
mm_data = Dataset(train=lambda partition: iter([{"text": batch}]), val=None,
                  records=rows, batch=rows)
mm_trainer = Trainer(mm_objective, optax.sgd(1e-4), key=jax.random.key(3),
                     mesh=MeshSpec(), layout=Layout(min_shard=2**30))
mm_state = mm_trainer.fit(mm_data, steps=1, log_every=1)
bundle.save("gemma3-tiny-step1", variables=mm_state.variables)
```

`bundle.save(directory, variables=...)` writes the trained weights with the source tensor names, together with the processor, so both Dew and Transformers can reload the directory. This includes Gemma 4's frozen standardization and clipping buffers in the `constants` collection, which stay bit-for-bit unchanged through training. When the weights exceed `max_shard_size` (default `"5GB"`), they are written as numbered shards with an index.

A dense decoder or source-layout checkpoint builds each tensor on the host only when writing its shard. A quantized source, a `from_model` export of a Gemma 4 or GLM-5-next model, and a DiffusionGemma export build the whole export first. When replacing an earlier export, Dew gives the new shards fresh names and writes the index last. This replaces the export in one step, so an interrupted save leaves the previous export intact. Files the export did not write, such as `model.fp16.safetensors`, are left alone.

Each family's processor emits what its reference implementation expects:

- Gemma 3 gives one fixed-resolution image per placeholder block.
- Llama 4 splits each image into local tiles and a global tile with separator tokens. It normalizes pixels in bfloat16, as the original implementation does, and the loader widens them to float32 exactly.
- Gemma 4 emits padded patch streams with 2D patch positions. It expands video placeholders, which the decoder maps to the pad embedding.
- Qwen 3.5 packs patches channel first, then time, with a grid per image. The loader derives the spatial rotary coordinates that the reference's `get_rope_index` computes.
- Gemma 3n uses the MobileNet-v5 encoder. It embeds its hard vision and audio vocabulary ranges through the multimodal embedders and keeps placeholder IDs for its per-layer inputs while masking the hard ranges, on both training and decode steps.
- DeepSeek-V4.1 ships no processor. Its release resizes and pads each image in its own `inference/image_processor.py`, which Dew does not reproduce, so the caller builds the inputs it writes. `pixel_values` is `[B, images, 3, H, W]`, normalized to [-1, 1], with H and W multiples of the patch size. Each image fills `rows * (columns + 1) + 2` positions of `image_token_id`, where rows and columns count the aligner's `downsample_ratio` squares over the patch grid. `image_indices` numbers those positions in reading order.

Audio clips have a mask that is True for valid frames. Gemma 4 inserts one placeholder per encoded frame. Gemma 3n inserts a fixed `audio_soft_tokens_per_image` per clip and fills the remaining slots with the embedder's padding token.

The Gemma 3n and Gemma 4 audio encoders are `dew.nn.audio.Gemma3nAudio` and `Gemma4Audio`, registered as the towers `gemma3n_audio` and `gemma4_audio`. `audio_config` reads the checkpoint's `audio_config` record and rejects unknown computational fields. `audio_weights` converts the tower's tensors and keeps Gemma 4's checkpointed clipping bounds in a frozen `constants` collection.

An encoder takes `input_features` shaped `[B, T, F]` and a boolean `input_features_mask` that is True for valid frames. It returns `AudioEncoding(features, mask)`, with the mask subsampled to the encoder's frame rate. A loaded source's processor (`Pretrained.load(...).processor`) uses the checkpoint's own Transformers feature extractor to convert 16 kHz mono waveforms into those arrays; it does not resample them. Gemma 3n projects audio through `Gemma3nProjectorModule.soft_embeddings`, without the scaling used for vision. Gemma 4 reuses `Gemma4ProjectorModule`, with its input width taken from `output_proj_dims`.

## Masked diffusion language models

`MaskedDiffusionObjective` trains a bidirectional decoder to recover masked tokens under MDLM's negative ELBO. As in MDLM's SUBS parameterization, the mask token has no probability mass, so cross entropy excludes its column from the partition. LLaDA's published training script includes the whole vocabulary, and its loss values differ by that term. Each row holds exactly `seq_len` tokens, with no next-token shift: `TokenWindows(seq_len=63)` produces rows of 64 IDs for a 64-token objective. The mask token needs an ID of its own, here the one after the last byte:

```python
from dew.diffusion.discrete import MDLM
from dew.objectives.diffusion import MaskedDiffusionObjective

masked_data = TokenWindows(path="data/stories-byte", seq_len=63,
                           loading=Loading(workers=0)).load(batch=16)
mask_id = tokenizer.vocab_size
process = MDLM(mask_id=mask_id)()
masked_model = CausalTransformer(vocab_size=mask_id + 1,
                                 emb_features=64, num_layers=2, num_heads=2,
                                 mlp_features=256, max_seq_len=64, causal=False)
masked_objective = MaskedDiffusionObjective(masked_model, process, seq_len=64)
masked_state = Trainer(masked_objective, optax.adamw(3e-3), key=jax.random.key(4)).fit(
    masked_data, steps=1000, log_every=500)
drawn = process.generate(masked_model, masked_state.variables,
                         [tokenizer.encode("One day, Lily")], 24, key=jax.random.key(5))
print(tokenizer.decode(drawn.tokens[0]))
```

```text
Training CausalTransformer from step 0 to 1000: 147,968 parameters, on 1 × cpu, batch 16, float32
step  500/1000  loss 0.6956  masked_accuracy 64.9%  masked_fraction 45.6%  step_time_ms 20.22  samples_per_sec 791.4  accepted 100.0%  0:00:11 left
step 1000/1000  loss 0.7562  masked_accuracy 64.4%  masked_fraction 51.6%  step_time_ms 18.57  samples_per_sec 861.4  accepted 100.0%
Trained 1000 steps in 0:00:20: first step after 1.16 s, then 51.7 step/s
94.3% of the wall time in steps, final loss 0.7562
One day, Lily. She da b tfte,dAt nigl
```

The loss is MDLM's negative ELBO per token. `process.generate` unmasks the 24 tokens after the prompt together, in 64 reverse steps by default (`steps=`). This small model produces fragments of its corpus rather than a sentence.

On TinyStories, with the GPT-2 tokenizer, a 4-layer, 256-wide model trained on 128-token rows at batch 64 took about five minutes for 4,000 steps on an L4. Its loss fell from 3.48 at step 1,000 to 2.78 at step 4,000, and validation perplexity reached 15.0. That perplexity is exp of the ELBO, an upper bound on the model's own perplexity, so it does not compare directly with an autoregressive decoder's. One sample of 48 tokens read:

> Once upon a time, in a small town, there lived a little girl named Lily. Mia loved whistle. She had an key telling to try to sleep. One day, she always to see her favorite mom, dad unate to have lots of.

LLaDA and Dream use this masked-token approach starting from their released weights. They predict masked tokens with bidirectional attention and need a mask token ID and a masked-diffusion objective. Switching from autoregressive training changes both attention and corruption, as well as the loss. For these models, `PretrainedMaskedDecoder.load(...).text_generation()`, `dew.pipeline(source_or_run)` and `MaskedDiffusionObjective.pipeline(state)` return `MaskedGeneration`. It runs Dew's MDLM algorithm (`DiscreteProcess` with `Unmask`); it does not implement LLaDA's or Dream's source-specific remasking and block-generation recipes.

```python
from dew.interop import PretrainedMaskedDecoder

llada = PretrainedMaskedDecoder.load(fixtures / "llada-tiny", dtype=jnp.float32, attention_impl="xla")
masked_task = llada.text_generation()
result = masked_task([[1, 2, 3]], 8, key=7, steps=16, n=2)
print(result.host().tokens)
```

```text
[[  1   2   3  34  55  40   2  69  66  55  40]
 [  1   2   3  88 125  84  90  97 111   3 112]]
```

Released sources that ship a tokenizer also accept text and provide `result.text`. This path handles text only and rejects media payloads and media token fields; numeric requests use `ModelInputs`. Attention validity, logical `[B, S]` positions, rotary coordinates and segment IDs pass through the masked model. The prompt stays fixed, even if it contains a literal mask ID. All requested response positions are refined together under bidirectional attention, and mask IDs are never drawn as generated tokens.

The default is 64 model evaluations, including the final clean prediction; a call can override `steps`. The `n` continuations are grouped by prompt, and continuation zero does not change when you ask for more. EOS is applied after the whole span is refined, so lengths include the first EOS and the rest is padded; generation does not stop early at an EOS. A request for zero tokens keeps the prompt and reports zero refinements. `CanvasGeneration` reports lengths, EOS flags and refinement counts, and has no autoregressive log-probabilities. This path does not accept autoregressive sampling, beam or logits controls.

A source generation control that native MDLM cannot follow is rejected by name. Neutral values are accepted, as are the shared budget, continuation count, EOS and padding metadata. MDLM does not use a KV cache, so `use_cache=False` is accepted and asking for a cache is not.

Saved masked-diffusion recipe runs keep their compute and storage precision, and `dew.pipeline`'s `ema` option selects live or EMA weights. The run record rebuilds Dew's MDLM with the default `Unmask` solver and 64 steps, without saving custom objective steps or solver choices set in code. Plain `Checkpoints` saves weights and training state, so you must reapply any custom task settings. The recipe also does not save an EOS policy; source checkpoints follow their own EOS metadata. For a task built from an objective in code, set EOS with `dataclasses.replace(task, eos_token_ids=(...))`.

## DiffusionGemma

DiffusionGemma uses uniform-vocabulary corruption, a causal prompt encoder, a bidirectional canvas and self-conditioning. `Pretrained.load` loads the complete model. Its `BlockProcess` generation policy refines whole canvases and commits clean tokens to the shared text cache. It returns `CanvasGeneration` with response lengths (including EOS), termination flags and per-row refinement counts, and no autoregressive likelihood fields.

The next example loads the tiny reference checkpoint, then tokenizes, generates, decodes, saves and reloads it. Its synthetic vocabulary has 64 tokens. The released model ID is `google/diffusiongemma-26B-A4B-it`, which downloads large weights and needs matching host and device memory.

```python
from tempfile import TemporaryDirectory

from dew.interop import PretrainedBlockDecoder

gemma = PretrainedBlockDecoder.load(fixtures / "diffusion-gemma-workflow", dtype=jnp.float32,
                                    attention_impl="xla", max_seq_len=32)
prompt = gemma.processor(["<bos> t5 t7 t9 t11"])
block_task = gemma.block_generation()
generated = block_task(prompt, 7, key=jax.random.key(11))
print(block_task.decode(generated))
with TemporaryDirectory() as checkpoint:
    gemma.save(checkpoint)
    restored = PretrainedBlockDecoder.load(checkpoint, dtype=jnp.float32, attention_impl="xla",
                                           max_seq_len=32)
    replay = restored.block_generation()(prompt, 7, key=jax.random.key(11))
    np.testing.assert_array_equal(replay.tokens, generated.tokens)
```

```text
('t37 t49 t62 t14 t23 t49 t34',)
```

The last canvas is refined at full width, then the response is cut to `max_new_tokens`. The prefix plus the canvas capacity, rounded up to whole canvases, must fit in `max_seq_len`. Generation defaults come from `generation_config.json`; pass a `BlockProcess` as `process=` to override them. The checkpoint's processor prepares media, which the model evaluates only during prompt prefill. The [training-contract note](../research/inference.md#diffusiongemma-training-contract-and-open-prerequisites) covers the public fine-tuning recipe and the original sampler-distillation and RL objective, which Google has not published.

### Block diffusion fine-tuning

`BlockDiffusionObjective` ports Google's public SFT adapter, released after the model. It samples a valid response canvas, corrupts the whole response with uniform-vocabulary noise, and runs self-conditioning with a detached first pass. It then combines the canvas loss and the encoder loss, each normalized per row on its own. The default time safety margin is 1e-4 and the self-conditioning probability is 0.5. It is not the unpublished sampler-distillation and RL objective.

The next example takes one optimizer step on the tiny reference model, with a synthetic vocabulary and unequal target support. This dense model lacks the routed experts and vision tower built by transformers' `DiffusionGemmaForBlockDiffusion`, so `save` refuses to export it. Its run checkpoint still saves the model.

For a run of the image-reading model, `Pretrained.from_run(run).save(destination)` writes the config from the run's model: the text stack, the Gemma 4 tower's `vision_config` and the canvas length. Dew places images by position and keeps no image token IDs, so transformers uses its defaults for them. Because transformers reads `generation_config.json` as a closed set of fields, the run must use a Hugging Face tokenizer with its complete record in its files; a run on Dew's byte vocabulary is refused.

```python
from dew.objectives.diffusion import BlockDiffusionObjective

sft_source = fixtures / "diffusion-gemma-sft"
sft_bundle = PretrainedBlockDecoder.load(sft_source, dtype=jnp.float32, attention_impl="xla", max_seq_len=32)
with np.load(sft_source / "reference.npz") as reference:
    train_tokens = np.tile(reference["tokens"], (jax.device_count(), 1))
block_data = Dataset(train=lambda partition: iter([{"text": train_tokens}]), val=None,
                     records=len(train_tokens), batch=len(train_tokens))
block_objective = BlockDiffusionObjective(sft_bundle.model, prompt_length=4,
                                          num_canvases=2, variables=sft_bundle.variables)
block_state = Trainer(block_objective, optax.sgd(0.001), key=jax.random.key(2)).fit(
    block_data, steps=1, log_every=1)
print("Optimizer updates:", int(block_state.updates))
```

```text
Training DiffusionGemma from step 0 to 1: 1,858 parameters, on 1 × cpu, batch 2, float32
step 1/1  loss 6.802  canvas_ce 3.259  encoder_ce 3.542  step_time_ms 77.14  samples_per_sec 25.93  accepted 100.0%
Trained 1 steps in 0:00:03: first step after 3.18 s
0.1% of the wall time in steps, final loss 6.802
Optimizer updates: 1
```

`recipes/lm/train.py --objective block_diffusion` runs the same objective; there is no separate recipe. It trains on complete token-window rows, and `data.seq_len + 1` must equal `block_prompt_tokens` plus a whole number of training canvases. `block_canvas_size` defaults to the checkpoint's canvas length. Packed documents are rejected, because their context boundaries differ. Block SFT logs `canvas_ce` and `encoder_ce`. It does not report autoregressive perplexity or use the autoregressive preview settings.

```bash
python recipes/lm/train.py data:token-windows --data.path data/diffusion-token-windows \
    --data.seq-len 511 --block-prompt-tokens 256 \
    --pretrained google/diffusiongemma-26B-A4B-it \
    --tokenizer google/diffusiongemma-26B-A4B-it --objective block_diffusion \
    --sample-tokens 0 --ema-decay None --optim.learning-rate 0.00015
```

The token files must use the checkpoint's tokenizer and contain the clean prompt prefix followed by the intended response canvases. Trainer checkpoints keep the optimizer and iterator state, while `Pretrained.save` writes a complete inference checkpoint in the source format.
