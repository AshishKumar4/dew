# Language models

`LMObjective` trains a decoder on next-token prediction: it reads rows of token IDs, feeds all but the last token to the model and scores its predictions against the following tokens with cross entropy. The decoder is usually `causal_transformer` from `dew.models`, and `dew.sampling.generate` draws text from the trained weights. The same page covers loading published checkpoints with `load_pretrained`, and the two non-autoregressive language objectives, `MaskedDiffusionObjective` and `BlockDiffusionObjective`.

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

This is the layout `tools/tokenize_text.py` writes: `train.bin` and `val.bin` hold the token IDs back to back, and `meta.json` records the tokenizer, vocabulary size, dtype and counts. The validation split is the head of the stream.

```python
import jax
import optax

from dew import Trainer, metrics, models
from dew.data import Loading, TokenWindows
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling, generate

data = TokenWindows(path="data/stories-byte", seq_len=64,
                    loading=Loading(workers=0)).load(batch=16)
model = models.build("causal_transformer", vocab_size=tokenizer.vocab_size,
                     emb_features=64, num_layers=2, num_heads=2, mlp_features=256,
                     max_seq_len=128)
objective = LMObjective(model, seq_len=64)
lm_state = Trainer(objective, optax.adamw(3e-3), key=jax.random.key(0)).fit(
    data, steps=300, log_every=100, eval_every=300, metrics=(metrics.perplexity(),))
continuation = generate(model, lm_state.params, [tokenizer.encode("One day, Lily")],
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

The training loss is near zero because four sentences repeat, and the held-out head of the same stream scores as low. Validation and the generation above both read the trained `lm_state.params`, since the objective keeps no moving average unless `ema_decay` is set.

## TinyStories

The same code trains a decoder on TinyStories, a corpus of short stories in simple English, with the GPT-2 tokenizer. Download the 22 MB validation file of TinyStories V2 and tokenize it; the tool holds out the first 1% of the tokens for validation:

```bash
hf download roneneldan/TinyStories TinyStoriesV2-GPT4-valid.txt \
    --repo-type dataset --local-dir data
python tools/tokenize_text.py --input data/TinyStoriesV2-GPT4-valid.txt \
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

On one Colab L4 GPU the run takes about three minutes. The training loss falls from 2.75 at step 500 to 1.95 at step 2,000, and validation perplexity reaches 8.7. At step 1,000 the EMA weights, which validation reads, are at perplexity 29.3. One run's greedy continuation reads:

> Once upon a time, there was a little girl named Lily. She had a big, red ball. Lily loved to play with her ball. One day, she saw a big box. She wanted to open it.

GPU reductions are not bitwise repeatable by default, so a second run can continue differently after the first sentence. [Post-training](post_training.md) continues a decoder with SFT, DPO and GRPO, and [Inference](inference.md) covers generation tasks.

## Tokenizers and token files

Use one tokenizer everywhere: data preparation, model construction, decoding and checkpoint export. `ByteTokenizer` has a vocabulary of 256 and uses ID 255 as its EOS. `HFTokenizer(name)` wraps a Hugging Face tokenizer with that model's vocabulary and chat template, and downloads its files on first use.

`tools/tokenize_text.py` (in a repository checkout) tokenizes a file, or every `.txt` file under a directory, into `train.bin`, `val.bin` and `meta.json`:

```bash
python tools/tokenize_text.py --input data/corpus.txt --out data/corpus-byte \
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

`LMObjective(model, seq_len, **options)` reads `batch["text"]` of shape `[B, seq_len + 1]`. Every auxiliary term is off until its argument is set.

| Argument | Default | Meaning |
|---|---|---|
| `ema_decay` | `None` | Decay of an EMA copy, which evaluation and previews then read. `None` keeps no copy. |
| `pad_id` | `None` | Token ID whose targets carry no loss. |
| `head_chunks` | `4` | Vocabulary slices the tiled head scores in. `1` is the full pass. |
| `head_tile` | `None` | Backward tile of the head, or `'whole'` / `'tiled'`. `None` keeps the whole logits where they fit. |
| `samples` | `None` | `Samples` configuration for generated previews at evaluation. |
| `pretrained` | `None` | A variables mapping to start from instead of a fresh init. |
| `loss_role` | `None` | Count only targets whose `text_roles` entry equals this `Role` (SFT). |
| `balance_rate` | `None` | Aux-loss-free routing-bias update rate for mixture layers. |
| `aux_loss_alpha`, `seq_aux` | `None`, `True` | DeepSeek V2 expert balance loss and its per-sequence form. |
| `router_z_loss` | `0.0` | ST-MoE router z-loss coefficient. |
| `mtp_weight` | `None` | Weight of DeepSeek V3's multi-token prediction loss. |
| `z_loss` | `0.0` | PaLM's squared log-partition auxiliary. |
| `qk_stats` | `False` | Report per-head attention logit maxima for `muonclip`. |
| `trainable` | `None` | `PathFilter` selecting the leaves the optimizer moves; the rest go under `frozen`. |
| `token_accuracy` | `True` | Report argmax accuracy. |

The step reports `ce`, `perplexity` and `token_accuracy`. `LMObjective` refuses a model built with `causal=False`.

### Loss and precision

The vocabulary loss runs in float32. The head's product follows the compute dtype, as torch autocast and MaxText run it: under bf16 compute the hidden states and the head multiply as bf16 and sum in float32, backward included, which is 1.5x faster on an L4 and 1.6x faster on an RTX 4080 than fp32 operands. At the default `matmul_precision` (unset or `"default"`) the logits are then rounded to bf16 and their gradient is rounded to bf16 once, as torch autocast's and MaxText's bf16 logits are; the softmax and the loss read them in float32. An fp32 model multiplies in fp32 ([Performance measurements](../performance.md)).

That rounding costs nothing measurable in training. Over 2000 steps of wikitext-103 on an RTX 4080, against float32 logits and a float32 gradient at the same seed, validation loss moved by at most 4.5e-4 for a 3-layer decoder trained from scratch and 1.6e-3 for a Qwen3-0.6B fine-tune; two seeds of either rounding are 1.3e-2 and 2.9e-3 apart on average. It makes the 3-layer decoder's step 13% faster and Qwen3-0.6B's 3% faster at 1 x 1024 tokens. Layout parity is where it shows: with the logits' gradient rounded once, a bf16 run on 4 RTX 3090s read 1.75 times its layout-parity bound at a dense model's final norm and 5758 times it at an MoE's expert gate_proj, against 0.47 and 0.41 with the gradient carried in float32. A run that compares layouts in bf16, or resumes onto a different mesh and expects the same numbers, sets `matmul_precision="highest"`, which keeps the head in float32. `tools/layout_parity.py` sets it for bf16 decoders. An LM run recorded before this change resumes with the new rounding, which is within the rerun spread above but is not bitwise the same as before. `token_accuracy=False` (`--no-token-accuracy` on the recipe) skips the argmax over every logit, which costs 0.77 ms of the head's 8.0 ms on a TPU v6e. Chunking the head with `head_chunks` can lower peak memory but adds work, and the backend compiler may rewrite it; the saving depends on vocabulary size, sequence length, batch size and the compiled executable.

With bf16 compute, Dew keeps the residual stream and every norm and sublayer output in bf16, each rounded where transformers rounds a bf16 tensor, as MaxText and Megatron (with `fp32_residual_connection=False`) do. `torch.autocast` instead keeps the residual stream and the norms in float32 and rounds at each matmul's input. The torch run with Dew's rounding points is autocast with the embedding output and every RMSNorm output cast to bf16 and the rotary table left in float32 (`tools/reference_runs/torch_lm.py --precision autocast-bf16-residual`). At those rounding points the first step's loss depends on the attention kernel as much as on the framework. On the first batch of a Qwen3-0.6B fine-tune (4 x 1024 tokens, one A100, `tools/reference_runs/step0_attention.py`):

| Run | Attention kernel | First-step loss minus float32 | Final hidden states, relative distance from torch's FlashAttention 2 run |
|---|---|---|---|
| torch | FlashAttention 2 | -1.60e-4 | 0 |
| torch | cuDNN | +8.17e-4 | 1.42e-2 |
| torch | math | -1.40e-4 | 1.42e-2 |
| Dew | cuDNN | +1.09e-3 | 1.38e-2 |
| Dew | XLA | +1.10e-3 | 1.40e-2 |

Plain `torch.autocast` with FlashAttention 2 sits at +4.9e-4. Over 256 steps of the same fine-tune, every 32-step window of Dew's loss and gradient norm stays within twice the distance from float32 of the torch run at Dew's rounding points, or twice that run's run-to-run spread where the spread is larger.

### EMA weights

`LMObjective` keeps no moving average unless `ema_decay` is set. Without one, `state.ema` is `None`, `state.averaged` raises `ValueError`, and previews and evaluation read the live variables. With `ema_decay=0.999`, `state.averaged` holds the EMA copy merged into the live variables, evaluation and previews read it, and the console labels the split `val (ema)`. The copy costs one more set of trained parameters. A checkpoint written with one setting does not restore into the other. An `LMRunConfig` record written before this default, which has no `ema_decay` field, reads back as 0.999, the average that run kept.

To measure validation perplexity, give the dataset a validation split and set `eval_every`, as in the example ([Evaluation and tracking](../guides/evaluation.md)). With a tracker and `fit(preview=True)`, `Samples` adds one generated preview per evaluation event, separate from the teacher-forced scoring of the batch.

### Gradient accumulation

With gradient accumulation, the cross-entropy and multi-token-prediction (MTP) losses are weighted by the mass of supported main targets, including the target-role masks at each MTP depth. Sequence-level router losses normalize by rows. Global router losses first add up the selected-slot counts and the score sums, then take their product. The routing bias stays fixed during an accumulation window and is updated from the window's total counts. A window with zero cross-entropy can still update when a router auxiliary loss is active. Accumulated and combined batches match only when their random draws are the same and mutable state follows its declared rules.

## Pretrained checkpoints

`load_pretrained(name_or_dir)` reads a local Hugging Face directory or a Hub identifier into a `Pretrained` bundle: the native Flax model, its variables, the checkpoint's processor or tokenizer, the source config and the generation defaults. `dew.pipeline(source)` wraps the same loader and returns a `TextGeneration` (a `BlockGeneration` for DiffusionGemma, a `MaskedGeneration` for LLaDA and Dream) with the weights placed on the current devices and the sampling policy and budget taken from the checkpoint.

`bundle.lm_objective(seq_len, **options)` builds an `LMObjective` starting from its loaded weights. With token files prepared using the checkpoint's tokenizer (`tools/tokenize_text.py --tokenizer Qwen/Qwen3-0.6B`), fine-tuning uses the same trainer as training from scratch:

```python
from dew.interop import load_pretrained

data = TokenWindows(path="data/qwen3-tokens", seq_len=512).load(batch=4)
bundle = load_pretrained("Qwen/Qwen3-0.6B", max_seq_len=512)
objective = bundle.lm_objective(seq_len=512)
state = Trainer(objective, optax.adamw(1e-5), key=jax.random.key(0)).fit(data, steps=100)
```

This downloads the Hub weights and needs memory for the model, gradients and optimizer. A bundle already supplies the initial variables, so passing `pretrained=` as well is refused. It also hands the objective its processor, so `objective.pipeline(state)` takes text prompts; `processor=` overrides it.

`bundle.lora(rank=, modules=, key=)` returns the same kind of bundle with a fresh low-rank adapter (LoRA) on the projections `modules` names, PEFT's `target_modules`. Its `lm_objective` trains the adapter's factors and leaves every other weight frozen. `tuned.adapter.save` writes PEFT's adapter directory, and `tuned.save` writes the source's layout with the factors merged into the kernels. Both read the trainer's `state.params` as it comes back:

```python
key = jax.random.key(0)
tuned = bundle.lora(rank=8, modules=("q_proj", "v_proj"), key=key)
objective = tuned.lm_objective(seq_len=512)
state = Trainer(objective, optax.adamw(1e-4), key=key).fit(data, steps=100)
tuned.adapter.save(state.params, "qwen3-adapter")
tuned.save("qwen3-merged", variables=state.params)
print(objective.pipeline(state)("The capital of France is", 8, key=key).text[0])
```

`bundle` itself is unchanged; `lora` returns a new value. Calling `lora` on an adapted bundle is refused, as is `trainable=` beside an adapter. `dew.lora.LoRA.fresh` and `LoRA.load` build the same adapter over any model and variables, for a model built from the registry or a pipeline component.

`save_pretrained_decoder` writes a trained `CausalTransformer` in the Hugging Face layout. This exports the decoder from the example and loads it back:

```python
import dew
from dew.interop import save_pretrained_decoder

save_pretrained_decoder(model, lm_state.params, "stories-decoder", tokenizer="byte",
                        generation_config={"do_sample": False, "max_new_tokens": 24})
task = dew.pipeline("stories-decoder", dtype="float32")
print(task.sampling)
result = task([tokenizer.encode("At night")], key=0).host()
print(tokenizer.decode(result.tokens[0]), result.lengths)
```

```text
Sampling(temperature=0.0, top_k=None, eos_id=None, pad_id=0, top_p=1.0, min_p=0.0)
At night, Lily went home. She wa [24]
```

The directory holds `config.json`, `generation_config.json` and `model.safetensors`. `generation_config.json` sets the task's defaults: `do_sample: false` becomes `temperature=0.0`, and `max_new_tokens` the budget, so the call passes neither. The byte vocabulary has no Hugging Face tokenizer files, so the export records only `tokenizer_name: "byte"` and the loaded task takes token rows. An export with an `HFTokenizer` carries the tokenizer's files and its task accepts strings and returns `result.text`. `save_pretrained_decoder` refuses a model whose computation the exported config would not reproduce, and names the field.

A Hub name such as `"Qwen/Qwen3-0.6B"` loads the same way; the published checkpoint needs enough host and device memory for its weights and cache. `LMObjective.policy(params)` returns the same kind of task bound to a training tree, and `SampledRollout` samples with it. To serve the weights with Ollama or vLLM, export them and point the runtime at the directory; the [README](https://github.com/AshishKumar4/dew/blob/main/README.md#exporting-a-decoder-and-serving-it) walks through it.

### Loading from the Hub

From the Hub, the loader fetches the configs and indexes first, then only the weight files the index's `weight_map` names, or `model.safetensors` when there is no index. Other copies in the repo stay on the Hub: Mistral's `consolidated.safetensors`, a pipeline's fp16 variants, and root single-file checkpoints. A tensor stored in two shards is refused. `Pretrained.revision` records the commit the Hub resolved, so a branch that moves later does not change what you loaded; it is `None` for a local directory.

`param_dtype` defaults to float32 master weights; `param_dtype="auto"` stores parameters in the checkpoint's own dtype: `dtype` in `config.json`, or else the first floating tensor's, as transformers' `dtype="auto"` reads it. `load_pretrained(..., mesh=MeshSpec(...), layout=Layout(...))`, which `dew.pipeline` calls, places each weight on its devices directly from the memory-mapped checkpoint, one device shard at a time, so the host never holds the whole translated model ([Distributed training](distributed.md) has the measurements).

A config field Dew cannot express is refused by name. The exceptions are the fields in `_INERT_FIELDS` (`dew/interop/hf_decoders.py`), which transformers 5.16.1 neither declares nor reads, such as SmolLM2's `transformers.js_config`, Qwen2.5's `use_mrope: false` and the Mamba2 ports' `rms_norm`. A family reads only the fields its transformers config class declares. A Llama config that names a sliding window is refused, because transformers' Llama attends to every key; the window is read only for a model type that applies it, such as Mistral or Ministral. MLX quantization is refused by name.

The real prompt plus the continuation must fit in `model.max_seq_len`; input padding takes no room. `Sampling` holds temperature, top-k, top-p, min-p, EOS and the output padding ID. For a loaded checkpoint, the source's `generation_config.json` fills it in, and `text_generation(sampling=...)` overrides it. Generation prepares the inputs on the host, then runs prefill and decode in one compiled call with one padded input shape and a cache cursor per row, so inputs with different valid lengths but the same padded shape reuse the executable.

The base checkpoints used below are not instruction-tuned chat models. With an instruction-tuned model, use the chat template its checkpoint documents. The [Supported models](../models.md) page lists which decoder families load, train, generate and export.

### Other weight formats

- **GGUF.** `load_pretrained(repo, gguf_file="Model-Q4_K_M.gguf")` reads one llama.cpp file (`pip install 'dewml[gguf]'`). The file's metadata becomes the config, and every tensor is dequantized to float32 on the host. The model then trains and generates like a safetensors load, and `param_dtype` sets how it is stored. The tokenizer comes from the file when the repo has none. The llama, qwen2 and qwen3 architectures load; anything else is refused. A repo that ships only GGUF files, loaded without `gguf_file`, lists its files and names the argument that loads one.
- **PyTorch pickles.** A repo with only `pytorch_model.bin` (or its index) also loads. If SFconvertbot has opened a safetensors pull request for that commit, Dew loads that `refs/pr/N` revision, logs which one, and records its commit in `Pretrained.revision`. Otherwise, with torch installed (`pip install 'dewml[torch]'`), Dew unpickles the files once with `torch.load(weights_only=True)`, which runs no code from the file. It saves them as safetensors under `~/.cache/dew/converted/` (`$XDG_CACHE_HOME/dew/converted/` when that is set), and later loads read that copy without importing torch. Without torch and without a pull request, the load is refused with the extra to install and the https://huggingface.co/spaces/safetensors/convert space. Local directories work the same way.
- **mamba_ssm checkpoints.** `state-spaces/mamba2-*` and other checkpoints in mamba_ssm's own format have a config.json with `d_model`, `n_layer` and `ssm_cfg` but no `model_type`. They load as the transformers Mamba2 port that transformers' conversion script would produce, and `save` writes that port.

### Model types without a Dew family

Loading works in three tiers.

1. **Native family.** The model type is registered, and a test pins its numbers against transformers. It gets Dew's attention kernels, sharding rules, cached generation and export.
2. **Verified mapping.** Some unregistered model types compute the Llama block, for example CWM. Before downloading the weights, `load_pretrained` builds transformers' own class for that type from the same config at a much smaller size, with random weights, loads it through Dew's Llama mapping and compares the logits in float32. If they match, the real load goes ahead as a native Dew model and emits one `VerifiedMappingWarning` that names tier 2, the transformers version and the measured error. The check needs torch (`pip install 'dewml[torch]'`); without torch the load is refused and the refusal names that extra. A type that computes something else is refused with the measured difference, for example SmolLM3 (layers without rotary positions), Granite (multipliers) or Phi-3 (fused projections).
3. **Generic, opt-in.** `load_pretrained(repo, fallback="torchax")` builds transformers' PyTorch model on the host and runs its forward through torchax. It compiles, differentiates and trains through `Trainer` and `LMObjective`. The variables keep the source's torch names and are split into `params`, which the optimizer updates, and `buffers`, which it never touches. `TorchLayout` places them on a mesh by name. This tier has no Dew kernels and no cached generation (`text_generation` refuses), and reads float weights only. It pins the torch version (`pip install 'dewml[torchax]'`), and the load keeps the torch model in host memory too, so plan for at least twice the checkpoint size. The load warns and names the tier.

### Multimodal checkpoints

A multimodal checkpoint loads as a `MultimodalTransformer`. It holds the text decoder, the vision tower and projector, and for Gemma 3n and Gemma 4 also the audio tower and its embedder, under the variable names `language_model`, `tower`, `projector`, `audio_tower` and `audio_projector`. The checkpoint's processor resizes, normalizes, patches and expands placeholders; Dew's `Processor` runs it and lays out its outputs row by row.

The next examples read the tiny checkpoints under `tests/fixtures/hf` of a Dew repository checkout. They run on CPU and show mechanics, not language quality. This one loads a tiny Gemma 3 and gives it images:

```python
from pathlib import Path

import numpy as np
from dew.interop import load_pretrained

fixtures = Path(dew.__file__).parents[2] / "tests" / "fixtures" / "hf"
source = fixtures / "gemma3-native-tiny"
bundle = load_pretrained(source, dtype="float32", max_seq_len=64)
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

The two rows above have different numbers of images, so the processor left-pads the shorter one. `inputs.token_fields["attention_mask"]` is the only record of which slots are valid, and padded slots take no cache. A batch with nothing to pad has no `attention_mask` at all; that is how the host says every slot is real. Without a mask, the model passes causality to the kernel as a flag and attention runs fused; an all-true mask would prevent that ([Performance measurements](../performance.md)). Code that handles both cases reads the field with `token_fields.get("attention_mask")`.

Padding belongs to each process's own rows, so on a process pool the processes agree about a missing mask before anything is assembled. A generation request creates the field on every process if any process has it, and leaves it out if none does. A training batch placed by `shard_batch` has it on every process. For text-only prompts, leave out `images`. Gemma 3n and Gemma 4 take `audio=[waveform, ...]`, one waveform per audio placeholder in reading order.

The processor checks token IDs, placeholder counts and media shapes on the host; the compiled model does no checking. The same `ModelInputs` goes to `model.apply`, the objective and `generate`. Media are evaluated at prefill, and decode steps read the cache.

To keep training, build the bundle's objective and feed the trainer `{"text": inputs}` batches:

```python
import optax
from dew.data.dataset import Dataset
from dew.training import Layout, MeshSpec

mm_objective = bundle.lm_objective(inputs.tokens.shape[1] - 1, pad_id=0)
rows = 2 * jax.device_count()
batch = inputs.take_rows(jax.numpy.arange(rows) % 2)
mm_data = Dataset(train=lambda partition: iter([{"text": batch}]), val=None,
                  records=rows, batch=rows)
mm_trainer = Trainer(mm_objective, optax.sgd(1e-4), key=jax.random.key(3),
                     mesh=MeshSpec(), layout=Layout(min_shard=2**30))
mm_state = mm_trainer.fit(mm_data, steps=1, log_every=1)
bundle.save("gemma3-tiny-step1", variables=mm_state.params)
```

`bundle.save(directory, variables=...)` writes the trained weights back under the source tensor names, together with the processor, so the directory loads again both in Dew and in Transformers. This includes Gemma 4's frozen standardization and clipping buffers, which live in the `constants` collection and stay bit-for-bit unchanged through training. Past `max_shard_size` (default `"5GB"`) the weights go out as numbered shards with their index. A dense decoder or a source-layout checkpoint builds each tensor on the host only when its shard is written; a quantized source, `save_pretrained_decoder` of a Gemma 4 or GLM-5-next model, and a DiffusionGemma export build the whole export first. Saving over an earlier export replaces it in one step: the new shards take fresh names and the index is written last, so a save that stops partway leaves the previous export whole, and files the export did not write, such as `model.fp16.safetensors`, are left alone.

Each family's processor emits what its reference implementation expects:

- Gemma 3 gives one fixed-resolution image per placeholder block.
- Llama 4 splits each image into local tiles and a global tile with separator tokens. It normalizes pixels in bfloat16, as the original implementation does, and the loader widens them to float32 exactly.
- Gemma 4 emits padded patch streams with 2D patch positions. It expands video placeholders, which the decoder maps to the pad embedding.
- Qwen 3.5 packs patches channel first, then time, with a grid per image. The loader derives the spatial rotary coordinates that the reference's `get_rope_index` computes.
- Gemma 3n uses the MobileNet-v5 encoder. It embeds its hard vision and audio vocabulary ranges through the multimodal embedders and keeps placeholder IDs for its per-layer inputs while masking the hard ranges, on both training and decode steps.
- DeepSeek-V4.1 ships no processor. Its release resizes and pads each image in its own `inference/image_processor.py`, which Dew does not reproduce, so the caller builds the inputs it writes. `pixel_values` is `[B, images, 3, H, W]`, normalized to [-1, 1], with H and W multiples of the patch size. Each image fills `rows * (columns + 1) + 2` positions of `image_token_id`, where rows and columns count the aligner's `downsample_ratio` squares over the patch grid. `image_indices` numbers those positions in reading order.

Audio clips carry a mask that is True for valid frames. Gemma 4 inserts one placeholder per encoded frame. Gemma 3n inserts a fixed `audio_soft_tokens_per_image` per clip and fills the remaining slots with the embedder's padding token.

The Gemma 3n and Gemma 4 audio encoders are `dew.nn.audio.Gemma3nAudio` and `Gemma4Audio`, registered as the towers `gemma3n_audio` and `gemma4_audio`. `audio_config` reads the checkpoint's `audio_config` record and rejects unknown computational fields. `audio_weights` converts the tower's own tensors and keeps Gemma 4's checkpointed clipping bounds in a frozen `constants` collection. An encoder takes `input_features` shaped `[B, T, F]` and a boolean `input_features_mask` that is True for valid frames, and returns `AudioEncoding(features, mask)`, with the mask subsampled to the encoder's frame rate. `dew.data.audio.AudioProcessor` builds the checkpoint's feature extractor from its `preprocessor_config.json` record and turns 16 kHz mono waveforms into those two arrays; it does not resample. Gemma 3n projects audio through `Gemma3nProjectorModule.soft_embeddings`, without the scaling used for vision. Gemma 4 reuses `Gemma4ProjectorModule`, with its input width taken from `output_proj_dims`.

## Masked diffusion language models

`MaskedDiffusionObjective` trains a bidirectional decoder to recover masked tokens under MDLM's negative ELBO. Each row holds exactly `seq_len` tokens; there is no next-token shift, so `TokenWindows(seq_len=63)`, whose rows hold 64 IDs, feeds a 64-token objective. The mask token needs an ID of its own, here the one after the last byte:

```python
from dew.diffusion.discrete import MDLM
from dew.objectives.diffusion import MaskedDiffusionObjective

masked_data = TokenWindows(path="data/stories-byte", seq_len=63,
                           loading=Loading(workers=0)).load(batch=16)
mask_id = tokenizer.vocab_size
process = MDLM(mask_id=mask_id)()
masked_model = models.build("causal_transformer", vocab_size=mask_id + 1,
                            emb_features=64, num_layers=2, num_heads=2,
                            mlp_features=256, max_seq_len=64, causal=False)
masked_objective = MaskedDiffusionObjective(masked_model, process, seq_len=64)
masked_state = Trainer(masked_objective, optax.adamw(3e-3), key=jax.random.key(4)).fit(
    masked_data, steps=1000, log_every=500)
drawn = process.generate(masked_model, masked_state.params,
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

The loss is MDLM's negative ELBO per token. `process.generate` unmasks the 24 tokens after the prompt together, in 64 reverse steps by default (`steps=`), so a model this small draws fragments of its corpus rather than a sentence. On TinyStories, with the GPT-2 tokenizer, a 4-layer, 256-wide model over 128-token rows at batch 64 took about five minutes for 4,000 steps on an L4. Its loss fell from 3.48 at step 1,000 to 2.78 at step 4,000, and validation perplexity reached 15.0. That perplexity is exp of the ELBO, an upper bound on the model's own, so it does not compare directly with an autoregressive decoder's. One draw of 48 tokens read:

> Once upon a time, in a small town, there lived a little girl named Lily. Mia loved whistle. She had an key telling to try to sleep. One day, she always to see her favorite mom, dad unate to have lots of.

LLaDA and Dream use this masked-token path from their released weights. They predict masked tokens with bidirectional attention and need a mask token ID and a masked-diffusion objective; swapping out the autoregressive loss is not enough, because the attention and the corruption process change too. For these models, `load_pretrained(...).text_generation()`, `dew.pipeline(source_or_run)` and `MaskedDiffusionObjective.pipeline(state)` return `MaskedGeneration`. It runs Dew's native MDLM algorithm (`DiscreteProcess` with `Unmask`), not the source-specific remasking and block-generation recipes of LLaDA or Dream.

```python
llada = load_pretrained(fixtures / "llada-tiny", dtype="float32", attention_impl="xla")
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

Saved masked-diffusion recipe runs keep their compute and storage precision, and `dew.pipeline`'s `ema` option picks live or EMA weights. The run record rebuilds native MDLM with its default `Unmask` sampler and 64 steps; it does not save custom objective steps or sampler choices set in code. Plain `Checkpoints` saves weights and training state, not these task settings, so a custom task configuration has to be applied again. The recipe does not save an EOS policy either. Source checkpoints follow their own EOS metadata. For a task built from an objective in code, set EOS with `dataclasses.replace(task, eos_token_ids=(...))`.

## DiffusionGemma

DiffusionGemma uses uniform-vocabulary corruption, a causal prompt encoder, a bidirectional canvas and self-conditioning. `load_pretrained` loads the complete model. Its `BlockProcess` generation policy refines whole canvases and commits clean tokens to the shared text cache. It returns `CanvasGeneration` with response lengths (including EOS), termination flags and per-row refinement counts, and no autoregressive likelihood fields.

The next example loads the tiny reference checkpoint, then tokenizes, generates, decodes, saves and reloads it. Its synthetic vocabulary has 64 tokens. The released model ID is `google/diffusiongemma-26B-A4B-it`, which downloads large weights and needs matching host and device memory.

```python
from tempfile import TemporaryDirectory

gemma = load_pretrained(fixtures / "diffusion-gemma-workflow", dtype="float32",
                        attention_impl="xla", max_seq_len=32)
prompt = gemma.processor(["<bos> t5 t7 t9 t11"])
block_task = gemma.block_generation()
generated = block_task(prompt, 7, key=jax.random.key(11))
print(block_task.decode(generated))
with TemporaryDirectory() as checkpoint:
    gemma.save(checkpoint)
    restored = load_pretrained(checkpoint, dtype="float32", attention_impl="xla", max_seq_len=32)
    replay = restored.block_generation()(prompt, 7, key=jax.random.key(11))
    np.testing.assert_array_equal(replay.tokens, generated.tokens)
```

```text
('t37 t49 t62 t14 t23 t49 t34',)
```

The last canvas is refined at full width, and the returned response is then cut to `max_new_tokens`. The prefix plus the canvas capacity, rounded up to whole canvases, must fit in `max_seq_len`. Generation defaults come from `generation_config.json`; pass a `BlockProcess` as `process=` to override them. Media go through the checkpoint's processor and run only during prompt prefill, not on every refinement. The [training-contract note](../research/inference.md#diffusiongemma-training-contract-and-open-prerequisites) separates the official fine-tuning recipe, which is public, from the original sampler-distillation and RL objective, which Google has not published.

### Block diffusion fine-tuning

`BlockDiffusionObjective` ports Google's public SFT adapter, released after the model. It samples a valid response canvas, corrupts the whole response with uniform-vocabulary noise, and runs self-conditioning with a detached first pass. It then combines the canvas loss and the encoder loss, each normalized per row on its own. The default time safety margin is 1e-4 and the self-conditioning probability is 0.5. It is not the unpublished sampler-distillation and RL objective.

The next example takes one optimizer step on the tiny reference model, with a synthetic vocabulary and unequal target support. The objective makes the per-layer scalars trainable, while the checkpoint's ordinary HF view keeps those tensors frozen, so the export passes the objective's native model.

```python
from dataclasses import replace

from dew.objectives.diffusion import BlockDiffusionObjective

sft_source = fixtures / "diffusion-gemma-sft"
sft_bundle = load_pretrained(sft_source, dtype="float32", attention_impl="xla", max_seq_len=32)
with np.load(sft_source / "reference.npz") as reference:
    train_tokens = np.tile(reference["tokens"], (jax.device_count(), 1))
block_data = Dataset(train=lambda partition: iter([{"text": train_tokens}]), val=None,
                     records=len(train_tokens), batch=len(train_tokens))
block_objective = BlockDiffusionObjective(sft_bundle.model, prompt_length=4,
                                          num_canvases=2, pretrained=sft_bundle.variables)
block_state = Trainer(block_objective, optax.sgd(0.001), key=jax.random.key(2)).fit(
    block_data, steps=1, log_every=1)
with TemporaryDirectory() as checkpoint:
    replace(sft_bundle, model=block_objective.model).save(checkpoint, variables=block_state.params)
    trained_bundle = load_pretrained(checkpoint, dtype="float32", attention_impl="xla",
                                     max_seq_len=32)
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

The token files must use the checkpoint's tokenizer and lay out the clean prompt prefix and the response canvases you intend. Trainer checkpoints keep the optimizer and iterator state. `Pretrained.save` writes a complete inference checkpoint in the source format instead.
