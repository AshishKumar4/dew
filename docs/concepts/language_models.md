# Training language models

This page assumes you have done the [first training run](../getting-started.md) and know how next-token prediction works. You do not need a pretrained model to run the first example. It uses a synthetic vocabulary of four tokens, so you can look at every input and output.

## Train a small decoder

```python
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew import Trainer
from dew.data import Dataset
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling, generate

row = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0], dtype=np.int32)
tokens = np.tile(row, (8, 1))
data = Dataset(train=lambda partition: itertools.repeat({"text": tokens}),
               val=None, records=8, batch=8)
model = CausalTransformer(vocab_size=4, emb_features=16, num_layers=1,
                          num_heads=2, mlp_features=32, max_seq_len=16,
                          dtype=jnp.float32, attention_impl="xla")
objective = LMObjective(model, seq_len=8)
trainer = Trainer(objective, optax.adam(0.01), key=jax.random.key(0))
state = trainer.fit(data, steps=30, log_every=15)
prompt = jnp.array([[0, 1]], dtype=jnp.int32)
result = generate(model, state.params, prompt, max_new_tokens=6,
                  key=jax.random.key(1), sampling=Sampling(temperature=0.0))
print("Generated token IDs:", np.asarray(result.tokens).tolist())
np.testing.assert_array_equal(np.asarray(result.tokens[:, :2]), np.asarray(prompt))
```

`LMObjective` reads token rows of shape `(B, S + 1)`. It feeds the first `S` tokens to the model and scores the predictions against the next `S` tokens. Each row here has nine tokens, so `seq_len=8`. Token IDs must be integers inside the model's vocabulary.

Causal attention stops a position from reading later tokens. The model has a hidden width of 16, two attention heads and a feed-forward width of 32. The example uses float32 and XLA attention so it also runs on CPU. These sizes are for learning the API. Do not use them to compare model quality or throughput.

The training loss should go down as the decoder learns the repeating pattern. `result.tokens` holds the prompt followed by six generated token IDs. `Sampling(temperature=0.0)` picks the highest-scoring token at each step, so the behavior log-probability of each pick is zero. You still pass the key explicitly. The exact output can change with the library version and the initialization.

## Tokenize real text

For real text, pick one tokenizer and use its vocabulary everywhere: data preparation, model construction, decoding and checkpoint loading. `ByteTokenizer` treats UTF-8 bytes as tokens, with a vocabulary of 256. A Hugging Face tokenizer uses the chosen model's vocabulary and chat template, and may download files the first time you use it.

From a repository checkout, prepare your own corpus and write token files:

```bash
mkdir -p data
printf 'A small corpus for a tokenizer demonstration.\n' > data/corpus.txt
python tools/tokenize_text.py --input data/corpus.txt --out data/corpus-byte --tokenizer byte --val-fraction 0.1
```

This writes `train.bin`, `val.bin` and `meta.json`. The binary files hold token IDs. The metadata records their dtype, counts and tokenizer. This tiny corpus only shows the preparation step. Before training, use enough text to fill the windows, batches and held-out split you ask for.

`TokenWindows(path, seq_len).load(batch=...)` reads fixed-width windows. `PackedTokens` packs documents together and adds `text_segment_ids` and `text_positions`. The objective skips padded targets and the transitions between documents. Packing can change which attention implementations you can use, because the model then needs a segment mask.

## Loss, precision, and evaluation

The vocabulary loss runs in float32. The head's product follows the compute dtype, as torch autocast and MaxText run it: under bf16 compute the states and the head multiply as bf16 and sum in float32, backward included, 1.6x faster on an L4 and an RTX 4080 than fp32 operands; an fp32 model multiplies in fp32 ([performance](../performance.md#kernel-choices-per-generation-2026-09-22)). The step reports `ce`, `perplexity` and `token_accuracy`. `token_accuracy=False` on `LMObjective`, `--no-token-accuracy` on the recipe, drops the accuracy and the pass over every logit it costs: 0.77 ms of the head's 8.0 ms on a TPU v6e. `head_chunks` sets how many vocabulary slices the loss scores; the default is four. Chunking can lower peak memory, but it adds work, and the backend compiler may rewrite it. How much memory it saves depends on the vocabulary size, sequence length, batch size and the compiled executable. There is no fixed saving.

With bf16 compute, Dew keeps the residual stream and every norm and sublayer output in bf16, each rounded where transformers rounds a bf16 tensor, as MaxText and Megatron (with `fp32_residual_connection=False`) do; `torch.autocast` instead keeps the residual stream and the norms in float32 and rounds at each matmul's input. The torch run with Dew's rounding points is autocast with the embedding output and every RMSNorm output cast to bf16 and the rotary table left in float32 (`tools/reference_runs/torch_lm.py --precision autocast-bf16-residual`). At those rounding points the first step's loss depends on the attention kernel as much as on the framework. On the first batch of a Qwen3-0.6B fine-tune (4 x 1024 tokens, one A100, `tools/reference_runs/step0_attention.py`):

| Run | Attention kernel | First-step loss minus float32 | Final hidden states, relative distance from torch's FlashAttention 2 run |
|---|---|---|---|
| torch | FlashAttention 2 | -1.60e-4 | 0 |
| torch | cuDNN | +8.17e-4 | 1.42e-2 |
| torch | math | -1.40e-4 | 1.42e-2 |
| Dew | cuDNN | +1.09e-3 | 1.38e-2 |
| Dew | XLA | +1.10e-3 | 1.40e-2 |

Plain `torch.autocast` with FlashAttention 2 sits at +4.9e-4. Over 256 steps of the same fine-tune, every 32-step window of Dew's loss and gradient norm stays within twice the distance from float32 of the torch run at Dew's rounding points, or twice that run's run-to-run spread where the spread is larger.

`LMObjective` keeps an EMA copy by default. Use `state.params` for the live variables and `state.averaged` when you want the moving-average copy. Evaluation reads the averaged variables when the objective keeps them. With `ema_decay=None` there is no copy: `state.ema` is `None`, `state.averaged` raises, and previews and evaluation read the live variables. A checkpoint written with one of these settings does not restore into the other.

To measure validation perplexity, pass a validation iterator and set `eval_every`, as shown in [evaluation and tracking](../guides/evaluation.md). With a tracker, `Samples` sets up one generated preview per event. That preview is separate from teacher-forced scoring of the complete batch.

With gradient accumulation, Dew weights the cross-entropy and multi-token-prediction (MTP) losses by the mass of supported main targets, including the target-role masks at each MTP depth. Sequence-level router losses normalize by rows. Global router losses first add up the selected-slot counts and the score sums, then take their product. The routing bias stays fixed during an accumulation window and is updated from the window's total counts. A window with zero cross-entropy can still update when a router auxiliary loss is active. Accumulated and combined batches match only when their random draws are the same and mutable state follows its declared rules.

## Generate from a checkpoint

`load_pretrained` reads a local Hugging Face directory or a Hub identifier into a `Pretrained` bundle. The bundle holds the native Flax model, its variables tree, the checkpoint's own processor or tokenizer, the source config and the generation defaults. `dew.pipeline(source)` is the simpler entry point over the same loader. It returns a `TextGeneration` (or a `BlockGeneration` for DiffusionGemma) with its weights placed once on the current device mesh, and with the sampling policy and budget taken from the checkpoint. [Inference](inference.md) covers placement, `seed`, `host()` and `text`, and the workflows that start from a trained objective, a run directory and a published checkpoint.

From the Hub, the loader fetches the configs and indexes first, then only the weight files the index's `weight_map` names, or `model.safetensors` when there is no index. Other copies in the repo stay on the Hub: Mistral's `consolidated.safetensors`, a pipeline's fp16 variants, and root single-file checkpoints. A tensor stored in two shards is refused. `Pretrained.revision` records the commit the Hub resolved, so a branch that moves later does not change what you loaded; it is `None` for a local directory. `param_dtype="auto"` stores parameters in the checkpoint's own dtype: `dtype` in `config.json`, or else the first floating tensor's, as transformers' `dtype="auto"` reads it. `load_pretrained(..., mesh=MeshSpec(...), layout=Layout(...))`, which `dew.pipeline` calls, places each weight on its devices directly from the memory-mapped checkpoint, one device shard at a time. The host never holds the whole translated model ([Distributed](distributed.md) has the measurements).

A config field Dew cannot express is refused by name. The exceptions are the fields in `_INERT_FIELDS` (`dew/interop/hf_decoders.py`), which transformers 5.16.1 neither declares nor reads, such as SmolLM2's `transformers.js_config`, Qwen2.5's `use_mrope: false` and the Mamba2 ports' `rms_norm`. A family reads only the fields its transformers config class declares. A Llama config that names a sliding window is refused, because transformers' Llama attends to every key; the window is read only if the model type is one that applies it, such as Mistral or Ministral. MLX quantization is refused by name.

The real prompt plus the continuation must fit in `model.max_seq_len`. Input padding takes up no room. `Sampling` holds temperature, top-k, top-p, min-p, EOS and the output padding id. For a loaded checkpoint, the source's `generation_config.json` fills it in, and `text_generation(sampling=...)` overrides it. Generation prepares the inputs on the host, then runs prefill and decode in one compiled call with one padded input shape and a cache cursor per row. Inputs with different valid lengths but the same padded shape reuse the executable.

```python
import dew

task = dew.pipeline("tests/fixtures/hf/gemma3-native-tiny", dtype="float32")
print(task("token7 token9", 3, seed=1).text[0])
```

This example runs offline on the tiny Gemma 3 fixture the wrapper tests use. A Hub name such as `"Qwen/Qwen3-0.6B"` works the same way with the `interop` extra installed and a download. A real checkpoint needs enough host and device memory for its weights and cache.

`LMObjective.policy(params)` returns the same kind of task, already bound to a training tree. `SampledRollout` samples with it. To run the weights in another runtime, export them and point Ollama or vLLM at the directory. The [README](https://github.com/AshishKumar4/dew/blob/main/README.md#exporting-a-decoder-and-serving-it) goes through this up to the client call.

The next example loads the same fixture with its processor and gives it images:

```python
import jax
import numpy as np
from dew.interop import load_pretrained
from dew.sampling import Sampling

source = "tests/fixtures/hf/gemma3-native-tiny"
bundle = load_pretrained(source, dtype="float32", max_seq_len=64)
images = np.load(f"{source}/raw_images.npy")          # uint8 [3, 32, 32, 3]
inputs = bundle.processor(["token7 <start_of_image> token9",
                           "token5 <start_of_image> token8 <start_of_image> token6"],
                          images=[[images[0]], [images[1], images[2]]])
logits = bundle.model.apply(bundle.variables, inputs.tokens, **inputs.kwargs())
task = bundle.text_generation(sampling=Sampling(temperature=0))
for text in task(inputs, 3, seed=1).text:
    print(text)
```

The two rows have different numbers of images, so the processor left-pads the shorter one. `inputs.token_fields["attention_mask"]` is the only record of which slots are valid, and padded slots take up no cache. A batch with nothing to pad has no `attention_mask` at all. That is how the host says every slot is real. Without a mask, the model passes causality to the kernel as a flag and attention runs fused; an all-true mask would prevent that (see `docs/performance.md`). If your code has to handle both cases, read the field with `token_fields.get("attention_mask")`.

Padding belongs to each process's own rows, so on a process pool the processes agree about a missing mask before anything is assembled. A generation request creates the field on every process if any process has it, and leaves it out if none does. A training batch placed by `shard_batch` has it on every process. For text-only prompts, leave out the `images` argument. Gemma 3n and Gemma 4 take `audio=[waveform, ...]`, one waveform per audio placeholder in reading order.

To keep training, hand the loaded variables to the objective and feed the trainer `{"text": inputs}` batches. `bundle.save(directory, variables=state.params)` writes the trained weights back under the source tensor names, together with the processor, so the directory loads again both here and in Transformers.

```python
import optax
from dew.data.dataset import Dataset
from dew.objectives.lm import LMObjective
from dew.training import Layout, MeshSpec, Trainer

objective = LMObjective(bundle.model, inputs.tokens.shape[1] - 1,
                        pretrained=bundle.variables, ema_decay=None, pad_id=0)
rows = 2 * jax.device_count()
batch = inputs.take_rows(jax.numpy.arange(rows) % 2)
data = Dataset(train=lambda partition: iter([{"text": batch}]), val=None, records=rows, batch=rows)
trainer = Trainer(objective, optax.sgd(1e-4), key=jax.random.key(3),
                  mesh=MeshSpec(), layout=Layout(min_shard=2**30))
state = trainer.fit(data, steps=1, log_every=1)
bundle.save("gemma3-tiny-step1", variables=state.params)
```

The base fixture is not an instruction-tuned chat model. When you load an instruction-tuned model, use the chat template its checkpoint documents. The [README model list](https://github.com/AshishKumar4/dew/blob/main/README.md#models) says which decoder families load, train, generate and export.

### Other weight formats

- **GGUF.** `load_pretrained(repo, gguf_file="Model-Q4_K_M.gguf")` reads one llama.cpp file (`pip install 'dew-ml[gguf]'`). The file's metadata becomes the config, and every tensor is dequantized to float32 on the host. The model then trains and generates like a safetensors load, and `param_dtype` sets how it is stored. The tokenizer comes from the file when the repo has none. The llama, qwen2 and qwen3 architectures load; anything else is refused. A repo that ships only GGUF files, loaded without `gguf_file`, lists its files and tells you which argument loads one.
- **PyTorch pickles.** A repo with only `pytorch_model.bin` (or its index) also loads. If SFconvertbot has opened a safetensors pull request for that commit, Dew loads that `refs/pr/N` revision, logs which one, and records its commit in `Pretrained.revision`. Otherwise, with torch installed (`pip install 'dew-ml[torch]'`), Dew unpickles the files once with `torch.load(weights_only=True)`, which runs no code from the file. It saves them as safetensors under `~/.cache/dew/converted/` (`$XDG_CACHE_HOME/dew/converted/` when that is set), and later loads read that copy without importing torch. Without torch and without a pull request, the load is refused with the extra to install and the https://huggingface.co/spaces/safetensors/convert space. Local directories work the same way.
- **mamba_ssm checkpoints.** `state-spaces/mamba2-*` and other checkpoints in mamba_ssm's own format have a config.json with `d_model`, `n_layer` and `ssm_cfg` but no `model_type`. They load as the transformers Mamba2 port that transformers' conversion script would produce, and `save` writes that port.

### Models Dew has no family for

Loading works in three tiers.

1. **Native family.** The model type is registered, and a test pins its numbers against transformers. You get Dew's attention kernels, sharding rules, cached generation and export. There is nothing extra to do.
2. **Verified mapping.** Some model types are not registered but compute the Llama block, for example CWM. Before downloading the weights, `load_pretrained` builds transformers' own class for that type from the same config at a much smaller size, with random weights. It loads that small model through Dew's Llama mapping and compares the logits in float32. If they match, the real load goes ahead as a native Dew model, with the same kernels, sharding and cached generation. You get one `VerifiedMappingWarning` that names tier 2, the transformers version and the measured error. The check needs torch (`pip install 'dew-ml[torch]'`); without torch the load is refused and the refusal names that extra. A type that computes something else is refused with the measured difference, for example SmolLM3 (layers without rotary positions), Granite (multipliers) or Phi-3 (fused projections).
3. **Generic, opt-in.** `load_pretrained(repo, fallback="torchax")` builds transformers' PyTorch model on the host and runs its forward through torchax. It still compiles, differentiates and trains through `Trainer` and `LMObjective`. The variables keep the source's torch names and are split in two: `params`, which the optimizer updates, and `buffers`, which it never touches. `TorchLayout` places them on a mesh by name. This tier has no Dew kernels and no cached generation (`text_generation` refuses), and it reads float weights only. It pins the torch version (`pip install 'dew-ml[torchax]'`), and the load keeps the torch model in host memory too, so plan for at least twice the checkpoint size. You get a warning that names the tier.

## Diffusion language models and media inputs

LLaDA and Dream predict masked tokens with bidirectional attention. They need a mask token ID and a masked-diffusion objective. Swapping out the autoregressive loss is not enough; the attention and the corruption process have to change too.

For these models, `load_pretrained(...).text_generation()`, `dew.pipeline(source_or_run)` and `MaskedDiffusionObjective.pipeline(state)` return `MaskedGeneration`. It runs Dew's native MDLM algorithm (`DiscreteProcess` with `Unmask`). It does not reproduce the source-specific remasking and block-generation recipes of LLaDA or Dream.

```python
from dew.interop import load_pretrained

bundle = load_pretrained("tests/fixtures/hf/llada-tiny", dtype="float32", attention_impl="xla")
task = bundle.text_generation()
result = task([[1, 2, 3]], 8, seed=7, steps=16, n=2)
print(result.host().tokens)
```

The tiny fixtures show the mechanics, not language quality. Released sources that ship a tokenizer also accept text and provide `result.text`. This path handles text only and rejects media payloads and media token fields. Numeric requests use `ModelInputs`. Attention validity, logical `[B, S]` positions, rotary coordinates and segment IDs pass through the masked model. The prompt stays fixed, even if it contains a literal mask ID. All requested response positions are refined together under bidirectional attention, and mask IDs are never drawn as generated tokens.

The default is 64 model evaluations, including the final clean prediction. A call can override `steps`. The `n` continuations are grouped by prompt, and continuation zero does not change when you ask for more. EOS is applied after the whole span is refined: lengths include the first EOS, and the rest is padded. This is not autoregressive early stopping. A request for zero tokens keeps the prompt and reports zero refinements. `CanvasGeneration` reports lengths, EOS flags and refinement counts, and has no autoregressive log-probabilities. This path does not accept autoregressive sampling, beam or logits controls.

If the source turns on a generation control that native MDLM cannot follow, Dew rejects it by name. Neutral values are accepted, as are the shared budget, continuation count, EOS and padding metadata. MDLM does not use a KV cache, so `use_cache=False` is accepted and asking for a cache is not.

Saved masked-diffusion recipe runs keep their compute and storage precision, and you pick live or EMA weights through the usual `dew.pipeline` options. The run record rebuilds native MDLM with its default `Unmask` sampler and 64 steps. It does not save custom objective steps or sampler choices you set in code. Plain `Checkpoints` saves weights and training state, not these task settings, so you must keep your custom task configuration and apply it again yourself. The recipe does not save an EOS policy either. Source checkpoints follow their own EOS metadata. For a task built from an objective in code, set EOS with `dataclasses.replace(task, eos_token_ids=(...))`. The recipe saves no other inference fields.

DiffusionGemma uses uniform-vocabulary corruption, a causal prompt encoder, a bidirectional canvas and self-conditioning. Load the complete model through the same `load_pretrained` interface as other published models. Its `BlockProcess` generation policy refines whole canvases and commits clean tokens to the shared text cache. It returns `CanvasGeneration` with response lengths (including EOS), termination flags and per-row refinement counts, and no autoregressive likelihood fields.

From a repository checkout, the next example loads the complete tiny reference checkpoint, then tokenizes, generates, decodes, saves and reloads it. It runs on CPU. Its synthetic vocabulary has 64 tokens, so it tests the workflow, not language quality. The released model ID is `google/diffusiongemma-26B-A4B-it`. Loading it downloads large weights and needs enough host and device memory.

```python
from tempfile import TemporaryDirectory

import jax
import numpy as np
from dew.interop import load_pretrained

source = "tests/fixtures/hf/diffusion-gemma-workflow"
bundle = load_pretrained(source, dtype="float32", attention_impl="xla", max_seq_len=32)
assert bundle.processor is not None
inputs = bundle.processor(["<bos> t5 t7 t9 t11"])
task = bundle.block_generation()
generated = task(inputs, 7, key=jax.random.key(11))
print(task.decode(generated))
with TemporaryDirectory() as checkpoint:
    bundle.save(checkpoint)
    restored = load_pretrained(checkpoint, dtype="float32", attention_impl="xla", max_seq_len=32)
    replay = restored.block_generation()(inputs, 7, key=jax.random.key(11))
    np.testing.assert_array_equal(replay.tokens, generated.tokens)
```

The last canvas is refined at full width, and the returned response is then cut to `max_new_tokens`. The prefix plus the canvas capacity, rounded up to whole canvases, must fit in `max_seq_len`. Generation defaults come from `generation_config.json`. To override them, pass a `BlockProcess` as `process=`. Media go through the checkpoint's processor and run only during prompt prefill, not on every refinement. The [training-contract note](../research/inference.md#diffusiongemma-training-contract-and-open-prerequisites) separates the official fine-tuning recipe, which is public, from the original sampler-distillation and RL objective, which Google has not published.

### Fine-tune with the official block loss

`BlockDiffusionObjective` ports Google's public SFT adapter, released after the model. It samples a valid response canvas, corrupts the whole response with uniform-vocabulary noise, and runs self-conditioning with a detached first pass. It then combines the canvas loss and the encoder loss, each normalized per row on its own. The default time safety margin is 1e-4 and the self-conditioning probability is 0.5. This is not the unpublished original sampler-distillation and RL objective.

The next example takes one real optimizer step on the tiny official reference model, on CPU. It uses a synthetic vocabulary and unequal target support on purpose. The objective makes the per-layer scalars trainable, while the checkpoint's ordinary HF view keeps those same tensors frozen. So when you export the trained variables, pass the objective's native model.

```python
from dataclasses import replace
from tempfile import TemporaryDirectory

import jax
import numpy as np
import optax
from dew import Dataset, Trainer
from dew.interop import load_pretrained
from dew.objectives.diffusion import BlockDiffusionObjective

training_source = "tests/fixtures/hf/diffusion-gemma-sft"
training_bundle = load_pretrained(training_source, dtype="float32", attention_impl="xla", max_seq_len=32)
with np.load(training_source + "/reference.npz") as reference:
    train_tokens = np.tile(reference["tokens"], (jax.device_count(), 1))
training_data = Dataset(train=lambda partition: iter([{"text": train_tokens}]), val=None,
                        records=len(train_tokens), batch=len(train_tokens))
block_objective = BlockDiffusionObjective(training_bundle.model, prompt_length=4,
                                         num_canvases=2, pretrained=training_bundle.variables)
block_trainer = Trainer(block_objective, optax.sgd(0.001), key=jax.random.key(2))
block_state = block_trainer.fit(training_data, steps=1, log_every=1)
with TemporaryDirectory() as checkpoint:
    replace(training_bundle, model=block_objective.model).save(checkpoint, variables=block_state.params)
    trained_bundle = load_pretrained(checkpoint, dtype="float32", attention_impl="xla", max_seq_len=32)
print("Optimizer updates:", int(block_state.updates))
```

The same objective is available in `recipes/lm/train.py`; there is no separate recipe for it. It trains on complete token-window rows, and `data.seq_len + 1` must equal `block_prompt_tokens` plus a whole number of training canvases. `block_canvas_size` defaults to the checkpoint's canvas length. Packed documents are rejected, because their context boundaries differ. Block SFT logs `canvas_ce` and `encoder_ce`. It does not report autoregressive perplexity or use the autoregressive preview settings. To generate text, use the shared pretrained interface.

```bash
python recipes/lm/train.py data:token-windows --data.path data/diffusion-token-windows \
    --data.seq-len 511 --block-prompt-tokens 256 \
    --pretrained google/diffusiongemma-26B-A4B-it \
    --tokenizer google/diffusiongemma-26B-A4B-it --objective block_diffusion \
    --sample-tokens 0 --ema-decay None --optim.learning-rate 0.00015
```

The token files must use the checkpoint's tokenizer and lay out the clean prompt prefix and the response canvases you intend. Trainer checkpoints keep the optimizer and iterator state. `Pretrained.save` writes a complete inference checkpoint in the source format instead.

### Multimodal checkpoints

A multimodal checkpoint loads as a `MultimodalTransformer`. It holds the text decoder, the vision tower and projector, and for Gemma 3n and Gemma 4 also the audio tower and its embedder. Their variable names are `language_model`, `tower`, `projector`, `audio_tower` and `audio_projector`. The checkpoint's processor does resizing, normalization, patching and placeholder expansion. Dew's `Processor` runs it and lays out its outputs row by row.

`ModelInputs` has three parts:

- `tokens` is `[B, S]`.
- `token_fields` holds `attention_mask`, `positions`, `image_indices` and `image_groups` (the soft feature and the image behind each slot, -1 for text), `audio_indices` for audio slots, and for Qwen 3.5 the three-axis `rotary_positions`.
- `conditioning` holds the media, padded to the row with the most images or clips. `pixel_values` is `[B, images, ...]` in the processor's own layout, with `image_position_ids` (Gemma 4) or `image_grid_thw` (Qwen 3.5) beside it. `input_features` and `input_features_mask` are `[B, clips, frames, mel]`.

The processor checks token ids, placeholder counts and media shapes on the host, and the compiled model does no checking. The same `ModelInputs` goes to `model.apply`, the objective and `generate`. Media are evaluated at prefill, and decode steps read the cache.

Each family's processor emits what its reference implementation expects:

- Gemma 3 gives one fixed-resolution image per placeholder block.
- Llama 4 splits each image into local tiles and a global tile with separator tokens. It normalizes pixels in bfloat16, as the original implementation does, and the loader widens them to float32 exactly.
- Gemma 4 emits padded patch streams with 2D patch positions. It expands video placeholders, which the decoder maps to the pad embedding.
- Qwen 3.5 packs patches channel first, then time, with a grid per image. The loader derives the spatial rotary coordinates that the reference's `get_rope_index` computes.
- Gemma 3n uses the MobileNet-v5 encoder. It embeds its hard vision and audio vocabulary ranges through the multimodal embedders and keeps placeholder ids for its per-layer inputs while masking the hard ranges, on both training and decode steps.

Audio clips carry a mask that is True for valid frames. Gemma 4 inserts one placeholder per encoded frame. Gemma 3n inserts a fixed `audio_soft_tokens_per_image` per clip and fills the remaining slots with the embedder's padding token.

`bundle.save` writes trained variables under the source tensor names. This includes Gemma 4's frozen standardization and clipping buffers, which live in the `constants` collection and stay bit-for-bit unchanged through training. The [README model list](https://github.com/AshishKumar4/dew/blob/main/README.md#models) lists the media each wrapper takes.

The Gemma 3n and Gemma 4 audio encoders are `dew.nn.audio.Gemma3nAudio` and `Gemma4Audio`, registered as the towers `gemma3n_audio` and `gemma4_audio`. `audio_config` reads the checkpoint's `audio_config` record and rejects unknown computational fields. `audio_weights` converts the tower's own tensors and keeps Gemma 4's checkpointed clipping bounds in a frozen `constants` collection. An encoder takes `input_features` shaped `[B, T, F]` and a boolean `input_features_mask` that is True for valid frames. It returns `AudioEncoding(features, mask)`, with the mask subsampled to the encoder's frame rate. `dew.data.audio.AudioProcessor` builds the checkpoint's feature extractor from its `preprocessor_config.json` record and turns 16 kHz mono waveforms into those two arrays. It does not resample. Gemma 3n projects audio through `Gemma3nProjectorModule.soft_embeddings`, without the scaling used for vision. Gemma 4 reuses `Gemma4ProjectorModule`, with its input width taken from `output_proj_dims`.
