# Training language models

This guide assumes the [first training run](../getting-started.md) and basic next-token prediction. You do not need a pretrained model to run the example. It uses a four-token synthetic vocabulary so you can inspect the entire input and output.

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
data = Dataset(train=lambda: itertools.repeat({"text": tokens}),
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

`LMObjective` reads token rows of shape `(B, S + 1)`. It feeds the first `S` tokens to the model and scores predictions against the next `S` tokens. Here each row has nine tokens, so `seq_len=8`. Token IDs must be integers inside the model's vocabulary.

The model's causal attention prevents a position from reading later tokens. Its hidden width is 16, with two attention heads and a feed-forward width of 32. The example uses float32 and XLA attention on CPU. These dimensions are for teaching, not model-quality or throughput comparisons.

The training loss should decrease as the decoder learns the repeating pattern. `result.tokens` contains the prompt followed by six token IDs. `Sampling(temperature=0.0)` chooses the highest-scoring token at each step; its behavior log-probability is zero. The key remains an explicit argument. Outputs may differ with library versions and initialization.

## Tokenize real text

For real text, choose a tokenizer and use its vocabulary consistently for data preparation, model construction, decoding, and checkpoint loading. `ByteTokenizer` represents UTF-8 bytes with a vocabulary of 256. A Hugging Face tokenizer uses the selected model's vocabulary and chat template and may download files on first use.

From a repository checkout, prepare your own corpus and create token files:

```bash
mkdir -p data
printf 'A small corpus for a tokenizer demonstration.\n' > data/corpus.txt
python tools/tokenize_text.py --input data/corpus.txt --out data/corpus-byte --tokenizer byte --val-fraction 0.1
```

This command writes `train.bin`, `val.bin`, and `meta.json`. The binary arrays store token IDs; metadata describes their dtype, counts, and tokenizer. This tiny corpus demonstrates preparation only. Use enough text to supply your requested windows, batches, and held-out split before training.

`TokenWindows(path, seq_len).load(batch=...)` reads fixed-width windows. `PackedTokens` combines documents and adds `text_segment_ids` and `text_positions`. The objective excludes padded targets and document-boundary transitions. Packing can change which attention implementation is usable because the model needs a segment mask.

## Loss, precision, and evaluation

The vocabulary loss runs in float32. `head_chunks` controls how many vocabulary slices it scores, with a default of four. Chunking can reduce peak memory but adds work and can be transformed by the backend compiler. The saved memory depends on vocabulary size, sequence length, batch size, and executable; it is not a universal fixed reduction.

`LMObjective` enables EMA by default. Use `state.params` for live variables and `state.averaged` when you deliberately want the moving-average copy. Evaluation reads averaged variables when the objective keeps them. `ema_decay=None` keeps no copy: `state.ema` is None, `state.averaged` raises, previews and evaluation read the live variables, and a checkpoint written with one configuration will not restore into the other.

To measure validation perplexity, provide a validation iterator and set `eval_every`, as shown in [evaluation and tracking](../guides/evaluation.md). With a tracker, `Samples` configures one generated preview per event, separate from complete-batch teacher-forced scoring.

Accumulation weights CE and MTP by the main supported-target mass, including target-role masks at each MTP depth. Sequence-router losses normalize by rows; global router losses pool selected-slot counts and score sums before their product. Routing bias stays fixed during the window and commits from its aggregate counts. A zero-CE window with active router auxiliary can still update. Combined-batch equivalence assumes identical stochastic realizations and the declared mutable-state semantics.

## Generate from a checkpoint

`load_pretrained` reads a local Hugging Face directory or a Hub identifier into a `Pretrained` bundle: the native Flax model, its explicit variables tree, the checkpoint's own processor or tokenizer, the source config and the generation defaults. Text decoders and the multimodal wrappers share this one entry point. `bundle.processor` turns prompts, images and waveforms into `ModelInputs` (tokens, row-aligned token fields and media conditioning), which `model.apply`, `LMObjective` batches and `bundle.generate` all accept; `bundle.generate` reads EOS and padding ids from the checkpoint's generation config. `Generation.tokens` has shape `(B, P + max_new_tokens)` and preserves the input token slots; `lengths` counts the valid continuation per row.

The real prompt plus continuation must fit `model.max_seq_len`; input padding consumes no capacity. `Sampling` carries temperature, top-k, top-p, min-p, EOS and the output padding id. `eos_id` accepts one id or a tuple of stop ids. EOS counts as a valid generated action. `Generation.lengths` counts response actions; `terminated` distinguishes EOS from the token budget. Likelihood arrays cover only the response: `behavior_log_probs` includes temperature/top-k/top-p/min-p, while `raw_log_probs` records the original policy. Ignore slots beyond each response length.

Generation prepares inputs on the host and runs prefill and decode in one compiled call. Each row has its own cache cursor. Invalid tokens consume no cache slots; paused recurrent rows retain their convolution history and recurrent memory. Different valid lengths at the same padded shape reuse the executable. Changing batch size, padded width, model or sampling settings can still compile a new executable. Media conditioning runs at prefill; subsequent steps use the native model's cache and logical positions.

On a mesh, rows split over the batch axes and parameters retain their placement. The decode executes a fixed number of steps. EOS disables cache writes and output recording for the finished row without skipping collectives. In a multi-process run, each process passes and receives its own rows. Processes validate inputs together and require matching input shapes and sampling settings. Invalid input on one rank raises on all ranks before device execution; this does not recover a failed device collective.

### Generate through a task

`TextGeneration` from `dew.inference` binds a decoder, one variables tree and the source's processor, so a call takes text or prepared inputs and returns the same `Generation` as `generate`. `bind` returns the task over other weights; a rollout binds a policy snapshot once and draws from it for the whole collection, with the actual and raw-policy likelihood of every action in the result.

```python
from dew.inference import TextGeneration
from dew.sampling import Sampling

policy = TextGeneration(model, variables, sampling=Sampling(temperature=0.8, top_k=40))
drawn = policy([[1, 2, 3], [4, 5, 6]], 32, key=jax.random.key(0))
later = policy.bind(state.params)
```

Serving stays outside Dew: export a checkpoint with `Pretrained.save` and serve it with vLLM or Ollama.

This example runs offline on the tiny Gemma 3 fixture that the wrapper tests use. A Hub name such as `"Qwen/Qwen3-0.6B"` works the same way with the `interop` extra and a download; a real checkpoint needs enough host and device memory for its weights and cache.

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
generated = bundle.generate(inputs, 3, key=jax.random.key(1), generation=Sampling(temperature=0))
width = inputs.tokens.shape[1]
for row, length in enumerate(generated.lengths):
    continuation = generated.tokens[row, width:width + int(length)]
    print(bundle.processor.decode(continuation[None])[0])
```

The rows have different image counts, so the processor left-pads the shorter one; `inputs.token_fields["attention_mask"]` is the sole validity source and the padded slots consume no cache. Text-only prompts skip the `images` argument, and Gemma 3n and Gemma 4 take `audio=[waveform, ...]`, one waveform per audio placeholder in reading order.

To continue training, hand the loaded variables to the objective and feed `{"text": inputs}` batches to the trainer; `bundle.save(directory, variables=state.params)` writes the trained weights back under the source tensor names with the processor, so the directory loads again here and in Transformers.

```python
import optax
from dew.data.dataset import Dataset
from dew.objectives.lm import LMObjective
from dew.training import Layout, MeshSpec, Trainer

objective = LMObjective(bundle.model, inputs.tokens.shape[1] - 1,
                        pretrained=bundle.variables, ema_decay=None, pad_id=0)
rows = 2 * jax.device_count()
batch = inputs.take_rows(jax.numpy.arange(rows) % 2)
data = Dataset(train=lambda: iter([{"text": batch}]), val=None, records=rows, batch=rows)
trainer = Trainer(objective, optax.sgd(1e-4), key=jax.random.key(3),
                  mesh=MeshSpec(), layout=Layout(min_shard=2**30))
state = trainer.fit(data, steps=1, log_every=1)
bundle.save("gemma3-tiny-step1", variables=state.params)
```

The base fixture is not an instruction-tuned chat assistant. Use the checkpoint's documented chat template when loading an instruction-tuned model. A translated configuration, tiny reference parity, and full-checkpoint execution are distinct checks. [Decoder family reference](../reference/model-families.md) lists the translation coverage and limitations.

## Diffusion language models and media inputs

LLaDA and Dream use bidirectional masked-token prediction. They require a mask token ID and a masked-diffusion objective; replacing an autoregressive loss without changing the attention and corruption process is not sufficient.

DiffusionGemma uses uniform-vocabulary corruption, a causal prompt encoder, a bidirectional canvas, and self-conditioning. Load the complete model through the same `load_pretrained` interface as other published models. Its `BlockProcess` generation policy refines full canvases, commits clean tokens to the shared text cache, and returns `CanvasGeneration`: response lengths including EOS, termination flags and per-row refinement counts, without autoregressive likelihood fields.

From a repository checkout, this CPU example loads the complete tiny reference checkpoint, tokenizes, generates, decodes, saves and reloads it. Its 64-token synthetic vocabulary tests the workflow, not language quality. The released model ID is `google/diffusiongemma-26B-A4B-it`; loading that ID downloads large weights and requires sufficient host/device memory.

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
print(task.decode(generated, inputs.tokens.shape[1]))
with TemporaryDirectory() as checkpoint:
    bundle.save(checkpoint)
    restored = load_pretrained(checkpoint, dtype="float32", attention_impl="xla", max_seq_len=32)
    replay = restored.block_generation()(inputs, 7, key=jax.random.key(11))
    np.testing.assert_array_equal(replay.tokens, generated.tokens)
```

The final canvas is refined at its full width, then the returned response is clipped to `max_new_tokens`. The prefix plus rounded-up canvas capacity must fit `max_seq_len`. Generation defaults come from `generation_config.json`; pass a `BlockProcess` as `process=` to override them. Media are prepared through the checkpoint processor and run only during prompt prefill, not once per refinement. The [training-contract note](../research/inference.md#diffusiongemma-training-contract-and-open-prerequisites) separates the available official fine-tuning recipe from the still-undisclosed original sampler-distillation/RL objective.

### Fine-tune with the official block loss

`BlockDiffusionObjective` ports Google's public post-release SFT adapter. It samples a valid response canvas, corrupts the entire response with uniform-vocabulary noise, performs detached-first-pass self-conditioning, and combines independently row-normalized canvas and encoder losses. The default time safety margin is 1e-4 and self-conditioning probability is 0.5. This is not the undisclosed original sampler-distillation/RL objective.

The following CPU example takes a real optimizer step on the tiny official-reference model. It deliberately uses a synthetic vocabulary and unequal target support. The objective prepares trainable layer scalars; the checkpoint's ordinary HF view keeps those same tensors frozen. Pass the objective's native model value when exporting the trained variables.

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
training_data = Dataset(train=lambda: iter([{"text": train_tokens}]), val=None,
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

The same objective is available in `recipes/lm/train.py`, not a second recipe. It uses complete token-window rows: `data.seq_len + 1` must equal `block_prompt_tokens` plus whole training canvases. `block_canvas_size` defaults to the checkpoint's canvas length. Packed documents are rejected because they have different context boundaries. Block SFT logs `canvas_ce` and `encoder_ce`; it does not report autoregressive perplexity or use the AR preview settings. Generate text through the shared pretrained interface when needed.

```bash
python recipes/lm/train.py data:token-windows --data.path data/diffusion-token-windows \
    --data.seq-len 511 --block-prompt-tokens 256 \
    --pretrained google/diffusiongemma-26B-A4B-it \
    --tokenizer google/diffusiongemma-26B-A4B-it --objective block_diffusion \
    --sample-tokens 0 --ema-decay None --optim.learning-rate 0.00015
```

Those token files must use the checkpoint tokenizer and arrange the intended clean prompt prefix and response canvases. The large-checkpoint command is not part of the CPU example and was not run here. Trainer checkpoints preserve optimizer and iterator state; `Pretrained.save` instead writes a complete source-format inference checkpoint.

A multimodal checkpoint loads as a `MultimodalTransformer`: the text decoder, the vision tower and projector, and for Gemma 3n and Gemma 4 the audio tower and its embedder, under the variable names `language_model`, `tower`, `projector`, `audio_tower` and `audio_projector`. The checkpoint's processor owns resizing, normalization, patching and placeholder expansion; Dew's `Processor` runs it and lays its outputs out row by row. `ModelInputs.tokens` is `[B, S]`; `token_fields` holds `attention_mask`, `positions`, `image_indices` and `image_groups` (the soft feature and the image behind each slot, -1 for text), `audio_indices` for audio slots and, for Qwen 3.5, the three-axis `rotary_positions`; `conditioning` holds the media, padded to the row with the most images or clips: `pixel_values` as `[B, images, ...]` in the processor's own layout, `image_position_ids` (Gemma 4) or `image_grid_thw` (Qwen 3.5) beside it, and `input_features` with `input_features_mask` as `[B, clips, frames, mel]`. The processor checks token ids, placeholder counts and media shapes on the host; the compiled model is pure. The same `ModelInputs` feeds `model.apply`, the objective and `generate`; media are evaluated at prefill and decode steps read the cache.

Per family, the processor emits what the reference expects. Gemma 3 gives one fixed-resolution image per placeholder block. Llama 4 tiles each image into local tiles and a global tile with separator tokens, normalizing pixels in bfloat16 as its original implementation does; the loader widens them to float32 exactly. Gemma 4 emits padded patch streams with 2D patch positions and expands video placeholders that the decoder maps to the pad embedding. Qwen 3.5 packs channel-then-time patches with a per-image grid, and the loader derives the spatial rotary coordinates the reference's `get_rope_index` computes. Gemma 3n uses the MobileNet-v5 encoder, embeds its hard vision and audio vocabulary ranges through the multimodal embedders, and keeps placeholder ids for its per-layer inputs while masking the hard ranges, on training and decode steps alike. Audio clips carry a mask that is True for valid frames; Gemma 4 inserts one placeholder per encoded frame, Gemma 3n a fixed `audio_soft_tokens_per_image` per clip with the embedder's padding token in the remaining slots.

`bundle.save` writes trained variables under the source tensor names, including Gemma 4's frozen standardization and clipping buffers, which live in the `constants` collection and stay bitwise through training. The [family reference](../reference/model-families.md) lists each wrapper's fixture and processor inputs.

Gemma 3n and Gemma 4 audio encoders are `dew.nn.audio.Gemma3nAudio` and `Gemma4Audio`, registered as towers `gemma3n_audio` and `gemma4_audio`. `audio_config` reads the checkpoint's `audio_config` record and rejects unknown computational fields; `audio_weights` converts the tower's own tensors, keeping Gemma 4's checkpointed clipping bounds in a frozen `constants` collection. An encoder takes `input_features` shaped `[B, T, F]` and a boolean `input_features_mask` that is True for valid frames, and returns `AudioEncoding(features, mask)` with the mask subsampled to the encoder's frame rate. `dew.data.audio.AudioProcessor` builds the checkpoint's feature extractor from its `preprocessor_config.json` record and converts 16 kHz mono waveforms into those two arrays without resampling. Gemma 3n projects audio through `Gemma3nProjectorModule.soft_embeddings`, without the vision-only scaling; Gemma 4 reuses `Gemma4ProjectorModule` with its input width taken from `output_proj_dims`.
