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

`generate(model, variables, prompt, max_new_tokens, key=..., sampling=Sampling(...), prompt_lengths=...)` accepts the complete Flax variables mapping, including its outer `params` key. Prompts have shape `(B, P)` and integer IDs. `prompt_lengths` contains each row's real suffix length; omit it for unpadded prompts. The `Generation` result has a `tokens` array of shape `(B, P + max_new_tokens)` with the original prompt preserved.

The real prompt plus continuation must fit `model.max_seq_len`. `Sampling` carries temperature, top-k, EOS and the output padding id. EOS counts as a valid generated action. `Generation.lengths` counts response actions; `terminated` distinguishes EOS from the token budget. Likelihood arrays cover only the response: `behavior_log_probs` includes temperature/top-k, while `raw_log_probs` records the original policy. Ignore slots beyond each response length.

Generation is a host operation. It trims left padding and runs the same compiled cached decoder for each exact prompt-length group. This keeps recurrent and latent caches free of padded input. Many distinct lengths create many compiled shapes and separate prefills. It is a correctness path with an execution cost, not continuous batching or a serving engine. Full-sequence RL rescoring left-aligns real tokens within one fixed shape.

On a mesh the rows of each group split over the batch axes and the parameters stay where the trainer placed them. The decode runs a fixed number of steps; a row that reached EOS keeps stepping on padding with its outputs masked, so the collectives never depend on sampled content. In a multi-process run every process passes its own rows and gets its own back. The groups and their sizes come from every process's prompt lengths, so cooperating processes issue identical collectives; a process with fewer rows in a group fills the shape with rows whose outputs are discarded.

Checkpoint loading needs the `interop` extra and may download substantial files. The following complete checkpoint-to-text example is not part of the offline quickstart and has not been run during this documentation validation:

```python
# runs elsewhere: downloads Qwen weights and requires enough host/device memory
import jax
import jax.numpy as jnp
from dew.interop import load_pretrained_decoder
from dew.sampling import Sampling, generate
from transformers import AutoTokenizer

name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(name)
model, variables, config = load_pretrained_decoder(name, max_seq_len=64)
encoded = tokenizer("A short prompt", return_tensors="np")
ids = jnp.asarray(encoded["input_ids"], dtype=jnp.int32)
generated = generate(model, variables, ids, max_new_tokens=24,
                     key=jax.random.key(1), sampling=Sampling(temperature=0.8, top_k=40,
                                                            eos_id=tokenizer.eos_token_id))
continuation = generated.tokens[0, ids.shape[1]:ids.shape[1] + int(generated.lengths[0])]
print(tokenizer.decode(continuation.tolist(), skip_special_tokens=True))
```

The displayed string contains only valid continuation tokens. `Generation.tokens` also contains the original prompt and padded response slots. The base model is not an instruction-tuned chat assistant. Use the checkpoint's documented chat template when loading an instruction-tuned model.

Use the returned model configuration when continuing training. A translated configuration, tiny reference parity, and full-checkpoint execution are distinct checks. [Decoder family reference](../reference/model-families.md) lists the translation coverage and limitations.

## Diffusion language models and vision inputs

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
generated = bundle.generate(inputs, 7, key=jax.random.key(11))
response = generated.tokens[:, inputs.tokens.shape[1]:]
print(bundle.processor.decode(response))
with TemporaryDirectory() as checkpoint:
    bundle.save(checkpoint)
    restored = load_pretrained(checkpoint, dtype="float32", attention_impl="xla", max_seq_len=32)
    replay = restored.generate(inputs, 7, key=jax.random.key(11))
    np.testing.assert_array_equal(replay.tokens, generated.tokens)
```

The final canvas is refined at its full width, then the returned response is clipped to `max_new_tokens`. The prefix plus rounded-up canvas capacity must fit `max_seq_len`. Generation defaults come from `generation_config.json`; pass a `BlockProcess` as `generation=` to override them. Media are prepared through the checkpoint processor and run only during prompt prefill, not once per refinement. The [training-contract note](../research/inference.md#diffusiongemma-training-contract-and-open-prerequisites) separates the available official fine-tuning recipe from the still-undisclosed original sampler-distillation/RL objective.

Multimodal wrapper translation separates decoder, vision tower, and projector variables. The projector produces soft image tokens, which enter the decoder at image positions. Supported paths and remaining restrictions are listed in the [capability reference](../reference/support.md). Ordinary `generate` is not a general multimodal preprocessing pipeline.

Gemma 3n uses the MobileNet-v5 encoder. It accepts floating processor output shaped `[B, C, H, W]` and emits `[B, R * R, 2048]` features; `R` is `msfa_output_resolution`, 16 by default. Its projector handles soft image features and hard IDs from the vision vocabulary. `model_inputs` receives the decoder's scaled token embeddings, original IDs, projected soft tokens, and precomputed integer image positions. Set `per_layer_input_vocab=decoder.per_layer_input_vocab`. The prepared dictionary supplies `tokens`, `input_embeddings`, and `embedding_positions` to `decoder.apply`.

`hard_embeddings` checks the interval `[vocab_offset, vocab_offset + vocab_size)`. `model_inputs` accepts only `[0, vocab_offset + vocab_size)`; it rejects negative and audio-vocabulary IDs before fusion. Both methods raise on invalid eager input. Compiled callers must use `jax.jit(checkify.checkify(...))` and call the returned `error.throw()` on the host before consuming the result or applying an update. Plain `jit` does not functionalize these checks. The private numerical path stays pure and does not clip or poison IDs.

This small projector demonstrates the checked boundary without a checkpoint:

```python
import jax
import jax.numpy as jnp
from jax.experimental import checkify
from dew.nn.vision import Gemma3nProjectorModule

projector = Gemma3nProjectorModule(vision_width=8, text_width=4,
                                  vocab_offset=16, vocab_size=3)
features = jnp.arange(8, dtype=jnp.float32).reshape(1, 1, 8)
variables = projector.init(jax.random.key(0), features)
soft_tokens = projector.apply(variables, features)
text_embeddings = jnp.ones((1, 4, 4), dtype=jnp.float32)
ids = jnp.array([[2, 16, 17, 18]], dtype=jnp.int32)
image_positions = jnp.array([[2]], dtype=jnp.int32)

def prepare(params, embeddings, tokens, soft, positions):
    return projector.apply(params, embeddings, tokens, soft, positions,
                           per_layer_input_vocab=16, method=projector.model_inputs)

checked_prepare = jax.jit(checkify.checkify(prepare))
error, inputs = checked_prepare(variables, text_embeddings, ids,
                                soft_tokens, image_positions)
error.throw()                         # On the host, before using inputs.
print(inputs["tokens"].tolist())      # [[2, 0, 0, 0]]

error, _ = checked_prepare(variables, text_embeddings, ids.at[0, 3].set(19),
                           soft_tokens, image_positions)
try:
    error.throw()
except ValueError:
    print("Rejected unsupported token ID")
```

After `error.throw()` succeeds, the prepared dictionary can enter the decoder. Valid vision IDs map to zero only for its smaller per-layer embedding table; the fused image and hard-vision embeddings retain their values.

Gemma 3n wrapper translation accepts image-only bundles whose audio config is absent or `None` and whose tensor files contain no audio component. Full released audio-bearing bundles remain unsupported. Translation checks the full vision record, its model type, and its `model_args` before tensor loading. Unknown computational fields, arbitrary timm backbones, classifier pooling, alternative norm layers, and feature-only timm wrappers are rejected.
