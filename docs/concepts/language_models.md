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
from dew.sampling import generate

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
                  key=jax.random.key(1), temperature=0.0)
print("Generated token IDs:", np.asarray(result).tolist())
assert result.shape == (1, 8)
np.testing.assert_array_equal(np.asarray(result[:, :2]), np.asarray(prompt))
```

`LMObjective` reads token rows of shape `(B, S + 1)`. It feeds the first `S` tokens to the model and scores predictions against the next `S` tokens. Here each row has nine tokens, so `seq_len=8`. Token IDs must be integers inside the model's vocabulary.

The model's causal attention prevents a position from reading later tokens. Its hidden width is 16, with two attention heads and a feed-forward width of 32. The example uses float32 and XLA attention on CPU. These dimensions are for teaching, not model-quality or throughput comparisons.

The training loss should decrease as the decoder learns the repeating pattern. Generation returns the prompt followed by six token IDs. `temperature=0.0` chooses the highest-scoring token at each step; the key is still an explicit API argument. The output may differ with library versions and initialization details. Inspect the continuation as a sequence of IDs in this synthetic vocabulary.

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

`LMObjective` enables EMA by default. Use `state.params` for live variables and `state.averaged` when you deliberately want the moving-average copy. Evaluation reads averaged variables for the built-in objective. Account for that additional copy when estimating device memory.

To measure validation perplexity, provide a validation iterator and set `eval_every`, as shown in [evaluation and tracking](../guides/evaluation.md). `Samples` configures generated preview text. Current evaluation can regenerate the same fixed-prompt preview across batches. Count distinct generated samples when reporting evaluation results.

The current accumulation path does not preserve combined token-mean gradients when microbatches have unequal numbers of valid targets. See [capabilities and limitations](../reference/support.md) before using accumulation with packing, padding, or SFT masks.

## Generate from a checkpoint

`generate(model, variables, prompt, max_new_tokens, key=..., temperature=..., top_k=...)` accepts the complete Flax variables mapping, including its outer `params` key. Prompts have shape `(B, P)` and int32 IDs. The returned array has shape `(B, P + max_new_tokens)` and includes the prompt.

The prompt plus continuation must fit `model.max_seq_len`. Generation prefills a KV cache and decodes a fixed number of new tokens. Current public controls are temperature and top-k; this API is not a serving engine with streaming requests or continuous batching.

Checkpoint loading needs the `interop` extra and may download substantial files. The following complete checkpoint-to-text example is not part of the offline quickstart and has not been run during this documentation validation:

```python
# runs elsewhere: downloads Qwen weights and requires enough host/device memory
import jax
import jax.numpy as jnp
from dew.interop import load_pretrained_decoder
from dew.sampling import generate
from transformers import AutoTokenizer

name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(name)
model, variables, config = load_pretrained_decoder(name, max_seq_len=64)
encoded = tokenizer("A short prompt", return_tensors="np")
ids = jnp.asarray(encoded["input_ids"], dtype=jnp.int32)
generated = generate(model, variables, ids, max_new_tokens=24,
                     key=jax.random.key(1), temperature=0.8, top_k=40)
continuation = generated[0, ids.shape[1]:]
print(tokenizer.decode(continuation.tolist(), skip_special_tokens=True))
```

The displayed string contains only the continuation; the returned token array also contains the prompt. The base model is not an instruction-tuned chat assistant. Use the checkpoint's documented chat template when loading an instruction-tuned model.

Use the returned model configuration when continuing training. A translated configuration, tiny reference parity, and full-checkpoint execution are distinct checks. [Decoder family reference](../reference/model-families.md) lists the translation coverage and limitations.

## Diffusion language models and vision inputs

LLaDA and Dream use bidirectional masked-token prediction. They require a mask token ID and a masked-diffusion objective; replacing an autoregressive loss without changing the attention and corruption process is not sufficient.

Diffusion Gemma denoises blocks with a causal prompt encoder, a bidirectional canvas, and self-conditioning. Dew provides its block process, cache prefill, and denoiser functions. This is a different process from masked diffusion.

Multimodal wrapper translation separates decoder, vision tower, and projector variables. The projector produces soft image tokens, which enter the decoder at image positions. Supported fixed-resolution paths and remaining restrictions are listed in the [capability reference](../reference/support.md). Ordinary `generate` is not a general multimodal preprocessing pipeline.
