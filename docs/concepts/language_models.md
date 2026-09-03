# Language models

An autoregressive language model is another objective on the same trainer. The mesh, the sharding, the EMA, the checkpoints and the logging are the ones diffusion and JEPA runs use; what is different is the data (packed token ids instead of images), the loss (cross entropy against the next token) and the validation artifact (text the model wrote).

## From text to batches

`tools/tokenize_text.py` turns a file or a directory of files into three files:

```bash
python tools/tokenize_text.py --input data/shakespeare.txt \
    --out data/shakespeare-byte --tokenizer byte --val-fraction 0.01
```

- `train.bin` and `val.bin` are flat arrays of token ids in the smallest unsigned dtype that holds the vocabulary, `uint8` up to 256 ids, `uint16` up to 65536 and `uint32` beyond, with no separators or padding.
- `meta.json` records the `tokenizer`, its `vocab_size`, the `dtype` of the id arrays and the `train_tokens` / `val_tokens` counts. The recipe reads the vocabulary from here, so the model is always built for the ids on disk.

`--tokenizer` is either `byte`, which is `dew.data.text.ByteTokenizer` over raw UTF-8 bytes with a vocabulary of 256, or the name of a HuggingFace tokenizer, which `dew.data.text.HFTokenizer` loads through `transformers.AutoTokenizer`. Both expose `encode(str) -> list[int]` and `decode(ids) -> str`, and nothing downstream needs anything else from them.

`dew.data.sources.text.TokenFileSource(path, seq_len)` is a random-access view over a memory-mapped `.bin`: record `i` is `tokens[i * seq_len : i * seq_len + seq_len + 1]`, so the records tile the file and the last token of one is the first of the next. `dew.data.dataloaders.get_token_dataset_grain` wraps that source in the same grain pipeline the image loaders use (an index sampler seeded from the run, `ShardByJaxProcess` so every host reads its own records, and worker processes for the reads) and yields batches of

```python
{"text": int32[B, seq_len + 1]}
```

The record is one id longer than the context on purpose. The inputs are `text[:, :-1]` and the targets are `text[:, 1:]`, so a batch carries its own labels and nothing has to be shifted twice.

## Packed documents

A flat stream has no document boundaries, so a fixed window can straddle two unrelated texts: attention runs across the seam and RoPE never restarts. `--pack` closes every document (input file) with the tokenizer's eos id and records that id in `meta.json` as `eos_id`:

```bash
python tools/tokenize_text.py --input data/corpus --out data/corpus-byte --tokenizer byte --pack
```

`data.pack_sequences` then selects `dew.data.dataloaders.get_packed_token_dataset_grain`, which is grain's `Dataset` API rather than its `DataLoader`, because packing is the reason grain gives for switching. `dew.data.sources.text.TokenDocumentSource` reads a record as one document, documents longer than the window are cut into consecutive pieces first (grain's packer refuses an over-long element), and `grain.experimental.FirstFitPackIterDataset` fills windows of `seq_len + 1` with whole documents. A batch is then

```python
{"text": int32[B, seq_len + 1],
 "text_segment_ids": int32[B, seq_len + 1],   # which document, 0 for padding
 "text_positions": int32[B, seq_len + 1]}     # position inside that document
```

Three things read those two arrays. The backbone takes `positions` and `segment_ids` as call arguments: RoPE rotates by the position inside the document, and the attention mask is causal and block-diagonal, so no query reaches another document or the padding. The objective drops the one target a packed row must not train on, the last token of a document predicting the first of the next, along with padding. And the kernel changes: cuDNN has no mask argument, so jax converts a bool mask into an additive bias of `-2**41` in the compute dtype (`combine_bias_and_mask`) and refuses odd sequence lengths in training once a bias is present, while the xla kernel takes the mask itself on every backend with the same fp32 softmax. A segment-masked batch therefore runs on xla. Passing neither argument leaves every unpacked run bit-identical.

That kernel costs, one NVIDIA GeForce RTX 4080 (16 GiB, driver 595.84), 3 layers of GPT-2 small's width in bf16, 20 steps, one process per row:

| shape | batch | kernel | ms/step | peak GiB |
|---|---|---|---:|---:|
| 16 x 512 | fixed window | cuDNN | 75.8 | 4.99 |
| 16 x 512 | packed, 4 documents | xla | 83.6 | 5.80 |
| 16 x 512 | fixed window, xla pinned | xla | 83.5 | 5.80 |
| 4 x 2048 | fixed window | cuDNN | 78.5 | 5.00 |
| 4 x 2048 | packed, 4 documents | xla | 108.2 | 8.34 |

The third row is where the cost is. A fixed window on the xla kernel costs what a packed batch costs on it, so the mask is free and the whole difference is the fused kernel. The xla path materializes the fp32 `[B, N, T, S]` logits per layer and keeps them for the backward pass, which is why the gap grows with the sequence: 0.81 GiB at 512, 3.34 GiB at 2048. Keeping cuDNN for a packed batch means training against the bias it builds from the mask, so it needs a parity check against xla on the card in question before it can be the default.

```bash
python tools/benchmark_step.py --preset small --architectures causal_transformer --steps 20
python tools/benchmark_step.py --preset small --architectures causal_transformer --steps 20 --packed-documents 4
python tools/benchmark_step.py --preset small --architectures causal_transformer --steps 20 --attention-impl xla

LONG='{"architecture": "causal_transformer", "config": {"vocab_size": 50304,
  "emb_features": 768, "num_layers": 3, "num_heads": 12, "mlp_ratio": 4,
  "max_seq_len": 2048}, "batch_size": 4, "seq_len": 2048, "dtype": "bfloat16"}'
python tools/benchmark_step.py --steps 20 --cases "[$LONG]"
python tools/benchmark_step.py --steps 20 --cases "[$LONG]" --packed-documents 4
```

Sharding happens before packing (each process slices the documents), and the iterator's position saves and restores with the checkpoint like the fixed-window one. `train_len` counts window-sized chunks, which is the windows a pass yields at most: every window first-fit emits holds at least one chunk, and which chunks share a window depends on the shuffle, so the exact number is not known until the pass has run. A recipe divides it by the batch size for `steps_per_epoch`.

## The objective

`dew.objectives.lm.LMObjective(model, seq_len, vocab_size=..., pad_id=None, head_chunks=4, samples=None)` is the whole learning problem:

- `loss` reads the model's final hidden states, multiplies them by its head matrix a vocabulary slice at a time, and returns the mean cross entropy of the targets in float32. Float32 is deliberate: a bfloat16 logsumexp over a large vocabulary loses enough precision to move the loss and the gradient with it. The slicing is why the full `[tokens, vocab]` logits tensor is never built, which at the small benchmark preset is 1.57 GiB read four times over; `head_chunks` is how many slices, and four is the measured best on one RTX 4080 at vocabulary 50,304 ([research](../research/lm-head.md)). The scalar comes with `ce`, `perplexity` and `token_accuracy`, which the trainer logs under `train/`. The accuracy is the same top-1 the whole row would have given, tie for tie, taken as a running best across the slices.
- The model has to expose the seam the loss reads: `hidden_states(tokens, train=..., positions=..., segment_ids=...)` for the states before the head, `head_weight(params)` for the `[width, vocab]` matrix the forward multiplies them by, and `final_logit_softcap` and `precision` so the loss applies what the forward would have. `CausalTransformer` does; a model with a bias on its head does not fit, because then the projection is not a matmul.
- `pad_id` excludes padded targets from the mean. It defaults to `None` because a fixed-window token file has no padding, and masking an id that is really in the data would drop those tokens from the average. A packed batch needs no pad id: `text_segment_ids` says which slots are padding, and those targets are excluded whatever the id.
- `ema` averages the whole parameter tree at `--trainer.ema-decay`, and the EMA copy is what validation reads.
- `input_shapes` is `{"tokens": ((seq_len,), jnp.int32)}`. This is how a run needs no `DiffusionInputConfig`: `ObjectiveTrainer(..., objective=objective, input_config=None)` takes the init batch from the objective, and the `(shape, dtype)` pair is what keeps token ids from being initialised as floats. An objective that declares neither is rejected at construction.

## Generation

`dew.sampling.text.generate(model, params, prompt, max_new_tokens, *, rng, temperature=1.0, top_k=None)` prefills the KV cache on the prompt and then runs one `lax.scan` over the decode steps, returning `int32[B, P + max_new_tokens]`. The prompt is `int32[B, P]`, `temperature=0` is greedy, and `top_k` truncates the distribution before sampling. `params` is the variables dict the trainer holds, so the EMA copy can be sampled from directly.

The objective generates during validation when it is given a `samples` dict:

```python
LMObjective(model, 256, vocab_size=meta["vocab_size"], samples={
    "prompt": tokenizer.encode("To be, or not to be"),
    "max_new_tokens": 128,
    "temperature": 0.8,
    "decode": tokenizer.decode,
})
```

The prompt in `samples` is one sequence of ids or several of the same length, which the objective batches into the `[B, P]` that `generate` takes. The sampling key is folded with the step so each validation writes different text, and `decode` is what turns the ids back into the string that gets logged.

## Pretrained checkpoints

`dew.interop.hf_decoders` loads a Hugging Face decoder into the same `CausalTransformer` a from-scratch run trains, because the backbone's parameter names are the HF ones and its config surface covers what these families vary:

```python
from dew.interop import load_pretrained_decoder, save_pretrained_decoder

model, variables, config = load_pretrained_decoder("Qwen/Qwen3-0.6B")
logits = model.apply(variables, tokens)              # [B, S, 151936] fp32
save_pretrained_decoder(model, variables, "out/qwen3-tuned",
                        tokenizer_name="Qwen/Qwen3-0.6B")
```

- `load_pretrained_decoder(name_or_dir, *, dtype='bfloat16', attention_impl='auto', max_seq_len=None, revision=None)` takes a hub repo id or a local directory in the HF layout. It downloads only `*.safetensors` and `*.json`, reads the shards as float32 without torch (`safetensors.numpy` cannot read bfloat16, so those leaves are widened here), and builds the model through the same `apply_precision_policy` a recipe uses, so `dtype` is the compute dtype and the parameters stay float32. `max_seq_len` defaults to the config's context clamped to 8192, since the KV cache is allocated at that length whether decoding uses it or not. The third return is the dew config the model was built from, which is what a run logs.
- `translate_config(hf_config)` is the field map on its own, and `translate_weights(tensors, config)` the key map: `.weight` of a Linear becomes a transposed `.kernel`, a norm's `.weight` becomes `.scale`, `embed_tokens.weight` becomes `embed_tokens.embedding`, and a tied `lm_head.weight` is dropped after checking it really is the copy of the embedding it claims to be.
- `save_pretrained_decoder(model, variables, directory, *, tokenizer_name=None)` writes `config.json`, `model.safetensors` and `generation_config.json` back in HF vocabulary: `gemma3_text` when the sandwich norms are on, `qwen3` when the q/k norms are, `llama` otherwise. transformers loads the result, and `load_pretrained_decoder` on it returns bitwise-equal parameters.

What loads today is `llama`, `qwen3` and `gemma3_text`. Of Gemma 3 that is `gemma-3-1b-pt`: the 4B, 12B and 27B are multimodal `gemma3` checkpoints, refused both for the vision tower nothing here runs and for the linear RoPE factor of 8 in their `text_config`. `qwen2` is refused by name: it biases q, k and v and leaves `o_proj` bias-free, and the backbone has one `attention_bias` flag for all four projections, so its checkpoints cannot load unchanged. Gemma needs four things beyond a Qwen: `sandwich_norms` for the norms on each sublayer's output (HF calls them `post_attention_layernorm` and `post_feedforward_layernorm`, which is why the rename exists), `scale_offset` for its `(1 + w)` norm scales, `embedding_scale` for the `sqrt(hidden)` on the embeddings, and `attention_scale` for `query_pre_attn_scalar ** -0.5` in place of `head_dim ** -0.5`.

A config field that changes what the model computes and has no counterpart here is a `ValueError` naming the field, not a silent skip: `attn_logit_softcapping`, `use_bidirectional_attention`, a `mlp_bias`, an activation that is neither silu nor tanh-gelu, and any `rope_scaling` other than plain rope. Llama 3's `llama3` scaling is refused, and so is the `linear` factor of 8 that Gemma 3 carries in the `text_config` of its 4B, 12B and 27B; `gemma-3-1b-pt` names no scaling at all, which is why it loads. A scaling spelled the old way, `{"type": "yarn", ...}` instead of `{"rope_type": "yarn", ...}`, is read as the scaling it is, as transformers reads it. What does not load at all yet, for want of the layers rather than the names: Qwen3.5's GatedDeltaNet mixer, and Gemma 4's per-layer embeddings and cross-layer KV sharing.

Parity is the acceptance bar for a family, not a nice-to-have. `tools/hf_reference.py` writes the fixtures under torch and transformers, and `tests/test_hf_decoders.py` compares logits at float32 on CPU: 8.3e-06 max absolute difference on a random-weight Qwen3, 3.3e-06 on a random-weight Gemma3, 6.1e-06 on a random-weight Llama, and on the real Qwen3-0.6B weights the same argmax at all 48 prompt positions with 1.4e-04 max difference over the reference's top 32 logits per position. Gemma 3's own weights are gated on the Hub, so what is tested there is the translation of the real `gemma-3-1b-pt` config plus a random-weight fixture, not Google's weights.

`recipes/lm/train.py --pretrained` continues training one of these:

```bash
python tools/tokenize_text.py --input data/corpus.txt --out data/corpus-qwen3 \
    --tokenizer Qwen/Qwen3-0.6B
python recipes/lm/train.py --data.dataset data/corpus-qwen3 --pretrained Qwen/Qwen3-0.6B \
    --tokenizer Qwen/Qwen3-0.6B --sequence-length 512 --data.batch-size 4 \
    --optim.learning-rate 1e-5
```

The checkpoint decides every architecture field, so `--model.config` may carry `max_seq_len` and nothing else, and the token files have to come from the checkpoint's own tokenizer: a `meta.json` written with a different one stops the run instead of training the embedding table against ids that mean something else.

## Running it

```bash
python recipes/lm/train.py --data.dataset data/shakespeare-byte \
    --sequence-length 256 --data.batch-size 32 --trainer.epochs 10 \
    --model.config '{"emb_features": 384, "num_layers": 6, "num_heads": 6}' \
    --sample-prompt "To be, or not to be" --sample-tokens 200
```

`--data.dataset` is the token directory, not a dataset name. `--sequence-length` is the context the model trains on; it reaches the loader as the record length and the model as its `max_seq_len`, which is also the size of the decode cache, so a `--sample-tokens` budget that outruns the training context raises that limit to fit it. `--tokenizer` has to name the tokenizer `meta.json` was written with, otherwise the run stops rather than decoding samples with the wrong vocabulary. Everything else is the shared configuration: `--optim.*` for the solver, `--trainer.fsdp-size` and `--optim.grad-accum-steps` for scaling, `--trainer.wandb-project` to log anywhere at all.

## What a run reports

Every logging tick writes `train/loss` and, from the objective's auxiliary metrics, `train/ce`, `train/perplexity` and `train/token_accuracy`, alongside the trainer's throughput numbers.

At the end of each epoch the validation loop runs the objective's validation step over `--data.val-steps-per-epoch` batches. It reports the teacher-forced cross entropy, which `dew.eval.get_perplexity_metric()` turns into `val/perplexity` (lower is better, and tracked as `best_val/perplexity`), and the generated ids, which the objective decodes into a `val/samples` table. `--trainer.best-tracker-metric` defaults to `val/perplexity` for this recipe, and it decides whether the run is published rather than which step is: the trainer compares this run against the project's best five on that metric, and a run among them pushes its newest checkpoint to the registry, with the `best` alias when it leads.
