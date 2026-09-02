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

`dew.data.sources.text.TokenFileSource(path, seq_len)` is a random-access view over a memory-mapped `.bin`: record `i` is `tokens[i * seq_len : i * seq_len + seq_len + 1]`, so the records tile the file and the last token of one is the first of the next. `dew.data.dataloaders.get_token_dataset_grain` wraps that source in the same grain pipeline the image loaders use - an index sampler seeded from the run, `ShardByJaxProcess` so every host reads its own records, and worker processes for the reads - and yields batches of

```python
{"text": int32[B, seq_len + 1]}
```

One id longer than the context on purpose: the inputs are `text[:, :-1]` and the targets are `text[:, 1:]`, so a batch carries its own labels and nothing has to be shifted twice.

## The objective

`dew.objectives.lm.LMObjective(model, seq_len, vocab_size=..., pad_id=None, samples=None)` is the whole learning problem:

- `loss` runs the model on the inputs, casts the logits to float32 and returns the mean cross entropy of the targets. Float32 is deliberate: a bfloat16 logsumexp over a large vocabulary loses enough precision to move the loss and the gradient with it. The scalar comes with `ce`, `perplexity` and `token_accuracy`, which the trainer logs under `train/`.
- `pad_id` excludes padded targets from the mean. It defaults to `None` because packed token files have no padding, and masking an id that is really in the data would drop those tokens from the average.
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
