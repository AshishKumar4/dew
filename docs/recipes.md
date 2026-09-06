# Run a training recipe

A recipe is a Python entry point that builds a model, loads data, chooses an objective, and calls `Trainer.fit`. Use a recipe when you want a command-line run with a saved configuration. Use the [getting-started tutorial](getting-started.md) when you want to build those pieces yourself in Python.

The repository contains LM, diffusion, and JEPA recipes. Their command lines come from typed configuration objects, so the flags follow the same structure. This page starts with a complete local language-model run. It creates its own corpus and trains from random initialization; it does not fetch data or model weights.

## Prepare a small corpus

Follow [installation](installation.md), activate your environment, and open a shell in the Dew checkout. The `recipes/` and `tools/` scripts belong to the checkout, not to the installed `dew` Python package. Save the checkout path before switching to an empty working directory:

```bash
export DEW_REPO="$PWD"
mkdir -p /tmp/dew-first-recipe
cd /tmp/dew-first-recipe
```

Use a new directory if that path already contains a run you want to preserve. Run this Python block there to create the only input file:

```python
from pathlib import Path

lines = [
    f"The number after {number} is {number + 1}.\n"
    for number in range(100)
]
corpus = Path("corpus.txt")
corpus.write_text("".join(lines) * 20, encoding="utf-8")
print("Wrote", corpus, "with", corpus.stat().st_size, "bytes")
```

Tokenize it with the byte tokenizer, which maps each UTF-8 byte to one of 256 IDs:

```bash
python "$DEW_REPO/tools/tokenize_text.py" \
    --input corpus.txt --out tokens --tokenizer byte --val-fraction 0.1
```

The command writes `tokens/train.bin`, `tokens/val.bin`, and `tokens/meta.json`. The metadata records the tokenizer, vocabulary size, storage dtype, and token counts. Keep those files together. The validation split is the beginning of the token stream, not a random sample. This repeated toy corpus is suitable for checking the workflow, but not for measuring generalization.

## Inspect the CLI, then train

Help performs no training:

```bash
python "$DEW_REPO/recipes/lm/train.py" --help
python "$DEW_REPO/recipes/lm/train.py" data:token-windows --help
```

Now run two updates on a small decoder:

```bash
python "$DEW_REPO/recipes/lm/train.py" data:token-windows \
    --data.path tokens --data.seq-len 16 \
    --data.loading.workers 0 --data.loading.threads 1 \
    --data.loading.read-buffer 2 --data.val-batches 2 \
    --model.dtype float32 --model.attention-impl reference \
    --model.config '{"emb_features": 16, "num_layers": 1, "num_heads": 2, "mlp_features": 32}' \
    --trainer.batch-size 8 --trainer.steps 2 --trainer.log-every 1 \
    --trainer.eval-every 2 --trainer.checkpoint-every 2 \
    --trainer.checkpoint-dir runs --trainer.name byte-demo \
    --trainer.multi-host False --sample-tokens 0
```

The first update includes JAX compilation, so it takes longer than later updates. You should see losses for steps 1 and 2, a validation report, and checkpoint output. `runs/byte-demo/run.json` records the resolved configuration next to the checkpoint directories. The final checkpoint holds the training state and data position. See [checkpoints](guides/checkpoints.md) before depending on resume behavior.

This is a training smoke run, not a useful language model. `--sample-tokens 0` disables text sampling, so validation only scores tokens. After increasing the training duration, you can request sample continuations with `--sample-prompt "The number after" --sample-tokens 32`. Sampling uses the EMA model; the recipe expands the decoder's context to accommodate the prompt plus those new tokens. Random byte sequences may decode to replacement characters early in training.

## Read a recipe configuration

The four shared parts are:

| Configuration | What you choose | CLI example |
| --- | --- | --- |
| `model` | Architecture, constructor fields, compute dtype, attention implementation. | `--model.architecture causal_transformer` |
| `data` | A dataset specification and its loading settings. | `data:token-windows --data.seq-len 16` |
| `optim` | Optimizer, learning rate, schedule, weight decay, clipping. | `--optim.learning-rate 0.0001` |
| `trainer` | Run length, batch size, checkpoint/logging intervals, device layout. | `--trainer.steps 1000` |

A dotted flag selects a field inside a configuration object. A subcommand such as `data:token-windows` selects a registered dataset type and makes that type's flags available. Architecture fields travel in one JSON object through `--model.config`; JSON keys keep their Python spellings, such as `num_layers`. Use the architecture's documented fields rather than guessing from a different model family.

`--trainer.steps` and `--trainer.epochs` are alternatives; set one, not both. The batch size is global across JAX processes. Loader workers and threads control host-side reading, not the device batch size. Zero workers avoids launching a worker process pool for this small run. See [data](concepts/data.md) before increasing those settings.

`eval_every` and `checkpoint_every` accept a step interval, `epoch`, or `None`. Epoch intervals require a known dataset size. A stream without a restorable read position cannot produce a resumable checkpoint; its checkpoint interval must be `None`. A configured checkpointer can still write the final state, so `None` should not be read as a promise that no files will exist. The [checkpoint guide](guides/checkpoints.md) describes the distinction.

`ModelConfig` defaults to bfloat16 computation while parameters remain float32. The offline example uses float32 and reference attention to keep device-specific kernel choices out of the first run. Device meshes and layout settings belong to [distributed training](concepts/distributed.md), not to the model's JSON fields.

The shared optimizer choices are `adam`, `adamw`, `lamb`, `muon`, and `muonclip`. Start with the recipe's optimizer unless you have a reason to change it. Muon partitions matrix updates from embedding, head, and normalization updates; MuonClip additionally needs attention QK statistics for its clipping operation. Choosing its name alone does not establish that a new objective emits those statistics.

With `trainer.wandb` unset, Dew logs to the terminal. Setting `--trainer.wandb.project` enables Weights & Biases; its offline setting prevents an online tracker session, but does not prevent dataset or model downloads elsewhere in a recipe. Profiling is opt-in through `trainer.profile`. The saved `run.json` records configuration, not your dataset contents, package environment, or source checkout. Archive those separately when you need to reproduce a run.

## Use your own token data or pretrained model

Replace `corpus.txt` with a UTF-8 file or a directory of `.txt` files, and rerun the tokenizer into a new output directory. Keep the tokenizer identity consistent between tokenization and training. The recipe reads the vocabulary from `meta.json`; there is no separate vocabulary-size flag. `data:token-windows` draws fixed-width windows. For document-aware packing, tokenize with `--pack`, then select `data:packed-tokens`; the packer resets attention boundaries and positions between documents.

`--pretrained` accepts a supported Hugging Face model directory or Hub ID. A Hub ID can trigger a download, authentication, and license requirements. For an offline run, prepare a local checkpoint and its tokenizer files before starting, then tokenize the corpus with that same tokenizer and pass matching `--tokenizer` and `--pretrained` values. The checkpoint selects the architecture; in that mode `--model.config` may only override `max_seq_len`. Consult [language models](concepts/language_models.md) for family-specific import and export boundaries rather than assuming any Hugging Face directory will work.

The LM recipe's `--objective` accepts `lm` or `masked_diffusion`. The latter trains a bidirectional masked-denoising model with its own mask-token requirements. It is not a flag for SFT, DPO, or GRPO; those data layouts and their Python workflows are in [post-training](concepts/post_training.md).

## Diffusion and JEPA recipes

Inspect these entry points without creating data or loading weights:

```bash
python "$DEW_REPO/recipes/diffusion/train.py" --help
python "$DEW_REPO/recipes/jepa/train.py" --help
```

A diffusion configuration adds a training `preset`, validation `sampler`, guidance, sampling steps, a text condition, and an optional autoencoder. Registry choices use subcommands such as `preset:edm` and `sampler:heun`. Guidance is a structured configuration, so its scale is `--guidance.scale`, not a bare numeric `--guidance` argument. The default configuration uses Oxford Flowers, a CLIP text encoder, and a CLIP validation metric. Running it can download both data and weights. Prepare the dataset, text encoder, and any evaluation models before a resource-constrained or offline run; selecting an offline tracker does not prepare them.

A JEPA configuration adds predictor fields, target-mask settings, an EMA momentum schedule for the target encoder, and optional representation probes. The encoder predicts representations of hidden image or video regions, rather than generating those pixels. Image and video dataset/model choices must agree. Set an explicit run length and prepare the chosen dataset before launching it.

Both recipes expose `main(config)` for use in a Python application. Their frozen dataclass configurations extend `RunConfig` with the fields above. Import `DiffusionRunConfig` from `dew.objectives.diffusion.config`; `JepaRunConfig` and `LmRunConfig` live in their respective checkout recipe modules. Call the recipe's `main`, which performs process setup and builds the objective and data, rather than calling `config.train` with unrelated objects. See the guides for [diffusion](guides/diffusion.md) and [representation learning](guides/representation-learning.md) before choosing their task-specific settings.

## Before increasing the run size

Check that the first small run reaches the intended number of updates, reports finite losses, reads the intended split, and writes to the intended directory. Increasing batch size, sequence length, or model width changes memory requirements. A two-step local result does not verify multi-host collectives, sustained input throughput, recovery after preemption, or final model quality. Known overflow/resume, unequal-mask accumulation, repeated-evaluation RNG, and prefetch-lifetime defects remain relevant to longer runs; use the checkpoint, evaluation, and data guides when planning around them.
