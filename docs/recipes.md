# Run a training recipe

> An AI assistant maintains this document. It is presented as-is.

A recipe is a Python script that builds a model, loads data, picks an objective, and calls `Trainer.fit`. Use a recipe when you want to start a run from the command line and keep its configuration on disk. Use the [getting-started tutorial](getting-started.md) when you want to put those pieces together yourself in Python.

The repository has recipes for language models (LM), diffusion, and JEPA. Their command-line flags come from typed configuration objects, so all three follow the same structure. This page starts with a complete language-model run on your own machine. It makes its own tiny corpus and trains from random initialization, so it downloads no data and no model weights.

## Prepare a small corpus

Follow [installation](installation.md), activate your environment, and open a shell in the Dew checkout. The `recipes/` and `tools/` scripts live in the checkout. They are not part of the installed `dew` package. Save the checkout path, then move to an empty working directory:

```bash
export DEW_REPO="$PWD"
mkdir -p /tmp/dew-first-recipe
cd /tmp/dew-first-recipe
```

If that directory already holds a run you want to keep, pick another one. Run this Python block there to create the one input file:

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

The command writes `tokens/train.bin`, `tokens/val.bin`, and `tokens/meta.json`. The metadata records the tokenizer, the vocabulary size, the storage dtype, and the token counts, so keep the three files together. The validation split is the first 10% of the token stream, not a random sample. This corpus repeats itself, so it is good for checking that the workflow runs and useless for measuring generalization.

## Inspect the CLI, then train

`--help` prints the flags and trains nothing:

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

The first update includes JAX compilation, so it is slower than the ones after it. You should see a loss for step 1 and step 2, a validation line with the perplexity, and a checkpoint in `runs/byte-demo/2`. `runs/byte-demo/run.json` records the full configuration next to the checkpoint directories. The final checkpoint holds the training state and the data position. Read [checkpoints](guides/checkpoints.md) before you rely on resuming.

Two steps do not make a useful language model. This run only checks that training works. `--sample-tokens 0` turns off text sampling, so validation only scores tokens. Once you train for longer, you can ask for sample continuations with `--sample-prompt "The number after" --sample-tokens 32`. Sampling uses the EMA weights, and the recipe makes the decoder's context long enough for the prompt plus the new tokens. Early in training, random bytes may decode to replacement characters.

## Read a recipe configuration

Every recipe shares these four parts:

| Configuration | What you choose | CLI example |
| --- | --- | --- |
| `model` | Architecture, constructor fields, compute dtype, attention implementation. | `--model.architecture causal_transformer` |
| `data` | A dataset specification and its loading settings. | `data:token-windows --data.seq-len 16` |
| `optim` | Optimizer, learning rate, schedule, weight decay, clipping. | `--optim.learning-rate 0.0001` |
| `trainer` | Run length, batch size, checkpoint/logging intervals, device layout. | `--trainer.steps 1000` |

A dotted flag sets a field inside a configuration object. A subcommand such as `data:token-windows` picks a registered dataset type and makes its flags available. All architecture fields go into one JSON object passed to `--model.config`. The JSON keys keep their Python spelling, such as `num_layers`. Only use fields that the chosen architecture accepts.

Set either `--trainer.steps` or `--trainer.epochs`, not both. The batch size is global across all JAX processes. Loader workers and threads control how the host reads data; they do not change the batch size on the device. With zero workers, the loader starts no pool of worker processes, which is enough for this small run. Read [data](concepts/data.md) before you raise those settings.

`eval_every` and `checkpoint_every` take a step interval, `epoch`, or `None`. An `epoch` interval needs a dataset with a known size. A stream that cannot save and restore its read position cannot produce a checkpoint you can resume from, so its checkpoint interval must be `None`. A configured checkpointer can still write the final state in that case, so `None` does not mean that no checkpoint files appear. The [checkpoint guide](guides/checkpoints.md) explains the difference.

`ModelConfig` computes in bfloat16 by default and stores parameters in float32. The example above uses float32 and reference attention so that no device-specific kernel is involved in your first run. Device meshes and layout settings belong in the trainer configuration and are explained in [distributed training](concepts/distributed.md). They are not model JSON fields.

`--optim.optimizer` takes `adam`, `adamw`, `lamb`, `muon`, or `muonclip`. Keep the recipe's optimizer unless you have a reason to change it. Muon treats matrix parameters separately from embedding, head, and normalization parameters. MuonClip also needs attention QK statistics for its clipping step, and a new objective must emit those statistics itself. Choosing `muonclip` does not make an objective produce them.

With `trainer.wandb` unset, Dew prints to the terminal and keeps a local tracking journal in `runs/<name>/tracking`. To use Weights & Biases, select it with `trainer.wandb:wandb` and then set `--trainer.wandb.project NAME`. `--trainer.wandb.offline` stops the tracker from opening an online session. It does not stop the recipe from downloading datasets or models. Profiling is off unless you set `trainer.profile`. The saved `run.json` records the configuration. It does not record your dataset contents, your package versions, or your source checkout, so archive those yourself if you need to reproduce a run.

## Use your own token data or pretrained model

Replace `corpus.txt` with a UTF-8 file or a directory of `.txt` files, and tokenize it again into a new output directory. Use the same tokenizer for tokenization and for training. The recipe reads the vocabulary size from `meta.json`, so there is no flag for it. `data:token-windows` cuts fixed-width windows from the stream. For packing whole documents, tokenize with `--pack` and select `data:packed-tokens`. The packed loader resets attention boundaries and positions between documents. `--pack` treats each input file as one document and needs a tokenizer with an EOS id; the byte tokenizer has none.

`--pretrained` takes a supported Hugging Face model directory or Hub ID. A Hub ID can start a download and can need authentication and a license agreement. To run offline, prepare a local checkpoint and its tokenizer, tokenize your data with that tokenizer, and pass matching `--tokenizer` and `--pretrained` values. The checkpoint decides the architecture. In this mode `--model.config` may hold `max_seq_len` and nothing else; its default is `{}`, and any other field makes the recipe refuse to load the checkpoint. [Language models](concepts/language_models.md) covers the import and export requirements.

The LM recipe's `--objective` takes `lm`, `masked_diffusion`, or `block_diffusion`. `masked_diffusion` trains a bidirectional masked-denoising model, which has its own mask-token requirements. With `--pretrained` it continues from the weights of a LLaDA or Dream checkpoint. Without it, it starts from a fresh initialization and needs `mask_token_id` in `--model.config` next to `causal=False`. `block_diffusion` fine-tunes a DiffusionGemma checkpoint and needs both `--pretrained` and `data:token-windows`; [language models](concepts/language_models.md) walks through it. None of these three objectives is SFT, DPO, or GRPO. Those data layouts and their Python workflows are in [post-training](concepts/post_training.md).

## Diffusion and JEPA recipes

You can look at these entry points without creating data or loading weights:

```bash
python "$DEW_REPO/recipes/diffusion/train.py" --help
python "$DEW_REPO/recipes/jepa/train.py" --help
```

A diffusion configuration adds a training `preset`, a validation `sampler`, guidance, the number of sampling steps, a text condition, and an optional autoencoder. You pick registered choices with subcommands such as `preset:edm` and `sampler:heun`. Guidance is a configuration object, so you set its scale with `--guidance.scale`; there is no bare numeric `--guidance` flag. By default the recipe trains on Oxford Flowers with a CLIP text encoder and scores validation with a CLIP metric. Oxford Flowers must be prepared first and passed as `--data.path` (see [installation](installation.md#prepare-tfds-data-separately)). The CLIP text encoder and the CLIP metric download `openai/clip-vit-large-patch14` from Hugging Face unless you have it cached. For an offline run or a machine with limited resources, prepare the dataset, the text encoder, and any evaluation models first. An offline tracker does not prepare any of them.

A JEPA configuration adds predictor fields, target-mask settings, an EMA momentum schedule for the target encoder, and optional representation probes. The predictor estimates the encoded features of hidden image or video regions. The dataset and the model must both be image or both be video. Set the run length explicitly and prepare the dataset before you start.

Both recipes expose `main(config)` for use from Python. Their configurations are frozen dataclasses that extend `RunConfig`. Import `DiffusionRunConfig` from `dew.objectives.diffusion.config`. `JepaRunConfig` and `LmRunConfig` are defined in the recipe files in the checkout. A recipe's `main` sets up the process and builds the matching objective and data. Read the [diffusion](guides/diffusion.md) and [representation learning](guides/representation-learning.md) guides before you choose task-specific settings.

## Before increasing the run size

Check that your first small run reaches the number of updates you asked for, reports finite losses, reads the split you meant, and writes to the directory you meant. A bigger batch size, sequence length, or model width needs more memory. A two-step run on one machine does not test multi-host collectives, sustained input throughput, recovery after preemption, or final model quality. For longer runs, read the checkpoint, evaluation, and data guides for what resume restores and what validation scores. For example, if validation shards have different lengths, the rows after the shortest shard ends are not scored.
