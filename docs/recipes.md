# Training recipes

A recipe is a command-line script in `recipes/` that builds a model, loads data, picks an objective and calls `Trainer.fit`, with every setting taken from a typed configuration and saved next to the checkpoints as `run.json`. The repository has recipes for language models (`recipes/lm/train.py`), diffusion (`recipes/diffusion/train.py`) and JEPA (`recipes/jepa/train.py`). The `recipes/` and `tools/` scripts live in the repository checkout; they are not part of the installed `dew` package. The [first training run](getting-started.md) assembles the same pieces in Python.

## Example

The example trains a small language model from random initialization on a corpus it writes itself, so it downloads no data and no weights. After [installation](installation.md), open a shell in the Dew checkout, save its path and move to an empty working directory:

```bash
export DEW_REPO="$PWD"
mkdir -p /tmp/dew-first-recipe
cd /tmp/dew-first-recipe
```

Write the corpus:

```python
from pathlib import Path

lines = [f"The number after {number} is {number + 1}.\n" for number in range(100)]
corpus = Path("corpus.txt")
corpus.write_text("".join(lines) * 20, encoding="utf-8")
print("Wrote", corpus, "with", corpus.stat().st_size, "bytes")
```

```text
Wrote corpus.txt with 53640 bytes
```

Tokenize it with the byte tokenizer, which maps each UTF-8 byte to one of 256 IDs:

```bash
python "$DEW_REPO/tools/tokenize_text.py" \
    --input corpus.txt --out tokens --tokenizer byte --val-fraction 0.1
```

```text
1 file(s), 0.1 MB of text
wrote tokens/train.bin (48276 tokens)
wrote tokens/val.bin (5364 tokens)
wrote tokens/meta.json: {"tokenizer": "byte", "vocab_size": 256, "dtype": "uint8", "train_tokens": 48276, "val_tokens": 5364, "eos_id": null}
```

`tokens/meta.json` records the tokenizer, the vocabulary size, the storage dtype and the token counts, so keep the three files together. The validation split is the first 10% of the token stream, not a random sample. This corpus repeats itself, so it checks the workflow and measures nothing about generalization.

`--help` prints the flags and trains nothing:

```bash
python "$DEW_REPO/recipes/lm/train.py" --help
python "$DEW_REPO/recipes/lm/train.py" data:token-windows --help
```

Train two updates of a small decoder:

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

```text
Number of devices: 1
Experiment_Name: byte-demo
Local tracking: /tmp/dew-first-recipe/runs/byte-demo/tracking
Training CausalTransformer from step 0 to 2: 6,720 parameters, on 1 × cpu, batch 8, float32
step 1/2  loss 5.738  ce 5.738  perplexity 310.4  token_accuracy 0.8%  step_time_ms 22.96  samples_per_sec 348.5  accepted 100.0%
step 2/2  loss 5.713  ce 5.713  perplexity 302.9  token_accuracy 0.0%  step_time_ms 6.611  samples_per_sec 1,210  accepted 100.0%
eval val at step 2: perplexity 314.7 (16 records in 0.10 s)
Trained 2 steps in 0:00:01: first step after 0.49 s, then 137.6 step/s
1.1% of the wall time in steps, final loss 5.713
```

The first update includes compilation. The run logs a loss at steps 1 and 2 and a validation perplexity at step 2, and writes a checkpoint to `runs/byte-demo/2` with `runs/byte-demo/run.json` beside it. The checkpoint holds the training state and the data position ([Checkpoints](guides/checkpoints.md)). `--sample-tokens 0` turns off text sampling, so validation only scores tokens. A longer run can ask for sample continuations with `--sample-prompt "The number after" --sample-tokens 32`; sampling uses the EMA weights, and the recipe makes the decoder's context long enough for the prompt plus the new tokens. Early in training, random bytes may decode to replacement characters.

## Configuration

Every recipe's configuration extends `RunConfig` with four parts:

| Part | Holds | CLI example |
| --- | --- | --- |
| `model` (`ModelConfig`) | Architecture, constructor fields, compute dtype, parameter dtype, attention implementation. | `--model.architecture causal_transformer` |
| `data` | A registered dataset specification and its loading settings. | `data:token-windows --data.seq-len 16` |
| `optim` (`OptimConfig`) | Optimizer, learning rate, schedule, weight decay, clipping, optimizer-state dtype. | `--optim.learning-rate 0.0001` |
| `trainer` (`TrainerConfig`) | Run length, batch size, checkpoint and logging intervals, mesh and layout, tracking. | `--trainer.steps 1000` |

A dotted flag sets a field inside a configuration object. A subcommand such as `data:token-windows` picks a registered dataset type and makes its flags available. All architecture fields go into one JSON object passed to `--model.config`, with their Python spelling (`num_layers`); use only fields the chosen architecture accepts. Meshes and layouts are trainer fields ([Distributed training](concepts/distributed.md)), not model fields.

| Setting | Default | Notes |
|---|---|---|
| `--model.dtype` | `bfloat16` | Compute dtype; parameters are stored in float32 unless `--model.param-dtype` says otherwise. |
| `--model.attention-impl` | `auto` | `auto`, `reference`, `xla`, `cudnn` or `tpu`. The example uses `reference` so no device-specific kernel runs. |
| `--optim.optimizer` | `adamw` | `adam`, `adamw`, `lamb`, `muon` or `muonclip`. |
| `--optim.learning-rate` | `2.7e-4` | The constant rate when no schedule is named. |
| `--optim.state-dtype` | `float32` | `bfloat16` stores Adam's moments stochastically rounded, for `adam` and `adamw` only. |
| `--trainer.steps` / `--trainer.epochs` | | Run length; set one, not both. |
| `--trainer.batch-size` | `32` | Global batch across all JAX processes. |
| `--trainer.eval-every`, `--trainer.checkpoint-every` | `epoch` | A step interval, `epoch`, or `None`. |
| `--trainer.checkpoint-dir`, `--trainer.name` | `./checkpoints`, none | The run's directory is `<checkpoint-dir>/<name>`. |
| `--trainer.multi-host` | `None` | Join the JAX process pool: `None` asks and continues alone when no cluster is configured, `True` requires the pool, `False` never asks. |
| `--trainer.xla-flags` | `None` | Extra `XLA_FLAGS`, applied before JAX opens a backend. |

Loader workers and threads control how the host reads data; they do not change the batch size on the device. With zero workers the loader starts no worker processes. [Training data](concepts/data.md) covers the loading settings.

An `epoch` interval needs a dataset with a known size. A stream that cannot save and restore its read position cannot produce a resumable checkpoint, so its checkpoint interval must be `None`; a configured checkpointer still writes the final state.

Muon treats matrix parameters separately from embedding, head and normalization parameters. MuonClip also needs attention QK statistics for its clipping step; the LM recipe turns them on (`qk_stats`) when the optimizer is `muonclip`, and a new objective must emit them itself. `--optim.state-dtype bfloat16` halves the optimizer state; its effect on step time depends on the device ([Performance measurements](performance.md)).

With `trainer.wandb` unset, the recipe prints to the terminal and keeps a local tracking journal in `<checkpoint-dir>/<name>/tracking`. To use Weights & Biases, select it with `trainer.wandb:wandb` and set `--trainer.wandb.project NAME`; `--trainer.wandb.offline` keeps the tracker from opening an online session but does not stop the recipe from downloading datasets or models. Profiling is off unless `trainer.profile` is set. `run.json` records the configuration, not the dataset contents, package versions or source checkout; archive those separately to reproduce a run.

## Token data and pretrained models

Replace `corpus.txt` with a UTF-8 file or a directory of `.txt` files, and tokenize it into a new output directory with the tokenizer the run will train with. The recipe reads the vocabulary size from `meta.json`, so there is no flag for it. `data:token-windows` cuts fixed-width windows from the stream. For whole documents, tokenize with `--pack` and select `data:packed-tokens`; the packed loader resets attention boundaries and positions between documents. `--pack` treats each input file as one document and writes the tokenizer's EOS ID after it; the byte tokenizer's EOS is ID 255, a byte that never occurs in UTF-8 text.

`--pretrained` takes a supported Hugging Face model directory, a Hub ID, or `repo@revision` for a branch, tag or commit. A Hub ID can start a download and can need authentication and a license agreement. `run.json` records a Hub source as `repo@commit`, the commit it resolved to. To run offline, prepare a local checkpoint and its tokenizer, tokenize the data with that tokenizer, and pass matching `--tokenizer` and `--pretrained` values. The checkpoint decides the architecture: `--model.config` may then hold `max_seq_len` and nothing else, and any other field makes the recipe refuse to load the checkpoint. [Language models](concepts/language_models.md) covers loading and export.

The LM recipe's `--objective` takes `lm`, `masked_diffusion` or `block_diffusion`. `masked_diffusion` trains a bidirectional masked-denoising model; with `--pretrained` it continues from a LLaDA or Dream checkpoint, and without it starts from a fresh initialization and needs `mask_token_id` in `--model.config` next to `causal=False`. `block_diffusion` fine-tunes a DiffusionGemma checkpoint and needs both `--pretrained` and `data:token-windows`. None of the three is SFT, DPO or GRPO; those run in Python ([Post-training](concepts/post_training.md)).

## Diffusion and JEPA recipes

```bash
python "$DEW_REPO/recipes/diffusion/train.py" --help
python "$DEW_REPO/recipes/jepa/train.py" --help
```

A diffusion configuration adds a training `preset`, a validation `sampler`, guidance, the number of sampling steps, a text condition and an optional autoencoder. Registered choices are picked with subcommands such as `preset:edm` and `sampler:heun`. Guidance is a configuration object, set with `--guidance.scale`; there is no bare numeric `--guidance` flag. By default the recipe trains on Oxford Flowers with a CLIP text encoder and scores validation with a CLIP metric. Oxford Flowers must be prepared first and passed as `--data.path` ([Installation](installation.md)). The CLIP text encoder and the CLIP metric download `openai/clip-vit-large-patch14` from Hugging Face unless it is cached. An offline tracker does not prepare any of these.

`--pretrained` fine-tunes a published diffusion pipeline in the diffusers layout: a Hub ID, `repo@revision` or a local directory holding SD 1.x/2.x/XL, SD3, Flux, FLUX.2, Qwen-Image or Z-Image. The pipeline decides the model, its text conditioning (all of its text encoders, passed to the model as `conditioning`) and its autoencoder, so `--model` then carries only `dtype`, `param_dtype` and `attention_impl`, and `text` and `autoencoder` stay unset. The pipeline runs at the data's resolution, and its images are in the autoencoder's own channels: Qwen-Image 2.1's are RGBA. `preset:none` trains on the convention the pipeline's scheduler reads, with the flow shift its sampler walks at that resolution (3.0 for SD3; for Flux, exp(mu) of its packed latent's token count). A scheduler file states how its checkpoint samples, not how it was trained, so this is Dew's fine-tuning choice; a preset of the same kind, such as `preset:flow --preset.shift 1.0`, replaces it. `run.json` records a Hub source as `repo@commit`. The same published architectures train from scratch by naming them in `--model.architecture` with their fields in `--model.config`, conditioned by a pipeline's text towers through `--text.encoder diffusion_text --text.checkpoint REPO`. `rl:flow-grpo` trains the model with Flow-GRPO instead of the denoising loss: it samples `--rl.groups` images per prompt through the flow SDE, scores them with the image metric `--rl.reward` names (higher must be better, as `clip_score`), and needs a flow preset.

A JEPA configuration adds predictor fields, target-mask settings, an EMA momentum schedule for the target encoder, and optional representation probes. The predictor estimates the encoded features of hidden image or video regions. The dataset and the model must both be image or both be video. Set the run length explicitly and prepare the dataset before starting.

Both recipes expose `main(config)` for use from Python; their configurations are frozen dataclasses that extend `RunConfig`. `DiffusionRunConfig` is importable from `dew.objectives.diffusion.config`; `JepaRunConfig` and `LmRunConfig` are defined in the recipe files. A recipe's `main` sets up the process and builds the matching objective and data. The [diffusion](guides/diffusion.md) and [representation learning](guides/representation-learning.md) guides cover task-specific settings.

## Scaling up

Before a longer run, check that a small run reaches the number of updates asked for, reports finite losses, reads the intended split and writes to the intended directory. A larger batch, sequence length or model width needs more memory. A two-step run on one machine does not test multi-host collectives, sustained input throughput, recovery after preemption or final model quality. [Checkpoints](guides/checkpoints.md), [Evaluation and tracking](guides/evaluation.md) and [Training data](concepts/data.md) describe what a resume restores and what validation scores; for example, when validation shards have different lengths, the rows after the shortest shard ends are not scored.
