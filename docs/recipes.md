# Training recipes

A recipe is a command-line script in `recipes/`. It builds a model, loads data, selects an objective and calls `Trainer.fit`. Every setting comes from a typed configuration saved beside the checkpoints as `run.json`.

Use `recipes/lm/train.py` for language models, `recipes/diffusion/train.py` for diffusion and `recipes/jepa/train.py` for JEPA. The `recipes/` and `tools/` scripts are in the repository checkout, outside the installed `dew` package. The [first training run](getting-started.md) uses the same objects directly in Python.

## Example

Train a small language model from random initialization on a corpus you create below, without downloading data or weights. After [installation](installation.md), open a shell in the Dew checkout. Save its path and move to an empty working directory:

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
dew tokenize \
    --input corpus.txt --out tokens --tokenizer byte --val-fraction 0.1
```

```text
wrote 48276 tokens to tokens/train.bin and 5364 to tokens/val.bin
tokens/meta.json: {"tokenizer": "byte", "vocab_size": 256, "dtype": "uint8", "train_tokens": 48276, "val_tokens": 5364, "eos_id": null}
```

Keep the three token files together. `tokens/meta.json` records the tokenizer, vocabulary size, storage dtype and token counts. Validation uses the first 10% of the token stream without random sampling. This repeating corpus checks the workflow only; it does not measure generalization.

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

The first update includes compilation. The run logs loss at steps 1 and 2 and validation perplexity at step 2. It saves a checkpoint at `runs/byte-demo/2`, with configuration in `runs/byte-demo/run.json`. The checkpoint stores training state and data position; see [Checkpoints](guides/checkpoints.md).

With `--sample-tokens 0`, validation scores tokens without generating text. For sample continuations in a longer run, use `--sample-prompt "The number after" --sample-tokens 32`. Sampling uses EMA weights. The recipe sets enough decoder context for the prompt and new tokens. Early in training, random bytes may decode to replacement characters.

## Configuration

Every recipe's configuration extends `RunConfig` with four parts:

| Part | Holds | CLI example |
| --- | --- | --- |
| `model` (`ModelConfig`) | Architecture, constructor fields, compute dtype, parameter dtype, attention implementation. | `--model.architecture causal_transformer` |
| `data` | A registered dataset specification and its loading settings. | `data:token-windows --data.seq-len 16` |
| `optim` (`OptimConfig`) | Optimizer, learning rate, schedule, weight decay, clipping, optimizer-state dtype. | `--optim.learning-rate 0.0001` |
| `trainer` (`TrainerConfig`) | Run length, batch size, checkpoint and logging intervals, mesh and layout, tracking. | `--trainer.steps 1000` |

A dotted flag sets a field in a configuration object. A subcommand such as `data:token-windows` selects a registered dataset type and exposes its flags. Pass architecture fields in one JSON object through `--model.config`, using their Python spelling, such as `num_layers`. Only include fields the chosen architecture accepts. Meshes and layouts belong to the trainer configuration; see [Distributed training](concepts/distributed.md).

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

Loader workers and threads control host data reading. Device batch size is unchanged. With zero workers, the loader starts no worker processes. See [Training data](concepts/data.md) for loading settings.

An `epoch` interval requires a dataset with a known size. If a stream cannot save and restore its read position, set its checkpoint interval to `None`. It cannot produce a resumable checkpoint, but a configured checkpointer still saves the final state.

Muon treats matrix parameters separately from embedding, head and normalization parameters. MuonClip also needs attention QK statistics for clipping. The LM recipe enables `qk_stats` when the optimizer is `muonclip`. A new objective must emit those statistics itself.

MuonClip rescales query and key kernels. Query/key normalization would cancel that scaling in the logits, so MuonClip rejects layers with it enabled. The causal transformer normalizes queries and keys by default. To train it with MuonClip, use `--optim.optimizer muonclip --model.config '{..., "qk_norm": false}'`.

`--optim.state-dtype bfloat16` halves optimizer-state storage. Its effect on step time depends on the device; see [Performance measurements](performance.md).

With `trainer.wandb` unset, the recipe prints to the terminal and writes a local journal in `<checkpoint-dir>/<name>/tracking`. For Weights & Biases, select `trainer.wandb:wandb` and set `--trainer.wandb.project NAME`. `--trainer.wandb.offline` prevents an online tracking session. It does not prevent dataset or model downloads.

Profiling is off unless you set `trainer.profile`. `run.json` records configuration only. To reproduce a run, separately archive dataset contents, package versions and the source checkout.

## Token data and pretrained models

To use your own corpus, replace `corpus.txt` with a UTF-8 file or a directory of `.txt` files. Tokenize into a new output directory with the tokenizer used for training. The recipe reads vocabulary size from `meta.json`; there is no vocabulary-size flag. `data:token-windows` reads fixed-width windows from the stream.

For whole documents, tokenize with `--pack` and select `data:packed-tokens`. The packed loader resets attention boundaries and positions between documents. `--pack` treats each input file as one document and appends the tokenizer's EOS ID. The byte tokenizer uses ID 255 for EOS, a byte that never occurs in UTF-8 text.

`--pretrained` accepts a supported Hugging Face model directory, a Hub ID, or `repo@revision` for a branch, tag or commit. A Hub ID may trigger a download requiring authentication and a license agreement. `run.json` records the resolved source as `repo@commit`.

For offline use, prepare a local checkpoint and its tokenizer. Tokenize the data with that tokenizer, then pass matching `--tokenizer` and `--pretrained` values. The checkpoint determines the architecture. In this mode, `--model.config` accepts only `max_seq_len`; any other field makes the recipe reject loading. See [Language models](concepts/language_models.md) for loading and export.

The LM recipe's `--objective` accepts `lm`, `masked_diffusion` or `block_diffusion`. Use `masked_diffusion` for bidirectional masked denoising. With `--pretrained`, it continues from a LLaDA or Dream checkpoint. Without it, training starts from fresh initialization and requires `mask_token_id` and `causal=False` in `--model.config`.

Use `block_diffusion` to fine-tune a DiffusionGemma checkpoint. It requires both `--pretrained` and `data:token-windows`. SFT, DPO and GRPO run in Python, outside these three recipe modes; see [Post-training](concepts/post_training.md).

## Diffusion and JEPA recipes

```bash
python "$DEW_REPO/recipes/diffusion/train.py" --help
python "$DEW_REPO/recipes/jepa/train.py" --help
```

A diffusion configuration adds a training `preset`, validation `solver`, guidance, sampling steps, text condition and optional autoencoder. Select registered choices with subcommands such as `preset:edm` and `sampler:heun`. Set guidance through its configuration object with `--guidance.scale`. There is no bare numeric `--guidance` flag.

By default, the recipe trains on Oxford Flowers with a CLIP text encoder and scores validation with a CLIP metric. First prepare Oxford Flowers and pass its path as `--data.path`; see [Installation](installation.md). The encoder and metric download `openai/clip-vit-large-patch14` from Hugging Face if it is not cached. An offline tracker does not prepare the dataset or models.

To fine-tune a published diffusion pipeline, pass `--pretrained` a Hub ID, `repo@revision` or local directory in the diffusers layout. Supported pipelines include SD 1.x/2.x/XL, SD3, Flux, FLUX.2, Qwen-Image and Z-Image.

The pipeline supplies the model, text encoders and autoencoder. It passes all text-encoder outputs to the model as `conditioning`. In this mode, `--model` holds only `dtype`, `param_dtype` and `attention_impl`. Leave `text` and `autoencoder` unset. Training uses the data's resolution and the autoencoder's image channels; Qwen-Image 2.1, for example, uses RGBA.

With `preset:none`, training uses the convention read by the pipeline's scheduler and the flow shift its sampler uses at that resolution. The shift is 3.0 for SD3. For Flux, it is exp(mu) of the packed latent's token count. A scheduler file describes sampling, not the original training procedure, so this is Dew's fine-tuning choice. To replace it, use a preset of the same kind, such as `preset:flow --preset.shift 1.0`. `run.json` records Hub sources as `repo@commit`.

You can train these architectures from scratch by setting `--model.architecture` and their fields in `--model.config`. For conditioning from a pipeline's text towers, use `--text.encoder diffusion_text --text.checkpoint REPO`.

For Flow-GRPO, select `rl:flow-grpo` with a flow preset. It trains with a reward objective in place of denoising loss. The rollout samples `--rl.groups` images per prompt through the flow SDE, then scores them with the image metric named by `--rl.reward`. The metric must reward higher values, as `clip_score` does.

A JEPA configuration adds predictor fields, target-mask settings, an EMA momentum schedule for the target encoder, and optional representation probes. The predictor estimates the encoded features of hidden image or video regions. The dataset and the model must both be image or both be video. Set the run length explicitly and prepare the dataset before starting.

To use either recipe from Python, call `main(config)`. Their configurations are frozen dataclasses extending `RunConfig`. Import `DiffusionRunConfig` from `dew.objectives.diffusion.config`. `JepaRunConfig` and `LmRunConfig` are defined in the recipe files. `main` sets up the process and builds the objective and data. See the [diffusion](guides/diffusion.md) and [representation learning](guides/representation-learning.md) guides for task-specific settings.

## Scaling up

Before a longer run, check that a small run reaches the requested updates, reports finite losses, reads the intended split and writes to the intended directory. Increasing batch size, sequence length or model width needs more memory. A two-step run on one machine does not test multi-host collectives, sustained input throughput, preemption recovery or final model quality.

See [Checkpoints](guides/checkpoints.md), [Evaluation and tracking](guides/evaluation.md) and [Training data](concepts/data.md) for resume and validation behavior. For example, if validation shards have different lengths, scoring stops at the shortest shard's end.
