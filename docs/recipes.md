# Training recipes

A recipe is a command-line script in `recipes/` that builds a model, loads data, picks an objective and calls `Trainer.fit`. Every setting comes from a typed configuration, and the recipe saves that configuration next to the checkpoints as `run.json`. The repository has recipes for language models (`recipes/lm/train.py`), diffusion (`recipes/diffusion/train.py`) and JEPA (`recipes/jepa/train.py`). The `recipes/` and `tools/` scripts are in the repository checkout and are not part of the installed `dew` package. The [first training run](getting-started.md) puts the same pieces together in Python.

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
dew tokenize \
    --input corpus.txt --out tokens --tokenizer byte --val-fraction 0.1
```

```text
wrote 48276 tokens to tokens/train.bin and 5364 to tokens/val.bin
tokens/meta.json: {"tokenizer": "byte", "vocab_size": 256, "dtype": "uint8", "train_tokens": 48276, "val_tokens": 5364, "eos_id": null}
```

`tokens/meta.json` records the tokenizer, the vocabulary size, the storage dtype and the token counts, so keep the three files together. The validation split is the first 10% of the token stream, not a random sample. This corpus repeats itself, so it tests the workflow and tells you nothing about generalization.

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
    --model.emb-features 16 --model.num-layers 1 --model.num-heads 2 --model.mlp-features 32 \
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

The first update includes compilation. The run logs a loss at steps 1 and 2 and a validation perplexity at step 2. It writes a checkpoint to `runs/byte-demo/2` and the configuration to `runs/byte-demo/run.json`. The checkpoint holds the training state and the position in the data stream ([Checkpoints](guides/checkpoints.md)).

`--sample-tokens 0` turns off text sampling, so validation only scores tokens. In a longer run you can ask for sample continuations with `--sample-prompt "The number after" --sample-tokens 32`. Sampling uses the EMA weights, and the recipe makes the decoder's context long enough for the prompt plus the new tokens. Early in training, random bytes may decode to replacement characters.

## Configuration

Every recipe's configuration extends `RunConfig` with five parts:

| Part | Holds | CLI example |
| --- | --- | --- |
| `model` (`ModelConfig`) | The model's class and its constructor fields, compute dtype and attention kernel among them. | `--model causal_transformer --model.num-layers 12` |
| `objective` (`ObjectiveConfig`) | The objective's class and the constructor arguments the run states. | `--objective lm --objective.ema-decay 0.999` |
| `data` | A dataset specification and its loading settings. | `data:token-windows --data.seq-len 16` |
| `optim` (`OptimConfig`) | Optimizer, learning rate, schedule, weight decay, clipping, optimizer-state dtype. | `--optim.learning-rate 0.0001` |
| `trainer` (`TrainerConfig`) | Run length, batch size, checkpoint and logging intervals, mesh and layout, tracking. | `--trainer.steps 1000` |

A dotted flag sets a field inside a configuration object. A subcommand such as `data:token-windows` picks a dataset type by its alias and makes its flags available. `--model` picks the model's class by alias or import path (`--model hybrid_dit`, `--model mypackage.models:Net`), and every field that class declares is then a flag of its own, typed by its annotation: `--model.num-layers 12`, `--model.dtype float32`, `--model.precision highest`. A field holding records, such as a UNet's `attention_configs`, takes one JSON value. Naming another class starts from that class's defaults and keeps the recipe's compute dtype. `--objective` works the same way: it picks the objective's class, and each argument of its constructor is a flag, `--objective.<argument>`, except those it takes positionally without a default, which the recipe builds from the model and the data. Meshes and layouts are trainer fields ([Distributed training](concepts/distributed.md)), not model fields.

| Setting | Default | Notes |
|---|---|---|
| `--model.dtype` | `bfloat16` | The model's compute dtype, which each recipe's default model states. Parameters are stored in float32. A run-wide matmul precision is `jax.default_matmul_precision`; a model's own is `--model.precision`. |
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

An `epoch` interval needs a dataset with a known size. A stream that cannot save and restore its read position cannot produce a resumable checkpoint, so `fit` refuses a checkpoint interval for it and you have to set `--trainer.checkpoint-every None`. A configured checkpointer still writes the final state.

Muon treats matrix parameters separately from embedding, head and normalization parameters. MuonClip also needs attention QK statistics for its clipping step. The LM recipe turns them on (`qk_stats`) when the optimizer is `muonclip`, and a new objective has to emit them itself. MuonClip clips by rescaling the query and key kernels. A layer that normalizes its queries and keys divides that scale back out of the logits, so MuonClip refuses such a layer. The causal transformer normalizes them by default, so to train it with MuonClip, pass `--optim.optimizer muonclip --model.no-qk-norm`.

`--optim.state-dtype bfloat16` halves the memory the optimizer state takes. Its effect on step time depends on the device ([Performance measurements](performance.md)).

With `trainer.wandb` unset, the recipe prints to the terminal and keeps a local tracking journal in `<checkpoint-dir>/<name>/tracking`. To use Weights & Biases, select it with `trainer.wandb:wandb` and set `--trainer.wandb.project NAME`. `--trainer.wandb.offline` stops the tracker from opening an online session, but the recipe can still download datasets and models. Profiling is off unless `trainer.profile` is set. `run.json` records the configuration only, so to reproduce a run, archive the dataset contents, the package versions and the source checkout separately.

## Token data and pretrained models

To train on your own text, replace `corpus.txt` with a UTF-8 file or a directory of `.txt` files. Tokenize it into a new output directory with the tokenizer the run will train with. The recipe reads the vocabulary size from `meta.json`, so there is no flag for it. `data:token-windows` cuts fixed-width windows from the token stream.

For whole documents, tokenize with `--pack` and add `--data.pack`. `--pack` treats each input file as one document and writes the tokenizer's EOS ID after it. The byte tokenizer's EOS is ID 255, a byte that never occurs in UTF-8 text. The packed loader then resets attention boundaries and positions between documents.

`--pretrained` takes a supported Hugging Face model directory, a Hub ID, or `repo@revision` for a branch, tag or commit. A Hub ID can start a download, which may need authentication and a license agreement. `run.json` records a Hub source as `repo@commit`, with the commit it resolved to. To run offline, prepare a local checkpoint and its tokenizer, tokenize the data with that tokenizer, and pass matching `--tokenizer` and `--pretrained` values. The checkpoint decides the architecture, so of its fields `--model.max-seq-len` alone may be set. With any other field, the recipe refuses to load the checkpoint. [Language models](concepts/language_models.md) covers loading and export.

The LM recipe's `--objective` takes `lm`, `masked_diffusion` or `block_diffusion`, each with its own constructor's arguments as flags: `--objective.mtp-weight 0.3` for `LMObjective`, say. `masked_diffusion` trains a bidirectional masked-denoising model, with an EMA (`--objective.ema-decay`, 0.999) unless set to `None`. With `--pretrained` it continues from a LLaDA or Dream checkpoint. Without it, the model starts from a fresh initialization, and needs `--model.mask-token-id` as well as `--model.no-causal`. `block_diffusion` fine-tunes a DiffusionGemma checkpoint and needs both `--pretrained` and `data:token-windows`, and `--objective.prompt-length`, the clean prefix of each row. The recipe does not run SFT, DPO or GRPO; those run in Python ([Post-training](concepts/post_training.md)).

## Diffusion and JEPA recipes

```bash
python "$DEW_REPO/recipes/diffusion/train.py" --help
python "$DEW_REPO/recipes/jepa/train.py" --help
```

A diffusion configuration adds a training `preset`, a text condition and an optional autoencoder. You pick a preset with a subcommand such as `preset:edm`. The objective (`--objective`, `diffusion` by default) holds the EMA, the condition dropout and how validation samples: `--objective.ema-decay 0.9999`, `--objective.steps 40`, and records for the solver and the guidance, `--objective.solver '{"class": "heun"}'` and `--objective.guidance '{"class": "dew.sampling.guidance:CFG", "fields": {"scale": 4.0}}'`. Another objective trains a loss of its own in place of the denoising loss: `mean_flow`, `shortcut`, `flow_grpo`, or a distillation of a saved run, `rcm`, `ladd` or `guidance_distillation` with `--objective.teacher-run <run directory>`. Each refuses a preset or a guidance its loss cannot train or sample with.

By default the recipe trains on Oxford Flowers with a CLIP text encoder and scores validation with a CLIP metric. Prepare Oxford Flowers first and pass it as `--data.path` ([Installation](installation.md)). The CLIP text encoder and the CLIP metric download `openai/clip-vit-large-patch14` from Hugging Face unless it is cached. An offline tracker does not prepare any of these.

`--pretrained` fine-tunes a published diffusion pipeline in the diffusers layout. It takes a Hub ID, `repo@revision` or a local directory holding SD 1.x/2.x/XL, SD3, Flux, FLUX.2, Qwen-Image or Z-Image. The pipeline decides the model, its autoencoder and its text conditioning, which comes from all of its text encoders and reaches the model as `conditioning`. So the model flags then set only `--model.dtype` and `--model.attention-impl`, the parameters load in float32, and `text` and `autoencoder` stay unset. The pipeline runs at the data's resolution, and its images use the autoencoder's own channels; Qwen-Image 2.1's, for example, are RGBA.

`preset:none` trains with the convention the pipeline's scheduler reads, using the flow shift its sampler applies at that resolution. That shift is 3.0 for SD3, and for Flux it is exp(mu), with mu linear in the packed latent's token count. A scheduler file says how its checkpoint samples and says nothing about how it was trained, so this is a choice Dew makes for fine-tuning. A preset of the same kind replaces it, for example `preset:flow --preset.shift 1.0`. `run.json` records a Hub source as `repo@commit`.

To train one of these published architectures from scratch, name it with `--model` and set its fields with `--model.<field>` flags. To condition it on a pipeline's text encoders, pass `--text.encoder diffusion_text --text.checkpoint REPO`.

`--objective flow_grpo` trains the model with Flow-GRPO in place of the denoising loss, and it needs a flow preset. For each prompt it samples `--objective.groups` images through the flow SDE and scores them with the image metric named by `--objective.reward`. That metric has to give better images higher scores, as `clip_score` does.

A JEPA configuration adds predictor fields, target-mask settings and optional representation probes; the target encoder's EMA momentum is the objective's (`--objective.momentum 0.996 1.0`), ramped over the whole run unless `--objective.momentum-steps` says otherwise. The predictor estimates the encoded features of hidden image or video regions. The dataset and the model must both be image or both be video. Set the run length explicitly and prepare the dataset before starting.

## Python experiments

`dew train` trains a run written in Python, with no recipe. The file builds a `RunConfig`, `run`, or a function `run` returning one, and the objective it names is built around its model:

```python
from dew.config import ModelConfig, ObjectiveConfig, RunConfig, TrainerConfig
from dew.data import HFOptions, HubDataset
from dew.inputs import Field, InputSpec
from dew.objectives.supervised import Accuracy, CrossEntropy

run = RunConfig(
    model=ModelConfig.from_model(MLP(hidden=32)),
    data=HubDataset(name="json", options=HFOptions(data_files="rows.jsonl")),
    objective=ObjectiveConfig("supervised", {
        "loss": CrossEntropy(), "metrics": (Accuracy(),), "inputs": InputSpec(Field("x", (2,)))}),
    trainer=TrainerConfig(steps=300, batch_size=64))
```

```bash
dew train experiment.py --set trainer.steps=2000 --set model.hidden=64
```

Each `--set path=value` changes the field at a dotted path, read as the field's own flag reads it: `trainer.steps=2000` is an int, `trainer.eval_every=None` is None, `objective.momentum=0.9 0.99` is a tuple, and a record (`objective.loss={"function": "pkg.losses:hinge"}`) is JSON. Under `model` and `objective`, a path names a constructor argument of the class the run names. A path the run does not declare is refused. Sweeps override the same paths.

`dew train` imports the file as the module its name names, so the model, loss and metrics it defines are recorded by import path (`experiment:MLP`), and a lambda is refused when the run is saved. Reading a record constructs only what its field declares: a model is a Flax module, an objective an `Objective`, and a loss or metric a function, or a class whose instances are called from Dew, Flax, the experiment's own module or a package `--trust` names. `dew train <checkpoint-dir>/<name>/run.json --trust experiment` rebuilds the run from its record alone, with `experiment.py` importable (`PYTHONPATH`); it continues from the run's checkpoints, or trains it again elsewhere with `--set trainer.checkpoint_dir=...`. A recipe's run, whose configuration extends `RunConfig`, trains through its recipe. [`examples/train_supervised.py`](../examples/train_supervised.py) is a complete experiment.

To run a recipe from Python, call its `main(config)`. The configurations are frozen dataclasses that extend `RunConfig`. You can import `DiffusionRunConfig` from `dew.objectives.diffusion.config`, while `JepaRunConfig` and `LmRunConfig` are defined in the recipe files. A recipe's `main` sets up the process and builds the matching objective and data. The [diffusion](guides/diffusion.md) and [representation learning](guides/representation-learning.md) guides cover task-specific settings.

## Scaling up

Before a longer run, check that a small run reaches the number of updates you asked for, reports finite losses, reads the intended split and writes to the intended directory. A larger batch, sequence length or model width needs more memory. A two-step run on one machine does not test multi-host collectives, sustained input throughput, recovery after preemption or final model quality.

[Checkpoints](guides/checkpoints.md) describes what a resume restores. [Evaluation and tracking](guides/evaluation.md) and [Training data](concepts/data.md) describe what validation scores. For example, when validation shards have different lengths, the rows past the end of the shortest shard are not scored.
