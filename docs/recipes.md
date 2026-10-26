# Training recipes

A recipe is a command-line script in `recipes/` that parses a run's typed configuration and trains it: `LMRunConfig.cli().run()`. The configuration's class builds the model, loads the data and picks the objective, and the run saves the configuration next to the checkpoints as `run.json`, as its class and its fields. The repository has recipes for language models (`recipes/lm/train.py`), diffusion (`recipes/diffusion/train.py`) and JEPA (`recipes/jepa/train.py`). The `recipes/` and `tools/` scripts are in the repository checkout and are not part of the installed `dew` package. The [first training run](getting-started.md) puts the same pieces together in Python.

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
python - <<'PY'
from pathlib import Path
import numpy as np
from dew.data import ByteTokenizer
from dew.data.sinks import export
ids = np.asarray(ByteTokenizer().encode(Path("corpus.txt").read_text()), np.int32)
held = int(len(ids) * 0.1)
metadata = {"token_store": True, "tokenizer": "byte", "vocab_size": 256,
            "dtype": "int32", "eos_id": None}
export([{"text": ids[held:]}], "tokens/train", format="array_record", metadata=metadata, length_field="text")
export([{"text": ids[:held]}], "tokens/val", format="array_record", metadata=metadata, length_field="text")
PY
```

```text
The train and validation stores contain 48276 and 5364 tokens respectively.
```

Each store's producer manifest records the tokenizer, vocabulary, dtype, row counts and length index. Validation is the head 10% of the token stream. This repeated corpus exercises the workflow rather than generalization.

`--help` prints the flags and trains nothing:

```bash
python "$DEW_REPO/recipes/lm/train.py" --help
```

The recipe's data is a chain, which it records with the run. Its settings change with `--set`: `--set data.source.url=tokens/train --set data.token_windows.seq_len=16 --set data.token_windows.batch=8` trains on the store above. A run that also validates on a store of its own names both chains in a small run file, which `dew train` reads ([Python experiments](#python-experiments)):

<!-- not run: the file `dew train byte_demo.py` reads -->
```python
# byte_demo.py
from dew.data import Loading, presets
from dew.data.sources import TokenFiles
from dew.config.lm import LMRunConfig

loading = Loading(workers=0, threads=1, read_buffer=2)
windows = {"seq_len": 16, "batch": 8}
run = LMRunConfig(
    data=TokenFiles("tokens/train").shuffle(seed=0).repeat().pipe(presets.token_windows, **windows)
    .loading(loading),
    validation=TokenFiles("tokens/val").pipe(presets.token_windows, **windows, records=16).loading(loading),
)
```

Train two updates of a small decoder:

```bash
dew train byte_demo.py \
    --set model.dtype=float32 --set model.attention_impl=reference \
    --set model.emb_features=16 --set model.num_layers=1 --set model.num_heads=2 --set model.mlp_features=32 \
    --set trainer.steps=2 --set trainer.log_every=1 \
    --set trainer.eval_every=2 --set trainer.checkpoint_every=2 \
    --set trainer.checkpoint_dir=runs --set trainer.name=byte-demo \
    --set trainer.multi_host=False --set max_new_tokens=0
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

`max_new_tokens=0` turns off text sampling, so validation only scores tokens. In a longer run you can ask for sample continuations with `--set prompt="The number after" --set max_new_tokens=32`. Sampling uses the EMA weights, and the recipe makes the decoder's context long enough for the prompt plus the new tokens. Early in training, random bytes may decode to replacement characters.

## Configuration

Every recipe's configuration extends `RunConfig` with five parts:

| Part | Holds | CLI example |
| --- | --- | --- |
| `model` (`ModelConfig`) | The model's class and its constructor fields, compute dtype and attention kernel among them. | `--model causal_transformer --model.num-layers 12` |
| `objective` (`ObjectiveConfig`) | The objective's class and the constructor arguments the run states. | `--objective lm --objective.ema-decay 0.999` |
| `data`, `validation` | Chains: a source, its steps and a preset, recorded with the run. | `--set data.token_windows.seq_len=16` |
| `optim` (`OptimConfig`) | Optimizer, constant or scheduled rate, weight decay, clipping, optimizer-state dtype. | `--optim.learning-rate 0.0001` |
| `trainer` (`TrainerConfig`) | Run length, checkpoint and logging intervals, mesh and layout, tracking. | `--trainer.steps 1000` |

A dotted flag sets a field inside a configuration object, and `--set path=value`, applied after the flags, sets one inside the run's data chain: `data.<preset>.<setting>` for the preset's settings and `data.source.url` for its store. `--model` picks the model's class by alias or import path (`--model hybrid_dit`, `--model mypackage.models:Net`), and every field that class declares is then a flag of its own, typed by its annotation: `--model.num-layers 12`, `--model.dtype float32`, `--model.precision highest`. A field holding records, such as a UNet's `attention_configs`, takes one JSON value. Naming another class starts from that class's defaults and keeps the recipe's compute dtype. `--objective` works the same way: it picks the objective's class, and each argument of its constructor is a flag, `--objective.<argument>`, except th…

Each setting's default is the recipe's own, which can differ from the configuration class's: the language-model recipe states its own optimizer settings (`LMRunConfig`). A recipe's `--help` prints every flag with the default that recipe runs with, for example `python recipes/lm/train.py --help`.

| Setting | Notes |
|---|---|
| `--model.dtype` | The model's compute dtype, which each recipe's default model states. Parameters are stored in float32. A run-wide matmul precision is `jax.default_matmul_precision`; a model's own is `--model.precision`. |
| `--model.attention-impl` | A name `AttentionImpl` declares ([dew/nn/attention.py](../src/dew/nn/attention.py)). The example uses `reference` so no device-specific kernel runs. |
| `--optim.optimizer` | `adam`, `adamw`, `lamb`, `muon` or `muonclip`. |
| `--optim.learning-rate` | The learning rate: a number, or a schedule's record as JSON (`'{"class": "cosine", "fields": {...}}'`). |
| `--optim.state-dtype` | `bfloat16` stores Adam's moments stochastically rounded, for `adam` and `adamw` only. |
| `--trainer.steps` / `--trainer.epochs` | Run length; set one, not both. |
| `--set data.<preset>.batch` | Global batch across all JAX processes, the preset's (`data.token_windows.batch`, `data.images.batch`). |
| `--trainer.eval-every`, `--trainer.checkpoint-every` | A step interval, `epoch`, or `None`. |
| `--trainer.checkpoint-dir`, `--trainer.name` | The run's directory is `<checkpoint-dir>/<name>`. |
| `--trainer.multi-host` | Join the JAX process pool: `None` asks and continues alone when no cluster is configured, `True` requires the pool, `False` never asks. |
| `--trainer.xla-flags` | Extra `XLA_FLAGS`, applied before JAX opens a backend. |

A chain's `workers(n)`, `threads(n)` or `loading(Loading(...))` control how the host reads data; they do not change the batch size on the device. With zero workers the loader starts no worker processes. [Training data](concepts/data.md) covers the loading settings.

An `epoch` interval needs a dataset with a known size. A stream that cannot save and restore its read position cannot produce a resumable checkpoint, so `fit` refuses a checkpoint interval for it and you have to set `--trainer.checkpoint-every None`. A configured checkpointer still writes the final state.

Muon treats matrix parameters separately from embedding, head and normalization parameters. MuonClip also needs the step's attention logit maxima for its clipping step. A run whose optimizer holds a QK-Clip asks its objective for them (`Step.qk_stats`): the LM, RL, preference, masked-diffusion and block-diffusion objectives return them (`Aux.qk_stats`), and a run whose objective returns none is refused rather than left unclipped, as a decision run is for now. MuonClip clips by rescaling the query and key kernels. A layer that normalizes its queries and keys divides that scale back out of the logits, so MuonClip refuses such a layer. The causal transformer normalizes them by default, so to train it with MuonClip, pass `--optim.optimizer muonclip --model.no-qk-norm`.

Two optimizers from Nested Learning (arXiv 2512.24695) are Optax transforms in `dew.optim`, composed in a Python run's own `optax.chain` rather than named by `--optim.optimizer`. `scale_by_m3` is M3, its Algorithm 1: Muon's orthogonalized momentum over Adam's second moment, plus a slow momentum that gathers each chunk of `frequency` updates and reaches the next chunk's updates, weighted by `alpha`; follow it with `optax.scale_by_learning_rate`, and partition a model's vectors to another optimizer as Muon's do. `delta_gradient_descent` steps each kernel to the minimizer of its eq. 56, forgetting what the kernel held along the step's input `x`: its update takes `inputs=`, the input of each kernel, and suits a memory or fast weight that sees one input a step.

`--optim.state-dtype bfloat16` halves the memory the optimizer state takes. Its effect on step time depends on the device ([Performance measurements](performance.md)).

With `trainer.wandb` unset, the recipe prints to the terminal and keeps a local tracking journal in `<checkpoint-dir>/<name>/tracking`. To use Weights & Biases, select it with `trainer.wandb:wandb` and set `--trainer.wandb.project NAME`; the run shows in W&B under its own name, or under `--trainer.wandb.name` where that is set, while its checkpoints stay under the run's name. `--trainer.wandb.offline` stops the tracker from opening an online session, but the recipe can still download datasets and models. Profiling is off unless `trainer.profile` is set. `run.json` records the configuration only, so to reproduce a run, archive the dataset contents, the package versions and the source checkout separately.

## Token data and pretrained models

To train on your own text, export records with the tokenizer the run will use. The producer manifest carries the vocabulary, dtype and length index, so the model does not guess them. The recipe's chain reads the committed store and cuts fixed-width windows.

For whole documents, preserve their boundaries when exporting, or append EOS with the producer's `--eos` flag, and pack them in a run file, which packs in the store's order and orders the packed rows after: `TokenFiles(url).pipe(presets.lm, None, seq_len=256, batch=None, pack="first-fit").shuffle(0).repeat().batch(32)`. The byte tokenizer's EOS is ID 255, which never occurs in UTF-8 text. Packing resets attention boundaries and positions between documents.

`--pretrained` takes a supported Hugging Face model directory, a Hub ID, or `repo@revision` for a branch, tag or commit. A Hub ID can start a download, which may need authentication and a license agreement. `run.json` records a Hub source as `repo@commit`, with the commit it resolved to. To run offline, prepare a local checkpoint and its tokenizer, tokenize the data with that tokenizer, and pass matching `--tokenizer` and `--pretrained` values. The checkpoint decides the architecture, so of its fields `--model.max-seq-len` alone may be set. With any other field, the recipe refuses to load the checkpoint. [Language models](concepts/language_models.md) covers loading and export.

The LM recipe's `--objective` takes `lm`, `masked_diffusion` or `block_diffusion`, each with its own constructor's arguments as flags: `--objective.mtp-weight 0.3` for `LMObjective`, say, and `--objective.prefix 64` (or `prompt`, on chat rows) for a prefix run ([Prefix blocks](concepts/language_models.md#prefix-blocks)). `masked_diffusion` trains a bidirectional masked-denoising model, with an EMA (`--objective.ema-decay`, 0.999) unless set to `None`. With `--pretrained` it continues from a LLaDA or Dream checkpoint. Without it, the model starts from a fresh initialization, and needs `--model.mask-token-id` as well as `--model.no-causal`. `block_diffusion` fine-tunes a DiffusionGemma checkpoint and needs both `--pretrained` and a token-store chain, and `--objective.prompt-length`, the clean prefix of each row. The recipe does not run SFT, distillation, DPO, GRPO or PPO; those run in Python, alone or chained with pretraining by `recipes/chain.py` ([Stage chains](concepts/post_training.md#stage-chains)).

## Diffusion and JEPA recipes

```bash
python "$DEW_REPO/recipes/diffusion/train.py" --help
python "$DEW_REPO/recipes/jepa/train.py" --help
```

A diffusion configuration adds a training `preset`, a text condition and an optional autoencoder. You pick a preset with a subcommand such as `preset:edm`. The objective (`--objective`, `diffusion` by default) holds the EMA, the condition dropout and how validation samples: `--objective.ema-decay 0.9999`, `--objective.steps 40`, and records for the solver and the guidance, `--objective.solver '{"class": "heun"}'` and `--objective.guidance '{"class": "dew.sampling.guidance:CFG", "fields": {"scale": 4.0}}'`. Another objective trains a loss of its own in place of the denoising loss: `mean_flow`, `shortcut`, `flow_grpo`, or a distillation of a saved run, `rcm`, `ladd` or `guidance_distillation` with `--teacher-run <run directory>`. Each refuses a p…

By default the recipe trains on Oxford Flowers with a CLIP condition encoder. Prepare TFDS separately and point the chain's source at it, `--set data.source.path=...` (see Installation). Large corpora use the same `images` preset over `Files` URLs, including ArrayRecord and Parquet, in a run file. Choose the read mode explicitly; preparation and provisioning are not step-zero fallbacks.

For a committed store, `--set data.source.url=...` is its URL. A URL catalog adds a fetch step before decoding, written in the chain. The CLIP condition and metric need their pretrained checkpoint; an offline tracker does not prepare it.

`--pretrained` fine-tunes a published diffusion pipeline in the diffusers layout. It takes a Hub ID, `repo@revision` or a local directory holding SD 1.x/2.x/XL, SD3, Flux, FLUX.2, Qwen-Image or Z-Image. The pipeline decides the model, its autoencoder and its text conditioning, which comes from all of its text encoders and reaches the model as `conditioning`. So the model flags then set only `--model.dtype` and `--model.attention-impl`, the parameters load in float32, and `text` and `autoencoder` stay unset. The pipeline runs at the data's resolution, and its images use the autoencoder's own channels; Qwen-Image 2.1's, for example, are RGBA.

`preset:none` trains with the convention the pipeline's scheduler reads, using the flow shift its sampler applies at that resolution. That shift is 3.0 for SD3, and for Flux it is exp(mu), with mu linear in the packed latent's token count. A scheduler file says how its checkpoint samples and says nothing about how it was trained, so this is a choice Dew makes for fine-tuning. A preset of the same kind replaces it, for example `preset:flow --preset.shift 1.0`. `run.json` records a Hub source as `repo@commit`.

To train one of these published architectures from scratch, name it with `--model` and set its fields with `--model.<field>` flags. To condition it on a pipeline's text encoders, pass `--text.encoder diffusion_text --text.checkpoint REPO`.

`--objective flow_grpo` trains the model with Flow-GRPO in place of the denoising loss, and it needs a flow preset. For each prompt it samples `--rollout.groups` images through the flow SDE and scores them with the image metric named by `--rollout.reward`. That metric has to give better images higher scores, as `clip_score`, the default, does. `--rollout.steps` counts the rollout's time points and `--rollout.train-steps` the transitions it trains on.

A JEPA configuration adds predictor fields, target-mask settings and optional representation probes; the target encoder's EMA momentum is the objective's (`--objective.momentum 0.996 1.0`), ramped over the whole run unless `--objective.momentum-steps` says otherwise. The predictor estimates the encoded features of hidden image or video regions. The dataset and the model must both be image or both be video. Set the run length explicitly and prepare the dataset before starting.

## Python experiments

`dew train` trains a run written in Python, with no recipe. The file builds a `RunConfig`, `run`, or a function `run` returning one, and the objective it names is built around its model:

<!-- not run: the file `dew train experiment.py` imports; its MLP is recorded by that module's import path -->
```python
from flax import linen as nn

from dew.config import ModelConfig, ObjectiveConfig, RunConfig, TrainerConfig
from dew.data import Files
from dew.data.steps.generic import Formatting
from dew.inputs import Field, InputSpec
from dew.objectives.supervised import Accuracy, CrossEntropy


class MLP(nn.Module):
    hidden: int = 32

    @nn.compact
    def __call__(self, x):
        return nn.Dense(2)(nn.relu(nn.Dense(self.hidden)(x)))


run = RunConfig(
    model=ModelConfig.from_model(MLP(hidden=32)),
    data=Files("rows.jsonl").map(Formatting()).shuffle(seed=0).repeat().batch(64),
    objective=ObjectiveConfig("supervised", {
        "loss": CrossEntropy(), "metrics": (Accuracy(),), "inputs": InputSpec(Field("x", (2,)))}),
    trainer=TrainerConfig(steps=300))
```

```bash
dew train experiment.py --set trainer.steps=2000 --set model.hidden=64
```

Each `--set path=value` changes the field at a dotted path, read as the field's own flag reads it: `trainer.steps=2000` is an int, `trainer.eval_every=None` is None, `objective.momentum=0.9 0.99` is a tuple, and a record (`objective.loss={"function": "pkg.losses:hinge"}`) is JSON. Under `model` and `objective`, a path names a constructor argument of the class the run names. A path the run does not declare is refused. Sweeps override the same paths.

`dew train` imports the file as the module its name names, so the model, loss and metrics it defines are recorded by import path (`experiment:MLP`), and a lambda is refused where the run states it. Reading a record constructs only what its field declares: a model is a Flax module, an objective an `Objective`, and a loss or metric a function, or a class whose instances are called from Dew, Flax, the experiment's own module or a package `--trust` names. `dew train <checkpoint-dir>/<name>/run.json --trust experiment` rebuilds the run from its record alone, with `experiment.py` importable (`PYTHONPATH`); it continues from the run's checkpoints, or trains it again elsewhere with `--set trainer.checkpoint_dir=...`. [`examples/train_supervised.py`](../examples/train_supervised.py) is a complete experiment. `dew train` also trains a recipe's run from its `run.json`, which names the run's class.

A run is a frozen dataclass that extends `RunConfig`: `LMRunConfig` (`dew.config.lm`), `DiffusionRunConfig` (`dew.config.diffusion`), `JepaRunConfig` (`dew.config.jepa`) and `DecisionRunConfig` (`dew.decision.config`). `config.run()` sets up the process, builds what the run trains (`prepare`), and trains it under `--trainer.name`, or `<objective>-<dataset>/date-<time>` when that is unset; the name is recorded in `run.json`, so training the record again continues the same run. A kind of run adds its own settings and builds from them in `prepare`. The [diffusion](guides/diffusion.md) and [representation learning](guides/representation-learning.md) guides cover task-specific settings.

## Scaling up

Before a longer run, check that a small run reaches the number of updates you asked for, reports finite losses, reads the intended split and writes to the intended directory. A larger batch, sequence length or model width needs more memory. A two-step run on one machine does not test multi-host collectives, sustained input throughput, recovery after preemption or final model quality.

[Checkpoints](guides/checkpoints.md) describes what a resume restores. [Evaluation and tracking](guides/evaluation.md) and [Training data](concepts/data.md) describe what validation scores. Every real validation row is scored, however the processes' shards differ in length ([Reproducibility and limits](guides/evaluation.md#reproducibility-and-limits)).
