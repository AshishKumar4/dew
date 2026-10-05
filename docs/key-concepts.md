# Key concepts

A Dew training run uses four objects:

| Object | What it does | When to write your own |
|---|---|---|
| Model, a `flax.linen.Module` | The forward computation and the shapes of its variables | You need a new architecture. The built-in ones are classes in `dew.nn.backbones`. |
| `Objective` | Initializing the variables, the loss, evaluation outputs, and which weights keep a moving average | You need a loss Dew does not have. |
| `Dataset` | The iterators that yield training and validation batches | Your data does not fit a built-in reader. |
| `Trainer` | The device mesh, the compiled step, the optimizer update, the moving average, checkpoints and logging | Configure it; do not subclass it. |

`Trainer.fit` returns a `TrainState` with the variables, optimizer state, root random key, step counters and moving-average copy. To continue a run, you also need the position in the data stream and the configuration used to build the objective and trainer. Checkpoints store the data position beside the state. A recipe saves the configuration in `run.json`.

## Example

Train a small decoder on one repeated sentence, then generate from it. This example downloads nothing and runs on a CPU.

```python
import jax
import numpy as np
import optax

from dew import Dataset, Trainer
from dew.data import ByteTokenizer
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling, generate

tokenizer = ByteTokenizer()
text = tokenizer.encode("dew trains jax models. " * 3)
rows = np.tile(np.asarray(text[:65], np.int32), (8, 1))
data = Dataset.from_records({"text": rows}, batch=8)

model = CausalTransformer(
    vocab_size=tokenizer.vocab_size,
    emb_features=64, num_layers=2, num_heads=4,
    mlp_features=256, max_seq_len=128)
objective = LMObjective(model, seq_len=64)
trainer = Trainer(objective, optax.adamw(3e-3),
                  key=jax.random.key(0))
state = trainer.fit(data, steps=100, log_every=25)

prompt = [tokenizer.encode("dew")]
out = generate(model, state.variables, prompt, max_new_tokens=40,
               key=jax.random.key(1),
               sampling=Sampling(temperature=0))
print(tokenizer.decode(out.tokens[0]))
```

Output on two cores of a shared workstation CPU:

```text
Training CausalTransformer from step 0 to 100: 147,840 parameters, on 1 × cpu, batch 8, float32
step  25/100  loss 0.03061  ce 0.03061  perplexity 1.031  token_accuracy 100.0%  step_time_ms 23.68  samples_per_sec 337.9  accepted 100.0%  0:00:01 left
step  50/100  loss 0.01018  ce 0.01018  perplexity 1.010  token_accuracy 100.0%  step_time_ms 24.52  samples_per_sec 326.2  accepted 100.0%  0:00:01 left
step  75/100  loss 0.006710  ce 0.006710  perplexity 1.007  token_accuracy 100.0%  step_time_ms 24.68  samples_per_sec 324.1  accepted 100.0%  0:00:01 left
step 100/100  loss 0.005113  ce 0.005113  perplexity 1.005  token_accuracy 100.0%  step_time_ms 22.84  samples_per_sec 350.2  accepted 100.0%
Trained 100 steps in 0:00:04: first step after 2.06 s, then 42.6 step/s
53.0% of the wall time in steps, final loss 0.005113
dew trains jax models. dew trains jax model
```

I recorded this output with stdout piped to a file. In a terminal, `fit` draws a live panel with the same numbers, a progress bar and a sparkline per metric. Each `step` line shows that step's loss and objective metrics. For `LMObjective`, these are cross entropy, perplexity and token accuracy. Step time and throughput cover the interval since the previous line.

Each row holds 65 byte IDs. `LMObjective(seq_len=64)` passes the first 64 to the model and scores predictions of the following 64, shifted by one position. `generate` returns the prompt followed by the new tokens. With `temperature=0`, it picks the most likely token at every step.

## Model

A model is a `flax.linen.Module` with `init` and `apply`. Its variables are a nested dictionary of arrays. Its computation is independent of training. Build one from its class with `from dew.nn.backbones import CausalTransformer`. Each class also has a registered name, such as `causal_transformer`. Recipes and saved runs use that name to rebuild the model from a configuration file.

Your own model registers with one line. Its constructor fields become the saved configuration:

```python
import flax.linen as nn

from dew.registry import models


@models("residual_mlp")
class ResidualMLP(nn.Module):
    """A denoiser `DiffusionObjective` can train: it maps a noisy sample, the
    noise level's embedding and optional text context to its prediction."""

    features: int = 64

    @nn.compact
    def __call__(self, x, temb, textcontext=None, train=False):
        return x + nn.Dense(x.shape[-1])(nn.gelu(nn.Dense(self.features)(x)))
```

Every checkpoint of this model records `"architecture": "residual_mlp"` and `"config": {"features": 64}`. `TextToImage.from_run` and `dew.pipeline` use that record to rebuild the model. For decoders, `TextGeneration.from_run` and `Pretrained.from_run` load runs the same way. A composite model's record includes each registered part. An adapted model records its base model and the adapter's rank, alpha and modules.

You need registration to load a run from its record. An unregistered model still trains, checkpoints and resumes normally. Its record uses the class name in lower case. The first checkpoint prints a warning with the registration line to add. Loading the run also reports that line. Once you add it, the same run loads.

Dew's modules name the logical axes of their parameters, such as `embed`, `heads` and `mlp`. The trainer maps those names onto the device mesh, so the model code does not change when the mesh does. [Distributed training](concepts/distributed.md) describes the mapping.

## Objective

An objective implements two methods:

- `init(key, variables=None)` returns the model's variables.
- `loss(variables, batch, step)` returns the loss as a `Ratio(total, mass)` and an `Aux` with metrics to log.

For validation, an objective can implement `evaluate`. For samples, it can implement `preview`. Its `ema` attribute selects weights for an exponential moving average (EMA). Its `shown` attribute maps metric names to `Shown` values that configure the training display. For an accuracy metric, `Shown(better="higher", percent=True)` colours an increase as progress and prints the value as a percentage.

Dew includes objectives for autoregressive language modeling (`LMObjective`), image and video diffusion (`DiffusionObjective`), masked and block diffusion over tokens (`MaskedDiffusionObjective`, `BlockDiffusionObjective`), JEPA (`JepaObjective`), preference and reinforcement learning (`DPOObjective`, `GRPOObjective`, `PPOObjective`, `FlowGRPOObjective`) and distillation (`DistillationObjective`). To add an architecture, write a module. To add a training method, write an objective. [Custom objectives](concepts/objectives.md) shows how.

## Dataset

`Dataset(train, val, records, batch)` holds two functions. `train(partition)` opens an endless stream of training batches. `val(partition)` opens one pass over validation records. The `DataPartition` argument, `partition`, specifies this process's share of each global batch when several processes train together.

`Dataset.from_records` builds a dataset from records in memory, as in the example. It reshuffles records every epoch, gives each process its share and saves the stream position in checkpoints.

To use a built-in reader, construct a specification and call `.load(batch=...)` to get a `Dataset`. `TokenWindows` reads tokenized text; `HFImages` reads image datasets on the Hugging Face Hub. Images arrive as `uint8` arrays in `[0, 255]`. Token windows are `int32` IDs under the key `"text"`. Grain readers record their position, so checkpoints resume the data stream where it stopped. See [Training data](concepts/data.md) for details.

![Dataset to global batch: train(partition) opens an iterator of host batches, and shard_batch assembles each into one jax.Array split over the mesh's batch axes.](assets/data-pipeline-light.svg)
![Dataset to global batch: train(partition) opens an iterator of host batches, and shard_batch assembles each into one jax.Array split over the mesh's batch axes.](assets/data-pipeline-dark.svg)

## Trainer

`Trainer(objective, optimizer, key=...)` takes an Optax optimizer and a JAX random key. `fit(dataset, steps=...)` places variables on the devices, compiles one training step and repeats it until the step counter reaches `steps`. It logs every `log_every` steps and evaluates every `eval_every` steps, between training steps. If you pass `checkpoints=Checkpoints(directory)`, it also saves every `checkpoint_every` steps.

![Trainer.fit: the state is placed on the mesh, the dataset's iterator feeds a prefetcher, and each compiled step runs the loss, the gradient, the optimizer update and the EMA update; logging, evaluation and checkpoints run on the host between steps.](assets/training-loop-light.svg)
![Trainer.fit: the state is placed on the mesh, the dataset's iterator feeds a prefetcher, and each compiled step runs the loss, the gradient, the optimizer update and the EMA update; logging, evaluation and checkpoints run on the host between steps.](assets/training-loop-dark.svg)

The same call runs on one device or many. `Trainer(..., mesh=MeshSpec(fsdp=4))` shards parameters and optimizer state over four devices. You can keep the objective, model and data unchanged.

The trainer logs the loss, objective metrics and an injected learning rate. It does not log the gradient norm, because that would add a reduction over every gradient each step. To track it, chain a transformation that stores the norm in the optimizer state. Read `state.opt_state[0]` after `fit`. With `clip_by_global_norm` next to it, XLA computes the norm once for both transformations, so tracking it adds no norm computation:

```python
import jax.numpy as jnp
import optax

record_norm = optax.GradientTransformation(
    lambda params: jnp.zeros(()), lambda updates, state, params=None: (updates, optax.tree.norm(updates)))
optimizer = optax.chain(record_norm, optax.clip_by_global_norm(1.0), optax.adamw(1e-3))
```

## Training state

`TrainState` keeps three counters. `step` counts attempts and, together with the root key, determines the next random draw. `microstep` counts accepted microbatches. `updates` counts optimizer updates, so it differs from `microstep` with gradient accumulation. `state.variables` holds the live variables. `state.averaged` is the same tree with the EMA weights substituted.

After training, call `objective.pipeline(state)` to create an inference task using the weights, such as text generation or text-to-image. Use `Pretrained.save` to write a checkpoint in its source format. [Checkpoints](guides/checkpoints.md) explains how to use each artifact.
