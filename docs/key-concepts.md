# Key concepts

A Dew training run is built from four objects. Each owns one part of the work:

| Object | Owns | Written by you when |
|---|---|---|
| Model, a `flax.linen.Module` | The forward computation and the shapes of its variables | You need a new architecture. `models.build(name, ...)` builds a registered one. |
| `Objective` | Initializing the variables, the loss, evaluation outputs, and which weights keep a moving average | You need a loss Dew does not have. |
| `Dataset` | The iterators that yield training and validation batches | Your data does not fit a built-in reader. |
| `Trainer` | The device mesh, the compiled step, the optimizer update, the moving average, checkpoints and logging | Never; it is configured, not subclassed. |

`Trainer.fit` returns a `TrainState`, the numerical state of the run: the variables, the optimizer state, the root random key, the step counters and the moving-average copy. Continuing a run also needs the data position, which checkpoints store beside the state, and the configuration that built the objective and trainer (a recipe writes it to `run.json`).

## Example

This trains a small decoder on one repeated sentence and generates from it. It downloads nothing and runs on a CPU.

```python
import itertools

import jax
import numpy as np
import optax

from dew import Dataset, Trainer, models
from dew.data import ByteTokenizer
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling, generate

tokenizer = ByteTokenizer()
text = tokenizer.encode("dew trains jax models. " * 3)
batch = {"text": np.tile(np.asarray(text[:65], np.int32), (8, 1))}
data = Dataset(train=lambda partition: itertools.repeat(batch),
               val=None, records=8, batch=8)

model = models.build(
    "causal_transformer", vocab_size=tokenizer.vocab_size,
    emb_features=64, num_layers=2, num_heads=4,
    mlp_features=256, max_seq_len=128)
objective = LMObjective(model, seq_len=64)
trainer = Trainer(objective, optax.adamw(3e-3),
                  key=jax.random.key(0))
state = trainer.fit(data, steps=100, log_every=25)

prompt = [tokenizer.encode("dew")]
out = generate(model, state.params, prompt, max_new_tokens=40,
               key=jax.random.key(1),
               sampling=Sampling(temperature=0))
print(tokenizer.decode(out.tokens[0]))
```

Output on four cores of a workstation CPU:

```text
Training from step 0 to 100 on 1 × cpu, 147,840 parameters
step  25/100  loss 0.0306  ce 0.0306  perplexity 1.0311  token_accuracy 1.0000  10.1 ms/step  789 samples/s  0:00:01 left
step  50/100  loss 0.0102  ce 0.0102  perplexity 1.0102  token_accuracy 1.0000  7.7 ms/step  1,033 samples/s  0:00:00 left
step  75/100  loss 0.0067  ce 0.0067  perplexity 1.0067  token_accuracy 1.0000  7.9 ms/step  1,017 samples/s  0:00:00 left
step 100/100  loss 0.0051  ce 0.0051  perplexity 1.0051  token_accuracy 1.0000  7.5 ms/step  1,068 samples/s  0:00:00 left
Trained 100 steps in 0:00:02: first step after 1.53 s, then 129.8 step/s
33.3% of the wall time in steps, final loss 0.0051
dew trains jax models. dew trains jax model
```

Each `step` line shows the loss and the objective's metrics (for `LMObjective`: cross entropy, perplexity and token accuracy) of that step, and the step time and throughput over the interval since the previous line. Each row holds 65 byte ids. `LMObjective(seq_len=64)` feeds the first 64 to the model and scores its predictions of the 64 that follow, shifted by one position. `generate` returns the prompt followed by the new tokens; `temperature=0` picks the most likely token at every step.

## Model

A model is a `flax.linen.Module` with `init` and `apply`. Its variables are a nested dictionary of arrays, and it has no knowledge of training. `models.build("causal_transformer", ...)` looks the class up in a registry by name, which is how recipes and saved runs rebuild a model from a configuration file. Importing the class gives the same module: `from dew.nn.backbones import CausalTransformer`.

Dew's modules name the logical axes of their parameters, such as `embed`, `heads` and `mlp`. The trainer maps those names onto the device mesh, so the model code does not change when the mesh does. [Distributed training](concepts/distributed.md) describes the mapping.

## Objective

An objective implements two methods:

- `init(key, variables=None)` returns the model's variables.
- `loss(variables, batch, step)` returns the loss as a `Mean(total, mass)` and an `Aux` with metrics to log.

It can also implement `evaluate` for validation and `preview` for samples, and it names the weights that keep an exponential moving average (EMA) in its `ema` attribute.

Dew ships objectives for autoregressive language modeling (`LMObjective`), image and video diffusion (`DiffusionObjective`), masked and block diffusion over tokens (`MaskedDiffusionObjective`, `BlockDiffusionObjective`), JEPA (`JepaObjective`), preference and reinforcement learning (`DPOObjective`, `GRPOObjective`, `PPOObjective`, `FlowGRPOObjective`) and distillation (`DistillationObjective`). A new kind of model is a new module; a new kind of training is a new objective. [Custom objectives](concepts/objectives.md) writes one.

## Dataset

`Dataset(train, val, records, batch)` holds two functions. `train(partition)` opens an endless stream of training batches; `val(partition)` opens one pass over the validation records. `partition` is a `DataPartition` that says which share of each global batch this process reads, which matters when several processes train together.

The built-in readers, such as `TokenWindows` for tokenized text and `HFImages` for image datasets on the Hugging Face Hub, are specifications whose `.load(batch=...)` returns a `Dataset`. Images arrive as `uint8` arrays in `[0, 255]` and token windows as `int32` ids under the key `"text"`. Readers built on Grain record their position, so a checkpoint resumes the data stream where it stopped. [Training data](concepts/data.md) covers the details.

![Dataset to global batch: train(partition) opens an iterator of host batches, and shard_batch assembles each into one jax.Array split over the mesh's batch axes.](assets/data-pipeline-light.svg)
![](assets/data-pipeline-dark.svg)

## Trainer

`Trainer(objective, optimizer, key=...)` takes an Optax optimizer and a JAX random key. `fit(dataset, steps=...)` places the variables on the devices, compiles one training step and runs it until the step counter reaches `steps`. Between steps it logs every `log_every` steps, evaluates every `eval_every` steps, and writes a checkpoint every `checkpoint_every` steps when the trainer was given `checkpoints=Checkpoints(directory)`.

![Trainer.fit: the state is placed on the mesh, the dataset's iterator feeds a prefetcher, and each compiled step runs the loss, the gradient, the optimizer update and the EMA update; logging, evaluation and checkpoints run on the host between steps.](assets/training-loop-light.svg)
![](assets/training-loop-dark.svg)

The same call runs on one device or many. `Trainer(..., mesh=MeshSpec(fsdp=4))` shards the parameters and optimizer state over four devices; the objective, the model and the data do not change.

## Training state

`TrainState` keeps three counters. `step` counts attempts, and together with the root key it determines the next random draw. `microstep` counts accepted microbatches. `updates` counts optimizer updates, which differs from `microstep` when gradients are accumulated. `state.params` holds the live variables, and `state.averaged` holds the same tree with the EMA weights in place.

After training, `objective.pipeline(state)` wraps the weights in an inference task, such as text generation or text-to-image, and `Pretrained.save` writes a checkpoint in its source's own format. [Checkpoints](guides/checkpoints.md) lists which artifact continues what.
