# Key concepts

A training run in Dew is four objects: a Flax model, an objective, a dataset and a trainer. Each owns one part of the work, and the boundaries between them are the API. Once you know which object owns what, you know where a change goes.

| Object | What it owns | When you write your own |
|---|---|---|
| Model, a `flax.linen.Module` | The forward computation and the shape of its variables | For a new architecture. `models.build(name, ...)` builds a registered one. |
| `Objective` | Initializing the variables, the loss, evaluation outputs, and which weights keep a moving average | For a loss Dew does not have. |
| `Dataset` | The iterators that yield training and validation batches | When your data does not fit a built-in reader. |
| `Trainer` | The device mesh, the compiled step, the optimizer update, the moving average, checkpoints and logging | You configure it; you do not subclass it. |

`Trainer.fit` returns a `TrainState`, which holds everything a run needs to continue: the variables, the optimizer state, the root random key, the step counters and the moving-average copy.

## One run, end to end

This trains a small decoder on one repeated sentence and then generates from it. It downloads nothing and runs on a CPU in about twenty seconds.

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

On four cores of a workstation CPU it prints:

```text
Training from step 0 to 100 on {'data': 1, 'expert': 1, 'fsdp': 1, 'tensor': 1, 'sequence': 1, 'stage': 1} (1 process(es))
step 25: loss 0.0306
step 50: loss 0.0102
step 75: loss 0.0067
step 100: loss 0.0051
Goodput: first step after 1.99 s, 31.6% of the wall time in steps
dew trains jax models. dew trains jax model
```

Each row holds 65 byte ids: `LMObjective(seq_len=64)` feeds the first 64 to the model and predicts the 64 that follow, one position later. The loss falls from 0.031 at step 25 to 0.005 at step 100, and the model continues the prompt with the sentence it learned. `records` is the number of training examples and `batch` the global batch size. `generate` returns the prompt followed by the new tokens, and `temperature=0` picks the most likely token at every step.

## Models are plain Flax modules

A model knows nothing about training. It is a `flax.linen.Module` with `init` and `apply`, and its variables are an ordinary nested dictionary of arrays. `models.build("causal_transformer", ...)` looks the class up in a registry by name, which is how recipes and saved runs rebuild a model from a configuration file. Importing the class gives the same module: `from dew.nn.backbones import CausalTransformer`.

Dew's modules name the logical axes of their parameters, such as `embed`, `heads` and `mlp`. The trainer maps those names onto devices, so the model code does not change when the mesh does. [Distributed training](concepts/distributed.md) explains the mapping.

## The objective says what is learned

An objective implements `init(key)`, which returns the model's variables, and `loss(variables, batch, step)`, which returns the loss and any metrics to log. It can also implement `evaluate` for validation and `preview` for samples, and it names the weights that keep an exponential moving average (EMA).

Dew ships objectives for autoregressive language modeling (`LMObjective`), image and video diffusion (`DiffusionObjective`), masked and block diffusion over tokens (`MaskedDiffusionObjective`, `BlockDiffusionObjective`), JEPA (`JepaObjective`), preference and reinforcement learning (`DPOObjective`, `GRPOObjective`, `PPOObjective`, `FlowGRPOObjective`) and distillation (`DistillationObjective`). A new kind of model is a new module; a new kind of training is a new objective. [Write a custom objective](concepts/objectives.md) walks through one.

## A dataset is a pair of iterator factories

`Dataset(train, val, records, batch)` holds two functions. `train(partition)` opens an endless, shuffled stream of batches; `val(partition)` opens one pass over the held-out records and then stops. `partition` says which share of each global batch this process reads, which matters once several processes train together.

The built-in readers, such as `TokenWindows` for tokenized text and `HFImages` for image datasets on the Hugging Face Hub, describe a dataset and build a `Dataset` with `.load(batch=...)`. Images arrive as `uint8` arrays in `[0, 255]` and token windows as `int32` ids under the key `"text"`. Readers built on Grain record their position, so a checkpoint can resume the data stream where it stopped. [Supply training data](concepts/data.md) covers the details.

## The trainer runs the loop

`Trainer(objective, optimizer, key=...)` takes an Optax optimizer and a JAX random key. `fit(dataset, steps=...)` initializes the variables on the devices, compiles one training step, and runs it until the step counter reaches `steps`. Along the way it updates the moving average, logs every `log_every` steps, evaluates every `eval_every` steps, and writes a checkpoint every `checkpoint_every` steps when the `Trainer` was given `checkpoints=Checkpoints(directory)`.

The same call runs on one device or many. `Trainer(..., mesh=MeshSpec(fsdp=4))` shards the parameters and optimizer state over four devices; the objective, the model and the data do not change.

## Training state

`TrainState` counts three things separately. `step` counts attempts, and together with the root key it decides the next random draw. `microstep` counts accepted microbatches. `updates` counts optimizer updates, which differs from `microstep` when you accumulate gradients. `state.params` holds the live variables, and `state.averaged` holds the same tree with the EMA weights in place, which usually sample better.

After training, `objective.pipeline(state)` wraps the weights in an inference task, such as text generation or text-to-image, and `Pretrained.save` writes a checkpoint in its source's own format. [Save and resume](guides/checkpoints.md) lists which artifact continues what.
