# Quickstart

This page trains a one-layer Flax model to fit the line `y = 2x + 1` on 32 synthetic points. The script downloads nothing and runs on a CPU in a few seconds. It shows the three things every Dew run needs: a `Dataset`, an `Objective` and a `Trainer`. [Installation](installation.md) comes first.

The blocks below form one script, `train.py`.

## Data

```python
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew import Trainer
from dew.data import Dataset
from dew.objectives.base import Aux, Mean, Objective, mean_loss

x = np.linspace(-1, 1, 32, dtype=np.float32).reshape(32, 1)
y = 2 * x + 1
batch = {"x": x, "y": y}
data = Dataset(train=lambda partition: itertools.repeat(batch), val=None,
               records=32, batch=32)
```

A batch is a dictionary of arrays whose first dimension is the batch. Here `x` and `y` are float32 arrays of shape `(32, 1)`.

`Dataset` holds functions that open iterators, not the iterators themselves:

| Argument | Meaning |
|---|---|
| `train` | `train(partition)` returns an iterator of training batches. This one repeats the same batch forever. |
| `val` | The same for validation batches, or `None` for no validation. |
| `records` | The number of training examples. |
| `batch` | The global batch size. |

Repeating one batch is enough to check that optimization works. A real run reads different batches and holds out validation records; [Training data](concepts/data.md) covers both.

## Objective

A Flax Linen module describes a computation and holds no variables. `model.init(key, sample)` creates the variables, and `model.apply(variables, inputs)` runs the computation with them. An `Objective` connects a model to the trainer: `init` returns the variables and `loss` returns the quantity to minimize.

```python
class Regression(Objective):
    def __init__(self, model):
        self.model = model

    def init(self, key, variables=None):
        sample = jnp.zeros((1, 1), dtype=jnp.float32)
        return self.model.init(key, sample)

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        errors = (prediction - batch["y"]) ** 2
        loss = Mean(jnp.sum(errors), jnp.asarray(errors.size))
        mse, _ = mean_loss(loss)
        return loss, Aux(metrics={"mse": mse})


model = nn.Dense(features=1)
objective = Regression(model)
```

`nn.Dense(features=1)` has a `(1, 1)` kernel and a bias: the slope and the intercept of the line. The sample passed to `init` sets the number of input features. It does not fix the batch size.

`loss(variables, batch, step)` returns two values:

- `Mean(total, mass)`, a sum and the count it is averaged over. The trainer adds totals and masses over a gradient-accumulation window and divides once, so the gradient is the gradient of the mean over the whole window. Here the mass is the number of squared errors.
- `Aux(metrics=...)`, scalars to log. `mean_loss` turns a `Mean` into its value.

`step` is a `Step`: `step.step` counts accepted microbatches and `step.key` is a fresh random key for this attempt. This loss is deterministic and uses neither.

## Training

```python
trainer = Trainer(objective, optax.sgd(learning_rate=0.1),
                  key=jax.random.key(0))
state = trainer.fit(data, steps=100, log_every=50)
```

`Trainer` takes the objective, an Optax optimizer and a JAX random key. The key fixes the initialization and every random draw during training. `fit` initializes the variables, places them on the device mesh, compiles one training step with `jax.jit` and runs it until the step counter reaches `steps`. It logs the loss every `log_every` steps. This call writes no checkpoints, since the trainer has no `Checkpoints`, and runs no validation, since `eval_every` is not set.

[Key concepts](key-concepts.md) shows the order of work inside `fit`.

## Result

```python
prediction = model.apply(state.params, x)
mse = float(jnp.mean((prediction - y) ** 2))
print(f"Final mean squared error: {mse:.6f}")
assert mse < 1e-4
```

```bash
JAX_PLATFORMS=cpu python train.py
```

```text
Training from step 0 to 100 on 1 × cpu, 2 parameters
step  50/100  loss 2.228e-04  mse 2.228e-04  0.5 ms/step  58,604 samples/s  0:00:00 left
step 100/100  loss 1.416e-07  mse 1.416e-07  0.4 ms/step  73,148 samples/s  0:00:00 left
Trained 100 steps in 0:00:00: first step after 0.24 s, then 2222.4 step/s
15.8% of the wall time in steps, final loss 1.416e-07
Final mean squared error: 0.000000
```

The first line names the devices and the parameter count. Each `step` line reports the loss and the objective's metrics (`mse` from `Aux`) averaged over the last `log_every` steps, the step time and the throughput. The last two lines report when the first step finished, which includes compilation, and the share of wall time spent in steps. On a terminal `fit` shows a live progress display instead; elsewhere, as here, it prints one line per logging interval. Only process 0 prints. Other backends and library versions print slightly different numbers.

`fit` returns a `TrainState`. `state.params` holds the trained variables, the tree `model.apply` takes. The state also holds the optimizer state, the root key and three counters: `step` counts attempts, `microstep` counts accepted microbatches, and `updates` counts optimizer updates. They differ when gradients are accumulated, or when dynamic loss scaling rejects a step with non-finite values. This objective keeps no moving average, so `state.averaged` raises an error.

## Changing the example

- More input features: give the batch shape `(batch_size, features)` and the `init` sample shape `(1, features)`.
- A nonlinear model: replace `nn.Dense` with your own Linen module. The batch fields and the objective must still agree on names, shapes and dtypes.
- Changing data: return an iterator over your batches instead of `itertools.repeat`. To resume from a checkpoint the iterator must save and restore its position; see [Training data](concepts/data.md) and [Checkpoints](guides/checkpoints.md).

[Custom objectives](concepts/objectives.md) adds evaluation and state to an objective. [Language models](concepts/language_models.md) uses a built-in objective on tokenized text.
