# Quickstart

This page trains a one-layer Flax model to fit the line `y = 2x + 1` on 32 synthetic points. The script downloads nothing and runs on a CPU in a few seconds. It shows the three things every Dew run needs: a `Dataset`, an `Objective` and a `Trainer`. [Installation](installation.md) comes first.

The blocks below form one script, `train.py`.

## Data

```python
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
data = Dataset.from_records({"x": x, "y": y}, batch=32)
```

`Dataset.from_records` takes the records as columns, here float32 arrays `x` and `y` of shape `(32, 1)` whose first axis is the record. A batch is a dictionary of the same fields whose first dimension is the batch. `batch=32` reads all 32 points every step.

The training stream reshuffles the records every epoch, from `seed` (0 unless given), and a checkpoint saves its position, so a resumed run continues where it stopped. With several processes, each one reads its own share of every batch. `validation=` takes held-out records in the same form; this example has none. [Training data](concepts/data.md) covers the readers for files, Hugging Face and TFDS datasets, and the `Dataset` value they all return.

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
Training Dense from step 0 to 100: 2 parameters, on 1 × cpu, batch 32, float32
step  50/100  loss 2.228e-04  mse 2.228e-04  step_time_ms 10.31  samples_per_sec 3,105  accepted 100.0%  0:00:01 left
step 100/100  loss 1.416e-07  mse 1.416e-07  step_time_ms 13.69  samples_per_sec 2,337  accepted 100.0%
Trained 100 steps in 0:00:02: first step after 0.59 s, then 83.7 step/s
66.6% of the wall time in steps, final loss 1.416e-07
Final mean squared error: 0.000000
```

The first line names the model, its parameter count, the devices, the global batch and the precision. Each `step` line reports the loss and the objective's metrics of that step, the step time and throughput averaged over the interval since the previous line, and whether the step was accepted (a step with non-finite values is rejected under dynamic loss scaling). The last two lines report when the first step finished, which includes compilation, and the share of wall time spent in steps. This is `fit`'s output when stdout is not a terminal, as in a pipe, a log file or CI; on a terminal it draws one live panel with the same numbers, a progress bar and a sparkline per metric. Only process 0 prints. The output above came from two cores of a shared workstation CPU, where reading the batch through Grain's threads takes most of each 10 ms step; other machines, backends and library versions print different timings.

`fit` returns a `TrainState`. `state.params` holds the trained variables, the tree `model.apply` takes. The state also holds the optimizer state, the root key and three counters: `step` counts attempts, `microstep` counts accepted microbatches, and `updates` counts optimizer updates. They differ when gradients are accumulated, or when dynamic loss scaling rejects a step with non-finite values. This objective keeps no moving average, so `state.averaged` raises an error.

## Changing the example

- More input features: give the batch shape `(batch_size, features)` and the `init` sample shape `(1, features)`.
- A nonlinear model: replace `nn.Dense` with your own Linen module. The batch fields and the objective must still agree on names, shapes and dtypes.
- Changing data: pass more records, a smaller `batch`, or `validation=` held-out records to `from_records`. Data that does not fit in memory comes from a dataset specification or `dew.data.load`; see [Training data](concepts/data.md) and [Checkpoints](guides/checkpoints.md).

[Custom objectives](concepts/objectives.md) adds evaluation and state to an objective. [Language models](concepts/language_models.md) uses a built-in objective on tokenized text.
