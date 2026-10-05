# Quickstart

Train a one-layer Flax model to fit `y = 2x + 1` on 32 synthetic points. The script downloads nothing and runs on a CPU in a few seconds. You will use a `Dataset`, an `Objective` and a `Trainer`, as in every Dew run. Follow [Installation](installation.md) first.

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
from dew.objectives.base import Aux, Ratio, Objective

x = np.linspace(-1, 1, 32, dtype=np.float32).reshape(32, 1)
y = 2 * x + 1
data = Dataset.from_records({"x": x, "y": y}, batch=32)
```

`Dataset.from_records` takes records as columns. Here, `x` and `y` are float32 arrays of shape `(32, 1)`, with one record per row. Each batch is a dictionary with the same fields and a batch dimension first. With `batch=32`, every step reads all 32 points.

The training stream reshuffles records every epoch using `seed`, which defaults to 0. Checkpoints save the stream position, so a resumed run continues where it stopped. With several processes, each reads its own share of every batch. To add held-out records, pass them in the same form to `validation=`. This example has none. [Training data](concepts/data.md) covers file, Hugging Face and TFDS readers, all of which return a `Dataset`.

## Objective

A Flax Linen module describes a computation but stores no variables. `model.init(key, sample)` creates the variables. `model.apply(variables, inputs)` runs the computation using them. In a Dew `Objective`, `init` returns these variables and `loss` returns the quantity the trainer minimizes.

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
        loss = Ratio(jnp.sum(errors), jnp.asarray(errors.size))
        mse, _ = loss.mean()
        return loss, Aux(metrics={"mse": mse})


model = nn.Dense(features=1)
objective = Regression(model)
```

`nn.Dense(features=1)` has a `(1, 1)` kernel for the slope and a bias for the intercept. The sample passed to `init` sets the number of input features. You can still change the batch size.

`loss(variables, batch, step)` returns two values:

- `Ratio(total, mass)` holds a sum and the count used to average it. The trainer adds totals and masses over a gradient-accumulation window, then divides once. This gives the gradient of the mean over the whole window. Here, the mass is the number of squared errors.
- `Aux(metrics=...)` holds scalars to log. Use `Ratio.mean` to get the value of a `Ratio`.

The `Step` argument provides `step.step`, the count of accepted microbatches, and `step.key`, a fresh random key for this attempt. This loss is deterministic and uses neither.

## Training

```python
trainer = Trainer(objective, optax.sgd(learning_rate=0.1),
                  key=jax.random.key(0))
state = trainer.fit(data, steps=100, log_every=50)
```

`Trainer` takes the objective, an Optax optimizer and a JAX random key. The key determines initialization and every random draw during training. `fit` initializes the variables and places them on the device mesh. It compiles one training step with `jax.jit`, then repeats it until the step counter reaches `steps`. It logs the loss every `log_every` steps.

This trainer has no `Checkpoints`, so the call writes no checkpoints. With `eval_every` unset, it runs no validation either.

[Key concepts](key-concepts.md) shows the order of work inside `fit`.

## Result

```python
prediction = model.apply(state.variables, x)
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

The first line lists the model, parameter count, devices, global batch and precision. Each `step` line gives that step's loss and objective metrics. Step time and throughput are averages since the previous line. The acceptance field records whether the step was accepted. With dynamic loss scaling, a step with non-finite values is rejected. The last two lines report when the first step finished, including compilation, and the share of wall time spent in steps.

`fit` prints this output when stdout is a pipe, log file or CI stream. In a terminal, it draws a live panel with the same numbers, a progress bar and a sparkline per metric. Only process 0 prints.

I recorded this output on two cores of a shared workstation CPU. Reading batches through Grain's threads took most of each 10 ms step. Other machines, backends and library versions give different timings.

`fit` returns a `TrainState`. Pass its trained variables, `state.variables`, to `model.apply`. The state also stores the optimizer state and root key. Its three counters track different work: `step` counts attempts, `microstep` counts accepted microbatches, and `updates` counts optimizer updates. These counts differ with gradient accumulation or when dynamic loss scaling rejects a step with non-finite values. This objective keeps no moving average, so `state.averaged` raises an error.

## Changing the example

- To add input features, use batch shape `(batch_size, features)` and `init` sample shape `(1, features)`.
- For a nonlinear model, replace `nn.Dense` with your own Linen module. The batch fields and objective must still agree on names, shapes and dtypes.
- To change the data, pass more records, a smaller `batch`, or held-out records through `validation=` to `from_records`. For data that does not fit in memory, use a dataset specification or `dew.data.load`. See [Training data](concepts/data.md) and [Checkpoints](guides/checkpoints.md).

[Custom objectives](concepts/objectives.md) adds evaluation and state to an objective. [Language models](concepts/language_models.md) uses a built-in objective on tokenized text.
