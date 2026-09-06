# Your first training run

This tutorial assumes Python, NumPy-style arrays, and the idea of minimizing a loss with gradients. It introduces the Flax Linen and JAX concepts needed for this example. Complete [installation](installation.md) first.

We will fit a line to 32 synthetic examples. The target is `y = 2x + 1`, so we can measure the trained model's error without a downloaded dataset or pretrained checkpoint. This is a training demonstration, not a benchmark or generalization result.

## Prepare a batch

Create `train.py` and add the following blocks in order. They form one complete script.

```python
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew import Trainer
from dew.data import Dataset
from dew.objectives.base import Aux, Objective

x = np.linspace(-1, 1, 32, dtype=np.float32).reshape(32, 1)
y = 2 * x + 1
batch = {"x": x, "y": y}
data = Dataset(train=lambda: itertools.repeat(batch), val=None,
               records=32, batch=32)
```

Each array has shape `(32, 1)`: the first dimension is the batch dimension, and the second contains one feature or target. Both arrays use float32. The mean squared error therefore has units of the target squared; here the target is dimensionless.

`Dataset` receives functions that open iterators. Its `train` function returns an endless repetition of this batch. `records=32` describes the number of training examples; `batch=32` is the global batch size. `val=None` means there is no validation split. Real datasets need independent training and validation records; repeating one batch only demonstrates optimization.

## Define initialization and loss

A Flax Linen module describes computation. Its variables are separate from the module object. `model.init(key, sample_input)` creates those variables; `model.apply(variables, input)` computes a prediction.

An `Objective` tells Dew how to initialize variables and compute the scalar loss to differentiate:

```python
class Regression(Objective):
    def __init__(self, model):
        self.model = model

    def init(self, key):
        sample = jnp.zeros((1, 1), dtype=jnp.float32)
        return self.model.init(key, sample)

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        mse = jnp.mean((prediction - batch["y"]) ** 2)
        return mse, Aux(metrics={"mse": mse})


model = nn.Dense(features=1)
objective = Regression(model)
```

`nn.Dense(features=1)` learns a matrix and a bias. For this input shape they represent the line's slope and intercept. The sample passed to `init` defines one input feature; it does not restrict later training to a batch size of one.

`loss` receives the full Flax variables tree, a batch, and a `Step` containing the training step number and random key. This loss is deterministic and does not use `step`. It returns a scalar and `Aux`, which carries training metrics and optional non-parameter updates. Dew differentiates the loss with respect to the `params` collection. The base `Objective` supplies an evaluation method that returns no artifacts, so evaluation is optional here.

## Optimize the parameters

```python
trainer = Trainer(objective, optax.sgd(learning_rate=0.1),
                  key=jax.random.key(0))
state = trainer.fit(data, steps=100, log_every=50)
```

The Optax optimizer applies stochastic gradient descent. `Trainer` initializes the variables, places them on the visible device mesh, compiles the training step, and consumes batches. The first call includes compilation work; its duration is not a steady-state throughput measurement.

The key fixes initialization and the training random stream. JAX keys are explicit values. For a stochastic loss, split `step.key` into separate keys for each random operation. Reusing the same key repeats the same random draw.

In this run there is no gradient accumulation or dynamic loss scaling. `steps=100` sets the final step target. The loop prints training loss every 50 steps. It writes no checkpoints because no `Checkpoints` object was supplied, and it performs no validation because `eval_every` is unset.

## Inspect the result

```python
prediction = model.apply(state.params, x)
mse = float(jnp.mean((prediction - y) ** 2))
print(f"Final mean squared error: {mse:.6f}")
assert mse < 1e-4
```

Run the file:

```bash
JAX_PLATFORMS=cpu python train.py
```

The validation run printed a loss of approximately `0.0002` at step 50 and `Final mean squared error: 0.000000` after step 100. Small differences across backends and library versions are expected. The numerical assertion checks the learned relation and tolerates those differences.

`state` is a `TrainState` containing the step, complete variables tree, optimizer state, random key, and optional moving-average variables. Use `state.params` with `model.apply`. This objective does not request an exponential moving average, so `state.averaged` is not available.

## Adapt the example

For a model with several input features, change the batch to `(batch_size, features)` and change the initialization sample to `(1, features)`. Keep target and prediction shapes compatible with the loss. Replace `nn.Dense` with a Linen module for a nonlinear model; the data and objective still need to agree on field names, shapes, and dtypes.

For data that changes between steps, return an iterator over your batches instead of `itertools.repeat`. To continue from checkpoints, that iterator must expose a restorable position. [Supplying training data](concepts/data.md) and [resuming training](guides/checkpoints.md) explain the requirements.

Next, read [writing a custom objective](concepts/objectives.md) for state and evaluation, or [language models](concepts/language_models.md) for a built-in objective with tokenized inputs.
