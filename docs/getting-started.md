# Your first training run

> An AI assistant maintains this document. It is presented as-is.

This tutorial assumes you know Python, NumPy-style arrays, and the idea of lowering a loss with gradients. It explains the Flax Linen and JAX ideas the example needs as they come up. Finish [installation](installation.md) first.

We will fit a line to 32 made-up examples. The target is `y = 2x + 1`, so you can measure the trained model's error yourself, without downloading a dataset or a pretrained checkpoint. The point is to see training work end to end. It says nothing about benchmarks or how a model generalizes.

## Prepare a batch

Create `train.py` and add the blocks below in order. Together they make one complete script.

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
data = Dataset(train=lambda: itertools.repeat(batch), val=None,
               records=32, batch=32)
```

Each array has shape `(32, 1)`. The first dimension is the batch, and the second holds one feature or one target. Both arrays are float32. The mean squared error has the units of the target squared, and here the target has no units.

`Dataset` takes functions that open iterators. Its `train` function returns this one batch repeated forever. `records=32` is the number of training examples, and `batch=32` is the global batch size. `val=None` means there is no validation split. A real dataset needs separate training and validation records. Repeating one batch only shows that the optimizer works.

## Define initialization and loss

A Flax Linen module describes a computation. It does not hold its variables. `model.init(key, sample_input)` creates the variables, and `model.apply(variables, input)` computes a prediction with them.

An `Objective` says how to initialize the variables and which loss statistics to differentiate:

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

`nn.Dense(features=1)` learns a weight matrix and a bias. With this input shape they are the slope and the intercept of the line. The sample passed to `init` tells Flax there is one input feature. It does not fix the batch size to one for training.

`loss` receives the full Flax variables tree, a batch, and a `Step`. It returns a `Mean(total, mass)` and an `Aux` holding training metrics. Dew adds up the totals and the masses over an accumulation window, and only then divides to get the gradient. Here the mass is the number of squared-error elements. Other objectives choose their own mass, so it is not always the batch size. `step.step` counts accepted microbatches, and `step.key` is the random key for the current attempt. This loss is deterministic, so it uses neither.

## Optimize the parameters

```python
trainer = Trainer(objective, optax.sgd(learning_rate=0.1),
                  key=jax.random.key(0))
state = trainer.fit(data, steps=100, log_every=50)
```

The Optax optimizer here is plain stochastic gradient descent. `Trainer` initializes the variables, places them on the visible device mesh, compiles the training step, and reads batches. The first call includes compilation, so its time does not tell you the steady-state speed.

The key fixes the initialization and the random stream used during training. In JAX a key is an explicit value. If your loss is random, split `step.key` into one key per random operation. Using the same key twice gives the same random draw twice.

This run uses no gradient accumulation and no dynamic loss scaling. `steps=100` is the step to stop at. The loop prints the training loss every 50 steps. It writes no checkpoints because we did not pass a `Checkpoints` object, and it runs no validation because `eval_every` is not set.

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

When I ran it, it printed a loss of about `0.0002` at step 50 and `Final mean squared error: 0.000000` after step 100. You may see small differences on another backend or with other library versions. The assertion checks that the model learned the line, and it allows for those differences.

`state.step` counts attempts, `state.microstep` counts accepted microbatches, and `state.updates` counts optimizer updates. `TrainState` also holds the variables, the optimizer state, the root key, the optional EMA, the loss-scaler history, and any unfinished accumulation window. Pass `state.params` to `model.apply`. This objective asks for no EMA, so `state.averaged` raises an error.

## Adapt the example

For a model with several input features, give the batch shape `(batch_size, features)` and the initialization sample shape `(1, features)`. Keep the target and prediction shapes compatible in the loss. For a nonlinear model, replace `nn.Dense` with your own Linen module. The data and the objective still have to agree on field names, shapes, and dtypes.

If your data changes between steps, return an iterator over your batches instead of `itertools.repeat`. To resume from a checkpoint, that iterator must be able to save and restore its position. [Supplying training data](concepts/data.md) and [resuming training](guides/checkpoints.md) explain what it needs.

Next, read [writing a custom objective](concepts/objectives.md) for state and evaluation, or [language models](concepts/language_models.md) for a built-in objective that takes tokenized input.
