# Writing a custom objective

This page assumes you have run [the first training example](../getting-started.md) and know Flax Linen's `init` and `apply` methods. An `Objective` says how to initialize a variables tree, how to compute a loss and, optionally, how to evaluate a batch. `Trainer` differentiates the loss, applies the optimizer and manages the training state.

## Initialization

Subclass `dew.objectives.base.Objective`. Implement `init(key, variables=None)` so it returns the complete Flax variables mapping, including a `params` collection. The tree can hold one model or several, as long as your loss reads the same structure. Callers pass only the key. The second parameter is for objectives that start from weights they hold, described below.

The trainer traces initialization to find the shapes, then initializes the variables directly in their device placement. Keep `init` pure. It should compute arrays from its key and configuration, without downloading weights or opening files. If you need external weights, load them yourself before you construct an objective that accepts pretrained variables.

The [regression tutorial](../getting-started.md#define-initialization-and-loss) has a complete custom objective. Register an objective only when a configuration needs to find it through a registry. You can pass an instance straight to `Trainer` without any decorator.

### Objectives that start from held weights

An objective that continues from a checkpoint, or keeps a frozen tower next to the model it trains, holds real arrays. The trainer passes those arrays into its compiled state construction as JIT arguments, never as constants compiled into the program. A 0.6B checkpoint compiled in as a constant takes 2.2 GiB inside the executable, which is over the 2 GiB limit for a compilation cache entry.

Two public methods describe the held arrays. `held_variables()` reports what the objective starts from. `init(key, variables=None)` initializes from what the caller passes. `Objective.initializer` combines them into one value a JIT accepts: `Partial(self.init)` when `held_variables()` is `None`, and `Partial(self.init, variables=held)` otherwise. `jax.tree_util.Partial` is a pytree whose bound arguments are its children, so the held tree arrives as data. An objective that builds its whole tree from the key only needs `init`. Here is one that holds weights:

```python
import jax.numpy as jnp

from dew import Objective


class Continued(Objective):
    def __init__(self, model, pretrained=None):
        self.model, self.pretrained = model, pretrained

    def held_variables(self):
        return self.pretrained

    def init(self, key, variables=None):
        pretrained = self.pretrained if variables is None else variables
        if pretrained is not None:
            return pretrained
        return self.model.init(key, jnp.zeros((1, 4), jnp.float32))
```

`variables=None` means "use the configured input", which is what a plain `init(key)` does. The trainer always passes the tree through the initializer, so nothing is read off the objective inside the trace.

The call goes through the public `init`. A subclass that overrides `init` therefore decides what the state holds, whether you call it directly or the trainer compiles it. Because the tree is an argument, it stays an argument however deeply `init` nests its own `jax.jit`. An objective that wraps another passes the held tree on to that objective's `init`. A subclass of an objective that holds weights must accept the second parameter. If it does not, the call raises instead of silently skipping it.

`Trainer.initial_state(initializer=None, key=None)` is the one place the state is built. Each `None` is filled in from the run. `place` resolves both inputs once and calls this method for the shapes and for the values. A `Trainer` subclass that overrides `initial_state` is therefore used on every path, and `trainer.initial_state()` still returns the state a run starts from.

## Loss and auxiliary values

`loss(variables, batch, step)` returns `(statistics, aux)`. Return `Mean(total, mass)` for additive terms that share one denominator, where the denominator is nonnegative and does not depend on the parameters. The default reducer divides the summed numerator by the summed mass, and treats zero mass as "no data". A plain scalar stands for one term with unit mass. Dew does not guess token or row weights from a scalar.

For a composite loss, return a Flax PyTree that your objective owns, whose leaves are additive sufficient statistics, and implement `reduce_loss(statistics) -> (value, has_data)`. Keep independent denominators separate. To differentiate directly, `scalar_loss(objective, variables, batch, step)` computes `(value, aux)` from the same statistics. A batch loss that is not additive needs its own decomposition. A local mean does not work as a general replacement.

`Aux(metrics=...)` holds scalar arrays to report next to the loss. With a tracker configured, the trainer records them as `train/<name>` at the logging interval. A metric in `Aux` is measured on the training batch. It is not a score over the whole validation set.

`Aux.variables` holds replacements for non-parameter state, such as BatchNorm statistics, applied in order after each accepted microbatch. `Aux.effects` holds additive observations for `apply_effects(variables, effects)`, which returns non-parameter replacements once per supported optimizer commit. Router balancing uses these deferred counts so its bias stays fixed within an accumulation window. `Aux.qk_stats` carries attention observations, and the trainer keeps the per-head maximum across accepted microbatches.

## Update non-parameter state

A variables tree is a nested mapping. Its outer keys are collections, such as `params` for trainable arrays and `batch_stats` for BatchNorm's running statistics. Each leaf is one array, such as a kernel, bias, mean or variance. A typical tree looks like this:

```text
variables
  params
    projection: kernel, bias
    norm: scale, bias
  batch_stats
    norm: mean, var
```

Linen returns the collections that changed from `apply(..., mutable=["batch_stats"])`. Pass those collections through `Aux.variables` so the trainer stores them with the updated parameters. This complete example trains a BatchNorm model and checks that the running mean was saved:

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


class NormalizedRegressor(nn.Module):
    @nn.compact
    def __call__(self, x, train=False):
        x = nn.BatchNorm(use_running_average=not train, name="norm")(x)
        return nn.Dense(features=1, name="projection")(x)


class StatefulRegression(Objective):
    def __init__(self):
        self.model = NormalizedRegressor()

    def init(self, key, variables=None):
        return self.model.init(key, jnp.zeros((1, 4), jnp.float32), train=True)

    def loss(self, variables, batch, step):
        prediction, updated = self.model.apply(
            variables, batch["x"], train=True, mutable=["batch_stats"],
        )
        errors = (prediction - batch["y"]) ** 2
        loss = Mean(jnp.sum(errors), jnp.asarray(errors.size))
        mse, _ = mean_loss(loss)
        return loss, Aux(metrics={"mse": mse}, variables=updated)


x = np.arange(32, dtype=np.float32).reshape(8, 4)
batch = {"x": x, "y": x.mean(axis=1, keepdims=True)}
data = Dataset(train=lambda partition: itertools.repeat(batch), val=None, records=8, batch=8)
objective = StatefulRegression()
trainer = Trainer(objective, optax.sgd(0.001), key=jax.random.key(0))
state = trainer.fit(data, steps=3, log_every=1)
running_mean = np.asarray(state.params["batch_stats"]["norm"]["mean"])
assert np.all(running_mean > 0)
print("Stored running mean:", running_mean)
```

`updated` is the complete replacement `batch_stats` collection from this call. Dew does not merge part of a collection: a nested leaf you leave out is not kept. Leave parameter changes to the optimizer, and do not return a replacement `params` collection through `Aux.variables`. At inference, call this model with `train=False` so it uses the stored running statistics. A few updates are not enough to calibrate BatchNorm on a real dataset.

The [core reference](../reference/core-api.md#collections-and-ema-selection) describes how collections and EMA weights are selected.

## Randomness

`Step.step` is the index of the accepted microbatch in the schedule. `Step.key` comes from the root key and the number of attempts consumed, so a rejected attempt does not reuse its random draws. Split this key when a loss needs several independent random operations.

A `TrainState` stores the run key. A deterministic step key alone does not make results bit-for-bit equal across devices, compiler versions, reduction orders or data-loader randomness that Dew does not track. Continuing a run exactly also needs a checkpoint of the relevant state and the iterator position.

## Moving-average variables

The base `Objective` has `ema=None`. Set an `EMASpec` only if your method uses an exponential moving average. It gives a decay schedule and a path filter that picks the leaves to average. JEPA picks its context encoder. DPO uses a decay of one to keep a frozen reference.

The trainer stores the selected copy in `TrainState.ema`. `state.averaged` lays it over the live variables, and raises when no EMA is configured. EMA arithmetic runs in at least fp32 and keeps explicit fp64, then stores each result in the leaf's initialized dtype. So there is no wider persistent EMA copy in memory. Frozen references with a decay of one stay exact. Count the selected copy when you size a run.

The trainer updates the EMA only on a supported optimizer commit. Shared `Mean` accumulation keeps one gradient tree, at least as precise as fp32, and a mass. Explicit float64 inputs keep their precision when JAX x64 is on. The finished gradient is converted to each parameter's dtype before Optax sees it, so the optimizer state keeps its expected dtypes.

Composite accumulation keeps the realized inputs and snapshots of the mutable state it read, then replays the scalar pullbacks with the final normalization coefficients. Keep the loss computation pure, and collect rollouts and external rewards before the compiled step. Replays do not apply mutable writes twice. BatchNorm calls on separate microbatches keep their sequential behavior, and do not equal one BatchNorm forward pass over the full batch.

## Evaluation

Override `evaluate(variables, batch, step)` to produce scoring artifacts for the complete coordinated batch. It can return one artifact, a tuple or `None`. Artifact types that exist today include token scores, image grids, text samples and representations. A `Metric[S]` declares which artifact type it reads, computes sufficient statistics per batch, merges them into state that the evaluation pass owns, and finalizes once.

Override `preview(variables, batch, step, *, scored=None)` for display work that runs once per event. The base implementation reuses the first scoring artifacts when there are any. All ranks run the numerical work and the gathers. Decode only on process zero, after every gather has finished.

Evaluation is opt-in when you call `fit`: pass validation data and set `eval_every`. Passing `metrics` on its own does not start it. Evaluation runs outside the compiled optimization step, so compile any expensive device work inside your evaluation code. [Evaluation and tracking](../guides/evaluation.md) describes scheduling, metrics and current limits.

## Choose a built-in objective

| Objective | Expected model and data |
|---|---|
| `LMObjective` | A decoder with hidden-state and vocabulary-head methods; integer token rows, optionally packing and role fields |
| `DiffusionObjective` | A model accepting noisy samples, noise levels, and configured conditions; image or video batches |
| `JepaObjective` | A context encoder and predictor; image or video batches and a masking specification |
| `DPOObjective` | A language decoder; chosen/rejected token pairs and completion masks |
| `GRPOObjective` | A language decoder; generated responses with old log probabilities, masks, rewards, and advantages |

Each built-in objective expects more from its model than any `flax.linen.Module`. Read the matching guide before you swap in a different model. For a different loss or state layout, write your own objective and state its field and method requirements.
