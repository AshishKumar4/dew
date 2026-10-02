# Custom objectives

An `Objective` defines what a run learns: how to initialize the variables, how to compute the loss and, optionally, how to evaluate a batch. `Trainer` differentiates the loss, applies the optimizer and manages the training state. This page describes each method of `dew.objectives.base.Objective` and ends with the built-in objectives. It assumes the [Quickstart](../getting-started.md) and Flax Linen's `init` and `apply`.

| Method or attribute | Required | Returns |
|---|---|---|
| `init(key, variables=None)` | Yes | The complete Flax variables tree, with a `params` collection |
| `loss(variables, batch, step)` | Yes | `(statistics, Aux)`; statistics is usually `Ratio(total, mass)` |
| `held_variables()` | No | Arrays the objective starts from, or `None` |
| `reduce_loss(statistics)` | For composite statistics | `(value, has_data)` |
| `evaluate(variables, batch, step)` | No | Scoring artifacts for a validation batch |
| `preview(variables, batch, step, *, scored=None)` | No | Display artifacts, once per evaluation event |
| `ema` | No | An `EMASpec`, or `None` for no moving average |

An objective is passed to `Trainer` as an instance. Registering it (`dew.registry`) is only needed when a configuration file must find it by name.

## init

`init(key, variables=None)` returns the complete variables mapping. The tree can hold one model or several, as long as `loss` reads the same structure. The trainer traces `init` to find the shapes and then runs it directly in each variable's device placement, so `init` must be pure: it computes arrays from the key and the configuration and does not download weights or open files. Load external weights before constructing the objective and pass them in, as below. The [Quickstart](../getting-started.md#objective) has a complete objective.

### Held weights

An objective that continues from a checkpoint, or keeps a frozen tower next to the model it trains, holds real arrays. The trainer passes those arrays into its compiled state construction as JIT arguments, never as constants compiled into the program. A 0.6B checkpoint compiled in as a constant takes 2.2 GiB inside the executable, which is over the 2 GiB limit for a compilation cache entry.

Two methods describe the held arrays. `held_variables()` returns what the objective starts from, and `init(key, variables=None)` initializes from what the caller passes. `Objective.initializer` combines them into one value `jax.jit` accepts: `Partial(self.init)` when `held_variables()` is `None`, and `Partial(self.init, variables=held)` otherwise. `jax.tree_util.Partial` is a pytree whose bound arguments are its children, so the held tree arrives as data. An objective that builds its whole tree from the key only needs `init`. An objective that holds weights:

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

`variables=None` means the objective's own configured input, which is what a plain `init(key)` uses. The trainer always passes the tree through the initializer, so nothing is read off the objective inside the trace. Because the tree is an argument, it stays an argument however deeply `init` nests its own `jax.jit`. An objective that wraps another passes the held tree on to that objective's `init`. A subclass of an objective that holds weights must accept the `variables` parameter; if it does not, the call raises.

`Trainer.initial_state(initializer=None, key=None)` is the one place the state is built; each `None` is filled in from the run. `Trainer.place` calls it once for the shapes and once for the values, so `trainer.initial_state()` returns the state a run starts from.

## loss

`loss(variables, batch, step)` returns `(statistics, aux)`. `Ratio(total, mass)` holds additive terms that share one denominator; the mass is nonnegative and does not depend on the parameters. The trainer adds totals and masses over an accumulation window and then divides, treating zero mass as no data. A plain scalar is one term with unit mass; Dew does not infer token or row weights from a scalar.

For a composite loss, return a pytree whose leaves are additive sufficient statistics and implement `reduce_loss(statistics) -> (value, has_data)`, keeping independent denominators separate. `scalar_loss(objective, variables, batch, step)` computes `(value, aux)` from the same statistics for direct differentiation. A loss that is not additive over the batch needs its own decomposition into additive statistics; a per-microbatch mean is not a substitute.

`Aux(metrics=...)` holds scalar arrays to report next to the loss. With a tracker configured, the trainer records them as `train/<name>` at the logging interval. They are measured on the training batch, not on the validation set.

`Aux.variables` holds replacements for non-parameter state, such as BatchNorm statistics, applied in order after each accepted microbatch. `Aux.effects` holds additive observations for `apply_effects(variables, effects)`, which returns non-parameter replacements once per supported optimizer commit. Router balancing uses these deferred counts so its bias stays fixed within an accumulation window. `Aux.qk_stats` carries attention observations, and the trainer keeps the per-head maximum across accepted microbatches.

## Non-parameter state

A variables tree is a nested mapping whose outer keys are collections, such as `params` for trainable arrays and `batch_stats` for BatchNorm's running statistics:

```text
variables
  params
    projection: kernel, bias
    norm: scale, bias
  batch_stats
    norm: mean, var
```

`apply(..., mutable=["batch_stats"])` returns the collections that changed. Returning them in `Aux.variables` makes the trainer store them with the updated parameters. This example trains a BatchNorm model and checks that the running mean was stored:

```python
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew import Trainer
from dew.data import Dataset
from dew.objectives.base import Aux, Ratio, Objective, mean_loss


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
        loss = Ratio(jnp.sum(errors), jnp.asarray(errors.size))
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

`updated` replaces the whole `batch_stats` collection; Dew does not merge part of a collection, so a leaf left out is dropped. The optimizer owns `params`, so `Aux.variables` must not carry a `params` collection. At inference, call this model with `train=False` to use the stored running statistics.

[API overview](../reference/core-api.md) describes how collections and EMA weights are selected.

## Randomness

`Step.step` is the number of accepted microbatches. `Step.key` is `jax.random.fold_in(root_key, state.step)`, where `state.step` counts attempts, so a rejected attempt does not reuse its random draws. Split it when a loss needs several independent random operations.

The root key is part of `TrainState`. A deterministic key does not make results bitwise equal across devices, compiler versions or reduction orders. Continuing a run exactly also needs the checkpointed state and the data iterator's position. On a GPU it also needs `--xla_gpu_deterministic_ops=true`, which orders the reductions; [checkpoints](../guides/checkpoints.md) covers autotuning, the other source XLA names, and what turning it off costs.

## EMA

The base `Objective` has `ema=None`. An `EMASpec` turns on an exponential moving average: a decay schedule and a path filter that selects the leaves to average. JEPA averages its context encoder; DPO uses a decay of one to keep a frozen reference.

The trainer stores the selected copy in `TrainState.ema`. `state.averaged` lays it over the live variables and raises when no EMA is configured. EMA arithmetic runs in at least fp32 (fp64 when the leaves are fp64) and stores each result in the leaf's initialized dtype, so there is no wider persistent copy. A decay of one keeps the reference exact. The copy counts toward memory.

The EMA is updated only when the optimizer commits an update. `Ratio` accumulation keeps one gradient tree, at least fp32, and a mass; float64 inputs keep their precision when JAX x64 is on. The finished gradient is cast to each parameter's dtype before Optax sees it, so the optimizer state keeps its dtypes.

Composite accumulation keeps each microbatch's inputs and snapshots of the mutable state it read, then replays the pullbacks with the final normalization coefficients. The loss must therefore be pure; collect rollouts and external rewards before the compiled step. Replays do not apply mutable writes twice. BatchNorm over separate microbatches keeps its sequential behavior and differs from one BatchNorm pass over the full batch.

## Evaluation

`evaluate(variables, batch, step)` returns scoring artifacts for a validation batch: one artifact, a tuple or `None`. Artifact types include token scores, image grids, text samples and representations. A `Metric[S]` declares the artifact type it reads, computes sufficient statistics per batch, merges them over the pass and finalizes once.

`preview(variables, batch, step, *, scored=None)` produces display artifacts once per evaluation event; the base implementation reuses the first scoring artifacts. Every process runs the numerical work and the gathers; decode only on process zero, after every gather has finished.

`fit` evaluates only when `eval_every` is set, and refuses an `eval_every` whose pass nothing would read: it needs `metrics`, or `preview=True` with a tracker. Evaluation runs outside the compiled training step, so `evaluate` should compile its own expensive device work. [Evaluation and tracking](../guides/evaluation.md) describes scheduling and metrics.

## Built-in objectives

| Objective | Expected model and data |
|---|---|
| `LMObjective` | A decoder with hidden-state and vocabulary-head methods; integer token rows, optionally packing and role fields |
| `DiffusionObjective` | A model accepting noisy samples, noise levels, and configured conditions; image or video batches |
| `JepaObjective` | A context encoder and predictor; image or video batches and a masking specification |
| `DPOObjective` | A language decoder; chosen/rejected token pairs and completion masks |
| `GRPOObjective` | A language decoder; generated responses with old log probabilities, masks, rewards, and advantages |

Each built-in objective expects more from its model than a plain `flax.linen.Module`; the matching guide lists the methods it calls.
