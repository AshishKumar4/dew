# Custom objectives

An `Objective` defines what a run learns. It creates the variables, computes the loss and, optionally, evaluates a batch. `Trainer` differentiates the loss, applies the optimizer and manages the training state. This page describes each method of `dew.objectives.base.Objective` and ends with a table of the built-in objectives. It assumes you know the [Quickstart](../getting-started.md) and Flax Linen's `init` and `apply`.

| Method or attribute | Required | Returns |
|---|---|---|
| `init(key, variables=None)` | Yes | The complete Flax variables tree, with a `params` collection |
| `loss(variables, batch, step)` | Yes | `(statistics, Aux)`; statistics is usually `Ratio(total, mass)` |
| `held_variables()` | No | Arrays the objective starts from, or `None` |
| `reduce_loss(statistics)` | For composite statistics | `(value, has_data)` |
| `evaluate(variables, batch, step)` | No | Scoring artifacts for a validation batch |
| `preview(variables, batch, step, *, scored=None)` | No | Display artifacts, once per evaluation event |
| `ema` | No | An `EMASpec`, or `None` for no moving average |
| `optimizer(tx, *, accumulation)` | No | The optimizer the trainer steps `params` with; `tx` by default |
| `averages(update)` | No | Whether an update moves the EMA; every update by default |

You pass `Trainer` an instance of the objective. A configuration file or a saved run names it by its import path and the constructor arguments it states (`ObjectiveConfig`).

## Supervised

A model trained on a loss over its outputs needs no objective of its own. `Supervised(model, loss, metrics=(), *, inputs)` applies any Flax module to the batch field `inputs.sample` names, and `loss(outputs, batch)` returns one loss per example (any trailing axes are averaged too):

```python
import optax

from dew import Supervised
from dew.inputs import Field, InputSpec
from dew.objectives.supervised import Accuracy


def cross_entropy(outputs, batch):
    return optax.softmax_cross_entropy_with_integer_labels(outputs, batch["label"])


objective = Supervised(MLP(), cross_entropy, (Accuracy(),), inputs=InputSpec(Field("x", (2,))))
```

The loss is the mean over the batch's rows, `Objective.row_mean`, so a validation batch's repeated rows count for nothing. Each metric is called the same way and reported as its own mean, under its function's name or its class's (`accuracy`). A loss or metric is a module-level function or a configured callable object, such as `CrossEntropy(labels="label")`: a run's record names it by import path, and saving a run whose loss is a lambda raises. Reading the record back builds a callable object only of a class from Dew, Flax or a package the reader trusts (`RunConfig.load(directory, trust=("mypackage",))`). [Recipes](../recipes.md#python-experiments) trains this objective from a Python experiment file.

## init

`init(key, variables=None)` returns the complete variables mapping. The tree can hold one model or several, as long as `loss` reads the same structure. The trainer traces `init` to find the shapes, then runs it again so that each variable is created directly on the devices it is placed on. So `init` must be pure. It computes arrays from the key and the configuration, and it does not download weights or open files. Load external weights before you construct the objective and pass them in, as the next section shows. The [Quickstart](../getting-started.md#objective) has a complete objective.

### Held weights

An objective that continues from a checkpoint, or keeps a frozen tower next to the model it trains, holds real arrays. The trainer passes those arrays as JIT arguments to the compiled function that builds the state, so they are never compiled into the program as constants. A 0.6B checkpoint compiled in as a constant takes 2.2 GiB inside the executable, which is over the 2 GiB limit for one compilation cache entry.

Two methods deal with the held arrays. `held_variables()` returns the tree the objective starts from, and `init(key, variables=None)` initializes from the tree the caller passes. `Objective.initializer` combines the two into one value that `jax.jit` accepts, `Partial(self.init)` when `held_variables()` is `None` and `Partial(self.init, variables=held)` otherwise. `jax.tree_util.Partial` is a pytree whose children are its bound arguments, so the held tree reaches the compiled function as data.

An objective that builds its whole tree from the key needs only `init`. The objective below holds weights:

```python
import jax.numpy as jnp

from dew import Objective


class Continued(Objective):
    def __init__(self, model, variables=None):
        self.model, self.variables = model, variables

    def held_variables(self):
        return self.variables

    def init(self, key, variables=None):
        start = self.variables if variables is None else variables
        if start is not None:
            return start
        return self.model.init(key, jnp.zeros((1, 4), jnp.float32))
```

The same objective trains a low-rank adapter without any change. Given `adapter.model` and `adapter.variables` from `dew.lora.LoRA(...).apply(model, variables, key=)`, it starts from the factors under `params` and the base weights under `frozen`. The adapted model adds the base back in itself, so the optimizer updates only the factors.

`variables=None` means the objective's own configured input, which is what a plain `init(key)` uses. The trainer always passes the tree through the initializer, so nothing is read off the objective inside the trace. Because the tree is an argument, it stays an argument however deeply `init` nests its own `jax.jit`. An objective that wraps another passes the held tree on to that objective's `init`. A subclass of an objective that holds weights must accept the `variables` parameter; if it does not, the call raises.

The state is built in one place, `Trainer.initial_state(initializer=None, key=None)`, and each argument left as `None` is taken from the run. `Trainer.place` calls it once for the shapes and once for the values, so `trainer.initial_state()` returns the state a run starts from.

## Several networks, several optimizers

An objective that trains more than one network can give each network its own optimizer. Two examples are a few-step student trained beside the fake score that criticizes it (rCM, DMD2), and a generator trained beside its discriminator. `optimizer(tx, *, accumulation)` returns the optimizer the trainer steps `params` with, built from the `tx` you gave `Trainer`.

In the example below, `optax.multi_transform` gives each network its own copy of `tx`, with its own moments, count and schedule. `optax.conditionally_mask` steps a copy only on its own network's updates, so Adam's momentum cannot move one network during another's update. The mask counts the trainer's committed updates, which at `accumulation=1` is also the `step.step` the loss reads. An objective that alternates refuses a larger accumulation, because one accumulation window would mix the two phases. `averages(update)` says which updates move the EMA:

```python
import jax
import optax

from dew import Objective
from dew.objectives.base import EMASpec, under


class Players(Objective):
    ema = EMASpec(decay=optax.constant_schedule(0.999), select=under("params", "gen"))

    def generating(self, update):
        return update % 2 == 0

    def loss(self, variables, batch, step):
        return jax.lax.cond(self.generating(step.step), self.generator_loss,
                            self.discriminator_loss, variables["params"], batch)

    def optimizer(self, tx, *, accumulation):
        if accumulation > 1:
            raise ValueError("the players alternate update by update; train with accumulation=1")
        return optax.multi_transform(
            {"gen": optax.conditionally_mask(tx, self.generating),
             "disc": optax.conditionally_mask(tx, lambda update: ~self.generating(update))},
            {"gen": "gen", "disc": "disc"})

    def averages(self, update):
        return self.generating(update)
```

`ConsistencyDistillationObjective` (rCM) is built this way. Its student and its fake score each update on their own phase, from their own copy of the optimizer, and its EMA follows the student's updates.

## loss

`loss(variables, batch, step)` returns `(statistics, aux)`. `Ratio(total, mass)` holds additive terms that share one denominator; the mass is nonnegative and does not depend on the parameters. `Objective.row_mean(values, batch)` is the `Ratio` of a per-row array's sum and count; compute every sum and batch-wide mean over the batch's rows with it, because a validation pass pads its last batch with repeated rows, which `VALID_ROWS` marks and `row_mean` leaves out. The trainer adds totals and masses over an accumulation window and then divides, treating zero mass as no data. A plain scalar is one term with unit mass; Dew does not infer token or row weights from a scalar.

For a composite loss, return a pytree whose leaves are additive sufficient statistics and implement `reduce_loss(statistics) -> (value, has_data)`, keeping independent denominators separate. `objective.scalar_loss(variables, batch, step)` computes `(value, aux)` from the same statistics for direct differentiation. A loss that is not additive over the batch needs its own decomposition into additive statistics; a per-microbatch mean is not a substitute.

A loss whose update rule is not its own derivative (e-prop's eligibility traces, a forward-gradient or evolution-strategies estimate, a synthetic gradient) returns its statistics through `Objective.with_gradients(stats, gradients, variables["params"])`, where `gradients` mirrors `stats` with each scalar statistic's gradient tree, or None for one no parameter moves. The statistics keep their values, and the trainer steps with the stated rule through the same microbatching, accumulation and sharding as any other loss.

`Aux(metrics=...)` holds scalar arrays to report next to the loss. With a tracker configured, the trainer records them as `train/<name>` at the logging interval. They are measured on the training batch, not on the validation set.

`Aux.variables` holds new values for non-parameter state, such as BatchNorm statistics, and the trainer applies them in order after each accepted microbatch. `Aux.effects` holds additive observations. Once per supported optimizer commit, the trainer passes them to `apply_effects(variables, effects)`, which returns the new non-parameter values. Router balancing uses these deferred counts so that its bias stays fixed within an accumulation window. `Aux.qk_stats` holds attention observations, and the trainer keeps the per-head maximum across accepted microbatches.

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
from dew.objectives.base import Aux, Objective


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
        loss = self.row_mean(errors, batch)
        mse, _ = loss.mean()
        return loss, Aux(metrics={"mse": mse}, variables=updated)


x = np.arange(32, dtype=np.float32).reshape(8, 4)
batch = {"x": x, "y": x.mean(axis=1, keepdims=True)}
data = Dataset(train=lambda partition: itertools.repeat(batch), val=None, records=8, batch=8)
objective = StatefulRegression()
trainer = Trainer(objective, optax.sgd(0.001), key=jax.random.key(0))
state = trainer.fit(data, steps=3, log_every=1)
running_mean = np.asarray(state.variables["batch_stats"]["norm"]["mean"])
assert np.all(running_mean > 0)
print("Stored running mean:", running_mean)
```

`updated` replaces the whole `batch_stats` collection. Dew does not merge part of a collection, so a leaf you leave out is dropped. Only the optimizer writes `params`, so `Aux.variables` must not contain a `params` collection. At inference, call this model with `train=False` to use the stored running statistics.

[API overview](../reference/core-api.md) describes how collections and EMA weights are selected.

## Randomness

`Step.step` is the number of accepted microbatches. `Step.key` is `jax.random.fold_in(root_key, state.step)`, where `state.step` counts attempts, so a rejected attempt does not reuse its random draws. Split it when a loss needs several independent random operations.

The root key is part of `TrainState`. A deterministic key does not make results bitwise equal across devices, compiler versions or reduction orders. To continue a run exactly you also need the checkpointed state and the data iterator's position, and on a GPU you need `--xla_gpu_deterministic_ops=true`, which fixes the order of reductions. [Checkpoints](../guides/checkpoints.md) covers autotuning, which XLA names as the other source of run-to-run differences, and what turning it off costs.

## EMA

The base `Objective` has `ema=None`. An `EMASpec` turns on an exponential moving average: a decay schedule and a path filter that selects the leaves to average. JEPA averages its context encoder; DPO uses a decay of one to keep a frozen reference.

The trainer stores the selected copy in `TrainState.ema`. `state.averaged` returns the live variables with the averaged leaves merged in, and raises when no EMA is configured. EMA arithmetic runs in at least fp32 (fp64 when the leaves are fp64), and each result is stored in the leaf's initialized dtype, so no wider copy is kept between updates. A decay of one keeps the reference exact. The averaged copy still counts toward memory.

The EMA is updated only when the optimizer commits an update. `Ratio` accumulation keeps one gradient tree, at least fp32, and a mass; float64 inputs keep their precision when JAX x64 is on. The finished gradient is cast to each parameter's dtype before Optax sees it, so the optimizer state keeps its dtypes.

Composite accumulation keeps each microbatch's inputs and snapshots of the mutable state it read, then replays the pullbacks with the final normalization coefficients. The loss must therefore be pure; collect rollouts and external rewards before the compiled step. Replays do not apply mutable writes twice. BatchNorm over separate microbatches keeps its sequential behavior and differs from one BatchNorm pass over the full batch.

## Evaluation

`evaluate(variables, batch, step)` returns scoring artifacts for a validation batch: one artifact, a tuple or `None`. Artifact types include token scores, image grids, text samples and representations. A `Metric[S]` declares the artifact type it reads, computes sufficient statistics per batch, merges them over the pass and finalizes once.

`preview(variables, batch, step, *, scored=None)` produces display artifacts once per evaluation event; the base implementation reuses the first scoring artifacts. Every process runs the numerical work and the gathers; decode only on process zero, after every gather has finished.

`fit` evaluates only when `eval_every` is set. It also needs `metrics`, or `preview=True` with a tracker, and it refuses an `eval_every` without either because nothing would read the pass. Evaluation runs outside the compiled training step, so `evaluate` should compile its own expensive device work. [Evaluation and tracking](../guides/evaluation.md) describes scheduling and metrics.

## Built-in objectives

| Objective | Expected model and data |
|---|---|
| `LMObjective` | A decoder with hidden-state and vocabulary-head methods; integer token rows, optionally packing and role fields |
| `DiffusionObjective` | A model accepting noisy samples, noise levels, and configured conditions; image or video batches |
| `JepaObjective` | A context encoder and predictor; image or video batches and a masking specification |
| `DPOObjective` | A language decoder; chosen/rejected token pairs and completion masks |
| `GRPOObjective` | A language decoder; generated responses with old log probabilities, masks, rewards, and advantages |

Each built-in objective expects more from its model than a plain `flax.linen.Module`; the matching guide lists the methods it calls.
