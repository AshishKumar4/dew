# Custom objectives

An `Objective` defines how a run initializes its variables, computes its loss and, optionally, evaluates a batch. `Trainer` differentiates the loss, applies the optimizer and manages the training state. The methods below belong to `dew.objectives.base.Objective`; the final table lists the built-in objectives. You should know the [Quickstart](../getting-started.md) and Flax Linen's `init` and `apply`.

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

Pass an objective instance to `Trainer`. You only need to register it (`dew.registry`) if a configuration file must find it by name.

## init

`init(key, variables=None)` returns the complete variables mapping. The tree can hold one model or several, as long as `loss` reads the same structure. The trainer traces `init` for shapes, then runs it in each variable's device placement. That requires a pure function: compute arrays from the key and configuration without downloading weights or opening files. Load external weights before constructing the objective and pass them in, as below. The [Quickstart](../getting-started.md#objective) has a complete objective.

### Held weights

An objective may hold arrays from a checkpoint or a frozen tower alongside the model it trains. The trainer passes these arrays into compiled state construction as JIT arguments, keeping them out of the program's constants. Compiling a 0.6B checkpoint as a constant takes 2.2 GiB inside the executable, exceeding the 2 GiB limit for a compilation cache entry.

Two methods handle these arrays: `held_variables()` returns the objective's starting tree, and `init(key, variables=None)` initializes from the tree the caller passes. `Objective.initializer` packages them for `jax.jit` as `Partial(self.init)` when `held_variables()` is `None`, or `Partial(self.init, variables=held)` otherwise. The bound arguments of `jax.tree_util.Partial` are pytree children, so JAX receives the held tree as data.

An objective that builds its whole tree from the key only needs `init`. For an objective that holds weights:

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

With `variables=None`, `init(key)` uses the objective's configured input. The trainer always passes the tree through the initializer, so the trace does not read it from the objective. It remains an argument even through nested `jax.jit` calls inside `init`. If an objective wraps another, it passes the held tree to the wrapped objective's `init`. Subclasses of objectives that hold weights must also accept `variables`; otherwise the call raises.

`Trainer.initial_state(initializer=None, key=None)` builds the starting state, using the run's settings for each `None` argument. `Trainer.place` calls it once for shapes and once for values, so `trainer.initial_state()` returns the state the run starts from.

## Several networks, several optimizers

When an objective trains several networks, it can give each one its own optimizer. Examples include a few-step student with its fake-score critic (rCM, DMD2), or a generator with its discriminator. `optimizer(tx, *, accumulation)` builds the optimizer for `params` from the transformation passed to `Trainer`.

`optax.multi_transform` gives each network a copy of `tx` with separate moments, count and schedule. `optax.conditionally_mask` steps each copy only on that network's updates, so Adam's momentum cannot move a network during another's update. The mask counts committed trainer updates; at `accumulation=1`, this is also the `step.step` the loss reads. An alternating objective refuses larger accumulation windows because they would mix phases. `averages(update)` selects which updates change the EMA:

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

`ConsistencyDistillationObjective` (rCM) uses this setup. Its student and fake score update in separate phases with their own optimizer copies, and the EMA follows the student's updates.

## loss

`loss(variables, batch, step)` returns `(statistics, aux)`. `Ratio(total, mass)` holds additive terms that share one denominator; the mass is nonnegative and does not depend on the parameters. The trainer adds totals and masses over an accumulation window and then divides, treating zero mass as no data. A plain scalar is one term with unit mass; Dew does not infer token or row weights from a scalar.

For a composite loss, return a pytree whose leaves are additive sufficient statistics and implement `reduce_loss(statistics) -> (value, has_data)`, keeping independent denominators separate. `objective.scalar_loss(variables, batch, step)` computes `(value, aux)` from the same statistics for direct differentiation. A loss that is not additive over the batch needs its own decomposition into additive statistics; a per-microbatch mean is not a substitute.

`Aux(metrics=...)` holds scalar arrays to report next to the loss. With a tracker configured, the trainer records them as `train/<name>` at the logging interval. They are measured on the training batch, not on the validation set.

The trainer applies `Aux.variables` replacements for non-parameter state, such as BatchNorm statistics, in order after each accepted microbatch. `Aux.effects` holds additive observations for `apply_effects(variables, effects)`, which returns non-parameter replacements once per supported optimizer commit. Router balancing defers these counts so the bias stays fixed within an accumulation window. For the attention observations in `Aux.qk_stats`, the trainer keeps the per-head maximum across accepted microbatches.

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
from dew.objectives.base import Aux, Ratio, Objective


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

`updated` replaces the whole `batch_stats` collection, so any leaf you leave out is dropped. Dew does not merge partial collections. The optimizer updates `params`; `Aux.variables` must not contain that collection. At inference, call this model with `train=False` to use the stored running statistics.

[API overview](../reference/core-api.md) describes how collections and EMA weights are selected.

## Randomness

`Step.step` is the number of accepted microbatches. `Step.key` is `jax.random.fold_in(root_key, state.step)`, where `state.step` counts attempts, so a rejected attempt does not reuse its random draws. Split it when a loss needs several independent random operations.

The root key is part of `TrainState`. A deterministic key does not make results bitwise equal across devices, compiler versions or reduction orders. Exact continuation also requires the checkpointed state and the data iterator's position. On a GPU, use `--xla_gpu_deterministic_ops=true` to order the reductions. [Checkpoints](../guides/checkpoints.md) covers autotuning (the other source XLA identifies) and the cost of disabling it.

## EMA

The base `Objective` has `ema=None`. An `EMASpec` turns on an exponential moving average: a decay schedule and a path filter that selects the leaves to average. JEPA averages its context encoder; DPO uses a decay of one to keep a frozen reference.

The trainer stores the selected copy in `TrainState.ema`. `state.averaged` overlays it onto the live variables and raises when no EMA is configured. EMA arithmetic uses at least fp32 (fp64 for fp64 leaves), then stores each result in the leaf's initialized dtype. There is no wider persistent copy, but the EMA copy still uses memory. A decay of one keeps the reference exact.

The EMA is updated only when the optimizer commits an update. `Ratio` accumulation keeps one gradient tree, at least fp32, and a mass; float64 inputs keep their precision when JAX x64 is on. The finished gradient is cast to each parameter's dtype before Optax sees it, so the optimizer state keeps its dtypes.

Composite accumulation keeps each microbatch's inputs and snapshots of the mutable state it read, then replays the pullbacks with the final normalization coefficients. The loss must therefore be pure; collect rollouts and external rewards before the compiled step. Replays do not apply mutable writes twice. BatchNorm over separate microbatches keeps its sequential behavior and differs from one BatchNorm pass over the full batch.

## Evaluation

`evaluate(variables, batch, step)` returns scoring artifacts for a validation batch: one artifact, a tuple or `None`. Artifact types include token scores, image grids, text samples and representations. A `Metric[S]` declares the artifact type it reads, computes sufficient statistics per batch, merges them over the pass and finalizes once.

`preview(variables, batch, step, *, scored=None)` produces display artifacts once per evaluation event; the base implementation reuses the first scoring artifacts. Every process runs the numerical work and the gathers; decode only on process zero, after every gather has finished.

To enable evaluation in `fit`, set `eval_every` and provide either `metrics` or `preview=True` with a tracker. An interval without either is refused because nothing would read the results. Evaluation runs outside the compiled training step, so `evaluate` should compile its own expensive device work. [Evaluation and tracking](../guides/evaluation.md) describes scheduling and metrics.

## Built-in objectives

| Objective | Expected model and data |
|---|---|
| `LMObjective` | A decoder with hidden-state and vocabulary-head methods; integer token rows, optionally packing and role fields |
| `DiffusionObjective` | A model accepting noisy samples, noise levels, and configured conditions; image or video batches |
| `JepaObjective` | A context encoder and predictor; image or video batches and a masking specification |
| `DPOObjective` | A language decoder; chosen/rejected token pairs and completion masks |
| `GRPOObjective` | A language decoder; generated responses with old log probabilities, masks, rewards, and advantages |

Each built-in objective expects more from its model than a plain `flax.linen.Module`; the matching guide lists the methods it calls.
