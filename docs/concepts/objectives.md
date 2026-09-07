# Writing a custom objective

This page assumes you have run [the first training example](../getting-started.md) and understand Flax Linen's `init` and `apply` methods. An `Objective` defines how to initialize a variables tree, compute a loss, and optionally evaluate a batch. `Trainer` differentiates the loss, applies the optimizer, and manages training state.

## Initialization

Subclass `dew.objectives.base.Objective`. Implement `init(key, variables=None)` to return the complete Flax variables mapping, including a `params` collection. It can contain one model or several models, as long as your loss interprets the same structure. Callers pass only the key; the second parameter is for objectives that start from held weights (below).

The trainer traces initialization to determine shapes and then initializes variables with their device placement. Keep `init` pure: it should compute arrays from its key and configuration, without downloading weights or opening files. Load external weights explicitly before constructing an objective that accepts pretrained variables.

The [regression tutorial](../getting-started.md#define-initialization-and-loss) includes a complete custom objective. Register an objective when a configuration needs to look it up through a registry. Passing an instance directly to `Trainer` requires no decorator.

### Objectives that start from held weights

An objective that continues from a checkpoint, or keeps a frozen tower beside the model it trains, holds real arrays. Those arrays cross into the trainer's compiled state construction as JIT arguments, never as compiled-in constants: a 0.6B checkpoint captured as a constant is 2.2 GiB inside the executable, past the 2 GiB limit on a compilation cache entry.

Two public methods describe the held arrays. `held_variables()` reports what the objective starts from; `init(key, variables=None)` initializes from what the caller supplies. `Objective.initializer` combines them into the one value a JIT accepts: `Partial(self.init)` when `held_variables()` is None, otherwise `Partial(self.init, variables=held)`. `jax.tree_util.Partial` is a pytree whose bound arguments are children, so the held tree arrives as data. An objective that draws its whole tree from the key implements only `init`:

```python
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

`variables=None` means resolve the configured input, which is what a plain `init(key)` does. The trainer always supplies the tree through the initializer, so nothing is read off the objective inside the trace.

The call dispatches through public `init`, so a subclass that overrides `init` decides what the state holds whether called directly or compiled by the trainer. Because the tree is an argument it stays one however deeply `init` nests its own `jax.jit`, and an objective that composes another passes the held tree to that objective's `init`. A subclass of a holding objective must accept the second parameter; otherwise the call raises rather than being bypassed.

`Trainer.initial_state(initializer=None, key=None)` is the single state implementation; each None resolves from the run. `place` resolves both inputs once and calls that method for shapes and for values, so a `Trainer` subclass overriding `initial_state` is honoured on every path, and `trainer.initial_state()` still returns the state a run starts from.


## Loss and auxiliary values

`loss(variables, batch, step)` returns `(statistics, aux)`. Return `Mean(total, mass)` for additive terms sharing one nonnegative, parameter-independent denominator. The default reducer divides the summed numerator by the summed mass and treats zero support as inactive. A plain scalar explicitly denotes one unit-mass term. Dew does not infer token or row weights from a scalar.

For a composite loss, return an objective-owned Flax PyTree whose leaves are additive sufficient statistics and implement `reduce_loss(statistics) -> (value, has_data)`. Keep independent denominators separate. For direct differentiation, `scalar_loss(objective, variables, batch, step)` derives `(value, aux)` from these same statistics. Arbitrary non-additive batch losses need their own decomposition; a local mean is not a general substitute.

`Aux(metrics=...)` contains scalar arrays to report with the loss. If a tracker is configured, the trainer records these values under `train/<name>` at the logging cadence. A metric in `Aux` is a training-batch measurement; it is not automatically a whole-validation-set score.

`Aux.variables` contains sequential accepted-microbatch replacements such as BatchNorm state. `Aux.effects` contains additive observations for `apply_effects(variables, effects)`, which returns nonparameter replacements once per supported optimizer commit. Router balancing uses deferred counts so its bias stays fixed within a window. `Aux.qk_stats` carries attention observations; the trainer retains per-head maxima across accepted microbatches.

## Update non-parameter state

A variables tree is a nested mapping. Its outer keys are collections, such as `params` for trainable arrays and `batch_stats` for BatchNorm's running statistics. A leaf is one array, such as a kernel, bias, mean, or variance. A typical shape is:

```text
variables
  params
    projection: kernel, bias
    norm: scale, bias
  batch_stats
    norm: mean, var
```

Linen returns changed collections from `apply(..., mutable=["batch_stats"])`. Pass those collections through `Aux.variables` so the trainer keeps them with the updated parameters. This complete example trains a BatchNorm model and verifies that the running mean was saved:

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
data = Dataset(train=lambda: itertools.repeat(batch), val=None, records=8, batch=8)
objective = StatefulRegression()
trainer = Trainer(objective, optax.sgd(0.001), key=jax.random.key(0))
state = trainer.fit(data, steps=3, log_every=1)
running_mean = np.asarray(state.params["batch_stats"]["norm"]["mean"])
assert np.all(running_mean > 0)
print("Stored running mean:", running_mean)
```

`updated` contains the complete replacement `batch_stats` collection from this call. Omitting a nested leaf is not a request to merge part of a collection. Keep parameter changes in the optimizer; do not return a replacement `params` collection through `Aux.variables`. At inference, call this model with `train=False` to use the stored running statistics. A few updates do not calibrate BatchNorm for a real dataset.

The [core reference](../reference/core-api.md#collections-and-ema-selection) describes collection and EMA selection contracts.

## Randomness

`Step.step` is the accepted-microbatch schedule index. `Step.key` derives from the root key and consumed-attempt count, so rejected attempts do not repeat their draws. Split this key when a loss needs independent random operations.

A `TrainState` stores the run key. A deterministic step key does not by itself guarantee bitwise equality across devices, compiler versions, different reduction orders, or untracked data-loader randomness. Exact continuation also depends on checkpointing the relevant state and iterator position.

## Moving-average variables

The base `Objective` has `ema=None`. Set an `EMASpec` only when the method uses an exponential moving average. It specifies a decay schedule and a path filter selecting the leaves to average. JEPA selects its context encoder; DPO uses unit decay for a frozen reference.

The trainer stores the selected copy in `TrainState.ema`. `state.averaged` overlays it onto live variables and raises when no EMA is configured. EMA calculations use at least fp32 and retain explicit fp64, then store each result in its initialized leaf dtype. This can round differently from earlier implicit dtype promotion, but does not allocate a wider persistent EMA. Unit-decay frozen references remain exact. Account for the selected copy when sizing a run.

The trainer updates EMA only on a supported optimizer commit. Shared `Mean` accumulation keeps one gradient tree at least as precise as fp32 and a mass; explicit float64 inputs keep their precision when JAX x64 is enabled. The completed gradient converts to each parameter's dtype before Optax, preserving its optimizer-state contract. Composite accumulation retains realized inputs and mutable read snapshots, then replays scalar pullbacks with final normalization coefficients. Keep loss computation pure; collect rollouts and external rewards before the compiled step. Replays do not apply mutable writes twice. Separate microbatch BatchNorm calls retain their sequential semantics and do not equal one full-batch BatchNorm forward.

## Evaluation

Override `evaluate(variables, batch, step)` to produce scoring artifacts for the complete coordinated batch. It can return one artifact, a tuple, or `None`. Existing types include token scores, image grids, text samples and representations. A `Metric[S]` declares which artifact type it reads, computes per-batch sufficient statistics, merges them into pass-owned state, and finalizes once. Override `preview(variables, batch, step, *, scored=None)` for once-per-event display work. The base implementation reuses the first scoring artifacts when available. All ranks run numerical work and gathers; decode only on process zero after all gathers finish.

Evaluation is opt-in at the training call: provide validation data and set `eval_every`. Passing `metrics` alone does not trigger it. Evaluation runs outside the compiled optimization step; compile expensive device computation within your evaluation implementation when needed. [Evaluation and tracking](../guides/evaluation.md) describes scheduling, metrics, and current limitations.

## Choose a built-in objective

| Objective | Expected model and data |
|---|---|
| `LMObjective` | A decoder with hidden-state and vocabulary-head methods; integer token rows, optionally packing and role fields |
| `DiffusionObjective` | A model accepting noisy samples, noise levels, and configured conditions; image or video batches |
| `JepaObjective` | A context encoder and predictor; image or video batches and a masking specification |
| `DPOObjective` | A language decoder; chosen/rejected token pairs and completion masks |
| `GRPOObjective` | A language decoder; generated responses with old log probabilities, masks, rewards, and advantages |

A built-in objective's model contract is more specific than `flax.linen.Module`. Use the corresponding guide before replacing its model. For a different loss or state layout, implement your own objective with explicit field and method requirements.
