# Writing a custom objective

This page assumes you have run [the first training example](../getting-started.md) and understand Flax Linen's `init` and `apply` methods. An `Objective` defines how to initialize a variables tree, compute a loss, and optionally evaluate a batch. `Trainer` differentiates the loss, applies the optimizer, and manages training state.

## Initialization

Subclass `dew.objectives.base.Objective`. Implement `init(key)` to return the complete Flax variables mapping, including a `params` collection. It can contain one model or several models, as long as your loss interprets the same structure.

The trainer traces initialization to determine shapes and then initializes variables with their device placement. Keep `init` pure: it should compute arrays from its key and configuration, without downloading weights or opening files. Load external weights explicitly before constructing an objective that accepts pretrained variables.

The [regression tutorial](../getting-started.md#define-initialization-and-loss) includes a complete custom objective. Register an objective when a configuration needs to look it up through a registry. Passing an instance directly to `Trainer` requires no decorator.

## Loss and auxiliary values

`loss(variables, batch, step)` returns `(scalar_loss, aux)`. The scalar must be differentiable with respect to `variables["params"]`. Batch fields are a contract between your data and objective: Dew does not infer whether a column is an image, a token sequence, or a label.

`Aux(metrics=...)` contains scalar arrays to report with the loss. If a tracker is configured, the trainer records these values under `train/<name>` at the logging cadence. A metric in `Aux` is a training-batch measurement; it is not automatically a whole-validation-set score.

Use `Aux.variables` for updated non-parameter collections, such as batch statistics. The update replaces the specified collections; it is not an optimizer update to the `params` collection. `Aux.qk_stats` supplies attention statistics to the optional QK-Clip optimizer path.

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
from dew.objectives.base import Aux, Objective


class NormalizedRegressor(nn.Module):
    @nn.compact
    def __call__(self, x, train=False):
        x = nn.BatchNorm(use_running_average=not train, name="norm")(x)
        return nn.Dense(features=1, name="projection")(x)


class StatefulRegression(Objective):
    def __init__(self):
        self.model = NormalizedRegressor()

    def init(self, key):
        return self.model.init(key, jnp.zeros((1, 4), jnp.float32), train=True)

    def loss(self, variables, batch, step):
        prediction, updated = self.model.apply(
            variables, batch["x"], train=True, mutable=["batch_stats"],
        )
        mse = jnp.mean((prediction - batch["y"]) ** 2)
        return mse, Aux(metrics={"mse": mse}, variables=updated)


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

`step` is a `Step` containing a scalar step number, a JAX key, and optional averaged variables. Use `jax.random.split(step.key, n)` when the loss needs several independent random draws. Do not create a new fixed key inside the loss, since that repeats noise or dropout masks.

A `TrainState` stores the run key. A deterministic step key does not by itself guarantee bitwise equality across devices, compiler versions, different reduction orders, or untracked data-loader randomness. Exact continuation also depends on checkpointing the relevant state and iterator position.

## Moving-average variables

The base `Objective` has `ema=None`. Set an `EMASpec` only when the method uses an exponential moving average. It specifies a decay schedule and a path filter selecting the leaves to average. JEPA selects its context encoder; DPO uses unit decay for a frozen reference.

The trainer stores the selected copy in `TrainState.ema`. `state.averaged` overlays that copy onto the live variables tree. It raises when the objective has no EMA. Some built-in objectives enable EMA by default, so account for the extra parameter storage when sizing a run.

The default training step updates EMA on optimizer-update boundaries when gradient accumulation is enabled. The current implementation has unresolved overflow/checkpoint clock issues and does not preserve token-weighted equivalence across unequal-mask microbatches. See [capabilities and limitations](../reference/support.md).

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
