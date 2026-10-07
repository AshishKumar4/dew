"""Any Flax model, trained on a loss over its outputs and the batch.

`Supervised(model, loss, metrics=(), inputs=InputSpec(Field("x", (4,))))`
applies the model to the batch field `inputs.sample` names, with no subclass
of `Objective`. `loss(outputs, batch)` returns one loss per example, any
trailing axes averaged too, and the objective is their mean over the batch's
rows (`Objective.row_mean`), so an evaluation batch's repeated rows count for
nothing. Each metric is called the same way and reported as its own mean,
under its function's name or its class's.

A loss or a metric is a module-level function or a configured callable
object (`CrossEntropy(labels="label")`): a run's record names it by import
path (`ObjectiveConfig`), and writing the record refuses a lambda.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from dew.inputs import InputSpec
from dew.objectives.base import Aux, Batch, Objective, Ratio, Source, Step, Variables

type Criterion = Callable[[jax.Array, Batch], jax.Array]
"""A loss or a metric: the model's outputs and the batch, to one value per example."""


class Supervised(Objective[Ratio]):
    """Trains `model`, a Flax module or a loaded source whose weights training
    starts from, on `loss`, reporting `metrics` beside it.

    `inputs.sample` is the batch field the model reads and its per-example
    shape, which the trainer checks the first batch against. A model with no
    starting weights initializes on integer zeros of that shape, which a model
    that embeds ids reads as ids and any other promotes to its own dtype.
    """

    def __init__(self, model: nn.Module | Source[nn.Module], loss: Criterion,
                 metrics: Sequence[Criterion] = (), *, inputs: InputSpec):
        if inputs.conditions or inputs.mask is not None:
            raise ValueError("Supervised feeds its model the sample field alone; a condition or a mask is "
                             "a generative objective's")
        named = {metric.__name__ if isinstance(metric, types.FunctionType) else type(metric).__name__.lower():
                 metric for metric in metrics}
        if len(named) != len(metrics) or "loss" in named:
            raise ValueError(f"the metrics' names {sorted(named)} repeat or include loss; each reports under "
                             "its own name")
        self.model = self.bind_model(model)
        self.criterion = loss
        self.metrics: Mapping[str, Criterion] = named
        self.sample = inputs.sample
        self.inputs = inputs

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        return self.model.init(key, jnp.zeros((1, *self.sample.shape), jnp.int32))

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        outputs = self.model.apply(variables, batch[self.sample.key], rngs={"dropout": step.key})
        # `mutable` is unset, so apply returns the output alone, not a pair.
        assert not isinstance(outputs, tuple)
        losses = self.criterion(outputs, batch)
        if jnp.ndim(losses) == 0:
            raise ValueError("the loss returns one loss per example, not their mean; Supervised "
                             "takes the mean over the batch's rows itself")
        return self.row_mean(losses, batch), Aux({name: self.row_mean(metric(outputs, batch), batch).mean()[0]
                                                  for name, metric in self.metrics.items()})


@dataclasses.dataclass(frozen=True)
class CrossEntropy:
    """The softmax cross entropy of logits over the last axis, against the
    integer labels in `labels`, in float32 or wider as the logits are."""

    labels: str = "label"

    def __call__(self, outputs: jax.Array, batch: Batch) -> jax.Array:
        logits = outputs.astype(jnp.promote_types(outputs.dtype, jnp.float32))
        return optax.softmax_cross_entropy_with_integer_labels(logits, batch[self.labels])


@dataclasses.dataclass(frozen=True)
class Accuracy:
    """1 where the largest logit is the label in `labels`, else 0."""

    labels: str = "label"

    def __call__(self, outputs: jax.Array, batch: Batch) -> jax.Array:
        return (jnp.argmax(outputs, axis=-1) == batch[self.labels]).astype(jnp.float32)


__all__ = ["Accuracy", "Criterion", "CrossEntropy", "Supervised"]
