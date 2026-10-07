"""Any Flax model, trained on a loss over its outputs and the batch.

`Supervised(model, loss, metrics=(), inputs=InputSpec(Field("x", (4,))))`
applies the model to the batch field `inputs.sample` names, with no subclass
of `Objective`. `loss(outputs, batch)` reads the model's output as it
returns it, an array or any tree of them (an autoencoder's reconstruction
and latent, say), and returns one loss per example, any trailing axes
averaged too; the objective is their mean over the batch's rows
(`Objective.row_mean`), so an evaluation batch's repeated rows count for
nothing. Each metric is called the same way and reported as its own mean,
under its function's name or its class's.

A loss or a metric is a module-level function or a configured callable
object (`CrossEntropy(labels="label")`): a run's record names it by import
path (`ObjectiveConfig`), and writing the record refuses a lambda.
"""

from __future__ import annotations

import dataclasses
import inspect
import types
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from dew.inputs import InputSpec
from dew.objectives.base import Aux, Batch, Objective, Ratio, Source, Step, Variables

type Outputs = jax.Array | tuple[Outputs, ...] | list[Outputs] | Mapping[str, Outputs]
"""What a model returns: an array, or a tree of them."""

type Criterion = Callable[[Outputs, Batch], jax.Array]
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
        call = inspect.signature(type(self.model).__call__).parameters.values()
        needed = [parameter.name for parameter in list(call)[2:] if parameter.default is parameter.empty
                  and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)]
        if needed:
            raise TypeError(f"Supervised calls its model on the sample alone, and "
                            f"{type(self.model).__name__}'s call also needs {', '.join(needed)}")
        self.criterion = loss
        self.metrics: Mapping[str, Criterion] = named
        self.sample = inputs.sample
        self.inputs = inputs

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        return self.model.init(key, jnp.zeros((1, *self.sample.shape), jnp.int32))

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        outputs = self.model.apply(variables, batch[self.sample.key], rngs={"dropout": step.key})
        losses = self.criterion(outputs, batch)
        if jnp.ndim(losses) == 0:
            raise ValueError("the loss returns one loss per example, not their mean; Supervised "
                             "takes the mean over the batch's rows itself")
        return self.row_mean(losses, batch), Aux({name: self.row_mean(metric(outputs, batch), batch).mean()[0]
                                                  for name, metric in self.metrics.items()})


def selected(outputs: Outputs, output: tuple[int | str, ...]) -> jax.Array:
    """The array at the path `output` within the model's outputs: an index
    into a tuple or a list, a key into a mapping, each in turn; the outputs
    themselves for an empty path."""
    for step in output:
        if isinstance(outputs, Mapping) and isinstance(step, str):
            outputs = outputs[step]
            continue
        if isinstance(outputs, (tuple, list)) and isinstance(step, int):
            outputs = outputs[step]
            continue
        raise TypeError(f"output {output} indexes the model's {type(outputs).__name__} with {step!r}")
    if not isinstance(outputs, jax.Array):
        raise TypeError(f"output {output} selects a {type(outputs).__name__}, not the logits; name "
                        "the path to them")
    return outputs


@dataclasses.dataclass(frozen=True)
class CrossEntropy:
    """The softmax cross entropy of the logits at `output` within the model's
    outputs (`selected`) over their last axis, against the integer labels in
    `labels`, in float32 or wider as the logits are."""

    labels: str = "label"
    output: tuple[int | str, ...] = ()

    def __call__(self, outputs: Outputs, batch: Batch) -> jax.Array:
        logits = selected(outputs, self.output)
        logits = logits.astype(jnp.promote_types(logits.dtype, jnp.float32))
        return optax.softmax_cross_entropy_with_integer_labels(logits, batch[self.labels])


@dataclasses.dataclass(frozen=True)
class Accuracy:
    """1 where the largest of the logits at `output` (`selected`) is the label in `labels`, else 0."""

    labels: str = "label"
    output: tuple[int | str, ...] = ()

    def __call__(self, outputs: Outputs, batch: Batch) -> jax.Array:
        logits = selected(outputs, self.output)
        return (jnp.argmax(logits, axis=-1) == batch[self.labels]).astype(jnp.float32)


__all__ = ["Accuracy", "Criterion", "CrossEntropy", "Outputs", "Supervised", "selected"]
