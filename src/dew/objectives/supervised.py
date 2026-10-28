"""Any Flax model, trained on a loss over its outputs and the batch.

`Supervised(model, loss, metrics=(), inputs="x")` applies the model to the
batch field "x", at the per-example shape the run's data gives it, with no
subclass of `Objective`; `inputs=InputSpec(Field("x", (4,)))` states the shape
instead. `loss(outputs, batch)` reads the model's output as it
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
import types
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.batch import FROZEN, Batch, Variables
from dew.inputs import InputSpec
from dew.nn import categorical
from dew.nn.precision import at_least_fp32
from dew.nn.protocols import Lacks
from dew.objectives.base import Aux, Objective, Ratio, Source, Step, model_rngs

type Outputs = jax.Array | tuple[Outputs, ...] | list[Outputs] | Mapping[str, Outputs]
"""What a model returns: an array, or a tree of them."""

type Criterion = Callable[[Outputs, Batch], jax.Array]
"""A loss or a metric: the model's outputs and the batch, to one value per example."""


class Supervised(Objective[Ratio]):
    """Trains `model`, a Flax module or a loaded source whose weights training
    starts from, on `loss`, reporting `metrics` beside it.

    `inputs` is the batch field the model reads: its name, whose per-example
    shape a run reads off its first batch as the model reads it, after the
    loader's device steps, before the model initializes (`Objective.named_inputs`),
    or an `InputSpec` whose sample states the shape too,
    which the trainer checks the first batch against. The model's
    collections besides its parameters update as it trains, a BatchNorm's
    running statistics among them. A model with no
    starting weights initializes on integer zeros of that shape, which a model
    that embeds ids reads as ids and any other promotes to its own dtype.

    `mode` names the keyword the model's call reads its mode from, `train`
    for a call `__call__(x, train=False)` whose BatchNorm and Dropout depend
    on it: the model is called with it True while it trains and False when
    it is initialized and when a validation pass scores it, which then
    writes nothing. None calls the model on the sample alone, in whatever
    mode its call defaults to.

    `work` is the analytical FLOPs of training on a batch, its rows' forward and backward, as the
    batch's host rows hold them: the numerator of the run's MFU (`train/flops`, `train/mfu`). None
    counts the model's own (`Objective.training_flops`), which a module Dew holds no count for leaves
    unknown, never estimated from the compiled program.
    """

    def __init__(self, model: nn.Module | Source[nn.Module], loss: Criterion,
                 metrics: Sequence[Criterion] = (), *, inputs: InputSpec | str, mode: str | None = None,
                 work: Callable[[Batch], float] | None = None):
        if isinstance(inputs, InputSpec) and (inputs.conditions or inputs.mask is not None):
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
        self.field = inputs if isinstance(inputs, str) else inputs.sample.key
        self.named_inputs, self.inputs = (inputs, None) if isinstance(inputs, str) else (None, inputs)
        self.mode = mode
        self.work = work

    def training_flops(self, model, batch: Batch) -> float:
        """The batch's training FLOPs as `work` declares them, or the model's own count."""
        return super().training_flops(model, batch) if self.work is None else float(self.work(batch))

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        if self.inputs is None:
            raise ValueError(f"the model's input {self.field!r} takes its shape from a run's first batch, "
                             "and nothing has read one yet: fit the trainer, or give inputs=InputSpec(...)")
        sample = jnp.zeros((1, *self.inputs.sample.shape), jnp.int32)
        mode = {} if self.mode is None else {self.mode: False}
        return self.model.init(key, sample, method="__call__", **mode)

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        # The model's own collections besides its parameters, such as a
        # BatchNorm's running statistics, update as it runs, and the trainer
        # keeps the updates (`Aux.variables`).
        held = [name for name in variables if name not in ("params", FROZEN)] if step.training else []
        mode = {} if self.mode is None else {self.mode: step.training}
        outputs, updates = self.model.apply(variables, batch[self.field],
                                            rngs=model_rngs(step.key, training=step.training),
                                            mutable=held, method="__call__", **mode)
        return self._scored(outputs, batch, dict(updates) if held else None)

    def scored_units(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, jax.Array]:
        """`loss`'s statistics, its model run as `loss` runs it, its collections open as `loss` opens them
        and their updates dropped, with each example's share of their total: the sum of its losses, its
        row weighted as `row_mean` weights it."""
        held = [name for name in variables if name not in ("params", FROZEN)] if step.training else []
        mode = {} if self.mode is None else {self.mode: step.training}
        rngs = model_rngs(step.key, training=step.training)
        outputs, _ = self.model.apply(variables, batch[self.field], rngs=rngs, mutable=held,
                                      method="__call__", **mode)
        losses = self.criterion(outputs, batch)
        weights = self.row_weights(batch, losses.shape[0])
        shares = losses.reshape(losses.shape[0], -1).astype(at_least_fp32(losses.dtype)).sum(-1) * weights
        return self._scored(outputs, batch, None)[0], shares

    def _scored(self, outputs: Outputs, batch: Batch, updates: Variables | None) -> tuple[Ratio, Aux]:
        losses = self.criterion(outputs, batch)
        if jnp.ndim(losses) == 0:
            raise ValueError("the loss returns one loss per example, not their mean; Supervised "
                             "takes the mean over the batch's rows itself")
        metrics = {name: self.row_mean(metric(outputs, batch), batch).mean()[0]
                   for name, metric in self.metrics.items()}
        return self.row_mean(losses, batch), Aux(metrics, variables=updates)


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
        raise Lacks("logits", f"output {output} selects a {type(outputs).__name__}, not the logits; name "
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
        return categorical.cross_entropy(selected(outputs, self.output), batch[self.labels])


@dataclasses.dataclass(frozen=True)
class Accuracy:
    """1 where the largest of the logits at `output` (`selected`) is the label in `labels`, else 0."""

    labels: str = "label"
    output: tuple[int | str, ...] = ()

    def __call__(self, outputs: Outputs, batch: Batch) -> jax.Array:
        logits = selected(outputs, self.output)
        return (jnp.argmax(logits, axis=-1) == batch[self.labels]).astype(jnp.float32)


__all__ = ["Accuracy", "Criterion", "CrossEntropy", "Outputs", "Supervised", "selected"]
