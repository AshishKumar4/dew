"""What a model can be asked for, as structural protocols.

An objective, a task or a server reads a model through these capabilities
instead of asking which class it is. A model has a capability by defining
the method; there is no Dew base class to inherit and no wrapper around a
model. A caller checks one with `isinstance(model, Logits)` and calls it
through Flax's own `apply`, with `rngs` and `mutable` passed to `apply`:

    logits = model.apply(variables, tokens, method='logits')
    states = model.apply(variables, tokens, train=True, method='hidden_states',
                         rngs={'dropout': key})
    table = model.apply(variables, method='output_table')

`Logits` and `HiddenStates` are token models' full-sequence reads, with no
cache: a call writes no `cache` collection. `DenoisingModel` is the call every
image and video denoiser already has, the raw network that
`dew.diffusion.process.Denoiser` wraps into a prediction a solver steps
with. `AffineHead` and `LogitsFromHidden` let a loss score the vocabulary
without a second trunk pass: `output_table` gives the matrix a tiled loss
contracts in place of the logits, and `logits_from_hidden` is the exact head
for a head no matrix alone gives.
`DenoisingModel` is an annotation only: every Flax module has a `__call__`, so
an `isinstance` check on it would hold for any model. `RequiresText`,
`IntervalModel` and `TimeScaled` are what a denoiser declares about its
conditions and its time: the text it cannot run without, the interval
duration it can embed, and how fast its time features turn.

The keywords a model takes past its tokens are `ModelKwarg` values:
`ModelInputs.kwargs()` gives a request's token fields and conditioning, and
callers add flags and absent values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax
from flax import struct
from flax.typing import PrecisionLike

from dew.nn.inputs import ModelKwarg

if TYPE_CHECKING:
    from dew.diffusion.process import DenoisingCondition
    from dew.nn.dit import TextContext

__all__ = ["AffineHead", "DenoisingModel", "HiddenStates", "IntervalModel", "Logits", "LogitsFromHidden",
           "ModelKwarg", "OutputTable", "RequiresText", "TimeScaled"]


@struct.dataclass
class OutputTable:
    """A vocabulary head as a matrix a loss contracts: `matrix` as the tree
    stores it, `[vocab, features]` when `vocab_major` and `[features, vocab]`
    otherwise, an optional fp32 `[vocab]` `bias` added after the product,
    the `softcap` applied last, and the `precision` the product runs at.

    Contracting the model's final states with it reproduces the model's own
    logits, bias, softcap and any adapter on the head included. Where no
    matrix does, `output_table` returns None, and the logits come from
    `logits_from_hidden` or `logits` instead.
    """

    matrix: jax.Array
    vocab_major: bool = struct.field(pytree_node=False)
    bias: jax.Array | None = None
    softcap: float | None = struct.field(pytree_node=False, default=None)
    precision: PrecisionLike = struct.field(pytree_node=False, default=None)


@runtime_checkable
class Logits(Protocol):
    """A token model's `[B, S, vocab]` fp32 logits over the whole sequence."""

    def logits(self, tokens: jax.Array, *, train: bool = False, **fields: ModelKwarg) -> jax.Array: ...


@runtime_checkable
class HiddenStates(Protocol):
    """A token model's `[B, S, features]` final states, the head's input."""

    def hidden_states(self, tokens: jax.Array, *, train: bool = False,
                      **fields: ModelKwarg) -> jax.Array: ...


class DenoisingModel(Protocol):
    """A diffusion network's output for `sample` at `time`, given its conditions:
    arrays, encoded text (`TextContext`) or a process's `DenoisingCondition`."""

    def __call__(self, sample: jax.Array, time: jax.Array, *, train: bool = False,
                 **conditions: ModelKwarg | TextContext | DenoisingCondition) -> jax.Array: ...


@runtime_checkable
class AffineHead(Protocol):
    """A model whose logits may be one matrix product of its final states.

    It reads the variables `apply` binds, so an adapter's frozen base weights
    are there too, and `apply` copies nothing: under a gradient the matrix is
    the caller's parameter, which a loss keeps for its backward pass.
    """

    def output_table(self) -> OutputTable | None: ...


@runtime_checkable
class LogitsFromHidden(Protocol):
    """The exact head from final states, for scoring states a loss already holds."""

    def logits_from_hidden(self, hidden: jax.Array) -> jax.Array: ...


@runtime_checkable
class RequiresText(Protocol):
    """A denoising model that cannot run without text, which reaches it under
    the model keyword `text_keyword` on every call: the text runs as a stream
    of its own through the blocks, or every block attends to it, so an
    unconditional call has nothing to run. A denoising model without this
    capability runs unconditionally."""

    @property
    def text_keyword(self) -> str: ...


@runtime_checkable
class IntervalModel(Protocol):
    """A denoising model that, with `interval` set, embeds the `duration` of
    the interval it predicts over beside its time, as MeanFlow's and shortcut
    models' networks do (`dew.diffusion.process.Process.interval`).

    Set, a call without a duration is the instantaneous prediction, the zero
    duration's; unset, a duration is refused. `interval` is a field, so
    `model.clone(interval=True)` is the model an interval process runs.
    """

    @property
    def interval(self) -> bool: ...


@runtime_checkable
class TimeScaled(Protocol):
    """A denoising model whose time features turn `time_scale` times faster
    than its unit table's: the model at scale s reading time t (and duration
    d) computes what it computes at s / k reading k t (and k d).

    A loss that differentiates the model in time, as MeanFlow's and sCM's do,
    needs the scale small (`dew.objectives.diffusion.few_step.SMOOTH_TIME_SCALE`).
    `time_scale` is a field, so `model.clone(time_scale=...)` sets it. Where
    the features are a table in the variables, as `FourierEmbedding`'s
    `constants` are, it sets the table `init` draws; variables loaded from
    another run carry that run's.
    """

    @property
    def time_scale(self) -> float: ...
