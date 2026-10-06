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
an `isinstance` check on it would hold for any model.

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

__all__ = ["AffineHead", "DenoisingModel", "HiddenStates", "Logits", "LogitsFromHidden", "ModelKwarg",
           "OutputTable"]


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
