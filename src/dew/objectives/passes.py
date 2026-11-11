"""How a loss weighs a looped model's passes (`PassWeights`), and what a validation reports of each
(`PassReports`).

A looped decoder (`dew.nn.backbones.decoder_stack.Loop`) runs a block of its layers several times; each pass
can leave the stack as the last one does, through the coda and the final norm (`Loop.exits`), and a loss
can score each pass's exit as it scores the last pass's. Weighing those terms trains every pass count at
once: a model trained so answers at one pass as well as at its last, where a loss of the last pass alone
leaves its earlier passes untrained. Any objective that scores a looped model's exits takes them: a
language model's (`LMObjective`) and a decision model's (`DecisionObjective`) alike.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import jax
from flax import struct

from dew.artifacts import Artifacts
from dew.nn.protocols import Looped, Looping
from dew.objectives.base import Ratio, Weighted

PassWeighting = Literal["mean"] | Sequence[float]
"""How a loss weighs a looped model's passes as a caller states it: "mean", every pass alike, or one weight a
pass in order, the last pass's last, summed to one."""


@dataclass(frozen=True)
class PassWeights:
    """How a loss weighs a looped model's passes (`PassWeighting`), each pass's term the loss of its exit.

    The loss is `Σ_k w_k loss_k`, `loss_k` the loss of pass k's exit of the one loop, one forward. With no
    cut (`Loop.backprop_steps` None) `loss_k` and its gradient are those of the model with its block run k
    times (`Looping.with_passes(k)`) on the same variables and batch. Under a cut, term k is the cut
    loop's exit k, as its exits are without weights: a pass the cut holds back exits without passing its
    gradient back, so its term trains what the cut leaves open (its coda and the head). It is not the
    separately truncated `with_passes(k)` model's, whose window ends at its own exit.
    "mean" weighs each of the passes the step runs alike (one weight `1/T` over `T` passes); stated weights
    are one a pass, nonnegative, summing to one within their own rounding (each a decimal rounded to the
    nearest double below one, so `n` of them sum to one within `n · 2^-53`). Weights that put everything on
    the last pass are the loss of the last pass alone, the model's own output, which the loss then scores as
    it scores any model's (`over`).
    """

    stated: Literal["mean"] | tuple[float, ...]

    @classmethod
    def of(cls, stated: PassWeighting | None) -> PassWeights | None:
        """`stated` checked (`PassWeighting`), None where it is None: the last pass alone."""
        if stated is None:
            return None
        if isinstance(stated, str):
            if stated != "mean":
                raise ValueError(f"pass_weights is 'mean' or one weight a pass, got {stated!r}")
            return cls("mean")
        weights = tuple(float(weight) for weight in stated)
        if not weights or not all(0.0 <= weight <= 1.0 for weight in weights):
            raise ValueError(f"pass_weights weighs each pass by a share in [0, 1], got {list(weights)}")
        if abs(math.fsum(weights) - 1.0) > len(weights) * 2.0 ** -53:
            raise ValueError(f"pass_weights' shares sum to one, got {list(weights)} "
                             f"(summing to {math.fsum(weights)!r})")
        return cls(weights)

    def over(self, steps: int) -> tuple[float, ...] | None:
        """Each of `steps` passes' weight in order, None where every weight is the last pass's: the loss of
        the last pass alone, which no pass's exit needs. Stated weights are refused for another count."""
        if self.stated == "mean":
            weights = (1.0 / steps,) * steps
        elif len(self.stated) != steps:
            raise ValueError(f"pass_weights weighs {len(self.stated)} passes, and the loop runs {steps}")
        else:
            weights = self.stated
        return None if all(weight == 0.0 for weight in weights[:-1]) else weights

    def of_model(self, model: object) -> tuple[float, ...] | None:
        """Each pass's weight over the passes `model`'s loop runs (`over`), refused for a model that runs
        no loop: it has one pass, whose exit is its output."""
        loop = model.loop if isinstance(model, Looped) else None
        if loop is None or not isinstance(model, Looping):
            raise ValueError(f"pass_weights weighs a looped model's passes, and a {type(model).__name__} "
                             "runs no loop (dew.nn.protocols.Looped)")
        return self.over(loop.steps)


def exiting[M](model: M) -> M:
    """`model` with every pass of its loop leaving as the last does (`Loop.exits`), the same parameters:
    what a loss that scores each pass's exit runs. The model as it stands where its loop already exits
    every pass (`Loop.exiting`: an exit gate)."""
    assert isinstance(model, Looped) and isinstance(model, Looping) and model.loop is not None
    if model.loop.exiting:
        return model
    passing = model.with_passes(exits=True)
    assert passing is not None, "a looped model keeps its loop"
    return passing


def weighed(weights: Sequence[float], terms: Sequence[jax.Array]) -> jax.Array:
    """`Σ_k weights[k] · terms[k]` in pass order, each term in its own dtype: the passes' terms of one loss,
    each a pass's per-token or per-question loss, combined before the loss's mean, whose mass every pass
    shares."""
    return functools.reduce(lambda total, term: total + term,
                            (weight * term for weight, term in zip(weights, terms, strict=True)))


@struct.dataclass
class PassReports:
    """What a validation scored of each pass of a looped model, its passes in order, beside the run's own
    report of its last pass (`Objective._validation_scores`): each pass's loss statistics, the run's loss
    had it weighed that pass alone, and the artifacts its metrics read. A validation pass that scores
    metrics reports each of them and the loss of each pass under `<split>/pass_<k>/`
    (`dew.training.evaluation`); one of the loss alone reports the run's."""

    passes: tuple[tuple[Ratio | Weighted, Artifacts], ...]
