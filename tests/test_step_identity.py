"""A data step's identity (`dew.data.plan.StepIdentity`) across Dew revisions: a field added to a
configuration after runs were saved with it (`dew._records.ADDED`)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dew._records import ADDED
from dew.data.plan import StepIdentity


@dataclasses.dataclass(frozen=True)
class Scaled:
    """A step as a run saved it, before it had `shift`."""

    scale: int


@dataclasses.dataclass(frozen=True)
class ScaledLater:
    """`Scaled`, given a field after runs were saved with it (`dew._records.ADDED`)."""

    scale: int
    shift: int = dataclasses.field(default=0, repr=False, metadata={ADDED: True})


@pytest.mark.parametrize("held", [False, True], ids=["repr", "configured"])
def test_a_field_added_after_runs_were_saved_keeps_their_identity_until_set(held):
    """A step given a field after runs were saved keeps the identity it had where a run leaves the field
    at its default, so the run resumes under the Dew that added it, whether its identity is its repr or
    its configured fields (a step that holds an array); one that sets the field is a step of its own."""
    weights = np.arange(3) if held else None

    def identity(step):
        named = StepIdentity.of(step)
        return named.config.replace(type(step).__qualname__, "Step")

    before = Scaled(2) if not held else HeldScaled(2, weights)
    after = ScaledLater(2) if not held else HeldScaledLater(2, weights)
    assert identity(after) == identity(before)
    changed = (ScaledLater(2, shift=1) if not held else HeldScaledLater(2, weights, shift=1))
    assert identity(changed) != identity(before)
    # Set in a configuration a step holds in a sequence or a mapping, each value is a step of its own.
    for held_in in (lambda step: (step,), lambda step: {"inner": step}):
        nested = [identity(Holding(held_in(ScaledLater(2, shift=shift)))) for shift in (0, 1, 2)]
        assert len(set(nested)) == 3 and nested[0] == identity(Holding(held_in(Scaled(2)))).replace(
            "Scaled(", "ScaledLater(")


@dataclasses.dataclass(frozen=True)
class Holding:
    """A step holding configurations in a sequence or a mapping."""

    inner: object


@dataclasses.dataclass(frozen=True)
class HeldScaled:
    """`Scaled`, holding an array, which its identity records by its configured fields."""

    scale: int
    weights: np.ndarray | None


@dataclasses.dataclass(frozen=True)
class HeldScaledLater:
    """`HeldScaled`, given a field after runs were saved with it."""

    scale: int
    weights: np.ndarray | None
    shift: int = dataclasses.field(default=0, repr=False, metadata={ADDED: True})
