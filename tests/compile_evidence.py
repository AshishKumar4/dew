"""The evidence a derivation gate holds a model's prediction to (tests/test_derived_plan.py): a model's
prediction beside what the compiler measured, and whether their decisions agree (`Evidence`)."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Prediction:
    """A model's prediction: its named terms (integers in `unit`), their total, the budget it compares the
    total with, and its decision."""
    terms: dict[str, int]
    total: int | None
    budget: int
    admits: bool
    unit: str
    detail: tuple = ()
    """What else the decision fixes, compared as the decision is."""


@dataclasses.dataclass(frozen=True)
class Measurement:
    """What the compiler measured: its named quantities (None where unmeasured), the decision they give, and
    the compiler."""
    quantities: dict[str, int | None]
    admits: bool
    compiler: str
    detail: tuple = ()


@dataclasses.dataclass(frozen=True)
class Evidence:
    """A case's evidence: the model's prediction and the compiler's measurement."""
    case_id: str
    predicted: Prediction
    compiled: Measurement

    @property
    def agrees(self) -> bool:
        """Whether the model decides as the compiler does (its decision and what it fixes)."""
        return (self.predicted.admits, self.predicted.detail) == (self.compiled.admits, self.compiled.detail)
