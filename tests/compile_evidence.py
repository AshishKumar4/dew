"""The evidence the hardware-free kernel gate holds a template's register model to (tests/kernel_compile.py):
each kernel's model prediction beside what ptxas measured, and whether their decisions agree (`Evidence`).
The gate fails on any kernel whose decisions disagree, either way (`disagreements`), listing its per-term
errors, signed."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Facts:
    """A kernel's facts: its key (the case and the kernel's name) and the target's generation."""
    key: tuple[str, ...]
    target: str


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
    """What else the decision fixes (a launch's programs a multiprocessor), compared as the decision is."""


@dataclasses.dataclass(frozen=True)
class Measurement:
    """What the compiler measured: its named quantities (None where unmeasured), the decision they give, and
    the compiler and its version."""
    quantities: dict[str, int | None]
    admits: bool
    compiler: str
    version: str = ""
    detail: tuple = ()


@dataclasses.dataclass(frozen=True)
class Evidence:
    """A kernel's evidence at a head: its facts, the model's prediction and the compiler's measurement."""
    case_id: str
    sha: str
    facts: Facts
    predicted: Prediction
    compiled: Measurement

    @property
    def agrees(self) -> bool:
        """Whether the model decides as the compiler does (its decision and what it fixes)."""
        return (self.predicted.admits, self.predicted.detail) == (self.compiled.admits, self.compiled.detail)

    @property
    def errors(self) -> dict[str, int]:
        """The model's terms less the compiler's quantities, each term both name, signed."""
        return {name: value - measured for name, value in self.predicted.terms.items()
                if isinstance(measured := self.compiled.quantities.get(name), int)}

    def describe(self) -> str:
        return (f"{self.case_id}: the model ({self.predicted.unit} {self.predicted.terms}, total "
                f"{self.predicted.total} against {self.predicted.budget}) decides {self.predicted.admits}"
                f"{self.predicted.detail or ''}, {self.compiled.compiler}'s {self.compiled.quantities} "
                f"{self.compiled.admits}{self.compiled.detail or ''}; errors {self.errors}")


def disagreements(records: list[Evidence]) -> list[str]:
    """Each kernel whose model decides otherwise than its compiler, described with its errors."""
    return [record.describe() for record in records if not record.agrees]
