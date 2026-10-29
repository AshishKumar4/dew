"""The evidence a hardware-free gate holds a model's decision to: each case's facts, what the model predicts,
what the compiler measured, and whether their decisions agree (`Evidence`). A gate fails on any case whose
decisions disagree, either way (`assert_agree`), and records every case with its per-term errors, signed and
unthresholded, so drift shows before it flips a decision. Each gate keeps its own model and terms: the kernel
templates' registers and shared memory a launch (tests/kernel_compile.py), a training step's device bytes
(the step-memory gate); they share only this record and its check."""

import dataclasses
import json
import os
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Facts:
    """A case's facts: its program key (the template and call, or model, policy and split), the target's
    generation and device count, its mesh and layout, its shapes, and the capacity its model decides within
    (None where the model states its own budget)."""
    key: tuple[str, ...]
    target: str
    devices: int = 1
    mesh: str = ""
    shapes: tuple = ()
    capacity: int | None = None


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
    """What the compiler measured: its named quantities (None where unmeasured), the decision they give, the
    compiler and its version, and the compile's seconds (None where unmeasured)."""
    quantities: dict[str, int | None]
    admits: bool
    compiler: str
    version: str = ""
    seconds: float | None = None
    detail: tuple = ()


@dataclasses.dataclass(frozen=True)
class Evidence:
    """A case's evidence at a head: its facts, the model's prediction and the compiler's measurement."""
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


RECORDS = "DEW_EVIDENCE_RECORDS"
"""A directory a gate run writes its evidence into (`write`), one JSON line a case, to compare heads."""


def write(records: list[Evidence], name: str) -> None:
    """`records` appended as JSON lines in `RECORDS`' directory where it is set, to a file of `name` and
    this process's own (a gate's invocations each add theirs; parallel workers and a test re-run in a
    process of its own write apart)."""
    if (directory := os.environ.get(RECORDS)) is None or not records:
        return
    lines = "".join(json.dumps({**dataclasses.asdict(record), "agrees": record.agrees,
                                "errors": record.errors}) + "\n" for record in records)
    with Path(directory, f"{name}-{os.getpid()}.jsonl").open("a") as file:
        file.write(lines)


def disagreements(records: list[Evidence]) -> list[str]:
    """Each case whose model decides otherwise than its compiler, described with its errors."""
    return [record.describe() for record in records if not record.agrees]


def assert_agree(records: list[Evidence], name: str) -> None:
    """Fails on any case whose model and compiler disagree, either way, listing each; writes every case
    (`write`) first."""
    write(records, name)
    failed = disagreements(records)
    assert not failed, "\n".join(failed)
