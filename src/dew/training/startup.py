"""A run's start, before its first step: its data prepared and its length counted (`run_data`), and each
phase of it reported (`Startup`) as it starts and when it ends with what it took, its seconds reported
once beside the first step's scalars, so a start that takes minutes says what it does and how long each
part took: the data's preparation, the count of an examples-bounded run's steps, the step's compile and
the evaluations' warming."""

import contextlib
import dataclasses
import logging
import time
from collections.abc import Iterator, Mapping
from typing import Protocol

from dew.data.compile import fitted
from dew.data.dataset import Loader, Reader
from dew.data.pipeline import Dataset, ObjectiveData
from dew.telemetry.instrumentation import count_compilation_cache
from dew.training.budget import run_length, run_steps, run_tokens
from dew.training.evaluation import EvalSuite
from dew.training.runtime import ready_process

_log = logging.getLogger(__name__)


@dataclasses.dataclass
class Startup:
    """The seconds each phase of a run's start took, by its name, reported once (`scalars`)."""

    seconds: dict[str, float] = dataclasses.field(default_factory=dict)
    reported: bool = False

    @contextlib.contextmanager
    def phase(self, name: str, doing: str, *, when: bool = True) -> Iterator[None]:
        """The phase `name` of the run's start, `doing` it, `when` the run is in it (a step's first
        compile, not a later shape's): logged as it starts and when it ends with what it took, its
        seconds added to `name`'s."""
        if not when:
            yield
            return
        _log.info("%s", doing)
        began = time.perf_counter()
        try:
            yield
        finally:
            took = time.perf_counter() - began
            self.seconds[name] = self.seconds.get(name, 0.0) + took
            _log.info("%s took %.1f s", doing, took)

    def scalars(self) -> dict[str, float]:
        """Each phase's seconds as `startup/<name>_seconds`, the first time they are asked for; none after."""
        if self.reported:
            return {}
        self.reported = True
        return {f"startup/{name}_seconds": seconds for name, seconds in self.seconds.items()}


class DataObjective(Protocol):
    """What a fit's data reads of its objective: its steps over the data (`Objective.data`) and whether
    it counts real tokens (`Objective.counts_tokens`)."""

    @property
    def counts_tokens(self) -> bool: ...

    def data(self) -> ObjectiveData: ...


def run_data(objective: DataObjective, dataset: Dataset | Loader,
             validation: Dataset | Mapping[str, Dataset | Reader | EvalSuite] | None, *,
             steps: int | None, epochs: int | None, tokens: int | None, examples: int | None,
             accumulation: int, startup: Startup,
             ) -> tuple[Loader, Mapping[str, Reader | EvalSuite] | None, int | None, int | None]:
    """A fit's data and its length: its length named once (`run_length`), the process readied, the data
    and its validation compiled with the objective's steps (`fitted`), its token budget and its
    microsteps (`run_tokens`, `run_steps`), each long phase reported (`Startup`)."""
    run_length(steps, epochs, tokens, examples, required=False)
    ready_process()
    with startup.phase("data", "preparing the data"):
        loader, validation = fitted(dataset, validation, objective=objective.data(), examples=examples)
    # A schedule whose last stage ends its data bounds the run as tokens= does (`run_tokens`).
    tokens = run_tokens(loader, tokens, counts_tokens=objective.counts_tokens)
    with startup.phase("count", "counting the steps the run's examples take", when=examples is not None):
        steps = run_steps(loader, steps, epochs, tokens, accumulation, examples)
    count_compilation_cache()
    return loader, validation, tokens, steps
