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
from typing import TYPE_CHECKING, Protocol

import flax.linen as nn

from dew._records import JSON
from dew.data.compile import Handed, fitted
from dew.data.dataset import DataPartition, Loader
from dew.data.pipeline import Dataset, ObjectiveData
from dew.telemetry.instrumentation import count_compilation_cache
from dew.training._fit import admitted
from dew.training.budget import run_length, run_steps, run_tokens
from dew.training.evaluation import EvalSuite, Split
from dew.training.runtime import ready_process

if TYPE_CHECKING:
    from dew.checkpoints import Resume
    from dew.training.trainer import Trainer

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
    it counts real tokens (`Objective.counts_tokens`), and what it hands it: the plan version its data
    reads by (`Objective.bind_plan`) and the widths its data bound (`Objective.bind_widths`)."""

    @property
    def counts_tokens(self) -> bool: ...

    def data(self) -> ObjectiveData: ...

    def bind_plan(self, version: int) -> None: ...

    def bind_widths(self, widths: Mapping[str, object]) -> None: ...


def run_data(objective: DataObjective, dataset: Dataset | Loader | Handed,
             validation: Dataset | Mapping[str, Dataset | Split | EvalSuite] | None, *,
             steps: int | None, epochs: int | None, tokens: int | None, examples: int | None,
             accumulation: int, startup: Startup,
             ) -> tuple[Loader, Mapping[str, Split | EvalSuite] | None, int | None, int | None, int | None]:
    """A fit's data and its length: its length named once (`run_length`), the process readied, the data
    and its validation compiled with the objective's steps (`fitted`), its token budget, its microsteps
    and its bound in examples, `epochs` of a read with no record count read as passes over its source
    examples among them (`Loader.pass_examples`) (`run_tokens`, `run_steps`), each long phase reported
    (`Startup`)."""
    run_length(steps, epochs, tokens, examples, required=False)
    ready_process()
    with startup.phase("data", "preparing the data"):
        loader, validation = fitted(dataset, validation, objective=objective.data(), examples=examples,
                                    epochs=epochs, steps=steps, tokens=tokens)
    # The plan version the data reads its rows by, a loader's own where it was compiled already (handed
    # data reads by none), then the widths it bound: what a saved run of the objective records.
    if loader.plan is not None:
        objective.bind_plan(loader.plan.version)
    objective.bind_widths(loader.widths)
    # A schedule whose last stage ends its data bounds the run as tokens= does (`run_tokens`).
    tokens = run_tokens(loader, tokens, counts_tokens=objective.counts_tokens)
    with startup.phase("count", "counting the steps the run's examples take", when=examples is not None):
        steps = run_steps(loader, steps, epochs, tokens, accumulation, examples)
    count_compilation_cache()
    return loader, validation, tokens, steps, loader.examples


def resume_admitted[LossT, EffectsT](trainer: "Trainer[LossT, EffectsT]", loader: Loader,
                                     resume: "Resume" = "auto") -> None:
    """Refuse a resume under `resume` whose saved data position `loader` would not read on from, as
    `trainer.fit` refuses it where it opens its data, before a run writes anything over what it resumes
    (`RunConfig.train`): the data fit reads, compiled once as the run reads it (`RunConfig.readied`), the
    position of this process's share at the step it continues (`Checkpoints.resumed`,
    `Checkpoints.position`), and the stream of that share fit opens (`Trainer._stream_partition`), under
    the layout's rules, restored and closed (`_fit.admitted`). Nothing where it continues no step or
    saved no position."""
    checkpoints = trainer.checkpoints
    step = None if checkpoints is None else checkpoints.resumed(resume)
    if checkpoints is None or step is None:
        return
    # The shares fit reads by, its layout placing the rows, the saved one's among them.
    with nn.logical_axis_rules(trainer.layout.axis_rules):
        saved = checkpoints.position(step, DataPartition.of(trainer.device_mesh))
        if saved is not None:
            admitted(loader, trainer._stream_partition(), saved)


def computation_differs(saved: JSON, now: JSON) -> list[str]:
    """Where the computation a checkpoint trained, its inference record (`Objective.inference_record`, which
    every step carries), and this run's differ: each path of the objective's class or its model's record where
    both state it and they disagree, the fields that say how layers execute aside (the
    configured path's own `RunConfig.EXECUTED`). Nothing where either records none (an unrecordable
    model's, or a step that predates the record): only positive evidence refuses (row 5723)."""
    from dew._records import without_fields
    from dew.config import RunConfig

    if not isinstance(saved, Mapping) or not isinstance(now, Mapping) or "unrecorded" in {*saved, *now}:
        return []

    def differing(before: JSON, after: JSON, path: str) -> list[str]:
        if isinstance(before, Mapping) and isinstance(after, Mapping):
            return [found for key in sorted({*before, *after})
                    for found in differing(before.get(key), after.get(key), f"{path}.{key}")]
        return [] if before == after else [path]

    return [found for name in ("objective", "model") if name in saved and name in now
            for found in differing(without_fields(saved[name], RunConfig.EXECUTED),
                                   without_fields(now[name], RunConfig.EXECUTED), name)]
