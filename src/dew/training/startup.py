"""A run's start, before its first step: its data prepared and its length counted (`run_data`), and each
phase of it reported (`Startup`) as it starts and when it ends with what it took, its seconds reported
once beside the first step's scalars, so a start that takes minutes says what it does and how long each
part took: the data's preparation, the count of an examples-bounded run's steps, the step's compile and
the evaluations' warming."""

import contextlib
import dataclasses
import functools
import logging
import time
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Protocol

import flax.linen as nn

from dew._records import JSON
from dew.data.compile import Handed, fitted
from dew.data.dataset import DataPartition, Loader
from dew.data.pipeline import Dataset, ObjectiveData
from dew.data.plan import DataResume
from dew.telemetry.instrumentation import count_compilation_cache
from dew.training._fit import data_resumed
from dew.training.budget import run_length, run_steps, run_tokens
from dew.training.evaluation import EvalSuite, Split
from dew.training.runtime import ready_process

if TYPE_CHECKING:
    from dew.checkpoints import Checkpoints, Resume
    from dew.training.trainer import Trainer

_log = logging.getLogger(__name__)


@dataclasses.dataclass
class Startup:
    """The seconds each phase of a run's start took, by its name, and whether its start took the lowering
    begun beside its step's compile (`options.Compiling.taken`; None where none was begun), reported once
    (`scalars`) with the seconds to its first step's result no phase holds: every second of a start is
    named, a phase's or that remainder's."""

    seconds: dict[str, float] = dataclasses.field(default_factory=dict)
    overlapped: bool | None = None
    reported: bool = False
    began: float = dataclasses.field(default_factory=time.perf_counter)
    """When the fit's start began, its plan's first phase (`dew.training.trainer.Trainer._fit_plan`)."""
    stepped: float | None = None
    """When its first step's result came (`perf_counter`), None before: with `began`, the one window every
    phase lies in."""

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

    def first_step(self, started: float) -> float:
        """Mark the first step's result as come now (`stepped`), and give its seconds since `started`."""
        self.stepped = time.perf_counter()
        return self.stepped - started

    def scalars(self) -> dict[str, float]:
        """Each phase's seconds as `startup/<name>_seconds`, the seconds from the start's beginning to its
        first step's result (`began`, `stepped`) no phase holds as `startup/other_seconds`, and
        `startup/start_overlapped` 1 where the start took the lowering begun beside its step's compile (its
        algebra passed once) and 0 where it lowered again, the first time they are asked for; none after."""
        if self.reported:
            return {}
        self.reported = True
        phases = {f"startup/{name}_seconds": seconds for name, seconds in self.seconds.items()}
        if self.stepped is not None:
            phases["startup/other_seconds"] = self.stepped - self.began - sum(self.seconds.values())
        if self.overlapped is None:
            return phases
        return {**phases, "startup/start_overlapped": float(self.overlapped)}


class DataObjective(Protocol):
    """What a fit's data reads of its objective: its steps over the data (`Objective.data`) and whether
    it counts real tokens (`Objective.counts_tokens`), and what it hands it: the widths its data bound
    (`Objective.bind_widths`)."""

    @property
    def counts_tokens(self) -> bool: ...

    def data(self) -> ObjectiveData: ...

    def bind_widths(self, widths: Mapping[str, object]) -> None: ...


def run_data(objective: DataObjective, dataset: Dataset | Loader | Handed,
             validation: Dataset | Mapping[str, Dataset | Split | EvalSuite] | None, *,
             steps: int | None, epochs: int | None, tokens: int | None, examples: int | None,
             accumulation: int, startup: Startup, resume: DataResume = "relaxed",
             ) -> tuple[Loader, Mapping[str, Split | EvalSuite] | None, int | None, int | None, int | None]:
    """A fit's data and its length: its length named once (`run_length`), the process readied, the data and
    its validation compiled with the objective's steps for a resume of `resume` (`fitted`), its token budget,
    its microsteps and its bound in examples, `epochs` of a read with no record count read as passes over
    its source examples among them (`Loader.pass_examples`) (`run_tokens`, `run_steps`), each long phase
    reported (`Startup`)."""
    run_length(steps, epochs, tokens, examples, required=False)
    ready_process()
    with startup.phase("data", "preparing the data"):
        loader, validation = fitted(dataset, validation, objective=objective.data(), examples=examples,
                                    epochs=epochs, steps=steps, tokens=tokens, resume=resume)
    # The widths the data bound: what a saved run of the objective records.
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
    the layout's rules, restored and closed (`_fit.data_resumed`). Nothing where it continues no step or
    saved no position; a relaxed resume (`Trainer.data_resume`) refuses no position whose data can start
    again, the one decision fit makes too (`_fit.data_resumed`)."""
    checkpoints = trainer.checkpoints
    step = resumed_step(trainer, resume)
    if checkpoints is None or step is None:
        return
    # The shares fit reads by, its layout placing the rows, the saved one's among them.
    with nn.logical_axis_rules(trainer.layout.axis_rules):
        data_resumed(loader, trainer._stream_partition(),
                     functools.partial(checkpoints.position, step, DataPartition.of(trainer.device_mesh)),
                     resume=trainer.data_resume, step=step)


def resumed_step[LossT, EffectsT](trainer: "Trainer[LossT, EffectsT]", resume: "Resume") -> int | None:
    """The step `trainer`'s run continues under `resume` (`Checkpoints.resumed`), as every process agrees it,
    or None where it starts afresh: refused, under "auto" too, where its directory holds steps none of which
    resumes and the newest trained another computation (`computation_differs`): another run's model alone,
    beside which a fresh run would write (row 9019). The one check of it, every site's (row 9632)."""
    from dew._coordination import agreed_same

    checkpoints = trainer.checkpoints
    if checkpoints is None:
        return None

    def continued() -> int | None:
        step, held = checkpoints.resumed(resume), max(checkpoints.committed(), default=None)
        if (step is None and held is not None and resume == "auto"
                and computation_differs(trainer, checkpoints, held)):
            raise ValueError(f"resume='auto' starts a run afresh here, and {checkpoints.directory} holds "
                             f"another run's model at step {held}, which no run resumes; give the new run a "
                             "directory of its own")
        return step

    return agreed_same("newest checkpoint", continued)


def computation_differs[LossT, EffectsT](trainer: "Trainer[LossT, EffectsT]", checkpoints: "Checkpoints",
                                         step: int) -> list[str]:
    """Where the computation `checkpoints`' `step` trained and `trainer`'s differ (`differing_records`):
    their inference records and what their steps compute beside the model (`PhasePasses.training_record`,
    a pass count's sampled law among it, row 10783)."""
    saved, now = checkpoints.training(step), trainer.phase_passes.training_record(trainer.objective)
    # A law the checkpoint records binds a run that draws none: a fixed-pass resume would stop drawing (a
    # checkpoint that records none, a fixed-pass run's, any run resumes).
    dropped = (["training.sampled_passes"] if isinstance(saved, dict) and "sampled_passes" in saved
               and "sampled_passes" not in now else [])
    return differing_records(checkpoints.artifact(step), trainer._artifact(), saved, now) + dropped


def differing_records(saved: JSON, now: JSON, trained: JSON, training: JSON) -> list[str]:
    """Where the computation a checkpoint trained and this run's differ: each path of its objective's class
    or its model's record in its inference record (`Objective.inference_record`), the fields that say how
    layers execute aside (the configured path's own `RunConfig.EXECUTED`), and each setting of what its
    objective's loss computes (`trained` against `training`, `Objective.training_record`; row 9020: an LM's
    `z_loss` changed and resumed), where both state it and they disagree. A loop's passes (`Loop.PASSED`,
    which a phase sets) and a field one of them predates are no evidence, nor is a record either holds none
    of (an unrecordable model's, a step that predates the record): only positive evidence refuses (row
    5723)."""
    from dew._records import without_fields
    from dew.config import RunConfig

    found = []
    match saved, now:
        case dict(), dict() if "unrecorded" not in {*saved, *now}:
            found = [differs for name in ("objective", "model") if name in saved and name in now
                     for differs in _differing(without_fields(saved[name], RunConfig.EXECUTED),
                                               without_fields(now[name], RunConfig.EXECUTED), name)]
    match trained, training:
        case dict(), dict():
            found += _differing(trained, training, "training")
    return found


def _differing(before: JSON, after: JSON, path: str) -> list[str]:
    """Each path at which `before` and `after` both state a value and disagree (`differing_records`)."""
    from dew.nn.backbones.decoder_stack import Loop

    # A loop's passes are the run's to set, a phase's own (`Loop.PASSED`).
    passed = Loop.PASSED if path.endswith(".loop") else ()
    match before, after:
        # A field one record holds and the other predates is no evidence: only fields both state compare.
        case dict(), dict():
            return [found for key in sorted((before.keys() & after.keys()) - set(passed))
                    for found in _differing(before[key], after[key], f"{path}.{key}")]
        case _:
            return [] if before == after else [path]
