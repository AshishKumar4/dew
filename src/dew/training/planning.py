"""The plan of a training step: the microbatches it reads its batch in and
the rung of the fit ladder it compiles at (`Trainer.compile`).

A step that does not fit its devices whole is read in microbatches, an even
split of every row shard's own rows, before it recomputes anything:
recomputing costs every token, where a smaller microbatch at the same rung
costs little (`FitSearch`). Compiles are the measurement, XLA's memory analysis of each candidate against
each device's free bytes (`dew.training.memory`), and the search bounds how
many a step takes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import gc
import json
import logging
import tempfile
import threading
import time
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.experimental.profiler import get_profiled_instructions_proto
from jax.sharding import AbstractMesh, Mesh, NamedSharding, PartitionSpec as P

from dew._coordination import from_every_process
from dew._devices import device_memory, pool_share, pool_statement
from dew._records import JSON
from dew.batch import Batch
from dew.data.dataset import rows_of
from dew.distributed.host import snapshot
from dew.distributed.placement import Layout, MeshSpec, Placement, explicit_axes, placed_bytes
from dew.mesh import LayoutRefused
from dew.nn.kernels.options import OuterProgram, compiler_options, owned_program, program_digest
from dew.nn.kernels.templates import report
from dew.objectives.base import RECORD, Objective, Pooled
from dew.state import TrainState, apart, together
from dew.telemetry.compiles import EVENTS, Compiles
from dew.telemetry.instrumentation import compiled_memory
from dew.telemetry.profile import Profiler, active_profile
from dew.training.ahead import handed
from dew.training.distributed import batch_divisor
from dew.training.interval import latest
from dew.training.memory import (
    Miss,
    climbed,
    compiled_if_it_fits,
    head_fused_of,
    head_tile_of,
    local_devices,
    missed,
    offloads,
    placeable,
    placed_beside,
    recompute_more,
    recompute_record,
    remat_half,
    remat_offloaded,
    rung_record,
    step_fits,
    temporaries_factor,
)
from dew.training.model_flops import TOKENS, TRAINING_FLOPS
from dew.training.profiling import PGLE_CAPTURE

if TYPE_CHECKING:
    from dew.training._lowering import KernelReport
    from dew.training.ahead import AheadOfTime
    from dew.training.rollout import Rollout
    from dew.training.trainer import StepFn

_log = logging.getLogger(__name__)

def read_whole(accumulation: int, split: int | None, step: Callable[..., StepFn] | None, layout: Layout,
               rollout: Rollout | None, mesh: MeshSpec | Mesh) -> str | None:
    """Why every step of a trainer reads its batch whole, or None where a plan may split it (`splits`):
    a custom `step`, a layout that keeps the variables on the host, a `rollout` or an Explicit mesh
    axis. Refuses the arguments no Trainer takes together."""
    if accumulation < 1:
        raise ValueError(f"accumulation must be at least 1, got {accumulation}")
    # Split under an Explicit axis, a step met a cotangent the split typed
    # float32[16@data,16,16] against its forward's float32[16,16,16].
    whole = ("a custom step owns its execution" if step is not None else
             "the layout keeps the variables on the host" if "variables" in layout.host else
             "a rollout writes a batch of its own" if rollout is not None else
             "an Explicit mesh axis types the step's rows" if explicit_axes(mesh) else None)
    if split is not None and (split < 1 or (whole is not None and split > 1)):
        raise ValueError(f"split reads each step's batch in that many microbatches, at least 1, and "
                         f"only 1 where {whole}; got {split}")
    if step is not None and "variables" in layout.host:
        raise ValueError(
            "Parameter-streamed training requires the trainer objective transaction; "
            "custom steps own their execution"
        )
    return whole


def splits(rows: Sequence[int] | None, batch: Batch, mesh: Mesh | AbstractMesh, microbatches: int,
           fixed: int | None) -> list[int]:
    """The splits a plan may read `batch` in on `mesh`, ascending: `fixed`
    where the run fixes one, else 1, the batch whole, and every count that
    cuts each batch the run reads, of `rows` (`batch`'s own where None), into
    microbatches whose rows the mesh's row shards divide (`batch_divisor`),
    every device holding whole rows of each; a batch of step-wide fields
    alone is read whole. A fixed split that leaves some device without whole
    rows is refused."""
    rows, divisor = rows or (rows_of(batch),), batch_divisor(mesh, microbatches)
    whole = [split for split in range(1, min(rows) // divisor + 1)
             if all(count % (split * divisor) == 0 for count in rows)]
    if fixed is None:
        return [1, *(split for split in whole if split > 1)]
    if fixed > 1 and fixed not in whole:
        raise LayoutRefused(
            f"this run reads each step's batch in {fixed} microbatches, and the batches it reads, of "
            f"{sorted(set(rows))} rows over {divisor} row shards of {dict(mesh.shape)}, cut into whole rows "
            f"on every device only in {whole}; read them in one of those, or leave the split to the plan")
    return [fixed]


@dataclasses.dataclass(frozen=True)
class Candidate:
    """A step a plan compiles: the shapes of the microbatch it reads, the
    loss's shapes, the narrow copies, the state's placement and shapes as it
    traces them, the bytes held beside it, the program lowered, None once its plan is admitted
    (`Planned.released`), and its kernel report (`dew.training._lowering.KernelReport`: its dispatches'
    routes and lowered launches)."""
    micro: Batch
    shapes: object
    copies: dict
    shardings: Placement[TrainState]
    prepared: TrainState
    held: int
    built: int
    """The bytes of `held` the step's arguments hold: the copies and accumulator its first call builds."""
    program: jax.stages.Lowered | None
    kernels: KernelReport
    sampled: Callable[[], jax.stages.Lowered] | None = None
    """The same step summarizing what the run inspects at its commit, lowered on call
    (`dew.training.inspection`); None in a run that inspects nothing."""
    spent: Mapping[str, float] = dataclasses.field(default_factory=dict)
    """The seconds JAX recorded of its step's work so far, by kind (`dew.telemetry.compiles.EVENTS`: traced,
    lowered, compiled), its lowering's (`FitSearch.lowering`), which its probe adds its compile to
    (`Probe.spent`)."""
    spread: tuple[str, ...] = ()
    """The mesh axes a projection of its step spread over as it traced (`dew.nn.sharding.Link.spread`), which
    a step kept apart from its trace (`dew.training._candidates.candidate_for`) sets again on the links of
    the trace that reads it."""
    @property
    def lowered(self) -> jax.stages.Lowered:
        """The program, which a plan's search and its guided compile read before it is released."""
        assert self.program is not None, "a plan's program is released once the plan is admitted"
        return self.program


class Relowered:
    """What a compile lowered afresh where a kernel spilled (`report.compiled`'s `relower`): each
    candidate `lower` gives, the last the one the executable compiled from, its lowering and kernel
    report the executable's."""

    def __init__(self, lower: Callable[[], Candidate]):
        self.lower, self.last = lower, None

    def __call__(self) -> jax.stages.Lowered:
        self.last = self.lower()
        return self.last.lowered

    def candidate(self, compiled: Candidate) -> Candidate:
        """The candidate the executable compiled from: `compiled`, the one first lowered, where the
        compile lowered none afresh."""
        return compiled if self.last is None else self.last


class SampledStep(NamedTuple):
    """A step that also summarizes what the run inspects (`Candidate.sampled`): the program it compiled
    from (`dew.nn.kernels.options.program_digest`) and its executable."""

    program: str
    executable: jax.stages.Compiled


def whole_step_program[L, E](objective: Objective[L, E]) -> OuterProgram:
    """`whole_step` over `objective`'s statistics (`Objective.reduce_loss`), the one program a split step
    runs on its microbatches' outputs, compiled under its towers' options, which a run compiled ahead
    compiles with its step (`dew.training.ahead`)."""
    return owned_program(functools.partial(whole_step, reduce=objective.reduce_loss,
                                           shown=frozenset(objective.shown)), *objective.towers())


def whole_step[L](outputs, reduce: Callable[[L], tuple[jax.Array, jax.Array]],
                  shown: frozenset[str] = frozenset()):
    """A step's outputs from those of the microbatches it was read in: its loss as an unsplit step logs it,
    whatever the split (docs/design/training.md, the logged loss): where its last microbatch commits, the
    update's, pooled over the window; elsewhere its own batch's, its microbatches' own statistics pooled
    (`Committed.own`) and reduced by `reduce` (`Objective.reduce_loss`); its metrics pooled by their own
    statistics (`_pooled`), a metric the objective shows (`shown`, which a run selects by) refused where it
    states none; finite and accepted where every microbatch was; and the commits' counts,
    tokens and norms summed, its inspection the latest part's that committed (`interval.latest`), as a
    run's interval keeps the latest sampled commit: a step of two commits (a window a rejection left part
    filled, then the closing part of a run's last batch) reports its second's summaries, never the two
    added. A pure function, which a step compiles as a program of its own (`dew.training.compiler`) or
    traces within its own (`dew.training.ahead`)."""
    losses, metrics, finite, accepted, committed = zip(*outputs, strict=True)
    report = {name: _pooled(name, [part[name] for part in metrics]) for name in metrics[-1]}
    if len(metrics) > 1:
        # A metric that states no pooling is one microbatch's, named so (`Aux.metrics`); a shown one, which
        # a run selects its best and its plateau by under its name, is refused.
        unpooled = sorted(name for name in shown & set(report) if not isinstance(report[name], Pooled))
        if unpooled:
            raise ValueError(f"the objective shows {', '.join(unpooled)}, which state no pooling over the "
                             "microbatches a step is read in; state each as a Ratio or Moments "
                             "(`Aux.metrics`), or read the step whole (split=1)")
        report = {(name if name.startswith(RECORD) or name in (TRAINING_FLOPS, TOKENS)
                   or isinstance(value, Pooled) else f"{name}/last_microbatch"): value
                  for name, value in report.items()}
    loss = losses[-1]
    if committed[-1].own is not None:
        owns = [part.own for part in committed]
        # Pooled as a window pools them (`Transaction.prepare_attempt`): a loss stated as one scalar as the
        # mean of its equal microbatches', statistics by their sums.
        scalar = all(isinstance(part, jax.Array) and part.ndim == 0 for part in owns)
        own = reduce(jax.tree.map(lambda *values: sum(values), *owns))[0]
        own = own / len(owns) if scalar else own
        loss = jnp.where(committed[-1].count > 0, loss, own.astype(loss.dtype))
    summed = jax.tree.map(lambda *values: sum(values),
                          *(dataclasses.replace(part, inspected=None, own=None) for part in committed))
    sample = None
    for part in committed:
        if part.inspected is not None:
            sample = (part.count, part.inspected) if sample is None else latest(
                (part.count, part.inspected), sample)
    # A pooled statistic leaves the step as its value, the number a caller reads (`Aux.metrics`).
    report = {name: value.mean()[0] if isinstance(value, Pooled) else value for name, value in report.items()}
    return (loss, report, jnp.all(jnp.stack(finite)), jnp.all(jnp.stack(accepted)),
            dataclasses.replace(summed, inspected=None if sample is None else sample[1]))


def _pooled(name: str, values: list):
    """A step's metric `name` from its microbatches' `values`: a recording step's evaluations every
    microbatch's in the order they ran (`RECORD`); a `Pooled` statistic by its own pooling
    (`Pooled.pooled`); any other its last microbatch's (`Aux.metrics`), the model FLOPs and tokens among
    them, which every microbatch counts of the whole batch it is read from (`step_counts`)."""
    if name.startswith(RECORD):
        return jnp.concatenate(values)
    if isinstance(values[-1], Pooled):
        return Pooled.pooled(values)
    return values[-1]


TRACING_COLLECTIONS = 50_000
"""The allocations between collections of Python's youngest generation while a plan search runs
(`held_objects`), past CPython's 700: its traces allocate millions of short-lived objects."""

_HELD = 0
"""The blocks of `held_objects` open, across nesting and threads."""

_HELD_GUARD = threading.Lock()

_THRESHOLDS: list[int] = []
"""Python's collection thresholds as the first open block of `held_objects` found them."""

@contextlib.contextmanager
def held_objects() -> Iterator[None]:
    """Within the block, the youngest generation of Python's collector collects every
    `TRACING_COLLECTIONS` allocations: a plan search's traces, compiles and bookkeeping make millions of
    short-lived objects, and each full collection their collections led to walked the long-lived ones
    beside them (row 2440: 59 full collections paused a Qwen3.5-4B step's compile_ahead 73 s of 342,
    FS2440-GC-PLAIN). The process's permanent generation is no part of it: a freeze is the process's own
    to make and undo (one made before Dew was imported reads as CPython 3.12's own immortal objects do, which
    each collection returns to it), so the block never freezes nor unfreezes. Nested and concurrent blocks
    share one threshold; the last to leave restores the process's."""
    global _HELD
    with _HELD_GUARD:
        if not _HELD:
            _THRESHOLDS[:] = gc.get_threshold()
            gc.set_threshold(TRACING_COLLECTIONS, *_THRESHOLDS[1:])
        _HELD += 1
    try:
        yield
    finally:
        with _HELD_GUARD:
            _HELD -= 1
            if not _HELD:
                gc.set_threshold(*_THRESHOLDS)


def planned(candidates: Sequence[int], first: int, fits: Callable[[int], bool],
            climb: Callable[[], bool], after: Callable[[int], int | None]) -> int | None:
    """The split a step reads its batch in, at the rung `climb` leaves the
    objective on, or None where no candidate fits.

    A candidate is a rung of the objective's ladder, which `climb` moves one
    up, and one of `candidates`, ascending: the batch read in that many equal
    microbatches. The best is the first that fits in rung-major order, the
    least recomputation and then the most rows a microbatch: a rung that
    recomputes the block costs every token (9-18% on Qwen3.5-0.8B at
    3,072-token rows, A100), where a smaller microbatch at the same rung
    costs little. A half rung, which recomputes only a few named values,
    costs little too, and `fits` tries it at each split it asks
    (`FitSearch`), so it is no rung here.

    `fits(split)`, a compile at the current rung, is asked of
    `candidates[first]` first (the batch whole, or a recorded plan, those
    before it taken as missed); past a miss, of the split the compile that
    missed computes as the first that fits (`after`, an index into
    `candidates`, from what it read: `Miss`), and so on from each compile
    that misses, upward, until one fits; where `after` names none, the
    objective climbs a rung and asks `candidates[first]` again. XLA fuses and
    schedules each split's program again, so its compile is the judge both
    ways: a miss there computes the next, and a split that fits past one not
    asked is asked of the split before it, down to one that misses, as a
    split coarser than one that misses misses too. A tight start is two
    compiles where the split computed is the next.
    """
    index: int | None = first
    while True:
        missed = first - 1
        while index is not None:
            if fits(candidates[index]):
                while index - 1 > missed and fits(candidates[index - 1]):
                    index -= 1
                return candidates[index]
            missed, index = index, after(index)
        if not climb():
            return None
        index = first


@dataclasses.dataclass(frozen=True)
class Probe:
    """One compile of a plan's search (`planned`): the split, the rows of
    its microbatch and the rung (`rung_record`) it compiled at, what
    the fit check read, this process's tightest headroom and XLA's
    temporaries and arguments and what the step places beside its temporaries (`beside`), None where none
    were read."""
    split: int
    rows: int
    rung: Mapping[str, JSON]
    fits: bool
    headroom: int | None
    temp: int | None
    argument: int | None
    ahead: bool = False
    """Whether the executable was compiled before this search, not by it: the step compiled ahead
    (`FitSearch.ahead`), or one a search before it compiled (`FitSearch.compiled`)."""
    compile_seconds: float = 0.0
    """Time spent compiling or loading this program, apart from its shape trace and lowering."""
    spent: Mapping[str, float] = dataclasses.field(default_factory=dict)
    """The seconds JAX recorded of the candidate's work, its traces, lowerings and backend compile by kind
    (`Candidate.spent` and its compile's), which the compiled-step record and the perf gate read."""
    beside: int | None = None
    """The bytes the fit check placed beside the step's temporaries (`placed_beside`): its outputs that
    write over no donated argument and what the loop holds outside the step (copies, accumulators, a state
    not placed yet). None where none were read."""

    @classmethod
    def read(cls, split: int, candidate: Candidate, rung: Mapping[str, JSON], fit: bool,
             headroom: int | None, executable: jax.stages.Compiled | None, held: int) -> Probe:
        """The compile of `candidate` at `rung`, its memory as XLA reads it where it compiled, `held` the
        bytes its fit check held beside it (`FitSearch.fitting`)."""
        stats = None if executable is None else compiled_memory(executable)
        beside = None if stats is None else placed_beside(stats, held)
        return cls(split, rows_of(candidate.micro), rung, fit, headroom,
                   None if stats is None else stats.temp, None if stats is None else stats.argument,
                   beside=beside)

    def record(self) -> dict[str, JSON]:
        return {"split": self.split, "rows": self.rows, **self.rung, "fits": self.fits,
                "headroom": self.headroom, "temp": self.temp, "argument": self.argument,
                "ahead": self.ahead,
                "compile_seconds": self.compile_seconds,
                "beside": self.beside, "spent": dict(self.spent)}

    def named(self) -> str:
        tile = ("" if self.rung["head_tile"] is None else ", the head tiled" if self.rung["head_fused"]
                else ", the head tiled, its gradient in the backward")
        defaults = ", XLA's default options" if self.rung["xla_defaults"] else ""
        return f"{self.split} x {self.rows} rows at remat {self.rung['remat']!r}{tile}{defaults}"


def unplanned(probes: Sequence[Probe]) -> str:
    """Why no candidate fits: every candidate the plan compiled, and the
    closest, the most headroom, with its numbers; where none read any, the
    last, which XLA refused."""
    gib = 1024 ** 3
    known = [probe for probe in probes if probe.headroom is not None]
    closest = max(known, key=lambda probe: probe.headroom or 0) if known else probes[-1]
    if closest.headroom is None or closest.temp is None or closest.argument is None:
        said = f"the last, {closest.named()}, XLA refused for memory"
    else:
        placed = "" if closest.beside is None else (
            f" and {closest.beside / gib:.2f} GiB more beside them (outputs no donated argument holds, and "
            "what the loop keeps outside the step)")
        said = (f"the closest, {closest.named()}, needs {closest.temp / gib:.2f} GiB of temporaries"
                f"{placed}, with {closest.argument / gib:.2f} GiB of arguments in use, "
                f"{-closest.headroom / gib:.2f} GiB more than its tightest device has")
    return (f"the training step fits no plan of its batch: {said}. Tried "
            f"{'; '.join(probe.named() for probe in probes)}. Shard the state over more devices "
            "(fsdp), train fewer parameters or shorter rows, or use a device with more memory")


class PlanRefused(RuntimeError):
    """No candidate fits. `probes` holds every program the search compiled, including refusals, so
    an offline gate reads failed admission evidence as objects rather than parsing the message, and
    `allocators` what each local device's allocator stated as the search refused (`device_memory`): its
    limit, its bytes in use, its largest free block and its pool, which the closest's headroom read."""

    def __init__(self, probes: Sequence[Probe], allocators: Sequence[Mapping[str, int]] = ()):
        self.probes, self.allocators = tuple(probes), tuple(allocators)
        gib = 1024 ** 3
        read = [memory for memory in self.allocators if 'bytes_limit' in memory and 'bytes_in_use' in memory]
        said = ""
        if read:
            tightest = min(read, key=placeable)
            names = ('bytes_limit', 'bytes_in_use', 'largest_free_block_bytes', 'pool_bytes')
            said = " Its tightest device's allocator: " + ", ".join(
                f"{name} {tightest[name] / gib:.2f} GiB" for name in names if name in tightest) + "."
        super().__init__(unplanned(self.probes) + said)


@dataclasses.dataclass(frozen=True)
class Planned:
    """A plan `FitSearch` found: the step reading its batch in `split`
    microbatches, its executable and the options it compiled under, whether it keeps XLA's default
    options and reads the parameters through narrow copies, and the rung it compiled
    at, as a record and as the objective's modules, head tile and whether that
    head is fused. `profiled` is the runs and the bytes of the profile that guided its executable
    (`guided`), None where none did."""
    split: int
    candidate: Candidate
    executable: jax.stages.Compiled
    options: jax.stages.CompilerOptions | None
    xla_defaults: bool
    narrow: bool
    rung: Mapping[str, JSON]
    modules: tuple
    tile: tuple[int, int] | None
    fused: bool
    profiled: tuple[int, int] | None = None
    sampled: SampledStep | None = None
    """The step that also summarizes what the run inspects (`Candidate.sampled`), compiled under the
    plan's options and admitted with it (`FitSearch.measured`); None in a run that inspects nothing."""

    def record(self) -> dict[str, JSON]:
        record: dict[str, JSON] = {"split": self.split, "rows": rows_of(self.candidate.micro), **self.rung}
        if self.profiled is not None:
            runs, profile = self.profiled
            record["pgle"] = {"runs": runs, "bytes": profile}
        return record

    def parts(self, mesh: Mesh) -> list[jax.Array]:
        """The microbatch index of each of the step's `split` calls, on every device of `mesh`."""
        return [jax.device_put(np.int32(part), NamedSharding(mesh, P())) for part in range(self.split)]

    def guided(self, profile: bytes, runs: int, mesh: Mesh, relower: Callable[[], Candidate]) -> Planned:
        """This plan with its program compiled again under the options its executable compiled under,
        guided by `profile` of `runs` runs (`Trials.profiled`) as JAX's jit guides AutoPGLE's (XLA's
        feedback-directed `fdo_profile`, which JAX shares across a pool's processes under AutoPGLE), and
        held to the fit check the plan passed: a schedule the profile guides may keep more alive at once,
        and one that no longer fits is refused. It compiles through the kernel templates' reports
        (`report.compiled`), the plan's step lowered afresh where a kernel spills (`relower`, `Relowered`).
        An empty profile, of a backend XLA profiles none of, leaves the executable as it is. The plan
        given up holds the executable it replaces, which goes with it and its program (`released`)."""
        if not profile:
            return dataclasses.replace(self, profiled=(runs, 0))
        relowered = Relowered(relower)
        executable = report.compiled(self.candidate.lowered, {**(self.options or {}), "fdo_profile": profile},
                                     relower=relowered)
        candidate = relowered.candidate(self.candidate)
        fit, headroom = step_fits(executable, mesh, candidate.held)
        if not fit:
            raise RuntimeError(f"the step guided by its profile (jax_enable_pgle) needs {-(headroom or 0)} "
                               "bytes more than its tightest device has, where unguided it fits; train it "
                               "without AutoPGLE")
        return dataclasses.replace(self, candidate=candidate, executable=executable,
                                   profiled=(runs, len(profile)))

    def released(self) -> Planned:
        """This plan as the run keeps it once admitted: its executable without the program it was lowered
        from. JAX keeps each executable compiled from a lowering while the lowering lives (its in-memory
        compilation cache, keyed by the lowered module), so releasing it lets every executable the search
        or a guided compile made of it go, the one the run steps alone held."""
        return dataclasses.replace(self, candidate=dataclasses.replace(self.candidate, program=None))


def released(trees) -> None:
    """Delete every buffer of `trees`, which nothing reads again: what a step lets go of before the next is
    planned or built, so the devices hold none of it beside what replaces it. An array already deleted
    stays so (`Array.delete` returns at once)."""
    for leaf in jax.tree.leaves(trees):
        leaf.delete()


@dataclasses.dataclass
class Trials:
    """Trial runs of a plan's steps on `batch` from `state`, each on a copy of the state as it stood
    (`run`), used to profile the admitted plan under AutoPGLE (`profiled`). A state the
    trials own (`owned`, `fit`'s) is stepped by the first trial
    and deleted, each later trial stepping a fresh copy from the host, so the devices hold one state at a
    time, and `state` is then a copy as it stood, for the run to go on from. One they borrow (a compile
    given none, `Trainer.compile`) is never stepped and stays `state`: each trial steps a copy beside it,
    freed as the trial ends, for which the devices must hold room (`profiled`)."""
    state: TrainState
    batch: Batch
    owned: bool = True
    restored: Callable[[], TrainState] | None = None
    """The moving state as it stood, copied to the host at the first trial (`snapshot`)."""

    def run[R](self, trial: Callable[[TrainState], R]) -> R:
        """`trial` on a copy of the state as it stood, which it may step and donate."""
        # The frozen tree is no step's to change or donate, and the objective's held tree shares it: it
        # is kept as it is, and the moving state alone is copied, freed and restored.
        moving, frozen = apart(self.state)
        if self.restored is None:
            self.restored = snapshot(moving)
        stepped = moving if self.owned else self.restored()
        outcome = trial(together(stepped, frozen))
        # What the trial did not donate of the state it stepped is freed too, so nothing holds a
        # reference to a state handed to a trial, on any backend.
        released(stepped)
        if self.owned:
            self.state = together(self.restored(), frozen)
        return outcome

    def profiled(self, step: Callable, planned: Planned, mesh: Mesh, runs: int) -> bytes:
        """XLA's profile of `runs` runs of `planned`'s executable, each over the microbatches of `batch`
        and waited on, as JAX's AutoPGLE waits on the runs it profiles
        (`jax.experimental.profiler.get_profiled_instructions_proto`), on a trial, which `step` steps once
        first, untraced, so what a step builds at its first call (its window, its narrow copies) is built
        and the trace holds the executable alone; empty on a backend XLA profiles none of. A process holds
        one profiling session at a time, so a capture of the caller's own running (`dew.Profiler`) is
        refused (`PGLE_CAPTURE`), as a fit's window is under AutoPGLE. Borrowed trials first hold the
        admitted step to room for the copy they step beside the caller's state, refused where there is
        none."""
        if active_profile() is not None:
            raise RuntimeError(PGLE_CAPTURE)
        if not self.owned:
            # The copy a borrowed trial steps sits beside the caller's state, which the plan admitted alone.
            copy = placed_bytes(apart(self.state)[0])
            fit, headroom = step_fits(planned.executable, mesh, planned.candidate.held + copy)
            if not fit:
                raise RuntimeError(
                    "AutoPGLE (jax_enable_pgle) profiles a step compiled with no trials on a copy of the "
                    f"state beside the caller's, which needs {-(headroom or 0)} bytes more than its tightest "
                    "device has; compile it with Trials(state, batch), which steps the state itself as fit "
                    "does, and step on from trials.state")

        def profile(state: TrainState) -> bytes:
            stepped, *_ = step(state, self.batch)
            current, held = apart(stepped)
            parts = planned.parts(mesh)
            with tempfile.TemporaryDirectory() as directory:
                # XLA reads the trace alone, so the capture writes no XProf reports.
                with Profiler(directory, reports=False) as profiler:
                    for _ in range(runs):
                        for part in parts:
                            current, *_ = planned.executable(current, held, self.batch, part)
                        jax.block_until_ready(current)
                (capture,) = profiler.captures
                (session,) = capture.glob("plugins/profile/*")
                return get_profiled_instructions_proto(str(session))

        return self.run(profile)


def compiled_key(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None) -> tuple[str, str]:
    """A search's key of the executable it compiles from `lowered` under `options` (`FitSearch.compiled`):
    the program's digest (`program_digest`), one for a lowering over a state's shapes before it is placed
    and over the placed state, and the options as text."""
    return program_digest(lowered), repr(sorted((options or {}).items()))


def settled(planned: Planned, probes: Sequence[Probe], search: FitSearch,
            frontier: Sequence[Probe] | None = None) -> dict[str, JSON]:
    """The admitted plan, its kernel report, every compile needed to find it, and the kernels those compiles
    found spilling and the launch
    configs refused where it searched (`FitSearch.kernels`), which the run's telemetry reports with it
    (`StepCompiled`); planned ahead with its `frontier`, each rung above the plan that fits at fewer
    microbatches (`FitSearch.above`). There is no alternative-policy timing or selection in the record."""
    _log.info("the step's %s", planned.candidate.kernels.summary())
    above: dict[str, JSON] = {} if frontier is None else {"frontier": [probe.record() for probe in frontier]}
    return {**planned.record(), "kernels": planned.candidate.kernels.record(),
            "probes": [probe.record() for probe in probes], **search.kernels,
            "pool": pool_record(local_devices(search.mesh)), **above}


def pool_record(devices: Sequence) -> dict[str, JSON]:
    """The allocator's limit a plan was made against, its tightest device's (`device_memory`), with the pool's
    share and its source (`dew._devices.pool_statement`): a plan held to JAX's 75% of the device, its
    backend opened before Dew sized the pool, is visible as such. Devices this process does not attach
    plan against the memory stated for them (`compile_only`), at the share their loading run takes."""
    limits = [limit for device in devices if (limit := device_memory(device).get("bytes_limit")) is not None]
    attached = set(jax.local_devices())
    stated = {"share": pool_share(), "source": "compile_only"} if not attached.issuperset(devices) else (
        pool_statement())
    return {"limit": min(limits, default=None), **stated}


def refusal_record(identity: str, config: Hashable, names: frozenset[str]) -> dict[str, JSON]:
    """A launch config refused for its kernels' spills (`report.refused_launches`), the config by its
    text."""
    kernels: list[JSON] = [*sorted(names)]
    return {"launch": identity, "config": repr(config), "kernels": kernels}


def rewrite_record(made) -> dict[str, JSON]:
    """A region the algebra pass rewrote as a compile traced its program (`dew.algebra.program.AlgebraReport`,
    `report.rewrites`): its equations, the sets admitted and the matches each fired, its iterations, nodes and
    limits, its price before and after, its seconds, the roots its splice took, and whether an earlier
    window of its structure derived it."""
    fired: dict[str, JSON] = dict(made.fired)
    limits: list[JSON] = list(made.limits)
    admitted: list[JSON] = list(made.admitted)
    return {"equations": made.equations, "admitted": admitted, "fired": fired, "iterations": made.iterations,
            "nodes": made.nodes, "limits": limits, "limited": made.limited, "before": made.before,
            "after": made.after, "seconds": made.seconds, "spliced": made.spliced, "reused": made.reused}


def spill_record(miss: report.Miss) -> dict[str, JSON]:
    """A kernel a compile found spilling, or a launch whose program's block XLA refused (`report.Miss`), its
    launch's config by its text."""
    return {"kernel": miss.kernel, "op": miss.op,
            "config": None if miss.config is None else repr(miss.config), "registers": miss.registers,
            "frame": miss.frame, "shared": miss.shared, "stepped": miss.stepped}


class Lower(Protocol):
    """A search's step at a split and whether it keeps its narrow copies, at the objective's rung, lowered
    (`FitSearch.lower`)."""

    def __call__(self, split: int, narrow: bool) -> Candidate: ...


class _Spent(Compiles):
    """Every program JAX traces, lowers or compiles while it is open, each kind of its events held."""

    COUNTED = frozenset(EVENTS.values())


@contextlib.contextmanager
def spending(candidate: Candidate | None = None) -> Iterator[dict[str, float]]:
    """The seconds JAX records of the work within the block, by kind (`EVENTS`: traced, lowered, compiled),
    added to `candidate`'s (`Candidate.spent`), in the dict it yields once the block ends."""
    held = dict({} if candidate is None else candidate.spent)
    with _Spent(None) as compiles:
        yield held
    for made in compiles.programs:
        held[made.kind] = held.get(made.kind, 0.0) + made.seconds


@dataclasses.dataclass
class FitSearch[LossT, EffectsT]:
    """A step's search over the fit ladder (`planned`): the first of the run's split candidates (`splits`)
    and the objective's rungs to fit the devices, each lowered by `lower` at the objective's rung and
    `narrow` once, compiled under its `options` or XLA's defaults on the half rung above, the head at the
    run's tile `base` at each remat rung (`recompute_more`). It holds what it lowered, compiled and
    measured, and where on the ladder it stands.

    The first rung climbed drops the narrow copies, and a plan above it keeps them dropped: taking them
    back would compile another plan once one fits. A split missing at a rung with a half rung
    (`remat_half`) is compiled there, the plan staying where it fits; climbing passes
    every half rung asked (`onward`). Each compile that misses names the split asked next (`after`);
    `resumed`, a checkpoint's rung, is warned of where it misses; where nothing fits it raises with the
    closest candidate's numbers. The objective is left at the last plan's rung.

    The ladder holds no host offload. XLA:GPU holds the optimizer state and the EMA whole on the device
    for the update wherever they rest between steps, so keeping them on the host frees nothing at a
    step's peak (a decoder's step on an A100 peaked at 8.427 GB either way, node receipt
    TL-a100-offload-1338-c); a run keeps them there by its layout (`Layout.host`) alone. A remat policy
    that keeps its residuals in pinned host memory ('minimal_offloaded') keeps as many on the device as
    'full' does (`tests/test_remat_rungs.py`' `_saved`: a decoder's 107,720 bytes both, 'save_dot_except_mlp'
    132,296), so it fits nowhere 'full' does not, and a run takes it by its own remat."""
    lower: Lower
    options: Callable[[Candidate], jax.stages.CompilerOptions | None]
    objective: Objective[LossT, EffectsT]
    mesh: Mesh
    xla_defaults: bool
    narrow: bool
    base: tuple[int, int] | None
    resumed: JSON
    probes: list[Probe] = dataclasses.field(default_factory=list)
    refusals: list[RuntimeError] = dataclasses.field(default_factory=list)
    lowered: dict[tuple[int, str], Candidate] = dataclasses.field(default_factory=dict)
    halves: set[str] = dataclasses.field(default_factory=set)
    """The remat rungs (`recompute_record`) the search has asked as half rungs."""
    gather: Callable[[object], list] = from_every_process
    """How every process's figure is read (`from_every_process`), so each asks the same split."""
    misses: dict[tuple[str, int], Miss] = dataclasses.field(default_factory=dict)
    """What each compile that did not fit read, by its rung and split (`Miss`), which names the split the
    search asks next (`after`)."""
    chosen: jax.stages.CompilerOptions | None = None
    fitted: tuple[Candidate, jax.stages.Compiled, jax.stages.CompilerOptions | None] | None = None
    """The plan the search last found to fit: its candidate, executable and the options it compiled under."""
    sampled: SampledStep | None = None
    """The last measured candidate's sampled step (`measured`)."""
    fitted_sampled: SampledStep | None = None
    """The fitted plan's sampled step, which summarizes what the run inspects (`Candidate.sampled`)."""
    ahead: AheadOfTime | None = None
    """A step compiled ahead (`dew.training.ahead`), run in place of the search's first compile where it
    is that program (`AheadOfTime.taken`), lowered as its own compile lowered it (`handed`: within its
    refusals, for its GPU, `fits_here`)."""
    compiled: dict[tuple[str, str], jax.stages.Compiled] = dataclasses.field(default_factory=dict)
    """Each executable this search, or one before it over the same candidates, compiled, by its program and
    its options (`compiled_key`): one a search before the state was placed compiled is taken
    here in place of compiling it again (`dew.training.compiler.StepCompiler.planned_unplaced`), judged on
    these devices as one compiled here is."""
    unplaced: int | None = None
    """The bytes a device holds of the state the search plans over that the devices hold none of yet, None
    where they hold it (`dew.training.compiler.StepCompiler.planned_unplaced`): each candidate is judged with
    them beside it, which its step's first call needs as it needs the copies and accumulators it builds,
    on the devices' free bytes in all (`memory.available`), not on their allocators' largest block, which
    placing the state reshapes (a GH200's fresh 4B run, its 52.56 GiB of state read beside the step in the
    one 38.88 GiB block its pool could give of 70.88 GiB free, 22.41 GiB short, fits). A candidate holds
    none of them (`Candidate.held`), so one cached for a
    placed state serves an unplaced one."""
    kernels: dict[str, JSON] = dataclasses.field(default_factory=dict)
    """What the templates' reports found where the search compiled (`admitted`): each kernel its compiles
    found spilling (`report.misses`, `spill_record`), each launch config they refused
    (`report.refused_launches`), each region the algebra pass read (`report.rewrites`) and each traced
    program's whole pass in seconds (`report.passes`), none the registry held before it searched."""

    def __post_init__(self):
        self.factor = temporaries_factor(self.mesh)
        self.started = self.remat_rung()
        record = recompute_record(self.objective)
        # Whether any module the run trains sets its own remat, which it compiles at alone.
        self.remat_set = any(entry is not None for entry in (record if isinstance(record, list)
                                                             else [record]))
        # Every rung keeps, XLA's defaults' included, the options the modules need to compute right, as the
        # step's own (`options`) are made; the target's own policy is the compile point's
        # (`report.targeted`).
        self.required = compiler_options(*self.objective.towers()) or None

    def admitted(self, candidates: Sequence[int], first: int) -> tuple[Planned, tuple[Probe, ...]]:
        """The first fitting plan the split-and-rung search finds (`planned`), and every compile it made:
        each candidate compiled and judged by XLA's memory analysis of it on the devices' allocators
        (`step_fits`), the ladder walked from its first rung, each compile that misses naming the split
        asked next (`after`). Once admission succeeds, no optional alternate policy is compiled."""
        seen, held, traced = len(report.misses()), set(report.refused_launches()), len(report.rewrites())
        passed = len(report.passes())
        split = planned(candidates, first, self.fits, self.climb, functools.partial(self.after, candidates))
        spills: list[JSON] = [spill_record(miss) for miss in report.misses()[seen:]]
        refused: list[JSON] = [refusal_record(*refusal) for refusal in report.refused_launches()
                               if refusal not in held]
        rewrites: list[JSON] = [rewrite_record(made) for made in report.rewrites()[traced:]]
        # Each program the search traced, its whole algebra pass in seconds (`report.passes`), beside the
        # regions' own: what the pass cost this plan's compiles.
        passes: list[JSON] = list(report.passes()[passed:])
        self.kernels = {"spills": spills, "refused": refused, "rewrites": rewrites, "passes": passes}
        if split is None or self.fitted is None:
            allocators = [device_memory(device) for device in local_devices(self.mesh)]
            raise PlanRefused(self.probes, allocators) from (self.refusals[-1] if self.refusals else None)
        return self.found(split), tuple(self.probes)

    def above(self, candidates: Sequence[int], split: int) -> tuple[Probe, ...]:
        """Past the admitted plan at `split`, each rung above it that fits at fewer of `candidates` than every
        plan before it, at the fewest that fit there, compiled: the plans a run's speed is chosen among
        offline (`dew.training.ahead.compile_ahead`'s frontier). A rung at as many microbatches as one below
        it recomputes more for none fewer, which no speed favours, so it is not compiled; a remat rung takes
        the run's own head again (`recompute_more`), so a rung may fit at fewer than the one before it, and
        the search reads the ladder to its end, or until a rung fits at the fewest. Where the ladder's top
        fits, the plan or a rung found (its head policies after it moving `current` on), its sibling that
        offloads the residuals it would recompute (`remat_offloaded`) is compiled at the last top that fit,
        its modules and head policy put back first: as many device bytes, another speed (row 1932). The
        objective is left at the last rung read, the run's to restore."""
        # A frontier rung that misses is no checkpoint's rung missing (`fits_here`).
        self.resumed = None
        fewer, found = list(candidates[:candidates.index(split)]), []
        top = (split, self.configuration()) if offloads(self.objective) else None
        while fewer and self.onward(lambda: recompute_more(self.objective, self.base)):
            asked = len(self.probes)
            fewest = planned(fewer, 0, self.fits_here, lambda: False, functools.partial(self.after, fewer))
            if fewest is not None:
                found.append(next(probe for probe in self.probes[asked:]
                                  if probe.fits and probe.split == fewest))
                fewer = fewer[:fewer.index(fewest)]
                if offloads(self.objective):
                    top = (fewest, self.configuration())
        if top is not None:
            self.restore(top[1])
            if remat_offloaded(self.objective) and self.fits_here(top[0]):
                found.append(self.probes[-1])
        return tuple(found)

    def configuration(self) -> tuple[tuple[nn.Module, ...], tuple[int, int] | None, bool]:
        """The objective's modules and its head's tile and fusion as they stand, which `restore` puts back."""
        return (tuple(entry.module for entry in self.objective.program_key()), head_tile_of(self.objective),
                head_fused_of(self.objective))

    def restore(self, configuration: tuple[tuple[nn.Module, ...], tuple[int, int] | None, bool]) -> None:
        """Put the objective's modules and head policy back as `configuration` holds them."""
        modules, tile, fused = configuration
        self.objective.substitute(list(modules))
        self.objective.restore_head(tile, fused=fused)

    def fitting(self, executable: jax.stages.Compiled | None,
                candidate: Candidate) -> tuple[bool, int | None]:
        """`step_fits` of `executable`, `candidate`'s compiled step or its sampled twin, with the state's
        `unplaced` bytes beside it where the devices hold none of it yet."""
        unplaced = self.unplaced or 0
        return step_fits(executable, self.mesh, self.held(candidate), candidate.built + unplaced,
                         placed=self.unplaced is None)

    def held(self, candidate: Candidate) -> int:
        """The bytes the fit check holds beside `candidate`'s step: its own (`Candidate.held`), and the
        state's `unplaced` bytes where the devices hold none of it yet."""
        return candidate.held + (self.unplaced or 0)

    def lowering(self, split: int) -> Candidate:
        """The step at `split` at the objective's rung and the search's `narrow`, lowered."""
        with spending() as spent:
            lowered = self.lower(split, self.narrow)
        return dataclasses.replace(lowered, spent=spent)

    def relowering(self, split: int) -> Candidate:
        """The step at `split` lowered afresh past a kernel that spilled (`Relowered`)."""
        return self.lower(split, self.narrow)

    def after(self, candidates: Sequence[int], index: int) -> int | None:
        """The index into `candidates` of the split asked after the compile of `candidates[index]` at the
        objective's rung missed: the first above it that compile computes fits (`Miss.fits_at`), the
        latest any process computes so every process asks the same; the next above it where the compile
        read no memory (an executable XLA refused for memory as it compiled); the last where none is
        computed to fit, XLA's compile the judge of the finest before the rung climbs; None past the
        last, the rung's to climb."""
        if index + 1 >= len(candidates):
            return None
        miss = self.misses.get((self.rung(), candidates[index]))
        if miss is None:
            return index + 1
        above = range(index + 1, len(candidates))
        computed = next((at for at in above if miss.fits_at(candidates[at])), len(candidates) - 1)
        return max(self.gather(computed))

    def current(self) -> dict[str, JSON]:
        return rung_record(self.objective, self.xla_defaults, narrow=self.narrow)

    def rung(self) -> str:
        return json.dumps(self.current(), sort_keys=True)

    def remat_rung(self) -> str:
        return json.dumps(recompute_record(self.objective), sort_keys=True)

    def onward(self, step: Callable[[], bool]) -> bool:
        """`step` the objective up its ladder past every half rung the search has asked, each missed."""
        moved = step()
        while moved and self.remat_rung() in self.halves:
            moved = step()
        return moved

    def fits(self, split: int) -> bool:
        """Whether the step at `split` fits, at the objective's rung or its half rung (`remat_half`), a
        compile asked at each. A run that states its remat has no half rung: its remat is its own."""
        if self.fits_here(split):
            return True
        held = None if self.remat_set else remat_half(self.objective)
        if held is None:
            return False
        self.halves.add(self.remat_rung())
        if self.fits_here(split):
            return True
        self.objective.substitute(held)
        return False

    def compiled_options(self, candidate: Candidate) -> jax.stages.CompilerOptions | None:
        """The options the search compiles `candidate` under: `required` on the half rung of XLA's defaults,
        else the step's own (`options`), so a step whose options are its towers' own is that rung's program
        (`climb`); the compile point adds the target's own (`report.targeted`)."""
        return self.required if self.xla_defaults else self.options(candidate)

    def measured(self, split: int, candidate: Candidate) -> tuple[
            jax.stages.CompilerOptions | None, Candidate, jax.stages.Compiled | None, Probe]:
        """`candidate`, the step at `split` at the objective's rung, compiled under the rung's options
        where it fits the devices, or for the search's first compile the step compiled ahead
        (`ahead`); the candidate its executable compiled from, lowered afresh where a kernel spilled
        (`compiled_if_it_fits`, `Relowered`); and the measure of its fit (`Probe`), the search's state
        else untouched. A run that inspects compiles the candidate's sampled step too
        (`Candidate.sampled`), at once with the step (`report.attempts`), or takes the one compiled ahead
        with the step (`AheadOfTime.sampled`), which
        must fit as the step must, so a rung fits only where both do; it is `sampled` until the next
        measure."""
        options = self.compiled_options(candidate)
        taken, ahead = self.ahead is not None, self.ahead
        with spending(candidate) as spent:
            began = time.perf_counter()
            early, firsts, relowered = None, [], None
            key = compiled_key(candidate.lowered, options)
            kept = self.compiled.get(key)
            if ahead is not None:
                executable, self.ahead = ahead.taken(candidate.lowered), None
            elif candidate.sampled is None and kept is not None:
                executable, taken = kept, True
            else:
                # A run that inspects compiles its sampled step beside the step, at once (`report.attempts`),
                # its compile kept only where the step fits, as the step's own is kept first.
                early = None if candidate.sampled is None else candidate.sampled()
                firsts = report.attempts([(candidate.lowered, options)]
                                         + ([] if early is None else [(early, options)]))
                relowered = Relowered(lambda: self.relowering(split))
                executable = compiled_if_it_fits(candidate.lowered, options, self.refusals, relower=relowered,
                                                 first=firsts[0])
                candidate = relowered.candidate(candidate)
                if executable is not None:
                    # Under the program it compiled: the relowered one past a spill, which a plan after it
                    # lowers directly, the spill's refusal held for the process.
                    self.compiled[compiled_key(candidate.lowered, options)] = executable
            fit, headroom = self.fitting(executable, candidate)
            self.sampled = None
            if fit and candidate.sampled is not None:
                if ahead is not None and ahead.sampled is not None:
                    # The sampled step compiled ahead with the step, lowered as its compile lowered it
                    # (`handed`), which its target compiles none of.
                    with handed(ahead):
                        sampled = candidate.sampled()
                    self.sampled = SampledStep(ahead.sampled.program,
                                               ahead.sampled.taken(sampled, jax.devices()))
                else:
                    # The one compiled beside the step's, but where the step was lowered afresh past a spill,
                    # which its sampled step is then too, as one lowered after the step's compile would be.
                    beside = early if relowered is not None and relowered.last is None else None
                    lowered = [candidate.sampled() if beside is None else beside]

                    def resampled() -> jax.stages.Lowered:
                        """The sampled step lowered afresh, its launches derived past a spilled one's
                        config."""
                        again = self.relowering(split).sampled
                        assert again is not None
                        lowered.append(again())
                        return lowered[-1]
                    compiled = compiled_if_it_fits(lowered[0], options, self.refusals, relower=resampled,
                                                   first=None if beside is None else firsts[1])
                    self.sampled = (None if compiled is None
                                    else SampledStep(program_digest(lowered[-1]), compiled))
                # Admitted by the plan whichever compiled it, ahead or here (training.md: the sampled step is
                # admitted by the same memory plan).
                fit, headroom = self.fitting(None if self.sampled is None else self.sampled.executable,
                                             candidate)
        seconds = time.perf_counter() - began
        probe = Probe.read(split, candidate, self.current(), fit, headroom, executable, self.held(candidate))
        probe = dataclasses.replace(probe, ahead=taken, compile_seconds=seconds, spent=spent)
        return options, candidate, executable, probe

    def fits_here(self, split: int) -> bool:
        key = (split, self.rung())
        if key not in self.lowered:
            # A step compiled ahead, its first compile, is lowered as its own compile lowered it (`handed`):
            # within its refusals, for its GPU; every other compile within this process's, for this GPU.
            with handed(self.ahead):
                self.lowered[key] = self.lowering(split)
        self.chosen, candidate, executable, probe = self.measured(split, self.lowered[key])
        self.lowered[key] = candidate
        fit, headroom = probe.fits, probe.headroom
        self.probes.append(probe)
        if not fit and headroom is not None:
            self.misses[(self.rung(), split)] = missed(executable, split, headroom, self.factor)
        if fit:
            assert executable is not None
            self.fitted, self.fitted_sampled = (candidate, executable, self.chosen), self.sampled
            return True
        if self.resumed is not None:
            _log.warning("the step does not fit the devices at the rung its checkpoint trained on (%s); from "
                         "here the resumed run computes otherwise than the run it continues", self.resumed)
            self.resumed = None
        return False


    def told(self, admitted: Planned) -> None:
        """Tell a run that kept the remat it set itself how to learn what else fits: a run compiles its
        own rung alone (row 2111), and planning it ahead at each policy, with each GPU's memory stated
        (`--trainer.compile-ahead.capacity-gib`), compiles the others offline."""
        remat = admitted.rung["remat"]
        if self.remat_set and json.dumps(remat, sort_keys=True) == self.started:
            _log.info("the run sets remat %r and compiles at it alone; --trainer.compile-ahead with "
                      "--trainer.compile-ahead.capacity-gib plans it on its GPUs offline, at each remat "
                      "policy asked, to tell what else fits", remat)

    def climb(self) -> bool:
        # The first rung drops the narrow copies, two bytes a parameter held all run for 1-2% of a step.
        if self.narrow and any(candidate.copies for candidate in self.lowered.values()):
            self.narrow = False
            self.lowered.clear()
            return True
        # XLA's default options, but `required`, are the half rung above the step's own where those
        # are another program: the Triton GEMM fusions they keep can take fewer temporaries (sm89).
        if not self.xla_defaults and self.chosen != self.required:
            self.xla_defaults = True
            return True
        self.xla_defaults = False
        self.lowered.clear()
        return self.onward(self.compiled_again)

    def compiled_again(self) -> bool:
        """Climb the objective one rung, as `recompute_more` does, saying it compiles the step again
        there: a run that states its remat climbs its head's rungs alone (`climbed` without its remat)."""
        moved = climbed(self.objective, self.base, remat=not self.remat_set)
        if moved is not None:
            _log.warning("the step does not fit the devices %s", moved)
        return moved is not None

    def found(self, split: int) -> Planned:
        assert self.fitted is not None
        modules = tuple(entry.module for entry in self.objective.program_key())
        return Planned(split, *self.fitted, self.xla_defaults, self.narrow, self.current(), modules,
                       head_tile_of(self.objective), head_fused_of(self.objective),
                       sampled=self.fitted_sampled)
