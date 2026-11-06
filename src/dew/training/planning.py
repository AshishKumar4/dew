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
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.experimental.profiler import get_profiled_instructions_proto
from jax.sharding import AbstractMesh, Mesh, NamedSharding, PartitionSpec as P

from dew._coordination import from_every_process
from dew._records import JSON
from dew.batch import Batch
from dew.data.dataset import rows_of
from dew.distributed.host import snapshot
from dew.distributed.placement import Layout, MeshSpec, Placement, explicit_axes, placed_bytes
from dew.mesh import LayoutRefused
from dew.nn.kernels.options import compiler_options, program
from dew.nn.kernels.templates import report
from dew.objectives.base import RECORD, Objective
from dew.state import TrainState, apart, together
from dew.telemetry.instrumentation import compiled_memory
from dew.telemetry.profile import Profiler, active_profile
from dew.training.ahead import handed, program_of
from dew.training.distributed import batch_divisor
from dew.training.footprint import Footprint
from dew.training.memory import (
    climbed,
    compiled_if_it_fits,
    derived_headroom,
    deviceless_options,
    head_fused_of,
    head_tile_of,
    judged,
    local_devices,
    offloads,
    pooled_headroom,
    recompute_more,
    recompute_record,
    remat_half,
    remat_offloaded,
    rung_record,
    step_fits,
    temporaries_factor,
)
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
    (`Planned.released`), its kernel launches (`dew.nn.kernels.tuning.tune`) and its kernel report
    (`dew.training._lowering.KernelReport`: its dispatches' routes and lowered launches)."""
    micro: Batch
    shapes: object
    copies: dict
    shardings: Placement[TrainState]
    prepared: TrainState
    held: int
    built: int
    """The bytes of `held` the step's arguments hold: the copies and accumulator its first call builds."""
    program: jax.stages.Lowered | None
    launches: str
    kernels: KernelReport
    footprint: Footprint | None
    """Its derived device footprint; None for a custom step not differentiating the objective."""
    sampled: Callable[[], jax.stages.Lowered] | None = None
    """The same step summarizing what the run inspects at its commit, lowered on call
    (`dew.training.inspection`); None in a run that inspects nothing."""

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
    from (`dew.training.ahead.program_of`) and its executable."""

    program: str
    executable: jax.stages.Compiled


@program
def whole_step(outputs):
    """A step's outputs from those of the microbatches it was read in: the
    last one's loss, pooled over the window where it commits, and its
    report, as each step of an accumulating run reports its own
    microbatch's; finite and accepted where every microbatch was; and the
    commits summed."""
    losses, metrics, finite, accepted, committed = zip(*outputs, strict=True)
    # A recording step's evaluations are every microbatch's, in the order they ran (`RECORD`).
    report = {name: jnp.concatenate([part[name] for part in metrics]) if name.startswith(RECORD) else value
              for name, value in metrics[-1].items()}
    return (losses[-1], report, jnp.all(jnp.stack(finite)), jnp.all(jnp.stack(accepted)),
            jax.tree.map(lambda *values: sum(values), *committed))


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


def judged_likely(headroom: int | None, under: int, *, forecast: bool) -> bool:
    """Whether a step is likely to fit: by the `headroom` the derivation judged, past `under`, the most a
    compile at its remat needed beyond its derivation, where it judged one; by the compiles' `forecast`
    where it judged none. The derivation counts what a split changes beside the activations (the
    accumulated gradient a split of two or more holds), which a forecast from a compile's temporaries
    does not, so where it judges it orders the search alone (row 3341: Qwen3.5-4B at 4 x 2048 on an H100
    compiled two microbatches the derivation judged 3.83 GiB short, on the forecast's word)."""
    if headroom is None:
        return forecast
    return headroom - under >= 0


def planned(candidates: Sequence[int], first: int, fits: Callable[[int], bool],
            climb: Callable[[], bool], likely: Callable[[int], bool] = lambda split: False,
            halved: Callable[[int], bool] | None = None) -> int | None:
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
    (`FitSearch`), so it is no rung here. Memory falls
    along both, so `fits(split)`, a compile at the current rung, is asked of
    `candidates[first]` first (the batch whole, or a recorded plan), then of the
    largest split at each rung until one fits, and that rung's smallest
    fitting split is found by halving the range between: the first compile,
    one a rung tried, and at most ceil(log2(len(candidates))) more. Memory is
    assumed to fall with the split; a state that replays its microbatches
    (statistics that are no `Ratio`) holds a slot of each, mutable
    collections included, which grows with it.
    MaxText's estimator searches batch sizes the same way, one policy at a
    time (src/maxtext/utils/estimator.py, `largest_batch_size`, at
    10073768f); it plans offline and returns the frontier, where a run
    takes its first point.

    Past a miss, the first split `likely` to fit by what the compiles
    measured (`Forecast`) or the derivation judged is asked in place of the
    largest, and where it misses, the next above it; the halving below a
    split that fits asks `halved`, which may rule a compile out
    (`FitSearch.refused`: Qwen3.5-4B at 4 x 2048 on an H100, two
    microbatches, judged 3.83 GiB short for their accumulated gradient,
    compiled a third time on main, row 3341). Replayed through it, the six compiles an A100-40 asked for
    Qwen3.5-0.8B's packed 8 x 3072 step are five (no largest split), the
    plan the same.
    """
    if fits(candidates[first]):
        return candidates[first]
    low, top = first, len(candidates) - 1
    while True:
        while low < top:
            index = next((index for index in range(low + 1, top) if likely(candidates[index])), top)
            if fits(candidates[index]):
                high = index
                while high - low > 1:
                    middle = (low + high) // 2
                    low, high = (low, middle) if (halved or fits)(candidates[middle]) else (middle, high)
                return candidates[high]
            low = index
        if not climb():
            return None
        low = -1


@dataclasses.dataclass(frozen=True)
class Probe:
    """One compile of a plan's search (`planned`): the split, the rows of
    its microbatch and the rung (`rung_record`) it compiled at, what
    the fit check read, this process's tightest headroom and XLA's
    temporaries and arguments, None where none were read, and what the
    candidate's shapes derived it would take (`Footprint`)."""
    split: int
    rows: int
    rung: Mapping[str, JSON]
    fits: bool
    headroom: int | None
    temp: int | None
    argument: int | None
    derived: Footprint | None
    ahead: bool = False
    """Whether the executable is the step compiled ahead (`FitSearch.ahead`), not the search's own."""
    compile_seconds: float = 0.0
    """Time spent compiling or loading this program, apart from its shape trace and lowering."""
    derived_headroom: int | None = None
    """The headroom the derivation judged the candidate would leave (`derived_headroom`), None where it
    judged none: beside `headroom`, which the compile measured, its error."""

    @classmethod
    def read(cls, split: int, candidate: Candidate, rung: Mapping[str, JSON], fit: bool,
             headroom: int | None, executable: jax.stages.Compiled | None) -> Probe:
        """The compile of `candidate` at `rung`, its memory as XLA reads it where it compiled."""
        stats = None if executable is None else compiled_memory(executable)
        return cls(split, rows_of(candidate.micro), rung, fit, headroom,
                   None if stats is None else stats.temp, None if stats is None else stats.argument,
                   candidate.footprint)

    def record(self) -> dict[str, JSON]:
        return {"split": self.split, "rows": self.rows, **self.rung, "fits": self.fits,
                "headroom": self.headroom, "temp": self.temp, "argument": self.argument,
                "derived": None if self.derived is None else self.derived.record(), "ahead": self.ahead,
                "compile_seconds": self.compile_seconds, "derived_headroom": self.derived_headroom}

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
        said = (f"the closest, {closest.named()}, needs {closest.temp / gib:.2f} GiB of temporaries beside "
                f"{closest.argument / gib:.2f} GiB of arguments, {-closest.headroom / gib:.2f} GiB more than "
                "its tightest device has")
    return (f"the training step fits no plan of its batch: {said}. Tried "
            f"{'; '.join(probe.named() for probe in probes)}. Shard the state over more devices "
            "(fsdp), train fewer parameters or shorter rows, or use a device with more memory")


class PlanRefused(RuntimeError):
    """No candidate fits. `probes` holds every program the search compiled, including refusals, so
    an offline gate reads failed admission evidence as objects rather than parsing the message."""

    def __init__(self, probes: Sequence[Probe]):
        self.probes = tuple(probes)
        super().__init__(unplanned(self.probes))


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
        for leaf in jax.tree.leaves(stepped):
            if not leaf.is_deleted():
                leaf.delete()
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


@dataclasses.dataclass
class Forecast:
    """What a plan's compiles measured, which orders its search (`planned`'s `likely`): each rung's
    (`rung_record`) compiles, as the split and every process's XLA temporaries, headroom and
    temporaries factor (`temporaries_factor`), None where it read none, gathered once a compile
    (`gather`, `from_every_process`); and the half rung each rung where a split missed moved to, None
    where it had none."""
    gather: Callable[[object], list] = from_every_process
    measured: dict[str, list[tuple[int, list]]] = dataclasses.field(default_factory=dict)
    half_of: dict[str, str | None] = dataclasses.field(default_factory=dict)

    def read(self, key: str, split: int, executable: jax.stages.Compiled | None, headroom: int | None,
             factor: int) -> None:
        """Keep what the compile of `split` at rung `key` measured on every process; every process
        calls this at every compile."""
        stats = None if executable is None else compiled_memory(executable)
        local = (None if stats is None or headroom is None
                 else (stats.temp, headroom, factor))
        self.measured.setdefault(key, []).append((split, self.gather(local)))

    def likely(self, key: str, split: int) -> bool:
        """Whether `split` is guessed to fit on every process that measured, at `key` or the half rung
        a miss there moved to: from each process's two compiles nearest above it, its headroom taken
        as a line in a microbatch's rows, or from one compile, its temporaries as all per row; no guess
        where no compile above it read memory. A guess only orders the search: it rules nothing out."""
        rungs = [key, *filter(None, [self.half_of.get(key)])]
        return any(guess is not None and guess >= 0 for guess in (self.guess(rung, split) for rung in rungs))

    def guess(self, key: str, split: int) -> int | None:
        """The least headroom a process is guessed to have at `split` and rung `key`, None where none
        measured above it."""
        compiles = [(asked, gathered) for asked, gathered in self.measured.get(key, ()) if asked < split]
        guesses = []
        for process in range(len(compiles[0][1]) if compiles else 0):
            points = sorted((asked, gathered[process]) for asked, gathered in compiles
                            if gathered[process] is not None)
            if len(points) == 1:
                (asked, (temporaries, headroom, factor)), = points
                guesses.append(headroom + factor * (temporaries - temporaries * asked // split))
            elif points:
                (near, (_, low, _)), (nearer, (_, high, _)) = points[-2:]
                slope = (high - low) / (1 / nearer - 1 / near)
                guesses.append(round(high + slope * (1 / split - 1 / nearer)))
        return min(guesses, default=None)


def settled(planned: Planned, probes: Sequence[Probe], search: FitSearch,
            frontier: Sequence[Probe] | None = None) -> dict[str, JSON]:
    """The admitted plan, its kernel report, every compile needed to find it, each rung the derivation
    read before them (`FitSearch.derived`), and the kernels those compiles found spilling and the launch
    configs refused where it searched (`FitSearch.kernels`), which the run's telemetry reports with it
    (`StepCompiled`); planned ahead with its `frontier`, each rung above the plan that fits at fewer
    microbatches (`FitSearch.above`). There is no alternative-policy timing or selection in the record."""
    _log.info("the step's %s", planned.candidate.kernels.summary())
    above: dict[str, JSON] = {} if frontier is None else {"frontier": [probe.record() for probe in frontier]}
    return {**planned.record(), "kernels": planned.candidate.kernels.record(),
            "probes": [probe.record() for probe in probes], "derived": list(search.derived), **search.kernels,
            **above}


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
    every half rung asked (`onward`). Each compile's measure (`Forecast`) orders the splits; `resumed`, a
    checkpoint's rung, is warned of where it misses; where nothing fits it raises with the closest
    candidate's numbers. The objective is left at the last plan's rung.

    The ladder holds no host offload. XLA:GPU holds the optimizer state and the EMA whole on the device
    for the update wherever they rest between steps, so keeping them on the host frees nothing at a
    step's peak (a decoder's step on an A100 peaked at 8.427 GB either way, node receipt
    TL-a100-offload-1338-c); a run keeps them there by its layout (`Layout.host`) alone. A remat policy
    that keeps its residuals in pinned host memory ('minimal_offloaded') keeps as many on the device as
    'full' does (`tests/test_remat_rungs.py`' `_saved`: a decoder's 107,720 bytes both, 'save_dot_except_mlp'
    132,296), so it fits nowhere 'full' does not, and a run takes it by its own remat."""
    lower: Callable[[int, bool, Footprint | None], Candidate]
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
    forecast: Forecast = dataclasses.field(default_factory=Forecast)
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
    capacity: int | None = None
    """The bytes a device that states none holds (`step_headroom`): a compile-only GPU's, as a step
    compiled ahead for it states them (`AheadTarget.capacity`)."""
    derive: Callable[[int, bool], Candidate] | None = None
    """The step at a split, at the objective's rung and `narrow`, traced and not lowered, for its derived
    footprint (`dew.training.footprint`), which picks the search's first rung (`derived_start`); None
    for a run that starts from a plan, recorded, compiled ahead or resumed, which the search tries first."""
    derived: list[dict[str, JSON]] = dataclasses.field(default_factory=list)
    """Each rung the derivation read before the first compile: its rung, terms, headroom and label."""
    judged_at: dict[tuple[str, int], int | None] = dataclasses.field(default_factory=dict)
    """The derived headroom of each candidate the derivation read, by its rung (`rung`) and split: what
    orders the compiles (`likely`), a compile's measure correcting it (`under`)."""
    copies_at: dict[tuple[str, int], bool] = dataclasses.field(default_factory=dict)
    """Whether each candidate the derivation read keeps narrow copies (`Candidate.copies`)."""
    footprints: dict[tuple[str, int], Footprint | None] = dataclasses.field(default_factory=dict)
    """The footprint of each candidate the derivation read, which its lowering keeps (`lowering`): traced
    again there, it was 22 of a Qwen3.5-4B step's 185 seconds of `compile_ahead` (row 4261). A spill's
    refusal, which changes the configs its kernels take, clears it (`measured`)."""
    under: dict[str, int] = dataclasses.field(default_factory=dict)
    """By remat rung (`remat_rung`), the most bytes a compile there needed past its derived headroom: a
    candidate there is judged to fit only where its derived headroom holds that much more, in proportion
    to its derived temporaries (`missed_at`, `shortfall`)."""
    missed_at: dict[str, int] = dataclasses.field(default_factory=dict)
    """By remat rung, the derived temporaries of the compile that set its `under`: what the derivation
    misses grows with what it derives, a step's activations with its microbatch, so a split of half the
    rows misses about half as much (a 64-token test step on a CPU: 963 KB short at one microbatch, 0.6 MB
    to spare at two, which a shortfall taken whole judged short and compiled four)."""
    overstated: set[str] = dataclasses.field(default_factory=set)
    """The remat rungs (`remat_rung`) a compile fit at that the derivation judged short: its refusals
    there rule no compile out (`refused`)."""
    kernels: dict[str, JSON] = dataclasses.field(default_factory=dict)
    """What the templates' reports found where the search compiled (`admitted`): each kernel its compiles
    found spilling (`report.misses`, `spill_record`) and each launch config they refused
    (`report.refused_launches`), none the registry held before it searched."""

    def __post_init__(self):
        self.factor = temporaries_factor(self.mesh, self.capacity)
        self.started = self.remat_rung()
        record = recompute_record(self.objective)
        # Whether any module the run trains sets its own remat, which it compiles at alone.
        self.remat_set = any(entry is not None for entry in (record if isinstance(record, list)
                                                             else [record]))
        # Every rung keeps, XLA's defaults' included, the options the modules need to compute right, as the
        # target compiles them (`deviceless_options`), as the step's own (`options`) are made.
        self.required = deviceless_options(self.mesh, compiler_options(*self.objective.towers()) or None)

    def admitted(self, candidates: Sequence[int], first: int) -> tuple[Planned, tuple[Probe, ...]]:
        """The first fitting plan found by the split-and-rung search, and every compile it made, its
        first rung the one the derivation picks (`derived_start`). Once admission succeeds, no optional
        alternate policy is compiled."""
        # A rung fits where its most microbatches do; the derivation then finds its split there, which
        # alone is compiled where it is right (`derived_split`).
        self.derived_start(candidates[-1])
        first = self.derived_split(candidates, first)
        seen, held, traced = len(report.misses()), set(report.refused_launches()), len(report.rewrites())
        split = planned(candidates, first, self.fits, self.climb, self.likely,
                        functools.partial(self.fits, pruned=True))
        spills: list[JSON] = [spill_record(miss) for miss in report.misses()[seen:]]
        refused: list[JSON] = [refusal_record(*refusal) for refusal in report.refused_launches()
                               if refusal not in held]
        rewrites: list[JSON] = [rewrite_record(made) for made in report.rewrites()[traced:]]
        self.kernels = {"spills": spills, "refused": refused, "rewrites": rewrites}
        if split is None or self.fitted is None:
            raise PlanRefused(self.probes) from (self.refusals[-1] if self.refusals else None)
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
            fewest = planned(fewer, 0, self.fits_here, lambda: False,
                             lambda split: self.forecast.likely(self.rung(), split))
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

    def derived_start(self, split: int) -> None:
        """Move the objective to the least-recompute rung whose derived footprint at `split`, the run's
        most microbatches, fits the devices (`derived_headroom`, labelled 'fits'), before anything
        compiles, so the search's first compile confirms it and a miss climbs on as a correction; the
        rungs read are `derived`. A rung the derivation cannot judge ('unknown': residuals over several
        devices) or devices that state no memory to judge one against ('unstated') end the walk where it
        started, at the base rung, as does a ladder no rung of which fits. A run that sets its remat
        compiles at it alone; its split is read there (`derived_split`). A 'fits' is the derivation's
        judgment, which the one compile confirms."""
        if self.derive is None or self.resumed is not None or self.remat_set:
            return
        base = self.configuration()
        label, narrow = self.walked(split)
        if label != "fits":
            self.restore(base)
            narrow = self.narrow
        elif self.remat_rung() != self.started:
            _log.info("the derivation says the step fits from remat %s; compiling there first",
                      self.remat_rung())
        self.narrow = narrow

    def walked(self, split: int) -> tuple[str, bool]:
        """Climb the objective from its rung to the least-recompute one whose derived footprint at `split`
        fits the devices, each rung read appended to `derived` (`judged`): the label the walk ends on, and
        whether the rung keeps the narrow copies. It ends where a rung fits, where the derivation cannot
        judge one ("unknown") and where the ladder ends; where no device states memory to judge a rung
        against (`judging`), it reads none ("unstated")."""
        assert self.derive is not None
        narrow, label = self.narrow, "refuses"
        if not self.judging():
            # A device that states no memory leaves every rung unjudged (`admitted`), so no footprint is
            # traced for it.
            self.derived.append({**self.current(), "narrow": narrow, "split": split, "headroom": None,
                                 "label": "unstated", "terms": None})
            return "unstated", narrow
        while True:
            headroom = self.judged(split, narrow)
            label = "unknown" if headroom is None else "fits" if headroom >= 0 else "refuses"
            if label != "refuses":
                break
            # The search's own climb drops the narrow copies first, at the same head and remat.
            if narrow and self.copies_at[self.candidate_key(split, narrow)]:
                narrow = False
                continue
            narrow = False
            if not self.onward(lambda: recompute_more(self.objective, self.base)):
                break
        return label, narrow

    def judged(self, split: int, narrow: bool) -> int | None:
        """The derived headroom of the step at `split` and `narrow` at the objective's rung, traced once a
        candidate (`judged_at`), and recorded beside the rungs `walked` reads; None where the derivation
        cannot judge it or no device states memory to judge it against. Every process reads it alike
        (`memory.pooled_headroom`), so each compiles the plan the others do."""
        key = self.candidate_key(split, narrow)
        if key not in self.judged_at:
            assert self.derive is not None
            candidate = self.derive(split, narrow)
            footprint = candidate.footprint
            headroom = pooled_headroom(None if footprint is None else derived_headroom(
                footprint, local_devices(self.mesh), self.capacity, candidate.built))
            label = "unknown" if headroom is None else "fits" if headroom >= 0 else "refuses"
            self.derived.append({**self.current(), "narrow": narrow, "split": split, "headroom": headroom,
                                 "label": label, "terms": None if footprint is None else footprint.record()})
            self.judged_at[key], self.copies_at[key] = headroom, bool(candidate.copies)
            self.footprints[key] = footprint
        return self.judged_at[key]

    def lowering(self, split: int) -> Candidate:
        """The step at `split` at the objective's rung and the search's `narrow`, lowered, with the
        footprint the derivation traced of it where it read it (`footprints`)."""
        return self.lower(split, self.narrow, self.footprints.get(self.candidate_key(split, self.narrow)))

    def relowering(self, split: int) -> Candidate:
        """The step at `split` lowered afresh past a kernel that spilled (`Relowered`), its footprint traced
        again: the kernel's next config can change the buffers the derivation traced at its first (a
        Triton attention backward's partial gradients, a block of queries each)."""
        return self.lower(split, self.narrow, None)

    def candidate_key(self, split: int, narrow: bool) -> tuple[str, int]:
        """The step at `split` and `narrow` at the objective's rung, as the derivation's judgments keep it."""
        return json.dumps({**self.current(), "narrow": narrow}, sort_keys=True), split

    def derived_split(self, candidates: Sequence[int], first: int) -> int:
        """The index of the candidate the search compiles first: at the rung the derivation picked, the
        fewest microbatches whose derived footprint fits the devices, found by halving the range below
        the most, which fits (`derived_start`, memory falling with the split as `planned` assumes); the
        derivation trusted where it refuses one, so the one compile at the most performant plan it admits
        decides (row 3341). `first` where the run starts from a plan (a recorded one, or a checkpoint's
        rung), where the derivation cannot judge the step, or where no candidate it reads fits."""
        if self.derive is None or self.resumed is not None or first != 0:
            return first
        if not self.judging():
            if not self.derived:
                # A run that sets its remat reads no rung's walk (`derived_start`), so its record says here
                # what a walk would, and its log what it asked for and is not told: nothing is judged, and
                # the compiles decide. A run that leaves its remat to the walk asked nothing of it.
                self.derived.append({**self.current(), "narrow": self.narrow, "split": candidates[-1],
                                     "headroom": None, "label": "unstated", "terms": None})
                _log.warning("the run sets remat %s, and its devices state no memory to judge its footprint "
                             "against (a CPU, or a compile-only GPU given no capacity), so the derivation "
                             "picks no plan and the compiles decide it", self.remat_rung())
            return first
        headroom = self.judged(candidates[-1], self.narrow)
        if headroom is None or headroom < 0:
            return first
        low, high = -1, len(candidates) - 1
        while high - low > 1:
            middle = (low + high) // 2
            judged = self.judged(candidates[middle], self.narrow)
            fits = judged is not None and judged >= 0
            low, high = (low, middle) if fits else (middle, high)
        return high

    def judgment(self, split: int) -> int | None:
        """The derivation's headroom of the step at `split` at the objective's rung, traced where the search
        has not judged it yet (a trace, where a compile it saves is a minute and more), None where it
        cannot judge."""
        if self.derive is not None and self.judging():
            return self.judged(split, self.narrow)
        return self.judged_at.get(self.candidate_key(split, self.narrow))

    def likely(self, split: int) -> bool:
        """Whether the step at `split` at the objective's rung is likely to fit, which orders the compiles
        past a miss (`planned`): as the derivation judges it, corrected by what the compiles measured;
        as the compiles' forecast guesses where the derivation cannot judge (`judged_likely`)."""
        return judged_likely(self.judgment(split), self.shortfall(split),
                             forecast=self.forecast.likely(self.rung(), split))

    def shortfall(self, split: int) -> int:
        """The bytes a step at `split` at the objective's remat is judged to need past its derived headroom
        (`under`): the most a compile there missed, scaled by the step's derived temporaries against that
        compile's (`missed_at`), or whole where either is not derived."""
        rung = self.remat_rung()
        under, missed = self.under.get(rung, 0), self.missed_at.get(rung)
        footprint = self.footprints.get(self.candidate_key(split, self.narrow))
        temporaries = None if footprint is None else footprint.temporaries
        if not under or not missed or temporaries is None:
            return under
        return under * temporaries // missed

    def learned(self, foreseen: int | None, measured: int | None, *, fit: bool, split: int,
                temporaries: int | None) -> None:
        """What a compile at the objective's remat at `split` teaches the derivation's next judgments there:
        the headroom it `measured` short of the one `foreseen` corrects them (`under`), in proportion to the
        derived `temporaries` it was short of (`missed_at`), and a compile that fit where the derivation, so
        corrected, judged it short rules its refusals out (`overstated`)."""
        rung = self.remat_rung()
        if foreseen is not None and fit and foreseen - self.shortfall(split) < 0:
            self.overstated.add(rung)
        if foreseen is not None and measured is not None and foreseen - measured > self.under.get(rung, 0):
            self.under[rung] = foreseen - measured
            if temporaries:
                self.missed_at[rung] = temporaries
            else:
                self.missed_at.pop(rung, None)

    def refused(self, split: int) -> bool:
        """Whether the derivation judges the step at `split` short past a miss, corrected by what a compile
        at the rung needed beyond it (`under`), which the halving below a split that fits does not compile
        (`planned`): no split it cannot judge, and none at a remat where a compile it judged short fit
        (`overstated`: a tiny model's derived peak, its fixed buffers, over-states the step many times)."""
        rung = self.remat_rung()
        if rung in self.overstated:
            return False
        derived = self.judgment(split)
        return derived is not None and derived - self.shortfall(split) < 0

    def judging(self) -> bool:
        """Whether every process's devices judge a derived footprint (`memory.judged`) at the search's
        capacity, each process answering alike: where one does not, every rung is unknown and no footprint
        is traced."""
        local = all(judged(device, self.capacity) for device in local_devices(self.mesh))
        return all(from_every_process(local))

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

    def fits(self, split: int, *, pruned: bool = False) -> bool:
        """Whether the step at `split` fits, at the objective's rung or its half rung (`remat_half`), a
        compile asked at each; `pruned`, a rung the derivation judges short asks none (`refused`), the
        half rung judged on its own."""
        if self.fits_here(split, pruned):
            return True
        missed, held = self.rung(), remat_half(self.objective)
        self.forecast.half_of[missed] = None if held is None else self.rung()
        if held is None:
            return False
        self.halves.add(self.remat_rung())
        if self.fits_here(split, pruned):
            return True
        self.objective.substitute(held)
        return False

    def compiled_options(self, candidate: Candidate) -> jax.stages.CompilerOptions | None:
        """The options the search compiles `candidate` under: `required` on the half rung of XLA's defaults,
        else the step's own (`options`), each under the compile-only policy (`deviceless_options`), which
        the search alone applies, so a step whose options are its towers' own is that rung's program
        (`climb`)."""
        return self.required if self.xla_defaults else deviceless_options(self.mesh, self.options(candidate))

    def measured(self, split: int, candidate: Candidate) -> tuple[
            jax.stages.CompilerOptions | None, Candidate, jax.stages.Compiled | None, Probe]:
        """`candidate`, the step at `split` at the objective's rung, compiled under the rung's options
        where it fits the devices, or for the search's first compile the step compiled ahead
        (`ahead`); the candidate its executable compiled from, lowered afresh where a kernel spilled
        (`compiled_if_it_fits`, `Relowered`); and the measure of its fit (`Probe`), the search's state
        else untouched. A run that inspects compiles the candidate's sampled step too
        (`Candidate.sampled`), or takes the one compiled ahead with the step (`AheadOfTime.sampled`), which
        must fit as the step must, so a rung fits only where both do; it is `sampled` until the next
        measure."""
        options = self.compiled_options(candidate)
        taken, ahead = self.ahead is not None, self.ahead
        began = time.perf_counter()
        if ahead is not None:
            executable, self.ahead = ahead.taken(candidate.lowered), None
        else:
            relowered = Relowered(lambda: self.relowering(split))
            executable = compiled_if_it_fits(candidate.lowered, options, self.refusals, relower=relowered)
            candidate = relowered.candidate(candidate)
            if relowered.last is not None:
                # A spill's refusal changes the configs every derivation traced: each is traced again.
                self.footprints.clear()
        fit, headroom = step_fits(executable, self.mesh, candidate.held, self.capacity, candidate.built)
        self.sampled = None
        if fit and candidate.sampled is not None and ahead is not None and ahead.sampled is not None:
            # The sampled step compiled ahead with the step, lowered as its compile lowered it (`handed`),
            # which its target compiles none of.
            with handed(ahead):
                sampled = candidate.sampled()
            self.sampled = SampledStep(ahead.sampled.program, ahead.sampled.taken(sampled, jax.devices()))
        elif fit and candidate.sampled is not None:
            lowered = [candidate.sampled()]

            def resampled() -> jax.stages.Lowered:
                """The sampled step lowered afresh, its launches derived past a spilled one's config."""
                again = self.lowering(split).sampled
                assert again is not None
                lowered.append(again())
                return lowered[-1]
            compiled = compiled_if_it_fits(lowered[0], options, self.refusals, relower=resampled)
            if len(lowered) > 1:
                self.footprints.clear()
            self.sampled = None if compiled is None else SampledStep(program_of(lowered[-1]), compiled)
            fit, headroom = step_fits(compiled, self.mesh, candidate.held, self.capacity, candidate.built)
        seconds = time.perf_counter() - began
        probe = Probe.read(split, candidate, self.current(), fit, headroom, executable)
        foreseen = pooled_headroom(None if candidate.footprint is None else derived_headroom(
            candidate.footprint, local_devices(self.mesh), self.capacity, candidate.built))
        probe = dataclasses.replace(probe, ahead=taken, compile_seconds=seconds, derived_headroom=foreseen)
        self.learned(foreseen, pooled_headroom(headroom), fit=fit, split=split,
                     temporaries=None if candidate.footprint is None else candidate.footprint.temporaries)
        if foreseen is not None and (foreseen >= 0) != fit:
            # A derived peak bounded from below (`Footprint.exact`) admits provisionally: its miss names
            # the need the compile found, which a sourced rule is to reach.
            footprint = candidate.footprint
            # The derived temporaries are what admission judges (`derived_headroom`): the gradient's peak,
            # or the staged commit's fields beside the gradients where they hold more.
            _log.warning("a model miss: the derivation judged %s %s with %d bytes to spare%s, and its "
                         "compile %s with %s, its temporaries %s bytes against derived temporaries of %s "
                         "(its peak %s, its staged commit %s; derived terms %s); the compile decides, and "
                         "the derivation is to fix",
                         probe.named(), "fits" if foreseen >= 0 else "does not fit", foreseen,
                         "" if footprint is None or footprint.exact else " (its peak a bound from below)",
                         "fits" if fit else "does not fit", headroom, probe.temp,
                         None if footprint is None else footprint.temporaries,
                         None if footprint is None else footprint.peak,
                         None if footprint is None else footprint.staged,
                         None if probe.derived is None else probe.derived.record())
        return options, candidate, executable, probe

    def fits_here(self, split: int, pruned: bool = False) -> bool:
        if pruned and self.refused(split):
            return False
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
        self.forecast.read(self.rung(), split, executable, headroom, self.factor)
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
        there."""
        moved = climbed(self.objective, self.base)
        if moved is not None:
            _log.warning("the step does not fit the devices %s", moved)
        return moved is not None

    def found(self, split: int) -> Planned:
        assert self.fitted is not None
        modules = tuple(entry.module for entry in self.objective.program_key())
        return Planned(split, *self.fitted, self.xla_defaults, self.narrow, self.current(), modules,
                       head_tile_of(self.objective), head_fused_of(self.objective),
                       sampled=self.fitted_sampled)
