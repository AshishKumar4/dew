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

import dataclasses
import json
import logging
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.profiler import get_profiled_instructions_proto
from jax.sharding import AbstractMesh, Mesh, NamedSharding, PartitionSpec as P

from dew._coordination import from_every_process
from dew._records import JSON
from dew.batch import Batch
from dew.data.dataset import rows_of
from dew.distributed.host import snapshot
from dew.distributed.placement import Layout, MeshSpec, Placement, explicit_axes, placed_bytes
from dew.mesh import LayoutRefused
from dew.nn.kernels.options import compiler_options
from dew.objectives.base import RECORD, Objective
from dew.state import TrainState, apart, together
from dew.telemetry.benchmark import samples
from dew.telemetry.profile import active_profile, capture_options
from dew.training.distributed import batch_divisor
from dew.training.memory import (
    compiled_if_it_fits,
    head_fused_of,
    head_tile_of,
    recompute_more,
    recompute_record,
    remat_half,
    remat_more,
    rung_record,
    step_fits,
    temporaries_factor,
)
from dew.training.profiling import PGLE_CAPTURE

if TYPE_CHECKING:
    from dew.training._lowering import KernelReport
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
    program: jax.stages.Lowered | None
    launches: str
    kernels: KernelReport

    @property
    def lowered(self) -> jax.stages.Lowered:
        """The program, which a plan's search and its guided compile read before it is released."""
        assert self.program is not None, "a plan's program is released once the plan is admitted"
        return self.program


@jax.jit
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


def planned(candidates: Sequence[int], first: int, fits: Callable[[int], bool],
            climb: Callable[[], bool], likely: Callable[[int], bool] = lambda split: False) -> int | None:
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
    measured (`Forecast`) is asked in place of the largest, and where it
    misses, the next above it: a guess orders the compiles and never moves
    the plan. Replayed through it, the six compiles an A100-40 asked for
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
                    low, high = (low, middle) if fits(candidates[middle]) else (middle, high)
                return candidates[high]
            low = index
        if not climb():
            return None
        low = -1


@dataclasses.dataclass(frozen=True)
class Probe:
    """One compile of a plan's search (`planned`): the split, the rows of
    its microbatch and the rung (`rung_record`) it compiled at, and what
    the fit check read, this process's tightest headroom and XLA's
    temporaries and arguments, None where none were read."""
    split: int
    rows: int
    rung: Mapping[str, JSON]
    fits: bool
    headroom: int | None
    temp: int | None
    argument: int | None

    @classmethod
    def read(cls, split: int, candidate: Candidate, rung: Mapping[str, JSON], fit: bool,
             headroom: int | None, executable: jax.stages.Compiled | None) -> Probe:
        """The compile of `candidate` at `rung`, its memory as XLA reads it where it compiled."""
        stats = None if executable is None else executable.memory_analysis()
        return cls(split, rows_of(candidate.micro), rung, fit, headroom,
                   None if stats is None else int(stats.temp_size_in_bytes),
                   None if stats is None else int(stats.argument_size_in_bytes))

    def record(self) -> dict[str, JSON]:
        return {"split": self.split, "rows": self.rows, **self.rung, "fits": self.fits,
                "headroom": self.headroom, "temp": self.temp, "argument": self.argument}

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

    def record(self) -> dict[str, JSON]:
        record: dict[str, JSON] = {"split": self.split, "rows": rows_of(self.candidate.micro), **self.rung}
        if self.profiled is not None:
            runs, profile = self.profiled
            record["pgle"] = {"runs": runs, "bytes": profile}
        return record

    def parts(self, mesh: Mesh) -> list[jax.Array]:
        """The microbatch index of each of the step's `split` calls, on every device of `mesh`."""
        return [jax.device_put(np.int32(part), NamedSharding(mesh, P())) for part in range(self.split)]

    def guided(self, profile: bytes, runs: int, mesh: Mesh) -> Planned:
        """This plan with its program compiled again under the options its executable compiled under,
        guided by `profile` of `runs` runs (`Trials.profiled`) as JAX's jit guides AutoPGLE's (XLA's
        feedback-directed `fdo_profile`, which JAX shares across a pool's processes under AutoPGLE), and
        held to the fit check the plan passed: a schedule the profile guides may keep more alive at once,
        and one that no longer fits is refused. An empty profile, of a backend XLA profiles none of, leaves
        the executable as it is. The plan given up holds the executable it replaces, which goes with it
        and its program (`released`)."""
        if not profile:
            return dataclasses.replace(self, profiled=(runs, 0))
        executable = self.candidate.lowered.compile({**(self.options or {}), "fdo_profile": profile})
        fit, headroom = step_fits(executable, mesh, self.candidate.held)
        if not fit:
            raise RuntimeError(f"the step guided by its profile (jax_enable_pgle) needs {-(headroom or 0)} "
                               "bytes more than its tightest device has, where unguided it fits; train it "
                               "without AutoPGLE")
        return dataclasses.replace(self, executable=executable, profiled=(runs, len(profile)))

    def released(self) -> Planned:
        """This plan as the run keeps it once admitted: its executable without the program it was lowered
        from. JAX keeps each executable compiled from a lowering while the lowering lives (its in-memory
        compilation cache, keyed by the lowered module), so releasing it lets every executable the search
        or a guided compile made of it go, the one the run steps alone held."""
        return dataclasses.replace(self, candidate=dataclasses.replace(self.candidate, program=None))


TIMED_STEPS, TIMED_REPEATS, TIMED_WARMUP = 10, 3, 2
"""A frontier's plans are each timed `TIMED_REPEATS` times over `TIMED_STEPS` steps (`step_seconds`),
after the warmup ones, each plan's own: a split step's first call builds its window."""


def step_seconds(step: Callable, state: TrainState, batch: Batch) -> list[float]:
    """Seconds a step of `step` takes on `batch` from `state`, each of `TIMED_REPEATS` samples the mean
    of `TIMED_STEPS` queued and synchronized once after `TIMED_WARMUP`
    (`dew.telemetry.benchmark.samples`), each step consuming the state the one before it returned, as
    training does."""
    held = [state]

    def stepped() -> TrainState:
        held[0], *_ = step(held[0], batch)
        return held[0]

    return samples(stepped, warmup=TIMED_WARMUP, repeats=TIMED_REPEATS, calls=TIMED_STEPS)


UNTIMED_FRONTIER = "DEW_UNTIMED_FRONTIER"
"""The environment variable that, set to 1, has a process keep a frontier's first plan untimed: Dew's
tests' setting (tests/lane_environment.py), never the library's. A frontier's choice follows its
clock, so two runs timed apart may train different programs, by design; a test whose property is a
resume's, two pools a run and its resume, would read that choice as its own, where no record pins a
pool's plan (`dew.training.rungs.recorded_rung`). The choice itself stays tested through `Trainer.compile`'s
own timer."""


def untimed_frontier() -> bool:
    """Whether this process keeps a frontier's first plan untimed (`UNTIMED_FRONTIER`)."""
    return os.environ.get(UNTIMED_FRONTIER) == "1"


def faster(timings: Sequence[Sequence[float]]) -> int:
    """The plan of a frontier's `timings` (each plan's samples, `step_seconds`) that runs faster: a later
    one only where its slowest sample beats every sample of the first's, the first otherwise. A choice
    within the timings' spread is the clock's noise, which would give two runs of one configuration
    different plans, and so different models and steps, where neither runs faster."""
    chosen = 0
    for index, timing in enumerate(timings[1:], start=1):
        if max(timing) < min(timings[chosen]):
            chosen = index
    return chosen


@dataclasses.dataclass
class Trials:
    """Trial runs of a plan's steps on `batch` from `state`, each on a copy of the state as it stood
    (`run`): an offline frontier's timing (`seconds`), where `frontier` asks for it, and AutoPGLE's
    profile of the plan admitted (`profiled`). Live compilation never invokes `seconds`. A state the
    trials own (`owned`, `fit`'s) is stepped by the first trial
    and deleted, each later trial stepping a fresh copy from the host, so the devices hold one state at a
    time, and `state` is then a copy as it stood, for the run to go on from. One they borrow (a compile
    given none, `Trainer.compile`) is never stepped and stays `state`: each trial steps a copy beside it,
    freed as the trial ends, for which the devices must hold room (`profiled`)."""
    state: TrainState
    batch: Batch
    owned: bool = True
    frontier: bool = True
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

    def seconds(self, steps: Sequence[Callable]) -> list[list[float]]:
        """Each of a frontier's `steps` timed on its own trial (`step_seconds`)."""
        return [self.run(lambda state, step=step: step_seconds(step, state, self.batch)) for step in steps]

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
                with jax.profiler.trace(directory, profiler_options=capture_options()):
                    for _ in range(runs):
                        for part in parts:
                            current, *_ = planned.executable(current, held, self.batch, part)
                        jax.block_until_ready(current)
                (capture,) = Path(directory).glob("plugins/profile/*")
                return get_profiled_instructions_proto(str(capture))

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
        stats = None if executable is None else executable.memory_analysis()
        local = (None if stats is None or headroom is None
                 else (int(stats.temp_size_in_bytes), headroom, factor))
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


def settled(planned: Planned, plans: Sequence[Planned], seconds: Sequence[Sequence[float]],
            probes: Sequence[Probe]) -> dict[str, JSON]:
    """The record of `planned`, of `plans` a frontier timed at `seconds` a
    step (none where one plan fit first): every compile the search made,
    `probes`, each timed plan's split, rows, rung and seconds, and the planned
    step's kernels (`KernelReport.record`: each dispatch's route and each
    kernel's lowered launches), which the run log states."""
    _log.info("the step's %s", planned.candidate.kernels.summary())
    return {**planned.record(), "kernels": planned.candidate.kernels.record(),
            "probes": [probe.record() for probe in probes],
            "timed": [{**plan.record(), "seconds": min(took)}
                      for plan, took in zip(plans, seconds, strict=True)] if seconds else None}


def opens_frontier(split: int, whole: int, mesh: Mesh) -> bool:
    """Whether a plan reading its batch in `split` microbatches is timed against the batch whole, in
    `whole`, a remat rung up (`FitSearch.plans`' frontier): where it splits the batch, or spans accelerators,
    whose collectives the rung's freed memory lets XLA's scheduler overlap with compute. Measured, not
    ranked: Qwen3.5-0.8B on an A100-40 ran 1205.6 ms a step in four microbatches, 1174.1 whole under
    'minimal'; Qwen3.5-9B by fsdp over two A100-80s left 168 of 169 ms of collectives exposed, 15 under
    'minimal'."""
    return split != whole or mesh.devices.size > 1


@dataclasses.dataclass
class FitSearch[LossT, EffectsT]:
    """A step's search over the fit ladder (`planned`): the first of the run's split candidates (`splits`)
    and the objective's rungs to fit the devices, each lowered by `lower` at the objective's rung and
    `narrow` once, compiled under its `options` or XLA's defaults on the half rung above, the head at the
    run's tile `base` at each remat rung (`recompute_more`). It holds what it lowered, compiled and
    measured, and where on the ladder it stands.

    The first rung climbed drops the narrow copies, which a plan above it, the frontier's too, takes
    back where its rung and split fit with them. A split missing at a rung with a half rung
    (`remat_half`) is compiled there, the plan staying where it fits; climbing and the frontier pass
    every half rung asked (`onward`). Each compile's measure (`Forecast`) orders the splits; `resumed`, a
    checkpoint's rung, is warned of where it misses; where nothing fits it raises with the closest
    candidate's numbers. The objective is left at the last plan's rung.

    The ladder holds no host offload. XLA:GPU holds the optimizer state and the EMA whole on the device
    for the update wherever they rest between steps, so keeping them on the host frees nothing at a
    step's peak (a decoder's step on an A100 peaked at 8.427 GB either way, node receipt
    TL-a100-offload-1338-c); a run keeps them there by its layout (`Layout.host`) alone. A remat policy
    that keeps its residuals in pinned host memory ('minimal_offloaded') keeps as many on the device as
    'full' does (`test_model_protocols`' `_saved`: a decoder's 107,720 bytes both, 'save_dot_except_mlp'
    132,296), so it fits nowhere 'full' does not, and a run takes it by its own remat."""
    lower: Callable[[int, bool], Candidate]
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
    dropped: str | None = None
    """The rung (`rung`) the narrow copies were dropped at."""

    def __post_init__(self):
        self.factor = temporaries_factor(self.mesh)
        # Every rung keeps, XLA's defaults' included, the options the modules need to compute right.
        self.required = compiler_options(*self.objective.towers()) or None

    def plans(self, candidates: Sequence[int], first: int, frontier: bool
              ) -> tuple[list[Planned], tuple[Probe, ...]]:
        """The plans to time, the first kept where none is timed, with every compile made: the ladder's
        first of `candidates` from `candidates[first]` and, where `frontier` asks and it opens a frontier
        (`opens_frontier`), the batch whole a remat rung up (`remat_more`), the head tiled and unfused,
        above any rung that dropped the copies."""
        split = planned(candidates, first, self.fits, self.climb,
                        lambda split: self.forecast.likely(self.rung(), split))
        if split is None or self.fitted is None:
            raise RuntimeError(unplanned(self.probes)) from (self.refusals[-1] if self.refusals else None)
        plans = [self.found(split)]
        if (frontier and opens_frontier(split, candidates[0], self.mesh)
              and self.onward(lambda: remat_more(self.objective))):
            self.objective.tile_head()
            self.objective.unfuse_head()
            self.xla_defaults, self.narrow = False, self.narrow and self.dropped is None
            self.lowered.clear()
            head = "the head tiled" if head_tile_of(self.objective) is not None else "the whole logits kept"
            _log.info("timing the plan against its batch whole at remat %r, %s",
                      recompute_record(self.objective), head)
            if self.fits(candidates[0]):
                plans.append(self.found(candidates[0]))
        return plans, tuple(self.probes)

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
        if self.fits_here(split):
            return True
        missed, held = self.rung(), remat_half(self.objective)
        self.forecast.half_of[missed] = None if held is None else self.rung()
        if held is None:
            return False
        self.halves.add(self.remat_rung())
        if self.fits_here(split):
            return True
        self.objective.substitute(held)
        return False

    def measured(self, split: int, candidate: Candidate
                 ) -> tuple[jax.stages.CompilerOptions | None, jax.stages.Compiled | None, Probe]:
        """`candidate`, the step at `split` at the objective's rung, compiled under the rung's options
        where it fits the devices, and the measure of its fit (`Probe`), the search's state untouched."""
        options = self.required if self.xla_defaults else self.options(candidate)
        executable = compiled_if_it_fits(candidate.lowered, options, self.refusals)
        fit, headroom = step_fits(executable, self.mesh, candidate.held)
        return options, executable, Probe.read(split, candidate, self.current(), fit, headroom, executable)

    def fits_here(self, split: int) -> bool:
        key = (split, self.rung())
        if key not in self.lowered:
            self.lowered[key] = self.lower(split, self.narrow)
        candidate = self.lowered[key]
        self.chosen, executable, probe = self.measured(split, candidate)
        fit, headroom = probe.fits, probe.headroom
        self.probes.append(probe)
        self.forecast.read(self.rung(), split, executable, headroom, self.factor)
        if fit:
            assert executable is not None
            self.fitted = candidate, executable, self.chosen
            return True
        if self.resumed is not None:
            _log.warning("the step does not fit the devices at the rung its checkpoint trained on (%s); from "
                         "here the resumed run computes otherwise than the run it continues", self.resumed)
            self.resumed = None
        return False

    def climb(self) -> bool:
        # The first rung drops the narrow copies, two bytes a parameter held all run for 1-2% of a step.
        if self.narrow and any(candidate.copies for candidate in self.lowered.values()):
            self.narrow = False
            self.dropped = self.rung()
            self.lowered.clear()
            return True
        # XLA's default options, but `required`, are the half rung above the step's own where those
        # are another program: the Triton GEMM fusions they keep can take fewer temporaries (sm89).
        if not self.xla_defaults and self.chosen != self.required:
            self.xla_defaults = True
            return True
        self.xla_defaults = False
        self.lowered.clear()
        return self.onward(lambda: recompute_more(self.objective, self.base))

    def found(self, split: int) -> Planned:
        if not self.narrow and self.dropped not in (None, self.rung()):
            # A rung above the copies' may recompute the casts they save: a plan keeps them where it fits.
            kept, self.narrow = self.fitted, True
            if not self.fits_here(split):
                self.fitted, self.narrow = kept, False
        assert self.fitted is not None
        modules = tuple(entry.module for entry in self.objective.program_key())
        return Planned(split, *self.fitted, self.xla_defaults, self.narrow, self.current(), modules,
                       head_tile_of(self.objective), head_fused_of(self.objective))
