"""The plan of a training step: the microbatches it reads its batch in and
the rung of the fit ladder it compiles at (`Trainer.compile`).

A step that does not fit its devices whole is read in microbatches, an even
split of every row shard's own rows, before it recomputes anything:
recomputing costs every token, where a smaller microbatch at the same rung
costs little. Compiles are the measurement, XLA's memory analysis of each
candidate against each device's free bytes (`dew.training.memory`), and the
search bounds how many a step takes.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh, Mesh

from dew.coordination import from_every_process
from dew.data.dataset import rows_of
from dew.nn.kernels.options import compiler_options
from dew.nn.sharding import LayoutRefused
from dew.objectives.base import RECORD, Batch, Objective
from dew.records import JSON
from dew.telemetry.benchmark import samples
from dew.training.distributed import Placement, batch_divisor
from dew.training.host import snapshot
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
from dew.training.state import TrainState

_log = logging.getLogger(__name__)


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
    traces them, the bytes held beside it and the program lowered."""
    micro: Batch
    shapes: object
    copies: dict
    shardings: Placement[TrainState]
    prepared: TrainState
    held: int
    program: jax.stages.Lowered


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
    (`plan_step`), so it is no rung here. Memory falls
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
    """A plan `plan_step` found: the step reading its batch in `split`
    microbatches and its executable, whether it keeps XLA's default options,
    and the rung it compiled at, as a record and as the objective's modules,
    head tile and whether that head is fused."""
    split: int
    candidate: Candidate
    executable: jax.stages.Compiled
    xla_defaults: bool
    rung: Mapping[str, JSON]
    modules: tuple
    tile: tuple[int, int] | None
    fused: bool

    def record(self) -> dict[str, JSON]:
        return {"split": self.split, "rows": rows_of(self.candidate.micro), **self.rung}


TIMED_STEPS, TIMED_WARMUP = 10, 2
"""The steps a frontier's plans are timed over (`step_seconds`), after the
warmup ones, each plan's own: a split step's first call builds its window."""


def step_seconds(step: Callable, state: TrainState, batch: Batch) -> float:
    """Seconds a step of `step` takes on `batch` from `state`, the mean of
    `TIMED_STEPS` queued and synchronized once after `TIMED_WARMUP`
    (`dew.telemetry.benchmark.samples`), each step consuming the state the
    one before it returned, as training does."""
    held = [state]

    def stepped() -> TrainState:
        held[0], *_ = step(held[0], batch)
        return held[0]

    return samples(stepped, warmup=TIMED_WARMUP, repeats=1, calls=TIMED_STEPS)[0]


@dataclasses.dataclass
class Timer:
    """Times a frontier's steps on `batch` from `state` (`step_seconds`):
    the first steps the state itself, which it then deletes, and each later
    one a fresh copy from the host, so the devices hold one state at a time.
    `state` is then a fresh copy for the run to go on from."""
    state: TrainState
    batch: Batch

    def __call__(self, steps: Sequence[Callable]) -> list[float]:
        restore = snapshot(self.state)
        seconds = [step_seconds(steps[0], self.state, self.batch)]
        # What the step did not donate is freed too, so the run holds no
        # reference to the state it handed over, on any backend.
        for leaf in jax.tree.leaves(self.state):
            if not leaf.is_deleted():
                leaf.delete()
        seconds += [step_seconds(step, restore(), self.batch) for step in steps[1:]]
        self.state = restore()
        return seconds


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


def settled(planned: Planned, plans: Sequence[Planned], seconds: Sequence[float],
            probes: Sequence[Probe]) -> dict[str, JSON]:
    """The record of `planned`, of `plans` a frontier timed at `seconds` a
    step (none where one plan fit first): every compile the search made,
    `probes`, and each timed plan's split, rows, rung and seconds."""
    return {**planned.record(), "probes": [probe.record() for probe in probes],
            "timed": [{**plan.record(), "seconds": took}
                      for plan, took in zip(plans, seconds, strict=True)] if seconds else None}


def plan_step[LossT, EffectsT](
        lower: Callable[[int], Candidate], options: Callable[[Candidate], jax.stages.CompilerOptions | None],
        candidates: Sequence[int], first: int, *, objective: Objective[LossT, EffectsT], mesh: Mesh,
        xla_defaults: bool, base: tuple[int, int] | None, resumed: JSON,
        frontier: bool) -> tuple[list[Planned], tuple[Probe, ...]]:
    """The first of `candidates` (`splits`) and the objective's rungs to fit
    the devices (`planned`), each lowered by `lower` at the objective's
    current rung once, and compiled under its `options`, or under XLA's
    defaults and `compiler_options` on the half rung above, the head
    returning to the run's own tile `base` at each remat rung
    (`recompute_more`). Where a split does not fit at a rung whose models
    have a half rung above it (`remat_half`, a decoder's delta rule
    recomputing its memory alone), the same split is compiled there, and the
    plan stays there where it fits: no plan splits the batch further at a
    rung where the half rung fits it whole, and climbing and the frontier
    pass every half rung asked (`onward`). What each compile measured
    (`Forecast`) orders the splits asked. `resumed`, the
    rung a checkpoint trained the shape on, is warned of where it does not
    fit. Where nothing fits, it raises with the closest candidate's numbers.
    Beside the plans come every compile the search made, the frontier's too.

    Where that plan splits the batch and `frontier` asks, the frontier's
    other point follows, one compile more: the batch in `candidates[0]`
    microbatches one remat rung up (`remat_more`) with the head tiled and
    unfused, the most memory one rung frees, where that fits. Which runs
    faster is measured, not ranked: on an A100-40, Qwen3.5-0.8B's full
    fine-tune at 8 x 3072 ran 1205.6 ms a step in four microbatches without
    recomputing, 1174.1 under 'minimal' whole, the head tiled. The objective
    is left at the last plan's rung, each plan holding its own (`Planned`)."""
    probes: list[Probe] = []
    refusals: list[RuntimeError] = []
    lowered: dict[tuple[int, str], Candidate] = {}
    halves: set[str] = set()  # the remat rungs (`recompute_record`) the search has asked as half rungs
    forecast, factor = Forecast(), temporaries_factor(mesh)
    chosen: jax.stages.CompilerOptions | None = None
    fitted: tuple[Candidate, jax.stages.Compiled] | None = None
    # Every rung keeps, XLA's defaults' included, the options the modules need to compute right.
    required = compiler_options(*(entry.module for entry in objective.program_key())) or None

    def rung() -> str:
        return json.dumps(rung_record(objective, xla_defaults), sort_keys=True)

    def remat_rung() -> str:
        return json.dumps(recompute_record(objective), sort_keys=True)

    def onward(step: Callable[[], bool]) -> bool:
        """`step` the objective up its ladder, past every half rung the search has asked: each split
        tried there missed, so a rung above it is the next to ask."""
        moved = step()
        while moved and remat_rung() in halves:
            moved = step()
        return moved

    def fits(split: int) -> bool:
        if fits_here(split):
            return True
        missed, held = rung(), remat_half(objective)
        forecast.half_of[missed] = None if held is None else rung()
        if held is None:
            return False
        halves.add(remat_rung())
        if fits_here(split):
            return True
        objective.substitute(held)
        return False

    def fits_here(split: int) -> bool:
        nonlocal resumed, chosen, fitted
        key = (split, rung())
        if key not in lowered:
            lowered[key] = lower(split)
        candidate = lowered[key]
        chosen = required if xla_defaults else options(candidate)
        executable = compiled_if_it_fits(candidate.program, chosen, refusals)
        fit, headroom = step_fits(executable, mesh, candidate.held)
        stats = None if executable is None else executable.memory_analysis()
        probes.append(Probe(split, rows_of(candidate.micro), rung_record(objective, xla_defaults), fit,
                            headroom, None if stats is None else int(stats.temp_size_in_bytes),
                            None if stats is None else int(stats.argument_size_in_bytes)))
        forecast.read(rung(), split, executable, headroom, factor)
        if fit:
            assert executable is not None
            fitted = candidate, executable
        elif resumed is not None:
            _log.warning("the step does not fit the devices at the rung its checkpoint trained on (%s); from "
                         "here the resumed run computes otherwise than the run it continues", resumed)
            resumed = None
        return fit

    def climb() -> bool:
        # XLA's default options, but `required`, are the half rung above the step's own where those
        # are another program: the Triton GEMM fusions they keep can take fewer temporaries (sm89).
        nonlocal xla_defaults
        if not xla_defaults and chosen != required:
            xla_defaults = True
            return True
        xla_defaults = False
        lowered.clear()
        return onward(lambda: recompute_more(objective, base))

    def found(split: int) -> Planned:
        assert fitted is not None
        modules = tuple(entry.module for entry in objective.program_key())
        return Planned(split, *fitted, xla_defaults, rung_record(objective, xla_defaults), modules,
                       head_tile_of(objective), head_fused_of(objective))

    split = planned(candidates, first, fits, climb, lambda split: forecast.likely(rung(), split))
    if split is None or fitted is None:
        raise RuntimeError(unplanned(probes)) from (refusals[-1] if refusals else None)
    plans = [found(split)]
    if frontier and split != candidates[0] and onward(lambda: remat_more(objective)):
        objective.tile_head()
        objective.unfuse_head()
        xla_defaults = False
        lowered.clear()
        _log.info("timing the plan against its batch whole at remat %r, %s", recompute_record(objective),
                     "the head tiled" if head_tile_of(objective) is not None else "the whole logits kept")
        if fits(candidates[0]):
            plans.append(found(candidates[0]))
    return plans, tuple(probes)
