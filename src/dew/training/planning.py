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
from dew._devices import device_memory, pool_share, pool_statement, stated_share
from dew._records import JSON
from dew.batch import Batch
from dew.data.dataset import rows_of
from dew.distributed.host import snapshot
from dew.distributed.placement import Layout, MeshSpec, Placement, explicit_axes, placed_bytes
from dew.mesh import LayoutRefused
from dew.nn.kernels.options import (
    OuterProgram,
    compiler_options,
    held,
    hold,
    owned_program,
    program_key,
    running,
    shapes_of,
    unhold,
)
from dew.nn.kernels.templates import report
from dew.objectives.base import RECORD, Objective, Pooled, head_fused_of, head_tile_of
from dew.state import TrainState, apart, together
from dew.telemetry.compiles import Ledger
from dew.telemetry.instrumentation import compiled_memory
from dew.telemetry.profile import Profiler, active_profile
from dew.training.ahead import handed
from dew.training.distributed import batch_divisor
from dew.training.interval import latest
from dew.training.memory import (
    LIMIT_EXIT,
    LiveSet,
    MemoryMiss,
    admission,
    compile_limit,
    compiled_if_it_fits,
    local_devices,
    offloads,
    placeable,
    placed_beside,
    recompute_more,
    recompute_record,
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
    a custom `step`, a layout that keeps the variables on the host where the run states no split (no plan
    splits the host's step: it reads its batch in the split stated, `StepCompiler._compile_host`), a
    `rollout` or an Explicit mesh axis. Refuses the arguments no Trainer takes together."""
    if accumulation < 1:
        raise ValueError(f"accumulation must be at least 1, got {accumulation}")
    # Split under an Explicit axis, a step met a cotangent the split typed
    # float32[16@data,16,16] against its forward's float32[16,16,16].
    whole = ("a custom step owns its execution" if step is not None else
             "the layout keeps the variables on the host" if "variables" in layout.host and split is None else
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
    """The bytes the step's arguments hold of what `held` charges, as XLA counts them, one allocation's
    rounding short of their charge (`allocated_bytes`): the copies and accumulator its first call builds."""
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
    joins: bool = False
    """Whether its updates join two or more Newton-Schulz stacks of one shape where its run states no group
    (`dew.newton_schulz.joining`): what its plan's release would part (`FitSearch.searched`)."""
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
    from (its lowering), which a run compiled ahead names it by (`program_digest`), and its executable.
    A plan the run steps by lets the lowering go (`Planned.released`): JAX holds every executable
    compiled from a lowering while the lowering lives, so a kept one held the sampled module resident
    through the run (base review of 2ffc61c8c, F2)."""

    lowered: jax.stages.Lowered | None
    executable: jax.stages.Compiled


def whole_step_program[L, E](objective: Objective[L, E], parted_work: bool = False) -> OuterProgram:
    """`whole_step` over `objective`'s statistics (`Objective.reduce_loss`), the one program a split step
    runs on its microbatches' outputs, compiled through Dew's compile point, which a run compiled ahead
    compiles with its step (`dew.training.ahead`); `parted_work` where each microbatch counts its own
    work (a sampled pass count's, `step_counts`)."""
    return owned_program(functools.partial(whole_step, reduce=objective.reduce_loss,
                                           shown=frozenset(objective.shown), parted_work=parted_work))


def whole_step[L](outputs, reduce: Callable[[L], tuple[jax.Array, jax.Array]],
                  shown: frozenset[str] = frozenset(), parted_work: bool = False):
    """A step's outputs from those of the microbatches it was read in: its loss as an unsplit step logs it,
    whatever the split (docs/design/training.md, the logged loss): where a microbatch of it commits, the
    latest such commit's update's, pooled over its window; elsewhere its own batch's, its microbatches'
    own statistics pooled (`Committed.own`) and reduced by `reduce` (`Objective.reduce_loss`); its
    metrics pooled by their own statistics (`_pooled`), a metric the objective shows (`shown`, which a
    run selects by) refused where it states none; finite and accepted where every microbatch was; and the
    commits' counts,
    tokens and norms summed, its inspection the latest part's that committed (`interval.latest`), as a
    run's interval keeps the latest sampled commit: a step of two commits (a window a rejection left part
    filled, then the closing part of a run's last batch) reports its second's summaries, never the two
    added. A pure function, which a step compiles as a program of its own (`dew.training.compiler`) or
    traces within its own (`dew.training.ahead`). Where each microbatch counted its own work
    (`parted_work`, a sampled pass count's), the step's is their sum."""
    losses, metrics, finite, accepted, committed = zip(*outputs, strict=True)
    report = {name: (sum(part[name] for part in metrics) if parted_work and name == TRAINING_FLOPS
                     else _pooled(name, [part[name] for part in metrics])) for name in metrics[-1]}
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
        # The latest part that commits logs its update's loss, its window pooled, whichever part it is: a
        # window a rejection left part filled commits at an earlier part, and the later ones open the next.
        loss = own.astype(loss.dtype)
        for part, update in zip(committed, losses, strict=True):
            loss = jnp.where(part.count > 0, update, loss)
    summed = jax.tree.map(lambda *values: sum(values),
                          *(dataclasses.replace(part, inspected=None, own=None, finite=None,
                                                unfinite_layer=None, unfinite_state=None)
                            for part in committed))
    walked = [part.unfinite_layer for part in committed if part.unfinite_layer is not None]
    # The state the step leaves is its last part's, whose committed state's report it keeps, a leaf's
    # index no sum over parts (row 8269).
    summed = dataclasses.replace(summed, unfinite_layer=functools.reduce(jnp.maximum, walked) if walked
                                 else None, unfinite_state=committed[-1].unfinite_state)
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


def walked(candidates: Sequence[int], first: int, fits: Callable[[int], bool],
           after: Callable[[int], int | None]) -> int | None:
    """The split a step reads its batch in at the rung the objective stands on, or None where none of
    `candidates` (ascending: the batch read in that many equal microbatches) fits there.

    `fits(split)`, a compile at the rung, is asked of `candidates[first]` first (the batch whole, a plan
    recorded or one a search before it found); past a miss, of the split that compile computes as the
    first that can fit (`after`, an index into `candidates`, from what it read: `MemoryMiss`), the finest
    where they compute none, and so on from each compile that misses, upward, until one fits or the finest
    misses. A split that fits past one not asked is asked of the split before it, down to one that misses;
    the split before `first` is taken as missed, by the record the start read or the miss that computed
    `first`. XLA fuses and schedules each split's program again, so its temporaries can fall faster than
    its rows and a coarser split fit that the figures said would not (a phased run's three-pass step fit at
    two microbatches where its whole batch's figures computed none); no figure bounds what a split frees,
    and the compile at each is the judge. A tight start at one rung is two compiles where the split
    computed is the fewest that fit.
    """
    index, missed = first, first - 1
    while index is not None:
        if fits(candidates[index]):
            while index - 1 > missed and fits(candidates[index - 1]):
                index -= 1
            return candidates[index]
        missed, index = index, after(index)
    return None


@dataclasses.dataclass(frozen=True)
class Probe:
    """One compile of a plan's search (`searched`, `walked`): the split, the rows of
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
    (`FitSearch.ahead`), or one a search before it compiled (`dew.nn.kernels.options.held`)."""
    compile_seconds: float = 0.0
    """Time spent compiling or loading this program, apart from its shape trace and lowering."""
    spent: Mapping[str, float] = dataclasses.field(default_factory=dict)
    """The seconds JAX recorded of the candidate's work, its traces, lowerings and backend compile by kind
    (`Candidate.spent` and its compile's), which the compiled-step record and the perf gate read."""
    beside: int | None = None
    """The bytes the fit check placed beside the step's temporaries (`placed_beside`): its outputs that
    write over no donated argument and what the loop holds outside the step (copies, accumulators, a state
    not placed yet). None where none were read."""
    admitted: tuple[int, int, int] | None = None
    """The terms the fit check admitted by (`memory.admission`: resident, beside, temporaries), the
    executable's whose fit it states (the sampled step's where it stands for the step, `FitSearch.measured`);
    None where it read none."""

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

    def __init__(self, probes: Sequence[Probe], allocators: Sequence[Mapping[str, int]] = (),
                 candidates: Sequence[int] = (), variants: bool = True):
        self.probes, self.allocators = tuple(probes), tuple(allocators)
        # The splits the search asked of each rung, and whether it chose a variant of its rungs from what
        # its misses read (its Newton-Schulz stacks parted, its narrow copies dropped), which a search on
        # more room may choose otherwise (`dew.training.compiler.StepCompiler.refuse_unplaced`).
        self.candidates, self.variants = tuple(candidates), variants
        # Whether the search began at a rung a record or a resumed checkpoint named (`StepCompiler._planned`).
        self.recorded_start = False
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
class Rung:
    """A place on the fit ladder (`FitSearch.ladder`): the modules the objective runs (their remat), its
    head's tile and fusion, whether the step compiles under XLA's defaults (`required` alone) and whether it
    reads narrow copies, all `FitSearch.at` puts the objective and the search at."""
    modules: tuple[nn.Module, ...]
    tile: tuple[int, int] | None
    fused: bool
    xla_defaults: bool
    narrow: bool
    ns_group: int | None = None

    @staticmethod
    def of(objective, xla_defaults: bool, narrow: bool, ns_group: int | None = None) -> Rung:
        """Where `objective` stands, under XLA's defaults or not, reading narrow copies or not, and joining
        its Newton-Schulz stacks `ns_group` at a time (all where None)."""
        return Rung(tuple(entry.module for entry in objective.program_key()), head_tile_of(objective),
                    head_fused_of(objective), xla_defaults, narrow, ns_group)

    def put(self, objective) -> None:
        """`objective` at this rung: its modules and its head's tile and fusion. The options and copies are
        its compiler's or its search's to set (`FitSearch.at`, `StepCompiler._planned`)."""
        objective.substitute(list(self.modules))
        objective.restore_head(self.tile, fused=self.fused)


@dataclasses.dataclass(frozen=True)
class Planned:
    """A plan `FitSearch` found: the step reading its batch in `split`
    microbatches, its executable and the options it compiled under, and the rung it compiled at, as a
    record (`rung`) and as the place on the ladder the objective and its compiler stand at (`at`: its
    modules, head tile and fusion, XLA's default options or not, narrow copies or not). `profiled` is the
    runs and the bytes of the profile that guided its executable (`guided`), None where none did."""
    split: int
    candidate: Candidate
    executable: jax.stages.Compiled
    options: jax.stages.CompilerOptions | None
    rung: Mapping[str, JSON]
    at: Rung
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
        executable = guided_compile(self.candidate.lowered, self.options, profile, relowered)
        candidate = relowered.candidate(self.candidate)
        fit, headroom = step_fits(executable, mesh, candidate.held)
        if not fit:
            raise RuntimeError(f"the step guided by its profile (jax_enable_pgle) needs {-(headroom or 0)} "
                               "bytes more than its tightest device has, where unguided it fits; train it "
                               "without AutoPGLE")
        return dataclasses.replace(self, candidate=candidate, executable=executable,
                                   profiled=(runs, len(profile)))

    def released(self) -> Planned:
        """This plan as the run keeps it once admitted: its executables without the programs they were
        lowered from, the step's and its sampled step's. JAX keeps each executable compiled from a lowering
        while the lowering lives (its in-memory compilation cache, keyed by the lowered module), so
        releasing them lets every executable the search or a guided compile made of them go, the ones the
        run steps alone held."""
        sampled = None if self.sampled is None else self.sampled._replace(lowered=None)
        return dataclasses.replace(self, candidate=dataclasses.replace(self.candidate, program=None),
                                   sampled=sampled)


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

            def once() -> None:
                nonlocal current
                for part in parts:
                    current, *_ = planned.executable(current, held, self.batch, part)
                jax.block_until_ready(current)

            return profile_of(once, runs)

        return self.run(profile)

    def profiled_programs(self, step: Callable, runs: int, platform: Callable[[jax.stages.Compiled], bool]
                          ) -> tuple[bytes, list[Ran]]:
        """XLA's profile of `runs` steps of `step` (a host step, whose programs are each an executable of
        its own run through Dew's compile point) on a trial (`profile_of`), after one untraced, as
        `profiled`'s; with each executable those steps ran that `platform` takes (an accelerator's), its
        program and its lowering, traced as the call traced it (`options.running`), which a guided compile
        reads (`guided_compile`). One profile serves every program: XLA keys each instruction's cost by its
        module's fingerprint. A trial borrowed on the host holds its copy in host memory, no device's."""
        if active_profile() is not None:
            raise RuntimeError(PGLE_CAPTURE)
        ran: dict[int, Ran] = {}

        @contextlib.contextmanager
        def recorded(program: OuterProgram, executable: jax.stages.Compiled, args: tuple, kwargs: dict):
            if id(executable) not in ran and platform(executable):
                # Over its arguments' shapes, which a later lowering (a spill's) reads after the trial
                # donated them; lowered here, under the contexts the call traced in.
                shaped, named = jax.tree.map(
                    lambda leaf: shapes_of(leaf) if isinstance(leaf, jax.Array) else leaf, (args, kwargs))
                lower = functools.partial(program.lower, *shaped, **named)
                ran[id(executable)] = (program, executable, lower(), lower)
            yield

        def profile(state: TrainState) -> bytes:
            current, *_ = step(state, self.batch)

            def once() -> None:
                nonlocal current
                current, *_ = step(current, self.batch)
                jax.block_until_ready(current)

            with running(recorded):
                return profile_of(once, runs)

        return self.run(profile), list(ran.values())


type Ran = tuple[OuterProgram, jax.stages.Compiled, jax.stages.Lowered, Callable[[], jax.stages.Lowered]]
"""An executable a profiled trial ran (`Trials.profiled_programs`): its program, it, its lowering and what
lowers it again (a spill's relowering, `report.compiled`)."""


def profile_of(run: Callable[[], None], runs: int) -> bytes:
    """XLA's profile of `runs` calls of `run`, each waiting on what it ran, as JAX's AutoPGLE waits on the
    runs it profiles, in one profiling session of its own (`jax.experimental.profiler
    .get_profiled_instructions_proto`): empty on a backend XLA profiles none of. The one place Dew reads
    JAX's profile of a trial (`Trials`)."""
    with tempfile.TemporaryDirectory() as directory:
        # XLA reads the trace alone, so the capture writes no XProf reports.
        with Profiler(directory, reports=False) as profiler:
            for _ in range(runs):
                run()
        (capture,) = profiler.captures
        (session,) = capture.glob("plugins/profile/*")
        return get_profiled_instructions_proto(str(session))


def guided_compile(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None, profile: bytes,
                   relower: Callable[[], jax.stages.Lowered]) -> jax.stages.Compiled:
    """`lowered` compiled again under `options` guided by `profile` (`profile_of`), as JAX's jit guides
    AutoPGLE's (XLA's feedback-directed `fdo_profile`), through the kernel templates' reports
    (`report.compiled`, `relower` lowering afresh where a kernel spills): a device plan's step
    (`Planned.guided`) and each program of a host step (`StepCompiler._guided_host`) alike."""
    return report.compiled(lowered, {**(options or {}), "fdo_profile": profile}, relower=relower)


def compiled_key(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None,
                 devices: Sequence) -> Hashable:
    """The key of the executable a plan compiles from `lowered` under `options` for `devices`, which the
    process holds across its trainers (`dew.nn.kernels.options.hold`): the program within this process
    (`program_key`), one for a lowering over a state's shapes before it is placed and over the placed state,
    the options as text, and the devices it loads on."""
    return program_key(lowered), repr(sorted((options or {}).items())), tuple(devices)


def settled(planned: Planned, probes: Sequence[Probe], search: FitSearch,
            frontier: Sequence[Probe] | None = None,
            lighter: Sequence[Probe] | None = None) -> dict[str, JSON]:
    """The admitted plan, its kernel report, every compile needed to find it, and the kernels those compiles
    found spilling and the launch configs refused where it searched (`FitSearch.kernels`), which the run's
    telemetry reports with it (`StepCompiled`); planned ahead with its `frontier`, each rung above the plan
    that fits at fewer microbatches (`FitSearch.above`), and each rung below it that the search passed over
    and that fits (`FitSearch.below`, `lighter`). There is no alternative-policy timing or selection in the
    record."""
    _log.info("the step's %s", planned.candidate.kernels.summary())
    above: dict[str, JSON] = {} if frontier is None else {"frontier": [probe.record() for probe in frontier]}
    if lighter is not None:
        above["lighter"] = [probe.record() for probe in lighter]
    return {**planned.record(), "kernels": planned.candidate.kernels.record(),
            "probes": [probe.record() for probe in probes], **search.kernels,
            "pool": pool_record(local_devices(search.mesh)),
            # Read at autotune 0 where compiled offline for a GPU kind (`report.autotuned`): its memory
            # XLA's default choices', which a tuned one on the device may exceed.
            "autotuned": report.autotuned(local_devices(search.mesh)), **above}


def pool_record(devices: Sequence) -> dict[str, JSON]:
    """The allocator's limit a plan was made against, its tightest device's (`device_memory`), with the pool's
    share and its source (`dew._devices.pool_statement`): a plan held to JAX's 75% of the device, its
    backend opened before Dew sized the pool, is visible as such. Devices this process does not attach
    plan against the memory stated for them (`compile_only`), at the share their loading run takes."""
    limits = [limit for device in devices if (limit := device_memory(device).get("bytes_limit")) is not None]
    attached = set(jax.local_devices())
    # Devices this process does not attach plan at the share their loading run states (`stated_share`).
    shares = [pool_share() if (share := stated_share(device)) is None else share
              for device in devices if device not in attached]
    stated = {"share": min(shares), "source": "compile_only"} if shares else pool_statement()
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
    """A kernel a compile refused, or a launch whose program's block XLA refused (`report.Miss`): every
    field the miss holds, read from its declaration so a field it gains is saved with the plan too, its
    launch's config by its text."""
    record = {field.name: getattr(miss, field.name) for field in dataclasses.fields(miss)}
    return {**record, "config": None if miss.config is None else repr(miss.config)}


class Lower(Protocol):
    """A search's step at a split, whether it keeps its narrow copies and the Newton-Schulz stacks it joins at
    a time (`ns_group`), at the objective's rung, lowered (`FitSearch.lower`)."""

    def __call__(self, split: int, narrow: bool, ns_group: int | None) -> Candidate: ...


@contextlib.contextmanager
def spending(candidate: Candidate | None = None) -> Iterator[dict[str, float]]:
    """The seconds JAX records of the work within the block, by kind (`EVENTS`: traced, lowered, compiled),
    added to `candidate`'s (`Candidate.spent`), in the dict it yields once the block ends."""
    held = dict({} if candidate is None else candidate.spent)
    with Ledger(None) as compiles:
        yield held
    for made in compiles.programs:
        held[made.kind] = held.get(made.kind, 0.0) + made.seconds


def batch_rows(candidate: Candidate) -> frozenset[tuple[int, int]]:
    """The (argument, axis) pairs of the batch's rows among `candidate`'s step's arguments (its state's
    moving and frozen trees, its batch, its part, `step_program`): each batch leaf's leading axis, by its
    flat index among them."""
    moving, frozen = apart(candidate.prepared)
    offset = len(jax.tree.leaves(moving)) + len(jax.tree.leaves(frozen))
    return frozenset((offset + index, 0) for index in range(len(jax.tree.leaves(candidate.micro))))


@dataclasses.dataclass
class FitSearch[LossT, EffectsT]:
    """A step's search over the fit ladder: the first of the run's split candidates (`splits`) and the
    objective's rungs that fit the devices, each lowered by `lower` at the rung once, compiled under its
    `options` or XLA's defaults, the head at the run's tile `base` at each remat rung (`ladder`). It holds
    what it lowered, compiled and measured, and the plan each rung fitted at.

    The rung it starts at (the run's own, a plan recorded or a search's before it) is asked first, its
    splits walked as each miss's figures leave possible (`walked`): most runs fit there. Past it the
    Newton-Schulz stacks its updates join where its run states no group (`Candidate.joins`) are parted one
    at a time, before any recomputation, and joined again where that frees nothing (row 5889); then the
    remat rungs, least recomputation first, each at the run's own head and options with the narrow copies
    the start held (`recomputing`): a tight start settles in its second compile where the next remat rung
    holds what its miss was short of (owner, 2026-11-21, BIBLE §6: two compiles a tight cold start). Past
    those, where none fits, the narrow copies' drop, judged by the two misses (where the step without them
    fell less short, the rungs after keep them dropped; where it fell no less short, a GPU's step casting its
    weights whole either way, they come back: `freed`), then the whole ladder in its order, each remat rung
    with its head's rungs under the step's options and XLA's defaults (`ladder`), every rung asked before
    the search gives up. Each attempt is one compile which, on a GPU, stops after buffer assignment where
    the step cannot fit (`LIMIT_EXIT`), so a miss costs its assignment and no code generation, and states its
    live set (`LiveSet`), from which the split asked next at that rung is computed (`MemoryMiss`). The plan
    is the first rung of that order that fits, at the coarsest split seen to fit there: a head's tile or
    XLA's defaults, which recompute nothing, can fit below the remat rung settled, which a run states by
    its rung (`--trainer.remat`, a head tile) or plans offline at each (`told`). What a rung keeps orders its
    residuals, not XLA's allocated peak (a scanned delta decoder at four microbatches needed 31.71 MB
    recomputing all but its scans and convolutions and 32.14 MB recomputing them too, a heavier rung needing
    more), so no rung is passed over for one above or below it on a figure; the log names every rung asked
    with what it needed.
    Each compile that misses names the split asked next (`after`); `resumed`, a checkpoint's rung, is
    warned of where it misses; where nothing fits it raises with the closest candidate's numbers. The
    objective is left at the plan's rung. A miss read from JAX's persistent cache states its live set as
    a fresh compile does (`peak_live` reads its stored buffer assignment, its argument map from the
    lowering's kept arguments).

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
    ns_group: int | None
    base: tuple[int, int] | None
    resumed: JSON
    probes: list[Probe] = dataclasses.field(default_factory=list)
    refusals: list[RuntimeError] = dataclasses.field(default_factory=list)
    lowered: dict[tuple[int, str], Candidate] = dataclasses.field(default_factory=dict)
    gather: Callable[[object], list] = from_every_process
    """How every process's figure is read (`from_every_process`), so each asks the same split."""
    misses: dict[tuple[str, int], MemoryMiss] = dataclasses.field(default_factory=dict)
    """What each compile that did not fit read, by its rung and split (`MemoryMiss`), which names the split
    the search asks next (`after`)."""
    chosen: jax.stages.CompilerOptions | None = None
    fits_at: dict[str, tuple[int, tuple[Candidate, jax.stages.Compiled, jax.stages.CompilerOptions | None],
                             SampledStep | None]] = dataclasses.field(default_factory=dict)
    """Each rung a split fitted at, by its record (`rung`, as `misses` holds a rung: a rung's modules are
    Flax modules, which need not hash), the last that fitted there (a walk's descent ends at its fewest):
    the split, its candidate, executable and the options it compiled under, and its sampled step. The plan
    is the one at the rung `searched` settles on (`found`)."""
    sampled: SampledStep | None = None
    """The last measured candidate's sampled step (`measured`), which summarizes what the run inspects."""
    ahead: AheadOfTime | None = None
    """A step compiled ahead (`dew.training.ahead`), run in place of the search's first compile where it
    is that program (`AheadOfTime.taken`), lowered as its own compile lowered it (`handed`: within its
    refusals, for its GPU, `fits_here`)."""
    provisional: bool = False
    """Whether the search judges devices other than the ones the step runs on as they will stand: before
    the state is placed (`unplaced`) or off the loop while an earlier phase holds them
    (`dew.training.ahead.LaterPlans`). It holds every executable it compiles (`dew.nn.kernels.options.hold`),
    one that missed among them, for the plan made on the devices as they stand to judge again in place of
    compiling it; a search that is not holds those that fit alone, and drops a held one that misses on its
    devices, which no plan after it asks for."""
    unplaced: int | None = None
    """The bytes a device holds of the state the search plans over that the devices hold none of yet, None
    where they hold it (`dew.training.compiler.StepCompiler.planned_unplaced`): each candidate is judged with
    them beside it, which its step's first call needs as it needs the copies and accumulators it builds,
    on the devices' free bytes in all (`memory.available`), not on their allocators' largest block, which
    placing the state reshapes (a GH200's fresh 4B run, its 52.56 GiB of state read beside the step in the
    one 38.88 GiB block its pool could give of 70.88 GiB free, 22.41 GiB short, fits). A candidate holds
    none of them (`Candidate.held`), so one cached for a
    placed state serves an unplaced one."""
    variants: bool = False
    """Whether its search chose a variant of its rungs from what its misses read (`searched`)."""
    asked: list[Rung] = dataclasses.field(default_factory=list)
    """The rungs its search asked past the one it stood on first (`searched`), which a plan made offline
    asks none of again (`below`)."""
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
        self.required = compiler_options() or None

    def admitted(self, candidates: Sequence[int], first: int) -> tuple[Planned, tuple[Probe, ...]]:
        """The first fitting plan the split-and-rung search finds (`searched`), and every compile it made:
        each candidate compiled and judged by XLA's memory analysis of it on the devices' allocators
        (`step_fits`), the ladder walked from its first rung, each compile that misses naming the split
        asked next (`after`). Once admission succeeds, no optional alternate policy is compiled."""
        seen, held, traced = len(report.misses()), set(report.refused_launches()), len(report.rewrites())
        passed = len(report.passes())
        split = self.searched(candidates, first)
        spills: list[JSON] = [spill_record(miss) for miss in report.misses()[seen:]]
        refused: list[JSON] = [refusal_record(*refusal) for refusal in report.refused_launches()
                               if refusal not in held]
        rewrites: list[JSON] = [rewrite_record(made) for made in report.rewrites()[traced:]]
        # Each program the search traced, its whole algebra pass in seconds (`report.passes`), beside the
        # regions' own: what the pass cost this plan's compiles.
        passes: list[JSON] = list(report.passes()[passed:])
        self.kernels = {"spills": spills, "refused": refused, "rewrites": rewrites, "passes": passes}
        if split is None:
            allocators = [device_memory(device) for device in local_devices(self.mesh)]
            raise PlanRefused(self.probes, allocators, candidates, self.variants) from (
                self.refusals[-1] if self.refusals else None)
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
        top = (split, self.rung_now()) if offloads(self.objective) else None
        while fewer and recompute_more(self.objective, self.base):
            asked = len(self.probes)
            fewest = walked(fewer, 0, self.fits_here, functools.partial(self.after, fewer))
            if fewest is not None:
                found.append(next(probe for probe in self.probes[asked:]
                                  if probe.fits and probe.split == fewest))
                fewer = fewer[:fewer.index(fewest)]
                if offloads(self.objective):
                    top = (fewest, self.rung_now())
        if top is not None:
            self.at(top[1])
            if remat_offloaded(self.objective) and self.fits_here(top[0]):
                found.append(self.probes[-1])
        return tuple(found)

    def fitting(self, executable: jax.stages.Compiled | None,
                candidate: Candidate) -> tuple[bool, int | None, tuple[int, int, int] | None]:
        """`step_fits` of `executable`, `candidate`'s compiled step or its sampled twin, with the state's
        `unplaced` bytes beside it where the devices hold none of it yet, and the terms it admitted by
        (`admission`), None where its executable states no memory."""
        held, built = self.held(candidate), candidate.built + (self.unplaced or 0)
        stats = None if executable is None else compiled_memory(executable)
        fit, headroom = step_fits(executable, self.mesh, held, built, placed=self.unplaced is None)
        return fit, headroom, None if stats is None else admission(stats, held, built)

    def held(self, candidate: Candidate) -> int:
        """The bytes the fit check holds beside `candidate`'s step: its own (`Candidate.held`), and the
        state's `unplaced` bytes where the devices hold none of it yet."""
        return candidate.held + (self.unplaced or 0)

    def lowering(self, split: int) -> Candidate:
        """The step at `split` at the objective's rung and the search's `narrow`, lowered."""
        with spending() as spent:
            lowered = self.lower(split, self.narrow, self.ns_group)
        return dataclasses.replace(lowered, spent=spent)

    def relowering(self, split: int) -> Candidate:
        """The step at `split` lowered afresh past a kernel that spilled (`Relowered`)."""
        return self.lower(split, self.narrow, self.ns_group)

    def after(self, candidates: Sequence[int], index: int) -> int | None:
        """The index into `candidates` of the split asked after the compile of `candidates[index]` at the
        objective's rung missed: the first above it that compile computes fits (`MemoryMiss.fits_at`), the
        latest any process computes so every process asks the same; the next above it where the compile
        read no memory (an executable XLA refused for memory as it compiled, or one whose memory this
        process states none of); the last where none is computed to fit, XLA's compile the judge of the
        finest before the rung is left: the figures take every temporary as scaled by a microbatch's rows,
        and a split's temporaries can fall faster than its rows (`walked`), so they rule no split out; None
        past the last. Every process gathers at every miss, whatever its own compile read, so their rounds
        stay in step."""
        if index + 1 >= len(candidates):
            return None
        miss = self.misses.get((self.rung(), candidates[index]))
        above = range(index + 1, len(candidates))
        computed = index + 1 if miss is None else next(
            (at for at in above if miss.fits_at(candidates[at])), len(candidates) - 1)
        return max(self.gather(computed))

    def current(self) -> dict[str, JSON]:
        return rung_record(self.objective, self.xla_defaults, narrow=self.narrow, ns_group=self.ns_group)

    def rung(self) -> str:
        return json.dumps(self.current(), sort_keys=True)

    def remat_rung(self) -> str:
        return json.dumps(recompute_record(self.objective), sort_keys=True)

    def compiled_options(self, candidate: Candidate) -> jax.stages.CompilerOptions | None:
        """The options the search compiles `candidate` under: `required` at a rung of XLA's defaults, else
        the step's own (`options`), so a step whose options are its towers' own is that rung's program
        (`ladder` asks none such); the compile point adds the target's own (`report.targeted`). On a GPU,
        each compile stops after buffer assignment past the bytes no step fits in (`limited`)."""
        return self.limited(candidate, self.required if self.xla_defaults else self.options(candidate))

    def limited(self, candidate: Candidate, options: jax.stages.CompilerOptions | None
                ) -> jax.stages.CompilerOptions | None:
        """`options` with the limit past which a GPU compile of `candidate` stops after buffer assignment
        (`LIMIT_EXIT`, `compile_limit`): an attempt that cannot fit costs its assignment, not its compile.
        As given where a device states no limit or is no GPU."""
        devices = local_devices(self.mesh)
        # The fit check's own held and built bytes (`fitting`): the state's unplaced bytes count in both.
        limit = (compile_limit(devices, candidate.held, candidate.built)
                 if all(device.platform == "gpu" for device in devices) else None)
        return options if limit is None else {**(options or {}), LIMIT_EXIT: limit}

    def compiled_key(self, lowered: jax.stages.Lowered,
                     options: jax.stages.CompilerOptions | None) -> Hashable:
        """`lowered`'s executable's key under `options` on the search's devices (`compiled_key`)."""
        return compiled_key(lowered, options, local_devices(self.mesh))

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
            looked = self.compiled_key(candidate.lowered, options)
            kept = held(looked)
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
            fit, headroom, admitted = self.fitting(executable, candidate)
            # Its lowering's own key, derived again only where a spill relowered it.
            key = looked if relowered is None or relowered.last is None else self.compiled_key(
                candidate.lowered, options)
            if relowered is not None and executable is not None and (fit or self.provisional):
                # Under the program it compiled: the relowered one past a spill, which a plan after it lowers
                # directly, the spill's refusal held for the process. One that misses on the devices the
                # step runs on is no step's, its device code held for nothing.
                hold(key, executable)
            elif kept is not None and executable is kept and not fit and not self.provisional:
                unhold(key)
            self.sampled = None
            if fit and candidate.sampled is not None:
                if ahead is not None and ahead.sampled is not None:
                    # The sampled step compiled ahead with the step, lowered as its compile lowered it
                    # (`handed`), which its target compiles none of.
                    with handed(ahead):
                        sampled = candidate.sampled()
                    self.sampled = SampledStep(sampled, ahead.sampled.taken(sampled, jax.devices()))
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
                                    else SampledStep(lowered[-1], compiled))
                # Admitted by the plan whichever compiled it, ahead or here (training.md: the sampled step is
                # admitted by the same memory plan).
                fit, headroom, admitted = self.fitting(
                    None if self.sampled is None else self.sampled.executable, candidate)
        seconds = time.perf_counter() - began
        probe = Probe.read(split, candidate, self.current(), fit, headroom, executable, self.held(candidate))
        probe = dataclasses.replace(probe, ahead=taken, compile_seconds=seconds, spent=spent,
                                    admitted=admitted)
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
        if not fit and headroom is not None and probe.temp is not None:
            live = None if executable is None else LiveSet.of(executable, batch_rows(candidate))
            # Its temporaries a microbatch's rows scale, as its live set states them, those it states no
            # provenance of among them (the most a split can free of them); every one where it states none.
            scaled = probe.temp if live is None else live.scaled + live.unknown
            self.misses[(self.rung(), split)] = MemoryMiss(split, scaled, -headroom, self.factor)
        if fit:
            assert executable is not None
            self.fits_at[self.rung()] = (split, (candidate, executable, self.chosen), self.sampled)
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

    def rung_now(self) -> Rung:
        """Where on the ladder the objective and the search stand (`Rung`)."""
        return Rung.of(self.objective, self.xla_defaults, self.narrow, self.ns_group)

    def at(self, rung: Rung) -> None:
        """Put the objective and the search at `rung`."""
        rung.put(self.objective)
        self.xla_defaults, self.narrow, self.ns_group = rung.xla_defaults, rung.narrow, rung.ns_group

    def ladder(self, start: Rung, narrow: bool, ns_group: int | None, *, remat_set: bool | None = None,
               defaults: tuple[bool, ...] | None = None) -> list[list[Rung]]:
        """The rungs above `start`, where the search began, by remat rung, least recomputation first: the
        places the one climb visits from `start` (`recompute_more`: at each remat rung its head's rungs,
        then the next remat rung with its head back at the run's tile), each under the step's options and
        then under XLA's defaults where those are another program (`compiled_options`: the Triton GEMM
        fusions they keep can take fewer temporaries, sm89), each reading the narrow copies where `narrow`
        and joining its Newton-Schulz stacks `ns_group` at a time.
        A run that states its remat has its head's rungs alone (docs/design/training.md, every choice can be
        overridden). The rungs at `start`'s remat and head are left out, asked already; the objective is
        read from `start` and left there, wherever the search stood before. `remat_set` and `defaults` state
        those two choices where a caller reads the ladder for another run's state (`below`): the search's
        own (`remat_set`, its options' twins) where None."""
        self.at(start)
        # The step's own options as its program reads them, the compile's stop past the limit no option of
        # the program's (`limited`): where they are XLA's defaults, a rung under them is the same program.
        own = {name: value for name, value in (self.chosen or {}).items() if name != LIMIT_EXIT} or None
        if defaults is None:
            defaults = (False, True) if own != self.required else (False,)
        remat_set = self.remat_set if remat_set is None else remat_set
        places = [Rung.of(self.objective, xla_defaults=False, narrow=narrow, ns_group=ns_group)]
        while recompute_more(self.objective, self.base):
            place = Rung.of(self.objective, xla_defaults=False, narrow=narrow, ns_group=ns_group)
            if remat_set and place.modules != start.modules:
                break
            places.append(place)
        ladder: list[list[Rung]] = []
        for place in places:
            if not ladder or ladder[-1][0].modules != place.modules:
                ladder.append([])
            ladder[-1].extend(dataclasses.replace(place, xla_defaults=on) for on in defaults)
        self.at(start)
        own = (start.modules, start.tile, start.fused, start.xla_defaults)
        ladder = [[rung for rung in rungs if (rung.modules, rung.tile, rung.fused, rung.xla_defaults) != own]
                  for rungs in ladder]
        return [rungs for rungs in ladder if rungs]

    def searched(self, candidates: Sequence[int], first: int) -> int | None:
        """The split the step fits at, the objective and the search left at its rung (`FitSearch`), or None
        where no rung fits: the rung it stands on first, from `candidates[first]` (`walked`, `after`); then,
        where its updates join Newton-Schulz stacks its run states no group for (`Candidate.joins`), the same
        rung joining one at a time, its copies released (up to 11.3 GB at a 4B step's peak for 82 ms a step,
        row 5669, where a remat rung recomputes its forward; row 5889), kept only where that brought it nearer
        (`freed`); then each remat rung of the ladder at the run's own head and options, the narrow copies
        as the start held them, least recomputation first (`recomputing`): a tight start's next compile is
        the rung that recomputes the activations its miss was short of (owner, 2026-11-21: a GH200's 4B
        step asked six head and options rungs and the copies' drop, none freeing its 0.47 GiB, before
        'minimal' fitted, 8 compiles where two were the start's contract, BIBLE §6). Past them, where none
        fits, the rest of the ladder in its order: where its step read narrow copies, the start's rung
        without them, whose miss and the one before it decide whether the rungs after read them (`freed`),
        then every rung of the ladder from the batch whole, its head's tiles and XLA's default options
        among them, until one fits. No rung is asked twice; every process asks the same."""
        start = self.rung_now()
        after = functools.partial(self.after, candidates)
        split = walked(candidates, first, self.fits_here, after)
        if split is not None:
            return split
        # The rungs walked already, all missed: none is asked again; and each as its record names it.
        self.asked = [start]
        named = [self.rung()]

        def ask(rung: Rung, said: str) -> int | None:
            self.asked.append(rung)
            self.at(rung)
            named.append(self.rung())
            _log.warning(said, json.dumps(self.current(), sort_keys=True))
            return walked(candidates, 0, self.fits_here, after)

        narrow, ns_group = self.narrow, self.ns_group
        if ns_group is None and any(candidate.joins for candidate in self.lowered.values()):
            kept = self.rung()
            split = ask(dataclasses.replace(start, ns_group=1), "the step does not fit the devices with its "
                        "Newton-Schulz stacks joined; compiling it again joining one at a time (%s)")
            if split is not None:
                self.said(named)
                return split
            ns_group = 1 if self.freed(kept, self.rung()) else None
            # The rungs past here chosen from what the misses read (`PlanRefused.variants`).
            self.variants = True
            if ns_group is None:
                _log.warning("the step's joined Newton-Schulz stacks free nothing it needs: it keeps them "
                             "joined and climbs on")
        for rung in self.recomputing(start, narrow, ns_group):
            if rung in self.asked:
                continue
            split = ask(rung, "the step does not fit the devices; compiling it again recomputing more at %s")
            if split is not None:
                self.said(named)
                return split
        if narrow and any(candidate.copies for candidate in self.lowered.values()):
            # The narrow copies, two bytes a parameter held all run for 1-2% of a step, dropped at the start's
            # rung where no recomputation fits with them.
            self.at(dataclasses.replace(start, ns_group=ns_group))
            kept = self.rung()
            split = ask(dataclasses.replace(start, narrow=False, ns_group=ns_group), "the step does not fit "
                        "the devices with its narrow copies; compiling it again without them (%s)")
            if split is not None:
                self.said(named)
                return split
            narrow = not self.freed(kept, self.rung())
            self.variants = True
            if narrow:
                # Dropped, the step fell no less short at the split both missed at: on a GPU the step casts
                # its weights whole either way (a GH200: 68.352 MB without its copies, 68.344 MB with them),
                # so the copies come back, their 1-2% of a step kept, and the ladder climbs with them.
                _log.warning("the step's narrow copies free nothing it needs: it keeps them and climbs on")
        for rung in (rung for group in self.ladder(start, narrow, ns_group) for rung in group):
            if rung in self.asked:
                continue
            split = ask(rung, "the step does not fit the devices; compiling it again at %s")
            if split is not None:
                self.said(named)
                return split
        self.said(named)
        return None

    def below(self, candidates: Sequence[int], origin: Rung, plan: Rung) -> tuple[Probe, ...]:
        """Each rung the ladder reads before the plan's from `origin`, the rung the run was configured at,
        cheapest first, that no search asked (`asked`; a plan taken from its record asked none), at the
        coarsest of `candidates` it fits at, compiled: the copies' drop at `origin`'s rung, then each rung
        the one climb visits (`ladder`) before the plan's, a head's tile or XLA's defaults at less
        recomputation, which a run settled recomputing in its fewest compiles (`searched`) leaves for a plan
        made offline to tell (`dew.training.ahead.compile_ahead`'s frontier). `plan` is the admitted rung,
        whatever the frontier above it asked since (`above`). Every rung's twin under XLA's defaults is read,
        whichever split's options it differs from (`step_options` reads a microbatch's size): where it is the
        step's own program at its split, its entry states the same figures as the own one's. None where the
        plan is no rung of that climb (the run's own, or one past a release the climb does not read). The
        objective is left at the last rung read, the run's to restore."""
        self.at(origin)
        # The climb as the configured run reads it: one that set its own remat climbs its head alone, whatever
        # remat a record then moved it to.
        record = recompute_record(self.objective)
        explicit = any(entry is not None for entry in (record if isinstance(record, list) else [record]))
        start = dataclasses.replace(origin, ns_group=plan.ns_group)
        copies = start.narrow and any(candidate.copies for candidate in self.lowered.values())
        order = [*([dataclasses.replace(start, narrow=False)] if copies else []),
                 *(rung for group in self.ladder(start, start.narrow, start.ns_group, remat_set=explicit,
                                                 defaults=(False, True)) for rung in group)]
        if plan not in order:
            return ()
        # A rung read here that misses is no checkpoint's rung missing (`fits_here`).
        self.resumed = None
        found = []
        for rung in order[:order.index(plan)]:
            if rung in self.asked:
                continue
            self.at(rung)
            # Every split in turn, coarsest first, to the first that fits: no forecast passes one over
            # (`walked` may, its miss's figures no bound on what the compiler allocates), as this report
            # states the coarsest split each fits at.
            if next((split for split in sorted(candidates) if self.fits_here(split)), None) is not None:
                found.append(self.probes[-1])
        return tuple(found)

    def recomputing(self, start: Rung, narrow: bool, ns_group: int | None) -> list[Rung]:
        """The ladder's remat rungs above `start`'s, each at the run's own head and the step's own options
        (each remat place's first, `ladder`), least recomputation first, reading the narrow copies where
        `narrow` and joining the Newton-Schulz stacks `ns_group` at a time: none for a run that states its
        remat, whose ladder is its head's rungs alone."""
        return [group[0] for group in self.ladder(start, narrow, ns_group)
                if group[0].modules != start.modules and not group[0].xla_defaults]

    def said(self, named: list[str]) -> None:
        """Log every rung the search asked (`named`, as `rung` names each), in the ladder's order, with the
        least it was short by where it missed (its misses' figures, `misses`); the last fits where one did."""
        def short(rung: str) -> str:
            missed = [miss.shortfall // self.factor for (at, _), miss in self.misses.items() if at == rung]
            return f"{rung}: short by {min(missed)} bytes" if missed else f"{rung}: fits"
        _log.info("the rungs the step's search asked, in the ladder's order: %s",
                  "; ".join(short(rung) for rung in named))

    def freed(self, kept: str, dropped: str) -> bool:
        """Whether a release (the narrow copies dropped, the Newton-Schulz stacks parted) brought the step
        nearer to fitting: at `dropped`, the rung released, at the split it last missed at, it fell less short
        than at `kept`, the rung before it, at that split, what was released and the temporaries its absence
        changes counted together; True where either compile read no memory. Every process gathers its
        verdict, as at every miss (`after`), so all keep it or none."""
        latest = next((miss for (rung, _), miss in reversed(self.misses.items()) if rung == dropped), None)
        before = None if latest is None else self.misses.get((kept, latest.split))
        held = before is not None and latest is not None and latest.shortfall >= before.shortfall
        return not any(self.gather(held))

    def found(self, split: int) -> Planned:
        """The plan at `split` on the rung the search stands at, as it fitted there (`fits_at`)."""
        at = self.rung_now()
        fitted, (candidate, executable, options), sampled = self.fits_at[self.rung()]
        assert fitted == split, (fitted, split)
        return Planned(split, candidate, executable, options, self.current(), at, sampled=sampled)
