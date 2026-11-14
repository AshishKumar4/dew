"""A run's step compiler (`StepCompiler`): where the fit ladder stands for the run, rung by rung and shape by
shape, and the links of its mesh's axes, and the compile of its step from what the run states of it
(`StepSettings`) and reads (`Stepping`), its result one value (`Compiled`)."""

from __future__ import annotations

import dataclasses
import functools
import logging
from collections.abc import Callable, Hashable, Mapping
from typing import TYPE_CHECKING, NamedTuple, Protocol

import jax
import numpy as np
from flax import linen as nn
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew._records import JSON, boolean, json_value, listed, record
from dew.batch import CLOSES_WINDOW, Batch
from dew.distributed.batches import batch_layout, batch_shardings, batch_signature
from dew.distributed.placement import Validation
from dew.mesh import SEQUENCE_AXIS, TENSOR_AXIS
from dew.nn.kernels.generation import gpu_target
from dew.nn.sharding import Link, Schedule
from dew.objectives.base import Objective
from dew.state import Accumulation, TrainState, apart, occupied, together
from dew.telemetry.instrumentation import StepFlops
from dew.training._candidates import Stepping, candidate_for, lowered_for
from dew.training._lowering import Lowered, routed
from dew.training.distributed import link_figure, link_figures, resumed_link_figures
from dew.training.memory import (
    climb_to,
    head_tile_of,
    layer_gathers,
    local_devices,
    memory_record,
    rung_record,
    step_fits,
    step_options,
)
from dew.training.model_flops import batch_consumed, batch_tokens, model_batch
from dew.training.narrow import holds_copies, narrowing
from dew.training.phase_passes import PhasePasses
from dew.training.planning import (
    Candidate,
    FitSearch,
    Planned,
    PlanRefused,
    Trials,
    held_objects,
    released,
    settled,
    splits,
    whole_step_program,
)
from dew.training.profiling import pgle_runs
from dew.training.rungs import keep_rung, plan_flops, recorded_rung
from dew.training.transaction import Transaction, outcome

if TYPE_CHECKING:
    from dew.training.ahead import AheadOfTime
    from dew.training.transaction import Committed

_log = logging.getLogger(__name__)


def freshly_held(state: TrainState) -> TrainState:
    """`state` as a fresh state holds it at an update's end (a phase of other passes planned afresh,
    `dew.training.trainer.Trainer._compiled_for`): the narrow copies another program's step made
    (`TrainState.compute`, made again from the variables, a pure function of them) released, their buffers
    deleted, for the run holds the state it hands over until the new step returns: only dropped, they stayed
    on the devices while the new step was planned (5.3 GiB of a Qwen3.5-4B's on a GH200, M-MC-3k). Its empty
    window it keeps: the new step's window is built over it, its buffers donated (`laid_out`), so the plan
    counts none of the window built beside the step but its leaves of other shapes (`taken`), where building
    one afresh beside the old one's 16.7 GiB refused the run's third phase (M-MC-3q)."""
    released([state.compute])
    return dataclasses.replace(state, compute=None)


def laid_out(state: TrainState, window: int) -> tuple[TrainState, Accumulation | None]:
    """`state` laid out for a step of `window` microbatches an update, and the window it let go: as it is
    where its window holds that many, else, at an update's end, its empty window let go (a commit cleared
    it but for its gradient sum, which the next window's first microbatch replaces unread) for the step to
    lay out its own over it (`builds_window`, `window_of`: each leaf of its shape and dtype donated to the new
    window's, `taken`, the rest released), its count of accepted microbatches read in the new window's
    (`microstep`, whose windows and steps the transaction counts by it: the same updates and steps). A state
    inside an update is refused another window: its open window is laid out for the one it pools
    (`occupancy`)."""
    size = int(state.window_size)
    if size == window:
        return state, None
    if occupied(state.window):
        raise ValueError(f"the state pools an open window of {size} microbatches, and the step it is handed "
                         f"to reads {window} an update: a step of another split begins at an update's end")
    # Counted on the host and placed as the counts were: an operation on the device arrays would compile a
    # program of its own in a run handed its step compiled ahead (row 5781).
    microstep = int(state.microstep) // size * window
    return dataclasses.replace(
        state, accumulation=None,
        microstep=jax.device_put(np.asarray(microstep, state.microstep.dtype), state.microstep.sharding),
        window_size=jax.device_put(np.int32(window), state.window_size.sharding)), state.window


def admitted_placed(planned: Planned, mesh: Mesh) -> tuple[bool, int | None]:
    """Whether `planned`, compiled before the state was placed (`StepCompiler.planned_unplaced`), fits the
    devices of `mesh` now that it is (`step_fits`), its step and, where it samples, its sampled step, and
    the step's headroom there."""
    held, built = planned.candidate.held, planned.candidate.built
    fit, headroom = step_fits(planned.executable, mesh, held, built)
    if fit and planned.sampled is not None:
        fit = step_fits(planned.sampled.executable, mesh, held, built)[0]
    return fit, headroom


class Unplaced(NamedTuple):
    """A fresh run's first step planned before its state was placed (`StepCompiler.planned_unplaced`): the
    signature of the batch it was planned over, its plan and what its compile found, and the record of its
    rung (`keep_rung`), kept only once the devices hold it with the state placed (`admitted_placed`)."""

    signature: Hashable
    planned: Planned
    compiled: Compiled
    keep: Callable[[], None]


class CompiledStep(Protocol):
    """A call returns the stepped state, the scalar loss, the objective's whole
    report (`Aux.metrics`), whether that loss was finite, and whether the
    step's microbatches were accepted, and the transaction's commit telemetry (None for a custom step);
    `sampled`, the commit telemetry also summarizes what the run inspects (`Committed.inspected`): the state
    it is handed laid out for its window first (`laid_out`)."""

    def __call__(self, state: TrainState, batch: Batch, /, sampled: bool = False) -> tuple[
            TrainState, jax.Array, dict[str, jax.Array], jax.Array, jax.Array, Committed | None]: ...


@dataclasses.dataclass(frozen=True)
class Compiled:
    """What a compile of a run's step found: the microbatches it reads a batch in (`split`), the record of
    its plan (`plan`, every compile its search made), the check that it fits (`memory`, `memory_record`),
    the executable it compiled and its FLOPs (`step_flops`), the links of the axes it traced with, the state
    it holds between steps as the layout validates it (`validation`), and the step itself (`step`), none
    where a compile ahead reads its plan alone. A CPU-owned run's compile plans nothing, and finds none of
    the rest."""

    split: int | None
    plan: JSON = None
    memory: JSON = None
    executable: jax.stages.Compiled | None = None
    step_flops: StepFlops | None = None
    links: Mapping[str, Link] = dataclasses.field(default_factory=dict)
    validation: Validation | None = None
    step: CompiledStep | None = None


def resumed_stepping[L, E](stepping: Stepping[L, E], window_size: int, *, held: bool, pools: bool,
                           replicas: bool, afresh: bool = False) -> Stepping[L, E]:
    """`stepping` at the split a state of `window_size` microbatches an update steps at: a state that pools
    a window part filled (`pools`), or DiLoCo's `replicas` once they have stepped, each counting their
    microbatches in their split's windows, which no other split reads, bind it, a stated other refused; a
    state that holds an empty window (`held`), or was laid out at more than one microbatch, takes it unless
    the run states another or its step is planned `afresh` (a phase of other passes at an update's end,
    `StepCompiler.afresh`), the step it is handed then laying its own window out over it (`laid_out`). A run's
    compile and a resume's step compiled ahead (`dew.training.ahead.compile_ahead`) both read it."""
    accumulation = stepping.settings.accumulation
    split, unaligned = divmod(window_size, accumulation)
    if unaligned or ((pools or replicas) and stepping.shape.split not in (None, split)):
        raise ValueError(f"the state pools windows of {window_size} microbatches, and this run updates "
                         f"every {accumulation} steps of {stepping.shape.split} microbatches")
    if pools or replicas or (not afresh and (held or split > 1) and stepping.shape.split is None):
        return dataclasses.replace(stepping, shape=dataclasses.replace(stepping.shape, split=split))
    return stepping


class StepCompiler[L, E]:
    """Where a run's fit ladder stands and the links its traces read, and the compile of its step
    (`compile`): its objective's rung as the first compile found it and as each shape compiled at it, the
    rungs a resumed checkpoint recorded, whether the step keeps XLA's default options and its narrow
    copies, each mesh axis's measured bandwidth, the links a run states (`links`) and a step compiled ahead
    (`ahead`), until a compile takes it. Everything else a compile reads is its call's."""

    def __init__(self, objective: Objective[L, E], links: Mapping[str, Link] | None,
                 ahead: AheadOfTime | None, phase_passes: PhasePasses[L, E]):
        self.objective = objective
        self.phase_passes = phase_passes
        self.ahead = ahead
        self._stated_links = dict(links or {})
        self._bandwidths: dict[tuple[Mesh, str], float | None] = {}
        # The ladder's rungs past the objective's head and remat that a step fit at (no narrow copies, XLA's
        # defaults) stay for later compiles; a resumed run starts at its checkpoint's rung (`climb_to`).
        self._narrow, self._xla_defaults = True, False
        self._resumed_rung: JSON = None
        # The rung each newly met shape starts from, as the first compile found the objective: its modules,
        # head tile, XLA option choice and record (`at_base_rung`); then the rung each shape compiled at, by
        # layout, and the rungs a resumed checkpoint recorded for its shapes.
        self._base_rung: tuple[tuple[nn.Module, ...], tuple[int, int] | None, bool, dict[str, JSON]] | None
        self._base_rung = None
        self._shape_rungs: dict[str, JSON] = {}
        self._resumed_shapes: dict[str, JSON] = {}
        self._resumed_links: dict[str, float | None] = {}
        # A fresh run's first step planned over its state's shapes before the state is placed
        # (`planned_unplaced`), by the signature of the batch it was planned over: its first compile's.
        self._unplaced: Unplaced | None = None
        # Whether the next compile plans a phase afresh at an update's end
        # (`dew.training.trainer.Trainer._compiled_for`): the empty window its state holds binds no split
        # (`resumed_stepping`), the new step's window built over it.
        self.afresh = False
        # The executables a search before the state was placed compiled where it held no plan
        # (`planned_unplaced`), which the run's first compile's search takes in place of compiling them again.
        self._executables: dict | None = None

    @property
    def narrow(self) -> bool:
        """Whether the step keeps its narrow copies, at the rung it stands at."""
        return self._narrow

    @property
    def stated_links(self) -> Mapping[str, Link]:
        """The links the run states by axis, measuring none of them (`links`)."""
        return self._stated_links

    def rung(self) -> dict[str, JSON]:
        """The fit ladder's rung this trainer's step compiles at: the
        objective's head tile, its trained modules' remat (`recompute_record`) and whether
        the step keeps XLA's default options, the half rung `compile` climbs
        to before the next. A checkpoint records it, since each decides the program a step runs."""
        return rung_record(self.objective, self._xla_defaults, narrow=self._narrow)

    def checkpoint_rung(self) -> JSON:
        """The rung a checkpoint records: the run's own, which every newly met
        shape starts from (`rung`'s fields as the first compile found them),
        and under `shapes` the rung each shape compiled at, by its layout
        (`batch_layout`). The split is the state's (`window_size`), and its
        link figures by placement (`link_figures`), a resume's own there."""
        base = self.rung() if self._base_rung is None else self._base_rung[3]
        links = link_figures(self._bandwidths)
        return {**base, 'shapes': dict(self._shape_rungs), **({'links': links} if links else {})}

    def resume_rung(self, rung: JSON) -> None:
        """Take a checkpoint's rung (`checkpoint_rung`): the objective climbs
        to the run's own rung, and each shape the checkpoint names climbs to
        its own when it is compiled (`at_base_rung`). A rung recorded before
        shapes had their own names none, so every shape starts at it."""
        self.climb_to(rung)
        self._resumed_links = {} if rung is None else resumed_link_figures(rung)
        if rung is not None:
            shapes = record(record(rung, 'rung').get('shapes', {}), 'rung shapes')
            self._resumed_shapes = {name: json_value(shape, f'rung of {name}')
                                    for name, shape in shapes.items()}

    def at_base_rung(self, batch: Batch) -> JSON:
        """Put the objective at the rung a newly met shape starts from, and
        return the rung a resumed checkpoint recorded for this shape, or None.

        Each shape fits on its own. A run bucketed by length meets a long bucket that climbs, and when that
        rung stayed, every later, shorter bucket compiled at it too, though it fitted with less recomputation
        (in one real mix, six shapes ran at 'minimal' that fitted with none, 15% slower a token). So each new
        shape starts where the first compile found the objective, its configured remat and head or the rung
        its checkpoint trained on, and climbs only as far as it must. A shape the checkpoint names climbs to
        that shape's own rung, so a resumed run compiles each shape as the run it continues did."""
        if self._base_rung is None:
            self._base_rung = (tuple(entry.module for entry in self.objective.program_key()),
                               head_tile_of(self.objective), self._xla_defaults, self.rung())
        else:
            modules, tile, defaults, rung = self._base_rung
            self.objective.substitute(list(modules))
            self.objective.restore_head(tile, fused=bool(rung['head_fused']))
            self._xla_defaults, self._narrow = defaults, bool(rung['narrow'])
            # The base modules at the pass count the step being compiled runs, whatever they were taken at.
            self.phase_passes.restored(self.objective)
        recorded = self._resumed_shapes.get(batch_layout(batch))
        if recorded is None:
            return None
        self.climb_to(recorded)
        return self.rung()

    def climb_to(self, rung: JSON) -> None:
        """Move the objective up to `rung`, a checkpoint's (`rung`), where
        it is above where the objective stands, never below it.

        The fit check reads the free memory its own process finds, and one that restores a state can find more
        than the one that built it and pick a lighter rung, another program. So the resumed run compiles the
        rung its checkpoint trained on, and climbs further only where that
        does not fit; a ladder rung the record does not name it never climbed."""
        if rung is None:
            return
        fields = record(rung, 'rung')
        climb_to(self.objective, fields)
        self._xla_defaults = self._xla_defaults or boolean(
            fields.get("xla_defaults", False), "rung xla_defaults"
        )
        self._narrow = self._narrow and boolean(fields.get("narrow", True), "rung narrow")
        self._resumed_rung = self.rung()


    @functools.cached_property
    def _ahead_links(self) -> dict[str, Link]:
        """The links the step compiled ahead traced with, at the run's first trace (`AheadOfTime.over`)."""
        return {} if self.ahead is None else self.ahead.over(self._stated_links)

    def links(self, mesh: Mesh) -> dict[str, Link]:
        """The links of `mesh`'s tensor and sequence axes, those it splits, for one trace. Every process
        lists the mesh's devices alike and `link_bandwidth` agrees one figure across the pool, so every
        process decides alike and compiles the same program. The peak is the fastest device's, on which
        the FLOPs a spread saves take the least time. Each axis's bandwidth is measured once per mesh, and
        not at all for a device the peak table does not name (a CPU's among them), where it decides
        nothing, or over a compile-only GPU (`dew.training.ahead`), which no process holds to measure; a
        stated link (`links`) as stated, a step compiled ahead's over it (`AheadOfTime.over`)."""
        devices = list(mesh.devices.flat)
        peaks = [peak for device in devices if (peak := gpu_target(device.device_kind).bf16_peak) is not None]
        peak = max(peaks) if len(peaks) == len(devices) else None
        links = {}
        for axis in (TENSOR_AXIS, SEQUENCE_AXIS):
            if mesh.shape.get(axis, 1) == 1:
                continue
            stated = self._ahead_links.get(axis, self._stated_links.get(axis))
            if stated is not None:
                links[axis] = Link(stated.bytes_per_second, stated.flops_per_second)
                self._bandwidths[mesh, axis] = stated.bytes_per_second
                continue
            if not all(device in jax.devices() for device in devices):
                links[axis] = Link(None, peak)
                continue
            if (mesh, axis) not in self._bandwidths:
                self._bandwidths[mesh, axis] = None if peak is None else link_figure(
                    mesh, axis, self._resumed_links)
            links[axis] = Link(self._bandwidths[mesh, axis], peak)
        return links

    def compile(self, stepping: Stepping, state: TrainState, batch: Batch,
                trials: Trials | None = None) -> Compiled:
        """Compile a transaction over the state and one already-produced global batch (`Trainer.compile`),
        the step and what its compile found (`Compiled`).

        The plan (`dew.training.planning`) takes the fewest microbatches at least recomputation, a
        transaction pooling `split` x `accumulation`. Batches split as the first plan or the pooling
        state's window holds, a state saved between updates at the split it was saved at unless the run
        states another; later shapes climb the ladder alone, but for a phase of other passes beginning at
        an update's end, which its trainer plans afresh (`Trainer._planned_afresh`). No admitted plan compiles
        or times optional alternatives, even with `trials`: speed selection is offline. Where JAX's AutoPGLE
        is on (`jax_enable_pgle`), the plan admitted runs its first runs on a trial, `trials`' or, where
        none are given, a copy of the state beside it, refused where the devices hold no room for it
        (`Trials.profiled`), and its program compiles again guided by their profile (`Planned.guided`),
        the executable it replaces released with its program (`Planned.released`). A compile-only GPU holds
        the memory stated for it (`dew._devices.device_memory`).
        """
        settings = stepping.settings
        if (target := settings.lower_for) is not None:
            # Shapes are all a lowering reads, so the state may be abstract (`jax.eval_shape`); it lowers at
            # the base rung.
            self.at_base_rung(batch)
            raise Lowered(target, *routed(lambda: lowered_for(stepping, state, batch, target, self.narrow)))
        # A state pools its window's microbatches where it holds one part filled: the window, laid out for
        # its split, binds it. Read on the host: a reduction on the devices compiles a program of the
        # occupancy's placement, a restored one's another than a fresh one's, which a resume compiles anew.
        pools = bool(occupied(state.window))
        afresh, self.afresh = self.afresh, False
        stepping = resumed_stepping(stepping, int(state.window_size), held=state.window is not None,
                                    pools=pools, replicas=state.inner is not None and int(state.step) > 0,
                                    afresh=afresh)
        if (state.scale is not None) != settings.dynamic_scale:
            raise ValueError("checkpoint dynamic-scaler configuration differs from this trainer")
        if settings.host_master:
            if self.ahead is not None or settings.inspection is not None:
                raise ValueError("a step compiled ahead or inspected runs where the devices hold the state; "
                                 "this layout keeps the variables on the host")
            return Compiled(stepping.shape.split, step=self._compile_host(stepping, state, batch))
        held, self._unplaced = self._unplaced, None
        executables, self._executables = self._executables, None
        mesh = settings.device_mesh
        fit, headroom = (False, None) if held is None or held.signature != batch_signature(batch) else (
            admitted_placed(held.planned, mesh))
        if held is not None and fit:
            # Planned and compiled while the state was built (`planned_unplaced`), and judged now the state is
            # placed: its plan, kept and its rung recorded, its record holding what the devices read.
            planned, compiled = held.planned, held.compiled
            held.keep()
            plan = record(compiled.plan, "plan")
            *earlier, kept = listed(plan["probes"], "probes", read=record, what="probes")
            probes = [*earlier, {**kept, "headroom": headroom}]
            memory = memory_record(planned.executable, local_devices(mesh), planned.candidate.held, fits=True)
            compiled = dataclasses.replace(compiled, plan={**plan, "probes": probes}, memory=memory)
        else:
            # Where the devices, the state placed, hold no block for it, the plan's search runs here, on them.
            planned, compiled, keep = self._plan(stepping, state, batch, trials, executables=executables)
            keep()
        return dataclasses.replace(compiled, step=self.runner(stepping, planned.released(),
                                                              settings.device_mesh))

    def planned_unplaced(self, stepping: Stepping, state: TrainState, batch: Batch,
                         unplaced: int) -> None:
        """A fresh run's first step planned and compiled over `state`, its shapes, before the `unplaced`
        bytes a device holds of it are placed, so the step compiles while the run's start builds the state
        (`Trainer._opened`), not after it: each candidate is judged on the devices' free bytes in all beside
        the state's (`FitSearch.unplaced`), and the run's first compile over `batch` judges the one taken
        against their allocators once the state is on them (`compile`, `admitted_placed`). It holds a plan
        only where its search refused nothing on the way to it: the bytes the devices
        hold before the state is placed may include arrays the start copies and lets go, so a candidate
        refused then may fit once it is, and the plan the run takes is the one its placed devices pick
        (a phased run's first phase read whole, bit for bit the run planned once its state was placed). Where
        it refused one, or none fits, that compile plans once the state is placed, over the traces this
        search kept (`candidate_for`), and the executables it compiled (`FitSearch.compiled`): nothing is
        traced or compiled twice."""
        executables: dict = {}
        self._executables = executables
        try:
            planned, compiled, keep = self._plan(stepping, state, batch, None, unplaced=unplaced,
                                                 executables=executables)
        except PlanRefused as refused:
            _log.info("the step fits no plan before its state is placed, planned once it is: %s", refused)
            return
        plan = record(compiled.plan, "plan")
        probes = listed(plan["probes"], "probes", read=record, what="probes")
        if any(not boolean(probe["fits"], "fits") for probe in probes):
            _log.info("the step's plan before its state is placed refused a candidate on the way; planned "
                      "once the state is placed")
            return
        self._unplaced = Unplaced(batch_signature(batch), planned, compiled, keep)

    def plan(self, stepping: Stepping, state: TrainState, batch: Batch,
             trials: Trials | None, frontier: bool = False) -> tuple[Planned, Compiled]:
        """The shared device-step planner: the plan it admitted and what its compile found (`Compiled`, its
        step none: the caller reads its lowering before releasing it, `compile`, or serializing its
        executable, `compile_ahead`), with `frontier` the rungs above it (`FitSearch.above`), its rung
        recorded (`keep_rung`)."""
        planned, compiled, keep = self._plan(stepping, state, batch, trials, frontier)
        keep()
        return planned, compiled

    def _plan(self, stepping: Stepping, state: TrainState, batch: Batch, trials: Trials | None,
              frontier: bool = False, *, unplaced: int | None = None, executables: dict | None = None
              ) -> tuple[Planned, Compiled, Callable[[], None]]:
        """`plan`, over a state whose devices hold `unplaced` bytes a device holds of it none of yet where
        given (`FitSearch.unplaced`), its search taking and keeping its executables in `executables` where
        given (`FitSearch.compiled`), and the record of its rung, which the caller keeps once it takes the
        plan (`keep_rung`)."""
        settings = stepping.settings
        mesh = settings.device_mesh
        links = self.links(mesh)
        shaped = self.at_base_rung(batch)
        resumed, self._resumed_rung = self._resumed_rung if shaped is None else shaped, None
        with stepping.traced_on(mesh, links) as schedule, held_objects():
            planned, flops, plan, validation, keep = self._planned(stepping, state, batch, mesh, schedule,
                                                                   resumed, trials, frontier, unplaced,
                                                                   executables)
            split, candidate = planned.split, planned.candidate
            self._shape_rungs[batch_layout(batch)] = self.rung()
            memory = memory_record(planned.executable, local_devices(mesh), candidate.held, fits=True)
        return planned, Compiled(split, plan, memory, planned.executable, flops and split * flops, links,
                                 validation), keep

    def runner(self, stepping: Stepping, planned: Planned,
               mesh: Mesh) -> CompiledStep:
        """The step `planned` compiled, which reads a batch in its microbatches."""
        settings = stepping.settings
        # The program compiled above, not the jit: a call through the jit traces and compiles its own,
        # without the step's compiler options, which was 25 s of an A100's cold start.
        split, candidate = planned.split, planned.candidate
        parts = planned.parts(mesh)
        window = split * settings.accumulation

        # The window size of the state this step last returned: a state it is handed with another (a
        # state another program's step returned, a cooldown's restored, a public caller's) is laid out for
        # this step's window first (`laid_out`).
        returned: list[jax.Array | None] = [None]
        # A split step's whole from its microbatches' outputs, its loss as an unsplit step logs it.
        whole = whole_step_program(self.objective)

        def run(current, batch, sampled: bool = False):
            # A sampled step summarizes what the run inspects (`Planned.sampled`); every other, the plan's.
            executable = planned.sampled.executable if sampled and planned.sampled is not None else (
                planned.executable)
            donor = None
            if current.inner is None and current.window_size is not returned[0]:
                current, donor = laid_out(current, window)
            with stepping.traced_on(mesh):
                # Each program plans its copies (`at_base_rung`), and a step reads the state with its own: a
                # state another program's step left holds that program's (a phase of other passes, row
                # 5560), which are made again from its variables, a pure function of them.
                if not candidate.copies:
                    # Copies the step reads none of, let go with their buffers: the state the run holds
                    # would keep them through the step's first call.
                    released([current.compute])
                    current = dataclasses.replace(current, compute=None)
                elif not holds_copies(current.compute, candidate.copies):
                    # The copies another program planned, which the step consumes with the state, are let go
                    # before its own are made: the plan's fit counts its own alone (`_candidates`).
                    released([current.compute])
                    # A run handed a step compiled ahead runs the program compiled with it: `fit` runs its
                    # steps within the hand (`Trainer._taking`).
                    copies = narrowing(candidate.copies, self.objective.towers(), candidate.shardings.compute)
                    current = dataclasses.replace(current, compute=copies(current.variables))
                if stepping.builds_window(current, candidate, window):
                    # The window, and with it the split, is the state's from here on, built over the one it
                    # replaces (a window laid out for another split, or a ramp's stage's of other shapes).
                    current = dataclasses.replace(stepping.window_of(current, candidate, window, donor),
                                                  window_size=jax.device_put(np.int32(window)))
                    # The moving state alone: the frozen tree stays the arrays the run holds.
                    moving, frozen = apart(current)
                    current = together(jax.device_put(moving, apart(candidate.shardings)[0]), frozen)
                elif donor is not None:
                    # A step that keeps no window (one microbatch an update) lets the one it was handed go.
                    released([donor])
                outputs = []
                moving, frozen = apart(current)
                for part in parts:
                    moving, *output = executable(moving, frozen, batch, part)
                    outputs.append(tuple(output))
                stepped = together(moving, frozen)
                returned[0] = stepped.window_size
                # A split step's metrics through its whole, which pools them and reads each as its value.
                return (stepped, *(whole(outputs) if split > 1 else outputs[0]))
        return run

    def _planned(self, stepping: Stepping, state: TrainState, batch: Batch,
                 mesh: Mesh, schedule: Schedule, resumed: JSON, trials: Trials | None,
                 frontier: bool, unplaced: int | None, executables: dict | None
                 ) -> tuple[Planned, StepFlops | None, JSON, Validation, Callable[[], None]]:
        """The plan `compile` reads `batch` in (`FitSearch`), from the run's splits (`splits`), the first that
        fits, guided under AutoPGLE (`Planned.guided`), with `plan` its record and the objective at its rung,
        a microbatch's FLOPs (`plan_flops`) and the state it holds between steps, as the layout validates
        it. A plan recorded beside the compilation cache
        (`recorded_rung`) is the first candidate, kept untimed; `resumed` is a checkpoint's rung; `frontier`
        asks the rungs above."""
        settings = stepping.settings
        replicated = NamedSharding(mesh, P())
        placement = batch_shardings(mesh, batch)
        candidates = splits(stepping.shape.rows, batch, mesh, settings.microbatches, stepping.shape.split)

        def lower(split: int, narrow: bool) -> Candidate:
            return candidate_for(stepping, state, batch, schedule, placement, replicated, split, narrow)

        # Read before anything traces, so a recorded rung is the first one lowered.
        record_path, key, recorded = recorded_rung(self.objective.program_key(), (state, batch), candidates,
                                                   mesh, settings.step)
        if self.ahead is not None and self.ahead.plan is not None:
            recorded = self.ahead.plan
        before = self.rung()
        self.climb_to(recorded)
        self._resumed_rung = None
        if self.rung() != before:
            _log.warning("taking the saved plan, from the ahead step or %s", record_path)
        first = None if recorded is None else record(recorded, 'recorded plan').get('split', 1)
        # The plan's one sizing of its largest layer, which every candidate's options read.
        gathered = layer_gathers(self.objective, state.variables, mesh.devices.size)
        search = FitSearch(
            lower, lambda candidate: step_options(
                self.objective, candidate.micro, candidate.prepared.variables, mesh, gathered),
            self.objective, mesh, self._xla_defaults, self._narrow, None if self._base_rung is None
            else self._base_rung[1], resumed, ahead=self.ahead,
            unplaced=unplaced, compiled={} if executables is None else executables)
        planned, probes = search.admitted(
            candidates, candidates.index(first) if isinstance(first, int) and first in candidates else 0)
        above = search.above(candidates, planned.split) if frontier else None
        for probe in above or ():
            _log.info("above the plan, fitting at fewer microbatches: %s", probe.named())
        # Consume the artifact only after its program was accepted; a refusal leaves it pending.
        self.ahead = search.ahead
        self.objective.substitute(list(planned.modules))
        self.objective.restore_head(planned.tile, fused=planned.fused)
        self._xla_defaults, self._narrow = planned.xla_defaults, planned.narrow
        if runs := pgle_runs():
            trials = Trials(state, batch, owned=False) if trials is None else trials
            run = self.runner(stepping, planned, mesh)
            relower = functools.partial(lower, planned.split, planned.narrow)
            planned = planned.guided(trials.profiled(run, planned, mesh, runs), runs, mesh, relower)
        # The state the chosen plan holds between steps, its accumulator and copies included; the
        # plan's fit check counted them before any is built (`Candidate.held`).
        validation = stepping.validation(planned.candidate.prepared)
        chosen = {**self.rung(), 'split': planned.split}
        flops, counted = plan_flops(recorded, planned.executable)
        keep = (functools.partial(keep_rung, record_path, key, {**chosen, 'flops': counted})
                if record_path is not None else lambda: None)
        plan = settled(planned, probes, search, above)
        if stepping.staging.read_whole:
            # A transformation given built that kept every layer whole on every stage (`StageLocal.of`).
            plan = {**plan, "layers_whole_for": list(stepping.staging.read_whole)}
        search.told(planned)
        if planned.split != candidates[0]:
            _log.info("the step fits the devices in the plan %s", plan)
        return planned, flops, plan, validation, keep

    def _compile_host(self, stepping: Stepping, state: TrainState, batch: Batch
                      ) -> CompiledStep:
        settings = stepping.settings
        from dew.distributed.host import transfer
        from dew.training.execution import HostExecution, frozen_trees, stack_sites
        cpu = settings.state_mesh
        sites = stack_sites(self.objective, state.variables)
        execution = HostExecution(self.objective, settings.layout, settings.device_mesh, cpu, sites,
                                  device=stepping.shape.device,
                                  gathered=layer_gathers(self.objective, state.variables,
                                                         settings.device_mesh.devices.size))
        cpu_batch = transfer(batch, batch_shardings(cpu, batch))
        with stepping.traced_on(cpu) as schedule:
            _, shapes = stepping.loss_shape(state, cpu_batch, settings.accumulation)
            stepping.staging.refuse_unpipelined(schedule, self.objective)
            prepared = stepping.initialize_accumulation(state, cpu_batch, shapes,
                                                        window=settings.accumulation, shape_only=True)
            placement = stepping.shardings(prepared)
            transaction = Transaction(self.objective, settings.optimizer, settings.accumulation, shapes,
                                      record=settings.record, examples=stepping.shape.examples)
            body = transaction.step(realize=execution.realize, host=True,
                                    frozen=tuple(path for path, _, _ in frozen_trees(state.variables, sites)))

        def run(current, batch, sampled: bool = False):
            # A host-master run inspects nothing (`compile` refuses it), so no step of it is sampled.
            with stepping.traced_on(cpu):
                batch = transfer(batch, batch_shardings(cpu, batch))
                if settings.accumulation > 1 and (current.accumulation is None or (
                        (stepping.shape.ramp or stepping.shape.stage_starts)
                        and not current.accumulation.replays(prepared.accumulation))):
                    current = stepping.initialize_accumulation(current, batch, shapes,
                                                               window=settings.accumulation)
                # The device steps run in the accelerator's forward (`HostExecution`).
                advanced, loss, aux, committed = body(current, model_batch(batch), 0, batch_tokens(batch),
                                                      batch.get(CLOSES_WINDOW))
                advanced, *stepped = outcome(current, advanced, loss, aux, committed, batch, 1,
                                             batch_consumed(batch), split=1)
                # A leaf already where its placement says stays the array it
                # is: the resident frozen collection crosses no boundary.
                return transfer(advanced, placement), *stepped
        return run

    # ------------------------------------------------------------------ The loop
    # ------------------------------------------------------------------
