"""A run's step compiler (`StepCompiler`): where the fit ladder stands for the run, rung by rung and shape by
shape, and the links of its mesh's axes, and the compile of its step from what the run states of it
(`StepSettings`) and reads (`Stepping`), its result one value (`Compiled`)."""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple, Protocol

import jax
import numpy as np
from flax import linen as nn
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew._coordination import from_every_process
from dew._records import JSON, boolean, integer, json_value, listed, record
from dew.batch import CLOSES_WINDOW, Batch
from dew.distributed.batches import batch_layout, batch_shardings, batch_signature
from dew.distributed.placement import Validation
from dew.mesh import SEQUENCE_AXIS, TENSOR_AXIS
from dew.nn.kernels.generation import gpu_target
from dew.nn.kernels.options import hold, trace_context
from dew.nn.kernels.templates import report
from dew.nn.sharding import Link, Schedule
from dew.objectives.base import Objective
from dew.state import Accumulation, TrainState, apart, occupied, together
from dew.telemetry.instrumentation import StepFlops
from dew.training._candidates import Stepping, _shaped, candidate_for, lowered_for
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
from dew.training.phase_passes import Holding, PhasePasses, SampledPasses
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
    from dew.training.interval import Interval
    from dew.training.transaction import Committed

_log = logging.getLogger(__name__)


def freshly_held(state: TrainState) -> TrainState:
    """`state` as a fresh state holds it at an update's end (a phase of other passes planned afresh,
    `dew.training.trainer.Trainer._compiled_for`), its narrow copies released, their buffers deleted, for
    the run holds the state it hands over until the new step returns: only dropped, they stayed on the
    devices while the new step was planned (5.3 GiB of a Qwen3.5-4B's on a GH200, M-MC-3k)."""
    released([state.compute])
    return fresh(state)


def fresh(state: TrainState) -> TrainState:
    """`state` as a fresh state holds it at an update's end, its arrays or its shapes: without the narrow
    copies another program's step made (`TrainState.compute`, made again from the variables, a pure
    function of them), its empty window kept, which the new step's window is built over, its buffers
    donated (`laid_out`), so a plan counts none of the window built beside the step but its leaves of
    other shapes (`taken`), where building one afresh beside the old one's 16.7 GiB refused a run's third
    phase (M-MC-3q). The boundary's state (`freshly_held`) and a later phase's planned off the loop
    (`dew.training.ahead.LaterPlans`) alike."""
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


class Plan(NamedTuple):
    """Where a step's plan stands (`StepCompiler._plans`): the rung it compiles at (`rung_record`) and the
    split it reads its batch in, or a resumed checkpoint's rung for its layout (`resumed`, no split)."""

    rung: JSON
    split: int | None
    resumed: bool = False


class Unplaced(NamedTuple):
    """A fresh run's first step planned before its state was placed (`StepCompiler.planned_unplaced`): the
    signature of the batch it was planned over, its plan and what its compile found, and the record of its
    rung (`keep_rung`), kept only once the devices hold it with the state placed (`admitted_placed`)."""

    signature: Hashable
    planned: Planned
    compiled: Compiled
    keep: Callable[[], None]

    def probes(self) -> tuple[tuple[Mapping[str, object], ...], int]:
        """Its search's probes (`dew.training.planning.Probe.record`) and the index of its own: the last that
        fit at its split and rung (a walk past it may have missed after, a descent's split before it)."""
        probes = listed(record(self.compiled.plan, "plan")["probes"], "probes", read=record, what="probes")
        rung = json.dumps(self.planned.rung, sort_keys=True)
        own = max(index for index, probe in enumerate(probes)
                  if probe["fits"] is True and probe["split"] == self.planned.split
                  and json.dumps({name: probe[name] for name in self.planned.rung}, sort_keys=True) == rung)
        return probes, own


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
    planned: Planned | None = None
    """The plan the step runs, as the run keeps it (`Planned.released`), which a sampled pass count's other
    counts are admitted at (`StepCompiler.admitted_at`); none where a compile ahead reads its plan alone."""


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


def _unlike(cap: Planned, other: Planned) -> list[str]:
    """What `other`'s step holds or returns otherwise than `cap`'s (`StepCompiler.admitted_at`), each leaf
    named by its path: the state it prepares, its window among them, with their shapes, dtypes and
    shardings, and its outputs (`out_info`), the stepped state and the metrics a step's whole pools."""
    def leaves(tree) -> dict[str, tuple]:
        return {jax.tree_util.keystr(path): (getattr(leaf, "shape", None), str(getattr(leaf, "dtype", leaf)),
                                             getattr(leaf, "sharding", None))
                for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]}

    unlike: list[str] = []
    for name, (held, given) in {"state": (cap.candidate.prepared, other.candidate.prepared),
                                "placement": (cap.candidate.shardings, other.candidate.shardings),
                                "outputs": (cap.executable.out_info, other.executable.out_info)}.items():
        ours, theirs = leaves(held), leaves(given)
        unlike += [f"{name}{path}" for path in sorted(ours.keys() | theirs.keys())
                   if ours.get(path) != theirs.get(path)]
    return unlike


@dataclasses.dataclass
class SampledStep:
    """The step of a run that draws its looped model's pass count per microbatch (`SampledPasses`): each
    count's plan at its cap's (`StepCompiler.admitted_at`) and its FLOPs a step, by count (`plans`,
    `flops`), the call running each of its `split` microbatches by its drawn count's program (`counts`,
    `drawn`) over the one state."""

    plans: Mapping[int, Planned]
    flops: Mapping[int, StepFlops | None]
    split: int
    run: CompiledStep | None = None
    counts: list[int] = dataclasses.field(default_factory=list)
    key: tuple[int, ...] | None = None

    def drawn(self, sampled: SampledPasses, key: jax.Array, interval: Interval,
              step: int) -> StepFlops | None:
        """The counts of the next call's microbatches, each drawn at its global position under the run's
        `key` (`SampledPasses.count`), the run's dispatched microbatches past them (`Interval.microbatches`;
        a checkpoint that counted none resumes at its `step`'s, said); and the call's FLOPs, each
        microbatch's at its count's program's."""
        if self.key is None:
            self.key = tuple(int(word) for word in np.asarray(jax.random.key_data(key)).ravel())
        if interval.microbatches is None:
            interval.microbatches = step * self.split
            _log.info("the run's checkpoint counted none of its dispatched microbatches: its pass counts "
                      "draw on from step %d's, microbatch %d", step, interval.microbatches)
        start, interval.microbatches = interval.microbatches, interval.microbatches + self.split
        self.counts = [sampled.count(self.key, start + part) for part in range(self.split)]
        total: StepFlops | None = None
        for count in self.counts:
            each = self.flops[count]
            if each is None:
                return None
            total = (1 / self.split) * each if total is None else total + (1 / self.split) * each
        return total

    def __call__(self, state: TrainState, batch: Batch, /, sampled: bool = False):
        assert self.run is not None and len(self.counts) == self.split, "a call's counts are drawn first"
        return self.run(state, batch, sampled)


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
        # The Newton-Schulz stacks the step joins at a time where the run states none (`ns_group`).
        self._ns_group: int | None = None
        self._resumed_rung: JSON = None
        # The rung each newly met shape starts from, as the first compile found the objective: its modules,
        # head tile, XLA option choice and record (`at_base_rung`).
        self._base_rung: tuple[tuple[nn.Module, ...], tuple[int, int] | None, bool, dict[str, JSON]] | None
        self._base_rung = None
        self._resumed_links: dict[str, float | None] = {}
        # A fresh run's first step planned over its state's shapes before the state is placed
        # (`planned_unplaced`), by the signature of the batch it was planned over: its first compile's.
        self._unplaced: Unplaced | None = None
        # Where each step's plan stands, by the batch's layout and the pass count it runs (`_plan_key`): the
        # rung and split this process's search found, which every later search of that step starts at
        # (`_planned`), so a process searches a step once; or the rung a resumed checkpoint recorded for
        # its layout (`resume_rung`, any pass count: `None`), which its first compile climbs to
        # (`at_base_rung`). A checkpoint records each layout's rung from it (`checkpoint_rung`).
        self._plans: dict[tuple[str, int | None], Plan] = {}
        # Whether the next compile plans a phase afresh at an update's end
        # (`dew.training.trainer.Trainer._compiled_for`): the empty window its state holds binds no split
        # (`resumed_stepping`), the new step's window built over it.
        self.afresh = False
        # The loss's shape-only traces (`loss_shaped`), until the next plan is made: the run's fit check's
        # (`Trainer._whole_validated`) and its plan's candidates' read one trace.
        self._loss_shapes: dict[Hashable, tuple[tuple, bool]] = {}

    @property
    def narrow(self) -> bool:
        """Whether the step keeps its narrow copies, at the rung it stands at."""
        return self._narrow

    @property
    def ns_group(self) -> int | None:
        """The Newton-Schulz stacks of one shape the step joins at a time where its run states none, at the
        rung it stands at (`dew.newton_schulz.released`): None joins them all. Every trace of the step reads
        it (`StepSettings.ns_group`, `Trainer._settings`)."""
        return self._ns_group

    @property
    def stated_links(self) -> Mapping[str, Link]:
        """The links the run states by axis, measuring none of them (`links`)."""
        return self._stated_links

    def rung(self) -> dict[str, JSON]:
        """The fit ladder's rung this trainer's step compiles at: the objective's head tile, its trained
        modules' remat (`recompute_record`) and whether the step keeps XLA's default options
        (`dew.training.planning.FitSearch.ladder`). A checkpoint records it, since each decides the program
        a step runs."""
        return rung_record(self.objective, self._xla_defaults, narrow=self._narrow, ns_group=self._ns_group)

    def checkpoint_rung(self) -> JSON:
        """The rung a checkpoint records: the run's own, which every newly met
        shape starts from (`rung`'s fields as the first compile found them),
        and under `shapes` the rung each shape compiled at, by its layout
        (`batch_layout`). The split is the state's (`window_size`), and its
        link figures by placement (`link_figures`), a resume's own there."""
        base = self.rung() if self._base_rung is None else self._base_rung[3]
        links = link_figures(self._bandwidths)
        shapes: dict[str, JSON] = {layout: plan.rung for (layout, _), plan in self._plans.items()}
        return {**base, 'shapes': shapes, **({'links': links} if links else {})}

    def resume_rung(self, rung: JSON) -> None:
        """Take a checkpoint's rung (`checkpoint_rung`): the objective climbs
        to the run's own rung, and each shape the checkpoint names climbs to
        its own when it is compiled (`at_base_rung`). A rung recorded before
        shapes had their own names none, so every shape starts at it."""
        self.climb_to(rung)
        self._resumed_links = {} if rung is None else resumed_link_figures(rung)
        if rung is not None:
            shapes = record(record(rung, 'rung').get('shapes', {}), 'rung shapes')
            self._plans.update({(name, None): Plan(json_value(shape, f'rung of {name}'), None, resumed=True)
                                for name, shape in shapes.items()})

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
            self._ns_group = None if rung['ns_group'] is None else integer(rung['ns_group'], "rung ns_group")
            # The base modules at the pass count the step being compiled runs, whatever they were taken at.
            self.phase_passes.restored(self.objective)
        recorded = self._plan_of(batch)
        if recorded is None or not recorded.resumed:
            return None
        self.climb_to(recorded.rung)
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
        # A group the record states is a release its run made: kept, as the ladder never joins them again.
        group = fields.get("ns_group")
        if group is not None:
            self._ns_group = integer(group, "rung ns_group")
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
        mesh = settings.device_mesh
        fit, headroom = (False, None) if held is None or held.signature != batch_signature(batch) else (
            admitted_placed(held.planned, mesh))
        probes, own = ([], 0) if held is None else held.probes()
        before = probes[own]["headroom"] if probes else None
        if held is not None and fit and headroom is not None and isinstance(before, int):
            # The placed devices hold more room than the search read where the start copied held arrays and
            # let their sources go: each candidate it saw miss holds the same bytes, so one whose miss that
            # room covers fits there, and the plan is searched again on the placed devices. Devices that
            # state no memory read none, and the held plan stands.
            more = headroom - before
            short = [(probe, -missed) for probe in probes
                     if probe["fits"] is False and isinstance(missed := probe["headroom"], int)]
            covered = [probe for probe, by in short if by <= more]
            _log.info("the step its search judged before the state was placed has %d bytes more room placed; "
                      "its misses were short by %s", more, [by for _, by in short])
            # Every process searches again or none does: each reads its own devices' room, and a search
            # runs its coordination rounds in every process alike (`FitSearch.gather`).
            if any(from_every_process(bool(covered))):
                _log.warning("a step its search judged before the state was placed misses by less than the "
                             "room the placed devices hold beside it (%d bytes more in this process): it is "
                             "searched again there (%s)", more,
                             "; ".join(f"split {probe['split']} at remat {probe['remat']}"
                                       for probe in covered) or "another process's misses")
                fit = False
                self._plans.pop(self._plan_key(batch), None)
        if held is not None and fit:
            # Planned and compiled while the state was built (`planned_unplaced`), and judged now the state is
            # placed: its plan, kept and its rung recorded, its record holding what the devices read.
            planned, compiled = held.planned, held.compiled
            held.keep()
            plan = record(compiled.plan, "plan")
            probes = [*probes[:own], {**probes[own], "headroom": headroom}, *probes[own + 1:]]
            memory = memory_record(planned.executable, local_devices(mesh), planned.candidate.held, fits=True)
            compiled = dataclasses.replace(compiled, plan={**plan, "probes": probes}, memory=memory)
        else:
            # Where the devices, the state placed, hold no block for it, the plan's search runs here, on them.
            planned, compiled, keep = self._plan(stepping, state, batch, trials)
            keep()
        kept = planned.released()
        return dataclasses.replace(compiled, step=self.runner(stepping, kept, settings.device_mesh),
                                   planned=kept)

    def planned_unplaced(self, stepping: Stepping, state: TrainState, batch: Batch,
                         unplaced: int) -> None:
        """A fresh run's first step planned and compiled over `state`, its shapes, before the `unplaced`
        bytes a device holds of it are placed, so the step compiles while the run's start builds the state
        (`Trainer._opened`), not after it: each candidate is judged on the devices' free bytes in all beside
        the state's (`FitSearch.unplaced`), and the run's first compile over `batch` judges the one taken
        against their allocators once the state is on them (`compile`, `admitted_placed`), its executable the
        one the run steps: the process searches its step once, and compiles its plan once. Where the plan
        does not fit the placed devices, or none fitted before the state was placed, that compile plans once
        the state is placed, from the plan's rung and split (`_plans`), not from the ladder's foot, its
        search finding the executables this one compiled, which the process holds
        (`dew.nn.kernels.options.held`)."""
        try:
            planned, compiled, keep = self._plan(stepping, state, batch, None, unplaced=unplaced)
        except PlanRefused as refused:
            _log.info("the step fits no plan before its state is placed, planned once it is: %s", refused)
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

    def sampled_step(self, holder: Holding[L, E], stepping: Callable[[], Stepping], cap: Compiled,
                     state: TrainState, batch: Batch) -> SampledStep:
        """The step of a run that draws its pass count per microbatch (`PhasePasses.sampled`) over `batch`:
        `cap`, its cap's step (`compile`), and every other count's admitted at its plan (`admitted_at`), each
        traced under its count's objective, `holder`'s and the compiler's (`PhasePasses.scoped`), by
        `stepping`. Its window pools each microbatch's statistics by their mean: one that replays them
        (`pooled_by_mean`) would replay each at the count closing its window, and is refused where it holds
        several (row 7884)."""
        sampled, planned = self.phase_passes.sampled, cap.planned
        assert sampled is not None and planned is not None, "a sampled count's cap compiled on the devices"
        # A window that pools its microbatches by their mean holds their gradient's sum and its mass
        # (`Trainer._initialize_accumulation`); one that replays them holds their batches instead.
        window = planned.candidate.prepared.accumulation
        if window is not None and window.mass is None:
            raise ValueError(f"{type(self.objective).__name__}'s statistics are no mean its window pools, so "
                             "its update replays each microbatch at the count that closes it: a pass count "
                             "drawn per microbatch takes them one microbatch an update (split and "
                             "accumulation 1), until replays run each at its own (row 7884)")
        plans, flops = {sampled.cap: planned}, {sampled.cap: cap.step_flops}
        # Compiled here, a step compiled ahead holding its cap's alone (`dew.training.ahead`).
        _log.info("the run draws its pass count per microbatch: it compiles the steps of counts %s at its "
                  "cap's plan", list(range(1, sampled.cap)))
        for count in range(1, sampled.cap):
            self.phase_passes.current = count
            with self.phase_passes.scoped(holder, self):
                plans[count], flops[count] = self.admitted_at(stepping(), state, batch, planned, count)
        self.phase_passes.current = sampled.cap
        with self.phase_passes.scoped(holder, self):
            capped = stepping()
        step = SampledStep(plans, flops, planned.split)
        step.run = self.runner(capped, planned, capped.settings.device_mesh,
                               parted=lambda: [plans[count] for count in step.counts])
        return step

    def admitted_at(self, stepping: Stepping, state: TrainState, batch: Batch, cap: Planned,
                    passes: int) -> tuple[Planned, StepFlops | None]:
        """The step of a sampled pass count's `passes` (`dew.training.phase_passes.SampledPasses`) at exactly
        the plan `cap` its cap's step runs: its split, its rung and the narrow copies the cap's step reads
        (`prepared_for`), so the window's microbatches pass one state between the counts' programs, part by
        part (`runner`'s `parted`), none remaking copies. Lowered and judged once on the devices
        (`FitSearch.fits_here`), with no search of its own: one that misses is refused, naming its count,
        its rung and the bytes it fell short by, and one whose state, window or loss's shapes are other than
        the cap's step's is refused, naming the leaves. Derived from the cap's plan at each admission, never
        recorded (`_plans`), so a resume admits each count as the run did."""
        settings = stepping.settings
        mesh = settings.device_mesh
        links = self.links(mesh)
        # The cap's rung over this count's modules: its remat, head and options, its passes this count's.
        cap.at.put(self.objective)
        self.phase_passes.restored(self.objective)
        stepping = dataclasses.replace(stepping, shape=dataclasses.replace(stepping.shape, split=cap.split))
        try:
            with stepping.traced_on(mesh, links) as schedule, held_objects():
                replicated, placement = NamedSharding(mesh, P()), batch_shardings(mesh, batch)
                shaped = self.loss_shaped(stepping, schedule)

                def lower(split: int, narrow: bool, ns_group: int | None) -> Candidate:
                    grouped = dataclasses.replace(shaped, settings=dataclasses.replace(shaped.settings,
                                                                                       ns_group=ns_group))
                    return candidate_for(grouped, state, batch, schedule, placement, replicated, split,
                                         narrow, copies=cap.candidate.copies)

                gathered = layer_gathers(self.objective, state.variables, mesh.devices.size)
                search = FitSearch(
                    lower, lambda candidate: step_options(
                        self.objective, candidate.micro, candidate.prepared.variables, mesh, gathered),
                    self.objective, mesh, cap.at.xla_defaults, cap.at.narrow, cap.at.ns_group,
                    None if self._base_rung is None else self._base_rung[1], None)
                if not search.fits_here(cap.split):
                    short = search.probes[-1].headroom
                    raise ValueError(f"the step of {passes} passes misses the devices at its cap's plan "
                                     f"({search.rung()}, {cap.split} microbatches a step)"
                                     + ("" if short is None else f" by {-short} bytes"))
                planned = search.found(cap.split)
        finally:
            self._loss_shapes.clear()
        unlike = _unlike(cap, planned)
        if unlike:
            raise ValueError(f"the step of {passes} passes holds {unlike} other than its cap's step does: a "
                             "sampled count's steps pass one state between them, its window pooling them all")
        flops, _ = plan_flops(None, planned.executable)
        return planned.released(), flops and cap.split * flops

    def _plan(self, stepping: Stepping, state: TrainState, batch: Batch, trials: Trials | None,
              frontier: bool = False, *, unplaced: int | None = None, provisional: bool = False
              ) -> tuple[Planned, Compiled, Callable[[], None]]:
        """`plan`, over a state whose devices hold `unplaced` bytes a device holds of it none of yet where
        given (`FitSearch.unplaced`), and the record of its rung, which the caller keeps once it takes the
        plan (`keep_rung`); a plan before its state is placed, or one `provisional` (a later phase's off the
        loop), holds each executable it compiles for the plan made on the devices as they stand
        (`FitSearch.provisional`)."""
        settings = stepping.settings
        mesh = settings.device_mesh
        links = self.links(mesh)
        shaped = self.at_base_rung(batch)
        resumed, self._resumed_rung = self._resumed_rung if shaped is None else shaped, None
        try:
            with stepping.traced_on(mesh, links) as schedule, held_objects():
                planned, flops, plan, validation, keep = self._planned(
                    stepping, state, batch, mesh, schedule, resumed, trials, frontier, unplaced,
                    provisional or unplaced is not None)
                split, candidate = planned.split, planned.candidate
                memory = memory_record(planned.executable, local_devices(mesh), candidate.held, fits=True)
        finally:
            # The plan made, its loss's traces are read no more (`loss_shaped`).
            self._loss_shapes.clear()
        return planned, Compiled(split, plan, memory, planned.executable, flops and split * flops, links,
                                 validation), keep

    def _plan_key(self, batch: Batch) -> tuple[str, int | None]:
        """The step a plan is kept for (`_plans`): its batch's layout and the pass count it runs
        (`PhasePasses.current`), a phase of other passes another program."""
        return batch_layout(batch), self.phase_passes.current

    def _plan_of(self, batch: Batch) -> Plan | None:
        """Where the step over `batch` stands (`_plans`): its own plan, or its layout's rung a resumed
        checkpoint recorded; None where neither."""
        layout, passes = self._plan_key(batch)
        return self._plans.get((layout, passes), self._plans.get((layout, None)))

    def runner(self, stepping: Stepping, planned: Planned, mesh: Mesh,
               parted: Callable[[], Sequence[Planned]] | None = None) -> CompiledStep:
        """The step `planned` compiled, which reads a batch in its microbatches: each by `planned`'s program,
        or by its own of the plans `parted` names for the step's microbatches at each call, programs at
        `planned`'s plan over the one state (a sampled pass count's, `admitted_at`)."""
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
        whole = whole_step_program(self.objective, parted_work=settings.sampled_passes is not None)

        def run(current, batch, sampled: bool = False):
            # A sampled step summarizes what the run inspects (`Planned.sampled`); every other, the plan's.
            plans = [planned] * split if parted is None else parted()
            executables = [held.sampled.executable if sampled and held.sampled is not None
                           else held.executable for held in plans]
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
                for part, executable in zip(parts, executables, strict=True):
                    moving, *output = executable(moving, frozen, batch, part)
                    outputs.append(tuple(output))
                stepped = together(moving, frozen)
                returned[0] = stepped.window_size
                # A split step's metrics through its whole, which pools them and reads each as its value.
                return (stepped, *(whole(outputs) if split > 1 else outputs[0]))
        return run

    def loss_shaped(self, stepping: Stepping, schedule: Schedule) -> Stepping:
        """`stepping` whose loss's shape-only trace (`Trainer._loss_shape`) is made once for what decides
        it: the window and split, the objective's rung, the trace's context (`options.trace_context`) and
        the state's and batch's types; read again by the run's fit check and its plan's candidates (a
        candidate lowered again past a spilled kernel, `Relowered`, or at its narrow copies). Kept until the
        next plan is made (`_planned`). Row 6320: nine identical traces, 61.6 s, of a 4B decision step's
        start (LF-PROMO3); the fit check's and the plan's were two of a CPU start's. Each trace is kept with
        what it told the `schedule` it ran under, whether a pipeline ran (`Schedule.pipelined`), which a read
        of it tells the schedule it is read under: a stage axis is refused where no pipeline ran
        (`StageLocal.refuse_unpipelined`), and a kept trace runs nothing."""
        traced = self._loss_shapes

        def loss_shape(view, rows, window: int, split: int, *, held=stepping.loss_shape) -> tuple:
            rung = rung_record(self.objective, xla_defaults=False, narrow=True)
            key = (window, split, json.dumps(rung, sort_keys=True), trace_context(), _shaped(view),
                   _shaped(rows))
            if key not in traced:
                before, schedule.pipelined = schedule.pipelined, False
                shapes = held(view, rows, window, split)
                traced[key] = shapes, schedule.pipelined
                schedule.pipelined = before or schedule.pipelined
            shapes, pipelined = traced[key]
            schedule.pipelined = schedule.pipelined or pipelined
            return shapes

        return dataclasses.replace(stepping, loss_shape=loss_shape)

    def _planned(self, stepping: Stepping, state: TrainState, batch: Batch,
                 mesh: Mesh, schedule: Schedule, resumed: JSON, trials: Trials | None,
                 frontier: bool, unplaced: int | None, provisional: bool
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
        stepping = self.loss_shaped(stepping, schedule)

        def lower(split: int, narrow: bool, ns_group: int | None) -> Candidate:
            # Its Newton-Schulz stacks joined as the rung asked releases them (`StepSettings.ns_group`).
            grouped = dataclasses.replace(stepping, settings=dataclasses.replace(stepping.settings,
                                                                                 ns_group=ns_group))
            return candidate_for(grouped, state, batch, schedule, placement, replicated, split, narrow)

        # Read before anything traces, so a recorded rung is the first one lowered.
        record_path, key, recorded = recorded_rung(self.objective.program_key(), (state, batch), candidates,
                                                   mesh, settings.step)
        source = record_path
        if self.ahead is not None and self.ahead.plan is not None:
            recorded, source = self.ahead.plan, "the ahead step"
        elif (searched := self._plan_of(batch)) is not None and not searched.resumed:
            recorded = {**record(searched.rung, 'plan'), 'split': searched.split}
            source = "this process's search of the step"
        before = self.rung()
        self.climb_to(recorded)
        self._resumed_rung = None
        if self.rung() != before:
            _log.warning("taking the saved plan, from %s", source)
        first = None if recorded is None else record(recorded, 'recorded plan').get('split', 1)
        # The plan's one sizing of its largest layer, which every candidate's options read.
        gathered = layer_gathers(self.objective, state.variables, mesh.devices.size)
        search = FitSearch(
            lower, lambda candidate: step_options(
                self.objective, candidate.micro, candidate.prepared.variables, mesh, gathered),
            self.objective, mesh, self._xla_defaults, self._narrow, self._ns_group,
            None if self._base_rung is None
            else self._base_rung[1], resumed, ahead=self.ahead,
            provisional=provisional, unplaced=unplaced)
        planned, probes = search.admitted(
            candidates, candidates.index(first) if isinstance(first, int) and first in candidates else 0)
        # The rungs above the plan are read for their memory alone, no device ever running them: offline
        # within a prepare too, as no device run has tuned their programs (`report.read_offline`).
        with report.read_offline():
            above = search.above(candidates, planned.split) if frontier else None
        for probe in above or ():
            _log.info("above the plan, fitting at fewer microbatches: %s", probe.named())
        # Consume the artifact only after its program was accepted; a refusal leaves it pending.
        self.ahead = search.ahead
        planned.at.put(self.objective)
        self._xla_defaults, self._narrow = planned.at.xla_defaults, planned.at.narrow
        self._ns_group = planned.at.ns_group
        if runs := pgle_runs():
            trials = Trials(state, batch, owned=False) if trials is None else trials
            run = self.runner(stepping, planned, mesh)
            relower = functools.partial(lower, planned.split, planned.at.narrow, planned.at.ns_group)
            asked = search.compiled_key(planned.candidate.lowered, planned.options)
            planned = planned.guided(trials.profiled(run, planned, mesh, runs), runs, mesh, relower)
            # The guided executable in the unguided one's place, as JAX's jit keeps its program's under
            # AutoPGLE: the one the run steps, the unguided one freed with the search.
            hold(asked, planned.executable)
        # The state the chosen plan holds between steps, its accumulator and copies included; the
        # plan's fit check counted them before any is built (`Candidate.held`).
        validation = stepping.validation(planned.candidate.prepared)
        chosen = {**self.rung(), 'split': planned.split}
        layout, passes = self._plan_key(batch)
        # This process's plan of the step, in place of a resumed checkpoint's rung for its layout.
        self._plans.pop((layout, None), None)
        self._plans[(layout, passes)] = Plan(self.rung(), planned.split)
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
                                      record=settings.record, examples=stepping.shape.examples,
                                      skip=settings.skip_nonfinite)
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
