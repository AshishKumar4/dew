"""A run's step compiler (`StepCompiler`): where the fit ladder stands for the run, rung by rung and shape by
shape, and the links of its mesh's axes, and the compile of its step from what the run states of it
(`StepSettings`) and reads (`Stepping`), its result one value (`Compiled`)."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import logging
import time
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple, Protocol

import jax
import numpy as np
from flax import linen as nn
from jax.sharding import AbstractMesh, Mesh, NamedSharding, PartitionSpec as P

from dew._coordination import agreed_count, from_every_process
from dew._records import JSON, boolean, integer, json_value, listed, record
from dew.batch import CLOSES_WINDOW, Batch, Variables, at_path, placed_at
from dew.distributed.batches import batch_layout, batch_shardings, batch_signature
from dew.distributed.placement import Validation, allocated_bytes, placed_bytes, placed_shapes
from dew.mesh import SEQUENCE_AXIS, TENSOR_AXIS
from dew.nn.kernels.generation import assumed, gpu_mesh, gpu_target
from dew.nn.kernels.options import (
    abstract_signature,
    collecting,
    hold,
    lowering,
    owned_program,
    shapes_of,
    trace_context,
)
from dew.nn.kernels.templates import report
from dew.nn.sharding import Link, Schedule
from dew.objectives.base import Objective
from dew.state import Accumulation, TrainState, apart, occupied, together
from dew.telemetry.compiles import Compiles, Ledger, Program, Span, owning
from dew.telemetry.instrumentation import StepFlops, compiled_memory
from dew.training._candidates import RunShape, Stepping, _shaped, candidate_for, forget, lowered_for
from dew.training._lowering import Lowered, routed
from dew.training.distributed import (
    link_figure,
    link_figures,
    microbatch,
    microbatch_shapes,
    resumed_link_figures,
)
from dew.training.memory import (
    climb_to,
    head_tile_of,
    layer_gathers,
    local_devices,
    memory_record,
    placed_beside,
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
    guided_compile,
    held_objects,
    released,
    settled,
    splits,
    whole_step_program,
)
from dew.training.profiling import pgle_runs
from dew.training.rungs import keep_rung, plan_flops, recorded_rung
from dew.training.transaction import Transaction, outcome_of

if TYPE_CHECKING:
    from dew.training.ahead import AheadOfTime
    from dew.training.interval import Interval
    from dew.training.transaction import Committed

_log = logging.getLogger(__name__)

UNPLACED_REFUSED = "the step fits no plan before its state is placed, planned once it is: %s"
"""What a fresh run's search of its first step before its state is placed logs where it finds no plan
(`StepCompiler.planned_unplaced`): once a fit (`FitRun.searched`)."""

PLACED_AS_CALLED = functools.partial(placed_shapes, weak=True)
"""A host step's pass over shapes moves each leaf as its shape placed where the move would place it, weakly
typed as the array its call takes is (`placed_shapes`'s `weak`)."""


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


def held_beside(stepping: Stepping, candidate: Candidate, current: TrainState, window: int,
                donor: Accumulation | None) -> TrainState:
    """`current` holding what `candidate`'s program reads beside the state a step leaves, built before it
    runs: its narrow copies and its window (`window` microbatches an update, built over `donor`, the one
    `laid_out` let go), what a fit check counts as built (`Candidate.built`): each call of the step's
    (`StepCompiler.runner`), copies a fresh run's first plan built already reused (`Building`)."""
    # Each program plans its copies (`at_base_rung`), and a step reads the state with its own: a state
    # another program's step left holds that program's (a phase of other passes, row 5560), which are made
    # again from its variables, a pure function of them.
    if not candidate.copies:
        # Copies the step reads none of, let go with their buffers: the state the run holds would keep them
        # through the step's first call.
        released([current.compute])
        current = dataclasses.replace(current, compute=None)
    elif not holds_copies(current.compute, candidate.copies):
        # The copies another program planned, which the step consumes with the state, are let go before its
        # own are made: the plan's fit counts its own alone (`_candidates`). A run handed a step compiled
        # ahead runs the program compiled with it: `fit` runs its steps within the hand (`Trainer._taking`).
        released([current.compute])
        copies = narrowing(candidate.copies, candidate.shardings.compute)
        current = dataclasses.replace(current, compute=copies(current.variables))
    if stepping.builds_window(current, candidate, window):
        # The window, and with it the split, is the state's from here on, built over the one it replaces (a
        # window laid out for another split, or a ramp's stage's of other shapes), the moving state alone
        # placed (the frozen tree stays the arrays the run holds).
        made = dataclasses.replace(stepping.window_of(current, candidate, window, donor),
                                   window_size=jax.device_put(np.int32(window)))
        moving, frozen = apart(made)
        return together(jax.device_put(moving, apart(candidate.shardings)[0]), frozen)
    if donor is not None:
        # A step that keeps no window (one microbatch an update) lets the one it was handed go.
        released([donor])
    return current


@dataclasses.dataclass
class Building:
    """A fresh run's state (its owned `trials`', which its loop steps, holding no copies), on which its first
    step's plan, made before the state was placed, builds its narrow copies (`narrowing`, a pure function of
    its variables) and is judged on the allocators holding them where they landed, beside its step what
    they do not hold yet (its window, its batches) at worst (`memory.placeable_beside`): M-MC-24c's H100
    admitted a 4B step whose 7.27 GB of copies then took the one 21.64 GB block its 20.76 GB of temporaries
    were admitted to. Their making is agreed across processes (`_coordination.agreed_count`: an allocator's
    refusal anywhere a miss everywhere, any other failure, an interrupt among them, the compile's end
    everywhere); a miss deletes them on every process, the state as it was handed (nothing of it was
    touched); a fit's first step runs on them (`Trials.state`, `held_beside` reusing copies it holds). Its
    window is the runner's to lay out, as every step's is."""
    stepping: Stepping
    trials: Trials
    before: TrainState | None = None
    """The state as it was handed, where its copies were taken (`release`)."""
    charged: tuple[int, int] = (0, 0)
    """The bytes its fit held beside the step and, of them, among its arguments built in place (`fits`)."""

    def release(self) -> None:
        """Delete the copies a fit built (a plan searched again after it, `StepCompiler.compile`), the state
        as it was handed."""
        if self.before is not None:
            released([self.trials.state.compute])
            self.trials.state, self.before = self.before, None

    def fits(self, planned: Planned) -> tuple[bool, int | None]:
        """Whether `planned` fits the devices, its narrow copies built where it reads any, its step and,
        where it samples, its sampled step, and the step's headroom."""
        candidate, mesh, state = planned.candidate, self.stepping.settings.device_mesh, self.trials.state
        if not candidate.copies:
            self.charged = (candidate.held, candidate.built)
            return self._judged(planned, *self.charged)
        made: list = []

        def built() -> tuple[Variables, int, int] | None:
            """The copies, their bytes on a device as its pool charges them (`allocated_bytes`) and as the
            step's arguments count them (`placed_bytes`), or None where an allocator refused them, registered
            as made as soon as they are returned (a refusal while they are placed deletes them too)."""
            try:
                with self.stepping.traced_on(mesh):
                    made.append(narrowing(candidate.copies, candidate.shardings.compute)(state.variables))
                jax.block_until_ready(made[-1])
            except jax.errors.JaxRuntimeError as error:
                if error.error_code_string != "RESOURCE_EXHAUSTED":
                    raise
                return None
            return made[-1], allocated_bytes(made[-1]), placed_bytes(made[-1])

        # One owner of the copies until the state takes them: every way out but that one deletes them, the
        # state holding them the mark of their transfer.
        try:
            outcome, refused = agreed_count("a fresh step's narrow copies", built, lambda held: held is None)
            if refused or outcome is None:
                return False, None
            copies, charge, payload = outcome
            # Their bytes, which the allocators now hold: no longer beside the step, nor among the bytes its
            # arguments build in place.
            self.charged = (candidate.held - charge, candidate.built - payload)
            fit, headroom = self._judged(planned, *self.charged)
            if fit:
                self.before, self.trials.state = state, dataclasses.replace(state, compute=copies)
            return fit, headroom
        finally:
            if not made or self.trials.state.compute is not made[-1]:
                released(made)

    def _judged(self, planned: Planned, held: int, built: int) -> tuple[bool, int | None]:
        """`step_fits` of `planned`'s step and, where it samples, its sampled step, `held` bytes beside it,
        `built` of them among its arguments."""
        mesh = self.stepping.settings.device_mesh
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


class HostStartup(NamedTuple):
    """What a host step's startup compiled over its state's shapes before its first step
    (`StepCompiler._compile_host`): JAX's work for each compile through Dew's compile point that was the
    startup's own (`telemetry.compiles.Ledger`'s programs of its `owning` block, each of its compile's
    `Span`; a thread its context is copied to among it, a concurrent run's or another thread's own none of
    it), and the seconds it took; its programs, each one compile (`Span.called`), how many, and by backend
    (its CPU companion's, its accelerator's) their seconds tracing and lowering, and compiling: the run's own
    compiles of a step whose steps then compile nothing."""

    spent: tuple[Program, ...]
    seconds: float
    pgle: Mapping[str, int] | None = None
    """Under JAX's AutoPGLE (`pgle_runs`), the trial steps its profile ran, its bytes and the accelerator's
    programs compiled again guided by it (`StepCompiler._guided_host`); None where it is off."""

    @classmethod
    def joined(cls, passes: Sequence[HostStartup]) -> HostStartup:
        """One startup of several passes' (a run that inspects steps over its sampled body too): their
        programs and seconds together."""
        return cls(tuple(each for passed in passes for each in passed.spent),
                   sum(passed.seconds for passed in passes))

    @property
    def spans(self) -> list[Span]:
        """Its compiles, each once, in the order their work was heard: those that built their executable,
        a program handed its executable (`options.taking`) none of them, whatever its lowering compiled to
        certify its launches."""
        return list({id(each.span): each.span for each in self.spent
                     if each.span is not None and each.span.built}.values())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted({span.called for span in self.spans}))

    @property
    def programs(self) -> int:
        return len(self.spans)

    @property
    def by_backend(self) -> dict[str, dict[str, float]]:
        """By the backend each compile targets, how many it compiled and their seconds tracing and lowering,
        and compiling."""
        by_backend: dict[str, dict[str, float]] = {}
        platforms: dict[int, str] = {}
        for span in self.spans:
            platform = span.platform
            assert platform is not None, "a startup's compiles have all returned"
            platforms[id(span)] = platform
            held = by_backend.setdefault(platform, {"programs": 0, "lowered": 0.0, "compiled": 0.0})
            held["programs"] += 1
        for each in self.spent:
            if each.span is not None and id(each.span) in platforms:
                held = by_backend[platforms[id(each.span)]]
                held["compiled" if each.kind == "compiled" else "lowered"] += each.seconds
        return by_backend

    def record(self) -> dict[str, JSON]:
        return {"names": list(self.names), "programs": self.programs, "seconds": self.seconds,
                "by_backend": {name: dict(held) for name, held in self.by_backend.items()},
                "pgle": None if self.pgle is None else dict(self.pgle)}


STEADY = 3
"""The steps a host step's startup pass steps over shapes before its state's types are those it steps from
(`StepCompiler._compile_host`): the state as placed, then as a step returns it."""


class HostRun:
    """A step that keeps its variables on the host, as its run calls it (`StepCompiler._compile_host`): each
    step's microbatches through the transaction's phases (`body`) and the glue between them, every
    program of which a startup pass compiled over its shapes (`startup`), so its steps compile nothing; a
    batch of shapes no pass stepped over (a ramp's stage's), or a step called in a trace context none
    stepped in (a float64 replay's precision), has a pass of its own before its step. A step that still
    traces or compiles (an input no pass foresaw) does, and says so: each program named in a warning and
    counted in the fit's record (`StepCompiler.counting`, `train/step_misses`), never silent; a run held
    to its loaded programs alone refuses it by its own `executable_only` (training.md §6)."""

    def __init__(self, stepping: Stepping, bodies: Mapping[bool, Callable], placement, prepared: TrainState,
                 shapes, split: int, window: int, empty, whole: Callable, frozen: tuple):
        # Its step's body, by whether the step is sampled (a run that inspects: `Trainer.inspection`), and
        # the paths of the frozen trees its state keeps beside the accelerator (`execution.resident`).
        self.stepping, self.bodies, self.placement, self.prepared = stepping, bodies, placement, prepared
        self.frozen = frozen
        self.shapes, self.split, self.window, self.whole = shapes, split, window, whole
        # The window built with the step, which its first step takes (`windowed`).
        self.empty = empty
        # Its glue between the transaction's phases, a program of its own as the transaction's is
        # (`Transaction.step`): a microbatch's reading of the batch, and its outcome.
        self.opened = owned_program(functools.partial(_opened, split=split))
        self.ended = owned_program(functools.partial(outcome_of, split=split))
        self.started: set[Hashable] = set()
        # Where its steps' misses are counted: the fit's interval (`StepCompiler.counting`).
        self.counted: Callable[[], Interval | None] = lambda: None

    def __call__(self, current, batch, sampled: bool = False):
        from dew.distributed.host import transfer

        cpu = self.stepping.settings.state_mesh
        with self.stepping.traced_on(cpu):
            batch = transfer(batch, batch_shardings(cpu, batch))
            current = self.windowed(current, batch)
            sampled = sampled and True in self.bodies
            if (abstract_signature(batch), trace_context(), sampled) not in self.started:
                self.startup(current, batch, sampled)
            with Compiles(None) as compiles, owning() as owner:
                stepped = self.stepped(current, batch, transfer, sampled)
        # What the step itself traced or compiled (`owning`): a steady host step traces nothing, so any of its
        # programs is a miss, one whose executable a kept file handed it too (`options.taking`), each named
        # once; a program another thread compiled meanwhile (a checkpoint's restore beside it) is none of it.
        missed = sorted({program.called for program in compiles.programs if program.owner is owner})
        if missed:
            interval = self.counted()
            if interval is not None:
                interval.step_misses = (interval.step_misses or 0) + len(missed)
            _log.warning("a host step traced %d programs its startup pass did not (%s): an input no pass "
                         "stepped over; %d so far", len(missed), ", ".join(missed),
                         0 if interval is None else interval.step_misses)
        return stepped

    def windowed(self, current: TrainState, batch: Batch) -> TrainState:
        """`current` with the window its step reads: the one built with the step at a run's first, a
        ramp's stage's built afresh where its window replays other shapes (`builds_window`)."""
        stepping, window, split = self.stepping, self.window, self.split
        if window > 1 and current.accumulation is None and self.empty is not None:
            empty, self.empty = self.empty, None
            return dataclasses.replace(current, accumulation=empty,
                                       window_size=jax.device_put(np.int32(window)))
        if window > 1 and (current.accumulation is None or (
                (stepping.shape.ramp or stepping.shape.stage_starts)
                and not current.accumulation.replays(self.prepared.accumulation))):
            # The window, and with it the split, the state's from here on, as a device step's.
            return dataclasses.replace(
                stepping.initialize_accumulation(current, microbatch_shapes(batch, split), self.shapes,
                                                 window=window, split=split),
                window_size=jax.device_put(np.int32(window)))
        return current

    def stepped(self, current: TrainState, batch: Batch, moved: Callable, sampled: bool = False) -> tuple:
        """One step over `current` and `batch`, values or shapes, each move made by `moved`; `sampled`, the
        step whose commit summarizes what the run inspects."""
        split, body = self.split, self.bodies[sampled]
        # Every step reads the state where its placement puts it, as the step before it returned it: the
        # programs its startup compiled over those placements are the ones each step calls.
        current = moved(current, self.placement)
        has_batch_stats = (split > 1 and current.window is not None
                           and current.window.batch_statistics is not None)
        pooled = self.stepping.model_batch(batch) if has_batch_stats else None
        outputs = []
        # Each microbatch an attempt of its own, its device work the accelerator's forward
        # (`HostExecution`), the step counted at its last.
        for part in range(split):
            at = np.int32(part)
            micro, tokens, closes, consumed = _microbatch_of(self.opened, batch, part, split)
            advanced, loss, aux, committed = body(current, micro, at, tokens, closes, pooled)
            count, clock, *output = self.ended(current.step, current.consumed, current.microstep,
                                               advanced.microstep, loss, aux, committed, batch,
                                               np.int32(part == split - 1), consumed)
            current = dataclasses.replace(advanced, step=count, consumed=clock)
            outputs.append(tuple(output))
        # A leaf already where its placement says stays the array it
        # is: the resident frozen collection crosses no boundary.
        return moved(current, self.placement), *(self.whole(outputs) if split > 1 else outputs[0])

    def startup(self, held: TrainState, batch: Batch, sampled: bool = False) -> HostStartup:
        """The step over `held`'s and `batch`'s shapes within `collecting` (the `sampled` step's, where it is
        one; a run that inspects steps over both at its start): every program it calls compiled
        there as it calls them over arrays, each of them its steps then run (`OuterProgram`'s `_shaped`); a
        pass that reads and writes nothing of the run. Its first step reads the state as given, each later
        one the state the step before returned, whose types its programs state (a weak type among them):
        the pass steps on its own results' shapes until they are the shapes it stepped from. What it
        compiled (`HostStartup`)."""
        began = time.perf_counter()
        with collecting(), Ledger(None) as ledger, owning() as owner:
            self.settled(held, batch, sampled, contextlib.nullcontext)
        self.started.add((abstract_signature(batch), trace_context(), sampled))
        passed = HostStartup(tuple(each for each in ledger.programs if each.owner is owner),
                             time.perf_counter() - began)
        _log.info("a host step's startup compiled %d programs over its state's shapes in %.1f s (%s): %s",
                  passed.programs, passed.seconds, passed.by_backend, ", ".join(passed.names))
        return passed

    def settled(self, held: TrainState, batch: Batch, sampled: bool,
                each: Callable[[], contextlib.AbstractContextManager]) -> None:
        """Step over `held`'s and `batch`'s shapes, each step within `each()`: its first step reads the state
        as given, each later one the state the step before returned, whose types its programs state (a weak
        type among them), until a step returns the shapes it stepped from, its steady steps' programs; a
        state whose types never settle is refused."""
        shaped = shapes_of(held)
        for _ in range(STEADY):
            with each():
                returned = self.stepped(shaped, shapes_of(batch), PLACED_AS_CALLED, sampled)[0]
            if abstract_signature(returned) == abstract_signature(shaped):
                return
            shaped = returned
        raise ValueError(f"a host step's state takes other types at each of its first {STEADY} steps: its "
                         "programs settle on none to compile ahead of its first step")

    def lowered(self, held: TrainState, batch: Batch, platforms: tuple[str, ...]
                ) -> list[tuple[str, jax.stages.Lowered]]:
        """Each program its steady steps call (`settled`), a sampled step's beside an unsampled one's where
        the run inspects, lowered over `held`'s and `batch`'s shapes for `platforms` where it runs on the
        accelerator and for the CPU where on its companion (`options.lowering`), none compiled: the step
        `Trainer(lower_for=)` lowers for a target this host is not."""
        programs: list[tuple[str, jax.stages.Lowered]] = []
        for sampled in self.bodies:
            last: list[tuple[str, jax.stages.Lowered]] = []
            self.settled(held, batch, sampled, functools.partial(_lowered_into, last, platforms))
            programs.extend(last)
        return programs


@contextlib.contextmanager
def _lowered_into(last: list[tuple[str, jax.stages.Lowered]], platforms: tuple[str, ...]):
    """A pass of `HostRun.lowered`'s, its lowerings `last`'s in place of an earlier pass's."""
    with lowering(platforms) as lowered:
        yield
    last[:] = lowered


def _opened(batch: Batch, part: jax.Array, *, split: int) -> tuple:
    """A host step's `part`th microbatch of `batch` (`microbatch`), its real tokens and the source examples
    it consumes (`_microbatch_of`)."""
    micro = microbatch(batch, part, split)
    return micro, batch_tokens(micro), batch_consumed(micro)


def _microbatch_of(opened: Callable, batch: Batch, part: int, split: int) -> tuple:
    """A host step's `part`th microbatch of `batch` (`_opened`, its program), the objective's batch of it,
    its real tokens, whether it closes the window and the source examples it consumes. A field with no rows
    is the batch's own array, which every microbatch reads whole, and the whole batch is its one microbatch:
    those the program would hand back as given, which the arrays a call is keyed by keep as they are."""
    micro, tokens, consumed = opened(batch, np.int32(part))
    micro = batch if split == 1 else jax.tree.map(
        lambda sliced, given: sliced if np.ndim(given) else given, micro, batch)
    return model_batch(micro), tokens, micro.get(CLOSES_WINDOW), consumed


def planned_over(batch: Batch, shape: RunShape) -> Hashable:
    """What a step's plan is planned over beside the state: its batch's shapes and the run's shape (how its
    batches are cut and counted, which every compile of its step reads)."""
    return batch_signature(batch), shape


class Unplaced(NamedTuple):
    """A fresh run's first step planned before its state was placed (`StepCompiler.planned_unplaced`): what
    it was planned over (`planned_over`: its batch's signature and the run's shape), its plan and what its
    compile found, and the record of its rung (`keep_rung`), kept only once the devices hold it with the
    state placed (`Building.fits`)."""

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
class DrawingStep:
    """The step of a run that draws its looped model's pass count per microbatch (`sampled`): each count's
    plan at its cap's (`StepCompiler.admitted_at`) and its FLOPs a step, by count (`plans`, `flops`), the
    call running each of its `split` microbatches by its drawn count's program (`counts`) over the one
    state. Each call's counts are drawn at the global positions of its microbatches under the run's `key`
    (`SampledPasses.count`), from the position it holds (`position`): a fit draws them before the call from
    the microbatches the run dispatched (`drawn`), and a call no fit drew for, as `Trainer.compile`'s caller
    makes (`dew inspect`, a step called by hand), draws its own from where it stands, its first at its
    state's step's microbatches."""

    plans: Mapping[int, Planned]
    flops: Mapping[int, StepFlops | None]
    split: int
    sampled: SampledPasses
    key: tuple[int, ...]
    """The run's key's words (`jax.random.key_data`), which every draw reads."""
    run: CompiledStep | None = None
    counts: list[int] = dataclasses.field(default_factory=list)
    """The counts of the call in hand: the next call's where `pending`, else the last call's."""
    pending: bool = False
    position: int | None = None
    """The global position of the next draw's first microbatch; None until a fit or the first call sets it."""

    def drawn(self, interval: Interval, step: int) -> StepFlops | None:
        """The counts of the next call's microbatches at the run's dispatched microbatches
        (`Interval.microbatches`, the interval's count past them): a run's first step, its fresh start's
        microbatch 0, a checkpoint's microbatches where it counted them, else (a checkpoint of a run that drew
        none) its `step`'s, said; and the call's FLOPs, each microbatch's at its count's program's."""
        if interval.microbatches is None:
            interval.microbatches = step * self.split
            if interval.last_saved is not None:
                _log.info("the run's checkpoint counted none of its dispatched microbatches: its pass counts "
                          "draw on from step %d's, microbatch %d", step, interval.microbatches)
        self.position = interval.microbatches
        interval.microbatches += self.split
        self._draw()
        self.pending = True
        total: StepFlops | None = None
        for count in self.counts:
            each = self.flops[count]
            if each is None:
                return None
            total = (1 / self.split) * each if total is None else total + (1 / self.split) * each
        return total

    def _draw(self) -> None:
        """The next call's counts, drawn at `position`, which moves past them."""
        assert self.position is not None
        self.counts = [self.sampled.count(self.key, self.position + part) for part in range(self.split)]
        self.position += self.split

    def __call__(self, state: TrainState, batch: Batch, /, sampled: bool = False):
        assert self.run is not None
        if not self.pending:
            # A call no fit drew for draws its own, its first at its state's step's microbatches.
            if self.position is None:
                self.position = int(state.step) * self.split
            self._draw()
        self.pending = False
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
        # The step compiled ahead a run was handed, whose counts' family a run that draws its pass count per
        # microbatch takes after its first compile took the cap's (`admitted_counts`).
        self.counts_ahead: AheadOfTime | None = None
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
        # The fit's interval its host steps count their misses in (`counting`), None outside a fit.
        self._counted: Interval | None = None
        # What a host step's startup compiled over its state's shapes (`_compile_host`), None before one.
        self.host_startup: HostStartup | None = None
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

    def checkpoint_rung(self) -> dict[str, JSON]:
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
            # Shapes are all a lowering reads, so the state may be abstract (`jax.eval_shape`); a device step
            # lowers at the base rung, a host step as its compile builds it.
            if settings.host_master:
                raise Lowered(target, *routed(lambda: self._lowered_host(stepping, state, batch, target)))
            self.at_base_rung(batch)
            raise Lowered(target, *routed(
                lambda: [("step", lowered_for(stepping, state, batch, target, self.narrow))]))
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
            if self.ahead is not None:
                raise ValueError("a step compiled ahead runs where the devices hold the state; this layout "
                                 "keeps the variables on the host")
            run = self._compile_host(stepping, state, batch)
            if runs := pgle_runs():
                self._guided_host(stepping, run, state, batch, trials, runs)
            # Its startup's record, AutoPGLE's outcome among it, the fit's compile report's plan
            # (`StepCompiled.plan`): a host step plans no rung, and its startup is what its compile did.
            assert self.host_startup is not None, "a host step's compile runs its startup pass"
            return Compiled(stepping.shape.split, plan={"startup": self.host_startup.record()}, step=run)
        held, self._unplaced = self._unplaced, None
        mesh = settings.device_mesh
        # A fresh run's first plan, made before its state was placed, judged with its narrow copies built on
        # the state its loop steps (`Building`): owned by its fit's trials, holding no copies (a start's own;
        # none an overridden start hands over), so a refused build touches nothing of the state. A compile
        # no loop steps after (a public caller's) takes no plan held for a fit.
        planned_here = (held is not None and held.signature == planned_over(batch, stepping.shape)
                        and trials is not None and trials.owned and trials.state.compute is None)
        judging = Building(stepping, trials) if planned_here and trials is not None else None
        held = None if judging is None else held
        fit, headroom = (False, None) if judging is None or held is None else judging.fits(held.planned)
        probes, own = ([], 0) if held is None else held.probes()
        before = probes[own]["headroom"] if probes else None
        if judging is not None and fit:
            # The placed devices hold more room than the search read where the start copied held arrays and
            # let their sources go: each candidate it saw miss holds the same bytes, so one whose miss that
            # room covers fits there, and the plan is searched again on the placed devices. Devices that
            # state no memory read none, and cover none: every process takes the round (`fit` is agreed).
            more = headroom - before if headroom is not None and isinstance(before, int) else None
            short = [(probe, -missed) for probe in probes
                     if probe["fits"] is False and isinstance(missed := probe["headroom"], int)]
            covered = [] if more is None else [probe for probe, by in short if by <= more]
            _log.info("the step its search judged before the state was placed has %s bytes more room placed; "
                      "its misses were short by %s", more, [by for _, by in short])
            # Every process searches again or none does: each reads its own devices' room, and a search
            # runs its coordination rounds in every process alike (`FitSearch.gather`).
            if any(from_every_process(bool(covered))):
                _log.warning("a step its search judged before the state was placed misses by less than the "
                             "room the placed devices hold beside it (%s bytes more in this process): it is "
                             "searched again there (%s)", more,
                             "; ".join(f"split {probe['split']} at remat {probe['remat']}"
                                       for probe in covered) or "another process's misses")
                fit = False
                judging.release()
                self._plans.pop(self._plan_key(batch), None)
        if judging is not None and held is not None and fit:
            # Planned and compiled while the state was built (`planned_unplaced`), and judged now the state is
            # placed: its plan, kept and its rung recorded, its record holding what the devices read.
            planned, compiled = held.planned, held.compiled
            held.keep()
            plan = record(compiled.plan, "plan")
            # The search's probes, its own read at the headroom the placed devices hold (what a step's need is
            # read from) with the bytes that judgement placed beside the step, not the unplaced search's
            # charge of a state the devices held none of (row 8962: a headroom and a charge of two
            # judgements in one record); and the placed judgement whole (`placed`): that headroom with its
            # copies built, and the bytes it held beside the step that the allocators did not.
            held_out = judging.charged[0]
            stats = compiled_memory(planned.executable)
            beside = None if stats is None else placed_beside(stats, held_out)
            probes = [*probes[:own], {**probes[own], "headroom": headroom, "beside": beside},
                      *probes[own + 1:]]
            memory = memory_record(planned.executable, local_devices(mesh), held_out, fits=True)
            compiled = dataclasses.replace(
                compiled, plan={**plan, "probes": probes, "placed": {"headroom": headroom, "held": held_out}},
                memory=memory)
        else:
            # Where the devices, the state placed, hold no block for it, the plan's search runs here, on them.
            planned, compiled, keep = self._plan(stepping, state, batch, trials)
            keep()
        kept = planned.released()
        return dataclasses.replace(compiled, step=self.runner(stepping, kept, settings.device_mesh),
                                   planned=kept)

    def anew(self) -> StepCompiler[L, E]:
        """A compiler of this one's objective, links, step compiled ahead and pass counts, nothing planned:
        where a run lays its state out otherwise than the plans this one made were for
        (`dew.training.streaming.laid_start`), whose ladder climbs and plans read no step of that layout."""
        return StepCompiler(self.objective, self._stated_links, self.ahead, self.phase_passes)

    @property
    def unplaced(self) -> Planned | None:
        """The plan a fresh run's first step took before its state was placed (`planned_unplaced`), None
        where its devices held none or none was made."""
        return None if self._unplaced is None else self._unplaced.planned

    def planned_unplaced(self, stepping: Stepping, state: TrainState, batch: Batch,
                         unplaced: int) -> Hashable:
        """A fresh run's first step planned and compiled over `state`, its shapes, before the `unplaced`
        bytes a device holds of it are placed, so the step compiles while the run's start builds the state
        (`Trainer._opened`), not after it: each candidate is judged on the devices' free bytes in all beside
        the state's (`FitSearch.unplaced`), and the run's first compile over `batch` judges the one taken
        against their allocators once the state is on them, its buffers built (`compile`, `Building.fits`),
        its executable the
        one the run steps: the process searches its step once, and compiles its plan once. Where the plan
        does not fit the placed devices, or none fitted before the state was placed, that compile plans once
        the state is placed, from the plan's rung and split (`_plans`), not from the ladder's foot, its
        search finding the executables this one compiled, which the process holds
        (`dew.nn.kernels.options.held`). What it searched over (`planned_over`), its plan held or none
        found, which the fit searching it keeps (`FitRun.searched`): no fit searches it twice before placing
        its state."""
        signature = planned_over(batch, stepping.shape)
        try:
            planned, compiled, keep = self._plan(stepping, state, batch, None, unplaced=unplaced)
        except PlanRefused as refused:
            _log.info(UNPLACED_REFUSED, refused)
            self._unplaced = None
            return signature
        self._unplaced = Unplaced(signature, planned, compiled, keep)
        return signature

    def plan(self, stepping: Stepping, state: TrainState, batch: Batch,
             trials: Trials | None, frontier: bool = False) -> tuple[Planned, Compiled]:
        """The shared device-step planner: the plan it admitted and what its compile found (`Compiled`, its
        step none: the caller reads its lowering before releasing it, `compile`, or serializing its
        executable, `compile_ahead`), with `frontier` the rungs above it (`FitSearch.above`), its rung
        recorded (`keep_rung`)."""
        planned, compiled, keep = self._plan(stepping, state, batch, trials, frontier)
        keep()
        return planned, compiled

    def drawing(self, holder: Holding[L, E], stepping: Callable[[], Stepping], compiled: Compiled,
                state: TrainState, batch: Batch, key: jax.Array) -> CompiledStep:
        """`compiled`'s step, or where the run draws its pass count per microbatch (`PhasePasses.sampled`)
        its every count's step at its cap's plan, drawing under the run's `key` (`sampled_step`)."""
        assert compiled.step is not None
        if self.phase_passes.sampled is None:
            return compiled.step
        return self.sampled_step(holder, stepping, compiled, state, batch, key)

    @staticmethod
    def drawn(held: tuple[CompiledStep, StepFlops | None], interval: Interval,
              step: int) -> tuple[CompiledStep, StepFlops | None]:
        """A compiled step and its FLOPs (`held`), a drawing step's (`DrawingStep`) its next call's counts
        drawn at the microbatches `interval` counted, its FLOPs theirs."""
        compiled, flops = held
        if isinstance(compiled, DrawingStep):
            flops = compiled.drawn(interval, step)
        return compiled, flops

    def sampled_step(self, holder: Holding[L, E], stepping: Callable[[], Stepping], cap: Compiled,
                     state: TrainState, batch: Batch, key: jax.Array) -> DrawingStep:
        """The step of a run that draws its pass count per microbatch (`PhasePasses.sampled`) over `batch`:
        `cap`, its cap's step (`compile`), and every other count's admitted at its plan (`admitted_at`), each
        traced under its count's objective, `holder`'s and the compiler's (`PhasePasses.scoped`), by
        `stepping`. Its window pools each microbatch's statistics by their mean: one that replays them
        (`pooled_by_mean`) would replay each at the count closing its window, and is refused where it holds
        several (row 7884)."""
        sampled, planned = self.phase_passes.sampled, cap.planned
        assert sampled is not None and planned is not None, "a sampled count's cap compiled on the devices"
        # A window that pools its microbatches by their mean holds their gradient's sum and its mass
        # (`Trainer._initialize_accumulation`), DiLoCo's its groups' (`TrainState.window`); one that replays
        # them holds their batches instead.
        window = planned.candidate.prepared.window
        if window is not None and window.mass is None:
            raise ValueError(f"{type(self.objective).__name__}'s statistics are no mean its window pools, so "
                             "its update replays each microbatch at the count that closes it: a pass count "
                             "drawn per microbatch takes them one microbatch an update (split and "
                             "accumulation 1), until replays run each at its own (row 7884)")
        plans, flops = {sampled.cap: planned}, {sampled.cap: cap.step_flops}
        for count, each in self.admitted_counts(holder, stepping, planned, state, batch):
            plans[count], flops[count] = each.released(), self._flops(each)
        self.phase_passes.current = sampled.cap
        with self.phase_passes.scoped(holder, self):
            capped = stepping()
        words = tuple(int(word) for word in np.asarray(jax.random.key_data(key)).ravel())
        step = DrawingStep(plans, flops, planned.split, sampled, words)
        step.run = self.runner(capped, planned, capped.settings.device_mesh,
                               parted=lambda: [plans[count] for count in step.counts])
        return step

    def admitted_counts(self, holder: Holding[L, E], stepping: Callable[[], Stepping], cap: Planned,
                        state: TrainState, batch: Batch) -> Iterator[tuple[int, Planned]]:
        """Every count but the cap of a run that draws its pass count per microbatch (`PhasePasses.sampled`)
        admitted at the cap's plan `cap` (`admitted_at`), each traced under its count's objective, `holder`'s
        and the compiler's (`PhasePasses.scoped`), by `stepping`, and handed on as it is admitted, so its
        caller releases or keeps each before the next compiles; each taken from the counts' family of the
        step compiled ahead where the run was handed one (`counts_ahead`, `AheadOfTime.at_count`), which
        its first admission consumes as a step compiled ahead is (a later batch's shapes compile their
        own), else compiled here, as the run says."""
        sampled = self.phase_passes.sampled
        assert sampled is not None, "a run that draws its pass count per microbatch"
        counts, family, self.counts_ahead = range(1, sampled.cap), self.counts_ahead, None
        if family is None:
            _log.info("the run draws its pass count per microbatch: it compiles the steps of counts %s at "
                      "its cap's plan", list(counts))
        try:
            for count in counts:
                self.phase_passes.current = count
                with self.phase_passes.scoped(holder, self):
                    planned = self.admitted_at(stepping(), state, batch, cap, count,
                                               ahead=None if family is None else family.at_count(count))
                yield count, planned
        finally:
            self.phase_passes.current = sampled.cap

    @staticmethod
    def _flops(planned: Planned) -> StepFlops | None:
        """A step's FLOPs at `planned`, a microbatch's (`plan_flops`) times its split."""
        flops, _ = plan_flops(None, planned.executable)
        return flops and planned.split * flops

    def admitted_at(self, stepping: Stepping, state: TrainState, batch: Batch, cap: Planned, passes: int,
                    ahead: AheadOfTime | None = None) -> Planned:
        """The step of a sampled pass count's `passes` (`dew.training.phase_passes.SampledPasses`) at exactly
        the plan `cap` its cap's step runs: its split, its rung and the narrow copies the cap's step reads
        (`prepared_for`), so the window's microbatches pass one state between the counts' programs, part by
        part (`runner`'s `parted`), none remaking copies. Lowered and judged once on the devices
        (`FitSearch.fits_here`), with no search of its own: one that misses is refused, naming its count,
        its rung and the bytes it fell short by, and one whose state, window or loss's shapes are other than
        the cap's step's is refused, naming the leaves. Derived from the cap's plan at each admission, never
        recorded (`_plans`), so a resume admits each count as the run did. `ahead`, its step compiled ahead
        (`AheadOfTime.at_count`), runs in place of its compile where it is that program. Unreleased: its
        caller releases it (`Planned.released`) once it has read its lowering, a step compiled ahead its
        program's digest."""
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
                    None if self._base_rung is None else self._base_rung[1], None, ahead=ahead)
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
        return planned

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
                current = held_beside(stepping, candidate, current, window, donor)
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
        # Consume the artifact only after its program was accepted; a refusal leaves it pending. A run
        # that draws its pass count per microbatch keeps its counts' family for their admission
        # (`admitted_counts`).
        if self.ahead is not None and search.ahead is None and self.phase_passes.sampled is not None:
            self.counts_ahead = self.ahead
        self.ahead = search.ahead
        planned.at.put(self.objective)
        self._xla_defaults, self._narrow = planned.at.xla_defaults, planned.at.narrow
        self._ns_group = planned.at.ns_group
        if runs := pgle_runs():
            trials = Trials(state, batch, owned=False) if trials is None else trials
            run = self.runner(stepping, planned, mesh)
            relower = functools.partial(lower, planned.split, planned.at.narrow, planned.at.ns_group)
            unguided, replaced = planned.candidate.lowered, planned.executable
            asked = search.compiled_key(unguided, planned.options)
            planned = planned.guided(trials.profiled(run, planned, mesh, runs), runs, mesh, relower)
            if planned.executable is not replaced:
                # The guided executable in the unguided one's place, as JAX's jit keeps its program's under
                # AutoPGLE: the one the run steps, the unguided one freed with the search and the step kept
                # for a refit. An empty profile replaces nothing, and every kept lowering stays.
                hold(asked, planned.executable)
                forget(unguided)
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

    def counting(self, interval: Interval, *, host: bool) -> Interval:
        """`interval`, a fit's first, which its host steps (`host`) count the programs they compile past
        their startup passes in from here on (`HostRun`, `Interval.step_misses`, zero at its start); a fit
        whose steps are no host's counts none."""
        self._counted = interval if host else None
        interval.step_misses = 0 if host else None
        return interval

    def prepared_host(self, stepping: Stepping, state: TrainState, batch: Batch) -> HostStartup:
        """A fresh host step's startup over `state`'s and `batch`'s shapes (`_compile_host`), nothing placed:
        what a build host compiles to warm the shared cache (`dew.training.ahead.prepare_host_step`)."""
        self._compile_host(stepping, state, batch)
        assert self.host_startup is not None, "a host step's compile runs its startup pass"
        return self.host_startup

    def _compile_host(self, stepping: Stepping, state: TrainState, batch: Batch) -> HostRun:
        """The host step over `state` and `batch` (`_host_run`), its startup run before its first step: each
        of its steps' bodies, a sampled step's as well where the run inspects (`HostRun.startup`)."""
        from dew.distributed.host import transfer

        run, held, cpu_batch = self._host_run(stepping, state, batch, stepping.settings.device_mesh, transfer)
        with stepping.traced_on(stepping.settings.state_mesh):
            passes = [run.startup(held, cpu_batch, sampled) for sampled in run.bodies]
        self.host_startup = HostStartup.joined(passes)
        return run

    def _guided_host(self, stepping: Stepping, run: HostRun, state: TrainState, batch: Batch,
                     trials: Trials | None, runs: int) -> None:
        """Under JAX's AutoPGLE, `run`'s steps profiled `runs` times on a trial (`trials`', or a copy of
        `state` beside it, in host memory: `Trials.profiled_programs`), after its startup, as a device plan
        is profiled before its run (`Planned.guided`); each program they ran on the accelerator compiled
        again guided by the profile (`guided_compile`) and run in its executable's place
        (`OuterProgram.guided`), so the run's steps run the guided programs and compile nothing. An empty
        profile, a backend XLA profiles none of, guides nothing. Its count is the startup's record
        (`HostStartup.pgle`)."""
        accelerator = {device.platform for device in stepping.settings.device_mesh.devices.flat} - {"cpu"}

        def on_accelerator(executable: jax.stages.Compiled) -> bool:
            return any(device.platform in accelerator
                       for sharding in (*jax.tree.leaves(executable.input_shardings),
                                        *jax.tree.leaves(executable.output_shardings))
                       for device in sharding.device_set)

        trials = Trials(state, batch, owned=False) if trials is None else trials
        # The trial steps a window of its own: the one built with the step stays the run's first step's
        # (`HostRun.windowed`), which then traces nothing (row 8503).
        kept = run.empty
        run.empty = None if kept is None else jax.tree.map(lambda leaf: leaf.copy(), kept)
        try:
            profile, ran = trials.profiled_programs(run, runs, on_accelerator)
        finally:
            run.empty = kept
        guided = ran if profile else []
        for program, executable, lowered, relower in guided:
            program.guided(executable, guided_compile(lowered, program.options, profile, relower))
        assert self.host_startup is not None, "a host step's compile runs its startup pass"
        self.host_startup = self.host_startup._replace(
            pgle={"runs": runs, "bytes": len(profile), "programs": len(guided)})
        # Zero included: an empty profile (a backend XLA profiles none of) guides nothing, said so.
        _log.info("AutoPGLE's profile of a host step's %d trial steps (%d bytes) guided %d of its %d "
                  "accelerator programs", runs, len(profile), len(guided), len(ran))

    def _lowered_host(self, stepping: Stepping, state: TrainState, batch: Batch, target: str
                      ) -> list[tuple[str, jax.stages.Lowered]]:
        """The host step's programs lowered for a GPU of the CUDA `target` this host is not (`lower_for`):
        built as `_compile_host` builds it, under the target's choices (`generation.assumed`) and over an
        abstract mesh of its devices (`gpu_mesh`) where its realizations run, its companion this host's CPUs,
        each program its steady steps call lowered and none compiled (`HostRun.lowered`)."""
        settings = stepping.settings
        # Shapes alone, a placed state's too: a lowering builds nothing (a window an array state's first
        # step would build among it).
        state, batch = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(np.shape(leaf), leaf.dtype),
                                    (state, batch))
        with assumed(target):
            accelerator = gpu_mesh(settings.device_mesh, target)
            run, held, cpu_batch = self._host_run(stepping, state, batch, accelerator, PLACED_AS_CALLED)
            # The frozen trees beside the target's devices, as beside this host's (`execution.resident`).
            variables = run.placement.variables
            for path in run.frozen:
                variables = placed_at(variables, path, jax.tree.map(
                    lambda held: NamedSharding(accelerator, held.spec, memory_kind=held.memory_kind),
                    at_path(variables, path)))
            run.placement = dataclasses.replace(run.placement, variables=variables)
            with stepping.traced_on(settings.state_mesh):
                return run.lowered(held, cpu_batch, ("cuda",))

    def _host_run(self, stepping: Stepping, state: TrainState, batch: Batch, accelerator: Mesh | AbstractMesh,
                  moved: Callable) -> tuple[HostRun, TrainState, Batch]:
        """The host step over `state` and `batch`, its realizations on `accelerator`: the run, the state its
        startup steps from (with the window its first step builds) and the batch on its CPU companion, placed
        there by `moved` (`transfer`, or `PLACED_AS_CALLED` for a batch of shapes, a lowering's)."""
        settings = stepping.settings
        from dew.training.execution import HostExecution, frozen_trees, stack_sites
        cpu = settings.state_mesh
        sites = stack_sites(self.objective, state.variables)
        execution = HostExecution(self.objective, settings.layout, accelerator, cpu, sites,
                                  device=stepping.shape.device,
                                  gathered=layer_gathers(self.objective, state.variables, accelerator.size))
        # The batch read in the split the run states (no plan chooses one: the step is the host's), its window
        # that many microbatches an update times the run's accumulation, as a device step's (`step_program`).
        split = stepping.shape.split or 1
        window = split * settings.accumulation
        cpu_batch = moved(batch, batch_shardings(cpu, batch))
        with stepping.traced_on(cpu) as schedule:
            micro = microbatch_shapes(cpu_batch, split)
            _, shapes = stepping.loss_shape(state, micro, window, split)
            stepping.staging.refuse_unpipelined(schedule, self.objective)
            prepared = stepping.initialize_accumulation(state, micro, shapes, window=window, split=split,
                                                        shape_only=True)
            placement = stepping.shardings(prepared)
            frozen = tuple(path for path, _, _ in frozen_trees(state.variables, sites))
            # Its step's body, and where the run inspects, a sampled step's (`Trainer.inspection`): the
            # one whose commit summarizes what it inspects, its forward differentiating the watched
            # outputs' probes beside its parameters (`HostExecution`), as a device run's sampled step.
            bodies = {sampled: Transaction(
                self.objective, settings.optimizer, window, shapes, split=split, record=settings.record,
                examples=stepping.shape.examples, skip=settings.skip_nonfinite,
                inspection=settings.inspection if sampled else None).step(
                    realize=execution.realize, host=True, frozen=frozen)
                for sampled in ((False, True) if settings.inspection is not None else (False,))}
        # A state of shapes alone (a build host's, warming the shared cache: `dew.training.ahead
        # .prepare_host_step`) holds its window's shapes.
        concrete = not isinstance(jax.tree.leaves(state.variables)[0], jax.ShapeDtypeStruct)
        # The window a run's first step builds (`builds_window`), built here, with the step: its first
        # step reads it as built, compiling and tracing nothing (`executable_only`).
        empty = (None if window == 1 or state.accumulation is not None else stepping.initialize_accumulation(
            state, microbatch_shapes(cpu_batch, split), shapes, window=window, split=split,
            shape_only=not concrete).accumulation)
        run = HostRun(stepping, bodies, placement, prepared, shapes, split, window, empty,
                      whole_step_program(self.objective), frozen)
        run.counted = lambda: self._counted
        held = state if empty is None else dataclasses.replace(
            state, accumulation=empty, window_size=jax.device_put(np.int32(window)))
        return run, held, cpu_batch

    # ------------------------------------------------------------------ The loop
    # ------------------------------------------------------------------
