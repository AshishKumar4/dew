"""Whether a fresh run streams its parameters from the host (row 1880): the plan its devices hold, priced
against the step that keeps the state on the host (`Layout(host=("variables",))`), each as a step time
(`dew.training.memory.StepPrice`).

A run whose devices hold no plan of its step trains streamed where the devices hold that one; where they
hold both, the cheaper step is taken: the device plan's step is its compute, a remat rung's recompute
among it, and the streamed step its compute or its host link's transfers, whichever is longer, the two
overlapping layer by layer, and then its update on the host (`StepPrice.seconds`). A run that states its
layout keeps it, and so does one the streamed step cannot run: a state streamed lives on a CPU companion
of the devices, which a process holds only where it was started with one (training.md, "Parameters on the
host"), and a part that states no host link (`GpuTarget.host_links`) prices no streamed step. Each refusal
is said by name where the plans are chosen (`refusal`, `Choice`)."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from dew._coordination import broadcast_from_process_zero
from dew._records import JSON, listed, record, text
from dew.batch import FROZEN
from dew.distributed.batches import placed_abstract, prefetched_bytes
from dew.distributed.host import companion_mesh
from dew.distributed.placement import Layout, placed_bytes
from dew.mesh import LayoutRefused
from dew.nn.kernels.generation import gpu_target
from dew.nn.protocols import ComputesIn, computes_in
from dew.telemetry.instrumentation import step_flops
from dew.training._fit import Start, handed, staged
from dew.training.diloco import DILOCO_ON_HOST
from dew.training.distributed import marked, microbatch_shapes
from dew.training.execution import HostExecution, frozen_trees, stack_sites, unscanned
from dew.training.memory import (
    StepPrice,
    Unpriced,
    device_price,
    host_memory_rate,
    host_update_seconds,
    layer_gathers,
    local_devices,
    step_fits,
    streamed_price,
)
from dew.training.model_flops import model_batch
from dew.training.planning import PlanRefused, Rung
from dew.training.transaction import loss_arguments

_log = logging.getLogger(__name__)

STREAMED = ("variables",)
"""The layout's host fields of a streamed run: its whole state the host's (`Layout.host`)."""


@dataclasses.dataclass(frozen=True)
class Choice:
    """The plans a run chose its layout among: whether its devices hold a plan of its given layout's step
    (`held`) and whether they hold the streamed step (`fits`), each one's step price where its part rates
    it (`device`, `streamed`; None otherwise), why the streamed one was not a candidate (`refused`), and
    whether it was taken (`streams`); or, for a run resuming a checkpoint (`resumed`), the layout that
    checkpoint trained in, laid out alike (`resumed_layout`), no plan priced. `streamed_headroom` is the
    bytes the streamed step's fit check found its tightest device had to spare (`step_fits`, negative where
    short), as a plan's probes record the device ladder's; None where that check did not run or read none."""

    held: bool
    fits: bool
    device: StepPrice | None
    streamed: StepPrice | None
    refused: str | None
    streams: bool
    resumed: bool = False
    streamed_headroom: int | None = None

    def record(self) -> dict[str, JSON]:
        return {"held": self.held, "fits": self.fits,
                "device": None if self.device is None else self.device.record(),
                "streamed": None if self.streamed is None else self.streamed.record(),
                "refused": self.refused, "streams": self.streams, "resumed": self.resumed,
                "streamed_headroom": self.streamed_headroom}

    def noted(self, error: BaseException) -> None:
        """`error`, a plan or a layout the devices do not hold, noted with what this choice found of the
        streamed step: why it was no candidate, or that the devices do not hold it either."""
        if self.refused is not None:
            error.add_note(f"Streaming the parameters from the host (Layout(host=('variables',))) was not "
                           f"considered: {self.refused}.")
        elif not self.streams and not self.fits:
            error.add_note("Streaming the parameters from the host (Layout(host=('variables',))) was "
                           "considered: the devices do not hold that step either.")


NOT_BUILT = 7908
"""The ledger row of the runs the streamed step is not built for yet (`refusal`)."""


def refusal(trainer, *, stated: bool, ahead: bool) -> str | None:
    """Why `trainer`'s run cannot stream its parameters from the host, None where it can. What the streamed
    step cannot be: a layout the run states (`stated`), a custom step, which owns its execution, a run
    lowering for another target (`lower_for`), whose steps this host prices on no devices of that target (a
    stated host layout lowers, `StepCompiler._lowered_host`), a run whose rollout makes its steps' batches
    (`TrainerAdvanced(rollout=)`), whose stream's first batch is the rollout's input, no step's, so no step is
    priced before a rollout runs (a stated host layout samples from a device copy of its weights,
    `dew.training.rollout.on_devices`), or a process holding no CPU companion of the devices, which JAX makes
    only before it initializes. And what it is not built for yet (`NOT_BUILT`; DiLoCo's groups by row
    `DILOCO_ON_HOST`): DiLoCo, a step compiled `ahead`, a pass count drawn per microbatch, a part whose table
    states no host link to price it by (`GpuTarget.host_links`: a TPU's but v6e's), and a frozen stack a plain
    loop runs (`frozen_loops`). Read before any plan is built."""
    unbuilt = (
        (trainer.diloco is not None, "DiLoCo's groups", DILOCO_ON_HOST),
        (ahead, "a step compiled ahead", NOT_BUILT),
        (trainer.phase_passes.sampled is not None, "a pass count drawn per microbatch (sampled_passes)",
         NOT_BUILT),
    )
    if stated:
        return "the run states its layout"
    if trainer.step is not None:
        return "a custom step owns its execution"
    if trainer.lower_for is not None:
        return "the run lowers its step for another target, whose devices price no plan here"
    if trainer.rollout is not None:
        return ("the run's rollout makes its steps' batches, which its stream's first batch is none of: no "
                "step is priced before a rollout runs")
    for refused, what, row in unbuilt:
        if refused:
            return f"the streamed step is not built for {what} yet (row {row})"
    target = gpu_target()
    if target.host_link_rate is None:
        return (f"the streamed step is not priced on {target.device_kind} yet: no host-link rate is stated "
                f"for it (`GpuTarget.host_links`; row {NOT_BUILT})")
    try:
        companion_mesh(trainer.device_mesh)
    except ValueError as missing:
        return f"the process holds no CPU companion of its devices: {missing}"
    # A frozen stack a plain loop runs, which no host layout holds in bank memory (`unscanned`), read from
    # the objective's draw's shapes (row 8336: a LoRA fit over an unscanned decoder raised where its
    # streamed start placed its frozen base; row 8765: one site refuses by name).
    return frozen_loops(trainer, trainer.objective.drawn_shapes())


def frozen_loops(trainer, variables) -> str | None:
    """Why a streamed layout of `trainer`'s run over `variables` (shapes alike) cannot place its frozen
    trees, None where it can: a stack among them a plain loop runs (`execution.unscanned`), which every
    tree's placement refuses (`execution.resident`)."""
    reasons = (unscanned(sites) for _, _, sites in frozen_trees(variables, stack_sites(trainer.objective,
                                                                                         variables)))
    return next((reason for reason in reasons if reason is not None), None)


def priced(plan, dtype) -> StepPrice:
    """The step of the device plan `plan` (`StepCompiler.planned_unplaced`'s), priced (`device_price`):
    its executable runs once a microbatch, `plan.split` of them a step."""
    flops = step_flops(plan.executable)
    return device_price(None if flops is None else plan.split * flops, gpu_target(), dtype)


def streamed(trainer, start: Start, batch, dtype, device_steps: tuple, update: float
             ) -> tuple[bool, int | None, StepPrice | None]:
    """Whether the devices hold the step of `trainer`, laid out streamed, that streams `start`'s parameters
    from the host over `batch`, the loader's `device_steps` run in it, its update on the host `update`
    seconds (`host_update_seconds` of `host_updated`), traced in the run's own context
    (`Trainer._traced_on`; `HostExecution.compiled_whole`), admitted before anything is placed as a
    plan's candidate is (`_candidates._candidate`: `step_fits`, the inputs each realization places counted
    as built, and the batches the run prefetches beside them, `prefetched_bytes`), and its price
    (`streamed_price`), None where its part rates no step computing in `dtype`, with the bytes its fit check
    found the tightest device had to spare (`step_fits`'s headroom, None where it read none). A run that
    inspects (`Trainer.inspection`) is admitted where its sampled step fits too, its forward differentiating
    the watched outputs' probes (`loss_arguments`' `watched`), as a device plan's rung is
    (`FitSearch.measured`). Its bytes are each device's: the shards its link moves, and the shards its
    process's update reads and writes on its host, every local device's."""
    mesh, abstract = trainer.device_mesh, start.abstract
    # The batch read in the split the run states, each microbatch an attempt of its own that fetches the
    # parameters again (`StepCompiler._compile_host`).
    split = trainer._shape.split or 1
    whole_batch, batch = batch, microbatch_shapes(batch, split)
    # The CPU companion read before the devices' mesh is in place: its probe runs a CPU program, which no
    # trace under a GPU's mesh may (M-MC-24b: refused on a GH200).
    companion = companion_mesh(mesh)
    watched = None if trainer.inspection is None else trainer.inspection.watched
    with trainer._traced_on(mesh):
        execution = HostExecution(trainer.objective, trainer.layout, mesh, companion,
                                  stack_sites(trainer.objective, abstract.variables), device=device_steps,
                                  gathered=layer_gathers(trainer.objective, abstract.variables,
                                                         mesh.devices.size))
        window = split * trainer.accumulation
        steps = []
        for sampled in (None, *(() if watched is None else (watched,))):
            arguments = loss_arguments(trainer.objective, abstract, batch, window, split, watched=sampled)
            steps.append(execution.compiled_whole(arguments[0], batch, arguments[1]))
    (compiled, (primal, frozen, rest, _, here)), *sampled_steps = steps
    devices = local_devices(mesh)
    built = max(placed_bytes((primal, frozen, rest, here), device=device, memory="device")
                for device in devices)
    prefetched = prefetched_bytes(whole_batch, jax.tree.map(lambda leaf: leaf.sharding, whole_batch),
                                  trainer._shape.prefetch)
    # A process whose compile XLA refused for memory (None) takes part in the agreement as one that does
    # not fit, as a plan's candidate's does: every process answers alike, and none waits on another.
    fits, headroom = step_fits(compiled, mesh, prefetched + built, built, placed=False)
    for sampled, _ in sampled_steps:
        # The tighter margin of the two, unknown where the sampled step's is.
        held, room = step_fits(sampled, mesh, prefetched + built, built, placed=False)
        fits = fits and held
        headroom = None if headroom is None or room is None else min(headroom, room)
    if not fits or compiled is None:
        return False, headroom, None
    # The parameters the step trains, the vjp's primal, moved there and back each attempt.
    parameters = max(placed_bytes(primal, device=device) for device in devices)
    # Read once an attempt, beside them, and differentiated by nothing: the averaged weights the step reads
    # (an EMA reference, `Step.ema`), and the banks of every frozen tree the host keeps, its frozen
    # parameters' (a LoRA base, the run's frozen collection) and its frozen roots' (`frozen_trees`: a
    # distillation's teacher whole), each forward's fetch of them from host memory; their other leaves stay
    # resident on the devices, which the admission counts.
    # The run's own frozen collection is among the primal's frozen leaves (`frozen`), the snapshot's
    # parameters holding it; its nested roots' trees are the snapshot's other collections' (`rest`).
    banked = [frozen, *(tree for path, tree, _ in frozen_trees(rest, execution.sites) if path != (FROZEN,))]
    averaged = max(placed_bytes(here.ema, device=device)
                   + placed_bytes(banked, device=device, memory="pinned_host") for device in devices)
    try:
        flops = step_flops(compiled)
        price = streamed_price(None if flops is None else split * flops, gpu_target(), dtype,
                               split * parameters, update, read_once=split * averaged)
    except Unpriced:
        price = None
    return fits, headroom, price


def host_updated(start: Start, companion: Mesh) -> int:
    """The bytes a streamed step's update reads and writes on this process's host (`host_update_seconds`'
    `updated`): each of its CPU companion's devices its shards of the parameters, the optimizer's state and
    the averaged weights, read and written, and the gradient's, read, `start` laid out streamed."""
    abstract = start.abstract
    weights, held = abstract.variables["params"], start.shardings.variables["params"]
    return sum(placed_bytes((weights, abstract.opt_state, abstract.ema),
                            (held, start.shardings.opt_state, start.shardings.ema), device=device)
               + placed_bytes(weights, held, device=device)
               for device in local_devices(companion))


def chosen(held: bool, fits: bool, device: StepPrice | None, stream: StepPrice | None) -> bool:
    """Whether the run streams: where its devices hold the streamed step (`fits`) and no plan of the given
    layout's (`held`), or hold both and the streamed step is priced the cheaper (a part that rates neither
    keeps its devices' plan)."""
    if not fits:
        return False
    if not held:
        return True
    return device is not None and stream is not None and stream.seconds < device.seconds


@contextlib.contextmanager
def noting(trainer) -> Iterator[None]:
    """Within the block, a plan or a layout the devices do not hold (`PlanRefused`, `LayoutRefused`) is noted
    with what the run's layout choice found of the streamed step (`Choice.noted`): a run never ends short of
    memory without saying whether streaming was considered, and why not."""
    try:
        yield
    except (PlanRefused, LayoutRefused) as refused:
        if trainer.layout_choice is not None:
            trainer.layout_choice.noted(refused)
        raise


def compute_dtype(trainer, start: Start) -> np.dtype:
    """The dtype the run's step computes in: its trained modules' compute dtype, each as its model states it
    (`dew.nn.protocols.ComputesIn`), the widest of them; a module that states none computes in its
    parameters' as `start` holds them (the run's `param_dtype` applied, loaded parameters their own), as Flax
    does."""
    stated = [entry.module.dtype for entry in trainer.objective.program_key()
              if entry.trained and isinstance(entry.module, ComputesIn) and entry.module.dtype is not None]
    if stated:
        return np.dtype(jnp.result_type(*stated))
    return np.dtype(computes_in(None, start.abstract.variables["params"]))


def resumed_layout(trainer, step: int | None) -> Layout | None:
    """The layout of the checkpoint `trainer`'s run resumes (as `resume` asks, `Checkpoints.resumed`, the
    step every process agrees), the given one's with the host fields its rung recorded (`host`,
    `Trainer._checkpoint_envelope`); None for a fresh run. A checkpoint that recorded none trained on the
    devices."""
    checkpoints = trainer.checkpoints
    if checkpoints is None or step is None:
        return None
    rung = checkpoints.rung(step)
    host = () if rung is None else listed(record(rung, "rung").get("host", []), "rung host", read=text,
                                          what="the layout's host fields")
    return dataclasses.replace(trainer._given_layout, host=host)


def said(choice: Choice) -> None:
    """The choice, said once a run: which plan it took and what each costs, the checkpoint's it resumed, or
    why the streamed one was not priced."""
    if choice.resumed:
        _log.info("the run resumes its checkpoint's layout: the parameters %s",
                  "streamed from the host" if choice.streams else "on the devices")
    elif choice.refused is not None:
        _log.info("the step does not stream its parameters from the host: %s", choice.refused)
    elif choice.streams:
        _log.warning("the step streams its parameters from the host (Layout(host=('variables',))): %s",
                     choice.record())
    else:
        _log.info("the step keeps its parameters on the devices: %s", choice.record())


def laid_start(trainer, run, dataset, resume) -> Start:
    """A run's start (`Trainer._started`) in its layout. A run resuming a checkpoint trained in its
    checkpoint's layout (`resumed_layout`), laid out alike. A fresh one in the layout its step is cheaper in:
    the plan its devices hold of the given layout's step (`Trainer._planned_unplaced`, which the run's first
    compile takes), priced, against the step that streams its parameters from the host, priced where its
    devices hold it, over the run's first batch (`Trainer._primed_first`, which the run then reads as its
    first), the choice every process's first's (`broadcast_from_process_zero`), said and kept
    (`Trainer.layout_choice`). A run that states a layout (`Layout()` among them: `Trainer`'s argument
    given, row 9016), or one the streamed step cannot run, keeps its layout, said by name."""
    stated = trainer._layout_stated
    # The step it resumes and its position, read once for the start (`Trainer._resumed`).
    known = trainer._resumed(run, resume)
    resumed = None if stated else resumed_layout(trainer, known.step)
    if resumed is not None:
        trainer._laid_out(resumed)
        trainer.layout_choice = Choice(held=False, fits=False, device=None, streamed=None, refused=None,
                                       streams=resumed.host == STREAMED, resumed=True)
        said(trainer.layout_choice)
        return trainer._started(resume, known)
    refused = refusal(trainer, stated=stated, ahead=trainer._compiler.ahead is not None)
    if refused is not None:
        trainer.layout_choice = Choice(held=False, fits=False, device=None, streamed=None, refused=refused,
                                       streams=False)
        said(trainer.layout_choice)
        return trainer._started(resume, known)
    # Each fit chooses afresh, from the layout it was given (a refit's of a run that streamed among them).
    given = trainer._given_layout
    trainer._laid_out(given)
    first = run.primed[2] if run.primed is not None else trainer._primed_first(run, dataset, known.position)
    # Staged as the stream the start read first stages it, as the run's first compile reads it.
    trainer._shape = staged(trainer._shape, run.stages)
    # As the loop hands its step the batch, counted on the host and its rows marked (`FitRun.marked`), so
    # the run's first compile takes the plan made here (`FitRun.searched`), where it planned
    # and compiled its step again (M-MC-26: 196 s of an H100's 375 s to its first step).
    partition = run.primed[0] if run.primed is not None else trainer._stream_partition()
    batch = handed(placed_abstract(trainer.device_mesh, marked(partition, trainer._counted_batch)(first)),
                   trainer._shape.examples)
    # The step's phase the first batch's, as the run's first compile takes it (`Trainer._compiled_fresh`),
    # and its objective, at that phase's pass count, as it stands before the device plan's search climbs
    # its ladder: the streamed layout starts from it where it is taken, both plans priced at that count.
    phases = trainer.phase_passes
    active = phases.objective_for(phases.at(first))
    rung = Rung.of(active, xla_defaults=False, narrow=True)
    try:
        start = trainer._started(resume, known)
        trainer._planned_unplaced(run, start, batch)
    except LayoutRefused:
        start = None
    plan = trainer._compiler.unplaced if start is not None else None
    # The device plan's search, released wherever the run turns from it: the streamed layout taken, or
    # the choice raising (`StepCompiler.abandoned`).
    devices_compiler = trainer._compiler
    try:
        trainer._laid_out(dataclasses.replace(given, host=STREAMED))
        # The device plan's climb and plans undone: the streamed step is priced at the rung its host step
        # starts at, the objective's own or the one its remat policy states (`StepCompiler.unsearched_rung`),
        # as it compiles there.
        devices_rung = Rung.of(active, xla_defaults=False, narrow=True)
        rung.put(active)
        trainer._compiler = devices_compiler.anew()
        trainer._compiler.unsearched_rung(active)
        laid = trainer._started(resume, known)
        dtype = compute_dtype(trainer, laid)
        unpriced = None
        try:
            device = None if plan is None else priced(plan, dtype)
        except Unpriced as rated:
            device, unpriced = None, str(rated)
        held = plan is not None
        # A held device plan gives way to the streamed step only where it is priced dearer (`chosen`), and the
        # streamed step is never priced under its update on the host (`host_update_seconds`): a held plan
        # priced no dearer than that update, or priced not at all (its dtype rated by no datasheet), is the
        # run's whatever the streamed step compiles to, so it compiles nothing (M-MC-26: the whole host
        # program, 56 s of an H100's start, for a device plan of 1.8 s against an update of 4.5 s). Every
        # process decides as the first does, as each takes the plan it chooses.
        update = host_update_seconds(host_updated(laid, companion_mesh(trainer.device_mesh)),
                                     host_memory_rate())
        kept = None if not held else (
            f"the device plan's step is unpriced, which keeps its plan ({unpriced})" if device is None
            else f"the device plan's step is priced no dearer than the streamed step's update on the host "
                 f"alone ({update:.3g} s)" if device.seconds <= update else None)
        # The first process's reason, which every process takes.
        kept = broadcast_from_process_zero(kept)
        if kept is not None:
            trainer.layout_choice = Choice(held, fits=False, device=device, streamed=None, refused=kept,
                                           streams=False)
        else:
            with phases.scoped(trainer, trainer._compiler):
                fits, headroom, price = streamed(trainer, laid, model_batch(batch), dtype,
                                                 trainer._shape.device,
                                                 update)
            trainer.layout_choice = Choice(held, fits, device, price, None,
                                           streams=broadcast_from_process_zero(
                                               chosen(held, fits, device, price)),
                                           streamed_headroom=headroom)
        said(trainer.layout_choice)
        if trainer.layout_choice.streams:
            devices_compiler.abandoned()
            return laid
    except BaseException:
        devices_compiler.abandoned()
        raise
    # The device plan taken: its compiler, which holds the plan the run's first compile takes, and its rung.
    trainer._compiler = devices_compiler
    devices_rung.put(active)
    trainer._laid_out(given)
    # The given layout's start, which its first compile takes the plan above over; or, its devices holding
    # none, its refusal, as a run that streams nothing refuses.
    return start if start is not None else trainer._started(resume, known)
