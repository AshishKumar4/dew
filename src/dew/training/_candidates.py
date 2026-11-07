"""The step a fit search lowers at each candidate (`dew.training.planning.FitSearch`): the loss traced
once for its shapes and the narrow copies it reads (`prepared_for`), the state its window of microbatches
pools, its program over one microbatch (`step_program`), lowered or traced for its footprint, with what the
fit check holds beside it (`candidate_for`), and the same step lowered for a GPU this process lacks
(`lowered_for`). The trainer reads each one's (`Trainer._planned`)."""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh, Mesh, NamedSharding, PartitionSpec as P

from dew.batch import CLOSES_WINDOW, Batch
from dew.distributed.batches import batch_shardings, prefetched_bytes
from dew.distributed.placement import Placement, placed_bytes, placed_shapes
from dew.nn.kernels import tuning
from dew.nn.kernels.generation import assumed, gpu_mesh
from dew.nn.kernels.options import owned_program
from dew.nn.sharding import Schedule
from dew.state import TrainState, apart, together
from dew.training import staging
from dew.training._lowering import KernelReport, Routed
from dew.training.diloco import group_of, windowed
from dew.training.distributed import microbatch, microbatch_shapes
from dew.training.footprint import Footprint, candidate_footprint
from dew.training.memory import local_devices
from dew.training.model_flops import batch_consumed, batch_tokens
from dew.training.narrow import narrow_copies, narrowed
from dew.training.planning import Candidate, splits
from dew.training.transaction import COMMIT_FIELDS, consumed_where_made, outcome, pooled_by_mean

if TYPE_CHECKING:
    from dew.training.trainer import Trainer


def candidate_for[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], state: TrainState, batch: Batch,
                                   schedule: Schedule, placement, replicated, split: int, narrow: bool | None,
                                   launches: str | None, lowering: bool = True, *,
                                   mesh: Mesh | AbstractMesh | None = None,
                                   platforms: tuple[str, ...] | None = None, capacity: int | None = None,
                                   footprint: Footprint | None = None) -> Candidate:
    """The step reading `batch` in `split` microbatches, lowered at the objective's current rung (traced
    alone where not `lowering`, for its derived footprint), with what the fit check holds beside it.
    `launches` are its kernel launches where a record holds them, which settle their configs before it
    lowers; otherwise, or where the run measures configs (`tune_kernels`), its lowering records them
    (`tune`). `mesh` and `platforms` lower it for other devices, untuned (`lowered_for`). `capacity` is
    the bytes a device that states none holds (`FitSearch.capacity`), at which its footprint is judged;
    `footprint`, the one its derivation traced, which it keeps (`FitSearch.footprints`)."""
    micro = microbatch_shapes(batch, split)
    shapes, copies, prepared = prepared_for(trainer, state, micro, schedule, split, narrow)
    shardings = trainer.shardings(prepared)
    if mesh is not None:
        shardings = jax.tree.map(lambda s: NamedSharding(mesh, s.spec, memory_kind=s.memory_kind),
                                 shardings)
    prepared = placed_shapes(prepared, shardings)
    # The check reads the state's arguments as already on the devices. The copies and accumulators
    # `run` builds on the first call are not yet, so they count with the prefetched batches beside
    # the step, a ramp stage's window too, which `run` builds beside the last, already resident.
    last = state.window
    rebuilt = last is not None and not last.replays(prepared.window)
    # A step lowered for other devices is checked against none of this process's.
    built = 0 if mesh is not None else placed_bytes([field for field, kept in (
        (prepared.compute, state.compute), (prepared.window, None if rebuilt else last))
        if kept is None and field is not None])
    held = 0 if mesh is not None else prefetched_bytes(batch, placement, trainer._prefetch) + built
    jitted = step_program(trainer, shapes, copies, shardings, placement, replicated, split)
    part = jax.ShapeDtypeStruct((), jnp.int32, sharding=replicated)
    # The batch as shapes with the placement it is read under, whoever lowers it: a lowering over the host
    # arrays a step compiled ahead reads (`first_step_batch`) and one over the placed arrays a run steps
    # are two texts of one program (`dew.training.ahead.program_of`), the types of a helper call's
    # arguments, their shardings among them, telling its functions apart in the module.
    given = placed_shapes(batch, placement)
    lower = Routed(functools.partial(
        lambda *arguments: jitted.trace(*arguments).lower(lowering_platforms=platforms),
        *apart(prepared), given, part))
    # The bytes its host-resident commit fields hold at once are its own staged cut's
    # (`staging.measured`): read as the step is traced, by its lowering or, lowering none, a trace alone.
    staging_measured = (trainer.step is None and mesh is None
                        and bool(set(COMMIT_FIELDS) & set(trainer.layout.host)))
    with staging.measured() as cuts:
        if trainer.tune_kernels and lowering and platforms is None:
            # The configs it measures are not the ones a derivation traced: its footprint is traced again.
            lowered, launches, footprint = *tuning.tune(lower, measure=True), None
        else:
            # The tuner is off (bible §6): each launch takes its derived config, nothing settled or read.
            lowered, launches = lower() if lowering else None, launches or "[]"
        if staging_measured and not lowering:
            jitted.trace(*apart(prepared), given, part)
    window = split * trainer.accumulation
    if footprint is None and shapes is not None and mesh is None:
        footprint = candidate_footprint(
            trainer.objective, trainer._model_shapes(micro), window, prepared, held,
            (local_devices(trainer.device_mesh) or [*trainer.device_mesh.devices.flat])[0], split,
            functools.partial(consumes_where_made, trainer, prepared, micro, window, shapes),
            max(cuts, default=0), capacity)
    # The sampled step lowers as the step does, over the same arguments, for the same platforms.
    watched = None if trainer.inspection is None else step_program(
        trainer, shapes, copies, shardings, placement, replicated, split, sampled=True)
    sampled = None if watched is None else functools.partial(
        lambda *arguments: watched.trace(*arguments).lower(lowering_platforms=platforms),
        *apart(prepared), given, part)
    return Candidate(micro, shapes, copies, shardings, prepared, held, built, lowered, launches,
                     lower.report() if lowering else KernelReport((), ()), footprint, sampled)


def consumes_where_made[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], prepared: TrainState,
                                         micro: Batch, window: int, shapes) -> bool:
    """Whether the step over `prepared` consumes each gradient where it is made
    (`transaction.consumed_where_made`), over the state its commit reads: under DiLoCo a group's own
    (`group_of`), not the outer optimizer's."""
    with group_of(prepared, micro) as (view, _):
        return consumed_where_made(trainer.optimizer, view, window, shared=pooled_by_mean(shapes[0]),
                                   progressed=trainer._examples is not None)


def prepared_for[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], state: TrainState, batch: Batch,
                                  schedule: Schedule, split: int, narrow: bool | None = None):
    """The loss's shapes (None under a custom step), the narrow copies and the state's shapes with the
    accumulation of an update's window, `split` microbatches a step (DiLoCo's groups' `windowed`), and
    the copies the step builds, as it traces them over microbatches of `batch`'s shapes, none where
    `narrow` is off."""
    window = split * trainer.accumulation
    with group_of(state, batch) as (view, rows):
        traced, shapes = (None, None) if trainer.step is not None else trainer._loss_shape(
            view, rows, window, split)
    if shapes is not None:
        trainer._staging.refuse_unpipelined(schedule, trainer.objective)
    prepared = windowed(lambda state, rows, *, shape_only: trainer._initialize_accumulation(
        state, rows, shapes, window=window, shape_only=shape_only), state, batch)
    # DiLoCo's groups read their own parameters, which no narrow copy shadows.
    copies = (narrow_copies(trainer.objective, state, batch, trainer._model_batch, window=window, split=split,
                            composed=trainer.step is None and not trainer.dynamic_scale, traced=traced)
              if (trainer._narrow if narrow is None else narrow) and trainer.diloco is None else {})
    prepared = dataclasses.replace(prepared, compute=jax.eval_shape(
        lambda variables: narrowed(variables, copies), prepared.variables) if copies else None)
    return shapes, copies, prepared


def step_program[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], shapes, copies: dict,
                                  shardings: Placement[TrainState], placement, replicated: NamedSharding,
                                  split: int, sampled: bool = False):
    """The step over one of the `split` microbatches of a batch
    (`microbatch`), which counts the step at its last microbatch. It takes the state apart
    (`apart`): the moving state, which it donates and returns advanced, and the frozen tree, which
    it reads and neither donates nor returns, so the frozen arrays stay the ones the run (and the
    objective's held tree) holds, and no output copies them."""
    # The transaction brings the optimizer state and the EMA the layout keeps on the host to the device
    # once its gradient is ready (`Transaction.past`); a custom step, and every other field, reads them
    # from its start.
    late = () if trainer.step is not None else tuple(set(COMMIT_FIELDS) & set(trainer.layout.host))
    body = trainer._step_body(shapes, copies, split, {field: getattr(shardings, field) for field in late},
                           sampled, shardings.variables["params"])
    body = body if trainer.diloco is None else trainer.diloco.step(body, trainer.device_mesh, split)
    moving, frozen = apart(shardings)

    def step(current, held, batch, part):
        # The body sees every field on the device; the out shardings
        # return the host-resident ones to pinned host memory.
        current = together(current, held)
        micro = microbatch(batch, part, split)
        early = [field for field in trainer.layout.host if field not in late]
        advanced, loss, aux, committed = body(trainer._fetched(current, shardings, early),
                                              trainer._model_batch(micro), part, batch_tokens(micro),
                                              micro.get(CLOSES_WINDOW))
        stepped, *rest = outcome(current, advanced, loss, aux, committed, batch, part == split - 1,
                                 batch_consumed(micro))
        return (apart(stepped)[0], *rest)

    returned = (moving, replicated, replicated, replicated, replicated, replicated)
    return owned_program(step, in_shardings=(moving, frozen, placement, replicated),
                         out_shardings=returned, donate_argnums=0)


def lowered_for[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], state: TrainState, batch: Batch,
                                 generation: str) -> jax.stages.Lowered:
    """The step `compile` builds first, traced as on a GPU of the CUDA `generation` and lowered for one
    (`dew.training._lowering`) on this process's devices. It traces on an abstract mesh of that GPU's
    devices, from which JAX's Pallas lowering reads the GPU, at the base rung and the plan's first split:
    a run that does not fit splits its batch or recomputes more, no other kernel. A CPU-owned state's step
    realizes its parameters on the host as it runs, and is not lowered."""
    if trainer.host_master:
        raise ValueError("lower_for lowers a step whose state the devices hold; this layout keeps "
                         "the variables on the host")
    mesh = gpu_mesh(trainer.device_mesh, generation)
    trainer._at_base_rung(batch)
    with assumed(generation), trainer._traced_on(mesh) as schedule:
        state, batch = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), (state, batch))
        split = splits(trainer._rows, batch, mesh, trainer.microbatches, trainer._split)[0]
        placement, replicated = batch_shardings(mesh, batch), NamedSharding(mesh, P())
        return candidate_for(trainer, state, batch, schedule, placement, replicated, split, None, None,
                               mesh=mesh, platforms=('cuda',)).lowered
