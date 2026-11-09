"""The step a fit search lowers at each candidate (`dew.training.planning.FitSearch`): the loss traced
once for its shapes and the narrow copies it reads (`prepared_for`), the state its window of microbatches
pools, its program over one microbatch (`step_program`), lowered or traced for its footprint, with what the
fit check holds beside it (`candidate_for`), and the same step lowered for a GPU this process lacks
(`lowered_for`). The trainer reads each one's (`Trainer._planned`)."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import threading
from collections.abc import Callable, Hashable, Mapping
from typing import TYPE_CHECKING

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AbstractMesh, Mesh, NamedSharding, PartitionSpec as P

from dew.batch import CLOSES_WINDOW, Batch
from dew.distributed.batches import batch_shardings, prefetched_bytes
from dew.distributed.placement import Placement, placed_bytes, placed_shapes
from dew.nn.kernels import tuning
from dew.nn.kernels.generation import assumed, gpu_mesh
from dew.nn.kernels.options import owned_program, trace_context
from dew.nn.kernels.templates import report
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
from dew.training.stages import StageLocal
from dew.training.transaction import COMMIT_FIELDS, consumed_where_made, outcome, pooled_by_mean

if TYPE_CHECKING:
    from dew.training.trainer import Trainer


@dataclasses.dataclass(frozen=True)
class Stepping[LossT, EffectsT]:
    """What a fit search's candidates read of the run beside its trainer's public fields, as the trainer
    stands as it plans (`Trainer._stepping`): its loss's traced shapes (`loss_shape`), where its step keeps
    a pipelined run (`staging`), its window's accumulation (`initialize_accumulation`), its model's batch
    and its shapes (`model_batch`, `model_shapes`), its step's body (`step_body`) and the fields it fetches
    (`fetched`), the batches it prefetches (`prefetch`), the examples clock it runs on (`examples`), the
    mesh it traces on (`traced_on`), and the rows and split its batches are cut in (`rows`, `split`): read
    here, none off the trainer's private members (row 4555)."""

    trainer: Trainer[LossT, EffectsT]
    loss_shape: Callable[..., tuple]
    staging: StageLocal
    initialize_accumulation: Callable[..., TrainState]
    model_batch: Callable[[Batch], Batch]
    model_shapes: Callable[[Batch], Batch]
    step_body: Callable[..., Callable]
    fetched: Callable[..., TrainState]
    prefetch: int
    examples: int | None
    traced_on: Callable[[Mesh | AbstractMesh], contextlib.AbstractContextManager[Schedule]]
    rows: tuple[int, ...] | None
    split: int | None


KEPT_CANDIDATES = 4
"""The traced steps a process keeps for a trainer built anew to read (`candidate_for`), the latest: each
holds its lowered module."""

_CANDIDATES: collections.OrderedDict[Hashable, Candidate] = collections.OrderedDict()
_KEEPING = threading.Lock()


class _Unkeyed:
    """A part of a step's settings that states no value (a function, an object compared by identity)."""


UNKEYED = _Unkeyed()


def frozen_value(value) -> Hashable:
    """`value` as a value a step's trace can be kept by: a mapping as its sorted items, a sequence as a
    tuple, a dataclass as its type and fields, recursively, a number, string or other hashable value as
    itself (a Flax module by its fields); `UNKEYED` where any part is a function, or an object of no value
    (one hashed by its identity)."""
    if isinstance(value, Mapping):
        items = tuple(sorted((key, frozen_value(item)) for key, item in value.items()))
        return UNKEYED if any(item is UNKEYED for _, item in items) else items
    if isinstance(value, list | tuple):
        parts = tuple(map(frozen_value, value))
        return UNKEYED if any(part is UNKEYED for part in parts) else (type(value), parts)
    if dataclasses.is_dataclass(value) and not isinstance(value, type | nn.Module):
        fields = tuple((field.name, frozen_value(getattr(value, field.name)))
                       for field in dataclasses.fields(value))
        return UNKEYED if any(item is UNKEYED for _, item in fields) else (type(value), fields)
    if ((callable(value) and not isinstance(value, nn.Module))
            or type(value).__hash__ in (None, object.__hash__)):
        return UNKEYED
    return value


def _shaped(tree) -> Hashable:
    """`tree`'s structure and each leaf's abstract value and placement (an array's, a NumPy array's, a
    shape's, a scalar's beside its type, which tells True from 1), or the leaf itself where it is none."""
    def leaf(value) -> Hashable:
        if isinstance(value, jax.Array | jax.ShapeDtypeStruct | np.ndarray | np.generic | bool | int | float
                      | complex):
            aval = jax.typeof(value)
            return type(value), aval.shape, aval.dtype, aval.weak_type, getattr(value, "sharding", None)
        return value
    leaves, structure = jax.tree.flatten(tree)
    return structure, tuple(map(leaf, leaves))


def _shapes(tree):
    """`tree` with each array leaf as its shape, dtype and placement, as a kept step holds a batch it
    traced over and none of its values."""
    return jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(np.shape(leaf), leaf.dtype,
                                                          sharding=getattr(leaf, "sharding", None))
                        if hasattr(leaf, "dtype") else leaf, tree)


def _footprinted(footprint: Footprint | None) -> Hashable:
    """`footprint` as a value, its resident bytes by name in order."""
    if footprint is None:
        return None
    return tuple((field.name, tuple(sorted(value.items())) if field.name == "resident" else value)
                 for field in dataclasses.fields(footprint) for value in (getattr(footprint, field.name),))


def candidate_for[LossT, EffectsT](stepping: Stepping[LossT, EffectsT], state: TrainState, batch: Batch,
                                   schedule: Schedule, placement, replicated, split: int, narrow: bool,
                                   launches: str | None, lowering: bool = True, *,
                                   mesh: Mesh | AbstractMesh | None = None,
                                   platforms: tuple[str, ...] | None = None, capacity: int | None = None,
                                   footprint: Footprint | None = None, derived: bool = True) -> Candidate:
    """`_candidate`'s step, or the one an equal trainer traced in this process over the same shapes
    (`Trainer._step_key`; row 3873: each fit of a resumed run traced its step again, 6.5 to 12.9 s, and
    lowered it, 9 s), its pipeline's schedule the one its microbatches make; a trainer whose step no value
    states traces its own."""
    stated = stepping.trainer._step_key()
    # What the step traces from beside its arguments and the trainer's settings: the compiles' refusals
    # its kernels' derivation reads (a relowering after a spill reads its proposal's), and the context.
    key = None if stated is None else (
        stated, _shaped(state), _shaped(batch), _shaped(placement), replicated, split, narrow,
        # No launch record lowers as an empty one (`_candidate`'s untuned lowering).
        launches or "[]",
        lowering, mesh, platforms, capacity, _footprinted(footprint), derived, stepping.rows, stepping.split,
        stepping.prefetch, stepping.examples, report.refused_launches(), trace_context())
    if key is not None:
        with _KEEPING:
            if (kept := _CANDIDATES.get(key)) is not None:
                _CANDIDATES.move_to_end(key)
                sampled = _sampled(stepping, kept, batch, placement, replicated, split, platforms)
                return dataclasses.replace(kept, sampled=sampled)
    candidate = _candidate(stepping, state, batch, schedule, placement, replicated, split, narrow, launches,
                           lowering, mesh=mesh, platforms=platforms, capacity=capacity, footprint=footprint,
                           derived=derived)
    if key is not None:
        # Kept detached from this trainer: its batch as shapes, its inspection's step none (`_sampled`
        # builds the reader's own).
        with _KEEPING:
            _CANDIDATES[key] = dataclasses.replace(candidate, micro=_shapes(candidate.micro), sampled=None)
            while len(_CANDIDATES) > KEPT_CANDIDATES:
                _CANDIDATES.popitem(last=False)
    return candidate


def _sampled[LossT, EffectsT](stepping: Stepping[LossT, EffectsT], candidate: Candidate, batch: Batch,
                              placement, replicated, split: int, platforms: tuple[str, ...] | None
                              ) -> Callable[[], jax.stages.Lowered] | None:
    """The sampled step `candidate`'s reader lowers (`Candidate.sampled`), its own trainer's, as the step
    lowers, over the same arguments, for the same platforms: none without an inspection."""
    if stepping.trainer.inspection is None:
        return None
    watched = step_program(stepping, candidate.shapes, candidate.copies, candidate.shardings, placement,
                           replicated, split, sampled=True)
    part = jax.ShapeDtypeStruct((), jnp.int32, sharding=replicated)
    given = placed_shapes(batch, placement)
    return functools.partial(lambda *arguments: watched.trace(*arguments).lower(lowering_platforms=platforms),
                             *apart(candidate.prepared), given, part)


def _candidate[LossT, EffectsT](stepping: Stepping[LossT, EffectsT], state: TrainState, batch: Batch,
                                schedule: Schedule, placement, replicated, split: int, narrow: bool,
                                launches: str | None, lowering: bool = True, *,
                                mesh: Mesh | AbstractMesh | None = None,
                                platforms: tuple[str, ...] | None = None, capacity: int | None = None,
                                footprint: Footprint | None = None, derived: bool = True) -> Candidate:
    """The step reading `batch` in `split` microbatches, lowered at the objective's current rung (traced
    alone where not `lowering`, for its derived footprint), with what the fit check holds beside it.
    `launches` are its kernel launches where a record holds them, which settle their configs before it
    lowers; otherwise, or where the run measures configs (`tune_kernels`), its lowering records them
    (`tune`). `mesh` and `platforms` lower it for other devices, untuned (`lowered_for`). `capacity` is
    the bytes a device that states none holds (`FitSearch.capacity`), at which its footprint is judged;
    `footprint`, the one its derivation traced, which it keeps (`FitSearch.footprints`). One whose
    executable is a step compiled ahead derives none (not `derived`): its compile judged the
    derivation where it was compiled, and a run handed it does none of that work on its devices."""
    trainer = stepping.trainer
    micro = microbatch_shapes(batch, split)
    shapes, copies, prepared = prepared_for(stepping, state, micro, schedule, split, narrow)
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
    held = 0 if mesh is not None else prefetched_bytes(batch, placement, stepping.prefetch) + built
    jitted = step_program(stepping, shapes, copies, shardings, placement, replicated, split)
    part = jax.ShapeDtypeStruct((), jnp.int32, sharding=replicated)
    # The batch as shapes with the placement it is read under, whoever lowers it: a lowering over the host
    # arrays a step compiled ahead reads (`first_step_batch`) and one over the placed arrays a run steps
    # are two texts of one program (`dew.nn.kernels.options.program_digest`), the types of a helper call's
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
    if derived and footprint is None and shapes is not None and mesh is None:
        footprint = candidate_footprint(
            trainer.objective, stepping.model_shapes(micro), window, prepared, held,
            (local_devices(trainer.device_mesh) or [*trainer.device_mesh.devices.flat])[0], split,
            functools.partial(consumes_where_made, stepping, prepared, micro, window, shapes),
            max(cuts, default=0), capacity)
    candidate = Candidate(micro, shapes, copies, shardings, prepared, held, built, lowered, launches,
                          lower.report() if lowering else KernelReport((), ()), footprint)
    return dataclasses.replace(candidate, sampled=_sampled(stepping, candidate, batch, placement, replicated,
                                                           split, platforms))


def consumes_where_made[LossT, EffectsT](stepping: Stepping[LossT, EffectsT], prepared: TrainState,
                                         micro: Batch, window: int, shapes) -> bool:
    """Whether the step over `prepared` consumes each gradient where it is made
    (`transaction.consumed_where_made`), over the state its commit reads: under DiLoCo a group's own
    (`group_of`), not the outer optimizer's."""
    trainer = stepping.trainer
    with group_of(prepared, micro) as (view, _):
        return consumed_where_made(trainer.optimizer, view, window, shared=pooled_by_mean(shapes[0]),
                                   progressed=stepping.examples is not None)


def prepared_for[LossT, EffectsT](stepping: Stepping[LossT, EffectsT], state: TrainState, batch: Batch,
                                  schedule: Schedule, split: int, narrow: bool):
    """The loss's shapes (None under a custom step), the narrow copies and the state's shapes with the
    accumulation of an update's window, `split` microbatches a step (DiLoCo's groups' `windowed`), and
    the copies the step builds, as it traces them over microbatches of `batch`'s shapes, none where
    `narrow` is off."""
    trainer = stepping.trainer
    window = split * trainer.accumulation
    with group_of(state, batch) as (view, rows):
        traced, shapes = (None, None) if trainer.step is not None else stepping.loss_shape(
            view, rows, window, split)
    if shapes is not None:
        stepping.staging.refuse_unpipelined(schedule, trainer.objective)
    prepared = windowed(lambda state, rows, *, shape_only: stepping.initialize_accumulation(
        state, rows, shapes, window=window, shape_only=shape_only), state, batch)
    # DiLoCo's groups read their own parameters, which no narrow copy shadows.
    copies = (narrow_copies(trainer.objective, state, batch, stepping.model_batch, window=window, split=split,
                            composed=trainer.step is None and not trainer.dynamic_scale, traced=traced)
              if narrow and trainer.diloco is None else {})
    prepared = dataclasses.replace(prepared, compute=jax.eval_shape(
        lambda variables: narrowed(variables, copies), prepared.variables) if copies else None)
    return shapes, copies, prepared


def step_program[LossT, EffectsT](stepping: Stepping[LossT, EffectsT], shapes, copies: dict,
                                  shardings: Placement[TrainState], placement, replicated: NamedSharding,
                                  split: int, sampled: bool = False):
    """The step over one of the `split` microbatches of a batch
    (`microbatch`), which counts the step at its last microbatch. It takes the state apart
    (`apart`): the moving state, which it donates and returns advanced, and the frozen tree, which
    it reads and neither donates nor returns, so the frozen arrays stay the ones the run (and the
    objective's held tree) holds, and no output copies them."""
    trainer = stepping.trainer
    # The transaction brings the optimizer state and the EMA the layout keeps on the host to the device
    # once its gradient is ready (`Transaction.past`); a custom step, and every other field, reads them
    # from its start.
    late = () if trainer.step is not None else tuple(set(COMMIT_FIELDS) & set(trainer.layout.host))
    body = stepping.step_body(shapes, copies, split, {field: getattr(shardings, field) for field in late},
                              sampled, shardings.variables["params"])
    body = body if trainer.diloco is None else trainer.diloco.step(body, trainer.device_mesh, split)
    moving, frozen = apart(shardings)

    def step(current, held, batch, part):
        # The body sees every field on the device; the out shardings
        # return the host-resident ones to pinned host memory.
        current = together(current, held)
        micro = microbatch(batch, part, split)
        early = [field for field in trainer.layout.host if field not in late]
        advanced, loss, aux, committed = body(stepping.fetched(current, shardings, early),
                                              stepping.model_batch(micro), part, batch_tokens(micro),
                                              micro.get(CLOSES_WINDOW))
        stepped, *rest = outcome(current, advanced, loss, aux, committed, batch, part == split - 1,
                                 batch_consumed(micro))
        return (apart(stepped)[0], *rest)

    returned = (moving, replicated, replicated, replicated, replicated, replicated)
    return owned_program(step, in_shardings=(moving, frozen, placement, replicated),
                         out_shardings=returned, donate_argnums=0)


def lowered_for[LossT, EffectsT](stepping: Stepping[LossT, EffectsT], state: TrainState, batch: Batch,
                                 generation: str, narrow: bool) -> jax.stages.Lowered:
    """The step `compile` builds first, traced as on a GPU of the CUDA `generation` and lowered for one
    (`dew.training._lowering`) on this process's devices. It traces on an abstract mesh of that GPU's
    devices, from which JAX's Pallas lowering reads the GPU, at the base rung its trainer put the
    objective at (`Trainer._at_base_rung`), its narrow copies where that rung keeps them (`narrow`), and
    the plan's first split: a run that does not fit splits its batch or recomputes more, no other kernel.
    A CPU-owned state's step realizes its parameters on the host as it runs, and is not lowered."""
    trainer = stepping.trainer
    if trainer.host_master:
        raise ValueError("lower_for lowers a step whose state the devices hold; this layout keeps "
                         "the variables on the host")
    mesh = gpu_mesh(trainer.device_mesh, generation)
    with assumed(generation), stepping.traced_on(mesh) as schedule:
        state, batch = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), (state, batch))
        split = splits(stepping.rows, batch, mesh, trainer.microbatches, stepping.split)[0]
        placement, replicated = batch_shardings(mesh, batch), NamedSharding(mesh, P())
        return candidate_for(stepping, state, batch, schedule, placement, replicated, split, narrow, None,
                               mesh=mesh, platforms=('cuda',)).lowered
