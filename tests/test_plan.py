"""A step's plan: the fewest microbatches at the least recomputation that fit.

Splitting a step's batch must be invisible to everything except memory use:
the update pools the same rows, the clocks count the same steps, and a
resumed run reads its batches as the run it continues did.
"""

import collections
import contextlib
import dataclasses
import functools
import gc
import itertools
import json
import logging
import struct
import types
import weakref
from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import FEATURES, Affine, Regression
from arm_arithmetic import float64_held
from export_support import Stalled
from jax.extend.backend import get_backend
from map_judge import marked_seeds, sharded
from port_support import row_names_port
from process_support import assert_exits_cleanly
from reference_error import as_given, assert_as_exact
from sharded import collectives
from token_support import write_token_corpus

from dew import _cache
from dew._records import JSON
from dew.checkpoints import Checkpoints
from dew.data import Loading, Memory, presets
from dew.data.compile import compile
from dew.data.pipeline import Pipeline
from dew.data.sources.tokens import TokenFiles
from dew.diffusion.discrete import MDLM
from dew.distributed.placement import Layout, MeshSpec, placed_bytes
from dew.mesh import LayoutRefused
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import DECODER_REMAT, REMAT_POLICIES, remat_record
from dew.nn.kernels import options
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.objectives.lm import LMObjective
from dew.objectives.lm.chunked import chunked_tile
from dew.optim import Muon, Optax, OptimConfig
from dew.registry import mixers
from dew.state import TrainState, apart
from dew.telemetry.compiles import Compiles
from dew.training import FitPolicy, Trainer, UpdateBatch
from dew.training.compiler import UNPLACED_REFUSED
from dew.training.distributed import microbatch
from dew.training.memory import (
    FirstRunFailed,
    climb_to,
    first_run,
    local_devices,
    memory_record,
    recompute_more,
    rung_record,
)
from dew.training.planning import PlanRefused, Trials, splits, walked

ROWS, SEQ = 32, 16
BATCH = {"text": jnp.asarray((np.arange(ROWS * (SEQ + 1)).reshape(ROWS, SEQ + 1) * 7) % 64, jnp.int32)}


@pytest.mark.parametrize("fitting, first, computed, plan, asked", [
    # The batch whole fits: one compile.
    (lambda split: True, 0, {}, 1, [1]),
    # The whole batch's miss computes two microbatches, which fit: two compiles.
    (lambda split: split >= 2, 0, {1: 1}, 2, [1, 2]),
    # It computes four, which fit, past two not asked: two is asked, and fits, the fewest.
    (lambda split: split >= 2, 0, {1: 2}, 2, [1, 4, 2]),
    # Four fit and two miss: four, the fewest.
    (lambda split: split >= 4, 0, {1: 2}, 4, [1, 4, 2]),
    # Its miss computes two, whose program misses too, and that miss computes four.
    (lambda split: split >= 4, 0, {1: 1, 2: 2}, 4, [1, 2, 4]),
    # Its miss computes none, and the finest, asked, misses: the rung cannot fit.
    (lambda split: False, 0, {1: 3}, None, [1, 8]),
    # A recorded plan is asked first and kept where it fits.
    (lambda split: split >= 2, 2, {}, 4, [4]),
])
def test_a_rungs_splits_are_asked_as_each_miss_computes(fitting, first, computed, plan, asked):
    """At one rung the split asked first, then, past each miss, the split that miss computes as the first
    that can fit (`after`, an index into the candidates, the finest where it computes none), until one fits,
    then each split before it not asked until one misses. The plan is the least that fits, a compile of its
    own judged it, and the search compiles no split twice and no more of them than `asked` lists, the cost
    the misses' computed splits save; the order it asks them in is its own (row 5904)."""
    compiled = []
    candidates = [1, 2, 4, 8]

    def fits(split):
        compiled.append(split)
        return fitting(split)

    def after(index):
        return computed.get(candidates[index])

    assert walked(candidates, first, fits, after) == plan
    assert plan is None or plan in compiled
    assert len(set(compiled)) == len(compiled) <= len(asked)


def test_every_process_gathers_the_split_asked_after_each_miss():
    """The split asked after a miss is the latest any process computes (`FitSearch.after`): every process
    gathers at every miss, its own compile read or not (a compile XLA refused for memory, or one whose
    memory this process states none of), so their rounds stay in step; past the last split none is asked
    and none gathers."""
    from types import SimpleNamespace

    from dew.training.memory import MemoryMiss
    from dew.training.planning import FitSearch

    gathered = []

    def gather(value):
        gathered.append(value)
        return [value, 2]

    def search(misses):
        return SimpleNamespace(misses=misses, rung=lambda: "rung", gather=gather)

    candidates = [1, 2, 4, 8]
    # This process read nothing of its miss and asks the next; another computes four, which all ask.
    assert FitSearch.after(search({}), candidates, 0) == 2 and gathered == [1]
    assert FitSearch.after(search({("rung", 1): MemoryMiss(1, 90, 50, 1)}), candidates, 0) == 2
    assert gathered == [1, 2]
    assert FitSearch.after(search({}), candidates, 3) is None and gathered == [1, 2]
    # A miss whose figures leave no split above it room asks the finest, XLA's compile the judge of it.
    assert FitSearch.after(search({("rung", 1): MemoryMiss(1, 30, 50, 1)}), candidates, 0) == 3


def test_every_process_keeps_or_drops_the_narrow_copies_together():
    """Whether the ladder keeps the narrow copies (`FitSearch.freed`) is every process's verdict
    gathered: a process whose step without them fell less short, beside one whose fell no less short,
    keeps them as that one does, so no two processes lower different steps and enter different compiles'
    collectives; and where every process's drop freed bytes, every one keeps them dropped."""
    from types import SimpleNamespace

    from dew.training.memory import MemoryMiss
    from dew.training.planning import FitSearch

    def search(without: int, others: bool):
        misses = {("with", 1): MemoryMiss(1, 50, 50, 1), ("without", 1): MemoryMiss(1, 50, without, 1)}
        return SimpleNamespace(misses=misses, gather=lambda held: [held, others])

    # This process's drop helped (40 short, where 50 with them); another's did not: both keep the copies.
    assert FitSearch.freed(search(40, others=True), "with", "without") is False
    # Neither process's drop left it no less short: both keep them dropped.
    assert FitSearch.freed(search(40, others=False), "with", "without") is True
    # This process's drop did not help: it keeps them, whatever another's did.
    assert FitSearch.freed(search(60, others=False), "with", "without") is False


def test_a_start_whose_head_alone_fits_tiled_settles_it_at_no_recomputation(monkeypatch):
    """A decoder whose 4,096-word head's logits are most of its step, read whole and without narrow copies,
    given room for its step with that head tiled and recomputing nothing: no remat rung at its whole head
    fits, so past them the search climbs the whole ladder from its own rung and settles the tiled head at no
    recomputation, not a recomputing rung's (row 6740's order: the ladder read from the start's rung, not
    from the last remat rung asked)."""
    from dew.nn.kernels.generation import chosen

    def decoder():
        return lm(split=1, vocab=4096)

    with chosen(narrow_copies="off"):
        tiled = decoder()
        assert tiled.objective.tile_head() is not None
        room = needs(monkeypatch, tiled)
        free(monkeypatch, room)
        trainer = decoder()
        state, _, _ = trainer.place()
        trainer.compile(state, BATCH, Trials(state, BATCH))
    probes = trainer.plan["probes"]
    assert any(probe["remat"] is not None for probe in probes), probes
    assert trainer.plan["head_tile"] is not None and trainer.plan["remat"] is None, probes


def test_a_miss_computes_the_split_its_temporaries_leave_room_at():
    """A compile that missed by its `shortfall`, its stranding counted: a split above it holds what its
    rows scale at their share (`MemoryMiss.fits_at`)."""
    from dew.training.memory import MemoryMiss

    # The batch whole: 90 bytes of temporaries its rows scale, 50 short.
    whole = MemoryMiss(1, 90, 50, 1)
    assert not whole.fits_at(2) and whole.fits_at(4)
    # 30 bytes of temporaries: they fall by no more than 30, never by 50.
    assert not any(MemoryMiss(1, 30, 50, 1).fits_at(split) for split in (2, 4, 8, 64))
    # An allocator that strands its temporaries needs half the bytes back.
    assert MemoryMiss(1, 90, 50, 2).fits_at(2)


def test_the_fit_ladder_tiles_the_head_fused_then_unfused_before_it_recomputes_more():
    """At each remat rung the ladder keeps the whole logits, then tiles the head with its gradient in the
    forward, the faster tiled head, then in its backward, whose tile holds less than the fused chunk's
    whole vocabulary, before it recomputes more with the whole logits back; a resumed run climbs to a
    recorded unfused rung alike."""
    def objective():
        model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                                  max_seq_len=SEQ)
        return LMObjective(model, seq_len=SEQ)

    climbing = objective()
    rungs = [rung_record(climbing, xla_defaults=False, narrow=True)]
    while len(rungs) < 5 and recompute_more(climbing):
        rungs.append(rung_record(climbing, xla_defaults=False, narrow=True))
    tile = list(chunked_tile())
    assert [(rung["head_tile"], rung["head_fused"], rung["remat"]) for rung in rungs] == [
        (None, False, None), (tile, True, None), (tile, False, None), (None, False, "minimal"),
        (tile, True, "minimal")]
    resumed = objective()
    climb_to(resumed, rungs[2])
    assert rung_record(resumed, xla_defaults=False, narrow=True) == rungs[2]
    # A head put back whole, as a new shape starts from the run's own rung, tiles fused again.
    resumed.restore_head(None, fused=False)
    assert resumed.tile_head() is not None
    assert rung_record(resumed, xla_defaults=False, narrow=True)["head_fused"]


def test_the_fit_ladder_tiles_a_masked_diffusion_head_before_it_recomputes_more():
    """A masked-diffusion objective runs its head by the policy every token objective takes
    (`dew.objectives.lm.head`): the ladder keeps its whole logits, then tiles it, its gradient in the
    backward as its loss fuses none, before it recomputes more with the whole logits back; a resumed
    run climbs to a recorded tiled rung alike."""
    model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=SEQ, causal=False)

    def objective():
        return MaskedDiffusionObjective(model, MDLM(mask_id=63)(), seq_len=SEQ)

    climbing = objective()
    rungs = [rung_record(climbing, xla_defaults=False, narrow=True)]
    while len(rungs) < 4 and recompute_more(climbing):
        rungs.append(rung_record(climbing, xla_defaults=False, narrow=True))
    tile = list(chunked_tile())
    assert [(rung["head_tile"], rung["head_fused"], rung["remat"]) for rung in rungs] == [
        (None, False, None), (tile, False, None), (None, False, "minimal"), (tile, False, "minimal")]
    resumed = objective()
    climb_to(resumed, rungs[1])
    assert rung_record(resumed, xla_defaults=False, narrow=True) == rungs[1]


def test_a_microbatch_is_every_splitth_row():
    """The `part`th microbatch takes rows part, part + split, ...
    (`microbatched`), a share of every device's rows; a field with no rows
    is the step's and every microbatch reads it whole."""
    batch = {"rows": jnp.arange(16).reshape(8, 2), "scale": jnp.asarray(3.0)}
    parts = [microbatch(batch, jnp.int32(part), 2) for part in range(2)]
    assert [part["rows"][:, 0].tolist() for part in parts] == [[0, 4, 8, 12], [2, 6, 10, 14]]
    assert all(float(part["scale"]) == 3.0 for part in parts)
    assert microbatch(batch, jnp.int32(0), 1) is batch


@pytest.mark.mesh(devices=4)
def test_a_microbatch_moves_no_row_between_devices():
    """Over four row shards the slice is local: the compiled program holds
    no collective."""
    mesh = MeshSpec(fsdp=4).build(jax.devices()[:4])
    rows = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp"))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    batch = jax.device_put(jnp.arange(32).reshape(16, 2), rows)
    sliced = jax.jit(lambda batch, part: jax.lax.with_sharding_constraint(
        microbatch(batch, part, 2), rows), in_shardings=(rows, replicated))
    moved = collectives(sliced.lower(batch, jax.device_put(jnp.int32(1), replicated)).compile().as_text())
    assert not moved, moved


def one_device(trainer):
    """`trainer` on one device, whose row shards every split divides."""
    trainer.device_mesh = MeshSpec().build(jax.devices()[:1])
    return trainer


def test_a_step_whose_rows_an_explicit_axis_types_is_read_whole():
    """Under an Explicit data axis the plan reads each batch whole, and a
    pinned split is refused."""
    model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=SEQ)
    spec = MeshSpec(explicit=("data",))
    with pytest.raises(ValueError):
        Trainer(LMObjective(model, seq_len=SEQ), optax.sgd(1e-2), key=jax.random.key(0),
                fit=FitPolicy(split=2), mesh=spec)
    trainer = Trainer(LMObjective(model, seq_len=SEQ), optax.sgd(1e-2), key=jax.random.key(0), mesh=spec)
    assert splits(None, BATCH, MeshSpec().build(jax.devices()[:1]), 1, trainer._shape.split) == [1]


EXECUTED: list[int] = []
"""One entry for each time a compiled program ran a `Counted` objective's loss."""


class Counted(LMObjective):
    """An `LMObjective` whose loss reports each run of a program over it (`jax.debug.callback`): a
    step's microbatch, or a trial's, counts once, which a compile alone does not."""

    def loss(self, variables, batch, step):
        jax.debug.callback(lambda: EXECUTED.append(1))
        return super().loss(variables, batch, step)


def lm(split="auto", accumulation=1, objective=LMObjective, *, seq=SEQ, width=16, layers=1, vocab=64,
       policy="auto", **fields):
    """A tiny LM's trainer on one device: its step's batch read in `split` microbatches, `accumulation`
    steps an update, recomputing as its fit `policy` asks (`FitPolicy.remat`), its model's `fields` given."""
    model = CausalTransformer(vocab_size=vocab, emb_features=width, num_layers=layers, num_heads=2,
                              mlp_features=2 * width, max_seq_len=seq, **fields)
    return one_device(Trainer(objective(model, seq_len=seq), optax.sgd(1e-2), key=jax.random.key(0),
                              update=UpdateBatch(batches=accumulation),
                              fit=FitPolicy(split=split, remat=policy)))


def test_a_step_whose_optimizer_state_is_a_class_made_in_its_factory_plans_and_steps():
    """A step whose optimizer keeps its state in a class its factory makes, as an optax-style factory may: a
    node no process names, which no file of programs states, is planned and compiled within this process by
    the program as JAX holds it (`program_key`), so the run plans and steps, where the plan's key of its
    executables stated its trees for a file and raised (train 961's F1)."""
    from optim_support import counting

    model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=SEQ)
    trainer = one_device(Trainer(LMObjective(model, seq_len=SEQ), counting(1e-2), key=jax.random.key(0)))
    state = trained(trainer, steps=2)
    assert int(state.opt_state.steps) == 2


def test_a_second_trainer_of_one_step_plans_on_the_executable_the_first_compiled():
    """A second trainer of a process, its step the same program on the same devices as an earlier one's,
    plans on the executable the earlier compiled (`dew.nn.kernels.options.hold`): its plan compiles nothing
    and reads nothing back from the persistent compilation cache, which the compile telemetry would name
    (`Compiles`, a backend compile or a cache read alike), where every new trainer read its step back (4.4 s
    of a small model's 11.9 s start on CPU, row 6995); it steps as the first."""
    from dew.telemetry.compiles import Compiles

    first = trained(lm(split=1), steps=1)
    second = lm(split=1)
    with Compiles(None) as compiles:
        state = trained(second, steps=1)
    assert [program.name for program in compiles.programs if program.kind == "compiled"
            and program.name == "jit(step)"] == []
    for one, other in zip(jax.tree.leaves(first.variables), jax.tree.leaves(state.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(one), np.asarray(other))


def scaled_on_the_host(scale: float) -> optax.GradientTransformation:
    """Plain descent whose rate a host callback answers (`jax.pure_callback`): `scale` is the callback's
    closure, so two of other scales trace one program text calling other callbacks."""
    def update(updates, state, params=None):
        del params
        rate = jax.pure_callback(lambda: np.float32(1e-2 * scale), jax.ShapeDtypeStruct((), jnp.float32))
        return jax.tree.map(lambda update: -rate * update, updates), state

    return optax.GradientTransformation(lambda params: optax.EmptyState(), update)


def test_a_second_trainer_whose_step_calls_another_host_callback_runs_its_own():
    """Two trainers of one step whose optimizers differ only in what their host callback answers (a rate
    `jax.pure_callback` reads from its closure, of one program text) each step by their own callback: the
    process holds an executable by the callbacks its program calls (`dew.nn.kernels.options.program_key`),
    where the second ran the first's executable, its rate, a wrong result with nothing to show it (the
    base review of c5eb090ed's F1)."""
    model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=SEQ)

    def run(scale: float):
        return one_device(Trainer(LMObjective(model, seq_len=SEQ), scaled_on_the_host(scale),
                                  key=jax.random.key(0), fit=FitPolicy(split=1)))

    once, twice = (trained(run(scale), steps=1) for scale in (1.0, 2.0))
    # From one start over one batch: the second's step at its own rate lands elsewhere.
    assert not all(np.array_equal(np.asarray(one), np.asarray(two)) for one, two in zip(
        jax.tree.leaves(once.variables["params"]), jax.tree.leaves(twice.variables["params"]), strict=True))


def trained(trainer, steps=3, batch=None):
    """The state after `steps` steps of `batch`, BATCH where None."""
    batch = BATCH if batch is None else batch
    state, _, _ = trainer.place()
    step = trainer.compile(state, batch)
    for _ in range(steps):
        state, *_ = step(state, batch)
    return state


LIMIT = 2 ** 40
"""The bytes every device reports it holds: one size, which a step's rung record names, whatever is free."""


def free(monkeypatch, room):
    """Every device reports `room` bytes of its LIMIT free (`Device.memory_stats`), the one reading of
    the devices the fit check takes; what a step needs is XLA's memory analysis of the step."""
    monkeypatch.setattr(jax.Device, "memory_stats",
                        lambda device: {"bytes_limit": LIMIT, "bytes_in_use": LIMIT - room})


def free_until_placed(monkeypatch, room):
    """Every device reports `room` bytes of its LIMIT free, less the arrays this process places on it after:
    the reading a device gives a fit that plans its step before placing its state, beside that state's
    bytes (`Trainer._planned_unplaced`), and judges it again once the state holds them, where `free`'s
    reading holds still."""
    def held(device) -> int:
        return sum(array.nbytes for array in jax.live_arrays() if device in array.devices())

    gc.collect()
    before = {device: held(device) for device in jax.devices()}
    monkeypatch.setattr(jax.Device, "memory_stats", lambda device: {
        "bytes_limit": LIMIT, "bytes_in_use": LIMIT - room + max(0, held(device) - before[device])})



def test_a_search_handed_to_a_compile_that_refuses_is_released_with_no_winner(monkeypatch):
    """A search made off the loop for the step a compile plans next (a later phase's,
    `dew.training.ahead.LaterPlans`) is handed to the compile (`StepCompiler.settle_with`): a compile whose
    plan is refused releases it with no winner (its every program dropped), never leaving it held with the
    run's executables and their admission (review of row 12131's P0: a boundary's refusal leaked it)."""
    from dew.training.planning import PlanRefused

    trainer = lm()
    state, _, _ = trainer.place()
    released: list[object] = []
    trainer._compiler.settle_with(released.append)
    free(monkeypatch, 1 << 10)
    with pytest.raises(PlanRefused):
        trainer.compile(state, BATCH)
    assert released == [None], released


def test_a_fresh_fit_whose_step_fits_no_plan_before_its_state_is_placed_is_refused_unplaced(monkeypatch):
    """A fresh fit on devices with room for its state and for no plan of its step is refused as its search
    before the state was placed found it, every candidate short by more than placing the state can free: its
    initializer never runs on the devices, and its step is lowered once a probe at most
    (`StepCompiler.refuse_unplaced`; row 11712: a 4B step on two H100s, short by 70 GiB, searched its four
    rungs again once its state was placed, 1.5 h). So too a start that copies held parameters on the devices,
    as a loaded checkpoint's are, freeing their bytes, far less than the step is short of."""
    from affine_run import Replaying
    from test_layout_validation import INITIALIZED, Watched

    from dew.data.dataset import Loader

    rows = Replaying(lambda position: {"text": np.asarray(BATCH["text"])})
    roomy = lm()
    free(monkeypatch, LIMIT)
    trained = roomy.fit(Loader(lambda partition: rows, None, None, ROWS), steps=1)
    assert roomy.validation is not None
    room = max(roomy.validation.device_bytes.values()) + 1024
    options.unhold()
    # Held arrays the start copies, the trained parameters on the device.
    held = {"params": trained.variables["params"]}
    lowered = []

    def lowering(event, seconds, *, fun_name=None, **_):
        if event == "/jax/core/compile/jaxpr_to_mlir_module_duration" and fun_name == "jit(step)":
            lowered.append(seconds)

    def refused(objective) -> tuple[PlanRefused, list[bool], int]:
        """The refusal of a fresh fit of `objective`, whether its initializer ran, its step's lowerings."""
        free_until_placed(monkeypatch, room)
        INITIALIZED.clear()
        lowered.clear()
        jax.monitoring.register_event_duration_secs_listener(lowering)
        try:
            with pytest.raises(PlanRefused) as raised:
                # Its split stated, the one candidate its search asks of every rung, as a run that states
                # its microbatches (the 4B start's) asks it.
                lm(split=1, objective=objective).fit(Loader(lambda partition: rows, None, None, ROWS),
                                                     steps=1)
        finally:
            jax.monitoring.unregister_event_duration_listener(lowering)
        jax.effects_barrier()
        return raised.value, list(INITIALIZED), len(lowered)

    # The held parameters lie on the device the run trains on, which the start copies and may let go.
    assert {device for leaf in jax.tree.leaves(held) for device in leaf.devices()} == {jax.devices()[0]}
    for objective in (Watched, functools.partial(Watched, variables=held)):
        refusal, ran, count = refused(objective)
        assert ran == [] and count <= len(refusal.probes), (ran, count, refusal.probes)
        # Each candidate it asked was measured short of the room.
        assert all(isinstance(probe.headroom, int) and probe.headroom < 0 for probe in refusal.probes)


def test_a_fresh_fit_whose_placed_devices_hold_its_step_trains_after_its_unplaced_search_refused(monkeypatch,
                                                                                                   caplog):
    """A fresh fit whose devices read, before its state is placed, room for its state and for no plan of its
    step, and once it is placed room enough: its search before placing refuses every candidate, the start
    places the state, and the search on the placed devices plans the step, which trains (row 11712's
    review: a refusal before placing is final only where a placed search could reach nothing more)."""
    from affine_run import Replaying
    from test_layout_validation import INITIALIZED, Watched

    from dew.data.dataset import Loader

    rows = Replaying(lambda position: {"text": np.asarray(BATCH["text"])})
    roomy = lm()
    free(monkeypatch, LIMIT)
    roomy.fit(Loader(lambda partition: rows, None, None, ROWS), steps=1)
    assert roomy.validation is not None
    state = max(roomy.validation.device_bytes.values())
    options.unhold()

    def held(device) -> int:
        return sum(array.nbytes for array in jax.live_arrays() if device in array.devices())

    gc.collect()
    before = {device: held(device) for device in jax.devices()}

    def memory_stats(device):
        # Room for the state alone until most of it lies on the device, then all of the limit.
        placed = held(device) - before[device] > state // 2
        return {"bytes_limit": LIMIT, "bytes_in_use": held(device) if placed else LIMIT - state - 1024}

    monkeypatch.setattr(jax.Device, "memory_stats", memory_stats)
    INITIALIZED.clear()
    trainer = lm(objective=Watched)
    with caplog.at_level(logging.INFO, logger="dew.training.compiler"):
        trained = trainer.fit(Loader(lambda partition: rows, None, None, ROWS), steps=1)
    jax.effects_barrier()
    # Refused before the state was placed, then planned on the placed devices: its plan's probes theirs, one
    # fitting (`StepCompiler.compile`'s own search).
    assert any(record.msg == UNPLACED_REFUSED for record in caplog.records)
    plan = trainer.plan
    assert isinstance(plan, dict) and any(probe["fits"] for probe in plan["probes"]), plan
    assert int(trained.step) == 1 and INITIALIZED

def test_a_fit_of_no_steps_on_devices_holding_no_plan_returns_and_its_trainer_trains_given_room(monkeypatch):
    """A fit of no steps on devices with room for no plan of its step returns its state, refused nothing
    (a run with no work reads no step), and its trainer fit again on devices with room for its step trains
    (row 11712's review: a refusal one fit kept was raised by the next)."""
    from affine_run import Replaying

    from dew.data.dataset import Loader

    rows = Replaying(lambda position: {"text": np.asarray(BATCH["text"])})
    roomy = lm()
    free(monkeypatch, LIMIT)
    roomy.fit(Loader(lambda partition: rows, None, None, ROWS), steps=1)
    assert roomy.validation is not None
    room = max(roomy.validation.device_bytes.values()) + 1024
    options.unhold()
    trainer = lm()
    free_until_placed(monkeypatch, room)
    assert int(trainer.fit(Loader(lambda partition: rows, None, None, ROWS), steps=0).step) == 0
    free(monkeypatch, LIMIT)
    assert int(trainer.fit(Loader(lambda partition: rows, None, None, ROWS), steps=1).step) == 1

def needs(monkeypatch, trainer, run=None):
    """The bytes the step `trainer` compiles first needs beside what a device holds: the headroom that
    compile left of all of LIMIT free, taken from LIMIT. `run` compiles it; None compiles it for BATCH."""
    free(monkeypatch, LIMIT)
    if run is None:
        trained(trainer, steps=0)
    else:
        run()
    # Measured as in a process apart: the run a test then counts compiles nothing of it.
    options.unhold()
    return LIMIT - trainer.plan["probes"][0]["headroom"]


def test_a_settled_plan_keeps_no_program_its_search_passed_over(monkeypatch):
    """A search that compiled the step whole, which missed, and split, which fits, keeps the plan's program
    alone once it settles: its executable held for the process (`options.hold`) and its traced step for a
    refit (`_candidates.candidate_for`), each other program it compiled dropped, whose lowering keys its
    loaded executable and, over several GPUs, its collectives' room outside XLA's pool for as long as it is
    kept (row 12131: 24 kept beside a 4B run's step left an all-reduce no room)."""
    from dew.training import _candidates

    whole, halves = needs(monkeypatch, lm(split=1)), needs(monkeypatch, lm(split=2))
    assert halves < whole, (halves, whole)
    free(monkeypatch, (whole + halves) // 2)
    options.release()
    trainer = lm()
    trained(trainer, steps=0)
    assert len(trainer.plan["probes"]) >= 2 and trainer.plan["split"] > 1, trainer.plan["probes"]
    kept = [kept.lowered for kept in _candidates._CANDIDATES.values()]
    assert len(kept) == 1, len(kept)
    assert len(options._HELD) == 1, list(options._HELD)


def test_a_provisional_plans_search_keeps_no_program_once_its_plan_is_taken(monkeypatch):
    """A fresh fit plans its step before placing its state (`Trainer._planned_unplaced`), its search keeping
    every program it compiled for that plan to take; once its compile takes the plan, its own or one
    searched on the placed devices, the programs it passed over are dropped, the plan's alone kept (review
    of row 12131's P0: a provisional search's losers and frontier rungs stayed loaded)."""
    from affine_run import Replaying

    from dew.data.dataset import Loader
    from dew.training import _candidates

    whole, halves = needs(monkeypatch, lm(split=1)), needs(monkeypatch, lm(split=2))
    assert halves < whole, (halves, whole)
    options.release()
    rows = Replaying(lambda position: {"text": np.asarray(BATCH["text"])})
    free(monkeypatch, LIMIT)
    roomy = lm()
    roomy.fit(Loader(lambda partition: rows, None, None, ROWS), steps=1)
    assert roomy.validation is not None
    state = max(roomy.validation.device_bytes.values())
    options.release()
    gc.collect()
    trainer = lm()
    free_until_placed(monkeypatch, state + (whole + halves) // 2)
    trained = trainer.fit(Loader(lambda partition: rows, None, None, ROWS), steps=1)
    plan = trainer.plan
    assert int(trained.step) == 1 and isinstance(plan, dict) and len(plan["probes"]) >= 2, plan
    gc.collect()
    assert len(_candidates._CANDIDATES) == 1, list(_candidates._CANDIDATES)
    assert len(options._HELD) == 1, list(options._HELD)


def test_a_batch_that_does_not_fit_whole_is_split_before_anything_is_recomputed(monkeypatch):
    """With room for the step at two microbatches and not at one, the plan
    reads the batch in microbatches under no recomputation, and the run
    trains as the whole batch does: the update pools the same rows, `step`
    counts steps and `microstep` microbatches."""
    whole, halves = needs(monkeypatch, lm(split=1)), needs(monkeypatch, lm(split=2))
    assert halves < whole, (halves, whole)
    unsplit = trained(lm(split=1))
    free(monkeypatch, (whole + halves) // 2)
    trainer = lm()
    state = trained(trainer)
    split = trainer.plan["split"]
    assert split > 1 and trainer.objective.model.remat is None
    # Whatever it compiled whole did not fit, and the split it trains at did.
    probes = trainer.plan["probes"]
    assert not any(probe["fits"] for probe in probes if probe["split"] == 1), probes
    assert any(probe["fits"] for probe in probes if probe["split"] == split), probes
    assert (int(state.step), int(state.microstep), int(state.updates)) == (3, 3 * split, 3)
    for planned_leaf, whole_leaf in zip(jax.tree.leaves(state.variables), jax.tree.leaves(unsplit.variables),
                                        strict=True):
        np.testing.assert_allclose(np.asarray(planned_leaf), np.asarray(whole_leaf), rtol=1e-4, atol=1e-5)


def test_each_shape_a_run_meets_starts_at_the_run_s_own_head_fusion():
    """A run whose own rung tiles its head unfused, as the objective came or a checkpoint recorded,
    compiles each later shape it meets unfused too: a new shape starts at the run's own rung, its
    head put back as the first compile found it."""
    trainer = lm()
    assert trainer.objective.tile_head() is not None and trainer.objective.unfuse_head() is not None
    state, _, _ = trainer.place()
    trainer.compile(state, BATCH)
    trainer.compile(state, {"text": BATCH["text"][: ROWS // 2]})
    rung = trainer._compiler.checkpoint_rung()
    assert [shaped["head_fused"] for shaped in rung["shapes"].values()] == [False, False], rung
    assert rung["head_tile"] is not None and not rung["head_fused"]


def test_a_later_run_starts_its_search_at_the_plan_an_earlier_one_recorded(monkeypatch):
    """The plan is recorded beside the compilation cache, read before the step
    traces, so a restart compiles the recorded candidate first, from the
    cache, and keeps it, with the FLOPs the first run counted for the same
    executable. (What the restart's start costs, its executable's text unread among it, the startup
    benchmark measures.)"""
    whole, halves = needs(monkeypatch, lm(split=1)), needs(monkeypatch, lm(split=2))
    free(monkeypatch, (whole + halves) // 2)
    first = lm()
    trained(first, steps=0)
    free(monkeypatch, LIMIT)
    again = lm()
    trained(again, steps=0)
    assert [probe["split"] for probe in again.plan["probes"]] == [first.plan["split"]]
    assert again.step_flops == first.step_flops


def a_recorded_plan_settles_its_kernel_launches_without_lowering_the_step_for_them():
    """A step's lowering records its kernel launches as it traces them (`tune`), with its plan,
    so each run lowers its step once: the first again only where settling the launches changed
    a config its trace read, and a later run, which settles them from the record first, never."""
    lowered = []

    def lowering(event, seconds, *, fun_name=None, **_):
        if event == "/jax/core/compile/jaxpr_to_mlir_module_duration" and fun_name == "jit(step)":
            lowered.append(seconds)

    counts = []
    settled = _cache.KERNEL_TUNES.get_global()
    jax.monitoring.register_event_duration_secs_listener(lowering)
    try:
        for _ in range(2):
            trained(lm(), steps=0)
            counts.append(len(lowered))
            lowered.clear()
            changed, settled = _cache.KERNEL_TUNES.get_global() != settled, _cache.KERNEL_TUNES.get_global()
            assert counts[-1] == 1 + changed
    finally:
        jax.monitoring.unregister_event_duration_listener(lowering)
    assert counts[1] == 1


@contextlib.contextmanager
def autopgle(runs: int) -> Iterator[None]:
    """JAX's AutoPGLE on for the block, profiling a program's first `runs` runs before compiling it
    again guided by them (`jax_enable_pgle`, `jax_pgle_profiling_runs`)."""
    before = jax.config.jax_enable_pgle, jax.config.jax_pgle_profiling_runs
    jax.config.update("jax_enable_pgle", val=True)
    jax.config.update("jax_pgle_profiling_runs", val=runs)
    try:
        yield
    finally:
        jax.config.update("jax_enable_pgle", val=before[0])
        jax.config.update("jax_pgle_profiling_runs", val=before[1])


@dataclasses.dataclass
class Handed(Trials):
    """Trials that profile the plan admitted, keep a weak reference to the executable it was admitted with
    (`unguided`) and the executables the backend holds then, it among them (`live`, by module), and hand
    back `profile` where given in place of the profile they measured."""
    profile: bytes | None = None
    unguided: weakref.ref | None = None
    live: collections.Counter | None = None

    def profiled(self, step, planned, mesh, runs):
        measured = super().profiled(step, planned, mesh, runs)
        self.unguided = weakref.ref(planned.executable)
        self.live = live_executables()
        return measured if self.profile is None else self.profile


def live_executables() -> collections.Counter:
    """The executables the backend holds once the garbage they are part of is collected, by their module's
    name: two counts' difference names what one holds the other does not."""
    gc.collect()
    return collections.Counter(executable.hlo_modules()[0].name
                               for executable in get_backend().live_executables())


def holding(modules: collections.Counter, depth: int = 8) -> dict[str, list[list[str]]]:
    """For each live executable of `modules`' names, the chains of objects that keep it, from the executable
    out to a module's namespace or a class, each named by its type (a namespace by its module's name), the
    first four found: what a test of an executable's release reads where one outlives its step."""
    gc.collect()
    found: dict[str, list[list[str]]] = {}
    for executable in get_backend().live_executables():
        name = executable.hlo_modules()[0].name
        if name not in modules:
            continue
        chains: list[list[str]] = []
        frontier, seen = [(executable, [type(executable).__name__])], {id(executable)}
        for _ in range(depth):
            reached = []
            for held, chain in frontier:
                for referrer in gc.get_referrers(held):
                    # The walk's own lists and frames hold what it walks, and keep nothing past it.
                    if (id(referrer) in seen or isinstance(referrer, types.FrameType)
                            or referrer is frontier or referrer is reached):
                        continue
                    seen.add(id(referrer))
                    namespace = isinstance(referrer, dict) and "__name__" in referrer
                    named = f"({referrer['__name__']})" if namespace else ""
                    step = [*chain, type(referrer).__name__ + named]
                    if namespace or isinstance(referrer, type):
                        chains.append(step)
                    else:
                        reached.append((referrer, step))
            frontier = reached
        found.setdefault(name, []).extend(chains[:4])
    return found


def test_a_plan_search_keeps_the_freeze_a_process_made_before_it():
    """A plan search raises Python's youngest collection threshold while it runs (`held_objects`) and
    leaves the permanent generation to the process: one that froze its objects before importing Dew, as one
    that forks its workers does, has its freeze as it made it after a step planned and compiled, Dew
    freezing and unfreezing nothing, and its thresholds as they were, in a fresh interpreter (row 3341's
    review)."""
    assert_exits_cleanly("""
import gc
import itertools

touched = []
freeze, unfreeze = gc.freeze, gc.unfreeze
gc.freeze = lambda: touched.append("freeze")
gc.unfreeze = lambda: touched.append("unfreeze")
gc.collect()
freeze()
thresholds = gc.get_threshold()

import sys
sys.path.insert(0, "tests")
from test_plan import BATCH, lm

trainer = lm()
state, _, _ = trainer.place()
trainer.compile(state, BATCH)
assert not touched and gc.get_threshold() == thresholds, (touched, gc.get_threshold())
from dew.training.planning import held_objects
with held_objects():
    assert gc.get_threshold()[0] > thresholds[0]
assert not touched and gc.get_threshold() == thresholds
unfreeze()
""")


COST = b"\x0a\x0c\x0a\x01x\x11" + struct.pack("<d", 1.0)
"""A profile of one instruction, `x`, at a microsecond, as XLA's ProfiledInstructionsProto writes it: what
an NVIDIA GPU's profile holds, which a CPU's compile takes and does not read."""


def stepped(trainer, steps: int = 3, trials: Callable[[TrainState], Trials] | None = None):
    """The state after `steps` steps of BATCH from the trainer's placed state, its step compiled on the
    `trials` of that state, from which it steps on, or on none, stepping that state itself; and the
    trials."""
    state, _, _ = trainer.place()
    held = None if trials is None else trials(state)
    # The garbage of the steps before collected, which the plan search would hold frozen while it runs
    # (`dew.training.planning.held_objects`) where an executable counted inside it is part of it.
    gc.collect()
    step = trainer.compile(state, BATCH, held)
    state = state if held is None else held.state
    for _ in range(steps):
        state, *_ = step(state, BATCH)
    return state, held


@pytest.mark.parametrize("split", [1, 2], ids=["whole-batch", "two-microbatches"])
def test_a_step_compiled_ahead_is_profiled_on_a_trial_under_autopgle(tmp_path, split):
    """Under JAX's AutoPGLE the step a plan compiles ahead of its run, which no jit call profiles, runs as
    many times as AutoPGLE profiles, each over the step's microbatches, on a trial: a copy of the state
    beside it, where its compile is given no trials, which the caller then steps as a run without
    AutoPGLE does; and under `fit`, whose admission keeps its first fitting plan
    and profiles it all the same. XLA's profile of a CPU holds nothing, so each plan keeps the executable
    it admitted and steps as the run without AutoPGLE does, bit for bit; a GPU's holds its program's
    timings, some bytes whose count is the profile's own (89 and 278 on an H100, row 4990), and its run
    steps through the program the profile guides, another program."""
    unguided, _ = stepped(lm(split=split))
    with autopgle(2):
        trainer, fitted = lm(split=split), windowed(tmp_path, split)
        guided, _ = stepped(trainer)
        fitted.fit(windows(token_files(tmp_path)), steps=1, checkpoint_every=None)
    profiles = [trainer.plan["pgle"], fitted.plan["pgle"]]
    assert [profile["runs"] for profile in profiles] == [2, 2], profiles
    assert int(guided.step) == 3
    if jax.default_backend() != "cpu":
        assert all(profile["bytes"] for profile in profiles), profiles
        return
    assert [profile["bytes"] for profile in profiles] == [0, 0], profiles
    for guided_leaf, unguided_leaf in zip(jax.tree.leaves(guided.variables),
                                          jax.tree.leaves(unguided.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(guided_leaf), np.asarray(unguided_leaf))


@pytest.mark.parametrize("earlier", [False, True])
def test_the_executable_a_profile_replaces_goes_with_its_program(earlier):
    """Under AutoPGLE a profile of the admitted step's runs compiles its program again, and the run keeps
    the guided executable alone: once the compile returns, the one the plan was admitted with is freed
    and the backend holds as many executables as it did with that one alive, though JAX keeps every
    executable it compiled of a program while the program's lowering lives. A handoff that kept the
    admitted executable, or kept both, would hold one more. Where an `earlier` trainer of the same program
    compiled the unguided step, the trainer takes it and lets it go too: no lowering the earlier trainer kept
    holds it alive beside the guided one (rows 7988, 8293: three steps live where two were, on a GH200)."""
    if earlier:
        trained(lm(), steps=0)
    with autopgle(2):
        trainer = lm()
        state, trials = stepped(trainer, trials=functools.partial(Handed, batch=BATCH, profile=COST))
    assert trainer.plan["pgle"] == {"runs": 2, "bytes": len(COST)}
    assert trials.unguided is not None and trials.unguided() is None
    live = live_executables()
    assert live == trials.live and int(state.step) == 3, (live - trials.live, trials.live - live,
                                                            holding(live - trials.live))


def test_a_profile_window_traces_steps_past_the_first_compile_and_is_refused_under_autopgle(tmp_path,
                                                                                            monkeypatch):
    """A fit's window opens once its warmup steps have run and its first step has compiled, so a window of
    no warmup traces the three steps after the first, never the compile; its reports are exported in a
    process of their own while the run trains on, a step after the window logging while the export waits
    (`Stalled`, row 2017), and the run's end waits for them, whole. Under JAX's AutoPGLE, which
    profiles each program's first runs in a profiling session of its own where a process holds one at a
    time, a fit with a window is refused before it trains, naming JAX's way to capture guided programs
    (`PGLE_CAPTURE`): JAX's sessions would fail inside the window, as the trainer's would."""
    pytest.importorskip("xprof.convert.raw_to_tool_data", reason="optional profile extra")
    from affine_run import Replaying
    from recording import RecordingTracker

    from dew.data.dataset import Loader
    from dew.training import ProfileWindow
    from dew.training.profiling import traced_steps

    stalled = Stalled.under(tmp_path / "stalled", monkeypatch)

    class Releasing(RecordingTracker):
        """The run's logs, the window's export released at the first log of a step after the window, once
        the export waits: a log the run can make only while the export is held where it runs beside it."""

        def __init__(self):
            super().__init__()
            self.stepped: list[int] = []

        def log(self, scalars, step):
            super().log(scalars, step)
            if step > 4 and not self.stepped:
                stalled.await_start()
                self.stepped.append(step)
                stalled.release()

    tracker = Releasing()

    def fit(directory):
        model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                                  max_seq_len=SEQ)
        window = ProfileWindow(str(directory), steps=3, warmup=0)
        trainer = one_device(Trainer(LMObjective(model, seq_len=SEQ), optax.sgd(1e-2), key=jax.random.key(0),
                                     tracker=tracker, profile=window))
        rows = Replaying(lambda position: {"text": np.asarray(BATCH["text"])})
        trainer.fit(Loader(lambda partition: rows, None, None, ROWS), steps=5, log_every=1)

    fit(tmp_path / "window")
    # A step after the window logged while its export waited, which the run's end waited for in turn.
    assert tracker.stepped == [5] and stalled.released
    (window,) = [value for _, value in tracker.artifacts if type(value).__name__ == "ProfileTraced"]
    (capture,) = (tmp_path / "window").glob("capture-*")
    assert window.steps == 3 and traced_steps(capture) == (1, 2, 3)
    assert json.loads((capture / "manifest.json").read_text())["export_status"] == "completed"
    with autopgle(2), pytest.raises(ValueError):
        fit(tmp_path / "pgle")
    assert not (tmp_path / "pgle").exists()


def test_a_run_that_fails_after_its_profile_window_still_reports_the_window(tmp_path):
    """A run whose window stopped and which fails steps later reports the window at its end, its reports
    written (row 2824): the record says where the capture is whatever the run's outcome, as its manifest
    says whether its export completed."""
    pytest.importorskip("xprof.convert.raw_to_tool_data", reason="optional profile extra")
    from affine_run import Replaying
    from recording import RecordingTracker

    from dew.data.dataset import Loader
    from dew.training import Observe, ProfileWindow

    tracker = RecordingTracker()

    class LaterStepFailed(RuntimeError):
        """The run's failure, steps after its window stopped."""

    def failing(state):
        if int(state.step) >= 4:
            raise LaterStepFailed
        return 0

    model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=SEQ)
    trainer = one_device(Trainer(LMObjective(model, seq_len=SEQ), optax.sgd(1e-2), key=jax.random.key(0),
                                 tracker=tracker, profile=ProfileWindow(str(tmp_path), steps=2, warmup=0)))
    rows = Replaying(lambda position: {"text": np.asarray(BATCH["text"])})
    with pytest.raises(LaterStepFailed):
        trainer.fit(Loader(lambda partition: rows, None, None, ROWS), steps=6, log_every=1,
                    observe=Observe(1, failing))
    (window,) = [value for _, value in tracker.artifacts if type(value).__name__ == "ProfileTraced"]
    (capture,) = tmp_path.glob("capture-*")
    assert window.steps == 2
    assert json.loads((capture / "manifest.json").read_text())["export_status"] == "completed"


def test_a_run_that_stops_inside_its_profile_window_stops_its_capture_and_reports_what_it_traced(tmp_path):
    """A run that stops before its window's steps have run (a pause at half its budget) stops the capture
    at its end, where a trace left running would take the next one down with it, and reports the window
    over every step it traced, the step it stopped at among them, its reports written."""
    pytest.importorskip("xprof.convert.raw_to_tool_data", reason="optional profile extra")
    from affine_run import Replaying
    from recording import RecordingTracker

    from dew.data.dataset import Loader
    from dew.training import Pause, ProfileWindow
    from dew.training.profiling import traced_steps

    tracker = RecordingTracker()
    model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=SEQ)
    trainer = one_device(Trainer(LMObjective(model, seq_len=SEQ), optax.sgd(1e-2), key=jax.random.key(0),
                                 tracker=tracker, checkpoints=Checkpoints(str(tmp_path / "run")),
                                 profile=ProfileWindow(str(tmp_path / "trace"), steps=4, warmup=0)))
    rows = Replaying(lambda position: {"text": np.asarray(BATCH["text"])})
    # The window opens past the first step's compile and holds the second and third, where the run pauses.
    trainer.fit(Loader(lambda partition: rows, None, None, ROWS), steps=6, log_every=1, stop=Pause(0.5))
    (window,) = [value for _, value in tracker.artifacts if type(value).__name__ == "ProfileTraced"]
    (capture,) = (tmp_path / "trace").glob("capture-*")
    assert window.steps == 2 and traced_steps(capture) == (1, 2)
    assert json.loads((capture / "manifest.json").read_text())["export_status"] == "completed"


def test_a_waiting_window_reports_in_a_run_that_cannot_create_a_process(tmp_path):
    """A window that waits (`ProfileWindow.wait`) converts its capture's reports in the run's own process as
    it stops: a fit in a program its sandbox forbids to create processes (the live site's guest policy, a
    seccomp filter on fork and clone, `site/live/container/guest_limits.py`) finishes with its window's
    reports written, where an exporter's process of its own could not start. The guest holds a uid of its
    own, as each of the site's kernels does (`site/live/container/gateway_manager.py`), here a user
    namespace's, so the policy's 256 threads are its own, beside another process of the test's uid holding
    300 threads, as a host's neighbouring runs do (row 4869: sharing their uid, its threads failed to start,
    pthread_create's EAGAIN, and the run aborted)."""
    pytest.importorskip("xprof.convert.raw_to_tool_data", reason="optional profile extra")
    import os
    import subprocess
    import sys
    from pathlib import Path

    import dew

    root = Path(dew.__file__).resolve().parents[2]
    script = f"""
import importlib.util, os, sys
sys.path.insert(0, {str(root / "tests")!r})
# A uid of its own, mapped to the test's, so the threads the policy counts are this guest's alone.
uid = os.getuid()
os.unshare(os.CLONE_NEWUSER)
with open("/proc/self/uid_map", "w") as handle:
    handle.write(f"{{uid}} {{uid}} 1\\n")
import jax, jax.numpy as jnp, numpy as np, optax
from affine_run import Replaying
from dew.data.dataset import Loader
from dew.distributed.placement import MeshSpec
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.training import ProfileWindow, Trainer
spec = importlib.util.spec_from_file_location("guest_limits",
                                              {str(root / "site/live/container/guest_limits.py")!r})
limits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(limits)
limits.install("train")
model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                          max_seq_len={SEQ})
trainer = Trainer(LMObjective(model, seq_len={SEQ}), optax.sgd(1e-2), key=jax.random.key(0),
                  profile=ProfileWindow({str(tmp_path / "trace")!r}, steps=2, warmup=0, wait=True))
trainer.device_mesh = MeshSpec().build(jax.devices()[:1])
text = np.asarray((np.arange({ROWS} * {SEQ + 1}).reshape({ROWS}, {SEQ + 1}) * 7) % 64, np.int32)
rows = Replaying(lambda position: dict(text=text))
trainer.fit(Loader(lambda partition: rows, None, None, {ROWS}), steps=3)
"""
    source = str(root / "src")
    path = os.pathsep.join([source, *filter(None, [os.environ.get("PYTHONPATH")])])
    # A neighbour of the test's uid holding 300 threads, past the policy's 256 for a uid.
    neighbour = subprocess.Popen([sys.executable, "-c", "import threading, time, sys\n"
                                  "for _ in range(300): threading.Thread(target=time.sleep, args=(600,), "
                                  "daemon=True).start()\nprint(flush=True)\ntime.sleep(600)"],
                                 stdout=subprocess.PIPE, text=True)
    try:
        neighbour.stdout.readline()
        subprocess.run([sys.executable, "-"], input=script, text=True, check=True,
                       env={**os.environ, "PYTHONPATH": path})
    finally:
        neighbour.kill()
        neighbour.wait()
    (capture,) = (tmp_path / "trace").glob("capture-*")
    record = json.loads((capture / "manifest.json").read_text())
    assert record["capture_status"] == record["export_status"] == "completed"
    assert any(report["status"] == "completed" for report in record["reports"])


def test_a_profile_window_spanning_a_checkpoints_evaluation_holds_the_training_steps_alone(tmp_path):
    """A window that spans the evaluation a checkpoint's ranking runs (its cadence's, every second step) and
    the save holds the training steps alone: the work between them runs outside its capture, which
    starts again at the next step's read, so the window's captures hold its four steps between them, each
    once, and no evaluation, which a capture marks wherever one runs inside it (`region`); and the run
    finishes, every checkpoint written. Inside the window an evaluation filled a step profile's capture."""
    pytest.importorskip("xprof.convert.raw_to_tool_data", reason="optional profile extra")
    import gzip

    from affine_run import Replaying
    from recording import RecordingTracker

    from dew.data.dataset import Loader
    from dew.training import ProfileWindow
    from dew.training.profiling import traced_steps

    def regions(capture):
        named = set()
        for trace in capture.glob("plugins/profile/*/*.trace.json.gz"):
            with gzip.open(trace, "rt") as stream:
                named.update(event.get("name") for event in json.load(stream)["traceEvents"])
        return named

    tracker = RecordingTracker()
    model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=SEQ)
    checkpoints = Checkpoints(str(tmp_path / "run"), keep=3)
    trainer = one_device(Trainer(LMObjective(model, seq_len=SEQ), optax.sgd(1e-2), key=jax.random.key(0),
                                 tracker=tracker, checkpoints=checkpoints,
                                 profile=ProfileWindow(str(tmp_path / "trace"), steps=4, warmup=0)))
    rows = Replaying(lambda position: {"text": np.asarray(BATCH["text"])})
    held = {"held": lambda partition: iter([{"text": np.asarray(BATCH["text"])}])}
    trainer.fit(Loader(lambda partition: rows, None, None, ROWS), steps=6, log_every=1, checkpoint_every=2,
                validation=held)
    assert checkpoints.committed() == [2, 4, 6]
    (window,) = [value for _, value in tracker.artifacts if type(value).__name__ == "ProfileTraced"]
    captures = sorted((tmp_path / "trace").glob("capture-*"))
    assert window.steps == 4 and len(captures) > 1
    assert sorted(step for capture in captures for step in traced_steps(capture)) == [1, 2, 3, 4]
    assert all(traced_steps(capture) and "evaluate" not in regions(capture) for capture in captures)


def test_a_borrowed_compile_under_autopgle_profiles_beside_the_state_only_where_the_devices_hold_both(
        monkeypatch):
    """Under AutoPGLE a compile given no trials profiles its step on a copy of the state beside the
    caller's, which it never steps. With room for the step and its state alone, it is refused before it
    copies anything, where trials that own the state (`fit`'s), which step it and restore it, profile at
    that room; with room for the copy too, it profiles and leaves the devices holding the caller's state
    alone, the copy freed. Both profile the step alike: no profile on a backend XLA profiles none of
    (CPU), one on a GPU (89 bytes on an H100, its timings' encoding varying)."""
    need = needs(monkeypatch, lm())
    state, _, _ = lm().place()
    copy = placed_bytes(apart(state)[0])

    def copies(state) -> int:
        """The arrays the backend holds shaped as the state's widest moving leaf, its own among them."""
        widest = max(jax.tree.leaves(apart(state)[0]), key=lambda leaf: leaf.size)
        return sum(live.shape == widest.shape and live.dtype == widest.dtype and not live.is_deleted()
                   for live in jax.live_arrays())

    free(monkeypatch, need + copy // 2)
    with autopgle(2):
        refused = lm()
        state, _, _ = refused.place()
        held = copies(state)
        with pytest.raises(RuntimeError):
            refused.compile(state, BATCH)
        assert copies(state) == held
        owned = lm()
        stepped(owned, trials=functools.partial(Trials, batch=BATCH))
        assert owned.plan["pgle"]["runs"] == 2
        free(monkeypatch, need + 2 * copy)
        borrowed = lm()
        state, _, _ = borrowed.place()
        held = copies(state)
        borrowed.compile(state, BATCH)
        profiled = borrowed.plan["pgle"]
        assert profiled["runs"] == 2 and bool(profiled["bytes"]) == bool(owned.plan["pgle"]["bytes"])
        assert copies(state) == held and not any(leaf.is_deleted() for leaf in jax.tree.leaves(state))


def regression_steps(x, *, pgle: bool):
    """The losses of four steps of SGD on `Regression` over `x`'s rows, eight a step, and its weights
    after them, the step compiled ahead and, with `pgle`, guided by a profile of its first runs."""
    trainer = one_device(Trainer(Regression(), optax.sgd(0.1), key=jax.random.key(0)))
    rows = [{"x": x[start:start + 8], "y": 2 * x[start:start + 8, :2]} for start in range(0, len(x), 8)]
    with autopgle(2) if pgle else contextlib.nullcontext():
        state, _, _ = trainer.place()
        trials = Trials(state, rows[0])
        step = trainer.compile(state, rows[0], trials)
    state, losses = trials.state, []
    for batch in rows:
        state, loss, *_ = step(state, batch)
        losses.append(loss)
    return {"loss": jnp.stack(losses), "params": state.variables["params"]}


def regression_fit(x, directory, split: int, steps: int):
    """`regression_steps`' run as a fit over `x`'s rows, eight a step in order, read in `split` microbatches
    and checkpointed every two steps in `directory`, resumed there where it holds a checkpoint: its weights
    after `steps` steps."""
    from dew.data import Memory
    from dew.data.pipeline import Pipeline

    rows = [{"x": row, "y": 2 * row[:2]} for row in x]
    data = compile(Pipeline(Memory(rows)).batch(8), read="memory", seed=0,
                   loading=Loading(workers=0, threads=1, read_buffer=1))
    # The optimizer stated as a config, which keys the step, so each seed's fits read the step the first
    # traced (`Trainer._step_key`).
    trainer = one_device(Trainer(Regression(), OptimConfig(optimizer=Optax('optax:sgd'), learning_rate=0.1),
                                 key=jax.random.key(0), fit=FitPolicy(split=split),
                                 checkpoints=Checkpoints(str(directory))))
    return trainer.fit(data, steps=steps, checkpoint_every=2).variables["params"]


def test_a_step_handed_a_state_another_split_left_at_an_updates_end_reads_it_in_its_own():
    """A compiled step is handed a state another step of another split returned at an update's end (a
    phase's, a cooldown's restored, a caller's): it lays the state out for its own window first, its count
    of accepted microbatches read in its own, so each update is one step; handed one inside an update, it
    refuses it (row 5558)."""
    halves = trained(lm(split=2), steps=2)
    quarters = lm(split=4)
    step = quarters.compile(halves, BATCH)
    state, *_ = step(halves, BATCH)
    assert (int(state.step), int(state.microstep), int(state.window_size)) == (3, 3 * 4, 4)
    pooled = trained(lm(split=2, accumulation=2), steps=1)
    with pytest.raises(ValueError):
        lm(split=4, accumulation=2).compile(pooled, BATCH)(pooled, BATCH)


@pytest.mark.check1_shards(seeds=128, per_task=16)
def test_a_run_resumed_at_a_finer_split_lands_where_the_uninterrupted_run_does(tmp_path, request):
    """A run saved at an update's end in two microbatches a step and resumed in four, a memory setting
    alone, lands where the run nobody stopped lands: its weights as exact as that run's (check 1), from
    the float64 run of its steps (`regression_truth`). Its 128 seeds were sized paired (`seeds_for`, the
    resumed run as `dew_fn`) on armada at main 2aa25af4e, where sizing them live cost 418 s of 522 fits
    (row 5673); its mark declares them, 16 a map task: a seed's three fits take about 3 s, so 32 a task
    ran 86 to 126 s on armada (row 6068), past the group target."""
    runs = itertools.count()

    @as_given
    def resumed(x):
        directory = tmp_path / f"resumed-{next(runs)}"
        regression_fit(x, directory, 2, 2)
        return {"params": regression_fit(x, directory, 4, 4)}

    @as_given
    def uninterrupted(x):
        return {"params": regression_fit(x, tmp_path / f"whole-{next(runs)}", 2, 4)}

    truthful = regression_truth()

    def truth(x):
        return {"params": truthful(x)["params"]}

    label = "a run resumed at a finer split"
    assert_as_exact(resumed, uninterrupted, regression_rows, label, truth_fn=truth,
                    seeds=marked_seeds(request))


def regression_truth() -> Callable:
    """`regression_steps`' run in float64, a function of its rows: from the weights its trainer draws, held
    in float64 here, each step's affine map, squared error and SGD step at 0.1 over its eight rows computed
    in float64 (check 1's truth, which the gate traces and compiles)."""
    trainer = one_device(Trainer(Regression(), optax.sgd(0.1), key=jax.random.key(0)))
    state, _, _ = trainer.place()
    start = float64_held(state.variables["params"])

    def loss(params, rows):
        return jnp.mean((Affine().apply({"params": params}, rows) - 2 * rows[:, :2]) ** 2)

    def truth(x):
        params, losses = start, []
        for begin in range(0, x.shape[0], 8):
            value, gradient = jax.value_and_grad(loss)(params, x[begin:begin + 8])
            params = jax.tree.map(lambda param, step: param - 0.1 * step, params, gradient)
            losses.append(value)
        return {"loss": jnp.stack(losses), "params": params}
    return truth


def test_a_compiled_steps_memory_is_xlas_analysis_whoever_reads_it():
    """A compiled step's memory (`compiled_memory`), which the planner's probe and forecast, the step's
    headroom and the run's memory record each read, is XLA's analysis of it, the same whoever reads it and
    whenever (row 4422: a count of the analysis's calls held its cost, which the startup benchmark
    measures)."""
    from dew.telemetry.instrumentation import compiled_memory

    rows = regression_rows(0)[0]
    executable = jax.jit(lambda x: jnp.tanh(x @ x.T).sum(axis=0)).lower(rows).compile()
    analysis = executable.memory_analysis()
    read = compiled_memory(executable)
    assert read is not None
    assert (read.argument, read.output, read.alias, read.temp) == (
        analysis.argument_size_in_bytes, analysis.output_size_in_bytes, analysis.alias_size_in_bytes,
        analysis.temp_size_in_bytes)
    assert compiled_memory(executable) == read


def regression_rows(seed: int) -> tuple[np.ndarray]:
    """Four steps' rows of one draw: 32 rows of FEATURES coordinates from a unit normal."""
    return (np.random.default_rng(seed).normal(size=(32, FEATURES)).astype(np.float32),)


GUIDED_SEEDS = 8
"""Check 1's declared draws of a step guided by its profile against the unguided step, sized on a GH200
(`seeds_for`, measurement M-MC-29): a GPU tier's prepare compiles a declared site's truths, and refuses a
sizing at its run (rows 7988, 8293)."""


def a_step_guided_by_its_profile_is_the_step_the_run_takes_and_trains_as_exactly():
    """On an NVIDIA GPU the profile of the admitted step's runs holds its instructions' times, its
    program compiles again guided by it, and the run steps the guided executable alone: the one the plan
    was admitted with is freed and the backend holds no more executables than with it alive, as it would
    not where the handoff is discarded. The guided steps' losses and weights are as exact as the unguided
    step's (check 1)."""
    with autopgle(2):
        trainer = lm()
        _, trials = stepped(trainer, trials=functools.partial(Handed, batch=BATCH))
    assert trainer.plan["pgle"]["runs"] == 2 and trainer.plan["pgle"]["bytes"] > 0
    assert trials.unguided is not None and trials.unguided() is None
    live = live_executables()
    assert live == trials.live, (live - trials.live, trials.live - live, holding(live - trials.live))
    guided = as_given(functools.partial(regression_steps, pgle=True))
    unguided = as_given(functools.partial(regression_steps, pgle=False))
    label, truth = "a step guided by its profile", regression_truth()
    assert_as_exact(guided, unguided, regression_rows, label, truth_fn=truth, seeds=GUIDED_SEEDS)


def test_a_pinned_split_is_read_as_given_and_its_state_holds_it():
    """`split` fixes the microbatches; the window pools `split` x
    `accumulation` of them, one update every `accumulation` steps, and the
    state's window holds the split, so a trainer that continues the state
    reads its steps alike."""
    trainer = lm(split=4, accumulation=2)
    state = trained(trainer, steps=4)
    assert [probe["split"] for probe in trainer.plan["probes"]] == [4]
    clocks = (int(state.step), int(state.microstep), int(state.updates), int(state.window_size))
    assert clocks == (4, 16, 2, 8)
    again = lm(accumulation=2)
    again.compile(state, BATCH)
    assert [probe["split"] for probe in again.plan["probes"]] == [4]


@pytest.mark.mesh(devices=2)
def test_a_split_must_cut_every_batch_the_run_reads_into_whole_rows():
    """A token budget's batches have other rows at each length; the plan
    splits only by counts that cut them all (24, 16, 12, 8, 6 and 4 rows
    take 1 or 2), and a pinned split that leaves some batch without whole
    rows is refused."""
    rows = (24, 16, 12, 8, 6, 4)
    one, two = MeshSpec().build(jax.devices()[:1]), MeshSpec().build(jax.devices()[:2])
    assert splits(rows, BATCH, one, 1, None) == [1, 2]
    assert splits(rows, BATCH, two, 1, None) == [1]
    splits(rows, BATCH, one, 1, 2)
    with pytest.raises(LayoutRefused):
        splits(rows, BATCH, one, 1, 4)


def raw(leaf):
    """A leaf's bits, a key's included."""
    return np.asarray(jax.random.key_data(leaf) if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key) else leaf)


def token_files(directory):
    """A token corpus written once, so a resume reads the files its checkpoint names: the same
    tokens to train and to validate on."""
    tokens = (np.arange(97, dtype=np.uint16) * 7) % 32
    return write_token_corpus(directory / "tokens", tokens, tokens, vocab_size=32)


def windows(corpus):
    """Windows of four tokens a token apart in a run's order, in batches of eight, validating on the
    first 32 windows a window apart."""
    train = TokenFiles(corpus.train).shuffle(7).repeat().pipe(presets.token_windows, seq_len=4, stride=1,
                                                               batch=8)
    held = TokenFiles(corpus.val).pipe(presets.token_windows, seq_len=4, batch=8, records=32)
    return compile(train, validation=held, read="memory", seed=7,
                   loading=Loading(workers=0, threads=1, read_buffer=1))


def windowed(directory, split, remat=None, objective=LMObjective, accumulation=1):
    model = CausalTransformer(vocab_size=32, emb_features=8, num_layers=2, num_heads=1, mlp_features=16,
                              max_seq_len=8, remat=remat)
    return one_device(Trainer(objective(model, seq_len=4), optax.adam(1e-2), key=jax.random.key(0),
                              update=UpdateBatch(batches=accumulation), fit=FitPolicy(split=split),
                              layout=Layout(min_shard=1, tolerance=1.0),
                              checkpoints=Checkpoints(str(directory / "run"))))


def test_a_resumed_run_reads_its_batches_in_the_microbatches_its_checkpoint_trained_on(tmp_path):
    """A resumed run that leaves its split open takes the checkpoint's and trains bit for bit as the
    uninterrupted run. Saved at an update's end, where its checkpoint holds no window, it may state a finer
    split, a memory setting alone: its window is laid out afresh, and it lands where the uninterrupted run
    does, within the reductions' rounding. Saved inside an update, its window holds that many microbatches,
    and a resume that states another split is refused."""
    corpus = token_files(tmp_path)
    expected = windowed(tmp_path / "whole", 2).fit(windows(corpus), steps=4, checkpoint_every=2)
    windowed(tmp_path / "split", 2).fit(windows(corpus), steps=2, checkpoint_every=2)
    resumed = windowed(tmp_path / "split", "auto")
    actual = resumed.fit(windows(corpus), steps=4, checkpoint_every=2)
    assert resumed.plan["split"] == 2
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert raw(left).tobytes() == raw(right).tobytes()
    windowed(tmp_path / "finer", 2).fit(windows(corpus), steps=2, checkpoint_every=2)
    finer = windowed(tmp_path / "finer", 4)
    landed = finer.fit(windows(corpus), steps=4, checkpoint_every=2)
    # Its accepted microbatches read in the finer window's: the same steps and updates.
    assert finer.plan["split"] == 4 and (int(landed.step), int(landed.microstep)) == (4, 4 * 4)
    windowed(tmp_path / "inside", 2, accumulation=2).fit(windows(corpus), steps=3, checkpoint_every=3)
    with pytest.raises(ValueError):
        windowed(tmp_path / "inside", 4, accumulation=2).fit(windows(corpus), steps=4, checkpoint_every=4)


@pytest.mark.mesh(devices=2)
def test_a_resumed_run_continues_its_admitted_plan_without_timing(tmp_path, monkeypatch):
    """A run over two devices keeps its admitted plan, and a checkpoint resumed from it continues
    that plan without compiling or timing optional alternatives."""
    corpus = token_files(tmp_path)

    def run(split):
        trainer = windowed(tmp_path / "frontier", split)
        trainer.device_mesh = MeshSpec().build(jax.devices()[:2])
        return trainer

    first = run(1)
    first.fit(windows(corpus), steps=2, checkpoint_every=2)
    assert first.plan["remat"] is None
    resumed = run("auto")
    resumed.fit(windows(corpus), steps=4, checkpoint_every=2)
    assert (resumed.plan["split"], resumed.plan["remat"]) == (first.plan["split"], first.plan["remat"])


def test_a_checkpoint_whose_step_climbed_to_recompute_its_layers_resumes_at_that_rung(tmp_path, monkeypatch):
    """A run whose devices hold its step only recomputing its layers (`remat`) climbs to that rung and
    checkpoints there, its model's record stating the recompute; a run resuming that checkpoint on devices
    that hold either computes the same step, a recompute saying how layers execute and not what they compute
    (`RunConfig.EXECUTED`), so it resumes, at the rung its checkpoint holds (`Trainer.plan`'s first probe;
    a resume was refused as another computation, `model.fields.remat`)."""
    corpus = token_files(tmp_path)

    def run():
        return windowed(tmp_path / "climbed", 1)

    def fitted(trainer, steps):
        trainer.fit(windows(corpus), steps=steps, log_every=100, checkpoint_every=1)
        options.unhold()
        return trainer

    free(monkeypatch, LIMIT)
    own = LIMIT - fitted(windowed(tmp_path / "own", 1), 1).plan["probes"][0]["headroom"]
    free(monkeypatch, LIMIT)
    full = LIMIT - fitted(windowed(tmp_path / "full", 1, remat="full"), 1).plan["probes"][0]["headroom"]
    assert full < own, (full, own)
    free(monkeypatch, (own + full) // 2)
    climbed = fitted(run(), 1)
    assert climbed.plan["remat"] is not None, climbed.plan
    free(monkeypatch, LIMIT)
    resumed = fitted(run(), 2)
    first = resumed.plan["probes"][0]
    assert first["remat"] == climbed.plan["remat"] and first["fits"], (first, climbed.plan)


def test_a_split_plan_does_not_compile_or_time_an_optional_faster_sibling(monkeypatch):
    """A batch that needs two microbatches keeps the first fitting plan, without compiling a
    different remat to ask whether it is faster. Even explicitly supplied timing trials are not run:
    the compile runs no program, and a step then runs its two microbatches alone (row 2230)."""
    free(monkeypatch, needs(monkeypatch, lm(split=1)) - 1)
    trainer = lm(objective=Counted)
    state, _, _ = trainer.place()
    trials = Trials(state, BATCH)
    EXECUTED.clear()
    with Compiles("step") as compiles:
        step = trainer.compile(state, BATCH, trials)
    jax.effects_barrier()
    assert (trainer.plan["split"], trainer.plan["remat"], len(EXECUTED)) == (2, None, 0)
    assert compiles.count == len(trainer.plan["probes"])
    assert all(probe["remat"] is None for probe in trainer.plan["probes"])
    state, loss, *_ = step(state, BATCH)
    jax.effects_barrier()
    assert np.isfinite(float(loss)) and len(EXECUTED) == 2


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("remat", [None, "minimal", "full"], ids=["default", "nonterminal", "terminal"])
@pytest.mark.parametrize("trials_given", [False, True], ids=["without-trials", "with-trials"])
def test_a_fitting_policy_compiles_once_even_with_frontier_trials(monkeypatch, remat, trials_given):
    """Over two devices, an admitted policy compiles once, whether default, explicitly nonterminal,
    or explicitly terminal, and whether or not supplied Trials would time another plan faster."""
    free(monkeypatch, LIMIT)
    policy = None if remat is None else REMAT_POLICIES[remat]
    trainer = lm(remat=policy, objective=Counted)
    trainer.device_mesh = MeshSpec().build(jax.devices()[:2])
    state, _, _ = trainer.place()
    trials = Trials(state, BATCH) if trials_given else None
    EXECUTED.clear()
    with Compiles("step") as compiles:
        trainer.compile(state, BATCH, trials)
    jax.effects_barrier()
    assert compiles.count == 1 and not EXECUTED
    assert trainer.plan["remat"] == remat
    assert len(trainer.plan["probes"]) == 1


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("remat", sorted(REMAT_POLICIES))
def test_fit_keeps_an_explicit_policy_without_a_frontier_probe(tmp_path, monkeypatch, remat):
    """A real fit over two devices keeps each explicit policy it is given once admitted, whether the
    policy has a further rung or is its ladder's last: its step compiles once and runs its one
    microbatch alone, no optional compile or trial hidden in fit's internally created Trials (row 2153;
    the compile-once gate, row 2286)."""
    free(monkeypatch, LIMIT)
    corpus = token_files(tmp_path)
    trainer = windowed(tmp_path, 1, remat=REMAT_POLICIES[remat], objective=Counted)
    trainer.device_mesh = MeshSpec().build(jax.devices()[:2])
    EXECUTED.clear()
    with Compiles("step") as compiles:
        state = trainer.fit(windows(corpus), steps=1, checkpoint_every=None, checkpoint_final=False,
                            eval_every=None)
    jax.effects_barrier()
    # The one step's one microbatch, and no trial beside it.
    assert compiles.count == 1 and int(state.step) == 1 and len(EXECUTED) == 1
    assert [probe["remat"] for probe in trainer.plan["probes"]] == [remat]


@pytest.mark.parametrize("split", [1, 2])
def test_a_plan_with_no_second_point_to_fit_is_not_timed(monkeypatch, split):
    """With room for the batch whole or two microbatches, only the compiles needed to find the
    admitted split appear in the plan. No optional remat candidate is compiled or timed."""
    free(monkeypatch, needs(monkeypatch, lm(split=split)))
    trainer = lm()
    state, _, _ = trainer.place()
    trials = Trials(state, BATCH)
    with Compiles("step") as compiles:
        trainer.compile(state, BATCH, trials)
    assert compiles.count == len(trainer.plan["probes"])
    assert trainer.objective.model.remat is None
    assert (trainer.plan["split"], None, True) in asked(trainer)
    assert all(remat is None for _, remat, _ in asked(trainer))


def asked(trainer) -> list[tuple[int, str | None, bool]]:
    """The compiles a plan's search made, as (split, remat, fits)."""
    return [(probe["split"], probe["remat"], probe["fits"]) for probe in trainer.plan["probes"]]


def assert_searched(trainer, chosen):
    """`trainer`'s plan, a search that settled on `chosen` (split, remat): the batch whole asked first, no
    compile asked twice (a split at a rung, its head and options), and no split asked at the plan's rung
    below the plan's but missed. Which splits are asked follows the memory each compile that missed read,
    which differs between devices: ascending from each miss to the first its figures leave room at
    (`after`)."""
    probes = trainer.plan["probes"]
    keys = [tuple(probe[name] for name in ("split", "remat", "head_tile", "head_fused", "xla_defaults",
                                            "narrow")) for probe in probes]
    assert keys[0][:2] == (1, None) and not probes[0]["fits"]
    assert len(set(keys)) == len(keys), keys
    assert (trainer.plan["split"], trainer.plan["remat"]) == chosen
    fitted = next(key for key, probe in zip(keys, probes, strict=True)
                  if probe["fits"] and key[:2] == chosen)
    assert not any(probe["fits"] for key, probe in zip(keys, probes, strict=True)
                   if key[1:] == fitted[1:] and key[0] < chosen[0])


DELTA = {"mixer": mixers.build("gated_delta_net", linear_num_key_heads=2, linear_num_value_heads=2,
                               linear_key_head_dim=8, linear_value_head_dim=8)}
"""A decoder of one gated delta net layer, whose ladder's first rung above none is 'all_but_delta_memory'."""

HALVED = {"seq": 256, "width": 64, "layers": 2, **DELTA}
"""A decoder of two gated delta net layers over 256 tokens, the delta rule's memory a share of the step
that that rung saves: at four microbatches of 32 rows the rung needs 3.4 to 3.7 MB less than no
recomputation on an A100 and an H100, where the one-layer, sixteen-token decoder's needs 512 bytes more
(row 3341's GPU cell)."""
HALVED_BATCH = {"text": jnp.asarray((np.arange(ROWS * 257).reshape(ROWS, 257) * 7) % 64, jnp.int32)}


def a_step_under_its_room_without_recomputation_plans_a_step_that_fits_there(monkeypatch):
    """With room for four microbatches when the delta rule recomputes its memory, less than four need
    without recomputation (the premise, checked first), the plan the walk derives fits that room by its own
    compile's figures (its probe's headroom), keeps the walk's contract (`assert_searched`: the batch whole
    asked first, no compile asked twice, none skipped at its rung below it that fit), and steps. Which plan
    is the device's to derive: a CPU's XLA fits no finer split
    without recomputation, and plans four microbatches at 'all_but_delta_memory'; a GH200's fits a finer
    one at none (M-MC-19, row 6720). The delta rule runs its port (`port_support.row_names_port`), whatever
    the generation's row names: the rung recomputes the memory the port names (`checkpoint_name`), and the
    portable route, an A10's, names none. What a split or the rung computes is check 1's at their own sites
    (test_memory_composition's recomputed delta memory, test_algebra_carry's split sums)."""
    rung = "all_but_delta_memory"
    with row_names_port("gated_delta_rule", "fla"):
        def measured(**fields) -> int:
            trainer = lm(split=4, **HALVED, **fields)
            return needs(monkeypatch, trainer, lambda: trained(trainer, steps=0, batch=HALVED_BATCH))

        room = measured(remat=rung)
        assert room < measured(), "the rung saves memory at four microbatches"
        free(monkeypatch, room)
        trainer = lm(**HALVED)
        state = trained(trainer, steps=1, batch=HALVED_BATCH)
    plan = trainer.plan
    chosen = (plan["split"], plan["remat"])
    own = [probe for probe in plan["probes"] if probe["fits"] and (probe["split"], probe["remat"]) == chosen]
    assert own and all(probe["headroom"] >= 0 for probe in own), (room, plan["probes"])
    assert_searched(trainer, chosen)
    assert int(state.step) == 1


LONG = 64
"""Rows of 64 tokens: a four-layer decoder's activations, which recomputation frees, outweigh its state."""
LONG_BATCH = {"text": jnp.asarray((np.arange(ROWS * (LONG + 1)).reshape(ROWS, LONG + 1) * 7) % 64, jnp.int32)}


H100 = 2 ** 40
"""The allocator limit of the compile-only H100 a census shape's need at each rung is read on: past any."""


def climbed(monkeypatch, target: str, fields: dict, batch: dict, room_of, request
            ) -> tuple[list, int, tuple, tuple | None, tuple | None, list[tuple]] | None:
    """A census shape's step at four microbatches on `target` ('cpu', its devices' memory stated through
    `free`; 'h100', a compile-only H100 compiled ahead, what a GPU start plans for): the places of the fit
    ladder in its order (`recompute_more`: at each remat rung its head's rungs, then the next), as the
    objective states each (`rung_record`); the room `room_of` the needs by remat at the run's own head, each
    such place's step compiled with room past all as it asks for it (no other place's need is compiled); the
    place the start's climb settles in that room, its rung unstated; and the first place of the ladder whose
    step, compiled in that room, fits: an exhaustive walk. XLA schedules a GPU's program to the memory its
    target states, so a place's need in that room is its compile there, not its need with room past all. A
    place is its head's tile, whether that head is fused, and its remat. A case its test marks `shards` walks
    its places over CI's map tasks, each compiling its own (`map_judge.sharded`): a map task records them and
    gets None, the judge reads them all and climbs."""
    from dew.training.memory import recompute_more, rung_record

    def described(objective) -> tuple:
        record = rung_record(objective, xla_defaults=False, narrow=False)
        return (record["head_tile"], record["head_fused"], record["remat"], False)

    def placed(trainer) -> tuple:
        plan = trainer.plan
        return (plan["head_tile"], plan["head_fused"], plan["remat"], plan["xla_defaults"])

    if target == "cpu":
        def planned(moves: int, room: int):
            free(monkeypatch, room)
            trainer = lm(split=4, **fields)
            for _ in range(moves):
                assert recompute_more(trainer.objective)
            state, _, _ = trainer.place()
            trainer.compile(state, batch, Trials(state, batch))
            return trainer
    else:
        from model_support import memory_limited_to

        from dew.nn.kernels.generation import compile_only_gpus
        from dew.training.ahead import compile_ahead

        def planned(moves: int, room: int):
            trainer = lm(split=4, **fields)
            trainer.device_mesh = MeshSpec().build(compile_only_gpus("sm90", 1, memory_limited_to(room)))
            for _ in range(moves):
                assert recompute_more(trainer.objective)
            compile_ahead(trainer, batch)
            return trainer

    room_past = LIMIT if target == "cpu" else H100
    ladder = lm(split=4, **fields).objective
    places = [described(ladder)]
    while recompute_more(ladder):
        places.append(described(ladder))
    # The needs at the run's own head, by remat, each compiled with room past all as `room_of` reads it.
    own = {remat: moves for moves, (tile, _, remat, _) in enumerate(places) if tile is None}

    def need(remat) -> int:
        trainer = planned(own[remat], room_past)
        assert placed(trainer)[:3] == places[own[remat]][:3], (placed(trainer), places[own[remat]])
        return room_past - trainer.plan["probes"][0]["headroom"]

    room = room_of(need)
    def fits(moves: int) -> bool:
        # The place's own step compiled in the room, its first compile the judge: where it misses, the
        # search past it may find nothing more at a remat the run now states (`PlanRefused`).
        try:
            return planned(moves, room).plan["probes"][0]["fits"]
        except PlanRefused as refused:
            return refused.probes[0].fits

    if request.node.get_closest_marker("shards") is None:
        # Each place's fit compiled where it is asked, once (no place past the one sought is compiled).
        fitted = functools.cache(fits)
    else:
        walked = sharded(f"{request.node.name} walk", lambda start, stop: [fits(moves) for moves in
                                                                            range(start, stop)], len(places))
        if walked is None:
            return None
        found = [fit for part in walked for fit in part]
        fitted = found.__getitem__
    # The remat rungs past the start's at the run's own head, in order: what a tight start asks next.
    preferred = [moves for moves, place in enumerate(places)
                 if moves and place[0] is None and place[2] != places[0][2]]
    settled = next((at for at, moves in enumerate(preferred) if fitted(moves)), None)
    recomputing = None if settled is None else places[preferred[settled]]
    # What a tight start asks, a probe a place at its fixed split: its own rung, then each remat rung at its
    # own head and options up to the first that fits.
    asks = None if settled is None else [places[0], *(places[moves] for moves in preferred[:settled + 1])]
    lightest = None if recomputing is not None else next(
        (place for moves, place in enumerate(places) if fitted(moves)), None)
    trainer = planned(0, room)
    probes = [(probe["head_tile"], probe["head_fused"], probe["remat"], probe["xla_defaults"])
              for probe in trainer.plan["probes"]]
    return places, room, placed(trainer), lightest, asks, probes


CENSUS = {
    # Four attention layers over 64 tokens, room between what 'minimal' and no rung need.
    "decoder": ({"seq": LONG, "width": 32, "layers": 4}, LONG_BATCH,
                lambda need: (need(None) + need("minimal")) // 2),
    # Two scanned gated delta net layers over 256 tokens, room for what 'minimal' needs.
    "delta-scanned": ({**HALVED, "scan_layers": True}, HALVED_BATCH, lambda need: need("minimal")),
}
"""The census shapes: their decoders' fields, batches, and the room each start is given of their needs by
remat at their own head, each read as asked (`climbed`): no other rung's need is compiled."""


@pytest.mark.parametrize(("shape", "target"), [
    pytest.param("decoder", "cpu", id="decoder-cpu"),
    pytest.param("decoder", "h100", id="decoder-h100"),
    pytest.param("delta-scanned", "h100", id="delta-scanned-h100",
                 marks=pytest.mark.shards(units=18, per_task=2)),
])
def test_a_starts_climb_settles_the_lightest_rung_an_exhaustive_walk_finds_fits(monkeypatch, shape, target,
                                                                                    request):
    """A start whose step does not fit at its own rung asks the remat rungs at its own head first and
    settles the first of them that fits (row 6740: a tight cold start's second compile), where an
    exhaustive walk compiling each place of the ladder in the same room finds one fits, and the lightest
    place it finds fits where none of them does, the decoder on the CPU and for a compile-only H100, the
    scanned delta decoder for the H100, what a GPU start plans for: its CPU case walked its ladder
    in 418 s of a train's call time past the train's bound (row 7414), and the CPU's walk is the decoder's.
    What a rung keeps orders its residuals, not XLA's peak (the scanned delta decoder needed more recomputing
    its scans and convolutions too than not), so a search that passed over a rung for one seen to miss
    settled a heavier plan than one that fits. The scanned delta decoder's walk of its 18 places runs over
    CI's map tasks, 2 a task: compiled in one it took 262 to 350 s of a train's task (row 8438), 6 a task
    469 s, its judge 168 s after (row 8524), 3 a task 299 s for its first, the heaviest (row 8557)."""
    fields, batch, room_of = CENSUS[shape]
    with row_names_port("gated_delta_rule", "fla"):
        climb = climbed(monkeypatch, target, fields, batch, room_of, request)
    if climb is None:
        return
    ladder, room, plan, lightest, asks, probes = climb
    if asks is not None:
        # A tight start asks its own rung, then the remat rungs at its own head and options in order, and
        # settles the first of them that fits: no head or options rung compiled on the way (row 6740).
        assert probes == [(*place[:3], False) for place in asks], (probes, asks)
        assert plan == probes[-1], (plan, probes)
    else:
        # Where no remat rung at its own head fits, the whole ladder: no later than the lightest place the
        # walk (under the step's own options alone) finds fits, a rung of XLA's defaults maybe earlier.
        places = [place[:3] for place in ladder]
        assert lightest is not None and places.index(plan[:3]) <= places.index(lightest[:3]), (
            ladder, room, plan, probes)


def test_a_plan_made_before_the_state_is_placed_is_searched_again_where_the_placed_devices_cover_a_miss(
        monkeypatch, caplog):
    """A fresh fit plans its step before its state is placed (`StepCompiler.planned_unplaced`), judging each
    candidate beside the state's bytes it will place: room for two microbatches and not the batch whole.
    Placed, the devices read no fewer free bytes (a fine-tune's copied sources let go as their copies are
    placed), room for the batch whole, whose miss that room covers: the step is searched again there and
    reads the batch whole, where the held plan's two microbatches ran the whole run."""
    # The step the run compiles, its split unstated, read whole and in halves.
    whole, halves = needs(monkeypatch, lm()), needs(monkeypatch, lm(split=2))
    assert halves < whole, (halves, whole)
    state, _, _ = lm().place()
    held = sum(leaf.nbytes for leaf in jax.tree.leaves(state) if isinstance(leaf, jax.Array))
    # Room for the batch whole once its state is placed, short of it while the state's bytes stand beside
    # it unplaced: the step the search compiles over the unplaced state needs as much as the placed one
    # within those bytes.
    free(monkeypatch, whole + held)
    trainer = lm()
    with caplog.at_level(logging.INFO, logger="dew.training.compiler"):
        trainer.fit(Pipeline(Memory({"text": np.asarray(BATCH["text"])})).repeat().batch(ROWS), steps=1)
    assert any("searched again there" in record.getMessage() for record in caplog.records), (
        held, whole, halves, [(probe["split"], probe["fits"], probe["headroom"]) for probe in
                              trainer.plan["probes"]], caplog.text)
    assert (trainer.plan["split"], trainer.plan["remat"]) == (1, None), trainer.plan["probes"]


def test_a_plan_at_no_recomputation_stops_before_any_optional_frontier(monkeypatch):
    """An A100's plan: no recomputation fits the batch at four microbatches and not in fewer, so the plan
    is four at no recomputation, which XLA's memory analysis of each compile decides, and no optional
    full-batch frontier is compiled."""
    free(monkeypatch, needs(monkeypatch, lm(split=4, **DELTA)))
    trainer = lm(**DELTA)
    state, _, _ = trainer.place()
    trials = Trials(state, BATCH)
    trainer.compile(state, BATCH, trials)
    assert (trainer.plan["split"], trainer.plan["remat"]) == (4, None)
    assert "frontier" not in trainer.plan


def test_a_decoder_that_fits_split_asks_its_splits_at_no_recomputation(monkeypatch):
    """A step that fits at four microbatches with no recomputation: the search asks the batch whole and
    the splits its misses compute, each at no recomputation, and stops there."""
    free(monkeypatch, needs(monkeypatch, lm(split=4)))
    trainer = lm()
    state, _, _ = trainer.place()
    trials = Trials(state, BATCH)
    trainer.compile(state, BATCH, trials)
    assert_searched(trainer, (4, None))
    assert all(remat is None for _, remat, _ in asked(trainer))


def test_a_step_the_fit_check_admits_is_within_the_limit_its_compile_stops_at():
    """The bytes past which a plan's GPU compile stops (`compile_limit`) are the fit check's own arithmetic
    solved for XLA's sum (arguments, outputs that write over none, temporaries): a 4.8B multimodal
    transformer's step (row 5805), its
    arguments 63.1 GB with 22.0 GB of them built on its first call, held beside it, its temporaries 18.9 GB,
    which the check admitted on a GH200's 96.4 GB, compiles whole within it, where the limit, its
    allocator's less all it held, counted the built arguments twice, stopped its compile at 74.4 GB and its
    first run refused it; a prefetched batch held beside the step outside its arguments lowers the limit by
    its bytes."""
    from model_support import memory_limited_to

    from dew.nn.kernels.generation import compile_only_gpus
    from dew.training.memory import compile_limit

    room, held, built = 96_395_591_680, 21_975_318_136, 21_975_318_136
    summed = 63_111_557_912 + 63_110_666_810 - 63_110_643_844 + 18_894_096_952
    devices = compile_only_gpus("sm90", 1, memory_limited_to(room))
    assert compile_limit(devices, held, built) >= summed
    batch = 2**20
    assert compile_limit(devices, held + batch, built) == compile_limit(devices, held, built) - batch


def test_a_step_that_fails_on_its_first_run_states_the_room_outside_xla_s_pool():
    """A newly compiled step whose first run fails on its devices is raised as the step's, from the
    devices' own error, which its message states, so a reader of the message alone tells an out-of-memory
    from any other failure, with the record of the check that said it fits and what each device holds now,
    each GPU's bytes free outside XLA's pool among it: a collective's communicators (NCCL) take their
    buffers there, beside cuBLAS's workspaces and the kernels loaded (row 2901). A first run that
    succeeds tells each GPU's free bytes there at the check and after it."""
    class Refused:
        """A step's loss whose run the devices refused, as a GPU reports it once it is awaited."""

        def block_until_ready(self):
            raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: the devices refused the step")

    mesh = MeshSpec().build(jax.devices()[:1])
    devices = local_devices(mesh)
    gpus = [device.id for device in devices if device.platform == "gpu"]
    compiled = jax.jit(lambda value: value * 2).lower(jnp.float32(0)).compile()
    record = memory_record(compiled, devices, 0, fits=True)
    assert [device["id"] for device in record["devices"] if device["outside"] is not None] == gpus
    loss = compiled(jnp.float32(1))
    observed = first_run(loss, 0, record, mesh)
    assert observed["checked"] == {device["id"]: device["outside"] for device in record["devices"]
                                   if device["outside"] is not None}
    assert list(observed["now"]) == gpus and all(free >= 0 for free in observed["now"].values())
    with pytest.raises(FirstRunFailed) as raised:
        first_run(Refused(), 0, record, mesh)
    assert isinstance(raised.value.__cause__, jax.errors.JaxRuntimeError)
    assert str(raised.value.__cause__) in str(raised.value)
    assert raised.value.fitted == record
    assert [(device["id"], device["outside"] is not None) for device in raised.value.now] == [
        (device.id, device.id in gpus) for device in devices]


@pytest.mark.parametrize("owned", [True, False], ids=["owned", "borrowed"])
def test_trials_step_a_copy_of_the_state_and_hand_back_a_fresh_one(owned):
    """An AutoPGLE trial runs a step on a copy of the state as it stood: trials that own the state
    (`fit`'s) step it and delete it, handing back a copy as it was; trials that borrow it (a compile given
    none) step a copy beside it and hand back the state itself, as it is. Either way the run trains on
    from where it stood."""
    trainer = lm(split=1)
    state, _, _ = trainer.place()
    step = trainer.compile(state, BATCH)
    before = [raw(leaf).copy() for leaf in jax.tree.leaves(state)]
    trials = Trials(state, BATCH, owned=owned)
    for _ in range(2):
        advanced = trials.run(lambda current: jax.block_until_ready(step(current, BATCH)[0]))
        assert int(advanced.step) == int(trials.state.step) + 1
    assert all(leaf.is_deleted() == owned for leaf in jax.tree.leaves(state))
    assert all(np.array_equal(raw(left), right)
               for left, right in zip(jax.tree.leaves(trials.state), before, strict=True))


def test_a_run_trains_only_its_requested_steps_without_timing_a_frontier(monkeypatch, tmp_path):
    """A fit trains its requested steps without spending extra updates on a speed comparison."""
    corpus, whole = token_files(tmp_path), windowed(tmp_path / "whole", 1)
    free(monkeypatch, needs(monkeypatch, whole, lambda: whole.fit(windows(corpus), steps=1,
                                                                  checkpoint_every=None)) - 1)
    trainer = windowed(tmp_path / "timed", "auto")
    with Compiles("step") as compiles:
        state = trainer.fit(windows(corpus), steps=3, checkpoint_every=None)
    assert compiles.count == len(trainer.plan["probes"])
    assert int(state.step) == 3


KEY = """
import json, sys
sys.path.insert(0, "tests")
from test_plan import BATCH, lm
from dew.training.rungs import rung_key
trainer = lm()
state, _, _ = trainer.place()
print(json.dumps(rung_key(trainer.objective.program_key(), (state, BATCH), [1, 2], trainer.device_mesh)))
"""


def test_two_processes_name_one_run_alike():
    """A record is found by the next process of the same run, so nothing the key names may be one
    process's alone, such as an address Python prints beside a function."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    keys = []
    for _ in range(2):
        done = subprocess.run([sys.executable, "-c", KEY], cwd=root, capture_output=True, text=True,
                              timeout=300)
        assert done.returncode == 0, done.stderr[-3000:]
        keys.append(json.loads(done.stdout.splitlines()[-1]))
    assert keys[0] is not None and keys[0] == keys[1]


@dataclasses.dataclass(frozen=True)
class Displayed:
    """A module configuration whose display is its instance's own, as a custom repr may make it."""

    width: int
    held: object = None

    def __repr__(self) -> str:
        return f"Displayed at {id(self):#x}"


def test_a_rung_key_states_its_modules_configuration_not_their_display():
    """A rung record's key states its modules as plain data (`dew._stated.stated`), not as they print: two
    equal configurations whose displays differ name one record, another width another; a module holding an
    array of JAX's, or a typed PRNG key, by its values, the same values one record and others another."""
    from dew.training.rungs import rung_key

    trainer = lm()
    state, _, _ = trainer.place()

    def key(module):
        return rung_key([module], (state, BATCH), [1, 2], trainer.device_mesh)

    assert key(Displayed(4)) == key(Displayed(4)) != key(Displayed(8))
    zeros, ones = (key(Displayed(4, held=values)) for values in (jnp.zeros(2), jnp.ones(2)))
    assert zeros is not None and zeros == key(Displayed(4, held=jnp.zeros(2))) != ones
    # A typed PRNG key by its key data, its key type beside it (train 1019's review, F1).
    first, second = (key(Displayed(4, held=jax.random.key(seed))) for seed in (0, 1))
    assert first is not None and first == key(Displayed(4, held=jax.random.key(0))) != second


def test_a_rung_key_holds_a_flag_given_twice_by_its_order(monkeypatch):
    """A flag given twice is read by its last value, so a rung record's key states the flags in their order:
    two orders of one flag's two values compute apart and key apart (row 10797: sorted, they were one key),
    and the same order keys alike."""
    import os

    from dew.training.rungs import rung_key

    trainer = lm()
    state, _, _ = trainer.place()
    flags = os.environ.get("XLA_FLAGS", "")

    def key(*settings):
        monkeypatch.setenv("XLA_FLAGS", " ".join((flags, *settings)))
        return rung_key(trainer.objective.program_key(), (state, BATCH), [1, 2], trainer.device_mesh)

    autotune_on, autotune_off = "--xla_gpu_autotune_level=4", "--xla_gpu_autotune_level=0"
    first = key(autotune_on, autotune_off)
    assert first is not None and first == key(autotune_on, autotune_off) != key(autotune_off, autotune_on)


SCALE = 2
"""A module setting the wrapped function below reads."""


def test_a_users_function_wrapped_as_a_librarys_states_the_globals_it_reads(monkeypatch):
    """A user's function that names a library's as the one it wraps (`functools.wraps`, which copies the
    library's module onto it) is the user's by the globals it was defined in, so its statement holds the
    module setting it reads, and a change to that setting states another function (row 10554: taken as the
    library's, its globals were dropped, and a rung record or truth store keyed a stale entry)."""
    import json

    from dew._stated import stated

    @functools.wraps(json.dumps)
    def scaled(value):
        return value * SCALE

    before = stated(scaled, "a wrapped function")
    monkeypatch.setitem(globals(), "SCALE", 3)
    assert stated(scaled, "a wrapped function") != before


def test_arrays_of_one_byte_layout_and_other_dtypes_fingerprint_apart():
    """float8 e4m3fn and e4m3fnuz, and int4 and uint4, share one byte layout (`dtype.str` '<V1'): an
    array of each holding the same bytes fingerprints apart, as the same values of one dtype do not."""
    import ml_dtypes

    from dew.data.plan import fingerprint

    raw = np.arange(8, dtype=np.uint8)
    pairs = ((ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e4m3fnuz), (ml_dtypes.int4, ml_dtypes.uint4))
    for first, second in pairs:
        assert np.dtype(first).str == np.dtype(second).str
        assert fingerprint(raw.view(first)) != fingerprint(raw.view(second))
        assert fingerprint(raw.view(first)) == fingerprint(raw.copy().view(first))



class Refused(jax.errors.JaxRuntimeError):
    """An XLA refusal with its status code, as a GPU's compiler raises it."""
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code

    @property
    def error_code_string(self):
        return self.code


OUT_OF_MEMORY = ("EXECUTION FAILED: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 7.83GiB with "
                 "allocator GPU_0_bfc on device 0. [tf-allocator-allocation-error='']")


def autotuned(*failures: str) -> str:
    """A GPU GEMM autotuner's NOT_FOUND message listing `failures`, as XLA's `PickBestConfig` writes it."""
    return ("All configs failed during profiling or were excluded from selection.\n"
            f"Failures ({len(failures)}):\n" + "\n".join(failures))


@pytest.mark.parametrize(("code", "message", "refused"), [
    ("RESOURCE_EXHAUSTED", "XLA:TPU's temporaries past HBM", True),
    ("NOT_FOUND", autotuned(OUT_OF_MEMORY, OUT_OF_MEMORY), True),
    ("NOT_FOUND", autotuned(OUT_OF_MEMORY, "EXECUTION FAILED: INTERNAL: an illegal address"), False),
    ("NOT_FOUND", autotuned(OUT_OF_MEMORY, "COMPILATION FAILED: INTERNAL: ptxas exited with 255"), False),
    ("NOT_FOUND", autotuned(OUT_OF_MEMORY, "REDZONE CHECK FAILED: the kernel wrote past its output"), False),
    ("NOT_FOUND", autotuned(OUT_OF_MEMORY, "WRONG RESULTS: Output disagrees with winning cluster"), False),
    ("NOT_FOUND", autotuned(OUT_OF_MEMORY, "cublas_config: Backend excluded from selection (CUBLAS)"), False),
    ("NOT_FOUND", "All configs failed during profiling or were excluded from selection.", False),
    ("NOT_FOUND", "no config for this fusion", False),
    ("INTERNAL", "a ptxas failure", False)],
    ids=["exhausted", "autotuned-out-of-memory", "autotuned-executed-otherwise",
         "autotuned-compilation-failed", "autotuned-redzone", "autotuned-wrong-results", "autotuned-excluded",
         "autotuned-nothing-listed", "not-found", "internal"])
def test_a_gemm_autotuner_out_of_memory_is_a_refusal_for_memory(code, message, refused):
    """XLA's refusal for memory, which the plan reads as the rung's miss and climbs past
    (`compiled_if_it_fits`), is RESOURCE_EXHAUSTED, or a GPU GEMM autotuner's NOT_FOUND where every failure
    it lists is a config's execution out of memory, as an A100's were at a whole head's GEMM (node receipt
    TL-a100-offload-1338-d). One failure of another kind beside them, a config that failed to compile, its
    redzone check or its results, or a backend excluded, and the error is raised as it came, as is any
    other."""
    from dew.training.memory import compiled_if_it_fits

    error = Refused(code, message)
    lowered = jax.jit(lambda: None).lower()

    class Program:
        # The lowering Dew's compile point reads the target's devices from (`report.targeted`).
        _lowering = lowered._lowering

        def compiler_ir(self, dialect):
            return lowered.compiler_ir(dialect)

        def compile(self, options):
            raise error

    refusals = []
    try:
        outcome = compiled_if_it_fits(Program(), None, refusals, relower=Program)
    except jax.errors.JaxRuntimeError as raised:
        outcome = raised
    assert (outcome, refusals) == ((None, [error]) if refused else (error, []))


@pytest.mark.parametrize("width, between", [
    # Its temporaries scaled by its rows (a narrow layer).
    (16, (1, 2)),
    # They fall faster than its rows.
    (16, (2, 4)),
    # Mostly fixed (a wide layer over few tokens: the gradients of its weights).
    (1024, (2, 4)),
])
def test_a_tight_start_settles_the_split_beside_one_that_missed_and_compiles_once_warm(monkeypatch, tmp_path,
                                                                                        width, between):
    """What a run pays at its start where its batch does not fit whole (the owner's contract: "a single
    trace and compile most often"): cold, the plan is the split seen to fit beside the next coarser seen
    to miss, those `between`, each split compiled once at most, no more than the splits there are; warm,
    the plan its cold start recorded (`keep_rung`, beside the compilation cache) is the first candidate,
    one compile, its step the one it runs. The devices' memory lies between the fewest's need and the least
    of the whole batch's and the first's, so both miss: the whole batch's need, its accumulator absent, is
    not the most of every split's."""
    whole, fewer, fewest = (needs(monkeypatch, lm(split=split, width=width)) for split in (1, *between))
    missing = min(whole, fewer)
    assert fewest < missing, (fewest, whole, fewer)
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path))
    free(monkeypatch, (missing + fewest) // 2)
    cold_run, warm = lm(width=width), lm(width=width)
    with Compiles("step") as compiles:
        trained(cold_run, steps=0)
    probes = cold_run.plan["probes"]
    assert cold_run.plan["split"] == between[1], probes
    assert any(probe["split"] == between[0] and not probe["fits"] for probe in probes), probes
    asked = [probe["split"] for probe in probes]
    assert len(set(asked)) == len(asked) <= ROWS and compiles.count <= len(asked), (compiles.count, probes)
    # A warm process: the cold one's record, none of its executables.
    options.unhold()
    with Compiles("step") as compiles:
        trained(warm, steps=0)
    assert compiles.count == 1 and warm.plan["split"] == between[1]


def test_a_run_that_sets_its_remat_compiles_its_step_once(monkeypatch, tmp_path):
    """An explicit full policy compiles one step, counted by the telemetry owner of JAX's public
    monitoring events rather than a private log's spelling (rows 2111, 2154)."""
    free(monkeypatch, LIMIT)
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path))
    trainer = lm(split=1, remat=REMAT_POLICIES["full"])
    with Compiles("step") as compiles:
        trained(trainer, steps=0)
    assert compiles.count == 1
    assert [probe["remat"] for probe in trainer.plan["probes"]] == ["full"]
    assert trainer.plan["remat"] == "full" and trainer.objective.model.remat == REMAT_POLICIES["full"]


def test_a_runs_fit_policy_sets_what_its_step_recomputes_for_every_module(monkeypatch):
    """A run's fit policy states its remat for every module its model states none of (`FitPolicy.remat`):
    "full" compiles one step at each module's whole recomputation, the top of its ladder; "none" one that
    recomputes nothing; a model that states its own remat beside a policy that states one is refused as
    its trainer is built (row 11934), and a later phase's planner over the objective its climb left, its
    model stating the policy's remat, plans on (`dew.training.ahead`; Train 1229's review: refused, every
    later phase compiled on the loop)."""
    from dew.training.compiler import StepCompiler
    from dew.training.phase_passes import PhasePasses

    free(monkeypatch, LIMIT)
    full, none = lm(split=1, policy="full"), lm(split=1, policy="none")
    with Compiles("step") as compiles:
        trained(full, steps=0)
        trained(none, steps=0)
    assert compiles.count == 2
    assert full.plan["remat"] == "full" and full.objective.model.remat == REMAT_POLICIES["full"]
    assert none.plan["remat"] is None and none.objective.model.remat is None
    with pytest.raises(ValueError):
        lm(policy="none", remat=REMAT_POLICIES["minimal"])
    planner = StepCompiler(full.objective, None, None, PhasePasses(full.objective), "full")
    assert planner.remat == "full"


def test_a_run_of_one_remat_takes_no_plan_a_run_of_another_recorded(monkeypatch, tmp_path):
    """Plans recorded beside the compilation cache are a fit policy's own (`rung_key`): a run of "auto"
    whose devices hold too little for its step at no recomputation records a plan that recomputes, and a
    run of "none" over the same step after it, given the room for none, plans its own at no recomputation,
    where it read the first's and was refused it (review of row 11934)."""
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path))
    plain, recomputing = needs(monkeypatch, lm(split=1)), needs(monkeypatch, lm(split=1, policy="full"))
    assert recomputing < plain, (recomputing, plain)
    free(monkeypatch, (plain + recomputing) // 2)
    tight = lm(split=1)
    trained(tight, steps=0)
    assert tight.plan["remat"] is not None, tight.plan
    options.unhold()
    free(monkeypatch, LIMIT)
    none = lm(split=1, policy="none")
    trained(none, steps=0)
    assert none.plan["remat"] is None and none.objective.model.remat is None


def test_a_checkpoints_shape_rung_its_run_s_remat_policy_refuses_is_not_climbed_to(monkeypatch):
    """A checkpoint whose run's own rung recomputes nothing and whose shape's rung climbed (a step that did
    not fit), resumed by a run whose fit policy recomputes nothing (`FitPolicy(remat="none")`): its own rung
    is taken and its shape's, which the policy refuses, is not, each judged against the objective as the
    run authored it, so the step compiles recomputing nothing (review of API01: the shapes' rungs were read
    unjudged)."""
    plain, recomputing = needs(monkeypatch, lm(split=1)), needs(monkeypatch, lm(split=1, policy="full"))
    assert recomputing < plain, (recomputing, plain)
    free(monkeypatch, (plain + recomputing) // 2)
    tight = lm(split=1)
    trained(tight, steps=0)
    climbed = tight._compiler.checkpoint_rung()
    assert tight.plan["remat"] is not None and climbed["shapes"], tight.plan
    options.unhold()
    free(monkeypatch, LIMIT)
    none = lm(split=1, policy="none")
    none._compiler.resume_rung({**none._compiler.rung(), "shapes": climbed["shapes"]})
    trained(none, steps=0)
    assert none.plan["remat"] is None and none.objective.model.remat is None, none.plan


def test_a_fits_start_traces_its_loss_once_for_its_fit_check_and_its_plan(monkeypatch):
    """A fit's check of the whole state its steps keep (`Trainer._whole_validated`) and its plan's
    candidates read one trace of its loss's shapes (`StepCompiler.loss_shaped`): a CPU start traced it
    twice, the fit check's and the plan's, 4.1 s and 0.6 s of a chat-format decision run's 55 s."""
    trainer = lm()
    data = Pipeline(Memory({"text": np.asarray(BATCH["text"])})).repeat().batch(ROWS)
    with Compiles(None) as compiles:
        trainer.fit(data, steps=1)
    traced = [program for program in compiles.programs if program.kind == "traced" and program.name == "loss"]
    assert len(traced) == 1, [program.where for program in traced]


def muon_lm(remat: str | None = None, **opts) -> Trainer:
    """A four-layer LM trained by Muon over one microbatch a step of two rows: its layers' matrices of one
    shape are stacks its Newton-Schulz joins where `opts` states no group, wide enough beside the rows'
    activations that the update's iteration holds the step's peak; recomputing as `remat` names where
    given."""
    model = CausalTransformer(vocab_size=64, emb_features=128, num_layers=4, num_heads=2, mlp_features=256,
                              max_seq_len=SEQ, **({} if remat is None else {"remat": REMAT_POLICIES[remat]}))
    config = OptimConfig(optimizer=Muon(**opts), learning_rate=1e-2)
    return one_device(Trainer(LMObjective(model, seq_len=SEQ), config, key=jax.random.key(0),
                              fit=FitPolicy(split=1)))


def fitted_once(trainer: Trainer) -> None:
    """`trainer` fit one step of two of BATCH's rows, its plan made as a fit makes it."""
    trainer.fit(Pipeline(Memory({"text": np.asarray(BATCH["text"][:2])})).repeat().batch(2), steps=1)


@functools.cache
def muon_needs() -> tuple[int, int, JSON]:
    """The rooms the cells below free for the Muon step (`muon_lm`) before its state is placed
    (`free_until_placed`), and the parted step's remat. Each holds the bytes of the arguments the devices
    hold none of yet, which a fit plans its step beside before placing them, and lies past what the step
    needs beside what a device holds with its Newton-Schulz stacks parted (`ns_group` 1): the first half way
    from there to what it needs with them joined, the second half way from the most of that and of what it
    needs with them grouped by four recomputing 'minimal', the first rung of that group's climb that frees
    anything, to what it needs grouped by four. Freed as `free` frees, without the arguments and never taken
    by them, no rung of that plan fitted and the placed devices' room passed the joined step: the fit
    searched its ladder again once placed (row 8606: 129 s a cell). Measured once."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        joined = needs(monkeypatch, trainer := muon_lm(), lambda: fitted_once(trainer))
        grouped = needs(monkeypatch, trainer := muon_lm(ns_group=4), lambda: fitted_once(trainer))
        recomputing = needs(monkeypatch, trainer := muon_lm("minimal", ns_group=4),
                            lambda: fitted_once(trainer))
        parted = needs(monkeypatch, trainer := muon_lm(ns_group=1), lambda: fitted_once(trainer))
    assert parted < joined and max(parted, recomputing) < grouped, (parted, recomputing, grouped, joined)
    arguments = trainer.plan["probes"][0]["argument"]
    return ((joined + parted) // 2 + arguments, (grouped + max(parted, recomputing)) // 2 + arguments,
            trainer.plan["remat"])


@pytest.mark.shared_setup("muon_needs")
def test_a_muon_step_that_fits_only_with_its_stacks_parted_parts_them_before_recomputing(monkeypatch):
    """A Muon step whose joined Newton-Schulz stacks' copies leave it short of the devices' room, and which
    fits with them parted, its split fixed: its plan parts them, joining one at a time (`ns_group` 1, the
    climb's release after the narrow copies, `FitSearch.searched`), and recomputes nothing more, where it
    climbed the remat rungs (row 5889: the joined copies held 11.3 GB of a 4B step's peak for 82 ms a
    step)."""
    room, _, remat = muon_needs()
    free_until_placed(monkeypatch, room)
    planned = muon_lm()
    fitted_once(planned)
    assert planned.plan["ns_group"] == 1, planned.plan
    assert planned.plan["remat"] == remat, planned.plan


@pytest.mark.shared_setup("muon_needs")
def test_a_muon_run_that_states_its_stacks_group_keeps_it(monkeypatch):
    """A Muon run that states its Newton-Schulz group keeps it, in the room where an unstated run's plan
    parts its stacks, and its own step fits recomputing: its group joins every stack, as the joined step
    does, and the planner asks no step of it to join them one at a time (`FitSearch.searched`), recomputing
    instead (row 8696: in all of the room no release was ever asked, so a stated group overridden failed
    nothing)."""
    _, room, _ = muon_needs()
    free_until_placed(monkeypatch, room)
    stated = muon_lm(ns_group=4)
    fitted_once(stated)
    groups = [probe["ns_group"] for probe in stated.plan["probes"]]
    assert stated.plan["ns_group"] is None and stated.plan["remat"] is not None, stated.plan
    assert groups and 1 not in groups, groups

def lowered_and_compiled(run: Callable[[], object]) -> tuple[int, int]:
    """How many times `run` lowered the step (JAX's lowering of `jit(step)`) and compiled it (`Compiles`)."""
    lowered = []

    def lowering(event, seconds, *, fun_name=None, **_):
        if event == "/jax/core/compile/jaxpr_to_mlir_module_duration" and fun_name == "jit(step)":
            lowered.append(seconds)

    jax.monitoring.register_event_duration_secs_listener(lowering)
    try:
        with Compiles("step") as compiles:
            run()
    finally:
        jax.monitoring.unregister_event_duration_listener(lowering)
    return len(lowered), compiles.count


@pytest.mark.parametrize("stated", [False, True], ids=["planned", "remat-stated"])
def test_a_fit_whose_step_fits_only_recomputing_searches_it_once_and_compiles_its_plan_once(monkeypatch,
                                                                                           stated):
    """A 9B decision run's start on two H100s (2026-11-14), and on four (11-15): a step no
    recomputation fits, and 'minimal' does, with room for the state beside it. A fit searches its step
    once, before its state is placed, and its first compile takes that plan: each candidate the search
    asked is lowered once and compiled once, and the plan's executable, compiled once, is the step the
    run takes. It searched twice, the second time from the ladder's foot over the placed state, every
    candidate lowered again (seven candidates, fourteen lowerings; the 9B run's 18 compiles, 71 minutes). A
    run that states the rung its search ends on lowers and compiles its step once."""
    fields = {"seq": LONG, "width": 32, "layers": 4, "split": 4}

    def measured(**more) -> int:
        trainer = lm(**fields, **more)
        return needs(monkeypatch, trainer, lambda: trained(trainer, steps=0, batch=LONG_BATCH))

    recomputing, whole = measured(remat=REMAT_POLICIES["minimal"]), measured()
    room = (recomputing + whole) // 2
    free(monkeypatch, room)
    trainer = lm(**fields, **({"remat": REMAT_POLICIES["minimal"]} if stated else {}))
    data = Pipeline(Memory({"text": np.asarray(LONG_BATCH["text"])})).repeat().batch(ROWS)
    lowered, compiled = lowered_and_compiled(lambda: trainer.fit(data, steps=1))
    probes = trainer.plan["probes"]
    assert trainer.plan["remat"] is not None and probes[-1]["fits"]
    # Each candidate asked lowered and compiled once at most, the plan's executable the step the run takes;
    # a run that states its rung compiles that alone.
    assert lowered <= len(probes) and compiled <= len(probes), (lowered, compiled, probes)
    assert not stated or (lowered, compiled, len(probes)) == (1, 1, 1), probes
    if not stated:
        assert probes[0]["remat"] is None and not probes[0]["fits"]
        # Only the rungs at or below the plan's recomputation are asked: a step that fits one remat rung
        # up asks what a climb to it asks, no heavier rung that fits and is not taken.
        keys = [tuple(json.dumps(probe[name]) for name in ("split", "remat", "head_tile", "head_fused",
                                                        "xla_defaults", "narrow")) for probe in probes]
        assert len(set(keys)) == len(keys), keys
        order = [None, *(remat_record(policy) for policy in DECODER_REMAT[1:])]
        assert all(order.index(probe["remat"]) <= order.index(trainer.plan["remat"]) for probe in probes), (
            [probe["remat"] for probe in probes])


def test_each_probe_of_a_held_plan_states_its_own_headroom(monkeypatch):
    """A fit's plan made before its state is placed, its state's bytes beside its step, taken once the
    state is placed (`StepCompiler.compile`): the headroom the placed devices read is the plan's own
    probe's, the last that fit at its split, where its search asked a coarser split after it, which missed
    (the batch whole's miss computes four microbatches; four fit; two, asked after, do not): every probe's
    headroom says what its verdict says, a missed one's no plan's."""
    from dew.data import Memory
    from dew.data.pipeline import Pipeline
    from dew.distributed.placement import placed_bytes

    state, _, _ = lm().place()
    beside = placed_bytes(state)
    fours = needs(monkeypatch, lm(split=4)) + beside
    halves = needs(monkeypatch, lm(split=2)) + beside
    assert fours < halves, (fours, halves)
    free(monkeypatch, (fours + halves) // 2)
    trainer = lm()
    trainer.fit(Pipeline(Memory({"text": np.asarray(BATCH["text"])})).repeat().batch(ROWS), steps=1)
    probes = trainer.plan["probes"]
    assert trainer.plan["split"] == 4 and [probe["split"] for probe in probes if not probe["fits"]][-1] == 2
    assert all((probe["headroom"] >= 0) is probe["fits"] for probe in probes), probes



def test_a_trainer_built_after_the_process_releases_its_programs_traces_its_step_again(monkeypatch):
    """A trainer built anew reads the step an equal trainer traced over the same shapes (`candidate_for`),
    tracing fewer programs than it did; once the process releases its programs (`options.release`), which
    drops the kept steps with the executables their lowerings key, the next traces it again (row 8411's
    review: the kept steps, 32 of them, outlived a release that dropped every executable but theirs)."""
    from dew.telemetry.compiles import Compiles

    free(monkeypatch, LIMIT)
    # One optimizer for every trainer, so each states the step the first traced (`Trainer._step_key`).
    sgd = optax.sgd(1e-2)
    model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=SEQ)

    def traced() -> int:
        trainer = one_device(Trainer(LMObjective(model, seq_len=SEQ), sgd, key=jax.random.key(0)))
        state, _, _ = trainer.place()
        with Compiles("step") as compiles:
            trainer.compile(state, BATCH)
        return [program.kind for program in compiles.programs].count("traced")

    first = traced()
    kept = traced()
    options.release()
    again = traced()
    assert kept < first and kept < again, (first, kept, again)
