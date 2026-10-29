"""A step's plan: the fewest microbatches at the least recomputation that fit.

Splitting a step's batch must be invisible to everything except memory use:
the update pools the same rows, the clocks count the same steps, and a
resumed run reads its batches as the run it continues did.
"""

import contextlib
import dataclasses
import functools
import gc
import logging
import struct
import weakref
from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import FEATURES, Regression
from jax.extend.backend import get_backend
from reference_error import as_given, assert_as_exact, seeds_for
from sharded import collectives
from token_support import write_token_corpus

from dew import cache
from dew.checkpoints import Checkpoints
from dew.data import Loading, presets
from dew.data.compile import compile
from dew.data.sources.tokens import TokenFiles
from dew.diffusion.discrete import MDLM
from dew.distributed.placement import Layout, MeshSpec, placed_bytes
from dew.mesh import LayoutRefused
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import REMAT_POLICIES
from dew.nn.kernels.generation import chosen
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.objectives.lm import LMObjective
from dew.objectives.lm.chunked import chunked_tile
from dew.registry import mixers
from dew.state import TrainState, apart
from dew.training import Trainer
from dew.training.distributed import microbatch
from dew.training.memory import climb_to, recompute_more, rung_record
from dew.training.planning import TIMED_REPEATS, UNTIMED_FRONTIER, Trials, planned, splits

ROWS, SEQ = 32, 16
BATCH = {"text": jnp.asarray((np.arange(ROWS * (SEQ + 1)).reshape(ROWS, SEQ + 1) * 7) % 64, jnp.int32)}


@pytest.mark.parametrize("fitting, first, plan, asked", [
    # The batch whole fits: one compile.
    (lambda rung, split: True, 0, (0, 1), [(0, 1)]),
    # Four microbatches fit with no recomputation: the largest split, then halving.
    (lambda rung, split: split >= 4, 0, (0, 4), [(0, 1), (0, 8), (0, 2), (0, 4)]),
    # No split fits without recomputing: one compile at that rung past the
    # first, then the next rung's fewest microbatches.
    (lambda rung, split: rung >= 1 and split >= 2, 0, (1, 2), [(0, 1), (0, 8), (1, 8), (1, 2), (1, 1)]),
    # A recorded plan is asked first and kept where it fits.
    (lambda rung, split: split >= 2, 2, (0, 4), [(0, 4)]),
    # Nothing fits: the largest split at every rung, then no plan.
    (lambda rung, split: False, 0, None, [(0, 1), (0, 8), (1, 8), (2, 8)]),
])
def test_the_plan_takes_the_least_recomputation_then_the_fewest_microbatches(fitting, first, plan, asked):
    """Candidates are (rung, split), rung-major; memory falls along both, so
    the search asks the first split, each rung's largest, and halves the
    range at the first rung where that fits."""
    rung, compiled = [0], []

    def fits(split):
        compiled.append((rung[0], split))
        return fitting(rung[0], split)

    def climb():
        rung[0] += 1
        return rung[0] < 3

    split = planned([1, 2, 4, 8], first, fits, climb)
    assert (None if split is None else (rung[0], split)) == plan
    assert compiled == asked


@pytest.mark.parametrize("likely, asked", [
    # Past a miss, the first split likely to fit is asked in place of the largest, and the range below
    # it halved as ever: the guess orders the compiles.
    (lambda split: split >= 4, [1, 4, 2]),
    # A guess that misses leads to the next likely split above it, or the largest.
    (lambda split: split == 2, [1, 2, 8, 4]),
])
def test_a_likely_split_is_asked_first_and_the_plan_is_the_same(likely, asked):
    """Four microbatches fit, two do not: the plan is four whatever the guess."""
    compiled = []

    def fits(split):
        compiled.append(split)
        return split >= 4

    assert planned([1, 2, 4, 8], 0, fits, lambda: False, likely) == 4
    assert compiled == asked


def test_a_forecast_guesses_from_every_processs_compiles_above_a_split():
    """One compile above a split: its temporaries taken as all per row; two: a line in the rows. A guess
    is every process's least; a process that read no memory is no guess, and a split with no compile
    above it is guessed nothing, not likely."""
    from dew.training.planning import Forecast

    # Two processes; the second's device holds 10 bytes less.
    forecast = Forecast(gather=lambda local: [local, None if local is None else (local[0], local[1] - 10,
                                                                                  local[2])])
    forecast.read("rung", 1, None, None, 1)
    assert forecast.guess("rung", 2) is None and not forecast.likely("rung", 2)
    forecast.measured.clear()
    # The batch whole needs 90 bytes of temporaries and misses by 50.
    forecast.measured["rung"] = [(1, [(90, -50, 1), (90, -60, 1)])]
    assert forecast.guess("rung", 2) == -60 + 45 and not forecast.likely("rung", 2)
    assert forecast.guess("rung", 4) == -60 + 90 - 22 and forecast.likely("rung", 4)
    forecast.measured["rung"].append((2, [(50, -10, 1), None]))
    # The first process's two compiles, as a line in 1/split, give 10 at four; the second's one, 8.
    assert forecast.guess("rung", 4) == min(-10 + 40 * (1 / 4 - 1 / 2) / (1 / 2 - 1), 8)
    # A miss at the rung moved to its half rung, whose compiles count too.
    forecast.half_of["rung"] = "half"
    forecast.measured["half"] = [(1, [(20, 80, 1), (20, 70, 1)])]
    assert forecast.likely("rung", 2)


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
        Trainer(LMObjective(model, seq_len=SEQ), optax.sgd(1e-2), key=jax.random.key(0), split=2, mesh=spec)
    trainer = Trainer(LMObjective(model, seq_len=SEQ), optax.sgd(1e-2), key=jax.random.key(0), mesh=spec)
    assert splits(None, BATCH, MeshSpec().build(jax.devices()[:1]), 1, trainer._split) == [1]


def lm(split=None, accumulation=1, **fields):
    model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=SEQ, **fields)
    return one_device(Trainer(LMObjective(model, seq_len=SEQ), optax.sgd(1e-2), key=jax.random.key(0),
                              split=split, accumulation=accumulation))


def trained(trainer, steps=3):
    """The state after `steps` steps of BATCH."""
    state, _, _ = trainer.place()
    step = trainer.compile(state, BATCH)
    for _ in range(steps):
        state, *_ = step(state, BATCH)
    return state


LIMIT = 2 ** 40
"""The bytes every device reports it holds: one size, which a step's rung record names, whatever is free."""


def free(monkeypatch, room):
    """Every device reports `room` bytes of its LIMIT free (`Device.memory_stats`), the one reading of
    the devices the fit check takes; what a step needs is XLA's memory analysis of the step."""
    monkeypatch.setattr(jax.Device, "memory_stats",
                        lambda device: {"bytes_limit": LIMIT, "bytes_in_use": LIMIT - room})


def needs(monkeypatch, trainer, run=None):
    """The bytes the step `trainer` compiles first needs beside what a device holds: the headroom that
    compile left of all of LIMIT free, taken from LIMIT. `run` compiles it; None compiles it for BATCH."""
    free(monkeypatch, LIMIT)
    if run is None:
        trained(trainer, steps=0)
    else:
        run()
    return LIMIT - trainer.plan["probes"][0]["headroom"]


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
    first, *_ = trainer.plan["probes"]
    assert (first["split"], first["fits"]) == (1, False)
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
    rung = trainer._checkpoint_rung()
    assert [shaped["head_fused"] for shaped in rung["shapes"].values()] == [False, False], rung
    assert rung["head_tile"] is not None and not rung["head_fused"]


def test_a_later_run_starts_its_search_at_the_plan_an_earlier_one_recorded(monkeypatch):
    """The plan is recorded beside the compilation cache, read before the step
    traces, so a restart compiles the recorded candidate first, from the
    cache, and keeps it, with the FLOPs the first run counted for the same
    executable."""
    whole, halves = needs(monkeypatch, lm(split=1)), needs(monkeypatch, lm(split=2))
    free(monkeypatch, (whole + halves) // 2)
    first = lm()
    trained(first, steps=0)
    free(monkeypatch, LIMIT)
    again = lm()
    counted = []
    as_text = jax.stages.Compiled.as_text

    def read_text(executable, *args, **kwargs):
        counted.append(executable)
        return as_text(executable, *args, **kwargs)

    monkeypatch.setattr(jax.stages.Compiled, "as_text", read_text)
    trained(again, steps=0)
    assert [probe["split"] for probe in again.plan["probes"]] == [first.plan["split"]]
    # The executable is the one the first run counted, so its FLOPs are the record's, its text unread.
    assert counted == [] and again.step_flops == first.step_flops


@pytest.mark.gpu("a step's Pallas launches are read from a lowering of their own on a CUDA GPU alone")
def test_a_recorded_plan_settles_its_kernel_launches_without_lowering_the_step_for_them():
    """A step's lowering records its kernel launches as it traces them (`tune`), with its plan,
    so each run lowers its step once: the first again only where settling the launches changed
    a config its trace read, and a later run, which settles them from the record first, never."""
    lowered = []

    def lowering(event, seconds, *, fun_name=None, **_):
        if event == "/jax/core/compile/jaxpr_to_mlir_module_duration" and fun_name == "jit(step)":
            lowered.append(seconds)

    counts = []
    settled = cache.KERNEL_TUNES.get_global()
    jax.monitoring.register_event_duration_secs_listener(lowering)
    try:
        for _ in range(2):
            trained(lm(), steps=0)
            counts.append(len(lowered))
            lowered.clear()
            changed, settled = cache.KERNEL_TUNES.get_global() != settled, cache.KERNEL_TUNES.get_global()
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
class Timed(Trials):
    """Trials whose frontier takes the `given` seconds, each frontier asked recorded in `asked`: the
    frontier's choice at timings the clock would give by chance."""
    given: list[list[float]] = dataclasses.field(default_factory=list)
    asked: list[int] = dataclasses.field(default_factory=list)

    def seconds(self, steps):
        self.asked.append(len(steps))
        return self.given


@dataclasses.dataclass
class Handed(Trials):
    """Trials that profile the plan admitted, keep a weak reference to the executable it was admitted with
    (`unguided`) and count the executables the backend holds then, it among them (`live`), and hand back
    `profile` where given in place of the profile they measured."""
    profile: bytes | None = None
    unguided: weakref.ref | None = None
    live: int = 0

    def profiled(self, step, planned, mesh, runs):
        measured = super().profiled(step, planned, mesh, runs)
        self.unguided = weakref.ref(planned.executable)
        self.live = live_executables()
        return measured if self.profile is None else self.profile


def live_executables() -> int:
    """The executables the backend holds once the garbage they are part of is collected."""
    gc.collect()
    return len(get_backend().live_executables())


COST = b"\x0a\x0c\x0a\x01x\x11" + struct.pack("<d", 1.0)
"""A profile of one instruction, `x`, at a microsecond, as XLA's ProfiledInstructionsProto writes it: what
an NVIDIA GPU's profile holds, which a CPU's compile takes and does not read."""


def stepped(trainer, steps: int = 3, trials: Callable[[TrainState], Trials] | None = None):
    """The state after `steps` steps of BATCH from the trainer's placed state, its step compiled on the
    `trials` of that state, from which it steps on, or on none, stepping that state itself; and the
    trials."""
    state, _, _ = trainer.place()
    held = None if trials is None else trials(state)
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
    AutoPGLE does; and under `fit`, whose untimed frontier (Dew's tests' setting) keeps its first plan
    and profiles it all the same. XLA's profile of a CPU holds nothing, so each plan keeps the executable
    it admitted."""
    unguided, _ = stepped(lm(split=split))
    with autopgle(2):
        trainer, fitted = lm(split=split), windowed(tmp_path, split)
        guided, _ = stepped(trainer)
        fitted.fit(windows(token_files(tmp_path)), steps=1, checkpoint_every=None)
    assert trainer.plan["pgle"] == fitted.plan["pgle"] == {"runs": 2, "bytes": 0}
    assert int(guided.step) == 3
    for guided_leaf, unguided_leaf in zip(jax.tree.leaves(guided.variables),
                                          jax.tree.leaves(unguided.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(guided_leaf), np.asarray(unguided_leaf))


def test_the_executable_a_profile_replaces_goes_with_its_program():
    """Under AutoPGLE a profile of the admitted step's runs compiles its program again, and the run keeps
    the guided executable alone: once the compile returns, the one the plan was admitted with is freed
    and the backend holds as many executables as it did with that one alive, though JAX keeps every
    executable it compiled of a program while the program's lowering lives. A handoff that kept the
    admitted executable, or kept both, would hold one more."""
    with autopgle(2):
        trainer = lm()
        state, trials = stepped(trainer, trials=functools.partial(Handed, batch=BATCH, profile=COST))
    assert trainer.plan["pgle"] == {"runs": 2, "bytes": len(COST)}
    assert trials.unguided is not None and trials.unguided() is None
    assert live_executables() == trials.live and int(state.step) == 3


def test_a_profile_window_traces_steps_past_the_first_compile_and_is_refused_under_autopgle(tmp_path):
    """A fit's window opens once its warmup steps have run and its first step has compiled, so a window of
    no warmup traces the three steps after the first, never the compile. Under JAX's AutoPGLE, which
    profiles each program's first runs in a profiling session of its own where a process holds one at a
    time, a fit with a window is refused before it trains, naming JAX's way to capture guided programs
    (`PGLE_CAPTURE`): JAX's sessions would fail inside the window, as the trainer's would."""
    pytest.importorskip("xprof.convert.raw_to_tool_data", reason="optional profile extra")
    from affine_run import Replaying
    from recording import RecordingTracker

    from dew.data.dataset import Loader
    from dew.training import ProfileWindow
    from dew.training.profiling import traced_steps

    tracker = RecordingTracker()

    def fit(directory):
        model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                                  max_seq_len=SEQ)
        window = ProfileWindow(str(directory), steps=3, warmup=0)
        trainer = one_device(Trainer(LMObjective(model, seq_len=SEQ), optax.sgd(1e-2), key=jax.random.key(0),
                                     tracker=tracker, profile=window))
        rows = Replaying(lambda position: {"text": np.asarray(BATCH["text"])})
        trainer.fit(Loader(lambda partition: rows, None, None, ROWS), steps=5, log_every=1)

    fit(tmp_path / "window")
    (window,) = [value for _, value in tracker.artifacts if type(value).__name__ == "ProfileWindow"]
    (capture,) = (tmp_path / "window").glob("capture-*")
    assert window.steps == 3 and traced_steps(capture) == (1, 2, 3)
    with autopgle(2), pytest.raises(ValueError):
        fit(tmp_path / "pgle")
    assert not (tmp_path / "pgle").exists()


def test_a_borrowed_compile_under_autopgle_profiles_beside_the_state_only_where_the_devices_hold_both(
        monkeypatch):
    """Under AutoPGLE a compile given no trials profiles its step on a copy of the state beside the
    caller's, which it never steps. With room for the step and its state alone, it is refused before it
    copies anything, where trials that own the state (`fit`'s), which step it and restore it, profile at
    that room; with room for the copy too, it profiles and leaves the devices holding the caller's state
    alone, the copy freed."""
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
        assert owned.plan["pgle"] == {"runs": 2, "bytes": 0}
        free(monkeypatch, need + 2 * copy)
        borrowed = lm()
        state, _, _ = borrowed.place()
        held = copies(state)
        borrowed.compile(state, BATCH)
        assert borrowed.plan["pgle"] == {"runs": 2, "bytes": 0}
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


def regression_rows(seed: int) -> tuple[np.ndarray]:
    """Four steps' rows of one draw: 32 rows of FEATURES coordinates from a unit normal."""
    return (np.random.default_rng(seed).normal(size=(32, FEATURES)).astype(np.float32),)


@pytest.mark.gpu(reason="XLA profiles the instructions of an NVIDIA GPU's programs alone")
def test_a_step_guided_by_its_profile_is_the_step_the_run_takes_and_trains_as_exactly():
    """On an NVIDIA GPU the profile of the admitted step's runs holds its instructions' times, its
    program compiles again guided by it, and the run steps the guided executable alone: the one the plan
    was admitted with is freed and the backend holds no more executables than with it alive, as it would
    not where the handoff is discarded. The guided steps' losses and weights are as exact as the unguided
    step's (check 1)."""
    with autopgle(2):
        trainer = lm()
        _, trials = stepped(trainer, trials=functools.partial(Handed, batch=BATCH))
    assert trainer.plan["pgle"]["runs"] == 2 and trainer.plan["pgle"]["bytes"] > 0
    assert trials.unguided is not None and trials.unguided() is None and live_executables() == trials.live
    guided = as_given(functools.partial(regression_steps, pgle=True))
    unguided = as_given(functools.partial(regression_steps, pgle=False))
    label = "a step guided by its profile"
    assert_as_exact(guided, unguided, regression_rows, label,
                    seeds=seeds_for(unguided, regression_rows, label, dew_fn=guided))


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


def windowed(directory, split):
    model = CausalTransformer(vocab_size=32, emb_features=8, num_layers=2, num_heads=1, mlp_features=16,
                              max_seq_len=8)
    return one_device(Trainer(LMObjective(model, seq_len=4), optax.adam(1e-2), key=jax.random.key(0),
                              split=split, layout=Layout(min_shard=1, tolerance=1.0),
                              checkpoints=Checkpoints(str(directory / "run"))))


def test_a_resumed_run_reads_its_batches_in_the_microbatches_its_checkpoint_trained_on(tmp_path):
    """The split is part of what a run resumes: the state's window holds
    that many microbatches. A resumed run that leaves it open takes the
    checkpoint's and trains bit for bit as the uninterrupted run; one that
    pins another is refused."""
    corpus = token_files(tmp_path)
    expected = windowed(tmp_path / "whole", 2).fit(windows(corpus), steps=4, checkpoint_every=2)
    windowed(tmp_path / "split", 2).fit(windows(corpus), steps=2, checkpoint_every=2)
    resumed = windowed(tmp_path / "split", None)
    actual = resumed.fit(windows(corpus), steps=4, checkpoint_every=2)
    assert resumed.plan["split"] == 2
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert raw(left).tobytes() == raw(right).tobytes()
    with pytest.raises(ValueError):
        windowed(tmp_path / "split", 4).fit(windows(corpus), steps=6, checkpoint_every=2)


@pytest.mark.mesh(devices=2)
def test_a_resumed_run_continues_the_plan_its_frontier_chose_untimed(tmp_path, monkeypatch):
    """A run over two devices times its frontier, the batch whole as it fits and a rung up, and keeps one;
    a run resumed from its checkpoint continues that plan without timing again, as the run it continues
    compiled it, so the two train one program whatever a second timing would say."""
    # The frontier timed, as the library times it: Dew's tests keep its first plan otherwise.
    monkeypatch.delenv(UNTIMED_FRONTIER)
    corpus = token_files(tmp_path)

    def run(split):
        trainer = windowed(tmp_path / "frontier", split)
        trainer.device_mesh = MeshSpec().build(jax.devices()[:2])
        return trainer

    first = run(1)
    first.fit(windows(corpus), steps=2, checkpoint_every=2)
    assert [timing["remat"] for timing in first.plan["timed"]] == [None, "minimal"]
    resumed = run(None)
    resumed.fit(windows(corpus), steps=4, checkpoint_every=2)
    assert resumed.plan["timed"] is None
    assert (resumed.plan["split"], resumed.plan["remat"]) == (first.plan["split"], first.plan["remat"])


@pytest.mark.parametrize("seconds, split, remat", [
    ([[0.2, 0.21], [0.1, 0.12]], 1, "minimal"), ([[0.1, 0.12], [0.2, 0.21]], 2, None),
    ([[0.2, 0.3], [0.1, 0.25]], 2, None)], ids=["whole", "split", "within-spread"])
def test_a_split_plan_is_timed_against_its_batch_whole_a_rung_up_and_the_faster_kept(
        monkeypatch, seconds, split, remat):
    """Where the devices hold a byte less than the batch whole needs at the
    least recomputation, it fits split there, and whole one remat rung up with
    the head tiled: both are compiled and timed, and the plan is the whole one
    only where its every sample beats the split one's (`faster`), the split one
    where their samples overlap, the objective at its rung, each plan's fastest
    sample recorded."""
    free(monkeypatch, needs(monkeypatch, lm(split=1)) - 1)
    trainer = lm()
    state, _, _ = trainer.place()
    trials = Timed(state, BATCH, given=seconds)
    step = trainer.compile(state, BATCH, trials)
    assert trials.asked == [2]
    assert (trainer.plan["split"], trainer.plan["remat"]) == (split, remat)
    assert trainer.objective.model.remat == (None if remat is None else REMAT_POLICIES[remat])
    assert [(plan["split"], plan["remat"], plan["seconds"]) for plan in trainer.plan["timed"]] == [
        (2, None, min(seconds[0])), (1, "minimal", min(seconds[1]))]
    state, loss, *_ = step(state, BATCH)
    assert np.isfinite(float(loss))


@pytest.mark.mesh(devices=2)
def test_a_plan_over_several_devices_that_fits_whole_is_timed_against_its_batch_whole_a_rung_up(monkeypatch):
    """Over two devices a plan that fits the batch whole without recomputing is timed against the batch
    whole one rung up too, whose freed memory lets the scheduler overlap the collectives, and the faster
    is kept; on one device the plan is never timed."""
    free(monkeypatch, LIMIT)
    trainer = lm()
    trainer.device_mesh = MeshSpec().build(jax.devices()[:2])
    state, _, _ = trainer.place()
    trials = Timed(state, BATCH, given=[[2.0], [1.0]])
    trainer.compile(state, BATCH, trials)
    assert trials.asked == [2]
    timings = [(plan["split"], plan["remat"]) for plan in trainer.plan["timed"]]
    assert timings == [(1, None), (1, "minimal")]
    assert (trainer.plan["split"], trainer.plan["remat"]) == (1, "minimal")
    alone = lm()
    state, _, _ = alone.place()
    alone_trials = Timed(state, BATCH)
    alone.compile(state, BATCH, alone_trials)
    assert alone_trials.asked == [] and alone.plan["timed"] is None


@pytest.mark.parametrize("split", [1, 2])
def test_a_plan_with_no_second_point_to_fit_is_not_timed(monkeypatch, split):
    """With room for the batch whole, or for two microbatches and not the
    batch whole a rung up, there is one plan, and nothing is timed; the
    frontier's compile that did not fit is in the plan's record."""
    free(monkeypatch, needs(monkeypatch, lm(split=split)))
    trainer = lm()
    state, _, _ = trainer.place()
    trials = Timed(state, BATCH)
    trainer.compile(state, BATCH, trials)
    assert trials.asked == []
    assert trainer.plan["timed"] is None
    assert trainer.objective.model.remat is None
    last = trainer.plan["probes"][-1]
    assert (last["split"], last["remat"], last["fits"]) == ((1, None, True) if trainer.plan["split"] == 1
                                                            else (1, "minimal", False))


def asked(trainer) -> list[tuple[int, str | None, bool]]:
    """The compiles a plan's search made, as (split, remat, fits)."""
    return [(probe["split"], probe["remat"], probe["fits"]) for probe in trainer.plan["probes"]]


def assert_searched(probes, chosen, below, frontier, *, half=None):
    """`probes`, a search that settled on `chosen` (split, remat): the batch whole asked first, no
    compile asked twice, every split that missed at no recomputation asked at the `half` rung next,
    `below`, the largest split under the plan, asked and missed at each rung it may fit at, a plan at
    the half rung missed at no recomputation before, and the `frontier` asked last. Which splits
    between are asked, and in what order, follows the memory each compile measured, which differs
    between devices (`Forecast`)."""
    assert probes[0][:2] == (1, None) and not probes[0][2]
    assert len({probe[:2] for probe in probes}) == len(probes)
    assert (*chosen, True) in probes and probes[-1] == (*frontier, False)
    for index, (split, remat, fits) in enumerate(probes[:-1]):
        if half is not None and remat is None and not fits:
            assert probes[index + 1] == (split, half, probes[index + 1][2])
    rungs = [None] if half is None else [None, half] if chosen[1] is None else [half]
    assert all((below, remat, False) in probes for remat in rungs)
    if half is not None and chosen[1] == half:
        # A plan at the half rung missed at no recomputation first.
        assert probes.index((chosen[0], None, False)) < probes.index((*chosen, True))


DELTA = {"mixer": mixers.build("gated_delta_net", linear_num_key_heads=2, linear_num_value_heads=2,
                               linear_key_head_dim=8, linear_value_head_dim=8)}
"""A decoder of one gated delta net layer, whose ladder holds the half rung 'all_but_delta_memory'."""


@pytest.mark.gpu("the GPU kernel's delta rule recomputes its memory alone; on the CPU the half rung "
                 "needs what no recomputation does")
def test_a_split_that_misses_tries_the_half_rung_before_the_batch_splits_further(monkeypatch):
    """With room for four microbatches when a delta rule recomputes its memory alone, which no
    recomputation needs more than, the plan is four at that half rung: no plan splits the batch further
    at a rung where the half rung fits it, and the search asks each split there after it misses. The
    delta rule runs its port (`chosen`), whatever the generation's row names: the half rung recomputes
    the memory the port names (`checkpoint_name`), and the portable route, an A10's, names none."""
    half = "all_but_delta_memory"
    with chosen(gated_delta_rule="fla"):
        free(monkeypatch, needs(monkeypatch, lm(split=4, remat=half, **DELTA)))
        trainer = lm(**DELTA)
        state, _, _ = trainer.place()
        trials = Timed(state, BATCH)
        trainer.compile(state, BATCH, trials)
        assert trials.asked == []
    assert (trainer.plan["split"], trainer.plan["remat"]) == (4, half)
    assert trainer.objective.model.remat == REMAT_POLICIES[half]
    assert_searched(asked(trainer), (4, half), 2, (1, "minimal_with_scan_and_conv"), half=half)


def test_a_plan_at_no_recomputation_times_its_frontier_past_the_half_rung(monkeypatch):
    """An A100's plan: no recomputation fits the batch at four microbatches and the half rung never
    fits it in fewer, so the plan is four at no recomputation, and the frontier, the batch whole one
    rung up, is asked at the first rung past the half rung, whose every split asked missed."""
    free(monkeypatch, needs(monkeypatch, lm(split=4, **DELTA)))
    trainer = lm(**DELTA)
    state, _, _ = trainer.place()
    trials = Timed(state, BATCH)
    trainer.compile(state, BATCH, trials)
    assert trials.asked == []
    assert (trainer.plan["split"], trainer.plan["remat"]) == (4, None)
    assert_searched(asked(trainer), (4, None), 2, (1, "minimal_with_scan_and_conv"),
                    half="all_but_delta_memory")


def test_a_decoder_without_a_half_rung_asks_its_splits_at_no_recomputation(monkeypatch):
    """Without a delta rule no rung is a half rung: the search asks the batch whole and the splits
    that find the fewest microbatches that fit, each at no recomputation, and the frontier one rung
    up."""
    free(monkeypatch, needs(monkeypatch, lm(split=4)))
    trainer = lm()
    state, _, _ = trainer.place()
    trials = Timed(state, BATCH)
    trainer.compile(state, BATCH, trials)
    assert trials.asked == []
    assert_searched(asked(trainer), (4, None), 2, (1, "minimal"))
    assert all(remat is None for _, remat, _ in asked(trainer)[:-1])


@pytest.mark.parametrize("owned", [True, False], ids=["owned", "borrowed"])
def test_trials_step_a_copy_of_the_state_and_hand_back_a_fresh_one(owned):
    """A frontier's timing runs each step on a trial of the state as it stood: trials that own the state
    (`fit`'s) step it and delete it, handing back a copy as it was; trials that borrow it (a compile given
    none) step a copy beside it and hand back the state itself, as it is. Either way the run trains on
    from where it stood."""
    trainer = lm(split=1)
    state, _, _ = trainer.place()
    step = trainer.compile(state, BATCH)
    before = [raw(leaf).copy() for leaf in jax.tree.leaves(state)]
    trials = Trials(state, BATCH, owned=owned)
    seconds = trials.seconds([step, step])
    assert len(seconds) == 2
    assert all(len(samples) == TIMED_REPEATS and min(samples) > 0 for samples in seconds)
    assert all(leaf.is_deleted() == owned for leaf in jax.tree.leaves(state))
    assert all(np.array_equal(raw(left), right)
               for left, right in zip(jax.tree.leaves(trials.state), before, strict=True))


def test_a_run_trains_on_from_the_state_its_plan_timed(monkeypatch, tmp_path):
    """Under `fit`, a frontier is timed on the run's own state, and the run
    trains on from a fresh copy of it: the timed steps count for nothing."""
    # The frontier timed, as the library times it: Dew's tests keep its first plan otherwise.
    monkeypatch.delenv(UNTIMED_FRONTIER)
    corpus, whole = token_files(tmp_path), windowed(tmp_path / "whole", 1)
    free(monkeypatch, needs(monkeypatch, whole, lambda: whole.fit(windows(corpus), steps=1,
                                                                  checkpoint_every=None)) - 1)
    trainer = windowed(tmp_path / "timed", None)
    state = trainer.fit(windows(corpus), steps=3, checkpoint_every=None)
    assert [plan["seconds"] > 0 for plan in trainer.plan["timed"]] == [True, True]
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
    assert keys[0] == keys[1]


def test_a_frontier_keeps_its_first_plan_unless_another_is_faster_beyond_the_spread():
    """`faster` takes a later plan only where its slowest sample beats every sample of the plan it would
    replace: timings that overlap are the clock's noise, and keep the first."""
    from dew.training.planning import faster

    assert faster([[2.0, 2.1], [1.0, 1.1]]) == 1
    assert faster([[1.0, 1.1], [2.0, 2.1]]) == 0
    assert faster([[1.0, 2.0], [0.9, 1.5]]) == 0
    assert faster([[3.0], [2.0], [1.0]]) == 2


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

    class Program:
        def compile(self, options):
            raise error

    refusals = []
    try:
        outcome = compiled_if_it_fits(Program(), None, refusals)
    except jax.errors.JaxRuntimeError as raised:
        outcome = raised
    assert (outcome, refusals) == ((None, [error]) if refused else (error, []))


def test_a_run_that_sets_its_remat_compiles_its_step_once(monkeypatch, caplog, tmp_path):
    """A run that sets remat 'full', with room for its step at every cheaper rung too, compiles its
    training step once, at its own rung, and trains there: its plan holds that one compile and no
    other rung's (row 2102). Each compile is the whole step's, which on a 4B's first step took most of
    30 minutes where a run compiled the rungs below its own to advise it of them. JAX states each
    compile it makes (`jax._src.dispatch`, "Finished XLA compilation of jit(step)")."""
    free(monkeypatch, LIMIT)
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path))
    trainer = lm(split=1, remat=REMAT_POLICIES["full"])
    with caplog.at_level(logging.DEBUG, logger="jax._src.dispatch"):
        trained(trainer, steps=0)
    steps = [record for record in caplog.records
             if record.getMessage().startswith("Finished XLA compilation of jit(step)")]
    assert len(steps) == 1
    assert [probe["remat"] for probe in trainer.plan["probes"]] == ["full"]
    assert trainer.plan["remat"] == "full" and trainer.objective.model.remat == REMAT_POLICIES["full"]
