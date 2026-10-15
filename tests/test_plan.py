"""A step's plan: the fewest microbatches at the least recomputation that fit.

Splitting a step's batch must be invisible to everything except memory use:
the update pools the same rows, the clocks count the same steps, and a
resumed run reads its batches as the run it continues did.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.checkpoints import Checkpoints
from dew.config.data import TextData, loader
from dew.data import Loading
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import REMAT_POLICIES
from dew.nn.sharding import LayoutRefused
from dew.objectives.lm import LMObjective
from dew.registry import mixers
from dew.training import Trainer, rungs
from dew.training.distributed import Layout, MeshSpec, microbatch
from dew.training.planning import Timer, planned, splits

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
    text = sliced.lower(batch, jax.device_put(jnp.int32(1), replicated)).compile().as_text()
    assert not re.search(r"all-gather|all-to-all|collective-permute|all-reduce", text), text


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
    monkeypatch.setattr(rungs, "step_flops", lambda executable: counted.append(executable))
    trained(again, steps=0)
    assert [probe["split"] for probe in again.plan["probes"]] == [first.plan["split"]]
    # The executable is the one the first run counted, so its FLOPs are the record's, its text unread.
    assert counted == [] and again.step_flops == first.step_flops


@pytest.mark.gpu("a step's Pallas launches are read from a lowering of their own on a CUDA GPU alone")
def test_a_recorded_plan_settles_its_kernel_launches_without_lowering_the_step_for_them():
    """A step's kernel launches are read from a lowering of its own (`tune`) and recorded with
    its plan, so a later run settles their configs from the record and lowers the step once."""
    lowered = []

    def lowering(event, seconds, *, fun_name=None, **_):
        if event == "/jax/core/compile/jaxpr_to_mlir_module_duration" and fun_name == "jit(step)":
            lowered.append(seconds)

    counts = []
    jax.monitoring.register_event_duration_secs_listener(lowering)
    try:
        for _ in range(2):
            trained(lm(), steps=0)
            counts.append(len(lowered))
            lowered.clear()
    finally:
        jax.monitoring.unregister_event_duration_listener(lowering)
    assert counts == [2, 1]


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
    """Token files written once, so a resume reads the files its checkpoint names."""
    tokens = (np.arange(97, dtype=np.uint16) * 7) % 32
    tokens.tofile(directory / "train.bin")
    tokens.tofile(directory / "val.bin")
    return directory


def windows(corpus):
    return loader(TextData(source=str(corpus), seq_len=4, stride=1), batch=8, read="memory", seed=7,
                  loading=Loading(workers=0, threads=1, read_buffer=1), validation_records=4 * 8)


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


@pytest.mark.parametrize("seconds, split, remat", [([0.2, 0.1], 1, "minimal"), ([0.1, 0.2], 2, None)])
def test_a_split_plan_is_timed_against_its_batch_whole_a_rung_up_and_the_faster_kept(
        monkeypatch, seconds, split, remat):
    """Where the devices hold a byte less than the batch whole needs at the
    least recomputation, it fits split there, and whole one remat rung up with
    the head tiled: both are compiled and timed, and the faster is the plan,
    the objective at its rung, both times recorded."""
    free(monkeypatch, needs(monkeypatch, lm(split=1)) - 1)
    trainer = lm()
    state, _, _ = trainer.place()
    timed = []
    step = trainer.compile(state, BATCH, lambda steps: (timed.append(len(steps)), seconds)[1])
    assert timed == [2]
    assert (trainer.plan["split"], trainer.plan["remat"]) == (split, remat)
    assert trainer.objective.model.remat == (None if remat is None else REMAT_POLICIES[remat])
    assert [(plan["split"], plan["remat"], plan["seconds"]) for plan in trainer.plan["timed"]] == [
        (2, None, seconds[0]), (1, "minimal", seconds[1])]
    state, loss, *_ = step(state, BATCH)
    assert np.isfinite(float(loss))


@pytest.mark.parametrize("split", [1, 2])
def test_a_plan_with_no_second_point_to_fit_is_not_timed(monkeypatch, split):
    """With room for the batch whole, or for two microbatches and not the
    batch whole a rung up, there is one plan, and nothing is timed; the
    frontier's compile that did not fit is in the plan's record."""
    free(monkeypatch, needs(monkeypatch, lm(split=split)))
    trainer = lm()
    state, _, _ = trainer.place()
    trainer.compile(state, BATCH, lambda steps: pytest.fail("a single plan was timed"))
    assert trainer.plan["timed"] is None
    assert trainer.objective.model.remat is None
    last = trainer.plan["probes"][-1]
    assert (last["split"], last["remat"], last["fits"]) == ((1, None, True) if trainer.plan["split"] == 1
                                                            else (1, "minimal", False))


DELTA = {"mixer": mixers.build("gated_delta_net", linear_num_key_heads=2, linear_num_value_heads=2,
                               linear_key_head_dim=8, linear_value_head_dim=8)}
"""A decoder of one gated delta net layer, whose ladder holds the half rung 'all_but_delta_memory'."""


@pytest.mark.gpu("the GPU kernel's delta rule recomputes its memory alone; on the CPU the half rung "
                 "needs what no recomputation does")
def test_a_split_that_misses_tries_the_half_rung_before_the_batch_splits_further(monkeypatch):
    """With room for four microbatches when a delta rule recomputes its memory alone, which no
    recomputation needs more than, the plan is four at that half rung: no plan splits the batch further
    at a rung where the half rung fits it, and the search asks each split there after it misses."""
    half = "all_but_delta_memory"
    free(monkeypatch, needs(monkeypatch, lm(split=4, remat=half, **DELTA)))
    trainer = lm(**DELTA)
    state, _, _ = trainer.place()
    trainer.compile(state, BATCH, lambda steps: pytest.fail("the plan at its first split was timed"))
    assert (trainer.plan["split"], trainer.plan["remat"]) == (4, half)
    assert trainer.objective.model.remat == REMAT_POLICIES[half]
    assert [(probe["split"], probe["remat"], probe["fits"]) for probe in trainer.plan["probes"]] == [
        (1, None, False), (1, "all_but_delta_memory", False), (32, None, True), (4, None, False),
        (4, "all_but_delta_memory", True), (2, "all_but_delta_memory", False),
        (1, "minimal_with_conv", False)]


def test_a_plan_at_no_recomputation_times_its_frontier_past_the_half_rung(monkeypatch):
    """An A100's plan: no recomputation fits the batch at four microbatches and the half rung never
    fits it in fewer, so the plan is four at no recomputation, and the frontier, the batch whole one
    rung up, is asked at the first rung past the half rung, whose every split asked missed."""
    free(monkeypatch, needs(monkeypatch, lm(split=4, **DELTA)))
    trainer = lm(**DELTA)
    state, _, _ = trainer.place()
    trainer.compile(state, BATCH, lambda steps: pytest.fail("a single plan was timed"))
    assert (trainer.plan["split"], trainer.plan["remat"]) == (4, None)
    assert [(probe["split"], probe["remat"], probe["fits"]) for probe in trainer.plan["probes"]] == [
        (1, None, False), (1, "all_but_delta_memory", False), (32, None, True), (4, None, True),
        (2, None, False), (2, "all_but_delta_memory", False), (1, "minimal_with_conv", False)]


def test_a_decoder_without_a_half_rung_asks_the_splits_it_did(monkeypatch):
    """Without a delta rule no rung is a half rung: the search asks the batch whole, the largest split,
    and halves the range, each at no recomputation, and the frontier one rung up."""
    free(monkeypatch, needs(monkeypatch, lm(split=4)))
    trainer = lm()
    state, _, _ = trainer.place()
    trainer.compile(state, BATCH, lambda steps: pytest.fail("a single plan was timed"))
    assert [(probe["split"], probe["remat"], probe["fits"]) for probe in trainer.plan["probes"]] == [
        (1, None, False), (32, None, True), (4, None, True), (2, None, False), (1, "minimal", False)]


def test_the_timer_steps_the_state_it_was_given_and_hands_back_a_fresh_copy():
    """Timing a frontier steps the state it was given, deletes it, and hands
    the run a copy of that state as it was, so the run trains on from where
    it stood."""
    trainer = lm(split=1)
    state, _, _ = trainer.place()
    step = trainer.compile(state, BATCH)
    before = [raw(leaf).copy() for leaf in jax.tree.leaves(state)]
    timer = Timer(state, BATCH)
    seconds = timer([step, step])
    assert len(seconds) == 2 and all(took > 0 for took in seconds)
    assert all(leaf.is_deleted() for leaf in jax.tree.leaves(state))
    assert all(np.array_equal(raw(left), right)
               for left, right in zip(jax.tree.leaves(timer.state), before, strict=True))


def test_a_run_trains_on_from_the_state_its_plan_timed(monkeypatch, tmp_path):
    """Under `fit`, a frontier is timed on the run's own state, and the run
    trains on from a fresh copy of it: the timed steps count for nothing."""
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
