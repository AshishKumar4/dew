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
from dew.data import Loading, TokenWindows
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import REMAT_POLICIES
from dew.nn.sharding import LayoutRefused
from dew.objectives.lm import LMObjective
from dew.training import Trainer, memory, planning, rungs, trainer as trainer_module
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
    pinned split is refused by name."""
    model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=SEQ)
    spec = MeshSpec(explicit=("data",))
    with pytest.raises(ValueError, match="only 1 where an Explicit mesh axis types the step's rows"):
        Trainer(LMObjective(model, seq_len=SEQ), optax.sgd(1e-2), key=jax.random.key(0), split=2, mesh=spec)
    trainer = Trainer(LMObjective(model, seq_len=SEQ), optax.sgd(1e-2), key=jax.random.key(0), mesh=spec)
    assert splits(None, BATCH, MeshSpec().build(jax.devices()[:1]), MeshSpec(), trainer._split) == [1]


def lm(split=None, accumulation=1):
    model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=SEQ)
    return one_device(Trainer(LMObjective(model, seq_len=SEQ), optax.sgd(1e-2), key=jax.random.key(0),
                              split=split, accumulation=accumulation))


def trained(trainer, steps=3):
    """The state after `steps` steps of BATCH, and the step's temporaries."""
    state, _, _ = trainer.place()
    step = trainer.compile(state, BATCH)
    for _ in range(steps):
        state, *_ = step(state, BATCH)
    return state, trainer.executable.memory_analysis().temp_size_in_bytes


def room(monkeypatch, bytes_free):
    """Give every compile `bytes_free` for its temporaries, as XLA's memory analysis reads them."""
    monkeypatch.setattr(memory, "step_headroom", lambda executable, devices, held=0:
                        bytes_free - executable.memory_analysis().temp_size_in_bytes)
    monkeypatch.setattr(trainer_module, "step_compiler_options",
                        lambda objective, tokens, frozen, devices: None)


def test_a_batch_that_does_not_fit_whole_is_split_before_anything_is_recomputed(monkeypatch):
    """With room for the step's temporaries at two microbatches and not at
    one, the plan reads the batch in microbatches under no recomputation,
    and the run trains as the whole batch does: the update pools the same
    rows, `step` counts steps and `microstep` microbatches."""
    whole, temps = trained(lm(split=1))
    _, halves = trained(lm(split=2))
    assert halves < temps, (halves, temps)
    room(monkeypatch, (temps + halves) // 2)
    trainer = lm()
    state, _ = trained(trainer)
    split = trainer.plan["split"]
    assert split > 1 and trainer.objective.model.remat is None
    first, *_ = trainer.plan["probes"]
    assert (first["split"], first["fits"]) == (1, False)
    assert (int(state.step), int(state.microstep), int(state.updates)) == (3, 3 * split, 3)
    for planned_leaf, whole_leaf in zip(jax.tree.leaves(state.variables), jax.tree.leaves(whole.variables),
                                        strict=True):
        np.testing.assert_allclose(np.asarray(planned_leaf), np.asarray(whole_leaf), rtol=1e-4, atol=1e-5)


def test_a_later_run_starts_its_search_at_the_plan_an_earlier_one_recorded(monkeypatch):
    """The plan is recorded beside the compilation cache, read before the step
    traces, so a restart compiles the recorded candidate first, from the
    cache, and keeps it, with the FLOPs the first run counted."""
    _, temps = trained(lm(split=1), steps=0)
    _, halves = trained(lm(split=2), steps=0)
    room(monkeypatch, (temps + halves) // 2)
    first = lm()
    trained(first, steps=0)
    room(monkeypatch, 2 * temps)
    again = lm()
    counted = []
    monkeypatch.setattr(rungs, "step_flops", lambda executable: counted.append(executable))
    trained(again, steps=0)
    assert [probe["split"] for probe in again.plan["probes"]] == [first.plan["split"]]
    # Every program came from the cache, so the step's FLOPs are the record's, its text unread.
    assert counted == [] and again.step_flops == first.step_flops


def test_a_pinned_split_is_read_as_given_and_its_state_holds_it():
    """`split` fixes the microbatches; the window pools `split` x
    `accumulation` of them, one update every `accumulation` steps, and the
    state's window holds the split, so a trainer that continues the state
    reads its steps alike."""
    trainer = lm(split=4, accumulation=2)
    state, _ = trained(trainer, steps=4)
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
    assert splits(rows, BATCH, one, MeshSpec(), None) == [1, 2]
    assert splits(rows, BATCH, two, MeshSpec(), None) == [1]
    with pytest.raises(LayoutRefused, match=r"of \[4, 6, 8, 12, 16, 24\] rows over 1 row shards of .*, cut "
                       r"into whole rows on every device only in \[1, 2\]"):
        splits(rows, BATCH, one, MeshSpec(), 4)


def test_a_step_that_fits_no_plan_is_refused_with_the_closest_candidates_numbers(monkeypatch):
    """Where no split at any rung fits, the run stops naming every candidate
    it compiled and the one closest to fitting, with what it needs."""
    room(monkeypatch, 0)
    trainer = lm()
    state, _, _ = trainer.place()
    with pytest.raises(RuntimeError, match=r"fits no plan of its batch: the closest, \d+ x \d+ rows at remat "
                       r".*, needs [\d.]+ GiB of temporaries beside [\d.]+ GiB of arguments, [\d.]+ GiB more "
                       r"than its tightest device has\. Tried 1 x 32 rows at remat None; 32 x 1 rows at "
                       r"remat None; 32 x 1 rows at remat None, the head tiled; 32 x 1 rows at remat "
                       r"'minimal'; 32 x 1 rows at remat 'minimal', the head tiled;"):
        trainer.compile(state, BATCH)


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
    return TokenWindows(path=str(corpus), seq_len=4, stride=1, seed=7, val_batches=4,
                        loading=Loading(workers=0, threads=1, read_buffer=1)).load(batch=8)


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
    with pytest.raises(ValueError, match="the state pools windows of 2 microbatches, and this run updates "
                       "every 1 steps of 4 microbatches"):
        windowed(tmp_path / "split", 4).fit(windows(corpus), steps=6, checkpoint_every=2)


def fitting(monkeypatch, trainer, fits):
    """Make each compile of `trainer`'s plan fit by `fits(split, remat,
    tiled)`, read off the split it lowered last and its objective's rung."""
    lowered = []
    lower = trainer._candidate

    def recorded(*args):
        lowered.append(args[5])
        return lower(*args)

    monkeypatch.setattr(trainer, "_candidate", recorded)
    monkeypatch.setattr(planning, "step_fits", lambda executable, mesh, held=0: (fits(
        lowered[-1], trainer.objective.model.remat, trainer.objective.head_tile is not None), 0))
    monkeypatch.setattr(trainer_module, "step_compiler_options",
                        lambda objective, tokens, frozen, devices: None)


@pytest.mark.parametrize("seconds, split, remat", [([0.2, 0.1], 1, "minimal"), ([0.1, 0.2], 2, None)])
def test_a_split_plan_is_timed_against_its_batch_whole_a_rung_up_and_the_faster_kept(
        monkeypatch, seconds, split, remat):
    """Where the batch fits only split at the least recomputation, and whole
    one remat rung up with the head tiled, both are compiled and timed, and
    the faster is the plan: the objective at its rung, both times recorded."""
    trainer = lm()
    fitting(monkeypatch, trainer, lambda split, remat, tiled: split >= 2 or (remat is not None and tiled))
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


@pytest.mark.parametrize("fits", [lambda split, remat, tiled: True, lambda split, remat, tiled: split >= 2])
def test_a_plan_with_no_second_point_to_fit_is_not_timed(monkeypatch, fits):
    """A batch that fits whole, or that fits whole at no rung above, has one
    plan, and nothing is timed; the frontier's compile that did not fit is
    in the plan's record."""
    trainer = lm()
    fitting(monkeypatch, trainer, fits)
    state, _, _ = trainer.place()
    trainer.compile(state, BATCH, lambda steps: pytest.fail("a single plan was timed"))
    assert trainer.plan["timed"] is None
    assert trainer.objective.model.remat is None
    last = trainer.plan["probes"][-1]
    assert (last["split"], last["remat"], last["fits"]) == ((1, None, True) if trainer.plan["split"] == 1
                                                            else (1, "minimal", False))


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
    trainer = windowed(tmp_path / "timed", None)
    fitting(monkeypatch, trainer, lambda split, remat, tiled: split >= 2 or (remat is not None and tiled))
    state = trainer.fit(windows(token_files(tmp_path)), steps=3, checkpoint_every=None)
    assert [plan["seconds"] > 0 for plan in trainer.plan["timed"]] == [True, True]
    assert int(state.step) == 3
