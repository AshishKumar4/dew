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
from token_support import write_token_corpus

from dew import cache
from dew.checkpoints import Checkpoints
from dew.data import Loading, presets
from dew.data.compile import compile
from dew.data.sources.tokens import TokenFiles
from dew.diffusion.discrete import MDLM
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import REMAT_POLICIES
from dew.nn.kernels.generation import chosen
from dew.nn.sharding import LayoutRefused
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.objectives.lm import LMObjective
from dew.objectives.lm.chunked import chunked_tile
from dew.registry import mixers
from dew.training import Trainer
from dew.training.distributed import Layout, MeshSpec, microbatch
from dew.training.memory import climb_to, recompute_more, rung_record
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
    rungs = [rung_record(climbing, xla_defaults=False)]
    while len(rungs) < 4 and recompute_more(climbing):
        rungs.append(rung_record(climbing, xla_defaults=False))
    tile = list(chunked_tile())
    assert [(rung["head_tile"], rung["head_fused"], rung["remat"]) for rung in rungs] == [
        (None, False, None), (tile, False, None), (None, False, "minimal"), (tile, False, "minimal")]
    resumed = objective()
    climb_to(resumed, rungs[1])
    assert rung_record(resumed, xla_defaults=False) == rungs[1]


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
        trainer.compile(state, BATCH, lambda steps: pytest.fail("the plan at its first split was timed"))
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
    trainer.compile(state, BATCH, lambda steps: pytest.fail("a single plan was timed"))
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
    trainer.compile(state, BATCH, lambda steps: pytest.fail("a single plan was timed"))
    assert_searched(asked(trainer), (4, None), 2, (1, "minimal"))
    assert all(remat is None for _, remat, _ in asked(trainer)[:-1])


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
