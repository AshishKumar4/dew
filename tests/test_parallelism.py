"""Sharding, FSDP and data-pipeline tests on a simulated 8-device CPU mesh.

The parity tests are the safety net for the partitioned step: a run over eight
devices, and over a sharded parameter tree, has to produce the numbers a plain
single-device optax loop produces, otherwise the collectives GSPMD derived are
not the ones we meant.
"""


import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from diffusion_support import DeterministicObjective
from flax import linen as nn
from jax.sharding import PartitionSpec as P
from recording import RecordingTracker
from sharded import assert_sharded
from token_support import write_token_corpus

from dew.artifacts import Representations
from dew.data import Loading
from dew.data.dataset import DataPartition, Loader, replayed
from dew.distributed.batches import DevicePrefetchIterator, shard_batch
from dew.objectives.base import Aux, Objective
from dew.training import Checkpoints, Layout, MeshSpec, Trainer
from dew.training.evaluation import WARM_BATCHES

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh


RES = 8
BATCH = 8
# The test model's parameters are far below the production shard threshold, so
# the threshold is lowered; at the production value "FSDP on" would replicate
# everything.
TINY = 256


def images():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(BATCH, RES, RES, 3)).astype(np.uint8)


def batches():
    batch = {"image": images()}
    while True:
        yield batch


def make_trainer(tmp_path=None, fsdp=1, optimizer=None, objective=None, tracker=None,
                 **kwargs):
    checkpoints = None if tmp_path is None else Checkpoints(str(tmp_path), keep=4)
    return Trainer(
        DeterministicObjective() if objective is None else objective,
        optax.adam(1e-3) if optimizer is None else optimizer,
        key=jax.random.key(0),
        mesh=MeshSpec(fsdp=fsdp),
        layout=Layout(min_shard=TINY),
        checkpoints=checkpoints,
        tracker=tracker,
        **kwargs,
    )


def run_losses(steps, **kwargs):
    """Per-step losses of a fit, as the tracker receives them; the trained
    parameters are split over the mesh's parameter axes."""
    tracker = RecordingTracker()
    trainer = make_trainer(tracker=tracker, **kwargs)
    state = trainer.fit(
        Loader(lambda partition: batches(), None, batch=BATCH, records=None), steps=steps, log_every=1)
    assert_sharded(state.variables["params"], trainer.device_mesh)
    return [scalars["train/loss"] for _, scalars in tracker.scalars if "train/loss" in scalars]


def reference_losses(trainer, steps):
    """The same run as a plain optax loop on one device: the yardstick a
    partitioned step has to reproduce."""
    objective, optimizer = trainer.objective, trainer.optimizer
    state = trainer.initial_state()
    device = jax.devices()[0]
    params = jax.device_put(state.variables, device)
    opt_state = jax.device_put(state.opt_state, device)

    @jax.jit
    def step(params, opt_state, batch, key):
        def loss_fn(trainable):
            from dew.objectives.base import Step
            return objective.scalar_loss({**params, "params": trainable}, batch,
                                  Step(step=jnp.zeros((), jnp.int32), key=key, ema=None))

        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params["params"])
        updates, opt_state = optimizer.update(grads, opt_state, params["params"])
        return {**params, "params": optax.apply_updates(params["params"], updates)}, opt_state, loss

    losses = []
    source = batches()
    for index in range(steps):
        batch = jax.device_put(next(source), device)
        params, opt_state, loss = step(params, opt_state, batch,
                                       jax.random.fold_in(state.key, index))
        losses.append(float(loss))
    return losses


# --------------------------------------------------------------------------
# Sharding heuristic
# --------------------------------------------------------------------------

class Decoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(8, use_bias=False, name="readout")(x)


class Readouts(nn.Module):
    """A top-level `readout` whose module boxes its kernel's axes, a plain
    `readout` of the same shape under `decoder`, and a kernel boxed with names
    no rule mentions."""

    @nn.compact
    def __call__(self, x):
        def boxed(name, names):
            return nn.Dense(8, use_bias=False, name=name, kernel_init=nn.with_logical_partitioning(
                nn.initializers.lecun_normal(), names))(x)

        return boxed("readout", (None, "vocab")) + Decoder(name="decoder")(x) + boxed(
            "foreign", ("foreign_a", "foreign_b"))


class BoxedReadout(Objective):
    ema = None

    def fresh_variables(self, key, held):
        return Readouts().init(key, jnp.ones((1, 16)))

    def loss(self, variables, batch, step):
        features = batch["image"].reshape(batch["image"].shape[0], -1)[:, :16].astype(jnp.float32) / 255
        return jnp.sum(Readouts().apply(variables, features) ** 2), Aux({})


def test_a_restored_run_places_its_boxed_parameters_by_their_names(tmp_path):
    """The trainer records the axes the model boxes with each checkpoint, and
    a restore onto a mesh (`Checkpoints.variables`, which `from_run` and
    `dew.pipeline` read) places by them, the state having held them unboxed."""
    layout = Layout(min_shard=1, rules=(("vocab", "fsdp"),))
    Trainer(BoxedReadout(), optax.adam(1e-3), key=jax.random.key(0), mesh=MeshSpec(fsdp=2), layout=layout,
            checkpoints=Checkpoints(str(tmp_path))).fit(
                Loader(lambda partition: batches(), None, batch=BATCH, records=None), steps=1)

    checkpoints = Checkpoints(str(tmp_path))
    assert checkpoints.boxed() == {("params", "readout", "kernel"): (None, "vocab"),
                                   ("params", "foreign", "kernel"): ("foreign_a", "foreign_b")}
    params = checkpoints.variables(mesh=MeshSpec(fsdp=2), layout=layout)["params"]
    assert params["readout"]["kernel"].sharding.spec == P(None, "fsdp")
    assert params["decoder"]["readout"]["kernel"].sharding.spec == P("fsdp")
    assert params["foreign"]["kernel"].sharding.spec == P("fsdp")


# --------------------------------------------------------------------------
# Batch placement and prefetch
# --------------------------------------------------------------------------

@pytest.mark.parametrize("devices", [1, 8])
def test_placed_batch_owns_its_reader_buffer(devices):
    # CPU zero-copy needs aligned storage; each row and device slice stays aligned.
    size = 1 << 20
    storage = np.empty(size * 4 + 63, np.uint8)
    offset = -storage.ctypes.data % 64
    source = storage[offset:offset + size * 4].view(np.float32).reshape(8, -1)
    source.fill(7)
    mesh = MeshSpec().build(jax.devices()[:devices])
    placed = shard_batch(mesh, {"tokens": source})["tokens"]
    placed.block_until_ready()

    for shard in placed.addressable_shards:
        assert not np.shares_memory(np.asarray(shard.data), source)
    source.fill(-1)
    np.testing.assert_array_equal(np.asarray(placed), np.full(source.shape, 7, np.float32))


def test_prefetch_iterator_preserves_order_and_terminates():
    mesh = MeshSpec().build()
    source = ({"x": np.full((jax.device_count(), 2), i, np.float32)} for i in range(5))
    with DevicePrefetchIterator(source, mesh, depth=2) as it:
        seen = [float(np.asarray(b["x"])[0, 0]) for b in it]
        assert seen == [0.0, 1.0, 2.0, 3.0, 4.0]
        with pytest.raises(StopIteration):
            next(it)


def test_a_held_prefetcher_places_no_batch_until_its_hold_ends():
    """A prefetcher held (`DevicePrefetchIterator.held`) reads on but places nothing until the hold ends,
    then hands over every batch in order. A run holds its stream so while it compiles, evaluates or
    saves: a batch placed during that work lands past the memory the work borrows, in the block the next
    step's temporaries take whole (row 2088)."""
    import threading

    released, asked = threading.Event(), threading.Event()

    def source():
        for value in range(3):
            if value == 1:
                released.wait()
            if value == 2:
                # The worker asks for a third batch only once it has placed the second.
                asked.set()
            yield {"x": np.full((jax.device_count(), 2), value, np.float32)}

    with DevicePrefetchIterator(source(), MeshSpec().build(), depth=4) as batches:
        first = next(batches)
        with batches.held():
            released.set()
            assert not asked.wait(timeout=1)
        assert asked.wait(timeout=60)
        seen = [float(np.asarray(batch["x"])[0, 0]) for batch in [first, *batches]]
    assert seen == [0.0, 1.0, 2.0]


def test_a_prefetcher_closed_while_held_stops_its_worker():
    """Closing a prefetcher inside a hold stops its worker, which waits out the hold with a batch it read
    and would otherwise place: the close returns, and the source is finalized."""
    import threading

    released, prepared, closed = threading.Event(), threading.Event(), threading.Event()

    def source():
        try:
            yield {"x": np.zeros((jax.device_count(), 2), np.float32)}
            released.wait()
            yield {"x": np.ones((jax.device_count(), 2), np.float32)}
        finally:
            closed.set()

    def prepare(batch):
        # The worker prepares a batch after it read it and before it asks to place it.
        if float(np.asarray(batch["x"])[0, 0]) == 1:
            prepared.set()
        return batch

    batches = DevicePrefetchIterator(source(), MeshSpec().build(), depth=4, prepare=prepare)
    next(batches)
    with batches.held():
        released.set()
        assert prepared.wait(timeout=60)
        batches.close()
    assert closed.is_set()


def test_prefetch_iterator_tracks_checkpointable_source_state():
    """The position handed out must be the consumed batch's, not the thread's."""
    import grain.python as pygrain

    loader = pygrain.DataLoader(
        data_source=pygrain.RangeDataSource(0, 64, 1),
        sampler=pygrain.IndexSampler(num_records=64, shuffle=False, seed=0, num_epochs=1,
                                     shard_options=pygrain.NoSharding()),
        operations=[pygrain.Batch(jax.device_count(), drop_remainder=True)],
        worker_count=0,
    )
    mesh = MeshSpec().build()
    with DevicePrefetchIterator(iter(loader), mesh, depth=2) as it:
        next(it)
        next(it)
        state = it.source_state
        expected = np.asarray(next(it))

        with DevicePrefetchIterator(iter(loader), mesh, depth=2,
                                    source_state=state) as resumed:
            assert np.array_equal(np.asarray(next(resumed)), expected)


def test_prefetch_iterator_resumes_a_packed_dataset_iterator(tmp_path):
    """The packed loader is grain's Loader API, whose iterator reports its
    position as a dict where the DataLoader reports JSON bytes. A checkpoint
    holds bytes, so the position has to arrive as bytes and go back as a
    dict."""
    from dew.data import presets
    from dew.data.compile import compile
    from dew.data.sources.tokens import TokenFiles

    corpus = write_token_corpus(tmp_path / "tokens", np.concatenate([np.arange(1, 9), [0]] * 40), eos_id=0)
    # Packed in the store's order, then ordered: the pack is one plan over the store.
    packed = TokenFiles(corpus.train).pipe(presets.lm, None, seq_len=8, batch=None, bins=2).shuffle(
        0).repeat().batch(jax.device_count())
    data = compile(packed, read="memory", seed=0, loading=Loading(workers=0))

    mesh = MeshSpec().build()
    with DevicePrefetchIterator(data.train(DataPartition()), mesh, depth=2) as it:
        next(it)
        state = it.source_state
        expected = np.asarray(next(it)["text"])
        assert isinstance(state, bytes), "a checkpoint carries the position as bytes"

        with DevicePrefetchIterator(data.train(DataPartition()), mesh, depth=2,
                                    source_state=state) as resumed:
            assert np.array_equal(np.asarray(next(resumed)["text"]), expected)


# --------------------------------------------------------------------------
# Numerical parity
# --------------------------------------------------------------------------

def test_the_partitioned_step_reproduces_a_single_device_loop():
    """Partitioning over eight devices must not change the maths."""
    steps = 20
    trainer = make_trainer(optimizer=optax.sgd(0.5))
    reference = reference_losses(trainer, steps)
    assert jax.device_count() > 1
    np.testing.assert_allclose(reference, run_losses(steps, optimizer=optax.sgd(0.5)),
                               rtol=2e-4, atol=2e-5)


def test_fsdp_losses_match_replicated():
    """Sharding the parameters must not change the loss trajectory."""
    steps = 20
    replicated = run_losses(steps, fsdp=1)
    fsdp = run_losses(steps, fsdp=2)
    assert np.isfinite(replicated).all()
    np.testing.assert_allclose(replicated, fsdp, rtol=2e-4, atol=2e-5)


# --------------------------------------------------------------------------
# FSDP actually shards
# --------------------------------------------------------------------------

def test_fsdp_shards_parameters_and_optimizer_state():
    state = make_trainer(fsdp=2).fit(
        Loader(lambda partition: batches(), None, batch=BATCH, records=None), steps=1, log_every=1)
    leaves = jax.tree.leaves(state.variables)
    sharded = [x for x in leaves if 'fsdp' in str(x.sharding.spec)]
    assert sharded, "no parameter was sharded over the fsdp axis"

    for param in sharded:
        local = param.addressable_shards[0].data
        assert local.size == param.size // 2, "shard is not half the global param"
        # Exactly the dimension the spec names is halved. Which dimension that
        # is belongs to the declarations, not to this test.
        split = [axis for axis, (whole, part) in enumerate(
            zip(param.shape, local.shape, strict=True)) if whole != part]
        assert len(split) == 1, f"{param.shape} -> {local.shape}"
        assert param.shape[split[0]] // 2 == local.shape[split[0]]
        assert param.sharding.spec[split[0]] == 'fsdp'

    # Adam moments and the EMA copy must follow the variables they track,
    # without the optimizer or the model ever describing a layout. The
    # optimizer tracks the params collection; this objective's EMA tracks
    # every collection.
    def specs(tree):
        return [x.sharding.spec for x in jax.tree.leaves(tree)]

    assert specs(state.variables["params"]) == specs(state.opt_state[0].mu)
    assert specs(state.variables) == specs(state.ema)


# --------------------------------------------------------------------------
# Checkpointing under sharding
# --------------------------------------------------------------------------

def test_sharded_checkpoint_roundtrips(tmp_path):
    trained = make_trainer(tmp_path, fsdp=2).fit(
        Loader(lambda partition: batches(), None, batch=BATCH, records=None), steps=1, log_every=1)
    restored = make_trainer(tmp_path, fsdp=2).fit(
        Loader(lambda partition: batches(), None, batch=BATCH, records=None), steps=1)

    assert int(restored.step) == 1
    for before, after in zip(jax.tree.leaves(trained.variables),
                             jax.tree.leaves(restored.variables), strict=True):
        assert before.sharding.spec == after.sharding.spec
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))


@pytest.mark.parametrize("written,restored", [(1, 8), (8, 1)])
def test_a_checkpoint_restores_across_the_whole_fsdp_range(tmp_path, written, restored):
    """A run resumes on the hardware it gets, not the hardware it left.

    Both directions of the widest change the simulated mesh allows: every
    parameter replicated, and every parameter split eight ways. A checkpoint
    is bytes, not arithmetic, so the values have to come back equal, the EMA
    copy's as much as the weights'.
    """
    trained = make_trainer(tmp_path, fsdp=written).fit(
        Loader(lambda partition: batches(), None, batch=BATCH, records=None), steps=1, log_every=1)
    before = {field: [np.asarray(leaf).copy() for leaf in jax.tree.leaves(getattr(trained, field))]
              for field in ("variables", "ema")}
    assert any(not np.array_equal(average, live) for average, live
               in zip(before["ema"], before["variables"], strict=True)), "the EMA is its weights"

    reopened = make_trainer(tmp_path, fsdp=restored).fit(
        Loader(lambda partition: batches(), None, batch=BATCH, records=None), steps=1)
    assert int(reopened.step) == 1
    for field, saved in before.items():
        leaves = jax.tree.leaves(getattr(reopened, field))
        for value, leaf in zip(saved, leaves, strict=True):
            assert value.dtype == leaf.dtype, field
            np.testing.assert_array_equal(value.view(np.uint8), np.asarray(leaf).view(np.uint8),
                                          err_msg=field)

        sharded = [leaf for leaf in leaves if 'fsdp' in str(leaf.sharding.spec)]
        assert bool(sharded) == (restored > 1), f"the restored {field} layout is not this run's"
        for leaf in sharded:
            assert leaf.addressable_shards[0].data.size == leaf.size // restored


def test_a_checkpoint_without_a_position_does_not_resume_from_the_top_of_the_stream(tmp_path):
    """A stream without get_state writes no position, and a resume from that
    checkpoint, which would read the stream again from its start, is refused."""
    stream = Loader(lambda partition: batches(), None, batch=BATCH, records=None)
    make_trainer(tmp_path).fit(stream, steps=1, log_every=1)
    _, position = Checkpoints(str(tmp_path)).restore(share=DataPartition())
    assert position is None
    with pytest.raises(ValueError):
        make_trainer(tmp_path).fit(stream, steps=2, log_every=1)


# --------------------------------------------------------------------------
# Gradient accumulation and the EMA clock
# --------------------------------------------------------------------------

def make_accumulating(tmp_path=None, accumulation=1):
    return make_trainer(
        tmp_path, fsdp=2, optimizer=optax.sgd(0.5), accumulation=accumulation,
        # A ramp, not a constant, so indexing the schedule by micro-step
        # instead of by update is visible in the result.
        objective=DeterministicObjective(optax.linear_schedule(0.9, 1.0, transition_steps=8)))


def micro_steps(trainer, count, state=None):
    """Run compiled transactions over the deterministic batch stream."""
    state = trainer.initial_state() if state is None else state
    source = batches()
    batch = next(source)
    step = trainer.compile(state, batch)
    for _ in range(count):
        state, *_ = step(state, batch)
        batch = next(source)
    return state


def snapshot(tree):
    return [np.asarray(x).copy() for x in jax.tree.leaves(tree)]


def moved(before, after):
    return any(not np.array_equal(a, b) for a, b in zip(before, after, strict=True))


def test_accumulated_ema_matches_a_plain_run_at_equal_update_counts():
    """k micro-batches of the same data must land where one big batch would,
    EMA included: same params, same decay index, same average."""
    accum, updates = 4, 3
    plain = micro_steps(make_accumulating(accumulation=1), updates)
    accumulated = micro_steps(make_accumulating(accumulation=accum), updates * accum)

    # the comparison is only meaningful if the EMA left its starting point
    assert moved(snapshot(plain.variables), snapshot(plain.ema))
    for a, b in zip(jax.tree.leaves(plain.variables), jax.tree.leaves(accumulated.variables), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7)
    for a, b in zip(jax.tree.leaves(plain.ema), jax.tree.leaves(accumulated.ema), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7)


def test_a_resume_inside_an_accumulation_window_keeps_the_ema_clock(tmp_path):
    """An accumulated run killed mid-window has to average like one run whole.

    The checkpoint is taken one micro-batch into a window, so it carries the
    half-filled accumulator, the MultiSteps counters and a step count that is
    not a multiple of the window. The EMA decay is a ramp indexed by completed
    updates, so a resume that took the micro-batch counter for the update
    counter, or started either counter over, moves the average a different
    distance from where the uninterrupted run put it.
    """
    accum, updates = 2, 3
    total = accum * updates * 2
    cut = accum * updates + 1

    whole = make_accumulating(tmp_path / "whole", accum).fit(
        Loader(replayed(lambda partition: batches()), None, batch=BATCH, records=None), steps=total)
    interrupted = make_accumulating(tmp_path / "split", accum).fit(
        Loader(replayed(lambda partition: batches()), None, batch=BATCH, records=None), steps=cut)
    assert int(interrupted.microstep) % accum == 1, "the cut is not mid-window"
    resumed = make_accumulating(tmp_path / "split", accum).fit(
        Loader(replayed(lambda partition: batches()), None, batch=BATCH, records=None), steps=total)

    assert int(resumed.step) == int(whole.step) == total
    assert moved(snapshot(whole.variables), snapshot(whole.ema))
    for expected, actual in zip(jax.tree.leaves(whole.variables),
                                jax.tree.leaves(resumed.variables), strict=True):
        np.testing.assert_allclose(np.asarray(expected), np.asarray(actual),
                                   rtol=1e-6, atol=1e-7)
    for expected, actual in zip(jax.tree.leaves(whole.ema),
                                jax.tree.leaves(resumed.ema), strict=True):
        np.testing.assert_allclose(np.asarray(expected), np.asarray(actual),
                                   rtol=1e-6, atol=1e-7)


# --------------------------------------------------------------------------
# Resuming a grain loader mid-epoch
# --------------------------------------------------------------------------

def grain_image_loader(num_records=256):
    """A checkpointable source that yields distinguishable image batches."""
    import grain.python as pygrain

    class ToImage(pygrain.MapTransform):
        def map(self, index):
            return {"image": np.full((RES, RES, 3), index, np.uint8)}

    return iter(pygrain.DataLoader(
        data_source=pygrain.RangeDataSource(0, num_records, 1),
        sampler=pygrain.IndexSampler(num_records=num_records, shuffle=False, seed=0,
                                     num_epochs=4, shard_options=pygrain.NoSharding()),
        operations=[ToImage(), pygrain.Batch(BATCH, drop_remainder=True)],
        worker_count=0,
    ))


def test_a_resumed_run_reads_the_batch_after_its_checkpoint(tmp_path):
    """A resumed run must not replay the batches it already trained on."""
    data = Loader(lambda partition: grain_image_loader(), None, batch=BATCH, records=BATCH * 64)
    make_trainer(tmp_path).fit(data, steps=3, log_every=1)
    _, position = Checkpoints(str(tmp_path)).restore(share=DataPartition())
    assert position is not None, "iterator position was never captured"

    mesh = MeshSpec().build()
    with DevicePrefetchIterator(grain_image_loader(), mesh,
                                source_state=position) as resumed, (
            DevicePrefetchIterator(grain_image_loader(), mesh)
    ) as fresh:
        for _ in range(3):
            next(fresh)
        np.testing.assert_array_equal(np.asarray(next(resumed)["image"]),
                                      np.asarray(next(fresh)["image"]))


# --------------------------------------------------------------------------
# The whole run, on a real mesh
# --------------------------------------------------------------------------

class PeakToPeak:
    """A real metric over the objective's artifacts.

    Deliberately not CLIP or FID: those download pretrained weights. What is
    under test is the metric plumbing (artifacts in, score out, logged),
    which is identical whatever the score means.
    """
    name = "sample_range"
    reads = Representations

    def __init__(self, seen):
        self.seen = seen

    def __call__(self, artifact, batch):
        self.seen.append((np.asarray(artifact.features).shape, np.asarray(batch["image"]).shape))
        return float(jnp.max(artifact.features) - jnp.min(artifact.features)), 1

    def merge(self, accumulated, contribution):
        return accumulated[0] + contribution[0], accumulated[1] + contribution[1]

    def finalize(self, accumulated):
        return accumulated[0] / accumulated[1]


def val_stream():
    source = batches()
    for _ in range(2):
        yield next(source)


@pytest.mark.parametrize("fsdp_size", [1, 4])
def test_fit_end_to_end_with_validation_and_metrics(tmp_path, fsdp_size):
    """A full run on the simulated 8-device mesh: training, the validation
    pass, a metric, the tracker, and the final checkpoint.

    Parametrized over pure data parallelism and a 2x4 data/fsdp mesh, because
    validation reads the *sharded* EMA parameters, a layout the training step
    alone never exercises.
    """
    seen = []
    tracker = RecordingTracker()
    trainer = make_trainer(tmp_path, fsdp=fsdp_size, tracker=tracker)
    state = trainer.fit(
        Loader(lambda partition: batches(), lambda partition: val_stream(), batch=BATCH, records=None),
        steps=2, log_every=1, eval_every=1,
        metrics=(PeakToPeak(seen),), preview=True)

    assert int(state.step) == 2
    sharded = [x for x in jax.tree.leaves(state.ema) if 'fsdp' in str(x.sharding.spec)]
    assert bool(sharded) == (fsdp_size > 1)
    # The pass that warms the evaluations before the first step, then one after step 1 and one
    # at the end, two batches each.
    assert seen == [((BATCH, 3), (BATCH, RES, RES, 3))] * (WARM_BATCHES + 4)
    scores = [s["val/sample_range"] for _, s in tracker.scalars if "val/sample_range" in s]
    assert len(scores) == 2 and all(np.isfinite(score) and score > 0 for score in scores)
    assert [step for step, value in tracker.artifacts if isinstance(value, Representations)] == [1, 2]
    assert Checkpoints(str(tmp_path)).latest == 2


class Releasing(PeakToPeak):
    """`PeakToPeak`, whose first call in the run's first validation pass after its warming one
    (`WARM_BATCHES` calls) lets the training stream read on (`released`) and checks that the stream
    places nothing before that pass ends: the stream asks for the batch after the one released
    (`asked`) only once it has placed that one."""

    def __init__(self, released, asked):
        super().__init__([])
        self.released, self.asked = released, asked

    def __call__(self, artifact, batch):
        if len(self.seen) == WARM_BATCHES:
            self.released.set()
            assert not self.asked.wait(timeout=1), "the training stream placed a batch during the pass"
        return super().__call__(artifact, batch)


def test_a_run_places_no_training_batch_while_it_evaluates():
    """A run whose training stream falls behind and catches up during a validation pass places the
    caught-up batch after the pass, beside the step that reads it, never during it: on a GPU, a batch
    placed while a pass holds memory lands past the pass's temporaries, inside the free block the next
    step takes whole, and a 4B's step that fit before its first pass found its block split after it
    (row 2088)."""
    import threading

    released, asked = threading.Event(), threading.Event()

    def stream():
        # The first step's batch and the one the pass after it reads ahead, then the rest once released.
        source = batches()
        yield next(source)
        yield next(source)
        released.wait()
        yield next(source)
        asked.set()
        yield from source

    state = make_trainer().fit(
        Loader(lambda partition: stream(), lambda partition: val_stream(), batch=BATCH, records=None),
        steps=2, log_every=1, eval_every=1, metrics=(Releasing(released, asked),))
    assert int(state.step) == 2


# --------------------------------------------------------------------------
# The optimizer parameter groups under sharding
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Sharding invariants at every fsdp width
# --------------------------------------------------------------------------
