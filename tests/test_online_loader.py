"""Streaming loader tests: what reaches the model when fetching goes wrong.

Most of these replace the fetcher pool with a stub producer. One simulates a
missing HF `datasets` to check that importing the streaming stack does not
need the `streaming` extra. The test that runs the real pool cannot stub the
fetch, because a worker that was not forked holds no copy of this process's
memory, so it asserts on what its workers report. Nothing here touches the
network. Every url is under .invalid, apart from the file:// url the fetch
test reads out of tmp_path.
"""

import functools
import multiprocessing
import os
import queue
import sys
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import PIL.Image
import pytest

from dew.data import Loading, online_loader
from dew.objectives.base import Aux, Objective
from dew.training import Checkpoints, Layout, MeshSpec, Trainer
from dew.data.online_loader import (
    DROPPED_SAMPLE, MediaBatchIterator, OnlineStreamingDataLoader,
)

BATCH = 4
# Comfortably longer than the iterator's timeout below, so the slow producer
# really does make it wait.
SLOW = 0.15
# The pool test spreads 12 rows over 2 workers.
POOL_ROWS = 12


@pytest.fixture
def stop():
    """Keeps a stub producer alive for the test, like the endless real one."""
    event = threading.Event()
    yield event
    event.set()


def _sample(index):
    return {
        "url": f"https://example.invalid/{index}.jpg",
        "caption": f"sample {index}",
        "image": np.full((8, 8, 3), index + 1, np.uint8),
        "original_height": 8,
        "original_width": 8,
    }


def _iterator(monkeypatch, producer, queue_timeout=0.05):
    """A MediaBatchIterator fed by `producer` instead of the fetcher pool.

    The iterator binds the producer from the module global as it is built, so
    patching the name keeps the real one, and its process pool, out of reach.
    """
    monkeypatch.setattr(online_loader, "parallel_media_loader", producer)
    return MediaBatchIterator(list(range(64)), batch_size=BATCH,
                              queue_timeout=queue_timeout)


class _StubDataset:
    """As much of an HF dataset as the loader and the fetcher pool read.

    Column oriented like HF's, so a shard is a dict of lists. `passes` bounds
    the producer's otherwise endless loop, so a test's worker pool shuts down
    with it.
    """

    def __init__(self, size, passes=1):
        self.size = size
        self.passes = passes

    def shard(self, num_shards, index):
        return self

    def shuffle(self, seed=0):
        if seed > self.passes:
            raise RuntimeError("stub dataset ran out of passes")
        return self

    def __len__(self):
        return self.size

    def __getitem__(self, window):
        indices = range(*window.indices(self.size))
        return {"url": [f"https://example.invalid/{i}.jpg" for i in indices],
                "caption": [f"caption {i}" for i in indices]}


def _image_file(root, index):
    """A file the real fetch can read. Over the minimum size, square, and not
    a solid colour, so the default processor keeps it."""
    pixels = np.random.RandomState(index).randint(0, 256, (48, 48, 3), np.uint8)
    path = root / f"{index}.png"
    PIL.Image.fromarray(pixels).save(path)
    return path


def _refuse_to_fork(*args, **kwargs):
    raise AssertionError("the fetch pool forked a process that runs JAX")


# ---------------------------------------------------------------------------------
# Slow fetching waits; it never invents data
# ---------------------------------------------------------------------------------

def test_slow_fetching_waits_instead_of_fabricating_samples(monkeypatch, stop, capsys):
    """A queue timeout waits rather than handing back zeros captioned "Timeout
    occurred while waiting for sample", which would train as data."""
    def slow(dataset, *, data_queue, **kwargs):
        for index in range(BATCH):
            stop.wait(SLOW)
            data_queue.put(_sample(index))
        stop.wait(10)

    iterator = _iterator(monkeypatch, slow)
    batch = next(iterator)

    assert [sample["caption"] for sample in batch] == [f"sample {i}" for i in range(BATCH)]
    assert all(int(sample["image"].max()) > 0 for sample in batch)
    assert "still fetching" in capsys.readouterr().out


def test_dropped_fetches_are_counted_and_never_yielded(monkeypatch, stop):
    def with_drops(dataset, *, data_queue, **kwargs):
        for index in range(BATCH):
            data_queue.put({DROPPED_SAMPLE: f"https://example.invalid/{index}.jpg"})
            data_queue.put(_sample(index))
        stop.wait(10)

    iterator = _iterator(monkeypatch, with_drops)
    batch = next(iterator)

    assert [sample["caption"] for sample in batch] == [f"sample {i}" for i in range(BATCH)]
    assert iterator.dropped == BATCH


def test_a_failed_fetch_leaves_a_drop_marker_for_the_iterator(monkeypatch):
    """The workers drop dead URLs in their own processes; the marker is how the
    count reaches the iterator."""
    monkeypatch.setattr(online_loader, "fetch_single_image", lambda *args, **kwargs: None)
    data_queue = queue.Queue()

    online_loader.map_image_sample("https://example.invalid/gone.jpg", "caption", data_queue)

    assert data_queue.get_nowait() == {DROPPED_SAMPLE: "https://example.invalid/gone.jpg"}


# ---------------------------------------------------------------------------------
# A producer that is gone ends iteration, and says why
# ---------------------------------------------------------------------------------

def test_an_exhausted_producer_stops_iteration(monkeypatch):
    def exhausted(dataset, *, data_queue, **kwargs):
        data_queue.put(_sample(0))

    iterator = _iterator(monkeypatch, exhausted)

    with pytest.raises(StopIteration):
        next(iterator)
    # and it stays stopped rather than blocking on the empty queue
    with pytest.raises(StopIteration):
        next(iterator)


def test_a_producer_that_dies_raises_its_own_error(monkeypatch):
    def broken(dataset, *, data_queue, **kwargs):
        raise RuntimeError("the fetch pool fell over")

    iterator = _iterator(monkeypatch, broken)

    with pytest.raises(RuntimeError, match="media fetcher died") as failure:
        next(iterator)
    assert "fell over" in str(failure.value.__cause__)


# ---------------------------------------------------------------------------------
# The prefetch thread's failures reach the consumer
# ---------------------------------------------------------------------------------

def test_a_collate_failure_surfaces_from_the_consumer(monkeypatch, stop):
    """The batch worker's exception reaches the consumer, rather than being
    printed while the run trains on whatever the next iteration produced."""
    def producer(dataset, *, data_queue, **kwargs):
        for index in range(BATCH):
            data_queue.put(_sample(index))
        stop.wait(10)

    monkeypatch.setattr(online_loader, "parallel_media_loader", producer)

    def collate(batch):
        raise ValueError("malformed sample in batch")

    loader = OnlineStreamingDataLoader(_StubDataset(64), batch_size=BATCH,
                                       collate_fn=collate, prefetch=2)

    with pytest.raises(RuntimeError, match="streaming data loader died") as failure:
        next(loader)
    assert "malformed sample" in str(failure.value.__cause__)


def test_batches_keep_flowing_until_the_collate_error(monkeypatch, stop):
    """Whatever was already collated is still delivered; the failure comes
    after it, not instead of it."""
    def producer(dataset, *, data_queue, **kwargs):
        for index in range(2 * BATCH):
            data_queue.put(_sample(index))
        stop.wait(10)

    monkeypatch.setattr(online_loader, "parallel_media_loader", producer)

    collated = []

    def collate(batch):
        if len(collated) == 1:
            raise ValueError("second batch is malformed")
        collated.append(batch)
        return {"captions": [sample["caption"] for sample in batch]}

    loader = OnlineStreamingDataLoader(_StubDataset(64), batch_size=BATCH,
                                       collate_fn=collate, prefetch=4)

    assert next(loader)["captions"] == [f"sample {i}" for i in range(BATCH)]
    with pytest.raises(RuntimeError, match="streaming data loader died"):
        next(loader)


# ---------------------------------------------------------------------------------
# The real worker pool
# ---------------------------------------------------------------------------------

def test_the_fetch_pool_starts_its_workers_without_forking(monkeypatch):
    """os.fork carries over only the calling thread, so a worker forked out of
    a training process inherits mutexes that JAX's other threads were holding,
    and hangs the first time it allocates. With os.fork refused the pool still
    has to start its workers and hear back from them."""
    monkeypatch.setattr(os, "fork", _refuse_to_fork)
    data_queue = online_loader.ResourceManager(max_queue_size=64).get_data_queue()
    producer_failure = []

    def produce():
        try:
            online_loader.parallel_media_loader(
                _StubDataset(POOL_ROWS), data_queue=data_queue, num_workers=2,
                num_threads=2, timeout=1, retries=0, image_shape=(64, 64),
                min_image_shape=(32, 32))
        except BaseException as error:
            producer_failure.append(error)

    threading.Thread(target=produce, daemon=True).start()

    reported = []
    deadline = time.monotonic() + 60
    while len(reported) < POOL_ROWS and time.monotonic() < deadline:
        try:
            reported.append(data_queue.get(timeout=0.5))
        except queue.Empty:
            # Nothing queued and the producer gone means no worker ever ran.
            assert not producer_failure, producer_failure[0]

    # Each url is under .invalid, so every fetch fails, and the marker a worker
    # queues for it proves the worker ran and its inherited queue crossed over.
    assert len(reported) == POOL_ROWS
    assert all(DROPPED_SAMPLE in entry for entry in reported)


def test_the_fetch_pool_covers_the_rows_past_an_even_split(monkeypatch):
    """Thirteen rows over two workers: an even split holds twelve, so the
    thirteenth row needs a shard of its own. A row no shard holds is a row no
    run ever trains on, however long the stream runs. The drop markers carry
    their urls, so two passes have to show every row."""
    monkeypatch.setattr(os, "fork", _refuse_to_fork)
    rows = POOL_ROWS + 1
    data_queue = online_loader.ResourceManager(max_queue_size=64).get_data_queue()
    producer_failure = []

    def produce():
        try:
            online_loader.parallel_media_loader(
                _StubDataset(rows, passes=2), data_queue=data_queue, num_workers=2,
                num_threads=2, timeout=1, retries=0, image_shape=(64, 64),
                min_image_shape=(32, 32))
        except BaseException as error:
            producer_failure.append(error)

    threading.Thread(target=produce, daemon=True).start()

    reported = []
    deadline = time.monotonic() + 120
    while len(reported) < 2 * rows and time.monotonic() < deadline:
        try:
            reported.append(data_queue.get(timeout=0.5))
        except queue.Empty:
            assert not producer_failure, producer_failure[0]

    assert len(reported) == 2 * rows
    assert all(DROPPED_SAMPLE in entry for entry in reported)
    seen = {entry[DROPPED_SAMPLE] for entry in reported}
    assert seen == {f"https://example.invalid/{i}.jpg" for i in range(rows)}


def test_a_fetched_image_reaches_the_queue_as_a_sample(tmp_path, monkeypatch):
    """What a worker does with one url, run here rather than in a worker, which
    holds no copy of a patched fetch. Of that path only the request header
    comes from HF datasets, which the streaming extra owns."""
    monkeypatch.setattr(online_loader, "_user_agent", lambda: "dew-tests")
    path = _image_file(tmp_path, 3)
    data_queue = queue.Queue()

    online_loader.map_image_sample(path.as_uri(), "caption 3", data_queue,
                                   image_shape=(64, 64), min_image_shape=(32, 32))

    sample = data_queue.get_nowait()
    assert sample["caption"] == "caption 3"
    assert sample["image"].shape == (64, 64, 3)
    assert int(sample["image"].max()) > 0
    assert (sample["original_height"], sample["original_width"]) == (48, 48)


# ---------------------------------------------------------------------------------
# The streaming extra
# ---------------------------------------------------------------------------------

def test_loading_rows_asks_for_the_streaming_extra(monkeypatch):
    """Importing this module must work without HF datasets; only actually
    loading a dataset needs it, and it has to say so.

    The absence is simulated so the test states the behaviour whether or not
    the extra is installed in the environment that runs it.
    """
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError, match=r"dew-ml\[streaming\]"):
        online_loader.load_rows(["some/hf/dataset"])


def test_feature_extractor_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="URL"):
        online_loader.default_feature_extractor({"caption": ["caption"]})
    with pytest.raises(ValueError, match="caption"):
        online_loader.default_feature_extractor({"url": ["https://example.invalid/0.jpg"]})


def test_batch_mapping_propagates_feature_extractor_errors():
    def broken(sample):
        raise RuntimeError("bad shard schema")

    with pytest.raises(RuntimeError, match="bad shard schema"):
        online_loader.map_batch({}, queue.Queue(), num_threads=1,
                               feature_extractor=broken)


# ---------------------------------------------------------------------------------
# The streaming spec: what a run gets from OnlineImages
# ---------------------------------------------------------------------------------

def _producer_of(rows, passes, stop):
    """A fetcher that walks `rows` records `passes` times, then stays alive."""

    def produce(dataset, *, data_queue, **kwargs):
        for _ in range(passes):
            for index in range(rows):
                data_queue.put({**_sample(index), "image": np.full((4, 4, 3),
                                                                   index + 1, np.uint8)})
        stop.wait(5)

    return produce


def _online_spec(monkeypatch, rows, passes, stop):
    """OnlineImages over a stub table, with the fetcher pool replaced."""
    from dew.data import OnlineImages

    monkeypatch.setattr(online_loader, "parallel_media_loader",
                        _producer_of(rows, passes, stop))
    monkeypatch.setattr(online_loader, "load_rows", lambda sources: _StubDataset(rows))
    return OnlineImages(sources=("fake_online",), image_size=4,
                        loading=Loading(workers=1))


def test_the_streaming_spec_repeats_its_records_instead_of_ending(monkeypatch, stop):
    """The streamer is documented as endless: the pool walks its shards in a
    loop and a run bounds training by steps, so `for batch in loader` must not
    stop at the end of the dataset."""
    rows, batch, passes = 12, 4, 3
    loader = _online_spec(monkeypatch, rows, passes, stop).load(batch=batch).train()

    seen = [[int(v) for v in next(loader)["image"][:, 0, 0, 0]]
            for _ in range(rows // batch * passes)]
    records = [value for row in seen for value in row]

    assert len(records) == rows * passes
    assert sorted(records[:rows]) == list(range(1, rows + 1)), "a pass repeats a record"
    assert set(records[rows:]) == set(records[:rows]), "the stream reads them again"


def test_the_streaming_spec_reports_its_records_and_holds_nothing_out(monkeypatch, stop):
    """`records` counts rows, like every grain spec's, so a recipe divides it
    by the batch for steps per epoch; a streaming run holds nothing out, so
    nothing downstream may report a validation pass over a held-out split."""
    data = _online_spec(monkeypatch, 12, 1, stop).load(batch=4)

    assert data.records == 12 and data.batch == 4 and data.steps_per_epoch == 3
    assert data.val is None
    assert len(next(data.train())["image"]) == 4


def test_the_streaming_spec_opens_nothing_before_the_stream_is_asked_for(monkeypatch, stop):
    """Loading resolves the rows; the fetcher pool and its thread start with
    `train()`, and the stream cannot record a position."""
    started = []
    producer = _producer_of(4, 1, stop)

    def produce(dataset, **kwargs):
        started.append(dataset)
        producer(dataset, **kwargs)

    data = _online_spec(monkeypatch, 4, 1, stop).load(batch=4)
    monkeypatch.setattr(online_loader, "parallel_media_loader", produce)
    assert started == []

    stream = data.train()
    assert len(next(stream)["image"]) == 4
    assert len(started) == 1
    assert not hasattr(stream, "get_state")


def test_the_streaming_spec_stops_when_its_fetcher_is_gone(monkeypatch):
    """The one thing that does end the stream: nothing left to wait for.

    The spec leaves the iterator's queue timeout at a minute, so the wait
    before it looks at the fetcher is shortened here. What is under test is
    that the wait ends in StopIteration rather than in a batch of zeros.
    """
    def exhausted(dataset, *, data_queue, **kwargs):
        for index in range(4):
            data_queue.put({**_sample(index), "image": np.full((4, 4, 3), 1, np.uint8)})

    from dew.data import OnlineImages

    monkeypatch.setattr(online_loader, "parallel_media_loader", exhausted)
    monkeypatch.setattr(online_loader, "MediaBatchIterator",
                        functools.partial(MediaBatchIterator, queue_timeout=0.05))
    monkeypatch.setattr(online_loader, "load_rows", lambda sources: _StubDataset(4))
    loader = OnlineImages(sources=("fake_online",), image_size=4,
                          loading=Loading(workers=1)).load(batch=4).train()

    assert len(next(loader)["image"]) == 4
    with pytest.raises(StopIteration):
        next(loader)


class Mean(Objective):
    """One scalar fitted to the batch mean: the smallest objective that reads a
    batch, so what is under test is the streaming data path and nothing else."""

    ema = None

    def init(self, key):
        return {"params": {"level": jnp.zeros(())}}

    def loss(self, params, batch, step):
        pixels = (jnp.asarray(batch["image"], jnp.float32) - 127.5) / 127.5
        return jnp.mean((pixels - params["params"]["level"]) ** 2), Aux({})


def _run(data, *, steps, checkpoints=None, checkpoint_every=None):
    trainer = Trainer(Mean(), optax.sgd(0.1), key=jax.random.key(0),
                      mesh=MeshSpec(), layout=Layout(), checkpoints=checkpoints)
    return trainer.fit(data, steps=steps, log_every=steps, eval_every=None,
                       checkpoint_every=checkpoint_every)


def test_the_streaming_spec_cannot_report_a_position(monkeypatch, stop):
    """The iterator is the single answer to whether a run can checkpoint: the
    fetch stream carries no get_state, and `tokenized` does not invent one."""
    data = _online_spec(monkeypatch, 12, 1, stop).load(batch=4)

    stream = data.train()
    assert not hasattr(stream, "get_state") and not hasattr(stream, "set_state")


def test_a_streaming_run_trains_when_it_never_checkpoints(monkeypatch, stop):
    """The whole point of the gate: an online stream is trainable, six steps of
    it, as long as the run does not ask for a position it cannot have."""
    data = _online_spec(monkeypatch, 24, 6, stop).load(batch=8)

    state = _run(data, steps=6)

    assert int(state.step) == 6
    assert jnp.isfinite(state.params["params"]["level"])


def test_a_streaming_run_that_asks_for_checkpoints_is_refused(monkeypatch, stop, tmp_path):
    """And the other half: the refusal is by name, before any training."""
    data = _online_spec(monkeypatch, 24, 6, stop).load(batch=8)

    with pytest.raises(ValueError, match=r"checkpoint_every=None"):
        _run(data, steps=6, checkpoints=Checkpoints(str(tmp_path / "ckpt")),
             checkpoint_every=2)
