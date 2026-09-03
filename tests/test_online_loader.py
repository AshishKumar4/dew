"""Streaming loader tests: what reaches the model when fetching goes wrong.

The shared environment has no HF `datasets`, which is the point of one of
these: importing the streaming stack must not need the `streaming` extra. Most
of the rest replace the fetcher pool with a stub producer; the two that do run
the real worker pool stub the fetch instead. Nothing here touches the network,
and every url is under .invalid so a stray fetch could not reach a host.
"""

import functools
import multiprocessing
import queue
import sys
import threading

import numpy as np
import PIL.Image
import pytest

from dew.data import online_loader
from dew.data.online_loader import (
    DROPPED_SAMPLE, MediaBatchIterator, OnlineStreamingDataLoader,
)

BATCH = 4
# Comfortably longer than the iterator's timeout below, so the slow producer
# really does make it wait.
SLOW = 0.15
# The pool tests: 12 rows over 2 workers, one batch per pass.
POOL_ROWS = 12
POOL_BATCH = 8


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


def _url_index(url):
    return int(url.rsplit("/", 1)[1].split(".")[0])


def _fetch_stub(url, timeout=None, retries=0):
    """A synthetic image the default processor accepts: over the minimum size,
    square, and not a solid colour."""
    pixels = np.random.RandomState(_url_index(url)).randint(0, 256, (48, 48, 3), np.uint8)
    return PIL.Image.fromarray(pixels)


def _fetch_stub_every_third_fails(url, timeout=None, retries=0):
    return None if _url_index(url) % 3 == 2 else _fetch_stub(url)


def _pool_iterator(passes):
    """A MediaBatchIterator on the real fetcher pool."""
    return MediaBatchIterator(_StubDataset(POOL_ROWS, passes=passes),
                              batch_size=POOL_BATCH, num_workers=2, num_threads=2,
                              image_shape=(64, 64), min_image_shape=(32, 32),
                              queue_timeout=10)


def _require_fork():
    """The stub fetch reaches the pool workers by fork inheritance; a spawned
    worker would re-import the real module and go looking for a network."""
    if multiprocessing.get_context().get_start_method() != "fork":
        pytest.skip("the fetch stub only reaches forked workers")


# ---------------------------------------------------------------------------------
# Slow fetching waits; it never invents data
# ---------------------------------------------------------------------------------

def test_slow_fetching_waits_instead_of_fabricating_samples(monkeypatch, stop, capsys):
    """A queue timeout used to hand back zeros captioned "Timeout occurred
    while waiting for sample", which then trained as data."""
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
    """The batch worker used to print the exception and carry on, so a run kept
    training on whatever the next iteration produced."""
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
# The real worker pool, with the fetch stubbed
# ---------------------------------------------------------------------------------

def test_the_worker_pool_delivers_real_samples(monkeypatch):
    """parallel_media_loader passed its multiprocessing.Queue to pool.map as an
    argument, which multiprocessing refuses to pickle, so the pool died before
    its first fetch and every batch was fabricated zeros."""
    _require_fork()
    monkeypatch.setattr(online_loader, "fetch_single_image", _fetch_stub)

    batch = next(_pool_iterator(passes=2))

    assert len(batch) == POOL_BATCH
    assert all(sample["image"].shape == (64, 64, 3) for sample in batch)
    assert all(int(sample["image"].max()) > 0 for sample in batch)
    assert {sample["caption"] for sample in batch} <= {
        f"caption {index}" for index in range(POOL_ROWS)}


def test_drops_in_the_worker_processes_reach_the_iterators_counter(monkeypatch):
    _require_fork()
    monkeypatch.setattr(online_loader, "fetch_single_image",
                        _fetch_stub_every_third_fails)

    iterator = _pool_iterator(passes=6)
    batches = [next(iterator) for _ in range(3)]

    assert all(len(batch) == POOL_BATCH for batch in batches)
    assert all("image" in sample for batch in batches for sample in batch)
    # A pass over the 12 rows queues 8 samples and 4 markers, and pool.map
    # finishes a pass before the next starts, so three batches of real samples
    # come with 12 markers, give or take one pass of cross-process flush order.
    assert 8 <= iterator.dropped <= 16

# ---------------------------------------------------------------------------------
# The streaming extra
# ---------------------------------------------------------------------------------

def test_loading_a_dataset_by_path_asks_for_the_streaming_extra(monkeypatch):
    """Importing this module must work without HF datasets; only actually
    loading a dataset needs it, and it has to say so.

    The absence is simulated so the test states the behaviour whether or not
    the extra is installed in the environment that runs it.
    """
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError, match=r"dew-ml\[streaming\]"):
        OnlineStreamingDataLoader("some/hf/dataset")


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
# The streaming factory: what a run gets from get_dataset_online
# ---------------------------------------------------------------------------------

class _StubTokenizer:
    """The collate's tokenizer; the real one downloads CLIP."""

    def __init__(self, tensor_type="np"):
        pass

    def __call__(self, captions):
        n = len(captions)
        return {"input_ids": np.zeros((n, 4), np.int32),
                "attention_mask": np.ones((n, 4), np.int32)}


def _producer_of(rows, passes, stop):
    """A fetcher that walks `rows` records `passes` times, then stays alive."""

    def produce(dataset, *, data_queue, **kwargs):
        for _ in range(passes):
            for index in range(rows):
                data_queue.put({**_sample(index), "image": np.full((4, 4, 3),
                                                                   index + 1, np.uint8)})
        stop.wait(5)

    return produce


def _online_factory(monkeypatch, rows, passes, stop, batch=4):
    from dew.data import dataloaders
    from dew.data.registry import onlineDatasetMap

    monkeypatch.setattr(dataloaders, "AutoTextTokenizer", _StubTokenizer)
    monkeypatch.setattr(online_loader, "parallel_media_loader",
                        _producer_of(rows, passes, stop))
    monkeypatch.setitem(onlineDatasetMap, "fake_online",
                        {"source": _StubDataset(rows)})
    return dataloaders.get_dataset_online("fake_online", batch_size=batch,
                                          worker_count=1, image_scale=4)


def test_the_streaming_factory_repeats_its_records_instead_of_ending(monkeypatch, stop):
    """The streamer is documented as endless: the pool walks its shards in a
    loop and a run bounds training by steps, so `for batch in loader` must not
    stop at the end of the dataset."""
    rows, batch, passes = 12, 4, 3
    data = _online_factory(monkeypatch, rows, passes, stop, batch=batch)
    loader = data["train"]()

    seen = [[int(v) for v in next(loader)["image"][:, 0, 0, 0]]
            for _ in range(rows // batch * passes)]
    records = [value for row in seen for value in row]

    assert len(records) == rows * passes
    assert sorted(records[:rows]) == list(range(1, rows + 1)), "a pass repeats a record"
    assert set(records[rows:]) == set(records[:rows]), "the stream reads them again"


def test_the_streaming_factory_reports_its_length_in_records(monkeypatch, stop):
    """train_len is records, like every grain factory's, and a recipe divides
    it by the batch size for steps per epoch."""
    data = _online_factory(monkeypatch, 12, 1, stop, batch=4)

    assert data["train_len"] == 12
    assert data["local_batch_size"] == 4 and data["global_batch_size"] == 4
    assert len(data["train"]()) == 12


def test_the_streaming_factory_stops_when_its_fetcher_is_gone(monkeypatch):
    """The one thing that does end the stream: nothing left to wait for.

    The factory leaves the iterator's queue timeout at a minute, so the wait
    before it looks at the fetcher is shortened here. What is under test is
    that the wait ends in StopIteration rather than in a batch of zeros.
    """
    def exhausted(dataset, *, data_queue, **kwargs):
        for index in range(4):
            data_queue.put({**_sample(index), "image": np.full((4, 4, 3), 1, np.uint8)})

    from dew.data import dataloaders
    from dew.data.registry import onlineDatasetMap

    monkeypatch.setattr(dataloaders, "AutoTextTokenizer", _StubTokenizer)
    monkeypatch.setattr(online_loader, "parallel_media_loader", exhausted)
    monkeypatch.setattr(online_loader, "MediaBatchIterator",
                        functools.partial(MediaBatchIterator, queue_timeout=0.05))
    monkeypatch.setitem(onlineDatasetMap, "fake_online", {"source": _StubDataset(4)})
    loader = dataloaders.get_dataset_online("fake_online", batch_size=4,
                                            worker_count=1, image_scale=4)["train"]()

    assert len(next(loader)["image"]) == 4
    with pytest.raises(StopIteration):
        next(loader)
