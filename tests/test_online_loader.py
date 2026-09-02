"""Streaming loader tests: what reaches the model when fetching goes wrong.

The shared environment has no HF `datasets`, which is the point of one of
these: importing the streaming stack must not need the `streaming` extra. The
rest replace the fetcher pool with a stub producer, so nothing here touches the
network or a process pool.
"""

import queue
import threading

import numpy as np
import pytest

from dew.data import online_loader
from dew.data.online_loader import (
    DROPPED_SAMPLE, MediaBatchIterator, OnlineStreamingDataLoader,
)

BATCH = 4
# Comfortably longer than the iterator's timeout below, so the slow producer
# really does make it wait.
SLOW = 0.15


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
    """As much of an HF dataset as the loader's constructor reads."""

    def __init__(self, size):
        self.size = size

    def shard(self, num_shards, index):
        return self

    def __len__(self):
        return self.size


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
# The streaming extra
# ---------------------------------------------------------------------------------

def test_loading_a_dataset_by_path_asks_for_the_streaming_extra():
    """Importing this module must work without HF datasets; only actually
    loading a dataset needs it, and it has to say so."""
    with pytest.raises(ImportError, match=r"dew-ml\[streaming\]"):
        OnlineStreamingDataLoader("some/hf/dataset")
