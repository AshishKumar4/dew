"""Repeats: a corpus read over and over, its passes, their draws and positions, and a pass rewound."""

import dataclasses
import itertools

import grain.python as grain
import numpy as np
import pytest

from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading
from dew.data.pipeline import Pipeline
from dew.data.sources import Memory

READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)


@dataclasses.dataclass(frozen=True)
class Draw(grain.RandomMapTransform):
    def random_map(self, element, rng):
        return {**element, "draw": np.int64(rng.integers(0, 2**32))}


def source(count=24):
    return Memory(tuple({"id": np.int32(index)} for index in range(count)))


class Cursor(grain.IterDataset):
    """A Grain dataset whose position names its own fields as it likes: here the place it has read
    to, under `counts`."""

    def __init__(self, rows):
        super().__init__()
        self.rows = rows

    def __iter__(self):
        return CursorRead(self.rows)


class CursorRead(grain.DatasetIterator):
    def __init__(self, rows):
        super().__init__()
        self.rows, self.at = rows, 0

    def __next__(self):
        if self.at == len(self.rows):
            raise StopIteration
        self.at += 1
        return self.rows[self.at - 1]

    def get_state(self):
        return {"corpus": "cursor", "counts": self.at}

    def set_state(self, state):
        self.at = state["counts"]


def test_a_repeat_rewinds_a_native_grain_position_as_the_dataset_wrote_it():
    """A repeat that rewinds a pass keeps its corpus's counts, and leaves the rest of the position
    as the iterators wrote it, a Grain dataset's own fields whatever their names."""
    from dew.data import from_grain

    rows = [{"id": np.int32(index)} for index in range(2)]
    data = compile(from_grain(Cursor(rows), "native").take(2).repeat().batch(1), read="streamed",
                   loading=READ)
    stream = data.train(DataPartition())
    try:
        assert [int(next(stream)["id"][0]) for _ in range(5)] == [0, 1, 0, 1, 0]
    finally:
        stream.close()


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_corpus_smaller_than_a_batch_fills_one_by_its_repeats(read):
    """A corpus of two records repeated fills a batch of eight, as Grain's repeat and batch do,
    and a run counted in steps reads it; a run counted in epochs is refused, since one pass
    over the two records fills no batch. Repeated after the batch, a pass holds no batch to
    replay, so the read is refused rather than replaying nothing forever, whatever limit
    follows, in every read; eight records make a batch either way."""
    for count, repeated in ((2, "before"), (2, "after"), (8, "after"), (2, "after, then take"),
                            (8, "after, then take")):
        rows = Pipeline(Memory([{"id": np.int32(index)} for index in range(count)]))
        chain = rows.repeat().batch(8) if repeated == "before" else rows.batch(8).repeat()
        data = compile(chain.take(1) if repeated.endswith("take") else chain, read=read, loading=READ)
        if count < 8 and repeated.startswith("after"):
            with pytest.raises(ValueError):
                refused = data.train(DataPartition())
                try:
                    next(refused)
                finally:
                    refused.close()
            continue
        stream = data.train(DataPartition())
        try:
            assert sorted(int(value) for value in next(stream)["id"]) == sorted(
                index % count for index in range(8))
        finally:
            stream.close()
        if count < 8:
            with pytest.raises(ValueError):
                data.epoch_steps(1)


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
@pytest.mark.parametrize("batched", ["before", "after"])
def test_an_endless_repeat_resumed_at_the_end_of_a_pass_reads_on(read, batched):
    """A position saved just as a pass of an endless repeat ends restores to that end, and the
    read goes on into the next pass as the uninterrupted one does, rather than taking the
    restored end for a pass that read nothing; a repeat after the batch reads its passes as a
    stream in every read."""
    rows = Pipeline(Memory([{"id": np.int32(index)} for index in range(2)]))
    data = compile(rows.repeat().batch(2) if batched == "before" else rows.batch(2).repeat(), read=read,
                   loading=READ)

    def ids(stream, count: int) -> list[int]:
        return [int(value) for batch in itertools.islice(stream, count) for value in batch["id"]]

    whole = data.train(DataPartition())
    try:
        expected = ids(whole, 3)
    finally:
        whole.close()
    stopped = data.train(DataPartition())
    try:
        head = ids(stopped, 1)
        state = stopped.get_state()
    finally:
        stopped.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(state)
        assert head + ids(resumed, 2) == expected
    finally:
        resumed.close()


def test_a_random_step_after_an_endless_repeat_draws_as_after_a_finite_one():
    """A streamed repeat read endlessly guards its passes without changing what a random step
    after it draws: the passes a finite repeat reads draw the same values."""
    rows = Pipeline(source(2)).take(2)

    def draws(chain) -> list[int]:
        data = compile(chain.random_map(Draw()).batch(2), read="streamed", loading=READ)
        stream = data.train(DataPartition())
        try:
            return [int(value) for batch in itertools.islice(stream, 2) for value in batch["draw"]]
        finally:
            stream.close()

    assert draws(rows.repeat()) == draws(rows.repeat(2))


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_repeat_after_the_batch_replays_its_first_pass_draws_included(read):
    """A repeat after the batch repeats the batches the first pass made, a random step's draws
    and all, in every read, and a resume reads on as the uninterrupted run does."""
    data = compile(Pipeline(source(2)).random_map(Draw()).batch(2).repeat(), read=read, loading=READ)

    def batches(stream, count: int) -> list[tuple[list[int], list[int]]]:
        return [(np.asarray(batch["id"]).tolist(), np.asarray(batch["draw"]).tolist())
                for batch in itertools.islice(stream, count)]

    whole = data.train(DataPartition())
    try:
        expected = batches(whole, 3)
    finally:
        whole.close()
    assert expected[0] == expected[1] == expected[2]
    stopped = data.train(DataPartition())
    try:
        head = batches(stopped, 1)
        state = stopped.get_state()
    finally:
        stopped.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(state)
        assert head + batches(resumed, 2) == expected
    finally:
        resumed.close()
