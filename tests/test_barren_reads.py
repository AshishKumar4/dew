"""Reads whose steps drop records under loading workers: a worker runs the steps and keeps each
record it drops as a marker, which the reading process drops, so a worker whose own inputs are
all dropped never holds up the others' (row 883)."""

from dataclasses import dataclass

import numpy as np
import pytest

from dew.data import Loading, Memory, Rejected, mix
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline

WORKERS = Loading(workers=3, threads=1, read_buffer=1, worker_buffer=1)


@dataclass(frozen=True)
class KeepsOne:
    """Keeps the record whose id is `kept` and rejects every other one, for good."""

    kept: int
    identity = "keeps-one@1"
    rejects = True

    def __call__(self, row):
        return row if int(row["id"]) == self.kept else Rejected("unkept")


@dataclass(frozen=True)
class KeepsEven:
    """Keeps the records whose id is even."""

    identity = "keeps-even@1"
    rejects = True

    def __call__(self, row):
        return row if int(row["id"]) % 2 == 0 else Rejected("odd")


def _ids(reader, batches: int) -> list[int]:
    try:
        return [int(value) for _ in range(batches) for value in np.asarray(next(reader)["id"])]
    finally:
        reader.close()


@pytest.mark.parametrize("repeat", ["written after the filter", "the share's own"])
@pytest.mark.parametrize("read", ["indexed", "memory"])
def test_workers_whose_inputs_a_filter_drops_read_on_from_their_shares_others(read, repeat):
    """Twelve inputs and three loading workers, a filter that keeps only the sixth: the two workers
    whose first inputs hold no kept one read the share's other inputs in their next passes, so the
    read goes on, delivering the kept record, whether the repeat is written after the filter or
    is the share's own."""
    kept = Pipeline(Memory([{"id": index} for index in range(12)])).filter_map(KeepsOne(5))
    chain = (kept.repeat() if repeat == "written after the filter" else kept).batch(1)
    reader = compile(chain, read=read, loading=WORKERS).train(DataPartition())
    assert _ids(reader, 6) == [5] * 6


def test_streamed_workers_whose_shards_a_filter_drops_read_on_from_their_shares_others():
    """Twelve one-row shards read by three loading workers, a random step and then a filter that
    keeps only one row: the workers whose shards hold no kept row read the share's other shards in
    their next passes, so the read goes on."""
    rows = Memory([{"id": index} for index in range(12)], shard_rows=1)
    chain = (Pipeline(rows).random_map(Untouched(), identity="untouched@1").filter_map(KeepsOne(5))
             .repeat().batch(1))
    reader = compile(chain, read="streamed", loading=WORKERS).train(DataPartition())
    assert _ids(reader, 6) == [5] * 6


@dataclass(frozen=True)
class Untouched:
    """A random step that leaves its record as it is, which keeps a stream's repeat where it is
    written, after the shards each worker reads."""

    identity = "untouched@1"

    def __call__(self, row, rng):
        return row


def test_a_repeated_read_by_workers_reads_each_kept_input_once_a_pass():
    """Two passes over twelve inputs by three loading workers, a filter keeping the even ones:
    each kept input is read once a pass, and nothing else."""
    evens = Pipeline(Memory([{"id": index} for index in range(12)])).filter_map(KeepsEven())
    chain = evens.repeat(2).batch(1)
    reader = compile(chain, read="indexed", loading=WORKERS).train(DataPartition())
    try:
        ids = sorted(int(value) for batch in reader for value in np.asarray(batch["id"]))
    finally:
        reader.close()
    assert ids == sorted(2 * list(range(0, 12, 2)))


@pytest.mark.parametrize("read", ["indexed", "streamed"])
def test_a_mixture_of_corpora_each_kept_in_one_workers_inputs_reads_on(read):
    """Two corpora of twelve inputs each, read by three loading workers, the first keeping only its
    sixth input and the second only its seventh, which fall to different workers: the mixture
    reads on, delivering both."""
    def corpus(name, kept):
        rows = Memory([{"id": index} for index in range(12)], shard_rows=1)
        return Pipeline(rows, name=name).filter_map(KeepsOne(kept)).repeat()

    reader = compile(mix([corpus("fifth", 5), corpus("sixth", 6)]).batch(1), read=read,
                     loading=WORKERS).train(DataPartition())
    assert set(_ids(reader, 6)) == {5, 6}


def test_a_workers_read_of_a_dropping_corpus_resumes_where_it_stopped():
    """A read by three loading workers of a corpus whose filter drops records, saved after its
    workers' first passes, resumes on the records the uninterrupted read reads next."""
    evens = Pipeline(Memory([{"id": index} for index in range(12)])).filter_map(KeepsEven())
    chain = evens.repeat().batch(1)
    data = compile(chain, read="indexed", loading=WORKERS)
    expected = _ids(data.train(DataPartition()), 16)
    stopped = data.train(DataPartition())
    try:
        for _ in range(10):
            next(stopped)
        saved = stopped.get_state()
    finally:
        stopped.close()
    resumed = data.train(DataPartition())
    resumed.set_state(saved)
    assert _ids(resumed, 6) == expected[10:]
