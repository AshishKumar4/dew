"""Reads whose steps drop records: each loading worker reads every input of its share in turn, so
one whose own inputs are all dropped reads on from the others' (row 883); a corpus or a share none
of whose inputs is ever kept is refused as the read opens, not waited on forever."""

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


def test_a_repeated_read_by_rotating_workers_reads_each_kept_input_once_a_pass():
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


@pytest.mark.parametrize("read", ["indexed", "memory", "streamed"])
def test_a_corpus_none_of_whose_inputs_is_kept_is_refused_as_its_read_opens(read):
    """A corpus whose filter keeps none of its inputs, mixed with one that keeps all: the read is
    refused as it opens, naming the corpus, rather than waiting on it forever."""
    kept = Pipeline(Memory([{"id": 2 * index} for index in range(6)], shard_rows=1), name="kept")
    dropped = Pipeline(Memory([{"id": 2 * index + 1} for index in range(6)], shard_rows=1),
                       name="dropped").filter_map(KeepsEven())
    chain = mix([kept.repeat(), dropped.repeat()], weights=[0.5, 0.5]).batch(2)
    reader = compile(chain, read=read, loading=Loading(workers=0, threads=1, read_buffer=1)).train(
        DataPartition())
    try:
        with pytest.raises(ValueError):
            next(reader)
    finally:
        reader.close()


def test_a_share_none_of_whose_inputs_is_kept_is_refused_as_its_read_opens():
    """Eight inputs in two shares and a filter keeping the even ones: the second share holds the
    odd ones, none kept, so its read is refused as it opens; the first reads its own."""
    chain = Pipeline(Memory([{"id": index} for index in range(8)])).filter_map(KeepsEven()).batch(1)
    data = compile(chain, read="indexed", loading=Loading(workers=0, threads=1, read_buffer=1))
    assert set(_ids(data.train(DataPartition(index=0, count=2)), 4)) <= {0, 2, 4, 6}
    reader = data.train(DataPartition(index=1, count=2))
    try:
        with pytest.raises(ValueError):
            next(reader)
    finally:
        reader.close()


def test_a_rotating_workers_read_resumes_where_it_stopped():
    """A read by three rotating loading workers saved after its workers' first passes resumes on the
    records the uninterrupted read reads next."""
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
