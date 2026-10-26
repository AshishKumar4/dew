"""Reads whose steps drop records under loading workers: a worker runs the steps and keeps each
record it drops as a marker, which the reading process drops, so a worker whose own inputs are
all dropped never holds up the others' (row 883)."""

import json
from dataclasses import dataclass

import grain.python as grain
import numpy as np
import pytest
from reader_support import assert_resumes_where_it_stopped

from dew._records import record
from dew.data import Loading, Memory, Rejected, mix
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline
from dew.data.sources.grain import from_grain

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


@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
def test_a_repeated_read_by_workers_reads_each_kept_input_once_a_pass(tmp_path, cached):
    """Two passes over twelve inputs by three loading workers, a filter keeping the even ones, read
    as it is or through a cache of what it keeps: each kept input is read once a pass, and nothing
    else."""
    evens = Pipeline(Memory([{"id": index} for index in range(12)])).filter_map(KeepsEven())
    chain = (evens.cache(str(tmp_path / "cache")) if cached else evens).repeat(2).batch(1)
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
    """A read by three loading workers of a corpus whose filter drops records, in batches of three
    that span what the workers hand on, saved mid-way, resumes on the records the uninterrupted read
    reads next."""
    evens = Pipeline(Memory([{"id": index} for index in range(12)])).filter_map(KeepsEven())
    data = compile(evens.repeat().batch(3), read="indexed", loading=WORKERS)
    assert_resumes_where_it_stopped(lambda: data.train(DataPartition()), _batches, 5, 6)


def _batches(reader, count: int) -> list[list[int]]:
    return [np.asarray(next(reader)["id"]).tolist() for _ in range(count)]


@dataclass(frozen=True)
class Twice:
    """Each record twice, the second marked, one kept for each input."""

    identity = "twice@1"
    max_fan_out = 2

    def flat_map(self, row):
        return [row, {**row, "id": int(row["id"]) + 100}]


def test_a_worker_with_no_inputs_ends_and_the_others_read_on_and_resume():
    """Two inputs read by three loading workers, each expanded to two records of which a filter keeps
    the even ones, repeated after: the worker with no input ends its pass at once, the others read
    on, and a position saved mid-way resumes on what the uninterrupted read reads next."""
    rows = Pipeline(Memory([{"id": 0}, {"id": 1}])).flat_map(Twice()).filter_map(KeepsEven())
    data = compile(rows.repeat().batch(2), read="indexed", loading=WORKERS)
    expected = assert_resumes_where_it_stopped(lambda: data.train(DataPartition()), _batches, 3, 4)[0]
    assert {value for batch in expected for value in batch} == {0, 100}


def test_a_written_workers_split_reads_on_whatever_the_loadings_workers():
    """Workers written into the chain, after a filter keeping one of twelve inputs, with the
    loading's own workers none: the read goes on, in a corpus of a mixture as on its own."""
    def kept(name):
        return Pipeline(Memory([{"id": index} for index in range(12)]), name=name).filter_map(KeepsOne(5))

    alone = kept("alone").repeat().workers(3).batch(1)
    loading = Loading(workers=0, threads=1, read_buffer=1)
    assert _ids(compile(alone, read="indexed", loading=loading).train(DataPartition()), 4) == [5] * 4
    mixed = mix([kept("written").repeat().workers(3), Pipeline(Memory([{"id": 7}]), name="other").repeat()])
    reader = compile(mixed.batch(1), read="indexed", loading=loading).train(DataPartition())
    assert set(_ids(reader, 8)) == {5, 7}


@dataclass(frozen=True)
class OnlyFive:
    def __call__(self, row):
        return int(row["id"]) == 5


def _grain_filter(rows):
    return rows.filter(OnlyFive())


def test_a_grain_filter_in_workers_reads_on():
    """A Grain step of the caller's that keeps one of twelve indexed inputs, read by three loading
    workers: what it drops goes on as markers, so the read goes on."""
    rows = Pipeline(Memory([{"id": index} for index in range(12)]))
    chain = rows.grain(_grain_filter, identity="only-five@1")
    reader = compile(chain.batch(1), read="indexed", loading=WORKERS).train(DataPartition())
    assert _ids(reader, 4) == [5] * 4


class Renamed(grain.MapTransform):
    """A Grain map of the caller's that reads its record's id."""

    def map(self, row):
        return {"id": int(row["id"]), "renamed": np.int32(1)}


def _grain_rename(rows):
    return rows.map(Renamed())


@pytest.mark.parametrize("before", ["a filter", "a store missing inputs"])
def test_a_grain_map_in_a_workers_read_reads_records_not_markers(before):
    """A Grain map of the caller's after a filter of Dew's, or over a store missing some inputs, read
    by three loading workers: the map reads only records, and the read delivers the kept ones."""
    records = [None if index % 3 == 0 else {"id": index} for index in range(12)]
    if before == "a filter":
        rows = Pipeline(Memory([{"id": index} for index in range(12)])).filter_map(KeepsEven())
        kept = set(range(0, 12, 2))
    else:
        rows = Pipeline(from_grain(grain.MapDataset.source(records), name="missing-thirds"))
        kept = {index for index in range(12) if index % 3}
    chain = rows.grain(_grain_rename, identity="renamed@1").repeat().batch(1)
    reader = compile(chain, read="indexed", loading=WORKERS).train(DataPartition())
    assert set(_ids(reader, 16)) == kept


class Parsed(grain.MapTransform):
    """A record from its JSON bytes, kept where its id is even."""

    identity = "parsed@1"
    rejects = True

    def map(self, row):
        record = json.loads(row)
        return record if record["id"] % 2 == 0 else Rejected("odd")


def test_a_workers_read_of_bytes_hands_them_on_before_a_prefetch():
    """A store of JSON bytes, prefetched and then parsed by a step that drops the odd ones, read by
    three loading workers: the workers hand the bytes on as they are, and the read parses them."""
    rows = from_grain(grain.MapDataset.source([json.dumps({"id": index}).encode() for index in range(12)]),
                      name="json-bytes")
    chain = Pipeline(rows).prefetch(1).filter_map(Parsed()).repeat().batch(1)
    reader = compile(chain, read="indexed", loading=WORKERS).train(DataPartition())
    assert set(_ids(reader, 12)) == set(range(0, 12, 2))


def _kept(name: str) -> Pipeline:
    rows = Pipeline(Memory([{"id": index} for index in range(12)]), name=name)
    return rows.filter_map(KeepsEven()).repeat()


def test_a_mixture_edit_that_moves_a_kept_corpus_to_other_workers_is_refused():
    """A mixture of two corpora read by three loading workers, the first in two and the second in
    one: a position saved there and restored where the second corpus leads, in two, is refused,
    and one restored where a third corpus joins in none, the first two as they were, reads on."""
    loading = Loading(workers=3, threads=1, read_buffer=1, worker_buffer=1)
    saved_from = compile(mix([_kept("a"), _kept("b")]).batch(1), read="indexed", loading=loading)
    reader = saved_from.train(DataPartition())
    try:
        next(reader)
        saved = reader.get_state()
    finally:
        reader.close()
    moved = compile(mix([_kept("b"), _kept("c")]).batch(1), read="indexed", loading=loading)
    moved = moved.train(DataPartition())
    try:
        with pytest.raises(ValueError):
            moved.set_state(saved)
    finally:
        moved.close()
    joined = compile(mix([_kept("a"), _kept("b"), _kept("c")]).batch(1), read="indexed",
                     loading=Loading(workers=2, threads=1, read_buffer=1, worker_buffer=1))
    two = compile(mix([_kept("a"), _kept("b")]).batch(1), read="indexed",
                  loading=Loading(workers=2, threads=1, read_buffer=1, worker_buffer=1))
    before = two.train(DataPartition())
    try:
        next(before)
        state = before.get_state()
    finally:
        before.close()
    resumed = joined.train(DataPartition())
    resumed.set_state(state)
    assert set(_ids(resumed, 4)) <= set(range(0, 12, 2))


def test_a_nested_mixture_reads_in_the_workers_it_writes():
    """A mixture written to read in three workers, inside another, with the loading's own workers
    none: its two corpora read in two and one of them, the other corpus in the reading process."""
    inner = mix([_kept("a"), _kept("b")]).workers(3)
    loading = Loading(workers=0, threads=1, read_buffer=1)
    data = compile(mix([inner, _kept("c")]).batch(1), read="indexed", loading=loading)
    reader = data.train(DataPartition())
    # The position a run saves records the workers each corpus read in.
    plan = record(json.loads(reader.get_state())["dew_data_plan"], "saved plan")
    pools = record(record(plan["topology"], "topology")["pools"], "pools")
    assert set(_ids(reader, 8)) <= set(range(0, 12, 2))
    assert sorted(pools.values()) == [0, 1, 2]
