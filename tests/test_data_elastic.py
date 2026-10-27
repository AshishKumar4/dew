"""Elastic reads (`Pipeline.elastic`): a per-share order read so a run resumes on any reader count.

A filtered, drawn stream is read per share, so its position holds only on the shares that saved it.
Read elastically, every reader reads the order one reader reads and keeps its share of each batch,
so its rows are one reader's and its position, the same on every reader, resumes on any count. The
pool's own resume is tests/test_multiprocess.py's
(`test_an_elastic_filtered_stream_resumes_on_another_process_count`)."""

import dataclasses
import itertools
import json
import threading
from typing import ClassVar

import numpy as np
import pytest
from data_support import WideDraw, listed

from dew import position
from dew.batch import VALID_ROWS
from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading
from dew.data.pipeline import Pipeline
from dew.data.sources import Files, Memory

READ = Loading(workers=0, threads=4, read_buffer=4)
BATCH = 8


def odd(row) -> bool:
    return bool(int(row["id"]) % 2)


def unthird(row) -> bool:
    """Keeps a record whose id three does not divide: some of every share's, at two or four shares,
    where a share of every other record read per share would keep none and read on forever."""
    return bool(int(row["id"]) % 3)


@dataclasses.dataclass(frozen=True)
class Squared:
    """The record's id squared, and the thread that squared it."""

    def __call__(self, row):
        return {**row, "squared": np.int64(row["id"]) ** 2, "thread": np.int64(threading.get_ident())}


def stream(records: int = 96, *, elastic: bool = True, epochs: int | None = None) -> Pipeline:
    """A shuffled stream whose filter and draw make it a per-share order, then a map of each row."""
    chain = (Pipeline(Memory(tuple({"id": np.int64(index)} for index in range(records))))
             .shuffle(5).repeat(epochs).filter(unthird).random_map(WideDraw()).map(Squared()).batch(BATCH))
    return chain.elastic() if elastic else chain


def rows(batch) -> list[tuple[int, int, int]]:
    """A batch's rows as (id, draw, squared), without the thread that mapped them."""
    return list(zip(np.asarray(batch["id"]).tolist(), np.asarray(batch["draw"]).tolist(),
                    np.asarray(batch["squared"]).tolist(), strict=True))


def reader(chain: Pipeline, index: int, count: int):
    return compile(chain, loading=READ, seed=3).train(DataPartition(index, count))


@pytest.mark.parametrize("count", [2, 4])
def test_each_elastic_reader_reads_its_share_of_each_batch_one_reader_reads(count):
    """Every reader of an elastic read keeps every count-th row of the batch one reader reads: its
    filter's survivors, their draws and the map of each, in one reader's order, where a per-share
    read deals the shuffled stream out first and filters and draws each share apart."""
    one = reader(stream(), 0, 1)
    whole = [rows(batch) for batch in itertools.islice(one, 4)]
    shares = [reader(stream(), index, count) for index in range(count)]
    for step in range(4):
        held = [rows(next(share)) for share in shares]
        assert [row for index in range(count) for row in held[index]] == [
            whole[step][at] for index in range(count) for at in range(index, BATCH, count)]
    per_share = [rows(next(reader(stream(elastic=False), index, count))) for index in range(count)]
    assert sorted(itertools.chain(*per_share)) != sorted(whole[0])


@dataclasses.dataclass(frozen=True)
class Counting:
    """The record as it is, its id noted in `calls`."""

    calls: ClassVar[list[int]] = []

    def __call__(self, row):
        Counting.calls.append(int(row["id"]))
        return row


def test_an_elastic_readers_maps_after_its_last_filter_run_on_its_own_rows_alone():
    """The map written after the read's last filter runs on a reader's own rows alone, not on the
    other readers' rows: half the rows of a batch on each of two."""
    chain = (Pipeline(Memory(tuple({"id": np.int64(index)} for index in range(64)))).shuffle(5).repeat()
             .filter(odd).map(Counting()).batch(BATCH).elastic())
    first = reader(chain, 0, 2)
    Counting.calls.clear()
    ids = [int(value) for value in np.asarray(next(first)["id"])]
    assert sorted(Counting.calls) == sorted(ids) and len(ids) == BATCH // 2


@pytest.mark.parametrize(("saved", "resumed"), [(2, 1), (2, 4), (4, 2)])
def test_an_elastic_position_is_every_readers_and_resumes_on_another_count(saved, resumed):
    """After three batches every reader of an elastic read holds the same position, of the kind a
    checkpoint lets any reader count read (`position.translates`); readers of another count restored
    from it read the batches one reader reads from there, each its share."""
    shares = [reader(stream(), index, saved) for index in range(saved)]
    for share in shares:
        listed(share, 3)
    states = {share.get_state() for share in shares}
    assert len(states) == 1
    (state,) = states
    assert position.translates(state) and position.decode(state) is None
    one = reader(stream(), 0, 1)
    whole = [rows(batch) for batch in itertools.islice(one, 5)][3:]
    again = [reader(stream(), index, resumed) for index in range(resumed)]
    for share in again:
        share.set_state(state)
    for step in range(2):
        held = [rows(next(share)) for share in again]
        assert [row for index in range(resumed) for row in held[index]] == [
            whole[step][at] for index in range(resumed) for at in range(index, BATCH, resumed)]


def test_a_finite_elastic_read_ends_on_every_reader_at_once_its_last_batch_filled():
    """A read of one pass (a training read without a repeat reads on without end) whose last batch
    holds fewer rows than the batch: every reader still reads it, its share's rows real and the rest
    filled and marked, and every reader ends after it together."""
    chain = (Pipeline(Memory(tuple({"id": np.int64(index)} for index in range(22)))).repeat(1).filter(odd)
             .batch(BATCH, drop_remainder=False).elastic())
    batches = [list(reader(chain, index, 4)) for index in range(4)]
    assert {len(read) for read in batches} == {2}
    last = [np.asarray(read[-1][VALID_ROWS]).tolist() for read in batches]
    real = [int(np.asarray(read[-1]["id"])[0]) for read in batches]
    # The last batch holds 17, 19 and 21: the fourth reader holds none of them.
    assert last == [[True, False], [True, False], [True, False], [False, False]]
    assert real[:3] == [17, 19, 21]


def worker_reader(chain: Pipeline, index: int, count: int, workers: int):
    loading = Loading(workers=workers, threads=2, read_buffer=2, worker_buffer=2)
    return compile(chain, read="indexed", loading=loading, seed=3).train(DataPartition(index, count))


@pytest.mark.parametrize(("saved", "resumed"), [((2, 3), (4, 2)), ((1, 2), (2, 0))],
                         ids=["2-readers-3-workers-to-4-readers-2-workers",
                              "1-reader-2-workers-to-2-readers-none"])
def test_an_elastic_read_in_loading_workers_reads_one_readers_order_and_resumes_on_any_count(saved, resumed):
    """Loading workers read an elastic read's inputs in their order, a batch's count at a time, so
    every reader's rows are those one reader reads in threads alone, whatever the count of either,
    and the position readers of one count save in workers of one count resumes readers of another
    in another count of workers, in the batches one reader reads from there."""
    one = reader(stream(), 0, 1)
    whole = [rows(batch) for batch in itertools.islice(one, 5)]
    (readers, workers), (again, more) = saved, resumed

    def assert_one_readers(shares, count, step):
        held = [rows(next(share)) for share in shares]
        assert [row for index in range(count) for row in held[index]] == [
            whole[step][at] for index in range(count) for at in range(index, BATCH, count)]

    shares = [worker_reader(stream(), index, readers, workers) for index in range(readers)]
    try:
        for step in range(3):
            assert_one_readers(shares, readers, step)
        states = {share.get_state() for share in shares}
    finally:
        for share in shares:
            share.close()
    assert len(states) == 1
    (state,) = states
    shares = [worker_reader(stream(), index, again, more) for index in range(again)]
    try:
        for share in shares:
            share.set_state(state)
        for step in range(3, 5):
            assert_one_readers(shares, again, step)
    finally:
        for share in shares:
            share.close()


def test_an_elastic_stream_refuses_loading_workers_and_elastic_and_per_share_positions_each_other(tmp_path):
    """A streamed elastic read refuses loading workers, which read a stream's shards each their own;
    elastic and per-share positions each refuse the other's read."""
    lines = "".join(json.dumps({"id": index}) + "\n" for index in range(96))
    (tmp_path / "ids.jsonl").write_text(lines)
    streamed = (Pipeline(Files(str(tmp_path / "ids.jsonl"), format="jsonl")).repeat().filter(odd).batch(BATCH)
                .elastic().read("streamed"))
    with pytest.raises(ValueError):
        compile(streamed, loading=Loading(workers=2), seed=3).train(DataPartition(0, 2))
    shared = reader(stream(elastic=False), 0, 2)
    listed(shared, 1)
    elastic = reader(stream(), 0, 2)
    with pytest.raises(ValueError):
        elastic.set_state(shared.get_state())
    listed(elastic, 1)
    with pytest.raises(ValueError):
        reader(stream(elastic=False), 0, 2).set_state(elastic.get_state())


def test_a_per_share_read_whose_filter_rejects_its_whole_share_is_refused_and_an_elastic_one_reads():
    """Read by two shares, a shuffled endless repeat of ids is cut to the even ids for the first share
    before its filter keeps the odd ones, so that share's filter rejects every record of every pass:
    its reader refuses the read once two whole passes are rejected, where it read on forever. The read
    runs on a thread held to a deadline, so that a reader that reads on fails this test. Read
    elastically, the same chain reads the odd ids on that share."""
    chain = Pipeline(Memory(tuple({"id": np.int64(index)} for index in range(96)))).shuffle(5).repeat()
    starved = chain.filter(odd).batch(BATCH)
    outcome = []

    def read():
        try:
            next(reader(starved, 0, 2))
        except ValueError as refused:
            outcome.append(refused)

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    thread.join(60)
    assert not thread.is_alive(), "the reader read on with every record of its share rejected"
    assert len(outcome) == 1 and isinstance(outcome[0], ValueError)
    ids = np.asarray(next(reader(starved.elastic(), 0, 2))["id"])
    assert len(ids) == BATCH // 2 and all(int(value) % 2 for value in ids)
