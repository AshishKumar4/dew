"""Elastic reads (`Pipeline.elastic`): a per-share order read so a run resumes on any reader count.

A filtered, drawn stream is read per share, so its position holds only on the shares that saved it.
Read elastically, every reader reads the order one reader reads and keeps its share of each batch,
so its rows are one reader's and its position, the same on every reader, resumes on any count. The
pool's own resume is tests/test_multiprocess.py's
(`test_an_elastic_filtered_stream_resumes_on_another_process_count`)."""

import dataclasses
import itertools
import json
import multiprocessing
import sys
import threading
from concurrent import futures
from typing import ClassVar

import numpy as np
import pytest
from data_support import WideDraw, listed

from dew import position
from dew.batch import VALID_ROWS
from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading, Ramp, ramped
from dew.data.pipeline import Pipeline, Rejected
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
    """Every reader of an elastic read keeps every count-th row of the batch one reader of the chain
    without `.elastic()` reads: its filter's survivors, their draws, keyed by their place among the
    rows kept, and the map of each, where a per-share read deals the shuffled stream out first and
    filters and draws each share apart."""
    one = reader(stream(elastic=False), 0, 1)
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
    one = reader(stream(elastic=False), 0, 1)
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
    in another count of workers, in the batches one reader reads from there, without `.elastic()`."""
    one = reader(stream(elastic=False), 0, 1)
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


def jsonl_ids(tmp_path, *, elastic: bool = True) -> Pipeline:
    """A JSON-lines store of ids streamed, endlessly, the odd ones kept."""
    (tmp_path / "ids.jsonl").write_text("".join(json.dumps({"id": index}) + "\n" for index in range(96)))
    chain = Pipeline(Files(str(tmp_path / "ids.jsonl"), format="jsonl")).repeat().filter(odd).batch(BATCH)
    return (chain.elastic() if elastic else chain).read("streamed")


def test_an_elastic_stream_without_workers_reads_its_share_of_one_readers_batches(tmp_path):
    """A streamed elastic read without loading workers reads, on each of two readers, its share of
    the batches one reader of the stream reads."""
    one = [np.asarray(batch["id"]).tolist() for batch in itertools.islice(
        reader(jsonl_ids(tmp_path, elastic=False), 0, 1), 3)]
    shares = [reader(jsonl_ids(tmp_path), index, 2) for index in range(2)]
    for step in range(3):
        held = [np.asarray(next(share)["id"]).tolist() for share in shares]
        assert held == [one[step][0::2], one[step][1::2]]


def test_an_elastic_stream_refuses_loading_workers_and_elastic_and_per_share_positions_each_other(tmp_path):
    """A streamed elastic read refuses loading workers, which read a stream's shards each their own;
    elastic and per-share positions each refuse the other's read."""
    with pytest.raises(ValueError):
        compile(jsonl_ids(tmp_path), loading=Loading(workers=2), seed=3).train(DataPartition(0, 2))
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
    runs in a process held to a deadline (`starved`). Read elastically, the same chain reads the odd
    ids on that share."""
    chain = Pipeline(Memory(tuple({"id": np.int64(index)} for index in range(96)))).shuffle(5).repeat()
    starving = chain.filter(odd).batch(BATCH)
    assert starved(starving, 0, 2)
    ids = np.asarray(next(reader(starving.elastic(), 0, 2))["id"])
    assert len(ids) == BATCH // 2 and all(int(value) % 2 for value in ids)


@dataclasses.dataclass(frozen=True)
class Unkept:
    """Keeps a record whose `field`'s first value is not `value`."""

    field: str
    value: int

    def __call__(self, row):
        return int(np.asarray(row[self.field]).flat[0]) != self.value


@dataclasses.dataclass(frozen=True)
class Pieces:
    """A record as `count` pieces, each its id and its place among them."""

    count: int

    def __call__(self, row):
        return [{"id": row["id"], "piece": np.int64(piece)} for piece in range(self.count)]


@pytest.mark.parametrize("workers", [0, 2])
def test_an_elastic_expansion_reads_one_readers_pieces_in_or_out_of_workers(workers):
    """An elastic read by index whose records expand reads, in loading workers or none, the pieces
    one reader of the chain without `.elastic()` reads: its workers hand on its inputs before the
    expansion, which no index reads, and the reading process expands them in order."""
    chain = Pipeline(Memory(tuple({"id": np.int64(index)} for index in range(24)))).shuffle(5).repeat()
    expanded = chain.flat_map(Pieces(3)).filter(Unkept("piece", 1)).batch(BATCH)

    def pieces(batch):
        return list(zip(np.asarray(batch["id"]).tolist(), np.asarray(batch["piece"]).tolist(), strict=True))

    whole = [pieces(batch) for batch in itertools.islice(reader(expanded, 0, 1), 3)]
    shares = [worker_reader(expanded.elastic(), index, 2, workers) for index in range(2)]
    try:
        for step in range(3):
            assert [pieces(next(share)) for share in shares] == [whole[step][0::2], whole[step][1::2]]
    finally:
        for share in shares:
            share.close()


class Inline(futures.Executor):
    """Grain's prefetch thread pool, which runs each read as the prefetch submits it, so the reads
    it asks for ahead are made before the read that asked returns. The pool is where reads leave the
    reading thread: with its threads, what a prefetch holds ahead is seen only by waiting for reads
    that should not come, and Grain names no other place to count them. A read asked for is counted
    as made, so the count is the prefetch's reads in flight and buffered together."""

    def __init__(self, max_workers: int, *, thread_name_prefix: str = ""):
        super().__init__()

    def submit(self, fn, /, *args, **kwargs):
        done = futures.Future()
        try:
            done.set_result(fn(*args, **kwargs))
        except BaseException as error:
            done.set_exception(error)
        return done


class CountedImages:
    """Sixty-four 64 KiB images by index, each read of one counted."""

    def __init__(self):
        self.reads = 0

    def __len__(self) -> int:
        return 64

    def __getitem__(self, index):
        self.reads += 1
        return {"image": np.full((128, 128, 4), index, np.uint8)}


def test_automatic_loading_reads_an_elastic_read_ahead_by_the_groups_it_reads(monkeypatch):
    """An elastic read by index reads its inputs in groups of a batch's count, so automatic loading
    sizes its read-ahead in groups: eight 64 KiB images make a 512 KiB group, and 2 MiB holds four
    of them ahead, where sized by the record it would hold sixteen, 8 MiB. Counted at the source
    from the reader's open through its first batch, Grain's prefetch filling its four places, handing
    one on and filling it again, the read reads four whole groups, 32 images, more than one that
    holds nothing ahead; the reads both make at open, a scan of every image, cancel out."""
    # Grain's prefetch reads the pool's class from `concurrent.futures` as it builds the pool.
    monkeypatch.setattr(futures, "ThreadPoolExecutor", Inline)

    def first_batch_reads(loading: Loading) -> int:
        images = CountedImages()
        chain = Pipeline(Memory(images)).shuffle(1).repeat().filter(Unkept("image", -1)).batch(BATCH)
        stream = compile(chain.elastic(), read="indexed", loading=loading, seed=0).train(DataPartition(0, 2))
        try:
            opened = images.reads
            next(stream)
            return images.reads - opened
        finally:
            stream.close()

    automatic = first_batch_reads(Loading(workers="auto", ram_budget_mb=2, threads=2))
    assert automatic - first_batch_reads(Loading(workers=0, threads=0)) == 4 * BATCH


REFUSED, READ_A_BATCH = 3, 4
"""How a starved read's process ends (`first_batch`): refused, or with a batch."""


def first_batch(chain: Pipeline, index: int, count: int, workers: int) -> None:
    """Read share `index` of `count` of `chain` in `workers` loading workers to its first batch, in a
    process of its own, its end its exit code (`REFUSED`, `READ_A_BATCH`)."""
    opened = worker_reader(chain, index, count, workers)
    try:
        next(opened)
    except ValueError:
        sys.exit(REFUSED)
    finally:
        opened.close()
    sys.exit(READ_A_BATCH)


def starved(chain: Pipeline, index: int, count: int, *, workers: int = 0) -> bool:
    """Whether share `index` of `count` of `chain` is refused before its first batch, read in a
    process held to a minute's deadline and killed past it, with its loading workers, so a reader
    that reads on forever fails the test that reads it and leaves nothing running behind it."""
    spawned = multiprocessing.get_context("spawn")
    process = spawned.Process(target=first_batch, args=(chain, index, count, workers))
    process.start()
    process.join(60)
    alive = process.is_alive()
    if alive:
        process.kill()
        process.join()
    assert not alive, "the reader read on with every record of its share rejected"
    assert process.exitcode in (REFUSED, READ_A_BATCH), f"the read ended {process.exitcode}"
    return process.exitcode == REFUSED


@pytest.mark.parametrize(("elastic", "workers"), [(False, 2), (True, 0), (True, 2)],
                         ids=["per-share-in-workers", "elastic", "elastic-in-workers"])
def test_a_filter_that_rejects_every_record_read_is_refused_in_workers_and_elastically(elastic, workers):
    """A per-share read in loading workers whose share a filter rejects whole, and an elastic read
    whose filter rejects the whole corpus, are refused once two passes are rejected, the bound
    carried to the reading process that drops the workers' markers."""
    chain = Pipeline(Memory(tuple({"id": np.int64(2 * index)} for index in range(24)))).shuffle(5).repeat()
    read = chain.filter(odd).batch(BATCH)
    assert starved(read.elastic() if elastic else read, 0, 2, workers=workers)


@dataclasses.dataclass(frozen=True)
class Kept:
    """Keeps a record whose `field` is `value`."""

    field: str
    value: int

    def __call__(self, row):
        return int(row[self.field]) == self.value


@pytest.mark.parametrize("chain", ["expanded", "drawn-before", "shuffled-after"])
def test_rejections_that_need_not_go_on_forever_are_read_through(chain):
    """Rejections past two passes refuse a read only where each pass is the same records: an
    expansion whose first five pieces are rejected before its sixth is kept, from one record; a
    filter of a draw made before the repeat, which Grain's repeat draws again each pass, keeping
    only the third pass's draw; and a window shuffle across passes, keeping one of two records,
    which puts five rejections in a row ahead of its kept record at the seed chosen. Each reads its
    kept record."""
    if chain == "expanded":
        read = Pipeline(Memory(({"id": np.int64(0)},))).repeat().flat_map(Pieces(6)).filter(Kept("piece", 5))
        expected = ("piece", 5)
    elif chain == "drawn-before":
        drawn = Pipeline(Memory(({"id": np.int64(0)},))).random_map(WideDraw()).repeat()
        draws = list(itertools.islice(worker_reader(drawn.batch(1), 0, 1, 0), 3))
        third = int(np.asarray(draws[2]["draw"])[0])
        read = drawn.filter(Kept("draw", third))
        expected = ("draw", third)
    else:
        records = Pipeline(Memory(({"id": np.int64(0)}, {"id": np.int64(1)}))).repeat()

        def first_kept(seed: int) -> int:
            ids = worker_reader(records.shuffle(seed, window=100).batch(1), 0, 1, 0)
            return [int(np.asarray(batch["id"])[0]) for batch in itertools.islice(ids, 40)].index(1)

        # A seed whose window puts five rejections or more ahead of the first kept record.
        seed = next(seed for seed in range(200) if first_kept(seed) >= 5)
        read = records.shuffle(seed, window=100).filter(Kept("id", 1))
        expected = ("id", 1)
    assert not starved(read.batch(1), 0, 1)
    field, value = expected
    assert int(np.asarray(next(worker_reader(read.batch(1), 0, 1, 0))[field])[0]) == value


@dataclasses.dataclass(frozen=True)
class OddOnly:
    """A record whose id is odd, or its rejection (`Pipeline.filter_map`)."""

    def __call__(self, row):
        return row if int(row["id"]) % 2 else Rejected("even")


@pytest.mark.parametrize("count", [2, 4])
def test_an_elastic_read_whose_map_rejects_records_reads_one_readers_order(count):
    """A map that rejects records (`filter_map`) decides which rows there are, as a filter does, so
    every elastic reader runs it over the whole order and keeps its share of each batch of what it
    kept: the shares' rows together are one reader's, every id odd (row 1912, where the read took
    the map as one record for one and its rows' mapping refused the rejected record)."""
    def chain(elastic: bool = True) -> Pipeline:
        built = (Pipeline(Memory(tuple({"id": np.int64(index)} for index in range(96))))
                 .shuffle(5).repeat(None).filter_map(OddOnly()).map(Squared()).batch(BATCH))
        return built.elastic() if elastic else built

    def squared(batch) -> list[tuple[int, int]]:
        return list(zip(batch["id"], batch["squared"], strict=True))

    one = [squared(batch) for batch in listed(reader(chain(), 0, 1), 6)]
    shares = [[squared(batch) for batch in listed(reader(chain(), index, count), 6)]
              for index in range(count)]
    for step, whole in enumerate(one):
        assert sorted(row for share in shares for row in share[step]) == sorted(whole)
        assert all(ident % 2 for ident, *_ in whole)


def test_an_elastic_read_refuses_a_batch_ramp_before_it_reads():
    """A batch ramp cuts its steps out of a global record count, which an elastic read's position,
    its order's, is not: an elastic read under a ramp is refused where the ramp opens it, not where
    its first save or restore would find no record count (row 1913)."""
    loader = ramped(compile(stream(), loading=READ, seed=3), Ramp(start=4, increment=4, samples=64))
    with pytest.raises(TypeError):
        loader.train(DataPartition(0, 2))


@dataclasses.dataclass(frozen=True)
class Fetched:
    """A record as a live fetch returns it, which may differ between reads (`Volatile`)."""

    volatile: bool = True

    def __call__(self, row):
        return row


def test_an_elastic_read_of_a_volatile_source_over_several_shares_is_refused():
    """Every elastic share reads its order's whole prefix on its own, so a source a fetch reads
    differently each time (`Volatile`) would order the shares apart: over two shares it is refused
    before any reads, and over one it reads (row 1914)."""
    chain = (Pipeline(Memory(tuple({"id": np.int64(index)} for index in range(32))))
             .shuffle(5).repeat(None).map(Fetched()).filter(odd).batch(BATCH).elastic())
    with pytest.raises(ValueError):
        reader(chain, 0, 2)
    assert len(listed(reader(chain, 0, 1), 2)) == 2

