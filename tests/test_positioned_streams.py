"""A stream that numbers its rows (`Pipeline.map_with_index`) over a text store with an offset index reads
the store's records as the index lists them, each at its index there, in parts of whole records each stating
its rows (`Files.positioned_manifest`, row 9051), as its indexed read numbers them."""

import dataclasses
import io
import itertools
import json

import numpy as np
import pytest

from dew.data import Files
from dew.data.compile import compile
from dew.data.dataset import Closeable, DataPartition, Loading
from dew.data.pipeline import Pipeline


def numbered(files: Files) -> list[tuple[int, int]]:
    """Each row `files`' positioned stream reads, as (its place there, its own id)."""
    chain = Pipeline(files).map_with_index(lambda index, row: {**row, "at": np.int64(index)}).batch(1)
    reader = compile(chain, read="streamed", loading=Loading(workers=0, threads=2, read_buffer=1)).train(
        DataPartition())
    try:
        return [(int(batch["at"][0]), int(batch["id"][0])) for batch in itertools.islice(reader, 40)]
    finally:
        reader.close()


def test_a_positioned_stream_of_an_indexed_jsonl_store_numbers_each_row_by_its_index(tmp_path):
    """A JSONL store of twenty records, cut in byte ranges of a few records each, whose offset index
    `provision` wrote: its positioned stream reads every record at its index, as its indexed read places
    them, where before its byte ranges stated no rows and the stream was refused."""
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(20)))
    files = Files(str(path), shard_bytes=40)
    files.provision()
    assert sorted(numbered(files)[:20]) == [(index, index) for index in range(20)]


def test_a_positioned_stream_reads_the_records_a_producers_subset_index_lists(tmp_path):
    """A JSONL store whose producer's index lists every other record (a subset of its lines): its
    positioned stream reads those, each at its place in the index, as its indexed read does, never the
    lines the index leaves out."""
    path = tmp_path / "rows.jsonl"
    lines = [json.dumps({"id": index}) + "\n" for index in range(20)]
    path.write_text("".join(lines))
    starts = np.cumsum([0, *(len(line) for line in lines)])[:-1]
    listed = [(0, int(starts[index]), len(lines[index]) - 1) for index in range(0, 20, 2)]
    encoded = io.BytesIO()
    np.save(encoded, np.array(listed, dtype=[("shard", "<u4"), ("offset", "<u8"), ("bytes", "<u8")]))
    (tmp_path / "rows.index.npy").write_bytes(encoded.getvalue())
    files = Files(str(path), shard_bytes=40)
    indexed = files.records("indexed")
    expected = [(place, indexed[place]["id"]) for place in range(len(indexed))]
    assert sorted(numbered(files)[:len(expected)]) == expected and expected[1] == (1, 2)


def test_a_positioned_streams_shards_are_named_alike_wherever_its_store_is_mounted(tmp_path):
    """One indexed JSONL store at two mounts (two hosts', a resume's after a move): its positioned stream's
    shards are named alike at both, by its files' places, so a position one saved resumes on the other (row
    9051's review: they were named by the mount's path)."""
    for mount in ("a", "b"):
        (tmp_path / mount / "data").mkdir(parents=True)
        path = tmp_path / mount / "data" / "rows.jsonl"
        path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(8)))
        Files(str(path), shard_bytes=40).provision()
    first, second = (Files(str(tmp_path / mount / "data" / "rows.jsonl"),
                           shard_bytes=40).positioned_manifest() for mount in ("a", "b"))
    assert [shard.id for shard in first.shards] == [shard.id for shard in second.shards]


def test_a_positioned_stream_behind_its_cache_publishes_it_once_it_reads_every_shard(tmp_path):
    """An indexed JSONL store read by a stream that numbers its rows before its cache, in several shards
    (two readers): the cache publishes the store's generation once the read has filled every positioned
    shard, so the next streamed read reads it back, numbering and running nothing (row 9051's confirm
    review: the fill counted the store's byte ranges, and no seal reached its publication)."""
    from dew.data.cache import Cache
    from dew.data.compile import iterate
    from dew.data.dataset import Loading
    from dew.data.pipeline import Pipeline

    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(12)))
    files = Files(str(path), shard_bytes=40)
    files.provision()
    cache = Cache(str(tmp_path / "cache"))
    numbered: list[int] = []

    def at(index, row):
        numbered.append(int(index))
        return {**row, "at": np.int64(index)}

    def values() -> list[tuple[int, int]]:
        chain = Pipeline(files).map_with_index(at, identity="numbered@1").cache(cache)
        rows = iterate(chain, read="streamed", loading=Loading(workers=0, threads=2, read_buffer=1))
        return sorted((int(row["at"]), int(row["id"])) for row in rows)

    first = values()
    count = len(numbered)
    again = values()
    assert first == again == [(index, index) for index in range(12)]
    assert count == 12 and len(numbered) == count


def _numbered(log: str, index, row):
    """`row` with the index a positioned stream numbered it by, the index written to `log`: what a worker
    process numbered, which its parent reads."""
    with open(log, "a") as written:
        written.write(f"{int(index)}\n")
    return {**row, "at": np.int64(index)}


def numbered_cache(tmp_path, rows: int):
    """A JSONL store of `rows` records read by a stream that numbers its rows before a cache, the cache, and
    the log of every row a read numbered (`_numbered`)."""
    import functools

    from dew.data.cache import Cache
    from dew.data.pipeline import Pipeline

    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(rows)))
    files = Files(str(path))
    files.provision()
    log = tmp_path / "numbered"
    log.touch()
    chain = Pipeline(files).map_with_index(functools.partial(_numbered, str(log)), identity="numbered@1")
    return chain, Cache(str(tmp_path / "cache")), log


def test_a_positioned_fill_by_two_shares_of_two_workers_is_replayed_by_one_reader(tmp_path):
    """An indexed JSONL store of a few rows read by a stream that numbers its rows before its cache: two
    shares of two worker processes each fill its cache, every row numbered, and one reader in the process
    then reads it back from the published generation, numbering none again: every topology reads the same
    fixed shards (row 11637: one shard cut by the readers' count named parts no other topology's fill
    found)."""
    from dew.data.compile import compile, iterate

    chain, cache, log = numbered_cache(tmp_path, 12)
    data = compile(chain.cache(cache).batch(2), read="streamed", loading=Loading(workers=2, threads=1,
                                                                                 read_buffer=1))
    filled = []
    for index in range(2):
        reader = data.train(DataPartition(index=index, count=2))
        assert isinstance(reader, Closeable)
        try:
            filled += [(int(at), int(id_)) for batch in (next(reader) for _ in range(6))
                       for at, id_ in zip(batch["at"], batch["id"], strict=True)]
        finally:
            reader.close()
    numbered = log.read_text().split()
    alone = iterate(chain.cache(cache), read="streamed", loading=Loading(workers=0))
    replayed = sorted((int(row["at"]), int(row["id"])) for row in alone)
    assert sorted(filled) == replayed == [(index, index) for index in range(12)]
    assert set(map(int, numbered)) == set(range(12)) and log.read_text().split() == numbered


def test_a_positioned_stream_of_fewer_records_than_its_readers_is_refused(tmp_path):
    """A positioned stream over a store of three records read by two shares of two workers is refused before
    it reads, as any store of fewer shards than its readers is: each share steps in lockstep with the
    others, which a share with no records cannot."""
    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline

    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(3)))
    files = Files(str(path))
    files.provision()
    chain = Pipeline(files).map_with_index(lambda index, row: {**row, "at": np.int64(index)},
                                           identity="numbered@1").batch(1)
    data = compile(chain, read="streamed", loading=Loading(workers=2, threads=1, read_buffer=1))
    with pytest.raises(ValueError):
        next(data.train(DataPartition(index=0, count=2)))


@pytest.mark.parametrize("records", [3, 64, 65, 200])
def test_a_positioned_streams_store_is_cut_into_as_many_shards_as_its_records_allow(tmp_path, records):
    """A positioned stream's store of a few hundred bytes is cut into one shard a record, or, past 64
    records, into 64 shards at the least, each of whole records in order: so many readers read it at once
    (row 11637's review: 65 records were cut into 33 shards of two, short of 34 readers)."""
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(records)))
    files = Files(str(path))
    files.provision()
    shards = files.positioned_manifest().shards
    assert len(shards) >= min(records, 64), len(shards)
    assert [(shard.start, shard.stop) for shard in shards] == [
        (start, min(start + shards[0].rows, records)) for start in range(0, records, shards[0].rows)]



def test_a_positioned_streams_store_is_cut_into_as_many_shards_as_its_readers_and_workers_read(tmp_path):
    """A positioned stream's store of 130 records, held in 65 ranges, read by 40 shares of two workers each
    is cut into exactly 80 ranges of whole records in order, one for each of its readers, their sizes
    differing by one record at most, every record read once (docs/design/data.md: a store of fewer shards
    than its readers × workers is cut into that many; train 1192's review: it was refused past 64 readers;
    Train 1240's: 1000 records for 128 readers were cut into 143, one reader dealt twice another's)."""
    from dew.data.shards import cut

    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(130)))
    files = Files(str(path))
    files.provision()
    shards = cut(files.positioned_manifest(), partition=DataPartition(index=0, count=40),
                 loading=Loading(workers=2)).shards
    assert len(shards) == 80, len(shards)
    assert [shard.start for shard in shards[1:]] == [shard.stop for shard in shards[:-1]]
    assert (shards[0].start, shards[-1].stop) == (0, 130)
    assert {shard.stop - shard.start for shard in shards} == {1, 2}


def test_a_positioned_read_cut_finer_than_its_store_fills_no_cache_another_topology_publishes(tmp_path):
    """A positioned stream's store of 160 records in 80 fixed ranges of two read before its cache by 160
    shares, each a range of one record (`subdivide`): a share whose range starts inside one of the
    store's (`records:1` of its range), read past its end, is keyed by a cut no fill of the store names
    and fills nothing, so one reader's read of the store's own ranges after it publishes the generation,
    and the next read numbers nothing again (review of row 11948: the cut's completions, named as the
    store's ranges are, blocked every later fill's publication; its share 0 began a range of the store's
    own)."""
    from dew.data.compile import compile, iterate

    chain, cache, log = numbered_cache(tmp_path, 160)
    loading = Loading(workers=0, threads=1, read_buffer=1)
    reader = compile(chain.cache(cache).batch(160), read="streamed", loading=loading).train(
        DataPartition(index=1, count=160))
    assert isinstance(reader, Closeable)
    try:
        # The share's row, its own range of one record, numbered as the store holds it, and the next, read
        # past the range's end, where a fill would complete.
        batches = list(itertools.islice(reader, 2))
    finally:
        reader.close()
    at, numbered = (int(value) for value in (batches[0]["at"][0], batches[0]["id"][0]))
    # The second record of one of the store's ranges of two (`records:1`), so its cut is none of them.
    assert len(batches) == 2 and at == numbered and at % 2 == 1, (at, numbered)

    def whole() -> list[tuple[int, int]]:
        rows = iterate(chain.cache(cache), read="streamed", loading=loading)
        return sorted((int(row["at"]), int(row["id"])) for row in rows)

    first = whole()
    numbered = log.read_text().split()
    assert first == whole() == [(index, index) for index in range(160)]
    assert log.read_text().split() == numbered


def test_a_position_saved_under_the_first_cut_of_a_positioned_store_is_refused_where_it_resumes(tmp_path):
    """A positioned stream's store cut finer than its ranges by the readers' count is cut by one rule
    (`EVEN_CUT`), which its shards fact holds (`ShardManifest.cut`): a position saved when the store's
    ranges were cut otherwise, whose fact names its shards alone, is refused where it resumes, naming the
    fact, never read at the rows its places now hold (Train 1240's review: 130 records for 80 readers moved
    one reader's rows from [1, 2) to [2, 4) under an unchanged fact)."""
    from dew.data.plan import PlanChanged
    from dew.data.sources.base import stream_manifest

    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(130)))
    files = Files(str(path))
    files.provision()
    chain = Pipeline(files).map_with_index(lambda index, row: {**row, "at": np.int64(index)}).batch(1)
    loading = Loading(workers=0, threads=1, read_buffer=1)
    reader = compile(chain, read="streamed", loading=loading).train(DataPartition())
    assert isinstance(reader, Closeable)
    try:
        list(itertools.islice(reader, 2))
        saved = json.loads(reader.get_state())
    finally:
        reader.close()
    manifest = stream_manifest(files, positioned=True)
    current, first = manifest.digest, dataclasses.replace(manifest, cut=None).digest
    assert current != first
    facts = saved["dew_data_plan"]["derived"]
    named = [key for key, value in facts.items() if value == current]
    assert named, facts
    for key in named:
        facts[key] = first
    resumed = compile(chain, read="streamed", loading=loading).train(DataPartition())
    assert isinstance(resumed, Closeable)
    try:
        with pytest.raises(PlanChanged) as refused:
            resumed.set_state(json.dumps(saved).encode())
        assert any(part.endswith("shards") for part in refused.value.differs), refused.value.differs
    finally:
        resumed.close()
