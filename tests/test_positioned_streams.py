"""A stream that numbers its rows (`Pipeline.map_with_index`) over a text store with an offset index reads
the store's records as the index lists them, each at its index there, in parts of whole records each stating
its rows (`Files.positioned_manifest`, row 9051), as its indexed read numbers them."""

import io
import itertools
import json

import numpy as np
import pytest

from dew.data import Files
from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading
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


def test_a_positioned_fill_by_two_shares_of_two_workers_is_replayed_by_one_reader(tmp_path):
    """An indexed JSONL store of a few rows read by a stream that numbers its rows before its cache: two
    shares of two worker processes each fill its cache, every row numbered, and one reader in the process
    then reads it back from the published generation, numbering none again: every topology reads the same
    fixed shards (row 11637: one shard cut by the readers' count named parts no other topology's fill
    found)."""
    import functools

    from dew.data.cache import Cache
    from dew.data.compile import compile, iterate
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline

    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(12)))
    files = Files(str(path))
    files.provision()
    cache = Cache(str(tmp_path / "cache"))
    log = tmp_path / "numbered"
    log.touch()
    chain = Pipeline(files).map_with_index(functools.partial(_numbered, str(log)), identity="numbered@1")
    data = compile(chain.cache(cache).batch(2), read="streamed", loading=Loading(workers=2, threads=1,
                                                                                 read_buffer=1))
    filled = []
    for index in range(2):
        reader = data.train(DataPartition(index=index, count=2))
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
    from dew.data.dataset import DataPartition, Loading
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

