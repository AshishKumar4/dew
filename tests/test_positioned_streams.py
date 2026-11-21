"""A stream that numbers its rows (`Pipeline.map_with_index`) over a text store with an offset index reads
the store's records as the index lists them, each at its index there, in parts of whole records each stating
its rows (`Files.positioned_manifest`, row 9051), as its indexed read numbers them."""

import io
import itertools
import json

import numpy as np

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
