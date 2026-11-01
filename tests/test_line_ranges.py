"""How a JSON-lines or text file is cut into byte ranges a streamed read reads (`Files.manifest`):
equal ranges of at least `shard_bytes`, each read from its first line start, which its reader finds
within its own bytes, and refused where one record runs through all of them."""

import io
import itertools
import json

import pytest

from dew.data.sources import Files


def test_an_empty_file_holds_no_range_beside_one_that_holds_records(tmp_path):
    """An empty file is cut into no range, as a file of no bytes holds no record, and the store's
    other file is cut into its own."""
    (tmp_path / "empty.jsonl").write_text("")
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps({"id": index}) + "\n" for index in range(3)))
    shards = Files(str(tmp_path), format="jsonl", shard_bytes=10).manifest().shards
    assert shards and all(shard.uri.endswith("rows.jsonl") for shard in shards)


def test_a_files_byte_ranges_are_equal_and_never_shorter_than_its_shard_bytes(tmp_path):
    """A JSON-lines file is cut into equal ranges of at least shard_bytes, so no short tail range
    lies wholly inside the record that ends the file: 40 records of about 30 bytes cut at 100 bytes
    make ranges of 100 to 200 bytes, each holding records of its own, and read them all once."""
    lines = [json.dumps({"id": index, "pad": "y" * (index % 5)}) for index in range(40)]
    (tmp_path / "rows.jsonl").write_text("".join(line + "\n" for line in lines))
    stored = Files(str(tmp_path / "rows.jsonl"), shard_bytes=100)
    shards = stored.manifest().shards
    assert all(100 <= shard.stop - shard.start < 200 for shard in shards)
    read = [[row["id"] for row in stored.manifest().reader(shard)] for shard in shards]
    assert all(read)
    assert list(itertools.chain(*read)) == list(range(40))


@pytest.mark.parametrize("kind", ["jsonl", "text"])
def test_a_record_longer_than_shard_bytes_is_refused_by_the_range_it_runs_through(tmp_path, kind):
    """A record longer than a range runs through the whole of it, so the range starts no record and
    a reader given only it would read nothing: its reader refuses it, naming shard_bytes and the
    remedy, having read no more of the record than its own range's bytes."""
    lines = [json.dumps({"id": 0}), json.dumps({"id": 1, "pad": "x" * 5_000}), json.dumps({"id": 2})]
    (tmp_path / "rows.jsonl").write_text("".join(line + "\n" for line in lines))
    stored = Files(str(tmp_path / "rows.jsonl"), format=kind, shard_bytes=1_000)
    spanned = [shard for shard in stored.manifest().shards
               if shard.start > len(lines[0]) + 1 and shard.stop < len(lines[0]) + len(lines[1]) + 2]
    assert spanned
    with pytest.raises(ValueError):
        list(stored.manifest().reader(spanned[0]))
    whole = Files(str(tmp_path / "rows.jsonl"), format=kind, shard_bytes=10_000)
    rows = [row for shard in whole.manifest().shards for row in whole.manifest().reader(shard)]
    assert [row["id"] if kind == "jsonl" else json.loads(row["text"])["id"] for row in rows] == [0, 1, 2]


def test_a_line_start_is_found_a_chunk_at_a_time_within_the_range_it_is_sought_in():
    """The search for a range's first line reads no byte at or past the range's end, so a reader of
    a range inside a long record reads that range's bytes alone, not the rest of the record."""
    from dew.data.sources.files import _line_start

    class Counted(io.BytesIO):
        read_bytes = 0

        def read(self, size=-1):
            data = super().read(size)
            Counted.read_bytes += len(data)
            return data

    file = Counted(b"a" * 3_000_000 + b"\n" + b"b\n")
    assert _line_start(file, 1_000, 2_000) == 2_000
    # The byte before the range, which says whether a line starts at its first, and the range's own.
    assert Counted.read_bytes == 2_000 - 1_000
    assert _line_start(file, 1_000) == 3_000_001
    assert _line_start(file, 3_000_001) == 3_000_001
