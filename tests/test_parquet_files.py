"""A streamed Parquet store read file by file from data plan version 10 (row 4131): no file's footer read
before the first record but the first file's, each file resumed within it, and a position saved under
version 9 read by 9."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fsspec.implementations.memory import MemoryFileSystem

from dew.data import Loading
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline
from dew.data.sources import Files
from dew.data.storage import Store

READ = Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1)
FILES, GROUPS, ROWS = 6, 3, 4


def written(name: str) -> str:
    """`FILES` Parquet files in memory, each of `GROUPS` row groups of `ROWS` rows, ids counting through
    them in order; their directory's URL."""
    root = Store(f"memory://parquet-files-{name}")
    for number in range(FILES):
        start = number * GROUPS * ROWS
        table = pa.Table.from_pylist([{"id": start + row} for row in range(GROUPS * ROWS)])
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink, row_group_size=ROWS)
        root.child(f"part-{number:02d}.parquet").write(sink.getvalue().to_pybytes())
    return root.url


def ids(batches) -> list[int]:
    return [int(value) for batch in batches for value in batch["id"]]


def test_a_streamed_parquet_store_opens_its_first_file_alone_before_its_first_record(monkeypatch):
    """A stream of a Parquet store compiled and read to its first record opens its first file alone: each
    file's footer is read as its reader reaches it, where its row groups as shards read every file's
    footer before any record (FineWeb-Edu's default configuration's 2410 files, 196 s: row 4131)."""
    url = written("opened")
    opened: list[str] = []
    native = MemoryFileSystem._open

    def counted(self, path, *args, **kwargs):
        opened.append(path.rsplit("/", 1)[-1])
        return native(self, path, *args, **kwargs)

    monkeypatch.setattr(MemoryFileSystem, "_open", counted)
    stream = compile(Pipeline(Files(url, format="parquet")).batch(2), read="streamed",
                     loading=READ).train(DataPartition())
    try:
        first = next(stream)
        assert ids([first]) == [0, 1]
        assert set(opened) == {"part-00.parquet"}, sorted(set(opened))
    finally:
        stream.close()


def test_a_streamed_parquet_store_resumes_within_a_file_where_it_stopped():
    """A stream of a Parquet store saved partway through a file's row groups resumes at the record after the
    last one read, in that file, and reads each file's records once, in order."""
    url = written("resumed")

    def opened():
        return compile(Pipeline(Files(url, format="parquet")).batch(2), read="streamed",
                       loading=READ).train(DataPartition())

    saving = opened()
    try:
        before = ids(next(saving) for _ in range(7))
        saved = saving.get_state()
        after = ids(saving)
    finally:
        saving.close()
    assert before + after == list(range(FILES * GROUPS * ROWS))
    resumed = opened()
    try:
        resumed.set_state(saved)
        assert ids(resumed) == after
    finally:
        resumed.close()


def test_a_parquet_streams_position_saved_under_plan_version_9_resumes_by_9_alone():
    """A position a version 9 read saved of a Parquet stream, whose shards 9 made of its row groups and 10
    makes of its files, resumes where 9 compiles the chain (`plan_version=9`) and is refused where this Dew's
    version does, naming the version to compile it by (`dew.data.plan.read_otherwise_at_10`); a stream of
    another store saved under 9 resumes under this Dew's version, which reads its order alike."""
    url = written("versions")
    lines = Store("memory://parquet-files-lines")
    lines.child("rows.jsonl").write(b"".join(b'{"id": %d}\n' % row for row in range(FILES * GROUPS * ROWS)))

    def opened(source, version=None):
        return compile(Pipeline(source).batch(2), read="streamed", loading=READ,
                       **({} if version is None else {"plan_version": version})).train(DataPartition())

    for source in (Files(url, format="parquet"), Files(lines.url, format="jsonl")):
        saving = opened(source, 9)
        try:
            ids(next(saving) for _ in range(5))
            saved = saving.get_state()
            expected = ids(saving)
        finally:
            saving.close()
        for version in (9, None):
            resumed = opened(source, version)
            try:
                if source.kind == "parquet" and version is None:
                    with pytest.raises(ValueError):
                        resumed.set_state(saved)
                    continue
                resumed.set_state(saved)
                assert ids(resumed) == expected
            finally:
                resumed.close()
