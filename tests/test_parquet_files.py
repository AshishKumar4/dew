"""A streamed Parquet store read file by file from data plan version 10 (row 4131): no file's footer read
before the first record but the first file's, each file resumed within it, and a position saved under
version 9 read by 9."""

import dataclasses
from typing import ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from data_support import LATEST
from fsspec.implementations.memory import MemoryFileSystem

from dew.data import Loading
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline
from dew.data.sources import Concatenate, Files, Memory
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


class Opening(MemoryFileSystem):
    """Memory's files, the name of each one opened noted (`opened`)."""

    protocol = "opening"
    opened: ClassVar[list[str]] = []

    @classmethod
    def _strip_protocol(cls, path):
        return super()._strip_protocol(str(path).replace("opening://", "memory://", 1))

    def _open(self, path, mode="rb", **kwargs):
        Opening.opened.append(str(path).rsplit("/", 1)[-1])
        return super()._open(path, mode, **kwargs)


def test_a_streamed_parquet_store_opens_its_first_file_alone_before_its_first_record():
    """A stream of a Parquet store compiled and read to its first record opens its first file alone: each
    file's footer is read as its reader reaches it, where its row groups as shards read every file's
    footer before any record (FineWeb-Edu's default configuration's 2410 files, 196 s: row 4131)."""
    import fsspec

    url = written("opened")
    fsspec.register_implementation("opening", Opening, clobber=True)
    Opening.opened.clear()
    stream = compile(Pipeline(Files(url.replace("memory://", "opening://", 1), format="parquet")).batch(2),
                     read="streamed", loading=READ, seed=0).train(DataPartition())
    try:
        first = next(stream)
        assert len(ids([first])) == 2
        assert len(set(Opening.opened)) == 1, sorted(set(Opening.opened))
    finally:
        stream.close()


def test_a_streamed_parquet_store_resumes_within_a_file_where_it_stopped():
    """A stream of a Parquet store saved partway through a file's row groups resumes at the record after the
    last one read, in that file, and reads each file's records once, in the seeded shard order."""
    url = written("resumed")

    def opened():
        return compile(Pipeline(Files(url, format="parquet")).batch(2), read="streamed", loading=READ,
                       seed=0, epochs=1).train(DataPartition())

    saving = opened()
    try:
        before = ids(next(saving) for _ in range(7))
        saved = saving.get_state()
        after = ids(saving)
    finally:
        saving.close()
    # File by file, each file's records in order, every file once, in the order the seed shuffles them.
    per_file = GROUPS * ROWS
    read = before + after
    files = [read[at:at + per_file] for at in range(0, len(read), per_file)]
    assert sorted(chunk[0] // per_file for chunk in files) == list(range(FILES))
    assert all(chunk == list(range(chunk[0], chunk[0] + per_file)) and chunk[0] % per_file == 0
               for chunk in files)
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
        return compile(Pipeline(source).repeat(1).batch(2), read="streamed", loading=READ,
                       **({} if version is None else {"plan_version": version})).train(DataPartition())

    for source in (Files(url, format="parquet"), Files(lines.url, format="jsonl")):
        saving = opened(source, 9)
        try:
            ids(next(saving) for _ in range(5))
            saved = saving.get_state()
            expected = ids(saving)
            assert len(expected) == FILES * GROUPS * ROWS - 10
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


def test_a_streamed_parquet_file_read_by_more_shares_than_files_places_each_record():
    """A stream of a Parquet store of one file, read by two shares, reads its row groups as its shards, each
    record numbered by its place in the store (`map_with_index`) whatever share reads it, as a read of row
    groups numbered them (review of row 4131)."""
    root = Store("memory://parquet-files-placed")
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist([{"id": row} for row in range(GROUPS * ROWS)]), sink,
                   row_group_size=ROWS)
    root.child("part-00.parquet").write(sink.getvalue().to_pybytes())
    # The batch size is global: four rows give each of the two shares two.
    chain = Pipeline(Files(root.url, format="parquet")).map_with_index(
        lambda index, row: {**row, "place": index}, identity="place@1").batch(4)
    read = compile(chain, read="streamed", loading=READ, epochs=1)
    placed = []
    for share in range(2):
        stream = read.train(DataPartition(share, 2))
        try:
            mine = [(int(batch["id"][at]), int(batch["place"][at])) for batch in stream for at in range(2)]
            assert len(mine) == (2 - share) * ROWS
            placed += mine
        finally:
            stream.close()
    assert sorted(placed) == [(row, row) for row in range(GROUPS * ROWS)]


def test_a_concatenation_of_parquet_files_reads_through_more_shares_than_files():
    """A concatenation of two Parquet stores of one file each, read by four shares, reads their row groups as
    its shards, each record once, as each store alone would (review of row 4131)."""
    stores = []
    for number in range(2):
        root = Store(f"memory://parquet-files-joined-{number}")
        sink = pa.BufferOutputStream()
        rows = [{"id": number * 100 + row} for row in range(GROUPS * ROWS)]
        pq.write_table(pa.Table.from_pylist(rows), sink, row_group_size=ROWS)
        root.child("part-00.parquet").write(sink.getvalue().to_pybytes())
        stores.append(Files(root.url, format="parquet"))
    # A batch of four, one row of it each share's.
    read = compile(Pipeline(Concatenate(tuple(stores))).batch(4), read="streamed", loading=READ, epochs=1)
    seen = []
    for share in range(4):
        stream = read.train(DataPartition(share, 4))
        try:
            seen += ids(stream)
        finally:
            stream.close()
    assert sorted(seen) == [number * 100 + row for number in range(2) for row in range(GROUPS * ROWS)]


def test_an_endless_stream_of_parquet_files_that_hold_no_row_is_refused():
    """An endless stream of a Parquet store whose files hold no row is refused as it reads, as a store whose
    shards state they hold none is, where it would read nothing forever (review of row 4131)."""
    root = Store("memory://parquet-files-empty")
    for number in range(3):
        sink = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pylist([], schema=pa.schema([("id", pa.int64())])), sink)
        root.child(f"part-{number:02d}.parquet").write(sink.getvalue().to_pybytes())
    with pytest.raises(ValueError):
        stream = compile(Pipeline(Files(root.url, format="parquet")).repeat().batch(2), read="streamed",
                         loading=READ).train(DataPartition())
        try:
            next(stream)
        finally:
            stream.close()


def test_a_published_parquet_stream_names_its_rows_by_its_files_counts():
    """A cache filled from a stream of a Parquet store publishes its rows named by its files and their counts
    read off their footers, reading no file's rows again to name them: each byte of a file's first row group
    is read once, by the stream, where a footer's read (Arrow's 64 KiB prefetch from the file's end) reaches
    none of it, the group lying more than that before the end (review of row 4131)."""
    import io
    from collections import Counter
    from uuid import uuid4

    import fsspec

    from dew.data.cache import Cache
    from dew.data.compile import iterate, publish

    first_group: dict[str, range] = {}
    read: Counter[str] = Counter()

    class Counted(io.BytesIO):
        """A file's bytes, counting those read of its first row group."""

        def __init__(self, path: str, payload: bytes):
            super().__init__(payload)
            self.path = path

        def _counted(self, start: int, count: int) -> None:
            group = first_group.get(self.path, range(0))
            read[self.path] += max(0, min(start + count, group.stop) - max(start, group.start))

        def read(self, size=-1):
            start = self.tell()
            chunk = super().read(size)
            self._counted(start, len(chunk))
            return chunk

        def readinto(self, buffer):
            start = self.tell()
            count = super().readinto(buffer)
            self._counted(start, count)
            return count

    class Counting(MemoryFileSystem):
        """Memory's files, each opened for reading as a `Counted`."""

        protocol = "counted"

        @classmethod
        def _strip_protocol(cls, path):
            return super()._strip_protocol(str(path).replace("counted://", "memory://", 1))

        def _open(self, path, mode="rb", **kwargs):
            opened = super()._open(path, mode, **kwargs)
            return Counted(self._strip_protocol(path), opened.getvalue()) if mode == "rb" else opened

    fsspec.register_implementation("counted", Counting, clobber=True)
    root = Store(f"counted://parquet-files-published-{uuid4().hex}")
    blobs = np.random.default_rng(0)
    for number in range(FILES):
        start = number * GROUPS * ROWS
        sink = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pylist([{"id": start + row, "blob": blobs.bytes(40_000)}
                                             for row in range(GROUPS * ROWS)]), sink, row_group_size=ROWS)
        payload = sink.getvalue().to_pybytes()
        group = pq.read_metadata(pa.BufferReader(payload)).row_group(0)
        chunks = [group.column(column) for column in range(group.num_columns)]
        begins = [min(offset for offset in (chunk.dictionary_page_offset, chunk.data_page_offset)
                      if offset is not None) for chunk in chunks]
        spans = range(min(begins), max(begin + chunk.total_compressed_size
                                       for begin, chunk in zip(begins, chunks, strict=True)))
        assert len(payload) - spans.stop > 64 * 1024
        child = root.child(f"part-{number:02d}.parquet")
        first_group[Counting._strip_protocol(child.url)] = spans
        child.write(payload)
    tagged = Pipeline(Files(root.url, format="parquet")).map(lambda row: {**row, "tag": row["id"] * 10},
                                                             identity="tag@1")
    cache = Cache(f"memory://parquet-files-published-cache-{uuid4().hex}")
    published = publish(tagged, cache, read="streamed")
    assert dict(read) == {path: len(spans) for path, spans in first_group.items()}
    rows = sorted((int(row["id"]), int(row["tag"])) for row in iterate(Pipeline(published.source()),
                                                                       read="indexed"))
    assert rows == [(row, row * 10) for row in range(FILES * GROUPS * ROWS)]


def test_dew_data_publish_publishes_a_parquet_stream_at_the_plan_version_it_names(tmp_path, capsys):
    """`dew data publish --plan-version` fills a Parquet stream's publication as a run reading by that plan
    version does, shards of its row groups at 9 and of its files at 10: each the publication `publish` makes
    at that version, two publications of one store (train 649's review)."""
    from uuid import uuid4

    from dew._cli.main import main
    from dew.data.cache import Cache
    from dew.data.compile import publish
    from dew.data.generation import Generation

    url = written(f"cli-{uuid4().hex}")
    chain = tmp_path / "chain.py"
    chain.write_text("from dew.data.pipeline import Pipeline\nfrom dew.data.sources import Files\n"
                     f"data = Pipeline(Files({url!r}, format='parquet'))\n")
    made = {}
    for version in (9, 10):
        cache = f"memory://parquet-files-cli-cache-{uuid4().hex}"
        capsys.readouterr()
        assert main(["data", "publish", str(chain), cache, "--plan-version", str(version)]) == 0
        published = Generation(capsys.readouterr().out.split()[-1]).manifest
        again = publish(Pipeline(Files(url, format="parquet")),
                        Cache(f"memory://parquet-files-api-{uuid4().hex}"), read="streamed",
                        plan_version=version)
        # The fill's dependency and its manifest are what the version decides; the rest is the generation's.
        made[version] = (published["dependency"], published["manifest"])
        assert made[version] == (again.manifest["dependency"], again.manifest["manifest"])
    assert made[9] != made[10]


@pytest.mark.parametrize("stores", [1, 2], ids=["one store", "two concatenated"])
def test_a_parquet_stream_read_by_more_shares_than_files_publishes_its_fill(tmp_path, stores):
    """A cached stream of Parquet stores of one file each, alone or concatenated, read whole by more shares
    than files, fills them by their row groups and publishes them as the stores' own shards
    (`ShardManifest.parts`), so a later read of them through the cache reads the generation back, running
    no step, where it refilled every run (train 654's review, row 4581; a concatenation's, train 666's,
    row 4656)."""
    from test_data_cache import Offset

    from dew.data.cache import Cache
    from dew.data.compile import iterate

    def files() -> list[Files]:
        made = []
        for store in range(stores):
            root = tmp_path / f"store{store}"
            if not root.is_dir():
                root.mkdir()
                pq.write_table(pa.Table.from_pylist([{"id": store * GROUPS * ROWS + row}
                                                     for row in range(GROUPS * ROWS)]),
                               root / "part-00.parquet", row_group_size=ROWS)
            made.append(Files(str(root), format="parquet"))
        return made

    def store():
        held = files()
        return held[0] if stores == 1 else Concatenate(tuple(held))

    total = stores * GROUPS * ROWS
    cache = Cache(str(tmp_path / "cache"))
    filled, reading = Offset(10), Offset(10)
    read = compile(Pipeline(store()).map(filled).cache(cache).batch(4), read="streamed", loading=READ,
                   epochs=1)
    read_ids = []
    shares = 2 * stores
    for share in range(shares):
        stream = read.train(DataPartition(share, shares))
        try:
            read_ids += ids(stream)
        finally:
            stream.close()
    assert sorted(read_ids) == list(range(total))
    rows = iterate(Pipeline(store()).map(reading).cache(cache), read="streamed")
    assert sorted(int(row["value"]) for row in rows) == [row + 10 for row in range(total)]
    assert sorted(filled.calls) == list(range(total)) and reading.calls == []


def test_a_parquet_store_reads_each_footer_once_however_many_reads_count_its_rows():
    """Every read that counts a streamed Parquet store's rows (a bound, a placed read, a pass's count, a
    publication: `ShardManifest.sized`) reads each file's footer once for the store, the first: the counts
    are the store's, kept for the reads after (train 654's review, row 4582)."""
    import fsspec

    url = written("counted-once").replace("memory://", "opening://", 1)
    fsspec.register_implementation("opening", Opening, clobber=True)
    Opening.opened.clear()
    store = Files(url, format="parquet")
    first = store.manifest(LATEST).sized(2)
    opened = len(Opening.opened)
    again = store.manifest(LATEST).sized(2)
    counts = [GROUPS * ROWS] * FILES
    assert [shard.rows for shard in first.shards] == [shard.rows for shard in again.shards] == counts
    assert len(Opening.opened) == opened and len(set(Opening.opened)) == FILES


@pytest.mark.parametrize("version", [5, 11])
def test_a_read_by_a_plan_version_this_dew_reads_by_none_of_is_refused(version):
    """Every read builds by its data plan version (`dew.data.compile._build`): a pass, a publication and an
    export at one this Dew reads by none of is refused, as a run's compile is, where 11 read as 10 and 5 as
    8 (train 654's review, row 4583); a publication of it is refused where a version it read as already
    published the store's generation too, which it would have returned."""
    from uuid import uuid4

    from dew.data.cache import Cache
    from dew.data.compile import iterate, publish

    chain = Pipeline(Files(written(f"version-{version}"), format="parquet"))
    with pytest.raises(ValueError):
        next(iter(iterate(chain, plan_version=version)))
    cache = Cache(f"memory://parquet-files-refused-{uuid4().hex}")
    with pytest.raises(ValueError):
        publish(chain, cache, plan_version=version)
    publish(chain, cache, plan_version=10 if version > 10 else 8)
    with pytest.raises(ValueError):
        publish(chain, cache, plan_version=version)
    # Refused before the read prepares anything: a memory read's steps, which it runs as it is compiled,
    # run none (train 667's review, row 4659).
    with pytest.raises(ValueError):
        compile(Pipeline(Memory([{"id": 0}, {"id": 1}])).map(Unread()).batch(2), read="memory",
                plan_version=version)


@dataclasses.dataclass(frozen=True)
class Unread:
    """A step a read that refuses its plan version never runs."""

    def __call__(self, row):
        raise RuntimeError("a refused read ran its steps")


def test_a_token_export_reads_its_documents_by_the_plan_version_it_names(tmp_path):
    """`dew data export --tokenizer --plan-version` exports a store's documents as tokens read by that plan
    version, as `dew.data.text.export_tokens(plan_version=)` does, where it was refused; a version this Dew
    reads by none of is refused (train 654's review, row 4584)."""
    from token_support import export_byte_tokens

    from dew.data.compile import iterate
    from dew.data.sources import TokenFiles

    documents = ["one", "two words", "three of them"]
    for name in ("nine", "eleven"):
        (tmp_path / name).mkdir()
    exported = export_byte_tokens(tmp_path / "nine", documents, "--plan-version", "9")
    read = [bytes(row["text"][:-1].astype(np.uint8)).decode() for row in iterate(Pipeline(TokenFiles(
        str(exported))), read="streamed")]
    assert read == documents
    with pytest.raises(ValueError):
        export_byte_tokens(tmp_path / "eleven", documents, "--plan-version", "11")
