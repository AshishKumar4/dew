"""A window shuffle over a stream of line files reads each window's records by index where its stream serves
them so (`dew.data.shards.Window`, row 12475): the records, in the order, the stream's window gives, its
position the stream's, read only as each is given. Each read is held to the same chain over the same files
read as a stream (`streaming`: no window served by index), its position saved and restored across the two."""

import contextlib
import gzip
import itertools
import json

import numpy as np
import pytest

from dew.data import Files, shards
from dew.data.compile import compile
from dew.data.dataset import Checkpointable, Closeable, DataPartition, Loading

LOADING = Loading(workers=0, threads=1, read_buffer=1)


def _files(directory, sizes: tuple[int, ...]) -> tuple[str, ...]:
    """JSONL files of `sizes` records each, their lines ended `\\r\\n` and `\\n` by turns, a blank line and a
    line of spaces among them and each file's last line unended."""
    directory.mkdir(exist_ok=True)
    paths = []
    for index, size in enumerate(sizes):
        lines = [json.dumps({"file": index, "id": row, "text": f"{index}-{row}"}) for row in range(size)]
        text = "".join(line + ("\r\n" if row % 2 else "\n") + ("\n" if row == 2 else "")
                       + ("   \n" if row == 4 else "") for row, line in enumerate(lines[:-1])) + lines[-1]
        path = directory / f"f{index}.jsonl"
        path.write_text(text)
        paths.append(str(path))
    return tuple(paths)


def _ids(batch) -> list[tuple[int, int]]:
    return [(int(file), int(row)) for file, row in zip(batch["file"], batch["id"], strict=True)]


def _kept(record):
    return {"file": np.int32(record["file"]), "id": np.int32(record["id"])}


def _read(files, **fields):
    """The JSONL `files` read, each record kept as its file and id (`_kept`)."""
    return Files(files, format="jsonl", **fields).map(_kept)


CHAINS = {
    # Read endlessly by the run, its root's own lifetime, no repeat written.
    "unrepeated": (lambda files: _read(files).shuffle(3, window=8).batch(2), (6,)),
    "repeated": (lambda files: _read(files).repeat().shuffle(3, window=8).batch(2), (5,)),
    "repeated-twice": (lambda files: _read(files).repeat().repeat(2).shuffle(3, window=8).batch(2), (5,)),
    # Two files read in the run's seeded order of them, one record a window.
    "two-files": (lambda files: _read(files).shuffle(3, window=1).batch(2), (1, 1)),
    "larger-than-its-window": (lambda files: _read(files).shuffle(3, window=4).batch(3), (9, 7)),
    "one-pass": (lambda files: _read(files).repeat(1).shuffle(3, window=4).batch(3), (9, 7)),
    "columns": (lambda files: _read(files, columns=("file", "id")).shuffle(3, window=8).batch(2), (6,)),
}


@pytest.fixture
def streaming(monkeypatch):
    """A context in which a sequential pass serves no window by index: the same chain over the same files
    read as a stream, as a source no index serves reads (`dew.data.shards._SequentialIterator.window`)."""
    @contextlib.contextmanager
    def streamed():
        with monkeypatch.context() as patched:
            patched.setattr(shards._SequentialIterator, "window", lambda self, limit: None)
            yield
    return streamed


def _reader(tmp_path, chain: str, loading: Loading = LOADING):
    """`chain`'s training read over its files."""
    build, sizes = CHAINS[chain]
    files = _files(tmp_path / "files", sizes)
    reader = compile(build(files), read="streamed", loading=loading, seed=7).train(DataPartition())
    assert isinstance(reader, Checkpointable), type(reader)
    return reader


def _taken(reader, steps: int, within=contextlib.nullcontext) -> list[list[tuple[int, int]]]:
    """The next `steps` batches of `reader`, read within `within`."""
    with within():
        return [_ids(batch) for batch in itertools.islice(reader, steps)]


@pytest.mark.parametrize("chain", sorted(set(CHAINS) - {"one-pass"}))
def test_a_window_read_by_index_gives_the_records_a_streams_window_gives(tmp_path, chain, streaming):
    """Each chain's read gives the batches its stream gives, in order, for twelve steps: a source read
    endlessly by the run with no repeat written, repeated once or twice, two files read in the run's seeded
    order of them, files larger than a window, and a projection of the records' columns; blank and
    whitespace lines no records."""
    assert _taken(_reader(tmp_path, chain), 12) == _taken(_reader(tmp_path, chain), 12, streaming)


def test_a_window_read_by_index_reads_the_records_it_gives(tmp_path, streaming):
    """A window of 64 over a source of 9 records, repeated, its map before the shuffle, reads by index the
    records its first batch gives, where its stream read the whole window, the source seven times over,
    before giving one."""
    files = _files(tmp_path / "files", (9,))
    reads = []

    def counted(record):
        reads.append(1)
        return _kept(record)

    def first(within) -> int:
        reads.clear()
        chain = Files(files, format="jsonl").map(counted).repeat().shuffle(3, window=64).batch(2)
        with within():
            reader = compile(chain, read="streamed", loading=LOADING, seed=7).train(DataPartition())
            next(reader)
            if isinstance(reader, Closeable):
                reader.close()
        return len(reads)

    served, streamed = first(contextlib.nullcontext), first(streaming)
    assert served <= 2 < 64 <= streamed, (served, streamed)


@pytest.mark.parametrize("taken", [1, 4, 5, 9])
def test_a_streams_saved_position_resumes_a_window_read_by_index_and_back(tmp_path, taken, streaming):
    """A position its stream saved after `taken` steps (mid-window, at a window's end, past a pass's end)
    resumes the read by index at the batch the stream gives next, and the position the read by index
    saves resumes the stream there."""
    streamed = _reader(tmp_path, "repeated")
    _taken(streamed, taken, streaming)
    with streaming():
        saved = streamed.get_state()
    served = _reader(tmp_path, "repeated")
    served.set_state(saved)
    assert _taken(served, 10) == _taken(streamed, 10, streaming)
    back = _reader(tmp_path, "repeated")
    with streaming():
        back.set_state(served.get_state())
    again = _reader(tmp_path, "repeated")
    again.set_state(served.get_state())
    assert _taken(back, 6, streaming) == _taken(again, 6)


def test_a_finite_window_read_by_index_ends_where_its_stream_ends(tmp_path, streaming):
    """A read of one pass ends, its last window smaller than the rest, where its stream's does, the same
    batches given; resumed from the stream's position within that last window, it gives the rest."""
    served = _taken(_reader(tmp_path, "one-pass"), 100)
    streamed = _reader(tmp_path, "one-pass")
    assert _taken(streamed, len(served) - 1, streaming) == served[:-1]
    resumed = _reader(tmp_path, "one-pass")
    with streaming():
        resumed.set_state(streamed.get_state())
    assert _taken(resumed, 100) == _taken(streamed, 100, streaming) == served[-1:]


def test_two_workers_read_a_window_by_index_as_their_stream_does(tmp_path):
    """Two loading workers, each a share of two files, read the batches the same chain gives over a
    gzipped copy of the files, which no index serves (the workers' processes read as they are built, no
    patch of this one reaching them)."""
    workers = Loading(workers=2, threads=1, read_buffer=1)
    served = _taken(_reader(tmp_path, "larger-than-its-window", workers), 8)
    plain = _files(tmp_path / "files", CHAINS["larger-than-its-window"][1])
    zipped = []
    for path in plain:
        with open(path, "rb") as text:
            data = gzip.compress(text.read())
        with open(f"{path}.gz", "wb") as copy:
            copy.write(data)
        zipped.append(f"{path}.gz")
    chain = CHAINS["larger-than-its-window"][0](tuple(zipped))
    streamed = compile(chain, read="streamed", loading=workers, seed=7).train(DataPartition())
    assert served == _taken(streamed, 8)


def test_an_elastic_window_saved_without_workers_resumes_with_two(tmp_path):
    """An elastic read of a window shuffle saved with no loading workers resumes with two at the batch it
    would give next."""
    def elastic(workers: int):
        files = _files(tmp_path / "files", (9, 7))
        chain = _read(files).repeat().shuffle(3, window=8).batch(2).elastic()
        loading = Loading(workers=workers, threads=1, read_buffer=1)
        reader = compile(chain, read="streamed", loading=loading, seed=7).train(DataPartition())
        assert isinstance(reader, Checkpointable), type(reader)
        return reader

    saved = elastic(0)
    _taken(saved, 3)
    resumed = elastic(2)
    resumed.set_state(saved.get_state())
    assert _taken(resumed, 6) == _taken(saved, 6)


def test_a_window_over_more_places_than_it_holds_open_reads_them_all(tmp_path, streaming):
    """A window over more one-record files than a pass holds open for reads by index at once
    (`dew.data.shards.MAPPED`) reads every record its stream reads, each file let go as others are asked."""
    files = _files(tmp_path / "files", (1,) * (3 * shards.MAPPED))
    chain = _read(files).shuffle(3, window=len(files)).batch(len(files))

    def read(within):
        with within():
            reader = compile(chain, read="streamed", loading=LOADING, seed=7).train(DataPartition())
            return sorted(_ids(next(reader)))

    assert read(contextlib.nullcontext) == read(streaming)
