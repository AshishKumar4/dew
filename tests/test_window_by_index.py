"""A window shuffle over a stream of line files reads each window's records by index where its stream serves
them so (`dew.data.shards.Window`, row 12475): the records, in the order, the stream's window gives, its
position the stream's, read only as each is given. Each read is held to the same chain over a gzipped copy
of its files, which no index serves, its position saved and restored across the two."""

import gzip
import itertools
import json

import numpy as np
import pytest

from dew.data import Files
from dew.data.compile import compile
from dew.data.dataset import Checkpointable, Closeable, DataPartition, Loading

LOADING = Loading(workers=0, threads=1, read_buffer=1)


def _files(directory, sizes: tuple[int, ...]) -> tuple[str, ...]:
    """JSONL files of `sizes` records each, their lines ended `\\r\\n` and `\\n` by turns, a blank line and a
    line of spaces among them and each file's last line unended; gzipped under a `streamed` directory."""
    directory.mkdir(exist_ok=True)
    paths = []
    for index, size in enumerate(sizes):
        lines = [json.dumps({"file": index, "id": row, "text": f"{index}-{row}"}) for row in range(size)]
        text = "".join(line + ("\r\n" if row % 2 else "\n") + ("\n" if row == 2 else "")
                       + ("   \n" if row == 4 else "") for row, line in enumerate(lines[:-1])) + lines[-1]
        streamed = directory.name == "streamed"
        path = directory / f"f{index}.jsonl{'.gz' if streamed else ''}"
        path.write_bytes(gzip.compress(text.encode()) if streamed else text.encode())
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
    "columns": (lambda files: _read(files, columns=("file", "id")).shuffle(3, window=8).batch(2), (6,)),
}


def _reader(tmp_path, chain: str, how: str, loading: Loading = LOADING, **compiled):
    """`chain`'s training read over its files under `how` (`served` by index, or `streamed`)."""
    build, sizes = CHAINS[chain]
    files = _files(tmp_path / how, sizes)
    compiled_chain = compile(build(files), read="streamed", loading=loading, seed=7, **compiled)
    reader = compiled_chain.train(DataPartition())
    assert isinstance(reader, Checkpointable), type(reader)
    return reader


@pytest.mark.parametrize("chain", sorted(CHAINS))
def test_a_window_read_by_index_gives_the_records_a_streams_window_gives(tmp_path, chain):
    """Each chain's read gives the batches its stream gives, in order, for twelve steps:
    a source read endlessly by the run with no repeat written, repeated once or twice, two files read
    in the run's seeded order of them, files larger than a window, and a projection of the records'
    columns; blank and whitespace lines no records."""
    served, streamed = _reader(tmp_path, chain, "served"), _reader(tmp_path, chain, "streamed")
    assert ([_ids(batch) for batch in itertools.islice(served, 12)]
            == [_ids(batch) for batch in itertools.islice(streamed, 12)])


def test_a_window_read_by_index_reads_the_records_it_gives(tmp_path):
    """A window of 64 over a source of 9 records, repeated, reads by index the records its first batch
    gives, where its stream read the whole window, the source seven times over, before giving one."""
    reads = {"served": 0, "streamed": 0}

    def counted(name: str):
        def row(record):
            reads[name] += 1
            return _kept(record)
        return row

    for name in reads:
        files = _files(tmp_path / name, (9,))
        chain = Files(files, format="jsonl").map(counted(name)).repeat().shuffle(3, window=64).batch(2)
        reader = compile(chain, read="streamed", loading=LOADING, seed=7).train(DataPartition())
        next(reader)
        if isinstance(reader, Closeable):
            reader.close()
    assert reads["served"] <= 2 < 64 <= reads["streamed"], reads


@pytest.mark.parametrize("taken", [1, 4, 5, 9])
def test_a_streams_saved_position_resumes_a_window_read_by_index_and_back(tmp_path, taken):
    """A position its stream saved after `taken` steps (mid-window, at a window's end, past a pass's end)
    resumes the read by index at the batch the stream gives next, and the position the read by index
    saves resumes the stream there."""
    streamed = _reader(tmp_path, "repeated", "streamed")
    for _ in range(taken):
        next(streamed)
    served = _reader(tmp_path, "repeated", "served")
    served.set_state(streamed.get_state())
    following = [_ids(batch) for batch in itertools.islice(streamed, 10)]
    assert [_ids(batch) for batch in itertools.islice(served, 10)] == following
    back = _reader(tmp_path, "repeated", "streamed")
    back.set_state(served.get_state())
    again = _reader(tmp_path, "repeated", "streamed")
    again.set_state(streamed.get_state())
    assert [_ids(batch) for batch in itertools.islice(back, 6)] == [
        _ids(batch) for batch in itertools.islice(again, 6)]


def test_a_finite_window_read_by_index_ends_where_its_stream_ends(tmp_path):
    """A read of one pass ends, its last window smaller than the rest, where its stream's does, the same
    batches given; resumed from the stream's position within that last window, it gives the rest."""
    served = list(_reader(tmp_path, "larger-than-its-window", "served", epochs=1))
    streamed = _reader(tmp_path, "larger-than-its-window", "streamed", epochs=1)
    given = [_ids(next(streamed)) for _ in range(len(served) - 1)]
    assert [_ids(batch) for batch in served[:-1]] == given
    resumed = _reader(tmp_path, "larger-than-its-window", "served", epochs=1)
    resumed.set_state(streamed.get_state())
    assert [_ids(batch) for batch in resumed] == [_ids(batch) for batch in streamed] == [_ids(served[-1])]


def test_two_workers_read_a_window_by_index_as_their_stream_does(tmp_path):
    """Two loading workers, each a share of two files, read the batches their stream gives."""
    workers = Loading(workers=2, threads=1, read_buffer=1)
    served = _reader(tmp_path, "larger-than-its-window", "served", workers)
    streamed = _reader(tmp_path, "larger-than-its-window", "streamed", workers)
    assert ([_ids(batch) for batch in itertools.islice(served, 8)]
            == [_ids(batch) for batch in itertools.islice(streamed, 8)])


def test_an_elastic_window_saved_without_workers_resumes_with_two(tmp_path):
    """An elastic read of a window shuffle saved with no loading workers resumes with two at the batch it
    would give next."""
    def elastic(workers: int):
        files = _files(tmp_path / "served", (9, 7))
        chain = _read(files).repeat().shuffle(3, window=8).batch(2).elastic()
        loading = Loading(workers=workers, threads=1, read_buffer=1)
        reader = compile(chain, read="streamed", loading=loading, seed=7).train(DataPartition())
        assert isinstance(reader, Checkpointable), type(reader)
        return reader

    saved = elastic(0)
    for _ in range(3):
        next(saved)
    resumed = elastic(2)
    resumed.set_state(saved.get_state())
    assert ([_ids(batch) for batch in itertools.islice(resumed, 6)]
            == [_ids(batch) for batch in itertools.islice(saved, 6)])
