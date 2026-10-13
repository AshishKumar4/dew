"""`dew.data.load` over the two providers, offline.

The TFDS half reads the prepared fixture `tools/datasets/tfds_reference.py`
wrote, and asserts that reading it imports no TensorFlow; this file is part
of what the no-TF environment runs, where absence is the fact rather than
the assertion. The Hugging Face half builds its tables and its streams from
local files and a local HTTP server, so nothing here downloads anything.
"""

import itertools
import json
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import pytest
from absl import flags

# grain's worker processes read absl flags; a test that never ran absl.app
# would trip UnparsedFlagAccessError at any worker_count > 0.
if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()

import dew.data
from dew.data import Checkpointable, DataPartition, HFOptions, Loading, TFDSOptions

FIXTURES = Path(__file__).parent / "fixtures" / "tfds"
PREPARED = FIXTURES / "dew_images" / "1.0.0"
READ = {"loading": Loading(workers=0, threads=1, read_buffer=4, worker_buffer=2)}
ROWS = 24


def image_and_label(record, rng):
    """The explicit preprocessing a tfds image record needs to become a batch."""
    return {"image": record["image"], "label": np.int32(record["label"])}


def just_index(record, rng):
    """The index of a row, plus a draw from the row's own rng."""
    return {"index": np.int32(record["index"]),
            "draw": np.int32(rng.integers(1 << 20))}


def indices(iterator, batches):
    return [[int(value) for value in batch["index"]]
            for batch in itertools.islice(iterator, batches)]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# A provider is a spec
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Prepared TFDS ArrayRecords
# ---------------------------------------------------------------------------

pytest.importorskip("tensorflow_datasets", reason="needs the tfds extra")


def test_the_load_function_and_the_spec_read_the_same_dataset():
    """`load` is the spec built and called, so the two routes cannot drift."""
    from dew.registry import datasets as registry

    through_load = dew.data.load("tfds/dew_images", batch=4,
                                 options=TFDSOptions(path=str(PREPARED)),
                                 preprocess=image_and_label, **READ)
    spec = registry["tfds"](name="dew_images", options=TFDSOptions(path=str(PREPARED)),
                            preprocess=image_and_label, **READ)
    through_spec = spec.load(batch=4)

    assert through_load.records == through_spec.records
    for left, right in zip(indices_of(through_load, 2), indices_of(through_spec, 2),
                           strict=True):
        np.testing.assert_array_equal(left, right)


def indices_of(data, batches):
    """The first pixel of each image of the first `batches` batches."""
    return [batch["image"][:, 0, 0, 0] for batch in itertools.islice(data.train(DataPartition()), batches)]


def test_prepared_records_reach_batches_without_importing_tensorflow():
    """Reading prepared ArrayRecords is a read: preparation is what needs
    TensorFlow, and a training process that imported it would be paying for a
    dependency it never calls."""
    data = dew.data.load("tfds/dew_images", batch=4, split="train", val_split="test",
                         val_batches=1, options=TFDSOptions(path=str(PREPARED)),
                         preprocess=image_and_label,
                         **READ)

    assert data.records == 16 and data.batch == 4 and data.steps_per_epoch == 4
    batch = next(iter(data.train(DataPartition())))
    assert sorted(batch) == ["image", "label"]
    assert batch["image"].shape == (4, 8, 8, 3) and batch["image"].dtype == np.uint8
    assert data.val is not None
    validation = list(data.val(DataPartition()))
    assert [int(v) for v in validation[0]["image"][:, 0, 0, 0]] == [26, 27, 28, 29]
    assert "tensorflow" not in sys.modules


def test_a_pass_over_the_prepared_split_reads_every_record_once():
    data = dew.data.load("tfds/dew_images", batch=4, split="train",
                         options=TFDSOptions(path=str(PREPARED)),
                         preprocess=image_and_label, **READ)

    epoch = list(itertools.islice(iter(data.train(DataPartition())), data.steps_per_epoch))
    pixels = np.concatenate([batch["image"][:, 0, 0, 0] for batch in epoch])
    np.testing.assert_array_equal(np.sort(pixels), np.arange(10, 26))


# ---------------------------------------------------------------------------
# Hugging Face, Arrow-backed
# ---------------------------------------------------------------------------

datasets = pytest.importorskip("datasets", reason="needs the streaming extra")


@pytest.fixture(scope="module")
def jsonl(tmp_path_factory):
    """`ROWS` rows of one integer column, as a local file `load_dataset` reads."""
    path = tmp_path_factory.mktemp("hf") / "rows.jsonl"
    path.write_text("\n".join(json.dumps({"index": i}) for i in range(ROWS)))
    return str(path)


def test_an_arrow_split_reads_by_index_and_carries_its_position(jsonl):
    data = dew.data.load("hf/json", batch=4, split="train", preprocess=just_index,
                         options=HFOptions(data_files=jsonl), **READ)

    assert data.records == ROWS and data.steps_per_epoch == 6
    stream = data.train(DataPartition())
    assert isinstance(stream, Checkpointable), "an Arrow split is random access"
    seen = indices(stream, 3)
    state = stream.get_state()
    rest = indices(stream, 2)
    resumed = data.train(DataPartition())
    resumed.set_state(state)

    assert indices(resumed, 2) == rest
    assert sorted(i for batch in seen for i in batch) == sorted(
        {i for batch in seen for i in batch}), "no record twice inside a pass"


def test_an_arrow_split_holds_a_named_validation_split(jsonl):
    data = dew.data.load("hf/json", batch=4, split="train", val_split="train",
                         val_batches=2, preprocess=just_index,
                         options=HFOptions(data_files=jsonl), **READ)

    assert data.val is not None
    assert indices(data.val(DataPartition()), 5) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_a_text_split_reads_as_token_windows_tokenized_once(tmp_path, monkeypatch):
    """`tokenizer=` tokenizes the split's text column into dew's cache: the
    stream is the rows, each ended with the eos id, and a second load reads
    those files without writing them again. The spec it builds is what a run
    config records."""
    from dew.config import RunConfig
    from dew.data import HubText, TokenWindows

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    texts = [f"{index}: to be, or not to be\n" * (index + 1) for index in range(12)]
    (tmp_path / "plays.jsonl").write_text("".join(json.dumps({"text": text}) + "\n" for text in texts))
    hub = HubText(name="json", options=HFOptions(data_files=str(tmp_path / "plays.jsonl")))
    first = dew.data.load("hf/json", batch=4, tokenizer="byte", seq_len=16, options=hub.options, **READ)
    written = {path.name: path.stat().st_mtime_ns for path in hub.directory.iterdir()}
    spec = TokenWindows(hub=hub, seq_len=16, **READ)
    batch = next(spec.load(batch=4).train(DataPartition()))
    stream = b"".join((hub.directory / split).read_bytes() for split in ("val.bin", "train.bin"))

    assert batch["text"].shape == (4, 17) and batch["text"].dtype == np.int32
    assert stream == b"".join(text.encode() + b"\xff" for text in texts)
    assert all(bytes(row.astype(np.uint8).tolist()) in stream for row in batch["text"])
    assert spec.load(batch=4).records == first.records
    assert {path.name: path.stat().st_mtime_ns for path in hub.directory.iterdir()} == written
    config = RunConfig(data=spec)
    assert RunConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


def test_a_split_the_caller_already_has_is_read_as_it_is():
    """`dataset=` reads a table a caller built, which is the route for rows
    that never came from the hub."""
    table = datasets.Dataset.from_dict({"index": list(range(8))})

    data = dew.data.load("hf/in-memory", batch=4, dataset=table,
                         preprocess=just_index, **READ)

    assert data.records == 8
    assert sorted(i for b in indices(data.train(DataPartition()), 2) for i in b) == list(range(8))


def test_rows_without_preprocess_arrive_as_arrays_of_32_bit_fields():
    """A row's lists and Python numbers become arrays, so a list column is
    one [batch, n] field rather than n separate ones, and the 64-bit types
    `datasets` hands back narrow to what a device holds."""
    table = datasets.Dataset.from_dict({
        "x": [[float(i), -float(i)] for i in range(8)],
        "label": list(range(8)),
        "name": [f"row {i}" for i in range(8)]})

    batch = next(dew.data.load("hf/in-memory", batch=4, dataset=table, **READ).train(DataPartition()))

    assert batch["x"].shape == (4, 2) and batch["x"].dtype == np.float32
    np.testing.assert_array_equal(batch["x"][:, 0], -batch["x"][:, 1])
    assert batch["label"].shape == (4,) and batch["label"].dtype == np.int32
    assert sorted(batch["name"].tolist()) == sorted(f"row {i}" for i in batch["label"].tolist())


# ---------------------------------------------------------------------------
# Hugging Face, streamed
# ---------------------------------------------------------------------------

def test_a_streamed_split_reports_no_length(jsonl):
    """A stream cannot be listed, so it reports no record count and the run
    gives its length in steps."""
    data = dew.data.load("hf/json", batch=4, split="train", streaming=True,
                         preprocess=just_index, options=HFOptions(data_files=jsonl), **READ)

    assert data.records is None and data.steps_per_epoch is None
    stream = data.train(DataPartition())
    try:
        read = indices(stream, 8)
    finally:
        stream.close()
    assert len(read) == 8, "a streamed training split reads pass after pass"
    assert sorted(i for batch in read[:6] for i in batch) == list(range(ROWS))


def _drawn(stream, batches):
    return [(int(i), int(d)) for batch in itertools.islice(stream, batches)
            for i, d in zip(batch["index"], batch["draw"], strict=True)]


def test_an_unshuffled_streamed_split_resumes_on_the_record_it_stopped_at(jsonl):
    """`datasets` restores an unshuffled stream exactly and grain composes
    that with the transform and the batch behind it, so both the records and
    their own draws come back."""
    data = dew.data.load("hf/json", batch=4, split="train", streaming=True,
                         preprocess=just_index, options=HFOptions(data_files=jsonl), **READ)

    stream = data.train(DataPartition())
    assert isinstance(stream, Checkpointable)
    _drawn(stream, 3)
    state = stream.get_state()
    rest = _drawn(stream, 3)
    stream.close()

    resumed = data.train(DataPartition())
    resumed.set_state(state)
    try:
        assert _drawn(resumed, 3) == rest
    finally:
        resumed.close()


@pytest.mark.parametrize("read_buffer", [1, 8])
@pytest.mark.parametrize("shuffle_buffer", [0, 8])
def test_a_streamed_validation_pass_is_ordered_whatever_the_tuning(
        jsonl, shuffle_buffer, read_buffer):
    """A pass is the split in its own order. `shuffle_buffer` is the training
    shuffle and does not reach it, and `Loading` is performance only, so
    neither may change which rows a score is over."""
    data = dew.data.load("hf/json", batch=4, split="train", val_split="train",
                         val_batches=2, streaming=True, shuffle_buffer=shuffle_buffer,
                         preprocess=just_index, options=HFOptions(data_files=jsonl),
                         loading=Loading(workers=0, threads=1, read_buffer=read_buffer,
                                         worker_buffer=2))

    assert data.val is not None
    passed = data.val(DataPartition())
    try:
        assert indices(passed, 3) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    finally:
        passed.close()


@pytest.mark.parametrize("batch", [5, 30])
def test_a_streamed_pass_scores_its_last_rows_and_marks_what_fills_their_batch(jsonl, batch):
    """24 rows at batch 5 end in four rows and one repeat; at batch 30, one
    batch of all 24 and six repeats. Every row is read once as a real row,
    every batch is whole, and `VALID_ROWS` marks the repeats."""
    from dew.objectives.base import VALID_ROWS

    data = dew.data.load("hf/json", batch=batch, split="train", val_split="train", streaming=True,
                         preprocess=just_index, options=HFOptions(data_files=jsonl), **READ)
    assert data.val is not None
    passed = data.val(DataPartition())
    try:
        batches = list(passed)
    finally:
        passed.close()
    assert {len(read["index"]) for read in batches} == {batch}
    real = [int(index) for read in batches
            for index, kept in zip(read["index"], read.get(VALID_ROWS, np.ones(batch, bool)), strict=True)
            if kept]
    assert real == list(range(ROWS))


def test_a_streamed_row_is_transformed_by_its_own_rng(jsonl):
    """The transform is grain's `random_map`, so a row's draw is keyed by the
    row and repeats across two readings of the same stream."""
    def read():
        data = dew.data.load("hf/json", batch=4, split="train", streaming=True,
                             preprocess=just_index, options=HFOptions(data_files=jsonl), **READ)
        stream = data.train(DataPartition())
        try:
            return [(int(i), int(d)) for batch in itertools.islice(stream, 3)
                    for i, d in zip(batch["index"], batch["draw"], strict=True)]
        finally:
            stream.close()

    first = read()
    assert read() == first
    assert len({draw for _, draw in first}) > 1, "the draws really vary"


@pytest.mark.parametrize("processes", [2, 4])
def test_a_streamed_split_is_shared_over_the_processes_without_losing_rows(jsonl, processes):
    """`IterableDataset.shard` raises when a split has fewer physical shards
    than the pool has shares, which would lose the ranks past the shard
    count. The public node split keeps one row in `world_size` instead, so
    every share reads and the shares are disjoint."""
    together = []
    for index in range(processes):
        data = dew.data.load("hf/json", batch=processes, split="train", streaming=True,
                             shuffle_buffer=4, preprocess=just_index, options=HFOptions(data_files=jsonl),
                             loading=Loading(workers=0, threads=1, read_buffer=4,
                                             worker_buffer=2))
        stream = data.train(DataPartition(index, processes))
        try:
            mine = [i for batch in indices(stream, ROWS // processes) for i in batch]
        finally:
            stream.close()
        assert mine, f"process {index} of {processes} read nothing"
        together.append(mine)
    flat = [index for mine in together for index in mine]
    assert len(flat) == len(set(flat)), "two processes read the same row"
    assert sorted(flat) == list(range(ROWS))


def _stream_of(rows):
    """A streamed split over `rows`, as `dataset=` hands one over."""
    return datasets.Dataset.from_dict({"index": list(rows)}).to_iterable_dataset()


def test_every_pass_over_a_given_streamed_dataset_starts_at_its_beginning():
    """Each pass reads a copy, so no pass and no later factory call inherits
    where an earlier one stopped, and the caller's own object is untouched."""
    rows = _stream_of(range(8))
    data = dew.data.load("hf/given", batch=4, dataset=rows, streaming=True, records=8,
                         preprocess=just_index, **READ)

    first = data.train(DataPartition())
    try:
        crossing = indices(first, 4)
    finally:
        first.close()
    again = data.train(DataPartition())
    try:
        fresh = indices(again, 1)
    finally:
        again.close()

    assert crossing == [[0, 1, 2, 3], [4, 5, 6, 7]] * 2, "a pass reads the whole stream"
    assert fresh == [[0, 1, 2, 3]], "a fresh stream starts at the first row"
    assert [int(row["index"]) for row in rows] == list(range(8)), "the caller's own"


# ---------------------------------------------------------------------------
# A streamed read over the network, bounded, and its reader cleaned up
# ---------------------------------------------------------------------------

class _Rows(BaseHTTPRequestHandler):
    """Serves the rows once per request, counting the requests it answered."""

    served = 0

    def do_GET(self):
        body = "\n".join(json.dumps({"index": i}) for i in range(ROWS)).encode()
        type(self).served += 1
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def served():
    """A local HTTP server holding the rows, and its url."""
    server = socketserver.TCPServer(("127.0.0.1", 0), _Rows)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/rows.jsonl"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _prefetchers() -> set[threading.Thread]:
    return {thread for thread in threading.enumerate()
            if "prefetch" in thread.name.lower()}


def test_a_streamed_read_over_http_is_bounded_and_its_reader_is_joined(served):
    """The rows come off a socket, the read ahead of the step is bounded by
    `worker_buffer` batches, and closing the stream leaves no reader behind:
    an unjoined prefetch thread would keep pulling the network after the run
    stopped asking."""
    seen: list[int] = []

    def counted(record, rng):
        seen.append(int(record["index"]))
        return just_index(record, rng)

    batch, ahead = 4, 2
    data = dew.data.load("hf/json", batch=batch, split="train", streaming=True,
                         preprocess=counted, options=HFOptions(data_files=served),
                         loading=Loading(workers=0, threads=1, read_buffer=4,
                                         worker_buffer=ahead))
    before = _prefetchers()

    stream = data.train(DataPartition())
    read = indices(stream, 1)
    readers = _prefetchers() - before
    time.sleep(0.5)
    settled = len(seen)
    time.sleep(0.5)
    stream.close()
    for reader in readers:
        reader.join(timeout=5)

    assert read == [[0, 1, 2, 3]]
    assert readers, "the read runs ahead of the step in a thread"
    # One batch handed out, `ahead` buffered, one being filled behind them.
    assert settled == len(seen) <= (ahead + 2) * batch, (
        f"the read ran {len(seen)} rows ahead of one batch")
    assert not any(reader.is_alive() for reader in readers), "the streamed reader was not joined"
    assert _Rows.served >= 1, "nothing was read over the network"
