"""`dew.data.load` over the two providers, offline.

The TFDS half reads the prepared fixture `tools/datasets/tfds_reference.py`
wrote, and asserts that reading it imports no TensorFlow; this file is part
of what the no-TF environment runs, where absence is the fact rather than
the assertion. The Hugging Face half builds its tables and its streams from
local files and a local HTTP server, so nothing here downloads anything.
"""

import itertools
import json
import shutil
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
from dew.data import Checkpointable, Loading

FIXTURES = Path(__file__).parent / "fixtures" / "tfds"
PREPARED = FIXTURES / "dew_images" / "1.0.0"
READ = dict(loading=Loading(workers=0, threads=1, read_buffer=4, worker_buffer=2))
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

@pytest.mark.parametrize("source, message", [
    ("dew_images", "names no provider"),
    ("tf/dew_images", "names no provider"),
    ("tfds", "and no dataset"),
    ("tfds/", "and no dataset"),
])
def test_a_source_string_that_names_no_provider_is_refused(source, message):
    with pytest.raises(ValueError, match=message):
        dew.data.load(source, batch=4)


def test_a_hub_name_keeps_its_slashes():
    from dew.data.providers import provider_of

    assert provider_of("hf/owner/name") == ("hf", "owner/name")
    assert provider_of("tfds/dew_images") == ("tfds", "dew_images")


# ---------------------------------------------------------------------------
# Prepared TFDS ArrayRecords
# ---------------------------------------------------------------------------

pytest.importorskip("tensorflow_datasets", reason="needs the tfds extra")


def test_prepared_records_reach_batches_without_importing_tensorflow():
    """Reading prepared ArrayRecords is a read: preparation is what needs
    TensorFlow, and a training process that imported it would be paying for a
    dependency it never calls."""
    data = dew.data.load("tfds/dew_images", batch=4, split="train", val_split="test",
                         val_batches=1, path=str(PREPARED), preprocess=image_and_label,
                         **READ)

    assert data.records == 16 and data.batch == 4 and data.steps_per_epoch == 4
    batch = next(iter(data.train()))
    assert sorted(batch) == ["image", "label"]
    assert batch["image"].shape == (4, 8, 8, 3) and batch["image"].dtype == np.uint8
    assert data.val is not None
    validation = list(data.val())
    assert [int(v) for v in validation[0]["image"][:, 0, 0, 0]] == [26, 27, 28, 29]
    assert "tensorflow" not in sys.modules


def test_a_pass_over_the_prepared_split_reads_every_record_once():
    data = dew.data.load("tfds/dew_images", batch=4, split="train", path=str(PREPARED),
                         preprocess=image_and_label, **READ)

    epoch = list(itertools.islice(iter(data.train()), data.steps_per_epoch))
    pixels = np.concatenate([batch["image"][:, 0, 0, 0] for batch in epoch])
    np.testing.assert_array_equal(np.sort(pixels), np.arange(10, 26))


def test_the_data_dir_above_a_prepared_version_resolves_by_name():
    """A caller who prepared into a data_dir names the builder, not the
    version directory TFDS chose inside it."""
    data = dew.data.load("tfds/dew_images", batch=4, path=str(FIXTURES),
                         preprocess=image_and_label, **READ)

    assert data.records == 16


def test_a_version_directory_and_a_version_name_together_are_refused():
    with pytest.raises(ValueError, match="names its own config and version"):
        dew.data.load("tfds/dew_images", batch=4, path=str(PREPARED), version="1.0.0",
                      preprocess=image_and_label, **READ)


def test_a_builder_config_or_version_the_prepared_data_does_not_hold_is_refused():
    for options, message in (({"config": "nope"}, "no 'nope' config"),
                             ({"version": "9.9.9"}, "no version '9.9.9'")):
        with pytest.raises(FileNotFoundError, match=message):
            dew.data.load("tfds/dew_images", batch=4, path=str(FIXTURES), **options)
    with pytest.raises(FileNotFoundError, match="no prepared 'other'"):
        dew.data.load("tfds/other", batch=4, path=str(FIXTURES))


def test_prepared_metadata_decides_which_dataset_a_directory_holds(tmp_path):
    """Resolving a directory from names and then trusting the names would
    read whatever sits at that path. The version directory is copied under
    another builder's name, and the metadata inside it still speaks."""
    elsewhere = tmp_path / "other_builder" / "1.0.0"
    elsewhere.parent.mkdir(parents=True)
    shutil.copytree(PREPARED, elsewhere)

    with pytest.raises(ValueError, match="holds builder 'dew_images'"):
        dew.data.load("tfds/other_builder", batch=4, path=str(tmp_path),
                      preprocess=image_and_label, **READ)


def test_a_split_the_prepared_data_does_not_hold_is_refused():
    with pytest.raises(ValueError, match="holds no split 'valid'"):
        dew.data.load("tfds/dew_images", batch=4, split="valid", path=str(PREPARED))


def test_a_half_copied_prepared_dataset_is_refused_before_the_run(tmp_path):
    """A missing shard otherwise raises inside a grain worker on the first
    record that needed it, steps into a run."""
    copy = tmp_path / "dew_images" / "1.0.0"
    copy.parent.mkdir(parents=True)
    shutil.copytree(PREPARED, copy)
    next(copy.glob("*train.array_record*")).unlink()

    with pytest.raises(FileNotFoundError, match="Missing prepared ArrayRecord shard"):
        dew.data.load("tfds/dew_images", batch=4, path=str(copy))


def test_the_tfds_provider_names_an_option_it_does_not_take():
    with pytest.raises(TypeError, match=r"does not take \['builder_name'\]"):
        dew.data.load("tfds/dew_images", batch=4, path=str(PREPARED), builder_name="x")


def test_the_tfds_provider_needs_a_prepared_path():
    with pytest.raises(ValueError, match="never prepares its own data"):
        dew.data.load("tfds/dew_images", batch=4)


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
                         data_files=jsonl, **READ)

    assert data.records == ROWS and data.steps_per_epoch == 6
    stream = data.train()
    assert isinstance(stream, Checkpointable), "an Arrow split is random access"
    seen = indices(stream, 3)
    state = stream.get_state()
    rest = indices(stream, 2)
    resumed = data.train()
    resumed.set_state(state)

    assert indices(resumed, 2) == rest
    assert sorted(i for batch in seen for i in batch) == sorted(
        set(i for batch in seen for i in batch)), "no record twice inside a pass"


def test_an_arrow_split_holds_a_named_validation_split(jsonl):
    data = dew.data.load("hf/json", batch=4, split="train", val_split="train",
                         val_batches=2, preprocess=just_index, data_files=jsonl, **READ)

    assert data.val is not None
    assert indices(data.val(), 5) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_a_record_count_that_disagrees_with_the_split_is_refused(jsonl):
    with pytest.raises(ValueError, match="disagrees with the 24 records"):
        dew.data.load("hf/json", batch=4, records=8, preprocess=just_index,
                      data_files=jsonl, **READ)


def test_an_option_the_loader_does_not_know_reaches_datasets(jsonl):
    """An unknown hf option is not dropped: `load_dataset` owns those names,
    so a misspelling raises from the library that owns it."""
    with pytest.raises(TypeError, match="not_a_real_option"):
        dew.data.load("hf/json", batch=4, preprocess=just_index, data_files=jsonl,
                      not_a_real_option=1, **READ)


# ---------------------------------------------------------------------------
# Hugging Face, streamed
# ---------------------------------------------------------------------------

def test_a_streamed_split_reports_no_length_and_no_position(jsonl):
    """A stream cannot be listed and cannot be put back where it stopped, so
    it reports neither. `Trainer.fit` refuses `checkpoint_every` over a
    stream without the state pair, which is the contract this path takes."""
    data = dew.data.load("hf/json", batch=4, split="train", streaming=True,
                         preprocess=just_index, data_files=jsonl, **READ)

    assert data.records is None and data.steps_per_epoch is None
    stream = data.train()
    try:
        assert not isinstance(stream, Checkpointable)
        assert not hasattr(stream, "get_state") and not hasattr(stream, "set_state")
        read = indices(stream, 8)
    finally:
        stream.close()
    assert len(read) == 8, "a streamed training split reads pass after pass"
    assert sorted(i for batch in read[:6] for i in batch) == list(range(ROWS))


def test_a_streamed_pass_ends_and_a_bounded_one_ends_sooner(jsonl):
    data = dew.data.load("hf/json", batch=4, split="train", val_split="train",
                         val_batches=2, streaming=True, preprocess=just_index,
                         data_files=jsonl, **READ)

    assert data.val is not None
    passed = data.val()
    try:
        assert len(indices(passed, 10)) == 2
    finally:
        passed.close()


def test_a_streamed_row_is_transformed_by_its_own_rng(jsonl):
    """The transform is grain's `random_map`, so a row's draw is keyed by the
    row and repeats across two readings of the same stream."""
    def read():
        data = dew.data.load("hf/json", batch=4, split="train", streaming=True,
                             preprocess=just_index, data_files=jsonl, **READ)
        stream = data.train()
        try:
            return [(int(i), int(d)) for batch in itertools.islice(stream, 3)
                    for i, d in zip(batch["index"], batch["draw"])]
        finally:
            stream.close()

    first = read()
    assert read() == first
    assert len({draw for _, draw in first}) > 1, "the draws really vary"


@pytest.mark.parametrize("processes", [2, 4])
def test_a_streamed_split_is_shared_over_the_processes_without_losing_rows(
        monkeypatch, jsonl, processes):
    """`IterableDataset.shard` raises when a split has fewer physical shards
    than the pool has processes, which would lose the ranks past the shard
    count. The public node split keeps one row in `world_size` instead, so
    every rank reads and the ranks are disjoint."""
    import jax

    monkeypatch.setattr(jax, "process_count", lambda: processes)
    together = []
    for index in range(processes):
        monkeypatch.setattr(jax, "process_index", lambda index=index: index)
        data = dew.data.load("hf/json", batch=processes, split="train", streaming=True,
                             preprocess=just_index, data_files=jsonl,
                             loading=Loading(workers=0, threads=1, read_buffer=0,
                                             worker_buffer=2))
        stream = data.train()
        try:
            mine = [i for batch in indices(stream, ROWS // processes) for i in batch]
        finally:
            stream.close()
        assert mine, f"process {index} of {processes} read nothing"
        together.append(mine)
    flat = [index for mine in together for index in mine]
    assert len(flat) == len(set(flat)), "two processes read the same row"
    assert sorted(flat) == list(range(ROWS))


def test_a_streamed_position_says_why_it_cannot_be_restored(jsonl):
    """The refusal names the reason rather than the missing method: the
    shuffle buffer, the shard assignment and the pass number are part of the
    order dew reads and are not in what `datasets` restores."""
    from dew.data.sources.hf_stream import HFRows

    rows = iter(HFRows("json", "train", options={"data_files": jsonl}, seed=0, rank=0,
                       world_size=1, buffer=0, epochs=1))
    with pytest.raises(NotImplementedError, match="reports no position"):
        rows.set_state({"epoch": 0})
    rows.close()


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


def _prefetchers() -> int:
    return sum(1 for thread in threading.enumerate()
               if "prefetch" in thread.name.lower())


def test_a_streamed_read_over_http_is_bounded_and_its_reader_is_joined(served):
    """The rows come off a socket, the buffer ahead of the step is grain's and
    bounded by `worker_buffer`, and closing the stream leaves no reader
    behind: an unjoined prefetch thread would keep pulling the network after
    the run stopped asking."""
    data = dew.data.load("hf/json", batch=4, split="train", streaming=True,
                         preprocess=just_index, data_files=served,
                         loading=Loading(workers=0, threads=1, read_buffer=4,
                                         worker_buffer=2))
    before = _prefetchers()

    stream = data.train()
    read = indices(stream, 3)
    during = _prefetchers()
    stream.close()
    for _ in range(100):
        if _prefetchers() == before:
            break
        time.sleep(0.05)

    assert len(read) == 3 and all(len(batch) == 4 for batch in read)
    assert sorted(i for batch in read for i in batch) == sorted(
        set(i for batch in read for i in batch))
    assert during > before, "the read runs ahead of the step in a thread"
    assert _prefetchers() == before, "the streamed reader was not joined"
    assert _Rows.served >= 1, "nothing was read over the network"
