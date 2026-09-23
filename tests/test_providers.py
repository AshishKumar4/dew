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
from dew.data import Checkpointable, DataPartition, HFOptions, Loading, TFDSOptions

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
# A provider is a registered spec
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, options", [
    ("hf", HFOptions(config="20231101.en", revision="main", num_proc=2)),
    ("tfds", TFDSOptions(path="/data/prepared", config="small", version="1.0.0")),
])
def test_a_provider_spec_round_trips_through_a_run_config(name, options):
    """`dew.data.load` used to hand back a Dataset, which a run config cannot
    hold: nothing named the provider, so `to_dict` had nothing to write and a
    resumed run could not say which dataset it had read."""
    from dew.config import RunConfig
    from dew.registry import datasets as registry

    spec = registry[name](name="owner/rows", split="train[:80%]", val_split="test",
                          val_batches=2, options=options, seed=3)
    config = RunConfig(data=spec)

    record = config.to_dict()
    assert record["data"]["name"] == name
    assert record["data"]["fields"]["options"]["config"] == options.config
    assert RunConfig.from_dict(json.loads(json.dumps(record))) == config


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


def test_the_data_dir_above_a_prepared_version_resolves_by_name():
    """A caller who prepared into a data_dir names the builder, not the
    version directory TFDS chose inside it."""
    data = dew.data.load("tfds/dew_images", batch=4,
                         options=TFDSOptions(path=str(FIXTURES)),
                         preprocess=image_and_label, **READ)

    assert data.records == 16


def test_a_version_named_beside_a_resolved_path_is_an_identity_constraint():
    """A caller who has the version directory may still say which version it
    must hold; the metadata answers, so a match reads and a mismatch stops."""
    data = dew.data.load("tfds/dew_images", batch=4,
                         options=TFDSOptions(path=str(PREPARED), version="1.0.0"),
                         preprocess=image_and_label, **READ)
    assert data.records == 16

    with pytest.raises(ValueError, match="holds version '1.0.0'"):
        dew.data.load("tfds/dew_images", batch=4,
                      options=TFDSOptions(path=str(PREPARED), version="2.0.0"),
                      preprocess=image_and_label, **READ)


def test_a_builder_config_or_version_the_prepared_data_does_not_hold_is_refused():
    for options, message in (({"config": "nope"}, "no 'nope' config"),
                             ({"version": "9.9.9"}, "no version '9.9.9'")):
        with pytest.raises(FileNotFoundError, match=message):
            dew.data.load("tfds/dew_images", batch=4,
                          options=TFDSOptions(path=str(FIXTURES), **options))
    with pytest.raises(FileNotFoundError, match="no prepared 'other'"):
        dew.data.load("tfds/other", batch=4, options=TFDSOptions(path=str(FIXTURES)))


def test_prepared_metadata_decides_which_dataset_a_directory_holds(tmp_path):
    """Resolving a directory from names and then trusting the names would
    read whatever sits at that path. The version directory is copied under
    another builder's name, and the metadata inside it still speaks."""
    elsewhere = tmp_path / "other_builder" / "1.0.0"
    elsewhere.parent.mkdir(parents=True)
    shutil.copytree(PREPARED, elsewhere)

    with pytest.raises(ValueError, match="holds builder 'dew_images'"):
        dew.data.load("tfds/other_builder", batch=4,
                      options=TFDSOptions(path=str(tmp_path)),
                      preprocess=image_and_label, **READ)


def test_a_split_the_prepared_data_does_not_hold_is_refused():
    with pytest.raises(ValueError, match="holds no split 'valid'"):
        dew.data.load("tfds/dew_images", batch=4, split="valid",
                      options=TFDSOptions(path=str(PREPARED)))


def test_a_half_copied_prepared_dataset_is_refused_before_the_run(tmp_path):
    """A missing shard otherwise raises inside a grain worker on the first
    record that needed it, steps into a run."""
    copy = tmp_path / "dew_images" / "1.0.0"
    copy.parent.mkdir(parents=True)
    shutil.copytree(PREPARED, copy)
    next(copy.glob("*train.array_record*")).unlink()

    with pytest.raises(FileNotFoundError, match="Missing prepared ArrayRecord shard"):
        dew.data.load("tfds/dew_images", batch=4, options=TFDSOptions(path=str(copy)))


def test_an_option_no_provider_knows_is_refused_by_the_signature():
    with pytest.raises(TypeError, match="builder_name"):
        dew.data.load("tfds/dew_images", batch=4,
                      options=TFDSOptions(path=str(PREPARED)),
                      builder_name="x")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="data_files"):
        TFDSOptions(path=str(PREPARED), data_files="x")  # type: ignore[call-arg]


def test_an_option_of_the_other_provider_is_named():
    """One value per provider, so an option of the other one is the wrong
    type rather than a name checked against a list."""
    with pytest.raises(TypeError, match="the tfds provider reads TFDSOptions"):
        dew.data.load("tfds/dew_images", batch=4,
                      options=HFOptions(data_dir="/tmp"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="the hf provider reads HFOptions"):
        dew.data.load("hf/json", batch=4,
                      options=TFDSOptions(path="/tmp"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="are the hf provider's"):
        dew.data.load("tfds/dew_images", batch=4,
                      options=TFDSOptions(path=str(PREPARED)), streaming=True)


def test_a_decoder_the_caller_supplies_reaches_the_builder():
    """`SkipDecoding` hands back the bytes on disk, which is what a run that
    decodes images itself wants; the option is TFDS's and is forwarded."""
    import tensorflow_datasets as tfds

    data = dew.data.load("tfds/dew_images", batch=4,
                         options=TFDSOptions(path=str(PREPARED),
                                             decoders={"image": tfds.decode.SkipDecoding()}),
                         preprocess=lambda record, rng: {"raw": np.frombuffer(
                             record["image"], np.uint8)[:4]},
                         **READ)

    assert data.records == 16
    assert next(iter(data.train(DataPartition())))["raw"].shape == (4, 4)
    assert "tensorflow" not in sys.modules


def test_the_tfds_provider_needs_a_prepared_path():
    with pytest.raises(ValueError, match="never prepares its own data"):
        dew.data.load("tfds/dew_images", batch=4)


# ---------------------------------------------------------------------------
# Hugging Face, Arrow-backed
# ---------------------------------------------------------------------------

datasets = pytest.importorskip("datasets", reason="needs the streaming extra")


@pytest.fixture(scope="module")
def one_row(tmp_path_factory):
    """A split of a single row, so a pool of four leaves three ranks empty."""
    path = tmp_path_factory.mktemp("hf-one") / "row.jsonl"
    path.write_text(json.dumps({"index": 0}))
    return str(path)


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
        set(i for batch in seen for i in batch)), "no record twice inside a pass"


def test_an_arrow_split_holds_a_named_validation_split(jsonl):
    data = dew.data.load("hf/json", batch=4, split="train", val_split="train",
                         val_batches=2, preprocess=just_index,
                         options=HFOptions(data_files=jsonl), **READ)

    assert data.val is not None
    assert indices(data.val(DataPartition()), 5) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_a_record_count_that_disagrees_with_the_split_is_refused(jsonl):
    with pytest.raises(ValueError, match="disagrees with the 24 records"):
        dew.data.load("hf/json", batch=4, records=8, preprocess=just_index,
                      options=HFOptions(data_files=jsonl), **READ)


def test_the_library_own_arguments_are_forwarded_with_their_own_types(jsonl):
    """`features`, `storage_options` and a split-mapped `data_files` are
    `load_dataset`'s arguments, so they reach it as they are."""
    typed = dew.data.load(
        "hf/json", batch=4, preprocess=just_index,
        options=HFOptions(
            data_files={"train": jsonl}, storage_options={},
            features=datasets.Features({"index": datasets.Value("int32")})), **READ)

    assert typed.records == ROWS
    assert next(iter(typed.train(DataPartition())))["index"].dtype == np.int32

    both = dew.data.load("hf/json", batch=4, preprocess=just_index,
                         options=HFOptions(data_files=[jsonl, jsonl]), **READ)
    assert both.records == 2 * ROWS, "a sequence of files is read as one split"


def test_a_split_the_caller_already_has_is_read_as_it_is():
    """`dataset=` reads a table a caller built, which is the route for rows
    that never came from the hub."""
    table = datasets.Dataset.from_dict({"index": list(range(8))})

    data = dew.data.load("hf/in-memory", batch=4, dataset=table,
                         preprocess=just_index, **READ)

    assert data.records == 8
    assert sorted(i for b in indices(data.train(DataPartition()), 2) for i in b) == list(range(8))


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
            for i, d in zip(batch["index"], batch["draw"])]


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


def test_a_shuffled_streamed_split_withholds_its_position(jsonl):
    """The buffer a shuffle draws from is not in what `datasets` restores, so
    a shuffled stream reports no position and `Trainer.fit` refuses
    checkpoints over it."""
    data = dew.data.load("hf/json", batch=4, split="train", streaming=True,
                         shuffle_buffer=4, preprocess=just_index, options=HFOptions(data_files=jsonl),
                         **READ)

    stream = data.train(DataPartition())
    try:
        assert not isinstance(stream, Checkpointable)
        assert not hasattr(stream, "get_state") and not hasattr(stream, "set_state")
        assert len(indices(stream, 3)) == 3
    finally:
        stream.close()


def test_a_streamed_share_with_no_rows_is_refused_rather_than_waited_on(one_row):
    """A split shared out row by row leaves a share with nothing when the pool
    is larger than the split. Reopening that share for ever would spin inside
    next(), where a shutdown request cannot be seen."""
    one = dew.data.load("hf/json", batch=4, split="train", streaming=True,
                        preprocess=just_index, options=HFOptions(data_files=one_row), **READ)

    stream = one.train(DataPartition(3, 4))
    try:
        with pytest.raises(ValueError, match="was given none of the rows"):
            next(iter(stream))
    finally:
        stream.close()


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


def test_a_streamed_pass_ends_and_a_bounded_one_ends_sooner(jsonl):
    data = dew.data.load("hf/json", batch=4, split="train", val_split="train",
                         val_batches=2, streaming=True, preprocess=just_index,
                         options=HFOptions(data_files=jsonl), **READ)

    assert data.val is not None
    passed = data.val(DataPartition())
    try:
        assert len(indices(passed, 10)) == 2
    finally:
        passed.close()


def test_a_streamed_row_is_transformed_by_its_own_rng(jsonl):
    """The transform is grain's `random_map`, so a row's draw is keyed by the
    row and repeats across two readings of the same stream."""
    def read():
        data = dew.data.load("hf/json", batch=4, split="train", streaming=True,
                             preprocess=just_index, options=HFOptions(data_files=jsonl), **READ)
        stream = data.train(DataPartition())
        try:
            return [(int(i), int(d)) for batch in itertools.islice(stream, 3)
                    for i, d in zip(batch["index"], batch["draw"])]
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


def test_a_shuffled_streamed_position_says_why_it_cannot_be_restored(jsonl):
    """The refusal names the reason rather than the missing method."""
    from dew.data.sources.hf_stream import HFRows

    def open_split():
        return datasets.load_dataset("json", data_files=jsonl, split="train",
                                     streaming=True)

    shuffled = HFRows(open_split, what="the rows", seed=0, rank=0, world_size=1,
                      shuffle_buffer=4, epochs=1, given=False)
    assert not shuffled.resumable
    rows = iter(shuffled)
    with pytest.raises(NotImplementedError, match="shuffle buffer it was drawing from"):
        rows.set_state({"epoch": 0, "read": 0, "rows": None})
    rows.close()


def _stream_of(rows):
    """A streamed split over `rows`, as `dataset=` hands one over."""
    return datasets.Dataset.from_dict({"index": list(rows)}).to_iterable_dataset()


def test_a_given_streamed_dataset_reports_no_position(jsonl):
    """A caller's stream arrives through transformations dew did not apply,
    an upstream `.shuffle` among them, and an IterableDataset cannot be asked
    what it has been through. Advertising a position over it would restore a
    run onto rows nobody verified."""
    for rows in (_stream_of(range(12)), _stream_of(range(12)).shuffle(seed=9,
                                                                     buffer_size=8)):
        data = dew.data.load("hf/given", batch=4, dataset=rows, streaming=True,
                             records=12, preprocess=just_index, **READ)
        stream = data.train(DataPartition())
        try:
            assert not isinstance(stream, Checkpointable)
            assert not hasattr(stream, "get_state")
            assert len(indices(stream, 2)) == 2
        finally:
            stream.close()


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


def test_a_given_streamed_position_says_which_route_resumes():
    from dew.data.sources.hf_stream import HFRows

    given = HFRows(lambda: _stream_of(range(8)), what="the rows", seed=0, rank=0,
                   world_size=1, shuffle_buffer=0, epochs=1, given=True)
    assert not given.resumable
    rows = iter(given)
    with pytest.raises(NotImplementedError, match="dew builds and can resume"):
        rows.set_state({"epoch": 0, "read": 0, "rows": None})
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
