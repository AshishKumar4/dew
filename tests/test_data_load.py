"""Named data and run specs read the same chains, with disjoint default holdouts."""

import itertools
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hub_support import native_builder
from token_support import write_token_corpus

from dew.config.data import HoldOut, ImageData, RecordsData, TextData
from dew.data import TFDS, Hub, Loading, Memory, load, presets
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.objectives.base import VALID_ROWS

FIXTURES = Path(__file__).parent / "fixtures"
READ = Loading(workers=0, threads=1, read_buffer=2, worker_buffer=1)


def batches(data, *, held=False):
    reader = data.val if held else data.train
    assert reader is not None
    stream = reader(DataPartition())
    try:
        taken = list(itertools.islice(stream, 1000 if held else data.records // data.batch))
        if held:
            assert next(stream, None) is None
        return taken
    finally:
        stream.close()


def documents(rows):
    found = set()
    for batch in rows:
        kept = batch.get(VALID_ROWS, np.ones(len(batch["text"]), bool))
        laid = np.asarray(batch["text_coordinates"].segments)
        for tokens, segments in zip(batch["text"][kept], laid[kept], strict=True):
            for segment in np.unique(segments):
                if segment:
                    found.add(bytes(tokens[segments == segment][:-1].tolist()))
    return found


@pytest.fixture
def hub(tmp_path, monkeypatch):
    import datasets
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import GitRefs

    builders = {}
    texts = {}
    for revision, offset in (("frozen-default", 0), ("frozen-release", 1000)):
        ids = list(range(offset, offset + 512))
        texts[revision] = [f"document {index:04d} " + "x" * 113 for index in ids]
        split = datasets.Dataset.from_dict({"id": ids, "text": texts[revision]})
        builders[revision] = native_builder({"train": split}, tmp_path / f"{revision}-files",
                                            tmp_path / revision)

    def info(self, repo, revision=None, **options):
        assert repo == "acme/simple"
        assert revision in (None, "release", "frozen-default", "frozen-release")
        return SimpleNamespace(sha="frozen-release" if revision in ("release", "frozen-release")
                               else "frozen-default")

    def builder(repo, *, revision, **options):
        assert repo == "acme/simple"
        return builders[revision]

    monkeypatch.setattr(HfApi, "dataset_info", info)
    monkeypatch.setattr(HfApi, "list_repo_files", lambda *args, **options: ["train.tar"])
    monkeypatch.setattr(HfApi, "list_repo_refs", lambda *args, **options: GitRefs(
        branches=[], converts=[], tags=[]))
    monkeypatch.setattr(datasets, "load_dataset_builder", builder)
    return builders, texts


@pytest.mark.parametrize("revision", [None, "release"])
def test_named_hub_records_prepare_the_pinned_split_and_hold_out_its_first_256(hub, revision):
    builders, _ = hub
    pinned = "frozen-release" if revision else "frozen-default"
    assert not Path(builders[pinned].cache_dir, "dataset_info.json").exists()
    data = load("hf/acme/simple" + ("@release" if revision else ""), batch=8, seed=19, loading=READ)
    assert Path(builders[pinned].cache_dir, "dataset_info.json").exists()
    assert data.records == data.held_out == 256
    train = [int(index) for batch in batches(data) for index in batch["id"]]
    held = [int(index) for batch in batches(data, held=True) for index in batch["id"]]
    offset = 1000 if revision else 0
    assert sorted(train) == list(range(offset + 256, offset + 512))
    assert held == list(range(offset, offset + 256))
    assert set(train).isdisjoint(held)


@pytest.mark.parametrize("options", [{"seq_len": 127}, {"tokenizer": "byte"}])
@pytest.mark.parametrize("named", [True, False])
def test_text_forms_pack_each_document_on_only_one_side_of_the_default_holdout(hub, options, named):
    _, texts = hub
    source = Hub("acme/simple")
    data = load("hf/acme/simple" if named else source, batch=1, loading=READ, **options)
    train, held = source.holdout(0.01)
    reference = compile(presets.lm(train, "byte", seq_len=options.get("seq_len", 256), batch=1),
                        validation=presets.lm(held, "byte", seq_len=options.get("seq_len", 256), batch=1),
                        read="memory", loading=READ)
    trained, scored = documents(batches(data)), documents(batches(data, held=True))
    assert trained and scored and trained.isdisjoint(scored)
    assert trained | scored == {text.encode() for text in texts["frozen-default"]}
    assert trained == documents(batches(reference))
    assert scored == documents(batches(reference, held=True))
    assert all(batch["text"].shape[1] == options.get("seq_len", 256) + 1 for batch in batches(data))


RACE = """
import io, sys, threading
from datasets.utils.tqdm import tqdm as bars
from tqdm.contrib.concurrent import thread_map
if sys.argv[1] == "held":
    from dew.data.sources.hub import hold_progress_lock
    hold_progress_lock()
started, opened, failures = threading.Event(), threading.Event(), []
def other():
    started.wait()
    bar = bars(total=1, file=io.StringIO())
    opened.set()
    mapped.join()
    try:
        bar.close()
    except AttributeError as error:
        failures.append(str(error))
def shard(index):
    started.set()
    opened.wait()
    return index
mapped = threading.Thread(target=lambda: thread_map(shard, [0], tqdm_class=bars, file=io.StringIO()))
closer = threading.Thread(target=other)
closer.start(); mapped.start(); closer.join()
print(failures)
"""
"""A `datasets` bar opened while another thread reads shards through tqdm's thread_map, as `datasets` reads
a split's Arrow shards, and closed once the map has ended, in a fresh interpreter: before any read or after
the lock a Hub read holds (`hold_progress_lock`)."""


def test_a_bar_alive_across_another_threads_shard_read_closes_once_a_hub_read_holds_the_lock():
    """`datasets`' thread_map deletes its bar class's lock under a bar of another thread's read that outlives
    the map, as two splits read at once do; after a Hub read holds the class's lock, the map restores it
    and the bar closes."""
    import subprocess
    import sys

    def race(mode: str) -> str:
        return subprocess.run([sys.executable, "-c", RACE, mode], capture_output=True, text=True,
                              check=True).stdout.strip()

    assert race("bare") != "[]"
    assert race("held") == "[]"


def test_a_refused_read_and_disabled_provisioning_leave_the_hub_cache_unprepared(hub):
    builders, _ = hub
    for options in ({"read": "streamed"}, {"read": "memory", "provision": False}):
        with pytest.raises(ValueError):
            load("hf/acme/simple", batch=8, loading=READ, **options)
        assert not Path(builders["frozen-default"].cache_dir, "dataset_info.json").exists()
    data = load("hf/acme/simple", batch=8, read="memory", loading=READ)
    assert data.plan.order["read"] == "memory"
    assert sorted(int(value) for batch in batches(data) for value in batch["id"]) == list(range(256, 512))


@pytest.mark.parametrize("name", ["records", "acme/simple", "hf/", "hf/@release", "hf/acme/simple@",
                                  "hf://datasets/acme/simple", "tfds/", "tfds/dew_images:",
                                  "tfds://dew_images", "data/train.jsonl"])
def test_other_names_are_refused_before_reading_a_source(name):
    with pytest.raises(ValueError):
        load(name, batch=1)


@pytest.mark.parametrize("version", [None, "1.0.0"])
def test_named_tfds_reads_the_prepared_fixture_and_refuses_an_oversized_holdout(tmp_path, monkeypatch,
                                                                           version):
    pytest.importorskip("tensorflow_datasets", reason="needs the tfds extra")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    directory = tmp_path / "cache" / "dew" / "tfds" / "dew_images" / "1.0.0"
    shutil.copytree(FIXTURES / "tfds" / "dew_images" / "1.0.0", directory)
    with pytest.raises(ValueError):
        load("tfds/dew_images" + (":" + version if version else ""), batch=4, loading=READ,
             provision=False)
    source = TFDS("dew_images", version=version)
    assert source.metadata()["rows"] == 16
    data = RecordsData(source, validation=HoldOut(records=4)).load(batch=4, read="memory", loading=READ)
    train = [int(image[0, 0, 0]) for batch in batches(data) for image in batch["image"]]
    held = [int(image[0, 0, 0]) for batch in batches(data, held=True) for image in batch["image"]]
    assert sorted(train) == list(range(14, 26)) and held == list(range(10, 14))
    assert set(train).isdisjoint(held)
    with pytest.raises((FileNotFoundError, ValueError)):
        load("tfds/dew_images:2.0.0", batch=4, loading=READ, provision=False)


def test_a_dataset_keeps_its_steps_while_its_record_default_is_held_out():
    source = Memory([{"id": index} for index in range(320)]).map(
        lambda row: {"id": row["id"], "twice": 2 * row["id"]})
    data = load(source, batch=8, loading=READ, provision=False)
    train, held = batches(data), batches(data, held=True)
    trained = {int(value) for batch in train for value in batch["id"]}
    scored = {int(value) for batch in held for value in batch["id"]}
    assert trained == set(range(256, 320)) and scored == set(range(256))
    assert trained.isdisjoint(scored)
    for batch in train + held:
        np.testing.assert_array_equal(batch["twice"], 2 * batch["id"])


def draw(row, rng):
    return {**row, "draw": rng.integers(2**16)}


def test_load_keeps_an_indexed_datasets_random_draws_and_resumed_position_at_the_requested_seed():
    source = Memory([{"id": index} for index in range(320)]).random_map(draw)
    options = {"batch": 8, "read": "indexed", "seed": 37, "loading": READ}
    data = load(source, **options)
    train, held = source.holdout(records=256)
    reference = compile(train.batch(8), validation=held.batch(8), read="indexed", seed=37, loading=READ)
    actual, expected = data.train(DataPartition()), reference.train(DataPartition())
    changed = load(source, **{**options, "seed": 38}).train(DataPartition())
    try:
        first = next(actual)
        np.testing.assert_array_equal(first["draw"], next(expected)["draw"])
        assert not np.array_equal(first["draw"], next(changed)["draw"])
        saved = actual.get_state()
        following = next(actual)
        resumed = load(source, **options).train(DataPartition())
        try:
            resumed.set_state(saved)
            restored = next(resumed)
            for field in following:
                np.testing.assert_array_equal(restored[field], following[field])
        finally:
            resumed.close()
    finally:
        actual.close()
        expected.close()
        changed.close()


def test_text_data_defaults_end_documents_and_hold_out_tokens_without_replacing_named_validation():
    source = [{"text": "v" * 10 + "t" * 990}]
    data = TextData(source, seq_len=1).load(batch=1, read="memory", loading=READ)
    train, held = batches(data), batches(data, held=True)
    assert data.records == 990 and len(held) == 9
    assert {int(token) for batch in train for token in batch["text"].flat} == {ord("t"), 255}
    assert {int(token) for batch in held for token in batch["text"].flat} == {ord("v")}
    named = TextData(source, seq_len=1, validation=[{"text": "z" * 10}]).load(
        batch=1, read="memory", loading=READ)
    assert named.records == 1000
    assert {int(token) for batch in batches(named, held=True) for token in batch["text"].flat} == {
        ord("z"), 255}
    entire = TextData(source, seq_len=1, token_holdout=None).load(batch=1, read="memory", loading=READ)
    assert entire.records == 1000 and entire.val is None


def test_a_token_store_keeps_its_producer_validation_split_with_text_data_defaults(tmp_path):
    write_token_corpus(tmp_path, np.arange(100, 201, dtype=np.uint16),
                       np.arange(500, 525, dtype=np.uint16))
    data = TextData(str(tmp_path), seq_len=4).load(batch=1, read="memory", loading=READ)
    assert data.records == 25
    trained = {int(token) for batch in batches(data) for token in batch["text"].flat}
    scored = {int(token) for batch in batches(data, held=True) for token in batch["text"].flat}
    assert trained == set(range(100, 201)) and scored == set(range(500, 525))
    assert trained.isdisjoint(scored)


def test_a_stream_of_images_holds_out_a_hash_fraction_or_nothing_where_its_default_slice_is_refused(
        tmp_path):
    """The default holdout, the first 256 records, reads by index, so a streamed read of a Parquet
    file refuses it before reading a row. The stream takes what that refusal names instead: a hash
    fraction, each record on one side of it, or no holdout."""
    import cv2
    import pyarrow as pa
    import pyarrow.parquet as pq

    from dew.data import Files

    rows = [{"image": {"bytes": cv2.imencode(".png", np.full((8, 8, 3), index % 256, np.uint8))[1].tobytes(),
                       "path": None}, "label": index} for index in range(320)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "images.parquet")
    source = Files(str(tmp_path / "images.parquet"))

    def labels(reader, rows=None):
        stream = reader(DataPartition())
        try:
            return [int(label) for batch in itertools.islice(stream, rows) for label in batch["label"]]
        finally:
            stream.close()

    with pytest.raises(ValueError):
        ImageData(source, image_size=4, augment=None).load(batch=1, read="streamed", loading=READ)
    data = ImageData(source, image_size=4, augment=None, validation=HoldOut(fraction=0.1, by="hash")).load(
        batch=1, read="streamed", loading=READ)
    assert data.val is not None
    scored = labels(data.val)
    # The training stream repeats a pass it shuffles in a window, so three passes' rows hold every record.
    trained = set(labels(data.train, 3 * 320))
    assert scored and trained.isdisjoint(scored) and sorted(trained | set(scored)) == list(range(320))
    whole = ImageData(source, image_size=4, augment=None, validation=None).load(batch=1, read="streamed",
                                                                               loading=READ)
    assert whole.val is None and sorted(set(labels(whole.train, 3 * 320))) == list(range(320))


def test_image_data_defaults_hold_out_256_records_and_resize_both_parts():
    source = [{"image": np.full((8, 8, 3), index % 256, np.uint8), "label": index}
              for index in range(320)]
    data = ImageData(source, image_size=4, augment=None).load(batch=8, read="memory", loading=READ)
    train, held = batches(data), batches(data, held=True)
    trained = {int(label) for batch in train for label in batch["label"]}
    scored = {int(label) for batch in held for label in batch["label"]}
    assert data.records == 64 and data.held_out == 256
    assert trained == set(range(256, 320)) and scored == set(range(256))
    assert trained.isdisjoint(scored)
    for batch in train + held:
        np.testing.assert_array_equal(batch["image"], np.broadcast_to(
            (batch["label"] % 256).astype(np.uint8)[:, None, None, None], (8, 4, 4, 3)))
