"""Hub, TFDS and file stores read through `compile`: a Hub split prepared at its pinned revision
when provisioned and left unprepared when a read is refused, and holdouts that keep each record on
one side."""

import itertools
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hub_support import native_builder

from dew.data import TFDS, Hub, Loading, presets
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline

FIXTURES = Path(__file__).parent / "fixtures"
READ = Loading(workers=0, threads=1, read_buffer=2, worker_buffer=1)


def run(source, *, held=256, batch=8, seed=0, read="indexed", provision=False):
    """`source` with its first `held` records held out to validate on and the rest as a run reads
    them: shuffled by `seed`, in windows of 64 where streamed, repeated, in batches of `batch`."""
    train, scored = Pipeline(source).holdout(records=held)
    order = train.shuffle(seed, window=64 if read == "streamed" else None).repeat()
    return compile(order.batch(batch), validation=scored.batch(batch), read=read, seed=seed, loading=READ,
                   provision=provision)


def batches(data, *, held=False):
    reader = data.val if held else data.train
    assert reader is not None
    stream = reader(DataPartition())
    try:
        taken = list(itertools.islice(stream, 1000 if held else data.steps_per_epoch))
        if held:
            assert next(stream, None) is None
        return taken
    finally:
        stream.close()


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
def test_hub_records_prepare_the_pinned_split_and_hold_out_its_first_256(hub, revision):
    builders, _ = hub
    pinned = "frozen-release" if revision else "frozen-default"
    assert not Path(builders[pinned].cache_dir, "dataset_info.json").exists()
    data = run(Hub("acme/simple", revision=revision), seed=19, read="memory", provision=True)
    assert Path(builders[pinned].cache_dir, "dataset_info.json").exists()
    assert data.records == data.held_out == 256
    train = [int(index) for batch in batches(data) for index in batch["id"]]
    held = [int(index) for batch in batches(data, held=True) for index in batch["id"]]
    offset = 1000 if revision else 0
    assert sorted(train) == list(range(offset + 256, offset + 512))
    assert held == list(range(offset, offset + 256))
    assert set(train).isdisjoint(held)


RACE = """
import importlib, io, sys, threading
from tqdm.contrib.concurrent import thread_map
bars = importlib.import_module(sys.argv[2]).tqdm
if sys.argv[1] == "read":
    from dew.data.sources.hub import hold_progress_lock
    hold_progress_lock()
if sys.argv[1] == "listing":
    from dew.data.sources.hub import split_files
    split_files(["data/train-00000.parquet"], "train")
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
"""A bar of the class `sys.argv[2]` names, opened while another thread maps shards through tqdm's
thread_map with that class, as `datasets` reads or prepares a split's shards and resolves a listing's
files, and closed once the map has ended, in a fresh interpreter: before any Hub call, or after a Hub
read (`hold_progress_lock`) or a listing's discovery (`split_files`)."""


@pytest.mark.parametrize("bars", ["datasets.utils.tqdm", "tqdm.auto"], ids=["datasets", "auto"])
@pytest.mark.parametrize("call", ["read", "listing"])
def test_a_bar_alive_across_another_threads_shard_read_closes_once_a_hub_call_holds_the_lock(call, bars):
    """tqdm's thread_map deletes its bar class's lock under a bar of another thread's map that outlives
    it, as two splits read at once, or two streamed sources discovered at once, do; after a Hub read or
    a listing's discovery, a lock tqdm's class holds is the one every map's class finds, so the map
    restores it and the bar closes, whichever class `datasets` lends it: its own for a listing's files
    and a split's shards, tqdm's default for a prepared split's."""
    import subprocess
    import sys

    def race(mode: str) -> str:
        return subprocess.run([sys.executable, "-c", RACE, mode, bars], capture_output=True, text=True,
                              check=True).stdout.strip()

    assert race("bare") != "[]"
    assert race(call) == "[]"


def test_a_refused_read_and_disabled_provisioning_leave_the_hub_cache_unprepared(hub):
    builders, _ = hub
    for options in ({"read": "streamed"}, {"read": "memory", "provision": False}):
        with pytest.raises(ValueError):
            run(Hub("acme/simple"), **options)
        assert not Path(builders["frozen-default"].cache_dir, "dataset_info.json").exists()
    data = run(Hub("acme/simple"), read="memory", provision=True)
    assert data.plan.order["read"] == "memory"
    assert sorted(int(value) for batch in batches(data) for value in batch["id"]) == list(range(256, 512))


@pytest.mark.parametrize("version", [None, "1.0.0"])
def test_tfds_reads_the_prepared_split_in_dews_cache_and_refuses_a_version_it_lacks(tmp_path, monkeypatch,
                                                                                   version):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    directory = tmp_path / "cache" / "dew" / "tfds" / "dew_images" / "1.0.0"
    shutil.copytree(FIXTURES / "tfds" / "dew_images" / "1.0.0", directory)
    source = TFDS("dew_images", version=version)
    assert source.metadata()["rows"] == 16
    data = run(source, held=4, batch=4, read="memory")
    train = [int(image[0, 0, 0]) for batch in batches(data) for image in batch["image"]]
    held = [int(image[0, 0, 0]) for batch in batches(data, held=True) for image in batch["image"]]
    assert sorted(train) == list(range(14, 26)) and held == list(range(10, 14))
    assert set(train).isdisjoint(held)
    with pytest.raises((FileNotFoundError, ValueError)):
        run(TFDS("dew_images", version="2.0.0"), batch=4, read="memory")


def test_a_stream_of_images_holds_out_a_hash_fraction_or_nothing_where_a_slice_is_refused(tmp_path):
    """A holdout of the first records reads by index, so a streamed read of a Parquet file refuses it
    before reading a row. The stream takes what that refusal names instead: a hash fraction, each
    record on one side of it, or no holdout."""
    import cv2
    import pyarrow as pa
    import pyarrow.parquet as pq

    from dew.data import Files

    rows = [{"image": {"bytes": cv2.imencode(".png", np.full((8, 8, 3), index % 256, np.uint8))[1].tobytes(),
                       "path": None}, "label": index} for index in range(320)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "images.parquet")
    source = Files(str(tmp_path / "images.parquet"))

    def loaded(train, held=None):
        return compile(train.shuffle(0, window=64).repeat().pipe(presets.images, 4, batch=1),
                       validation=None if held is None else held.pipe(presets.images, 4, batch=1),
                       read="streamed", loading=READ)

    def labels(reader, rows=None):
        stream = reader(DataPartition())
        try:
            return [int(label) for batch in itertools.islice(stream, rows) for label in batch["label"]]
        finally:
            stream.close()

    with pytest.raises(ValueError):
        loaded(*source.holdout(records=256))
    data = loaded(*source.holdout(0.1))
    assert data.val is not None
    scored = labels(data.val)
    # The training stream repeats a pass it shuffles in a window, so three passes' rows hold every record.
    trained = set(labels(data.train, 3 * 320))
    assert scored and trained.isdisjoint(scored) and sorted(trained | set(scored)) == list(range(320))
    whole = loaded(source)
    assert whole.val is None and sorted(set(labels(whole.train, 3 * 320))) == list(range(320))


