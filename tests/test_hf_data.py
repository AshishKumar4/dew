"""Hugging Face datasets read through grain's random access.

Everything here is local: the tables are built in memory with
`Dataset.from_dict` and the hub is never called, `load_dataset` is replaced
where the name route is under test. The file needs the `datasets` package
itself, which is the streaming extra, so it skips without it.
"""

import hashlib
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from absl import flags

# grain's worker processes read absl flags; a test that never ran absl.app
# would trip UnparsedFlagAccessError at any worker_count > 0.
if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()

import grain.python as pygrain

datasets = pytest.importorskip("datasets")

from dew.data import HFImages, images  # noqa: E402
from dew.data.sources.hf import HFDatasetSource  # noqa: E402

RECORDS = 16
IMAGE_SIZE = 12
SCALE = 8


def _table(records=RECORDS, size=IMAGE_SIZE):
    """A tiny image and caption dataset, the shape a hub image dataset has."""
    from PIL import Image

    images_ = [
        Image.fromarray(
            np.random.RandomState(i).randint(0, 256, (size, size, 3), dtype=np.uint8))
        for i in range(records)
    ]
    return datasets.Dataset.from_dict({
        "image": images_,
        "caption": [f"caption number {i}" for i in range(records)],
        "index": list(range(records)),
    })


def keep_captions(captions):
    """A caption reader that hands the words back, so a test reads what the
    dataset wrote before a run's encoder tokenizes it."""
    return {"caption": np.asarray(captions)}


def _captions(*records):
    """The captions those records carry, in table order."""
    return [f"caption number {i}" for i in records]


@pytest.fixture
def hub(monkeypatch):
    """Hub names resolve to a local table, and record the load arguments."""
    calls = []

    def load_dataset(name, split=None, **kwargs):
        calls.append({"name": name, "split": split, **kwargs})
        return _table()

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    return calls


def _hub_images(**fields):
    return HFImages(name="acme/pets", image_size=SCALE, augmentation="none", worker_count=0,
                    read_threads=1, read_buffer=1, worker_buffer=1, **fields)


# ---------------------------------------------------------------------------------
# The source itself
# ---------------------------------------------------------------------------------

def test_a_wrapped_dataset_indexes_like_the_table():
    table = _table(records=4)
    source = HFDatasetSource(dataset=table)

    assert len(source) == len(table) == 4
    record = source[2]
    assert sorted(record) == ["caption", "image", "index"]
    assert record["caption"] == "caption number 2" and record["index"] == 2


def test_image_columns_come_back_as_arrays_not_pil_objects():
    """The transforms are numpy and cv2; a PIL image would reach cv2.resize."""
    record = HFDatasetSource(dataset=_table(records=2))[0]

    assert isinstance(record["image"], np.ndarray)
    assert record["image"].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert record["image"].dtype == np.uint8


def test_a_source_needs_a_name_or_a_dataset():
    with pytest.raises(ValueError, match="hub dataset name or a loaded dataset"):
        HFDatasetSource()


def test_pickling_leaves_the_table_behind_and_still_reads_the_records():
    """grain pickles the source per worker; the Arrow table must not ride
    along, so an in-memory table is written out and reopened instead."""
    source = HFDatasetSource(dataset=_table(records=4))
    payload = pickle.dumps(source)

    assert b"caption number 3" not in payload  # the rows stayed behind
    assert len(payload) < 1024

    reloaded = pickle.loads(payload)
    assert len(reloaded) == 4
    assert reloaded[3]["caption"] == "caption number 3"
    assert np.array_equal(reloaded[3]["image"], source[3]["image"])


def test_a_named_dataset_reloads_by_name_and_split(hub):
    source = HFDatasetSource(name="acme/pets", split="validation")
    assert len(source) == RECORDS
    assert hub == [{"name": "acme/pets", "split": "validation"}]

    payload = pickle.dumps(source)
    assert b"caption number 3" not in payload

    reloaded = pickle.loads(payload)
    assert reloaded[3]["caption"] == "caption number 3"
    # The worker's copy went back to load_dataset rather than carrying rows.
    assert hub == [{"name": "acme/pets", "split": "validation"}] * 2


def test_the_table_loads_once_under_concurrent_reads(monkeypatch):
    """grain reads a source from several threads at once. Two of them loading
    the same table raced inside datasets and tore down tqdm's lock."""
    loads = []

    def slow_load_dataset(name, split=None, **kwargs):
        loads.append(name)
        time.sleep(0.05)  # hold the first load open while the others arrive
        return _table()

    monkeypatch.setattr(datasets, "load_dataset", slow_load_dataset)
    source = HFDatasetSource(name="acme/pets")

    with ThreadPoolExecutor(max_workers=8) as pool:
        captions = list(pool.map(lambda index: source[index]["caption"], range(8)))

    assert captions == [f"caption number {i}" for i in range(8)]
    assert loads == ["acme/pets"]


def test_records_do_not_depend_on_worker_count():
    """Batch composition follows worker_count, record content must not: each
    worker takes its own slice of the index stream."""

    def by_record(worker_count):
        source = HFDatasetSource(dataset=_table())
        sampler = pygrain.IndexSampler(num_records=len(source), shuffle=True, seed=7,
                                       num_epochs=1, shard_options=pygrain.NoSharding())
        loader = pygrain.DataLoader(
            data_source=source, sampler=sampler,
            operations=[pygrain.Batch(4, drop_remainder=True)],
            worker_count=worker_count)
        return {int(index): (batch["image"][position].tobytes(),
                             str(batch["caption"][position]))
                for batch in loader
                for position, index in enumerate(batch["index"])}

    serial, parallel = by_record(0), by_record(2)
    assert sorted(serial) == list(range(RECORDS))
    assert serial == parallel


# ---------------------------------------------------------------------------------
# HFImages through the image pipeline
# ---------------------------------------------------------------------------------

def test_a_hub_dataset_spec_builds_the_image_pipeline(hub):
    """No registry entry per dataset and no path: the name carries the table."""
    data = _hub_images(val_batches=None).load(batch=4, tokenize=keep_captions)

    assert hub == [{"name": "acme/pets", "split": "train"}]
    assert data.records == RECORDS and data.val is None

    batch = next(data.train())
    assert batch["image"].shape == (4, SCALE, SCALE, 3)
    # The train stream shuffles, so the captions are the table's, in some order.
    assert set(map(str, batch["caption"])) <= set(_captions(*range(RECORDS)))
    # A caption dataset carries no class index, and the transform invents none.
    assert "label" not in batch


def test_the_split_is_a_field(hub):
    _hub_images(split="validation", val_batches=None).load(batch=4)

    assert hub == [{"name": "acme/pets", "split": "validation"}]


def test_the_caption_comes_from_the_record(hub):
    """The image transform is the one every image dataset uses; only where
    the caption comes from differs.

    The validation loader reads the held-out records in table order, so which
    caption belongs in which row is known here.
    """
    data = _hub_images(val_batches=1).load(batch=2, tokenize=keep_captions)
    batch = next(data.val())

    assert list(map(str, batch["caption"])) == _captions(0, 1)


def test_a_hub_dataset_holds_its_validation_batches_out_of_training(hub):
    data = _hub_images(val_batches=1).load(batch=4, tokenize=keep_captions)

    assert data.records == RECORDS - 4 and data.steps_per_epoch == 3
    batch = next(data.val())
    assert batch["image"].shape == (4, SCALE, SCALE, 3)
    # The held-out records, in table order.
    assert list(map(str, batch["caption"])) == _captions(0, 1, 2, 3)
