"""Hugging Face datasets read through grain's random access.

Everything here is local: the tables are built in memory with
`Dataset.from_dict` and the hub is never called, `load_dataset` is replaced
where the name route is under test. The file needs the `datasets` package
itself, which is the streaming extra, so it skips without it.
"""

import pickle

import grain.python as pygrain
import numpy as np
import pytest
from absl import flags

from dew.data import DataPartition, HFImages, Loading
from dew.data.sources.hf import HFDatasetSource

# grain's worker processes read absl flags; a test that never ran absl.app
# would trip UnparsedFlagAccessError at any worker_count > 0.
if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()


datasets = pytest.importorskip("datasets")


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

    # `path` is what the library calls its first parameter; the double has to
    # agree with it, because dew names every argument it forwards.
    def load_dataset(path, split=None, **kwargs):
        # The dataset and the split are what a reload has to get right; the
        # rest of what dew names are the library's own defaults.
        calls.append({"name": path, "split": split})
        return _table()

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    return calls


def _hub_images(**fields):
    return HFImages(name="acme/pets", image_size=SCALE, augmentation="none",
                    loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1),
                    **fields)


# ---------------------------------------------------------------------------------
# The source itself
# ---------------------------------------------------------------------------------

def test_a_wrapped_dataset_indexes_like_the_table_with_images_as_arrays():
    """The transforms are numpy and cv2; a PIL image would reach cv2.resize."""
    source = HFDatasetSource(dataset=_table(records=4))
    record = source[2]
    expected = np.random.RandomState(2).randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    assert len(source) == 4 and sorted(record) == ["caption", "image", "index"]
    assert record["caption"] == "caption number 2" and record["index"] == 2
    assert isinstance(record["image"], np.ndarray) and np.array_equal(record["image"], expected)


def test_pickling_leaves_the_table_behind_and_still_reads_the_records():
    """grain pickles the source per worker; the Arrow table must not ride
    along, so an in-memory table is written out and reopened instead."""
    source = HFDatasetSource(dataset=_table(records=4))
    payload = pickle.dumps(source)

    reloaded = pickle.loads(payload)
    assert len(reloaded) == 4
    assert reloaded[3]["caption"] == "caption number 3"
    assert np.array_equal(reloaded[3]["image"], source[3]["image"])


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

    batch = next(data.train(DataPartition()))
    assert batch["image"].shape == (4, SCALE, SCALE, 3)
    # The train stream shuffles, so the captions are the table's, in some order.
    assert set(map(str, batch["caption"])) <= set(_captions(*range(RECORDS)))
    # A caption dataset carries no class index, and the transform invents none.
    assert "label" not in batch


def test_a_hub_dataset_scores_a_named_split_instead_of_the_head(hub):
    """The hub dataset that ships a validation split was being scored on
    records held out of training instead, which cost the run those records
    and scored nothing the dataset's authors held out."""
    data = _hub_images(val_split="validation", val_batches=1).load(
        batch=4, tokenize=keep_captions)

    assert data.records == RECORDS, "the named split holds nothing out"
    assert data.val is not None
    assert len(list(data.val(DataPartition()))) == 1, "val_batches bounds the pass"
    assert list(map(str, next(data.val(DataPartition()))["caption"])) == _captions(0, 1, 2, 3)
    # The training records come from `split` and the pass from `val_split`,
    # each opened as its own source.
    assert {call["split"] for call in hub} == {"train", "validation"}


# ---------------------------------------------------------------------------------
# Columns, uncaptioned sets and validation augmentation
# ---------------------------------------------------------------------------------

def _classes(records=RECORDS, size=IMAGE_SIZE):
    """A class-labelled image table without captions, laid out as CIFAR-10 is:
    the image under "img", the class under "label". Record i is one flat
    colour, stored grey, 16-bit grey, as a palette, or fully transparent."""
    from PIL import Image

    def stored(index):
        grey = np.full((size, size), 16 * index, np.uint8)
        return (Image.fromarray(grey),
                Image.fromarray(grey.astype(np.uint16) * 256 + 255),
                Image.fromarray(np.dstack([grey] * 3)).quantize(),
                Image.fromarray(np.dstack([grey] * 3 + [grey * 0])))[index % 4]

    return datasets.Dataset.from_dict({
        "img": [stored(i) for i in range(records)],
        "label": [i % 10 for i in range(records)],
    })


@pytest.fixture
def classes(monkeypatch):
    monkeypatch.setattr(datasets, "load_dataset", lambda path, split=None, **kwargs: _classes())


def test_an_uncaptioned_class_dataset_reads_as_rgb_whatever_mode_it_stores(classes):
    """Loaded without a caption reader, a split with no caption column needs
    no caption_columns=(), and grey, 16-bit, palette and alpha images reach
    the batch as the RGB colour they show, a transparent one as white."""
    data = _hub_images(image_column="img", val_batches=1).load(batch=4)

    batch = next(data.val(DataPartition()))
    assert batch["image"].shape == (4, SCALE, SCALE, 3) and batch["image"].dtype == np.uint8
    np.testing.assert_array_equal(batch["image"][:, 0, 0], np.repeat([[0], [16], [32], [255]], 3, 1))
    np.testing.assert_array_equal(batch["label"], [0, 1, 2, 3])
    assert "caption" not in batch


def test_validation_is_scored_on_unaugmented_images(hub):
    """Training augments; a validation pass that crops, flips and jitters
    scores different images from the ones a reference metric reads."""
    def first_validation(**fields):
        data = HFImages(name="acme/pets", image_size=SCALE, val_batches=1,
                        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1),
                        **fields).load(batch=4, tokenize=keep_captions)
        return next(data.val(DataPartition()))["image"]

    plain = first_validation(augmentation="none")
    np.testing.assert_array_equal(first_validation(augmentation="flip_jitter"), plain)
    np.testing.assert_array_equal(
        first_validation(augmentation="flip_jitter", crop_scale=(0.5, 0.5), augmentation_size=SCALE * 2),
        plain)
