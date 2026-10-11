"""Prepared TFDS ArrayRecords decoded and batched without dataset preparation.

The twenty constant RGB images come from tools/datasets/tfds_reference.py.
These tests run with the tfds extra, including in the Python 3.14 environment
where TensorFlow is not installed. They require no downloads or preparation.
"""

import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from dew.data import TFDS, DataPartition, HoldOut, Images, Loading, load
from dew.data.images import decode_image
from dew.data.steps.image import ImageRecord

tfds = pytest.importorskip("tensorflow_datasets", reason="needs the tfds extra")

FIXTURE = Path(__file__).parent / "fixtures" / "tfds" / "dew_images" / "1.0.0"


def test_prepared_records_carry_encoded_pixels_labels_and_caption_names():
    """The source hands out the bytes on disk; `record` decodes them, so a
    worker pays one decode at the resolution it needs and no more."""
    spec = Images(source=TFDS("dew_images", path=str(FIXTURE), split="all",
        decoders={"image": tfds.decode.SkipDecoding()}), caption_templates=("a photo of a {}",))
    source = spec.source.records("indexed")
    assert len(source) == 20
    for index in range(len(source)):
        record = source[index]
        assert isinstance(record["image"], bytes)
        np.testing.assert_array_equal(decode_image(record["image"]), np.full((8, 8, 3), index + 10, np.uint8))
        assert record["label"] == index % 2
        result = ImageRecord(8, 8, templates=spec.caption_templates, classes=("red",
            "blue")).random_map(record, np.random.default_rng(0))
        caption, label = result["caption"], result["label"]
        assert ("red" if label == 0 else "blue") in caption.split()
    test_split = TFDS("dew_images", path=str(FIXTURE), split="test",
        decoders={"image": tfds.decode.SkipDecoding()}).records("indexed")
    assert [int(decode_image(test_split[i]["image"])[0, 0, 0]) for i in range(len(test_split))] == [
        26,
        27,
        28,
        29,
    ]


def test_a_prepared_source_describes_the_directory_and_split_it_reads():
    """A saved position names the order it counts into by naming its source,
    and the builder's own source describes itself by its address in this
    process, which no resume can match."""

    source = TFDS("dew_images", path=str(FIXTURE), split="all",
        decoders={"image": tfds.decode.SkipDecoding()})
    other = TFDS("dew_images", path=str(FIXTURE), split="test",
        decoders={"image": tfds.decode.SkipDecoding()})

    assert str(FIXTURE) in source.describe() and ":all:" in source.describe()
    assert other.snapshot() != source.snapshot()


@pytest.mark.parametrize("workers", [0, 2])
def test_prepared_records_reach_grain_batches_without_split_overlap(workers):
    data = load(Images(source=TFDS("dew_images", path=str(FIXTURE), split="all",
        decoders={"image": tfds.decode.SkipDecoding()}),
                       image_size=8, augment=None, validation=HoldOut(records=4),
                       caption_templates=("a photo of a {}",)), batch=4, read="memory",
                loading=Loading(workers=workers, threads=1, read_buffer=8, worker_buffer=2))
    source = data.train(DataPartition())
    train = list(itertools.islice(source, 4))
    validation = list(data.val(DataPartition()))
    assert data.records == 16
    train_pixels = np.concatenate([row["image"][:, 0, 0, 0] for row in train])
    val_pixels = np.concatenate([row["image"][:, 0, 0, 0] for row in validation])
    np.testing.assert_array_equal(np.sort(train_pixels), np.arange(14, 30))
    np.testing.assert_array_equal(val_pixels, np.arange(10, 14))
    for row in train + validation:
        assert row["image"].dtype == np.uint8
        np.testing.assert_array_equal(row["label"], (row["image"][:, 0, 0, 0] - 10) % 2)
        np.testing.assert_array_equal(
            row["image"], np.broadcast_to(row["image"][:, :1, :1, :1], (4, 8, 8, 3)))


def test_explicit_caption_names_override_the_prepared_metadata(tmp_path):
    labels = tmp_path / "names.txt"
    labels.write_text("fern\nrose\n")
    spec = Images(source=TFDS("dew_images", path=str(FIXTURE), split="all",
        decoders={"image": tfds.decode.SkipDecoding()}), labels=str(labels),
            caption_templates=("a photo of a {}",))
    source = spec.source.records("indexed")
    result = ImageRecord(8, 8, templates=spec.caption_templates,
        classes=tuple(labels.read_text().splitlines())).random_map(source[1], np.random.default_rng(0))
    caption, label = result["caption"], result["label"]
    assert label == 1 and "rose" in caption.split()


def test_an_unprepared_directory_requests_external_preparation(tmp_path):
    with pytest.raises(FileNotFoundError, match="has no prepared ArrayRecord data"):
        TFDS("dew_images", path=str(tmp_path), split="all").records("indexed")
    assert list(tmp_path.iterdir()) == []


def test_a_tfrecord_preparation_is_refused_before_reading(tmp_path):
    directory = tmp_path / "prepared"
    shutil.copytree(FIXTURE, directory)
    metadata_path = directory / "dataset_info.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["fileFormat"] = "tfrecord"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="dew reads ArrayRecords"):
        TFDS("dew_images", path=str(directory), split="all").records("indexed")


def test_missing_records_request_preparation_only_for_the_selected_split(tmp_path):
    """A missing shard otherwise raises inside a grain worker on the first
    record that needed it, steps into a run."""
    directory = tmp_path / "prepared"
    shutil.copytree(FIXTURE, directory)
    (directory / "dew_images-train.array_record-00000-of-00001").unlink()
    available = TFDS("dew_images", path=str(directory), split="test",
        decoders={"image": tfds.decode.SkipDecoding()}).records("indexed")
    assert int(decode_image(available[0]["image"])[0, 0, 0]) == 26
    with pytest.raises(FileNotFoundError, match="Missing prepared ArrayRecord shard"):
        TFDS("dew_images", path=str(directory), split="train").records("indexed")
