"""Prepared TFDS ArrayRecords decoded and batched without dataset preparation.

The twenty constant RGB images come from tools/datasets/tfds_reference.py.
These tests run with the tfds extra, including in the Python 3.14 environment
where TensorFlow is not installed. They require no downloads or preparation.
"""

import itertools
from pathlib import Path

import numpy as np
import pytest

from dew.config.data import HoldOut, ImageData, loader
from dew.data import TFDS, Loading
from dew.data.dataset import DataPartition
from dew.data.image_codec import decode_image
from dew.data.steps.image import ImageRecord

tfds = pytest.importorskip("tensorflow_datasets", reason="needs the tfds extra")

FIXTURE = Path(__file__).parent / "fixtures" / "tfds" / "dew_images" / "1.0.0"


def test_prepared_records_carry_encoded_pixels_labels_and_caption_names():
    """The source hands out the bytes on disk; `record` decodes them, so a
    worker pays one decode at the resolution it needs and no more."""
    spec = ImageData(source=TFDS("dew_images", path=str(FIXTURE), split="all",
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


@pytest.mark.parametrize("workers", [0, 2])
def test_prepared_records_reach_grain_batches_without_split_overlap(workers):
    data = loader(ImageData(source=TFDS("dew_images", path=str(FIXTURE), split="all",
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
