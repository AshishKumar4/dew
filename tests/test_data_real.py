"""The prepared Oxford Flowers corpus through its real data source.

Prepare oxford_flowers102 as ArrayRecords in a separate environment, then
set DEW_FLOWERS_PATH to the version directory printed as builder.data_dir.
The tfds extra reads the prepared data; TensorFlow is a preparation
dependency and is not required in this test environment.

This integration test needs the externally prepared full corpus, so it
retains its network marker. The small offline reader fixture lives in
test_tfds_read.py.

    DEW_FLOWERS_PATH=/data/oxford_flowers102/2.1.1 python -m pytest tests/test_data_real.py -q
"""

import itertools
import os

import numpy as np
import pytest

pytest.importorskip("tensorflow_datasets", reason="needs the tfds extra")

from dew.config.data import Augment, HoldOut, ImageData, loader
from dew.data import TFDS, Loading
from dew.data.dataset import DataPartition

pytestmark = pytest.mark.network

BATCH = 8
SIZE = 64
# oxford_flowers102 is 8189 images across its three splits, and the loader
# reads split="all".
RECORDS = 8189
VAL_RECORDS = 4 * BATCH


@pytest.fixture(scope="module")
def flowers():
    """The Dataset a recipe loads from the registered spec, in-process."""
    return loader(ImageData(source=TFDS("oxford_flowers102", path=os.environ["DEW_FLOWERS_PATH"],
                                        split="all"),
                       image_size=SIZE, augment=Augment(flip=True, jitter=True),
                       caption_templates=("a photo of a {}",), validation=HoldOut(records=VAL_RECORDS)),
                batch=BATCH, read="memory", loading=Loading(workers=0))


def labels_of(batch):
    return [int(label) for label in batch["label"]]


def test_train_batches_carry_resized_images_and_labels(flowers):
    assert flowers.records == RECORDS - VAL_RECORDS
    assert flowers.batch == BATCH

    train = flowers.train(DataPartition())
    first, second = next(train), next(train)

    for batch in (first, second):
        assert batch["image"].shape == (BATCH, SIZE, SIZE, 3)
        # uint8 out of the loader: the objective is what rescales to [-1, 1],
        # and sending floats over the worker queues would quadruple its cost.
        assert batch["image"].dtype == np.uint8
        assert batch["image"].max() > 0, "an all-black batch means decode failed"

        assert batch["label"].shape == (BATCH,)

    assert labels_of(first) != labels_of(second), "the train sampler is not shuffling"


def test_the_validation_pass_reads_the_split_once_and_stops(flowers):
    """The pass is the four held-out batches and nothing after them.

    grain's DataLoader batched inside each worker, and the loader defaults to
    eight of them, so a batch came out of one worker's four records read
    twice and the pass ran on for good. Two records of Flowers are never the
    same image, so counting distinct rows counts records.
    """
    rows = [image.tobytes() for batch in itertools.islice(flowers.val(DataPartition()), 8)
            for image in batch["image"]]

    assert len(rows) == VAL_RECORDS
    assert len(set(rows)) == VAL_RECORDS, "the pass read a record twice"
