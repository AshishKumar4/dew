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

from dew.data import TFDS, Loading, images, presets
from dew.data.compile import compile, held
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
    """The corpus through the images preset, its first records held out: the training loader and
    the validation reader, in-process."""
    store = TFDS("oxford_flowers102", path=os.environ["DEW_FLOWERS_PATH"], split="all")
    train, val = store.holdout(records=VAL_RECORDS)
    caption = images.caption(templates=("a photo of a {}",))
    loader = compile(train.shuffle(0).repeat().pipe(presets.images, SIZE, batch=BATCH, augment=True,
                                                    caption=caption, rejects=False),
                     read="memory", loading=Loading(workers=0))
    validation = held(val.pipe(presets.images, SIZE, batch=BATCH, caption=caption, rejects=False)
                      .read("memory").loading(Loading(workers=0)))
    return loader, validation


def labels_of(batch):
    return [int(label) for label in batch["label"]]


def test_train_batches_carry_resized_images_and_labels(flowers):
    loader, _ = flowers
    assert loader.records == RECORDS - VAL_RECORDS
    assert loader.batch == BATCH

    train = loader.train(DataPartition())
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

    Grain's DataLoader once batched inside each worker, so a batch came out of
    one worker's four records read twice and the pass ran on for good. Two
    records of Flowers are never the same image, so counting distinct rows
    counts records.
    """
    _, validation = flowers
    rows = [image.tobytes() for batch in itertools.islice(validation(DataPartition()), 8)
            for image in batch["image"]]

    assert len(rows) == VAL_RECORDS
    assert len(set(rows)) == VAL_RECORDS, "the pass read a record twice"
