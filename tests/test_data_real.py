"""One real dataset, loaded from its spec exactly as a recipe does.

Every other data test stubs the source: they check the registries, the
transforms and the split logic, but nothing has ever taken a real TFDS
dataset off disk, resized it, tokenized its captions and handed back batches.
That is the path a run actually depends on, and its failure modes (a missing
label file, a tokenizer that returns the wrong width, a validation loader that
hands back training records) are invisible to a stubbed source.

Needs the tfds extra and the ~330 MB Oxford Flowers download, so it is skipped
unless tensorflow_datasets imports and is marked network:

    uv venv /tmp/tfdsenv
    uv pip install --python /tmp/tfdsenv/bin/python -e '.[test,tfds]'
    /tmp/tfdsenv/bin/python -m pytest tests/test_data_real.py -q
"""

import itertools
import numpy as np
import pytest

pytest.importorskip("tensorflow_datasets", reason="needs the tfds extra")

from dew.data import Loading, OxfordFlowers

pytestmark = pytest.mark.network

BATCH = 8
SIZE = 64
# CLIP-L/14's context length, which AutoTextTokenizer pads every caption to.
CAPTION_TOKENS = 77
# oxford_flowers102 is 8189 images across its three splits, and the loader
# reads split="all".
RECORDS = 8189
# Validation uses the run's local batch size and holds out four batches.
VAL_BATCH = BATCH
VAL_RECORDS = 4 * BATCH


@pytest.fixture(scope="module")
def flowers():
    """The Dataset a recipe loads from the registered spec, in-process."""
    return OxfordFlowers(image_size=SIZE, loading=Loading(workers=0)).load(batch=BATCH)


def labels_of(batch):
    return [int(label) for label in batch["label"]]


def test_train_batches_carry_resized_images_and_tokenized_captions(flowers):
    assert flowers.records == RECORDS - VAL_RECORDS
    assert flowers.batch == BATCH

    train = flowers.train()
    first, second = next(train), next(train)

    for batch in (first, second):
        assert batch["image"].shape == (BATCH, SIZE, SIZE, 3)
        # uint8 out of the loader: the objective is what rescales to [-1, 1],
        # and sending floats over the worker queues would quadruple its cost.
        assert batch["image"].dtype == np.uint8
        assert batch["image"].max() > 0, "an all-black batch means decode failed"

        assert batch["text"]["input_ids"].shape == (BATCH, CAPTION_TOKENS)
        assert batch["text"]["attention_mask"].shape == (BATCH, CAPTION_TOKENS)
        # Every caption comes from a prompt template, so no row may be all pad
        assert (batch["text"]["attention_mask"].sum(axis=1) > 2).all()
        assert batch["label"].shape == (BATCH,)

    assert labels_of(first) != labels_of(second), "the train sampler is not shuffling"


def test_validation_reads_different_records_in_a_stable_order(flowers):
    """The val loader must not be a random slice of the training stream."""
    train_labels = labels_of(next(flowers.train()))
    val = next(flowers.val())
    again = next(flowers.val())

    assert val["image"].shape == (VAL_BATCH, SIZE, SIZE, 3)
    assert val["image"].dtype == np.uint8
    assert val["text"]["input_ids"].shape == (VAL_BATCH, CAPTION_TOKENS)

    # Canonical order, so two fresh iterators see the same records; the
    # shuffled train stream does not start there.
    assert labels_of(val) == labels_of(again)
    assert labels_of(val)[:BATCH] != train_labels


def test_the_validation_pass_reads_the_split_once_and_stops(flowers):
    """The pass is the four held-out batches and nothing after them.

    grain's DataLoader batched inside each worker, and the loader defaults to
    eight of them, so a batch came out of one worker's four records read
    twice and the pass ran on for good. Two records of Flowers are never the
    same image, so counting distinct rows counts records.
    """
    rows = [image.tobytes() for batch in itertools.islice(flowers.val(), 8)
            for image in batch["image"]]

    assert len(rows) == VAL_RECORDS
    assert len(set(rows)) == VAL_RECORDS, "the pass read a record twice"
