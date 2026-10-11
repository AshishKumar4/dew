"""Image recipes decode Arrow records and preserve captions and held-out rows."""

import io
import itertools

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from dew.data import Augment, DataPartition, Files, HoldOut, Images, Loading, load
from dew.data.steps.image import EncodeCaptions
from dew.objectives.base import VALID_ROWS

SCALE = 8
READ = Loading(workers=0, threads=1, read_buffer=2, worker_buffer=1)


def keep_captions(captions):
    return {"caption": np.asarray(captions)}


def image_table(path, count=16, *, struct=False):
    payloads = []
    for index in range(count):
        with io.BytesIO() as output:
            Image.fromarray(np.full((12, 12, 3), index + 1, np.uint8)).save(output, format="PNG")
            payloads.append(output.getvalue())
    images = [{"bytes": payload, "path": None} for payload in payloads] if struct else payloads
    pq.write_table(pa.table({"image": images, "caption": [f"caption {index}" for index in range(count)],
                            "label": list(range(count))}), path, row_group_size=3)
    return Files(str(path), format="parquet")


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_parquet_images_and_captions_reach_real_worker_batches(tmp_path, workers, read):
    source = image_table(tmp_path / "images.parquet")
    data = load(Images(source=source, image_size=SCALE, augment=None, validation=source),
                batch=4, read=read,
                loading=Loading(workers=workers, threads=1, read_buffer=2, worker_buffer=1),
                host=(EncodeCaptions(keep_captions),))
    assert data.val is not None
    stream = data.val(DataPartition())
    try:
        batches = list(stream)
    finally:
        stream.close()
    assert len(batches) == 4
    labels = np.concatenate([batch["label"] for batch in batches])
    assert labels.tolist() == list(range(16))
    for batch in batches:
        assert batch["image"].dtype == np.uint8
        np.testing.assert_array_equal(batch["image"][:, 0, 0, 0], batch["label"] + 1)
        assert batch["caption"].tolist() == [f"caption {index}" for index in batch["label"]]


def test_hugging_face_image_structs_decode_their_embedded_bytes(tmp_path):
    source = image_table(tmp_path / "hub-images.parquet", 4, struct=True)
    data = load(Images(source=source, image_size=SCALE, augment=None), batch=4,
                read="memory", loading=READ)
    stream = data.train(DataPartition())
    try:
        batch = next(stream)
    finally:
        stream.close()
    np.testing.assert_array_equal(batch["image"][:, 0, 0, 0], batch["label"] + 1)
    assert sorted(batch["label"].tolist()) == list(range(4))


def test_a_second_validation_source_leaves_every_training_record_available(tmp_path):
    train = image_table(tmp_path / "train.parquet", 16)
    held = image_table(tmp_path / "held.parquet", 5)
    data = load(Images(source=train, validation=held, image_size=SCALE, augment=None),
                batch=4, read="memory", loading=READ)
    assert data.records == 16 and data.held_out == 0 and data.val is not None
    stream = data.val(DataPartition())
    try:
        batches = list(stream)
    finally:
        stream.close()
    assert len(batches) == 2
    assert np.concatenate([batch["label"][batch.get(VALID_ROWS, np.ones(4, bool))]
                           for batch in batches]).tolist() == list(range(5))


def test_a_head_holdout_is_disjoint_from_the_training_epoch(tmp_path):
    source = image_table(tmp_path / "images.parquet", 16)
    data = load(Images(source=source, validation=HoldOut(records=8), image_size=SCALE, augment=None),
                batch=4, read="memory", loading=READ)
    assert data.records == 8 and data.val is not None
    training, validation = data.train(DataPartition()), data.val(DataPartition())
    try:
        train_labels = np.concatenate([batch["label"] for batch in itertools.islice(training, 2)])
        held_labels = np.concatenate([batch["label"] for batch in validation])
    finally:
        training.close()
        validation.close()
    assert sorted(train_labels.tolist()) == list(range(8, 16))
    assert held_labels.tolist() == list(range(8))


def test_validation_never_applies_the_training_crop_or_colour_jitter(tmp_path):
    source = image_table(tmp_path / "images.parquet", 8)
    images = []
    for augment in (None, Augment(flip=True, jitter=True, crop_scale=(0.5, 0.9))):
        data = load(Images(source=source, validation=source, image_size=SCALE,
                           staging_size=16, augment=augment), batch=4, read="memory", loading=READ)
        assert data.val is not None
        stream = data.val(DataPartition())
        try:
            images.append(next(stream)["image"])
        finally:
            stream.close()
    np.testing.assert_array_equal(images[0], images[1])


def test_an_unconditional_batch_discards_host_caption_strings(tmp_path):
    source = image_table(tmp_path / "images.parquet", 4)
    data = load(Images(source=source, image_size=SCALE, augment=None), batch=4, read="memory", loading=READ)
    stream = data.train(DataPartition())
    try:
        batch = next(stream)
    finally:
        stream.close()
    assert set(batch) == {"image", "label"}
