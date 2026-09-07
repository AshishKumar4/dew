"""Prepared TFDS ArrayRecords decoded and batched without dataset preparation.

The twenty constant RGB images come from tools/datasets/tfds_reference.py.
These tests run with the tfds extra, including in the Python 3.14 environment
where TensorFlow is not installed. They require no downloads or preparation.
"""

import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

pytest.importorskip("tensorflow_datasets", reason="needs the tfds extra")

from dew.data import Loading, OxfordFlowers

FIXTURE = Path(__file__).parent / "fixtures" / "tfds" / "dew_images" / "1.0.0"


def test_prepared_records_decode_pixels_labels_and_caption_names():
    spec = OxfordFlowers(path=str(FIXTURE))
    source = spec.source()
    assert len(source) == 20
    for index in range(len(source)):
        record = source[index]
        np.testing.assert_array_equal(record["image"], np.full((8, 8, 3), index + 10, np.uint8))
        assert record["label"] == index % 2
        _, caption, label = spec.record(record, np.random.default_rng(0))
        assert ("red" if label == 0 else "blue") in caption.split()
    test_split = OxfordFlowers(path=str(FIXTURE), split="test").source()
    assert [int(test_split[i]["image"][0, 0, 0]) for i in range(len(test_split))] == [26, 27, 28, 29]


@pytest.mark.parametrize("workers", [0, 2])
def test_prepared_records_reach_grain_batches_without_split_overlap(workers):
    data = OxfordFlowers(
        path=str(FIXTURE), image_size=8, augmentation="none", val_batches=1,
        loading=Loading(workers=workers, threads=1, read_buffer=8, worker_buffer=2),
    ).load(batch=4)
    source = data.train()
    train = list(itertools.islice(source, 4))
    validation = list(data.val())
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
    spec = OxfordFlowers(path=str(FIXTURE), labels=str(labels))
    source = spec.source()
    _, caption, label = spec.record(source[1], np.random.default_rng(0))
    assert label == 1 and "rose" in caption.split()


def test_an_unprepared_directory_requests_external_preparation(tmp_path):
    with pytest.raises(ValueError, match="prepared TFDS ArrayRecords"):
        OxfordFlowers().source()
    with pytest.raises(FileNotFoundError, match="separate environment"):
        OxfordFlowers(path=str(tmp_path)).source()
    assert list(tmp_path.iterdir()) == []


def test_a_tfrecord_preparation_is_refused_before_reading(tmp_path):
    directory = tmp_path / "prepared"
    shutil.copytree(FIXTURE, directory)
    metadata_path = directory / "dataset_info.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["fileFormat"] = "tfrecord"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="dew reads ArrayRecords"):
        OxfordFlowers(path=str(directory)).source()


def test_missing_records_request_preparation_only_for_the_selected_split(tmp_path):
    directory = tmp_path / "prepared"
    shutil.copytree(FIXTURE, directory)
    (directory / "dew_images-train.array_record-00000-of-00001").unlink()
    available = OxfordFlowers(path=str(directory), split="test").source()
    assert int(available[0]["image"][0, 0, 0]) == 26
    with pytest.raises(FileNotFoundError, match="Missing prepared ArrayRecord shard"):
        OxfordFlowers(path=str(directory), split="train").source()
