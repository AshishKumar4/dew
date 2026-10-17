"""Prepared TFDS ArrayRecords decoded and batched without dataset preparation.

The twenty constant RGB images come from tools/datasets/tfds_reference.py.
These tests run with the tfds extra, including in the Python 3.14 environment
where TensorFlow is not installed. They require no downloads or preparation.
"""

import itertools
import shutil
from collections import Counter
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from fsspec.implementations.local import LocalFileSystem

from dew.config.data import HoldOut, ImageData, loader
from dew.data import TFDS, Loading, Memory, images
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.image_codec import decode_image
from dew.data.storage import Store

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
        result = next(Memory([record]).map(images.DecodeImage("image", rejects=False))
                      .random_map(images.caption(templates=spec.caption_templates,
                                                 classes=("red", "blue"))).batch(1).iterate())
        caption, label = result["caption"][0], result["label"][0]
        assert ("red" if label == 0 else "blue") in caption.split()
    test_split = TFDS("dew_images", path=str(FIXTURE), split="test",
        decoders={"image": tfds.decode.SkipDecoding()}).records("indexed")
    assert [int(decode_image(test_split[i]["image"])[0, 0, 0]) for i in range(len(test_split))] == [
        26,
        27,
        28,
        29,
    ]


@pytest.mark.parametrize("location", ["local", "prepared", "mirror"])
def test_streamed_tfds_pins_one_fingerprint_for_all_virtual_shards(tmp_path, monkeypatch, location):
    """Virtual shards reuse the prepared split's pinned physical snapshot."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    directory = (tmp_path / "cache" / "dew" / "tfds" / "dew_images" / "1.0.0"
                 if location == "prepared" else tmp_path / "prepared")
    shutil.copytree(FIXTURE, directory)
    source = TFDS("dew_images", path=None if location == "prepared" else str(directory),
                  split="all", shard_rows=2,
                  decoders={"image": tfds.decode.SkipDecoding()})
    if location == "mirror":
        remote = Store(f"memory://tfds-snapshot-{uuid4().hex}/1.0.0")
        for path in directory.iterdir():
            remote.child(path.name).write(path.read_bytes())
        source = TFDS("dew_images", path=remote.url, split="all", shard_rows=2,
                      decoders={"image": tfds.decode.SkipDecoding()}).provision()
        directory = Path(source.directory)
    assert source.metadata()["rows"] == 20
    opened = Counter()
    open_file = LocalFileSystem.open

    def count(self, path, *args, **kwargs):
        filename = Path(path)
        if filename.parent == directory:
            opened[filename.name] += 1
        return open_file(self, path, *args, **kwargs)

    monkeypatch.setattr(LocalFileSystem, "open", count)
    reader = compile(source.repeat().batch(2), read="streamed",
                     loading=Loading(workers=0, threads=1, read_buffer=1)).train(DataPartition())
    try:
        identity = reader.identity()
        pixels = [int(decode_image(image)[0, 0, 0]) for batch in itertools.islice(reader, 10)
                  for image in batch["image"]]
        assert sorted(pixels) == list(range(10, 30))
        assert reader.identity() == identity
        if location != "local":
            source.provision()
        assert int(decode_image(source.records("indexed")[0]["image"])[0, 0, 0]) == 10
    finally:
        reader.close()
    assert opened["dataset_info.json"] == 2
    assert opened["features.json"] == opened["label.labels.txt"] == 1
    assert opened["dew_images-train.array_record-00000-of-00001"] == 1
    assert opened["dew_images-test.array_record-00000-of-00001"] == 1


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
