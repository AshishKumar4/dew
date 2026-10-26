"""The images preset decodes Arrow records and preserves captions and held-out rows."""

import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from data_support import keep_captions
from PIL import Image

from dew.batch import VALID_ROWS
from dew.data import Files, Loading, presets
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.images import EncodeCaptions

SCALE = 8
READ = Loading(workers=0, threads=1, read_buffer=2, worker_buffer=1)


def run(train, validation=None, *, read="memory", loading=READ, host=(), field="image"):
    """`train`'s images as a run reads them, shuffled, in windows of 64 where streamed, and repeated,
    and `validation`'s read once, each resized to SCALE pixels, its other fields as they are, in
    batches of four read by the `host` steps."""
    def pictures(dataset):
        return dataset.pipe(presets.images, SCALE, batch=4, field=field, rejects=False, host=host)

    order = train.shuffle(0, window=64 if read == "streamed" else None).repeat()
    held = None if validation is None else pictures(validation)
    return compile(pictures(order), validation=held, read=read, loading=loading)


def image_table(path, count=16, *, struct=False):
    payloads = []
    for index in range(count):
        with io.BytesIO() as output:
            Image.fromarray(np.full((12, 12, 3), index + 1, np.uint8)).save(output, format="PNG")
            payloads.append(output.getvalue())
    stored = [{"bytes": payload, "path": None} for payload in payloads] if struct else payloads
    pq.write_table(pa.table({"image": stored, "caption": [f"caption {index}" for index in range(count)],
                            "label": list(range(count))}), path, row_group_size=3)
    return Files(str(path), format="parquet")


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_parquet_images_and_captions_reach_real_worker_batches(tmp_path, workers, read):
    source = image_table(tmp_path / "images.parquet")
    loading = Loading(workers=workers, threads=1, read_buffer=2, worker_buffer=1)
    data = run(source, source, read=read, loading=loading, host=(EncodeCaptions(keep_captions),))
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
    data = run(source)
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
    data = run(train, held)
    assert data.records == 16 and data.held_out == 0 and data.val is not None
    stream = data.val(DataPartition())
    try:
        batches = list(stream)
    finally:
        stream.close()
    assert len(batches) == 2
    assert np.concatenate([batch["label"][batch.get(VALID_ROWS, np.ones(4, bool))]
                           for batch in batches]).tolist() == list(range(5))


def test_an_uncaptioned_class_dataset_reads_as_rgb_whatever_mode_it_stores(tmp_path):
    payloads = []
    for index in range(4):
        grey = np.full((12, 12), 16 * index, np.uint8)
        rgb = np.dstack([grey] * 3)
        stored = (Image.fromarray(grey),
                  Image.fromarray(grey.astype(np.uint16) * 256 + 255),
                  Image.fromarray(rgb).quantize(),
                  Image.fromarray(np.dstack([rgb, grey * 0])))[index]
        with io.BytesIO() as output:
            stored.save(output, format="PNG")
            payloads.append(output.getvalue())
    path = tmp_path / "classes.parquet"
    pq.write_table(pa.table({"img": payloads, "label": list(range(4))}), path)
    source = Files(str(path), format="parquet")
    data = run(source, source, field="img")
    stream = data.val(DataPartition())
    try:
        batch = next(stream)
    finally:
        stream.close()
    assert batch["image"].shape == (4, SCALE, SCALE, 3) and batch["image"].dtype == np.uint8
    np.testing.assert_array_equal(batch["image"][:, 0, 0], np.repeat([[0], [16], [32], [255]], 3, 1))
    np.testing.assert_array_equal(batch["label"], [0, 1, 2, 3])
    assert "caption" not in batch


@pytest.mark.parametrize("streamed", [False, True])
def test_a_native_dataset_names_its_classes_in_a_configured_caption(streamed):
    """A Hugging Face dataset handed over natively (`from_hf`), indexed or streamed, states its
    ClassLabel names as a prepared Hub dataset does, so a configured caption names the class
    rather than its number."""
    import datasets

    from dew.data import from_hf
    from dew.data.images import Caption, ConfigureCaption
    from dew.data.pipeline import Configured

    native = datasets.Dataset.from_dict({"label": [1, 0]}, features=datasets.Features(
        {"label": datasets.ClassLabel(names=["cat", "dog"])}))
    native = native.to_iterable_dataset() if streamed else native
    caption = Configured(ConfigureCaption(Caption("a photo of a {label}")), needs=())
    captioned = from_hf(native, "labels@1" if streamed else None).map(caption)
    assert [row["caption"] for row in captioned] == ["a photo of a dog", "a photo of a cat"]
