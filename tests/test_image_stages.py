"""Image stages, Dew's own and foreign libraries', draw from each record's generator."""

import numpy as np
import pytest

from dew.config.data import HoldOut, ImageData, loader
from dew.data import Files, Loading, Memory, images
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.sources.codec import pack_dict_of_byte_arrays
from dew.data.steps.image import EncodeCaptions


def pictures(count: int = 48) -> Memory:
    rng = np.random.default_rng(0)
    return Memory([{"image": rng.integers(0, 256, (24, 32, 3), np.uint8), "id": index}
                   for index in range(count)])


def stage(kind: str):
    if kind == "albumentations":
        A = pytest.importorskip("albumentations")

        return images.albumentations(A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=1.0),
                                                A.RandomCrop(16, 16)]))
    if kind == "torchvision":
        v2 = pytest.importorskip("torchvision.transforms.v2")

        return images.torchvision(v2.Compose([v2.RandomHorizontalFlip(), v2.ColorJitter(0.4, 0.4, 0.4),
                                              v2.RandomCrop(16)]))
    return images.random_resized_crop(16, scale=(0.3, 1.0))


@pytest.mark.parametrize("kind", ["albumentations", "torchvision", "dew"])
def test_an_augmentation_draws_the_same_at_any_thread_count_and_after_a_resume(kind):
    """Eight reader threads give every record the draws one thread gives it, and a reader resumed
    from a saved position continues with the draws an uninterrupted one makes."""
    chain = pictures().shuffle(0).repeat().random_map(stage(kind)).batch(8)

    def read(threads, count, state=None):
        data = compile(chain, read="indexed",
                       loading=Loading(workers=0, threads=threads, read_buffer=2 * threads, worker_buffer=1))
        reader = data.train(DataPartition())
        try:
            if state is not None:
                reader.set_state(state)
            batches = [next(reader) for _ in range(count)]
            return batches, reader.get_state()
        finally:
            reader.close()

    single, _ = read(1, 6)
    threaded, _ = read(8, 6)
    for one, many in zip(single, threaded, strict=True):
        np.testing.assert_array_equal(one["id"], many["id"])
        np.testing.assert_array_equal(one["image"], many["image"])
    assert any(not np.array_equal(batch["image"][0], batch["image"][1]) for batch in single)
    _, saved = read(8, 3)
    resumed, _ = read(8, 3, saved)
    for expected, found in zip(single[3:], resumed, strict=True):
        np.testing.assert_array_equal(expected["image"], found["image"])


def keep_captions(captions):
    return {"caption": np.asarray(captions)}


@pytest.mark.parametrize("captions", ["record", "template", "templates", "empty"])
def test_prepared_arrayrecords_and_public_image_steps_deliver_the_same_batches(tmp_path, captions):
    """Prepared pixel bytes and class captions follow one chain for training and validation."""
    from array_record.python.array_record_module import ArrayRecordWriter

    path = tmp_path / "images.array_record"
    writer = ArrayRecordWriter(str(path), "group_size:1")
    rng = np.random.default_rng(0)
    for index in range(20):
        pixels = rng.integers(0, 256, (20, 24, 3), np.uint8)
        writer.write(pack_dict_of_byte_arrays({
            "shape": np.asarray(pixels.shape[:2], np.int32).tobytes(),
            "image": pixels.tobytes(), "label": np.asarray(index % 2, np.int32).tobytes(),
            "caption": f"picture {index}".encode(), "url": f"catalog/{index}".encode(),
        }))
    writer.close()
    settings = {"record": {}, "template": {"caption": "a photo of {label}"},
                "templates": {"caption_templates": ("a {}", "a photo of {label}")},
                "empty": {"caption_columns": ()}}[captions]
    classes = ("rose", "tulip")
    source = Files(str(path))
    recipe = ImageData(source=source, image_size=16, staging_size=32, augment=None, classes=classes,
                       validation=HoldOut(records=4), **settings)
    loading = Loading(workers=0, threads=1, read_buffer=2)
    host = (EncodeCaptions(keep_captions),)
    expected = loader(recipe, batch=4, read="indexed", seed=37, loading=loading, host=host)
    train, held = source.holdout(records=4)

    def prepared(dataset):
        chain = dataset.map(images.decode("image", rejects=False)).map(images.resize(16))
        if captions == "templates":
            return chain.random_map(images.caption(templates=settings["caption_templates"], classes=classes))
        if captions == "empty":
            return chain.map(images.caption(classes=classes, columns=()))
        return chain.map(images.caption(settings.get("caption"), classes=classes))

    actual = compile(prepared(train.shuffle(37).repeat()).batch(4),
                     validation=prepared(held).batch(4, drop_remainder=False),
                     read="indexed", seed=37, loading=loading, host=host)
    left, right = expected.train(DataPartition()), actual.train(DataPartition())
    try:
        for _ in range(4):
            one, other = next(left), next(right)
            assert one["image"].shape == (4, 16, 16, 3) and one["image"].dtype == np.uint8
            assert set(one) == set(other) == {"image", "label", "caption", "url"}
            for name in one:
                np.testing.assert_array_equal(one[name], other[name])
        for one, other in zip(expected.val(DataPartition()), actual.val(DataPartition()), strict=True):
            for name in one:
                np.testing.assert_array_equal(one[name], other[name])
    finally:
        left.close()
        right.close()
