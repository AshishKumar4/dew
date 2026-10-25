"""Image stages, Dew's own and foreign libraries', draw from each record's generator."""

import numpy as np
import pytest
from data_support import keep_captions

from dew.data import Files, Loading, Memory, images, presets
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.images import EncodeCaptions
from dew.data.sources.codec import pack_dict_of_byte_arrays


def pictures(count: int = 48) -> Memory:
    rng = np.random.default_rng(0)
    return Memory([{"image": rng.integers(0, 256, (24, 32, 3), np.uint8), "id": index}
                   for index in range(count)])


def stage(kind: str):
    if kind == "albumentations":
        import albumentations as A

        return images.Albumentations(A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=1.0),
                                                A.RandomCrop(16, 16)]))
    if kind == "torchvision":
        from torchvision.transforms import v2

        return images.Torchvision(v2.Compose([v2.RandomHorizontalFlip(), v2.ColorJitter(0.4, 0.4, 0.4),
                                              v2.RandomCrop(16)]))
    return images.RandomResizedCrop(16, scale=(0.3, 1.0))


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


def _read(chain, threads, count):
    data = compile(chain, read="indexed",
                   loading=Loading(workers=0, threads=threads, read_buffer=2 * threads, worker_buffer=1))
    reader = data.train(DataPartition())
    try:
        return [next(reader) for _ in range(count)]
    finally:
        reader.close()


def test_stages_holding_one_albumentations_pipeline_each_draw_their_own_record_s_seed():
    """Two stages around one Compose, read by eight threads: no thread's seed lands between another's
    seed and call, so every record's draws are the ones one thread makes."""
    import albumentations as A

    compose = A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=1.0), A.RandomCrop(20, 20)])
    chain = (pictures().shuffle(0).repeat().random_map(images.Albumentations(compose))
             .random_map(images.Albumentations(compose)).batch(8))

    for one, many in zip(_read(chain, 1, 8), _read(chain, 8, 8), strict=True):
        np.testing.assert_array_equal(one["image"], many["image"])


def test_a_bucket_is_the_size_nearest_an_image_s_aspect_covered_and_cut_at_the_centre():
    """A wide, a tall and a square image each take the size nearest their aspect, cover it and keep
    its centre: a horizontal ramp's middle column stays in the middle."""
    sizes = ((16, 16), (12, 20), (20, 12))
    ramp = np.broadcast_to(np.arange(60, dtype=np.uint8)[None, :, None] * 4, (20, 60, 3))
    rows = [{"image": np.ascontiguousarray(ramp)}, {"image": np.zeros((40, 24, 3), np.uint8)},
            {"image": np.zeros((30, 31, 3), np.uint8)}]

    bucketed = [images.bucket(sizes).map(row) for row in rows]

    assert [int(row["bucket"]) for row in bucketed] == [1, 2, 0]
    assert [row["image"].shape for row in bucketed] == [(12, 20, 3), (20, 12, 3), (16, 16, 3)]
    # The ramp, 60 columns wide, covers 20 at 36 wide; the cut keeps columns 8 to 28 of 36, its centre.
    covered = images.Resize((12, 36), "area").map({"image": np.ascontiguousarray(ramp)})["image"]
    np.testing.assert_array_equal(bucketed[0]["image"], covered[:, 8:28])
    with pytest.raises(ValueError):
        images.bucket(((16, 0),))


@pytest.mark.parametrize("captions", ["record", "template", "templates", "empty"])
def test_the_images_preset_over_prepared_arrayrecords_delivers_what_its_public_steps_do(tmp_path, captions):
    """Prepared pixel bytes and class captions follow one chain for training and validation: the
    images preset's, and the public steps it is written as."""
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
    classes = ("rose", "tulip")
    caption = {"record": images.caption(classes=classes),
               "template": images.caption("a photo of {label}", classes=classes),
               "templates": images.caption(templates=("a {}", "a photo of {label}"), classes=classes),
               "empty": images.caption(classes=classes, columns=())}[captions]
    train, held = Files(str(path)).holdout(records=4)
    loading = Loading(workers=0, threads=1, read_buffer=2)
    host = (EncodeCaptions(keep_captions),)

    def preset(dataset):
        # Staged at 32 pixels for an augmentation, of which there is none: resized to 16 directly.
        return dataset.pipe(presets.images, 16, batch=4, staging_size=32, rejects=False, caption=caption)

    expected = compile(preset(train.shuffle(37).repeat()), validation=preset(held), read="indexed", seed=37,
                       loading=loading, host=host)

    def prepared(dataset):
        chain = dataset.map(images.DecodeImage("image", rejects=False)).map(images.Resize(16))
        return chain.random_map(caption) if captions == "templates" else chain.map(caption)

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


def test_host_image_augmentation_matches_across_workers_and_resumes_exactly():
    rows = [{**row, "label": np.int64(row["id"] % 3)} for row in pictures().values]
    chain = Memory(rows).shuffle(7).repeat().pipe(
        presets.images, 16, batch=8, staging_size=24, rejects=False,
        augment=images.RandomCrop(16, scale=(0.6, 0.9), flip=True, jitter=True),
        caption=images.caption(templates=("a {label}", "a photo of {}"), classes=("rose", "tulip", "lily")),
        host=(EncodeCaptions(keep_captions),))

    def read(workers, count, state=None):
        data = compile(chain, read="indexed", seed=7,
                       loading=Loading(workers=workers, threads=1, read_buffer=2, worker_buffer=1))
        reader = data.train(DataPartition())
        try:
            if state is not None:
                reader.set_state(state)
            batches = [next(reader) for _ in range(count)]
            return batches, reader.get_state()
        finally:
            reader.close()

    serial, _ = read(0, 6)
    workers, _ = read(2, 6)
    for one, many in zip(serial, workers, strict=True):
        assert one["image"].shape == (8, 16, 16, 3) and one["image"].dtype == np.uint8
        assert one["label"].dtype == np.int32
        for name in one:
            np.testing.assert_array_equal(one[name], many[name])
    _, saved = read(2, 3)
    resumed, _ = read(2, 3, saved)
    for expected, found in zip(serial[3:], resumed, strict=True):
        for name in expected:
            np.testing.assert_array_equal(expected[name], found[name])


@pytest.mark.parametrize("size", [3, 1])
def test_a_side_and_its_square_resize_alike(size):
    """Resize(n) and Resize((n, n)) ask for one geometry, and by default pick one interpolation
    for it, cubic up and area down, so they make the same pixels."""
    row = {"image": np.array([[[64, 64, 64], [192, 192, 192]]], np.uint8)}
    np.testing.assert_array_equal(images.Resize(size).map(row)["image"],
                                  images.Resize((size, size)).map(row)["image"])


def test_a_center_crop_cuts_a_large_image_s_central_square_and_grows_a_small_one_first():
    """CenterCrop cuts the central square of an image at least its size as it is, and resizes a
    smaller one bicubically until its shorter side is that size before cutting."""
    large = np.arange(6 * 9 * 3, dtype=np.uint8).reshape(6, 9, 3)
    small = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3) * 10
    cropped = list(Memory([{"image": large}, {"image": small}]).map(images.CenterCrop(4)))
    np.testing.assert_array_equal(cropped[0]["image"], large[1:5, 2:6])
    grown = images.Resize((4, 6), "bicubic").map({"image": small})["image"]
    np.testing.assert_array_equal(cropped[1]["image"], grown[:, 1:5])


def test_an_image_processor_writes_its_pixel_values_beside_the_image_it_reads():
    """Processor runs a Hugging Face image processor on the image in its field and writes the
    processor's own pixel values under its output, keeping the image."""
    from pathlib import Path

    from transformers import CLIPImageProcessorPil

    processor = CLIPImageProcessorPil.from_pretrained(str(Path(__file__).parent / "fixtures/clip/tiny"))
    image = np.random.default_rng(0).integers(0, 256, (12, 10, 3), np.uint8)
    (row,) = Memory([{"photo": image}]).map(images.Processor(processor, field="photo", to="pixels"))
    np.testing.assert_array_equal(row["photo"], image)
    expected = processor(images=image, return_tensors="np")["pixel_values"][0]
    np.testing.assert_array_equal(row["pixels"], expected)


def test_a_saved_position_is_refused_by_a_run_that_tokenizes_its_captions_otherwise():
    """A diffusion run's captions are tokenized by its condition's encoder, so a position saved
    under eight caption tokens is refused by a run that tokenizes to sixteen: the batches differ,
    though every step's code is the same."""
    from dew.diffusion.presets import Flow
    from dew.inputs import CharTable, Condition, Field, InputSpec
    from dew.nn.backbones import SimpleDiT
    from dew.objectives.diffusion import DiffusionObjective
    from dew.sampling import Euler

    def data(tokens: int):
        condition = Condition(CharTable.from_pretrained(tokens=tokens), field="text")
        objective = DiffusionObjective(
            SimpleDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1), Flow(),
            InputSpec(Field("image", (16, 16, 3)), conditions={"textcontext": condition}),
            solver=Euler(), steps=2)
        chain = pictures(8).shuffle(0).repeat().pipe(
            presets.images, 16, batch=4, augment=True, rejects=False,
            caption=images.caption("a photograph of a flower in a field"))
        return compile(chain, read="memory", loading=Loading(workers=0, threads=1, read_buffer=1),
                       objective=objective.data())

    saving = data(8).train(DataPartition())
    try:
        assert np.asarray(next(saving)["text"]["input_ids"]).shape == (4, 8)
        state = saving.get_state()
    finally:
        saving.close()
    for tokens, refused in ((8, False), (16, True)):
        resumed = data(tokens).train(DataPartition())
        try:
            if refused:
                with pytest.raises(ValueError):
                    resumed.set_state(state)
            else:
                resumed.set_state(state)
        finally:
            resumed.close()
