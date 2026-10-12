"""Image stages, Dew's own and foreign libraries', draw from each record's generator."""

import numpy as np
import pytest

from dew.data import Loading, Memory, images
from dew.data.compile import compile
from dew.data.dataset import DataPartition


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
    A = pytest.importorskip("albumentations")

    compose = A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=1.0), A.RandomCrop(20, 20)])
    chain = (pictures().shuffle(0).repeat().random_map(images.albumentations(compose))
             .random_map(images.albumentations(compose)).batch(8))

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
    # The ramp, 60 columns wide, covers 20 at 36 wide; the cut keeps columns 8 to 28 of 36, whose
    # middle is the ramp's middle, 118.
    assert abs(float(bucketed[0]["image"][:, 9:11].mean()) - 118) < 8
    with pytest.raises(ValueError, match="positive"):
        images.bucket(((16, 0),))
