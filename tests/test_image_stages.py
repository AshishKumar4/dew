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
        import albumentations as A

        return images.albumentations(A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=1.0),
                                                A.RandomCrop(16, 16)]))
    if kind == "torchvision":
        from torchvision.transforms import v2

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
