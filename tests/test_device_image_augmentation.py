"""Batch augmentation agrees with OpenCV and follows the saved data position."""

import dataclasses
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from steady_state import guarded, steady_state

from dew.data import DataPartition, Loading
from dew.data.images import ImageDataset


@dataclasses.dataclass(frozen=True)
class Images(ImageDataset):
    def source(self, split=None):
        return [{"image": np.random.default_rng(index).integers(
            0, 256, (19, 23, 3), dtype=np.uint8), "label": index} for index in range(24)]

    def record(self, element, rng):
        return element["image"], "a flower", element["label"]


def spec(**kwargs):
    return Images(image_size=12, val_batches=0, seed=37,
                  loading=Loading(workers=0, threads=1, read_buffer=2), **kwargs)


def test_device_pipeline_returns_resident_uint8_and_resumes_exactly():
    data = spec(augmentation_backend="device", crop_scale=(0.5, 0.9)).load(batch=4)
    stream = data.train(DataPartition())
    resumed = data.train(DataPartition())
    try:
        first = next(stream)
        assert isinstance(first["image"], jax.Array)
        assert first["image"].dtype == jnp.uint8
        assert first["image"].shape == (4, 12, 12, 3)
        assert set(first) == {"image", "label"}
        saved = stream.get_state()
        expected = next(stream)
        resumed.set_state(saved)
        actual = next(resumed)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])
        assert not np.array_equal(first["image"], expected["image"])
    finally:
        stream.close()
        resumed.close()


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("crop", [(0, 0, 17, 21), (3, 4, 9, 11), (16, 20, 1, 1)])
def test_supplied_parameters_match_opencv_at_highest_precision(order, crop):
    from dew.data.image_augmentation import ImageParameters, apply_device, apply_host

    image = np.random.default_rng(7).integers(0, 256, (17, 21, 3)).astype(np.float64)
    parameters = ImageParameters(np.asarray(crop), np.asarray(True),
                                 np.asarray([1.13, 0.97, 0.86]), np.asarray(order))
    # Observed on RTX 4080/OpenCV 5.0.0: maximum fp64 error 6.55e-6,
    # with identical rounded uint8 codes in all 18 cases.
    # OpenCV resize.cpp uses HResizeLinear<double, double, float>: even for
    # fp64 pixels each axis's coefficient is rounded to fp32 (error <= eps/2).
    # Each corner's product therefore errs by <= eps + O(eps**2); four
    # corners bounded by 255 give <= 4*255*eps + O(eps**2). In infinity norm,
    # brightness/contrast/saturation gains are <= 1.2, 1.1 and 1.4 over the
    # configured ranges: their product 1.848 < 2. The factor 8 covers this
    # amplification, with margin for fp64 arithmetic and the eps**2 terms.
    # After rounding this is at most one uint8 code value.
    device = jax.devices()[0]
    if device.platform == "tpu":
        device = jax.devices("cpu")[0]  # TPU has no float64 operations.
    with jax.enable_x64(), jax.default_device(device):
        actual = jax.jit(lambda x, p: apply_device(x, p, 13))(image, parameters)
        expected = apply_host(image, parameters, 13)
    np.testing.assert_allclose(actual, expected, atol=255 * 8 * np.finfo(np.float32).eps, rtol=0)
    rounded = np.clip(np.rint(actual), 0, 255).astype(np.int16)
    host_rounded = np.clip(np.rint(expected), 0, 255).astype(np.int16)
    assert np.max(np.abs(rounded - host_rounded)) <= 1


def test_each_data_key_keeps_its_draw_across_batch_sizes_and_order():
    from dew.data.image_augmentation import augment_batch

    image = np.random.default_rng(4).integers(0, 256, (4, 16, 16, 3), dtype=np.uint8)
    keys = np.random.default_rng(2).integers(0, 2**32, (4, 2), dtype=np.uint32)
    run = jax.jit(lambda images, raw: augment_batch(images, raw, size=12,
                  flip=True, jitter=True, crop_scale=(0.4, 0.9)))
    together = run(image, keys)
    separately = jnp.concatenate([run(image[i:i+1], keys[i:i+1]) for i in range(4)])
    np.testing.assert_array_equal(together, separately)
    np.testing.assert_array_equal(run(image[::-1], keys[::-1]), together[::-1])
    assert not np.array_equal(run(image, keys ^ np.uint32(1)), together)


def test_production_fp32_crop_and_colour_are_within_one_uint8_code():
    from dew.data.image_augmentation import ImageParameters, apply_device, apply_host

    image = np.random.default_rng(13).integers(0, 256, (160, 160, 3), dtype=np.uint8)
    parameters = ImageParameters(np.asarray([5, 11, 143, 139]), np.asarray(True),
                                 np.asarray([1.19, 1.05, 0.8]), np.asarray([2, 1, 0]))
    expected = apply_host(image.astype(np.float64), parameters, 128)
    actual = jax.jit(lambda x, p: apply_device(x, p, 128))(image, parameters)
    rounded = np.clip(np.rint(actual), 0, 255).astype(np.int16)
    reference = np.clip(np.rint(expected), 0, 255).astype(np.int16)
    assert np.max(np.abs(rounded - reference)) <= 1


def test_model_float64_mode_changes_no_data_draw():
    from dew.data.image_augmentation import augment_batch

    image = np.random.default_rng(4).integers(0, 256, (4, 16, 16, 3), dtype=np.uint8)
    keys = np.random.default_rng(2).integers(0, 2**32, (4, 2), dtype=np.uint32)
    run = jax.jit(lambda images, raw: augment_batch(images, raw, size=12,
                  flip=True, jitter=True, crop_scale=(0.4, 0.9)))
    with jax.enable_x64(False):
        expected = run(image, keys)
    with jax.enable_x64():
        actual = run(image, keys)
    np.testing.assert_array_equal(actual, expected)


def test_device_none_keeps_the_old_resize_bit_identical():
    host = spec(augmentation="none").load(batch=4).train(DataPartition())
    device = spec(augmentation="none", augmentation_backend="device").load(batch=4).train(DataPartition())
    try:
        np.testing.assert_array_equal(next(device)["image"], next(host)["image"])
    finally:
        host.close()
        device.close()


def test_device_pixels_are_not_fetched_back_when_the_trainer_places_them():
    from dew.training.distributed import build_mesh, shard_batch

    stream = spec(augmentation_backend="device").load(batch=4).train(DataPartition())
    try:
        batch = next(stream)
        mesh = build_mesh(devices=[jax.devices()[0]])
        with guarded(host_to_device="allow"):
            placed = shard_batch(mesh, batch)
            jax.block_until_ready(placed)
        np.testing.assert_array_equal(placed["image"], batch["image"])
    finally:
        stream.close()


def test_device_draws_follow_record_keys_across_threads_and_partitions():
    original = spec(augmentation_backend="device", crop_scale=(0.6, 0.9))
    threaded = dataclasses.replace(original, loading=Loading(workers=0, threads=4, read_buffer=8))
    whole = original.load(batch=4).train(DataPartition())
    parallel = threaded.load(batch=4).train(DataPartition())
    shares = [original.load(batch=4).train(DataPartition(index=i, count=2)) for i in range(2)]
    try:
        for _ in range(3):
            expected = next(whole)
            actual = next(parallel)
            split = [next(stream) for stream in shares]
            for name in expected:
                np.testing.assert_array_equal(actual[name], expected[name])
                joined = np.stack([np.asarray(part[name]) for part in split], axis=1).reshape(
                    np.shape(expected[name]))
                np.testing.assert_array_equal(joined, expected[name])
    finally:
        for stream in [whole, parallel, *shares]:
            stream.close()


def test_grain_workers_only_decode_and_keep_the_parent_device_draws():
    original = spec(augmentation_backend="device", crop_scale=(0.6, 0.9))
    worker_spec = dataclasses.replace(original, loading=Loading(workers=1, threads=1, read_buffer=2))
    parent = original.load(batch=4).train(DataPartition())
    worker = worker_spec.load(batch=4).train(DataPartition())
    try:
        for _ in range(3):
            expected, actual = next(parent), next(worker)
            assert isinstance(actual["image"], jax.Array)
            for name in expected:
                np.testing.assert_array_equal(actual[name], expected[name])
    finally:
        parent.close()
        worker.close()


@pytest.mark.parametrize("options", [{"crop_scale": (0, 1)}, {"crop_scale": (0.9, 0.5)},
                                    {"augmentation_size": 0}, {"augmentation_backend": "bad"}])
def test_invalid_augmentation_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        spec(**options)


def test_a_device_augmented_stream_steadies_into_compiled_reads_the_loop_waits_on_alone():
    """Once a few batches have been augmented and placed, the stream the
    trainer reads compiles nothing more for records of other sizes and
    crops, and handing the loop a placed batch moves nothing it did not ask
    for (`steady_state`): the augmentation and the placement stay on the
    devices and on the prefetch worker."""
    from dew.training.distributed import DevicePrefetchIterator, build_mesh

    mesh = build_mesh(devices=[jax.devices()[0]])
    stream = spec(augmentation_backend="device", crop_scale=(0.5, 0.9)).load(batch=4).train(DataPartition())
    with DevicePrefetchIterator(stream, mesh) as prefetch:
        for _ in range(3):
            jax.block_until_ready(next(prefetch))
        with steady_state():
            batches = [next(prefetch) for _ in range(3)]
            jax.block_until_ready(batches)
    assert all(batch["image"].shape == (4, 12, 12, 3) for batch in batches)
