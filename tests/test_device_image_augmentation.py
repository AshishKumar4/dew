"""Batch augmentation agrees with OpenCV's resize and torchvision's
ColorJitter, and follows the saved data position."""

import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from steady_state import guarded, steady_state

from dew.data import Loading, Memory, images, presets
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.images import EncodeCaptions

TORCHVISION = np.load(Path(__file__).resolve().parent / "fixtures" / "colour_jitter" / "torchvision.npz")


READ = Loading(workers=0, threads=1, read_buffer=2)


ROWS = [{"image": np.random.default_rng(index).integers(0, 256, (19, 23, 3), dtype=np.uint8), "label": index}
        for index in range(24)]


def loaded(*, batch=4, seed=37, loading=READ, objective=None, augment=True, crop_scale=(1.0, 1.0),
           staging_size=None, jitter=True):
    """`ROWS` in a run's order, shuffled by `seed`, resized to 12 pixels (staged at `staging_size`
    where given) and captioned, the caption dropped before a batch reaches the run; with `augment`,
    cropped to `crop_scale` of their area, flipped and, with `jitter`, colour-jittered in the compiled
    step."""
    device = images.DeviceAugment(12, crop_scale=crop_scale, flip=True, jitter=jitter) if augment else False
    chain = Memory(ROWS).shuffle(seed).repeat().pipe(
        presets.images, 12, batch=batch, augment=device, staging_size=staging_size, rejects=False,
        caption=images.caption("a flower"), host=(EncodeCaptions(),))
    return compile(chain, read="memory", seed=seed, loading=loading, objective=objective)


def augmented(data, batch):
    """`batch` as the compiled training step reads it: after the loader's device steps."""
    for step in data.device:
        batch = step(batch)
    return batch


def test_the_image_pipeline_benchmark_times_the_augmentation_the_loader_leaves_to_the_device():
    """The benchmark's device arms read what a training step reads: images staged at 24 pixels,
    cropped, flipped and jittered by the loader's device steps to the 12 pixels the step takes,
    the same pixels the steps make applied one by one."""
    from tool_support import load

    benchmark = load("benchmark_image_pipeline")
    data = loaded(crop_scale=(0.5, 0.9), staging_size=24)
    stream = data.train(DataPartition())
    try:
        batch = next(stream)
    finally:
        stream.close()
    timed = benchmark.device_steps(data.device)(batch)
    assert np.asarray(batch["image"]).shape == (4, 24, 24, 3)
    assert timed["image"].shape == (4, 12, 12, 3)
    expected = augmented(data, {name: value for name, value in batch.items() if name != "caption"})
    np.testing.assert_array_equal(np.asarray(timed["image"]), np.asarray(expected["image"]))


def test_device_pipeline_reads_host_batches_the_step_augments_and_resumes_exactly():
    """The reader delivers host arrays with each record's key; the device steps, which the
    compiled training step runs, make uint8 images of the output size from them, the same after
    a resume."""
    data = loaded(crop_scale=(0.5, 0.9))
    assert data.device, "the augmentation runs in the compiled step"
    stream = data.train(DataPartition())
    resumed = data.train(DataPartition())
    try:
        read = next(stream)
        assert not isinstance(read["image"], jax.Array) and "image_augmentation_key" in read
        first = augmented(data, read)
        assert isinstance(first["image"], jax.Array)
        assert first["image"].dtype == jnp.uint8
        assert first["image"].shape == (4, 12, 12, 3)
        assert set(first) == {"image", "label"}
        saved = stream.get_state()
        expected = augmented(data, next(stream))
        resumed.set_state(saved)
        actual = augmented(data, next(resumed))
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])
        assert not np.array_equal(first["image"], expected["image"])
        from dew.distributed.batches import DevicePrefetchIterator
        from dew.distributed.placement import MeshSpec

        mesh = MeshSpec().build([jax.devices()[0]])
        with DevicePrefetchIterator(stream, mesh) as prefetch:
            for _ in range(3):
                jax.block_until_ready(next(prefetch))
            with steady_state():
                batches = [next(prefetch) for _ in range(3)]
                jax.block_until_ready(batches)
    finally:
        stream.close()
        resumed.close()


@pytest.mark.mesh
def test_the_device_steps_jitter_rows_an_explicit_mesh_axis_types():
    """A run whose data axis is Explicit (`MeshSpec(explicit=("data",))`) places its rows typed
    by that axis; the compiled device steps crop, flip and jitter them into the pictures the
    unplaced batch makes, within one uint8 code of a compilation's rounding."""
    from dew.distributed.batches import shard_batch, split_fields
    from dew.distributed.placement import MeshSpec

    data = loaded(batch=8, seed=5, crop_scale=(0.5, 0.9))
    stream = data.train(DataPartition())
    try:
        staged, _ = split_fields(next(stream))
    finally:
        stream.close()
    expected = augmented(data, staged)
    mesh = MeshSpec(explicit=("data",)).build(jax.devices()[:8])
    with jax.set_mesh(mesh):
        placed = shard_batch(mesh, staged)
        found = jax.jit(lambda rows: augmented(data, rows))(placed)
    difference = np.asarray(found["image"], np.int32) - np.asarray(expected["image"], np.int32)
    assert found["image"].shape == (8, 12, 12, 3) and np.abs(difference).max() <= 1


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("crop", [(0, 0, 17, 21), (3, 4, 9, 11), (16, 20, 1, 1)])
def test_supplied_parameters_match_opencv_at_highest_precision(order, crop):
    from dew.data.image_augmentation import ImageParameters, apply_device, apply_host

    image = np.random.default_rng(7).integers(0, 256, (17, 21, 3)).astype(np.float64)
    parameters = ImageParameters(np.asarray(crop), np.ones((), dtype=bool),
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


@pytest.mark.parametrize("case", range(6))
def test_the_jitter_is_torchvisions_colorjitter_in_every_order(case):
    """torchvision 0.29's own `adjust_brightness`, `adjust_contrast` and
    `adjust_saturation` (tools/colour_jitter_reference.py), in each of the
    six orders, on images whose brightened pixels leave the range, so each
    op's clamp and the grey weights decide them. The device path and the
    host path in float64, over the whole image with no flip (the resize is
    then the identity), are torchvision's float64 run to 1e-9; in float32
    the device path and the host loader's `jitter_host` round to the code
    torchvision's float32 run rounds to, or one next to it."""
    from dew.data.image_augmentation import ImageParameters, apply_device, apply_host, jitter_host

    image = TORCHVISION[f"{case}/image"]
    factors, order = TORCHVISION[f"{case}/factors"], TORCHVISION[f"{case}/order"]
    side = image.shape[0]
    parameters = ImageParameters(np.asarray([0, 0, side, side]), np.zeros((), dtype=bool), factors, order)
    device = jax.devices()[0]
    if device.platform == "tpu":
        device = jax.devices("cpu")[0]  # TPU has no float64 operations.
    with jax.enable_x64(), jax.default_device(device):
        exact = jax.jit(lambda x, p: apply_device(x, p, side))(image.astype(np.float64), parameters)
    for label, got in (("device", exact), ("host", apply_host(image.astype(np.float64), parameters, side))):
        np.testing.assert_allclose(np.asarray(got), TORCHVISION[f"{case}/jittered_f64"], rtol=0, atol=1e-9,
                                   err_msg=label)
    reference = np.rint(TORCHVISION[f"{case}/jittered"])
    for got in (jax.jit(lambda x, p: apply_device(x, p, side))(image, parameters),
                jitter_host(image.astype(np.float32), factors, order)):
        assert np.abs(np.clip(np.rint(np.asarray(got)), 0, 255) - reference).max() <= 1


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
    parameters = ImageParameters(np.asarray([5, 11, 143, 139]), np.ones((), dtype=bool),
                                 np.asarray([1.19, 1.05, 0.8]), np.asarray([2, 1, 0]))
    expected = apply_host(image.astype(np.float64), parameters, 128)
    actual = jax.jit(lambda x, p: apply_device(x, p, 128))(image, parameters)
    rounded = np.clip(np.rint(actual), 0, 255).astype(np.int16)
    reference = np.clip(np.rint(expected), 0, 255).astype(np.int16)
    assert np.max(np.abs(rounded - reference)) <= 1


def test_without_augmentation_each_picture_is_its_record_resized_by_opencvs_area_rule():
    """With augmentation off, a training batch holds each record's pixels resized to the output
    size by OpenCV's area interpolation, as `cv2.resize` makes them from the record itself."""
    import cv2

    rows = {int(row["label"]): row["image"] for row in ROWS}
    stream = loaded(augment=False).train(DataPartition())
    try:
        batch = next(stream)
    finally:
        stream.close()
    for label, image in zip(np.asarray(batch["label"]), np.asarray(batch["image"]), strict=True):
        expected = cv2.resize(rows[int(label)], (12, 12), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(image, expected)


def test_device_pixels_are_not_fetched_back_when_the_trainer_places_them():
    from dew.distributed.batches import shard_batch
    from dew.distributed.placement import MeshSpec

    stream = loaded().train(DataPartition())
    try:
        batch = next(stream)
        mesh = MeshSpec().build([jax.devices()[0]])
        with guarded(allow=("host_to_device",)):
            placed = shard_batch(mesh, batch)
            jax.block_until_ready(placed)
        np.testing.assert_array_equal(placed["image"], batch["image"])
    finally:
        stream.close()


def test_device_draws_follow_record_keys_across_threads_and_partitions():
    threaded = Loading(workers=0, threads=4, read_buffer=8)
    whole = loaded(crop_scale=(0.6, 0.9)).train(DataPartition())
    parallel = loaded(crop_scale=(0.6, 0.9), loading=threaded).train(DataPartition())
    shares = [loaded(crop_scale=(0.6, 0.9)).train(DataPartition(index=i, count=2)) for i in range(2)]
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
    worker_loading = Loading(workers=1, threads=1, read_buffer=2)
    data = loaded(crop_scale=(0.6, 0.9))
    parent = data.train(DataPartition())
    worker = loaded(crop_scale=(0.6, 0.9), loading=worker_loading).train(DataPartition())
    try:
        for _ in range(3):
            expected, actual = augmented(data, next(parent)), augmented(data, next(worker))
            assert isinstance(actual["image"], jax.Array)
            for name in expected:
                np.testing.assert_array_equal(actual[name], expected[name])
    finally:
        parent.close()
        worker.close()


@pytest.mark.parametrize("host", [False, True], ids=["device", "host-master"])
def test_a_run_checks_its_inputs_against_what_the_device_steps_make(host):
    """Pictures staged at 24 pixels reach the compiled step, which crops them to the model's 12: the
    run checks its declared inputs against the device steps' output, not the staged batch, and a
    run whose state the host holds augments in its accelerator forward too."""
    import optax

    from dew import Field, InputSpec, Trainer
    from dew.diffusion.presets import Flow
    from dew.nn.backbones import SimpleDiT
    from dew.objectives.diffusion import DiffusionObjective
    from dew.sampling import Euler
    from dew.training import Layout

    objective = DiffusionObjective(SimpleDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2,
                                             mlp_ratio=1),
                                   Flow(), InputSpec(Field("image", (12, 12, 3))), solver=Euler(), steps=2)
    data = loaded(batch=8, seed=0, staging_size=24, objective=objective.data())
    assert np.shape(next(iter(data.train(DataPartition())))["image"])[1:] == (24, 24, 3)
    layout = Layout(host=("variables",) if host else (), min_shard=1, tolerance=1.)
    state = Trainer(objective, optax.adam(1e-3), key=0, layout=layout).fit(data, steps=2, log_every=1)
    assert int(state.step) == 2


@pytest.mark.mesh(devices=8)
def test_a_device_steps_rows_keep_their_explicit_axis_where_the_run_reads_them():
    """Pictures staged at 24 pixels and cropped to 12 by the compiled step, on eight devices whose
    data axis is Explicit and the rest Auto: the device steps' output the run checks and traces its
    loss on keeps the rows split over that axis, as the step's own trace types them, and the run
    trains. Cropped and flipped, not jittered: jitter's colour switch inside a loop does not trace
    over rows an Explicit axis types (a finding of its own)."""
    import optax

    from dew import Field, InputSpec, Trainer
    from dew.diffusion.presets import Flow
    from dew.distributed.batches import shard_batch, split_fields
    from dew.nn.backbones import SimpleDiT
    from dew.objectives.diffusion import DiffusionObjective
    from dew.sampling import Euler
    from dew.training import Layout, MeshSpec

    objective = DiffusionObjective(SimpleDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2,
                                             mlp_ratio=1),
                                   Flow(), InputSpec(Field("image", (12, 12, 3))), solver=Euler(), steps=2)
    data = loaded(batch=8, seed=0, staging_size=24, jitter=False, objective=objective.data())
    trainer = Trainer(objective, optax.adam(1e-3), key=0, mesh=MeshSpec(explicit=("data",)),
                      layout=Layout(min_shard=1, tolerance=1.))
    state = trainer.fit(data, steps=2, log_every=1)
    assert int(state.step) == 2
    staged, _ = split_fields(next(iter(data.train(DataPartition()))))
    # Under the run's mesh, where the step and its checks trace.
    with jax.set_mesh(trainer.device_mesh):
        image = trainer._model_shapes(shard_batch(trainer.device_mesh, staged))["image"]
    assert image.shape == (8, 12, 12, 3)
    assert image.sharding is not None and image.sharding.spec[0] == "data"


def test_a_host_step_written_after_a_device_step_is_refused_in_every_spelling():
    """A device step runs in the compiled step, after the batch's host steps: a host step written
    after it, in the same map or a later one, would read the staged pixels, and is refused."""
    from dew.data import images

    def remember(batch):
        return {**batch, "seen": np.asarray(batch["image"]).sum(axis=(1, 2, 3))}

    rows = Memory([{"image": np.zeros((16, 16, 3), np.uint8)} for _ in range(8)])
    keyed = rows.random_map(images.AugmentationKey()).repeat().batch(4)
    with pytest.raises(ValueError):
        keyed.map(images.DeviceAugment(12), remember)
    augmented = keyed.map(images.DeviceAugment(12))
    with pytest.raises(ValueError):
        augmented.map(remember)
    from dew.data.compile import compile

    assert compile(keyed.map(remember).map(images.DeviceAugment(12)), read="memory", loading=READ).device
