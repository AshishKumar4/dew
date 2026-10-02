"""Decode/resize and synchronized Oxford Flowers batch throughput on one corpus.

    python tools/benchmark_image_pipeline.py --flowers /data/oxford_flowers102/2.1.1 \
        --images 512 --repeats 5 --batch 32 --steps 100 --out image-pipeline.json

Decoder timings read the same encoded JPEG bytes from RAM, use RGB/stored
orientation and the same OpenCV area/cubic final resize. TFDS's NumPy image
decoder is the one its ArrayRecord data source uses, not a tf.data graph.
The pipeline timing includes Grain, decode/resize, augmentation, transfer and
a synchronized device reduction. It measures input throughput, not training.
The separate training timing runs a small pixel-EDM objective through the
Trainer's compiled, prefetched update, with compilation and warmup excluded.
"""

import hashlib
import importlib.metadata
import io
import itertools
import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
import tyro
from PIL import Image

from dew.data import DataPartition, Loading, OxfordFlowers
from dew.data.image_augmentation import ImageParameters, apply_device, apply_host
from dew.data.images import decode_image, resize_image


@dataclass
class Config:
    flowers: str
    images: int = 512
    repeats: int = 5
    image_size: int = 128
    batch: int = 32
    steps: int = 100
    warmup: int = 5
    threads: int = 4
    decoder_only: bool = False
    training: bool = True
    """Also time the real Trainer's prefetched pixel-diffusion update."""
    out: Path | None = None


def decoders(size):
    feature = tfds.features.Image(shape=(None, None, 3))

    def pillow(encoded):
        with Image.open(io.BytesIO(encoded)) as image:
            return resize_image(np.asarray(image.convert("RGB")), size)

    runs = {
        "opencv_full": lambda encoded: resize_image(decode_image(encoded), size),
        "opencv_reduced": lambda encoded: resize_image(decode_image(encoded, at_least=size), size),
        "pillow": pillow,
        "tfds_numpy": lambda encoded: resize_image(feature.decode_example_np(encoded), size),
    }
    try:
        import torch
        from torchvision.io import ImageReadMode, decode_image as torchvision_decode
    except ImportError:
        return runs

    def torchvision(encoded):
        tensor = torch.from_numpy(np.frombuffer(encoded, np.uint8).copy())
        pixels = torchvision_decode(tensor, mode=ImageReadMode.RGB, apply_exif_orientation=False)
        return resize_image(pixels.permute(1, 2, 0).numpy(), size)

    runs["torchvision"] = torchvision
    return runs


def decoder_timings(encoded, size, repeats):
    report = {}
    reference = [resize_image(decode_image(value), size).astype(np.int16) for value in encoded]
    for name, decode in decoders(size).items():
        outputs = [decode(value) for value in encoded]
        error = max(int(np.max(np.abs(output.astype(np.int16) - target)))
                    for output, target in zip(outputs, reference, strict=True))
        durations = []
        for _ in range(repeats):
            started = time.perf_counter()
            for value in encoded:
                decode(value)
            durations.append(time.perf_counter() - started)
        report[name] = {"images_per_second": len(encoded) / statistics.median(durations),
                        "seconds": durations, "max_pixel_error_vs_full_cv2": error}
    return report


def augmentation_accuracy():
    """The three crop geometries and six jitter orders used by the parity tests."""
    device = jax.devices()[0]
    if device.platform == "tpu":
        device = jax.devices("cpu")[0]
    maximum, rounded = 0.0, 0
    with jax.enable_x64(), jax.default_device(device):
        image = np.random.default_rng(7).integers(0, 256, (17, 21, 3)).astype(np.float64)
        run = jax.jit(lambda x, p: apply_device(x, p, 13))
        for crop, order in itertools.product([(0, 0, 17, 21), (3, 4, 9, 11), (16, 20, 1, 1)],
                                             itertools.permutations(range(3))):
            parameters = ImageParameters(crop=np.asarray(crop), flip=True,
                                         factors=np.asarray([1.13, 0.97, 0.86]), order=np.asarray(order))
            actual = np.asarray(run(image, parameters))
            expected = apply_host(image, parameters, 13)
            maximum = max(maximum, float(np.max(np.abs(actual - expected))))
            error = (np.clip(np.rint(actual), 0, 255).astype(np.int16)
                     - np.clip(np.rint(expected), 0, 255).astype(np.int16))
            rounded = max(rounded, int(np.max(np.abs(error))))
    return {"device": device.device_kind, "dtype": "float64", "cases": 18,
            "maximum_absolute_error": maximum, "maximum_uint8_code_error": rounded,
            "bound": float(255 * 8 * np.finfo(np.float32).eps)}


def pipeline_timing(config, backend, crop_scale):
    spec = OxfordFlowers(path=config.flowers, image_size=config.image_size,
                         augmentation_backend=backend, crop_scale=crop_scale, val_batches=0,
                         loading=Loading(workers=0, threads=config.threads, read_buffer=2 * config.batch))
    stream = spec.load(batch=config.batch).train(DataPartition())
    consume = jax.jit(lambda pixels: jnp.mean(pixels.astype(jnp.float32)))
    durations, cpu_times = [], []
    try:
        for _ in range(config.warmup):
            jax.block_until_ready(consume(next(stream)["image"]))
        for _ in range(config.repeats):
            cpu_started = time.process_time()
            started = time.perf_counter()
            for _ in range(config.steps):
                jax.block_until_ready(consume(next(stream)["image"]))
            durations.append(time.perf_counter() - started)
            cpu_times.append(time.process_time() - cpu_started)
    finally:
        stream.close()
    samples = config.steps * config.batch
    return {"images_per_second": samples / statistics.median(durations),
            "wall_seconds": durations, "host_cpu_seconds": cpu_times,
            "host_cpu_ms_per_batch": 1000 * statistics.median(cpu_times) / config.steps}


def training_timing(config, backend):
    import optax

    from dew.diffusion.presets import EDM
    from dew.inputs import Field, InputSpec
    from dew.nn.backbones import SimpleDiT
    from dew.objectives.diffusion import DiffusionObjective
    from dew.training import Trainer
    from dew.training.distributed import DevicePrefetchIterator

    spec = OxfordFlowers(path=config.flowers, image_size=config.image_size,
                         augmentation_backend=backend, val_batches=0,
                         loading=Loading(workers=0, threads=config.threads, read_buffer=2 * config.batch))
    model = SimpleDiT(patch_size=16, emb_features=32, num_layers=1, num_heads=4,
                      dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, attention_impl="reference")
    objective = DiffusionObjective(model, EDM(regime="pixel"),
                                  InputSpec(Field("image", (config.image_size, config.image_size, 3))),
                                  ema_decay=None)
    trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0))
    state, _, _ = trainer.place()
    source = spec.load(batch=config.batch).train(DataPartition.of(trainer.device_mesh))
    durations = []
    with DevicePrefetchIterator(source, trainer.device_mesh) as batches:
        step = trainer.compile(state, next(batches))
        for _ in range(config.warmup):
            state, loss, _, _, _ = step(state, next(batches))
        jax.block_until_ready(state)
        for _ in range(config.repeats):
            started = time.perf_counter()
            for _ in range(config.steps):
                state, loss, _, _, _ = step(state, next(batches))
            jax.block_until_ready(state)
            durations.append(time.perf_counter() - started)
    if not np.isfinite(float(loss)):
        raise ValueError("pixel-diffusion benchmark produced a non-finite loss")
    return {"images_per_second": config.steps * config.batch / statistics.median(durations),
            "wall_seconds": durations, "final_loss": float(loss),
            "model": "SimpleDiT patch=16 width=32 layers=1 heads=4 fp32/HIGHEST",
            "updates": int(state.updates)}


def main(config):
    cv2.setNumThreads(1)
    source = OxfordFlowers(path=config.flowers).source()
    encoded = [source[index]["image"] for index in range(min(config.images, len(source)))]
    if not encoded:
        raise ValueError("the Flowers corpus contains no images")
    cpu = platform.processor()
    if not cpu:
        with open("/proc/cpuinfo") as handle:
            cpu = next(line.split(":", 1)[1].strip() for line in handle if line.startswith("model name"))
    versions = {"jax": jax.__version__, "opencv": cv2.__version__, "tfds": tfds.__version__,
                "pillow": importlib.metadata.version("pillow")}
    try:
        versions["torchvision"] = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError:
        versions["torchvision"] = "not installed"
    report = {"hardware": {"cpu": cpu, "device": jax.devices()[0].device_kind},
              "versions": versions,
              "images": len(encoded), "corpus_sha256": hashlib.sha256(b"".join(encoded)).hexdigest(),
              "config": {name: value for name, value in vars(config).items() if name != "out"},
              "augmentation_parity": augmentation_accuracy(),
              "decoders": decoder_timings(encoded, config.image_size, config.repeats)}
    if not config.decoder_only:
        # Each backend reads the same shuffled records. Warm the whole source
        # before either, so the second backend gets no cold-storage advantage.
        for index in range(len(source)):
            source[index]
        report["pipeline"] = {
            f"{backend}_{mode}": pipeline_timing(config, backend, scale)
            for mode, scale in (("flip_jitter", (1.0, 1.0)), ("crop_flip_jitter", (0.6, 1.0)))
            for backend in ("host", "device")
        }
        if config.training:
            report["training"] = {backend: training_timing(config, backend) for backend in ("host", "device")}
    text = json.dumps(report, indent=2)
    if config.out is not None:
        config.out.parent.mkdir(parents=True, exist_ok=True)
        config.out.write_text(text + "\n")
    print(text)
    return report


if __name__ == "__main__":
    main(tyro.cli(Config))
