"""Train a text-to-image DiT from scratch on Oxford Flowers across a TPU slice.

Prepare records once at the training resolution. Training then reads pixels
with a memcpy, without decoding JPEGs:

    python -c "import tensorflow_datasets as tfds; tfds.builder('oxford_flowers102', \\
        data_dir='~/.cache/dew/datasets').download_and_prepare(\\
        file_format='array_record')"
    python tools/prepare_images.py --dataset tfds_images \\
        --data-path ~/.cache/dew/datasets/oxford_flowers102/2.1.1 \\
        --split all --image-size 256 --out prepared/flowers-256

Then launch the same file on every worker of the slice:

    python examples/train_flowers_tpu.py --data prepared/flowers-256 --steps 200000

`--data` accepts a TFDS version directory as `tfds_images` or the output of
`prepare_images.py` as `array_record_images`. The smoke run writes a few
synthetic records in the second layout and trains on one CPU device:

    JAX_PLATFORMS=cpu python examples/train_flowers_tpu.py --smoke --out /tmp/flowers-smoke
"""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from PIL import Image

import dew
from dew.artifacts import uint8_pixels
from dew.config import ModelConfig, OptimConfig, TrainerConfig
from dew.data import ArrayRecordImages, DataPartition, Loading, TFDSImages
from dew.data.images import pack_dict_of_byte_arrays
from dew.diffusion.presets import EDM
from dew.eval import FID, CLIPScore
from dew.nn.backbones import SimpleDiT
from dew.objectives.diffusion import DiffusionRunConfig, TextCondition
from dew.sampling import CFG
from dew.sampling.solvers import Heun
from dew.training import MeshSpec, ProfileWindow, prepare_process
from dew.training.optim import Cosine

PROMPTS = ("a water lily", "a sunflower", "a red rose", "a purple orchid")

# This committed CLIP checkpoint has 77 context tokens and 32 features.
# The smoke run uses the same conditioning path as the full run without Hub access.
SMOKE_CLIP = Path(__file__).resolve().parents[1] / "tests/fixtures/clip/tiny"
# This FID extractor has one-sixteenth of each channel width and untrained
# parameters. It computes the same pooled statistics and distance for the fixture.
SMOKE_INCEPTION = (Path(__file__).resolve().parents[1]
                   / "tests/fixtures/inception/tiny/inception_v3_fid.safetensors")


@dataclass
class Config:
    data: Path | None = None
    """The prepared records: a TFDS version directory or a prepare_images.py output."""
    out: Path = Path("runs/flowers-tpu")
    image_size: int = 256
    batch_size: int = 256
    steps: int = 200_000
    learning_rate: float = 1e-4
    sampling_steps: int = 40
    guidance: float = 3.0
    clip_model: str = "openai/clip-vit-large-patch14"
    """CLIP checkpoint used for both conditioning and CLIPScore."""
    inception_weights: str | None = None
    """FID extractor weights file. If unset, download the published checkpoint.
    --smoke uses the committed tiny checkpoint."""
    model: dict = field(default_factory=lambda: {
        "patch_size": 2, "emb_features": 1024, "num_layers": 24, "num_heads": 16})
    smoke: bool = False
    """Train the synthetic fixture for a few steps on one device."""


def synthetic_records(directory: Path, count: int, size: int) -> None:
    """Write `count` captioned noise images in prepare_images.py's layout.

    Each index seeds one image, matching the pixel fixture in tests/test_data.py.
    The smoke run therefore reads the same records as the data tests.
    """
    from array_record.python.array_record_module import ArrayRecordWriter

    names = ("rose", "tulip", "lotus", "orchid", "marigold")
    directory.mkdir(parents=True, exist_ok=True)
    writer = ArrayRecordWriter(str(directory / "synthetic-00000-of-00001.array_record"),
                               options="group_size:1")
    for index in range(count):
        pixels = np.random.RandomState(index).randint(0, 256, (size, size, 3), np.uint8)
        writer.write(pack_dict_of_byte_arrays({
            "image": pixels.tobytes(),
            "shape": np.asarray(pixels.shape[:2], np.int32).tobytes(),
            "caption": f"a photo of a {names[index % len(names)]}".encode(),
            "label": np.asarray([index % len(names)], np.int32).tobytes()}))
    writer.close()


def smoke_config(config: Config, out: Path) -> DiffusionRunConfig:
    """Configure a small version of the run for one device."""
    synthetic_records(out / "data", count=16, size=16)
    return DiffusionRunConfig(
        model=ModelConfig.from_model(SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2,
                                               dtype=jnp.float32)),
        data=ArrayRecordImages(path=str(out / "data"), image_size=16, augmentation="none",
                               val_batches=1, loading=Loading(workers=0, threads=1,
                                                              read_buffer=2, worker_buffer=1)),
        preset=EDM(regime="pixel"),
        solver=Heun(),
        guidance=CFG(2.0),
        sampling_steps=2,
        ema_decay=0.9,
        text=TextCondition(encoder="clip_text", checkpoint=config.clip_model),
        # PSNR exercises validation without downloads. Both fid and clip_score
        # would download metric weights.
        val_metrics=("psnr",),
        optim=OptimConfig(learning_rate=1e-3),
        trainer=TrainerConfig(checkpoint_dir=str(out / "checkpoints"), batch_size=4, steps=3,
                              log_every=1, eval_every=3, checkpoint_every=3,
                              mesh=MeshSpec(fsdp=1), multi_host=False,
                              compilation_cache_dir=None))


def slice_config(config: Config) -> DiffusionRunConfig:
    """Configure a CLIP-conditioned DiT with EMA over the whole TPU slice."""
    if config.data is None:
        raise ValueError("--data is the prepared record directory; --smoke writes its own")
    path = str(config.data.expanduser())
    prepared = (ArrayRecordImages(path=path) if (config.data / "manifest.json").is_file()
                else TFDSImages(path=path))
    return DiffusionRunConfig(
        model=ModelConfig.from_model(SimpleDiT(**config.model, dtype=jnp.bfloat16)),
        data=replace(prepared, image_size=config.image_size, val_batches=4),
        preset=EDM(regime="pixel"),
        solver=Heun(),
        guidance=CFG(config.guidance),
        sampling_steps=config.sampling_steps,
        ema_decay=0.9999,
        text=TextCondition(encoder="clip_text", checkpoint=config.clip_model),
        val_metrics=("fid", "clip_score"),
        optim=OptimConfig(weight_decay=0.01, schedule=Cosine(
            peak=config.learning_rate, warmup_steps=2000, init=config.learning_rate,
            end=2e-4)),
        trainer=TrainerConfig(checkpoint_dir=str(config.out / "checkpoints"),
                              batch_size=config.batch_size, steps=config.steps,
                              log_every=50, eval_every=2000, checkpoint_every=2000,
                              profile=ProfileWindow(str(config.out / "profile"), steps=5,
                                                    warmup=20),
                              multi_host=True))


def held_out(run: DiffusionRunConfig) -> np.ndarray:
    """The uint8 images of the validation pass, which the run never trained on."""
    data = run.data.load(batch=run.trainer.batch_size)
    if data.val is None:
        raise ValueError("scoring FID needs a held-out split: set data.val_batches")
    # This process's share of the pass on the run's plain data-parallel mesh.
    share = DataPartition.of(run.trainer.mesh.build())
    return np.concatenate([np.asarray(batch["image"], np.uint8) for batch in data.val(share)])


def grid(images: np.ndarray, path: Path) -> None:
    """The sampled rows side by side as one PNG."""
    pixels = uint8_pixels(images)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(list(pixels), axis=1)).save(path)


def main(config: Config) -> Path:
    if config.smoke:
        config = replace(config, clip_model=str(SMOKE_CLIP), inception_weights=str(SMOKE_INCEPTION))
    run = smoke_config(config, config.out) if config.smoke else slice_config(config)
    prepare_process(run.trainer.wandb, run.trainer.multi_host, run.trainer.xla_flags,
                    run.trainer.compilation_cache_dir, layout=run.trainer.layout)
    if not config.smoke:
        # Counting devices opens the backend, so the slice's processes must join
        # first. In a pool, the count includes devices on every host.
        run = replace(run, trainer=replace(run.trainer, mesh=MeshSpec(fsdp=jax.device_count())))

    objective = run.build()
    data = run.data.load(batch=run.trainer.batch_size, tokenize=objective.inputs.tokenize)
    name = "smoke" if config.smoke else f"flowers-{config.image_size}"
    run.train(objective, data, name=name, metrics=run.build_eval_metrics(),
              summary={"prompts": list(PROMPTS)})

    run_dir = Path(run.trainer.checkpoint_dir) / name
    pipe = dew.pipeline(str(run_dir))
    drawn = pipe(list(PROMPTS), steps=run.sampling_steps, guidance=config.guidance,
                 solver=Heun(), key=1).host().images
    grid(drawn, config.out / "samples.png")

    generated = uint8_pixels(drawn)
    report = {"clip_score": CLIPScore(config.clip_model).score(generated, list(PROMPTS)),
              "fid": FID(weights=config.inception_weights).score(generated, held_out(run))}
    (config.out / "eval.json").write_text(json.dumps(report, indent=2))
    print(f"samples {config.out / 'samples.png'}  {report}")
    return run_dir


if __name__ == "__main__":
    main(tyro.cli(Config))
