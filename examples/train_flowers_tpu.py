"""Text-to-image DiT from scratch on Oxford Flowers, across a TPU slice.

Prepare the TFDS ArrayRecords once:

    python -c "import tensorflow_datasets as tfds; tfds.builder('oxford_flowers102', \\
        data_dir='~/.cache/dew/datasets').download_and_prepare(\\
        file_format='array_record')"

Then launch the same file on every worker of the slice:

    python examples/train_flowers_tpu.py \\
        --data ~/.cache/dew/datasets/oxford_flowers102/2.1.1 --steps 200000

`--data` reads whichever of two layouts it is given: the TFDS version
directory, its captions made from the class names, or a directory of
captioned ArrayRecords Dew published (its `manifest.json`). The smoke run
writes a handful of synthetic records in that second layout and trains on
them on one CPU device:

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
from dew.config import MetricConfig, ModelConfig, ObjectiveConfig, OptimConfig, TrainerConfig
from dew.config.diffusion import DiffusionRunConfig
from dew.coordination import collective_host
from dew.data import TFDS, Files, Loading, images, presets
from dew.data.sources.codec import pack_dict_of_byte_arrays
from dew.diffusion.presets import EDM
from dew.inputs import TextCondition
from dew.nn.backbones import SimpleDiT
from dew.optim import Cosine
from dew.sampling import CFG
from dew.sampling.solvers import Heun
from dew.training import LocalTracker, MeshSpec, ProfileWindow, prepare_process

PROMPTS = ("a water lily", "a sunflower", "a red rose", "a purple orchid")

# The tiny CLIP text tower committed for the tests: a real checkpoint, 77
# tokens of context and 32 features, so the smoke run exercises the same
# conditioning path as the real one without reaching the Hub.
SMOKE_CLIP = Path(__file__).resolve().parents[1] / "tests/fixtures/clip/tiny"
# The FID extractor at a sixteenth of every channel width, drawn rather than
# trained: the same pooled statistics and distance, on numbers of its own.
SMOKE_INCEPTION = (Path(__file__).resolve().parents[1]
                   / "tests/fixtures/inception/tiny/inception_v3_fid.safetensors")


@dataclass
class Config:
    data: Path | None = None
    """The records: a TFDS version directory, or a directory of captioned ArrayRecords Dew published."""
    out: Path = Path("runs/flowers-tpu")
    image_size: int = 256
    batch_size: int = 256
    steps: int = 200_000
    learning_rate: float = 1e-4
    sampling_steps: int = 40
    guidance: float = 3.0
    clip_model: str = "openai/clip-vit-large-patch14"
    """The checkpoint CLIPScore is read from, for conditioning and for scoring."""
    inception_weights: str | None = None
    """The FID extractor's parameters as a file; unset downloads the published
    checkpoint. --smoke reads the committed tiny one instead."""
    model: dict = field(default_factory=lambda: {
        "patch_size": 2, "emb_features": 1024, "num_layers": 24, "num_heads": 16})
    smoke: bool = False
    """Train the synthetic fixture on one device for a few steps instead."""


def synthetic_records(directory: Path, count: int, size: int) -> None:
    """`count` captioned noise images as captioned ArrayRecords.

    The pixels are tests/test_data.py's own fixture, one image per index from
    that index's seed, so the smoke run reads the records the data tests read.
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
    """The same run at the size a laptop finishes: one device, tiny everything."""
    synthetic_records(out / "data", count=16, size=16)
    train, held = Files(str(out / "data" / "*.array_record"), format="array_record").holdout(records=4)
    loading = Loading(workers=0, threads=1, read_buffer=2, worker_buffer=1)
    return DiffusionRunConfig(
        model=ModelConfig.from_model(SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2,
                                               dtype=jnp.float32)),
        data=train.shuffle(seed=0).repeat().pipe(presets.images, 16, batch=4).loading(loading),
        validation=held.pipe(presets.images, 16, batch=4).loading(loading),
        preset=EDM(regime="pixel"),
        objective=ObjectiveConfig("diffusion", {"solver": Heun(), "guidance": CFG(2.0), "steps": 2,
                                                "ema_decay": 0.9}),
        text=TextCondition(encoder="clip_text", checkpoint=config.clip_model),
        val_metrics=(MetricConfig("fid", {"weights": config.inception_weights}),
                     MetricConfig("clip_score", {"modelname": config.clip_model})),
        optim=OptimConfig(learning_rate=1e-3),
        trainer=TrainerConfig(checkpoint_dir=str(out / "checkpoints"), steps=3,
                              log_every=1, eval_every=3, checkpoint_every=3,
                              mesh=MeshSpec(fsdp=1), multi_host=False,
                              compilation_cache_dir=None))


def slice_config(config: Config) -> DiffusionRunConfig:
    """The real run: a DiT over the whole slice, CLIP-conditioned, EMA'd."""
    if config.data is None:
        raise ValueError("--data is the prepared record directory; --smoke writes its own")
    path = str(config.data.expanduser())
    records = (config.data / "manifest.json").is_file() or (config.data / "_dew_manifest.json").is_file()
    # Prepared records carry their captions; the TFDS source's are its class names.
    source = (Files(path, format="array_record") if records
              else TFDS("oxford_flowers102", split="all", path=path))
    caption = None if records else images.caption("a photo of a {label}")
    train, held = source.holdout(records=4 * config.batch_size)
    return DiffusionRunConfig(
        model=ModelConfig.from_model(SimpleDiT(**config.model, dtype=jnp.bfloat16)),
        data=train.shuffle(seed=0).repeat().pipe(presets.images, config.image_size, batch=config.batch_size,
                                                 augment=True, caption=caption),
        validation=held.pipe(presets.images, config.image_size, batch=config.batch_size, caption=caption),
        preset=EDM(regime="pixel"),
        objective=ObjectiveConfig("diffusion", {"solver": Heun(), "guidance": CFG(config.guidance),
                                                "steps": config.sampling_steps, "ema_decay": 0.9999}),
        text=TextCondition(encoder="clip_text", checkpoint=config.clip_model),
        val_metrics=(MetricConfig("fid", {"weights": config.inception_weights}),
                     MetricConfig("clip_score", {"modelname": config.clip_model})),
        optim=OptimConfig(weight_decay=0.01, learning_rate=Cosine(
            peak_value=config.learning_rate, warmup_steps=2000, init_value=config.learning_rate,
            end_value=2e-4)),
        trainer=TrainerConfig(checkpoint_dir=str(config.out / "checkpoints"),
                              steps=config.steps,
                              log_every=50, eval_every=2000, checkpoint_every=2000,
                              profile=ProfileWindow(str(config.out / "profile"), steps=5,
                                                    warmup=20),
                              multi_host=True))


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
        # Only now: counting devices opens the backend, which has to come
        # after the slice's processes join, and a pool's count is every
        # host's devices.
        run = replace(run, trainer=replace(run.trainer, mesh=MeshSpec(fsdp=jax.device_count())))

    # The objective encodes each batch's captions with its text condition as the run reads them.
    objective = run.build()
    name = "smoke" if config.smoke else f"flowers-{config.image_size}"
    # The run scores FID and CLIPScore over the held-out split at its end, on every process's
    # share at once, with the extractor and the CLIP the flags name (`val_metrics`), and logs
    # them under its last step.
    state = run.train(objective, run.data, validation=run.validation, name=name,
                      metrics=run.build_eval_metrics(),
                      summary={"prompts": list(PROMPTS)})

    # The run directory read back as a pipeline, which draws over every process's
    # devices; process 0 gathers the grid's rows and writes both files once.
    run_dir = Path(run.trainer.checkpoint_dir) / name
    pipe = dew.pipeline(str(run_dir))
    drawn = pipe(list(PROMPTS), steps=objective.steps, guidance=config.guidance, solver=Heun(), key=1)
    images = collective_host(drawn.images, phase="the sample grid", held_by="first")
    if images is not None:
        grid(images[:len(PROMPTS)], config.out / "samples.png")
        with LocalTracker(run_dir / "tracking") as journal:
            final = {metric: value for step, scores in journal.scalars() if step == int(state.step)
                     for metric, value in scores.items() if metric.startswith("val/")}
        (config.out / "eval.json").write_text(json.dumps(final, indent=2))
        print(f"samples {config.out / 'samples.png'}  {final}")
    return run_dir


if __name__ == "__main__":
    main(tyro.cli(Config))
