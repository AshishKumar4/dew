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
from dew.data import TFDS, Dataset, Files, Loading, images, presets
from dew.data.sources.codec import pack_dict_of_byte_arrays
from dew.diffusion.presets import EDM
from dew.distributed import collective_host
from dew.eval.fid import FID
from dew.eval.images import CLIPScore
from dew.inputs import CLIPText, Condition, Field, InputSpec
from dew.nn.backbones import SimpleDiT
from dew.objectives.diffusion import DiffusionObjective
from dew.optim import Cosine, OptimConfig
from dew.sampling import CFG
from dew.sampling.solvers import Heun
from dew.training import Checkpoints, LocalTracker, MeshSpec, ProfileWindow, Trainer, prepare_process

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


def smoke_data(out: Path) -> tuple[Dataset, Dataset]:
    """The same chains over synthetic records written here, at the size a laptop finishes."""
    synthetic_records(out / "data", count=16, size=16)
    train, held = Files(str(out / "data" / "*.array_record"), format="array_record").holdout(records=4)
    loading = Loading(workers=0, threads=1, read_buffer=2, worker_buffer=1)
    return (train.shuffle(seed=0).repeat().pipe(presets.images, 16, batch=4).loading(loading),
            held.pipe(presets.images, 16, batch=4).loading(loading))


def slice_data(config: Config) -> tuple[Dataset, Dataset]:
    """The real run's chains, every record but four batches held out for FID and CLIPScore."""
    if config.data is None:
        raise ValueError("--data is the prepared record directory; --smoke writes its own")
    path = str(config.data.expanduser())
    records = (config.data / "manifest.json").is_file() or (config.data / "_dew_manifest.json").is_file()
    # Prepared records carry their captions; the TFDS source's are its class names.
    source = (Files(path, format="array_record") if records
              else TFDS("oxford_flowers102", split="all", path=path))
    caption = None if records else images.caption("a photo of a {label}")
    train, held = source.holdout(records=4 * config.batch_size)
    return (train.shuffle(seed=0).repeat().pipe(presets.images, config.image_size, batch=config.batch_size,
                                                augment=True, caption=caption),
            held.pipe(presets.images, config.image_size, batch=config.batch_size, caption=caption))


def grid(images: np.ndarray, path: Path) -> None:
    """The sampled rows side by side as one PNG."""
    pixels = uint8_pixels(images)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(list(pixels), axis=1)).save(path)


def main(config: Config) -> Path:
    if config.smoke:
        config = replace(config, clip_model=str(SMOKE_CLIP), inception_weights=str(SMOKE_INCEPTION),
                         image_size=16)
    # The slice's processes join one pool before anything counts devices: a pool's count is every host's.
    prepare_process(multi_host=not config.smoke)
    train, held = smoke_data(config.out) if config.smoke else slice_data(config)
    size = config.image_size
    # Images conditioned on CLIP text; the objective encodes each batch's captions as the trainer reads them.
    inputs = InputSpec(sample=Field("image", (size, size, 3)),
                       conditions={"textcontext": Condition(CLIPText.from_pretrained(config.clip_model))})
    model = (SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2, output_channels=3,
                       dtype=jnp.float32) if config.smoke
             else SimpleDiT(**config.model, output_channels=3, dtype=jnp.bfloat16))
    steps = 2 if config.smoke else config.sampling_steps
    objective = DiffusionObjective(model, EDM(regime="pixel"), inputs, solver=Heun(),
                                   guidance=CFG(2.0 if config.smoke else config.guidance), steps=steps,
                                   ema_decay=0.9 if config.smoke else 0.9999)
    optimizer = (OptimConfig(learning_rate=1e-3) if config.smoke else OptimConfig(
        weight_decay=0.01, learning_rate=Cosine(peak_value=config.learning_rate, warmup_steps=2000,
                                                init_value=config.learning_rate, end_value=2e-4)))
    name = "smoke" if config.smoke else f"flowers-{config.image_size}"
    run_dir = config.out / "checkpoints" / name
    total = 3 if config.smoke else config.steps
    trainer = Trainer(objective, optimizer.build(total), key=jax.random.key(0),
                      mesh=MeshSpec(fsdp=1 if config.smoke else jax.device_count()),
                      checkpoints=Checkpoints(str(run_dir)), tracker=LocalTracker(run_dir / "tracking"),
                      profile=None if config.smoke else ProfileWindow(str(config.out / "profile"), steps=5,
                                                                      warmup=20))
    # The run scores FID and CLIPScore over the held-out split at its end, on every process's share at
    # once, with the extractor and the CLIP the flags name, and logs them under its last step.
    every = 3 if config.smoke else 2000
    scores = [FID(weights=config.inception_weights), CLIPScore(modelname=config.clip_model)]
    state = trainer.fit(train, validation=held, steps=total, log_every=1 if config.smoke else 50,
                        eval_every=every, checkpoint_every=every, metrics=scores)

    # The run directory read back as a pipeline, which draws over every process's
    # devices; process 0 gathers the grid's rows and writes both files once.
    pipe = dew.pipeline(str(run_dir))
    drawn = pipe(list(PROMPTS), steps=steps, guidance=config.guidance, solver=Heun(), key=1)
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
