"""Sample the pretrained text-to-image model over prompts, seeds and samplers.

Run from the repository root on a GPU, with a checkpoint step and the
training config saved beside it:

    python examples/sample_text_to_image.py --checkpoint cmbd8bia/1350000 \
        --config cmbd8bia/cmbd8bia_config.json

or with a Dew run directory, read through `TextToImage.from_run`:

    python examples/sample_text_to_image.py --checkpoint RUN_DIR

The recorded grids come from the first form: a hybrid DiT of state-space and
attention blocks trained with Dew, at step 1350000 of its EMA weights. Its
denoiser holds 175.6M parameters, beside a 123.1M-parameter CLIP text encoder
and an 83.7M-parameter Stable Diffusion autoencoder, and it samples 256x256
images. The second form has been run on the small text-conditioned run that
`examples/train_flowers_tpu.py --smoke` writes.

Each sampler draws one batch per seed, holding every prompt. The output
directory gets one PNG per image, one grid per sampler (rows are prompts,
columns are seeds) and manifest.json, which records each image's prompt, seed,
sampler, steps and guidance, and each batch's wall time. The first batch of a
sampler includes compiling it, or reading it from JAX's compilation cache.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

from dew.artifacts import uint8_pixels
from dew.interop.flaxdiff import load_flaxdiff
from dew.sampling import CFG, DPMSolverMultistep, EulerAncestral, Heun, TextToImage

# Each sampler's solver, step count and classifier-free guidance.
SAMPLERS = {
    "heun40": (Heun(), 40, CFG(5.0)),
    "dpm2m20": (DPMSolverMultistep(), 20, CFG(5.0)),
    "ea100": (EulerAncestral(), 100, CFG(5.0)),
}


@dataclass
class Config:
    checkpoint: Path
    """A Dew run directory, or a checkpoint step directory with --config."""
    config: Path | None = None
    """The training config (JSON) saved with a checkpoint step; given, --checkpoint is that step."""
    jax_version: str = "0.5.3"
    """The jax version the checkpoint step was written under."""
    out: Path = Path("runs/sample-text-to-image")
    """A new directory for the images, grids and manifest."""
    prompts: tuple[str, ...] = (
        "a tropical beach with palm trees and turquoise water",
        "a colorful hot air balloon over a green valley",
        "a red fox in a snowy forest",
        "a stained glass window with geometric patterns",
        "a bowl of ramen with an egg and green onions",
        "the northern lights over a frozen lake at night",
    )
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    samplers: tuple[str, ...] = tuple(SAMPLERS)
    negative: str | None = None
    """The prompt for the unconditional branch of classifier-free guidance; None is the empty prompt."""


def load(config: Config) -> TextToImage:
    if config.config is None:
        return TextToImage.from_run(str(config.checkpoint))
    return load_flaxdiff(config.checkpoint, json.loads(config.config.read_text()),
                         jax_version=config.jax_version)


def grid(title: str, prompts, seeds, images: np.ndarray) -> Image.Image:
    """`images[row, column]` as one image, labelled with prompts and seeds.

    The background is grey, so the white or black bands the model draws along
    some images' edges stay visible as part of those images."""
    rows, columns, height, width = images.shape[:4]
    font = ImageFont.load_default(size=14)
    label, gap = 22, 4
    heading = title + "   seeds " + ", ".join(map(str, seeds))
    # As wide as the images, or as the longest caption where it is wider.
    widest = max(font.getlength(caption) for caption in (heading, *prompts)) + 8
    sheet = Image.new("RGB", (max(columns * (width + gap) - gap, int(widest)), label + rows * (label + height)),
                      "#b0b0b0")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), heading, fill="black", font=font)
    for row, prompt in enumerate(prompts):
        top = label + row * (label + height)
        draw.text((4, top + 4), prompt, fill="black", font=font)
        for column in range(columns):
            sheet.paste(Image.fromarray(images[row, column]), (column * (width + gap), top + label))
    return sheet


def main(config: Config):
    out = config.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    prompts = list(config.prompts)
    started = time.perf_counter()
    pipe = load(config)
    print(f"Loaded in {time.perf_counter() - started:.1f} s on {jax.devices()[0].device_kind}")
    manifest = {"checkpoint": str(config.checkpoint), "device": jax.devices()[0].device_kind,
                "negative": config.negative, "batches": [], "images": []}
    for name in config.samplers:
        solver, steps, guidance = SAMPLERS[name]
        (out / name).mkdir()
        columns = []
        for seed in config.seeds:
            started = time.perf_counter()
            prepared = pipe.prepare(prompts, seed=seed, steps=steps, unconditional=config.negative)
            images = pipe(prepared, seed=seed, steps=steps, sampler=solver, guidance=guidance).host().images
            seconds = time.perf_counter() - started
            print(f"{name}: seed {seed}, {len(prompts)} images in {seconds:.1f} s")
            pixels = uint8_pixels(np.asarray(images, np.float32))
            columns.append(pixels)
            manifest["batches"].append({"sampler": name, "seed": seed, "images": len(prompts),
                                        "seconds": round(seconds, 2)})
            for row, prompt in enumerate(prompts):
                file = f"{name}/p{row}_s{seed}.png"
                Image.fromarray(pixels[row]).save(out / file)
                manifest["images"].append({
                    "file": file, "prompt": prompt, "seed": seed, "sampler": type(solver).__name__,
                    "steps": steps, "guidance": guidance.scale, "guidance_interval": guidance.interval})
        title = f"{type(solver).__name__}, {steps} steps, guidance {guidance.scale}"
        if guidance.interval != (0.0, 1.0):
            title += f" over progress {guidance.interval}"
        grid(title, prompts, config.seeds, np.stack(columns, axis=1)).save(out / f"grid-{name}.png")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved grids, images and manifest.json to {out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
