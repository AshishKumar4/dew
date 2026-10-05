#!/usr/bin/env python3
"""Write tests/fixtures/diffusers_handoff.npz: Diffusers' SDXL base walking
the first half of a trajectory and its refiner the second.

Over the `xl` and `refiner` pipelines of tests/fixtures/tiny_diffusers.tar.xz,
Diffusers 0.34.0's StableDiffusionXLPipeline (the base's PyTorch twins:
UNet, VAE and both CLIP towers, with its DDIM scheduler file) runs "cat"
against "dog" at guidance 3 for four steps with `denoising_end=0.5`, from
the `xl` reference's noise, and returns its latents. The
StableDiffusionXLImg2ImgPipeline refiner takes those latents as `image`
with `denoising_start=0.5`, which adds no noise and walks the steps below
the same cutoff, and returns its latents and its decoded images. Both run
in float32 and in float64 (`diffusers_wan_reference.float64` for the
models, `diffusers_wan_reference.float64_scheduler` for the DDIM
tables, built in float64).

    PYTHONPATH=src python tools/diffusers_handoff_reference.py
"""

from __future__ import annotations

import contextlib
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

# Also restores the transformers names Diffusers 0.34.0's pipelines import.
from tools.diffusers_wan_reference import float64, float64_scheduler  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "diffusers_handoff.npz"
STEPS, GUIDANCE, CUTOFF = 4, 3.0, 0.5
PROMPT, NEGATIVE = "cat", "dog"


def pipelines(root: Path, dtype: torch.dtype):
    import diffusers
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    base = root / "xl"
    config = diffusers.DDIMScheduler.load_config(base, subfolder="scheduler")
    scheduler = diffusers.DDIMScheduler.from_config(config)
    first = diffusers.StableDiffusionXLPipeline(
        vae=diffusers.AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=dtype),
        text_encoder=CLIPTextModel.from_pretrained(base, subfolder="text_encoder", dtype=dtype),
        text_encoder_2=CLIPTextModelWithProjection.from_pretrained(base, subfolder="text_encoder_2",
                                                                   dtype=dtype),
        tokenizer=CLIPTokenizer.from_pretrained(base, subfolder="tokenizer"),
        tokenizer_2=CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2"),
        unet=diffusers.UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=dtype),
        scheduler=scheduler)
    second = diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained(root / "refiner", torch_dtype=dtype)
    return first, second


def run(root: Path, noise: np.ndarray, wide: bool) -> dict[str, np.ndarray]:
    from diffusers.schedulers import scheduling_ddim

    dtype = torch.float64 if wide else torch.float32
    scope = contextlib.ExitStack()
    if wide:
        scope.enter_context(float64())
        scope.enter_context(float64_scheduler(scheduling_ddim))
    with scope:
        base, refiner = pipelines(root, dtype)
        latents = torch.from_numpy(np.moveaxis(noise, -1, 1)).to(dtype)
        common = {"prompt": PROMPT, "negative_prompt": NEGATIVE, "num_inference_steps": STEPS,
                  "guidance_scale": GUIDANCE}
        with torch.no_grad():
            prefix = base(**common, latents=latents, denoising_end=CUTOFF, output_type="latent").images
            final = refiner(**common, image=prefix, denoising_start=CUTOFF, output_type="latent").images
            images = refiner.vae.decode(final / refiner.vae.config.scaling_factor).sample
    return {"prefix": np.moveaxis(prefix.numpy(), 1, -1), "final": np.moveaxis(final.numpy(), 1, -1),
            "images": np.moveaxis(images.numpy(), 1, -1)}


def main() -> None:
    import diffusers
    import transformers

    with tempfile.TemporaryDirectory() as extracted:
        with tarfile.open(ROOT / "tests" / "fixtures" / "tiny_diffusers.tar.xz") as archive:
            archive.extractall(extracted, members=[member for member in archive.getmembers()
                                                   if member.name.split("/")[0] in ("xl", "refiner")],
                               filter="data")
        root = Path(extracted)
        noise = np.load(root / "xl" / "reference.npz")["noise"]
        arrays = {"noise": noise}
        for precision, wide in (("fp32", False), ("fp64", True)):
            arrays.update({f"{precision}.{name}": value for name, value in run(root, noise, wide).items()})
    arrays["versions"] = np.asarray([diffusers.__version__, transformers.__version__, torch.__version__])
    np.savez_compressed(FIXTURE, **arrays)
    gap = float(np.abs(arrays["fp32.final"] - arrays["fp64.final"]).max())
    print(f"{FIXTURE}: refiner latents fp32 off float64 by {gap:.3g}")


if __name__ == "__main__":
    main()
