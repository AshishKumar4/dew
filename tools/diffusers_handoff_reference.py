"""Diffusers' SDXL base walking the first half of a trajectory and its refiner the second, the reference
tools/check_native_diffusion.py's `handoff` holds Dew's to (`handed`).

Over the `xl` and `refiner` pipelines of tests/fixtures/tiny_diffusers.tar.xz, Diffusers'
StableDiffusionXLPipeline (the base's PyTorch twins: UNet, VAE and both CLIP towers, with its DDIM scheduler
file) runs "cat" against "dog" at guidance 3 for four steps with `denoising_end=0.5` from a given noise, and
returns its latents. The StableDiffusionXLImg2ImgPipeline refiner takes those latents as `image` with
`denoising_start=0.5`, which adds no noise and walks the steps below the same cutoff, and returns its latents
and its decoded images. Both run in float32 or in float64 (`torch_precision.float64` for the models,
`diffusers_wan_reference.float64_scheduler` for the DDIM tables, built in float64).
"""

from __future__ import annotations

import contextlib
import functools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.diffusers_reference_helpers import restore_transformers_names
from tools.diffusers_wan_reference import float64_scheduler
from tools.torch_precision import float64

restore_transformers_names()

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


@functools.cache
def built(root: Path, wide: bool):
    """The base and refiner pipelines under `root`, at float64 where `wide`, built once."""
    from diffusers.schedulers import scheduling_ddim

    with float64() if wide else contextlib.nullcontext(), \
            float64_scheduler(scheduling_ddim) if wide else contextlib.nullcontext():
        return pipelines(root, torch.float64 if wide else torch.float32)


def handed(root: Path, noise: np.ndarray, *, wide: bool) -> dict[str, np.ndarray]:
    """The base's latents (`prefix`), the refiner's (`final`) and its decoded `images`, channels last, from
    `noise` (channels last), in float32 or (`wide`) float64."""
    from diffusers.schedulers import scheduling_ddim

    dtype = torch.float64 if wide else torch.float32
    base, refiner = built(Path(root), wide)
    with float64() if wide else contextlib.nullcontext(), \
            float64_scheduler(scheduling_ddim) if wide else contextlib.nullcontext(), torch.no_grad():
        latents = torch.from_numpy(np.ascontiguousarray(np.moveaxis(np.asarray(noise), -1, 1))).to(dtype)
        common = {"prompt": PROMPT, "negative_prompt": NEGATIVE, "num_inference_steps": STEPS,
                  "guidance_scale": GUIDANCE}
        prefix = base(**common, latents=latents, denoising_end=CUTOFF, output_type="latent").images
        final = refiner(**common, image=prefix, denoising_start=CUTOFF, output_type="latent").images
        images = refiner.vae.decode(final / refiner.vae.config.scaling_factor).sample
    return {"prefix": np.moveaxis(prefix.numpy(), 1, -1), "final": np.moveaxis(final.numpy(), 1, -1),
            "images": np.moveaxis(images.numpy(), 1, -1)}
