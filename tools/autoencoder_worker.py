"""Diffusers' own VAE and SD3's latent normalization (tests/fixtures/vae/sd3-tiny, Diffusers 0.34.0
`AutoencoderKL`, SD3's shift and scale) run on drawn clips and noise, in float32 and float64: the reference
arm of tools/venv_reference.py's `vae-contract` site (its protocol tools/reference_worker.py), run in Dew's
own environment (its torch and diffusers), on the CPU.

    python tools/autoencoder_worker.py vae-contract    `IN OUT` lines on stdin, each `done`

IN holds each draw's `frames` `[N, H, W, 3]`, a clip's frames each encoded as its own image, and `noise`,
the standard normal Dew's encoder draws for the clip's key at the latents' shape (channels last), handed to
Diffusers through its `randn_tensor` so the two sample from one draw. OUT gets `<draw>.reference.<leaf>`
and `<draw>.truth.<leaf>` in float32 and float64 (the model's float64 twin under `torch_precision.widened`):
the `latents` SD3's img2img pipeline draws and normalizes (`pipeline_stable_diffusion_3_img2img.py:700-702`:
`retrieve_latents` of `vae.encode`, which is `DiagonalGaussianDistribution.sample`, then
`(z - shift_factor) * scaling_factor`, the one line taken as written), the `normalized_mean`, and the frames
its `vae.decode` makes of the normalized mean taken back the way SD3's pipeline takes latents back
(`pipeline_stable_diffusion_3.py:1128`: `z / scaling_factor + shift_factor`, also as written), all channels
last.
"""

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from diffusers import AutoencoderKL
from diffusers.models.autoencoders import vae as diffusers_vae
from reference_worker import serve
from torch_precision import float64_twin, widened

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "tests" / "fixtures" / "vae" / "sd3-tiny"


def channels_first(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.moveaxis(value, -1, 1))


def sampled(encoder_output, epsilon: torch.Tensor) -> torch.Tensor:
    """`retrieve_latents`' sample, its `randn_tensor` handing over `epsilon`.
    The pipeline module imports transformers names diffusers_wan_reference
    shims, so it is imported after that."""
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import retrieve_latents

    def handed(shape, **options):
        return epsilon.to(options["dtype"])

    with mock.patch.object(diffusers_vae, "randn_tensor", handed):
        return retrieve_latents(encoder_output, sample_mode="sample")


def walk(model, frames: np.ndarray, noise: np.ndarray, dtype) -> dict[str, np.ndarray]:
    model = model.to(dtype)
    config = model.config
    pixels = torch.from_numpy(channels_first(frames)).to(dtype)
    encoded = model.encode(pixels)
    mean = encoded.latent_dist.mean
    latents = (sampled(encoded, torch.from_numpy(channels_first(noise))) - config.shift_factor) * (
        config.scaling_factor)
    normalized_mean = (mean - config.shift_factor) * config.scaling_factor
    decoded = model.decode(normalized_mean / config.scaling_factor + config.shift_factor).sample
    arrays = {"latents": latents, "normalized_mean": normalized_mean, "decoded": decoded}
    return {key: np.moveaxis(value.detach().numpy(), 1, -1) for key, value in arrays.items()}


def run(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's latents, normalized mean and decoded frames, in float32 and float64."""
    model = AutoencoderKL.from_pretrained(TINY).eval()
    wide = float64_twin(model)
    out = {}
    with torch.no_grad():
        for name in sorted({key.split(".", 1)[0] for key in cases}):
            frames, noise = cases[f"{name}.frames"], cases[f"{name}.noise"]
            for leaf, value in walk(model, frames, noise, torch.float32).items():
                out[f"{name}.reference.{leaf}"] = value
            with widened():
                for leaf, value in walk(wide, frames, noise, torch.float64).items():
                    out[f"{name}.truth.{leaf}"] = value
    return out


if __name__ == "__main__":
    if sys.argv[1:] != ["vae-contract"]:
        raise SystemExit(__doc__)
    torch.set_num_threads(1)
    serve(run)
