#!/usr/bin/env python3
"""Write the autoencoder-contract fixture with Diffusers' own VAE and posterior.

Over the 16-channel VAE tests/fixtures/vae/sd3-tiny holds (Diffusers 0.34.0
`AutoencoderKL`, SD3's shift and scale), the tool records what Diffusers
computes for a clip's frames, each encoded as its own image:

- the latents SD3's img2img pipeline draws and normalizes
  (`pipeline_stable_diffusion_3_img2img.py:700-702`: `retrieve_latents` of
  `vae.encode`, which is `DiagonalGaussianDistribution.sample`, then
  `(z - shift_factor) * scaling_factor`, the one line taken as written);
- the frames its `vae.decode` makes of the normalized mean latents, taken
  back the way SD3's pipeline takes latents back
  (`pipeline_stable_diffusion_3.py:1128`: `z / scaling_factor +
  shift_factor`, also as written);
- `DiagonalGaussianDistribution(moments).sample()` on moments whose
  log-variances run past its [-30, 20] clamp on both sides.

Every draw is the standard normal JAX's `jax.random.normal` gives Dew's
encoder for the same key and shape (KEY, channels last), handed to
Diffusers through its `randn_tensor`, so the two sample from one draw. Each
is recorded in float32 and float64 (`diffusers_wan_reference.widened`), the
truth tests/reference_error.py measures both from.

Run with the Dew test environment's torch, diffusers and jax, on CPU:

    python tools/autoencoder_contract_reference.py OUTPUT.npz
"""

import json
import sys
from pathlib import Path
from unittest import mock

import jax
import numpy as np
import torch
from diffusers import AutoencoderKL
from diffusers.models.autoencoders import vae as diffusers_vae
from diffusers_wan_reference import widened

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "tests" / "fixtures" / "vae" / "sd3-tiny"
CLIP = (2, 3, 16, 16)  # batch, frames, height, width
KEY = 7
SEED = 71


def channels_first(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.moveaxis(value, -1, 1))


def draw(shape: tuple[int, ...]) -> np.ndarray:
    """JAX's standard normal draw for KEY at `shape` (channels last)."""
    return np.asarray(jax.random.normal(jax.random.key(KEY), shape, np.float32))


def sampled(encoder_output, epsilon: torch.Tensor) -> torch.Tensor:
    """`retrieve_latents`' sample, its `randn_tensor` handing over `epsilon`.
    The pipeline module imports transformers names diffusers_wan_reference
    shims, so it is imported after that."""
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import retrieve_latents

    def handed(shape, **options):
        return epsilon.to(options["dtype"])

    with mock.patch.object(diffusers_vae, "randn_tensor", handed):
        return retrieve_latents(encoder_output, sample_mode="sample")


def walk(model, frames, moments, dtype) -> dict[str, np.ndarray]:
    model = model.to(dtype)
    config = model.config
    pixels = torch.from_numpy(channels_first(frames)).to(dtype)
    encoded = model.encode(pixels)
    mean = encoded.latent_dist.mean
    epsilon = torch.from_numpy(channels_first(draw(tuple(np.moveaxis(mean.detach().numpy(), 1, -1).shape))))
    latents = (sampled(encoded, epsilon) - config.shift_factor) * config.scaling_factor
    normalized_mean = (mean - config.shift_factor) * config.scaling_factor
    decoded = model.decode(normalized_mean / config.scaling_factor + config.shift_factor).sample
    posterior = diffusers_vae.DiagonalGaussianDistribution(
        torch.from_numpy(channels_first(moments)).to(dtype))
    noise = torch.from_numpy(channels_first(draw(moments[..., moments.shape[-1] // 2:].shape)))
    with mock.patch.object(diffusers_vae, "randn_tensor", lambda shape, **kw: noise.to(kw["dtype"])):
        drawn = posterior.sample()
    arrays = {"latents": latents, "normalized_mean": normalized_mean, "decoded": decoded, "posterior": drawn}
    return {key: np.moveaxis(value.detach().numpy(), 1, -1) for key, value in arrays.items()}


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    torch.set_num_threads(1)
    model = AutoencoderKL.from_pretrained(TINY).eval()
    rng = np.random.default_rng(SEED)
    frames = rng.uniform(-1, 1, (CLIP[0] * CLIP[1], CLIP[2], CLIP[3], 3)).astype(np.float32)
    moments = rng.standard_normal((2, 4, 4, 8)).astype(np.float32)
    moments[..., 4:] *= 20  # log-variances from about -60 to 60, past the clamp both ways
    arrays: dict[str, np.ndarray] = {"frames": frames, "moments": moments}
    with torch.no_grad():
        single = walk(model, frames, moments, torch.float32)
        with widened():
            truth = walk(model, frames, moments, torch.float64)
    arrays.update({f"fp32.{key}": value for key, value in single.items()})
    arrays.update({f"fp64.{key}": value for key, value in truth.items()})
    meta = {"diffusers": __import__("diffusers").__version__, "torch": torch.__version__,
            "jax": jax.__version__, "clip": CLIP, "key": KEY}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(sys.argv[1], **arrays)
    clamped = int((np.abs(moments[..., 4:]) > 20).sum())
    print(f"{sys.argv[1]}: {Path(sys.argv[1]).stat().st_size / 1e6:.2f} MB, {len(arrays)} arrays, "
          f"{clamped} log-variances past the clamp")


if __name__ == "__main__":
    main()
