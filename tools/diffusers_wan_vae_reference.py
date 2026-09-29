"""The actual Wan 2.1 video VAE, saved and walked, for tests/fixtures.

A tiny `AutoencoderKLWan` from diffusers 0.34.0 is constructed with the
published architecture's controls (four levels, two residual blocks each,
time halved at the second and third downsampling, the mid-block attention)
at narrow widths, saved with `save_pretrained` - its real config and its
real safetensors - and run as `WanPipeline` runs it: `vae.encode(video)`
walks the frames in the source's causal chunks (the first frame, then four
at a time) and `vae.decode(z)` one latent frame at a time. Recorded in
float32, channel-first as `[B, C, T, H, W]`:

- the posterior mean and standard deviation of a nine-frame rectangular
  batch, and the gradient of `sum(mean * probe_mean + std * probe_std)`
  with respect to the pixels and every parameter;
- the decode of a fixed three-frame latent and the gradient of
  `sum(pixels * probe)` with respect to the latent and every parameter;
- the posterior mean of the batch's first frame alone, an image.

Weights are the source's own initialization with every RMS gamma and bias
moved off its init, rounded to bfloat16-representable values so the fixture
compresses; they are still float32 tensors.

    python tools/diffusers_wan_vae_reference.py OUTPUT_DIR
    python tools/diffusers_wan_vae_reference.py bundle OUTPUT_DIR tests/fixtures/wan_vae.tar.xz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

DIFFUSERS = "0.34.0"
CONFIG = {"base_dim": 8, "z_dim": 4, "dim_mult": [1, 2, 2, 2], "num_res_blocks": 2, "attn_scales": [],
          "temperal_downsample": [False, True, True], "dropout": 0.0}
BATCH, FRAMES = 2, 9
HEIGHT, WIDTH = 32, 48
SEED = 37


def build(root: Path) -> dict[str, np.ndarray]:
    from diffusers import AutoencoderKLWan

    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    z_dim = CONFIG["z_dim"]
    latents_mean = torch.randn(z_dim, generator=generator).mul(0.5).tolist()
    latents_std = torch.rand(z_dim, generator=generator).add(0.5).tolist()
    model = AutoencoderKLWan(**CONFIG, latents_mean=latents_mean, latents_std=latents_std).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("gamma", "bias")):
                parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
            parameter.copy_(parameter.to(torch.bfloat16).float())
    model.save_pretrained(root / "vae", safe_serialization=True)
    named = list(model.named_parameters())

    def gradients(loss, inputs):
        every = inputs + [value for _, value in named]
        found = torch.autograd.grad(loss, every, allow_unused=True)
        return [torch.zeros_like(value) if gradient is None else gradient
                for gradient, value in zip(found, every, strict=True)]

    video = (torch.rand((BATCH, 3, FRAMES, HEIGHT, WIDTH), generator=generator) * 2 - 1).requires_grad_()
    posterior = model.encode(video).latent_dist
    mean, std = posterior.mean, posterior.std
    probe_mean = torch.randn(mean.shape, generator=generator)
    probe_std = torch.randn(std.shape, generator=generator)
    encoded = gradients((mean * probe_mean).sum() + (std * probe_std).sum(), [video])

    latent = torch.randn(mean.shape, generator=generator).requires_grad_()
    pixels = model.decode(latent).sample
    probe = torch.randn(pixels.shape, generator=generator)
    decoded = gradients((pixels * probe).sum(), [latent])

    with torch.no_grad():
        image_mean = model.encode(video[:, :, :1].detach()).latent_dist.mean

    arrays = {
        "video": video.detach().numpy(), "mean": mean.detach().numpy(), "std": std.detach().numpy(),
        "probe_mean": probe_mean.numpy(), "probe_std": probe_std.numpy(),
        "encode.grad_video": encoded[0].numpy(),
        "latent": latent.detach().numpy(), "pixels": pixels.detach().numpy(), "probe": probe.numpy(),
        "decode.grad_latent": decoded[0].numpy(), "image_mean": image_mean.numpy(),
    }
    for (name, _), from_encode, from_decode in zip(named, encoded[1:], decoded[1:], strict=True):
        arrays[f"encode.grad.{name}"] = from_encode.numpy()
        arrays[f"decode.grad.{name}"] = from_decode.numpy()
    clamped = float((pixels.detach().abs() >= 1).float().mean())
    print(f"parameters {len(named)} ({sum(value.numel() for _, value in named)} values); "
          f"latent {tuple(mean.shape)}; pixels {tuple(pixels.shape)}; clamped pixels {clamped:.2%}")
    return arrays


def bundle(directory: str, destination: str) -> None:
    """Pack the saved VAE and the recorded arrays for the suite."""
    import tarfile

    root = Path(directory)
    with tarfile.open(destination, "w:xz") as archive:
        for path in sorted(root.iterdir()):
            archive.add(path, arcname=path.name)
    print(f"{destination}: {Path(destination).stat().st_size / 1e6:.2f} MB")


def main(destination: str) -> None:
    import diffusers

    if diffusers.__version__ != DIFFUSERS:
        raise RuntimeError(f"recorded against diffusers {DIFFUSERS}, not {diffusers.__version__}")
    torch.set_num_threads(2)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "wan_vae.npz", **build(root))
    record = {"diffusers": DIFFUSERS, "config": CONFIG, "batch": BATCH, "frames": FRAMES,
              "height": HEIGHT, "width": WIDTH, "seed": SEED}
    (root / "wan_vae.json").write_text(json.dumps(record, indent=1) + "\n")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"{root}: {size / 1e6:.2f} MB")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "bundle":
        bundle(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dew-wan-vae-reference")
