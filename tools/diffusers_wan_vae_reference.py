"""The actual Wan 2.1 video VAE, saved and walked, for tests/fixtures.

A tiny `AutoencoderKLWan` from diffusers 0.39.0 is constructed with the
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

`bfloat16 FIXTURE_DIR OUTPUT.npz` walks the saved fixture again in bfloat16
(`bfloat16_walk`), for tests/fixtures/wan_vae_bf16.npz.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers_reference_helpers import drawn_wan_vae
from reference_vjp import recorded_vjp
from reference_releases import RELEASES, require

DIFFUSERS = RELEASES["diffusers"]["diffusers"]
CONFIG = {"base_dim": 8, "z_dim": 4, "dim_mult": [1, 2, 2, 2], "num_res_blocks": 2, "attn_scales": [],
          "temperal_downsample": [False, True, True], "dropout": 0.0}
BATCH, FRAMES = 2, 9
HEIGHT, WIDTH = 32, 48
SEED = 37


def build(root: Path) -> dict[str, np.ndarray]:
    from diffusers import AutoencoderKLWan

    model, generator = drawn_wan_vae(AutoencoderKLWan, CONFIG, SEED, root / "vae")
    named = list(model.named_parameters())

    video = (torch.rand((BATCH, 3, FRAMES, HEIGHT, WIDTH), generator=generator) * 2 - 1).requires_grad_()
    posterior = model.encode(video).latent_dist
    mean, std = posterior.mean, posterior.std
    encoded = recorded_vjp("encode", {"probe_mean": mean, "probe_std": std}, {"video": video}, named,
                           generator)

    latent = torch.randn(mean.shape, generator=generator).requires_grad_()
    pixels = model.decode(latent).sample
    decoded = recorded_vjp("decode", {"probe": pixels}, {"latent": latent}, named, generator)

    with torch.no_grad():
        image_mean = model.encode(video[:, :, :1].detach()).latent_dist.mean

    arrays = {
        "video": video.detach().numpy(), "mean": mean.detach().numpy(), "std": std.detach().numpy(),
        "latent": latent.detach().numpy(), "pixels": pixels.detach().numpy(),
        "image_mean": image_mean.numpy(), **encoded, **decoded,
    }
    clamped = float((pixels.detach().abs() >= 1).float().mean())
    print(f"parameters {len(named)} ({sum(value.numel() for _, value in named)} values); "
          f"latent {tuple(mean.shape)}; pixels {tuple(pixels.shape)}; clamped pixels {clamped:.2%}")
    return arrays


def bfloat16_walk(fixture: Path) -> dict[str, np.ndarray]:
    """The saved fixture VAE walked in bfloat16 as `WanPipeline` runs it under
    `torch_dtype=torch.bfloat16`: one nine-frame video's posterior and the
    decode of one latent, with the input gradients of both. Inputs and probes
    are bfloat16 values; everything is recorded as float32."""
    from diffusers import AutoencoderKLWan

    model = AutoencoderKLWan.from_pretrained(fixture / "vae", torch_dtype=torch.bfloat16).eval()
    generator = torch.Generator().manual_seed(SEED + 1)

    def bfloat16(tensor):
        return tensor.bfloat16().float()

    video = torch.rand((1, 3, FRAMES, HEIGHT, WIDTH), generator=generator) * 2 - 1
    video = video.bfloat16().requires_grad_()
    posterior = model.encode(video).latent_dist
    mean, std = posterior.mean, posterior.std
    probe_mean = bfloat16(torch.randn(mean.shape, generator=generator))
    probe_std = bfloat16(torch.randn(std.shape, generator=generator))
    ((mean.float() * probe_mean).sum() + (std.float() * probe_std).sum()).backward()

    latent = torch.randn(mean.shape, generator=generator).bfloat16().requires_grad_()
    pixels = model.decode(latent).sample
    probe = bfloat16(torch.randn(pixels.shape, generator=generator))
    (pixels.float() * probe).sum().backward()
    tensors = {"video": video, "mean": mean, "std": std, "probe_mean": probe_mean, "probe_std": probe_std,
               "encode.grad_video": video.grad, "latent": latent, "pixels": pixels, "probe": probe,
               "decode.grad_latent": latent.grad}
    return {name: tensor.detach().float().numpy() for name, tensor in tensors.items()}


def main(destination: str) -> None:
    import diffusers

    require("diffusers")
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
        from diffusers_dc_ae_reference import bundle

        bundle(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 3 and sys.argv[1] == "bfloat16":
        import diffusers

        require("diffusers")
        torch.set_num_threads(2)
        np.savez_compressed(sys.argv[3], **bfloat16_walk(Path(sys.argv[2])))
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dew-wan-vae-reference")
