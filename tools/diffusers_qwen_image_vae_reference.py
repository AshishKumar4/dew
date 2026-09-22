"""The actual Qwen-Image 2.1 VAE, saved and walked, for tests/fixtures.

A tiny `AutoencoderKLQwenImage21` from diffusers 6256aa76 is constructed with
the published architecture's controls (residual blocks, RGBA, four spatial
halvings, the published temporal flags, no patching) at narrow widths, saved
with `save_pretrained` - its real config and its real safetensors - and run
the way `QwenImage21Pipeline` runs it on one frame: `vae.encode(image[:, :,
None])` for the posterior and `vae.decode(z[:, :, None])[0][:, :, 0]` for the
pixels. Recorded in float32, in the source's channel-first layout with the
frame axis dropped:

- the posterior mean and its clamped-log-variance standard deviation for a
  fixed rectangular RGBA batch, and the gradient of `sum(mean * probe_mean +
  std * probe_std)` with respect to the pixels and every parameter;
- the decode of a fixed latent and the gradient of `sum(pixels * probe)` with
  respect to the latent and every parameter.

A parameter a walk never reads (the other half of the autoencoder, and the
`time_conv` weights only a second frame reaches) is recorded with a zero
gradient. Weights are the source's own initialization with every RMS gamma
and bias moved off its init, rounded to bfloat16-representable values so the
fixture compresses; they are still float32 tensors.

Run in the isolated reference environment on CPU, then pack the output:

    python tools/diffusers_qwen_image_vae_reference.py OUTPUT_DIR
    python tools/diffusers_qwen_image_vae_reference.py bundle OUTPUT_DIR tests/fixtures/qwen_image_vae.tar.xz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

DIFFUSERS_COMMIT = "6256aa7666cedd47443adc8f82da9a10e110b09c"
CONFIG = {"base_dim": 4, "decoder_base_dim": 6, "z_dim": 8, "dim_mult": [1, 2, 2, 4, 4], "num_res_blocks": 1,
          "attn_scales": [], "temperal_downsample": [False, True, True, True], "dropout": 0.0,
          "is_residual": True, "in_channels": 4, "out_channels": 4, "patch_size": None,
          "scale_factor_temporal": 8, "scale_factor_spatial": 16}
BATCH = 2
HEIGHT, WIDTH = 32, 48
SEED = 29


def build(root: Path) -> dict[str, np.ndarray]:
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21

    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    z_dim = CONFIG["z_dim"]
    latents_mean = torch.randn(z_dim, generator=generator).mul(0.5).tolist()
    latents_std = torch.rand(z_dim, generator=generator).add(0.5).tolist()
    model = AutoencoderKLQwenImage21(**CONFIG, latents_mean=latents_mean, latents_std=latents_std).eval()
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

    image = (torch.rand((BATCH, CONFIG["in_channels"], HEIGHT, WIDTH), generator=generator) * 2 - 1
             ).requires_grad_()
    posterior = model.encode(image[:, :, None]).latent_dist
    mean, std = posterior.mean[:, :, 0], posterior.std[:, :, 0]
    probe_mean = torch.randn(mean.shape, generator=generator)
    probe_std = torch.randn(std.shape, generator=generator)
    encoded = gradients((mean * probe_mean).sum() + (std * probe_std).sum(), [image])

    latent = torch.randn(mean.shape, generator=generator).requires_grad_()
    pixels = model.decode(latent[:, :, None], return_dict=False)[0][:, :, 0]
    probe = torch.randn(pixels.shape, generator=generator)
    decoded = gradients((pixels * probe).sum(), [latent])

    arrays = {
        "image": image.detach().numpy(), "mean": mean.detach().numpy(), "std": std.detach().numpy(),
        "probe_mean": probe_mean.numpy(), "probe_std": probe_std.numpy(),
        "encode.grad_image": encoded[0].numpy(),
        "latent": latent.detach().numpy(), "pixels": pixels.detach().numpy(), "probe": probe.numpy(),
        "decode.grad_latent": decoded[0].numpy(),
    }
    for (name, _), from_encode, from_decode in zip(named, encoded[1:], decoded[1:], strict=True):
        arrays[f"encode.grad.{name}"] = from_encode.numpy()
        arrays[f"decode.grad.{name}"] = from_decode.numpy()
    clamped = float((pixels.detach().abs() >= 1).float().mean())
    print(f"parameters {len(named)} ({sum(value.numel() for _, value in named)} values); "
          f"latent {tuple(mean.shape)}; |mean| <= {np.abs(arrays['mean']).max():.3g}; "
          f"std in [{arrays['std'].min():.3g}, {arrays['std'].max():.3g}]; clamped pixels {clamped:.2%}")
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

    torch.set_num_threads(2)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    arrays = build(root)
    np.savez_compressed(root / "qwen_image_vae.npz", allow_pickle=False, **arrays)
    record = {"diffusers": diffusers.__version__, "commit": DIFFUSERS_COMMIT, "config": CONFIG,
              "batch": BATCH, "height": HEIGHT, "width": WIDTH, "seed": SEED}
    (root / "qwen_image_vae.json").write_text(json.dumps(record, indent=1) + "\n")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"{root}: {size / 1e6:.2f} MB")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "bundle":
        bundle(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dew-qwen-image-vae-reference")
