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
from diffusers_reference_helpers import drawn_wan_vae
from reference_vjp import recorded_vjp

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

    model, generator = drawn_wan_vae(AutoencoderKLQwenImage21, CONFIG, SEED, root / "vae")
    named = list(model.named_parameters())

    image = (torch.rand((BATCH, CONFIG["in_channels"], HEIGHT, WIDTH), generator=generator) * 2 - 1
             ).requires_grad_()
    posterior = model.encode(image[:, :, None]).latent_dist
    mean, std = posterior.mean[:, :, 0], posterior.std[:, :, 0]
    encoded = recorded_vjp("encode", {"probe_mean": mean, "probe_std": std}, {"image": image}, named,
                           generator)

    latent = torch.randn(mean.shape, generator=generator).requires_grad_()
    pixels = model.decode(latent[:, :, None], return_dict=False)[0][:, :, 0]
    decoded = recorded_vjp("decode", {"probe": pixels}, {"latent": latent}, named, generator)

    arrays = {
        "image": image.detach().numpy(), "mean": mean.detach().numpy(), "std": std.detach().numpy(),
        "latent": latent.detach().numpy(), "pixels": pixels.detach().numpy(), **encoded, **decoded,
    }
    clamped = float((pixels.detach().abs() >= 1).float().mean())
    print(f"parameters {len(named)} ({sum(value.numel() for _, value in named)} values); "
          f"latent {tuple(mean.shape)}; |mean| <= {np.abs(arrays['mean']).max():.3g}; "
          f"std in [{arrays['std'].min():.3g}, {arrays['std'].max():.3g}]; clamped pixels {clamped:.2%}")
    return arrays


def main(destination: str) -> None:
    import diffusers

    torch.set_num_threads(2)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    arrays = build(root)
    np.savez_compressed(root / "qwen_image_vae.npz", **arrays)
    record = {"diffusers": diffusers.__version__, "commit": DIFFUSERS_COMMIT, "config": CONFIG,
              "batch": BATCH, "height": HEIGHT, "width": WIDTH, "seed": SEED}
    (root / "qwen_image_vae.json").write_text(json.dumps(record, indent=1) + "\n")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"{root}: {size / 1e6:.2f} MB")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "bundle":
        from diffusers_dc_ae_reference import bundle

        bundle(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dew-qwen-image-vae-reference")
