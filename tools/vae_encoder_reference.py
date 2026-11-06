#!/usr/bin/env python3
"""Write the VAE encoder fixture with Diffusers' own Encoder.

Diffusers 0.34.0's `AutoencoderKL` encoder (`diffusers.models.autoencoders.vae
.Encoder`, its moments doubled for the mean and log-variance) at the
geometry tests/test_models.py's encoder runs at: widths (32, 64), one resnet
per level, 8 norm groups, 4 latent channels, 17x17 images whose odd side
the stride-two downsample pads. Every parameter is moved off its
initialization, the group norms' unit scales and zero shifts included (and
rounded to bfloat16-representable values, so the fixture compresses). The
tool records those weights; the test runs Diffusers' encoder on them,
in float32 and in its float64 twin (`torch_precision.widened`), over the
images and probes it draws, through `walk`.

Run with the Dew test environment's torch and diffusers, on CPU:

    python tools/vae_encoder_reference.py OUTPUT.npz
"""

import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import torch
from diffusers import AutoencoderKL

CONFIG = {"in_channels": 3, "out_channels": 3, "latent_channels": 4,
          "down_block_types": ["DownEncoderBlock2D"] * 2, "up_block_types": ["UpDecoderBlock2D"] * 2,
          "block_out_channels": [32, 64], "layers_per_block": 1, "norm_num_groups": 8, "act_fn": "silu"}
SIZE = 17
SEED = 67


def walk(encoder, image, probe, dtype):
    """The moments at `dtype`, and the gradients of `sum(moments * probe)` with
    respect to the image and every parameter."""
    encoder = encoder.to(dtype)
    image = image.to(dtype).requires_grad_()
    named = list(encoder.named_parameters())
    moments = encoder(image)
    grads = torch.autograd.grad((moments * probe.to(dtype)).sum(), [image] + [p for _, p in named])
    arrays = {"moments": moments, "grad_image": grads[0]}
    for (name, _), gradient in zip(named, grads[1:], strict=True):
        arrays[f"grad_param.{name}"] = gradient
    return {key: value.detach().numpy() for key, value in arrays.items()}


def main():
    from reference_source import saved_fixture

    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    encoder = AutoencoderKL.from_config(CONFIG).encoder.eval()
    generator = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
            parameter.copy_(parameter.to(torch.bfloat16).float())
    arrays = {f"param.{name}": value.detach().numpy().astype(ml_dtypes.bfloat16).view(np.uint16)
              for name, value in encoder.named_parameters()}
    meta = {"diffusers": __import__("diffusers").__version__, "torch": torch.__version__, "config": CONFIG,
            "size": SIZE}
    saved_fixture(Path(sys.argv[1]), arrays, meta)
    print(f"{sys.argv[1]}: {Path(sys.argv[1]).stat().st_size / 1e6:.2f} MB, {len(arrays)} arrays")


if __name__ == "__main__":
    main()
