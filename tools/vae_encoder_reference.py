#!/usr/bin/env python3
"""Write the VAE encoder fixture with Diffusers' own Encoder.

Diffusers 0.34.0's `AutoencoderKL` encoder (`diffusers.models.autoencoders.vae
.Encoder`, its moments doubled for the mean and log-variance) at the
geometry tests/test_models.py's encoder runs at: widths (32, 64), one resnet
per level, 8 norm groups, 4 latent channels, 17x17 images whose odd side
the stride-two downsample pads. Every parameter is moved off its
initialization, the group norms' unit scales and zero shifts included (and
rounded to bfloat16-representable values, so the fixture compresses). The
tool records the moments and, against a fixed cotangent, the gradients of
the image and of every parameter, at a batch of one and of four: once in
float32 and once in float64 (`torch_precision.widened`), the truth
tests/reference_error.py measures both from.

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
BATCHES = (1, 4)
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
    from torch_precision import float64_twin, widened

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
    image = torch.randn((max(BATCHES), CONFIG["in_channels"], SIZE, SIZE), generator=generator)
    arrays: dict[str, np.ndarray] = {"image": image.numpy()}
    arrays.update({f"param.{name}": value.detach().numpy().astype(ml_dtypes.bfloat16).view(np.uint16)
                   for name, value in encoder.named_parameters()})
    # The float64 twin, built inside `float64` so its constructor computes in float64, with these weights.
    wide = float64_twin(encoder, lambda: AutoencoderKL.from_config(CONFIG).encoder)
    for batch in BATCHES:
        rows = image[:batch]
        with torch.no_grad():
            shape = encoder(rows).shape
        probe = torch.randn(shape, generator=generator)
        arrays[f"{batch}/probe"] = probe.numpy()
        single = walk(encoder, rows, probe, torch.float32)
        arrays.update({f"{batch}/fp32.{key}": value for key, value in single.items()})
        with widened():
            truth = walk(wide, rows, probe, torch.float64)
        arrays.update({f"{batch}/fp64.{key}": value for key, value in truth.items()})
        gap = np.abs(single["moments"] - truth["moments"]).max()
        print(f"batch {batch}: moments {tuple(shape)}, fp32 off float64 by {gap:.3g}")
    meta = {"diffusers": __import__("diffusers").__version__, "torch": torch.__version__, "config": CONFIG,
            "batches": BATCHES, "size": SIZE}
    saved_fixture(Path(sys.argv[1]), arrays, meta)
    print(f"{sys.argv[1]}: {Path(sys.argv[1]).stat().st_size / 1e6:.2f} MB, {len(arrays)} arrays")


if __name__ == "__main__":
    main()
