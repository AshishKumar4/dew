"""The actual DC-AE, saved and walked, for tests/fixtures.

Two tiny `AutoencoderDC`s from diffusers 0.34.0 are constructed with the
controls the published checkpoints use, at narrow widths:

- `conv`, as `dc-ae-f32c32-sana-1.1`: strided-convolution downsampling,
  nearest-neighbour upsampling, RMS norms and SiLU throughout, a multiscale
  (5x5) QKV branch in every EfficientViT block, and a residual stack at the
  first level;
- `shuffle`, as `dc-ae-f32c32-in-1.0`: pixel-unshuffle and pixel-shuffle
  resampling, no first level (its input and output convolutions are
  resampling blocks), batch norms and ReLU in the decoder's residual levels,
  and no multiscale branch.

Each is saved with `save_pretrained` - its real config and its real
safetensors - and recorded channel-first, as `vae.encode(x).latent` and
`vae.decode(z).sample` compute them, in float32 and, over the same inputs and
probes, in float64 under `fp64.` keys (`reference_vjp.twice`):

- the latent of a fixed rectangular batch, and the gradient of
  `sum(latent * probe_latent)` with respect to the pixels and every parameter;
- the decode of a fixed latent and the gradient of `sum(pixels * probe)` with
  respect to the latent and every parameter;
- the encode and decode of a batch small enough that the deepest level's
  attention takes the quadratic path (height * width <= attention_head_dim).

Weights are the source's own initialization with every norm's weight and
bias, and every batch norm's running statistics, moved off their init,
rounded to bfloat16-representable values so the fixture compresses; they are
still float32 tensors.

    python tools/diffusers_dc_ae_reference.py OUTPUT_DIR
    python tools/diffusers_dc_ae_reference.py bundle OUTPUT_DIR tests/fixtures/dc_ae.tar.xz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from reference_vjp import Draws, recorded_vjp, twice
from reference_releases import RELEASES, require

DIFFUSERS = RELEASES["diffusers"]["diffusers"]
COMMON = {"in_channels": 3, "latent_channels": 8, "attention_head_dim": 8,
          "encoder_block_types": ["ResBlock", "ResBlock", "EfficientViTBlock"],
          "decoder_block_types": ["ResBlock", "ResBlock", "EfficientViTBlock"],
          "encoder_block_out_channels": [8, 16, 32], "decoder_block_out_channels": [8, 16, 32]}
CONFIGS = {
    "conv": {**COMMON, "encoder_layers_per_block": [1, 1, 2], "decoder_layers_per_block": [1, 1, 2],
             "encoder_qkv_multiscales": [[], [], [5]], "decoder_qkv_multiscales": [[], [], [5]],
             "downsample_block_type": "Conv", "upsample_block_type": "interpolate",
             "decoder_norm_types": "rms_norm", "decoder_act_fns": "silu", "scaling_factor": 0.41407},
    "shuffle": {**COMMON, "encoder_layers_per_block": [0, 1, 2], "decoder_layers_per_block": [0, 2, 2],
                "encoder_qkv_multiscales": [[], [], []], "decoder_qkv_multiscales": [[], [], []],
                "downsample_block_type": "pixel_unshuffle", "upsample_block_type": "pixel_shuffle",
                "decoder_norm_types": ["batch_norm", "batch_norm", "rms_norm"],
                "decoder_act_fns": ["relu", "relu", "silu"], "scaling_factor": 0.3189},
}
BATCH = 2
HEIGHT, WIDTH = 32, 48
# The deepest level of a 4x4-downscaling model sees 2x4 = 8 positions of this,
# which is attention_head_dim: the quadratic path.
SMALL = (8, 16)
SEED = 31


def build(config: dict, root: Path) -> dict[str, np.ndarray]:
    from diffusers import AutoencoderDC

    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    model = AutoencoderDC(**config).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.copy_(0.1 * torch.randn(module.running_mean.shape, generator=generator))
                module.running_var.copy_(torch.rand(module.running_var.shape, generator=generator) + 0.5)
                module.running_mean.copy_(module.running_mean.to(torch.bfloat16).float())
                module.running_var.copy_(module.running_var.to(torch.bfloat16).float())
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
            parameter.copy_(parameter.to(torch.bfloat16).float())
    model.save_pretrained(root, safe_serialization=True)
    draw = Draws(generator)

    def walk(model, dtype) -> dict[str, np.ndarray]:
        named = list(model.named_parameters())
        image = (draw("image", (BATCH, config["in_channels"], HEIGHT, WIDTH), dtype, uniform=True) * 2 - 1
                 ).requires_grad_()
        latent = model.encode(image).latent
        encoded = recorded_vjp("encode", {"probe_latent": latent}, {"image": image}, named, draw)
        code = draw("code", latent.shape, dtype).requires_grad_()
        pixels = model.decode(code).sample
        decoded = recorded_vjp("decode", {"probe": pixels}, {"code": code}, named, draw)
        small = draw("small", (BATCH, config["in_channels"], *SMALL), dtype, uniform=True) * 2 - 1
        with torch.no_grad():
            small_latent = model.encode(small).latent
            small_code = draw("small_code", small_latent.shape, dtype)
            small_pixels = model.decode(small_code).sample
        return {"image": image.detach().numpy(), "latent": latent.detach().numpy(),
                "code": code.detach().numpy(), "pixels": pixels.detach().numpy(),
                "small": small.numpy(), "small_latent": small_latent.numpy(),
                "small_code": small_code.numpy(), "small_pixels": small_pixels.numpy(), **encoded, **decoded}

    arrays = twice(walk, model)
    print(f"{root.name}: parameters {len(list(model.parameters()))}; latent {arrays['latent'].shape}; "
          f"|latent| <= {np.abs(arrays['latent']).max():.3g}; "
          f"|pixels| <= {np.abs(arrays['pixels']).max():.3g}")
    return arrays


def bundle(directory: str, destination: str) -> None:
    """Pack every entry of `directory`, the saved models and the recorded
    arrays, into the xz tarball `destination` for the suite. The diffusers
    reference tools that bundle a whole directory share it."""
    import tarfile

    root = Path(directory)
    with tarfile.open(destination, "w:xz") as archive:
        for path in sorted(root.iterdir()):
            archive.add(path, arcname=path.name)
    print(f"{destination}: {Path(destination).stat().st_size / 1e6:.2f} MB")


def main(destination: str) -> None:
    import diffusers

    require("diffusers")
    torch.set_num_threads(2)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    for name, config in CONFIGS.items():
        arrays = build(config, root / name)
        np.savez_compressed(root / name / "reference.npz", **arrays)
    record = {"diffusers": DIFFUSERS, "configs": CONFIGS, "batch": BATCH, "height": HEIGHT,
              "width": WIDTH, "small": SMALL, "seed": SEED}
    (root / "dc_ae.json").write_text(json.dumps(record, indent=1) + "\n")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"{root}: {size / 1e6:.2f} MB")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "bundle":
        bundle(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dew-dc-ae-reference")
