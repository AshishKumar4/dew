"""The actual DC-AE, saved and walked, for tests/fixtures.

Three tiny `AutoencoderDC`s are constructed, each by the diffusers release its record names (`conv`
and `shuffle` 0.34.0's, `plain` 0.39.0's), with the
controls the published checkpoints use, at narrow widths:

- `conv`, as `dc-ae-f32c32-sana-1.1`: strided-convolution downsampling,
  nearest-neighbour upsampling, RMS norms and SiLU throughout, a multiscale
  (5x5) QKV branch in every EfficientViT block, and a residual stack at the
  first level;
- `shuffle`, as `dc-ae-f32c32-in-1.0`: pixel-unshuffle and pixel-shuffle
  resampling, no first level (its input and output convolutions are
  resampling blocks), batch norms and ReLU in the decoder's residual levels,
  and no multiscale branch;
- `plain`, `conv` without either end's channel shortcut and with a SiLU before
  the decoder's last convolution (`encoder_out_shortcut`, `decoder_in_shortcut`,
  `decoder_conv_act_fn`).

Each is saved with `save_pretrained` - its real config and its real
safetensors - and recorded in float32, channel-first, as `vae.encode(x).latent`
and `vae.decode(z).sample` compute them:

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

    python tools/diffusers_dc_ae_reference.py OUTPUT_DIR conv shuffle    # with diffusers 0.34.0
    python tools/diffusers_dc_ae_reference.py OUTPUT_DIR plain           # with the test extra's diffusers
    python tools/diffusers_dc_ae_reference.py bundle OUTPUT_DIR tests/fixtures/dc_ae.tar.xz

Named variants are recorded into OUTPUT_DIR beside what it holds, each by its cohort's release (`COHORTS`),
the record keeping the others'; a run records one cohort's variants.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from diffusers_reference_helpers import on_bfloat16_grid
from reference_releases import require
from reference_vjp import recorded_vjp

if TYPE_CHECKING:
    from diffusers.models.autoencoders.autoencoder_dc import AutoencoderDC

COHORTS = {"conv": "diffusers-0.34", "shuffle": "diffusers-0.34", "plain": "diffusers"}
"""The reference release each variant is recorded by (`reference_releases.RELEASES`): a run records the
variants of the release installed, one cohort at a time."""
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
    # `conv` with neither end's channel shortcut and a SiLU before the decoder's last convolution: the
    # controls AutoencoderDC takes beside the published checkpoints' defaults.
    "plain": {**COMMON, "encoder_layers_per_block": [1, 1, 2], "decoder_layers_per_block": [1, 1, 2],
              "encoder_qkv_multiscales": [[], [], [5]], "decoder_qkv_multiscales": [[], [], [5]],
              "downsample_block_type": "Conv", "upsample_block_type": "interpolate",
              "decoder_norm_types": "rms_norm", "decoder_act_fns": "silu", "scaling_factor": 0.41407,
              "encoder_out_shortcut": False, "decoder_in_shortcut": False, "decoder_conv_act_fn": "silu"},
}
BATCH = 2
HEIGHT, WIDTH = 32, 48
# The deepest level of a 4x4-downscaling model sees 2x4 = 8 positions of this,
# which is attention_head_dim: the quadratic path.
SMALL = (8, 16)
SEED = 31


def encoded(model: AutoencoderDC, image: torch.Tensor) -> torch.Tensor:
    from diffusers.models.autoencoders.vae import EncoderOutput

    output = model.encode(image)
    assert isinstance(output, EncoderOutput)  # return_dict's default.
    return output.latent


def decoded(model: AutoencoderDC, code: torch.Tensor) -> torch.Tensor:
    from diffusers.models.autoencoders.vae import DecoderOutput

    output = model.decode(code)
    assert isinstance(output, DecoderOutput)  # return_dict's default.
    return output.sample


def build(config: dict, root: Path) -> dict[str, np.ndarray]:
    from diffusers.models.autoencoders.autoencoder_dc import AutoencoderDC

    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    model = AutoencoderDC(**config).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                mean, var = module.running_mean, module.running_var
                assert mean is not None and var is not None  # track_running_stats, BatchNorm2d's default.
                mean.copy_(0.1 * torch.randn(mean.shape, generator=generator))
                var.copy_(torch.rand(var.shape, generator=generator) + 0.5)
                mean.copy_(mean.to(torch.bfloat16).float())
                var.copy_(var.to(torch.bfloat16).float())
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
    on_bfloat16_grid(model)
    model.save_pretrained(root, safe_serialization=True)
    named = list(model.named_parameters())

    image = (torch.rand((BATCH, config["in_channels"], HEIGHT, WIDTH), generator=generator) * 2 - 1
             ).requires_grad_()
    latent = encoded(model, image)
    encode = recorded_vjp("encode", {"probe_latent": latent}, {"image": image}, named, generator)

    code = torch.randn(latent.shape, generator=generator).requires_grad_()
    pixels = decoded(model, code)
    decode = recorded_vjp("decode", {"probe": pixels}, {"code": code}, named, generator)

    small = torch.rand((BATCH, config["in_channels"], *SMALL), generator=generator) * 2 - 1
    with torch.no_grad():
        small_latent = encoded(model, small)
        small_code = torch.randn(small_latent.shape, generator=generator)
        small_pixels = decoded(model, small_code)

    arrays = {
        "image": image.detach().numpy(), "latent": latent.detach().numpy(),
        "code": code.detach().numpy(), "pixels": pixels.detach().numpy(),
        "small": small.numpy(), "small_latent": small_latent.numpy(),
        "small_code": small_code.numpy(), "small_pixels": small_pixels.numpy(), **encode, **decode,
    }
    print(f"{root.name}: parameters {len(named)} ({sum(value.numel() for _, value in named)} values); "
          f"latent {tuple(latent.shape)}; |latent| <= {np.abs(arrays['latent']).max():.3g}; "
          f"|pixels| <= {np.abs(arrays['pixels']).max():.3g}")
    return arrays


def bundle(directory: str, destination: str) -> None:
    """Pack every entry of `directory`, the saved models and the recorded
    arrays, into the xz tarball `destination` for the suite. The diffusers
    reference tools that bundle a whole directory share it."""
    from fixture_archive import write

    write(destination, ((path, path.name) for path in sorted(Path(directory).iterdir())))
    print(f"{destination}: {Path(destination).stat().st_size / 1e6:.2f} MB")


def main(destination: str, variants: list[str]) -> None:
    """Record `variants`, one cohort's (`COHORTS`), into `destination`, beside the variants it holds,
    whose releases its record keeps."""
    unknown = sorted(set(variants) - set(CONFIGS))
    if unknown:
        raise ValueError(f"{unknown} are no variants of {sorted(CONFIGS)}")
    if not variants:
        raise ValueError(f"name the variants to record, one cohort's at a time: {COHORTS}")
    cohorts = {COHORTS[name] for name in variants}
    if len(cohorts) != 1:
        raise ValueError(f"the variants asked are recorded by {sorted(cohorts)}: record each cohort's apart, "
                         "in its own environment")
    (cohort,) = cohorts
    release = require(cohort)["diffusers"]
    torch.set_num_threads(2)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    held = root / "dc_ae.json"
    releases = json.loads(held.read_text()).get("releases", {}) if held.exists() else {}
    for name in variants:
        arrays = build(CONFIGS[name], root / name)
        np.savez_compressed(root / name / "reference.npz", allow_pickle=False, **arrays)
        releases[name] = release
    record = {"releases": releases, "configs": CONFIGS, "batch": BATCH, "height": HEIGHT, "width": WIDTH,
              "small": SMALL, "seed": SEED}
    held.write_text(json.dumps(record, indent=1) + "\n")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"{root}: {size / 1e6:.2f} MB")


def seeds(variants: list[str]) -> None:
    """The truths recordings of `variants`, one cohort's (`COHORTS`, `record_walk_truths`): each saved
    model of tests/fixtures/dc_ae.tar.xz reloaded and walked, its encode and decode of a rectangular batch
    and of a small one (tests/autoencoder_walks.py `DC_AE`), every parameter's gradient kept, as
    tests/fixtures/dc_ae_<variant>_seeds.npz. No count of draws resolves a DC-AE walk's float32 errors
    (its ReLU attention's tiny normalizers leave each draw's to one or two positions in both arms, 512
    draws, row 705): each walk holds its function."""
    from autoencoder_walks import DC_AE
    from diffusers.models.autoencoders.autoencoder_dc import AutoencoderDC
    from reference_vjp import FIXTURES, record_walk_truths, save_walks, unpacked

    (cohort,) = {COHORTS[name] for name in variants}
    release = require(cohort)["diffusers"]
    torch.set_num_threads(2)
    calls = {"encode": encoded, "decode": decoded}
    with unpacked(FIXTURES / "dc_ae.tar.xz") as root:
        for name in variants:
            model = AutoencoderDC.from_pretrained(root / name).eval()
            names = [key for key, _ in model.named_parameters()]
            arrays = {}
            for walk, draw in DC_AE[name].items():
                call = calls[walk.split()[-1]]

                def walked(model, x, call=call):
                    return (call(model, x),)

                arrays.update(record_walk_truths(f"{name} {walk}", model, walked, draw, names))
            save_walks(f"dc_ae_{name}_seeds.npz", arrays, [f"{name} {walk}" for walk in DC_AE[name]],
                       reference=f"AutoencoderDC (diffusers {release})", revision=f"diffusers=={release}",
                       command=f"python tools/diffusers_dc_ae_reference.py seeds {name}")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "bundle":
        bundle(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 2 and sys.argv[1] == "seeds":
        seeds(sys.argv[2:])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dew-dc-ae-reference", sys.argv[2:])
