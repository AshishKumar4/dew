#!/usr/bin/env python3
"""Write the 16-channel VAE fixtures tests/test_vae_16ch.py checks against.

Everything here runs under torch and diffusers, which dew does not depend on,
so this is the only place the reference autoencoder is executed. The fixtures
it writes are what CI compares against.

Set up the venv once (torch CPU, diffusers, safetensors):

    uv venv /tmp/vaeref --python 3.12
    uv pip install --python /tmp/vaeref/bin/python torch diffusers \
        safetensors numpy \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple
    /tmp/vaeref/bin/python tools/vae_reference.py

--out DIR writes the fixtures somewhere other than tests/fixtures/vae.

What lands in tests/fixtures/vae:

- sd3-tiny/: a random-weight 16-channel AutoencoderKL in the diffusers layout
  (config.json + diffusion_pytorch_model.safetensors), shaped like the SD3.5
  and Flux VAE where it matters: `latent_channels` 16, `use_quant_conv` and
  `use_post_quant_conv` false, and the SD3 `shift_factor`/`scaling_factor`.
  Two stages instead of four, so it is a few megabytes rather than 320. The
  reference's fp32 encode (the posterior mean) and decode of one committed
  image, so the parity test needs no network.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "vae"

# The SD3/Flux latent space in miniature: 16 channels, no quant convs, the
# shift and scale of the real config. Built through `from_config`, the path a
# loaded config.json takes, because the constructor declares its block lists
# as one-element tuples.
TINY_CONFIG: dict[str, object] = dict(
    in_channels=3, out_channels=3, latent_channels=16,
    down_block_types=["DownEncoderBlock2D"] * 2,
    up_block_types=["UpDecoderBlock2D"] * 2,
    block_out_channels=[8, 16], layers_per_block=1, norm_num_groups=4,
    act_fn="silu", scaling_factor=1.5305, shift_factor=0.0609,
    use_quant_conv=False, use_post_quant_conv=False, sample_size=32,
)
IMAGE = {"seed": 0, "shape": [1, 16, 16, 3]}


def synthetic_image(recipe) -> np.ndarray:
    """The uint8 image the fixture was run on, reproduced the same way by the
    test."""
    return np.random.RandomState(recipe["seed"]).randint(
        0, 256, tuple(recipe["shape"]), dtype=np.uint8)


def tiny_model() -> AutoencoderKL:
    torch.manual_seed(0)
    model = AutoencoderKL.from_config(TINY_CONFIG)
    if not isinstance(model, AutoencoderKL):
        raise TypeError(f"from_config returned {type(model).__name__}, not the model")
    model.eval()
    return model


def write_tiny(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model = tiny_model()

    pixels = synthetic_image(IMAGE)
    # The objective's [-1, 1] scale, NHWC, as dew hands it to an autoencoder.
    sample = (pixels.astype(np.float32) - 127.5) / 127.5
    torch_sample = torch.tensor(sample).permute(0, 3, 1, 2)
    with torch.no_grad():
        encoded = model.encode(torch_sample)
        if not isinstance(encoded, AutoencoderKLOutput):
            raise TypeError(f"encode returned {type(encoded).__name__}, not its output")
        latent = encoded.latent_dist.mean
        decoded = model.decode(latent)
        if not isinstance(decoded, DecoderOutput):
            raise TypeError(f"decode returned {type(decoded).__name__}, not its output")
    # Back to NHWC, and the scaled latent the diffusion model would see.
    latent_nhwc = latent.permute(0, 2, 3, 1).numpy()
    decoded_nhwc = decoded.sample.permute(0, 2, 3, 1).numpy()
    mse = float(np.mean((decoded_nhwc - sample) ** 2))
    print(f"latent {latent_nhwc.shape}, decoded {decoded_nhwc.shape}, "
          f"reference round trip mse {mse:.4f}, psnr {10 * np.log10(4.0 / mse):.2f} dB")

    from safetensors.torch import save_file

    model.save_config(directory)
    save_file({name: tensor.clone() for name, tensor in model.state_dict().items()},
              directory / "diffusion_pytorch_model.safetensors")
    (directory / "inputs.json").write_text(json.dumps({"image": IMAGE}, indent=2) + "\n")
    np.savez(directory / "reference.npz", sample=sample,
             latent=latent_nhwc, decoded=decoded_nhwc)
    size = sum(p.stat().st_size for p in directory.iterdir()) / 2 ** 20
    print(f"wrote {directory} ({size:.1f} MB)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=FIXTURES)
    write_tiny(parser.parse_args(argv).out / "sd3-tiny")


if __name__ == "__main__":
    main()
