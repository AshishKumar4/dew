"""The actual RAE, saved and walked, for tests/fixtures.

Three tiny `AutoencoderRAE`s from diffusers 0.40.0 (with transformers
4.57.1, whose encoder tensor names the published checkpoints carry), one per
frozen encoder the published checkpoints use, at narrow widths, and a tiny
transformers `Dinov2Model` (no registers; REPA's encoder), `dinov2_plain`:

- `dinov2`: DINOv2 with four registers, 14-pixel patches at 112 pixels, whose
  37x37 position table is resized (bicubic, antialiased) to the 8x8 grid; the
  decoder paints 16-pixel patches, so a 128x128 image is shrunk to the
  encoder's input as the published 256 -> 224 is. It carries per-position
  latent standard deviations and no mean, as the published DINOv2 one does.
- `siglip2`: SigLIP, 16-pixel patches at 128 pixels, whose 16x16 position
  table is resized to the grid; per-position latent mean and deviation, as
  the published SigLIP and MAE ones.
- `mae`: ViT-MAE, 16-pixel patches at 128 pixels, whose 14x14 table is
  resized to the grid, and no latent statistics. SigLIP's and MAE's decoders
  paint 8-pixel patches, so a 64x64 image is enlarged to the encoder's input.

Each is saved with `save_pretrained` - its real config and safetensors - and
recorded in float32, channel-first:

- the latent of a fixed batch in [0, 1], and the gradient of
  `sum(latent * probe_latent)` with respect to the pixels;
- the decode of a fixed latent and the gradient of `sum(pixels * probe)`
  with respect to the latent and every decoder parameter.

The encoder is frozen in the source (`requires_grad_(False)`), so it has no
parameter gradient to record. Weights are the source's own initialization
with every norm, bias, layer scale and token moved off its init, rounded to
bfloat16-representable values so the fixture compresses; they are still
float32 tensors.

`published` records what the three published checkpoints compute, at pinned
revisions, for `tests/fixtures/rae_published.npz`: the latent of a smooth
256x256 image (its first 32 channels) and the decode of a fixed standard
normal latent (its top-left 64x64 pixels), in float64, and the patch tokens
`facebook/dinov2-base` gives the image at 224 pixels (its first 32 channels), and how far the
source's own float32 run lands from each. SigLIP's residual stream reaches
several hundred, so float32 rounding alone moves its latent by 2e-5 of the
largest value. Both inputs are built from numpy alone, so a test rebuilds
them without the fixture.

    python tools/diffusers_rae_reference.py OUTPUT_DIR
    python tools/diffusers_rae_reference.py bundle OUTPUT_DIR tests/fixtures/rae.tar.xz
    python tools/diffusers_rae_reference.py published tests/fixtures/rae_published.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

DIFFUSERS, TRANSFORMERS = "0.40.0", "4.57.1"
DECODER = {"decoder_hidden_size": 64, "decoder_num_hidden_layers": 2, "decoder_num_attention_heads": 4,
           "decoder_intermediate_size": 128}
CONFIGS = {
    "dinov2": {**DECODER, "encoder_type": "dinov2", "encoder_hidden_size": 128, "encoder_num_hidden_layers": 2,
               "encoder_patch_size": 14, "encoder_input_size": 112, "patch_size": 16},
    "siglip2": {**DECODER, "encoder_type": "siglip2", "encoder_hidden_size": 64, "encoder_num_hidden_layers": 1,
                "encoder_patch_size": 16, "encoder_input_size": 128, "patch_size": 8,
                "encoder_norm_mean": [0.5, 0.5, 0.5], "encoder_norm_std": [0.5, 0.5, 0.5]},
    "mae": {**DECODER, "encoder_type": "mae", "encoder_hidden_size": 64, "encoder_num_hidden_layers": 1,
            "encoder_patch_size": 16, "encoder_input_size": 128, "patch_size": 8},
}
STATISTICS = {"dinov2": ("std",), "siglip2": ("mean", "std"), "mae": ()}
BATCH = {"dinov2": 1, "siglip2": 2, "mae": 2}
SEED = 41


def build(name: str, root: Path) -> dict[str, np.ndarray]:
    from diffusers import AutoencoderRAE

    config = CONFIGS[name]
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    grid = config["encoder_input_size"] // config["encoder_patch_size"]
    shape = (config["encoder_hidden_size"], grid, grid)
    statistics = {}
    if "mean" in STATISTICS[name]:
        statistics["latents_mean"] = torch.randn(shape, generator=generator).mul(0.5)
    if "std" in STATISTICS[name]:
        statistics["latents_std"] = torch.rand(shape, generator=generator).add(0.5)
    model = AutoencoderRAE(**config, **statistics).eval()
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            if key.endswith(("bias", "lambda1", "_token", "tokens")) or "norm" in key:
                parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
            parameter.copy_(parameter.to(torch.bfloat16).float())
    model.save_pretrained(root, safe_serialization=True)
    decoder = [(key, value) for key, value in model.named_parameters() if key.startswith("decoder.")]
    for _, value in decoder:
        value.requires_grad_(requires_grad=True)

    size = grid * config["patch_size"]
    image = torch.rand((BATCH[name], 3, size, size), generator=generator).requires_grad_()
    latent = model.encode(image).latent
    probe_latent = torch.randn(latent.shape, generator=generator)
    (grad_image,) = torch.autograd.grad((latent * probe_latent).sum(), [image])

    code = torch.randn(latent.shape, generator=generator).requires_grad_()
    pixels = model.decode(code).sample
    probe = torch.randn(pixels.shape, generator=generator)
    decoded = torch.autograd.grad((pixels * probe).sum(), [code] + [value for _, value in decoder])

    arrays = {
        "image": image.detach().numpy(), "latent": latent.detach().numpy(),
        "probe_latent": probe_latent.numpy(), "encode.grad_image": grad_image.numpy(),
        "code": code.detach().numpy(), "pixels": pixels.detach().numpy(), "probe": probe.numpy(),
        "decode.grad_code": decoded[0].numpy(),
    }
    for (key, _), gradient in zip(decoder, decoded[1:], strict=True):
        arrays[f"decode.grad.{key}"] = gradient.numpy()
    print(f"{name}: parameters {sum(value.numel() for value in model.parameters())}; latent {tuple(latent.shape)}; "
          f"pixels {tuple(pixels.shape)}; |latent| <= {np.abs(arrays['latent']).max():.3g}")
    return arrays


PUBLISHED = {
    "dinov2": ("nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08", "eb1744fb3fc4269bb37a804a1afeed1194bc105c"),
    "siglip2": ("nyu-visionx/RAE-siglip2-base-p16-i256-ViTXL-n08", "9ed221bb479c83f610d3b3924f8a4a2f9a763b05"),
    "mae": ("nyu-visionx/RAE-mae-base-p16-ViTXL-n08", "2dc45f481873a1c7159e7b94d7b85c7e33b7f2d9"),
}
DINOV2 = ("facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415")
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
LATENT_CHANNELS, CROP = 32, 64


def smooth_image(size: int = 256) -> np.ndarray:
    """A smooth `[1, 3, size, size]` image in [0, 1] with a little noise."""
    y, x = np.mgrid[0:size, 0:size] / 32.0
    image = np.stack([np.sin(x + y), np.cos(2 * x - y), np.sin(3 * y) * np.cos(x)])[None]
    image = 0.5 + 0.4 * image + 0.05 * np.random.default_rng(0).standard_normal(image.shape)
    return np.clip(image, 0, 1).astype(np.float32)


def normal_latent(channels: int, grid: int) -> np.ndarray:
    """A fixed standard normal `[1, channels, grid, grid]` latent."""
    return np.random.default_rng(1).standard_normal((1, channels, grid, grid)).astype(np.float32)


def from_published(cls, repo: str, revision: str):
    """`cls` built from the checkpoint's config and loaded with its tensors.
    `from_pretrained` fails here: its weight init touches the final layer
    norm's affine parameters, which the source removes. Only the decoder
    table it recomputes is left unread, as `from_pretrained` leaves it."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    model = cls.from_config(cls.load_config(repo, revision=revision)).eval()
    tensors = load_file(hf_hub_download(repo, "diffusion_pytorch_model.safetensors", revision=revision))
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    if missing or unexpected != ["decoder.decoder_pos_embed"]:
        raise RuntimeError(f"{repo}: missing {missing}, unexpected {unexpected}")
    return model


def published(destination: str) -> None:
    from diffusers import AutoencoderRAE

    def walk(model, dtype):
        grid = model.config.encoder_input_size // model.config.encoder_patch_size
        image = torch.from_numpy(smooth_image()).to(dtype)
        latent = torch.from_numpy(normal_latent(model.config.encoder_hidden_size, grid)).to(dtype)
        with torch.no_grad():
            return (model.encode(image).latent.numpy()[:, :LATENT_CHANNELS],
                    model.decode(latent).sample.numpy()[:, :, :CROP, :CROP])

    def gap(actual, exact):
        return np.abs(actual - exact).max() / max(1.0, np.abs(exact).max())

    arrays = {}
    for name, (repo, revision) in PUBLISHED.items():
        model = from_published(AutoencoderRAE, repo, revision)
        single = walk(model, torch.float32)
        exact = walk(model.double(), torch.float64)
        for key, value, rounded in zip(("latent", "pixels"), exact, single, strict=True):
            arrays[f"{name}.{key}"] = value
            arrays[f"{name}.{key}.float32_gap"] = np.float64(gap(rounded, value))
        print(name, {key: float(arrays[f"{name}.{key}.float32_gap"]) for key in ("latent", "pixels")})
        del model
    from transformers import Dinov2Model

    model = Dinov2Model.from_pretrained(DINOV2[0], revision=DINOV2[1]).eval()
    mean, std = (np.asarray(value, np.float32).reshape(1, 3, 1, 1) for value in IMAGENET)
    pixels = torch.from_numpy((smooth_image(224) - mean) / std)
    with torch.no_grad():
        single = model(pixels).last_hidden_state[:, 1:, :LATENT_CHANNELS].numpy()
        exact = model.double()(pixels.double()).last_hidden_state[:, 1:, :LATENT_CHANNELS].numpy()
    arrays["dinov2_plain.tokens"] = exact
    arrays["dinov2_plain.tokens.float32_gap"] = np.float64(gap(single, exact))
    print("dinov2_plain", float(arrays["dinov2_plain.tokens.float32_gap"]))
    np.savez_compressed(destination, **arrays)
    print(f"{destination}: {Path(destination).stat().st_size / 1e6:.2f} MB")


PLAIN = {"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 1, "image_size": 518, "patch_size": 14}
PLAIN_INPUT = 112


def build_plain(root: Path) -> dict[str, np.ndarray]:
    """A tiny transformers `Dinov2Model`, REPA's encoder, saved, and its
    patch tokens for a normalized 112-pixel batch (the table resized to the
    8x8 grid) with the gradient of a probe against the pixels."""
    from transformers import Dinov2Config, Dinov2Model

    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    model = Dinov2Model(Dinov2Config(**PLAIN)).eval()
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            if key.endswith(("bias", "lambda1", "token")) or "norm" in key:
                parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
            parameter.copy_(parameter.to(torch.bfloat16).float())
    model.save_pretrained(root, safe_serialization=True)
    pixels = torch.randn((1, 3, PLAIN_INPUT, PLAIN_INPUT), generator=generator).requires_grad_()
    tokens = model(pixels).last_hidden_state[:, 1:]
    probe = torch.randn(tokens.shape, generator=generator)
    (grad,) = torch.autograd.grad((tokens * probe).sum(), [pixels])
    return {"pixels": pixels.detach().numpy(), "tokens": tokens.detach().numpy(), "probe": probe.numpy(),
            "grad_pixels": grad.numpy()}


def bundle(directory: str, destination: str) -> None:
    """Pack the saved autoencoders and the recorded arrays for the suite."""
    import tarfile

    root = Path(directory)
    with tarfile.open(destination, "w:xz") as archive:
        for path in sorted(root.iterdir()):
            archive.add(path, arcname=path.name)
    print(f"{destination}: {Path(destination).stat().st_size / 1e6:.2f} MB")


def main(destination: str) -> None:
    import diffusers
    import transformers

    if (diffusers.__version__, transformers.__version__) != (DIFFUSERS, TRANSFORMERS):
        raise RuntimeError(f"recorded against diffusers {DIFFUSERS} and transformers {TRANSFORMERS}, "
                           f"not {diffusers.__version__} and {transformers.__version__}")
    torch.set_num_threads(2)
    root = Path(destination)
    for name in CONFIGS:
        (root / name).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / name / "reference.npz", **build(name, root / name))
    (root / "dinov2_plain").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "dinov2_plain" / "reference.npz", **build_plain(root / "dinov2_plain"))
    record = {"diffusers": DIFFUSERS, "transformers": TRANSFORMERS, "configs": CONFIGS, "statistics": STATISTICS,
              "batch": BATCH, "plain": PLAIN, "plain_input": PLAIN_INPUT, "seed": SEED}
    (root / "rae.json").write_text(json.dumps(record, indent=1) + "\n")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"{root}: {size / 1e6:.2f} MB")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "bundle":
        bundle(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 2 and sys.argv[1] == "published":
        published(sys.argv[2])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dew-rae-reference")
