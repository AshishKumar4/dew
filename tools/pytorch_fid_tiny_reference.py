#!/usr/bin/env python3
"""Write the FID extractor fixture with pytorch-fid's own InceptionV3.

pytorch-fid's src/pytorch_fid/inception.py (v0.3.0, commit 0a754fb8, checked
against its SHA-256) builds the FID InceptionV3 over torchvision's
`Inception3` and runs it as FID does: bilinear resize to 299, `2 * x - 1`,
pool3. The tool builds it at a sixteenth of every channel width, the width
of Dew's committed tiny extractor, by handing torchvision's blocks a
`BasicConv2d` whose convolutions and norms divide their widths by 16 (the
three-channel input aside); every block, pool and branch is the upstream's.
The release weights are not fetched: `fid_inception_v3` loads the model's
own initialized state instead, and every weight and running statistic is
then drawn from a seed (rounded to bfloat16-representable values, so the
fixture compresses).

It records the state dict under pytorch-fid's names, 8-bit images (read as
`pixels / 255` in [0, 1]) at a size it upsamples and one it downsamples, and the pool3 features in float32
and float64 (the whole model in double), the truth tests/reference_error.py
measures both from.

Run with the Dew test environment's torch and torchvision, on CPU:

    python tools/pytorch_fid_tiny_reference.py OUTPUT.npz
"""

import hashlib
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models import inception as torchvision_inception

REPO, COMMIT = "mseitzer/pytorch-fid", "0a754fb8e66021700478fd365b79c2eaa316e31b"
PATH = "src/pytorch_fid/inception.py"
DIGEST = "5c5af6e5f71dfdad17d85c3b87b6751c0cc515ca8e389ae90211573e520abf17"
CACHE = Path.home() / ".cache" / "dew" / "upstream" / "pytorch-fid" / COMMIT
DIVISOR = 16
SIZES = (64, 320)
BATCH = 2
SEED = 79


class NarrowConv2d(torchvision_inception.BasicConv2d):
    """torchvision's BasicConv2d at a sixteenth of its widths."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        narrowed = in_channels if in_channels == 3 else in_channels // DIVISOR
        super().__init__(narrowed, out_channels // DIVISOR, **kwargs)


def pytorch_fid():
    """pytorch-fid's inception module at `COMMIT`."""
    local = CACHE / PATH
    if not local.is_file():
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(urllib.request.urlopen(
            f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{PATH}").read())
    found = hashlib.sha256(local.read_bytes()).hexdigest()
    if found != DIGEST:
        raise RuntimeError(f"{local} has SHA-256 {found}, not the pinned {DIGEST}")
    spec = importlib.util.spec_from_file_location("pytorch_fid_inception", local)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build():
    """pytorch-fid's FID InceptionV3, narrowed, every weight drawn from SEED,
    and the torchvision Inception3 its blocks are taken from, whose state
    dict names them as the release does."""
    module = pytorch_fid()
    torchvision_inception.BasicConv2d = NarrowConv2d
    built = []

    def own_state(url, progress=True):
        # `fid_inception_v3` loads what this returns into its `inception`.
        inception = sys._getframe(1).f_locals["inception"]
        built.append(inception)
        return inception.state_dict()

    module.load_state_dict_from_url = own_state
    torch.manual_seed(SEED)
    model = module.InceptionV3(resize_input=True, normalize_input=True, use_fid_inception=True).eval()
    generator = torch.Generator().manual_seed(SEED + 1)
    [inception] = built
    with torch.no_grad():
        for name, value in inception.state_dict().items():
            if name.endswith("num_batches_tracked"):
                continue
            if name.endswith("conv.weight"):
                fan_in = value[0].numel()
                drawn = torch.randn(value.shape, generator=generator) * np.sqrt(2.0 / fan_in)
            elif name.endswith(("bn.weight", "running_var")):
                drawn = 0.5 + torch.rand(value.shape, generator=generator)
            else:
                drawn = 0.25 * torch.randn(value.shape, generator=generator)
            value.copy_(drawn.to(torch.bfloat16).float())
    return model, inception


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    torch.set_num_threads(1)
    model, inception = build()
    rng = np.random.default_rng(SEED)
    arrays: dict[str, np.ndarray] = {}
    for name, value in inception.state_dict().items():
        if not name.endswith("num_batches_tracked") and not name.startswith("fc."):
            arrays[f"state/{name}"] = value.numpy()
    for size in SIZES:
        pixels = rng.integers(0, 256, (BATCH, 3, size, size), dtype=np.uint8)
        arrays[f"{size}/pixels"] = pixels
        images = pixels.astype(np.float32) / 255
        with torch.no_grad():
            [single] = model.float()(torch.from_numpy(images))
            [truth] = model.double()(torch.from_numpy(images).double())
        arrays[f"{size}/fp32.features"] = single.reshape(BATCH, -1).numpy()
        arrays[f"{size}/fp64.features"] = truth.reshape(BATCH, -1).numpy()
        gap = np.abs(arrays[f"{size}/fp32.features"] - arrays[f"{size}/fp64.features"]).max()
        print(f"{size}x{size}: features {single.shape[1]}, fp32 off float64 by {gap:.3g}")
    meta = {"repo": REPO, "commit": COMMIT, "path": PATH, "sha256": DIGEST, "divisor": DIVISOR,
            "torch": torch.__version__, "torchvision": torchvision.__version__}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(sys.argv[1], **arrays)
    print(f"{sys.argv[1]}: {Path(sys.argv[1]).stat().st_size / 1e6:.2f} MB, {len(arrays)} arrays")


if __name__ == "__main__":
    main()
