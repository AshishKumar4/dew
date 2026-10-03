#!/usr/bin/env python3
"""Write the DeepSeek GPU-kernel codec fixture with the releases' own kernels.

Two of DeepSeek's storage rules exist only as GPU kernels, so they run on a
CUDA GPU here, each from its release's inference/kernel.py fetched at the
pinned commit and checked against its SHA-256:

- V3's `weight_dequant` (Triton, deepseek-ai/DeepSeek-V3 at adecc0ef):
  FP8 E4M3 blocks times their float32 `weight_scale_inv`, 128 x 128, partial
  blocks masked, to float32. `v3/<rows>x<cols>/` holds the codes, scales and
  the kernel's output.
- V4's and V4.1's `fp4_act_quant` (TileLang `fp4_quant_kernel`, E8M0 scales,
  DeepSeek-V4-Flash at 60d8d707 and V4.1-Flash at dba1be0a): bfloat16 rows
  in groups of 32 to packed E2M1 pairs and one E8M0 byte per group. `v4/`
  and `v4_1/` hold the bfloat16 input's bits and the kernel's codes and
  scale bytes, over rows that reach its rounding rules: a power-of-two
  group scale exactly at a power, a hair above one, ties between E2M1 values,
  an all-zero group, a negative zero, values past E2M1's 6, and magnitudes
  over 80 binades.

Run on a CUDA GPU in the environment tools/deepseek_v41_reference.py
describes (torch 2.10.0+cu128, tilelang 0.1.8, triton 3.6.0). TileLang
compiles with nvcc, which takes a host gcc of at most 14:

    NVCC_PREPEND_FLAGS="-ccbin /path/to/gcc-14" \\
        ~/.cache/dew/reference-venvs/deepseek-v41/bin/python tools/deepseek_kernels_reference.py
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "codecs" / "deepseek_kernels.npz"
CACHE = Path.home() / ".cache" / "dew" / "upstream"
KERNELS = {
    "v3": ("https://raw.githubusercontent.com/{repo}/{commit}/inference/kernel.py",
           "deepseek-ai/DeepSeek-V3", "adecc0efbe2fda18945734168fce6e0df0d804c3",
           "ffee39fb93658b6ddd783c0932b1445cc4a007a061c0de2a40ac2a3b1563cc00"),
    "v4": ("https://huggingface.co/{repo}/resolve/{commit}/inference/kernel.py",
           "deepseek-ai/DeepSeek-V4-Flash", "60d8d70770c6776ff598c94bb586a859a38244f1",
           "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2"),
    "v4_1": ("https://huggingface.co/{repo}/resolve/{commit}/inference/kernel.py",
             "deepseek-ai/DeepSeek-V4.1-Flash", "dba1be0a40aa45a94ad051997016db3960a90277",
             "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"),
}
V3_SHAPES = ((256, 384), (130, 259), (5, 7))


def kernel(release: str):
    """One release's inference/kernel.py at its pinned commit, checked."""
    url, repo, commit, digest = KERNELS[release]
    local = CACHE / repo / commit / "inference" / "kernel.py"
    if not local.is_file():
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(urllib.request.urlopen(url.format(repo=repo, commit=commit)).read())
    found = hashlib.sha256(local.read_bytes()).hexdigest()
    if found != digest:
        raise RuntimeError(f"{local} has SHA-256 {found}, not the pinned {digest}")
    spec = importlib.util.spec_from_file_location(f"{release}_kernel", local)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fp4_rows(rng) -> np.ndarray:
    """bfloat16 rows of 8 groups of 32 that reach the quantizer's rules."""
    spread = np.exp2(rng.integers(-40, 40, (40, 8))).repeat(32, axis=1)
    rows = rng.standard_normal((40, 256)) * spread
    rows[0, :3] = [6.0, -3.0, 0.75]           # amax 6: scale exactly 1, 0.75 ties to even
    rows[1, :2] = [3.0, -1.4]                 # amax 3: scale exactly 0.5
    rows[2, :2] = [3.015625, 1.0]             # a hair above 3: the scale rounds up
    rows[3, :32] = 0.0                        # an all-zero group
    rows[3, 35] = -0.0
    rows[4, :2] = [0.7495, 6.0]               # reaches the 0.75 tie through bfloat16
    rows[5, :2] = [2.0 ** -120, -(2.0 ** -122)]
    rows[6, :8] = [1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, 4.5]  # every E2M1 midpoint
    rows[7, :2] = [7.0, -6.5]                 # past E2M1's 6 under the group's scale
    return rows.astype(np.float32)


def fp4(release: str, destination: str) -> None:
    """One release's `fp4_act_quant` over `fp4_rows`, in a process of its
    own: TileLang caches a compiled kernel by its name, and both releases
    name theirs `fp4_quant_kernel`."""
    bits = torch.from_numpy(fp4_rows(np.random.default_rng(29))).to(torch.bfloat16)
    packed, scales = kernel(release).fp4_act_quant(bits.cuda(), 32)
    np.savez(destination, input=bits.view(torch.int16).numpy(), codes=packed.view(torch.uint8).cpu().numpy(),
             scales=scales.view(torch.uint8).cpu().numpy())


def main():
    if not torch.cuda.is_available():
        raise SystemExit("the release kernels run on a CUDA GPU")
    if sys.argv[1:2] == ["--fp4"]:
        fp4(*sys.argv[2:4])
        return
    torch.manual_seed(0)
    rng = np.random.default_rng(23)
    arrays: dict[str, np.ndarray] = {}

    v3 = kernel("v3")
    for rows, cols in V3_SHAPES:
        codes = (rng.integers(0, 0x7F, (rows, cols), dtype=np.uint8)
                 | rng.choice(np.uint8([0, 0x80]), (rows, cols)))
        grid = (-(-rows // 128), -(-cols // 128))
        scales = (rng.standard_normal(grid) * np.exp2(rng.integers(-30, 30))).astype(np.float32)
        scales.flat[0] = np.float32(2.0 ** -127)  # subnormal products
        weight = torch.from_numpy(codes).view(torch.float8_e4m3fn).cuda()
        out = v3.weight_dequant(weight, torch.from_numpy(scales).cuda())
        label = f"v3/{rows}x{cols}"
        arrays[f"{label}/codes"], arrays[f"{label}/scales"] = codes, scales
        arrays[f"{label}/dequantized"] = out.float().cpu().numpy()

    with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
        for release in ("v4", "v4_1"):
            destination = Path(directory) / f"{release}.npz"
            subprocess.run([sys.executable, __file__, "--fp4", release, str(destination)], check=True)
            with np.load(destination) as written:
                arrays.update({f"{release}/{part}": written[part] for part in ("input", "codes", "scales")})

    meta = {"kernels": {release: {"repo": repo, "commit": commit, "sha256": digest}
                        for release, (_, repo, commit, digest) in KERNELS.items()},
            "torch": torch.__version__, "device": torch.cuda.get_device_name()}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE, **arrays)
    print(f"{FIXTURE}: {FIXTURE.stat().st_size / 1e6:.2f} MB, {len(arrays)} arrays, {meta['device']}")


if __name__ == "__main__":
    main()
