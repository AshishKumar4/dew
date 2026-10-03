#!/usr/bin/env python3
"""Write the DeepSeek block-FP8 fixture with DeepGEMM's own weight cast.

`per_block_cast_to_fp8` of deepseek-ai/DeepGEMM's deep_gemm/utils/math.py,
fetched at commit 057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7 and checked
against its SHA-256, is the cast that writes DeepSeek's block-scaled FP8
weights. The file imports only torch, so it runs on the CPU, loaded on its
own rather than through the GPU package. The fixture holds the operands and
what the upstream writes for them, both ways (float32 scales, and ue8m0
scales rounded up to powers of two):

- `random/<rows>x<cols>/<block>`: float32 weights spread over 50 binades,
  both signed zeros and values under the amax floor among them, in whole and
  partial blocks;
- `bf16/<rows>x<cols>/<block>`: a bfloat16 weight, stored as its bits;
- `edge`: one-element blocks 448 * 2 ** k * (1 + n * 2 ** -23) for k in
  -22..118 and n in 0, 1, 4, 8, 64, whose scale quotient sits n float32
  ULPs above a power of two;

each with `/codes` (the E4M3FN bytes) and `/scales[/ue8m0]` (float32). Run
from the checkout with any torch that has float8_e4m3fn on the CPU:

    python tools/deepgemm_fp8_reference.py
"""

import hashlib
import importlib.util
import json
import urllib.request
from pathlib import Path

import ml_dtypes
import numpy as np
import torch

REPO = "deepseek-ai/DeepGEMM"
COMMIT = "057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7"
SOURCE = "deep_gemm/utils/math.py"
SHA256 = "9cdecd0c0a0a8b3c64af87107464c91510876110eb80bdb485ff1f74eda8ea0e"
CACHE = Path.home() / ".cache" / "dew" / "upstream" / "DeepGEMM" / COMMIT
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "codecs" / "deepgemm_fp8.npz"
RANDOM = (((256, 384), 128), ((130, 259), 128), ((5, 7), 128), ((48, 32), 16), ((7, 3), 2))


def upstream():
    """DeepGEMM's math.py at `COMMIT`, fetched once and checked."""
    path = CACHE / SOURCE
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{SOURCE}"
        path.write_bytes(urllib.request.urlopen(url).read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"{path} has SHA-256 {digest}, not the pinned {SHA256}")
    spec = importlib.util.spec_from_file_location("deepgemm_math", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spread(shape, seed):
    rng = np.random.default_rng(seed)
    weight = (rng.standard_normal(shape) * np.exp2(rng.integers(-30, 20, shape))).astype(np.float32)
    weight.flat[:4] = np.array([0.0, -0.0, 1e-12, -1e-12], np.float32)
    weight[0, :min(shape[1], 3)] = 0.0
    return weight


def main():
    math = upstream()
    arrays = {}

    def cast(label, weight, block):
        for ue8m0, suffix in ((False, ""), (True, "/ue8m0")):
            codes, scales = math.per_block_cast_to_fp8(weight, ue8m0, gran_k=block)
            arrays[f"{label}/codes{suffix}"] = codes.view(torch.uint8).numpy()
            arrays[f"{label}/scales{suffix}"] = scales.numpy()

    for seed, (shape, block) in enumerate(RANDOM):
        label = f"random/{shape[0]}x{shape[1]}/{block}"
        arrays[f"{label}/weight"] = spread(shape, seed)
        cast(label, torch.from_numpy(arrays[f"{label}/weight"]), block)
    bf16 = spread((130, 259), len(RANDOM)).astype(ml_dtypes.bfloat16)
    arrays["bf16/130x259/128/weight"] = bf16.view(np.uint16)
    bits = torch.from_numpy(bf16.view(np.uint16).astype(np.int16))
    cast("bf16/130x259/128", bits.view(torch.bfloat16), 128)
    ks = np.arange(-22, 119)
    quotients = np.concatenate([np.exp2(ks.astype(np.float32)) * np.float32(1 + ulps * 2.0 ** -23)
                                for ulps in (0, 1, 4, 8, 64)]).astype(np.float32)
    arrays["edge/weight"] = (np.float32(448) * quotients)[:, None]
    cast("edge", torch.from_numpy(arrays["edge/weight"]), 1)
    meta = {"repo": REPO, "commit": COMMIT, "source": SOURCE, "sha256": SHA256, "torch": torch.__version__}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE, **arrays)
    print(f"{FIXTURE}: {FIXTURE.stat().st_size / 1e6:.2f} MB, {len(arrays)} arrays, "
          f"torch {torch.__version__}")


if __name__ == "__main__":
    main()
