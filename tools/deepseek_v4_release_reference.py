#!/usr/bin/env python3
"""Write the DeepSeek-V4 / V4.1 storage fixture with the releases' own reading code.

The releases ship the code that reads their `.scale` storage on the CPU:

- inference/convert.py's `main`, which dequantizes every attention `wo_a`
  FP8 weight against its block scales to bfloat16 (E8M0 bytes 0 and 254
  among the scales, so subnormal and overflowing products included) and, under
  `--expert-dtype fp8`, turns every FP4 routed expert into FP8 with
  `cast_e2m1fn_to_e4m3fn`. Fetched at DeepSeek-V4-Flash 60d8d707 (128 x 128
  blocks) and DeepSeek-V4.1-Flash dba1be0a (32 x 32 blocks), SHA-256
  checked, and run on a tiny checkpoint in the release's own layout: one
  wo_a under E8M0 scales and one under float32 ones (the Base releases),
  and one FP4 expert whose scales stay within the 2 ** 6 span each FP8
  block can carry, so the conversion is lossless. The expert's dense
  values are torch's reading of the FP8 and E8M0 it wrote, block by block.
- V4.1's inference/model.py `ParallelEngramEmbedding.forward`, the
  n-gram table's lookup, which dequantizes each row against one scale per
  32 values and returns bfloat16. model.py imports the release's TileLang
  kernels and its vision and engram modules: engram.py, vision.py and
  image_processor.py are fetched at the same commit, and `kernel` is a
  module whose functions raise if called, which the lookup never does.

The fixture holds the stored bytes each was given and what it returned,
under `v4/` and `v4_1/`. Run from the checkout with torch 2.14 on the CPU:

    python tools/deepseek_v4_release_reference.py
"""

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "codecs" / "deepseek_v4_release.npz"
CACHE = Path.home() / ".cache" / "dew" / "upstream"
RELEASES = {
    "v4": ("deepseek-ai/DeepSeek-V4-Flash", "60d8d70770c6776ff598c94bb586a859a38244f1", 128),
    "v4_1": ("deepseek-ai/DeepSeek-V4.1-Flash", "dba1be0a40aa45a94ad051997016db3960a90277", 32),
}
SHA256 = {
    ("v4", "inference/convert.py"): "912acfc20bdd9ae4dbd5bde9dc7c8e61f6d27b6826d3ac2d052b2534c0881454",
    ("v4_1", "inference/convert.py"): "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76",
    ("v4_1", "inference/model.py"): "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
    ("v4_1", "inference/engram.py"): "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
    ("v4_1", "inference/vision.py"): "5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c",
    ("v4_1", "inference/image_processor.py"):
        "482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272",
}
KERNELS = ("act_quant", "fp4_act_quant", "fp4_gemm", "fp8_gemm", "hc_split_sinkhorn", "sparse_attn")


def fetched(release: str, path: str) -> Path:
    """One release file at its pinned commit, checked against its SHA-256."""
    repo, commit, _ = RELEASES[release]
    local = CACHE / repo / commit / path
    if not local.is_file():
        local.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{repo}/resolve/{commit}/{path}"
        local.write_bytes(urllib.request.urlopen(url).read())
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    if digest != SHA256[release, path]:
        raise RuntimeError(f"{local} has SHA-256 {digest}, not the pinned {SHA256[release, path]!r}")
    return local


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def e4m3(rng, shape) -> torch.Tensor:
    """Random E4M3FN codes, both NaN codes left out."""
    codes = rng.integers(0, 0x7F, shape, dtype=np.uint8) | rng.choice(np.uint8([0, 0x80]), shape)
    return torch.from_numpy(codes).view(torch.float8_e4m3fn)


def e8m0(exponents) -> torch.Tensor:
    return torch.from_numpy(np.asarray(exponents, np.uint8)).view(torch.float8_e8m0fnu)


def raw(tensor: torch.Tensor) -> np.ndarray:
    """A tensor's bytes as a NumPy array of the same width."""
    width = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[tensor.element_size()]
    return tensor.contiguous().view(width).numpy().copy()


def convert(release: str, rng, arrays: dict) -> None:
    """The release's convert.py `main` over a one-layer, one-expert checkpoint."""
    _, _, block = RELEASES[release]
    module = load(f"{release}_convert", fetched(release, "inference/convert.py"))
    out, inputs = 2 * block, 3 * block
    # E8M0 bytes 0 and 254 take the products to float32 subnormals and past 448 * 2 ** 127.
    exponents = rng.integers(100, 150, (2, 3))
    exponents.flat[:2] = 0, 254
    stored = {
        "model.layers.0.self_attn.wo_a.weight": e4m3(rng, (out, inputs)),
        "model.layers.0.self_attn.wo_a.weight_scale_inv": e8m0(exponents),
        "model.layers.1.self_attn.wo_a.weight": e4m3(rng, (out, inputs)),
        "model.layers.1.self_attn.wo_a.weight_scale_inv": torch.exp2(
            torch.from_numpy(rng.integers(-27, 23, (2, 3)).astype(np.float32))),
        "model.layers.0.mlp.experts.0.w1.weight": torch.from_numpy(
            rng.integers(-128, 128, (out, inputs // 2), dtype=np.int8)),
    }
    # The scales of each FP8 block span 2 ** 6 at most, all a 32 x 32 block
    # (or 128 x 128) can carry, so the release's cast to FP8 is exact.
    base = rng.integers(115, 135, (out // block, inputs // block))
    spread = rng.integers(0, 7, (out, inputs // 32))
    stored["model.layers.0.mlp.experts.0.w1.weight_scale_inv"] = e8m0(
        np.repeat(np.repeat(base, block, axis=0), block // 32, axis=1) + spread)
    with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
        source, target = Path(directory) / "source", Path(directory) / "target"
        source.mkdir()
        save_file(stored, source / "model.safetensors")
        if release == "v4":
            module.main(str(source), str(target), 1, 1, "fp8")
        else:
            module.main(str(source), str(target), 1, "fp8")
        written = load_file(target / "model0-mp1.safetensors")
    prefix = f"{release}/"
    for name, tensor in stored.items():
        arrays[prefix + "stored/" + name] = raw(tensor)
    for layer in (0, 1):
        arrays[prefix + f"wo_a/{layer}"] = raw(written[f"layers.{layer}.attn.wo_a.weight"])
    fp8 = written["layers.0.ffn.experts.0.w1.weight"]
    scale = written["layers.0.ffn.experts.0.w1.scale"]
    blocks = scale.float().repeat_interleave(block, 0).repeat_interleave(block, 1)
    arrays[prefix + "expert/fp8"] = raw(fp8)
    arrays[prefix + "expert/scale"] = raw(scale)
    arrays[prefix + "expert/dense"] = (fp8.float() * blocks).numpy()


def engram(rng, arrays: dict) -> None:
    """V4.1's `ParallelEngramEmbedding.forward` over a 9-row table."""
    kernel = types.ModuleType("kernel")
    for name in KERNELS:
        def refuse(*args, _name=name, **kwargs):
            raise RuntimeError(f"the engram lookup called the {_name} kernel")
        setattr(kernel, name, refuse)
    sys.modules["kernel"] = kernel
    for name in ("engram", "vision", "image_processor"):
        load(name, fetched("v4_1", f"inference/{name}.py"))
    model = load("v4_1_model", fetched("v4_1", "inference/model.py"))
    table = model.ParallelEngramEmbedding(9, 256)
    weight, scale = e4m3(rng, (9, 256)), rng.integers(110, 140, (9, 8))
    scale.flat[:2] = 0, 254
    table.weight.data, table.scale.data = weight, e8m0(scale)
    indices = torch.tensor([[0, 3, 8, 1], [5, 5, 2, 7]])
    with torch.no_grad():
        values = table(indices)
    arrays["v4_1/engram/weight"], arrays["v4_1/engram/scale"] = raw(weight), raw(e8m0(scale))
    arrays["v4_1/engram/indices"], arrays["v4_1/engram/values"] = indices.numpy(), raw(values)


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(17)
    arrays: dict[str, np.ndarray] = {}
    for release in RELEASES:
        convert(release, rng, arrays)
    engram(rng, arrays)
    meta = {"releases": {release: {"repo": repo, "commit": commit, "block": block}
                         for release, (repo, commit, block) in RELEASES.items()},
            "sha256": {f"{release}/{path}": digest for (release, path), digest in SHA256.items()},
            "torch": torch.__version__}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE, **arrays)
    print(f"{FIXTURE}: {FIXTURE.stat().st_size / 1e6:.2f} MB, {len(arrays)} arrays, "
          f"torch {torch.__version__}")


if __name__ == "__main__":
    main()
