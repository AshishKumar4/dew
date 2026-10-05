#!/usr/bin/env python3
"""Write the U-ViT fixture with Bao et al.'s own model.

baofff/U-ViT's libs/uvit_t2i.py and libs/timm.py, fetched at commit
ce551708dc9cde9818d2af7d84dfadfeb7bd9034 and checked against their SHA-256,
build tiny text-conditioned U-ViTs whose every parameter is moved off its
initialization (and rounded to bfloat16-representable values, so the fixture
compresses). Each is called as U-ViT's continuous-time SDE calls it, on the
time t times 999, and the tool records, against a fixed cotangent, the
gradients of the image, the text states and every parameter: once in float32
and once in float64 (`diffusers_wan_reference.widened`, which widens the
sinusoids' float32 `arange` and `.float()`), the truth tests/reference_error.py
measures both from. Attention runs U-ViT's "math" mode: its "flash" mode
casts q, k and v to float32.

Cases:
- `published`: U-ViT's mscoco configuration, narrowed (mlp_time_embed off, the
  3x3 output convolution on, as its constructor defaults);
- `variant`: the time MLP on, no output convolution, an odd number of heads.

Run with the Dew test environment's torch, on CPU:

    python tools/uvit_reference.py OUTPUT.npz
"""

import hashlib
import importlib.util
import json
import sys
import types
import urllib.request
from pathlib import Path

import ml_dtypes
import numpy as np
import torch

REPO, COMMIT = "baofff/U-ViT", "ce551708dc9cde9818d2af7d84dfadfeb7bd9034"
FILES = {
    "libs/timm.py": "f1e414e5060038baba231ba30959bb1274839ebb1ba7649d3564288731a6babd",
    "libs/uvit_t2i.py": "2537800bd7251ea68f268b9125f5fbbfc64c1680bbc659767f37c737da468894",
}
CACHE = Path.home() / ".cache" / "dew" / "upstream" / "U-ViT" / COMMIT
CASES = {
    "published": {"img_size": 8, "patch_size": 2, "in_chans": 4, "embed_dim": 32, "depth": 4, "num_heads": 4,
                  "mlp_ratio": 4, "qkv_bias": False, "mlp_time_embed": False, "clip_dim": 12,
                  "num_clip_token": 5, "conv": True},
    "variant": {"img_size": 8, "patch_size": 4, "in_chans": 3, "embed_dim": 30, "depth": 2, "num_heads": 3,
                "mlp_ratio": 2, "qkv_bias": False, "mlp_time_embed": True, "clip_dim": 7,
                "num_clip_token": 3, "conv": False},
}
TIMES = (0.25, 0.9)
SEED = 47


def uvit():
    """U-ViT's libs at `COMMIT`, as the package `uvit_libs`, in its math attention."""
    package = types.ModuleType("uvit_libs")
    package.__path__ = []
    sys.modules["uvit_libs"] = package
    modules = {}
    for path, digest in FILES.items():
        local = CACHE / path
        if not local.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}"
            local.write_bytes(urllib.request.urlopen(url).read())
        found = hashlib.sha256(local.read_bytes()).hexdigest()
        if found != digest:
            raise RuntimeError(f"{local} has SHA-256 {found}, not the pinned {digest}")
        name = "uvit_libs." + Path(path).stem
        spec = importlib.util.spec_from_file_location(name, local)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    t2i = modules["uvit_libs.uvit_t2i"]
    t2i.ATTENTION_MODE = "math"
    return t2i


def walk(model, inputs, probe, dtype):
    """One call at `dtype`, and the gradients of `sum(output * probe)` with
    respect to the image, the text states and every parameter."""
    model = model.to(dtype)
    image, context = (inputs[name].to(dtype).requires_grad_() for name in ("image", "context"))
    named = list(model.named_parameters())
    output = model(image, inputs["timesteps"].to(dtype), context)
    grads = torch.autograd.grad((output * probe.to(dtype)).sum(), [image, context] + [p for _, p in named])
    arrays = {"output": output, "grad_image": grads[0], "grad_context": grads[1]}
    for (name, _), gradient in zip(named, grads[2:], strict=True):
        arrays[f"grad_param.{name}"] = gradient
    return {key: value.detach().numpy() for key, value in arrays.items()}


def main():
    from diffusers_wan_reference import widened

    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    torch.set_num_threads(2)
    t2i = uvit()
    arrays: dict[str, np.ndarray] = {}
    for case, config in CASES.items():
        torch.manual_seed(SEED)
        model = t2i.UViT(**config).eval()
        generator = torch.Generator().manual_seed(SEED + 1)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
                parameter.copy_(parameter.to(torch.bfloat16).float())
        times = torch.tensor(TIMES, dtype=torch.float32)
        inputs = {
            "image": torch.randn((len(TIMES), config["in_chans"], config["img_size"], config["img_size"]),
                                 generator=generator),
            "context": torch.randn((len(TIMES), config["num_clip_token"], config["clip_dim"]),
                                   generator=generator),
            "timesteps": times * 999,
        }
        with torch.no_grad():
            shape = model(inputs["image"], inputs["timesteps"], inputs["context"]).shape
        probe = torch.randn(shape, generator=generator)
        record = {"times": times.numpy(), "image": inputs["image"].numpy(),
                  "context": inputs["context"].numpy(), "probe": probe.numpy()}
        record.update({f"param.{name}": value.detach().numpy().astype(ml_dtypes.bfloat16).view(np.uint16)
                       for name, value in model.state_dict().items()})
        single = walk(model, inputs, probe, torch.float32)
        record.update({f"fp32.{key}": value for key, value in single.items()})
        with widened():
            truth = walk(model, inputs, probe, torch.float64)
        record.update({f"fp64.{key}": value for key, value in truth.items()})
        arrays.update({f"{case}/{key}": value for key, value in record.items()})
        gap = np.abs(record["fp32.output"] - record["fp64.output"]).max()
        print(f"{case}: output {tuple(shape)}, fp32 off float64 by {gap:.3g}")
    meta = {"repo": REPO, "commit": COMMIT, "files": FILES, "cases": CASES, "times": TIMES,
            "torch": torch.__version__}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(sys.argv[1], **arrays)
    print(f"{sys.argv[1]}: {Path(sys.argv[1]).stat().st_size / 1e6:.2f} MB, {len(arrays)} arrays")


if __name__ == "__main__":
    main()
