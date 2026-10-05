#!/usr/bin/env python3
"""Write the AWQ and GPTQ fixture with AutoAWQ's and gptqmodel's own packing.

GPTQ is gptqmodel 7.5.0, imported: `PackableQuantLinear.pack_block` packs a
Linear against given scales, zero points and g_idx into its internal (v2)
layout; `convert_gptq_v2_to_v1_format_module` writes a v1 ('gptq')
checkpoint's zeros from it, and `convert_gptq_v1_to_v2_format_module` reads
them back; `dequantize_weight` is `PackableQuantLinear`'s unpacking. The fixture
records `pack_block`'s Python path; the tool refuses to write it unless the
native pack extension packs the same words and `TorchLinear`'s eval-time
dequantize gives the same weight.

AWQ is AutoAWQ 0.2.9 (casper-hansen/AutoAWQ at tag v0.2.9, 88e4c76b),
whose package imports CUDA kernels and the model zoo on import: its four
files the gemm path runs (utils/module.py, utils/utils.py,
utils/packing_utils.py, modules/linear/gemm.py) are fetched at that commit,
checked against their SHA-256, and loaded under empty `awq` package
modules. `WQLinear_GEMM.from_linear` packs and `dequantize_gemm` unpacks.

The fixture holds the operands each library was given and what it wrote:

- `gptq/<bits>/<sym|asym>/<groups|actorder>/`: `weight` [out, in] float32
  near the grid, `grid_scales` and `grid_zeros` [groups, out], `g_idx`
  [in]; `qweight`, `qzeros` (v2), `qzeros_v1`, `scales` (fp16), and
  `dequantized` [in, out] (fp16 values as float32) from both layouts;
- `awq/<group>/`: the same for AutoAWQ's gemm layout, 4 bits;
- `checkpoint/<awq|gptq>/<tensor>`: qwen3-tiny's projections quantized
  per group of 16 inputs to min/max grids (the operands) and packed by each
  library, the GPTQ zeros in the v1 checkpoint format.

Environment (~/.cache/dew/reference-venvs/int-quant): torch 2.14.0+cpu,
gptqmodel 7.5.0, accelerate 1.15.0, safetensors 0.8.0, and a C++ compiler
for gptqmodel's pack extension. Run from the checkout:

    ~/.cache/dew/reference-venvs/int-quant/bin/python tools/integer_codecs_reference.py
"""

import copy
import hashlib
import importlib.util
import json
import os
import re
import sys
import types
import urllib.request
from pathlib import Path

import gptqmodel
import numpy as np
import torch
from gptqmodel.nn_modules.qlinear import PackableQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.model import convert_gptq_v1_to_v2_format_module, convert_gptq_v2_to_v1_format_module
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "codecs" / "integer.npz"
GPTQMODEL = "7.5.0"
AWQ_REPO, AWQ_COMMIT = "casper-hansen/AutoAWQ", "88e4c76b20755db275574e6a03c83c84ba3bece5"
AWQ_FILES = {
    "awq/utils/module.py": "5268e366ff3e910267d0ee0b25db7a7792b3b86fcd72bec11dba7fd4aa5d3b01",
    "awq/utils/utils.py": "a6a05a2058d5e530bb0448f1bd5a4e1c3da5b0b0a2aa3981cd0c8ad11ea82069",
    "awq/utils/packing_utils.py": "65eab3eabe3f55e300ffbab5feac59c49322d985f42dcda4e2288859fb9a4abe",
    "awq/modules/linear/gemm.py": "a3ff4f116e01cfaa905440af4606c61089c788bf0dad7442314e0ad074843cf2",
}
CACHE = Path.home() / ".cache" / "dew" / "upstream" / "AutoAWQ" / AWQ_COMMIT
PROJECTION = re.compile(r"(q|k|v|o|gate|up|down)_proj\.weight$")


def autoawq():
    """AutoAWQ's gemm modules at `AWQ_COMMIT`, under empty package modules."""
    for package in ("awq", "awq.utils", "awq.modules", "awq.modules.linear"):
        module = types.ModuleType(package)
        module.__path__ = []
        sys.modules[package] = module
    loaded = {}
    for path, digest in AWQ_FILES.items():
        local = CACHE / path
        if not local.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/{AWQ_REPO}/{AWQ_COMMIT}/{path}"
            local.write_bytes(urllib.request.urlopen(url).read())
        found = hashlib.sha256(local.read_bytes()).hexdigest()
        if found != digest:
            raise RuntimeError(f"{local} has SHA-256 {found}, not the pinned {digest}")
        name = path.removesuffix(".py").replace("/", ".")
        spec = importlib.util.spec_from_file_location(name, local)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    gemm = loaded["awq.modules.linear.gemm"]
    if gemm.awq_ext is not None or gemm.TRITON_AVAILABLE:
        raise RuntimeError("AutoAWQ found its CUDA or triton kernels; the fixture records its torch "
                           "path only")
    return loaded["awq.utils.packing_utils"], gemm


def grid(rng, out: int, groups: int, bits: int, *, sym: bool):
    """fp16-representable scales and integer zeros [groups, out]; asymmetric
    zeros span the whole code range, the ends included."""
    scales = np.exp2(rng.uniform(-8, -2, (groups, out))).astype(np.float16).astype(np.float32)
    maxq = (1 << bits) - 1
    if sym:
        zeros = np.full((groups, out), (maxq + 1) // 2, np.int64)
    else:
        zeros = rng.integers(0, maxq + 1, (groups, out))
        zeros[0, :2] = (0, maxq)
    return scales, zeros


def near_grid(rng, scales, zeros, g_idx, bits):
    """A [out, in] weight within 0.45 of a step of a code in range, so the
    rounding is never a tie and nothing saturates."""
    codes = rng.integers(0, 1 << bits, (len(g_idx), scales.shape[1]))
    step = scales[g_idx]
    weight = (codes - zeros[g_idx]) * step + rng.uniform(-0.45, 0.45, codes.shape) * step
    return weight.astype(np.float32).T.copy()


def gptq_packed(weight, scales, zeros, g_idx, bits, group, *, sym: bool, act: bool, extension: bool):
    """A TorchLinear packed by gptqmodel's `pack_block`, through its Python
    path or its native extension."""
    out, in_features = weight.shape
    module = TorchLinear(bits=bits, group_size=group, sym=sym, desc_act=act, in_features=in_features,
                         out_features=out, bias=False, enable_wf_unsqueeze=True)
    linear = torch.nn.Linear(in_features, out, bias=False)
    linear.weight.data = torch.from_numpy(weight)
    os.environ["GPTQMODEL_DISABLE_PACK_EXT" if not extension else "GPTQMODEL_FORCE_PACK_EXT"] = "1"
    os.environ.pop("GPTQMODEL_FORCE_PACK_EXT" if not extension else "GPTQMODEL_DISABLE_PACK_EXT", None)
    module.pack_block(linear, torch.from_numpy(scales.T.copy()), torch.from_numpy(zeros.T.copy()),
                      torch.from_numpy(g_idx.astype(np.int32)))
    os.environ["GPTQMODEL_DISABLE_PACK_EXT"] = "1"
    os.environ.pop("GPTQMODEL_FORCE_PACK_EXT", None)
    return module


def gptq_pack(weight, scales, zeros, g_idx, bits, group, *, sym: bool, act: bool):
    """gptqmodel's packing of one Linear, its v1 zeros, and both dequantizations."""
    module = gptq_packed(weight, scales, zeros, g_idx, bits, group, sym=sym, act=act, extension=False)
    native = gptq_packed(weight, scales, zeros, g_idx, bits, group, sym=sym, act=act, extension=True)
    if not (torch.equal(native.qweight, module.qweight) and torch.equal(native.qzeros, module.qzeros)):
        raise RuntimeError("pack_block's native extension parts from its Python path")
    record = {"qweight": module.qweight.numpy().copy(), "qzeros": module.qzeros.numpy().copy(),
              "scales": module.scales.numpy().copy(), "g_idx": module.g_idx.numpy().copy()}
    record["dequantized"] = PackableQuantLinear.dequantize_weight(module).float().numpy()
    cached = TorchLinear.dequantize_weight(module.eval()).float().numpy()
    if not np.array_equal(cached, record["dequantized"]):
        raise RuntimeError("TorchLinear's eval dequantize parts from PackableQuantLinear's")
    v1 = copy.deepcopy(module)
    config = QuantizeConfig(bits=bits, group_size=group, sym=sym, desc_act=act, offload_to_disk=False)
    convert_gptq_v2_to_v1_format_module(v1, config)
    record["qzeros_v1"] = v1.qzeros.numpy().copy()
    convert_gptq_v1_to_v2_format_module(v1, bits=bits, pack_dtype=torch.int32)
    record["dequantized_v1"] = PackableQuantLinear.dequantize_weight(v1).float().numpy()
    return record


def awq_pack(gemm, packing, weight, scales, zeros, group):
    """AutoAWQ's gemm packing of one Linear and its dequantization."""
    out, in_features = weight.shape
    linear = torch.nn.Linear(in_features, out, bias=False)
    linear.weight.data = torch.from_numpy(weight)
    module = gemm.WQLinear_GEMM.from_linear(linear, 4, group, scales=torch.from_numpy(scales),
                                            zeros=torch.from_numpy(zeros.astype(np.float32)))
    record = {"qweight": module.qweight.numpy().copy(), "qzeros": module.qzeros.numpy().copy(),
              "scales": module.scales.numpy().copy()}
    record["dequantized"] = packing.dequantize_gemm(module.qweight, module.qzeros, module.scales, 4,
                                                    group).float().numpy()
    return record


def min_max_grid(weight, group: int = 16, bits: int = 4):
    """A projection's per-group min/max grid and the weight on it: scales
    [groups, out] fp16, zeros [groups, out], and (code - zero) * scale."""
    w = weight.astype(np.float32).T  # [in, out]
    grouped = w.reshape(-1, group, w.shape[1])
    low, high = grouped.min(axis=1), grouped.max(axis=1)
    maxq = (1 << bits) - 1
    scales = ((high - low) / maxq).astype(np.float16).astype(np.float32)
    zeros = np.clip(np.round(-low / scales), 0, maxq).astype(np.int64)
    codes = np.clip(np.round(grouped / scales[:, None] + zeros[:, None]), 0, maxq)
    on_grid = ((codes - zeros[:, None]) * scales[:, None]).reshape(w.shape)
    return on_grid.astype(np.float32).T.copy(), scales, zeros


def main():
    if gptqmodel.__version__ != GPTQMODEL:
        raise RuntimeError(f"recorded against gptqmodel {GPTQMODEL}, not {gptqmodel.__version__}")
    torch.set_num_threads(2)
    packing, gemm = autoawq()
    rng = np.random.default_rng(5)
    arrays: dict[str, np.ndarray] = {}

    def keep(prefix, record):
        arrays.update({f"{prefix}/{part}": value for part, value in record.items()})

    in_features, out, group = 128, 64, 32
    for bits in (2, 4, 8):
        for sym in (False, True):
            for act in (False, True):
                label = f"gptq/{bits}/{'sym' if sym else 'asym'}/{'actorder' if act else 'groups'}"
                g_idx = np.arange(in_features) // group
                if act:
                    g_idx = rng.permutation(g_idx)
                scales, zeros = grid(rng, out, in_features // group, bits, sym=sym)
                weight = near_grid(rng, scales, zeros, g_idx, bits)
                record = gptq_pack(weight, scales, zeros, g_idx, bits, group, sym=sym, act=act)
                keep(label, {"weight": weight, "grid_scales": scales, "grid_zeros": zeros, **record})
    for group in (16, 32, 128):
        scales, zeros = grid(rng, out, in_features // group, 4, sym=False)
        g_idx = np.arange(in_features) // group
        weight = near_grid(rng, scales, zeros, g_idx, 4)
        keep(f"awq/{group}", {"weight": weight, "grid_scales": scales, "grid_zeros": zeros,
                              **awq_pack(gemm, packing, weight, scales, zeros, group)})

    dense = load_file(ROOT / "tests" / "fixtures" / "hf" / "qwen3-tiny" / "model.safetensors")
    for name, weight in sorted(dense.items()):
        if not PROJECTION.search(name):
            continue
        on_grid, scales, zeros = min_max_grid(weight)
        stem = name.removesuffix(".weight")
        awq = awq_pack(gemm, packing, on_grid, scales, zeros, 16)
        for part in ("qweight", "qzeros", "scales"):
            arrays[f"checkpoint/awq/{stem}.{part}"] = awq[part]
        g_idx = np.arange(on_grid.shape[1]) // 16
        gptq = gptq_pack(on_grid, scales, zeros, g_idx, 4, 16, sym=False, act=False)
        arrays.update({f"checkpoint/gptq/{stem}.qweight": gptq["qweight"],
                       f"checkpoint/gptq/{stem}.qzeros": gptq["qzeros_v1"],
                       f"checkpoint/gptq/{stem}.scales": gptq["scales"],
                       f"checkpoint/gptq/{stem}.g_idx": gptq["g_idx"]})

    meta = {"gptqmodel": gptqmodel.__version__, "autoawq": {"repo": AWQ_REPO, "commit": AWQ_COMMIT,
                                                            "sha256": AWQ_FILES},
            "torch": torch.__version__}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE, **arrays)
    print(f"{FIXTURE}: {FIXTURE.stat().st_size / 1e6:.2f} MB, {len(arrays)} arrays")


if __name__ == "__main__":
    main()
