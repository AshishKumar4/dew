#!/usr/bin/env python3
"""Measure ModelOpt's own NVFP4 arithmetic on a pinned real layer.

Reads config/index/header and 32 q_proj rows by HTTP range, about 100KB of
weight payload. ModelOpt 0.47.0's own NVFP4QTensor dequantizes those files;
its TensorQuantizer fake-quantizes activations under their stored input
scale on the RTX 4080. Results are diagnostic, not loader support: CT's
inverse-global representation must reproduce them before any conversion
could be a parity reference.

Environment (~/.cache/dew/reference-venvs/nvfp4-modelopt) borrows CUDA torch
from agent-engines-vllm and CT 0.17.1 from nvfp4-storage on PYTHONPATH. Run
through dew-gpu-run --short, with an explicit scratch .npz output path.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import requests
import torch

# CUDA torch has already loaded from agent-engines-vllm. Resolve CT from
# the pinned CPU-reference environment, whose torch must not replace it.
sys.path.insert(0, str(Path.home() / ".cache/dew/reference-venvs/nvfp4-storage/lib/python3.12/site-packages"))

from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme, QuantizationStatus
from compressed_tensors.quantization.lifecycle.forward import fake_quantize, forward_quantize
from compressed_tensors.quantization.quant_args import round_to_quantized_type_args
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.qtensor import NVFP4QTensor

REPO = "nvidia/Qwen3-14B-NVFP4"
REVISION = "bc39319a4dc265d9bbb9a9731bc52c4988d9ece7"
STEM = "model.layers.0.self_attn.q_proj"


def fetch(url: str, start: int, end: int) -> bytes:
    response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    response.raise_for_status()
    if response.status_code != 206 or len(response.content) != end - start + 1:
        raise ValueError(f"Hub did not return the requested byte range: {response.status_code}")
    return response.content


def main() -> None:
    out = Path(sys.argv[1])
    base = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
    response = requests.get(base + "model.safetensors.index.json", timeout=120)
    response.raise_for_status()
    index = response.json()["weight_map"]
    url = base + index[STEM + ".weight"]
    size = struct.unpack("<Q", fetch(url, 0, 7))[0]
    header = json.loads(fetch(url, 8, size + 7))
    tensors = {}
    for part in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
        meta = header[STEM + "." + part]
        shape = list(meta["shape"])
        start, end = meta["data_offsets"]
        if len(shape) == 2:
            row = (end - start) // shape[0]
            end, shape[0] = start + 32 * row, 32
        payload = fetch(url, 8 + size + start, 8 + size + end - 1)
        dtype = {"U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32}[meta["dtype"]]
        tensors[part] = torch.frombuffer(bytearray(payload), dtype=dtype).reshape(shape).cuda()
    packed = tensors["weight"]
    width = packed.shape[-1] * 2
    qtensor = NVFP4QTensor(torch.Size((32, width)), torch.bfloat16, packed)
    weight = qtensor.dequantize(scale=tensors["weight_scale"], double_scale=tensors["weight_scale_2"],
                               block_sizes={-1: 16})
    args = QuantizationArgs(num_bits=4, type="float", strategy="tensor_group", group_size=16,
                            symmetric=True, dynamic="local", scale_dtype=torch.float8_e4m3fn)
    scheme = QuantizationScheme(targets=["Linear"], weights=args.model_copy(update={"dynamic": False}),
                                input_activations=args)
    converted = {"weight_packed": packed, "weight_scale": tensors["weight_scale"],
                 "weight_global_scale": (1 / tensors["weight_scale_2"]).reshape(1)}
    ct_weight = NVFP4PackedCompressor.decompress(converted, scheme)["weight"]
    generator = torch.Generator(device="cuda").manual_seed(2043)
    inputs = torch.randn(16, width, generator=generator, device="cuda", dtype=torch.bfloat16)
    inputs[0] = 0
    inputs[1, :16] = 1e-5
    quantizer = TensorQuantizer(QuantizerAttributeConfig(
        num_bits=(2, 1), block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        constant_amax=float(tensors["input_scale"].reshape(())) * 2688)).cuda()
    module = torch.nn.Linear(width, 32, bias=False, device="cuda", dtype=torch.bfloat16)
    module.input_global_scale = torch.nn.Parameter(
        (1 / tensors["input_scale"]).reshape(1), requires_grad=False)
    module.quantization_status = QuantizationStatus.COMPRESSED
    with torch.no_grad():
        own = quantizer(inputs)
        ct = forward_quantize(module, inputs, "input", args)
        ct_fp32 = forward_quantize(module, inputs.float(), "input", args).bfloat16()
        values = inputs.float().unflatten(-1, (-1, 16))
        amax = values.abs().amax(-1)
        g = quantizer.amax.float() / 2688
        local = (amax / (6 * g)).clamp(max=448).to(torch.float8_e4m3fn).float() * g
        same_scale = fake_quantize(inputs.float(), local, None, args).bfloat16()
        safe = torch.where(local >= 1e-5, local, 1)
        guarded = fake_quantize(inputs.float(), safe, None, args).bfloat16()
        reciprocal_quotient = values * (1 / safe)[..., None]
        reciprocal_codes = round_to_quantized_type_args(reciprocal_quotient, args, min=-6, max=6)
        reciprocal = (reciprocal_codes * safe[..., None]).flatten(-2).bfloat16()
        divide_quotient = values / safe[..., None]
        bad = torch.nonzero(own != guarded)
        examples = []
        for row, column in bad[:8].tolist():
            group, element = column // 16, column % 16
            examples.append({"row": row, "column": column, "input": float(inputs[row, column]),
                             "scale": float(safe[row, group]),
                             "quotient_divide": float(divide_quotient[row, group, element]),
                             "quotient_reciprocal": float(reciprocal_quotient[row, group, element]),
                             "modelopt": float(own[row, column]),
                             "torch_divide": float(guarded[row, column])})
    arrays = {name: value.float().cpu().numpy() for name, value in
              {"inputs": inputs, "modelopt_qdq": own, "ct_qdq": ct,
               "modelopt_weight": weight, "ct_weight": ct_weight,
               "ct_fp32_qdq": ct_fp32, "modelopt_scale_qdq": same_scale, "guarded_qdq": guarded,
               "reciprocal_qdq": reciprocal,
               "input_scale": tensors["input_scale"]}.items()}
    np.savez(out, **arrays)
    print(json.dumps({"repo": REPO, "revision": REVISION, "torch": torch.__version__,
                      "device": torch.cuda.get_device_name(),
                      "decoded_weight_mismatches": int((weight != ct_weight).sum()),
                      "qdq_mismatches": int((own != ct).sum()),
                      "ct_fp32_mismatches": int((own != ct_fp32).sum()),
                      "combined_scale_mismatches": int((own != same_scale).sum()),
                      "combined_scale_with_floor_mismatches": int((own != guarded).sum()),
                      "reciprocal_multiply_mismatches": int((own != reciprocal).sum()),
                      "remaining_examples": examples,
                      "max_qdq_difference": float((own - ct).abs().max())}, indent=2))


if __name__ == "__main__":
    main()
