#!/usr/bin/env python3
"""Write the compressed-tensors fixture with compressed-tensors' own compressors.

For each weight scheme below, a fresh [48, 96] bfloat16 weight is quantized
the way a checkpoint's Linear is: `calculate_qparams` over the min and max of
each tensor, channel, group or block, then the format's compressor
(`PackedQuantizationCompressor`, `FloatQuantizationCompressor`,
`IntQuantizationCompressor`). The fixture holds, per scheme:

- `<scheme>/<part>`: the stored tensors the compressor writes;
- `<scheme>/dequantized`: the library's `decompress` of them, float32;
- `<scheme>/trained` and `<scheme>/requantized/<part>`: a weight moved off
  the grid, including values past the code range, and what the compressor
  writes for it against the same scales and zero points.

bfloat16 and the fp8 codes have no NumPy dtype here, so they are stored as
their raw bits (int16, uint8) with the dtype in `<scheme>/dtypes`.

Environment (~/.cache/dew/reference-venvs/kimi-k3): torch 2.8.0,
compressed-tensors 0.17.1. Run from the checkout:

    ~/.cache/dew/reference-venvs/kimi-k3/bin/python tools/compressed_tensors_reference.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from compressed_tensors.compressors.naive_quantized.base import (
    FloatQuantizationCompressor,
    IntQuantizationCompressor,
)
from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.utils.helpers import calculate_qparams

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "codecs" / "compressed_tensors.npz"

SCHEMES = {
    "pack_int4_group_sym": (PackedQuantizationCompressor, {"num_bits": 4, "type": "int", "strategy": "group",
                                                               "group_size": 32, "symmetric": True}),
    "pack_int4_group_asym": (PackedQuantizationCompressor, {"num_bits": 4, "type": "int", "strategy": "group",
                                                                "group_size": 32, "symmetric": False}),
    "pack_int4_actorder": (PackedQuantizationCompressor, {"num_bits": 4, "type": "int", "strategy": "group",
                                                              "group_size": 32, "symmetric": True, "actorder": "group"}),
    "pack_int8_channel": (PackedQuantizationCompressor, {"num_bits": 8, "type": "int", "strategy": "channel",
                                                             "symmetric": True}),
    "fp8_tensor": (FloatQuantizationCompressor, {"num_bits": 8, "type": "float", "strategy": "tensor", "symmetric": True}),
    "fp8_channel": (FloatQuantizationCompressor, {"num_bits": 8, "type": "float", "strategy": "channel",
                                                      "symmetric": True}),
    "fp8_block": (FloatQuantizationCompressor, {"num_bits": 8, "type": "float", "strategy": "block",
                                                    "block_structure": [16, 32], "symmetric": True}),
    "int8_channel": (IntQuantizationCompressor, {"num_bits": 8, "type": "int", "strategy": "channel", "symmetric": True}),
}
FORMATS = {PackedQuantizationCompressor: "pack-quantized", FloatQuantizationCompressor: "float-quantized",
           IntQuantizationCompressor: "int-quantized"}


def extremes(weight: torch.Tensor, args: QuantizationArgs) -> tuple[torch.Tensor, torch.Tensor]:
    """The min and max each scale covers, in the shape `calculate_qparams` takes."""
    if args.strategy == "tensor":
        return weight.amin().reshape(1), weight.amax().reshape(1)
    if args.strategy == "channel":
        return weight.amin(-1, keepdim=True), weight.amax(-1, keepdim=True)
    if args.strategy == "group":
        groups = weight.unflatten(-1, (-1, args.group_size))
        return groups.amin(-1), groups.amax(-1)
    rows, columns = args.block_structure
    blocks = weight.unflatten(0, (-1, rows)).unflatten(-1, (-1, columns))
    return blocks.amin((1, 3)), blocks.amax((1, 3))


def raw(tensor: torch.Tensor) -> tuple[np.ndarray, str]:
    name = str(tensor.dtype).removeprefix("torch.")
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy(), name
    if tensor.dtype == torch.float8_e4m3fn:
        return tensor.view(torch.uint8).numpy(), name
    return tensor.numpy(), name


def main() -> None:
    arrays, dtypes = {}, {}
    for index, (label, (compressor, fields)) in enumerate(SCHEMES.items()):
        generator = torch.Generator().manual_seed(7000 + index)
        weight = (torch.randn(48, 96, generator=generator) * 0.05).to(torch.bfloat16)
        args = QuantizationArgs(**fields)
        scheme = QuantizationScheme(targets=["Linear"], weights=args)
        low, high = extremes(weight.float(), args)
        scale, zero = calculate_qparams(low, high, args)
        scale = scale.to(torch.bfloat16)
        state = {"weight": weight, "weight_scale": scale}
        if not args.symmetric:
            state["weight_zero_point"] = zero
        if args.actorder == "group":
            state["weight_g_idx"] = torch.randperm(96, generator=generator) // args.group_size
            state["weight_g_idx"] = state["weight_g_idx"].to(torch.int32)
        stored = compressor.compress(dict(state), scheme)
        decompressed = compressor.decompress(dict(stored), scheme)["weight"].float()
        trained = weight.float() + torch.randn(48, 96, generator=generator) * 0.02
        trained[0, :4] = torch.tensor([10.0, -10.0, 1e-8, -1e-8])
        requantized = compressor.compress({**state, "weight": trained.to(torch.bfloat16)}, scheme)
        dtypes[label] = {"format": FORMATS[compressor], "weights": fields}
        for part, tensor in stored.items():
            arrays[f"{label}/{part}"], dtypes[label][part] = raw(tensor)
        for part, tensor in requantized.items():
            arrays[f"{label}/requantized/{part}"], _ = raw(tensor)
        arrays[f"{label}/dequantized"] = decompressed.numpy()
        arrays[f"{label}/trained"], _ = raw(trained.to(torch.bfloat16))
    arrays["dtypes"] = np.frombuffer(json.dumps(dtypes).encode(), np.uint8)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, **arrays)
    print(f"{FIXTURE}: {', '.join(SCHEMES)}")


if __name__ == "__main__":
    main()
