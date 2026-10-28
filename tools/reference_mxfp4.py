"""compressed-tensors' MXFP4 codec, which the reference tools encode and decode its checkpoints with: the
scheme a quantized checkpoint's Linear carries (moonshotai/Kimi-K3's config group at f831ab6, which Dew's
`PACKED_MXFP4_WEIGHTS` reads) and the library's own `calculate_qparams` and `MXFP4PackedCompressor`,
independent of Dew's codec.

Environment: torch and compressed-tensors 0.17.1 (the kimi-k3 reference venv, or Dew's test environment).
"""

import torch
from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.utils.helpers import calculate_qparams

ARGS = QuantizationArgs(num_bits=4, type="float", strategy="group", group_size=32,
                        symmetric=True, dynamic=False, scale_dtype=torch.uint8)
SCHEME = QuantizationScheme(targets=["Linear"], weights=ARGS)


def encoded(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One Linear weight's packed E2M1 codes and E8M0 scale bytes, computed in the weight's own
    dtype as the library does: `calculate_qparams` over each group's min and max, then the compressor
    over the module's state, with the zero point the lifecycle carries below COMPRESSED
    (quantization/lifecycle/apply.py:115)."""
    groups = weight.unflatten(-1, (-1, ARGS.group_size))
    scale, zero_point = calculate_qparams(groups.amin(-1), groups.amax(-1), ARGS)
    state = {"weight": weight, "weight_scale": scale, "weight_zero_point": zero_point}
    compressed = MXFP4PackedCompressor.compress(state, SCHEME)
    assert set(compressed) == {"weight_packed", "weight_scale"}, sorted(compressed)
    return compressed["weight_packed"].contiguous(), compressed["weight_scale"].contiguous()


def decoded(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The library's decompression of one packed pair, computed in bfloat16 as the library does and
    widened to float32."""
    state = {"weight_packed": packed, "weight_scale": scale}
    return MXFP4PackedCompressor.decompress(state, SCHEME)["weight"].float()
