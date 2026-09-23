#!/usr/bin/env python3
"""Write the compressed-tensors MXFP4 fixture with compressed-tensors' own encoder.

A fresh [output, input] weight is encoded the way a quantized checkpoint's
Linear is: `calculate_qparams` over each group's min and max
(quantization/utils/helpers.py:50-140), then `MXFP4PackedCompressor.compress`
over the module's state, which carries the zero point the lifecycle forces
while a model is below COMPRESSED (quantization/lifecycle/apply.py:115).
The weight holds what the export rule has to get right: an all-zero group
with -0.0 in it, groups whose largest magnitude has a mantissa fraction of
exactly 0.75 (rounds up to the next power of two) and just under it (rounds
down and saturates at 6), every E2M1 tie, small negatives that round to
code 8 (-0), a -0.0 the zero point turns into code 0, a group small enough
that its exponent clamps at byte 0, a large group, a quotient that is a
float32 subnormal and underflows in bfloat16, and rows of N(0, 0.08).

The library computes in the weight's dtype, so the same weight is encoded
three times: as float32, rounded to bfloat16 and rounded to float16. In
float16 the smallest groups' scales underflow to zero and take the
library's substitute scale of 1 (`calculate_qparams` step 5, the eps of a
uint8 scale_dtype), which it writes as byte 127.

Environment (~/.cache/dew/reference-venvs/kimi-k3): torch 2.8.0,
compressed-tensors 0.17.1. Run from the checkout:

    ~/.cache/dew/reference-venvs/kimi-k3/bin/python tools/compressed_tensors_mxfp4_reference.py
"""

from pathlib import Path

import numpy as np
import torch
from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.utils.helpers import calculate_qparams

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "codecs" / "compressed_tensors_mxfp4.npz"

# moonshotai/Kimi-K3's one config group (config.json at f831ab6), the scheme
# Dew's `PACKED_MXFP4_WEIGHTS` reads.
ARGS = QuantizationArgs(num_bits=4, type="float", strategy="group", group_size=32,
                        symmetric=True, dynamic=False, scale_dtype=torch.uint8)
SCHEME = QuantizationScheme(targets=["Linear"], weights=ARGS)

TIES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
"""Halfway between neighbouring E2M1 magnitudes; each rounds to the even code."""

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
"""The dtypes the weight is encoded in, by the name the fixture keys carry."""


def crafted_groups() -> np.ndarray:
    """Sixteen groups of 32, two rows of the weight."""
    groups = np.zeros((16, 32), np.float32)
    groups[0, 3] = groups[0, 17] = -0.0
    # Largest magnitude 6: scale 2 ** 0, so each value is its own quotient.
    groups[1, :16] = [6.0, *TIES, -6.0, *(-np.float32(TIES))]
    groups[1, 16:24] = [-0.1, -0.2, -0.24, 0.1, 0.26, -0.26, -0.0, 0.0]
    # 1.75 = 1.11b: the mantissa fraction 0.75 rounds up to 2 (scale byte 126).
    groups[2, :4] = [1.75, -1.0, 0.3, -0.7]
    # Just under 1.75 rounds down to 1 (byte 125): 1.7499999 / 0.25 saturates at 6.
    groups[3, :4] = [np.nextafter(np.float32(1.75), np.float32(0)), 1.5, -1.6, 0.2]
    # The same fraction at 2 ** -3, as a negative largest magnitude.
    groups[4, :3] = [-0.21875, 0.1, 0.05]
    # 2 ** -130: 127 - 130 - 2 clamps to byte 0.
    groups[5, :3] = [2.0 ** -130, -(2.0 ** -131), 2.0 ** -133]
    groups[6] = np.linspace(-1000.0, 1000.0, 32, dtype=np.float32)
    # Scale 16: -2 ** -130 / 16 is a float32 subnormal (code 8) and a
    # bfloat16 underflow to -0.0, which the zero point makes code 0.
    groups[7, :4] = [100.0, -(2.0 ** -130), 2.0 ** -130, -1e-30]
    rng = np.random.default_rng(5150)
    groups[8:] = rng.normal(0.0, 0.3, (8, 32)).astype(np.float32)
    return groups


def fresh_weight() -> np.ndarray:
    rng = np.random.default_rng(5151)
    weight = rng.normal(0.0, 0.08, (64, 256)).astype(np.float32)
    weight[:2] = crafted_groups().reshape(2, 256)
    return weight


def compress(weight: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """compressed-tensors' packed codes and E8M0 scale bytes for one Linear weight."""
    groups = weight.unflatten(-1, (-1, ARGS.group_size))
    scale, zero_point = calculate_qparams(groups.amin(-1), groups.amax(-1), ARGS)
    state = {"weight": weight, "weight_scale": scale, "weight_zero_point": zero_point}
    compressed = MXFP4PackedCompressor.compress(state, SCHEME)
    assert set(compressed) == {"weight_packed", "weight_scale"}, sorted(compressed)
    return compressed["weight_packed"].numpy(), compressed["weight_scale"].numpy()


def main() -> None:
    weight = torch.from_numpy(fresh_weight())
    arrays = {}
    for name, dtype in DTYPES.items():
        cast = weight.to(dtype)
        packed, scale = compress(cast)
        assert packed.dtype == np.uint8 and scale.dtype == np.uint8
        # The raw bits, since NumPy has no bfloat16.
        arrays[f"weight_{name}"] = cast.view(torch.int32 if name == "float32" else torch.int16).numpy()
        arrays[f"weight_packed_{name}"], arrays[f"weight_scale_{name}"] = packed, scale
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, **arrays)
    for name in DTYPES:
        crafted = arrays[f"weight_scale_{name}"][:2].reshape(-1)
        print(f"{name}: crafted scale bytes {crafted.tolist()}")
    print(f"{FIXTURE}: {tuple(weight.shape)} weight in {', '.join(DTYPES)}")


if __name__ == "__main__":
    main()
