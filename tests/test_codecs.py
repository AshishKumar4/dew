"""The checkpoint codecs of `dew.interop.codecs`, held to their formats' own rules.

compressed-tensors' MXFP4 export is held to the bytes compressed-tensors
0.17.1 wrote for one fresh weight in three dtypes
(tools/compressed_tensors_mxfp4_reference.py).
"""

from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from dew.interop import codecs

FIXTURE = Path(__file__).parent / "fixtures" / "codecs" / "compressed_tensors_mxfp4.npz"

STORED = {"float32": np.float32, "bfloat16": ml_dtypes.bfloat16, "float16": np.float16}
"""The dtypes the fixture's weight is encoded in, by the name its keys carry."""


def fixture_weight(dtype: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fixture's weight in `dtype` and the codes and exponent bytes the library wrote."""
    with np.load(FIXTURE) as fixture:
        return (fixture[f"weight_{dtype}"].view(STORED[dtype]), fixture[f"weight_packed_{dtype}"],
                fixture[f"weight_scale_{dtype}"])


@pytest.mark.parametrize("dtype", sorted(STORED))
def test_the_compressed_tensors_export_writes_the_librarys_bytes(dtype):
    """Both tensors of the pair, atol 0, over the fixture's crafted groups
    and 62 rows of N(0, 0.08). compressed-tensors computes in the weight's
    dtype, so each dtype pins bytes of its own: bfloat16 underflows one
    subnormal quotient to -0.0, which the zero point makes code 0 where
    float32 keeps code 8, and float16 underflows the smallest groups'
    scales to zero, which the library replaces by 1 and writes as byte 127."""
    weight, codes, exponents = fixture_weight(dtype)
    bias = np.arange(3, dtype=np.float32)

    packed = codecs.pack_packed_mxfp4({"m.weight": weight, "m.bias": bias}, ("m.weight",))

    assert set(packed) == {"m.weight_packed", "m.weight_scale", "m.bias"}
    assert packed["m.bias"] is bias
    np.testing.assert_array_equal(packed["m.weight_packed"], codes)
    np.testing.assert_array_equal(packed["m.weight_scale"], exponents)


def test_the_crafted_groups_take_the_rules_exponents():
    """The fixture's first eight float32 groups by the rule's own steps:
    zero clamps to byte 0; 6 = 1.5 * 4 has mantissa fraction 0.5 and rounds
    down to 4, byte 127 + 2 - 2; 1.75 has fraction 0.75 and rounds up to 2,
    byte 126, where one ulp under it rounds down to 1, byte 125; -0.21875 =
    -1.75 * 2 ** -3 rounds up to 2 ** -2, byte 123; 2 ** -130 rounds to 0
    and clamps to byte 0; 1000 = 1.95 * 512 rounds up to 1024, byte 135;
    100 = 1.5625 * 64 rounds down to 64, byte 131."""
    _, _, exponents = fixture_weight("float32")
    assert exponents[0, :8].tolist() == [0, 127, 126, 125, 123, 0, 135, 131]


@pytest.mark.parametrize("dtype", sorted(STORED))
def test_an_untrained_re_export_writes_the_same_values(dtype):
    """Decoding what the library wrote and encoding it again, in the same
    dtype, gives back the values and the exponents, atol 0: a nonzero
    group's largest quotient is 4 or 6, which the rule maps back to its
    exponent. Only a code 8 (-0.0) may come back as code 0, the zero
    point's doing, which is the same value."""
    _, codes, exponents = fixture_weight(dtype)
    source = {"m.weight_packed": codes, "m.weight_scale": exponents}
    decoded = codecs.read_packed_mxfp4_tensor(source, "m.weight")

    again = codecs.pack_packed_mxfp4({"m.weight": decoded.astype(STORED[dtype])}, ("m.weight",))

    np.testing.assert_array_equal(codecs.read_packed_mxfp4_tensor(again, "m.weight"), decoded)
    np.testing.assert_array_equal(again["m.weight_scale"], exponents)
    before, after = (np.stack([packed & 15, packed >> 4]) for packed in (codes, again["m.weight_packed"]))
    moved = before != after
    assert set(before[moved].tolist()) <= {8} and set(after[moved].tolist()) <= {0}


def test_a_group_rounding_past_the_largest_power_of_two_is_refused():
    """3e38 has mantissa fraction 0.76 over 2 ** 127, so it rounds up to
    2 ** 128, past float32, whose exponent byte would be E8M0's NaN."""
    weight = np.zeros((1, 32), np.float32)
    weight[0, 0] = 3e38
    with pytest.raises(ValueError, match="reserved NaN 0xff"):
        codecs.quantize_packed_mxfp4(weight)
