"""The checkpoint codecs of `dew.interop.codecs`, held to their formats' own rules.

compressed-tensors' MXFP4 export is held to the bytes compressed-tensors
0.17.1 wrote for one fresh weight in three dtypes
(tools/compressed_tensors_mxfp4_reference.py). Every codec reads a region of
a decoded tensor from the blocks it covers, which the formats' formulas,
taken element by element, check on views far larger than memory.
"""

from collections.abc import Callable
from functools import partial
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from test_quantized import decode_e4m3fn

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


# --------------------------------------------------------------------------
# Regions: what a streamed load asks for, one shard of one tensor at a time
# --------------------------------------------------------------------------

def encoded(kind: str) -> tuple[dict[str, np.ndarray], str, Callable[..., np.ndarray]]:
    """A tensor stored in `kind`'s format, its decoded name and the format's
    reader. Each shape ends on a partial block or a partial block row where
    the format has one."""
    rng = np.random.default_rng(7)
    if kind == "stored":
        return {"b": rng.integers(-9, 9, (70, 300), dtype=np.int32)}, "b", partial(codecs.read_fp8_tensor, block=32)
    if kind == "fp8":
        tensors = codecs.pack_fp8({"w": rng.standard_normal((70, 300)).astype(np.float32)}, ("w",), 32, ue8m0=False)
        return tensors, "w", partial(codecs.read_fp8_tensor, block=32)
    if kind == "gpt_oss":
        tensors = codecs.pack_mxfp4({"w": rng.standard_normal((3, 96, 70)).astype(np.float32)}, ("w",))
        return tensors, "w", codecs.read_mxfp4_tensor
    weight = rng.standard_normal((70, 256)).astype(np.float32)
    return codecs.pack_packed_mxfp4({"m.weight": weight}, ("m.weight",)), "m.weight", codecs.read_packed_mxfp4_tensor


REGIONS_2D = ((slice(3, 50, 3),), (slice(None), slice(17, 250, 5)), (slice(69, 2, -7), slice(None, None, -1)),
              (slice(5, 5),), (slice(-9, None), slice(33, 34)))
REGIONS_3D = ((slice(1, 3),), (slice(None), slice(17, 90, 5), slice(None, None, -2)),
              (slice(2, 0, -1), slice(31, 33), slice(64, 70)), (slice(0, 0),))


@pytest.mark.parametrize("kind, region", [
    *((kind, region) for kind in ("stored", "fp8", "compressed_tensors") for region in REGIONS_2D),
    *(("gpt_oss", region) for region in REGIONS_3D)])
def test_a_region_decodes_to_the_whole_tensors_values_there(kind, region):
    """What `jax.make_array_from_callback` asks of a streamed load: any
    region of a decoded tensor, strided, reversed or empty, on block edges
    or off them, is the whole decode at that region, bit for bit, -0.0
    included, and an unquantized tensor keeps its stored dtype."""
    tensors, name, read = encoded(kind)
    whole = read(tensors, name)

    part = read(tensors, name, region)

    assert part.dtype == whole.dtype
    np.testing.assert_array_equal(part.view(np.uint32), whole[region].view(np.uint32))


def hankel(base: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """A read-only [rows, cols] view holding base[i + j] at (i, j), with no memory of its own."""
    return np.lib.stride_tricks.as_strided(base, shape, (base.itemsize, base.itemsize), writeable=False)


def e2m1_values(codes: np.ndarray) -> np.ndarray:
    """E2M1 codes by their bit fields: bit 3 the sign, the rest a magnitude index."""
    magnitudes = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
    return np.where(codes & 8, np.float32(-1), np.float32(1)) * magnitudes[codes & 7]


def indices(rows: slice, cols: slice) -> tuple[np.ndarray, np.ndarray]:
    """A region's row and column indices, broadcasting against each other."""
    return np.arange(rows.start, rows.stop)[:, None], np.arange(cols.start, cols.stop, cols.step)[None, :]


def test_a_region_of_a_tensor_too_large_to_decode_reads_only_its_blocks():
    """Tensors far past memory, as views that hold none: the FP8 weight
    decodes whole to 1 TiB of float32 and the compressed-tensors one to 128
    GiB. A region across block and group edges decodes from the blocks and
    groups it covers alone, and equals each format's formula there, taken
    element by element."""
    rng = np.random.default_rng(3)

    codes = rng.integers(0, 0x7f, 2 ** 20 + 2 ** 18, dtype=np.uint8) | rng.choice(np.uint8([0, 0x80]), 2 ** 20 + 2 ** 18)
    scales = np.ldexp(np.float32(1), rng.integers(-12, 12, 2 ** 13 + 2 ** 11)).astype(np.float32)
    fp8 = {"w": hankel(codes.view(codecs.E4M3), (2 ** 20, 2 ** 18)), "w_scale_inv": hankel(scales, (2 ** 13, 2 ** 11))}
    rows, cols = slice(123_390, 123_395), slice(70_001, 70_100, 3)
    r, c = indices(rows, cols)
    expected = decode_e4m3fn(codes[r + c]) * scales[r // 128 + c // 128]
    np.testing.assert_array_equal(codecs.read_fp8_tensor(fp8, "w", (rows, cols), block=128).view(np.uint32),
                                  expected.view(np.uint32))

    packed = rng.integers(0, 256, 2 ** 22 + 2 ** 12, dtype=np.uint8)
    exponents = rng.integers(115, 140, 2 ** 22 + 2 ** 8, dtype=np.uint8)
    mx = {"m.weight_packed": hankel(packed, (2 ** 22, 2 ** 12)), "m.weight_scale": hankel(exponents, (2 ** 22, 2 ** 8))}
    rows, cols = slice(4_000_001, 4_000_006), slice(5_001, 5_100, 3)
    r, c = indices(rows, cols)
    expected = (e2m1_values((packed[r + c // 2] >> (4 * (c % 2))) & 15)
                * np.ldexp(np.float32(1), exponents[r + c // 32].astype(np.int32) - 127))
    np.testing.assert_array_equal(codecs.read_packed_mxfp4_tensor(mx, "m.weight", (rows, cols)).view(np.uint32),
                                  expected.view(np.uint32))
