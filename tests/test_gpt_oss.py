"""GPT OSS parity with transformers 5.16.1, from tools/gpt_oss_reference.py.

The MXFP4 encoder is held to the bytes transformers' own encoder wrote
(`quantize_to_mxfp4` over `downcast_to_mxfp_torch` of
kernels-community/gpt-oss-triton-kernels, `write_mxfp4_encoder`), and its
bytes are read back through the released reader, transformers'
`convert_moe_packed_tensors`.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop.codecs import MXFP4, dequantize_mxfp4, quantize_mxfp4
from dew.nn.gpt_oss import GptOssMLP

FIXTURES = Path(__file__).parent / "fixtures" / "gpt_oss"

ENCODER = FIXTURES / "mxfp4_encoder.npz"
"""Weights and the blocks and scales transformers' encoder wrote for them, by domain."""

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
"""Every E2M1 magnitude, in code order."""


def released_decode(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """dew's bytes through the released reader, at the bf16 it ships."""
    import torch
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors
    return convert_moe_packed_tensors(torch.from_numpy(np.array(blocks)),
                                      torch.from_numpy(np.array(scales))).float().numpy()


def group_codes(blocks: np.ndarray) -> np.ndarray:
    """The 32 codes a group's 16 bytes hold, low nibble first."""
    codes = np.empty((*blocks.shape[:-1], 32), np.uint8)
    codes[..., 0::2], codes[..., 1::2] = blocks & 15, blocks >> 4
    return codes


def test_biased_interleaved_experts_match_reference():
    """The fp32 budget covers measured fixture and implementation roundoff.
    The fixture differs from the float64 equation by up to 7.8e-5; measured
    cross-implementation differences are 3.4e-5 locally and 6.1e-5 on CI.
    Dropping the down bias moves the output by 2.57.
    """
    with np.load(FIXTURES / "moe.npz") as fixture:
        arrays = {name: jnp.asarray(value) for name, value in fixture.items()}
    params = {
        "router": {"kernel": arrays["router.weight"].T, "bias": arrays["router.bias"]},
        "experts": {name.removeprefix("experts."): value for name, value in arrays.items()
                    if name.startswith("experts.")},
    }
    module = GptOssMLP(16, 24, 4, 2)

    def run(tree: dict) -> jax.Array:
        return jnp.asarray(module.apply({"params": tree}, arrays["hidden"]))

    np.testing.assert_allclose(run(params), arrays["output"], atol=1.6e-4, rtol=0)
    params["experts"]["down_proj_bias"] = jnp.zeros_like(params["experts"]["down_proj_bias"])
    assert float(jnp.max(jnp.abs(run(params) - arrays["output"]))) > 0.5


def test_mxfp4_matches_reference_dequantization_exactly():
    """The fixture's bf16 output, atol 0; and every code at the two scale
    bytes that decode the smallest magnitudes to float32 subnormals, which
    XLA on CPU flushes to zero and the released reader keeps."""
    with np.load(FIXTURES / "mxfp4.npz") as fixture:
        actual = dequantize_mxfp4(fixture["blocks"], fixture["scales"])
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, fixture["output"])

    every_code = np.tile(np.arange(16, dtype=np.uint8), 2)
    blocks = np.tile(every_code[0::2] | (every_code[1::2] << 4), (2, 1, 1, 1))
    scales = np.array([0, 1], np.uint8).reshape(2, 1, 1)
    decoded = dequantize_mxfp4(blocks, scales)
    theirs = released_decode(blocks, scales)
    assert np.count_nonzero((decoded != 0) & (np.abs(decoded) < np.finfo(np.float32).tiny)) == 16
    np.testing.assert_array_equal(decoded, theirs)
    np.testing.assert_array_equal(np.signbit(decoded), np.signbit(theirs))


def test_mxfp4_encodes_every_codepoint_and_the_released_reader_agrees():
    """One group holding all sixteen E2M1 values twice, so its scale is 1 and
    each value is its own code. atol 0, observed difference 0, both zeros
    kept apart."""
    weight = np.tile(np.concatenate([E2M1, -E2M1]), 2).reshape(1, 32, 1)
    blocks, scales = quantize_mxfp4(weight)

    assert (blocks.shape, blocks.dtype) == ((1, 1, 1, 16), np.uint8)
    assert (scales.shape, scales.dtype) == ((1, 1, 1), np.uint8)
    np.testing.assert_array_equal(scales, [[[127]]])
    np.testing.assert_array_equal(group_codes(blocks).reshape(-1),
                                  np.tile(np.arange(16, dtype=np.uint8), 2))
    decoded = released_decode(blocks, scales)
    np.testing.assert_array_equal(decoded, weight)
    np.testing.assert_array_equal(np.signbit(decoded), np.signbit(weight))


@pytest.mark.parametrize("domain", ["midpoints", "random_bf16", "mixed", "trained", "every_bf16_magnitude"])
def test_mxfp4_encodes_what_the_released_encoder_encodes(domain):
    """Both bytes of every group against what the released encoder wrote,
    atol 0. The domains: E2M1 midpoints of both signs at the scales 2 ** 0,
    2 ** -3, 2 ** 10 and 2 ** -126, beside values bf16 carries onto one, all
    of which the kernel rounds away from zero; random bf16 bit patterns, so
    every exponent, both signs, the subnormals and the zeros are in there;
    groups where one large value sets the scale and the rest sit under the
    grid; trained-shaped weights; and one group per finite bf16 magnitude,
    so the scale's round-up meets every exponent and mantissa."""
    with np.load(ENCODER) as fixture:
        weight, blocks, scales = (fixture[f"{domain}_{part}"] for part in ("weight", "blocks", "scales"))

    ours = quantize_mxfp4(weight)

    np.testing.assert_array_equal(ours[1], scales)
    np.testing.assert_array_equal(ours[0], blocks)


def test_mxfp4_keeps_a_group_of_bf16_subnormals():
    """2 ** -127 is a bf16 subnormal, and the released encoder writes it as
    the 0.5 code at the 2 ** -126 scale, so the reader gives it back exactly.
    A subnormal beside a normal large enough to set the scale is a code below
    the grid, and rounds to zero as the reference's own product does."""
    weight = np.full((1, 32, 1), np.ldexp(1.0, -127), np.float32)
    weight[0, 5, 0] = -weight[0, 5, 0]
    blocks, scales = quantize_mxfp4(weight)
    decoded = released_decode(blocks, scales)

    assert int(scales[0, 0, 0]) == 1
    assert set(group_codes(blocks).reshape(-1).tolist()) == {1, 9}
    np.testing.assert_array_equal(decoded, weight)
    np.testing.assert_array_equal(np.signbit(decoded), np.signbit(weight))

    beside = np.full((1, 32, 1), np.ldexp(1.0, -130), np.float32)
    beside[0, 0, 0] = 1.0
    blocks, scales = quantize_mxfp4(beside)
    assert int(scales[0, 0, 0]) == 125
    assert set(group_codes(blocks).reshape(-1).tolist()) == {0, 6}


def test_mxfp4_rounds_an_fp32_weight_through_bf16_first():
    """0.7495 is nearer 0.5 than 1.0 in the E2M1 grid and still comes back as
    1.0: bf16 carries it onto the 0.75 midpoint, which rounds up. That second
    rounding is the released encoder's own first step, so an fp32-trained
    weight can land one code away from where a single rounding would put it."""
    weight = np.full((1, 32, 1), 0.7495, np.float32)
    weight[0, 0, 0] = 6.0
    blocks, scales = quantize_mxfp4(weight)

    assert int(scales[0, 0, 0]) == 127
    assert int(group_codes(blocks)[0, 0, 0, 1]) == 2
    assert released_decode(blocks, scales)[0, 1, 0] == 1.0


def test_mxfp4_re_encodes_a_checkpoint_group_onto_the_same_values():
    """The fixture's 1024 values, decoded and encoded again: every one comes
    back at atol 0, and the bytes come back unchanged wherever the group's
    largest code is +-6 or +-4. A group whose largest code is +-3 is written
    at half that scale with its codes doubled, which is the same numbers in
    other bytes -- this encoder canonicalizes a group's scale, it does not
    copy the source's."""
    with np.load(FIXTURES / "mxfp4.npz") as fixture:
        blocks, scales, values = (fixture[name] for name in ("blocks", "scales", "output"))
    again_blocks, again_scales = quantize_mxfp4(values)
    np.testing.assert_array_equal(released_decode(again_blocks, again_scales), values)
    canonical = np.isin(np.max(group_codes(blocks) & 7, axis=-1), (6, 7))
    np.testing.assert_array_equal(again_blocks[canonical], blocks[canonical])
    np.testing.assert_array_equal(again_scales[canonical], scales[canonical])

    threes = np.full((1, 32, 1), 3.0, np.float32)
    blocks, scales = quantize_mxfp4(threes)
    assert int(scales[0, 0, 0]) == 126
    assert set(group_codes(blocks).reshape(-1).tolist()) == {7}
    np.testing.assert_array_equal(released_decode(blocks, scales), threes)


def test_mxfp4_zero_and_smallest_groups_take_the_floor_of_the_format():
    """A group with nothing in it takes the 0x00 scale byte and still tells
    -0.0 from 0.0. A group down at the fp32 normal floor encodes exactly at
    the 2 ** -126 scale the round-up stops on. Below that floor nothing
    survives: a group of bf16 subnormals decodes to zero."""
    zeros = np.zeros((1, 32, 1), np.float32)
    zeros[0, 3, 0] = -0.0
    blocks, scales = quantize_mxfp4(zeros)
    decoded = released_decode(blocks, scales)
    assert int(scales[0, 0, 0]) == 0
    np.testing.assert_array_equal(decoded, zeros)
    np.testing.assert_array_equal(np.signbit(decoded), np.signbit(zeros))

    # Both groups want a scale below the floor and get 2 ** -126, byte 1: the
    # quotient that picks it is 2 ** -128 and 2 ** -127, subnormal both times.
    for exponent, code in ((-126, 2), (-125, 4)):
        smallest = np.full((1, 32, 1), np.ldexp(1.0, exponent), np.float32)
        smallest[0, 7, 0] *= 1.5
        blocks, scales = quantize_mxfp4(smallest)
        assert int(scales[0, 0, 0]) == 1
        assert int(group_codes(blocks)[0, 0, 0, 0]) == code
        np.testing.assert_array_equal(released_decode(blocks, scales), smallest)

    subnormal = np.full((1, 32, 1), np.ldexp(1.0, -133), np.float32)
    blocks, scales = quantize_mxfp4(subnormal)
    np.testing.assert_array_equal(released_decode(blocks, scales), np.zeros_like(subnormal))


@pytest.mark.parametrize("weight,message", [
    (np.full((1, 32, 1), np.nan, np.float32), "reserved E8M0 NaN scale"),
    (np.full((1, 32, 1), np.inf, np.float32), "reserved E8M0 NaN scale"),
    (np.zeros((1, 48, 1), np.float32), "multiple of the 32-value group"),
    (np.zeros((32, 1), np.float32), "multiple of the 32-value group"),
    (np.zeros((1, 32, 1), np.int32), "encodes float weights"),
])
def test_mxfp4_refuses_what_the_format_cannot_hold(weight, message):
    """E2M1 encodes neither an infinity nor a NaN, and a stored group is
    always exactly 32 values, so a half group has nowhere to go."""
    with pytest.raises(ValueError, match=message):
        quantize_mxfp4(weight)


def test_pack_mxfp4_writes_back_the_recorded_stems_and_nothing_else():
    """A tensor that arrived as a float weight -- an expert bias, the router,
    the attention sinks, the embeddings a head ties to -- comes back as the
    same object, and the recorded stem comes back as a pair the released
    reader decodes to what was unpacked. A recorded stem the tensors have
    lost is refused, and so is half a pair on the way in."""
    with np.load(FIXTURES / "mxfp4.npz") as fixture:
        blocks, scales = fixture["blocks"], fixture["scales"]
    stem = "model.layers.0.mlp.experts.gate_up_proj"
    untouched = {f"{stem}_bias": np.arange(6, dtype=np.float32),
                 "model.layers.0.self_attn.sinks": np.arange(4, dtype=np.float32),
                 "model.layers.0.mlp.router.weight": np.eye(3, dtype=np.float32),
                 "model.embed_tokens.weight": np.eye(4, dtype=np.float32)}
    tensors = {f"{stem}_blocks": blocks, f"{stem}_scales": scales, **untouched}
    assert MXFP4.names(tensors) == (stem,)

    unpacked = MXFP4.dequantize(tensors)
    assert set(unpacked) == {stem, *untouched}
    packed = MXFP4.requantize(unpacked, (stem,))
    assert set(packed) == set(tensors)
    for name, value in untouched.items():
        assert packed[name] is value
    np.testing.assert_array_equal(
        released_decode(packed[f"{stem}_blocks"], packed[f"{stem}_scales"]), unpacked[stem])

    with pytest.raises(ValueError, match="is not among the tensors to write"):
        MXFP4.requantize({name: value for name, value in unpacked.items() if name != stem}, (stem,))
    with pytest.raises(ValueError, match=r"holds no .*_scales"):
        MXFP4.names({name: value for name, value in tensors.items()
                     if not name.endswith("_scales")})
