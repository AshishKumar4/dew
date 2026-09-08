"""GPT OSS parity with transformers 5.16.1, from tools/gpt_oss_reference.py.

The MXFP4 encoder is checked the other way round: its bytes go through the
released reader, transformers' own `convert_moe_packed_tensors`, at test
time, because a fixture cannot hold the decode of bytes the test just wrote
and the released encoder (triton_kernels' `downcast_to_mxfp`) needs a GPU.
What the encoder is meant to emit -- the group's scale and each element's
code -- is stated here from the format instead, by exact power-of-two search
and by comparing against the grid's midpoints, so nothing in this file
repeats dew's own arithmetic.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.gpt_oss import (
    GptOssMLP, dequantize_mxfp4, mxfp4_stems, pack_mxfp4, quantize_mxfp4, unpack_mxfp4,
)

FIXTURES = Path(__file__).parent / "fixtures" / "gpt_oss"

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
"""Every E2M1 magnitude, in code order."""

MIDPOINTS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)
"""Halfway between neighbouring magnitudes, each one exact in fp32."""


def released_decode(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """dew's bytes through the released reader, at the bf16 it ships."""
    import torch
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors
    return convert_moe_packed_tensors(torch.from_numpy(np.array(blocks)),
                                      torch.from_numpy(np.array(scales))).float().numpy()


def nearest_codes(values: np.ndarray) -> np.ndarray:
    """The nearest E2M1 code, ties to even, from the grid's midpoints alone."""
    below = np.searchsorted(MIDPOINTS, np.abs(values), side='left')
    upto = np.searchsorted(MIDPOINTS, np.abs(values), side='right')
    # The two counts differ exactly on a tie, where the even code wins.
    index = np.where(below != upto, np.where(below % 2 == 0, below, upto), below)
    return (index + 8 * np.signbit(values)).astype(np.uint8)


def released_scale_byte(largest: np.ndarray) -> np.ndarray:
    """The released encoder's own scale expression, in numpy.

    triton_kernels' `downcast_to_mxfp_torch` under ROUND_UP: the group's
    largest magnitude over 6, rounded up to a power of two in the fp32 bits,
    read back as the E8M0 exponent byte. numpy keeps the subnormal quotient
    that XLA flushes, so this is the rule dew has to reproduce and not the
    arithmetic it uses to get there.
    """
    bits = (np.asarray(largest, np.float32) / np.float32(6)).view(np.uint32)
    return (((bits + 0x007fffff) & 0x7f800000) >> 23).astype(np.uint8)


def group_codes(blocks: np.ndarray) -> np.ndarray:
    """The 32 codes a group's 16 bytes hold, low nibble first."""
    codes = np.empty((*blocks.shape[:-1], 32), np.uint8)
    codes[..., 0::2], codes[..., 1::2] = blocks & 15, blocks >> 4
    return codes


def encode(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return tuple(np.asarray(part) for part in quantize_mxfp4(jnp.asarray(weight)))


def bf16_magnitudes() -> np.ndarray:
    """Every non-negative bf16 value that is normal or zero, as fp32."""
    import ml_dtypes
    patterns = np.arange(1 << 15, dtype=np.uint16).view(ml_dtypes.bfloat16).astype(np.float32)
    return patterns[np.isfinite(patterns) & ((patterns == 0) | (patterns >= np.ldexp(1.0, -126)))]


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
    """bf16 output, atol 0; maximum observed absolute difference 0."""
    with np.load(FIXTURES / "mxfp4.npz") as fixture:
        actual = dequantize_mxfp4(jnp.asarray(fixture["blocks"]), jnp.asarray(fixture["scales"]))
        assert actual.dtype == jnp.bfloat16
        np.testing.assert_array_equal(actual.astype(jnp.float32), fixture["output"])


def test_mxfp4_encodes_every_codepoint_and_the_released_reader_agrees():
    """One group holding all sixteen E2M1 values twice, so its scale is 1 and
    each value is its own code. atol 0, observed difference 0, both zeros
    kept apart."""
    weight = np.tile(np.concatenate([E2M1, -E2M1]), 2).reshape(1, 32, 1)
    blocks, scales = encode(weight)

    assert (blocks.shape, blocks.dtype) == ((1, 1, 1, 16), np.uint8)
    assert (scales.shape, scales.dtype) == ((1, 1, 1), np.uint8)
    np.testing.assert_array_equal(scales, [[[127]]])
    np.testing.assert_array_equal(group_codes(blocks).reshape(-1),
                                  np.tile(np.arange(16, dtype=np.uint8), 2))
    decoded = released_decode(blocks, scales)
    np.testing.assert_array_equal(decoded, weight)
    np.testing.assert_array_equal(np.signbit(decoded), np.signbit(weight))


def test_mxfp4_scale_and_codes_are_the_ones_the_format_names():
    """Every group of a trained-shaped weight against the released scale
    expression and the codes read off the grid's midpoints: atol 0 on both.
    The scale rounds up, so no value reaches the saturating clamp, and no
    group takes 0xff, the byte E8M0 reserves for NaN."""
    weight = (np.random.default_rng(2).standard_normal((2, 64, 3)) * 0.7).astype(np.float32)
    blocks, scales = encode(weight)
    rows = np.asarray(jnp.asarray(weight).astype(jnp.bfloat16).astype(jnp.float32))
    groups = rows.swapaxes(1, 2).reshape(2, 3, -1, 32)
    codes = group_codes(blocks)

    assert scales.shape == groups.shape[:-1] and scales.max() < 0xff
    for index in np.ndindex(groups.shape[:-1]):
        byte = int(released_scale_byte(np.max(np.abs(groups[index]))))
        assert int(scales[index]) == byte
        scale = np.float32(np.ldexp(1.0, byte - 127))
        assert np.max(np.abs(groups[index])) / scale <= 6
        np.testing.assert_array_equal(codes[index], nearest_codes(groups[index] / scale))


def test_mxfp4_scale_matches_the_released_rule_on_every_bf16_magnitude():
    """One group per bf16 magnitude that fp32 holds as a normal, every one of
    them, against the released expression: atol 0. This is where dew's
    arithmetic has to differ from the reference's -- XLA flushes the
    subnormal quotient of a group below 6 * 2 ** -126, where the reference
    rounds it up to the 2 ** -126 scale -- so the byte is checked over the
    whole domain rather than at a few points."""
    magnitudes = bf16_magnitudes()
    weight = np.repeat(magnitudes, 32).reshape(1, -1, 1)
    _, scales = encode(weight)
    np.testing.assert_array_equal(scales.reshape(-1), released_scale_byte(magnitudes))



def test_mxfp4_rounds_an_fp32_weight_through_bf16_first():
    """0.7495 is nearer 0.5 than 1.0 in the E2M1 grid and still comes back as
    1.0: bf16 carries it onto the 0.75 tie, which rounds to even. That second
    rounding is the released encoder's own first step, so an fp32-trained
    weight can land one code away from where a single rounding would put it."""
    weight = np.full((1, 32, 1), 0.7495, np.float32)
    weight[0, 0, 0] = 6.0
    blocks, scales = encode(weight)

    assert int(scales[0, 0, 0]) == 127
    assert int(nearest_codes(np.float32(0.7495))) == 1
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
    again_blocks, again_scales = encode(values)
    np.testing.assert_array_equal(released_decode(again_blocks, again_scales), values)
    canonical = np.isin(np.max(group_codes(blocks) & 7, axis=-1), (6, 7))
    np.testing.assert_array_equal(again_blocks[canonical], blocks[canonical])
    np.testing.assert_array_equal(again_scales[canonical], scales[canonical])

    threes = np.full((1, 32, 1), 3.0, np.float32)
    blocks, scales = encode(threes)
    assert int(scales[0, 0, 0]) == 126
    assert set(group_codes(blocks).reshape(-1).tolist()) == {7}
    np.testing.assert_array_equal(released_decode(blocks, scales), threes)


def test_mxfp4_zero_and_smallest_groups_take_the_floor_of_the_format():
    """A group with nothing in it takes the 0x00 scale byte and still tells
    -0.0 from 0.0. A group down at the fp32 normal floor still encodes
    exactly, at the 2 ** -126 scale the round-up stops on -- the quotient
    that picks it is subnormal, which is the one place XLA's arithmetic
    would have written the zero group instead. Below that floor nothing
    survives: a group of bf16 subnormals decodes to zero."""
    zeros = np.zeros((1, 32, 1), np.float32)
    zeros[0, 3, 0] = -0.0
    blocks, scales = encode(zeros)
    decoded = released_decode(blocks, scales)
    assert int(scales[0, 0, 0]) == 0
    np.testing.assert_array_equal(decoded, zeros)
    np.testing.assert_array_equal(np.signbit(decoded), np.signbit(zeros))

    # Both groups want a scale below the floor and get 2 ** -126, byte 1: the
    # quotient that picks it is 2 ** -128 and 2 ** -127, subnormal both times.
    for exponent, code in ((-126, 2), (-125, 4)):
        smallest = np.full((1, 32, 1), np.ldexp(1.0, exponent), np.float32)
        smallest[0, 7, 0] *= 1.5
        blocks, scales = encode(smallest)
        assert int(scales[0, 0, 0]) == 1
        assert int(group_codes(blocks)[0, 0, 0, 0]) == code
        np.testing.assert_array_equal(released_decode(blocks, scales), smallest)

    subnormal = np.full((1, 32, 1), np.ldexp(1.0, -133), np.float32)
    blocks, scales = encode(subnormal)
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
        quantize_mxfp4(jnp.asarray(weight))


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
    assert mxfp4_stems(tensors) == (stem,)

    unpacked = unpack_mxfp4(tensors)
    assert set(unpacked) == {stem, *untouched}
    packed = pack_mxfp4(unpacked, (stem,))
    assert set(packed) == set(tensors)
    for name, value in untouched.items():
        assert packed[name] is value
    np.testing.assert_array_equal(
        released_decode(packed[f"{stem}_blocks"], packed[f"{stem}_scales"]), unpacked[stem])

    with pytest.raises(ValueError, match="not among the tensors to write back"):
        pack_mxfp4({name: value for name, value in unpacked.items() if name != stem}, (stem,))
    with pytest.raises(ValueError, match="holds no .*_scales"):
        mxfp4_stems({name: value for name, value in tensors.items()
                     if not name.endswith("_scales")})
