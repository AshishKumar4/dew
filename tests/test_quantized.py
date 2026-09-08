"""Block-scaled FP8 dequantization against DeepSeek's reference formula.

`weight_dequant` in deepseek-ai/DeepSeek-V3's inference/kernel.py computes
y[i, j] = float32(x[i, j]) * s[i // block, j // block] with the partial
blocks masked. The tests hold the port to that, bit for bit in float32, on a
hand-computed case and on random fp8 blocks, and hold the loader hook to
pairing each weight with its `_scale_inv` partner.

The encoder's reference is the cast that produces *weights* in this format,
`per_block_cast_to_fp8` in deep_gemm/utils/math.py of deepseek-ai/DeepGEMM,
and not `act_quant` in the same DeepSeek file, which reduces over 1 x 128
activation tiles and divides where the weight cast multiplies by the
block's reciprocal. `torch_per_block_cast` below is that reference ported
to torch, and `quantize_fp8_blocks` is held to its bytes.

Two readers check the bytes rather than the arrays. `decode_e4m3fn` reads
E4M3FN's bit fields by the OCP FP8 definition, so a written file is decoded
by something that shares no code with ml_dtypes, torch or the writer. The
network tests read DeepSeek's own shipped bytes, dequantize them, and run
the encoder over the result: V3's float32 scales and V3.2-Exp's ue8m0
scales both come back bit for bit, which is the only oracle that can say
the rule here is the rule that made the checkpoints.

Format correctness and quantization loss are separate claims and are
tested apart. Every byte-level assertion is exact; the loss a trained fp32
weight takes on its way into the format is asserted against the format's
own rounding bound, and reported by tools/fp8_reexport_budget.py.
"""

import json
import os
import shutil
import struct
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
import pytest

from dew.interop import load_pretrained
from dew.interop.quantized import (AMAX_FLOOR, BLOCK, E4M3_MAX, SCALE_SUFFIX,
                                   dequantize_checkpoint, dequantize_fp8_blocks, fp8_block,
                                   pack_fp8, quantize_fp8_blocks, scaled_names)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hf" / "deepseek-v3-tiny"

FP8 = ml_dtypes.float8_e4m3fn


def reference(weight, scale_inv, block):
    """DeepSeek's kernel, one element at a time."""
    out = np.empty(weight.shape, np.float32)
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            out[i, j] = np.float32(weight[i, j]) * np.float32(scale_inv[i // block, j // block])
    return out


def bits(x):
    return np.asarray(x, np.float32).view(np.uint32)


def test_a_hand_computed_case_with_a_partial_column_block():
    """2 x 2 blocks over a 4 x 3 weight: the third column is a block of its
    own, with the scales of the second block column."""
    weight = np.arange(1, 13, dtype=np.float32).reshape(4, 3).astype(FP8)
    scale_inv = np.array([[0.5, 2.0], [4.0, 0.25]], np.float32)
    expected = np.array([[0.5, 1.0, 6.0], [2.0, 2.5, 12.0],
                         [28.0, 32.0, 2.25], [40.0, 44.0, 3.0]], np.float32)

    out = dequantize_fp8_blocks(weight, scale_inv, block=2)

    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("shape", [(256, 384), (576, 200), (5, 7)])
def test_random_blocks_equal_the_reference_bit_for_bit(shape):
    """Random fp8 values (denormals and both zeros included) under random
    scales, shapes with whole blocks and with a partial last block in each
    dimension, equal the reference formula in every float32 bit."""
    rng = np.random.default_rng(0)
    weight = (rng.standard_normal(shape) * 64).astype(np.float32).astype(FP8)
    weight[0, :3] = np.array([0.0, -0.0, 2 ** -9], FP8)
    blocks = (-(-shape[0] // BLOCK), -(-shape[1] // BLOCK))
    scale_inv = np.exp2(rng.integers(-12, 4, size=blocks)).astype(np.float32) * 1.7

    out = dequantize_fp8_blocks(weight, scale_inv)

    assert np.array_equal(bits(out), bits(reference(weight, scale_inv, BLOCK)))


def test_a_widened_weight_dequantizes_like_the_fp8_one():
    """The loader reads every tensor as fp32; fp8 widens exactly, so the
    dequantized weight is the same either way."""
    rng = np.random.default_rng(1)
    weight = (rng.standard_normal((130, 140)) * 8).astype(np.float32).astype(FP8)
    scale_inv = rng.uniform(0.01, 0.1, size=(2, 2)).astype(np.float32)

    assert np.array_equal(bits(dequantize_fp8_blocks(weight.astype(np.float32), scale_inv)),
                          bits(dequantize_fp8_blocks(weight, scale_inv)))


def test_a_scale_grid_that_does_not_fit_the_weight_is_refused():
    weight = np.zeros((576, 256), FP8)
    with pytest.raises(ValueError, match=r"\(5, 2\) scale, got \(4, 2\)"):
        dequantize_fp8_blocks(weight, np.ones((4, 2), np.float32))


def test_the_loader_hook_dequantizes_every_paired_weight_and_nothing_else():
    """A checkpoint read as fp32: the weight with a `_scale_inv` partner
    leaves dequantized and the partner is gone; every other tensor is the
    same object it was."""
    rng = np.random.default_rng(2)
    quantized = (rng.standard_normal((256, 256)) * 8).astype(np.float32).astype(FP8)
    scale_inv = rng.uniform(0.01, 0.1, size=(2, 2)).astype(np.float32)
    dense = rng.standard_normal((16, 16)).astype(np.float32)
    norm = np.ones((16,), np.float32)
    tensors = {
        "model.layers.0.mlp.up_proj.weight": quantized.astype(np.float32),
        "model.layers.0.mlp.up_proj.weight_scale_inv": scale_inv,
        "model.layers.0.self_attn.o_proj.weight": dense,
        "model.norm.weight": norm,
    }

    out = dequantize_checkpoint(tensors, BLOCK)

    assert set(out) == {"model.layers.0.mlp.up_proj.weight",
                        "model.layers.0.self_attn.o_proj.weight", "model.norm.weight"}
    assert np.array_equal(bits(out["model.layers.0.mlp.up_proj.weight"]),
                          bits(dequantize_fp8_blocks(quantized, scale_inv)))
    assert out["model.layers.0.self_attn.o_proj.weight"] is dense
    assert out["model.norm.weight"] is norm
    assert "model.layers.0.mlp.up_proj.weight_scale_inv" in tensors, "the input was mutated"


def test_a_scale_without_its_weight_is_refused():
    with pytest.raises(ValueError, match="scales model.layers.0.mlp.up_proj.weight, which"):
        dequantize_checkpoint({"model.layers.0.mlp.up_proj.weight_scale_inv": np.ones((1, 1))},
                              BLOCK)


def test_a_scale_in_a_checkpoint_whose_config_names_no_block_is_refused():
    with pytest.raises(ValueError, match="names no fp8 quantization_config"):
        dequantize_checkpoint({"a.weight": np.ones((4, 4)), "a.weight_scale_inv": np.ones((1, 1))},
                              None)


DEEPSEEK_V3 = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
               "weight_block_size": [128, 128]}
DEEPSEEK_V32 = {**DEEPSEEK_V3, "scale_fmt": "ue8m0"}


def test_the_config_names_the_block_for_fp8_and_nothing_for_the_rest():
    """The two DeepSeek configs as they are on the hub give 128; no
    quantization_config and another method (GPT-OSS's mxfp4) give None."""
    assert fp8_block({"quantization_config": DEEPSEEK_V3}) == 128
    assert fp8_block({"quantization_config": DEEPSEEK_V32}) == 128
    assert fp8_block({"model_type": "llama"}) is None
    assert fp8_block({"quantization_config": {"quant_method": "mxfp4"}}) is None


@pytest.mark.parametrize("field, value", [("fmt", "e5m2"), ("weight_block_size", None),
                                          ("weight_block_size", [1, 128]),
                                          ("weight_block_size", [128.0, 128.0]),
                                          ("weight_block_size", [True, True]),
                                          ("weight_block_size", [0, 0])])
def test_an_fp8_config_outside_deepseeks_format_is_refused(field, value):
    with pytest.raises(ValueError, match="square blocks and nothing else"):
        fp8_block({"quantization_config": {**DEEPSEEK_V3, field: value}})


def write_safetensors(path, tensors):
    """One safetensors file from named arrays, fp8 included, without torch."""
    names = {np.dtype(np.float32): "F32", np.dtype(FP8): "F8_E4M3"}
    header, offset = {}, 0
    for name, array in tensors.items():
        header[name] = {"dtype": names[array.dtype], "shape": list(array.shape),
                        "data_offsets": [offset, offset + array.nbytes]}
        offset += array.nbytes
    encoded = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for array in tensors.values():
            handle.write(np.ascontiguousarray(array).tobytes())


def torch_per_block_cast(weight, block, scale_fmt):
    """`per_block_cast_to_fp8` of deep_gemm/utils/math.py, run in torch.

    The reference operation for operation, `ceil_to_ue8m0` and the padding
    out to whole blocks included, giving back the raw bytes and the scales
    so the comparison never passes through ml_dtypes.
    """
    import torch

    def ceil_to_ue8m0(value):
        raw = value.abs().float().view(torch.int)
        exponent = ((raw >> 23) & 0xFF) + (raw & 0x7FFFFF).bool().int()
        return (exponent.clamp(1, 254) << 23).view(torch.float)

    values = torch.from_numpy(np.ascontiguousarray(weight, np.float32))
    rows, cols = values.shape
    padded = torch.zeros((-(-rows // block) * block, -(-cols // block) * block),
                         dtype=values.dtype)
    padded[:rows, :cols] = values
    view = padded.view(-1, block, padded.size(1) // block, block)
    amax = view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    scale = amax / 448.0
    if scale_fmt == "ue8m0":
        scale = ceil_to_ue8m0(scale)
    cast = (view * (1.0 / scale)).to(torch.float8_e4m3fn)
    return (cast.view_as(padded)[:rows, :cols].contiguous().view(torch.uint8).numpy(),
            scale.view(view.size(0), view.size(2)).numpy())


def decode_e4m3fn(codes):
    """E4M3FN bytes as float32, from their bit fields alone.

    OCP FP8's E4M3: a sign bit, a four-bit exponent biased by seven, three
    mantissa bits, no infinity, and S.1111.111 its only NaN. Spelled out
    here so what a file holds is read by something that shares no code with
    the writer, ml_dtypes and torch alike.
    """
    codes = np.asarray(codes, np.uint8)
    exponent = ((codes >> 3) & 0xF).astype(np.int32)
    mantissa = (codes & 0x7).astype(np.float32)
    magnitude = np.where(exponent == 0, mantissa * np.float32(2.0 ** -9),
                         (np.float32(1.0) + mantissa / np.float32(8.0))
                         * np.exp2((exponent - 7).astype(np.float32)))
    signed = np.where((codes >> 7) == 1, -magnitude, magnitude).astype(np.float32)
    return np.where((exponent == 0xF) & (mantissa == 7), np.float32(np.nan), signed)


def read_safetensors(path):
    """One safetensors file as {name: (dtype name, shape, payload bytes)}."""
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
        body = handle.read()
    return {name: (meta["dtype"], tuple(meta["shape"]),
                   body[meta["data_offsets"][0]:meta["data_offsets"][1]])
            for name, meta in header.items() if name != "__metadata__"}


def expanded(scale_inv, shape, block):
    """A block scale grid stretched over the elements each scale covers."""
    grid = np.repeat(np.repeat(scale_inv, block, axis=0), block, axis=1)
    return grid[:shape[0], :shape[1]]


def test_a_checkpoint_with_block_scales_loads_dequantized(tmp_path):
    """The tiny DeepSeek fixture re-shipped the way the real one is: the
    config names the block, two projections are fp8 with `weight_scale_inv`
    partners (one of them with a partial block), everything else is as it
    was. `load_pretrained` lands the dequantized weight, transposed
    as every kernel is, and the untouched tensors bit for bit."""
    from dew.interop.hf_decoders import _read_shard
    directory = tmp_path / "fp8"
    shutil.copytree(FIXTURE, directory)
    block = 16
    config = json.loads((directory / "config.json").read_text())
    config["quantization_config"] = {**DEEPSEEK_V3, "weight_block_size": [block, block]}
    (directory / "config.json").write_text(json.dumps(config))
    tensors = _read_shard(directory / "model.safetensors")
    quantized = ("model.layers.0.mlp.up_proj.weight",        # [48, 32]: 3 x 2 blocks
                 "model.layers.0.self_attn.kv_b_proj.weight")  # [64, 8]: a partial column block
    shipped = dict(tensors)
    expected = {}
    for name in quantized:
        shipped[name], shipped[name + SCALE_SUFFIX] = quantize_fp8_blocks(tensors[name], block)
        expected[name] = dequantize_fp8_blocks(shipped[name], shipped[name + SCALE_SUFFIX], block)
        assert not np.array_equal(expected[name], tensors[name]), "quantization changed nothing"
    write_safetensors(directory / "model.safetensors", shipped)

    variables = load_pretrained(str(directory), dtype="float32",
                                attention_impl="reference").variables

    params = variables["params"]
    up = params["layers_0"]["mlp"]["up_proj"]["kernel"]
    kv_b = params["layers_0"]["self_attn"]["kv_b_proj"]["kernel"]
    assert np.array_equal(bits(up), bits(expected["model.layers.0.mlp.up_proj.weight"].T))
    assert np.array_equal(bits(kv_b), bits(expected["model.layers.0.self_attn.kv_b_proj.weight"].T))
    assert np.array_equal(params["layers_0"]["mlp"]["gate_proj"]["kernel"],
                          tensors["model.layers.0.mlp.gate_proj.weight"].T)


# --------------------------------------------------------------------------
# The encoder
# --------------------------------------------------------------------------

HAND = np.array([[448.0, -224.0, 896.0],
                 [0.0, -0.0, -7.0],
                 [28.0, 0.109375, 0.875],
                 [-14.0, 3.5, -0.0546875]], np.float32)
"""A 4 x 3 weight whose every 2 x 2 block has 448 * 2 ** k for its amax, so
each block's scale comes out exactly 2 ** k and every element scales onto an
e4m3 value exactly. The third column is a partial block of its own and the
second block holds both zeros."""


def test_a_hand_computed_encoding_with_a_partial_column_block():
    """Each block's amax maps to 448 and the rest of the block divides by
    the same scale. The scales are the four amaxes over 448 -- 1, 2, 1/16
    and 2 ** -9 -- the partial third column is scaled on its own, and -0.0
    keeps its sign because the encoding multiplies rather than compares.
    """
    codes, scale_inv = quantize_fp8_blocks(HAND, block=2)

    np.testing.assert_array_equal(
        scale_inv, np.array([[1.0, 2.0], [0.0625, 2.0 ** -9]], np.float32))
    np.testing.assert_array_equal(
        decode_e4m3fn(codes.view(np.uint8)),
        np.array([[448.0, -224.0, 448.0], [0.0, -0.0, -3.5],
                  [448.0, 1.75, 448.0], [-224.0, 56.0, -28.0]], np.float32))
    assert codes.view(np.uint8)[1, 1] == 0x80, "-0.0 was written as +0.0"
    np.testing.assert_array_equal(dequantize_fp8_blocks(codes, scale_inv, block=2), HAND)


@pytest.mark.parametrize("shape, block", [((256, 384), 128), ((576, 200), 128),
                                          ((5, 7), 128), ((130, 259), 128),
                                          ((48, 32), 16), ((7, 3), 2)])
@pytest.mark.parametrize("scale_fmt", [None, "ue8m0"], ids=["float32-scales", "ue8m0-scales"])
def test_the_encoder_writes_the_reference_casts_bytes(shape, block, scale_fmt):
    """`quantize_fp8_blocks` against `per_block_cast_to_fp8` in torch: whole
    blocks and a partial block in each dimension, magnitudes spread over 50
    binades so no two blocks share a scale, and both signed zeros and values
    below the amax floor among the elements."""
    rng = np.random.default_rng(0)
    weight = (rng.standard_normal(shape)
              * np.exp2(rng.integers(-30, 20, shape))).astype(np.float32)
    weight.flat[:4] = np.array([0.0, -0.0, 1e-12, -1e-12], np.float32)
    weight[0, :min(shape[1], 3)] = 0.0

    codes, scale_inv = quantize_fp8_blocks(weight, block, scale_fmt=scale_fmt)

    reference_codes, reference_scales = torch_per_block_cast(weight, block, scale_fmt)
    assert np.array_equal(codes.view(np.uint8), reference_codes)
    assert np.array_equal(scale_inv.view(np.uint32), reference_scales.view(np.uint32))


def test_a_zero_block_takes_the_amax_floors_scale():
    """The reference's `clamp(1e-4)`: without it an all-zero block's scale
    is zero and every element of it 0/0. With it the block has a positive
    scale, its zeros stay zeros, and both signed zeros keep their sign."""
    codes, scale_inv = quantize_fp8_blocks(
        np.array([[0.0, -0.0], [-0.0, 0.0]], np.float32), block=2)

    assert scale_inv[0, 0] == np.float32(AMAX_FLOOR) / np.float32(E4M3_MAX)
    np.testing.assert_array_equal(codes.view(np.uint8),
                                  np.array([[0, 0x80], [0x80, 0]], np.uint8))
    assert np.all(np.isfinite(dequantize_fp8_blocks(codes, scale_inv, block=2)))


@pytest.mark.parametrize("amax, code", [(1e-9, 2), (1e-10, 0)])
def test_a_block_under_the_amax_floor_scales_against_the_floor(amax, code):
    """A block whose amax is under the floor is not stretched to fill the
    range: it keeps the floor's scale, so 1e-9 lands two subnormal steps up
    and 1e-10 rounds away altogether. That is the reference's behaviour, and
    the reason a re-export cannot promise to carry a tiny block."""
    codes, scale_inv = quantize_fp8_blocks(np.full((2, 2), amax, np.float32), block=2)

    assert scale_inv[0, 0] == np.float32(AMAX_FLOOR) / np.float32(E4M3_MAX)
    assert np.array_equal(codes.view(np.uint8), np.full((2, 2), code, np.uint8))


def test_a_ue8m0_scale_is_the_power_of_two_the_float_scale_rounds_up_to():
    """`ceil_to_ue8m0` is an integer rule on the float32 bits. It agrees
    with 2 ** ceil(log2(scale)) evaluated in float64, leaves the mantissa
    empty, and only ever moves a scale up, which is what keeps the scaled
    magnitudes inside the format's range."""
    rng = np.random.default_rng(1)
    weight = (rng.standard_normal((64, 64))
              * np.exp2(rng.integers(-40, 40, (64, 64)))).astype(np.float32)

    _, float_scales = quantize_fp8_blocks(weight, 16)
    _, power_scales = quantize_fp8_blocks(weight, 16, scale_fmt="ue8m0")

    np.testing.assert_array_equal(
        power_scales,
        np.exp2(np.ceil(np.log2(float_scales.astype(np.float64)))).astype(np.float32))
    assert np.all(power_scales.view(np.uint32) & 0x7FFFFF == 0), "not powers of two"
    assert np.all(power_scales >= float_scales)
    assert np.all(power_scales < 2.0 * float_scales)


@pytest.mark.parametrize("scale_fmt", [None, "ue8m0"], ids=["float32-scales", "ue8m0-scales"])
def test_the_encoding_moves_a_weight_only_as_far_as_the_format_rounds(scale_fmt):
    """Nearest-even into e4m3 inside a block scaled by `scale_inv` cannot
    move an element further than half the gap between the codes it falls
    between: 2 ** -4 of its own magnitude where it is normal, half of the
    block's 2 ** -9 subnormal step where it is not. Nothing else may move
    it, and something has to, or no rounding happened at all."""
    rng = np.random.default_rng(2)
    weight = (rng.standard_normal((64, 48))
              * np.exp2(rng.integers(-8, 8, (64, 48)))).astype(np.float32)

    codes, scale_inv = quantize_fp8_blocks(weight, 16, scale_fmt=scale_fmt)

    error = np.abs(dequantize_fp8_blocks(codes, scale_inv, 16) - weight)
    step = expanded(scale_inv, weight.shape, 16)
    assert np.all(error <= np.maximum(np.abs(weight) * np.float32(2.0 ** -4),
                                      step * np.float32(2.0 ** -10)))
    assert np.max(error) > 0.0


@pytest.mark.parametrize("scale_fmt", [None, "ue8m0"], ids=["float32-scales", "ue8m0-scales"])
def test_the_encoding_does_not_depend_on_the_dtype_or_the_backend_it_arrives_on(scale_fmt):
    """A trained leaf reaches the writer as whatever the run held it in: a
    jax array, a bfloat16 master, an fp32 one. All of them widen exactly
    into the float32 the reference reduces and scales in, and the arithmetic
    is NumPy's on a host copy, so none of them can pick up XLA's
    flush-to-zero on CPU. Magnitudes down to 1e-30 are in the weight because
    that is where a subnormal quotient would appear if one could."""
    rng = np.random.default_rng(5)
    weight = (rng.standard_normal((48, 32)) * 1e-30).astype(np.float32)
    weight[0, 0] = 3.0

    narrowed = weight.astype(ml_dtypes.bfloat16)

    assert np.all(quantize_fp8_blocks(weight, 16, scale_fmt=scale_fmt)[1]
                  >= np.finfo(np.float32).tiny), "a subnormal scale"
    # The jax leaf holds the float32 values; the bfloat16 master holds its
    # own, which widen into float32 exactly. Each has to give the bytes its
    # float32 counterpart gives.
    for counterpart, arrived in ((weight, jax.numpy.asarray(weight)),
                                 (narrowed.astype(np.float32), narrowed)):
        expected = quantize_fp8_blocks(counterpart, 16, scale_fmt=scale_fmt)
        actual = quantize_fp8_blocks(arrived, 16, scale_fmt=scale_fmt)
        assert np.array_equal(actual[0].view(np.uint8), expected[0].view(np.uint8))
        assert np.array_equal(actual[1].view(np.uint32), expected[1].view(np.uint32))


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_a_weight_that_is_not_finite_is_refused_rather_than_written(value):
    """E4M3FN encodes no infinity, and a nonfinite element takes its block's
    amax and so its scale with it: the reference's own arithmetic would
    write the whole block as NaN under a scale of NaN. The writer names the
    tensor instead."""
    weight = np.ones((4, 4), np.float32)
    weight[2, 3] = value

    with pytest.raises(ValueError, match="1 value.*not finite"):
        quantize_fp8_blocks(weight, block=2)


@pytest.mark.parametrize("scale_fmt", ["e8m0", "float", "ue4m3", ""])
def test_a_scale_format_this_writer_does_not_produce_is_refused(scale_fmt):
    """A source declaring a scale format with no published rounding law here
    is refused at export rather than written under an invented one. Reading
    such a checkpoint is untouched: a stored scale is a float32 multiplier
    whatever rounded it, which is why `fp8_block` never looks at this."""
    with pytest.raises(ValueError, match="names no scale format"):
        quantize_fp8_blocks(np.ones((2, 2), np.float32), 2, scale_fmt=scale_fmt)


def test_a_weight_that_is_not_a_matrix_is_refused():
    with pytest.raises(ValueError, match=r"\[rows, cols\] weight, got shape \(2, 2, 2\)"):
        quantize_fp8_blocks(np.ones((2, 2, 2), np.float32))


def test_the_quantized_names_come_off_the_sources_own_scale_partners():
    """What a loader has to record before it dequantizes: which tensors the
    source shipped quantized, read off the `_scale_inv` partners in
    checkpoint order rather than guessed from a name or a shape. After
    `dequantize_checkpoint` the partners are gone and an fp8 weight is an
    fp32 array like every other one, so the record cannot be taken later."""
    tensors = {"model.embed_tokens.weight": np.ones((4, 4), np.float32),
               "model.layers.0.mlp.up_proj.weight": np.ones((4, 4), FP8),
               "model.layers.0.mlp.up_proj.weight_scale_inv": np.ones((1, 1), np.float32),
               "model.layers.0.input_layernorm.weight": np.ones(4, np.float32),
               "model.layers.0.self_attn.o_proj.weight": np.ones((4, 4), FP8),
               "model.layers.0.self_attn.o_proj.weight_scale_inv": np.ones((1, 1), np.float32)}

    names = scaled_names(tensors)

    assert names == ("model.layers.0.mlp.up_proj.weight",
                     "model.layers.0.self_attn.o_proj.weight")
    dense = dequantize_checkpoint(tensors, BLOCK)
    assert scaled_names(dense) == (), "the record cannot be taken after the load"
    assert all(dense[name].dtype == np.float32 for name in names)


PACK_CONFIG = {"quantization_config": {**DEEPSEEK_V3, "weight_block_size": [16, 16]}}


def test_the_packer_writes_the_named_tensors_and_leaves_the_rest():
    """`pack_fp8` re-encodes the names it is handed and nothing else, so a
    source that shipped its embeddings and its norms dense re-exports them
    dense, as the same arrays rather than as copies or casts of them."""
    rng = np.random.default_rng(3)
    quantized = "model.layers.0.mlp.up_proj.weight"
    tensors = {"model.embed_tokens.weight": rng.standard_normal((32, 16)).astype(np.float32),
               quantized: rng.standard_normal((48, 32)).astype(np.float32),
               "model.layers.0.input_layernorm.weight": np.ones(16, np.float32)}

    written = pack_fp8(tensors, (quantized,), PACK_CONFIG)

    assert set(written) == set(tensors) | {quantized + SCALE_SUFFIX}
    assert written[quantized].dtype == FP8
    assert written[quantized + SCALE_SUFFIX].shape == (3, 2)
    for name in ("model.embed_tokens.weight", "model.layers.0.input_layernorm.weight"):
        assert written[name] is tensors[name]
    assert len(tensors) == 3, "the input was mutated"


def test_the_packer_takes_its_block_and_scale_format_from_the_sources_config():
    """The export declares the format it was read under, so the block and
    the scale format come off the source's own `quantization_config` and not
    off anything the trained tensors suggest."""
    rng = np.random.default_rng(4)
    weight = rng.standard_normal((48, 32)).astype(np.float32)

    for block, scale_fmt in ((16, None), (32, "ue8m0"), (128, None), (48, "ue8m0")):
        quantization = {**DEEPSEEK_V3, "weight_block_size": [block, block],
                        **({} if scale_fmt is None else {"scale_fmt": scale_fmt})}
        written = pack_fp8({"a.weight": weight}, ("a.weight",),
                           {"quantization_config": quantization})
        codes, scales = quantize_fp8_blocks(weight, block, scale_fmt=scale_fmt)
        assert np.array_equal(written["a.weight"].view(np.uint8), codes.view(np.uint8))
        assert np.array_equal(written["a.weight" + SCALE_SUFFIX], scales)


def test_the_packer_refuses_a_recorded_tensor_it_was_not_handed():
    """A source tensor that was quantized and is missing from the export
    would otherwise be written dense under a config that calls it
    quantized, which is the lie this whole path exists to avoid."""
    with pytest.raises(ValueError, match="up_proj.weight was quantized"):
        pack_fp8({"a.weight": np.ones((16, 16), np.float32)},
                 ("model.layers.0.mlp.up_proj.weight",), PACK_CONFIG)


def test_the_packer_refuses_to_overwrite_an_existing_scale_partner():
    tensors = {"a.weight": np.ones((16, 16), np.float32),
               "a.weight_scale_inv": np.ones((1, 1), np.float32)}

    with pytest.raises(ValueError, match="a.weight_scale_inv is already"):
        pack_fp8(tensors, ("a.weight",), PACK_CONFIG)


def test_the_packer_refuses_a_config_that_declares_no_fp8_blocks():
    with pytest.raises(ValueError, match="this config declares none"):
        pack_fp8({"a.weight": np.ones((16, 16), np.float32)}, ("a.weight",), {})


# --------------------------------------------------------------------------
# A trained source, back out in the format it came in
# --------------------------------------------------------------------------

REEXPORT_BLOCK = 16

QUANTIZED_TENSORS = (
    "model.layers.0.mlp.up_proj.weight",            # [48, 32]: 3 x 2 whole blocks
    "model.layers.0.self_attn.kv_b_proj.weight",    # [64, 8]: a partial column block
    "model.layers.0.self_attn.q_a_proj.weight",     # [8, 32]: a partial row block
    "model.layers.1.mlp.experts.3.up_proj.weight",  # [16, 32]: one expert of a stack
)


def source_tensors(loaded, variables):
    """The dict `Pretrained.save` assembles from the source layouts: source
    names in source orientation, which is where a packer meets an export."""
    mode = loaded.model.layer_scalar
    return {**loaded.retained_tensors,
            **{layout.name: layout.export(variables, mode) for layout in loaded.weight_layouts}}


def one_training_step(variables):
    """Every leaf moved off the source's values, deterministically, so what
    goes out is not what came in."""
    rng = np.random.default_rng(0)
    return jax.tree_util.tree_map(
        lambda leaf: (np.asarray(leaf)
                      + 0.02 * rng.standard_normal(np.shape(leaf))).astype(np.float32),
        variables)


@pytest.fixture(scope="module", params=[None, "ue8m0"],
                ids=["float32-scales", "ue8m0-scales"])
def reexport(request, tmp_path_factory):
    """The tiny DeepSeek fixture shipped the way the real checkpoints are,
    loaded, trained, and written back out in its own format.

    Four projections carry fp8 blocks with `_scale_inv` partners: whole
    blocks, a partial column block, a partial row block, and one expert of a
    stacked mixture. The config declares the block, and for one of the two
    parameters V3.2's `scale_fmt`. Run once per format for the tests below,
    which take the pipeline apart rather than run it again.
    """
    from dew.interop.hf_decoders import _load_shards, _read_shard
    scale_fmt = request.param
    root = tmp_path_factory.mktemp("fp8-reexport")
    directory, destination = root / "source", root / "reexport"
    shutil.copytree(FIXTURE, directory)
    quantization = {**DEEPSEEK_V3, "weight_block_size": [REEXPORT_BLOCK] * 2,
                    **({} if scale_fmt is None else {"scale_fmt": scale_fmt})}
    config = {**json.loads((directory / "config.json").read_text()),
              "quantization_config": quantization}
    (directory / "config.json").write_text(json.dumps(config))
    shipped = dict(_read_shard(directory / "model.safetensors"))
    for name in QUANTIZED_TENSORS:
        shipped[name], shipped[name + SCALE_SUFFIX] = quantize_fp8_blocks(
            shipped[name], REEXPORT_BLOCK, scale_fmt=scale_fmt)
    write_safetensors(directory / "model.safetensors", shipped)

    names = scaled_names(_load_shards(directory))
    loaded = load_pretrained(str(directory), dtype="float32", attention_impl="reference")
    values = one_training_step(loaded.variables)
    dense = source_tensors(loaded, values)
    written = pack_fp8(dense, names, config)
    shutil.copytree(directory, destination)
    write_safetensors(destination / "model.safetensors", written)
    reloaded = load_pretrained(str(destination), dtype="float32", attention_impl="reference")
    return {"scale_fmt": scale_fmt, "names": names, "dense": dense, "written": written,
            "loaded": loaded, "values": values, "reloaded": reloaded,
            "destination": destination}


def test_the_recorded_names_are_the_ones_the_source_shipped_quantized(reexport):
    """Read off the written checkpoint in its own order. The tensors the
    fixture ships dense, the other three dozen, are not in the record."""
    assert reexport["names"] == QUANTIZED_TENSORS


def test_a_trained_source_writes_the_reference_cast_of_its_trained_weights(reexport):
    """The format claim, exact: every byte of every re-encoded tensor is the
    byte `per_block_cast_to_fp8` produces for the trained fp32 weight the
    layouts assembled, and every scale is the same float32."""
    for name in reexport["names"]:
        codes, scales = torch_per_block_cast(reexport["dense"][name], REEXPORT_BLOCK,
                                             reexport["scale_fmt"])
        assert np.array_equal(reexport["written"][name].view(np.uint8), codes), name
        assert np.array_equal(reexport["written"][name + SCALE_SUFFIX].view(np.uint32),
                              scales.view(np.uint32)), name


def test_the_written_file_decodes_through_a_reader_that_shares_no_code(reexport):
    """The bytes on disk rather than the arrays in memory: the header names
    F8_E4M3 and F32, decoding the payload by E4M3FN's bit fields gives the
    weight ml_dtypes holds, and multiplying by the file's own scales gives
    the dequantized weight in every float32 bit."""
    raw = read_safetensors(reexport["destination"] / "model.safetensors")

    for name in reexport["names"]:
        dtype, shape, payload = raw[name]
        scale_dtype, scale_shape, scale_payload = raw[name + SCALE_SUFFIX]
        assert (dtype, scale_dtype) == ("F8_E4M3", "F32"), name
        codes = np.frombuffer(payload, np.uint8).reshape(shape)
        scale_inv = np.frombuffer(scale_payload, "<f4").reshape(scale_shape)
        assert np.array_equal(bits(decode_e4m3fn(codes)),
                              bits(reexport["written"][name])), name
        assert np.array_equal(
            bits(decode_e4m3fn(codes) * expanded(scale_inv, shape, REEXPORT_BLOCK)),
            bits(dequantize_fp8_blocks(reexport["written"][name], scale_inv,
                                       REEXPORT_BLOCK))), name


def test_every_tensor_the_source_shipped_dense_is_written_unchanged(reexport):
    """A re-export in the source's format touches the recorded tensors and
    nothing else: the norms, the embeddings, the routing biases and the
    projections the source left dense go out as the trained float32 arrays
    they came from."""
    dense, written, names = reexport["dense"], reexport["written"], reexport["names"]

    untouched = [name for name in dense if name not in names]
    assert len(untouched) > 30
    for name in untouched:
        assert written[name] is dense[name], name


def test_reloading_the_re_export_recovers_the_blocks_it_wrote(reexport):
    """The loader over the export: each re-encoded tensor comes back as
    exactly the dequantization of the pair on disk, and each untouched one
    as the bytes that were written. This is where the format round trip
    closes; the values it closes on are the rounded ones, not the trained
    ones, which is the next test's business."""
    written, names = reexport["written"], reexport["names"]
    recovered = source_tensors(reexport["reloaded"], reexport["reloaded"].variables)

    for name in names:
        expected = dequantize_fp8_blocks(written[name], written[name + SCALE_SUFFIX],
                                         REEXPORT_BLOCK)
        assert np.array_equal(bits(recovered[name]), bits(expected)), name
    for name, value in reexport["dense"].items():
        if name not in names:
            assert recovered[name].dtype == value.dtype, name
            assert recovered[name].tobytes() == value.tobytes(), name


def test_the_re_export_loses_only_what_the_format_rounds(reexport):
    """The lossy claim, kept apart from the exact ones above. A trained fp32
    weight and the weight the export gives back differ by nearest-even into
    e4m3 inside the block's scale and by nothing else. Worst case measured
    over the four tensors is 0.94 of that bound and 5.9% of the value, for
    the float32 and the ue8m0 scales alike."""
    dense, written = reexport["dense"], reexport["written"]
    worst = 0.0

    for name in reexport["names"]:
        want, scale_inv = dense[name], written[name + SCALE_SUFFIX]
        error = np.abs(dequantize_fp8_blocks(written[name], scale_inv, REEXPORT_BLOCK) - want)
        step = expanded(scale_inv, want.shape, REEXPORT_BLOCK)
        assert np.all(error <= np.maximum(np.abs(want) * np.float32(2.0 ** -4),
                                          step * np.float32(2.0 ** -10))), name
        worst = max(worst, float(np.max(error / np.maximum(np.abs(want), 1e-30))))
    assert 0.0 < worst <= 2.0 ** -4


def test_the_re_exported_model_tracks_the_trained_model(reexport):
    """What that loss costs the model, which is a third claim again. The
    re-export is not the trained model: four of its projections keep four
    mantissa bits of the trained values. On the fixture's own prompts the
    logits move by 0.41 at most against a logit range of +/- 4.1, and 95.8%
    of the argmax choices are unchanged."""
    ids = np.asarray(np.load(FIXTURE / "input_ids.npy"), np.int32)

    trained = np.asarray(reexport["loaded"].model.apply(reexport["values"], ids))
    recovered = np.asarray(
        reexport["reloaded"].model.apply(reexport["reloaded"].variables, ids))

    assert 0.0 < np.max(np.abs(trained - recovered)) < 1.0
    assert np.mean(trained.argmax(-1) == recovered.argmax(-1)) >= 0.9


# --------------------------------------------------------------------------
# The real checkpoint
# --------------------------------------------------------------------------

REPO = "deepseek-ai/DeepSeek-V3"
SHARD = "model-00001-of-000163.safetensors"
# [576, 7168] with a [5, 56] scale: the one projection of the model whose
# rows do not fill their last block.
TENSOR = "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"

SOURCES = (("deepseek-ai/DeepSeek-V3", None), ("deepseek-ai/DeepSeek-V3.2-Exp", "ue8m0"))
"""The two released formats, and the `scale_fmt` each config declares. Layer
zero's kv_a_proj lives in the first shard of both."""


def fetch(url, start, end):
    import requests
    response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    response.raise_for_status()
    return response.content


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("DEW_NETWORK_TESTS") != "1",
                    reason="reads two tensors of DeepSeek-V3 from the hub; DEW_NETWORK_TESTS=1 runs it")
def test_deepseek_v3_kv_a_proj_dequantizes_like_the_reference():
    """The real tensor pair, read by byte range from the checkpoint's first
    shard: fp8 weight, fp32 scale with a partial block row, equal to the
    reference bit for bit, finite, and within fp8's range times the scales."""
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(REPO, SHARD)
    length = struct.unpack("<Q", fetch(url, 0, 7))[0]
    header = json.loads(fetch(url, 8, 7 + length))
    data = 8 + length

    def tensor(name, dtype):
        meta = header[name]
        start, end = meta["data_offsets"]
        raw = fetch(url, data + start, data + end - 1)
        return np.frombuffer(raw, dtype).reshape(meta["shape"])

    assert header[TENSOR]["dtype"] == "F8_E4M3"
    weight = tensor(TENSOR, FP8)
    scale_inv = tensor(TENSOR + "_scale_inv", np.float32)
    assert weight.shape == (576, 7168) and scale_inv.shape == (5, 56)

    out = dequantize_checkpoint({TENSOR: weight.astype(np.float32),
                                 TENSOR + "_scale_inv": scale_inv}, BLOCK)[TENSOR]

    assert np.array_equal(bits(out), bits(reference(weight, scale_inv, BLOCK)))
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 448 * np.max(scale_inv)


@pytest.mark.network
@pytest.mark.parametrize("repo, scale_fmt", SOURCES,
                         ids=["deepseek-v3", "deepseek-v32-exp"])
@pytest.mark.skipif(os.environ.get("DEW_NETWORK_TESTS") != "1",
                    reason="reads two tensors of a DeepSeek checkpoint from the hub; DEW_NETWORK_TESTS=1 runs it")
def test_the_encoder_reproduces_the_bytes_deepseek_shipped(repo, scale_fmt):
    """The oracle no synthetic weight can stand in for. DeepSeek's own
    quantizer wrote the shipped pair, so dequantizing it and running this
    encoder over the result has to land back on the shipped bytes -- and it
    does, scales and codes both, for V3's float32 scales and V3.2-Exp's
    ue8m0 scales, on the [576, 7168] projection whose last block row is
    partial.

    The two formats are not interchangeable and this says so: V3.2-Exp's
    scales are all powers of two, only 5% of its blocks hold a 448, and
    re-encoding it as though its config named no `scale_fmt` reproduces
    3.98% of its bytes. Reading the declared format off the config is
    therefore load-bearing, not decoration.
    """
    from huggingface_hub import hf_hub_url
    quantization = json.loads(fetch(hf_hub_url(repo, "config.json"), 0, 200_000)
                              )["quantization_config"]
    assert quantization.get("scale_fmt") == scale_fmt
    block = quantization["weight_block_size"][0]
    url = hf_hub_url(repo, SHARD)
    length = struct.unpack("<Q", fetch(url, 0, 7))[0]
    header = json.loads(fetch(url, 8, 7 + length))
    data = 8 + length

    def tensor(name, dtype):
        meta = header[name]
        start, end = meta["data_offsets"]
        return np.frombuffer(fetch(url, data + start, data + end - 1), dtype).reshape(meta["shape"])

    weight = tensor(TENSOR, FP8)
    scale_inv = tensor(TENSOR + SCALE_SUFFIX, np.float32)
    assert weight.shape == (576, 7168) and scale_inv.shape == (5, 56)
    assert np.all(scale_inv.view(np.uint32) & 0x7FFFFF == 0) == (scale_fmt == "ue8m0"), (
        "ue8m0 ships powers of two and the float32 format does not")

    codes, scales = quantize_fp8_blocks(
        dequantize_fp8_blocks(weight, scale_inv, block), block, scale_fmt=scale_fmt)

    assert np.array_equal(scales.view(np.uint32), scale_inv.view(np.uint32))
    assert np.array_equal(codes.view(np.uint8), weight.view(np.uint8))
    if scale_fmt is not None:
        wrong, _ = quantize_fp8_blocks(
            dequantize_fp8_blocks(weight, scale_inv, block), block)
        assert np.mean(wrong.view(np.uint8) == weight.view(np.uint8)) < 0.1
