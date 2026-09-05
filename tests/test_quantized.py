"""Block-scaled FP8 dequantization against DeepSeek's reference formula.

`weight_dequant` in deepseek-ai/DeepSeek-V3's inference/kernel.py computes
y[i, j] = float32(x[i, j]) * s[i // block, j // block] with the partial
blocks masked. The tests hold the port to that, bit for bit in float32, on a
hand-computed case and on random fp8 blocks, and hold the loader hook to
pairing each weight with its `_scale_inv` partner.
"""

import json
import os
import shutil
import struct
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from dew.interop import load_pretrained_decoder
from dew.interop.quantized import BLOCK, dequantize_checkpoint, dequantize_fp8_blocks, fp8_block

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


def quantize(weight, block=16):
    """A weight as DeepSeek ships one: fp8 blocks with their inverse scales,
    each block divided by its own max / 448 (inference/kernel.py act_quant's
    rule, applied per 2-D block)."""
    rows, cols = weight.shape
    scale_inv = np.zeros((-(-rows // block), -(-cols // block)), np.float32)
    quantized = np.zeros(weight.shape, FP8)
    for i in range(scale_inv.shape[0]):
        for j in range(scale_inv.shape[1]):
            tile = weight[i * block:(i + 1) * block, j * block:(j + 1) * block]
            scale_inv[i, j] = np.max(np.abs(tile)) / 448.0
            quantized[i * block:(i + 1) * block, j * block:(j + 1) * block] = (
                tile / scale_inv[i, j]).astype(FP8)
    return quantized, scale_inv


def test_a_checkpoint_with_block_scales_loads_dequantized(tmp_path):
    """The tiny DeepSeek fixture re-shipped the way the real one is: the
    config names the block, two projections are fp8 with `weight_scale_inv`
    partners (one of them with a partial block), everything else is as it
    was. `load_pretrained_decoder` lands the dequantized weight, transposed
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
        shipped[name], shipped[name + "_scale_inv"] = quantize(tensors[name], block)
        expected[name] = dequantize_fp8_blocks(shipped[name], shipped[name + "_scale_inv"], block)
        assert not np.array_equal(expected[name], tensors[name]), "quantization changed nothing"
    write_safetensors(directory / "model.safetensors", shipped)

    _, variables, _ = load_pretrained_decoder(str(directory), dtype="float32",
                                              attention_impl="reference")

    params = variables["params"]
    up = params["layers_0"]["mlp"]["up_proj"]["kernel"]
    kv_b = params["layers_0"]["self_attn"]["kv_b_proj"]["kernel"]
    assert np.array_equal(bits(up), bits(expected["model.layers.0.mlp.up_proj.weight"].T))
    assert np.array_equal(bits(kv_b), bits(expected["model.layers.0.self_attn.kv_b_proj.weight"].T))
    assert np.array_equal(params["layers_0"]["mlp"]["gate_proj"]["kernel"],
                          tensors["model.layers.0.mlp.gate_proj.weight"].T)


# --------------------------------------------------------------------------
# The real checkpoint
# --------------------------------------------------------------------------

REPO = "deepseek-ai/DeepSeek-V3"
SHARD = "model-00001-of-000163.safetensors"
# [576, 7168] with a [5, 56] scale: the one projection of the model whose
# rows do not fill their last block.
TENSOR = "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"


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
