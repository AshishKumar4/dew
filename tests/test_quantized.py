"""DeepSeek's block-scaled FP8, read against `weight_dequant` and written
against DeepGEMM's `per_block_cast_to_fp8`.

tools/deepseek_kernels_reference.py runs V3's Triton `weight_dequant`,
fetched at a pinned commit, on a CUDA GPU over fixed codes and scales, and
`dequantize_fp8_blocks` is held to its float32 bits (tests/fixtures/codecs/
deepseek_kernels.npz). tools/deepgemm_fp8_reference.py runs `per_block_cast_to_fp8` of DeepGEMM's
deep_gemm/utils/math.py, fetched at a pinned commit, on fixed operands, and
`quantize_fp8_blocks` is held to its bytes (tests/fixtures/codecs/
deepgemm_fp8.npz). `decode_e4m3fn` reads a written file by E4M3FN's bit
fields, sharing no code with ml_dtypes, torch or the writer. The network
tests run the encoder over DeepSeek's own shipped bytes.
"""

import json
import os
import shutil
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from interop_support import fetch, respelled

from dew.interop import Pretrained
from dew.interop.codecs import (
    BLOCK,
    SCALE_SUFFIX,
    dequantize_fp8_blocks,
    fp8_blocks,
    fp8_format,
    quantize_fp8_blocks,
    scaled_names,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hf" / "deepseek-v3-tiny"

FP8 = ml_dtypes.float8_e4m3fn


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


@pytest.mark.parametrize("shape", ["256x384", "130x259", "5x7"])
def test_blocks_dequantize_as_v3s_weight_dequant_kernel_does(shape):
    """V3's `weight_dequant`, run on a GPU: random E4M3 codes (subnormals and
    both zeros among them) under float32 scales over 60 binades, one of them
    2 ** -127 so its products are float32 subnormals, in whole blocks and a
    partial last block in each dimension. Every float32 bit equal."""
    with np.load(Path(__file__).resolve().parent / "fixtures" / "codecs" / "deepseek_kernels.npz") as kernels:
        codes, scales = kernels[f"v3/{shape}/codes"], kernels[f"v3/{shape}/scales"]
        expected = kernels[f"v3/{shape}/dequantized"]

    out = dequantize_fp8_blocks(codes.view(FP8), scales)

    assert np.array_equal(bits(out), bits(expected))


def test_a_widened_weight_dequantizes_like_the_fp8_one():
    """The loader reads every tensor as fp32; fp8 widens exactly, so the
    dequantized weight is the same either way."""
    rng = np.random.default_rng(1)
    weight = (rng.standard_normal((130, 140)) * 8).astype(np.float32).astype(FP8)
    scale_inv = rng.uniform(0.01, 0.1, size=(2, 2)).astype(np.float32)

    assert np.array_equal(bits(dequantize_fp8_blocks(weight.astype(np.float32), scale_inv)),
                          bits(dequantize_fp8_blocks(weight, scale_inv)))


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

    out = fp8_blocks(BLOCK).dequantize(tensors)

    assert set(out) == {"model.layers.0.mlp.up_proj.weight",
                        "model.layers.0.self_attn.o_proj.weight", "model.norm.weight"}
    assert np.array_equal(bits(out["model.layers.0.mlp.up_proj.weight"]),
                          bits(dequantize_fp8_blocks(quantized, scale_inv)))
    assert out["model.layers.0.self_attn.o_proj.weight"] is dense
    assert out["model.norm.weight"] is norm
    assert "model.layers.0.mlp.up_proj.weight_scale_inv" in tensors, "the input was mutated"


DEEPSEEK_V3 = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
               "weight_block_size": [128, 128]}
def write_safetensors(path, tensors):
    """One safetensors file from named arrays, fp8 included, without torch.

    Published by rename, as `safetensors_io._publish` does: a re-ship writes
    arrays that are memmaps of `path`, and truncating that inode under them
    is SIGBUS on the next read."""
    names = {np.dtype(np.float32): "F32", np.dtype(FP8): "F8_E4M3"}
    header, offset = {}, 0
    for name, array in tensors.items():
        header[name] = {"dtype": names[array.dtype], "shape": list(array.shape),
                        "data_offsets": [offset, offset + array.nbytes]}
        offset += array.nbytes
    encoded = json.dumps(header).encode()
    path = Path(path)
    temporary = path.with_name(f".{path.name}.writing")
    with open(temporary, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for array in tensors.values():
            handle.write(np.ascontiguousarray(array).tobytes())
    os.replace(temporary, path)


def deepgemm():
    """DeepGEMM's casts of the fixture's operands (tools/deepgemm_fp8_reference.py)."""
    with np.load(Path(__file__).resolve().parent / "fixtures" / "codecs" / "deepgemm_fp8.npz") as loaded:
        return dict(loaded)


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


@pytest.mark.parametrize("block", [(16, 16), (1, 16), (8, 16)], ids=["square", "row", "rectangle"])
def test_a_checkpoint_with_block_scales_loads_dequantized(tmp_path, block):
    """The tiny DeepSeek fixture re-shipped the way the real one is: the
    config names the block, square as DeepSeek's or any rectangle
    transformers' FineGrainedFP8Config allows, two projections are fp8 with
    `weight_scale_inv` partners (one of them with a partial block),
    everything else is as it was. `Pretrained.load` lands the dequantized
    weight, transposed as every kernel is, and the untouched tensors bit for bit."""
    from dew.interop.safetensors_io import read_file
    directory = respelled(FIXTURE, tmp_path / "fp8", lambda config: {
        **config, "quantization_config": {**DEEPSEEK_V3, "weight_block_size": list(block)}})
    tensors, _ = read_file(directory / "model.safetensors")
    quantized = ("model.layers.0.mlp.up_proj.weight",        # [48, 32]: 3 x 2 blocks
                 "model.layers.0.self_attn.kv_b_proj.weight")  # [64, 8]: a partial column block
    shipped = dict(tensors)
    expected = {}
    for name in quantized:
        shipped[name], shipped[name + SCALE_SUFFIX] = quantize_fp8_blocks(tensors[name], block)
        expected[name] = dequantize_fp8_blocks(shipped[name], shipped[name + SCALE_SUFFIX], block)
        assert not np.array_equal(expected[name], tensors[name]), "quantization changed nothing"
    write_safetensors(directory / "model.safetensors", shipped)

    variables = Pretrained.load(str(directory), dtype="float32",
                                attention_impl="reference").variables

    params = variables["params"]
    up = params["layers_0"]["mlp"]["up_proj"]["kernel"]
    kv_b = params["layers_0"]["self_attn"]["kv_b_proj"]["kernel"]
    assert np.array_equal(bits(up), bits(expected["model.layers.0.mlp.up_proj.weight"].T))
    assert np.array_equal(bits(kv_b), bits(expected["model.layers.0.self_attn.kv_b_proj.weight"].T))
    assert np.array_equal(params["layers_0"]["mlp"]["gate_proj"]["kernel"],
                          tensors["model.layers.0.mlp.gate_proj.weight"].T)


def test_an_fp8_checkpoint_loads_and_saves_without_the_prediction_depths_it_drops(tmp_path):
    """GLM's tiny fixture shipped as fp8 in its trunk and its prediction depth: `prediction_depths=0`
    leaves the depth's quantized weights and scales unread, so the load binds the trunk's scales alone,
    and its save requantizes the trunk and reloads to the same logits."""
    from dew.interop.safetensors_io import read_file
    directory = respelled(FIXTURE.parent / "glm4-moe-tiny", tmp_path / "fp8", lambda config: {
        **config, "quantization_config": {**DEEPSEEK_V3, "weight_block_size": [16, 16]}})
    tensors, _ = read_file(directory / "model.safetensors")
    shipped = dict(tensors)
    for name in ("model.layers.0.self_attn.q_proj.weight", "model.layers.2.eh_proj.weight"):
        shipped[name], shipped[name + SCALE_SUFFIX] = quantize_fp8_blocks(tensors[name], (16, 16))
    write_safetensors(directory / "model.safetensors", shipped)

    loaded = Pretrained.load(str(directory), dtype="float32", attention_impl="reference",
                             prediction_depths=0)
    assert loaded.model.num_nextn_predict_layers == 0
    loaded.save(tmp_path / "saved")
    reloaded = Pretrained.load(str(tmp_path / "saved"), dtype="float32", attention_impl="reference")
    assert reloaded.model.num_nextn_predict_layers == 0
    ids = jnp.arange(8, dtype=jnp.int32)[None] % loaded.model.vocab_size
    np.testing.assert_array_equal(np.asarray(reloaded.model.apply(reloaded.variables, ids)),
                                  np.asarray(loaded.model.apply(loaded.variables, ids)))


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


@pytest.mark.parametrize("case", ["random/256x384/128", "random/130x259/128", "random/5x7/128",
                                  "random/48x32/16", "random/7x3/2", "bf16/130x259/128"])
@pytest.mark.parametrize("ue8m0", [False, True], ids=["float32-scales", "ue8m0-scales"])
def test_the_encoder_writes_the_upstream_casts_bytes(case, ue8m0):
    """`quantize_fp8_blocks` against DeepGEMM's `per_block_cast_to_fp8`:
    whole blocks and a partial block in each dimension, magnitudes spread
    over 50 binades so no two blocks share a scale, both signed zeros and
    values below the amax floor among the elements, and a bfloat16 weight,
    every code byte and every float32 scale bit equal."""
    reference = deepgemm()
    weight = reference[f"{case}/weight"]
    if case.startswith("bf16"):
        weight = weight.view(ml_dtypes.bfloat16)
    suffix = "/ue8m0" if ue8m0 else ""

    codes, scale_inv = quantize_fp8_blocks(weight, int(case.rsplit("/", 1)[1]), ue8m0=ue8m0)

    assert np.array_equal(codes.view(np.uint8), reference[f"{case}/codes{suffix}"])
    assert np.array_equal(scale_inv.view(np.uint32), reference[f"{case}/scales{suffix}"].view(np.uint32))


@pytest.mark.parametrize("ue8m0", [False, True], ids=["float32-scales", "ue8m0-scales"])
def test_the_encoding_moves_a_weight_only_as_far_as_the_format_rounds(ue8m0):
    """Nearest-even into e4m3 inside a block scaled by `scale_inv` cannot
    move an element further than half the gap between the codes it falls
    between: 2 ** -4 of its own magnitude where it is normal, half of the
    block's 2 ** -9 subnormal step where it is not. Nothing else may move
    it, and something has to, or no rounding happened at all."""
    rng = np.random.default_rng(2)
    weight = (rng.standard_normal((64, 48))
              * np.exp2(rng.integers(-8, 8, (64, 48)))).astype(np.float32)

    codes, scale_inv = quantize_fp8_blocks(weight, 16, ue8m0=ue8m0)

    error = np.abs(dequantize_fp8_blocks(codes, scale_inv, 16) - weight)
    step = expanded(scale_inv, weight.shape, 16)
    assert np.all(error <= np.maximum(np.abs(weight) * np.float32(2.0 ** -4),
                                      step * np.float32(2.0 ** -10)))
    assert np.max(error) > 0.0


@pytest.mark.parametrize("ue8m0", [False, True], ids=["float32-scales", "ue8m0-scales"])
def test_the_encoding_does_not_depend_on_the_dtype_or_the_backend_it_arrives_on(ue8m0):
    """A jax leaf, a bfloat16 master and an fp32 array all widen exactly
    into the float32 the reference reduces in, on the host. Magnitudes down
    to 1e-30 are in the weight because that is where a subnormal quotient
    would appear if one could."""
    rng = np.random.default_rng(5)
    weight = (rng.standard_normal((48, 32)) * 1e-30).astype(np.float32)
    weight[0, 0] = 3.0

    narrowed = weight.astype(ml_dtypes.bfloat16)

    assert np.all(quantize_fp8_blocks(weight, 16, ue8m0=ue8m0)[1]
                  >= np.finfo(np.float32).tiny), "a subnormal scale"
    # The jax leaf holds the float32 values; the bfloat16 master holds its
    # own, which widen into float32 exactly. Each has to give the bytes its
    # float32 counterpart gives.
    for counterpart, arrived in ((weight, jax.numpy.asarray(weight)),
                                 (narrowed.astype(np.float32), narrowed)):
        expected = quantize_fp8_blocks(counterpart, 16, ue8m0=ue8m0)
        actual = quantize_fp8_blocks(arrived, 16, ue8m0=ue8m0)
        assert np.array_equal(actual[0].view(np.uint8), expected[0].view(np.uint8))
        assert np.array_equal(actual[1].view(np.uint32), expected[1].view(np.uint32))


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


@pytest.fixture(scope="module", params=[False, True], ids=["float32-scales", "ue8m0-scales"])
def reexport(request, tmp_path_factory):
    """The tiny DeepSeek fixture shipped the way the real checkpoints are,
    loaded, trained, and written back out in its own format.

    Four projections carry fp8 blocks with `_scale_inv` partners: whole
    blocks, a partial column block, a partial row block, and one expert of a
    stacked mixture. The config declares the block, and for one of the two
    parameters V3.2's `scale_fmt`. Run once per format for the tests below,
    which take the pipeline apart rather than run it again.
    """
    from dew.interop.safetensors_io import read_file
    from dew.interop.sources import load_shards
    ue8m0 = request.param
    root = tmp_path_factory.mktemp("fp8-reexport")
    destination = root / "reexport"
    quantization = {**DEEPSEEK_V3, "weight_block_size": [REEXPORT_BLOCK] * 2,
                    **({"scale_fmt": "ue8m0"} if ue8m0 else {})}
    directory = respelled(FIXTURE, root / "source",
                          lambda config: {**config, "quantization_config": quantization})
    shipped = dict(read_file(directory / "model.safetensors")[0])
    for name in QUANTIZED_TENSORS:
        shipped[name], shipped[name + SCALE_SUFFIX] = quantize_fp8_blocks(
            shipped[name], REEXPORT_BLOCK, ue8m0=ue8m0)
    write_safetensors(directory / "model.safetensors", shipped)

    names = scaled_names(load_shards(directory))
    loaded = Pretrained.load(str(directory), dtype="float32", attention_impl="reference")
    values = one_training_step(loaded.variables)
    dense = source_tensors(loaded, values)
    written = fp8_blocks(REEXPORT_BLOCK, ue8m0=ue8m0).requantize(dense, names)
    shutil.copytree(directory, destination)
    write_safetensors(destination / "model.safetensors", written)
    reloaded = Pretrained.load(str(destination), dtype="float32", attention_impl="reference")
    return {"ue8m0": ue8m0, "names": names, "dense": dense, "written": written,
            "loaded": loaded, "values": values, "reloaded": reloaded,
            "destination": destination}


def test_a_trained_source_writes_the_cast_of_its_trained_weights(reexport):
    """Every byte of every re-encoded tensor is the cast of the trained fp32
    weight the layouts assembled, and every scale the same float32: the
    export writes what `quantize_fp8_blocks` gives, which the fixture above
    holds to DeepGEMM's bytes."""
    for name in reexport["names"]:
        codes, scales = quantize_fp8_blocks(reexport["dense"][name], REEXPORT_BLOCK,
                                            ue8m0=reexport["ue8m0"])
        assert np.array_equal(reexport["written"][name].view(np.uint8), codes.view(np.uint8)), name
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
REVISIONS = {"deepseek-ai/DeepSeek-V3": "e815299b0bcbac849fa540c768ef21845365c9eb",
             "deepseek-ai/DeepSeek-V3.2-Exp": "194c67e12b1b0d6df0ef373ddcf215bc84027409"}
# [576, 7168] with a [5, 56] scale: the one projection of the model whose
# rows do not fill their last block.
TENSOR = "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"

SOURCES = (("deepseek-ai/DeepSeek-V3", None), ("deepseek-ai/DeepSeek-V3.2-Exp", "ue8m0"))
"""The two released formats, and the `scale_fmt` each config declares. Layer
zero's kv_a_proj lives in the first shard of both."""





@pytest.mark.network
@pytest.mark.skipif(os.environ.get("DEW_NETWORK_TESTS") != "1",
                    reason="reads two tensors of DeepSeek-V3 from the hub; DEW_NETWORK_TESTS=1 runs it")
def test_deepseek_v3_kv_a_proj_dequantizes_within_its_range():
    """The real tensor pair at a pinned commit, read by byte range from the
    checkpoint's first shard: fp8 weight, fp32 scale with a partial block
    row, finite, and within fp8's range times the scales. The decoder is
    held to the release kernel's bits by the fixture test above."""
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(REPO, SHARD, revision=REVISIONS[REPO])
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

    out = fp8_blocks(BLOCK).dequantize({TENSOR: weight.astype(np.float32),
                                        TENSOR + "_scale_inv": scale_inv})[TENSOR]

    assert np.array_equal(bits(out), bits(dequantize_fp8_blocks(weight, scale_inv)))
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 448 * np.max(scale_inv)


@pytest.mark.network
@pytest.mark.parametrize("repo, scale_fmt", SOURCES,
                         ids=["deepseek-v3", "deepseek-v32-exp"])
@pytest.mark.skipif(
    os.environ.get("DEW_NETWORK_TESTS") != "1",
    reason="reads two tensors of a DeepSeek checkpoint from the hub; DEW_NETWORK_TESTS=1 runs it",
)
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
    quantization = json.loads(fetch(hf_hub_url(repo, "config.json", revision=REVISIONS[repo]), 0, 200_000)
                              )["quantization_config"]
    assert quantization.get("scale_fmt") == scale_fmt
    block, ue8m0 = fp8_format(quantization)
    assert ue8m0 == (scale_fmt == "ue8m0")
    url = hf_hub_url(repo, SHARD, revision=REVISIONS[repo])
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
    assert np.all(scale_inv.view(np.uint32) & 0x7FFFFF == 0) == ue8m0, (
        "ue8m0 ships powers of two and the float32 format does not")

    codes, scales = quantize_fp8_blocks(
        dequantize_fp8_blocks(weight, scale_inv, block), block, ue8m0=ue8m0)

    assert np.array_equal(scales.view(np.uint32), scale_inv.view(np.uint32))
    assert np.array_equal(codes.view(np.uint8), weight.view(np.uint8))
    if ue8m0:
        wrong, _ = quantize_fp8_blocks(
            dequantize_fp8_blocks(weight, scale_inv, block), block)
        assert np.mean(wrong.view(np.uint8) == weight.view(np.uint8)) < 0.1
