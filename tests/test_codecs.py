"""The checkpoint codecs of `dew.interop.codecs`, held to their formats' own rules.

compressed-tensors' MXFP4 export is held to the bytes compressed-tensors
0.17.1 wrote for one fresh weight in three dtypes
(tools/compressed_tensors_mxfp4_reference.py).

DeepSeek-V4's `.scale` storage is read against the release's own
dequantization (inference/convert.py and model.py of
deepseek-ai/DeepSeek-V4-Flash and V4.1-Flash), transcribed to NumPy from the
formats' bit fields and sharing no code with the codec, and its FP4 encoder
against the release's kernel rule. deepseek-v4-tiny, stored the way V4-Flash
and V4-Flash-Base store theirs, loads to its decoded twin's variables and
saves back in its own storage. The network test holds both directions to
real tensors of V4-Flash, V4-Flash-Base and V4.1-Flash at pinned commits.
"""

import json
import os
import re
import struct
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
import pytest
from test_quantized import decode_e4m3fn, fetch

from dew.interop import codecs, load_pretrained
from dew.interop.safetensors_io import _STORED_DTYPES, read_weights, save_hf_layout

FIXTURES = Path(__file__).parent / "fixtures" / "hf"

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

    packed = codecs.PACKED_MXFP4.requantize({"m.weight": weight, "m.bias": bias}, ("m.weight",))

    assert set(packed) == {"m.weight_packed", "m.weight_scale", "m.bias"}
    assert packed["m.bias"] is bias
    np.testing.assert_array_equal(packed["m.weight_packed"], codes)
    np.testing.assert_array_equal(packed["m.weight_scale"], exponents)


@pytest.mark.parametrize("dtype", sorted(STORED))
def test_an_untrained_re_export_writes_the_same_values(dtype):
    """Decoding what the library wrote and encoding it again, in the same
    dtype, gives back the values and the exponents, atol 0: a nonzero
    group's largest quotient is 4 or 6, which the rule maps back to its
    exponent. Only a code 8 (-0.0) may come back as code 0, the zero
    point's doing, which is the same value."""
    _, codes, exponents = fixture_weight(dtype)
    source = {"m.weight_packed": codes, "m.weight_scale": exponents}
    decoded = codecs.PACKED_MXFP4.read(source, "m.weight")

    again = codecs.PACKED_MXFP4.requantize({"m.weight": decoded.astype(STORED[dtype])}, ("m.weight",))

    np.testing.assert_array_equal(codecs.PACKED_MXFP4.read(again, "m.weight"), decoded)
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
# DeepSeek-V4 `.scale` storage, against the release's own dequantization
# --------------------------------------------------------------------------

FP4_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)
"""inference/convert.py's FP4_TABLE, which reads code 8 as +0.0."""


def e8m0(scale: np.ndarray) -> np.ndarray:
    """E8M0 bytes as float32 2 ** (b - 127), 255 as NaN, from the bytes alone."""
    exponent = scale.view(np.uint8).astype(np.int32)
    return np.where(exponent == 255, np.float32(np.nan), np.ldexp(np.float32(1), exponent - 127)).astype(np.float32)


def release_scale(scale: np.ndarray) -> np.ndarray:
    """A `.scale` as float32: E8M0 exponents, or the Base releases' float32 powers of two."""
    return scale if scale.dtype == np.float32 else e8m0(scale)


def release_fp8(weight: np.ndarray, scale: np.ndarray, block: int) -> np.ndarray:
    """inference/model.py's FP8 `Linear`: element (i, j) times scale[i // block,
    j // block] over a ceil(out / block) x ceil(in / block) grid."""
    i, j = np.indices(weight.shape)
    return decode_e4m3fn(weight.view(np.uint8)) * release_scale(scale)[i // block, j // block]


def release_fp4(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """convert.py's read of an FP4 expert (`cast_e2m1fn_to_e4m3fn`): FP4_TABLE
    of each byte's low nibble, then its high one, times the scale of its 32 inputs."""
    codes = packed.view(np.uint8)
    values = np.stack([FP4_TABLE[codes & 0x0F], FP4_TABLE[(codes >> 4) & 0x0F]], axis=-1)
    return values.reshape(codes.shape[0], -1) * np.repeat(e8m0(scale), 32, axis=1)


def release_engram(weight: np.ndarray, scale: np.ndarray, block: int) -> np.ndarray:
    """model.py's `ParallelEngramEmbedding.forward`: values.float().unflatten(-1,
    (-1, block)) * scales.float().unsqueeze(-1)."""
    rows = decode_e4m3fn(weight.view(np.uint8)).reshape(weight.shape[0], -1, block)
    return (rows * release_scale(scale)[..., None]).reshape(weight.shape)


def e4m3_codes(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Random E4M3FN bytes, both NaN codes left out."""
    codes = rng.integers(0, 0x7f, shape, dtype=np.uint8) | rng.choice(np.uint8([0, 0x80]), shape)
    return codes.view(codecs.E4M3)


def e8m0_codes(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Random E8M0 bytes around 1, with byte 0 (2 ** -127) and byte 254 (2 ** 127) in."""
    exponents = rng.integers(110, 140, shape, dtype=np.uint8)
    exponents.flat[:2] = 0, 254
    return exponents.view(ml_dtypes.float8_e8m0fnu)


@pytest.mark.parametrize("block, shape", [(32, (70, 100)), (128, (130, 257))])
@pytest.mark.parametrize("scale_dtype", ["float8_e8m0fnu", "float32"])
def test_a_v4_fp8_linear_decodes_as_the_release_reads_it(block, shape, scale_dtype):
    """V4.1's 32 x 32 blocks and V4's 128 x 128, each shape ending on a
    partial block both ways, under E8M0 scales (V4-Flash, V4.1-Flash) and
    float32 ones (the Base releases): bit for bit, the overflow of 448 *
    2 ** 127 to infinity and the subnormal products of byte 0 included."""
    rng = np.random.default_rng(11)
    weight = e4m3_codes(rng, shape)
    scale = e8m0_codes(rng, (-(-shape[0] // block), -(-shape[1] // block)))
    if scale_dtype == "float32":
        scale = e8m0(scale)
    tensors = {"layers.0.attn.wq_a.weight": weight, "layers.0.attn.wq_a.scale": scale}

    with np.errstate(over="ignore"):
        decoded = codecs.deepseek_v4(block, fp4_experts=True).read(tensors, "layers.0.attn.wq_a.weight")
        expected = release_fp8(weight, scale, block)

    np.testing.assert_array_equal(decoded.view(np.uint32), expected.view(np.uint32))


def test_a_v4_fp4_expert_decodes_as_the_release_reads_it():
    """Every byte three times, so every code in both nibbles, under
    exponents down to byte 0, where 0.5 * 2 ** -127 is a float32 subnormal:
    the release's values, and its bits but for code 8, which its table
    reads as +0.0 and the codec as -0.0, so that it encodes back to 8."""
    rng = np.random.default_rng(12)
    packed = rng.permutation(np.tile(np.arange(256, dtype=np.uint8), 3)).reshape(6, 128).view(np.int8)
    scale = e8m0_codes(rng, (6, 8))
    tensors = {"mtp.0.ffn.experts.3.w2.weight": packed, "mtp.0.ffn.experts.3.w2.scale": scale}

    with np.errstate(over="ignore"):
        decoded = codecs.deepseek_v4(128, fp4_experts=True).read(tensors, "mtp.0.ffn.experts.3.w2.weight")
        expected = release_fp4(packed, scale)

    np.testing.assert_array_equal(decoded, expected)
    codes = np.stack([packed.view(np.uint8) & 15, packed.view(np.uint8) >> 4], axis=-1).reshape(6, -1)
    np.testing.assert_array_equal(np.signbit(decoded), np.signbit(expected) | (codes == 8))


def test_a_v4_1_engram_row_decodes_as_the_release_reads_it():
    """V4.1's n-gram hash table: a row of 256 values under one scale per 32."""
    rng = np.random.default_rng(13)
    weight, scale = e4m3_codes(rng, (9, 256)), e8m0_codes(rng, (9, 8))
    tensors = {"layers.14.engram.embed.weight": weight, "layers.14.engram.embed.scale": scale}

    with np.errstate(over="ignore"):
        decoded = codecs.deepseek_v4(32, fp4_experts=True).read(tensors, "layers.14.engram.embed.weight")
        expected = release_engram(weight, scale, 32)

    np.testing.assert_array_equal(decoded.view(np.uint32), expected.view(np.uint32))


def test_the_v4_fp4_encoder_follows_the_releases_kernel():
    """`fp4_quant_kernel` under E8M0 scales (inference/kernel.py), group by
    group. Amax 6: 6 * float32(1 / 6) rounds to exactly 1, byte 127, and
    0.75 ties to the even 1.0. Amax 3 rounds to exactly 0.5, byte 126, so
    3 is code 7 and -1.4 / 0.5 = -2.8 rounds to -3. Amax 3.015625 is
    0.5026 after the multiply, whose mantissa bits round it up to byte 127.
    A zero group takes the 6 * 2 ** -126 floor, byte 1, and keeps -0.0 as
    code 8. 0.7495 reaches the 0.75 tie through bf16 first. Amax 2 ** -120
    is 1.33 * 2 ** -123 over 6, byte 5, so 2 ** -120 is 4 (code 6)."""
    groups = np.zeros((6, 32), np.float32)
    groups[0, :3] = [6.0, -3.0, 0.75]
    groups[1, :2] = [3.0, -1.4]
    groups[2, :2] = [3.015625, 1.0]
    groups[3, 3] = -0.0
    groups[4, :2] = [0.7495, 6.0]
    groups[5, :2] = [2.0 ** -120, -(2.0 ** -122)]

    packed, scale = codecs.quantize_deepseek_v4_fp4(groups)

    assert (packed.dtype, packed.shape, scale.dtype, scale.shape) == (
        np.int8, (6, 16), ml_dtypes.float8_e8m0fnu, (6, 1))
    assert scale.view(np.uint8)[:, 0].tolist() == [127, 126, 127, 1, 127, 5]
    codes = np.stack([packed.view(np.uint8) & 15, packed.view(np.uint8) >> 4], axis=-1).reshape(6, 32)
    assert codes[:, :4].tolist() == [[7, 13, 2, 0], [7, 13, 0, 0], [5, 2, 0, 0],
                                     [0, 0, 0, 8], [2, 7, 0, 0], [6, 10, 0, 0]]


def test_each_v4_layout_moves_a_weight_by_at_most_its_grid_step_and_holds_still_after():
    """Fresh weights through `deepseek_v4(...).requantize` and back. An FP8 value moves
    at most half an E4M3 step, 2 ** -4 of itself, or 2 ** -10 of its
    block's scale below E4M3's normals. An FP4 value moves at most 2 ** -9
    of itself in the bf16 rounding plus half an E2M1 step at its group's
    scale s, which is s / 4 below 1 and a quarter of the value above. A
    second pass writes the first pass's values exactly."""
    rng = np.random.default_rng(14)
    dense = {"layers.0.attn.wkv.weight": rng.standard_normal((70, 100)).astype(np.float32),
             "layers.0.ffn.experts.1.w3.weight": rng.standard_normal((40, 96)).astype(np.float32),
             "layers.1.engram.embed.weight": rng.standard_normal((9, 256)).astype(np.float32),
             "layers.0.ffn.gate.weight": rng.standard_normal((4, 100)).astype(np.float32)}
    names = ("layers.0.attn.wkv.weight", "layers.0.ffn.experts.1.w3.weight", "layers.1.engram.embed.weight")
    read = codecs.deepseek_v4(32, fp4_experts=True).read

    stored = codecs.deepseek_v4(32, fp4_experts=True).requantize(dense, names)

    assert stored["layers.0.ffn.gate.weight"] is dense["layers.0.ffn.gate.weight"]
    assert [stored[name].dtype for name in names] == [codecs.E4M3, np.int8, codecs.E4M3]
    for name in names:
        decoded, weight = read(stored, name), dense[name]
        scale = release_scale(stored[name.removesuffix("weight") + "scale"])
        if name.endswith("w3.weight"):
            group_scale = np.repeat(scale, 32, axis=1)
            rounded = weight.astype(ml_dtypes.bfloat16).astype(np.float32)
            bound = np.abs(weight) * 2.0 ** -9 + np.maximum(group_scale / 4, np.abs(rounded) / 4)
        else:
            unit = 32
            cell = scale[np.arange(weight.shape[0])[:, None] // (1 if "engram" in name else unit),
                         np.arange(weight.shape[1])[None, :] // unit]
            bound = np.maximum(np.abs(weight) * 2.0 ** -4, cell * 2.0 ** -10)
        assert np.all(np.abs(decoded - weight) <= bound), name
        again = codecs.deepseek_v4(32, fp4_experts=True).requantize({name: decoded}, (name,))
        np.testing.assert_array_equal(read(again, name), decoded)


E4M3_PAIR = np.zeros((2, 32), codecs.E4M3)
E8M0_BYTE = np.ones((1, 1), ml_dtypes.float8_e8m0fnu)
FP8_CONFIG = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}
EXPERT = "layers.0.ffn.experts.0.w1.weight"


def v4_read(tensors: dict[str, np.ndarray], name: str, fp4_experts: bool = True) -> np.ndarray:
    return codecs.deepseek_v4(32, fp4_experts=fp4_experts).read(tensors, name)


@pytest.mark.parametrize("refused, message", [
    (lambda: codecs.source_quantization({"quantization_config": {**FP8_CONFIG, "scale_fmt": "float"},
                                         "expert_dtype": "fp4"}), "scale_fmt 'float'.*'ue8m0'"),
    (lambda: codecs.source_quantization({"quantization_config": {**FP8_CONFIG, "scale_fmt": "ue8m0",
                                                                 "expert_dtype": "nvfp4"}}), "expert_dtype 'nvfp4'"),
    (lambda: codecs.deepseek_v4(128, fp4_experts=True).names({"a.scale": E8M0_BYTE}), r"a\.scale .*a\.weight"),
    (lambda: v4_read({EXPERT: E4M3_PAIR, EXPERT[:-6] + "scale": E8M0_BYTE}, EXPERT),
     r"experts\.0\.w1\.weight .*int8 \[out, in / 2\].*got float8_e4m3fn \(2, 32\)"),
    (lambda: v4_read({EXPERT: np.zeros((2, 16), np.int8), EXPERT[:-6] + "scale": E8M0_BYTE}, EXPERT, False),
     r"experts\.0\.w1\.weight .*float8_e4m3fn.*got int8 \(2, 16\)"),
    (lambda: v4_read({"l.wkv.weight": np.zeros((2, 32), ml_dtypes.bfloat16), "l.wkv.scale": E8M0_BYTE},
                     "l.wkv.weight"), r"l\.wkv\.weight .*got bfloat16 \(2, 32\)"),
    (lambda: v4_read({"l.engram.embed.weight": E4M3_PAIR, "l.engram.embed.scale": E8M0_BYTE}, "l.engram.embed.weight"),
     r"l\.engram\.embed\.weight .*\(2, 32\) and \(1, 1\)"),
    (lambda: codecs.deepseek_v4(128, fp4_experts=True).scale_dtype({"a.weight": E4M3_PAIR, "a.scale": E8M0_BYTE, "b.weight": E4M3_PAIR,
                                             "b.scale": np.ones((1, 1), np.float32)}),
     r"\['float32', 'float8_e8m0fnu'\]"),
], ids=["scale-fmt", "expert-dtype", "scale-without-weight", "fp4-expert-dtype", "pairs-under-fp8",
        "fp8-dtype", "engram-grid", "mixed-scale-dtypes"])
def test_a_v4_checkpoint_refuses_what_its_format_cannot_hold(refused, message):
    """Each refusal names the tensor or the field, and what it holds."""
    with pytest.raises(ValueError, match=message):
        refused()


V4_TINY = Path(__file__).parent / "fixtures" / "hf" / "deepseek-v4-tiny"

V4_QUANTIZED = re.compile(r"(attn\.(wq_a|wq_b|wkv|wo_a|wo_b)|attn\.indexer\.wq_b|ffn\.shared_experts\.w[123]"
                          r"|ffn\.experts\.\d+\.w[123]|^mtp\.\d+\.[eh]_proj)\.weight$")
"""The Linears DeepSeek-V4-Flash ships beside a `.scale` (the headers of its
shards at 60d8d707): every attention projection but the compressors', the
indexer's query, the shared and routed experts, and the prediction depth's
input projections."""


def v4_release_storage(directory: Path, experts: str, scale_dtype: str) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """deepseek-v4-tiny stored as V4-Flash stores its weights, under
    `directory`/quantized, and the same weights decoded by the release's
    formulas, dense, under `directory`/dense.

    An MX group is 32 inputs and the fixture's experts are 16 wide, so their
    width goes to 32, their tensors redrawn at the fixture's own spread.
    Seed 9 is searched: the stored weights route every token to the same
    experts in transformers 5.16.1 and in Dew, whose logits then agree to
    5.7e-6; eight of the first nine seeds leave a routing score tied closely
    enough that the two break it differently."""
    rng = np.random.default_rng(9)
    dense = {}
    for name, tensor in read_weights(V4_TINY).items():
        if "experts." in name:
            shape = (32, tensor.shape[1]) if name.endswith(("w1.weight", "w3.weight")) else (tensor.shape[0], 32)
            tensor = (rng.standard_normal(shape) * np.std(tensor)).astype(np.float32)
        dense[name] = tensor
    names = tuple(name for name in dense if V4_QUANTIZED.search(name))
    stored = codecs.deepseek_v4(128, fp4_experts=experts == "fp4", scale_dtype=scale_dtype).requantize(dense, names)
    decoded = dict(dense)
    for name in names:
        weight, scale = stored[name], stored[name.removesuffix("weight") + "scale"]
        decoded[name] = release_fp4(weight, scale) if weight.dtype == np.int8 else release_fp8(weight, scale, 128)
    config = {**json.loads((V4_TINY / "config.json").read_text()), "moe_intermediate_size": 32}
    released = json.loads((V4_TINY.parent / "deepseek-v4-flash" / "config.json").read_text())
    save_hf_layout(stored, {**config, "expert_dtype": experts,
                            "quantization_config": released["quantization_config"]}, directory / "quantized")
    save_hf_layout(decoded, config, directory / "dense")
    return stored, names


@pytest.mark.parametrize("experts, scale_dtype", [("fp4", "float8_e8m0fnu"), ("fp8", "float32")],
                         ids=["v4-flash", "v4-flash-base"])
def test_a_v4_checkpoint_in_the_release_storage_loads_and_saves_in_it(tmp_path, experts, scale_dtype):
    """V4-Flash's storage, FP4 routed experts under E8M0 scales, and
    V4-Flash-Base's, FP8 experts under float32 ones. The stored checkpoint
    loads to exactly the variables of its dense twin, whose weights the
    release's formulas decoded, so no `.scale` reaches the family. Saved
    untrained, it writes the source's names, dtypes and scale dtype, and
    weights that decode to the source's values."""
    stored, names = v4_release_storage(tmp_path, experts, scale_dtype)

    loaded = load_pretrained(tmp_path / "quantized", dtype="float32", attention_impl="reference")
    dense = load_pretrained(tmp_path / "dense", dtype="float32", attention_impl="reference")
    loaded.save(tmp_path / "export")

    assert set(loaded.quantized_tensors) == set(names)
    jax.tree_util.tree_map_with_path(
        lambda path, ours, theirs: np.testing.assert_array_equal(ours, theirs, err_msg=jax.tree_util.keystr(path)),
        loaded.variables, dense.variables)
    written = read_weights(tmp_path / "export")
    assert set(written) == set(stored)
    read = codecs.deepseek_v4(128, fp4_experts=experts == "fp4").read
    for name, value in stored.items():
        assert written[name].dtype == value.dtype, name
        if name in names:
            np.testing.assert_array_equal(read(written, name), read(stored, name), err_msg=name)
        elif not name.endswith(".scale"):
            np.testing.assert_array_equal(written[name], value, err_msg=name)


# --------------------------------------------------------------------------
# The releases' own tensors
# --------------------------------------------------------------------------

V4_RELEASES = {
    "deepseek-ai/DeepSeek-V4-Flash": "60d8d70770c6776ff598c94bb586a859a38244f1",
    "deepseek-ai/DeepSeek-V4-Flash-Base": "8855555deef230a27a21a8d6f294b7b7497759b6",
    "deepseek-ai/DeepSeek-V4.1-Flash": "dba1be0a40aa45a94ad051997016db3960a90277",
}
"""The pinned commits the network tests read."""


def released_tensor(repo: str, name: str, rows: tuple[int, int] | None) -> np.ndarray:
    """One tensor of a release, or a run of its rows, read by byte range from
    its shard at the pinned commit."""
    from huggingface_hub import hf_hub_download, hf_hub_url

    revision = V4_RELEASES[repo]
    index = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json", revision=revision)).read_text())
    url = hf_hub_url(repo, index["weight_map"][name], revision=revision)
    length = struct.unpack("<Q", fetch(url, 0, 7))[0]
    meta = json.loads(fetch(url, 8, 7 + length))[name]
    start, end = meta["data_offsets"]
    shape = list(meta["shape"])
    if rows is not None:
        row = (end - start) // shape[0]
        start, end, shape[0] = start + rows[0] * row, start + rows[1] * row, rows[1] - rows[0]
    return np.frombuffer(fetch(url, 8 + length + start, 8 + length + end - 1),
                         _STORED_DTYPES[meta["dtype"]]).reshape(shape)


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("DEW_NETWORK_TESTS") != "1",
                    reason="reads six tensors of DeepSeek-V4-Flash, V4-Flash-Base and V4.1-Flash and their "
                           "scales from the hub; "
                           "DEW_NETWORK_TESTS=1 runs it")
@pytest.mark.parametrize("repo, name, rows, block, fp4_experts", [
    ("deepseek-ai/DeepSeek-V4-Flash", "layers.0.attn.wkv.weight", None, 128, True),
    ("deepseek-ai/DeepSeek-V4-Flash", "layers.0.ffn.experts.0.w2.weight", None, 128, True),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", "layers.0.attn.wkv.weight", None, 128, False),
    ("deepseek-ai/DeepSeek-V4.1-Flash", "layers.0.attn.wkv.weight", None, 32, True),
    ("deepseek-ai/DeepSeek-V4.1-Flash", "layers.0.ffn.experts.0.w1.weight", None, 32, True),
    ("deepseek-ai/DeepSeek-V4.1-Flash", "layers.1.engram.embed.weight", (200_000_000, 200_002_048), 32, True),
], ids=["v4-fp8", "v4-fp4", "v4-base-fp8-float32-scale", "v41-fp8", "v41-fp4", "v41-engram"])
def test_a_released_v4_tensor_decodes_as_the_release_reads_it_and_encodes_back(repo, name, rows, block, fp4_experts):
    """Real tensors at the pinned commits, one engram table read as 2048 of
    its 384 M rows. Decoding matches the release's formulas bit for bit
    (FP4: in value, code 8 aside). Encoding the decoded weight writes the
    shipped bytes back: every FP4 group, whose largest code is 4 or 6, and
    every FP8 block or engram group but those whose largest code is 224,
    which the ceil rule moves to half the scale with the same values."""
    partner = name.removesuffix("weight") + "scale"
    weight, scale = released_tensor(repo, name, rows), released_tensor(repo, partner, rows)
    read = codecs.deepseek_v4(block, fp4_experts=fp4_experts).read
    layout = codecs.deepseek_v4_layout(name, fp4_experts)

    decoded = read({name: weight, partner: scale}, name)
    again = codecs.deepseek_v4(block, fp4_experts=fp4_experts, scale_dtype=scale.dtype.name).requantize(
        {name: decoded}, (name,))

    np.testing.assert_array_equal(read(again, name), decoded)
    if layout == "fp4":
        np.testing.assert_array_equal(decoded, release_fp4(weight, scale))
        np.testing.assert_array_equal(again[name].view(np.uint8), weight.view(np.uint8))
        np.testing.assert_array_equal(again[partner].view(np.uint8), scale.view(np.uint8))
        return
    expected = release_fp8(weight, scale, block) if layout == "blocks" else release_engram(weight, scale, block)
    np.testing.assert_array_equal(decoded.view(np.uint32), expected.view(np.uint32))
    magnitudes = np.abs(weight.astype(np.float32))
    if layout == "blocks":
        largest = magnitudes.reshape(scale.shape[0], block, scale.shape[1], block).max(axis=(1, 3))
    else:
        largest = magnitudes.reshape(*scale.shape, block).max(axis=-1)
    moved = again[partner].astype(np.float32) != scale.astype(np.float32)
    np.testing.assert_array_equal(moved, largest == 224)
    np.testing.assert_array_equal(again[partner].astype(np.float32)[moved], scale.astype(np.float32)[moved] / 2)
    kept = np.repeat(np.repeat(~moved, block if layout == "blocks" else 1, axis=0), block, axis=1)
    np.testing.assert_array_equal(again[name].view(np.uint8)[kept], weight.view(np.uint8)[kept])


# --------------------------------------------------------------------------
# AWQ and GPTQ integer groups
# --------------------------------------------------------------------------

def test_an_awq_word_holds_its_columns_in_awq_order():
    """Nibble j of an AWQ gemm word is column AWQ_ORDER[j], and the weight is
    (code - zero) * scale taken in fp16, transposed to torch's [out, in]."""
    word = np.array([[sum((j + 1) << (4 * j) for j in range(8))]], np.uint32).view(np.int32)
    zeros = np.array([[sum(1 << (4 * j) for j in range(8))]], np.uint32).view(np.int32)
    tensors = {"m.qweight": word, "m.qzeros": zeros, "m.scales": np.full((1, 8), 0.1, np.float16)}
    decoded = codecs.awq(4, 1).decode(tensors, "m.weight")
    codes = np.empty(8)
    codes[list(codecs.AWQ_ORDER)] = np.arange(1, 9)
    expected = ((codes - 1).astype(np.float32) * np.float32(np.float16(0.1))).astype(np.float16)
    np.testing.assert_array_equal(decoded, expected.astype(np.float32)[:, None])


def test_a_gptq_v1_zero_is_one_above_what_it_stores_and_its_groups_follow_g_idx():
    """GPTQ packs eight inputs per word down each column; a v1 ('gptq')
    checkpoint stores each zero one below itself, and g_idx puts each input
    in its group, which act-order permutes."""
    codes = np.array([[(i + c) % 16 for c in range(8)] for i in range(16)])  # [in, out]
    stored_zeros = np.stack([np.arange(8), 14 - np.arange(8)])  # [groups, out]
    scales = np.array([[0.5] * 8, [0.25] * 8], np.float16)
    groups = np.array([1, 0] * 8, np.int32)
    tensors = {"m.qweight": codecs._words(codes.T, 4).T, "m.qzeros": codecs._words(stored_zeros, 4),
               "m.scales": scales, "m.g_idx": groups}
    for v1, offset in ((True, 1), (False, 0)):
        decoded = codecs.gptq(4, v1=v1).decode(tensors, "m.weight")
        expected = (codes - (stored_zeros[groups] + offset)) * scales[groups].astype(np.float32)
        np.testing.assert_array_equal(decoded, expected.T, err_msg=f"v1={v1}")


def integer_checkpoint(directory: Path, method: str) -> dict[str, np.ndarray]:
    """qwen3-tiny's Linear weights as a 4-bit AWQ or GPTQ checkpoint in the
    libraries' layout, groups of 16 inputs with per-group min/max grids."""
    source = read_weights(FIXTURES / "qwen3-tiny")
    stored: dict[str, np.ndarray] = {}
    for name, weight in source.items():
        if not re.search(r"(q|k|v|o|gate|up|down)_proj\.weight$", name):
            stored[name] = weight
            continue
        w = np.asarray(weight, np.float32).T  # [in, out]
        grouped = w.reshape(-1, 16, w.shape[1])
        low, high = grouped.min(axis=1), grouped.max(axis=1)
        scales = ((high - low) / 15).astype(np.float16)
        zeros = np.clip(np.round(-low / scales.astype(np.float32)), 0, 15).astype(np.int32)
        codes = np.clip(np.round(grouped / scales[:, None] + zeros[:, None]), 0, 15).astype(np.int32)
        codes = codes.reshape(w.shape)
        stem = name.removesuffix(".weight")
        if method == "awq":
            order = list(codecs.AWQ_ORDER)
            def pack(values):
                return codecs._words(values.reshape(values.shape[0], -1, 8)[..., order].reshape(values.shape), 4)
            stored |= {stem + ".qweight": pack(codes), stem + ".qzeros": pack(zeros), stem + ".scales": scales}
        else:
            stored |= {stem + ".qweight": codecs._words(codes.T, 4).T, stem + ".qzeros": codecs._words(zeros - 1, 4),
                       stem + ".scales": scales, stem + ".g_idx": (np.arange(w.shape[0]) // 16).astype(np.int32)}
    config = json.loads((FIXTURES / "qwen3-tiny" / "config.json").read_text())
    config["quantization_config"] = ({"quant_method": "awq", "bits": 4, "group_size": 16, "version": "gemm",
                                      "zero_point": True} if method == "awq" else
                                     {"quant_method": "gptq", "bits": 4, "group_size": 16, "sym": False,
                                      "desc_act": False})
    save_hf_layout(stored, config, directory)
    for asset in (FIXTURES / "qwen3-tiny").glob("*.json"):
        if asset.name != "config.json":
            (directory / asset.name).write_bytes(asset.read_bytes())
    return stored


@pytest.mark.parametrize("method", ["awq", "gptq"])
def test_an_integer_checkpoint_saves_back_its_own_bytes_and_refuses_a_value_off_its_grid(tmp_path, method):
    """Loaded, an AWQ or GPTQ checkpoint runs on (code - zero) * scale; saved
    untrained it writes every stored tensor back byte for byte. A trained
    value its source grid cannot hold is refused: AutoAWQ's packing would
    spill it into the neighbouring codes and gptqmodel's would clamp it."""
    stored = integer_checkpoint(tmp_path / "source", method)
    loaded = load_pretrained(tmp_path / "source", dtype="float32", attention_impl="reference")
    loaded.save(tmp_path / "export")
    written = read_weights(tmp_path / "export")
    assert set(written) == set(stored)
    for name, value in stored.items():
        assert written[name].dtype == value.dtype, name
        np.testing.assert_array_equal(written[name], value, err_msg=name)

    variables = jax.tree.map(np.array, loaded.variables)
    kernel = variables["params"]["layers_0"]["self_attn"]["q_proj"]["kernel"]
    kernel[0, 0] = 1e3
    with pytest.raises(ValueError, match=r"q_proj\.weight: 1 trained values fall outside the source's 4-bit grid"):
        loaded.save(tmp_path / "trained", variables=variables)


# --------------------------------------------------------------------------
# compressed-tensors pack-, float- and int-quantized weights
# --------------------------------------------------------------------------

CT_FIXTURE = Path(__file__).parent / "fixtures" / "codecs" / "compressed_tensors.npz"
"""tools/compressed_tensors_reference.py's output: compressed-tensors 0.17.1's
own compressors and decompressors on nine weight schemes."""

CT_SCHEMES = ["pack_int4_group_sym", "pack_int4_group_asym", "pack_int4_actorder", "pack_int8_channel",
              "fp8_tensor", "fp8_channel", "fp8_block", "int8_channel", "nvfp4"]


def ct_case(label: str) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]],
                                 dict[str, np.ndarray], dict]:
    """One scheme's stored tensors, its requantized tensors (under the module
    name `m`) for the trained weight held in bfloat16 and in float32, those
    trained weights, and the quantization_config."""
    fixture = np.load(CT_FIXTURE)
    meta = json.loads(bytes(fixture["dtypes"]).decode())[label]
    views = {"bfloat16": ml_dtypes.bfloat16, "float8_e4m3fn": ml_dtypes.float8_e4m3fn}

    def read(key: str, part: str) -> np.ndarray:
        value = fixture[key]
        dtype = meta.get(part)
        return value.view(views[dtype]) if dtype in views else value

    parts = [key.split("/", 1)[1] for key in fixture.files if key.startswith(f"{label}/")
             and key.count("/") == 1 and not key.endswith(("dequantized", "trained"))]
    stored = {f"m.{part}": read(f"{label}/{part}", part) for part in parts}
    requantized = {kind: {f"m.{key.rsplit('/', 1)[1]}": read(key, key.rsplit("/", 1)[1]) for key in fixture.files
                          if key.startswith(f"{label}/{kind}/")} for kind in ("requantized", "requantized32")}
    trained = {"requantized": fixture[f"{label}/trained"].view(ml_dtypes.bfloat16),
               "requantized32": fixture[f"{label}/trained32"]}
    weights = {**meta["weights"], "dynamic": False}
    config = {"quantization_config": {"quant_method": "compressed-tensors", "format": meta["format"],
                                      "config_groups": {"group_0": {"targets": ["Linear"], "weights": weights}}}}
    return stored, requantized, trained, config


@pytest.mark.parametrize("label", CT_SCHEMES)
def test_a_compressed_tensors_weight_decodes_and_saves_as_the_library_does(label):
    """Decoded as compressed-tensors' own `decompress` returns it, bit for bit,
    and a trained weight, some of it past the code range, writes back the
    bytes the library's compressor writes against the same scales, whether the
    model holds it in bfloat16 or, fine-tuned in float32, in float32."""
    stored, requantized, trained, config = ct_case(label)
    codec = codecs.source_quantization(config, grid=stored)
    assert codec is not None and codec.names(stored) == ("m.weight",)
    np.testing.assert_array_equal(codec.decode(stored, "m.weight"),
                                  np.load(CT_FIXTURE)[f"{label}/dequantized"])
    for kind, weight in trained.items():
        written = codec.requantize({"m.weight": weight}, ("m.weight",))
        assert set(written) == set(requantized[kind]), kind
        for name, value in requantized[kind].items():
            assert written[name].dtype == value.dtype, (kind, name)
            np.testing.assert_array_equal(written[name].view(np.uint8), value.view(np.uint8), err_msg=f"{kind} {name}")


@pytest.mark.parametrize("change, message", [
    ({"format": "marlin-24"}, "compressed-tensors format 'marlin-24'"),
    ({"kv_cache_scheme": {"num_bits": 8}}, "kv_cache_scheme"),
    ({"config_groups": {"g": {"weights": {"num_bits": 4, "type": "int", "strategy": "group", "group_size": 32},
                              "input_activations": {"num_bits": 8, "type": "float", "dynamic": False}}}},
     "input_activations have dynamic=False"),
])
def test_a_compressed_tensors_config_this_loader_cannot_read_is_refused_by_name(change, message):
    _, _, _, config = ct_case("pack_int4_group_sym")
    config["quantization_config"] |= change
    with pytest.raises(ValueError, match=message):
        codecs.source_quantization(config)
