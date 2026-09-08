"""What re-exporting a trained source in DeepSeek's fp8 block format costs.

Three numbers that are three different claims, reported apart because they
have different kinds of answer:

- **bytes**: the emitted weight against `per_block_cast_to_fp8` of
  deep_gemm/utils/math.py run in torch, and the emitted scales against its
  scales. Exact or broken; the number is a byte disagreement count.
- **weight**: the trained float32 weight against the weight the export
  gives back, as a fraction of the format's own rounding bound -- half the
  gap between neighbouring e4m3 codes inside the block's scale. Anything at
  or under 1 is the format rounding and nothing else.
- **logits**: what that rounding costs the model on the source's own
  prompts, against the trained model's logits, with the logit range beside
  it so the number can be read.

`--repo` measures the bytes claim against a released checkpoint instead of
a fixture: DeepSeek's own quantizer produced the shipped pair, so
dequantizing it and re-encoding has to reproduce it, and this is where that
is checked without a GPU or a full download -- two tensors by HTTP byte
range.

    python tools/fp8_reexport_budget.py --source tests/fixtures/hf/deepseek-v3-tiny
    python tools/fp8_reexport_budget.py --repo deepseek-ai/DeepSeek-V3.2-Exp
"""
import argparse
import json
import shutil
import struct
import tempfile
from pathlib import Path

import jax
import ml_dtypes
import numpy as np

from dew.interop import load_pretrained
from dew.interop.quantized import (SCALE_SUFFIX, dequantize_fp8_blocks, pack_fp8,
                                   quantize_fp8_blocks, scaled_names)

E4M3 = ml_dtypes.float8_e4m3fn

TENSOR = "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
SHARD = "model-00001-of-000163.safetensors"


def torch_reference(weight, block, scale_fmt):
    """`per_block_cast_to_fp8` in torch, as raw bytes and float32 scales."""
    import torch

    def ceil_to_ue8m0(value):
        raw = value.abs().float().view(torch.int)
        exponent = ((raw >> 23) & 0xFF) + (raw & 0x7FFFFF).bool().int()
        return (exponent.clamp(1, 254) << 23).view(torch.float)

    values = torch.from_numpy(np.ascontiguousarray(weight, np.float32))
    rows, cols = values.shape
    padded = torch.zeros((-(-rows // block) * block, -(-cols // block) * block), dtype=values.dtype)
    padded[:rows, :cols] = values
    view = padded.view(-1, block, padded.size(1) // block, block)
    scale = view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4) / 448.0
    if scale_fmt == "ue8m0":
        scale = ceil_to_ue8m0(scale)
    cast = (view * (1.0 / scale)).to(torch.float8_e4m3fn)
    return (cast.view_as(padded)[:rows, :cols].contiguous().view(torch.uint8).numpy(),
            scale.view(view.size(0), view.size(2)).numpy())


def write_safetensors(path, tensors):
    """One safetensors file from named arrays, fp8 included, without torch."""
    names = {np.dtype(np.float32): "F32", np.dtype(E4M3): "F8_E4M3"}
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


def source_tensors(loaded, variables):
    """The dict `Pretrained.save` assembles from the source layouts."""
    mode = loaded.model.layer_scalar
    return {**loaded.retained_tensors,
            **{layout.name: layout.export(variables, mode) for layout in loaded.weight_layouts}}


def fixture_budget(source, block, scale_fmt, quantized):
    """The three numbers, over a fixture shipped as a quantized source."""
    root = Path(tempfile.mkdtemp(prefix="fp8-budget-"))
    try:
        from dew.interop.hf_decoders import _load_shards, _read_shard
        directory, destination = root / "source", root / "reexport"
        shutil.copytree(source, directory)
        quantization = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                        "weight_block_size": [block, block],
                        **({} if scale_fmt is None else {"scale_fmt": scale_fmt})}
        config = {**json.loads((directory / "config.json").read_text()),
                  "quantization_config": quantization}
        (directory / "config.json").write_text(json.dumps(config))
        shipped = dict(_read_shard(directory / "model.safetensors"))
        for name in quantized:
            shipped[name], shipped[name + SCALE_SUFFIX] = quantize_fp8_blocks(
                shipped[name], block, scale_fmt=scale_fmt)
        write_safetensors(directory / "model.safetensors", shipped)

        names = scaled_names(_load_shards(directory))
        loaded = load_pretrained(str(directory), dtype="float32", attention_impl="reference")
        rng = np.random.default_rng(0)
        values = jax.tree_util.tree_map(
            lambda leaf: (np.asarray(leaf)
                          + 0.02 * rng.standard_normal(np.shape(leaf))).astype(np.float32),
            loaded.variables)
        dense = source_tensors(loaded, values)
        written = pack_fp8(dense, names, config)
        shutil.copytree(directory, destination)
        write_safetensors(destination / "model.safetensors", written)
        reloaded = load_pretrained(str(destination), dtype="float32", attention_impl="reference")

        disagreements, slack, relative = 0, 0.0, 0.0
        for name in names:
            codes, scales = torch_reference(dense[name], block, scale_fmt)
            disagreements += int(np.count_nonzero(written[name].view(np.uint8) != codes))
            disagreements += int(np.count_nonzero(
                written[name + SCALE_SUFFIX].view(np.uint32) != scales.view(np.uint32)))
            want, scale_inv = dense[name], written[name + SCALE_SUFFIX]
            error = np.abs(dequantize_fp8_blocks(written[name], scale_inv, block) - want)
            step = np.repeat(np.repeat(scale_inv, block, 0), block, 1)[:want.shape[0], :want.shape[1]]
            bound = np.maximum(np.abs(want) * np.float32(2.0 ** -4), step * np.float32(2.0 ** -10))
            slack = max(slack, float(np.max(error / bound)))
            relative = max(relative, float(np.max(error / np.maximum(np.abs(want), 1e-30))))

        ids = np.asarray(np.load(Path(source) / "input_ids.npy"), np.int32)
        trained = np.asarray(loaded.model.apply(values, ids))
        recovered = np.asarray(reloaded.model.apply(reloaded.variables, ids))
        return {"scale_fmt": scale_fmt, "block": block, "tensors": len(names),
                "dense_tensors": len(dense) - len(names),
                "byte_disagreements_vs_reference": disagreements,
                "weight_error_over_format_bound": round(slack, 4),
                "weight_max_relative_error": round(relative, 5),
                "logit_max_shift": round(float(np.max(np.abs(trained - recovered))), 4),
                "logit_rms_shift": round(float(np.sqrt(np.mean((trained - recovered) ** 2))), 5),
                "logit_range": round(float(np.max(np.abs(trained))), 3),
                "argmax_unchanged": round(float(np.mean(trained.argmax(-1)
                                                        == recovered.argmax(-1))), 4)}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def repo_budget(repo):
    """The bytes claim against a released checkpoint, two tensors by range."""
    import requests
    from huggingface_hub import hf_hub_url

    def fetch(url, start, end):
        response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=300)
        response.raise_for_status()
        return response.content

    quantization = json.loads(fetch(hf_hub_url(repo, "config.json"), 0, 200_000))["quantization_config"]
    block, scale_fmt = quantization["weight_block_size"][0], quantization.get("scale_fmt")
    url = hf_hub_url(repo, SHARD)
    length = struct.unpack("<Q", fetch(url, 0, 7))[0]
    header = json.loads(fetch(url, 8, 7 + length))
    data = 8 + length

    def tensor(name, dtype):
        meta = header[name]
        start, end = meta["data_offsets"]
        return np.frombuffer(fetch(url, data + start, data + end - 1), dtype).reshape(meta["shape"])

    weight, scale_inv = tensor(TENSOR, E4M3), tensor(TENSOR + SCALE_SUFFIX, np.float32)
    dense = dequantize_fp8_blocks(weight, scale_inv, block)
    codes, scales = quantize_fp8_blocks(dense, block, scale_fmt=scale_fmt)
    mismatched, _ = quantize_fp8_blocks(dense, block,
                                        scale_fmt=None if scale_fmt else "ue8m0")
    return {"repo": repo, "tensor": TENSOR, "shape": list(weight.shape), "block": block,
            "declared_scale_fmt": scale_fmt,
            "shipped_scales_are_powers_of_two": bool(np.all(scale_inv.view(np.uint32) & 0x7FFFFF == 0)),
            "blocks_whose_amax_code_is_448": round(float(np.mean(
                np.maximum.reduceat(
                    np.maximum.reduceat(np.abs(weight.astype(np.float32)),
                                        np.arange(0, weight.shape[0], block), 0),
                    np.arange(0, weight.shape[1], block), 1) == 448.0)), 4),
            "code_bytes_reproduced": round(float(np.mean(
                codes.view(np.uint8) == weight.view(np.uint8))), 6),
            "scales_reproduced": bool(np.array_equal(scales.view(np.uint32),
                                                     scale_inv.view(np.uint32))),
            "code_bytes_under_the_other_scale_fmt": round(float(np.mean(
                mismatched.view(np.uint8) == weight.view(np.uint8))), 6)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("tests/fixtures/hf/deepseek-v3-tiny"),
                        help="a local source directory to ship quantized and re-export")
    parser.add_argument("--repo", help="a released checkpoint to check the encoder against")
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--quantize", nargs="*", default=[
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.self_attn.kv_b_proj.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.1.mlp.experts.3.up_proj.weight"])
    arguments = parser.parse_args()
    if arguments.repo:
        print(json.dumps(repo_budget(arguments.repo), indent=2))
        return
    for scale_fmt in (None, "ue8m0"):
        print(json.dumps(fixture_budget(arguments.source, arguments.block, scale_fmt,
                                        tuple(arguments.quantize)), indent=2))


if __name__ == "__main__":
    main()
