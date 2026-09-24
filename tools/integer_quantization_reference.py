"""Dew's AWQ and GPTQ decoding against the libraries' own, on released checkpoints.

    PYTHONPATH=src python tools/integer_quantization_reference.py <awq packing_utils.py> <out.json>

For each pinned repo: every quantized Linear decoded by Dew against the
library's torch dequantizer (autoawq 0.2.9 `dequantize_gemm`, loaded from the
file given, since the package imports its CUDA kernels; gptqmodel's
`TorchLinear.dequantize_weight` (7.5)), which must agree bit for bit; Dew's
logits against transformers' load of the same repo, in float32 and bfloat16,
and against transformers' plain float32 model on the libraries' dequantized
weights, which Dew's float32 logits meet to fp32 rounding (1.1e-4 to
2.6e-4 measured on the 0.5B models);
and a save of the untrained load, whose weight files must hold the source's
tensors byte for byte and which transformers must load to the same logits.
Needs torch, transformers and gptqmodel; runs where they are installed.
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from dew.interop import codecs, load_pretrained
from dew.interop.safetensors_io import read_weights

REPOS = {
    "Qwen/Qwen2.5-0.5B-Instruct-AWQ": "db09cd27ead7fee40cdee309693cf83601b9c899",
    "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4": "c34a4a91629f09f73a285f32dbd26106b033c654",
    "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8": "c68601e5424e69cdaa6e073673e3c94db27b4397",
}
IDS = np.array([[151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198]], np.int64)


def library_weight(awq_utils, config: dict, tensors: dict, name: str) -> np.ndarray:
    """The library's dequantized [out, in] weight, as its torch dequantizer returns it."""
    stem = name.removesuffix(".weight")
    part = {suffix: torch.from_numpy(np.array(tensors[stem + suffix])) for suffix in (".qweight", ".qzeros", ".scales")}
    bits, group = config["bits"], config["group_size"]
    if config["quant_method"] == "awq":
        return awq_utils.dequantize_gemm(part[".qweight"], part[".qzeros"], part[".scales"], bits, group).T.numpy()
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.utils.model import convert_gptq_v1_to_v2_format_module

    g_idx = torch.from_numpy(np.array(tensors[stem + ".g_idx"]))
    layer = TorchLinear(bits=bits, group_size=group, sym=config.get("sym", True), desc_act=config["desc_act"],
                             in_features=g_idx.shape[0], out_features=part[".scales"].shape[1], bias=False,
                             register_buffers=True)
    layer.qweight.data, layer.scales.data, layer.g_idx.data = part[".qweight"], part[".scales"], g_idx
    # A v1 checkpoint's zeros, converted by gptqmodel's own loader rule.
    layer.qzeros.data = part[".qzeros"]
    convert_gptq_v1_to_v2_format_module(layer, bits, torch.int32)
    layer.post_init()
    return layer.dequantize_weight().T.float().numpy()


def logits(directory: Path, dtype: str) -> np.ndarray:
    loaded = load_pretrained(directory, dtype=dtype, param_dtype=dtype, attention_impl="reference")
    return np.asarray(loaded.model.apply(loaded.variables, IDS.astype(np.int32)), np.float32)


def transformers_logits(directory: Path) -> np.ndarray:
    # float32 on CPU, so the reference carries no fp16 rounding of its own.
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32, device_map="cpu")
    with torch.no_grad():
        return model(torch.from_numpy(IDS)).logits.float().numpy()


def dense_logits(directory: Path, tensors: dict, library: dict[str, np.ndarray]) -> np.ndarray:
    """transformers' plain float32 model of the same architecture, every Linear
    dense, holding the library's own dequantized weights and the checkpoint's
    other tensors: the quantized model's function without a quantized kernel."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(directory)
    del config.quantization_config
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32).eval()
    parts = {part for name in library for suffix in (".qweight", ".qzeros", ".scales", ".g_idx")
             for part in (name.removesuffix(".weight") + suffix,)}
    state = {name: torch.from_numpy(np.asarray(value, np.float32)) for name, value in tensors.items()
             if name not in parts}
    state |= {name: torch.from_numpy(weight) for name, weight in library.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or [name for name in missing if name != "lm_head.weight"]:
        raise ValueError(f"dense reference: missing {missing[:4]}, unexpected {unexpected[:4]}")
    model.tie_weights()
    with torch.no_grad():
        return model(torch.from_numpy(IDS)).logits.numpy()


def main() -> None:
    spec = importlib.util.spec_from_file_location("awq_packing_utils", sys.argv[1])
    assert spec is not None and spec.loader is not None
    awq_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(awq_utils)
    report = {}
    for repo, revision in REPOS.items():
        directory = Path(snapshot_download(repo, revision=revision))
        config = json.loads((directory / "config.json").read_text())["quantization_config"]
        tensors = read_weights(directory)
        codec = codecs.source_quantization({"quantization_config": config})
        assert codec is not None
        names = codec.names(tensors)
        library = {name: library_weight(awq_utils, config, tensors, name).astype(np.float32) for name in names}
        worst = max(float(np.max(np.abs(codec.decode(tensors, name) - library[name]))) for name in names)
        theirs = transformers_logits(directory)
        dense = dense_logits(directory, tensors, library)
        ours = logits(directory, "float32")
        entry = {"revision": revision, "weights": len(names), "max_abs_vs_library": worst,
                 "logits_fp32_vs_dense_library_fp32": float(np.max(np.abs(ours - dense))),
                 "transformers_quantized_vs_dense_library_fp32": float(np.max(np.abs(theirs - dense))),
                 "logits_fp32_vs_transformers_fp32": float(np.max(np.abs(ours - theirs))),
                 "logits_bf16_vs_transformers_fp32": float(np.max(np.abs(logits(directory, "bfloat16") - theirs)))}
        with tempfile.TemporaryDirectory() as scratch:
            # 'auto' keeps each dense tensor in the dtype the source stored it in.
            loaded = load_pretrained(directory, dtype="float32", param_dtype="auto", attention_impl="reference")
            loaded.save(scratch)
            written = read_weights(scratch)
            entry["saved_bytes_differ"] = sorted(
                {*(set(written) ^ set(tensors)),
                 *(f"{name} {written[name].dtype}{written[name].shape} vs {tensors[name].dtype}{tensors[name].shape}"
                   for name in set(written) & set(tensors)
                   if written[name].dtype != tensors[name].dtype or not np.array_equal(
                       np.asarray(written[name]).view(np.uint8), np.asarray(tensors[name]).view(np.uint8)))})[:8]
            entry["saved_logits_vs_transformers"] = float(np.max(np.abs(transformers_logits(Path(scratch)) - theirs)))
        report[repo] = entry
        print(json.dumps({repo: entry}), flush=True)
    Path(sys.argv[2]).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
