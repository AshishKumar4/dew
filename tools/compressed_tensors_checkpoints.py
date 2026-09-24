#!/usr/bin/env python3
"""Dew's compressed-tensors decoding against the library's on released checkpoints.

Two steps in two environments, since compressed-tensors needs torch and Dew
needs JAX:

    ~/.cache/dew/reference-venvs/kimi-k3/bin/python tools/compressed_tensors_checkpoints.py dense <out>
    PYTHONPATH=src python tools/compressed_tensors_checkpoints.py compare <out>

`dense` decompresses every quantized Linear of each pinned repo with the
library's own compressor (compressed-tensors 0.17.1 `decompress`) and writes a
dense checkpoint, config without quantization_config, to <out>/<repo>.
`compare` then reports, per repo: Dew's decode against those weights (max
|diff|, which must be 0); Dew's logits in float32 and bfloat16 against
transformers' plain float32 model on the dense checkpoint, the quantized
model's function with no quantized kernel; and a save of the untrained load,
whose stored tensors must equal the source's byte for byte.
"""

import json
import shutil
import sys
from pathlib import Path

REPOS = {
    "RedHatAI/Qwen3-0.6B-quantized.w4a16": "3bceedd23534ac8ef59952f72c94790b27938e42",
    "RedHatAI/Qwen3-0.6B-FP8-BLOCK": "5def00c475640c0527e3a7067b1a26b94decefb5",
    "apolloparty/Qwen3-0.6B-NVFP4A16": "9537cb799d1c61c761104b39833c4d66fd382219",
}
IDS = [[151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198]]


def dense(out: Path) -> None:
    import torch
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.quantization import QuantizationScheme
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    for repo, revision in REPOS.items():
        source = Path(snapshot_download(repo, revision=revision))
        config = json.loads((source / "config.json").read_text())
        quantization = config.pop("quantization_config")
        group = next(iter(quantization["config_groups"].values()))
        scheme = QuantizationScheme(targets=group["targets"], weights=group["weights"])
        compressor = BaseCompressor.load_from_registry(quantization["format"])
        tensors = {}
        for shard in sorted(source.glob("*.safetensors")):
            tensors |= load_file(shard)
        stems = sorted({name.removesuffix(".weight_scale") for name in tensors if name.endswith(".weight_scale")})
        for stem in stems:
            state = {name.removeprefix(stem + "."): tensors.pop(name) for name in list(tensors)
                     if name.startswith(stem + ".") and name.removeprefix(stem + ".").startswith("weight")}
            tensors[stem + ".weight"] = compressor.decompress(state, scheme)["weight"].to(torch.float32)
        target = out / repo.replace("/", "--")
        target.mkdir(parents=True, exist_ok=True)
        save_file({name: value.contiguous() for name, value in tensors.items()}, target / "model.safetensors")
        (target / "config.json").write_text(json.dumps(config, indent=2))
        for asset in source.glob("tokenizer*"):
            shutil.copy(asset, target / asset.name)
        print(repo, len(stems), "weights decompressed", flush=True)


def compare(out: Path) -> None:
    import tempfile

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    from dew.interop import codecs, load_pretrained
    from dew.interop.safetensors_io import read_weights

    ids = np.asarray(IDS, np.int32)
    report = {}
    for repo, revision in REPOS.items():
        source = Path(snapshot_download(repo, revision=revision))
        reference = out / repo.replace("/", "--")
        stored, library = read_weights(source), read_weights(reference)
        config = json.loads((source / "config.json").read_text())
        codec = codecs.source_quantization(config)
        assert codec is not None
        names = codec.names(stored)
        worst = max(float(np.max(np.abs(codec.decode(stored, name) - library[name]))) for name in names)
        model = AutoModelForCausalLM.from_pretrained(reference, dtype=torch.float32).eval()
        with torch.no_grad():
            theirs = model(torch.from_numpy(ids.astype(np.int64))).logits.numpy()
        del model
        entry = {"revision": revision, "weights": len(names), "max_abs_vs_library": worst}
        for dtype in ("float32", "bfloat16"):
            loaded = load_pretrained(source, dtype=dtype, param_dtype=dtype, attention_impl="reference")
            ours = np.asarray(loaded.model.apply(loaded.variables, ids), np.float32)
            entry[f"logits_{dtype}_vs_dense_library_fp32"] = float(np.max(np.abs(ours - theirs)))
        with tempfile.TemporaryDirectory() as scratch:
            loaded = load_pretrained(source, dtype="float32", param_dtype="auto", attention_impl="reference")
            loaded.save(scratch)
            written = read_weights(scratch)
            entry["saved_bytes_differ"] = sorted(
                {*(set(written) ^ set(stored)),
                 *(name for name in set(written) & set(stored)
                   if written[name].dtype != stored[name].dtype
                   or not np.array_equal(np.asarray(written[name]).view(np.uint8),
                                         np.asarray(stored[name]).view(np.uint8)))})[:8]
        report[repo] = entry
        print(json.dumps({repo: entry}), flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    {"dense": dense, "compare": compare}[sys.argv[1]](Path(sys.argv[2]))
