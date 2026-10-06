#!/usr/bin/env python3
"""Write local NVFP4 input-QDQ checkpoints through CT 0.17.1's CPU reader.

The two files differ only in their declared input scale_dtype: absent
(as RedHatAI/Qwen3-32B-NVFP4 declares) or torch.float8_e4m3fn (CT's NVFP4
preset). Weights use the published bf16 decompression, then are promoted
to fp32 for the rounding comparison. Both files run through transformers
5.16.1 from_pretrained; only redundant weight QDQ is disabled, checked
bit for bit before recording. Input QDQ stays enabled.

Float64 widens the model's continuous operations. The quantizer itself
still computes its fp32 codes/scales before widening the QDQ result,
since changing a discontinuous quantizer's dtype changes its function.
Its fp32 result is separately recorded per Linear call, together with
the input, scale and weights, for a bit-for-bit codec/quantizer test.

Environment (~/.cache/dew/reference-venvs/nvfp4-storage): torch 2.14.0+cpu,
transformers 5.16.1, compressed-tensors 0.17.1, diffusers 0.34.0. Run:

    PYTHONPATH=src ~/.cache/dew/reference-venvs/nvfp4-storage/bin/python tools/nvfp4_qdq_reference.py
"""

import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme, QuantizationStatus
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.quantization.utils.helpers import generate_gparam
from diffusers_wan_reference import float64
from nvfp4_storage_reference import ARGS, compress
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "codecs" / "nvfp4_qdq"


def checkpoint(e4m3: bool) -> None:
    torch.manual_seed(2038)
    config = Qwen3Config(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                        max_position_embeddings=64, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    inputs = QuantizationArgs(**ARGS["nvfp4"], dynamic="local",
                              scale_dtype=torch.float8_e4m3fn if e4m3 else None)
    scheme = QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(**ARGS["nvfp4"]),
                                input_activations=inputs)
    ids = torch.tensor([[2, 8, 17, 5, 23, 9, 51, 43, 19, 6, 65, 11],
                        [31, 16, 3, 19, 42, 11, 78, 39, 61, 52, 36, 17]])
    calibration = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name != "lm_head":
            hooks.append(module.register_forward_pre_hook(
                lambda _, args, name=name: calibration.update({name: args[0].detach()})))
    with torch.no_grad():
        model(ids, use_cache=False)
    for hook in hooks:
        hook.remove()
    tensors = {name: value.clone() for name, value in model.state_dict().items() if name != "lm_head.weight"}
    for name in calibration:
        stored = compress(tensors.pop(name + ".weight"), scheme, "nvfp4")
        value = calibration[name]
        # Non-power-of-two globals catch both dropping it and reciprocal mistakes.
        stored["input_global_scale"] = generate_gparam(value.amin(), value.amax()).reshape(1)
        tensors.update({name + "." + part: value for part, value in stored.items()})
    directory = ROOT / ("e4m3" if e4m3 else "unrounded")
    directory.mkdir(parents=True, exist_ok=True)
    config.quantization_config = {
        "quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
        "quantization_status": "compressed", "dequantize": True,
        "config_groups": {"group_0": scheme.model_dump(mode="json")}, "ignore": ["lm_head"],
    }
    config.save_pretrained(directory)
    save_file(tensors, directory / "model.safetensors", metadata={"format": "pt"})
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32,
                                               attn_implementation="eager").eval()
    torch.nn.Module.float(model)
    with torch.no_grad():
        reference = model(ids, use_cache=False).logits.numpy()
    for module in model.modules():
        if hasattr(module, "quantization_status"):
            module.quantization_status = QuantizationStatus.COMPRESSED
    with torch.no_grad():
        np.testing.assert_array_equal(model(ids, use_cache=False).logits.numpy(), reference)
    arrays = {"ids": ids.numpy(), "logits": reference}
    for name, module in model.named_modules():
        if name not in calibration:
            continue
        value = calibration[name]
        with torch.no_grad():
            quantized = forward_quantize(module, value, "input", inputs)
            quantized_bf16 = forward_quantize(module, value.bfloat16(), "input", inputs)
        arrays[name + "/inputs"] = value.numpy()
        arrays[name + "/qdq"] = quantized.numpy()
        arrays[name + "/qdq_bf16"] = quantized_bf16.float().numpy()
        arrays[name + "/global"] = module.input_global_scale.numpy()
        arrays[name + "/weight"] = module.weight.detach().numpy()
    probe = torch.randn(4, 64, generator=torch.Generator().manual_seed(2041))
    probe[0] = 0
    probe[1] = -0.0
    probe[2, :16] = 1e-20
    probe[3, :16] = torch.tensor([0, -0.0, 0.25, -0.25, 0.75, -0.75, 1.25, -1.25,
                                  1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5, -5])
    module = model.model.layers[0].self_attn.q_proj
    with torch.no_grad():
        arrays["probe/inputs"] = probe.numpy()
        arrays["probe/qdq"] = forward_quantize(module, probe, "input", inputs).numpy()
        arrays["probe/qdq_bf16"] = forward_quantize(module, probe.bfloat16(), "input", inputs).float().numpy()
        arrays["probe/global"] = module.input_global_scale.numpy()
    exact = copy.deepcopy(model)
    direct_float = torch.Tensor.float

    def quantize_wide(module, value, base_name, args):
        # The quantizer defines the discrete function in fp32. Its stored
        # globals must stay fp32 even though the surrounding model widens.
        overall = module.input_global_scale
        module.input_global_scale = torch.nn.Parameter(direct_float(overall), requires_grad=False)
        try:
            return forward_quantize(module, direct_float(value), base_name, args).double()
        finally:
            module.input_global_scale = overall

    with float64(), torch.no_grad(), patch(
            "compressed_tensors.quantization.lifecycle.forward.forward_quantize", quantize_wide):
        arrays["logits_f64"] = torch.nn.Module.double(exact)(ids, use_cache=False).logits.numpy()
    for module in model.modules():
        module.quantization_enabled = False
    with torch.no_grad():
        arrays["weight_only_logits"] = model(ids, use_cache=False).logits.numpy()
    np.savez(directory / "reference.npz", **arrays)
    rms = float(np.sqrt(np.mean((reference - arrays["logits_f64"]) ** 2)))
    apart = float(np.sqrt(np.mean((reference - arrays["weight_only_logits"]) ** 2)))
    print(directory.name, "reference RMS from float64", rms,
          "QDQ RMS from weight-only", apart)


if __name__ == "__main__":
    checkpoint(e4m3=False)
    checkpoint(e4m3=True)
