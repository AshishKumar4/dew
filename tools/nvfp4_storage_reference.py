#!/usr/bin/env python3
"""Write tiny compressed-tensors checkpoints and their same-file CPU references.

Weights are compressed by compressed-tensors 0.17.1 and read back through
transformers 5.16.1's from_pretrained. NVFP4's published decoder always
returns bf16 weights. Its bf16 CPU reader proves removing redundant weight
QDQ leaves that forward bit for bit unchanged. A second fp32 reader keeps
dense checkpoint tensors in fp32; only its decoded bf16 Linear weights need
promotion before the fp32 forward. FP8 is read and run directly in fp32.
The truth is those decoded weights run in float64, including the norm,
RoPE and softmax pins widened by diffusers_wan_reference.float64. NVFP4's
unrounded fp32 dequantize is recorded too, and its bf16 rounding must equal
the published decoder. There are no activation quantizers in these files.

The separate QDQ fixture records real compressed-tensors Linear forwards
with static FP8, dynamic FP8 and local NVFP4 inputs, and the same decoded
weights without input QDQ. Decompressing the weights leaves the activation
quantizer enabled. These three forwards demonstrate why a weight-only
loader must refuse each activation scheme.

Environment (~/.cache/dew/reference-venvs/nvfp4-storage): torch 2.14.0+cpu,
transformers 5.16.1, compressed-tensors 0.17.1, diffusers 0.34.0. Run:

    PYTHONPATH=src ~/.cache/dew/reference-venvs/nvfp4-storage/bin/python \
        tools/nvfp4_storage_reference.py
"""

from pathlib import Path

import numpy as np
import torch
from compressed_tensors.compressors.naive_quantized.base import FloatQuantizationCompressor
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme, QuantizationStatus
from compressed_tensors.quantization.lifecycle.forward import dequantize, set_forward_quantized
from compressed_tensors.quantization.utils.helpers import calculate_qparams, generate_gparam
from diffusers_wan_reference import float64
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "codecs" / "storage"
ARGS = {
    "fp8": {"num_bits": 8, "type": "float", "strategy": "tensor", "symmetric": True},
    "nvfp4": {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16,
              "symmetric": True},
}
COMPRESSORS = {"fp8": FloatQuantizationCompressor, "nvfp4": NVFP4PackedCompressor}
FORMATS = {"fp8": "float-quantized", "nvfp4": "nvfp4-pack-quantized"}


def compress(weight: torch.Tensor, scheme: QuantizationScheme, kind: str) -> dict[str, torch.Tensor]:
    """The library's calibrated grid and compressor, including NVFP4's global scale."""
    args = scheme.weights
    assert args is not None
    values = weight.float()
    if kind == "nvfp4":
        groups = values.unflatten(-1, (-1, 16))
        low, high = groups.amin(-1), groups.amax(-1)
        overall = generate_gparam(values.amin(), values.amax()).reshape(1)
    else:
        low, high, overall = values.amin().reshape(1), values.amax().reshape(1), None
    scale, zero = calculate_qparams(low, high, args, global_scale=overall)
    state = {"weight": weight, "weight_scale": scale, "weight_zero_point": zero}
    if overall is not None:
        state["weight_global_scale"] = overall
    return COMPRESSORS[kind].compress(state, scheme)


def checkpoint(kind: str) -> None:
    """Read the same files through transformers before recording logits or decoded weights."""
    torch.manual_seed(2037)
    config = Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                        max_position_embeddings=32, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    scheme = QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(**ARGS[kind]))
    tensors = {name: value.clone() for name, value in model.state_dict().items() if name != "lm_head.weight"}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or name == "lm_head":
            continue
        weight = tensors.pop(name + ".weight")
        tensors.update({name + "." + part: value for part, value in compress(weight, scheme, kind).items()})
    directory = ROOT / kind
    directory.mkdir(parents=True, exist_ok=True)
    config.quantization_config = {
        "quant_method": "compressed-tensors", "format": FORMATS[kind], "quantization_status": "compressed",
        "config_groups": {"group_0": scheme.model_dump(mode="json")}, "ignore": ["lm_head"],
        "dequantize": True,
    }
    config.save_pretrained(directory)
    save_file(tensors, directory / "model.safetensors", metadata={"format": "pt"})
    loaded = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32,
                                                attn_implementation="eager").eval()
    ids = torch.tensor([[2, 8, 17, 5, 23, 9], [31, 16, 3, 19, 42, 11]])
    published_reader = (AutoModelForCausalLM.from_pretrained(directory, dtype=torch.bfloat16,
                                                           attn_implementation="eager").eval()
                        if kind == "nvfp4" else loaded)
    with torch.no_grad():
        published = published_reader(ids, use_cache=False).logits.float().numpy()
    decoded = {name: value.detach().float().numpy() for name, value in loaded.state_dict().items()
               if name.endswith(".weight") and name != "lm_head.weight"}
    for module in published_reader.modules():
        module.quantization_enabled = False
    with torch.no_grad():
        after = published_reader(ids, use_cache=False).logits.float().numpy()
        np.testing.assert_array_equal(after, published)
    for module in loaded.modules():
        module.quantization_enabled = False
    with torch.no_grad():
        # PreTrainedModel.float refuses quantized models even after their
        # weights are decoded. Module.float promotes that decoded tree.
        reference = torch.nn.Module.float(loaded)(ids, use_cache=False).logits.numpy()
    if kind == "nvfp4":
        for name in tuple(decoded):
            stem = name.removesuffix(".weight")
            if stem + ".weight_packed" not in tensors:
                continue
            packed = tensors[stem + ".weight_packed"]
            values = unpack_fp4_from_uint8(packed, packed.shape[0], 2 * packed.shape[1]).float()
            unrounded = dequantize(values, tensors[stem + ".weight_scale"].float(),
                                   global_scale=tensors[stem + ".weight_global_scale"], dtype=torch.float32)
            np.testing.assert_array_equal(unrounded.bfloat16().float().numpy(), decoded[name])
            decoded[name + "/unrounded"] = unrounded.numpy()
    with float64(), torch.no_grad():
        truth = torch.nn.Module.double(loaded)(ids, use_cache=False).logits.numpy()
    np.savez(directory / "reference.npz", ids=ids.numpy(), logits=reference, logits_f64=truth,
             published_logits=published, **decoded)
    print(kind, "reference RMS from float64", float(np.sqrt(np.mean((reference - truth) ** 2))))


def activation_forwards() -> None:
    """CPU QDQ runs through the library's wrapped Linear, after real decompression."""
    arrays = {}
    for label, kind, dynamic in (("fp8_static", "fp8", False), ("fp8_dynamic", "fp8", True),
                                 ("nvfp4_local", "nvfp4", "local")):
        generator = torch.Generator().manual_seed(7301)
        module = torch.nn.Linear(32, 24, bias=False)
        weight = torch.randn(24, 32, generator=generator) * 0.08
        inputs = torch.randn(4, 7, 32, generator=generator)
        args = QuantizationArgs(**ARGS[kind], dynamic=dynamic)
        scheme = QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(**ARGS[kind]),
                                    input_activations=args)
        stored = compress(weight, scheme, kind)
        if kind == "nvfp4":
            stored["input_global_scale"] = generate_gparam(inputs.amin(), inputs.amax()).reshape(1)
        elif not dynamic:
            scale, _ = calculate_qparams(inputs.amin().reshape(1), inputs.amax().reshape(1), args)
            stored["input_scale"] = scale
        decoded = COMPRESSORS[kind].decompress(stored, scheme)
        dtype = decoded["weight"].dtype
        module.weight = torch.nn.Parameter(decoded["weight"])
        inputs = inputs.to(dtype)
        for name, value in decoded.items():
            if name != "weight":
                module.register_buffer(name, value)
        module.quantization_scheme = scheme
        module.quantization_status = QuantizationStatus.DECOMPRESSED
        set_forward_quantized(module)
        with torch.no_grad():
            output = module(inputs)
            scheme.input_activations = None
            no_input_qdq = module(inputs)
            module.quantization_enabled = False
            dense = module(inputs)
        np.testing.assert_array_equal(no_input_qdq.float().numpy(), dense.float().numpy())
        assert not torch.equal(output, dense)
        arrays.update({label + "/" + name: value.float().numpy() for name, value in
                       {"inputs": inputs, "weight": module.weight.detach(), "qdq": output,
                        "weight_only": dense}.items()})
        rms = float(torch.mean((output.float() - dense.float()) ** 2).sqrt())
        print(label, "QDQ RMS from weight-only", rms)
    np.savez(ROOT / "activation_forwards.npz", **arrays)


def main() -> None:
    for kind in ARGS:
        checkpoint(kind)
    activation_forwards()


if __name__ == "__main__":
    main()
