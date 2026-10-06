#!/usr/bin/env python3
"""Write ModelOpt NVFP4 fixtures with the author's exporter, reader and fake quantizer.

ModelOpt 0.47.0 quantizes a bf16 Qwen3 and export_hf_checkpoint writes the
tiny published-format files. NVFP4QTensor.dequantize reads those files;
ModelOpt's fake-quant kernel reads each stored input multiplier; only its
sm89 FP8 conversion is corrected to the direct RN cast the torch and
deployment paths specify. The uncorrected outputs are recorded separately.
The fp32 reference is transformers 5.16.1 on those decoded weights and
input quantizers. Float64 widens continuous operations, keeping the same
fp32 discrete input quantizer before widening its result.

The real mode reads 32 q_proj rows at a pinned NVIDIA Qwen3-14B revision,
about 100KB of payload, and records the same author reader/quantizer.
No whole model is downloaded. ModelOpt's division order, scale floor and
FP4 tie rule are in kernels/quantization/gemm/fp4_kernel_hopper.py:76-99
and common/nvfp4_quant.py:33-63,123-125 at the pinned package version.
TensorRT-LLM dc6f88de69ce869dc0ce095c888a70705a8b663c,
cpp/tensorrt_llm/kernels/quantization.cuh:499-503, uses the direct float
CUDA E4M3 constructor too. The conversion helper is checked against
torch's direct cast over 1M values and E4M3 midpoint neighbours.

Environment (~/.cache/dew/reference-venvs/nvfp4-modelopt): ModelOpt 0.47.0,
CUDA torch 2.13.0+cu130 (borrowed from agent-engines-vllm), transformers
5.16.1 (borrowed from nvfp4-storage), diffusers 0.34.0. Run on RTX 4080
through dew-gpu-run --short with those site-packages on PYTHONPATH:

    python tools/modelopt_nvfp4_reference.py tiny
    ionice -c3 nice -n19 python tools/modelopt_nvfp4_reference.py real
"""

import copy
import json
import struct
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import requests
import torch

# CUDA torch must load before the pinned transformers environment's CPU
# torch can appear on sys.path. The same kernel then serves both references.
sys.path.insert(0, str(Path.home() / ".cache/dew/reference-venvs/nvfp4-storage/lib/python3.12/site-packages"))

import modelopt.torch.quantization as mtq
import triton
import triton.language as tl
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.kernels.quantization.gemm.fp4_kernel_hopper import fp4_fake_quant_kernel as sm89_kernel
from modelopt.torch.quantization.qtensor import FP8QTensor, NVFP4QTensor
from modelopt.torch.quantization.tensor_quant import fp8_eager
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3ForCausalLM
from triton.language.extra.cuda import libdevice

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "codecs" / "modelopt"
REPO = "nvidia/Qwen3-14B-NVFP4"
REVISION = "bc39319a4dc265d9bbb9a9731bc52c4988d9ece7"
STEM = "model.layers.0.self_attn.q_proj"


@triton.jit
def rn_scale(block_amax, global_scale):
    """The author's scale expression with direct E4M3 RN, as its torch/deployment paths use."""
    normalized = tl.minimum(block_amax / (6.0 * global_scale), 448.0)
    bits = normalized.to(tl.uint32, bitcast=True)
    exponent = tl.maximum(((bits >> 23) & 255).to(tl.int32) - 127, -6) - 3
    quantum = ((exponent + 127).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    rounded = libdevice.rint(normalized / quantum) * quantum
    return rounded * global_scale


@triton.jit
def conversion_probe(values, output, global_scale, count, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(values + indices, mask=indices < count, other=0)
    converted = rn_scale(value, tl.load(global_scale))
    tl.store(output + indices, converted, mask=indices < count)


def validate_conversion() -> None:
    rng = np.random.default_rng(2051)
    values = rng.uniform(0, 448, 1_000_000).astype(np.float32)
    codes = torch.arange(127, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    finite = codes[torch.isfinite(codes)].cpu().numpy()
    midpoint = (finite[:-1] + finite[1:]) / 2
    values = np.concatenate([values, midpoint, np.nextafter(midpoint, np.inf),
                             np.nextafter(midpoint, -np.inf)])
    tensor = torch.from_numpy(values).cuda()
    global_scale = torch.tensor([1 / 6], device="cuda", dtype=torch.float32)
    output = torch.empty_like(tensor)
    conversion_probe[(triton.cdiv(tensor.numel(), 256),)](
        tensor, output, global_scale, tensor.numel(), BLOCK=256)
    expected = tensor.clamp(max=448).to(torch.float8_e4m3fn).float() * global_scale
    np.testing.assert_array_equal(output.cpu().numpy().view(np.uint32),
                                  expected.cpu().numpy().view(np.uint32))
    print("direct RN conversion", tensor.numel(), "values match torch in every bit")


# Clone the author's exact kernel body; only its conversion helper changes.
# On sm89 the original cast lowers through fp16 RTZ, which its torch scale
# cast and deployment CUDA constructor do not specify.
_globals = {**sm89_kernel.fn.__globals__, "fp8_quantize_scale": rn_scale}
_fn = types.FunctionType(sm89_kernel.fn.__code__, _globals,
                         sm89_kernel.fn.__name__, sm89_kernel.fn.__defaults__, sm89_kernel.fn.__closure__)
_fn.__annotations__ = sm89_kernel.fn.__annotations__
fp4_fake_quant_kernel = triton.jit(_fn)


def author_qdq(x: torch.Tensor, global_scale: torch.Tensor, *, corrected: bool = True) -> torch.Tensor:
    """The author's kernel, with the file's multiplier rather than a reconstructed amax."""
    shape = x.shape
    values = x.reshape(-1, shape[-1]).contiguous()
    rows, columns = values.shape
    out = torch.empty_like(values)
    kernel = fp4_fake_quant_kernel if corrected else sm89_kernel
    kernel[(triton.cdiv(rows, 4), triton.cdiv(columns, 128))](
        values, out, rows, columns, global_scale, values.stride(0), values.stride(1),
        out.stride(0), out.stride(1), BLOCK_SIZE=16, TILE_M=4, TILE_N=128,
        NUM_FP4_BLOCKS=8, OUT_DTYPE={torch.float32: tl.float32, torch.bfloat16: tl.bfloat16}[values.dtype])
    return out.reshape(shape)


def author_weight(stored: dict[str, torch.Tensor], stem: str) -> torch.Tensor:
    packed = stored[stem + ".weight"]
    if packed.dtype == torch.float8_e4m3fn:
        quantized = FP8QTensor(packed.shape, torch.bfloat16, packed)
        return quantized.dequantize(scale=stored[stem + '.weight_scale'])
    shape = torch.Size((packed.shape[0], 2 * packed.shape[1]))
    quantized = NVFP4QTensor(shape, torch.bfloat16, packed)
    return quantized.dequantize(scale=stored[stem + ".weight_scale"],
                               double_scale=stored[stem + ".weight_scale_2"], block_sizes={-1: 16})


def tiny(*, mixed: bool = False) -> None:
    from diffusers_wan_reference import float64

    torch.manual_seed(2049)
    config = Qwen3Config(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                        max_position_embeddings=64, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    config.architectures = ["Qwen3ForCausalLM"]
    ids = torch.tensor([[2, 8, 17, 5, 23, 9, 51, 43], [31, 16, 3, 19, 42, 11, 78, 39]], device="cuda")
    model = Qwen3ForCausalLM(config).to("cuda", dtype=torch.bfloat16).eval()
    scheme = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    if mixed:
        scheme = {'algorithm': 'max', 'quant_cfg': [
            {'quantizer_name': '*', 'enable': False},
            {'quantizer_name': '*self_attn.*weight_quantizer', 'cfg': {'num_bits': (4, 3), 'axis': None}},
            {'quantizer_name': '*self_attn.*input_quantizer', 'cfg': {'num_bits': (4, 3), 'axis': None}},
            {'quantizer_name': '*mlp.*weight_quantizer', 'cfg': {
                'num_bits': (2, 1), 'block_sizes': {-1: 16, 'type': 'dynamic', 'scale_bits': (4, 3)}}},
        ]}
    model = mtq.quantize(model, scheme,
                         forward_loop=lambda module: module(ids, use_cache=False))
    directory = ROOT / ('mixed' if mixed else 'tiny')
    export_hf_checkpoint(model, dtype=torch.bfloat16, export_dir=directory)
    stored = load_file(directory / "model.safetensors", device="cuda")
    config = Qwen3Config.from_pretrained(directory)
    del config.quantization_config
    config._attn_implementation = "eager"
    reference = Qwen3ForCausalLM(config).to("cuda").eval()
    quantized = [name.removesuffix('.weight_scale') for name in stored if name.endswith('.weight_scale')]
    suffixes = (".weight_scale", ".weight_scale_2", ".input_scale")
    parts = {stem + suffix for stem in quantized for suffix in suffixes}
    state = {name: value.float() for name, value in stored.items() if name not in parts}
    arrays = {"ids": ids.cpu().numpy()}
    for stem in quantized:
        state[stem + ".weight"] = author_weight(stored, stem).float()
        arrays[stem + "/weight"] = state[stem + ".weight"].cpu().numpy()
    missing, unexpected = reference.load_state_dict(state, strict=False)
    assert missing == ["lm_head.weight"] and not unexpected, (missing, unexpected)
    reference.tie_weights()
    direct_float = torch.Tensor.float
    original_to = torch.Tensor.to
    hooks = []
    for stem in quantized:
        module = reference.get_submodule(stem)
        if stem + '.input_scale' not in stored:
            continue
        overall = stored[stem + ".input_scale"]
        is_fp8 = stored[stem + '.weight'].dtype == torch.float8_e4m3fn

        def input_hook(_, args, overall=overall, stem=stem, is_fp8=is_fp8):
            value = args[0]
            def quantize(x):
                if is_fp8:
                    with patch.object(torch.Tensor, 'to', original_to):
                        return fp8_eager(x, overall * 448)
                return author_qdq(x, overall)
            # The same discrete kernel runs in fp32 in the widened model.
            if value.dtype == torch.float64:
                return (quantize(direct_float(value)).double(),)
            output = quantize(value)
            arrays[stem + "/forward_inputs"] = value.detach().cpu().numpy()
            arrays[stem + "/forward_qdq"] = output.detach().cpu().numpy()
            if not is_fp8:
                arrays[stem + "/forward_sm89_qdq"] = author_qdq(value, overall, corrected=False).cpu().numpy()
            return (output,)

        hooks.append(module.register_forward_pre_hook(input_hook))
    with torch.no_grad():
        arrays["logits"] = reference(ids, use_cache=False).logits.cpu().numpy()
    with float64(), torch.no_grad():
        arrays["logits_f64"] = reference.double()(ids, use_cache=False).logits.cpu().numpy()
    for hook in hooks:
        hook.remove()
    generator = torch.Generator(device="cuda").manual_seed(2050)
    probe = torch.randn(16, 64, generator=generator, device="cuda")
    probe[0] = 0
    probe[1, :16] = 1e-5
    for stem in quantized:
        if stem + '.input_scale' not in stored:
            continue
        overall = stored[stem + ".input_scale"]
        arrays[stem + "/inputs"] = probe.cpu().numpy()
        arrays[stem + "/global"] = overall.cpu().numpy()
        if stored[stem + '.weight'].dtype == torch.float8_e4m3fn:
            arrays[stem + '/qdq'] = fp8_eager(probe, overall * 448).cpu().numpy()
            arrays[stem + '/qdq_bf16'] = fp8_eager(probe.bfloat16(), overall * 448).float().cpu().numpy()
        else:
            arrays[stem + "/qdq"] = author_qdq(probe, overall).cpu().numpy()
            arrays[stem + "/qdq_bf16"] = author_qdq(probe.bfloat16(), overall).float().cpu().numpy()
            arrays[stem + "/sm89_qdq"] = author_qdq(probe, overall, corrected=False).cpu().numpy()
    np.savez(directory / "reference.npz", **arrays)
    assert torch.Tensor.to is original_to
    print("tiny", directory, "reference RMS", float(np.sqrt(np.mean(
        (arrays["logits"] - arrays["logits_f64"]) ** 2))))
    differences = {stem: int(np.count_nonzero(
        arrays[stem + "/forward_qdq"].view(np.uint32)
        != arrays[stem + "/forward_sm89_qdq"].view(np.uint32))) for stem in quantized
        if stem + '/forward_sm89_qdq' in arrays}
    print("sm89 conversion-only forward differences", json.dumps(differences))


def fetch(url: str, start: int, end: int) -> bytes:
    response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    response.raise_for_status()
    if response.status_code != 206 or len(response.content) != end - start + 1:
        raise ValueError(f"Hub did not return the requested byte range: {response.status_code}")
    return response.content


def real() -> None:
    base = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
    response = requests.get(base + "model.safetensors.index.json", timeout=120)
    response.raise_for_status()
    url = base + response.json()["weight_map"][STEM + ".weight"]
    size = struct.unpack("<Q", fetch(url, 0, 7))[0]
    header = json.loads(fetch(url, 8, size + 7))
    tensors = {}
    for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
        meta = header[STEM + "." + suffix]
        shape, offsets = list(meta["shape"]), meta["data_offsets"]
        start, end = offsets
        if len(shape) == 2:
            end, shape[0] = start + 32 * ((end - start) // shape[0]), 32
        payload = fetch(url, 8 + size + start, 8 + size + end - 1)
        dtype = {"U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32}[meta["dtype"]]
        tensors["m." + suffix] = torch.frombuffer(bytearray(payload), dtype=dtype).reshape(shape).cuda()
    directory = ROOT / "real"
    directory.mkdir(parents=True, exist_ok=True)
    save_file({name: value.cpu() for name, value in tensors.items()}, directory / "model.safetensors")
    generator = torch.Generator(device="cuda").manual_seed(2043)
    width = tensors["m.weight"].shape[1] * 2
    inputs = torch.randn(16, width, generator=generator, device="cuda", dtype=torch.bfloat16)
    inputs[0] = 0
    inputs[1, :16] = 1e-5
    weight = author_weight(tensors, "m")
    np.savez(directory / "reference.npz", inputs=inputs.float().cpu().numpy(),
             weight=weight.float().cpu().numpy(),
             qdq=author_qdq(inputs, tensors["m.input_scale"]).float().cpu().numpy(),
             qdq_f32=author_qdq(inputs.float(), tensors["m.input_scale"]).cpu().numpy(),
             sm89_qdq=author_qdq(inputs, tensors["m.input_scale"], corrected=False).float().cpu().numpy(),
             global_scale=tensors["m.input_scale"].cpu().numpy())
    print("real", REPO, REVISION, directory)


if __name__ == "__main__":
    validate_conversion()
    if sys.argv[1] == "tiny":
        tiny()
    elif sys.argv[1] == 'mixed':
        tiny(mixed=True)
    else:
        real()
