"""ModelOpt's NVFP4 checkpoints (tests/fixtures/codecs/modelopt/<name>, tools/modelopt_nvfp4_reference.py's)
run by their author's forward over drawn ids, in float32 and float64: the reference arm of
tools/venv_reference.py's `codec-modelopt-<name>` sites (its protocol tools/reference_worker.py), on the
CPU, where Dew's arm runs in its test.

    python tools/modelopt_worker.py codec-modelopt-NAME    `IN OUT` lines on stdin, each `done`

IN holds each draw's `<draw>.ids` `[B, T]`; OUT gets `<draw>.reference.logits` and `<draw>.truth.logits`
(`author_models`: the weights decoded as the author decodes them, each input quantizer registered on its
layer in both runs). The models load once, on the first request.

The reference runs on the platform its arm is judged beside (row 5244): a float32 product's accumulation is
its platform's, and on a GPU the author's GEMMs round less than any CPU's, Dew's and torch's alike (one FMA
chain a product on XLA:CPU, the bits torch's CPU forward gives too), which a GPU recording would charge to
Dew. Its input quantization is the author's own on the CPU: an FP8 input through its TensorQuantizer's CPU
path (`fp8_eager`), an NVFP4 one through `NVFP4QTensor`, whose decisions are its GPU kernel's (every value
at every linear of both checkpoints on a GH200's capture, IE-5244-author64-1); its TensorQuantizer takes
the dynamic-block kernel only on a GPU (modelopt/torch/quantization/tensor_quant.py:172 at 0.47.0).
"""

import functools
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from modelopt.torch.quantization.qtensor import FP8QTensor, NVFP4QTensor
from modelopt.torch.quantization.tensor_quant import fp8_eager
from safetensors.torch import load_file
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
CODECS = ROOT / "tests" / "fixtures" / "codecs" / "modelopt"


def author_weight(stored: dict[str, torch.Tensor], stem: str) -> torch.Tensor:
    """Layer `stem`'s weight decoded as the author decodes it: an FP8 one by `FP8QTensor`, an NVFP4 one by
    `NVFP4QTensor` over its block and global scales."""
    packed = stored[stem + ".weight"]
    if packed.dtype == torch.float8_e4m3fn:
        quantized = FP8QTensor(packed.shape, torch.bfloat16, packed)
        return quantized.dequantize(scale=stored[stem + '.weight_scale'])
    shape = torch.Size((packed.shape[0], 2 * packed.shape[1]))
    quantized = NVFP4QTensor(shape, torch.bfloat16, packed)
    return quantized.dequantize(scale=stored[stem + ".weight_scale"],
                               double_scale=stored[stem + ".weight_scale_2"], block_sizes={-1: 16})


def nvfp4_qdq(x: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """The author's NVFP4 input quantization on the CPU (`NVFP4QTensor`): 16-value blocks, each scale its
    amax over 6 times the stored global scale in E4M3."""
    quantized, scale, double_scale = NVFP4QTensor.quantize(x, 16, weights_scaling_factor_2=global_scale)
    return quantized.dequantize(scale=scale, double_scale=double_scale, block_sizes={-1: 16})


def author_models(directory: Path, quantize: Callable = nvfp4_qdq, device: str = "cpu",
                  arrays: dict | None = None):
    """The author's forward of the checkpoint at `directory` on `device`: its weights decoded as the author
    decodes them (`author_weight`), the float32 reference and its float64 twin (`float64_twin`, built inside
    `float64` so its rotary tables are float64's), each input quantizer registered on its layer in both, an
    NVFP4 one `quantize(x, global_scale)`, the discrete step run in fp32 in the widened model; with `arrays`,
    each quantized layer's forward inputs and QDQ outputs recorded into it. Returns the two models, the
    stored tensors, the quantized stems and the hooks."""
    from _dew_torch.precision import float64_twin

    stored = load_file(directory / "model.safetensors", device=device)
    config = Qwen3Config.from_pretrained(directory)
    del config.quantization_config
    config._attn_implementation = "eager"
    reference = Qwen3ForCausalLM(config).to(device).eval()
    quantized = [name.removesuffix('.weight_scale') for name in stored if name.endswith('.weight_scale')]
    suffixes = (".weight_scale", ".weight_scale_2", ".input_scale")
    parts = {stem + suffix for stem in quantized for suffix in suffixes}
    state = {name: value.float() for name, value in stored.items() if name not in parts}
    for stem in quantized:
        state[stem + ".weight"] = author_weight(stored, stem).float()
        if arrays is not None:
            arrays[stem + "/weight"] = state[stem + ".weight"].cpu().numpy()
    missing, unexpected = reference.load_state_dict(state, strict=False)
    assert missing == ["lm_head.weight"] and not unexpected, (missing, unexpected)
    reference.tie_weights()
    # The float64 truth is the reference's twin (`float64_twin`), built inside `float64` so its rotary
    # tables are float64's, each input quantizer registered on its layer in both.
    wide = float64_twin(reference)
    direct_float = torch.Tensor.float
    original_to = torch.Tensor.to
    hooks = []
    for stem in quantized:
        if stem + '.input_scale' not in stored:
            continue
        overall = stored[stem + ".input_scale"]
        is_fp8 = stored[stem + '.weight'].dtype == torch.float8_e4m3fn

        def input_hook(_, args, overall=overall, stem=stem, is_fp8=is_fp8):
            value = args[0]
            def quantized_input(x):
                if is_fp8:
                    with patch.object(torch.Tensor, 'to', original_to):
                        return fp8_eager(x, overall * 448)
                return quantize(x, overall)
            # The same discrete step runs in fp32 in the widened model.
            if value.dtype == torch.float64:
                return (quantized_input(direct_float(value)).double(),)
            output = quantized_input(value)
            if arrays is not None:
                arrays[stem + "/forward_inputs"] = value.detach().cpu().numpy()
                arrays[stem + "/forward_qdq"] = output.detach().cpu().numpy()
            return (output,)

        hooks += [model.get_submodule(stem).register_forward_pre_hook(input_hook)
                  for model in (reference, wide)]
    return reference, wide, stored, quantized, hooks


@functools.cache
def models(name: str):
    """The author's float32 reference and float64 twin of checkpoint `name`."""
    reference, wide, *_ = author_models(CODECS / name)
    return reference, wide


def run(name: str, cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's logits, in float32 and float64."""
    from _dew_torch.precision import float64

    reference, wide = models(name)
    out = {}
    for draw in sorted({key.split(".", 1)[0] for key in cases}):
        ids = torch.from_numpy(np.asarray(cases[f"{draw}.ids"], np.int64))
        with torch.no_grad():
            out[f"{draw}.reference.logits"] = reference(ids, use_cache=False).logits.numpy()
        with float64(), torch.no_grad():
            out[f"{draw}.truth.logits"] = wide(ids, use_cache=False).logits.numpy()
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].startswith("codec-modelopt-"):
        raise SystemExit(__doc__)
    sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "src")]
    from reference_worker import serve

    torch.set_num_threads(1)
    serve(functools.partial(run, sys.argv[1].removeprefix("codec-modelopt-")))
