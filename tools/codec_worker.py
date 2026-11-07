"""The compressed-tensors checkpoints under tests/fixtures/codecs, each run by the format author's own CPU
reader (compressed-tensors 0.17.1 under transformers 5.16.1) on drawn ids, in float32 and float64: the
reference arm of tools/venv_reference.py's `codec-<format>-<kind>` sites (its protocol
tools/reference_worker.py), on the CPU.

    python tools/codec_worker.py codec-storage-KIND    `IN OUT` lines on stdin, each `done`
    python tools/codec_worker.py codec-nvfp4_qdq-KIND

`storage` is weight-only FP8 or NVFP4 (tools/nvfp4_storage_reference.py's files): the decoded weights run
in float32 with weight QDQ off, and in float64 (`float64_twin`, its rotary table float64's). `nvfp4_qdq` is
NVFP4 with local input QDQ under either scale rounding (tools/nvfp4_qdq_reference.py's files): the input
quantizer stays on, and in the float64 run it computes its discrete function in float32 on the float32 cast
of its input and widens the result (`quantize_wide`), since changing a discontinuous quantizer's dtype
changes its function. IN holds each draw's `<draw>.ids` `[B, T]`; OUT gets `<draw>.reference.logits` and
`<draw>.truth.logits`.
"""

import contextlib
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from reference_worker import serve

CODECS = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "codecs"


def readers(directory: Path, *, input_qdq: bool):
    """The checkpoint at `directory` read in float32 and its float64 twin, each a function of ids to its
    logits, the twin's run inside `float64` (with its input quantizer kept in float32 where `input_qdq`)."""
    from compressed_tensors.quantization import QuantizationStatus
    from compressed_tensors.quantization.lifecycle.forward import forward_quantize
    from transformers import AutoModelForCausalLM

    from tools.torch_precision import float64, float64_twin

    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32, attn_implementation="eager")
    # PreTrainedModel.float refuses quantized models even after their weights are decoded; Module.float
    # promotes that decoded tree.
    model = torch.nn.Module.float(model).eval()
    exact = float64_twin(model, lambda: AutoModelForCausalLM.from_pretrained(
        directory, dtype=torch.float64, attn_implementation="eager"))
    for network in (model, exact):
        for module in network.modules():
            if input_qdq and hasattr(module, "quantization_status"):
                module.quantization_status = QuantizationStatus.COMPRESSED
            elif not input_qdq:
                module.quantization_enabled = False
    direct_float = torch.Tensor.float

    def quantize_wide(module, value, base_name, args):
        # The quantizer's stored global stays float32 though the model around it widens.
        overall = module.input_global_scale
        module.input_global_scale = torch.nn.Parameter(direct_float(overall), requires_grad=False)
        try:
            return forward_quantize(module, direct_float(value), base_name, args).double()
        finally:
            module.input_global_scale = overall

    def narrow(ids):
        with torch.no_grad():
            return model(torch.from_numpy(ids).long(), use_cache=False).logits.numpy()

    def wide(ids):
        widening = (patch("compressed_tensors.quantization.lifecycle.forward.forward_quantize", quantize_wide)
                    if input_qdq else contextlib.nullcontext())
        with float64(), torch.no_grad(), widening:
            return exact(torch.from_numpy(ids).long(), use_cache=False).logits.numpy()
    return narrow, wide


def run(site: str, cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's logits, in float32 and float64."""
    family, kind = site.removeprefix("codec-").split("-")
    narrow, wide = readers(CODECS / family / kind, input_qdq=family == "nvfp4_qdq")
    out = {}
    for name in sorted({key.split(".", 1)[0] for key in cases}):
        ids = np.asarray(cases[f"{name}.ids"])
        out[f"{name}.reference.logits"] = narrow(ids)
        out[f"{name}.truth.logits"] = wide(ids)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].startswith("codec-"):
        raise SystemExit(__doc__)
    sys.path.insert(0, str(CODECS.parents[2]))
    torch.set_num_threads(1)
    serve(lambda cases: run(sys.argv[1], cases))
