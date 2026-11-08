"""The compressed-tensors checkpoints under tests/fixtures/codecs: weight-only storage (FP8 and NVFP4,
tools/nvfp4_storage_reference.py's files) and NVFP4 with input QDQ under either local-scale rounding
(tools/nvfp4_qdq_reference.py's), each run by the format author's own CPU reader
(compressed-tensors 0.17.1 under transformers 5.16.1) on drawn ids, in float32 and float64: the reference
arm of tools/venv_reference.py's `codec-<family>-<kind>` sites (its protocol tools/reference_worker.py), on
the CPU.

    python tools/codec_worker.py codec-FAMILY-KIND    `IN OUT` lines on stdin, each `done`

The decoded weights run in float32 with weight QDQ off and the input QDQ on where the checkpoint declares
it, and in float64 (`float64_twin`, its rotary table float64's, the float32 read's values in it), the
input QDQ there as published (`nvfp4_qdq_reference.quantized_as_published`). IN holds each draw's
`<draw>.ids` `[B, T]`; OUT gets `<draw>.reference.logits` and `<draw>.truth.logits`.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from reference_worker import drawn, serve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CODECS = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "codecs"
FAMILIES = ("storage", "nvfp4_qdq")
"""The checkpoint families the worker runs: weight-only storage, and NVFP4 with input QDQ."""


def run(family: str, kind: str, cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's logits, in float32 and float64."""
    from compressed_tensors.quantization import QuantizationStatus
    from nvfp4_qdq_reference import quantized_as_published
    from transformers import AutoModelForCausalLM

    from _dew_torch.precision import float64, float64_twin

    directory = CODECS / family / kind
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32, attn_implementation="eager")
    # PreTrainedModel.float refuses quantized models even after their weights are decoded; Module.float
    # promotes that decoded tree.
    model = torch.nn.Module.float(model).eval()
    exact = float64_twin(model, lambda: AutoModelForCausalLM.from_pretrained(
        directory, dtype=torch.float64, attn_implementation="eager"))
    for module in (*model.modules(), *exact.modules()):
        if family == "storage":
            module.quantization_enabled = False
        elif hasattr(module, "quantization_status"):
            # The decoded weights' QDQ again is redundant; the input QDQ stays (tools/nvfp4_qdq_reference.py).
            module.quantization_status = QuantizationStatus.COMPRESSED
    def answer(inputs):
        ids = torch.from_numpy(np.asarray(inputs["ids"])).long()
        with torch.no_grad():
            reference = model(ids, use_cache=False).logits.numpy()
        with float64(), torch.no_grad(), quantized_as_published():
            return {"logits": reference}, {"logits": exact(ids, use_cache=False).logits.numpy()}

    return drawn(cases, answer)


if __name__ == "__main__":
    family, _, kind = sys.argv[1].removeprefix("codec-").rpartition("-") if len(sys.argv) == 2 else ("",) * 3
    if family not in FAMILIES:
        raise SystemExit(__doc__)
    sys.path.insert(0, str(CODECS.parents[2] / "tools"))
    torch.set_num_threads(1)
    serve(lambda cases: run(family, kind, cases))
