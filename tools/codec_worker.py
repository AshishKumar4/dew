"""The weight-only compressed-tensors checkpoints under tests/fixtures/codecs/storage (FP8 and NVFP4,
tools/nvfp4_storage_reference.py's files), each run by the format author's own CPU reader
(compressed-tensors 0.17.1 under transformers 5.16.1) on drawn ids, in float32 and float64: the reference
arm of tools/venv_reference.py's `codec-storage-<kind>` sites (its protocol tools/reference_worker.py), on
the CPU.

    python tools/codec_worker.py codec-storage-KIND    `IN OUT` lines on stdin, each `done`

The decoded weights run in float32 with weight QDQ off, and in float64 (`float64_twin`, its rotary table
float64's, the float32 read's values in it). IN holds each draw's `<draw>.ids` `[B, T]`; OUT gets
`<draw>.reference.logits` and `<draw>.truth.logits`.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from reference_worker import serve

STORAGE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "codecs" / "storage"


def run(kind: str, cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's logits, in float32 and float64."""
    from transformers import AutoModelForCausalLM

    from tools.torch_precision import float64, float64_twin

    directory = STORAGE / kind
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32, attn_implementation="eager")
    # PreTrainedModel.float refuses quantized models even after their weights are decoded; Module.float
    # promotes that decoded tree.
    model = torch.nn.Module.float(model).eval()
    exact = float64_twin(model, lambda: AutoModelForCausalLM.from_pretrained(
        directory, dtype=torch.float64, attn_implementation="eager"))
    for module in (*model.modules(), *exact.modules()):
        module.quantization_enabled = False
    out = {}
    for name in sorted({key.split(".", 1)[0] for key in cases}):
        ids = torch.from_numpy(np.asarray(cases[f"{name}.ids"])).long()
        with torch.no_grad():
            out[f"{name}.reference.logits"] = model(ids, use_cache=False).logits.numpy()
        with float64(), torch.no_grad():
            out[f"{name}.truth.logits"] = exact(ids, use_cache=False).logits.numpy()
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].startswith("codec-storage-"):
        raise SystemExit(__doc__)
    sys.path.insert(0, str(STORAGE.parents[3]))
    torch.set_num_threads(1)
    serve(lambda cases: run(sys.argv[1].removeprefix("codec-storage-"), cases))
