"""llama.cpp reading the decoder Dew exports (tools/llama_cpp_reference.py's), over drawn ids, in float32,
beside transformers' float64 run of the same export: the recorded arm of tools/venv_reference.py's
`llama-cpp` site, the one its test judges (`judged`; its protocol tools/reference_worker.py), on the CPU.

    python tools/llama_cpp_worker.py llama-cpp    `IN OUT` lines on stdin, each `done`

On its first request it writes the export (Dew beneath its environment: a .pth onto Dew's site-packages,
with cmake, ninja and sentencepiece 0.2.2 above it), converts it with llama.cpp's own convert_hf_to_gguf.py
to an F32 GGUF and builds libllama with the logits program (`llama_cpp_reference.fetch`, `build`), all
under ~/.cache/dew/upstream and the worker's scratch directory. IN holds each draw's `<draw>.ids` `[1, T]`;
OUT gets `<draw>.reference.logits` (libllama's float32, an F32 KV cache, one thread) and
`<draw>.truth.logits` (transformers' float64 on the export, `_dew_torch.precision.float64`).
"""

import functools
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


@functools.cache
def prepared() -> tuple[Path, Path, torch.nn.Module]:
    """The logits program, the GGUF it reads and transformers' float64 model of the same export."""
    from tools.llama_cpp_reference import build, converted, export, fetch, float64_model

    scratch = Path(tempfile.mkdtemp(prefix="llama-cpp-worker-"))
    export(scratch / "export")
    source = fetch()
    gguf = scratch / "model-f32.gguf"
    converted(source, scratch / "export", gguf, sys.executable)
    return build(source), gguf, float64_model(scratch / "export")


def run(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's logits from libllama, and transformers' float64 truth."""
    from reference_worker import drawn

    from _dew_torch.precision import float64
    from tools.llama_cpp_reference import llama_logits

    program, gguf, model = prepared()

    def evaluated(inputs):
        ids = np.asarray(inputs["ids"], np.int32)
        reference = {"logits": llama_logits(program, gguf, ids)}
        with float64(), torch.no_grad():
            truth = {"logits": model(torch.from_numpy(ids).long()).logits.numpy()}
        return reference, truth

    return drawn(cases, evaluated)


if __name__ == "__main__":
    if sys.argv[1:] != ["llama-cpp"]:
        raise SystemExit(__doc__)
    sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "src")]
    # The reference environment's own build tools (cmake, ninja) first, as its interpreter is.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    torch.set_num_threads(1)
    from reference_worker import serve

    serve(run)
