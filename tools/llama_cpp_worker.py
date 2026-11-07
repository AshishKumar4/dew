"""llama.cpp reading the decoder Dew exports (tools/llama_cpp_reference.py's), over drawn ids, in float32,
beside transformers' float64 run of the same export: the reference arm of tools/venv_reference.py's
`llama-cpp` site (its protocol tools/reference_worker.py), on the CPU.

    python tools/llama_cpp_worker.py llama-cpp    `IN OUT` lines on stdin, each `done`

On its first request it writes the export (Dew beneath its environment: a .pth onto Dew's site-packages,
with cmake, ninja and sentencepiece 0.2.2 above it), converts it with llama.cpp's own convert_hf_to_gguf.py
to an F32 GGUF and builds libllama with the logits program (`llama_cpp_reference.fetch`, `build`), all
under ~/.cache/dew/upstream and the worker's scratch directory. IN holds each draw's `<draw>.ids` `[1, T]`;
OUT gets `<draw>.reference.logits` (libllama's float32, an F32 KV cache, one thread) and
`<draw>.truth.logits` (transformers' float64 on the export, `torch_precision.float64`).
"""

import functools
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


@functools.cache
def prepared() -> tuple[Path, Path, torch.nn.Module]:
    """The logits program, the GGUF it reads and transformers' float64 model of the same export."""
    from transformers import AutoModelForCausalLM

    from tools.llama_cpp_reference import build, export, fetch
    from tools.torch_precision import float64

    scratch = Path(tempfile.mkdtemp(prefix="llama-cpp-worker-"))
    export(scratch / "export")
    source = fetch()
    program = build(source)
    gguf = scratch / "model-f32.gguf"
    subprocess.run([sys.executable, str(source / "convert_hf_to_gguf.py"), str(scratch / "export"),
                    "--outtype", "f32", "--outfile", str(gguf)], check=True, capture_output=True)
    with float64():
        model = AutoModelForCausalLM.from_pretrained(scratch / "export", dtype=torch.float64,
                                                     attn_implementation="eager").eval()
    return program, gguf, model


def run(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's logits from libllama, and transformers' float64 truth."""
    from tools.torch_precision import float64

    program, gguf, model = prepared()
    out = {}
    for name in sorted({key.split(".", 1)[0] for key in cases}):
        ids = np.asarray(cases[f"{name}.ids"], np.int32)
        with tempfile.TemporaryDirectory() as scratch:
            given, made = Path(scratch) / "ids.bin", Path(scratch) / "logits.bin"
            given.write_bytes(ids[0].tobytes())
            subprocess.run([str(program), str(gguf), str(given), str(made)], check=True, capture_output=True)
            vocab = model.config.vocab_size
            out[f"{name}.reference.logits"] = np.fromfile(made, np.float32).reshape(1, ids.shape[1], vocab)
        with float64(), torch.no_grad():
            out[f"{name}.truth.logits"] = model(torch.from_numpy(ids).long()).logits.numpy()
    return out


if __name__ == "__main__":
    if sys.argv[1:] != ["llama-cpp"]:
        raise SystemExit(__doc__)
    sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "src")]
    # The reference environment's own build tools (cmake, ninja) first, as its interpreter is.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    torch.set_num_threads(1)
    from reference_worker import serve

    serve(run)
