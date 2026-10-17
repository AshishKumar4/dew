"""Scratch, not for landing (ledger 560): record the joint release's logits on fla's own kernels
(tools/joint_checkpoint_reference.py --fla) in an environment of the recorder's pinned versions, and
return logits_fla.npz in the failure message."""
import base64
import gc
import lzma
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENV = Path.home() / "ug-joint-fla-venv"


@pytest.mark.gpu(reason="fla's kernels run on CUDA")
def test_joint_fla():
    import jax
    from jax.extend import backend

    jax.clear_caches()
    backend.clear_backends()
    gc.collect()
    started, lines = time.time(), []
    sys.path.insert(0, str(ROOT / "tools"))
    from joint_checkpoint_reference import FLA_PINS

    uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
    python = VENV / "bin" / "python"
    if not python.exists():
        subprocess.run([uv, "venv", "-q", "--python", "3.12", str(VENV)], check=True)
    subprocess.run([uv, "pip", "install", "-q", "--python", str(python), f"torch=={FLA_PINS['torch']}",
                    "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
    subprocess.run([uv, "pip", "install", "-q", "--python", str(python), "numpy", "safetensors",
                    "huggingface_hub", "pillow", f"transformers=={FLA_PINS['transformers']}",
                    f"fla-core=={FLA_PINS['fla-core']}"], check=True)
    lines.append(f"env ready at {time.time() - started:.0f} s")
    done = subprocess.run([str(python), "tools/joint_checkpoint_reference.py", "--fla"], cwd=ROOT,
                          env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'tools'}",
                               "TRITON_F32_DEFAULT": "ieee"},
                          capture_output=True, text=True, timeout=2400)
    lines.append(f"recorder exit {done.returncode} at {time.time() - started:.0f} s\n"
                 f"{(done.stdout + done.stderr)[-3000:]}")
    out = ROOT / "tests" / "fixtures" / "joint_checkpoint" / "tiny" / "logits_fla.npz"
    if out.exists():
        import numpy as np

        with np.load(out) as found, np.load(out.with_name("logits.npz")) as torch_math:
            for key in found.files:
                lines.append(f"{key}: max |fla - torch| {np.max(np.abs(found[key] - torch_math[key])):.3e}, "
                             f"|torch - f64| {np.max(np.abs(torch_math[key] - torch_math[key + '/f64'])):.3e}")
        lines.append("FILE logits_fla.npz " + base64.b64encode(lzma.compress(out.read_bytes())).decode())
    pytest.fail("\n".join(lines))
