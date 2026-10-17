"""Probe, not for main: the released Dory under Dew against vLLM with PR #60149's model, bf16 on
this GPU, both cache modes."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent / "dory_vllm"
BASE = "153a17166e5d07bc38184dedeea542d2e8885e55"
HEAD = "55076950b6c155f384d6766d63563c3bc20a0e3f"
WHEEL = f"https://wheels.vllm.ai/{BASE}/vllm-0.32.0rc1.dev307%2Bg153a17166-cp38-abi3-manylinux_2_28_x86_64.whl"
OVERLAY = ["vllm/model_executor/models/dory.py", "vllm/model_executor/models/registry.py",
           "vllm/transformers_utils/config.py", "vllm/transformers_utils/configs/__init__.py",
           "vllm/transformers_utils/configs/dory.py"]


@pytest.mark.gpu(reason="probe")
def test_probe_dory_against_vllm(tmp_path):
    lines = []
    plain = {k: v for k, v in os.environ.items() if not k.startswith(("DEW_XLA_CACHE", "VIRTUAL_ENV", "PYTHON",
                                                                       "XLA_", "JAX_"))}
    plain["HF_HOME"] = str(tmp_path / "hf")
    venv = tmp_path / "vllm"
    built = subprocess.run(f"uv venv -q -p 3.12 {venv} && VIRTUAL_ENV={venv} uv pip install -q '{WHEEL}'",
                           shell=True, capture_output=True, text=True, timeout=2400, env=plain)
    if built.returncode:
        pytest.fail("vLLM install failed\n" + built.stderr[-3000:], pytrace=False)
    site = next(venv.glob("lib/python3.12/site-packages"))
    for path in OVERLAY:
        fetched = subprocess.run(["curl", "-sfL", "-o", str(site / path),
                                  f"https://raw.githubusercontent.com/shan18/vllm/{HEAD}/{path}"],
                                 capture_output=True, text=True)
        lines.append(f"overlay {path}: {fetched.returncode}")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"],
                         capture_output=True, text=True).stdout
    lines.append(smi.strip())
    records = {}
    for mode in ("per_loop", "last_loop"):
        out = tmp_path / f"{mode}.json"
        ran = subprocess.run([str(venv / "bin/python"), str(HERE / "record.py"), mode, str(out)], capture_output=True,
                             text=True, timeout=3600, env=plain)
        lines.append(f"vllm {mode}: exit {ran.returncode} " + ("" if out.exists() else ran.stderr[-4000:]))
        records[mode] = out
        if not out.exists():
            pytest.fail("\n".join(lines), pytrace=False)
    dew_env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "HF_HOME": plain["HF_HOME"],
               "PYTHONPATH": os.pathsep.join(filter(None, ["src", os.environ.get("PYTHONPATH")]))}
    dew_env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    ran = subprocess.run([sys.executable, str(HERE / "dew_side.py"), str(records["per_loop"]),
                          str(records["last_loop"])], capture_output=True, text=True, timeout=3600, env=dew_env)
    found = [line for line in ran.stdout.splitlines() if line.startswith("DEWSIDE ")]
    lines.append(json.dumps(json.loads(found[0][8:]), indent=1) if found else f"dew exit {ran.returncode}\n"
                 + ran.stderr[-4000:])
    for mode in ("per_loop", "last_loop"):
        lines.append(f"{mode} vllm: " + json.dumps([row["generated"] for row in json.load(open(records[mode]))]))
    pytest.fail("\n".join(lines), pytrace=False)
