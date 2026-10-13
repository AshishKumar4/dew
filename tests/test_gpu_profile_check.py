"""tools/gpu_profile_check.py: the self-check a GPU session runs first. Its
judging reads XProf's reports and the captures' numbers, and runs anywhere;
the captures themselves want a GPU."""

import json
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "gpu_profile_check.py"


def test_the_check_runs_end_to_end_and_fails_without_a_gpu(tmp_path):
    """On a CPU both captures run and are recorded, and the check fails: no
    device plane holds the step, as on a GPU session whose JAX lost its GPU.
    Each capture's own hlo_stats is read, which sits in the capture's
    directory below the profiler's."""
    done = subprocess.run([sys.executable, str(TOOL), "--out", str(tmp_path)], capture_output=True, text=True,
                          timeout=900)
    result = json.loads((tmp_path / "check.json").read_text())
    assert done.returncode == 1, done.stderr[-2000:]
    assert set(result["variants"]) == {"default", "attributed"}
    for found in result["variants"].values():
        assert "error" not in found, found
        assert found["platform"] == "cpu" and found["busy_ms_per_step"] == 0
        assert found["attributed_ops"] is not None, found
    assert result["versions"]["jax"]
