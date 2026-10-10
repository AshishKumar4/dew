"""tools/gpu_profile_check.py: the self-check a GPU session runs first. Its
judging reads XProf's reports and the captures' numbers, and runs anywhere;
the captures themselves want a GPU."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "gpu_profile_check.py"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("gpu_profile_check", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hlo_stats(tmp_path, names, column="hlo_op_name"):
    reports = tmp_path / "reports"
    reports.mkdir()
    table = {"cols": [{"id": "rank"}, {"id": column}],
             "rows": [{"c": [{"v": rank}, {"v": name}]} for rank, name in enumerate(names)]}
    (reports / "hlo_stats.json").write_text(json.dumps([table]))
    return reports


@pytest.mark.parametrize("names, column, count", [
    (["XlaModule/loop_add_fusion", "XlaModule/gemm"], "hlo_op_name", 0),
    (["XlaModule/DelayKernel", "loop_add_fusion", "gemm_fusion_dot.3"], "hlo_op_name", 2),
    (["loop_add_fusion"], "op", None),
], ids=["modules only", "two ops", "no op column"])
def test_ops_are_counted_apart_from_whole_modules(tmp_path, names, column, count):
    from dew.telemetry.profile import attributed_ops

    assert attributed_ops(hlo_stats(tmp_path, names, column)) == count


def capture(ops=12, busy=0.4, wall=1.0, **more):
    return {"attributed_ops": ops, "busy_ms_per_step": busy, "wall_ms_per_step": wall, **more}


@pytest.mark.parametrize("variants, failing", [
    ({"default": capture(ops=0), "attributed": capture()}, []),
    ({"default": capture(ops=0), "attributed": capture(ops=0)}, ["attributed: hlo_stats names no op"]),
    ({"default": capture(busy=0.0), "attributed": capture()}, ["default: 0.000 ms of device time"]),
    ({"default": capture(), "attributed": capture(busy=1.5)}, ["attributed: 1.500 ms of device time"]),
    ({"default": {"error": "exit 1: no GPU"}, "attributed": capture()}, ["default: exit 1: no GPU"]),
], ids=["passes", "no op named", "no device time", "more device time than wall", "a capture failed"])
def test_the_check_fails_on_lost_attribution_or_impossible_timing(check, variants, failing):
    """The default capture's attribution is recorded, not judged: on JAX
    0.11.2 it names no op, so a release that changes that shows in the
    record. With command buffers off it must name ops, and every capture's
    device time a step must be positive and no more than its wall time."""
    failures = check.judged({"variants": variants})
    assert len(failures) == len(failing)
    for failure, expected in zip(failures, failing, strict=True):
        assert failure.startswith(expected), failure


def test_the_check_runs_end_to_end_and_fails_without_a_gpu(tmp_path):
    """On a CPU both captures run and are recorded, and the check fails: no
    device plane holds the step, as on a GPU session whose JAX lost its GPU."""
    done = subprocess.run([sys.executable, str(TOOL), "--out", str(tmp_path)], capture_output=True, text=True,
                          timeout=900)
    result = json.loads((tmp_path / "check.json").read_text())
    assert done.returncode == 1, done.stderr[-2000:]
    assert set(result["variants"]) == {"default", "attributed"}
    for found in result["variants"].values():
        assert "error" not in found, found
        assert found["platform"] == "cpu" and found["busy_ms_per_step"] == 0
    assert result["versions"]["jax"]
