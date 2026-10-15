"""The performance gate's verdict (tools/perf_gate.py), on samples written here."""

import json
import subprocess
from pathlib import Path

import pytest
from tool_support import load

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gate():
    return load("perf_gate")


@pytest.mark.parametrize("head, code", [
    ({"gated": 10.0, "shown": 14.0}, 0),                   # a regression of an ungated row only shows
    ({"gated": "not run: it broke", "shown": 10.0}, 1),     # a broken row fails the gate
])
def test_the_report_fails_on_a_broken_row_and_not_on_an_ungated_one(gate, tmp_path, monkeypatch, head, code):
    rows = [gate.Row("gated", "ms", "").__dict__, gate.Row("shown", "ms", "", gated=False).__dict__]
    results = {"base": "main", "head": "head", "rows": rows,
               "samples": {"main": [{"gated": 10.0, "shown": 10.0}, {"gated": 10.1, "shown": 10.1}],
                           "head": [head, head]}}
    (tmp_path / "gate.json").write_text(json.dumps(results))
    monkeypatch.setattr(sys, "argv", ["perf_gate.py", "report", str(tmp_path / "gate.json")])
    assert gate.main() == code


def after(command, flag):
    return command[command.index(flag) + 1]


def standing_in(gate, monkeypatch, *, passed: bool) -> list:
    """Stand every tool of the battery in with one writing its output format,
    and return the commands run. The reference either passes its claim or
    measures a head at 0.72 of it and exits 1."""
    outputs = {
        "benchmark_step.py": lambda command: [{"ms_per_step": 5.0}],
        "benchmark_forward.py": lambda command: {after(command, "--cases"): 5.0},
        "benchmark_lm_serving.py": lambda command: {"sweep": [
            {"slots": int(slots), "repeats": [{"output_tokens_per_second": rate} for rate in (5.0, 6.0)]}
            for slots in after(command, "--slots").split(",")]},
        "benchmark_attention.py": lambda command: [
            {"implementation": implementation, "forward_backward_ms": 5.0}
            for implementation in command[command.index("--implementations") + 1:
                                          command.index("--sequence-lengths")]],
        "benchmark_image_pipeline.py": lambda command: {
            "pipeline": {"device_crop_flip_jitter": {"images_per_second": 5.0}}},
        "qwen35_pair.py": lambda command: {"comparison": {
            "dew_tokens_per_s": 720.0, "dew_over_torch": 1.0 if passed else 0.72, "tolerance": 0.01,
            "performance_claim_passed": passed}},
    }
    commands = []

    def child(command, **kwargs):
        commands.append(command)
        if command[1] == "-c":
            printed = ("NVIDIA A100-SXM4-80GB" if "device_kind" in command[2]
                       else json.dumps({"save_ms": 5.0, "restore_ms": 6.0}))
            return subprocess.CompletedProcess(command, 0, printed + "\n", "")
        tool = Path(command[1]).name
        flag = "--json-out" if "--json-out" in command else "--out"
        Path(after(command, flag)).write_text(json.dumps(outputs[tool](command)))
        failed = tool == "qwen35_pair.py" and not passed
        return subprocess.CompletedProcess(command, int(failed), "measured a slower head" * failed, "")

    monkeypatch.setattr(gate.subprocess, "run", child)
    return commands


def test_external_reference_subprocess_failure_keeps_its_measurement(gate, tmp_path, monkeypatch):
    """Exit 1 from an external reference still carries measured evidence through the battery, and a
    tree that measures no hardware row starts no process to read the device kind."""
    from types import SimpleNamespace

    commands = standing_in(gate, monkeypatch, passed=False)
    args = SimpleNamespace(only=["reference"], claim=True, model="unused", flowers="unused",
                           torch_python="torch", out=tmp_path / "gate.json")
    name = next(row.name for row in gate.BATTERY if row.head_only)
    sample = gate.measure(ROOT, args, ROOT / "tools", is_head=True)
    assert sample[name] == {"exit_code": 1, "value": 720.0, "ratio": 0.72,
                            "tolerance": 0.01, "passed": False}
    assert Path(sample[f"evidence {name}"]).is_file()
    assert len(commands) == 2
    assert gate.measure(ROOT, args, ROOT / "tools", is_head=False) == {}
    assert len(commands) == 2


def test_a_head_reads_every_row_of_the_battery_from_the_tool_it_names(gate, tmp_path, monkeypatch):
    """Each probe runs the tool its rows name and reads every one of its rows
    from that tool's output, so a head whose tools all ran has a measurement
    for every row of the battery."""
    from types import SimpleNamespace

    standing_in(gate, monkeypatch, passed=True)
    args = SimpleNamespace(only=[], claim=False, model="unused", flowers="unused", torch_python="torch",
                           out=tmp_path / "gate.json")
    sample = gate.measure(ROOT, args, ROOT / "tools", is_head=True)
    for row in gate.BATTERY:
        value = sample[row.name]
        assert value["exit_code"] == 0 if row.head_only else gate.measured(value), (row.name, value)
