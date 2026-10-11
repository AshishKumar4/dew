"""The performance gate's verdict (tools/perf_gate.py), on samples written here."""

import importlib.util
import json
import subprocess
import sys
from functools import partial
from pathlib import Path

import pytest
from tool_support import load

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("writer", ["perf_gate", "benchmark_step", "config"])
def test_interrupted_evidence_and_config_saves_keep_the_completed_file(writer, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from dew import files
    from dew.cli import config

    out = tmp_path / "completed"
    out.write_text("previous contents")

    def interrupted(source, target):
        raise OSError("interrupted publication")

    monkeypatch.setattr(files.os, "replace", interrupted)
    if writer == "config":
        save = partial(config.save, config.TpuConfig(), out)
    elif writer == "perf_gate":
        tool = load(writer)
        monkeypatch.setattr(tool, "measure", lambda *args: {"fixture": 1.0})
        args = SimpleNamespace(base="base=/tmp/base", head="head=/tmp/head", rounds=1,
                               only=["fixture"], out=out)
        save = partial(tool.run, args)
    else:
        tool = load(writer)
        monkeypatch.setattr(tool, "build_cases", lambda config: [SimpleNamespace(label="fixture")])
        monkeypatch.setattr(tool, "measure", lambda *args: {"ms_per_step": 1, "samples_per_sec": 1})
        save = partial(tool.run, tool.BenchmarkConfig(json_out=str(out)))
    with pytest.raises(OSError, match="interrupted publication"):
        save()
    assert out.read_text() == "previous contents"
    assert list(tmp_path.iterdir()) == [out]


def test_report_command_keeps_dew_out_of_its_cold_import():
    code = ("import runpy, sys\nrunpy.run_path('tools/perf_gate.py', run_name='cold_import')\n"
            "assert not any(n == 'dew' or n.startswith('dew.') for n in sys.modules)\n")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("perf_gate", ROOT / "tools/perf_gate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("unit, base, head, state", [
    ("ms", [100.0, 101.0, 100.5], [100.8, 101.2, 100.2], "level"),        # inside the noise
    ("ms", [100.0, 101.0, 100.5], [110.0, 111.0, 109.5], "regressed"),    # slower past the band
    ("ms", [100.0, 108.0, 100.5], [106.0, 109.0, 107.0], "level"),        # 6% slower, inside an 8% spread
    ("ms", [100.0, 101.0, 100.5], [90.0, 91.0, 90.5], "faster"),
    ("items/s", [1000.0, 1010.0, 1005.0], [900.0, 905.0, 910.0], "regressed"),  # fewer items a second
    ("items/s", [1000.0, 1010.0, 1005.0], [1100.0, 1105.0, 1110.0], "faster"),
    ("ms", [100.0, 101.0, 100.5], [92.0, 100.8, 91.0], "level"),         # faster median, ranges overlap
    ("ms", ["not run: no such tool"], [100.0, 101.0], "not compared"),  # an older base without the tool
    ("ms", [100.0, 101.0], ["not run: it broke"] * 2, "broken"),
    ("ms", ["not run: no data"] * 2, ["not run: no data"] * 2, "incomplete"),
    ("ms", [100.0] * 3, [100.0, "not run: OOM", "not run: timeout"], "partial"),  # one round of three ran
    ("ms", [100.0] * 3, [100.0, None, None], "partial"),                         # rounds that wrote no row
    ("ms", [100.0] * 3, [100.0, float("nan"), 101.0], "partial"),
    ("ms", [100.0] * 3, [100.0, 0.0, 101.0], "partial"),
    ("items/s", [1000.0] * 3, [float("inf"), 1000.0, 1000.0], "partial"),
    ("ms", [100.0, "not run: OOM", 101.0], [100.0] * 3, "partial"),              # the base's rounds too
    ("ms", [100.0] * 3, [float("nan")] * 3, "broken"),                           # no head round measured
])
def test_a_row_moves_only_past_its_own_noise(gate, unit, base, head, state):
    """A row regresses, or is faster, when the head's median moves past
    either tree's spread across rounds (at least 2%) and the two trees'
    samples do not overlap. A row the base could not run in any round is
    not compared, one the head could not run in any round is broken, and
    one neither ran leaves the battery incomplete. A row either tree
    measured in only some rounds, or as NaN, an infinity or zero, is
    partial, and is not judged on the rounds that ran."""
    assert gate.verdict(gate.Row("row", unit, ""), base, head)[0] == state


def test_the_report_fails_on_a_regressed_row_and_writes_the_table(gate, tmp_path, monkeypatch):
    rows = [gate.Row("fast", "ms", "").__dict__, gate.Row("slow", "ms", "").__dict__]
    results = {"base": "main", "head": "head", "rows": rows,
               "samples": {"main": [{"fast": 10.0, "slow": 10.0}, {"fast": 10.1, "slow": 10.1}],
                           "head": [{"fast": 10.05, "slow": 12.0}, {"fast": 10.0, "slow": 12.2}]}}
    (tmp_path / "gate.json").write_text(json.dumps(results))
    monkeypatch.setattr(sys, "argv", ["perf_gate.py", "report", str(tmp_path / "gate.json"),
                                      "--table", str(tmp_path / "gate.md")])
    assert gate.main() == 1
    table = (tmp_path / "gate.md").read_text()
    assert "| slow | ms | 10.05 | 12.10 | +20.4% | 2.0% | regressed |" in table
    assert "| fast | ms |" in table and "| level |" in table


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


@pytest.mark.parametrize("head, rounds, shown", [
    ([{"row": 10.0}, {}, {"row": 10.1}], 3, "partial (head 1 of 3 rounds unmeasured)"),
    ([{"row": 10.0}, {"row": float("nan")}, {"row": 10.1}], 3, "partial (head 1 of 3 rounds unmeasured)"),
    ([{"row": 10.0}], 3, "partial (head 2 of 3 rounds unmeasured)"),
], ids=["a missing row key", "nan", "an interrupted run"])
def test_the_report_fails_on_a_partial_row(gate, tmp_path, monkeypatch, head, rounds, shown):
    """A head that measured a row in only some of its rounds fails the gate
    and says how many rounds it lacks, rather than passing on the rest."""
    results = {"base": "main", "head": "head", "rounds": rounds, "rows": [gate.Row("row", "ms", "").__dict__],
               "samples": {"main": [{"row": 10.0}, {"row": 10.1}, {"row": 10.05}], "head": head}}
    (tmp_path / "gate.json").write_text(json.dumps(results))
    monkeypatch.setattr(sys, "argv", ["perf_gate.py", "report", str(tmp_path / "gate.json"),
                                      "--table", str(tmp_path / "gate.md")])
    assert gate.main() == 1
    assert f"| row | ms | - | - | - | - | {shown} |" in (tmp_path / "gate.md").read_text()


@pytest.mark.parametrize("code, ratio, state, status", [
    (0, 1.10, "passed", 0), (1, 0.72, "failed", 1), (1, None, "failed", 1),
])
def test_external_reference_row_judges_only_head_and_reports_exit_and_ratio(
        gate, tmp_path, monkeypatch, capsys, code, ratio, state, status):
    """The base need not beat an external reference for a head that now beats it to qualify."""
    row = gate.Row("external reference", "items/s", "external.py", head_only=True)
    measurement = {"exit_code": code, "value": 110.0 if code == 0 else 72.0,
                   "ratio": ratio, "tolerance": 0.01, "passed": code == 0}
    record = {"base": "main", "head": "head", "rounds": 2, "rows": [row.__dict__],
              "samples": {"main": [{}, {}], "head": [{row.name: measurement}] * 2}}
    out = tmp_path / "reference.json"
    out.write_text(json.dumps(record))
    monkeypatch.setattr(sys, "argv", ["perf_gate.py", "report", str(out)])
    assert gate.main() == status
    printed = capsys.readouterr().out
    assert f"{state} (exit codes {code}, {code})" in printed
    if ratio is not None:
        assert f"{ratio:.3f}x reference" in printed
    verdict, numbers = gate.verdict(row, [None, None], [measurement, measurement])
    assert verdict == state
    assert numbers["exit_codes"] == [code, code]
    assert numbers.get("ratio") == ratio


def test_external_reference_row_fails_closed_on_missing_or_partial_head(gate):
    row = gate.Row("external reference", "items/s", "external.py", head_only=True)
    good = {"exit_code": 0, "value": 110.0, "ratio": 1.10, "tolerance": 0.01, "passed": True}
    assert gate.verdict(row, [], [None, None])[0] == "incomplete"
    assert gate.verdict(row, [], [good, None])[0] == "incomplete"
    missing = "not measured on this hardware"
    assert gate.verdict(row, [], [good, missing])[0] == "incomplete"
    assert gate.verdict(row, [], [{**good, "exit_code": 1}, missing])[0] == "failed"


def test_external_reference_subprocess_failure_keeps_its_measurement(gate, tmp_path, monkeypatch):
    """Exit 1 from an external reference still carries measured evidence through the battery, and a
    tree that measures no hardware row starts no process to read the device kind."""
    from types import SimpleNamespace

    commands = []

    def child(command, **kwargs):
        commands.append(command)
        if "-c" in command:
            return subprocess.CompletedProcess(command, 0, "NVIDIA A100-SXM4-80GB\n", "")
        out = Path(command[command.index("--out") + 1])
        out.write_text(json.dumps({"comparison": {"dew_tokens_per_s": 720.0, "dew_over_torch": 0.72,
                                                  "tolerance": 0.01, "performance_claim_passed": False}}))
        return subprocess.CompletedProcess(command, 1, "measured a slower head", "")

    monkeypatch.setattr(gate.subprocess, "run", child)
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
