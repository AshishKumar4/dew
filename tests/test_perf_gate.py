"""The performance gate's verdict (tools/perf_gate.py), on samples written here."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


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
    ("ms", ["not run: no such tool"], [100.0, 101.0], "not compared"),
])
def test_a_row_regresses_only_past_its_own_noise(gate, unit, base, head, state):
    """A row regresses when the head's median is worse than the base's by
    more than either tree's spread across rounds (at least 2%) and the two
    ranges do not overlap; a row one tree could not run is not compared."""
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
