"""Scratch: why a train's rows of a file with cells are not reused by the next train."""

import json

from armada_support import loaded
from test_armada_ci import armada_task, cached_checkout


def test_probe(ci, tmp_path, monkeypatch, cache_worker):
    (tmp_path / "tests/helper.py").write_text("VALUE = 1\n")
    (tmp_path / "tests/test_mixed.py").write_text(
        "import pytest\nfrom pathlib import Path\n"
        "def note(name):\n    with Path(f'runs-{name}').open('a') as runs: runs.write('x')\n"
        "def test_a():\n    note('a')\n"
        "@pytest.mark.whole_suite\ndef test_b():\n    note('b')\n"
        "def test_c():\n    import helper\n    note('c')\n    assert helper.VALUE\n"
        "def test_d():\n    note('d')\n")
    (tmp_path / "tests/test_durations.json").write_text(
        json.dumps({f"tests/test_mixed.py::test_{name}": 150.0 for name in "abcd"}))
    url, stored, _ = cache_worker
    verdict = cached_checkout(ci, tmp_path, monkeypatch, cache_worker)

    def counts():
        return [len((tmp_path / f"runs-{name}").read_text()) if (tmp_path / f"runs-{name}").exists() else 0
                for name in "abcd"]

    armada_task(ci, monkeypatch, verdict, "on")
    print("COUNTS1", counts())
    common = ci.shared("3.12")
    for key, body in sorted(stored.items()):
        if body and "inputs" in key and key.endswith(("1-of-2", "2-of-2")):
            record = json.loads(body)
            recomputed = ci.cache_key(common, "tests/test_mixed.py", ci.parsed_row(record["row"]["name"])[2],
                                      record["depends"], None, ci.cells_ran(record["row"], whole=False))
            print("RECORD", key.split("/")[-1], "key", (record["key"] or "None")[:12], "recomputed", recomputed[:12],
                  "cells", record["row"].get("whole_suite"), "depends", len(record["depends"] or []))
            if record["key"] != recomputed:
                for item in record["depends"]:
                    print("   DEP", item)
    armada_task(ci, monkeypatch, verdict, "on")
    print("COUNTS2", counts())
