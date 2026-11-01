import json
import sys
import time

import pytest
import test_armada_ci as base
from armada_support import loaded, written


@pytest.fixture
def ci(tmp_path, monkeypatch):
    return loaded(tmp_path, monkeypatch)


def test_debug_deadline(ci, tmp_path, monkeypatch, cache_worker):
    verdict, _ = base.planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    names = "cde"
    written(tmp_path, {"test_c.py": "import time\nfrom pathlib import Path\n" + "".join(
        f"def test_{name}():\n    Path('runs-{name}').write_text('x')\n    time.sleep(3)\n"
        for name in names)})
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {f"tests/test_c.py::test_{name}": 250.0 for name in names}))
    groups = " ".join(f"tests/test_c.py#{index}/3@GenuineIntel-6-143:0000" for index in (1, 2, 3))
    monkeypatch.setattr(sys, "argv", ["ci.py", "task", "--python=3.12", f"--tests={groups}",
                                      f"--out={verdict}", "--cache=on", "--deadline=1"])
    start = time.monotonic()
    calls = []
    alone, tested, looked = ci.alone, ci.tested, ci.looked_up

    def alone_(*args, **kwargs):
        calls.append(("alone", round(time.monotonic() - start, 2), args[1], args[2], args[4]))
        result = alone(*args, **kwargs)
        calls.append(("alone done", round(time.monotonic() - start, 2)))
        return result

    def tested_(*args, **kwargs):
        calls.append(("tested", round(time.monotonic() - start, 2), args[1], args[2], args[4]))
        return tested(*args, **kwargs)

    def looked_(*args, **kwargs):
        calls.append(("looked_up", round(time.monotonic() - start, 2), args[1], args[2]))
        return looked(*args, **kwargs)

    monkeypatch.setattr(ci, "alone", alone_)
    monkeypatch.setattr(ci, "tested", tested_)
    monkeypatch.setattr(ci, "looked_up", looked_)
    code = ci.main()
    print("main", round(time.monotonic() - start, 2), "code", code)
    for call in calls:
        print(call)
    print(verdict.read_text()[:2000])
    for path in sorted(tmp_path.glob("runs-*")):
        print(path.name, path.stat().st_mtime)
    for path in sorted(verdict.parent.iterdir()):
        print(path.name, path.stat().st_size, path.read_text(errors="replace")[-600:] if path.is_file() else "")
    raise AssertionError("debug output above")
