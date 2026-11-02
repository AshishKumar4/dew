import json
import sys
from pathlib import Path

import pytest
from armada_support import lifetime_task, loaded

import test_armada_recorder as base


@pytest.fixture
def ci(tmp_path, monkeypatch):
    return loaded(tmp_path, monkeypatch)


def test_fork_debug(ci, tmp_path, monkeypatch, cache_worker):
    started = "import os, subprocess, sys\ndef test_child():\n    "
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "plain.py": "def helper(): return 1\n",
        "reexec_fork.py": "import os, sys\n"
                          "if os.environ.get('REEXECED') != '1':\n"
                          "    child = os.fork()\n"
                          "    if child == 0:\n"
                          "        bare = {name: value for name, value in os.environ.items()\n"
                          "                if name != 'DEW_EXECUTED_FOLLOWER'}\n"
                          "        bare['REEXECED'] = '1'\n"
                          "        os.execve(sys.executable, [sys.executable, __file__], bare)\n"
                          "    os.waitpid(child, 0)\n"
                          "    again = {**os.environ, 'REEXECED': '1'}\n"
                          "    os.execve(sys.executable, [sys.executable, __file__], again)\n"
                          "sys.path.insert(0, 'tests')\nimport plain\nassert plain.helper()\n",
        "test_reexecs_fork.py": started + "again = [sys.executable, 'tests/reexec_fork.py']\n"
                                          "    subprocess.run(again, check=True)\n"}, {})
    found = []
    for path in sorted(tmp_path.rglob("*.log")):
        found.append((str(path.relative_to(tmp_path)), path.read_text()[-1500:]))
    common = ci.shared("3.12")
    record = ci.recorded(ci.record_key("3.12", "3.12:tests/test_reexecs_fork.py", ci.device_class(), common), "token")
    print(json.dumps({"reused": reused(), "logs": found, "record": {k: (v if k != "depends" else len(v or [])) for k, v in (record or {}).items()}}, indent=1, default=str)[:12000])
    for name in ("verdict-0.json",):
        print((tmp_path / name).read_text()[:3000])
    raise AssertionError("debug output above")
