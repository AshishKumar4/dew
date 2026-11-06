"""tools/lint_assertions.py's rules, as tools/lint_slop.py reports them: bible §7's banned assertions in a
test, each case reported and its neighbours spared."""

import pytest
from tool_support import linted

BANNED_ASSERTION_LINTS = {
    "SLOP008": ({
        "src/sparx/store.py": "class Store:\n    def _open(self):\n        pass\n",
        "tests/test_store.py": (
            "from unittest import mock\n"
            "from sparx.store import Store\n"
            "def test_spied(monkeypatch):\n"
            "    store = Store()\n"
            "    monkeypatch.setattr(store, '_open', lambda: None)\n"
            "    with mock.patch.object(store, '_open', wraps=store._open):\n"
            "        pass\n"
            "def test_public_and_dunder(monkeypatch):\n"
            "    store = Store()\n"
            "    monkeypatch.setattr(store, 'path', 'x')\n"
            "    monkeypatch.setattr(store, '__len__', lambda: 1)\n"),
    }, "tests/test_store.py", (5, 6)),
    "SLOP011": ({
        "src/sparx/gate.py": "def main():\n    print('failed: quiet')\n",
        "tests/test_gate.py": (
            "from sparx import gate\n"
            "def test_table(capsys):\n"
            "    def judged():\n"
            "        code = gate.main()\n"
            "        return code, capsys.readouterr().out\n"
            "    code, table = judged()\n"
            "    assert code is None and 'failed: quiet' in table\n"
            "    code, table = judged()\n"
            "    table = table.split()\n"
            "    assert 'quiet' in table\n"),
    }, "tests/test_gate.py", (7,)),
    "SLOP029": ({
        "src/sparx/cli.py": "def main():\n    pass\n",
        "tests/test_cli.py": (
            "import subprocess\n"
            "import json\n"
            "def test_refused_by_its_exit_alone():\n"
            "    done = subprocess.run(['sparx', '--bad'], capture_output=True)\n"
            "    assert done.returncode != 0, done.stderr\n"
            "def test_call_refused():\n"
            "    assert subprocess.call(['sparx', '--bad'])\n"
            "def test_refused_as_a_crash_exits():\n"
            "    assert subprocess.run(['sparx', '--bad']).returncode == 1\n"
            "def test_its_own_code():\n"
            "    assert subprocess.call(['sparx', '--bad']) == 2\n"
            "def test_refused_with_its_state():\n"
            "    done = subprocess.run(['sparx', '--bad'], capture_output=True)\n"
            "    assert done.returncode == 2\n"
            "    assert json.loads(done.stdout) == {'refused': 'bad'}\n"
            "def test_runs():\n"
            "    assert subprocess.run(['sparx']).returncode == 0\n"),
    }, "tests/test_cli.py", (3, 6, 8)),
}
"""Bible §7's banned assertions the slop lint enforces (owner, 2026-11-06), a case each, by code: the files
linted, the file the findings are in, their lines. SLOP008 (class 3): a private attribute of an object the
test made, spied on or replaced; a public attribute or a dunder is not. SLOP011 (class 5): a sentence looked
for in captured output that a helper returns, read through its unpacked name; a name rebound to its parse is
not.
SLOP029 (class 7): a test whose every assertion is that a process failed, from its exit status alone (1, an
uncaught error's, among them); one that asserts the state the refusal reports as well, a code of the process's
own, or that a process ran (exit 0, its own checks inside it), is not."""


@pytest.mark.parametrize("code", sorted(BANNED_ASSERTION_LINTS))
def test_the_banned_assertions_the_lint_reads_are_reported_and_the_rest_spared(code, tmp_path):
    files, where, lines = BANNED_ASSERTION_LINTS[code]
    assert linted(tmp_path, files) == [(where, line, code) for line in lines]
