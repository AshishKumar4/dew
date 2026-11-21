"""tools/lint_slop.py's case tables: row 1878's lints and bible §7's banned assertions in a test
(tools/lint_assertions.py), each case reported and its neighbours spared."""

import pytest
from tool_support import linted

ROW_1878_LINTS = {
    "SLOP008": ({
        "src/sparx/store.py": "class Store:\n    def _open(self):\n        pass\ndef open(path):\n    pass\n",
        "tests/test_store.py": (
            "from sparx.store import Store\n"
            "import sparx.store as store\n"
            "def test_private(monkeypatch):\n"
            "    monkeypatch.setattr(Store, '_open', lambda self: None)\n"
            "def test_public(monkeypatch):\n"
            "    monkeypatch.setattr(store, 'open', lambda path: None)\n"
            "def test_a_seams_own_private(monkeypatch):\n"
            "    monkeypatch.setattr('sparx.store.os._exit', lambda code: None)\n"
            "def test_a_first_party_private_under_a_seam_word(monkeypatch):\n"
            "    monkeypatch.setattr('sparx.sources.hub.Hub._selected', lambda self: None)\n"),
    }, "tests/test_store.py", (4, 10)),
    "SLOP011": ({
        "src/sparx/graph.py": "EDGES = 1\n",
        "tests/test_graph.py": (
            "def test_table(table):\n"
            "    assert report.table().splitlines() == ['   a  b', 'x  1  2']\n"
            "    assert table.plain == 'data 4,824 rows  ·  0 rejected'\n"
            "    assert '4,824' in table.split()\n"
            "    assert table.title == 'data rows'\n"
            "    assert table.index('b') + 1 == len('   a  b')\n"
            "    assert json.loads(payload) == ['a  b']\n"
            "    assert parsed.name == 'two  spaces'\n"),
    }, "tests/test_graph.py", (2, 3)),
}
"""Row 1878's two lint cases, by code: the files linted, the file the findings are in, their lines.
SLOP008: a private attribute of a first-party object is the code under test, so replacing it is
reported even where its name holds a seam's (`Checkpoints._open`); a public seam of the same name and a
standard-library module's own private attribute are not. SLOP011: a rendering (its text read through
`plain`, `table()`, captured output or `str`) compared whole with a string, or a list of strings, with a
run of two or more spaces between words is a spacing snapshot; its words and numbers looked for, a
one-space string, a column end measured and parsed data holding two spaces are not."""


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


IR_TEXT_LINTS = {
    "SLOP011": ({
        "src/sparx/graph.py": "def parse(text):\n    return text.split()\n",
        "tests/test_graph.py": (
            "import re\n"
            "from pathlib import Path\n"
            "import jax\n"
            "import sparx.graph\n"
            "from sparx.graph import parse\n"
            "ROOT = Path(__file__).parents[1]\n"
            "def test_source():\n"
            "    text = (ROOT / 'src' / 'sparx' / 'graph.py').read_text()\n"
            "    assert 'def parse' in text\n"
            "def test_module_file():\n"
            "    lines = Path(sparx.graph.__file__).read_text().splitlines()\n"
            "    assert len(lines) > 1\n"
            "def test_opened():\n"
            "    with open(ROOT / 'src' / 'sparx' / 'graph.py') as handle:\n"
            "        assert handle.read()\n"
            "def test_jaxpr(f, x):\n"
            "    assert 'dot_general' in str(jax.make_jaxpr(f)(x))\n"
            "def test_lowered(f, x):\n"
            "    program = jax.jit(f).lower(x).as_text()\n"
            "    assert not re.findall(r'while', program)\n"
            "def test_formatted(f, x):\n"
            "    jaxpr = jax.make_jaxpr(f)(x)\n"
            "    assert 'sin' in f'{jaxpr}'\n"
            "def test_structure(f, x):\n"
            "    jaxpr = jax.make_jaxpr(f)(x)\n"
            "    assert 'sin' in {equation.primitive.name for equation in jaxpr.eqns}\n"
            "def test_parsed(f, x):\n"
            "    assert parse(jax.jit(f).lower(x).as_text())\n"
            "def test_own_file():\n"
            "    assert (ROOT / 'tests' / 'data.txt').read_text()\n"
            "def test_rebound(f, x):\n"
            "    program = jax.jit(f).lower(x).as_text()\n"
            "    program = parse(program)\n"
            "    assert program\n"
            "def test_a_strings_case(name):\n"
            "    mode = name.lower()\n"
            "    assert f'{mode}' == 'x' and 'a' in str(re.compile('a'))\n"),
    }, "tests/test_graph.py", (9, 12, 15, 17, 20, 23)),
}
"""Row 4512's case: SLOP011 on a test asserting on text it read from the package's source (a path under
src/sparx/, joined from "src" and the package, or a module's `__file__`, read or opened) or from a program's
IR (a jaxpr printed, a lowering's or compilation's `as_text()`), through the names it binds and the string
operations it matches with (`lint_texts`). The program's structure read as objects, the text handed to the
package's own parser, a file of the tests' own, a name bound again to something else, and a `lower` or
`compile` of no program (a string's case, a regular expression) are not reported."""


CASES = [(code, *case) for table in (ROW_1878_LINTS, BANNED_ASSERTION_LINTS, IR_TEXT_LINTS)
         for code, case in sorted(table.items())]
"""Every table's cases: a code may have one in each."""


@pytest.mark.parametrize(("code", "files", "where", "lines"), CASES)
def test_each_lint_reports_its_cases_and_spares_the_rest(code, files, where, lines, tmp_path):
    assert linted(tmp_path, files) == [(where, line, code) for line in lines]
