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
    "SLOP033": ({
        "src/sparx/tiles.py": "def derived(n):\n    return n\n",
        "tests/test_tiles.py": (
            "from sparx import tiles\n"
            "def test_derived_stages_pinned():\n"
            "    chosen = tiles.derived(8)\n"
            "    assert chosen.stages == 3\n"
            "def test_a_block_pinned_off_the_call():\n"
            "    assert tiles.choose(8).block_m == 128\n"
            "def test_each_offered_pinned():\n"
            "    for config in tiles.offered(8):\n"
            "        assert config.consumers >= 2\n"
            "def test_annotated_and_chained():\n"
            "    chosen: tiles.Config = tiles.derived(8)\n"
            "    assert tiles.budget(8) >= chosen.block_n > 16\n"
            "def test_legal():\n"
            "    chosen = tiles.derived(8)\n"
            "    assert 0 < tiles.budget(8) <= chosen.stages * chosen.block_m\n"
            "def test_rebound_to_its_own_config():\n"
            "    config = tiles.derived(8)\n"
            "    config = tiles.Config(stages=3)\n"
            "    assert config.stages == 3\n"
            "def test_another_field():\n"
            "    chosen = tiles.derived(8)\n"
            "    assert chosen.dtype_bits == 16\n"),
    }, "tests/test_tiles.py", (4, 6, 9, 12)),
}
"""Bible §7's banned assertions the slop lint enforces (owner, 2026-11-06), a case each, by code: the files
linted, the file the findings are in, their lines. SLOP008 (class 3): a private attribute of an object the
test made, spied on or replaced; a public attribute or a dunder is not. SLOP011 (class 5): a sentence looked
for in captured output that a helper returns, read through its unpacked name; a name rebound to its parse is
not.
SLOP029 (class 7): a test whose every assertion is that a process failed, from its exit status alone (1, an
uncaught error's, among them); one that asserts the state the refusal reports as well, a code of the process's
own, or that a process ran (exit 0, its own checks inside it), is not.
SLOP033 (class 2): a choice the package derives (a config a call named for choosing returns, bound,
annotated or iterated, or the call's own) pinned to a number beside it by its stages, consumers or a block
size; a legality bound computed against the choice, a name bound again to the test's own config, and
another field are not."""


COMPILE_WIDTH_LINTS = {
    "SLOP032": ({
        "src/sparx/build.py": (
            "import functools\n"
            "import re\n"
            "from concurrent.futures import ThreadPoolExecutor as Pool\n"
            "from sparx.report import compile_threads, xla_compiled\n"
            "def one(program):\n"
            "    return program.compile()\n"
            "def wide(programs):\n"
            "    with Pool(8) as pool:\n"
            "        return list(pool.map(lambda program: program.compile(), programs))\n"
            "def by_its_function(programs):\n"
            "    with Pool(max_workers=len(programs)) as pool:\n"
            "        return list(pool.map(one, programs))\n"
            "def bound(lowered):\n"
            "    pool = Pool(4)\n"
            "    return pool.submit(lowered.compile).result()\n"
            "def partial(programs):\n"
            "    return list(Pool(4).map(functools.partial(one), programs))\n"
            "def past_the_width(programs):\n"
            "    with Pool(compile_threads() + 8) as pool:\n"
            "        return list(pool.map(one, programs))\n"
            "def capped(programs):\n"
            "    with Pool(min(4, compile_threads())) as pool:\n"
            "        return list(pool.map(one, programs))\n"
            "def single(lowered):\n"
            "    with Pool(1) as pool:\n"
            "        return pool.submit(lowered.compile).result()\n"
            "def beside_a_compile(lowered, paths):\n"
            "    lowered.compile()\n"
            "    pattern = re.compile('x')\n"
            "    with Pool(8) as pool:\n"
            "        return list(pool.map(pattern.match, paths))\n"
            "def through_the_compile_point(lowered):\n"
            "    with Pool(8) as pool:\n"
            "        return pool.submit(xla_compiled, lowered, None).result()\n"),
        "src/sparx/nn/kernels/templates/report.py": (
            "from concurrent.futures import ThreadPoolExecutor\n"
            "def compiled_at_once(programs):\n"
            "    with ThreadPoolExecutor(8) as pool:\n"
            "        return list(pool.map(lambda program: program.compile(), programs))\n"),
    }, "src/sparx/build.py", (8, 11, 14, 17, 19)),
}
"""SLOP032: a pool handed work that compiles by its own `compile` (a lambda, a function of the module, a
partial of one, the bound method), by any name the executor is imported under, at a width of its own (a
number, the work's length, one the compile width does not cap); one capped by the compile width
(`compile_threads()`), one worker, a pool beside a compile that hands it none, a compile through the compile
point (`xla_compiled`), and the width's own module are not."""


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


CASES = [(code, *case)
         for table in (ROW_1878_LINTS, BANNED_ASSERTION_LINTS, IR_TEXT_LINTS, COMPILE_WIDTH_LINTS)
         for code, case in sorted(table.items())]
"""Every table's cases: a code may have one in each."""


@pytest.mark.parametrize(("code", "files", "where", "lines"), CASES)
def test_each_lint_reports_its_cases_and_spares_the_rest(code, files, where, lines, tmp_path):
    assert linted(tmp_path, files) == [(where, line, code) for line in lines]
