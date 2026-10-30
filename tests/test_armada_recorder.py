"""What Dew's CI records of a test file's run (tools/armada/lifetime.py, followed.py, executed.py), whose
record keys the file's row in the test cache (tools/armada/ci.py's `cache_key`): the code its run executed,
read, imported and started, for the process's whole life, and its children's."""

import json

import pytest
from armada_support import ON_SRC, PASSING, cached, interpreter, lifetime_task, loaded, reusable_now, written
from tool_support import load


@pytest.fixture
def ci(tmp_path, monkeypatch):
    return loaded(tmp_path, monkeypatch)


def test_a_python_child_joins_its_files_record(ci, tmp_path, monkeypatch, cache_worker):
    """A Python child a test file starts with its environment records itself
    (tools/armada/followed.py), once and silently though its site-packages are
    processed again: its reads of the backend join the file's evidence, and
    the files it executed and the pages it read the file's key, which holds
    until one changes. A child given an environment of its own, or killed
    before it wrote its record, leaves its file without a key."""
    url, _, _ = cache_worker
    cached(monkeypatch, url)
    started = "import os, subprocess, sys\ndef test_child():\n    "
    files = {
        "devices.py": "class Device:\n    platform = 'gpu'\ndef placed(device): return device.platform\n",
        "plain.py": "def helper(): return 1\n",
        "child.py": "import site, sysconfig\n"
                    "site.addsitedir(sysconfig.get_paths()['purelib'])\n"
                    "import devices, plain\n"
                    "assert devices.placed(devices.Device()) == 'gpu' and plain.helper()\n"
                    "assert open('notes/page.md').read()\n",
        "test_child.py": started + "done = subprocess.run([sys.executable, 'tests/child.py'], check=True,\n"
                                   "                      capture_output=True, text=True)\n"
                                   "    assert not done.stderr, done.stderr\n",
        "test_quiet.py": started + "subprocess.run([sys.executable, '-c', 'pass'], check=True)\n",
        "test_killed.py": started + "child = subprocess.Popen([sys.executable, '-c', 'input()'], stdin=-1)\n"
                                    "    child.kill()\n    child.wait()\n",
        "test_own_environment.py": started + "subprocess.run([sys.executable, '-c', 'pass'], check=True,\n"
                                             "                   env={'PATH': os.environ['PATH']})\n",
    }
    written(tmp_path, files)
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes/page.md").write_text("page\n")
    interpreter(tmp_path, "3.12")
    tests = [f"tests/{name}" for name in files if name.startswith("test_")]
    ci.task("3.12", tests, "", tmp_path / "verdict.json", 600.0)

    def reused() -> list[bool]:
        return reusable_now(ci, tests)

    assert reused() == [True, True, False, False]
    (tmp_path / "notes/page.md").write_text("page again\n")
    assert reused() == [False, True, False, False]
    (tmp_path / "notes/page.md").write_text("page\n")
    assert reused() == [True, True, False, False]
    (tmp_path / "tests/plain.py").write_text("def helper(): return 2 - 1\n")
    assert reused() == [False, True, False, False]
    assert ci.gpu_tier("3.12") == {"tests/test_child.py": ["tests/devices.py:placed"],
                                   "tests/test_killed.py": ["changed"],
                                   "tests/test_own_environment.py": ["changed"]}


def test_what_a_conftest_starts_before_any_test_is_every_files_input(ci, tmp_path, monkeypatch,
                                                                     cache_worker):
    """A Python child a conftest starts before the first test, which no test file owns, runs for every
    file the process runs: the code it executes is each one's input, so an edit to it reruns them all. A
    start a conftest makes that no event follows (a shell) leaves every file without a key, and their
    tests still pass."""
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "conftest.py": "import subprocess, sys\nsubprocess.run([sys.executable, 'src/x.py'], check=True)\n",
        **PASSING}, {"x.py": "print('x')\n"})
    assert reused() == [True, True]
    (tmp_path / "src/x.py").write_text("print('x, again')\n")
    assert reused() == [False, False]
    (tmp_path / "tests/conftest.py").write_text(
        "import subprocess\nsubprocess.run(['sh', '-c', 'true'], check=True)\n")
    ci.task("3.12", ["tests/test_a.py", "tests/test_b.py"], "", tmp_path / "verdict.json", 600.0)
    assert reused() == [False, False]


@pytest.mark.parametrize("late", [
    "import pytest\n@pytest.fixture(scope='session', autouse=True)\ndef held():\n    yield\n"
    "    import late\n",
    "import pytest\n@pytest.hookimpl(trylast=True)\ndef pytest_sessionfinish(session):\n    import late\n",
    "import atexit\natexit.register(lambda: __import__('late'))\n",
], ids=["a session fixture's finalizer", "a trylast sessionfinish", "an atexit"])
def test_what_runs_after_the_last_test_is_every_files_input(ci, tmp_path, monkeypatch, cache_worker, late):
    """What a session runs after its last test, which no test file owns (a session fixture's finalizer,
    a late sessionfinish hook, an atexit a conftest registered), is every file's input: an edit to it
    reruns them all."""
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {"conftest.py": ON_SRC + late,
                                                                     **PASSING},
                           {"late.py": "VALUE = 1\n"})
    assert reused() == [True, True]
    (tmp_path / "src/late.py").write_text("VALUE = 2\n")
    assert reused() == [False, False]


@pytest.mark.parametrize("hook", [
    "def pytest_fixture_post_finalizer(fixturedef, request):\n    if fixturedef.scope == 'session':\n"
    "        import late\n",
    "@pytest.hookimpl(hookwrapper=True)\ndef pytest_fixture_post_finalizer(fixturedef, request):\n"
    "    yield\n    if fixturedef.scope == 'session':\n        import late\n",
], ids=["a hook", "a hook wrapper"])
def test_what_a_session_fixtures_post_finalizer_hook_runs_is_every_files_input(ci, tmp_path, monkeypatch,
                                                                                cache_worker, hook):
    """What a hook runs once a session fixture's ending is done (pytest_fixture_post_finalizer, a hook of the
    root's conftest, whose session node it serves), a plain hook or a wrapper around the others, is every
    file's input: an edit to it reruns them all."""
    (tmp_path / "conftest.py").write_text(
        ON_SRC + "import pytest\n@pytest.fixture(scope='session', autouse=True)\ndef held():\n    yield\n"
        + hook)
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, PASSING, {"late.py": "VALUE = 1\n"})
    assert reused() == [True, True]
    (tmp_path / "src/late.py").write_text("VALUE = 2\n")
    assert reused() == [False, False]


@pytest.mark.parametrize("rewritten", [False, True], ids=["plain", "under pytest's assertion rewriting"])
def test_a_module_imported_by_name_is_its_files_input_and_no_earlier_files(ci, tmp_path, monkeypatch,
                                                                             cache_worker, rewritten):
    """A module a test imports by name at run time (importlib.import_module) is its file's input as an
    import statement's is, whichever finder pytest put first (its assertion rewriter, for a name a conftest
    registered): a helper of that name added under tests/, which leads sys.path, reruns the file, and no
    file the process ran before it, which never saw that module."""
    registered = "import pytest\npytest.register_assert_rewrite('tabnanny')\n"
    conftest = {"conftest.py": registered} if rewritten else {}
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        **conftest, "test_b.py": PASSING["test_b.py"],
        "test_named.py": "import importlib\ndef test_named():\n"
                         "    assert importlib.import_module('tabnanny')\n"}, {})
    assert reused() == [True, True]
    (tmp_path / "tests/tabnanny.py").write_text("")
    assert reused() == [True, False]


def test_a_file_a_session_parameter_revisits_takes_what_ran_between_its_visits(ci, tmp_path, monkeypatch,
                                                                                cache_worker):
    """A session-scoped parameter runs A[0], B[0], A[1], B[1]: what B's first visit executed and cached
    serves A's second, so it is A's input though A ran first; an edit to it reruns A."""
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "conftest.py": ON_SRC + "import pytest\n@pytest.fixture(scope='session', params=[0, 1])\n"
                                "def visit(request):\n    return request.param\n",
        "holder.py": "VALUES = {}\n",
        "test_a.py": "import holder\ndef test_a(visit):\n"
                     "    assert visit == 0 or holder.VALUES['m'] == 1\n",
        "test_b.py": "import holder\ndef test_b(visit):\n    import m\n"
                     "    holder.VALUES.setdefault('m', m.VALUE)\n"},
        {"m.py": "VALUE = 1\n"})
    assert reused() == [True, True]
    (tmp_path / "src/m.py").write_text("VALUE = 2\n")
    assert reused() == [False, False]


def test_each_start_needs_its_own_record_though_two_start_alike(ci, tmp_path, monkeypatch, cache_worker):
    """Two children a file starts alike are two starts, each owing a record: two that record themselves
    keep the file's key, and one that records itself beside one that starts with no site (-S), which no
    recorder sees, leave the file without one, so an edit to what the second ran cannot reuse its row."""
    plain = "    subprocess.run([sys.executable, '-c', 'pass'], check=True)\n"
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "test_two.py": "import subprocess, sys\ndef test_two():\n" + plain * 2,
        "test_twice.py": "import subprocess, sys\ndef test_twice():\n" + plain
                         + "    subprocess.run([sys.executable, '-S', 'src/other.py'], check=True)\n",
        "test_b.py": PASSING["test_b.py"]}, {"other.py": "print('other')\n"})
    assert reused() == [True, False, True]


MODULE = """import functools
LIMIT = 3


def scaled(x, by=2):
    return x * by


def unused(x):
    return x - 1


class Shape:
    sides = 4

    def area(self, side):
        return side * side

    def perimeter(self, side):
        return self.sides * side
"""


@pytest.mark.parametrize(("edit", "rerun"), [
    (("def unused(x):\n    return x - 1", "def unused(x):\n    return x - 2"), [False, False]),
    (("    return x * by", "    return by * x + 0"), [True, True]),
    (("        return side * side", "        return side ** 2"), [False, True]),
    (("        return self.sides * side", "        return side * self.sides"), [False, False]),
    (("def scaled(x, by=2):", "def scaled(x, by=3):"), [True, True]),
    (("LIMIT = 3", "LIMIT = 4"), [True, True]),
    (("    sides = 4", "    sides = 5"), [True, True]),
    (("import functools\n", "import functools\n\n\n# a note above\n\n"), [False, False]),
], ids=["a function no test ran", "a function the first file ran", "a method the last file ran",
        "a method no test ran", "a default", "a module's constant", "a class's attribute", "lines moved"])
def test_an_edit_reruns_the_files_whose_run_executed_what_it_changed(ci, tmp_path, monkeypatch,
                                                                       cache_worker, edit, rerun):
    """A file's key covers the code its run executed, each function it ran and the top level of each module
    it imported (all but function bodies: its imports, constants, classes' attributes and methods'
    signatures, defaults and decorators), not a module's whole text: an edit to a function no run executed,
    or one that only moves lines, reruns no file, and an edit to what a run executed reruns it and the files
    its process ran after it, which what it ran may serve."""
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "test_a.py": ON_SRC + "import m\ndef test_a():\n    assert m.scaled(2) == 4\n",
        "test_b.py": ON_SRC + "import m\ndef test_b():\n    assert m.Shape().area(3) == 9\n"},
        {"m.py": MODULE})
    assert reused() == [True, True]
    old, changed = edit
    (tmp_path / "src/m.py").write_text(MODULE.replace(old, changed))
    assert reused() == [not each for each in rerun]


@pytest.mark.parametrize(("seen", "edit"), [
    ("assert m.unused.__doc__ == 'Less one.'", ('"""Less one."""', '"""Less two."""')),
    ("import inspect\n    assert inspect.isgeneratorfunction(m.counted)", ("    yield x", "    return x")),
    ("assert 1 in m.unused.__code__.co_consts", ("    return x - 1", "    return x - 2")),
    ("namespace = {}\n    exec(m.__spec__.loader.get_source('m'), namespace)\n"
     "    assert namespace['unused'](2) == 1", ("    return x - 1", "    return x - 2")),
    ("pass", ("    return x - 1", "    return x - 1\n    break")),
], ids=["a docstring", "a generator's kind", "a function's code", "a module's text it executed itself",
        "a body that no longer compiles"])
def test_what_a_run_sees_of_code_it_never_entered_is_its_input(ci, tmp_path, monkeypatch, cache_worker,
                                                               seen, edit):
    """What a run sees of a function it never entered is its input as the code it ran is: the function
    value's docstring and kind (a generator's), its code object (`__code__`), the module's text it compiled
    itself, and whether the module still compiles, which a body left out would hide."""
    module = ('import sys\n\n\ndef unused(x):\n    """Less one."""\n    return x - 1\n\n\n'
              'def counted(x):\n    yield x\n')
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "test_seen.py": ON_SRC + f"import m\ndef test_seen():\n    {seen}\n"}, {"m.py": module})
    assert reused() == [True]
    old, changed = edit
    (tmp_path / "src/m.py").write_text(module.replace(old, changed))
    assert reused() == [False]


COMPILED = (
    "import types\n\n\nclass Meta(type):\n    def __new__(mcls, name, bases, namespace):\n"
    "        namespace['celled'] = '__classcell__' in namespace\n"
    "        return super().__new__(mcls, name, bases, namespace)\n\n\n"
    "class Plain(metaclass=Meta):\n    def unused(self):\n        return 1\n\n\n"
    "def outer():\n    def inner(x=(yield 1)):\n        pass\n    return inner\n")


@pytest.mark.parametrize(("seen", "edit"), [
    ("assert isinstance(m.outer(), types.GeneratorType)", ("x=(yield 1)", "x=1")),
    ("assert not m.Plain.celled", ("        return 1", "        return super()")),
], ids=["a yield in a nested default", "a method's super()"])
def test_what_compiling_a_body_makes_of_the_code_around_it_is_its_modules_input(ci, tmp_path,
                                                                                 monkeypatch, cache_worker,
                                                                                 seen, edit):
    """What compiling a function's body makes of the code around it, seen without entering the body, is
    its module's top level's input: whether calling the function builds a generator (a yield in a nested
    definition's default, which runs in its scope), and the cell a class body makes for a method's super()."""
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "test_seen.py": ON_SRC + f"import types\nimport m\ndef test_seen():\n    {seen}\n"},
        {"m.py": COMPILED})
    assert reused() == [True]
    old, changed = edit
    (tmp_path / "src/m.py").write_text(COMPILED.replace(old, changed))
    assert reused() == [False]


def test_a_module_imported_from_its_bytecode_alone_is_its_files_input(ci, tmp_path, monkeypatch,
                                                                       cache_worker):
    """A module imported from bytecode with no source beside it is its whole bytecode: an edit to it reruns
    the file that imported it."""
    import py_compile

    url, _, _ = cache_worker
    cached(monkeypatch, url)
    written(tmp_path, {"test_bytecode.py": ON_SRC + "import m\ndef test_bytecode():\n    assert m.VALUE\n"})
    (tmp_path / "src").mkdir()
    interpreter(tmp_path, "3.12")

    def placed(value: int) -> None:
        source = tmp_path / "src/m_source.py"
        source.write_text(f"VALUE = {value}\n")
        py_compile.compile(str(source), cfile=str(tmp_path / "src/m.pyc"), dfile=str(tmp_path / "src/m.py"))
        source.unlink()

    placed(1)
    ci.task("3.12", ["tests/test_bytecode.py"], "", tmp_path / "verdict.json", 600.0)
    assert reusable_now(ci, ["tests/test_bytecode.py"]) == [True]
    placed(2)
    assert reusable_now(ci, ["tests/test_bytecode.py"]) == [False]


def test_a_process_ending_1_after_its_tests_passed_is_red(ci, tmp_path, monkeypatch):
    """A pytest process that ends 1 though every test it reported passed (a late hook's os._exit) is no
    green: its tests ran in a process that failed, and every row it held is red."""
    written(tmp_path, {"conftest.py": "import os, pytest\n@pytest.hookimpl(trylast=True)\n"
                                      "def pytest_sessionfinish(session):\n    os._exit(1)\n", **PASSING})
    interpreter(tmp_path, "3.12")
    ci.task("3.12", ["tests/test_a.py", "tests/test_b.py"], "", tmp_path / "verdict.json", 600.0)
    rows = json.loads((tmp_path / "verdict.json").read_text())["rows"]
    assert [(row["exitCode"], "ended 1 after its tests passed" in row["output"]) for row in rows] \
        == [(1, True)] * 2


def test_each_files_record_is_its_own_where_files_share_a_process(ci, tmp_path, monkeypatch, cache_worker):
    """Each input moves exactly the files that may have seen it (row 2642): a fixture's arrival reruns its
    prober, the process's last file, alone; a helper shadowing a name only a file's child imported reruns
    that file alone; a module one file imported lazily reruns it and every file the process ran after it,
    which what it left may serve; and two files whose records share inputs never take each other's row."""
    child = ("import site, sysconfig; site.addsitedir(sysconfig.get_paths()['purelib']); "
             "import tabnanny")
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "test_child.py": f"import subprocess, sys\ndef test_child():\n"
                         f"    subprocess.run([sys.executable, '-c', {child!r}], check=True)\n",
        "test_lazy.py": ON_SRC + "import holder\ndef test_lazy():\n    import lazy\n"
                                 "    holder.VALUES['lazy'] = lazy.VALUE\n",
        "test_later.py": "import holder\ndef test_later():\n    assert holder.VALUES.get('lazy', 1) == 1\n",
        "test_probe.py": "import os\ndef test_probe():\n"
                         "    assert not os.path.exists('tests/fixtures/f.txt')\n",
        "holder.py": "VALUES = {}\n"}, {"lazy.py": "VALUE = 1\n"})
    assert reused() == [True, True, True, True]
    (tmp_path / "tests/fixtures").mkdir()
    (tmp_path / "tests/fixtures/f.txt").write_text("")
    assert reused() == [True, True, True, False]
    (tmp_path / "tests/fixtures/f.txt").unlink()
    (tmp_path / "tests/tabnanny.py").write_text("")
    assert reused() == [False, True, True, True]
    (tmp_path / "tests/tabnanny.py").unlink()
    (tmp_path / "src/lazy.py").write_text("VALUE = 2\n")
    assert reused() == [True, False, False, False]
    (tmp_path / "src/lazy.py").write_text("VALUE = 1\n")
    common = ci.shared("3.12")
    names = ("tests/test_lazy.py", "tests/test_later.py")
    found = [ci.reused("3.12", name, "", common)["name"] for name in names]
    assert found == [f"3.12:{name}" for name in names]


def test_a_files_children_hold_what_they_held_at_once_and_each_start_must_leave_its_record(tmp_path):
    """The children's logs join the file that started them (`followed.gathered`): their files, and the
    GPU they held, the most a child that took the parent's fraction used, those children as shares and
    the bytes of those that named their own, counted only while they lived at once. A file is followed
    only where every start it and its children saw left a closed log naming it and none started a
    process it did not follow; one whose child's log never closed, killed before its end, is not."""
    followed = load("followed", "tools/armada")
    gib = 2 ** 30

    def child(index, file, began, ended, pool=None, inherited=True, expected=0, unfollowed=False, peak=None,
              closed=True):
        lines = [f"owner\t{file}", f"inherited\t{int(inherited)}", f"began\t{began}", f"file\tsrc/{index}.py",
                 *["start\tfollowed"] * expected, *(["unfollowed\t1"] if unfollowed else []),
                 *([f"pool\t{pool}"] if pool else []), *([f"peak\t{peak}"] if peak is not None else []),
                 *([f"ended\t{ended}", "closed\t1"] if closed else [])]
        (tmp_path / f"{index}.log").write_text("\n".join(lines) + "\n")

    child(1, "tests/a.py", 0, 10, pool=6 * gib, peak=5 * gib)
    child(2, "tests/a.py", 5, 15, pool=6 * gib, expected=1, peak=gib)
    child(3, "tests/a.py", 20, 30, pool=3 * gib, inherited=False)
    child(4, "tests/a.py", 25, 35, pool=2 * gib, inherited=False)
    child(5, "tests/a.py", 40, 41)
    child(6, "tests/b.py", 0, 1)
    child(7, "tests/c.py", 0, 1)
    child(8, "tests/c.py", 0, 1, closed=False)
    child(9, "tests/d.py", 0, 1, unfollowed=True)
    found = followed.gathered(tmp_path, {"tests/a.py": 4, "tests/b.py": 2, "tests/c.py": 2, "tests/d.py": 1})
    assert found["tests/a.py"] == {"files": {f"src/{index}.py" for index in range(1, 6)}, "reads": set(),
                                   "followed": True, "peak": 5 * gib, "shares": 2, "pools": 5 * gib}
    assert [found[name]["followed"] for name in ("tests/b.py", "tests/c.py", "tests/d.py")] == [False] * 3
