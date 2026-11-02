"""What Dew's CI records of a test file's run (tools/armada/lifetime.py, followed.py, executed.py), whose
record keys the file's row in the test cache (tools/armada/ci.py's `cache_key`): the code its run executed,
read, imported and started, for the process's whole life, and its children's."""

import ast
import json
import sys

import pytest
from armada_support import (
    ON_SRC,
    PASSING,
    cached,
    fresh_tasks,
    interpreter,
    lifetime_task,
    loaded,
    reusable_now,
    written,
)
from tool_support import REPO_ROOT, load


@pytest.fixture
def ci(tmp_path, monkeypatch):
    return loaded(tmp_path, monkeypatch)


def test_a_python_child_joins_its_files_record(ci, tmp_path, monkeypatch, cache_worker):
    """A Python child a test file starts with its environment records itself
    (tools/armada/followed.py), once and silently though its site-packages are
    processed again: its reads of the backend join the file's evidence, and
    the files it executed and the pages it read the file's key, which holds
    until one changes, though it leaves through os._exit, it writes each as it
    happens, or execs itself again (dew.devices' restart under XLA's flags),
    whose new image writes on into the log of the old. A child given an
    environment of its own, one that execs itself again with no site (-S),
    whose .pth never runs, or one killed before it wrote its record, leaves its
    file without a key."""
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
        "exit.py": "import os, plain\nassert plain.helper()\nos._exit(0)\n",
        "reexec.py": "import os, sys\nimport plain\n"
                     "if os.environ.get('REEXECED') != '1':\n"
                     "    again = {**os.environ, 'REEXECED': '1'}\n"
                     "    os.execve(sys.executable, [sys.executable, __file__], again)\n"
                     "assert plain.helper()\n",
        # A new image with no site, whose .pth never runs: what it executes no log holds.
        "reexec_bare.py": "import os, sys\n"
                          "if os.environ.get('REEXECED') != '1':\n"
                          "    again = {**os.environ, 'REEXECED': '1'}\n"
                          "    os.execve(sys.executable, [sys.executable, '-S', __file__], again)\n"
                          "sys.path.insert(0, 'tests')\nimport plain\nassert plain.helper()\n",
        # The same with an option and its own argument before -S (review: `utf8` read as the program).
        "reexec_bare_x.py": "import os, sys\n"
                            "if os.environ.get('REEXECED') != '1':\n"
                            "    again = {**os.environ, 'REEXECED': '1'}\n"
                            "    bare = [sys.executable, '-X', 'utf8', '-S', __file__]\n"
                            "    os.execve(sys.executable, bare, again)\n"
                            "sys.path.insert(0, 'tests')\nimport plain\nassert plain.helper()\n",
        "reexec_bare_w.py": "import os, sys\n"
                            "if os.environ.get('REEXECED') != '1':\n"
                            "    again = {**os.environ, 'REEXECED': '1'}\n"
                            "    bare = [sys.executable, '-W', 'ignore', '-S', __file__]\n"
                            "    os.execve(sys.executable, bare, again)\n"
                            "sys.path.insert(0, 'tests')\nimport plain\nassert plain.helper()\n",
        # The same through the environment's base interpreter, which is no recorder's (review: its path's real
        # file is this Python's, but started by its own path it has neither the environment nor its `.pth`).
        "reexec_base.py": "import os, sys\n"
                          "if os.environ.get('REEXECED') != '1':\n"
                          "    again = {**os.environ, 'REEXECED': '1'}\n"
                          "    os.execve(sys._base_executable, [sys._base_executable, __file__], again)\n"
                          "sys.path.insert(0, 'tests')\nimport plain\nassert plain.helper()\n",
        # The same with the option grouped with another (review: `-BX utf8` read the `utf8` as the program).
        "reexec_bare_bx.py": "import os, sys\n"
                             "if os.environ.get('REEXECED') != '1':\n"
                             "    again = {**os.environ, 'REEXECED': '1'}\n"
                             "    bare = [sys.executable, '-BX', 'utf8', '-S', __file__]\n"
                             "    os.execve(sys.executable, bare, again)\n"
                             "sys.path.insert(0, 'tests')\nimport plain\nassert plain.helper()\n",
        "test_child.py": started + "done = subprocess.run([sys.executable, 'tests/child.py'], check=True,\n"
                                   "                      capture_output=True, text=True)\n"
                                   "    assert not done.stderr, done.stderr\n",
        "test_quiet.py": started + "subprocess.run([sys.executable, '-c', 'pass'], check=True)\n",
        "test_exits.py": started + "subprocess.run([sys.executable, 'tests/exit.py'], check=True)\n",
        "test_reexecs.py": started + "subprocess.run([sys.executable, 'tests/reexec.py'], check=True)\n",
        "test_reexecs_bare.py": started + "subprocess.run([sys.executable, 'tests/reexec_bare.py'],\n"
                                          "                   check=True)\n",
        "test_reexecs_bare_x.py": started + "subprocess.run([sys.executable, 'tests/reexec_bare_x.py'],\n"
                                            "                   check=True)\n",
        "test_reexecs_bare_w.py": started + "subprocess.run([sys.executable, 'tests/reexec_bare_w.py'],\n"
                                            "                   check=True)\n",
        "test_reexecs_bare_bx.py": started + "subprocess.run([sys.executable, 'tests/reexec_bare_bx.py'],\n"
                                             "                   check=True)\n",
        "test_reexecs_base.py": started + "subprocess.run([sys.executable, 'tests/reexec_base.py'],\n"
                                          "                   check=True)\n",
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
    fresh_tasks(ci, tests, tmp_path)

    def reused() -> list[bool]:
        return reusable_now(ci, tests)

    assert reused() == [True, True, True, True, False, False, False, False, False, False, False]
    (tmp_path / "notes/page.md").write_text("page again\n")
    assert reused() == [False, True, True, True, False, False, False, False, False, False, False]
    (tmp_path / "notes/page.md").write_text("page\n")
    assert reused() == [True, True, True, True, False, False, False, False, False, False, False]
    (tmp_path / "tests/plain.py").write_text("def helper(): return 2 - 1\n")
    assert reused() == [False, True, False, False, False, False, False, False, False, False, False]
    # A keyed file whose inputs moved is in as changed; one whose records keep no key (a child unfollowed) as
    # unrecorded, as no record is evidence of it.
    assert ci.gpu_tier("3.12") == {"tests/test_child.py": ["tests/devices.py:placed"],
                                   "tests/test_exits.py": ["changed"],
                                   "tests/test_killed.py": ["unrecorded"],
                                   "tests/test_own_environment.py": ["unrecorded"],
                                   "tests/test_reexecs.py": ["changed"],
                                   "tests/test_reexecs_bare.py": ["unrecorded"],
                                   "tests/test_reexecs_bare_bx.py": ["unrecorded"],
                                   "tests/test_reexecs_base.py": ["unrecorded"],
                                   "tests/test_reexecs_bare_w.py": ["unrecorded"],
                                   "tests/test_reexecs_bare_x.py": ["unrecorded"]}


def test_a_multiprocessing_child_joins_its_files_record_unless_a_forkserver_runs_it(
        ci, tmp_path, monkeypatch, cache_worker):
    """A child multiprocessing spawns (a `Process` of its `spawn` context: Grain's loading workers, whose
    start Python 3.12 audits no event of) records itself as a Python child any other start makes does
    (test_a_python_child_joins_its_files_record), though this process runs with -S, which multiprocessing
    passes on to the children it spawns: the functions they ran and the pages they read are their file's
    key, those of a child one of them spawns in turn too (and the resource tracker it starts, which runs
    the standard library alone, records nothing), and an edit to what only a child ran reruns the file. A
    forked child, whose code runs under its parent's hooks and writes into its log, joins the record as it
    is; one a forkserver runs, which no hook reaches, leaves its file without a key."""
    url, _, _ = cache_worker
    cached(monkeypatch, url)
    spawned = ("import multiprocessing\ndef test_child():\n"
               "    child = multiprocessing.get_context('spawn').Process(")
    started = "child.start()\n    child.join()\n    assert child.exitcode == 0\n"
    files = {
        "plain.py": "def helper(): return 1\n",
        "inner.py": "def inner(): return 2\n",
        "forked.py": "def in_fork(): return 3\ndef in_mp_fork(): return 4\n",
        "worker.py": "import multiprocessing\nimport plain\n"
                     "def work(path):\n    assert plain.helper() and open(path).read()\n"
                     "def spawn_inner():\n    import inner\n"
                     "    context = multiprocessing.get_context('spawn')\n"
                     "    done = context.Event()\n"
                     "    child = context.Process(target=inner.inner)\n"
                     "    child.start()\n    child.join()\n    done.set()\n    assert child.exitcode == 0\n",
        "test_spawn.py": "import worker\n" + spawned + "target=worker.work, args=('notes/page.md',))\n    "
                         + started,
        "test_nested.py": "import worker\n" + spawned + "target=worker.spawn_inner)\n    " + started,
        "test_os_fork.py": "import os, forked\ndef test_child():\n    pid = os.fork()\n    if pid == 0:\n"
                           "        try:\n            forked.in_fork()\n"
                           "        finally:\n            os._exit(0)\n"
                           "    assert os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) == 0\n",
        "test_fork.py": "import multiprocessing, forked\ndef test_child():\n"
                        "    child = multiprocessing.get_context('fork').Process(target=forked.in_mp_fork)\n"
                        "    " + started,
        "test_forkserver.py": "import multiprocessing, os\ndef test_child():\n"
                              "    context = multiprocessing.get_context('forkserver')\n"
                              "    child = context.Process(target=os.getpid)\n    " + started,
    }
    written(tmp_path, files)
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes/page.md").write_text("page\n")
    interpreter(tmp_path, "3.12")
    tests = [f"tests/{name}" for name in files if name.startswith("test_")]
    fresh_tasks(ci, tests, tmp_path)

    def reused() -> list[bool]:
        return reusable_now(ci, tests)

    def edited(name: str, source: str) -> list[bool]:
        """Whether each file is reused with `name` as `source`, which `name` is restored from after."""
        path = tmp_path / name
        original = path.read_text()
        path.write_text(source)
        try:
            return reused()
        finally:
            path.write_text(original)

    assert reused() == [True, True, True, True, False]
    assert edited("notes/page.md", "page again\n") == [False, True, True, True, False]
    assert edited("tests/inner.py", "def inner(): return 3 - 1\n") == [True, False, True, True, False]
    assert edited("tests/plain.py", "def helper(): return 2 - 1\n") == [False, True, True, True, False]
    # What only a forked child ran reruns its file alone, each file in a process of its own (ci.py's `alone`).
    assert edited("tests/forked.py", "def in_fork(): return 3\ndef in_mp_fork(): return 5 - 1\n") == [
        True, True, True, False, False]
    assert edited("tests/forked.py", "def in_fork(): return 4 - 1\ndef in_mp_fork(): return 4\n") == [
        True, True, False, True, False]


def test_a_native_media_binary_keeps_its_files_key_and_what_it_opened_is_that_keys_input(
        ci, tmp_path, monkeypatch, cache_worker):
    """The media binary imageio-ffmpeg ships, named for its platform and version, and the other native
    programs of `followed.TOOLS` the environment or the system owns run no checkout code and hold no GPU: a
    file that starts one keeps its key, whose inputs include the data files the program opened under tests/
    (the watch's: it raises no Python audit event). A script under the same name, which may run any code,
    leaves its file without a key, the checkout's or a test's alike (`followed.tool`)."""
    url, _, _ = cache_worker
    cached(monkeypatch, url)
    written(tmp_path, {"clip.bin": "frames\n"})
    (tmp_path / "tests/bin").mkdir()
    script = tmp_path / "tests/bin/ffmpeg-linux-x86_64-v7.0.2"
    script.write_text("#!/bin/sh\ncat tests/clip.bin\n")
    script.chmod(0o755)
    written(tmp_path, {
        "test_media.py": "import imageio_ffmpeg, subprocess\ndef test_run():\n"
                         "    program = imageio_ffmpeg.get_ffmpeg_exe()\n"
                         "    subprocess.run([program, '-v', 'quiet', '-i', 'tests/clip.bin'],\n"
                         "                   capture_output=True)\n",
        "test_shell_script.py": "import subprocess\ndef test_run():\n"
                                "    subprocess.run(['tests/bin/ffmpeg-linux-x86_64-v7.0.2'], check=True,\n"
                                "                   capture_output=True)\n"})
    interpreter(tmp_path, "3.12")
    tests = ["tests/test_media.py", "tests/test_shell_script.py"]
    fresh_tasks(ci, tests, tmp_path)
    assert reusable_now(ci, tests) == [True, False]
    (tmp_path / "tests/clip.bin").write_text("more frames\n")
    assert reusable_now(ci, tests) == [False, False]


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


def test_each_start_needs_its_own_record_though_two_start_alike(ci, tmp_path, monkeypatch, cache_worker):
    """Two children a file starts alike are two starts, each owing a record: two that record themselves
    keep the file's key, and one that records itself beside one that starts with no site (-S), which no
    recorder sees, leave the file without one, so an edit to what the second ran cannot reuse its row."""
    plain = "    subprocess.run([sys.executable, '-c', 'pass'], check=True)\n"
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "test_two.py": "import subprocess, sys\ndef test_two():\n" + plain * 2,
        "test_b.py": PASSING["test_b.py"],
        "test_twice.py": "import subprocess, sys\ndef test_twice():\n" + plain
                         + "    subprocess.run([sys.executable, '-S', 'src/other.py'], check=True)\n"},
        {"other.py": "print('other')\n"})
    assert reused() == [True, True, False]


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
    (("    return x * by", "    return by * x + 0"), [True, False]),
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
    or one that only moves lines, reruns no file, and an edit to what a run executed reruns that file alone,
    each file in a process of its own (ci.py's `alone`)."""
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
    ("import inspect\n    assert str(inspect.signature(m.unused)) == '(x)'",
     ("def unused(x):", "def unused(y):")),
    ("import inspect\n    assert inspect.isgeneratorfunction(m.counted)", ("    yield x", "    return x")),
    ("assert 'one' in m.unused.__code__.co_consts", ("    return x, 'one'", "    return x, 'two'")),
    ("namespace = {}\n    exec(m.__spec__.loader.get_source('m'), namespace)\n"
     "    assert namespace['unused'](2) == (2, 'one')", ("    return x, 'one'", "    return x, 'two'")),
    ("pass", ("    return x, 'one'", "    return x, 'one'\n    break")),
], ids=["a bare docstring read", "a signature read", "a generator's kind", "a function's code",
        "a module's text it executed itself",
        "a body that no longer compiles"])
def test_what_a_run_sees_of_code_it_never_entered_is_its_input(ci, tmp_path, monkeypatch, cache_worker,
                                                               seen, edit):
    """What a run sees of a function it never entered is its input as the code it ran is: its docstring,
    which a bare `__doc__` read takes with no code and no audit seeing it, and its signature, both kept in its
    module's top level (ci.py's `Hollow`), the function value's kind (a generator's), its code object
    (`__code__`), the module's text it compiled itself, and whether the module still compiles, which a body
    left out would hide."""
    # The body's constant a string, which every Python keeps in co_consts (3.14 loads a small int inline).
    module = ('import sys\n\n\ndef unused(x):\n    """Less one."""\n    return x, \'one\'\n\n\n'
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
    fresh_tasks(ci, ["tests/test_bytecode.py"], tmp_path)
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


def test_each_files_record_is_its_own_where_files_share_a_task(ci, tmp_path, monkeypatch, cache_worker):
    """Each input moves exactly the files that may have seen it (row 2642): a fixture's arrival reruns its
    prober, the task's last file, alone; a helper shadowing a name only a file's child imported, where the
    child's import searched for it, reruns that file alone; a module one file imported lazily reruns it
    alone, as each file runs in a process of its own (ci.py's `alone`) that nothing it left can serve; and
    two files whose records share inputs never take each other's row."""
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
    # The child's import searched the checkout's root (its `-c`'s ''), not tests/, which a helper there
    # cannot shadow.
    (tmp_path / "tests/tabnanny.py").write_text("")
    assert reused() == [True, True, True, True]
    (tmp_path / "tests/tabnanny.py").unlink()
    (tmp_path / "tabnanny.py").write_text("")
    assert reused() == [False, True, True, True]
    (tmp_path / "tabnanny.py").unlink()
    (tmp_path / "src/lazy.py").write_text("VALUE = 2\n")
    assert reused() == [True, False, True, True]
    (tmp_path / "src/lazy.py").write_text("VALUE = 1\n")
    common = ci.shared("3.12")
    names = ("tests/test_lazy.py", "tests/test_later.py")
    found = [ci.reused("3.12", name, "", common)["name"] for name in names]
    assert found == [f"3.12:{name}" for name in names]


def test_a_file_that_digests_dews_source_keys_on_what_it_ran_not_on_every_source_file(
        ci, tmp_path, monkeypatch, cache_worker):
    """`dew.cache.package_digest` reads every source file of the package only to name the records Dew's
    caches keep (the rung cache's, a verified mapping's, a saved source's export), and each file's process
    has a cache directory of its own (ci.py's `alone`), so no record another file wrote serves it: its reads
    are no input (followed.py's `DIGEST`). A file that calls it runs again for an edit of the code it ran,
    the digest's own included, and not for one of a module it never ran, nor for a release's version (train
    396 reused 1 of 2,279 rows for its version bump; ledger 3220: the digest keyed 37-45% of rows on every
    source file)."""
    import shutil

    package = tmp_path / "src" / "dew"
    shutil.copytree(REPO_ROOT / "src" / "dew", package, ignore=shutil.ignore_patterns("__pycache__"))
    (package / "_version.py").write_text('__version__ = "0.2.0"\n')
    (package / "spare.py").write_text("def spare():\n    return 1\n")
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "test_digested.py": ON_SRC + "from dew.cache import package_digest\n"
                                     "def test_digested():\n    assert package_digest()\n"}, {})
    assert reused() == [True]
    (package / "_version.py").write_text('__version__ = "0.3.0"\n')
    assert reused() == [True]
    (package / "spare.py").write_text("def spare():\n    return 2\n")
    assert reused() == [True]
    cache = package / "cache.py"
    source = cache.read_text()
    assert source.count('+ b"\\0" +') == 1
    cache.write_text(source.replace('+ b"\\0" +', '+ b"\\1" +'))
    assert reused() == [False]


def test_a_marker_registered_in_pyproject_reruns_the_files_that_use_it_and_its_other_parts_none(
        ci, tmp_path, monkeypatch, cache_worker):
    """pytest's load of pyproject.toml as it starts is no input of any row (followed.py's `configuring`):
    its pytest settings are every row's (ci.py's `pytest_settings`), a marker's registration the input of the
    files whose tests carry that marker however they came by it, a helper's alias among them (`marks`,
    executed.py's), and its other parts the input of a test that reads them itself alone (ledger 3220: train
    416 registered two markers and reran all 3,445 rows)."""
    project = ("[project]\nname = 'x'\n\n[tool.pytest.ini_options]\naddopts = \"--strict-markers\"\n"
               "markers = [\"tidy: {}\"]\n")
    (tmp_path / "pyproject.toml").write_text(project.format("a tidy test"))
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "test_marked.py": "import pytest\n@pytest.mark.tidy\ndef test_marked():\n    pass\n",
        # A marker a helper names, which the test file never spells.
        "helpers_marks.py": "import pytest\ntidy = pytest.mark.tidy\n",
        "test_aliased.py": "from helpers_marks import tidy\n@tidy\ndef test_aliased():\n    pass\n",
        "test_plain.py": "def test_plain():\n    pass\n",
        # A test's own read of the project's table, which pytest's settings read is not.
        "test_project.py": "import tomllib\ndef test_project():\n"
                           "    with open('pyproject.toml', 'rb') as project:\n"
                           "        assert tomllib.load(project)['project']['name'] == 'x'\n"}, {})
    assert reused() == [True, True, True, True]
    (tmp_path / "pyproject.toml").write_text(project.format("a tidier test"))
    assert reused() == [False, False, True, False]
    (tmp_path / "pyproject.toml").write_text(project.format("a tidy test").replace("'x'", "'y'"))
    assert reused() == [True, True, True, False]
    (tmp_path / "pyproject.toml").write_text(project.format("a tidy test").replace("--strict-markers", "-ra"))
    assert reused() == [False, False, False, False]


def one_task(ci, tmp_path, monkeypatch, cache_worker, files: dict[str, str], fresh: bool):
    """`files` under tests/ written and run as one 3.12 task, in one container, fresh (`fresh`, no task
    ran in it) or not, with the test cache on: whether each test file's row is reused now, in order."""
    url, _, _ = cache_worker
    cached(monkeypatch, url)
    written(tmp_path, files)
    interpreter(tmp_path, "3.12")
    tests = [f"tests/{name}" for name in files if name.startswith("test_")]
    ci.task("3.12", tests, "", tmp_path / "verdict.json", 600.0, fresh=fresh)
    rows = json.loads((tmp_path / "verdict.json").read_text())["rows"]
    assert all(row["exitCode"] == 0 for row in rows), rows
    return lambda: reusable_now(ci, tests)


# A reader of what the writer left in shared scratch by a native tool, which no audit of Python's could see.
LEFT = {
    "build.py": "def build(): return 1\n",
    "test_writer.py": "from build import build\ndef test_writer():\n"
                      "    open('/tmp/dew-left-result', 'w').write('value=' + str(build()) + '\\n')\n",
    "test_reader.py": "import subprocess\ndef test_reader():\n"
                      "    assert 'text' in subprocess.run(['file', '-b', '/tmp/dew-left-result'],"
                      " capture_output=True, text=True, check=True).stdout\n"}


def test_only_the_first_file_to_execute_in_a_fresh_container_keeps_a_key(
        ci, tmp_path, monkeypatch, cache_worker):
    """A file run after another in its container may take anything the first left (a file, a process, its
    memory), however it takes it, so it keeps no key: an edit to what the first left would leave its green row
    standing. The first, run from the state the environment's snapshot left, keeps its key (Main,
    2026-11-01: sound by construction, no write audited)."""
    reused = one_task(ci, tmp_path, monkeypatch, cache_worker, LEFT, fresh=True)
    assert reused() == [True, False]


def test_no_file_of_a_container_another_task_ran_in_keeps_a_key(ci, tmp_path, monkeypatch, cache_worker):
    """A container armada does not say is fresh (no ARMADA_FRESH) may hold what an earlier task left: no file
    run in it keeps a key, the first among them."""
    reused = one_task(ci, tmp_path, monkeypatch, cache_worker, LEFT, fresh=False)
    assert reused() == [False, False]


def test_a_reused_row_runs_nothing_so_the_first_file_to_execute_after_it_keeps_its_key(
        ci, tmp_path, monkeypatch, cache_worker):
    """A reused row executes nothing in the container, so the first file that runs after it still runs from
    the snapshot's state and keeps its key."""
    keyed = one_task(ci, tmp_path, monkeypatch, cache_worker, {"test_a.py": PASSING["test_a.py"]}, fresh=True)
    assert keyed() == [True]
    written(tmp_path, {"test_b.py": PASSING["test_b.py"]})
    ci.task("3.12", ["tests/test_a.py", "tests/test_b.py"], "", tmp_path / "verdict.json", 600.0, fresh=True)
    assert reusable_now(ci, ["tests/test_a.py", "tests/test_b.py"]) == [True, True]


def test_a_marker_a_deselected_test_carries_is_its_files_input(ci, tmp_path, monkeypatch, cache_worker):
    """A marker a test carries by a helper's alias is its file's input though the run's selection sets the
    test aside (review of the reuse-keys head): with --strict-markers the helper's import fails the file's
    collection once the marker's registration goes, so the file's row reruns."""
    project = ("[project]\nname = 'x'\n\n[tool.pytest.ini_options]\n"
               "addopts = \"--strict-markers -m 'not tidy'\"\nmarkers = [\"tidy: {}\"]\n")
    (tmp_path / "pyproject.toml").write_text(project.format("a tidy test"))
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "helpers_marks.py": "import pytest\ntidy = pytest.mark.tidy\n",
        "test_mixed.py": "from helpers_marks import tidy\ndef test_plain():\n    pass\n"
                         "@tidy\ndef test_tidy():\n    pass\n"}, {})
    assert reused() == [True]
    (tmp_path / "pyproject.toml").write_text(project.format("a tidier test"))
    assert reused() == [False]


def test_a_run_that_rewrites_its_own_input_keeps_no_key(ci, tmp_path, monkeypatch, cache_worker):
    """A run that rewrites an input it read keeps no key, as its key would hash what it left, not what it
    read (review of the reuse-keys head: a test reads a fixture, overwrites it and asserts on what it read; a
    later checkout holding the overwrite would reuse a run that never saw it). A run that left its inputs as
    it found them, the same fixture among them, keeps its key."""
    (tmp_path / "tests/fixtures").mkdir(parents=True)
    (tmp_path / "tests/fixtures/value.txt").write_text("good")
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "test_rewrites.py": "from pathlib import Path\ndef test_rewrites():\n"
                            "    found = Path('tests/fixtures/value.txt')\n    read = found.read_text()\n"
                            "    found.write_text('bad')\n    assert read == 'good'\n",
        "test_reads.py": "from pathlib import Path\ndef test_reads():\n"
                         "    assert Path('tests/fixtures/value.txt').read_text()\n"}, {})
    assert reused() == [False, True]


def test_a_run_that_rewrites_what_a_link_it_read_names_keeps_no_key(ci, tmp_path, monkeypatch, cache_worker):
    """A run that rewrites, through a link it read, the file the link names keeps no key, as its key reads the
    link's target (review: the link's own change time stands, so the rewrite went unseen and a later checkout
    holding the rewrite reused the run). A run that only reads through the link keeps its key. The pages sit
    under docs/, which no watch of tests/ or src/ follows to the link's target."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/value.md").write_text("good")
    (tmp_path / "docs/current.md").symlink_to("value.md")
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "test_rewrites.py": "from pathlib import Path\ndef test_rewrites():\n"
                            "    found = Path('docs/current.md')\n    read = found.read_text()\n"
                            "    found.write_text('bad')\n    assert read == 'good'\n",
        "test_reads.py": "from pathlib import Path\ndef test_reads():\n"
                         "    assert Path('docs/current.md').read_text()\n"}, {})
    assert reused() == [False, True]


def test_a_marker_a_child_pytest_looks_up_is_its_starters_input(ci, tmp_path, monkeypatch, cache_worker):
    """A pytest a test starts, without the recorder's plugin (its plugins not loaded), reads its settings
    from pyproject.toml as an input of the starting file (review: the child's read was set aside as the
    parent's plugin records markers, which the child's never did), so a marker registration its strict
    collection needs reruns the file when it changes."""
    project = ("[project]\nname = 'x'\n\n[tool.pytest.ini_options]\n"
               "addopts = \"--strict-markers\"\nmarkers = [\"tidy: {}\"]\n")
    (tmp_path / "pyproject.toml").write_text(project.format("a tidy test"))
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "helpers_marks.py": "import pytest\nunused = pytest.mark.tidy\n",
        "child_case.py": "import sys\nsys.path.insert(0, 'tests')\nimport helpers_marks\n"
                         "def test_case():\n    assert helpers_marks.unused\n",
        "test_starts.py": "import os, subprocess, sys\ndef test_starts():\n"
                          "    child = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',\n"
                          "             'tests/child_case.py']\n"
                          "    alone = {**os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}\n"
                          "    subprocess.run(child, check=True, env=alone)\n"}, {})
    assert reused() == [True]
    (tmp_path / "pyproject.toml").write_text(project.format("a tidier test"))
    assert reused() == [False]


def test_the_settings_a_pytest_a_test_runs_in_its_process_reads_are_the_tests_input(
        ci, tmp_path, monkeypatch, cache_worker):
    """A pytest a test runs in its own process (`pytest.main`), a session the recorder's plugin does not
    record, reads its settings as an input of the test's file (review: every configuration read in the process
    was set aside as the recorded session's), so a registration the inner session reads reruns the file."""
    project = "[project]\nname = 'x'\n\n[tool.pytest.ini_options]\nmarkers = [\"tidy: {}\"]\n"
    (tmp_path / "pyproject.toml").write_text(project.format("a tidy test"))
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "inner_case.py": "def test_inner(pytestconfig):\n"
                         "    assert any('tidy' in line for line in pytestconfig.getini('markers'))\n",
        "test_inner.py": "import pytest\ndef test_inner():\n"
                         "    inner = ['-q', '-p', 'no:cacheprovider', 'tests/inner_case.py']\n"
                         "    assert pytest.main(inner) == 0\n"}, {})
    assert reused() == [True]
    (tmp_path / "pyproject.toml").write_text(project.format("a tidier test"))
    assert reused() == [False]


def test_the_registrations_a_conftests_configure_reads_are_every_files_input(
        ci, tmp_path, monkeypatch, cache_worker):
    """A conftest's own pytest_configure reading the marker registrations (`config.getini("markers")`) makes
    them an input of the files it configures (review: a conftest's hook ran before the recorder's, and its
    read went unseen), so a registration it needs reruns them when it changes."""
    project = "[project]\nname = 'x'\n\n[tool.pytest.ini_options]\nmarkers = [\"tidy: {}\"]\n"
    (tmp_path / "pyproject.toml").write_text(project.format("a tidy test"))
    # The conftest's hook asks to run first too, as the recorder's would: no order of hooks hides its read.
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "conftest.py": "import pytest\n@pytest.hookimpl(tryfirst=True)\ndef pytest_configure(config):\n"
                       "    assert any('tidy' in line for line in config.getini('markers'))\n",
        "test_plain.py": "def test_plain():\n    pass\n"}, {})
    assert reused() == [True]
    (tmp_path / "pyproject.toml").write_text(project.format("a tidier test"))
    assert reused() == [False]


def test_a_marker_registered_in_a_string_setting_is_its_users_input(ci, tmp_path, monkeypatch, cache_worker):
    """A marker registered in pytest's settings as one string of lines, as pytest also takes them, is the
    input of each file that uses it, read as pytest reads the string (review: the string was read a character
    at a time, so a registration's removal went unseen and a strict collection's failure was reused green)."""
    project = "[project]\nname = 'x'\n\n[tool.pytest.ini_options]\nmarkers = \"\"\"\n{}\n\"\"\"\n"
    (tmp_path / "pyproject.toml").write_text(project.format("tidy: a tidy test\nslow: a slow test"))
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "test_tidy.py": "import pytest\n@pytest.mark.tidy\ndef test_tidy():\n    pass\n"}, {})
    assert reused() == [True]
    (tmp_path / "pyproject.toml").write_text(project.format("slow: a slow test"))
    assert reused() == [False]


def test_a_marker_a_helper_looks_up_and_no_test_carries_is_its_importers_input(
        ci, tmp_path, monkeypatch, cache_worker):
    """A marker a helper looks up but never applies is its importer's input (review of the reuse-keys head):
    with --strict-markers the lookup fails once the registration goes, failing the importer's collection,
    though no test carries the marker; so the importer's row reruns when the registration changes."""
    project = ("[project]\nname = 'x'\n\n[tool.pytest.ini_options]\n"
               "addopts = \"--strict-markers\"\nmarkers = [\"tidy: {}\"]\n")
    (tmp_path / "pyproject.toml").write_text(project.format("a tidy test"))
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "helpers_marks.py": "import pytest\nunused = pytest.mark.tidy\n",
        "test_plain.py": "import helpers_marks\ndef test_plain():\n    assert helpers_marks.unused\n"}, {})
    assert reused() == [True]
    (tmp_path / "pyproject.toml").write_text(project.format("a tidier test"))
    assert reused() == [False]


def test_a_store_linked_beneath_the_files_caches_to_one_elsewhere_is_no_private_store(
        ci, tmp_path, monkeypatch):
    """A store Dew's digest reads consult (`CACHES`) that resolves outside the file's own caches is no
    private one, though its path lies beneath them by a link (review of the reuse-keys head), so its digest
    reads are keyed (followed.py's `private`). JAX's own setting aside: none is imported here."""
    followed = ci.followed
    monkeypatch.setitem(sys.modules, "jax", None)
    own, elsewhere = tmp_path / "caches", tmp_path / "foreign"
    own.mkdir()
    elsewhere.mkdir()
    (own / "rungs").symlink_to(elsewhere)
    monkeypatch.setenv(followed.PRIVATE, str(own))
    monkeypatch.setenv("XDG_CACHE_HOME", str(own))
    for name in followed.CACHES:
        monkeypatch.setenv(name, str(own / "plain"))
    assert followed.private()
    monkeypatch.setenv(followed.CACHES[0], str(own / "rungs"))
    assert not followed.private()


@pytest.mark.parametrize("plugin", ["executed", "release", "split"])
def test_each_hook_a_plugin_of_the_harness_defines_is_one_pytest_registers(plugin):
    """pytest registers a plugin's function as a hook only where its name starts with `pytest_` (its plugin
    manager's `parse_hookimpl_opts`), whatever `specname` the hookimpl names: one named otherwise never runs
    and no other test says so (executed.py's marker hook went unregistered, so a marker a helper gave a test
    was never keyed)."""
    source = ast.parse((REPO_ROOT / "tools/armada" / f"{plugin}.py").read_text())
    unregistered = [function.name for function in source.body if isinstance(function, ast.FunctionDef)
                    and any("hookimpl" in ast.unparse(each) for each in function.decorator_list)
                    and not function.name.startswith("pytest_")]
    assert unregistered == []


def test_a_copy_of_the_interpreter_under_a_tools_name_is_no_tool(ci, tmp_path, monkeypatch, cache_worker):
    """A program named for a tool that is a copy of the interpreter, or made since the task began, runs code
    no record follows (followed.py's `tool`): the file that starts one has no key, as one starting the
    system's own native tool, run before it, keeps it."""
    home = tmp_path.parent / f"{tmp_path.name}-home"
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {
        "test_system.py": "import subprocess\ndef test_system():\n"
                          "    subprocess.run(['git', '--version'], check=True, capture_output=True)\n",
        "test_copied.py": "import os, shutil, subprocess, sys\ndef test_copied():\n"
                          f"    os.makedirs({str(home)!r}, exist_ok=True)\n"
                          "    program = shutil.copy(os.path.realpath(sys.executable),\n"
                          f"                          {str(home / 'git')!r})\n"
                          "    subprocess.run([program, '-S', '-c', 'pass'], check=False)\n"}, {})
    assert reused() == [True, False]


def test_each_file_of_a_task_finds_its_cache_and_scratch_directories_empty(ci, tmp_path):
    """No record one file's process leaves in its caches (XDG_CACHE_HOME: Dew's, the compilation cache and
    a step's rungs beside it) or its scratch (TMPDIR, pytest's tmp_path among it) serves another file of its
    task (`alone`), which no key would hold, as writes there are its own: what lets a file's key
    leave out what its taskmates ran, and the source digest those records are named by (followed.py's
    `DIGEST`, `private`)."""
    (tmp_path / "tests").mkdir(exist_ok=True)
    for name in ("first", "second"):
        (tmp_path / f"tests/test_{name}.py").write_text(
            "import os, tempfile\nfrom pathlib import Path\n"
            f"def test_{name}():\n"
            "    for record in (Path(os.environ['XDG_CACHE_HOME']) / 'dew' / 'left.txt',\n"
            "                   Path(tempfile.gettempdir()) / 'left.txt'):\n"
            "        assert not record.exists()\n"
            "        record.parent.mkdir(parents=True, exist_ok=True)\n"
            "        record.write_text('x')\n")
    interpreter(tmp_path, "3.12")
    out = tmp_path / "verdict.json"
    ci.task("3.12", ["tests/test_first.py", "tests/test_second.py"], "", out, 600.0)
    assert [row["exitCode"] for row in json.loads(out.read_text())["rows"]] == [0, 0]


def test_a_files_children_hold_what_they_held_at_once_and_each_start_must_leave_its_record(tmp_path):
    """The children's logs join the file that started them (`followed.gathered`): their files, and the
    GPU they held, the most a child that took the parent's fraction used, those children as shares and
    the bytes of those that named their own, counted only while they lived at once. A file is followed
    only where every start it and its children saw left a log naming it and none started a process it
    did not follow; a child killed before its end left its log as far as it got, which is what it ran,
    but its pool is not known (`measured`), as the child reads it as its last atexit runs."""
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
                                   "followed": True, "measured": True, "peak": 5 * gib, "shares": 2,
                                   "pools": 5 * gib}
    assert [found[name]["followed"] for name in ("tests/b.py", "tests/c.py", "tests/d.py")] == [
        False, True, False]
    assert [found[name]["measured"] for name in ("tests/c.py", "tests/d.py")] == [False, True]
