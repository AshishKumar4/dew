"""What a test file's key covers beyond what its run used (tools/armada/ci.py's `cache_key`). Train 410's two
small product heads moved the key of 1,288 of the 1,563 rows the test cache holds a key for, 749 of them rows
that ran none of the code the heads changed (tests/test_check2.py, a file of arithmetic, among them). Each
case below is such a head's edit to a package a file imports through a hub module and never calls into: a
module added beside the package's (`src/dew/interop/code_identity.py`: `packages` keys the listing of every
directory above a module a run executed, and 82% of the rows executed a module of src/dew/interop), a function
added to a module the hub imports (`LoadName` and `load_names` in `src/dew/interop/sources.py`: a module's top
level is a unit of every file that imported it, 81% of the rows), a docstring or a signature edited (train
395's docstring edits in `newton_schulz.py` and `optim.py` moved 84% of the rows). Here the file's own test
calls one function of one module.

The cases that must rerun stay: an edit to the function the file called, and a module that arrives under a
name the file's run tried to import and did not find."""

import pytest
from armada_support import ON_SRC, lifetime_task, loaded

PACKAGE = {
    "pkg/__init__.py": "",
    "pkg/hub.py": "from pkg import sibling\n\n\ndef hub():\n    return sibling.value()\n",
    "pkg/sibling.py": 'def value():\n    """The value."""\n    return 1\n\n\ndef spare(x, by=2):\n'
                      '    """Spare."""\n    return x * by\n',
    "pkg/pure.py": "def double(x):\n    return x * 2\n",
}
"""A package whose hub imports a sibling, as `dew.config` and `dew.nn.blocks` import theirs."""

PURE = (ON_SRC + "from pkg import hub, pure\n\n\ndef test_pure():\n    assert pure.double(2) == 4\n")
"""A file that imports the hub, as 80% of the suite's files do, and calls one function of one module."""

TRIES = (ON_SRC + "def test_tries():\n    try:\n        import pkg.tried\n    except ImportError:\n"
         "        found = False\n    else:\n        found = True\n    assert found in (True, False)\n")
"""A file whose run tries to import a module that is not there yet."""


@pytest.fixture
def ci(tmp_path, monkeypatch):
    return loaded(tmp_path, monkeypatch)


def task(ci, tmp_path, monkeypatch, cache_worker, tests):
    for name in PACKAGE:
        (tmp_path / "src" / name).parent.mkdir(parents=True, exist_ok=True)
    return lifetime_task(ci, tmp_path, monkeypatch, cache_worker, tests, PACKAGE)


def appended(src, name: str, text: str) -> None:
    (src / name).write_text((src / name).read_text() + text)


def replaced(src, name: str, *edits: tuple[str, str]) -> None:
    text = (src / name).read_text()
    for old, new in edits:
        assert old in text, old
        text = text.replace(old, new)
    (src / name).write_text(text)


@pytest.mark.parametrize(("name", "edit", "reused_after"), [
    ("a module added beside the package's",
     lambda src: (src / "pkg/added.py").write_text("def added():\n    return 1\n"), True),
    ("a docstring edited in a module the hub imports",
     lambda src: replaced(src, "pkg/sibling.py", ('"""Spare."""', '"""Spare, edited."""')), False),
    ("a function's signature and body edited, never called",
     lambda src: replaced(src, "pkg/sibling.py", ("def spare(x, by=2):", "def spare(x, by=3):"),
                          ("return x * by", "return by * x")), False),
], ids=["module added", "docstring edited", "signature edited"])
def test_an_edit_to_code_a_file_imported_and_never_called_leaves_its_row_where_it_cannot_observe_it(
        ci, tmp_path, monkeypatch, cache_worker, name, edit, reused_after):
    """A head's edit in a package the file imports through its hub, which no test of the file calls into: a
    module added where no import of the file searched leaves its row (an import keys the candidates it was
    searched for in, `contents`); a docstring or a signature edited reruns it, as a module's top level keeps
    every definition's docstring, defaults and annotations, which a function value shows unrun (`__doc__`,
    `inspect.signature`) and which may fail as the module imports (a default of 1/0)."""
    reused = task(ci, tmp_path, monkeypatch, cache_worker, {"test_pure.py": PURE})
    assert reused() == [True]
    edit(tmp_path / "src")
    assert reused() == [reused_after], name


@pytest.mark.xfail(strict=True, reason="a new top-level function is a name in every importer's module: "
                                       "leaving it out of the key needs what a run used of a module's names, "
                                       "which no record "
                                       "holds (a name's shadowing, a removal an importer's import never saw)")
def test_a_function_added_to_a_module_a_file_imported_leaves_its_row(ci, tmp_path, monkeypatch, cache_worker):
    """`LoadName` and `load_names` added to src/dew/interop/sources.py reran the 1,260 files that imported
    it."""
    reused = task(ci, tmp_path, monkeypatch, cache_worker, {"test_pure.py": PURE})
    assert reused() == [True]
    appended(tmp_path / "src", "pkg/sibling.py", "\n\ndef added(x):\n    return x\n")
    assert reused() == [True]


def test_an_edit_to_the_function_a_file_called_reruns_it(ci, tmp_path, monkeypatch, cache_worker):
    """The control: the one function the file calls, edited, reruns it."""
    reused = task(ci, tmp_path, monkeypatch, cache_worker, {"test_pure.py": PURE})
    assert reused() == [True]
    (tmp_path / "src/pkg/pure.py").write_text("def double(x):\n    return x + x\n")
    assert reused() == [False]


def test_a_default_changed_on_the_function_a_file_called_reruns_it(ci, tmp_path, monkeypatch, cache_worker):
    """The control: a default of the one function the file calls, changed, reruns it, its call reading the
    default through the function's own unit."""
    package = {**PACKAGE, "pkg/pure.py": "def double(x, by=2):\n    return x * by\n"}
    for name in package:
        (tmp_path / "src" / name).parent.mkdir(parents=True, exist_ok=True)
    reused = lifetime_task(ci, tmp_path, monkeypatch, cache_worker, {"test_pure.py": PURE}, package)
    assert reused() == [True]
    (tmp_path / "src/pkg/pure.py").write_text("def double(x, by=3):\n    return x * by\n")
    assert reused() == [False]


def test_an_import_added_to_a_module_a_file_imported_reruns_it(ci, tmp_path, monkeypatch, cache_worker):
    """The control: an import added at the top level of a module the file imports runs at the file's import,
    as any statement there does, so the file reruns."""
    reused = task(ci, tmp_path, monkeypatch, cache_worker, {"test_pure.py": PURE})
    assert reused() == [True]
    replaced(tmp_path / "src", "pkg/hub.py",
             ("from pkg import sibling\n", "import json\nfrom pkg import sibling\n"))
    assert reused() == [False]


def test_a_module_arriving_under_a_name_a_run_tried_to_import_reruns_it(ci, tmp_path, monkeypatch,
                                                                         cache_worker):
    """The control: `import pkg.tried` found nothing; the module that arrives under that name is what the same
    import now finds, so the file whose run tried it reruns."""
    reused = task(ci, tmp_path, monkeypatch, cache_worker, {"test_tries.py": TRIES})
    assert reused() == [True]
    (tmp_path / "src/pkg/tried.py").write_text("VALUE = 1\n")
    assert reused() == [False]
