"""tools/check_tool_types.py: each tool's environment read from its COHORTS or PEP 723 header, a tool whose
cohorts conflict refused, and a verdict's key covering every module of the checkout the tool imports, through
their own imports, so a change to a module a tool imports, the tool untouched, checks it again (row 7933)."""

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "check_tool_types.py"
PINS = {"torch": {"torch": "2.14.0"}, "transformers": {"transformers": "5.22.2"},
        "ouro": {"transformers": "4.56.2"},
        "laya": {"laya": "git+https://github.com/o/laya@" + "a" * 40, "transformers": "5.22.2"}}


@pytest.fixture
def gate(tmp_path, monkeypatch):
    """The gate's module, its checkout `tmp_path`: src/dew/{shared,deep}.py (shared imports deep),
    tools/{helper,tool}.py and a package tools/runs/ with a relative import, and Dew's lock."""
    spec = importlib.util.spec_from_file_location("check_tool_types_under_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    files = {"src/dew/__init__.py": "", "src/dew/shared.py": "from dew import deep\n",
             "src/dew/deep.py": "X = 1\n", "tools/helper.py": "import json\n",
             "tools/tool.py": "import helper\nfrom dew.shared import deep\n",
             "tools/runs/__init__.py": "", "tools/runs/a.py": "from . import b\n",
             "tools/runs/b.py": "Y = 2\n",
             "tools/other.py": "Z = 3\n", "locks/cpu.txt": "jax==1\n"}
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    return module


@pytest.mark.parametrize(("declared", "cohorts"), [
    ('COHORTS = "torch"', (("torch",),)),
    ('COHORTS = ["transformers", "torch"]', (("torch", "transformers"),)),
    ('COHORTS: dict = {"a": "torch", "b": ["transformers", "torch"], "c": "torch"}',
     (("torch",), ("torch", "transformers"))),
    ('COHORT = "torch"', (("torch",),)), ("X = 1", None)])
def test_a_tools_cohorts_are_read_from_its_module_level_cohorts(gate, tmp_path, declared, cohorts) -> None:
    (tmp_path / "tools" / "t.py").write_text(f"import os\n{declared}\n")
    assert gate.cohorts_of(tmp_path / "tools" / "t.py") == cohorts


def test_a_tools_environments_are_its_cohorts_pins_its_headers_or_dews(gate, tmp_path) -> None:
    """A tool's environment holds the pins of the cohorts it runs in together; one a variant's cohorts where
    COHORTS maps variants; a PEP 723 header's dependencies where it has one (`uv run` reads only that); Dew's
    own where it names neither; and none where cohorts it runs in together pin a distribution twice."""
    tools = tmp_path / "tools"
    (tools / "paired.py").write_text('COHORTS = ["torch", "transformers"]\n')
    (tools / "script.py").write_text('# /// script\n# requires-python = ">=3.12"\n'
                                     '# dependencies = ["z3-solver==4.15.4"]\n# ///\nimport z3\n')
    (tools / "variants.py").write_text('COHORTS = {"new": "transformers", "old": "ouro"}\n')
    (tools / "mixed.py").write_text('COHORTS = ["transformers", "ouro"]\n')
    (tools / "unknown.py").write_text('COHORTS = {"a": "torch", "b": "nothing"}\n')
    assert gate.environments_of(tools / "paired.py", PINS) == [
        ("torch+transformers", ["torch==2.14.0", "transformers==5.22.2"])]
    assert gate.environments_of(tools / "script.py", PINS)[0][1] == ["z3-solver==4.15.4"]
    assert gate.environments_of(tools / "other.py", PINS) == [(gate.DEW, [])]
    assert gate.environments_of(tools / "variants.py", PINS) == [
        ("ouro", ["transformers==4.56.2"]), ("transformers", ["transformers==5.22.2"])]
    (tools / "source.py").write_text('COHORT = "laya"\n')
    [(name, requirements)] = gate.environments_of(tools / "source.py", PINS)
    assert requirements == [f"laya @ git+https://github.com/o/laya@{'a' * 40}", "transformers==5.22.2"]
    assert gate.SOURCE.match(requirements[0])
    with pytest.raises(ValueError, match="pin transformers at 4.56.2 and 5.22.2.*split it"):
        gate.environments_of(tools / "mixed.py", PINS)
    with pytest.raises(ValueError, match="nothing, which RELEASES does not"):
        gate.environments_of(tools / "unknown.py", PINS)


def test_a_tools_key_covers_what_it_imports_through_their_imports_and_nothing_else(gate, tmp_path) -> None:
    """A tool's key moves with any module it imports, however deep (tools/tool.py imports dew.shared, which
    imports dew.deep), with its environment and with the shared part; not with a file it does not import."""
    tool = tmp_path / "tools" / "tool.py"
    names = {path.relative_to(tmp_path).as_posix() for path in gate.closure(tool)}
    assert names == {"tools/tool.py", "tools/helper.py", "src/dew/__init__.py", "src/dew/shared.py",
                     "src/dew/deep.py"}
    relative = {path.relative_to(tmp_path).as_posix() for path in gate.closure(tmp_path / "tools/runs/a.py")}
    assert relative == {"tools/runs/a.py", "tools/runs/__init__.py", "tools/runs/b.py"}
    before = gate.key(tool, [], b"shared")
    (tmp_path / "tools" / "other.py").write_text("Z = 4\n")
    assert gate.key(tool, [], b"shared") == before
    (tmp_path / "src" / "dew" / "deep.py").write_text("X = 2\n")
    moved = gate.key(tool, [], b"shared")
    assert moved != before
    assert len({moved, gate.key(tool, ["torch==2.14.0"], b"shared"), gate.key(tool, [], b"other")}) == 3
    (tmp_path / "locks" / "cpu.txt").write_text("jax==2\n")
    assert gate.key(tool, [], b"shared") != moved
