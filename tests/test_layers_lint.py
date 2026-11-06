"""tools/lint_layers.py: the layering contracts read from a source tree's imports, on fixture trees."""

from pathlib import Path

import pytest
from tool_support import load

layers = load("lint_layers")

CONTRACT = {"the data layer imports no trainer": layers.Contract(("dew.data",), ("dew.data.hub",),
                                                                 ("dew.training",))}


def tree(root: Path, modules: dict[str, str]) -> Path:
    """A `dew` package under `root` holding `modules`, by dotted name, each with its source."""
    for name, source in modules.items():
        path = root.joinpath(*name.split(".")).with_suffix(".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return root


@pytest.mark.parametrize("source", [
    "import dew.training\n",
    "from dew.training import Trainer\n",
    "from dew import training\n",
    "from ..training import Trainer\n",
    "def read():\n    from dew.training.trainer import Trainer\n",
    "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from dew.training import Trainer\n",
])
def test_an_import_of_a_forbidden_package_is_a_crossing_however_written(tmp_path, source):
    root = tree(tmp_path, {"dew.data.plan": source, "dew.training.trainer": ""})
    assert list(layers.crossings(root, CONTRACT)) == [
        layers.Crossing("the data layer imports no trainer", "dew.data.plan", "dew.training")]


def test_a_module_the_contract_leaves_out_or_an_allowed_import_is_no_crossing(tmp_path):
    root = tree(tmp_path, {"dew.data.hub": "import dew.training\n",
                           "dew.data.plan": "import dew.nn\nfrom . import hub\n",
                           "dew.nn.layers": "import dew.training\n"})
    assert list(layers.crossings(root, CONTRACT)) == []


def test_a_package_init_resolves_its_relative_imports_from_itself(tmp_path):
    root = tree(tmp_path, {"dew.data.__init__": "from .plan import Plan\nfrom ..training import Trainer\n",
                           "dew.data.plan": ""})
    assert list(layers.crossings(root, CONTRACT)) == [
        layers.Crossing("the data layer imports no trainer", "dew.data", "dew.training")]


def test_the_gate_fails_a_tree_with_a_crossing_and_passes_a_clean_one(tmp_path, monkeypatch):
    monkeypatch.setattr(layers, "LAYERS", CONTRACT)
    crossed = tree(tmp_path / "crossed", {"dew.data.plan": "import dew.training\n"})
    clean = tree(tmp_path / "clean", {"dew.data.plan": "import dew.nn\n"})
    assert layers.main([str(clean)]) == 0
    assert layers.main([str(crossed)]) == 1
