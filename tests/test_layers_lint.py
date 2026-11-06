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


def test_a_kernel_reaching_above_the_network_layer_crosses_its_contract(tmp_path):
    """The kernels' contract (row 4340): a module under dew.nn.kernels importing a feature package above the
    network layer is a crossing; the network layer and telemetry are its own."""
    root = tree(tmp_path, {"dew.nn.kernels.report": "from dew.training import Trainer\n",
                           "dew.nn.kernels.timed": "import dew.telemetry\nfrom dew.nn import layers\n",
                           "dew.training.trainer": "", "dew.telemetry.__init__": "", "dew.nn.layers": ""})
    found = [crossing for crossing in layers.crossings(root) if crossing.module.startswith("dew.nn.kernels")]
    contract = "kernels import nothing above the network layer"
    assert found == [layers.Crossing(contract, "dew.nn.kernels.report", "dew.training")]


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
