"""tools/lint_layers.py: the layering contracts read from a source tree's imports, on fixture trees."""

from pathlib import Path

import pytest
from tool_support import load

layers = load("lint_layers")

CONTRACT = {"the data layer imports no trainer": layers.Contract(("dew.data",), ("dew.data.hub",),
                                                                 ("dew.training",))}
CLOSED = {"telemetry imports no trainer": layers.Contract(("dew.telemetry",), (), ("dew.training",),
                                                           closure=True),
          "data imports no trainer": layers.Contract(("dew.data",), (), ("dew.training",))}


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


def test_a_kernel_reaching_above_the_network_layer_crosses_its_contract(tmp_path):
    """The kernels' contract (row 4340): a module under dew.nn.kernels importing a feature package above the
    network layer is a crossing; the network layer and telemetry are its own."""
    root = tree(tmp_path, {"dew.nn.kernels.report": "from dew.training import Trainer\n",
                           "dew.nn.kernels.timed": "import dew.telemetry\nfrom dew.nn import layers\n",
                           "dew.training.trainer": "", "dew.telemetry.__init__": "", "dew.nn.layers": ""})
    contract = "kernels import nothing above the network layer"
    found = [crossing for crossing in layers.crossings(root) if crossing.contract == contract]
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


def test_a_contract_over_its_closure_is_crossed_by_a_chain_its_imports_run(tmp_path):
    """A module whose imports run a forbidden package through others (a function's import among them, a
    package's initializer, a lazy package's export it names) crosses a contract over its closure, the chain
    named; a contract of direct imports alone is not crossed by the same chain (rows 5988-5990)."""
    root = tree(tmp_path, {
        "dew.__init__": "_EXPORTS = {'Trainer': 'dew.training.trainer'}\n",
        "dew.telemetry.sink": "def write():\n    from dew.nn import timing\n",
        "dew.data.plan": "from dew.nn import timing\n",
        "dew.nn.__init__": "", "dew.nn.timing": "from dew import Trainer\n",
        "dew.training.__init__": "", "dew.training.trainer": ""})
    assert list(layers.crossings(root, CLOSED)) == [layers.Crossing(
        "telemetry imports no trainer", "dew.telemetry.sink", "dew.training",
        ("dew.nn.timing", "dew.training"))]


@pytest.mark.parametrize(("modules", "chain"), [
    ({"dew.telemetry.sink": "from dew.bridge import Reader\n",
      "dew.bridge.__init__": "_EXPORTS = {'Reader': 'dew.nn'}\n",
      "dew.nn.__init__": "_EXPORTS = {'Reader': 'dew.training.reader'}\n",
      "dew.training.__init__": "", "dew.training.reader": ""}, ("dew.training",)),
    ({"dew.telemetry.sink": "import dew.bridge\n", "dew.bridge.__init__": "from dew import *\n",
      "dew.__init__": "_EXPORTS = {'Trainer': 'dew.training.trainer'}\n", "dew.training.__init__": "",
      "dew.training.trainer": ""}, ("dew.bridge", "dew.training")),
], ids=["a lazy export read off another lazy package", "a star import of a lazy package"])
def test_a_lazy_lookup_runs_the_module_its_name_resolves_to(tmp_path, modules, chain):
    """A name a lazy package exports from another lazy package is read off that one in turn
    (`dew._exports.lookup`'s getattr), and a star import of a lazy package reads every name it exports: the
    module each resolves to runs, and crosses a contract over its closure."""
    assert list(layers.crossings(tree(tmp_path, modules), CLOSED)) == [
        layers.Crossing("telemetry imports no trainer", "dew.telemetry.sink", "dew.training", chain)]


def test_the_contracts_hold_telemetry_and_the_algebra_over_their_closure_and_the_network_layer_directly(
        tmp_path):
    """The contracts as declared (`LAYERS`): telemetry's and the algebra's are crossed by a chain through a
    package whose initializer alone imports the trainer; the network layer's no-trainer contract by its own
    import of it, not by that chain (row 5992's bridge still open)."""
    root = tree(tmp_path, {"dew.telemetry.sink": "import dew.bridge.child\n",
                           "dew.algebra.terms": "import dew.bridge.child\n",
                           "dew.nn.client": "import dew.bridge.child\n",
                           "dew.nn.direct": "import dew.training\n",
                           "dew.bridge.__init__": "import dew.training\n", "dew.bridge.child": "",
                           "dew.training.__init__": ""})
    held = ("telemetry imports nothing above the network layer",
            "the algebra imports nothing above the network layer", "the network layer imports no trainer")
    assert sorted(crossing for crossing in layers.crossings(root) if crossing.contract in held) == sorted([
        layers.Crossing(held[0], "dew.telemetry.sink", "dew.training", ("dew.bridge", "dew.training")),
        layers.Crossing(held[1], "dew.algebra.terms", "dew.training", ("dew.bridge", "dew.training")),
        layers.Crossing(held[2], "dew.nn.direct", "dew.training")])


@pytest.mark.parametrize(("modules", "contracts"), [
    ({"dew.data.hub": "import dew.training\n", "dew.data.plan": "import dew.nn\nfrom . import hub\n",
      "dew.nn.layers": "import dew.training\n"}, CONTRACT),
    ({"dew.telemetry.sink": "import dew.nn.timing\n",
      "dew.nn.timing": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import dew.training\n",
      "dew.training.__init__": ""}, CLOSED),
], ids=["a module left out or an allowed import", "a type checker's import in a closure"])
def test_an_import_no_contract_forbids_at_run_time_is_no_crossing(tmp_path, modules, contracts):
    """A module a contract leaves out, or an import it allows, crosses nothing; nor does a chain through an
    import under TYPE_CHECKING, which a type checker alone reads and which runs no module, cross a contract
    over its closure, though the module's own such import still crosses directly."""
    assert list(layers.crossings(tree(tmp_path, modules), contracts)) == []
