"""The API reference check resolves Dew names across a page's code units.

A tutorial imports in one cell and calls in the next, and a docs page imports
in an early block that later blocks rely on, so the names one unit binds are
the next unit's names too. site/scripts/gen_api.py reads `src/dew` with griffe
and imports neither JAX nor Dew; CI runs this beside `gen_api.py --check`.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("griffe")

SCRIPT = Path(__file__).resolve().parents[1] / "site" / "scripts" / "gen_api.py"


@pytest.fixture(scope="module")
def gen_api():
    spec = importlib.util.spec_from_file_location("gen_api", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_api"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def documented(gen_api):
    package = gen_api.load()
    return (package, *gen_api.documented(package))


def test_a_name_bound_in_one_cell_is_resolved_in_the_next(gen_api, documented, tmp_path):
    notebook = tmp_path / "two.ipynb"
    notebook.write_text(json.dumps({"cells": [
        {"cell_type": "code", "source": ["import dew\n"]},
        {"cell_type": "markdown", "source": ["`dew.not_here` in prose is not code\n"]},
        {"cell_type": "code", "source": ["dew.removed_api()\n", "dew.Profiler\n"]},
    ]}))
    assert gen_api.unresolved_in("two.ipynb", notebook, *documented) == [
        "`dew.removed_api`: dew has no public removed_api"]


def test_a_docs_page_carries_its_imports_from_block_to_block(gen_api, documented, tmp_path):
    page = tmp_path / "page.md"
    page.write_text("```python\nfrom dew.training import MeshSpec\n```\n\nText.\n\n"
                    "```python\nMeshSpec().build()\nMeshSpec.build_mesh()\n```\n")
    assert gen_api.unresolved_in("page.md", page, *documented) == [
        "`MeshSpec.build_mesh`: dew.training.distributed.MeshSpec has no build_mesh"]


def test_an_instance_of_a_dew_class_is_held_to_its_members_across_cells(gen_api, documented, tmp_path):
    """`name = Class(...)` binds an instance: a later cell's attribute it
    does not have is reported, and so is calling a value it holds as though
    it were a method, which is what a method renamed to make room for a
    field of its old name leaves behind. A name assigned otherwise too is
    not held to the class."""
    notebook = tmp_path / "instance.ipynb"
    notebook.write_text(json.dumps({"cells": [
        {"cell_type": "code", "source": [
            "from dew.objectives.diffusion import DiffusionObjective\n",
            "objective = DiffusionObjective(model, process, inputs)\n",
            "other = DiffusionObjective(model, process, inputs)\n",
            "other = something_else()\n"]},
        {"cell_type": "code", "source": [
            "objective.model_variables(params)\n",
            "objective.pipeline(state)\n",
            "objective.process.denoiser(model, params, {})\n",
            "objective.variables(params)\n",
            "objective.sampler\n",
            "other.anything(params)\n"]},
    ]}))
    assert gen_api.unresolved_in("instance.ipynb", notebook, *documented) == [
        "`objective.sampler`: dew.objectives.diffusion.objective.DiffusionObjective has no sampler",
        "`objective.variables(...)`: variables is a value "
        "dew.objectives.diffusion.objective.DiffusionObjective holds, not a method"]


def test_names_that_differ_only_in_case_link_to_their_own_headings(gen_api, documented):
    """`dew.sampling` documents the class `Sample` and the function `sample`,
    whose headings render as `#sample` and `#sample-1`; each links to its own."""
    _, pages, home = documented
    urls = gen_api.linked(pages, home).by_path
    assert urls["dew.sampling.Sample"] != urls["dew.sampling.sample"]
    assert {urls["dew.sampling.Sample"].rsplit("#", 1)[1], urls["dew.sampling.sample"].rsplit("#", 1)[1]} == {
        "sample", "sample-1"}


def test_a_module_that_declares_all_needs_its_own_page(gen_api, documented):
    """With `dew.nn.multimodal`'s page dropped, the module itself is
    reported, not only the names it exports."""
    package, pages, home = documented
    assert gen_api.undocumented(package, pages, home) == []
    page = "dew.nn.multimodal"
    without = gen_api.undocumented(package, {path: held for path, held in pages.items() if path != page},
                                   {name: path for name, path in home.items() if path != page})
    assert f"{page} declares __all__ but has no API page; add it to the groups" in without
