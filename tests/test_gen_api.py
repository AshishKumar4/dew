"""The API reference check resolves Dew names across a page's code units.

A tutorial imports in one cell and calls in the next, and a docs page imports
in an early block that later blocks rely on, so the names one unit binds are
the next unit's names too. site/scripts/gen_api.py reads `src/dew` with griffe
and imports neither JAX nor Dew; CI runs this beside `gen_api.py --check`.
"""

import ast
import json
import re

import pytest
from tool_support import load

pytest.importorskip("griffe")


@pytest.fixture(scope="module")
def gen_api():
    return load("gen_api", "site/scripts")


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


def unresolved_page(gen_api, documented, page, text: str) -> list[str]:
    """What the API check reports of a docs page holding `text`."""
    page.write_text(text)
    return gen_api.unresolved_in(page.name, page, *documented)


NO_BUILD_MESH = "`MeshSpec.build_mesh`: dew.distributed.MeshSpec has no build_mesh"


def test_a_docs_page_carries_its_imports_from_block_to_block(gen_api, documented, tmp_path):
    assert unresolved_page(gen_api, documented, tmp_path / "page.md",
                           "```python\nfrom dew.training import MeshSpec\n```\n\nText.\n\n"
                           "```python\nMeshSpec().build()\nMeshSpec.build_mesh()\n```\n") == [NO_BUILD_MESH]


def test_a_name_a_scope_binds_hides_a_dew_import_in_that_scope_alone(gen_api, documented, tmp_path):
    """A parameter or a comprehension's target named like an import is the
    scope's own name, as Python reads it (tools/lint_scopes.py), so its uses
    there are not Dew's, nor is an instance's name a parameter takes; the
    import still stands everywhere else."""
    assert unresolved_page(gen_api, documented, tmp_path / "scoped.md",
                           "```python\nfrom dew.training import MeshSpec\n\n"
                           "def build(MeshSpec):\n    return MeshSpec.anything()\n\n"
                           "names = [MeshSpec.other for MeshSpec in specs]\n"
                           "mesh = MeshSpec()\n\ndef reads(mesh):\n    return mesh.not_a_member\n\n"
                           "MeshSpec.build_mesh()\n```\n") == [NO_BUILD_MESH]


def test_each_scope_reads_the_instances_and_aliases_it_binds_itself(gen_api, documented, tmp_path):
    """An instance an earlier block bound is held to its class inside a
    later constructor's arguments, one a function binds is held there, and
    an alias a function imports is that function's own, however the module
    spells the same name."""
    assert unresolved_page(gen_api, documented, tmp_path / "instances.md",
                           "```python\nfrom dew.training import MeshSpec\nfrom dew import sampling as api\n"
                           "mesh = MeshSpec()\n```\n\n"
                           "```python\nother = MeshSpec(mesh.build_mesh())\napi.Sampling\n\n"
                           "def make():\n    from dew import training as api\n    spec = MeshSpec()\n"
                           "    spec.not_here\n    return api.Sampling\n```\n") == [
        "`spec.not_here`: dew.distributed.MeshSpec has no not_here",
        "`api.Sampling`: dew.training has no public Sampling",
        "`mesh.build_mesh`: dew.distributed.MeshSpec has no build_mesh"]


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


@pytest.fixture
def reference_package(gen_api, tmp_path, monkeypatch):
    source = tmp_path / "src" / "example"
    source.mkdir(parents=True)
    package = gen_api.Package("example", tmp_path, "https://example.org/", [("API", ["example"])],
                              overview="overview.md", usage=["overview.md"])
    monkeypatch.setattr(gen_api, "P", package)
    monkeypatch.setattr("sys.argv", ["gen_api.py"])
    return source / "__init__.py", tmp_path / package.overview


@pytest.mark.parametrize("control,field", [("steps", "updates"), ("tokens", "compute")])
def test_source_changes_reach_overview_signatures_fields_and_return_types(
        gen_api, reference_package, control, field):
    source, overview = reference_package
    code = f"""from collections.abc import Callable
from dataclasses import dataclass

@dataclass
class State:
    step: int
    {field}: int = 0

class Trainer:
    def fit(self, dataset, /, *, {control}: int | None = None) -> State: ...

CompiledStep = Callable[[State, dict], tuple[State, float, dict, bool, bool, dict | None]]
"""
    source.write_text(code)
    marker = "<!-- api: example.Trainer.fit example.State example.CompiledStep -->"
    before, after = "# Overview\n\nContracts.\n\n", "\n\nUsage.\n"
    overview.write_text(before + marker + after)
    gen_api.main()
    prefix, block, suffix = re.split(r"```text\n(.*?)\n```", overview.read_text(), maxsplit=1, flags=re.S)
    assert prefix == before
    assert suffix.startswith(after)
    fit, state, compiled = block.split("\n\n")
    rendered_fit = ast.parse(fit.replace("def Trainer.fit(", "def fit(") + ":\n    pass").body[0]
    rendered_state = ast.parse(state.replace("class State(", "def State(") + ":\n    pass").body[0]
    tree = ast.parse(code)
    state_source, trainer_source, compiled_source = tree.body[2:]
    fit_source = trainer_source.body[0]
    fit_source.args.posonlyargs.pop(0)
    assert ast.dump(rendered_fit.args) == ast.dump(fit_source.args)
    assert ast.dump(rendered_fit.returns) == ast.dump(fit_source.returns)
    assert [(arg.arg, ast.dump(arg.annotation)) for arg in rendered_state.args.args] == [
        (held.target.id, ast.dump(held.annotation)) for held in state_source.body]
    assert [ast.dump(value) for value in rendered_state.args.defaults] == [
        ast.dump(held.value) for held in state_source.body if held.value is not None]
    assert ast.dump(ast.parse(compiled).body[0]) == ast.dump(compiled_source)


@pytest.mark.parametrize("reference", ["example.removed", "example.Model.removed", "example"])
def test_check_rejects_an_unresolvable_or_nondeclaration_reference(
        gen_api, reference_package, monkeypatch, reference):
    source, overview = reference_package
    source.write_text("class Model:\n    def fit(self): ...\n")
    overview.write_text(f"# Overview\n\n<!-- api: {reference} -->\n")
    monkeypatch.setattr("sys.argv", ["gen_api.py", "--check"])
    with pytest.raises(SystemExit) as stopped:
        gen_api.main()
    assert stopped.value.code == 1
    assert not (source.parents[2] / gen_api.P.content).exists()
