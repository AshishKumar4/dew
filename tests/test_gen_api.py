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
