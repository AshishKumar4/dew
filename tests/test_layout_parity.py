"""Layout references depend on every implementation that builds or judges them."""

import dataclasses
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def test_benchmark_cases_can_be_read_without_either_framework():
    """The torch-only environment can read presets without loading Dew or JAX."""
    program = (
        "import json, sys; import benchmark_cases as cases; "
        "assert not any(name in sys.modules for name in ('jax', 'torch', 'dew')); "
        "print(json.dumps([(case.architecture, case.batch_size, case.seq_len, case.orders) "
        "for case in cases.small_cases('bfloat16') if case.architecture == 'causal_transformer']))"
    )
    completed = subprocess.run([sys.executable, "-I", "-c",
                                f"import sys; sys.path.insert(0, {str(TOOLS)!r}); {program}"],
                               capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout) == [["causal_transformer", 16, 512, False]] * 2


def test_case_labels_keep_the_mesh_fields_in_the_trainers_order(monkeypatch):
    """The standalone case names mesh axes in the order the trainer lays them out."""
    from dew.training import MeshSpec

    monkeypatch.syspath_prepend(str(TOOLS))
    cases = importlib.import_module("benchmark_cases")
    names = [field.name for field in dataclasses.fields(MeshSpec)]
    mesh = dict.fromkeys(reversed(names), 2)
    assert cases.mesh_label(mesh) == "-".join(f"{name}2" for name in names)
    assert cases.mesh_label(dict.fromkeys(names, 1)) == "data"


@pytest.mark.parametrize("changed", ["benchmark_cases", "benchmark_models"])
def test_a_changed_benchmark_owner_refuses_the_cached_reference(tmp_path, monkeypatch, changed):
    """A change to either shared owner invalidates a persisted layout reference."""
    monkeypatch.syspath_prepend(str(TOOLS))
    tool = importlib.import_module("layout_parity")
    dependency = importlib.import_module(changed)
    copied = tmp_path / f"{changed}.py"
    copied.write_bytes(Path(dependency.__file__).read_bytes())
    monkeypatch.setattr(dependency, "__file__", str(copied))
    tool.source_digest.cache_clear()
    try:
        case = importlib.import_module("benchmark_cases").Case("causal_transformer", seq_len=2)
        batch = {"text": np.array([[0, 1, 2]], np.int32)}
        name, _ = tool._cache_name("reference", case, batch, steps=1)
        path = tmp_path / name
        tool._write(path, {"weight": np.array([1.0, -2.5])}, {})
        reader = tool.References(tmp_path)
        assert reader._path("reference", case, batch, steps=1)[0] == path
        copied.write_bytes(copied.read_bytes() + b"\n# Changed implementation bytes.\n")
        tool.source_digest.cache_clear()
        with pytest.raises(FileNotFoundError, match="--prepare"):
            reader._path("reference", case, batch, steps=1)
        assert path.exists()
    finally:
        tool.source_digest.cache_clear()
