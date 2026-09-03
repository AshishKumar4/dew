"""The direction of the import arrow around `dew.rl`.

`docs/design/plan.md`, section 5.1, makes `dew.rl` a subpackage instead of a
distribution of its own, and the invariant that keeps that decision reversible
is that the arrow never turns around: `dew.rl` may read `dew`, and nothing
under `dew` outside `dew.rl` and `dew.objectives.rl` may read `dew.rl`. If it
holds, splitting the package out later is a directory move and a
`pyproject.toml` entry.

Two of these read the source with `ast`, which sees an import a module never
executes. The third imports the package in a fresh interpreter and asks what
came with it, which sees an import the source does not name.
"""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src"
PACKAGE = SOURCE / "dew"
RL = PACKAGE / "rl"
OBJECTIVES_RL = PACKAGE / "objectives" / "rl"

FORBIDDEN = ("dew.training", "dew.data", "dew.objectives", "dew.registry")
"""The seams `dew.rl` is not allowed to know about: the trainer, the data path,
the objectives that compose it and the registry that builds models. An
advantage estimator that reaches any of them has stopped being array math."""


def imported_modules(path):
    """Every module a file imports, relative imports resolved to absolute.

    A relative import at level N starts N packages up from the file, and for
    `__init__.py` that is the package the file defines, so both cases drop the
    file's own name first.
    """
    parts = path.relative_to(SOURCE).with_suffix("").parts
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = ".".join(parts[:-node.level]
                                + ((node.module,) if node.module else ()))
            else:
                base = node.module or ""
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def python_files(directory):
    return sorted(path for path in directory.rglob("*.py"))


def test_the_rl_package_imports_no_trainer_no_data_and_no_objective():
    offenders = {
        str(path.relative_to(SOURCE)): sorted(
            name for name in imported_modules(path)
            if name.startswith(FORBIDDEN))
        for path in python_files(RL)
    }
    offenders = {path: names for path, names in offenders.items() if names}

    assert offenders == {}


def test_nothing_outside_the_rl_packages_imports_dew_rl():
    """`dew.objectives.rl` is the one place the arrow may be followed, because
    an RL objective is what composes these functions."""
    allowed = (RL, OBJECTIVES_RL)
    offenders = {}
    for path in python_files(PACKAGE):
        if any(path.is_relative_to(directory) for directory in allowed):
            continue
        names = sorted(name for name in imported_modules(path)
                       if name == "dew.rl" or name.startswith("dew.rl."))
        if names:
            offenders[str(path.relative_to(SOURCE))] = names

    assert offenders == {}


def test_importing_dew_rl_loads_nothing_else_from_dew():
    """What the source says and what the interpreter does are different claims.
    This one is the transitive closure. After `import dew.rl`, every `dew`
    module in `sys.modules` is `dew` itself or one of the package's own, so an
    import two files deep cannot smuggle the trainer in."""
    program = ("import dew.rl, sys; "
               "print(sorted(name for name in sys.modules if name.startswith('dew')))")
    result = subprocess.run([sys.executable, "-c", program], text=True,
                            capture_output=True,
                            env={"PYTHONPATH": str(SOURCE), "JAX_PLATFORMS": "cpu",
                                 "PATH": "/usr/bin:/bin"})

    assert result.returncode == 0, result.stderr
    loaded = ast.literal_eval(result.stdout.strip())
    assert {"dew", "dew.rl", "dew.rl.advantage", "dew.rl.surrogate"} <= set(loaded)
    assert [name for name in loaded
            if name != "dew" and not name.startswith("dew.rl")] == []


@pytest.mark.parametrize("module", ["advantage", "surrogate"])
def test_every_function_a_module_defines_is_re_exported(module):
    """`docs/design/plan.md` section 5.2 puts one family per module and the
    build order rules out an `rl.py` monolith, so `dew.rl` is the flat surface
    over both files. A function that lands in one of them and not in `__all__`
    is a function nobody can reach through the package."""
    import dew.rl

    submodule = importlib.import_module(f"dew.rl.{module}")
    defined = {name for name, value in vars(submodule).items()
               if not name.startswith("_") and callable(value)
               and getattr(value, "__module__", None) == submodule.__name__}

    assert defined <= set(dew.rl.__all__), \
        f"not re-exported: {sorted(defined - set(dew.rl.__all__))}"
