"""tools/gen_exports.py writes a lazy package's TYPE_CHECKING imports and `__all__` from its `_EXPORTS`, the
one place the surface is declared: on a scratch package, one whose three lists disagree fails the check,
is rewritten to agree, and then imports each name lazily and passes ruff as the tree's own packages do."""

import ast
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "gen_exports.py"

PACKAGE = '''"""A lazy package."""

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparx.shapes import Circle

__version__ = "1.0"

_EXPORTS = {"Circle": "sparx.shapes", "Square": "sparx.shapes", "area": "sparx.measures",
            "PI": "sparx.measures"}


def __getattr__(name: str) -> type | Callable:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)


__all__ = [
    "Circle",
]
'''


def tree(root: Path) -> Path:
    """A lazy package, sparx, under root, the root the tool is given."""
    (root / "sparx").mkdir()
    (root / "sparx" / "__init__.py").write_text(PACKAGE)
    (root / "sparx" / "shapes.py").write_text("class Circle:\n    pass\n\n\nclass Square:\n    pass\n")
    (root / "sparx" / "measures.py").write_text(
        "PI = 3.0\n\n\ndef area(radius):\n    return PI * radius**2\n")
    return root


def generate(root: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *arguments, str(root / "sparx")], capture_output=True,
                          text=True, timeout=120)


def surface(root: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """The package's three lists, as a type checker and `import *` read them: `_EXPORTS`, the TYPE_CHECKING
    imports (name to module) and `__all__`."""
    tree = ast.parse((root / "sparx" / "__init__.py").read_text())
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and node.targets[0].id in ("_EXPORTS", "__all__")}
    checking = next(node for node in tree.body if isinstance(node, ast.If))
    imported = {alias.name: statement.module for statement in checking.body for alias in statement.names}
    return values["_EXPORTS"], imported, values["__all__"]


def test_a_package_whose_lists_disagree_is_rewritten_from_its_exports(tmp_path):
    root = tree(tmp_path)
    assert generate(root, "--check").returncode == 1
    assert generate(root).returncode == 0
    assert generate(root, "--check").returncode == 0
    exports, imported, names = surface(root)
    assert imported == exports and sorted(names) == sorted([*exports, "__version__"])
    script = "import sparx; print(sparx.area(1.0), sparx.Square.__name__)"
    lazy = subprocess.run([sys.executable, "-c", script], cwd=root, env={"PYTHONPATH": str(root)},
                          capture_output=True, text=True, check=True)
    assert lazy.stdout.split() == ["3.0", "Square"]
    ruff = subprocess.run([sys.executable, "-m", "ruff", "check", "--isolated", "--line-length=110",
                           "--select=I001,RUF022,F401,F811", str(root / "sparx" / "__init__.py")],
                          capture_output=True, text=True)
    assert ruff.returncode == 0, ruff.stdout


def test_an_export_moved_to_another_module_moves_its_import(tmp_path):
    root = tree(tmp_path)
    generate(root)
    package = root / "sparx" / "__init__.py"
    package.write_text(package.read_text().replace('"area": "sparx.measures"', '"area": "sparx.shapes"'))
    assert generate(root, "--check").returncode == 1
    generate(root)
    exports, imported, _ = surface(root)
    assert imported == exports and imported["area"] == "sparx.shapes"
