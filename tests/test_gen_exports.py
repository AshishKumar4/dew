"""tools/gen_exports.py writes a lazy package's TYPE_CHECKING imports and `__all__` from its `_EXPORTS`, the
one place the surface is declared: on a scratch package, one whose three lists disagree fails the check and
is rewritten so that each consumer reads the same surface: a program's lazy lookup, which binds nothing at
import, `from package import *`, and the imports a type checker reads, run here with TYPE_CHECKING on, each
binding its name to the object its module defines; and ruff's gate passes it. (The suite's environment holds
no type checker; the pyright gate reads src/dew's own packages.)"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "gen_exports.py"

PACKAGE = '''"""A lazy package."""

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparx.shapes import Circle

__version__ = "1.0"

_EXPORTS = {"Circle": "sparx.shapes", "Model10": "sparx.shapes", "Model2": "sparx.shapes",
            "area": "sparx.measures", "PI": "sparx.measures"}


def __getattr__(name: str) -> type | Callable:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)


__all__ = [
    "Circle",
]
'''

CONSUMERS = '''import importlib, json, sys, typing
typing.TYPE_CHECKING = sys.argv[1] == "static"  # on: the imports a type checker reads run, binding each name
import sparx
exports = sparx._EXPORTS
bound = {name: value is getattr(importlib.import_module(exports[name]), name)
         for name, value in vars(sparx).items() if name in exports}
namespace = {}
exec("from sparx import *", namespace)
print(json.dumps({"bound": bound, "star": sorted(name for name in namespace if name != "__builtins__"),
                  "area": sparx.area(1.0), "shapes": [sparx.Model2.__name__, sparx.Model10.__name__]}))
'''


def tree(root: Path) -> Path:
    """A lazy package, sparx, under root."""
    (root / "sparx").mkdir()
    (root / "sparx" / "__init__.py").write_text(PACKAGE)
    (root / "sparx" / "shapes.py").write_text(
        "class Circle:\n    pass\n\n\nclass Model2:\n    pass\n\n\nclass Model10:\n    pass\n")
    (root / "sparx" / "measures.py").write_text(
        "PI = 3.0\n\n\ndef area(radius):\n    return PI * radius**2\n")
    return root


def generate(root: Path, *arguments: str) -> int:
    return subprocess.run([sys.executable, str(TOOL), *arguments, str(root / "sparx")], capture_output=True,
                          text=True, timeout=120).returncode


def consumed(root: Path, reading: str) -> dict:
    """What the package's consumers read, imported as a program does (`runtime`: each name is bound only
    when the lazy lookup reads it) or with its TYPE_CHECKING imports run (`static`: as a type checker reads
    them, each bound by its own import, the lazy lookup aside): the names it binds, each whether it is the
    object its module defines, `from sparx import *`, and the exports called."""
    done = subprocess.run([sys.executable, "-c", CONSUMERS, reading], cwd=root,
                          env={"PYTHONPATH": str(root)}, capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def test_a_package_whose_lists_disagree_is_rewritten_for_every_consumer(tmp_path):
    root = tree(tmp_path)
    assert generate(root, "--check") == 1
    assert generate(root) == 0
    assert generate(root, "--check") == 0
    star, shapes = ["Circle", "Model10", "Model2", "PI", "__version__", "area"], ["Model2", "Model10"]
    assert consumed(root, "runtime") == {"bound": {}, "star": star, "area": 3.0, "shapes": shapes}
    bound = dict.fromkeys(["Circle", "Model10", "Model2", "PI", "area"], True)
    assert consumed(root, "static") == {"bound": bound, "star": star, "area": 3.0, "shapes": shapes}
    ruff = subprocess.run([sys.executable, "-m", "ruff", "check", "--config", str(REPO / "pyproject.toml"),
                           "--select=I001,RUF022,F401,F811", str(root / "sparx" / "__init__.py")],
                          capture_output=True, text=True)
    assert ruff.returncode == 0, ruff.stdout


def test_an_export_moved_to_another_module_is_read_from_there(tmp_path):
    root = tree(tmp_path)
    generate(root)
    shapes, measures, package = (root / "sparx" / name
                                 for name in ("shapes.py", "measures.py", "__init__.py"))
    shapes.write_text(shapes.read_text() + "\n\ndef area(radius):\n    return 3.0 * radius**2\n")
    measures.write_text("PI = 3.0\n")
    package.write_text(package.read_text().replace('"area": "sparx.measures"', '"area": "sparx.shapes"'))
    assert generate(root, "--check") == 1
    assert generate(root) == 0
    assert consumed(root, "runtime")["area"] == 3.0
    assert consumed(root, "static")["bound"]["area"]
