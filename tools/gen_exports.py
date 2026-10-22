"""Generate a lazily exporting package's import surface from its `_EXPORTS`, the one place it is declared:
each `__init__.py` under the package (src/dew by default) that assigns `_EXPORTS` (a name to the module that
defines it) has its `if TYPE_CHECKING:` imports and its `__all__` written from it, so a type checker, an
editor and `from package import *` read the surface without running the lazy lookup, and none of the three
can drift.
`_EXPORTS` is a literal dict of names to absolute module names.

    python tools/gen_exports.py [--check] [PACKAGE]

ruff orders and wraps the imports and `__all__` (I001, RUF022), so the written file is the one the ruff gate
passes; `__all__` also names `__version__` where the package assigns it."""

import argparse
import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def ruffed(text: str, path: Path) -> str:
    """`text`, the source of `path`, with its imports and `__all__` ordered and wrapped as ruff (the suite's
    environment's, which the ruff gate runs) orders them under the repository's configuration."""
    command = [sys.executable, "-m", "ruff", "check", "--fix", "--quiet", "--select", "I001,RUF022",
               "--config", str(REPO / "pyproject.toml"), "--stdin-filename", str(path), "-"]
    return subprocess.run(command, input=text, capture_output=True, text=True, check=True).stdout


def imports(exports: dict[str, str]) -> list[str]:
    """The `if TYPE_CHECKING:` block importing each export from its module, one statement a module, in
    `_EXPORTS`' order; `ruffed` orders and wraps them."""
    named = {module: [name for name, owner in exports.items() if owner == module]
             for module in exports.values()}
    return ["if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)",
            *(f"    from {module} import {', '.join(names)}" for module, names in named.items())]


def written(path: Path) -> str:
    """`path` with its TYPE_CHECKING imports and `__all__` written from its `_EXPORTS`."""
    text = path.read_text()
    tree = ast.parse(text)
    assigned = {target.id: node for node in tree.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)}
    exports = ast.literal_eval(assigned["_EXPORTS"].value)
    names = [*exports, *(["__version__"] if "__version__" in assigned else [])]
    checking = next(node for node in tree.body if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                    and node.test.id == "TYPE_CHECKING")
    lines = text.splitlines()
    spans = sorted([(checking.lineno, checking.end_lineno, imports(exports)),
                    (assigned["__all__"].lineno, assigned["__all__"].end_lineno,
                     ["__all__ = [", *(f'    "{name}",' for name in names), "]"])], reverse=True)
    for start, end, block in spans:
        lines[start - 1:end] = block
    return ruffed("\n".join(lines) + "\n", path)


def packages(package: Path) -> list[Path]:
    """Every `__init__.py` under `package` that assigns `_EXPORTS`."""
    return sorted(path for path in package.rglob("__init__.py") if "\n_EXPORTS = " in path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Check the written blocks without writing them.")
    parser.add_argument("package", nargs="?", type=Path, default=REPO / "src" / "dew")
    args = parser.parse_args()
    stale = []
    for path in packages(args.package):
        text = written(path)
        if text != path.read_text():
            stale.append(path.relative_to(args.package.parent))
            if not args.check:
                path.write_text(text)
    if args.check and stale:
        sys.exit(f"gen_exports: run python tools/gen_exports.py to rewrite {', '.join(map(str, stale))}")
    print(f"gen_exports: {len(packages(args.package))} packages")


if __name__ == "__main__":
    main()
