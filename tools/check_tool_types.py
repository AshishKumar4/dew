"""Every script under tools/ type-checked by pyright in the environment it runs in (row 7933).

A tool runs in Dew's own environment unless it names the reference releases it records against: a module-level
`COHORTS` (tools/reference_releases.py's RELEASES names: one, a list, or a mapping whose values name them), or
a PEP 723 header, which `uv run` reads (tools/superoptimize_bits.py). Each cohort set's tools are checked in a
virtual environment holding exactly its pins, built here by uv and kept under `ENVIRONMENTS` by its pins'
digest, so a reference library's types are its own release's: with its import unresolved, pyright reads its
every use as Unknown and finds none of the tool's errors. A tool whose cohorts pin one distribution at two
releases runs in no one environment, and is refused: it is split into a file a cohort.

A tool's verdict is kept by its key (`key`): pyright's release and configuration, the stubs, its environment's
pins and the code of every module it imports from the checkout, followed through their own imports
(`closure`). A key green before is not checked again: a commit that changes no tool and nothing a tool imports
runs no pyright here.

    python tools/check_tool_types.py            # every tool, its verdict printed; exit 1 where any is red
    python tools/check_tool_types.py --list     # each cohort set's tools, checking nothing

"""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tomllib
import types
from collections.abc import Iterable, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYRIGHT = "pyright@1.1.406"
"""The release tools/check_gates' pyright gate runs, so both read types alike."""
ROOTS = ("src", "tools", "tests")
"""Where an import of a tool's is looked up in the checkout, as pyproject.toml's extraPaths and Dew's install
put them on its path."""
ENVIRONMENTS = Path(".venv-tools")
"""Each cohort set's environment, `<digest>` under it, made once and read by every run after."""
DEW = "dew"
"""The environment of a tool that names no cohort: Dew's own, locks/cpu.txt's."""
STORE = "tools-types"
"""The records store's prefix for each green verdict (tools/armada/records.py)."""
TORCH_CPU = "https://download.pytorch.org/whl/cpu"
"""Where a cohort's torch comes from: its CPU build, which pyright reads as the CUDA build's types."""
PYTHON = "3.12"


def loaded(path: Path) -> types.ModuleType:
    """The module at `path`, loaded by its path, as tools/armada/ci.py loads its neighbours."""
    spec = importlib.util.spec_from_file_location(f"check_tool_types_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def releases() -> Mapping[str, Mapping[str, str]]:
    """tools/reference_releases.py's RELEASES."""
    return loaded(ROOT / "tools" / "reference_releases.py").RELEASES


def cohorts_of(path: Path) -> tuple[str, ...] | None:
    """The cohorts tool `path`'s module-level `COHORTS` names (a name, a list or tuple of them, or a mapping
    whose values are either), sorted; None where it names none."""
    for node in ast.parse(path.read_text(), str(path)).body:
        if isinstance(node, ast.Assign):
            targets, expression = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, expression = [node.target], node.value
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id in ("COHORT", "COHORTS") for target in targets):
            value = ast.literal_eval(expression)
            values = value.values() if isinstance(value, dict) else [value]
            return tuple(sorted({name for each in values
                                 for name in ([each] if isinstance(each, str) else each)}))
    return None


SCRIPT = re.compile(r"^# /// script\s*$(?P<body>.*?)^# ///\s*$", re.MULTILINE | re.DOTALL)
"""A PEP 723 header: its `# ` lines between `# /// script` and `# ///`."""


def header_of(path: Path) -> dict | None:
    """Tool `path`'s PEP 723 header, read as TOML; None where it has none."""
    found = SCRIPT.search(path.read_text())
    if found is None:
        return None
    return tomllib.loads("\n".join(line.removeprefix("#").removeprefix(" ")
                                   for line in found["body"].splitlines()))


def environment_of(path: Path, pins: Mapping[str, Mapping[str, str]]) -> tuple[str, list[str]]:
    """Tool `path`'s environment: its name (`DEW`, `script:<file>`, or its cohorts joined by `+`) and the
    requirements it is made of (none for Dew's). A tool whose cohorts pin a distribution at two releases
    raises ValueError, naming them."""
    header = header_of(path)
    if header is not None:
        return f"script:{path.relative_to(ROOT).as_posix()}", sorted(header.get("dependencies", []))
    cohorts = cohorts_of(path)
    if not cohorts:
        return DEW, []
    if unknown := [cohort for cohort in cohorts if cohort not in pins]:
        raise ValueError(f"{path}: COHORTS names {', '.join(unknown)}, which RELEASES does not")
    held: dict[str, str] = {}
    for cohort in cohorts:
        for distribution, release in pins[cohort].items():
            if held.setdefault(distribution, release) != release:
                raise ValueError(f"{path}: its cohorts {', '.join(cohorts)} pin {distribution} at "
                                 f"{held[distribution]} and {release}; no one environment holds both: "
                                 "split it into a file a cohort")
    return "+".join(cohorts), sorted(f"{distribution}=={release}" for distribution, release in held.items())


def imported(path: Path) -> set[str]:
    """The dotted names tool or module `path` imports, absolute and its relative ones resolved, with each
    `from a import b` also as `a.b` (b may be a module)."""
    names = set()
    package = path.relative_to(ROOT).with_suffix("").parts[1:-1] if path.is_relative_to(ROOT) else ()
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = ".".join([*package[:len(package) - node.level + 1], *(node.module or "").split(".")]
                            if node.level else (node.module or "").split(".")).strip(".")
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
    return names


def located(name: str) -> list[Path]:
    """The checkout's files an import of `name` executes: each package's __init__.py on the way and the
    module, looked up under `ROOTS` in order, the first root holding its first part."""
    parts = name.split(".")
    for root in ROOTS:
        base = ROOT / root
        if not ((base / parts[0]).is_dir() or (base / f"{parts[0]}.py").is_file()):
            continue
        found = []
        for depth in range(1, len(parts) + 1):
            here = base.joinpath(*parts[:depth])
            if (here / "__init__.py").is_file():
                found.append(here / "__init__.py")
            elif here.with_suffix(".py").is_file():
                found.append(here.with_suffix(".py"))
                break
            elif not here.is_dir():
                break
        return found
    return []


def closure(path: Path) -> list[Path]:
    """Tool `path` and every module of the checkout it imports, followed through their own imports."""
    seen, frontier = {path}, [path]
    while frontier:
        for name in imported(frontier.pop()):
            for found in located(name):
                if found not in seen:
                    seen.add(found)
                    frontier.append(found)
    return sorted(seen)


def common() -> bytes:
    """What every tool's key holds: pyright's release, its configuration in pyproject.toml, and the stubs."""
    digest = hashlib.sha256(PYRIGHT.encode())
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pyright"]
    digest.update(json.dumps(configuration, sort_keys=True).encode())
    for stub in sorted((ROOT / "stubs").rglob("*")):
        if stub.is_file():
            digest.update(stub.relative_to(ROOT).as_posix().encode() + stub.read_bytes())
    return digest.digest()


def key(path: Path, requirements: Iterable[str], shared: bytes) -> str:
    """Tool `path`'s verdict's key: `shared` (`common`), its environment (its requirements, or Dew's lock
    where it names none) and the code of its `closure`, each file by its path and bytes."""
    digest = hashlib.sha256(shared)
    held = list(requirements)
    digest.update(json.dumps(held).encode() if held else (ROOT / "locks" / "cpu.txt").read_bytes())
    for each in closure(path):
        digest.update(each.relative_to(ROOT).as_posix().encode() + each.read_bytes())
    return digest.hexdigest()


def tools() -> list[Path]:
    """Every Python file under tools/."""
    return sorted(path for path in (ROOT / "tools").rglob("*.py") if ".venv" not in path.parts)


def made(requirements: list[str]) -> Path:
    """The interpreter of the environment holding exactly `requirements`, made once under `ENVIRONMENTS` by
    their digest (a torch from its CPU index); a failed install raises CalledProcessError."""
    where = ROOT / ENVIRONMENTS / hashlib.sha256(json.dumps(requirements).encode()).hexdigest()[:16]
    python = where / "bin" / "python"
    if not (where / "made").is_file():
        subprocess.run(["uv", "venv", "--quiet", "--clear", "--python", PYTHON, str(where)], check=True)
        subprocess.run(["uv", "pip", "install", "--quiet", "--python", str(python),
                        "--extra-index-url", TORCH_CPU, "--index-strategy", "unsafe-best-match",
                        *requirements], check=True)
        (where / "made").write_text(json.dumps(requirements))
    return python


def checked(python: str, paths: list[Path]) -> dict[Path, list[str]]:
    """Each of `paths` with pyright's errors in it, read with `python`'s environment."""
    ran = subprocess.run(["uvx", PYRIGHT, "--pythonpath", python, "--outputjson",
                          *(str(path) for path in paths)], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "NODE_OPTIONS": "--max-old-space-size=8192"}, check=False)
    found: dict[Path, list[str]] = {path: [] for path in paths}
    for diagnostic in json.loads(ran.stdout)["generalDiagnostics"]:
        if diagnostic["severity"] == "error":
            at, line = Path(diagnostic["file"]), diagnostic["range"]["start"]["line"] + 1
            said = f"{diagnostic['message'].splitlines()[0]} ({diagnostic.get('rule', '')})"
            found.setdefault(at, []).append(f"{at.relative_to(ROOT)}:{line}: {said}")
    return found


def records() -> types.ModuleType | None:
    """tools/armada/records.py, the CI store the verdicts are kept in, or None where this run has no token."""
    if not os.environ.get("DEW_XLA_CACHE_TOKEN") or not os.environ.get("DEW_XLA_CACHE_URL"):
        return None
    return loaded(ROOT / "tools" / "armada" / "records.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="each cohort set's tools, checking nothing")
    args = parser.parse_args()
    pins, shared, store = releases(), common(), records()
    groups: dict[tuple[str, tuple[str, ...]], list[Path]] = {}
    red = 0
    for path in tools():
        try:
            name, requirements = environment_of(path, pins)
        except ValueError as refused:
            print(f"refused: {refused}")
            red += 1
            continue
        groups.setdefault((name, tuple(requirements)), []).append(path)
    if args.list:
        for (name, _), paths in sorted(groups.items()):
            print(f"{name}: {len(paths)} tools: {' '.join(str(path.relative_to(ROOT)) for path in paths)}")
        return 0
    token = os.environ.get("DEW_XLA_CACHE_TOKEN", "")
    for (name, requirements), paths in sorted(groups.items()):
        keys = {path: key(path, requirements, shared) for path in paths}
        wanted = [path for path in paths
                  if store is None or store.recorded(f"{STORE}/{keys[path]}", token) is None]
        print(f"{name}: {len(paths)} tools, {len(paths) - len(wanted)} green before", flush=True)
        if not wanted:
            continue
        try:
            python = str(made(list(requirements))) if requirements else sys.executable
        except subprocess.CalledProcessError as failed:
            print(f"red: {name}'s environment ({' '.join(requirements)}) was not made: {failed}")
            red += len(wanted)
            continue
        for path, errors in checked(python, wanted).items():
            if errors:
                red += 1
                print("\n".join(errors))
            elif store is not None and path in keys:
                store.written(f"{STORE}/{keys[path]}", token, b"{}")
    print(f"tool types: {red} tool(s) red" if red else "tool types: green")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
