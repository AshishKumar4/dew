"""Every script under tools/ type-checked by pyright in the environment it runs in (row 7933).

A tool runs in Dew's own environment unless it names the reference releases it records against: a module-level
`COHORTS` (tools/reference_releases.py's RELEASES names: one or a list, run in together, or a mapping of the
tool's variants to them, each variant run in its own), or a PEP 723 header, which `uv run` reads
(tools/superoptimize_bits.py). Each cohort set's tools are checked in a virtual environment holding exactly
its pins, built here by uv and kept under `ENVIRONMENTS` by its pins' digest, so a reference library's types
are its own release's: with its import unresolved, pyright reads its every use as Unknown and finds none of
the tool's errors. Cohorts run in together that pin one distribution at two releases make no one
environment, and are refused: the tool is split into a file a cohort.

A tool's verdict is kept by its key (`key`): pyright's release and configuration, the stubs, its environment's
pins and the code of every module it imports from the checkout, followed through their own imports
(`closure`). A key green before is not checked again: a commit that changes no tool and nothing a tool imports
runs no pyright here.

From the checkout's root, which `tools` is a package of (its siblings imported as `tools.<name>`):

    python -m tools.check_tool_types            # every tool, its verdict printed; exit 1 where any is red
    python -m tools.check_tool_types --list     # each cohort set's tools, checking nothing
    python -m tools.check_tool_types --environment dew    # one environment's tools alone
    python -m tools.check_tool_types tools/x.py tools/y.py  # these tools alone
    python -m tools.check_tool_types --shard 2/4    # the second of four shards (tools/check_gates' gates)

"""

import argparse
import ast
import functools
import hashlib
import json
import os
import re
import subprocess
import sys
import sysconfig
import tomllib
import types
from collections.abc import Iterable, Mapping
from pathlib import Path

from tools import lint_scopes as scopes

ROOT = Path(__file__).resolve().parents[1]
PYRIGHT = "pyright@1.1.406"
"""The release tools/check_gates' pyright gate runs, so both read types alike."""
ENVIRONMENTS = Path(".venv-tools")
"""Each cohort set's environment, `<digest>` under it, made once and read by every run after."""
DEW = "dew"
"""The environment of a tool that names no cohort: Dew's own, locks/cpu.txt's."""
STORE = "tools-types"
"""The records store's prefix for each green verdict (tools/armada/records.py)."""
TORCH_CPU = "https://download.pytorch.org/whl/cpu"
"""Where a cohort's torch comes from: its CPU build, which pyright reads as the CUDA build's types."""
PYTHON = "3.12"


def releases() -> Mapping[str, Mapping[str, str]]:
    """tools/reference_releases.py's RELEASES."""
    from tools.reference_releases import RELEASES

    return RELEASES


def cohorts_of(path: Path) -> tuple[tuple[str, ...], ...] | None:
    """The environments tool `path`'s module-level `COHORTS` names, each as its cohorts, sorted: one for a
    name or a list or tuple of them, which it runs in together; one a distinct value for a mapping, whose
    keys are the tool's variants, each run in its value's cohorts alone (tools/diffusers_dc_ae_reference.py
    records each variant under its own diffusers release). None where it names none."""
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
            return tuple(sorted({tuple(sorted({each} if isinstance(each, str) else set(each)))
                                 for each in values}))
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


def environments_of(path: Path, pins: Mapping[str, Mapping[str, str]]) -> list[tuple[str, list[str]]]:
    """Tool `path`'s environments (`cohorts_of`), each its name (`DEW`, `script:<file>`, or its cohorts joined
    by `+`) and the requirements it is made of (none for Dew's). Cohorts a tool runs in together that pin a
    distribution at two releases raise ValueError, naming them."""
    header = header_of(path)
    if header is not None:
        return [(f"script:{path.relative_to(ROOT).as_posix()}", sorted(header.get("dependencies", [])))]
    environments = cohorts_of(path)
    if not environments:
        return [(DEW, [])]
    if unknown := sorted({cohort for cohorts in environments for cohort in cohorts if cohort not in pins}):
        raise ValueError(f"{path}: COHORTS names {', '.join(unknown)}, which RELEASES does not")
    made = []
    for cohorts in environments:
        held: dict[str, str] = {}
        for cohort in cohorts:
            for distribution, release in pins[cohort].items():
                if held.setdefault(distribution, release) != release:
                    raise ValueError(f"{path}: its cohorts {', '.join(cohorts)} pin {distribution} at "
                                     f"{held[distribution]} and {release}; no one environment holds both: "
                                     "split it into a file a cohort")
        pinned = (f"{distribution} @ {release}" if release.startswith("git+")
                  else f"{distribution}=={release}" for distribution, release in held.items())
        made.append(("+".join(cohorts), sorted(pinned)))
    return made


def imported(path: Path) -> set[str]:
    """The dotted names tool or module `path` imports, absolute and its relative ones resolved, with each
    `from a import b` also as `a.b` (b may be a module): `lint_scopes.modules_imported`, Dew's one reader of
    them."""
    package = path.relative_to(ROOT).with_suffix("").parts[1:-1] if path.is_relative_to(ROOT) else ()
    return scopes.modules_imported(ast.parse(path.read_text(), str(path)), list(package))


def roots_of(path: Path) -> tuple[str, ...]:
    """Where an import of the file `path` is looked up, in pyright's order as `checked` configures it: its
    pyproject.toml execution environment's extraPaths where it sits under one's root (tools/armada's scripts
    import their neighbours by name), else the checkout itself (`import tools.x`, "") and then
    pyproject.toml's extraPaths."""
    for root, extra in environment_roots():
        if path.is_relative_to(root):
            return extra
    return ("", *settings().get("extraPaths", []))


@functools.cache
def settings() -> Mapping:
    """pyproject.toml's pyright settings, read once: every file of every closure asks (`roots_of`), and a read
    each took the gate past its deadline."""
    configuration = ROOT / "pyproject.toml"
    if not configuration.is_file():
        return {}
    return tomllib.loads(configuration.read_text()).get("tool", {}).get("pyright", {})


def environment_roots() -> tuple[tuple[Path, tuple[str, ...]], ...]:
    """pyproject.toml's pyright execution environments, each its root and extraPaths."""
    return tuple((ROOT / each["root"], tuple(each.get("extraPaths", [])))
                 for each in settings().get("executionEnvironments", []))


def located(name: str, roots: tuple[str, ...]) -> list[Path]:
    """The checkout's files an import of `name` executes: each package's __init__.py on the way and the
    module, looked up under `roots` in order, the first root holding its first part as a module or a
    package (a directory of no Python, such as tools/ci beside tools/armada's ci.py, is neither)."""
    parts = name.split(".")
    for root in roots:
        base = ROOT / root
        if not ((base / parts[0] / "__init__.py").is_file() or (base / f"{parts[0]}.py").is_file()
                or ((base / parts[0]).is_dir() and any((base / parts[0]).glob("*.py")))):
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
        here = frontier.pop()
        for name in imported(here):
            for found in located(name, roots_of(here)):
                if found not in seen:
                    seen.add(found)
                    frontier.append(found)
    return sorted(seen)


def common() -> bytes:
    """What every tool's key holds: pyright's release, its configuration in pyproject.toml, the stubs, and
    this gate's own code, which says how pyright reads them."""
    digest = hashlib.sha256(PYRIGHT.encode() + Path(__file__).read_bytes())
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pyright"]
    digest.update(json.dumps(configuration, sort_keys=True).encode())
    for stub in sorted((ROOT / "stubs").rglob("*")):
        if stub.is_file():
            digest.update(stub.relative_to(ROOT).as_posix().encode() + stub.read_bytes())
    return digest.digest()


def key(path: Path, requirements: Iterable[str], shared: bytes) -> str:
    """Tool `path`'s verdict's key: `shared` (`common`), its environment (its requirements, and Dew's lock,
    whose environment every one reads beneath its own: `beneath`) and the code of its `closure`, each file by
    its path and bytes."""
    digest = hashlib.sha256(shared)
    digest.update(json.dumps(list(requirements)).encode() + (ROOT / "locks" / "cpu.txt").read_bytes())
    for each in closure(path):
        digest.update(each.relative_to(ROOT).as_posix().encode() + each.read_bytes())
    return digest.hexdigest()


def tools() -> list[Path]:
    """Every Python file under tools/."""
    return sorted(path for path in (ROOT / "tools").rglob("*.py") if ".venv" not in path.parts)


SOURCE = re.compile(r"^(?P<name>[\w.-]+) @ git\+(?P<url>https://github\.com/[\w.-]+/[\w.-]+)"
                    r"@(?P<sha>[0-9a-f]{40})$")
"""A RELEASES pin that is a public repository at a commit (`name @ git+https://github.com/org/repo@<sha>`):
the reference a site runs from its checkout (tools/venv_reference.py's SITES), where no release of it is
published."""


def made(requirements: list[str]) -> Path:
    """The interpreter of the environment holding `requirements` over Dew's own (`beneath`), made
    once under `ENVIRONMENTS` by their digest (a torch from its CPU index, a CUDA extension's build skipped,
    as pyright reads only its Python); a failed install raises CalledProcessError. A repository pin (`SOURCE`)
    is never built: it is fetched at its commit beside the environment and put on its path by a .pth file,
    as its site puts the checkout on its worker's path, which pyright reads as the environment's own; its
    own dependencies are the cohort's other pins."""
    where = ROOT / ENVIRONMENTS / hashlib.sha256(json.dumps(requirements).encode()).hexdigest()[:16]
    python = where / "bin" / "python"
    if (where / "made").is_file():
        return python
    subprocess.run(["uv", "venv", "--quiet", "--clear", "--python", PYTHON, str(where)], check=True)
    released = [each for each in requirements if not SOURCE.match(each)]
    if released:
        # The pins are the versions the reference's environment held, over what a release of them declares
        # (Kimi K3's: compressed-tensors 0.17.1, which declares torch>=2.10, beside torch 2.8.0).
        (where / "pins.txt").write_text("".join(f"{each}\n" for each in released))
        subprocess.run(["uv", "pip", "install", "--quiet", "--python", str(python), "--extra-index-url",
                        TORCH_CPU, "--index-strategy", "unsafe-best-match", "--override",
                        str(where / "pins.txt"), *released], check=True, env={**os.environ, **SKIPPED_BUILDS})
    purelib = "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    site = subprocess.run([str(python), "-c", purelib], capture_output=True, text=True,
                          check=True).stdout.strip()
    # One .pth, read after the environment's own packages, its lines in order: each repository pin's checkout
    # (a src layout's packages under its src/, as an editable install puts them on the path), then Dew's
    # environment beneath them all (`beneath`).
    paths = []
    for source in filter(None, map(SOURCE.match, requirements)):
        checkout = where / "sources" / source["name"]
        subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "fetch", "--quiet", "--depth=1", source["url"],
                        source["sha"]], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--quiet", "FETCH_HEAD"], check=True)
        paths.append(checkout / "src" if (checkout / "src").is_dir() else checkout)
    paths.append(beneath(Path(site), where / "beneath"))
    Path(site, "_tool_types.pth").write_text("".join(f"{path}\n" for path in paths))
    (where / "made").write_text(json.dumps(requirements))
    return python


def beneath(site: Path, where: Path) -> Path:
    """Dew's environment as one directory of links, `where`, but for what the environment at `site` holds
    itself: a reference tool runs Dew's side and the reference's in one file, so its environment reads Dew's
    beneath its own pins, and a distribution both hold is the pin's alone, which pyright, preferring a typed
    release (py.typed) wherever two are on its path, would not keep (2026-11-19: Dew's transformers hid
    diffusers-pipelines' 4.49.0, and its Flax CLIP models with it)."""
    where.mkdir(exist_ok=True)
    own = {entry.name.partition("-")[0].lower() for entry in site.iterdir()}
    for entry in Path(sysconfig.get_paths()["purelib"]).iterdir():
        if entry.suffix != ".pth" and entry.name.partition("-")[0].lower() not in own:
            (where / entry.name).symlink_to(entry)
    return where


SKIPPED_BUILDS = {"MAMBA_SKIP_CUDA_BUILD": "TRUE", "CAUSAL_CONV1D_SKIP_CUDA_BUILD": "TRUE"}
"""A release that compiles CUDA kernels at install, installed without them: pyright reads its Python alone."""


def pyright_report(root: Path, overrides: dict, paths: list[str], written: Path,
                   arguments: tuple[str, ...] = ()) -> list[dict]:
    """Pyright's diagnostics for `paths` of checkout `root`: `root`'s pyproject.toml configuration, every path
    in it made absolute, since pyright reads them from the configuration's own directory (`written`, where it
    is written), the checkout first among its import roots (read in place, the configuration's directory is
    the root `import tools.x` resolves from), with `overrides` over it, run from `root` with `arguments`."""
    configuration = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["pyright"]
    configuration = {**{name: value for name, value in configuration.items() if name != "include"},
                     "stubPath": str(root / configuration["stubPath"]),
                     "extraPaths": [str(root), *(str(root / path) for path in configuration["extraPaths"])],
                     "executionEnvironments": [
                         {**each, "root": str(root / each["root"]),
                          "extraPaths": [str(root / path) for path in each.get("extraPaths", [])]}
                         for each in configuration.get("executionEnvironments", [])],
                     **overrides}
    written.write_text(json.dumps(configuration))
    ran = subprocess.run(["uvx", PYRIGHT, "--project", str(written), "--outputjson", *arguments, *paths],
                         cwd=root, capture_output=True, text=True,
                         env={**os.environ, "NODE_OPTIONS": "--max-old-space-size=8192"}, check=False)
    report = json.loads(ran.stdout or "{}")
    if "generalDiagnostics" not in report:
        raise SystemExit(f"pyright gave no report: {ran.stderr[-2000:]}")
    return report["generalDiagnostics"]


def checked(environment: Path, paths: list[Path]) -> dict[Path, list[str]]:
    """Each of `paths` with pyright's errors in it (`pyright_report`), its imports resolved in the virtual
    environment `environment`, named in the configuration (a configuration's `venv` outranks
    `--pythonpath`, so a reference tool read Dew's environment, its libraries unresolved)."""
    (ROOT / ENVIRONMENTS).mkdir(exist_ok=True)
    diagnostics = pyright_report(ROOT, {"venvPath": str(environment.parent), "venv": environment.name},
                                 [str(path) for path in paths],
                                 ROOT / ENVIRONMENTS / f"pyright-{environment.name}.json")
    found: dict[Path, list[str]] = {path: [] for path in paths}
    for diagnostic in diagnostics:
        if diagnostic["severity"] == "error":
            at, line = Path(diagnostic["file"]), diagnostic["range"]["start"]["line"] + 1
            said = f"{diagnostic['message'].splitlines()[0]} ({diagnostic.get('rule', '')})"
            # A file outside the checkout is named whole (an interpreter's library pyright read through a
            # tool), red against the tools it was checked with.
            named = at.relative_to(ROOT) if at.is_relative_to(ROOT) else at
            found.setdefault(at, []).append(f"{named}:{line}: {said}")
    return found


def records() -> types.ModuleType | None:
    """tools/armada/records.py, the CI store the verdicts are kept in, or None where this run has no token."""
    if not os.environ.get("DEW_XLA_CACHE_TOKEN") or not os.environ.get("DEW_XLA_CACHE_URL"):
        return None
    from tools.armada import records

    return records


def shard_of(spelled: str) -> tuple[int, int]:
    """`K/N` as the zero-based shard index and the shard count."""
    index, _, count = spelled.partition("/")
    if not (index.isdigit() and count.isdigit() and 1 <= int(index) <= int(count)):
        raise argparse.ArgumentTypeError(f"{spelled} is no K/N with 1 <= K <= N")
    return int(index) - 1, int(count)


def sharded(groups: list[tuple[tuple[str, tuple[str, ...]], list[Path]]], index: int, count: int
            ) -> list[tuple[tuple[str, tuple[str, ...]], list[Path]]]:
    """Shard `index` of `count` of the environments' tools: every `count`-th of Dew's own (`DEW`), whose one
    pyright run over all of them outlasts a CI gate (1713 s on armada, 2026-11-19), and every `count`-th other
    environment whole, as each is built for its tools alone."""
    dew = [(group, paths[index::count]) for group, paths in groups if group[0] == DEW]
    others = [entry for entry in groups if entry[0][0] != DEW][index::count]
    return [entry for entry in dew + others if entry[1]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="each cohort set's tools, checking nothing")
    parser.add_argument("--environment", action="append", default=[],
                        help="check only this environment's tools (a name --list prints); repeatable")
    parser.add_argument("--shard", type=shard_of, default=(0, 1), metavar="K/N",
                        help="check the K-th of N shards (`sharded`), one CI gate each; K counts from 1")
    parser.add_argument("paths", nargs="*", type=Path, help="check only these tools")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        # Dew's own tools are checked in this interpreter's environment, which a bare interpreter is not
        # (2026-11-19: a system python3.14 read every Dew tool against its standard library alone).
        parser.error(f"{sys.executable} is no virtual environment: run this with Dew's "
                     "(.venv-3.12/bin/python)")
    pins, shared, store = releases(), common(), records()
    groups: dict[tuple[str, tuple[str, ...]], list[Path]] = {}
    red = 0
    named = {(ROOT / path).resolve() for path in args.paths}
    for path in tools():
        if named and path.resolve() not in named:
            continue
        try:
            environments = environments_of(path, pins)
        except ValueError as refused:
            print(f"refused: {refused}")
            red += 1
            continue
        for name, requirements in environments:
            groups.setdefault((name, tuple(requirements)), []).append(path)
    if args.list:
        for (name, _), paths in sorted(groups.items()):
            print(f"{name}: {len(paths)} tools: {' '.join(str(path.relative_to(ROOT)) for path in paths)}")
        return 0
    token = os.environ.get("DEW_XLA_CACHE_TOKEN", "")
    for (name, requirements), paths in sharded(sorted(groups.items()), *args.shard):
        if args.environment and name not in args.environment:
            continue
        keys = {path: key(path, requirements, shared) for path in paths}
        wanted = [path for path in paths
                  if store is None or store.recorded(f"{STORE}/{keys[path]}", token) is None]
        print(f"{name}: {len(paths)} tools, {len(paths) - len(wanted)} green before", flush=True)
        if not wanted:
            continue
        try:
            environment = made(list(requirements)).parents[1] if requirements else Path(sys.prefix)
        except subprocess.CalledProcessError as failed:
            print(f"red: {name}'s environment ({' '.join(requirements)}) was not made: {failed}")
            red += len(wanted)
            continue
        for path, errors in checked(environment, wanted).items():
            if errors:
                red += 1
                print("\n".join(errors))
            elif store is not None and path in keys:
                store.written(f"{STORE}/{keys[path]}", token, b"{}")
    print(f"tool types: {red} tool(s) red" if red else "tool types: green")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
