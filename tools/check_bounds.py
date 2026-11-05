#!/usr/bin/env python3
"""Dew at the bounds of its declared dependency ranges, in clean environments (row 862).

    python tools/check_bounds.py declared
    python tools/check_bounds.py matrix
    python tools/check_bounds.py run {lowest-direct,highest} dist/dewml-*.whl [--extra NAME] [--python 3.12]
    python tools/check_bounds.py resolve {lowest-direct,highest} --extra NAME

`declared` is the ranges gate (tools/check_gates): every requirement the library installs, the base's and
each runtime extra's, states a lower bound or an exact pin, so a lowest set means something. It reads
pyproject.toml alone.

`matrix` prints the legs CI runs (.github/workflows/ci.yml's `bounds` job) from the same declarations: the
base requirements at both bounds, each runtime extra that carries a range at both, and each extra whose
package needs hardware the runner has not, resolved at both. `[tool.dew.bounds]` in pyproject.toml names
the extras that are tooling and those that need hardware.

`run` is one leg. It resolves pyproject.toml's requirements, the base's and `--extra`'s, under constraints.txt
(the fork's one jax, and Dew's index) with uv at the lowest compatible version of each direct requirement and
the highest of everything else (`lowest-direct`, the strategy uv's documentation recommends a library's CI
run), or at the highest of all; installs that set and the built wheel, with nothing else, into a new virtual
environment; and runs tools/bounds_workflow.py there from outside the checkout. A result is the resolver's
compatible solution, not each package at its own bound: the summary says what each direct requirement resolved
to and, where that is above the floor it declares, that this environment resolved it so.

`resolve` resolves an accelerator extra at one bound for each Python tested and each machine pyproject.toml
names for it, and installs nothing: the plugins load only on their hardware, where the suite runs them, but a
floor with no wheel for a supported Python or machine fails here, at the declaration, as it would at a
user's install.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"
RESOLUTIONS = ("lowest-direct", "highest")
FLOOR = {">=", ">", "~=", "==", "==="}
"""The operators that state a lower bound: a floor, a compatible release's, or an exact pin."""
DISTRIBUTION = "dewml"


@dataclasses.dataclass(frozen=True)
class Declared:
    """What pyproject.toml declares about the library's requirements."""

    base: tuple[Requirement, ...]
    extras: dict[str, tuple[Requirement, ...]]
    """Every extra's requirements but its references to the library's own other extras."""
    tooling: frozenset[str]
    """The extras the library does not run on: the environments of its tests, tools, examples and site."""
    resolve_only: dict[str, tuple[str, ...]]
    """The extras whose packages need hardware the runner has not, by the machines each installs on."""

    @property
    def runtime(self) -> dict[str, tuple[Requirement, ...]]:
        return {name: found for name, found in self.extras.items() if name not in self.tooling}


def declared(path: Path = PYPROJECT) -> Declared:
    """`path`'s requirements, parsed by packaging, and the `[tool.dew.bounds]` table that classifies its
    extras."""
    document = tomllib.loads(path.read_text())
    project, bounds = document["project"], document["tool"]["dew"]["bounds"]

    def parsed(lines: list[str]) -> tuple[Requirement, ...]:
        found = (Requirement(line) for line in lines)
        return tuple(item for item in found if canonicalize_name(item.name) != DISTRIBUTION)

    extras = {name: parsed(lines) for name, lines in project["optional-dependencies"].items()}
    tooling = frozenset(bounds["tooling"])
    resolve_only = {name: tuple(machines) for name, machines in bounds["resolve-only"].items()}
    named = tooling | resolve_only.keys()
    if named - extras.keys() or tooling & resolve_only.keys():
        raise SystemExit("[tool.dew.bounds] names extras pyproject.toml does not declare, or classifies one "
                         f"twice: {sorted((named - extras.keys()) | (tooling & resolve_only.keys()))}")
    return Declared(parsed(project["dependencies"]), extras, tooling, resolve_only)


def floored(requirement: Requirement) -> bool:
    """Whether `requirement` states a lower bound: `>=`, `>`, `~=`, or `==` (exact or `.*`)."""
    return any(specifier.operator in FLOOR for specifier in requirement.specifier)


def pinned(requirement: Requirement) -> bool:
    """Whether `requirement` admits one release: only `==` of a version with no wildcard."""
    return bool(requirement.specifier) and all(
        specifier.operator in ("==", "===") and not specifier.version.endswith(".*")
        for specifier in requirement.specifier)


def unfloored(found: Declared) -> list[str]:
    """The requirements of the base and of every runtime extra that state no lower bound, each with its
    home."""
    homes = [("dependencies", item) for item in found.base]
    homes += [(f"extra {name}", item) for name, group in found.runtime.items() for item in group]
    return [f"{home}: {item}" for home, item in homes if not floored(item)]


def pythons() -> list[str]:
    """The Python versions tested, oldest first: tools/armada/ci.py's, from the classifiers, the one owner."""
    done = subprocess.run([sys.executable, str(REPO / "tools/armada/ci.py"), "pythons"], capture_output=True,
                          text=True, check=True, cwd=REPO)
    return done.stdout.split()


def legs(found: Declared) -> list[dict[str, str]]:
    """Each leg CI runs, for GitHub's matrix: its name, mode (run or resolve), resolution, Python and extra.
    The base and each runtime extra with a range (one pinned to a release is the suite's, at its one) run at
    the oldest Python for their lowest-direct set, whose old releases have no wheels for the newest, and at
    the newest for their highest; each accelerator extra is resolved at both bounds."""
    tested = pythons()
    chosen = {"lowest-direct": tested[0], "highest": tested[-1]}
    ranged = [name for name, group in sorted(found.runtime.items()) if name not in found.resolve_only
              and any(floored(item) and not pinned(item) for item in group)]
    made = [{"name": f"{resolution} {extra or 'base'}, {python}", "mode": "run", "resolution": resolution,
             "python": python, "extra": extra}
            for extra in ("", *ranged) for resolution, python in chosen.items()]
    made += [{"name": f"resolve {name}, {resolution}", "mode": "resolve", "resolution": resolution,
              "python": "", "extra": name}
             for name in sorted(found.resolve_only) for resolution in RESOLUTIONS]
    return made


def uv(*arguments: str | Path, cwd: Path = REPO, phase: str) -> None:
    """uv run to its end with its output on this one's, `phase` named where it fails."""
    print("+ uv", *arguments, flush=True)
    done = subprocess.run(["uv", *map(str, arguments)], cwd=cwd, text=True, check=False)
    if done.returncode:
        raise SystemExit(f"{phase} failed: uv {' '.join(map(str, arguments[:3]))} exited {done.returncode}")


def resolved(compiled: Path) -> dict[str, Version]:
    """Each package a compiled requirements file pins, by canonical name."""
    found = {}
    for line in compiled.read_text().splitlines():
        if matched := re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", line):
            found[canonicalize_name(matched.group(1))] = Version(matched.group(2))
    return found


def floor_of(requirement: Requirement) -> Version | None:
    """The release `requirement` names as its lower bound, where it states one."""
    versions = [Version(specifier.version.removesuffix(".*")) for specifier in requirement.specifier
                if specifier.operator in FLOOR]
    return max(versions, default=None)


def table(requirements: tuple[Requirement, ...], compiled: dict[str, Version]) -> list[str]:
    """Each direct requirement, its declared floor, what the resolution gave it and, where that is above the
    floor, an observation of this environment: another package's requirement, or this index's, moved it."""
    lines = ["| requirement | floor | resolved | observation |", "|---|---|---|---|"]
    for requirement in sorted(requirements, key=lambda item: canonicalize_name(item.name)):
        floor, got = floor_of(requirement), compiled.get(canonicalize_name(requirement.name))
        note = ""
        if got is None:
            note = "not installed here (its marker is false)"
        elif floor is not None and got > floor:
            note = "resolved above its declared floor in this environment"
        lines.append(f"| {requirement.name} | {floor or ''} | {got or ''} | {note} |")
    return lines


def slug(name: str) -> str:
    """`name` as a file name."""
    return re.sub(r"[^A-Za-z0-9.]+", "-", name).strip("-")


def identity(wheel: Path, interpreter: Path) -> list[str]:
    """What an execution is, beyond its pins: the wheel, uv, the interpreter's patch version, the platform."""
    version = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True).stdout.strip()
    python = subprocess.run([str(interpreter), "-c", "import sys; print(sys.version.replace(chr(10), ' '))"],
                            capture_output=True, text=True, check=True).stdout.strip()
    return [f"- wheel: {wheel.name} sha256 {hashlib.sha256(wheel.read_bytes()).hexdigest()}", f"- {version}",
            f"- Python {python}", f"- platform: {platform.platform()} {platform.machine()}"]


def attributed(outcome: dict, compiled: dict[str, Version], direct: dict[str, Requirement]) -> str:
    """A failed step's report: its traceback, and the distributions its frames were in, each marked where it
    is a direct requirement of this leg resolved at its floor. Where the frames point is a lead, not a
    cause."""
    lines = [f"### FAILED {outcome['name']}", "", "```", outcome["error"].rstrip(), "```", ""]
    if outcome["distributions"]:
        lines.append("Distributions of the failing frames (a lead, not a cause):")
        for name in outcome["distributions"]:
            key = canonicalize_name(name.split("==")[0])
            floor = None if key not in direct else floor_of(direct[key])
            at = " (a direct requirement, at its floor)" if floor is not None and compiled.get(key) == floor \
                else ""
            lines.append(f"- {name}{at}")
    return "\n".join(lines)


def run(resolution: str, wheel: Path, extras: list[str], python: str, output: Path) -> int:
    """One leg: resolve, install, run the workflow, write the summary into `output`; 0 if every step
    passed."""
    found = declared()
    unknown = [extra for extra in extras if extra not in found.runtime or extra in found.resolve_only]
    if unknown:
        raise SystemExit(f"{unknown} are not runtime extras that run here: tooling and accelerator extras "
                         "do not")
    leg = slug(f"{resolution}-{'-'.join(extras) or 'base'}-{python}")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dew-bounds-") as directory:
        scratch = Path(directory)
        environment, compiled_file = scratch / "env", output / f"bounds-{leg}.txt"
        interpreter = environment / "bin" / "python"
        uv("venv", "--python", python, environment, phase="environment")
        selected = [flag for extra in extras for flag in ("--extra", extra)]
        # The index lines go into the file (`--emit-index-url`): an install from it alone finds Dew's jax.
        uv("pip", "compile", "pyproject.toml", "-c", "constraints.txt", *selected, "--resolution", resolution,
           "--python", interpreter, "--torch-backend", "cpu", "--no-build", "--no-header", "--emit-index-url",
           "-o", compiled_file, phase="resolve")
        uv("pip", "install", "--python", interpreter, "--torch-backend", "cpu", "--no-build", "-r",
           compiled_file, cwd=scratch, phase="install")
        uv("pip", "install", "--python", interpreter, "--no-deps", wheel, cwd=scratch, phase="install")
        report = output / f"bounds-{leg}.json"
        command = [str(interpreter), "-P", str(REPO / "tools/bounds_workflow.py"), "--scratch",
                   str(scratch / "work"), "--report", str(report), *selected]
        print("+", *command, flush=True)
        status = subprocess.run(command, cwd=scratch, check=False).returncode
        reported = json.loads(report.read_text()) if report.is_file() else []
        compiled = resolved(compiled_file)
        requirements = found.base + tuple(item for extra in extras for item in found.extras[extra])
        summary = [f"## bounds: {leg}", "", *identity(wheel, interpreter),
                   f"- resolved set: bounds-{leg}.txt", "", *table(requirements, compiled), ""]
        direct = {canonicalize_name(item.name): item for item in requirements}
        summary += [attributed(outcome, compiled, direct) for outcome in reported if not outcome["passed"]]
        if status and not reported:
            summary.append(f"The workflow exited {status} before it reported a step.")
    text = "\n".join(summary) + "\n"
    (output / f"bounds-{leg}.md").write_text(text)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as target:
            target.write(text)
    return 1 if status else 0


def resolve(resolution: str, extra: str) -> int:
    """`extra` resolved at one bound, for each Python tested and each machine pyproject.toml names for it."""
    found = declared()
    if extra not in found.resolve_only:
        raise SystemExit(f"{extra} is not an accelerator extra [tool.dew.bounds] resolves: "
                         f"{sorted(found.resolve_only)}")
    for python in pythons():
        for machine in found.resolve_only[extra]:
            uv("pip", "compile", "pyproject.toml", "-c", "constraints.txt", "--extra", extra, "--resolution",
               resolution, "--python-version", python, "--python-platform", f"{machine}-manylinux_2_35",
               "--torch-backend", "cpu", "--no-build", "--no-header", "--emit-index-url", "-o", os.devnull,
               phase=f"resolve {extra} on Python {python}, {machine}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("declared", help="fail a runtime requirement that states no lower bound")
    commands.add_parser("matrix", help="print the legs CI runs, as GitHub's matrix")
    running = commands.add_parser("run", help="one leg: resolve, install into a clean environment, run")
    running.add_argument("resolution", choices=RESOLUTIONS)
    running.add_argument("wheel", type=Path)
    running.add_argument("--extra", action="append", default=[])
    running.add_argument("--python", help="the environment's Python: the oldest tested for lowest-direct, "
                         "the newest for highest")
    running.add_argument("--output", type=Path, default=Path("bounds"),
                         help="where the resolved set, the report and the summary are written")
    resolving = commands.add_parser("resolve", help="an accelerator extra resolved at one bound")
    resolving.add_argument("resolution", choices=RESOLUTIONS)
    resolving.add_argument("--extra", required=True)
    args = parser.parse_args()
    if args.command == "declared":
        problems = unfloored(declared())
        for problem in problems:
            print(f"states no lower bound: {problem}", file=sys.stderr)
        return 1 if problems else 0
    if args.command == "matrix":
        print(json.dumps({"include": legs(declared())}))
        return 0
    if args.command == "resolve":
        return resolve(args.resolution, args.extra)
    chosen = pythons()
    python = args.python or (chosen[0] if args.resolution == "lowest-direct" else chosen[-1])
    return run(args.resolution, args.wheel.resolve(), args.extra, python, args.output.resolve())


if __name__ == "__main__":
    sys.exit(main())
