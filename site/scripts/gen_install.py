"""Generate the install facts other files repeat from their owners: the CUDA builds' minima
(dew.builds.CUDA_BUILDS), Dew's package index (dew.builds.PACKAGE_INDEX), Dew's least
Python (pyproject.toml's requires-python), the JAX it requires (its dependencies), the pip every pip install
takes first (tools/pip.txt) and the least uv (pyproject.toml's [tool.uv] required-version), with the
bootstrap that installs it (site/scripts/uv_bootstrap.sh), into the standalone installer, armada's setup, dew
tpu setup's script, constraints.txt, the Colab notebooks' install, the site's scripts that install Dew
(site/package.json), docs/installation.md's table and JAX requirement, and the least Python and the pip the
README and docs/installation.md state; and holds docs/installation.md's extras table to the extras
pyproject.toml declares."""

import argparse
import ast
import json
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "armada"))
from ci import PYTHONS  # the tested Pythons, pyproject.toml's classifiers'

SHELL = ("# BEGIN install facts (site/scripts/gen_install.py)", "# END install facts")
TABLE = ("<!-- BEGIN CUDA builds (site/scripts/gen_install.py) -->", "<!-- END CUDA builds -->")
LEAST = ("<!-- BEGIN least Python (site/scripts/gen_install.py) -->", "<!-- END least Python -->")
CONSTRAINT = ("# BEGIN package index (site/scripts/gen_install.py)", "# END package index")
COLAB = ("// BEGIN package index (site/scripts/gen_install.py)", "// END package index")
PIP = ("<!-- BEGIN pip (site/scripts/gen_install.py) -->", "<!-- END pip -->")
JAX = ("<!-- BEGIN JAX requirement (site/scripts/gen_install.py) -->", "<!-- END JAX requirement -->")
COMMANDS = ("<!-- BEGIN install commands (site/scripts/gen_install.py) -->", "<!-- END install commands -->")
UV = ("# BEGIN uv bootstrap (site/scripts/gen_install.py)", "# END uv bootstrap")
EXTRAS = ("<!-- BEGIN extras (site/scripts/gen_install.py checks them against pyproject.toml) -->",
          "<!-- END extras -->")


def facts() -> tuple[list[tuple[str, int, tuple[int, int]]], str, str]:
    """The CUDA builds, the package index and the least Python, read as literals so the installer
    stays independent of Dew's runtime imports."""
    source = ast.parse((REPO / "src/dew/builds.py").read_text())

    def literal(name: str):
        return next(ast.literal_eval(node.value) for node in source.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name
                            for target in node.targets))

    builds, index = literal("CUDA_BUILDS"), literal("PACKAGE_INDEX")
    required = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["requires-python"]
    least = re.fullmatch(r">=\s*(\d+\.\d+)", required)
    if least is None:
        raise SystemExit(f"gen_install reads a requires-python of the form >=X.Y, not {required!r}")
    return builds, index, least[1]


def documented_extras(text: str) -> set[str]:
    """The extras docs/installation.md's table names: each backquoted name in a row's first cell, between
    the `EXTRAS` markers. The table's descriptions are the page's own; its names are pyproject.toml's."""
    rows = text.split(EXTRAS[0] + "\n")[1].split(EXTRAS[1])[0].splitlines()[2:]
    return {name for row in rows for name in re.findall(r"`([^`]+)`", row.split("|")[1])}


def commands(builds: list[tuple[str, int, tuple[int, int]]], index: str) -> list[str]:
    """The install table: each build's uv and pip command, a CUDA build's row naming its least driver."""
    rows = [("CPU", "dewml"),
            *((f"NVIDIA GPU, driver {driver}+ (CUDA {build.removeprefix('cuda')})", f'"dewml[{build}]"')
              for build, driver, _, _ in builds),
            ("Google TPU VM", '"dewml[tpu]"')]
    return ["| Hardware | uv | pip |", "|---|---|---|",
            *(f"| {hardware} | `uv pip install {package} --extra-index-url {index}` | "
              f"`pip install {package} --extra-index-url {index}` |" for hardware, package in rows)]


def named(text: str, index: str, newest: str) -> str:
    """`text`, a page, with every install it shows naming Dew's package index, and every environment it makes
    for Dew (`uv venv --python X` alone on its line) the newest Python CI tests."""
    text = re.sub(r"--extra-index-url [^\s`'\"]+", f"--extra-index-url {index}", text)
    return re.sub(r"^uv venv --python [\d.]+$", f"uv venv --python {newest}", text, flags=re.MULTILINE)


def between(text: str, markers: tuple[str, str], lines: list[str]) -> str:
    """`text` with the lines between its `markers` replaced by `lines`."""
    before, block = text.split(markers[0] + "\n")
    _, after = block.split(markers[1])
    return before + "\n".join([markers[0], *lines, ""]) + markers[1] + after


def indexed(text: str, index: str) -> str:
    """`text`, a package.json, with every `uv run` of its scripts that installs Dew from the checkout
    (`--with-editable`) reading Dew's package index, where Dew's JAX build is served."""
    for script in json.loads(text)["scripts"].values():
        commands = []
        for command in script.split(" && "):
            if "uv run" in command and "--with-editable" in command:
                command = re.sub(r" --extra-index-url \S+", "", command)
                command = re.sub(r"(--with-editable \S+)",
                                 lambda found: f"{found[1]} --extra-index-url {index}", command)
            commands.append(command)
        before = json.dumps(script, ensure_ascii=False)
        after = json.dumps(" && ".join(commands), ensure_ascii=False)
        if text.count(before) != 1:
            raise SystemExit(f"gen_install finds the script {before} {text.count(before)} times in "
                             f"site/package.json")
        text = text.replace(before, after)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Check the generated blocks without writing them.")
    args = parser.parse_args()
    builds, index, python = facts()
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    project = pyproject["project"]
    pip = next(line for line in (REPO / "tools/pip.txt").read_text().splitlines()
               if line and not line.startswith("#"))
    pip_version = re.search(r"/pip-([\d.]+)-", pip)[1]
    uv = re.fullmatch(r">=\s*([\d.]+)", pyproject["tool"]["uv"]["required-version"])[1]
    # The bootstrap each installer runs before any checkout exists to read pyproject.toml from.
    bootstrap = [f"uv_least={uv}", *(REPO / "site/scripts/uv_bootstrap.sh").read_text().splitlines()]
    jax = next(requirement for requirement in project["dependencies"] if re.match(r"jax\b", requirement))
    page = (REPO / "docs/installation.md").read_text()
    extras = set(project["optional-dependencies"])
    if documented_extras(page) != extras:
        raise SystemExit(f"gen_install: docs/installation.md's extras table lacks "
                         f"{sorted(extras - documented_extras(page))} and names "
                         f"{sorted(documented_extras(page) - extras)}, which pyproject.toml does not declare")
    shell = [f"python_min={python}", f"package_index={index}", f"pip_least='{pip}'",
             "cuda_builds='" + " ".join(f"{build}:{driver}:{major}.{minor}"
                                        for build, driver, (major, minor), _ in builds) + "'"]
    table = ["| Build | CUDA | NVIDIA driver | GPU (SM) |", "|---|---|---|---|"]
    for build, driver, (major, minor), cuda in builds:
        table.append(f"| `{build}` | {cuda // 1000}.{cuda % 1000 // 10} or later | {driver} or later, or an older "
                     f"one under NVIDIA's cuda-compat | {major}.{minor} or later |")
    least = [f"Dew needs Python {python} or newer, and CI tests {' and '.join(PYTHONS)}."]
    advice = [f"Use pip {pip_version} or newer and uv {uv} or newer: older ones trust the roots of "
              "eMudhra's compromised certificate authorities (October 2026). Update pip from its wheel by "
              "hash, so the update holds whatever the older pip trusts, and uv through the system's "
              "certificates once they are updated:", "",
              "```bash", f'python -m pip install "{pip}"', "UV_SYSTEM_CERTS=true uv self update", "```"]
    stale = []
    for path, blocks in ((REPO / "site/public/install.sh", ((SHELL, shell), (UV, bootstrap))),
                         (REPO / "tools/armada/setup.sh", ((UV, bootstrap),)),
                         (REPO / "src/dew/cli/tpu_setup.py", ((UV, bootstrap),)),
                         (REPO / "constraints.txt", ((CONSTRAINT, [f"--extra-index-url {index}"]),)),
                         (REPO / "site/scripts/colab-notebooks.mjs",
                          ((COLAB, [f"const PACKAGE_INDEX = '{index}';"]),)),
                         (REPO / "site/package.json", ()),
                         (REPO / "docs/installation.md", ((TABLE, table), (LEAST, least), (PIP, advice),
                                                           (JAX, [f"Dew requires `{jax}`."]),
                                                           (COMMANDS, commands(builds, index)))),
                         (REPO / "README.md", ((LEAST, least), (PIP, advice),
                                               (COMMANDS, commands(builds, index))))):
        original = generated = path.read_text()
        for markers, lines in blocks:
            generated = between(generated, markers, lines)
        if path.name == "package.json":
            generated = indexed(generated, index)
        if path.suffix == ".md":
            generated = named(generated, index, PYTHONS[-1])
        if generated != original:
            stale.append(str(path.relative_to(REPO)))
            if not args.check:
                path.write_text(generated)
    if args.check and stale:
        raise SystemExit(f"gen_install: run python site/scripts/gen_install.py to update {', '.join(stale)}")
    print(f"gen_install: {len(builds)} CUDA builds, Python {python}")


if __name__ == "__main__":
    main()
