"""Generate the install facts other files repeat from their owners: the CUDA builds' minima
(dew.telemetry.builds.CUDA_BUILDS), Dew's package index (dew.telemetry.builds.PACKAGE_INDEX) and
Dew's least Python (pyproject.toml's requires-python), into the standalone installer, constraints.txt,
the Colab notebooks' install, docs/installation.md's table and the least Python the README and
docs/installation.md state."""

import argparse
import ast
import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHELL = ("# BEGIN install facts (site/scripts/gen_install.py)", "# END install facts")
TABLE = ("<!-- BEGIN CUDA builds (site/scripts/gen_install.py) -->", "<!-- END CUDA builds -->")
LEAST = ("<!-- BEGIN least Python (site/scripts/gen_install.py) -->", "<!-- END least Python -->")
CONSTRAINT = ("# BEGIN package index (site/scripts/gen_install.py)", "# END package index")
COLAB = ("// BEGIN package index (site/scripts/gen_install.py)", "// END package index")


def facts() -> tuple[list[tuple[str, int, tuple[int, int]]], str, str]:
    """The CUDA builds, the package index and the least Python, read as literals so the installer
    stays independent of Dew's runtime imports."""
    source = ast.parse((REPO / "src/dew/telemetry/builds.py").read_text())

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


def between(text: str, markers: tuple[str, str], lines: list[str]) -> str:
    """`text` with the lines between its `markers` replaced by `lines`."""
    before, block = text.split(markers[0] + "\n")
    _, after = block.split(markers[1])
    return before + "\n".join([markers[0], *lines, ""]) + markers[1] + after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Check the generated blocks without writing them.")
    args = parser.parse_args()
    builds, index, python = facts()
    shell = [f"python_min={python}", f"package_index={index}"]
    table = ["| Build | CUDA | NVIDIA driver | GPU (SM) |", "|---|---|---|---|"]
    for build, driver, (major, minor), cuda in builds:
        shell.extend((f"{build}_driver={driver}", f"{build}_sm={major}.{minor}"))
        table.append(f"| `{build}` | {cuda // 1000}.{cuda % 1000 // 10} or later | {driver} or later, or an older "
                     f"one under NVIDIA's cuda-compat | {major}.{minor} or later |")
    least = [f"Dew needs Python {python} or newer."]
    stale = []
    for path, blocks in ((REPO / "site/public/install.sh", ((SHELL, shell),)),
                         (REPO / "constraints.txt", ((CONSTRAINT, [f"--extra-index-url {index}"]),)),
                         (REPO / "site/scripts/colab-notebooks.mjs",
                          ((COLAB, [f"const PACKAGE_INDEX = '{index}';"]),)),
                         (REPO / "docs/installation.md", ((TABLE, table), (LEAST, least))),
                         (REPO / "README.md", ((LEAST, least),))):
        original = generated = path.read_text()
        for markers, lines in blocks:
            generated = between(generated, markers, lines)
        if generated != original:
            stale.append(str(path.relative_to(REPO)))
            if not args.check:
                path.write_text(generated)
    if args.check and stale:
        raise SystemExit(f"gen_install: run python site/scripts/gen_install.py to update {', '.join(stale)}")
    print(f"gen_install: {len(builds)} CUDA builds, Python {python}")


if __name__ == "__main__":
    main()
