"""Generate the installer's CUDA minima from dew.telemetry.builds.CUDA_BUILDS."""

import argparse
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BEGIN = "# BEGIN CUDA minima (site/scripts/gen_install.py)"
END = "# END CUDA minima"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the generated block without writing it.")
    args = parser.parse_args()
    # Reading the literal keeps this standalone installer independent of Dew's runtime imports.
    source = ast.parse((REPO / "src/dew/telemetry/builds.py").read_text())
    builds = next(ast.literal_eval(node.value) for node in source.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(name, ast.Name) and name.id == "CUDA_BUILDS" for name in node.targets))
    lines = [BEGIN]
    for build, driver, (major, minor), _ in builds:
        lines.extend((f"{build}_driver={driver}", f"{build}_sm={major}.{minor}"))
    lines.append(END)
    path = REPO / "site/public/install.sh"
    original = path.read_text()
    before, block = original.split(BEGIN + "\n")
    _, after = block.split(END)
    generated = before + "\n".join(lines) + after
    if args.check:
        if generated != original:
            raise SystemExit("gen_install: run python site/scripts/gen_install.py to update the CUDA minima")
    else:
        path.write_text(generated)
    print(f"gen_install: {len(builds)} CUDA builds")


if __name__ == "__main__":
    main()
