"""site/scripts/gen_install.py writes the install facts their owners declare into the files that repeat them,
and holds docs/installation.md's extras table to the extras pyproject.toml declares: on a scratch copy of the
files it reads and writes, a new extra fails it until the table names it, a new JAX requirement is written
into the page, and the standalone installer picks the CUDA build dew.telemetry.builds picks, a build added
there included."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dew.telemetry import builds

REPO = Path(__file__).resolve().parents[1]
READ = ("pyproject.toml", "constraints.txt", "README.md", "docs/installation.md", "tools/pip.txt",
        "site/scripts/gen_install.py", "site/scripts/colab-notebooks.mjs", "site/package.json",
        "site/public/install.sh")


def scratch(root: Path) -> Path:
    """The files the generator reads and writes, and the package module that declares the CUDA builds."""
    for name in READ:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / name, root / name)
    shutil.copytree(REPO / "src", root / "src", ignore=lambda _, names: [
        name for name in names if name.endswith(".py") and name != "builds.py"])
    # The tested Pythons, which tools/armada/ci.py reads from pyproject.toml's classifiers.
    shutil.copytree(REPO / "tools/armada", root / "tools/armada")
    return root


def generate(root: Path, *arguments: str) -> int:
    return subprocess.run([sys.executable, str(root / "site/scripts/gen_install.py"), *arguments],
                          capture_output=True, text=True, timeout=120).returncode


def test_an_extra_the_page_does_not_name_fails_until_its_table_does(tmp_path):
    root = scratch(tmp_path)
    assert generate(root, "--check") == 0
    pyproject = root / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace("\ndelta = ", '\nsparx = ["numpy"]\ndelta = ', 1))
    assert generate(root, "--check") == 1
    assert generate(root) == 1
    page = root / "docs/installation.md"
    page.write_text(page.read_text().replace("| `delta` |", "| `sparx` | NumPy |\n| `delta` |", 1))
    assert generate(root, "--check") == 0


def test_a_new_jax_requirement_is_written_into_the_page(tmp_path):
    root = scratch(tmp_path)
    pyproject = root / "pyproject.toml"
    requirement = next(line.strip().strip('",') for line in pyproject.read_text().splitlines()
                       if line.strip().startswith('"jax>='))
    pyproject.write_text(pyproject.read_text().replace(requirement, "jax>=0.12.0,<0.12.1", 1))
    assert generate(root, "--check") == 1
    assert generate(root) == 0
    assert generate(root, "--check") == 0


def test_a_new_package_index_is_the_one_the_installer_and_the_pages_name(tmp_path):
    root = scratch(tmp_path)
    table = root / Path(builds.__file__).resolve().relative_to(REPO)
    table.write_text(table.read_text().replace(builds.PACKAGE_INDEX, "https://index.example/simple", 1))
    assert generate(root, "--check") == 1
    assert generate(root) == 0
    assert generate(root, "--check") == 0
    assert installed(root, "NVIDIA L4, 580.82.07, 8.9", index=True) == "https://index.example/simple"


def installed(root: Path, answer: str, *options: str, index: bool = False) -> str:
    """What the standalone installer, run with `options` on a machine whose nvidia-smi answers `answer`,
    asks uv to install, or with `index` the package index it names: uv, nvidia-smi and the environment's
    Python stand in, so nothing is installed."""
    stand_ins = root / "bin"
    stand_ins.mkdir(exist_ok=True)
    scripts = {
        "nvidia-smi": f"printf '%s\\n' '{answer}'",
        # `uv venv ... TARGET` makes the environment, whose Python answers the installer's two questions.
        "uv": 'case $1 in venv) for target; do :; done; mkdir -p "$target/bin";'
              ' cp "$0.python" "$target/bin/python";; pip) printf "%s\\n" "$*" >> "$0.log";; esac',
        "uv.python": 'case $2 in *importlib*) echo "0.1.0 with jax 0.11.2 on gpu";; esac',
    }
    for name, body in scripts.items():
        (stand_ins / name).write_text(f"#!/bin/sh\n{body}\n")
        (stand_ins / name).chmod(0o755)
    script = root / "site/public/install.sh"
    done = subprocess.run(["sh", str(script), "-y", "--venv", str(root / "env"), *options], cwd=root,
                          env={"PATH": f"{stand_ins}:/usr/bin:/bin", "HOME": str(root)},
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    command = (stand_ins / "uv.log").read_text().split()
    return command[command.index("--extra-index-url") + 1] if index else command[-1]


@pytest.mark.parametrize("answer", ["NVIDIA L4, 580.82.07, 8.9", "NVIDIA T4, 560.35.03, 7.5",
                                    "NVIDIA K80, 470.82.01, 3.7", "NVIDIA V100, 520.61.05, 7.0"])
def test_the_installer_installs_the_build_the_driver_and_gpus_run(tmp_path, answer):
    root = scratch(tmp_path)
    _, driver, sm = (field.strip() for field in answer.split(","))
    build = builds.newest_build((driver, tuple(map(int, sm.split(".")))))
    assert installed(root, answer) == (f"dewml[{build}]" if build else "dewml")


def test_a_build_the_table_adds_is_the_installers_too(tmp_path):
    root = scratch(tmp_path)
    table = root / Path(builds.__file__).resolve().relative_to(REPO)
    newer = '("cuda14", 600, (8, 0), 14000), '
    table.write_text(table.read_text().replace("CUDA_BUILDS = (", f"CUDA_BUILDS = ({newer}", 1))
    assert generate(root) == 0
    assert installed(root, "NVIDIA H100, 600.10, 9.0") == "dewml[cuda14]"
    assert installed(root, "NVIDIA H100, 600.10, 9.0", "--cuda", "12") == "dewml[cuda12]"
