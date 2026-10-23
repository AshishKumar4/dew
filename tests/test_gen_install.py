"""site/scripts/gen_install.py writes the install facts their owners declare into the files that repeat them,
and holds docs/installation.md's extras table to the extras pyproject.toml declares: on a scratch copy of the
files it reads and writes, a new extra fails it until the table names it, a new JAX requirement is written
into the page, and the standalone installer picks the CUDA build dew.builds picks, a build added
there included."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dew import builds

REPO = Path(__file__).resolve().parents[1]
READ = ("pyproject.toml", "constraints.txt", "README.md", "docs/installation.md", "tools/pip.txt",
        "site/package.json", "site/public/install.sh")


def scratch(root: Path) -> Path:
    """The files the generator reads and writes, and the package module that declares the CUDA builds."""
    for name in READ:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / name, root / name)
    shutil.copytree(REPO / "src", root / "src", ignore=lambda _, names: [
        name for name in names if name.endswith(".py") and name not in ("builds.py", "tpu_setup.py")])
    # The generator and what it reads beside it, and the tested Pythons, which tools/armada/ci.py reads from
    # pyproject.toml's classifiers.
    for directory in ("site/scripts", "tools/armada"):
        shutil.copytree(REPO / directory, root / directory, ignore=shutil.ignore_patterns("__pycache__"))
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
    command, _ = installed(root, "NVIDIA L4, 580.82.07, 8.9")
    assert command[command.index("--extra-index-url") + 1] == "https://index.example/simple"


UV = """#!/bin/sh
case $1 in
--version) echo "uv @VERSION@" ;;
venv) for target; do :; done; mkdir -p "$target/bin"; cp "$STAND_INS/python" "$target/bin/python" ;;
pip) echo "@VERSION@ $*" >> "$STAND_INS/uv.log" ;;
esac
"""
"""A uv of release @VERSION@: it makes an environment whose Python answers the installer's two questions,
and logs each install it is asked for, its release first."""

STAND_INS = {
    "python": 'case $2 in *importlib*) echo "0.1.0 with jax 0.11.2 on gpu";; esac',
    # astral's installer for the release the URL names, which puts that uv in ~/.local/bin.
    "curl": 'for url; do :; done; echo "$url" >> "$STAND_INS/curl.log"; release=${url#*/uv/}; '
            'release=${release%/install.sh}; printf \'mkdir -p "$HOME/.local/bin"; '
            'sed "s/@VERSION@/%s/" "$STAND_INS/uv.template" > "$HOME/.local/bin/uv"; '
            'chmod +x "$HOME/.local/bin/uv"\\n\' "$release"',
}


def installed(root: Path, answer: str, *options: str, uv: str = "99.0.0") -> tuple[list[str], list[str]]:
    """What the standalone installer, run with `options` on a machine whose nvidia-smi answers `answer`
    and whose uv is release `uv`, asks: the uv install it runs, the release that ran it first, and the URLs
    it fetched. uv, curl, nvidia-smi and the environment's Python stand in, so nothing is installed."""
    stand_ins = root / "bin"
    stand_ins.mkdir(exist_ok=True)
    scripts = {**STAND_INS, "nvidia-smi": f"printf '%s\\n' '{answer}'"}
    for name, body in scripts.items():
        (stand_ins / name).write_text(f"#!/bin/sh\n{body}\n")
        (stand_ins / name).chmod(0o755)
    (stand_ins / "uv.template").write_text(UV)
    (stand_ins / "uv").write_text(UV.replace("@VERSION@", uv))
    (stand_ins / "uv").chmod(0o755)
    script = root / "site/public/install.sh"
    done = subprocess.run(["sh", str(script), "-y", "--venv", str(root / "env"), *options], cwd=root,
                          env={"PATH": f"{stand_ins}:/usr/bin:/bin", "HOME": str(root),
                               "STAND_INS": str(stand_ins)},
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    fetched = stand_ins / "curl.log"
    return ((stand_ins / "uv.log").read_text().splitlines()[-1].split(),
            fetched.read_text().split() if fetched.exists() else [])


@pytest.mark.parametrize("answer", ["NVIDIA L4, 580.82.07, 8.9", "NVIDIA T4, 560.35.03, 7.5",
                                    "NVIDIA K80, 470.82.01, 3.7", "NVIDIA V100, 520.61.05, 7.0"])
def test_the_installer_installs_the_build_the_driver_and_gpus_run(tmp_path, answer):
    root = scratch(tmp_path)
    _, driver, sm = (field.strip() for field in answer.split(","))
    build = builds.newest_build((driver, tuple(map(int, sm.split(".")))))
    assert installed(root, answer)[0][-1] == (f"dewml[{build}]" if build else "dewml")


def test_a_build_the_table_adds_is_the_installers_too(tmp_path):
    root = scratch(tmp_path)
    table = root / Path(builds.__file__).resolve().relative_to(REPO)
    newer = '("cuda14", 600, (8, 0), 14000), '
    table.write_text(table.read_text().replace("CUDA_BUILDS = (", f"CUDA_BUILDS = ({newer}", 1))
    assert generate(root) == 0
    assert installed(root, "NVIDIA H100, 600.10, 9.0")[0][-1] == "dewml[cuda14]"
    assert installed(root, "NVIDIA H100, 600.10, 9.0", "--cuda", "12")[0][-1] == "dewml[cuda12]"


def floored(root: Path, least: str) -> Path:
    """`root`'s scratch copy with pyproject.toml's least uv `least`, its installers written from it."""
    scratch(root)
    pyproject = root / "pyproject.toml"
    floor = next(line for line in pyproject.read_text().splitlines() if line.startswith("required-version"))
    pyproject.write_text(pyproject.read_text().replace(floor, f'required-version = ">={least}"', 1))
    assert generate(root) == 0
    return root


@pytest.mark.parametrize(("have", "fetched", "running"), [
    ("0.13.9", ["https://astral.sh/uv/0.14.2/install.sh"], "0.14.2"),
    ("0.14.2-rc.1", ["https://astral.sh/uv/0.14.2/install.sh"], "0.14.2"),
    ("0.14.2+build.1", [], "0.14.2+build.1"), ("0.14.10", [], "0.14.10")])
def test_an_older_uv_is_replaced_by_the_least_one_which_runs_the_install(tmp_path, have, fetched, running):
    command, asked = installed(floored(tmp_path, "0.14.2"), "NVIDIA L4, 580.82.07, 8.9", uv=have)
    assert asked == fetched and command[:3] == [running, "pip", "install"]
