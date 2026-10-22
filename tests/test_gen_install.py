"""site/scripts/gen_install.py writes the install facts their owners declare into the files that repeat them,
and holds docs/installation.md's extras table to the extras pyproject.toml declares: on a scratch copy of the
files it reads and writes, a new extra fails it until the table names it, and a new JAX requirement is written
into the page."""

import shutil
import subprocess
import sys
from pathlib import Path

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
    assert "jax>=0.12.0,<0.12.1" in (root / "docs/installation.md").read_text()
    assert requirement not in (root / "docs/installation.md").read_text()
