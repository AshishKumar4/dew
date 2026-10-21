"""A reference environment (tools/environments.py `reference`, `--reference`): torch's, an engine's, named
by its installation and built once, by the builder a checkout's own environment is built by. uv stands in
on PATH, writing what it was asked."""

import os
import subprocess
import sys
from pathlib import Path

from tool_support import REPO_ROOT, load

environments = load("environments")

UV = """#!/bin/sh
echo "$*" >> {asked}
if [ "$1" = venv ]; then for last; do :; done; mkdir -p "$last/bin" && touch "$last/bin/python"; fi
"""


def standing_in_uv(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """uv, standing in on PATH: the file of what it was asked, one line a call, and the environment
    that finds it."""
    tools = tmp_path / "bin"
    tools.mkdir()
    asked = tmp_path / "asked"
    (tools / "uv").write_text(UV.format(asked=asked))
    (tools / "uv").chmod(0o755)
    return asked, {**os.environ, "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}"}


def installs(asked: Path) -> list[str]:
    return [line for line in asked.read_text().splitlines() if line.startswith("pip install")]


def test_a_reference_is_named_by_its_installation_and_built_once(tmp_path, monkeypatch):
    """A requirements file's lines, not its path, name a reference's environment: the same lines at
    another path reuse it, and changed lines are built anew beside it, which a running job may hold. One
    whose build never finished is started over."""
    asked, env = standing_in_uv(tmp_path)
    monkeypatch.setenv("PATH", env["PATH"])
    requirements = tmp_path / "torch.txt"
    requirements.write_text("torch==2.10.0\n")
    cache = tmp_path / "envs"

    python = environments.reference("torch", ["-r", str(requirements)], cache=cache)
    venv = python.parents[1]
    assert venv.parent == cache.resolve() and venv.name.startswith("torch-") and (venv / "complete").is_file()
    assert installs(asked) == [f"pip install -q --python {python} -r {requirements}"]

    moved = tmp_path / "moved.txt"
    moved.write_text(requirements.read_text())
    assert environments.reference("torch", ["-r", str(moved)], cache=cache) == python
    assert len(installs(asked)) == 1

    requirements.write_text("torch==2.11.0\n")
    changed = environments.reference("torch", ["-r", str(requirements)], cache=cache)
    assert changed != python and (venv / "complete").is_file()
    assert len(installs(asked)) == 2

    (changed.parents[1] / "complete").unlink()
    (changed.parents[1] / "left-by-the-unfinished-build").touch()
    assert environments.reference("torch", ["-r", str(requirements)], cache=cache) == changed
    assert not (changed.parents[1] / "left-by-the-unfinished-build").exists()
    assert len(installs(asked)) == 3


def test_the_command_line_takes_a_references_installation_whole_after_the_separator(tmp_path):
    """Everything after `--` is uv's installation, its own options too (a torch backend), and the Python
    printed is the environment's, made at the Python asked for; a reference with no installation, or a
    tree beside a reference, is refused before anything is built."""
    asked, env = standing_in_uv(tmp_path)
    tool = [sys.executable, str(REPO_ROOT / "tools" / "environments.py")]
    cache = tmp_path / "envs"
    printed = subprocess.run([*tool, "--reference", "fla", "--python", "3.13", "--cache", str(cache), "--",
                              "--torch-backend", "cu126", "fla==0.5.2"],
                             env=env, capture_output=True, text=True, check=True).stdout
    python = Path(printed.strip())
    assert python.parents[2] == cache.resolve() and python.parents[1].name.startswith("fla-")
    assert asked.read_text().splitlines() == [
        f"venv -q --python 3.13 {python.parents[1]}",
        f"pip install -q --python {python} --torch-backend cu126 fla==0.5.2"]

    for refused in (["--reference", "fla"], [str(tmp_path), "--reference", "fla", "--", "fla==0.5.2"]):
        ran = subprocess.run([*tool, "--cache", str(cache), *refused], env=env, capture_output=True,
                             text=True)
        assert ran.returncode == 2, ran.stderr
    assert len(asked.read_text().splitlines()) == 2
