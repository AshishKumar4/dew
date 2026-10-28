"""A reference environment (tools/environments.py `reference`, `--reference`): torch's, an engine's, named
by its installation and built once, by the builder a checkout's own environment is built by. uv stands in
on PATH, writing what it was asked."""

import hashlib
import json
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


def test_a_references_overrides_and_constraints_name_it_by_their_lines(tmp_path, monkeypatch):
    """A reference that installs a lock with overrides (torch's CUDA build over a CPU lock, say) or
    constraints is named by those files' lines too: a changed override builds anew, and the same lines at
    another path reuse the environment."""
    asked, env = standing_in_uv(tmp_path)
    monkeypatch.setenv("PATH", env["PATH"])
    lock, override = tmp_path / "lock.txt", tmp_path / "override.txt"
    lock.write_text("numpy==2.3.0\n")
    override.write_text("torch==2.14.1+cu128\n")
    cache = tmp_path / "envs"
    arguments = ["-r", str(lock), "--override", str(override)]
    python = environments.reference("torch", arguments, cache=cache)
    moved = tmp_path / "moved.txt"
    moved.write_text(override.read_text())
    assert environments.reference("torch", ["-r", str(lock), "--override", str(moved)], cache=cache) == python
    override.write_text("torch==2.15.0+cu128\n")
    assert environments.reference("torch", arguments, cache=cache) != python
    assert len(installs(asked)) == 2


def test_a_file_a_references_file_reads_names_it_too(tmp_path, monkeypatch):
    """A reference's override that reads another file (`-r`, `-c`, relative to it, however deep, its
    operand quoted or named by the environment) is named by that file's lines as well, each file apart: a
    changed pin in the included file alone builds anew, as does a pin moved between the two, the same lines
    reuse the environment, and a file reading itself is read once."""
    asked, env = standing_in_uv(tmp_path)
    monkeypatch.setenv("PATH", env["PATH"])
    (tmp_path / "locks").mkdir()
    lock, override, included = (tmp_path / "lock.txt", tmp_path / "locks/override.txt",
                                tmp_path / "locks/torch.txt")
    lock.write_text("numpy==2.3.0\n")
    override.write_text("-r torch.txt\ntransformers==5.19\n")
    included.write_text("--constraint=torch.txt\ntorch==2.8.0+cu128\n")
    cache = tmp_path / "envs"
    arguments = ["-r", str(lock), "--override", str(override)]
    python = environments.reference("torch", arguments, cache=cache)
    assert environments.reference("torch", arguments, cache=cache) == python
    included.write_text("--constraint=torch.txt\ntorch==2.9.0+cu128\n")
    changed = environments.reference("torch", arguments, cache=cache)
    assert changed != python
    # A pin moved from the included file into the one reading it is a change, though their lines join alike.
    child = tmp_path / "locks/child.txt"
    override.write_text("-c child.txt\npackaging\n")
    child.write_text("six==1.16.0\n")
    constrained = environments.reference("torch", arguments, cache=cache)
    override.write_text("-c child.txt\npackaging\nsix==1.16.0\n")
    child.write_text("")
    assert environments.reference("torch", arguments, cache=cache) not in (python, changed, constrained)
    # An operand quoted as a shell word, or named by the environment, is read as pip and uv read it.
    (tmp_path / "locks/cuda pins.txt").write_text("nvidia-cublas==12.8\n")
    (tmp_path / "locks/selected.txt").write_text("triton==3.4\n")
    monkeypatch.setenv("PINS", "selected.txt")
    override.write_text('-r torch.txt\n-c "cuda pins.txt"  # the CUDA wheels\n-r ${PINS}\n')
    quoted = environments.reference("torch", arguments, cache=cache)
    (tmp_path / "locks/selected.txt").write_text("triton==3.5\n")
    assert environments.reference("torch", arguments, cache=cache) != quoted
    # A name holding a lone quote is read as written, as uv reads it.
    (tmp_path / "locks/pins'2026.txt").write_text("tokenizers==0.22\n")
    override.write_text("-r pins'2026.txt\n")
    literal = environments.reference("torch", arguments, cache=cache)
    (tmp_path / "locks/pins'2026.txt").write_text("tokenizers==0.23\n")
    assert environments.reference("torch", arguments, cache=cache) != literal
    assert len(installs(asked)) == 8


def test_the_command_line_takes_a_references_installation_whole_after_the_separator(tmp_path):
    """Everything after `--` is uv's installation, its own options too (a torch backend), whatever spelling
    of --reference argparse takes, and the Python printed is the environment's, made at the Python asked
    for; a reference with no installation, or a tree beside a reference, is refused before anything is
    built."""
    asked, env = standing_in_uv(tmp_path)
    tool = [sys.executable, str(REPO_ROOT / "tools" / "environments.py")]
    cache = tmp_path / "envs"
    printed = subprocess.run([*tool, "--ref", "fla", "--python", "3.13", "--cache", str(cache), "--",
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


def test_a_references_python_and_its_literal_arguments_are_its_own(tmp_path, monkeypatch):
    """Two Pythons asked for one installation are two environments (3.12's keeps the name it always had), and
    an installation's arguments reach uv as written, whatever they hold."""
    asked, env = standing_in_uv(tmp_path)
    monkeypatch.setenv("PATH", env["PATH"])
    cache = tmp_path / "envs"
    default = environments.reference("x", ["numpy"], cache=cache)
    other = environments.reference("x", ["numpy"], python="3.13", cache=cache)
    assert default != other
    # The name every node's reference environments already have.
    assert default.parents[1].name == "x-" + hashlib.sha256(json.dumps(["numpy"]).encode()).hexdigest()[:12]
    braced = tmp_path / "{python}" / "requirements.txt"
    braced.parent.mkdir()
    braced.write_text("numpy\n")
    python = environments.reference("y", ["-r", str(braced)], cache=cache)
    assert installs(asked)[-1] == f"pip install -q --python {python} -r {braced}"


def test_a_checkout_named_after_the_separator_is_built(tmp_path):
    """`--` before TREE is argparse's own for a checkout, as it always was: the tree after it is built."""
    asked, env = standing_in_uv(tmp_path)
    tree = tmp_path / "tree"
    (tree / "locks").mkdir(parents=True)
    (tree / "tools").mkdir()
    (tree / "locks" / "cpu.txt").write_text("numpy\n")
    (tree / "tools" / "install_environment.sh").write_text(f"echo installed \"$@\" >> {asked}\n")
    (tree / ".python-version").write_text("3.12\n")
    printed = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "environments.py"), "--lock", "cpu",
                              "--cache", str(tmp_path / "envs"), "--", str(tree)],
                             env=env, capture_output=True, text=True, check=True).stdout
    python = Path(printed.strip())
    assert python.parents[1].name.startswith("cpu-")
    assert asked.read_text().splitlines()[-1] == f"installed cpu -q --python {python}"

