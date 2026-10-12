"""The host PyTorch installer resolves a single build flavor for both packages."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("options", [
    ["--system"],
    ["--python", "/a virtual environment/bin/python"],
    ["-q"],
])
def test_the_host_pair_resolves_from_one_index(options, tmp_path):
    uv = tmp_path / "uv"
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    uv.chmod(0o755)
    result = subprocess.run(
        ["sh", str(ROOT / "tools/install_host_torch.sh"), *options],
        env={**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
        text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == [
        "pip", "install", *options, "--upgrade", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cpu",
    ]


def test_a_failed_host_pair_install_fails_its_caller(tmp_path):
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 23\n")
    uv.chmod(0o755)
    result = subprocess.run(
        ["sh", str(ROOT / "tools/install_host_torch.sh"), "--system"],
        env={**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
        text=True, capture_output=True, check=False)
    assert result.returncode == 23
