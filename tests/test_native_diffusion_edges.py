"""Numerical regressions for nondefault published diffusion controls."""
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def edges(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-diffusion-edges")
    with tarfile.open(ROOT / "tests/fixtures/native_diffusion_edges.tar.xz") as archive:
        archive.extractall(directory, filter="data")
    return directory


@pytest.mark.parametrize(
    "case", ["ddim-linspace", "pndm-linspace", "ddim-clip", "euler-zero-snr", "dpm-karras", "norm", "odd"]
)
def test_native_source_control_and_gradient_parity(edges, case):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/check_native_diffusion_edges.py"), str(edges), case],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "JAX_PLATFORMS": "cpu",
            "USE_TF": "0",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "1",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
