"""Numerical regressions for nondefault published diffusion controls."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from interop_support import extract_fixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def edges(tmp_path_factory):
    return extract_fixture(ROOT / "tests/fixtures/native_diffusion_edges.tar.xz",
                           tmp_path_factory.mktemp("native-diffusion-edges"))


@pytest.mark.parametrize(
    "case", ["ddim-linspace", "pndm-linspace", "ddim-clip", "euler-zero-snr", "dpm-karras", "norm", "odd",
             "dropped"]
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
