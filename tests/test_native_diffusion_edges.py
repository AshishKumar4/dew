"""Numerical regressions for nondefault published diffusion controls."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from interop_support import extract_fixture
from whole_suite import timed

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def edges(tmp_path_factory):
    return extract_fixture(ROOT / "tests/fixtures/native_diffusion_edges.tar.xz",
                           tmp_path_factory.mktemp("native-diffusion-edges"))


PAST_TARGET = {"parity": {"dpm-karras": 171}}
"""The cases past a group's 120 s (`timed`), each its seconds in train 270's CI (66a222273, ledger 2133):
DPM-Karras's schedule; every train walks the same native-source path under DDIM, DDIM-clip, Euler and
PNDM, row 647's proof."""


@pytest.mark.parametrize("case", timed(PAST_TARGET, "parity", ["ddim-linspace", "ddim-clip", "euler-zero-snr",
                                                                "dpm-karras", "norm", "odd", "dropped"]))
def test_native_source_control_and_gradient_parity(edges, case):
    checked(edges, case)


@pytest.mark.check1_shards(seeds=128, per_task=32)
def test_native_source_control_and_gradient_parity_under_pndm(edges):
    """`checked` over PNDM's linspace walk, whose 128 draws, each its own table's walk compiled again, took
    143 s in one task (ledger 3896): CI draws them in map tasks of 32, the tool's process inheriting the
    task's draws (`map_judge`)."""
    checked(edges, "pndm-linspace")


def checked(edges, case):
    """tools/check_native_diffusion_edges.py's `case` over the extracted fixture, in a process of its own."""
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
