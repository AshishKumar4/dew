"""Numerical regressions for nondefault published diffusion controls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from dew.diffusion.schedules.source import SourceSchedule
from dew.interop.diffusion import unet_fields

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def edges(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-diffusion-edges")
    with tarfile.open(ROOT / "tests/fixtures/native_diffusion_edges.tar.xz") as archive:
        archive.extractall(directory, filter="data")
    return directory


@pytest.mark.parametrize("case", ["ddim-linspace", "pndm-linspace", "ddim-clip", "euler-zero-snr", "dpm-karras", "norm", "odd"])
def test_native_source_control_and_gradient_parity(edges, case):
    result = subprocess.run([sys.executable, str(ROOT / "tools/check_native_diffusion_edges.py"), str(edges), case],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "JAX_PLATFORMS": "cpu", "USE_TF": "0",
             "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("control", ["thresholding", "use_beta_sigmas", "use_lu_lambdas"])
def test_unimplemented_active_scheduler_controls_do_not_change_meaning(edges, control):
    with np.load(edges / "schedulers.npz") as reference:
        config = json.loads(str(reference["dpm-karras.config"]))
    with pytest.raises(ValueError):
        SourceSchedule.from_config({**config, control: True})


def test_source_pndm_does_not_substitute_epsilon_history_for_velocity(edges):
    with np.load(edges / "schedulers.npz") as reference:
        config = json.loads(str(reference["pndm-linspace.config"]))
    with pytest.raises(ValueError):
        SourceSchedule.from_config({**config, "prediction_type": "v_prediction"})


@pytest.mark.parametrize("control,value", [("center_input_sample", True), ("resnet_time_scale_shift", "scale_shift"),
                                           ("act_fn", "mish"), ("conv_in_kernel", 1)])
def test_unimplemented_unet_operations_are_explicit(edges, control, value):
    config = json.loads((edges / "unet/config.json").read_text())
    with pytest.raises(ValueError):
        unet_fields({**config, control: value})
