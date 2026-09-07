"""Native diffusion against authentic saved checkpoint/config/tokenizer oracles.

References run Diffusers 0.34.0 / Transformers 4.49.0 in isolation. Tests here
run Dew-native models, preprocessing, Process and solvers on the current stack.
Each worker process releases its JAX executable caches after the comparison.
"""
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def saved_pipelines(tmp_path_factory):
    destination = tmp_path_factory.mktemp("saved-diffusion")
    with tarfile.open(ROOT / "tests/fixtures/tiny_diffusers.tar.xz") as archive:
        archive.extractall(destination, filter="data")
    return destination


def run_check(*arguments):
    result = subprocess.run([sys.executable, str(ROOT / "tools/check_native_diffusion.py"), *map(str, arguments)],
                            cwd=ROOT, capture_output=True, text=True, timeout=240,
                            env={**os.environ, "JAX_PLATFORMS": "cpu", "USE_TF": "0",
                                 "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "2",
                                 "OPENBLAS_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("task", ["sd", "xl", "img2img", "inpaint", "xl-img2img", "xl-inpaint", "refiner", "safety"])
def test_native_bundle_trajectory_update_and_source_roundtrip(saved_pipelines, task):
    run_check(saved_pipelines / task)


@pytest.mark.parametrize("solver", ["pndm-prk", "pndm-plms", "lms", "lms-v", "lms-karras", "euler", "dpm"])
def test_native_solver_consumes_the_complete_source_grid(saved_pipelines, solver):
    run_check(saved_pipelines / "sd", "--grids", ROOT / "tests/fixtures/diffusers_pipeline_schedulers.npz", "--case", solver)


@pytest.mark.parametrize("task", ["sd", "xl"])
def test_raw_and_prepared_native_inputs_agree(saved_pipelines, task):
    run_check(saved_pipelines / task, "--case", "prepared")


def test_source_roundtrip_preserves_generation_geometry(saved_pipelines):
    run_check(saved_pipelines / "xl", "--case", "geometry")


def test_checkpoint_policy_controls_omitted_negative_conditioning(saved_pipelines):
    run_check(saved_pipelines / "xl", "--case", "negative-policy")


def test_caption_dropout_keeps_mask_conditioned_gradients(saved_pipelines):
    run_check(saved_pipelines / "inpaint", "--case", "mask-dropout")


@pytest.mark.parametrize("case", ["sd1", "sd2", "sdxl-inpaint", "refiner"])
def test_multilevel_unet_geometry_and_input_gradient(tmp_path, case):
    with tarfile.open(ROOT / "tests/fixtures/native_unet_geometry.tar.xz") as archive:
        archive.extractall(tmp_path, filter="data")
    run_check(tmp_path / case, "--model")
