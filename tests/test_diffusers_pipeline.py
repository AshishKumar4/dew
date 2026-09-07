"""Actual saved image pipelines with authentic configs, tokenizers and weights.

References: Diffusers 0.34.0 / Transformers 4.49.0. Runtime uses the declared
current Transformers stack. Full scheduler grids use the same official Flax
UNet on both sides; extended SDXL/checker fixtures use official Torch pipelines.
The reference/checker tools record numerical tolerances and observed errors.
"""
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

pytest.importorskip("diffusers")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def saved_pipelines(tmp_path_factory):
    destination = tmp_path_factory.mktemp("saved-diffusers")
    with tarfile.open(ROOT / "tests/fixtures/tiny_diffusers.tar.xz") as archive:
        archive.extractall(destination, filter="data")
    return destination


def run_reference_check(script, directory, *arguments):
    # Each process releases its executable caches, including autodiff graphs.
    result = subprocess.run([sys.executable, str(ROOT / "tools" / script), str(directory), *arguments],
                            cwd=ROOT, capture_output=True, text=True, timeout=180,
                            env={**os.environ, "JAX_PLATFORMS": "cpu", "USE_TF": "0",
                                 "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "2",
                                 "OPENBLAS_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("task", ["sd", "xl", "img2img", "inpaint"])
def test_saved_pipeline_trajectory_training_and_reload(saved_pipelines, task):
    arguments = ("--no-train",) if task == "img2img" else ()
    run_reference_check("check_diffusers_pipeline.py", saved_pipelines / task, *arguments)


@pytest.mark.parametrize("task", ["xl-img2img", "xl-inpaint", "refiner", "safety"])
def test_extended_saved_pipeline_and_retained_components(saved_pipelines, task):
    run_reference_check("check_diffusers_extended.py", saved_pipelines / task)


@pytest.mark.parametrize("scheduler", ["pndm-prk", "pndm-plms", "lms", "lms-v", "lms-karras", "euler"])
def test_complete_source_scheduler_trajectory(saved_pipelines, scheduler):
    run_reference_check("check_diffusers_extended.py", saved_pipelines / "sd", "--case", scheduler)


@pytest.mark.parametrize("task", ["sd", "xl"])
def test_raw_and_prepared_inputs_share_the_trajectory(saved_pipelines, task):
    run_reference_check("check_diffusers_extended.py", saved_pipelines / task, "--case", "prepared")


def test_saved_resolution_preserves_sdxl_generation(saved_pipelines):
    run_reference_check("check_diffusers_extended.py", saved_pipelines / "xl", "--case", "geometry")


def test_sdxl_checkpoint_controls_omitted_negative_prompts(saved_pipelines):
    run_reference_check("check_diffusers_extended.py", saved_pipelines / "xl", "--case", "negative-policy")


def test_inpainting_caption_dropout_preserves_mask_conditioned_gradients(saved_pipelines):
    run_reference_check("check_diffusers_extended.py", saved_pipelines / "inpaint", "--case", "mask-dropout")
