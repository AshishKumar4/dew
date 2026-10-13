"""Native diffusion against authentic saved checkpoint/config/tokenizer oracles.

References run Diffusers 0.34.0 / Transformers 4.49.0 in isolation. Tests here
run Dew-native models, preprocessing, Process and solvers on the current stack.
Each worker process releases its JAX executable caches after the comparison.
"""
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def saved_pipelines(tmp_path_factory):
    destination = tmp_path_factory.mktemp("saved-diffusion")
    with tarfile.open(ROOT / "tests/fixtures/tiny_diffusers.tar.xz") as archive:
        archive.extractall(destination, filter="data")
    return destination


def run_check(*arguments):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/check_native_diffusion.py"), *map(str, arguments)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "USE_TF": "0",
            "PYTHONPATH": str(ROOT / "src"),
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "1",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "task", ["sd", "xl", "img2img", "inpaint", "xl-img2img", "xl-inpaint", "refiner", "safety"]
)
def test_native_bundle_trajectory_update_and_source_roundtrip(saved_pipelines, task):
    run_check(saved_pipelines / task)


@pytest.mark.parametrize("task", ["sd", "xl", "safety"])
def test_a_streamed_bundle_matches_the_source(saved_pipelines, task):
    """The same checks, on the pipeline streamed onto a mesh a leaf at a time
    (tests/test_pipeline_streaming.py)."""
    run_check(saved_pipelines / task, "--streamed")


@pytest.mark.parametrize("solver", ["pndm-prk", "pndm-plms", "lms", "lms-v", "lms-karras", "euler", "dpm"])
def test_native_solver_consumes_the_complete_source_grid(saved_pipelines, solver):
    run_check(
        saved_pipelines / "sd",
        "--grids",
        ROOT / "tests/fixtures/diffusers_pipeline_schedulers.npz",
        "--case",
        solver,
    )


@pytest.mark.parametrize("task", ["sd", "xl"])
def test_raw_and_prepared_native_inputs_agree(saved_pipelines, task):
    run_check(saved_pipelines / task, "--case", "prepared")


def test_source_roundtrip_preserves_generation_geometry(saved_pipelines):
    run_check(saved_pipelines / "xl", "--case", "geometry")


def test_checkpoint_policy_controls_omitted_negative_conditioning(saved_pipelines):
    run_check(saved_pipelines / "xl", "--case", "negative-policy")


def test_the_sdxl_base_hands_its_partial_walk_to_the_refiner_as_diffusers_does(saved_pipelines):
    """The base walks the steps at or above the cutoff and stops undecoded,
    and the refiner continues from those latents (`prepare(initial=)`)
    below it, held to tests/reference_error.py's rule against Diffusers'
    own SDXL base with denoising_end and refiner with denoising_start
    (tools/diffusers_handoff_reference.py): the base's latents, the
    refiner's latents and their decoded images. docs/concepts/inference.md
    shows the mapping from strength and denoising_end/start to the grid."""
    run_check(saved_pipelines / "xl", "--case", "handoff")


def test_a_prepared_mask_is_the_inpainting_pipelines_own_at_every_level(saved_pipelines):
    """`prepare(mask=)` binarizes and shrinks a mask as Diffusers 0.34.0's
    inpainting pipeline does (its mask processor, then nearest
    interpolation), at each of the 256 byte levels."""
    run_check(saved_pipelines / "inpaint", "--case", "mask-levels")


def test_caption_dropout_keeps_mask_conditioned_gradients(saved_pipelines):
    run_check(saved_pipelines / "inpaint", "--case", "mask-dropout")


@pytest.mark.parametrize("case", ["sd1", "sd2", "sdxl-inpaint", "refiner"])
def test_multilevel_unet_geometry_and_input_gradient(tmp_path, case):
    with tarfile.open(ROOT / "tests/fixtures/native_unet_geometry.tar.xz") as archive:
        archive.extractall(tmp_path, filter="data")
    run_check(tmp_path / case, "--model")


def test_public_diffusion_export_preserves_mapped_snapshot_when_republished(saved_pipelines, tmp_path):
    import numpy as np

    from dew.interop import Pretrained

    source = Pretrained.load(saved_pipelines / "sd", dtype="float32", attention_impl="xla")
    original = {layout.name: layout.export(source.variables) for layout in source.weight_layouts}
    destination = tmp_path / "export"
    source.save(destination)
    mapped = Pretrained.load(destination, dtype="float32", attention_impl="xla")
    assert {layout.name for layout in mapped.weight_layouts} == set(original)
    for layout in mapped.weight_layouts:
        np.testing.assert_array_equal(layout.export(mapped.variables), original[layout.name])

    changed = next(layout for layout in mapped.weight_layouts if layout.name == "unet/conv_in.bias")

    def replace_leaf(tree, path):
        name, *tail = path
        value = replace_leaf(tree[name], tail) if tail else np.asarray(tree[name]) + np.float32(1.)
        return {**tree, name: value}

    updated = replace_leaf(mapped.variables, changed.paths[0])
    mapped.save(destination, variables=updated)
    for layout in mapped.weight_layouts:
        np.testing.assert_array_equal(layout.export(mapped.variables), original[layout.name])
    latest = Pretrained.load(destination, dtype="float32", attention_impl="xla")
    for layout in latest.weight_layouts:
        expected = (
            original[layout.name] + np.float32(1.0) if layout.name == changed.name else original[layout.name]
        )
        np.testing.assert_array_equal(layout.export(latest.variables), expected)


def test_a_resized_xl_conditioning_rebuilds_from_its_record_at_its_own_size(saved_pipelines):
    """An XL pipeline loaded at 64x48 instead of its own 32x32 states that size in its time ids, and
    the conditioner its run record rebuilds states the same, where it used to read the pipeline's
    own size back."""
    from dew.inputs import Condition
    from dew.interop.pretrained import load_diffusion_source

    source = load_diffusion_source(str(saved_pipelines / "xl"), dtype="float32", attention_impl="xla",
                                   size=(64, 48))
    condition = source.inputs.conditions["conditioning"]
    assert condition.encoder.time_id_values == (64, 48, 0, 0, 64, 48)
    assert source.inputs.sample.shape[:2] == (64, 48)
    reloaded = Condition.from_json(condition.to_json(), params=condition.encoder.params)
    assert reloaded.encoder.time_id_values == condition.encoder.time_id_values
