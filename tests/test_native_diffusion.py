"""Native diffusion against authentic saved checkpoint/config/tokenizer oracles.

References run Diffusers 0.34.0 / Transformers 4.49.0 in isolation. Tests here
run Dew-native models, preprocessing, Process and solvers on the current stack.
Each worker process releases its JAX executable caches after the comparison.
"""
import json
import os
import shutil
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


def _declared(saved_pipelines, tmp_path, case: str, class_name: str) -> Path:
    """A copy of the fixture directory declaring `class_name` instead."""
    directory = tmp_path / case
    shutil.copytree(saved_pipelines / case, directory)
    index = json.loads((directory / "model_index.json").read_text())
    index["_class_name"] = class_name
    (directory / "model_index.json").write_text(json.dumps(index))
    return directory


@pytest.mark.parametrize("case, declared, held", [
    ("sd", "StableDiffusionXLPipeline", "StableDiffusionPipeline"),
    ("xl", "StableDiffusionPipeline", "StableDiffusionXLPipeline"),
])
def test_a_directory_declaring_another_familys_pipeline_is_refused(
        saved_pipelines, tmp_path, case, declared, held):
    """SD and XL share the UNet denoiser, so the component check cannot tell
    them apart; the gate reads the pipeline family and names both classes."""
    from dew.interop.pretrained import load_pretrained

    directory = _declared(saved_pipelines, tmp_path, case, declared)
    with pytest.raises(ValueError, match=f"{declared}.*{held}"):
        load_pretrained(str(directory), dtype="float32", attention_impl="xla")


@pytest.mark.parametrize("case, declared, guidance", [
    ("sd", "StableDiffusionImg2ImgPipeline", 7.5),
    ("xl", "StableDiffusionXLImg2ImgPipeline", 5.0),
])
def test_a_matching_image_task_variant_loads(saved_pipelines, tmp_path, case, declared, guidance):
    """A pipeline of the denoiser's own family keeps loading with its own
    defaults: each img2img variant is accepted and guides at its scale."""
    from dew.interop.pretrained import load_pretrained

    directory = _declared(saved_pipelines, tmp_path, case, declared)
    loaded = load_pretrained(str(directory), dtype="float32", attention_impl="xla")
    assert loaded.text_to_image().guidance.scale == guidance


@pytest.mark.parametrize("family", ["sd", "xl", "safety", "sd3", "flux"])
def test_public_source_precision_covers_denoiser_and_frozen_component_weights(saved_pipelines, tmp_path, family):
    import jax.numpy as jnp
    from test_interop import assert_parameter_storage

    from dew.interop import load_pretrained

    if family in ("sd3", "flux"):
        with tarfile.open(ROOT / f"tests/fixtures/{family}_source.tar.xz") as archive:
            archive.extractall(tmp_path, filter="data")
        directory = tmp_path / "pipeline"
    else:
        directory = saved_pipelines / family
    masters = load_pretrained(directory, dtype="bfloat16", attention_impl="xla")
    native = load_pretrained(directory, dtype="float32", param_dtype="bfloat16", attention_impl="xla")
    assert masters.model.dtype == jnp.bfloat16
    assert native.model.dtype == jnp.float32

    def parameter(path):
        return (path[0] in ("params", "autoencoder")
                or path[:2] == ("encoders", "conditioning")
                or path[:3] in (("encoders", "safety", "vision_model"),
                               ("encoders", "safety", "visual_projection")))

    assert_parameter_storage(masters.variables, native.variables, parameter)


def test_component_binding_preserves_large_integer_indices():
    import numpy as np

    from dew.interop.diffusion import record_layouts

    indices = np.asarray([1, 16777217], np.int64)
    values, _ = record_layouts("component", {"indices": indices}, lambda name: (name,),
                               ("buffers",), param_dtype="bfloat16")
    value = values["indices"]
    assert isinstance(value, np.ndarray) and value.dtype == np.int64
    np.testing.assert_array_equal(value, indices)


def test_public_diffusion_export_preserves_mapped_snapshot_when_republished(saved_pipelines, tmp_path):
    import numpy as np

    from dew.interop import load_pretrained

    source = load_pretrained(saved_pipelines / "sd", dtype="float32", attention_impl="xla")
    original = {layout.name: layout.export(source.variables) for layout in source.weight_layouts}
    destination = tmp_path / "export"
    source.save(destination)
    mapped = load_pretrained(destination, dtype="float32", attention_impl="xla")
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
    latest = load_pretrained(destination, dtype="float32", attention_impl="xla")
    for layout in latest.weight_layouts:
        expected = original[layout.name] + np.float32(1.) if layout.name == changed.name else original[layout.name]
        np.testing.assert_array_equal(layout.export(latest.variables), expected)


@pytest.mark.parametrize("family", ["sd", "xl", "sd3", "flux"])
def test_conditioner_rebuild_uses_supplied_weights_without_reading_any_source_shard(
        saved_pipelines, tmp_path, monkeypatch, family):
    from dataclasses import replace

    import jax
    import jax.numpy as jnp
    import numpy as np

    from dew.inputs.diffusion import DiffusionConditioner
    from dew.inputs.encoders import rebuild
    from dew.interop import diffusion, load_pretrained

    if family in ("sd3", "flux"):
        with tarfile.open(ROOT / f"tests/fixtures/{family}_source.tar.xz") as archive:
            archive.extractall(tmp_path, filter="data")
        directory = tmp_path / "pipeline"
    else:
        directory = saved_pipelines / family
    source = load_pretrained(directory, dtype="float32", attention_impl="xla")
    assert source.inputs is not None
    encoder = source.inputs.conditions["conditioning"].encoder
    assert isinstance(encoder, DiffusionConditioner)
    tokens = encoder.tokenize(["a red bird"])
    from fnmatch import fnmatch
    from types import SimpleNamespace

    import huggingface_hub

    cache = tmp_path / "hub" / ("a" * 40)
    selected = set(encoder.names)
    if encoder.t5 is not None:
        selected.add(encoder.t5.name)
    supplied = False

    def snapshot(repo_id, *, revision=None, allow_patterns=None, dry_run=False):
        if dry_run:
            return [SimpleNamespace(filename=path.relative_to(directory).as_posix())
                    for path in directory.rglob("*") if path.is_file()]
        # Model the Hub storage boundary: forbidden weight files cannot be
        # transferred even if the caller would never open them afterward.
        for source_file in directory.rglob("*"):
            if not source_file.is_file():
                continue
            relative = source_file.relative_to(directory)
            if not any(fnmatch(relative.as_posix(), pattern) for pattern in allow_patterns):
                continue
            if source_file.suffix == ".safetensors" and (supplied or relative.parts[0] not in selected):
                raise AssertionError(f"unrequested weight download: {relative}")
            destination = cache / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, destination)
        return str(cache)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    ordinary = DiffusionConditioner.from_pretrained("fixture/conditioner", dtype="float32", revision="main")
    for actual, reference in zip(jax.tree.leaves(ordinary.encode(ordinary.params, tokens)),
                                 jax.tree.leaves(encoder.encode(encoder.params, tokens)), strict=True):
        np.testing.assert_array_equal(actual, reference)

    saved = jax.tree.map(lambda leaf: (jnp.asarray(leaf) + 0.015625).astype(jnp.bfloat16), encoder.params)
    expected = replace(encoder, params=saved).encode(saved, tokens)

    def forbid_weights(*args, **kwargs):
        raise AssertionError("conditioner reconstruction opened source weights")

    monkeypatch.setattr(diffusion, "component_tensors", forbid_weights)
    supplied = True
    rebuilt = rebuild("diffusion_text", {**encoder.to_json(), "checkpoint": "fixture/conditioner",
                                         "revision": "main", "param_dtype": "float32"}, params=saved)
    actual = rebuilt.encode(rebuilt.params, tokens)

    for reference, value in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(value, reference)
    for actual, original in zip(jax.tree.leaves(rebuilt.params), jax.tree.leaves(saved), strict=True):
        assert actual.dtype == original.dtype == jnp.bfloat16
        np.testing.assert_array_equal(actual, original)

