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

import numpy as np
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


SCHEDULER_GRIDS = ROOT / "tests/fixtures/diffusers_pipeline_schedulers.npz"


def grid_cases() -> list[str]:
    """The source schedulers tools/diffusers_scheduler_pipeline_reference.py
    walked the saved SD pipeline with, one config each in SCHEDULER_GRIDS."""
    with np.load(SCHEDULER_GRIDS) as grids:
        return sorted(name.removesuffix(".config") for name in grids.files if name.endswith(".config"))


@pytest.mark.parametrize("solver", grid_cases())
def test_native_solver_consumes_the_complete_source_grid(saved_pipelines, solver):
    run_check(saved_pipelines / "sd", "--grids", SCHEDULER_GRIDS, "--case", solver)


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


def test_a_diffusion_export_shards_each_component_at_the_bound_it_is_given(saved_pipelines, tmp_path):
    """Saved with a shard bound below its components' sizes, a pipeline writes each component's
    weights as shards and an index under the stem and shard names its library reads (the
    denoiser's and the VAE's `diffusion_pytorch_model`, the text encoder's `model`), which
    Diffusers' Hub download keeps, no single file beside them, and loads back to the same
    tensors."""
    import json

    import numpy as np

    from dew.interop import Pretrained
    from tools.diffusers_reference_helpers import restore_transformers_names

    restore_transformers_names()
    from diffusers.pipelines.pipeline_loading_utils import variant_compatible_siblings

    source = Pretrained.load(saved_pipelines / "sd", dtype="float32", attention_impl="xla")
    destination = tmp_path / "export"
    source.save(destination, max_shard_size=20_000)
    # What Diffusers downloads of the export, published as `push_to_hub` stages it.
    downloaded, _ = variant_compatible_siblings(
        [path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()])
    for component, stem in (("unet", "diffusion_pytorch_model"), ("vae", "diffusion_pytorch_model"),
                            ("text_encoder", "model")):
        index = json.loads((destination / component / f"{stem}.safetensors.index.json").read_text())
        shards = {f"{component}/{name}" for name in index["weight_map"].values()}
        assert len(shards) > 1 and shards <= set(downloaded)
        assert not (destination / component / f"{stem}.safetensors").exists()
    reloaded = Pretrained.load(destination, dtype="float32", attention_impl="xla")
    for layout in reloaded.weight_layouts:
        np.testing.assert_array_equal(layout.export(reloaded.variables), layout.export(source.variables))


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


def test_a_bfloat16_safety_head_decides_as_its_checkpoint_states(saved_pipelines, tmp_path):
    """The safety head keeps its scoring state, the concept embeddings and
    thresholds, in FP32 whatever the weights are stored in, so a load at
    `param_dtype="bfloat16"` blocks and allows what the checkpoint's own
    thresholds decide. The fixture's checker, all ones, allows the reference
    image, as the reference recorded. A second checker sets one concept to the
    reference's own embedding of that image at threshold 0.9985, between the
    bf16 values 0.99609375 and 1.0: the published rule rounds its score,
    cosine 1 less 0.9985, to 0.002 and blocks the image, where a threshold
    narrowed to 1.0 scores 0 and allows it. The image's own direction moves
    the cosine only at second order in the bf16 tower's error, far inside the
    0.001 left before the score rounds to zero."""
    import shutil

    import jax.numpy as jnp
    from safetensors.numpy import load_file, save_file

    from dew.interop import Pretrained

    source = saved_pipelines / "safety"
    reference = np.load(source / "reference.npz")
    concept = tmp_path / "safety"
    shutil.copytree(source, concept)
    checker = concept / "safety_checker" / "model.safetensors"
    tensors = load_file(str(checker))
    embeds, thresholds = tensors["concept_embeds"].copy(), tensors["concept_embeds_weights"].copy()
    embeds[0], thresholds[0] = reference["checker_embeddings"][0], 0.9985
    save_file({**tensors, "concept_embeds": embeds, "concept_embeds_weights": thresholds}, str(checker),
              metadata={"format": "pt"})
    images = jnp.asarray(reference["images"] * 2 - 1)
    for name, directory, expected in (("fixture", source, reference["checker_flags"]),
                                      ("concept", concept, [True])):
        for storage in ("float32", "bfloat16"):
            loaded = Pretrained.load(str(directory), dtype="float32", param_dtype=storage,
                                     attention_impl="xla")
            blocked = np.all(np.asarray(loaded.finish(loaded.variables, images)) == -1, axis=(1, 2, 3))
            np.testing.assert_array_equal(blocked, expected, err_msg=f"the {name} checker at {storage}")
