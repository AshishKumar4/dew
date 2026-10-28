"""Native diffusion against authentic saved checkpoint/config/tokenizer oracles.

References run Diffusers 0.34.0 / Transformers 4.49.0 in isolation. Tests here
run Dew-native models, preprocessing, Process and solvers on the current stack.
Each worker process releases its JAX executable caches after the comparison.
"""
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
from interop_support import extract_fixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def saved_pipelines(tmp_path_factory):
    return extract_fixture(ROOT / "tests/fixtures/tiny_diffusers.tar.xz",
                           tmp_path_factory.mktemp("saved-diffusion"))


def run_check(*arguments, failing: bool = False) -> str:
    """The checker's output; it must exit 0, or, where `failing` admits a reported failure, 1."""
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
    assert result.returncode in ((0, 1) if failing else (0,)), result.stdout + result.stderr
    return result.stdout


STREAMED = ("sd", "xl", "safety")
"""The flows that also load their pipeline streamed onto a mesh a leaf at a time
(tests/test_pipeline_streaming.py) and hold it to the source beside the whole load."""


@pytest.mark.parametrize(
    "task", ["sd", "xl", "img2img", "inpaint", "xl-img2img", "xl-inpaint", "refiner", "safety"]
)
def test_native_bundle_trajectory_update_and_source_roundtrip(saved_pipelines, task):
    run_check(saved_pipelines / task, *(["--streamed"] if task in STREAMED else []))


SCHEDULER_GRIDS = ROOT / "tests/fixtures/diffusers_pipeline_schedulers.npz"
EVALUATION_SEEDS = 8
"""The pipeline evaluation's declared draws (check 1), sized by `seeds_for` paired with the evaluation on
armada."""


def grid_cases() -> list[str]:
    """The source schedulers tools/diffusers_scheduler_pipeline_reference.py
    walked the saved SD pipeline with, one config each in SCHEDULER_GRIDS."""
    with np.load(SCHEDULER_GRIDS) as grids:
        return sorted(name.removesuffix(".config") for name in grids.files if name.endswith(".config"))


@pytest.fixture(scope="module")
def walked_grids(saved_pipelines) -> dict:
    """Every source grid walked by the saved SD pipeline, loaded once in one process, each case's
    errors and, where it is off the source, its failure (`check_grids`)."""
    printed = run_check(saved_pipelines / "sd", "--grids", SCHEDULER_GRIDS, failing=True)
    return json.loads(printed.strip().splitlines()[-1])["grids"]


@pytest.mark.parametrize("solver", grid_cases())
def test_native_solver_consumes_the_complete_source_grid(walked_grids, solver):
    assert "failed" not in walked_grids[solver], walked_grids[solver]["failed"]


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
    # Saved over itself, the directory holds the latest export alone, under names no earlier index
    # held.
    earlier: set[str] = set()
    for save in range(3):
        source.save(destination, max_shard_size=20_000)
        files = [path.relative_to(destination).as_posix() for path in destination.rglob("*")
                 if path.is_file()]
        # What Diffusers downloads of the directory, published as it is.
        downloaded, _ = variant_compatible_siblings(files)
        for component, stem in (("unet", "diffusion_pytorch_model"), ("vae", "diffusion_pytorch_model"),
                                ("text_encoder", "model")):
            index = json.loads((destination / component / f"{stem}.safetensors.index.json").read_text())
            shards = {f"{component}/{name}" for name in index["weight_map"].values()}
            written = {file for file in files if file.startswith(f"{component}/{stem}-")}
            assert len(shards) > 1 and written == shards
            assert not (destination / component / f"{stem}.safetensors").exists()
            assert shards <= set(downloaded) if save == 0 else not shards & earlier
            earlier |= shards
        reloaded = Pretrained.load(destination, dtype="float32", attention_impl="xla")
        for layout in reloaded.weight_layouts:
            np.testing.assert_array_equal(layout.export(reloaded.variables), layout.export(source.variables))


def test_a_resized_xl_conditioning_rebuilds_from_its_record_at_its_own_size(saved_pipelines):
    """An XL pipeline loaded at 64x48 instead of its own 32x32 states that size in its time ids, and
    the conditioner its run record rebuilds states the same, where it used to read the pipeline's
    own size back."""
    from dew.inputs import Condition
    from dew.interop.pretrained import load_diffusion_source
    from dew.registry import to_record

    source = load_diffusion_source(str(saved_pipelines / "xl"), dtype="float32", attention_impl="xla",
                                   size=(64, 48))
    condition = source.inputs.conditions["conditioning"]
    assert condition.encoder.time_id_values == (64, 48, 0, 0, 64, 48)
    assert source.inputs.sample.shape[:2] == (64, 48)
    reloaded = Condition.bound(to_record(condition, Condition), params=condition.encoder.params)
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


@pytest.mark.parametrize("scheduler", ["shipped", "lms"])
def test_an_objective_on_a_loaded_pipeline_samples_on_the_pipelines_own_grid(saved_pipelines, tmp_path,
                                                                              scheduler):
    """A loaded pipeline samples on its scheduler's own grid and ends as its solver does, without a
    closing denoise (`SourceTask.grid`), which its training process's times and finish do not give;
    an LMS scheduler's solver runs on that grid alone. An objective trained from the pipeline, built
    directly or by a run config, keeps both: its evaluation is the solver's walk over that grid from
    the same noise, whatever process trains, its task draws the pipeline's own task's images, and the
    grid a saved run records rebuilds a task that draws them too."""
    import dataclasses
    import shutil

    import jax
    import jax.numpy as jnp
    from diffusion_stubs import PROMPTS, batch_for
    from diffusion_support import images_of
    from reference_error import as_given, assert_as_exact, widened

    from dew.config import ObjectiveConfig
    from dew.config.diffusion import DiffusionRunConfig
    from dew.diffusion.schedules import DiscreteNoiseScheduler
    from dew.inference.images import ImageSettings
    from dew.interop.pretrained import load_diffusion_source
    from dew.objectives import Step
    from dew.objectives.diffusion import DiffusionObjective
    from dew.registry import from_record, to_record
    from dew.sampling import sample

    directory = tmp_path / "sd"
    shutil.copytree(saved_pipelines / "sd", directory)
    if scheduler == "lms":
        with np.load(SCHEDULER_GRIDS) as grids:
            (directory / "scheduler" / "scheduler_config.json").write_text(str(grids["lms.config"]))
    source = load_diffusion_source(str(directory), dtype="float32", attention_impl="xla")
    size = source.inputs.sample.shape[0]
    settings = {"guidance": None, "steps": 3, "ema_decay": None}
    objective = DiffusionObjective(source, **settings)
    other = DiscreteNoiseScheduler(np.linspace(1e-3, 0.2, 20), p2_loss_weight_gamma=0)
    retrained = DiffusionObjective(source, **settings,
                                   process=dataclasses.replace(source.process, schedule=other))
    model = DiffusionRunConfig().model
    configured = DiffusionRunConfig(
        pretrained=str(directory), preset=None,
        model=dataclasses.replace(model, fields={**model.fields, "dtype": "float32",
                                                 "attention_impl": "xla"}),
        data=images_of("tfds://oxford_flowers102", size), val_metrics=(),
        objective=ObjectiveConfig("diffusion", {**settings, "unconditional_prob": 0.0})).build()
    variables = source.variables
    batch = batch_for(objective, size)
    rows = batch["image"].shape[0]

    def stepped(seed):
        """A seed's step and the noise its evaluation draws from the step's key, which the walks start
        from too."""
        key = jax.random.fold_in(jax.random.PRNGKey(4), int(seed))
        x_T = source.task.grid(3)[0].noise(jax.random.split(key)[0], (rows, *objective.latent_shape))
        return Step(jnp.asarray(0), key, None), x_T

    def walked(source, objective, variables, step, x_T):
        """The pipeline's solver over its own grid from `x_T`, with no closing denoise, decoded."""
        process, times = source.task.grid(3)
        denoise = objective.denoiser(variables, objective.encoded_conditions(variables, batch), None, process)
        latents = sample(denoise, x_T, solver=source.schedule.solver, key=jax.random.split(step.key)[1],
                         times=np.asarray(times), final_denoise=False)
        return jnp.clip(source.autoencoder.decode(variables["autoencoder"], latents), -1.0, 1.0)

    def evaluation(seed):
        return objective.evaluate(variables, batch, stepped(seed)[0]).images

    def reference(seed):
        return walked(source, objective, variables, *stepped(seed))

    def truth(seed):
        # The seed's noise drawn once in float32, as the arms draw it, then widened: drawn under x64, the
        # process would draw other noise and walk another trajectory, and the gate runs every truth under
        # x64 (`reference_error.truth_of`), so the draw turns it off (row 2064). The denoiser in float64
        # over its weights widened; the text encoder and the decoder, whose roundings the evaluation and
        # the reference make alike, stay as they are.
        with jax.enable_x64(new_val=False):
            step, x_T = stepped(seed)
        assert x_T.dtype == jnp.float32, x_T.dtype
        with jax.enable_x64(new_val=True):
            wide = dataclasses.replace(source, model=source.model.clone(dtype=jnp.float64))
            return np.asarray(walked(wide, DiffusionObjective(wide, **settings), widened(variables), step,
                                     jnp.asarray(x_T, jnp.float64)))

    # On par with the solver's own walk from the same noise over the seeds' step keys (check 1).
    assert_as_exact(as_given(evaluation), as_given(reference), lambda seed: (np.int32(seed),),
                    "an evaluation on the pipeline's grid", seeds=EVALUATION_SEEDS, truth_fn=as_given(truth))
    step = stepped(0)[0]
    evaluated = evaluation(0)
    for built in (retrained, configured):
        np.testing.assert_array_equal(built.evaluate(variables, batch, step).images, evaluated)

    own = source.text_to_image()(PROMPTS, steps=3, guidance=None, key=0).host().images
    task = objective.build_task(variables)
    np.testing.assert_array_equal(task(PROMPTS, steps=3, guidance=None, key=0).host().images, own)
    from dew.diffusion.schedules.source import SourceGrid

    recorded = from_record(ImageSettings, to_record(configured.task_settings(), ImageSettings))
    rebuilt = dataclasses.replace(task, grid=from_record(SourceGrid, recorded.grid))
    np.testing.assert_array_equal(rebuilt(PROMPTS, steps=3, guidance=None, key=0).host().images, own)
