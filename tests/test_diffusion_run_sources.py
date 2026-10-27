"""What `DiffusionRunConfig` reaches beyond a scratch model on CLIP text: a
published pipeline to fine-tune, a published family trained from scratch on
its own text towers, and Flow-GRPO on a reward.

The pipelines are the committed tiny SD3, Flux and Qwen-Image sources, whose
forwards the family tests match against diffusers.
"""

import dataclasses
import json
import math
import shutil
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from diffusion_stubs import PROMPTS, batch_for
from diffusion_support import images_of
from interop_support import allocated_bytes
from reference_error import as_given, assert_as_exact

from dew.config import ObjectiveConfig, TrainerConfig
from dew.config.diffusion import DiffusionRunConfig, FlowRolloutConfig
from dew.diffusion.presets import Flow, ResolutionShift
from dew.diffusion.process import Process
from dew.diffusion.schedules.source import SourceSchedule
from dew.inference import TextToImage
from dew.inputs import TextCondition
from dew.objectives import Step
from dew.objectives.diffusion import Alignment, DiffusionObjective
from dew.objectives.diffusion.end_to_end import AUTOENCODER, LATENT_STATS, TUNED_WEIGHTS, EndToEnd
from dew.objectives.rl.flow import OLD_LOG_PROBS_KEY, REWARDS_KEY, FlowGRPOObjective
from dew.registry import ModelConfig, to_record
from dew.sampling import Euler
from dew.training import Trainer

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def pipelines(tmp_path_factory):
    """Each family's pipeline at `<family>/pipeline`: the flow transformers, and the tiny SD UNet
    pipeline, whose DDIM grid ends short of a clean sample."""
    root = tmp_path_factory.mktemp("pipelines")
    for family in ("sd3", "flux", "qwen_image"):
        with tarfile.open(FIXTURES / f"{family}_source.tar.xz") as archive:
            archive.extractall(root / family, filter="data")
    with tarfile.open(FIXTURES / "tiny_diffusers.tar.xz") as archive:
        archive.extractall(root / "unet", members=[member for member in archive.getmembers()
                                                   if member.name.split("/")[0] == "sd"], filter="data")
    (root / "sd").mkdir()
    (root / "unet" / "sd").rename(root / "sd" / "pipeline")
    return root


def precision() -> ModelConfig:
    """The default model record at float32 on the XLA kernel."""
    model = DiffusionRunConfig().model
    return dataclasses.replace(model, fields={**model.fields, "dtype": "float32", "attention_impl": "xla"})


def test_a_shape_only_pipeline_load_retains_geometry_without_eager_native_copies(pipelines, tmp_path,
                                                                               monkeypatch):
    """A shape-only load preserves every component's parameter geometry and storage dtype.
    Checkpoint validation still requires valid safetensors files. A delegating NumPy allocation
    recorder measures host buffers, not implementation calls: deferred linear leaves allocate fewer
    native-layout bytes than an eager load of the same already-warmed pipeline."""
    from dew.interop import Pretrained

    original = pipelines / "sd3" / "pipeline"
    expected = Pretrained.load(original, dtype="float32", attention_impl="xla", on_host=True)
    selected = tmp_path / "source"
    shutil.copytree(original, selected)
    _, eager_bytes = allocated_bytes(monkeypatch, lambda: Pretrained.load(
        selected, dtype="float32", attention_impl="xla", on_host=True))
    loaded, deferred_bytes = allocated_bytes(monkeypatch, lambda: Pretrained.load(
        selected, dtype="float32", attention_impl="xla", weights=False))
    assert eager_bytes > 0 and deferred_bytes < eager_bytes
    shapes = jax.tree.leaves(loaded.variables)
    assert all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in shapes)
    assert jax.tree.structure(loaded.variables) == jax.tree.structure(expected.variables)
    for actual, wanted in zip(shapes, jax.tree.leaves(expected.variables), strict=True):
        assert actual.shape == wanted.shape and actual.dtype == wanted.dtype


def value(objective, params, batch) -> float:
    loss, _ = objective.loss(params, batch, Step(jnp.asarray(0), jax.random.PRNGKey(5), None))
    return float(loss.total / loss.mass)


@pytest.mark.parametrize("family", ["sd3", "flux", "qwen_image"])
def test_a_run_fine_tunes_a_published_pipeline_and_rebinds_it_without_weights(
        family, pipelines, tmp_path):
    """The run trains the pipeline's own transformer under its own text
    conditioning, and a saved tree rebuilds over the directory's metadata
    alone: the rebuilt objective scores the trained weights exactly, from a
    copy of the pipeline with every weight file deleted. Rebound over the
    published pipeline, the tree exports through the layouts its source's
    headers give, which are the loaded source's own, and reloads leaf for
    leaf; the copy without weight files has none to export through, and its
    save writes nothing."""
    directory = pipelines / family / "pipeline"
    size = 16 if family != "qwen_image" else 32
    config = DiffusionRunConfig(pretrained=str(directory), preset=None, model=precision(),
                                data=images_of("tfds://oxford_flowers102", size),
                                    val_metrics=(),
                                objective=ObjectiveConfig("diffusion",
                                                          {"solver": Euler(), "guidance": None, "steps": 2,
                                                           "unconditional_prob": 0.0, "ema_decay": None}))
    objective = config.build()
    assert set(objective.inputs.conditions) == {"conditioning"}
    batch = batch_for(objective, size)
    trainer = Trainer(objective, optax.sgd(1e-2), key=jax.random.PRNGKey(3))
    initial = trainer.initial_state()
    before = value(objective, initial.variables, batch)
    state, _, _, _, accepted, _ = trainer.compile(initial, batch)(initial, batch)
    assert bool(accepted)
    trained = value(objective, state.variables, batch)
    assert trained < before

    metadata = tmp_path / "metadata"
    shutil.copytree(directory, metadata, ignore=shutil.ignore_patterns("*.safetensors"))
    rebuilt = dataclasses.replace(config, pretrained=str(metadata)).build(variables=state.variables)
    assert value(rebuilt, state.variables, batch) == trained

    # The saved tree rebound over the published pipeline writes its trained
    # weights back by the layouts the source's headers give, and the tensors
    # no leaf holds (Qwen-Image's vision tower) as the source stores them.
    from dew.interop.pretrained import load_diffusion_source
    from dew.interop.safetensors_io import read_weights

    def rebound(source: Path):
        return load_diffusion_source(str(source), dtype="float32", attention_impl="xla", size=(size, size),
                                     variables=state.variables)

    with pytest.raises(ValueError):
        rebound(metadata).save(tmp_path / "refused")
    assert not (tmp_path / "refused").exists()
    published = load_diffusion_source(str(directory), dtype="float32", attention_impl="xla",
                                      size=(size, size))
    pipeline = rebound(directory)
    assert sorted(pipeline.weight_layouts, key=lambda layout: layout.name) == sorted(
        published.weight_layouts, key=lambda layout: layout.name)
    pipeline.save(tmp_path / "export")
    again = load_diffusion_source(str(tmp_path / "export"), dtype="float32", attention_impl="xla",
                                  size=(size, size))
    assert jax.tree.structure(again.variables) == jax.tree.structure(pipeline.variables)
    for ours, theirs in zip(jax.tree.leaves(again.variables), jax.tree.leaves(pipeline.variables),
                            strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
    assert set(pipeline.retained_tensors) == set(published.retained_tensors)
    for name in published.retained_tensors:
        component, _, stored = name.partition("/")
        written, source = (read_weights(root / component)[stored]
                           for root in (tmp_path / "export", directory))
        assert written.dtype == source.dtype and written.tobytes() == source.tobytes(), name


_OFFLINE_REBIND = """
import json, sys
from dew.interop.pretrained import load_diffusion_source

published = load_diffusion_source(sys.argv[1], dtype="float32", attention_impl="xla", size=(16, 16))
pipeline = load_diffusion_source("dew/flux-tiny", dtype="float32", attention_impl="xla", size=(16, 16),
                                 variables=published.variables)
try:
    pipeline.save(sys.argv[2])
except ValueError as refused:
    print(json.dumps({"saved": False, "refused": str(refused)}))
else:
    print(json.dumps({"saved": True}))
"""


@pytest.mark.parametrize("cached", ["whole", "metadata"])
def test_a_pipeline_rebinds_saved_variables_offline_and_exports_them_by_the_headers_it_holds(
        pipelines, tmp_path, cached):
    """A saved tree rebound over a Hub pipeline needs no Hub request: offline, over a cache
    holding the whole pipeline at one commit, it binds and exports by the headers of the files
    the cache holds, and the export reloads leaf for leaf; over a cache holding the commit's
    listing and configs but no weight file, it binds, and its export, which has no header to
    write by, refuses and writes nothing."""
    import os
    import subprocess
    import sys

    from interop_support import hub_cache

    from dew.interop.pretrained import load_diffusion_source

    directory = pipelines / "flux" / "pipeline"
    commit = hub_cache(tmp_path / "hub", "dew/flux-tiny", directory)
    snapshot = tmp_path / "hub" / "models--dew--flux-tiny" / "snapshots" / commit
    if cached == "metadata":
        for weights in snapshot.rglob("*.safetensors"):
            weights.unlink()
    env = {**os.environ, "HF_HUB_CACHE": str(tmp_path / "hub"), "HF_HUB_OFFLINE": "1",
           "DEW_HUB_ANSWERS": str(tmp_path / "answers"),
           "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    done = subprocess.run([sys.executable, "-c", _OFFLINE_REBIND, str(directory), str(tmp_path / "export")],
                          env=env, check=False, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-3000:]
    [outcome] = [json.loads(line) for line in done.stdout.split("\n") if line.startswith("{")]
    assert outcome["saved"] == (cached == "whole"), outcome
    if cached == "metadata":
        assert not (tmp_path / "export").exists()
        return
    published = load_diffusion_source(str(directory), dtype="float32", attention_impl="xla", size=(16, 16))
    again = load_diffusion_source(str(tmp_path / "export"), dtype="float32", attention_impl="xla",
                                  size=(16, 16))
    assert jax.tree.structure(again.variables) == jax.tree.structure(published.variables)
    for ours, theirs in zip(jax.tree.leaves(again.variables), jax.tree.leaves(published.variables),
                            strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))


def test_a_pipeline_rebinds_over_a_sharded_copy_without_its_shards(pipelines, tmp_path):
    """A local copy of a pipeline whose transformer is sharded behind its index, with the shards
    deleted, rebinds a saved tree from its metadata alone, as it does unsharded, and its save,
    with no header to write by, refuses and writes nothing."""
    from dew.interop.pretrained import load_diffusion_source
    from dew.interop.safetensors_io import read_weights, save_sharded

    copy = tmp_path / "sharded"
    shutil.copytree(pipelines / "flux" / "pipeline", copy)
    weights = read_weights(copy / "transformer")
    for stored in (copy / "transformer").glob("*.safetensors"):
        stored.unlink()
    save_sharded(weights, copy / "transformer", "20KB", stem="diffusion_pytorch_model")
    assert len(list((copy / "transformer").glob("*.safetensors"))) > 1
    published = load_diffusion_source(str(copy), dtype="float32", attention_impl="xla", size=(16, 16))
    for stored in copy.rglob("*.safetensors"):
        stored.unlink()
    rebound = load_diffusion_source(str(copy), dtype="float32", attention_impl="xla", size=(16, 16),
                                    variables=published.variables)
    for ours, theirs in zip(jax.tree.leaves(rebound.variables), jax.tree.leaves(published.variables),
                            strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
    with pytest.raises(ValueError):
        rebound.save(tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


_HUB_PIPELINE = ("hf-internal-testing/tiny-flux-pipe", "a98bc4ec6a80c47c477ed22d7e81c79872d9c723")

_ONLINE_PUBLISHED = """
import sys
import jax, numpy as np
from flax import serialization
from dew.interop.pretrained import load_diffusion_source

published = load_diffusion_source(sys.argv[1], revision=sys.argv[2], dtype="float32", attention_impl="xla",
                                  size=(16, 16))
open(sys.argv[3], "wb").write(serialization.msgpack_serialize(jax.tree.map(np.asarray, published.variables)))
"""

_REBOUND = """
import json, sys
from flax import serialization
from dew.interop.pretrained import load_diffusion_source

variables = serialization.msgpack_restore(open(sys.argv[3], "rb").read())
pipeline = load_diffusion_source(sys.argv[1], revision=sys.argv[2], dtype="float32", attention_impl="xla",
                                 size=(16, 16), variables=variables)
pipeline.save(sys.argv[4])
print(json.dumps({"saved": True}))
"""


def test_a_pipeline_is_one_whatever_its_root_config_unless_that_names_a_model(pipelines, tmp_path):
    """A pipeline directory loads as its components whatever else its root holds, a config.json too,
    for a run as for a load (`_pipeline_source`); one whose config.json names a model is that model,
    a decoder shipping an index for its sampler, and a run training a pipeline refuses it."""
    from dew.interop.pretrained import load_diffusion_source

    copy = tmp_path / "pipeline"
    shutil.copytree(pipelines / "flux" / "pipeline", copy)
    (copy / "config.json").write_text("{}")
    loaded = load_diffusion_source(str(copy), dtype="float32", attention_impl="xla", size=(16, 16))
    assert jax.tree.leaves(loaded.variables)
    (copy / "config.json").write_text(json.dumps({"model_type": "diffusion_gemma"}))
    with pytest.raises(ValueError):
        load_diffusion_source(str(copy), dtype="float32", attention_impl="xla", size=(16, 16))


@pytest.mark.network
def test_a_hub_pipeline_rebound_online_exports_offline_from_its_recorded_headers(tmp_path):
    """A saved tree rebound over a Hub pipeline whose weights it never downloads reads the
    export's headers and file listing over HTTP and records them (`hub_answer`); an offline
    process over a cache of the same commit's metadata alone then rebinds and exports by those
    records, and the export reloads leaf for leaf as the published pipeline."""
    import os
    import subprocess
    import sys

    from dew.interop.pretrained import load_diffusion_source

    repo, revision = _HUB_PIPELINE
    src = str(Path(__file__).resolve().parents[1] / "src")

    def run(script, cache, *arguments, offline=False):
        env = {**os.environ, "HF_HUB_CACHE": str(tmp_path / cache), "PYTHONPATH": src,
               "DEW_HUB_ANSWERS": str(tmp_path / "answers"), "HF_HUB_OFFLINE": "1" if offline else "0"}
        done = subprocess.run([sys.executable, "-c", script, repo, revision, *map(str, arguments)], env=env,
                              check=False, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr[-3000:]

    run(_ONLINE_PUBLISHED, "whole", tmp_path / "variables.msgpack")
    run(_REBOUND, "metadata", tmp_path / "variables.msgpack", tmp_path / "online")
    assert not list((tmp_path / "metadata").rglob("*.safetensors"))
    run(_REBOUND, "metadata", tmp_path / "variables.msgpack", tmp_path / "offline", offline=True)
    published = load_diffusion_source(repo, revision=revision, dtype="float32", attention_impl="xla",
                                      size=(16, 16))
    again = load_diffusion_source(str(tmp_path / "offline"), dtype="float32", attention_impl="xla",
                                  size=(16, 16))
    assert jax.tree.structure(again.variables) == jax.tree.structure(published.variables)
    for ours, theirs in zip(jax.tree.leaves(again.variables), jax.tree.leaves(published.variables),
                            strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))


def test_a_pretrained_pipeline_trains_its_autoencoder_end_to_end(pipelines, tmp_path):
    """A recipe trains the loaded KL posterior and denoiser and publishes the tuned autoencoder,
    and the trained tree rebuilds over the pipeline's metadata alone, scoring as the run did."""
    with tarfile.open(FIXTURES / "rae.tar.xz") as archive:
        archive.extractall(tmp_path / "representation", filter="data")
    alignment = Alignment("proj_out", source=str(tmp_path / "representation" / "dinov2_plain"),
                          resolution=56, width=8)
    config = DiffusionRunConfig(
        pretrained=str(pipelines / "sd3" / "pipeline"), preset=None, model=precision(), val_metrics=(),
        data=images_of("tfds://oxford_flowers102", 16),
        objective=ObjectiveConfig("diffusion", {
            "solver": Euler(), "guidance": None, "steps": 2, "ema_decay": None,
            "alignment": alignment, "end_to_end": EndToEnd(perceptual_weight=0., discriminator_weight=0.)}))
    objective = config.build()
    batch = batch_for(objective, 16)
    trainer = Trainer(objective, optax.sgd(1e-3), key=jax.random.key(3))
    initial = trainer.initial_state()
    before = jax.tree.map(np.asarray, initial.variables)
    trained, _, _, _, accepted, _ = trainer.compile(initial, batch)(initial, batch)
    assert bool(accepted) and int(trained.updates) == 1
    for name in (AUTOENCODER, "proj_out"):
        assert any(not np.array_equal(start, end) for start, end in zip(
            jax.tree.leaves(before["params"][name]),
            jax.tree.leaves(trained.variables["params"][name]), strict=True))
    assert not np.array_equal(before[LATENT_STATS]["mean"], trained.variables[LATENT_STATS]["mean"])
    task = TextToImage.from_objective(objective, trained.variables)
    for actual, expected in zip(jax.tree.leaves(task.variables["autoencoder"][TUNED_WEIGHTS]),
                                 jax.tree.leaves(trained.variables["params"][AUTOENCODER]), strict=True):
        np.testing.assert_array_equal(actual, expected)

    metadata = tmp_path / "metadata"
    shutil.copytree(pipelines / "sd3" / "pipeline", metadata, ignore=shutil.ignore_patterns("*.safetensors"))
    rebuilt = dataclasses.replace(config, pretrained=str(metadata)).build(variables=trained.variables)
    assert value(rebuilt, trained.variables, batch) == value(objective, trained.variables, batch)


@pytest.mark.parametrize("family", ["sd3", "flux", "qwen_image", "sd"])
def test_a_run_config_builds_the_objective_the_python_api_builds(family, pipelines):
    """The wiring half of a configured run: `DiffusionRunConfig` over a
    published pipeline builds what `DiffusionObjective(Pretrained.load(...))`
    builds with the same settings at the data's resolution
    (`load_diffusion_source(size=)`): the same model, process (its
    resolution shift too), tokenized ids and autoencoder kind over the
    same initial variables, so both score a batch alike, through the VAE
    and the text towers. The loss the two share is held to Diffusers'
    train_dreambooth_flux.py statements on matched draws in
    tests/test_flow_matching.py (ad63cc6a). Both sample by the same policy, the
    pipeline's own grid and finish: their evaluations draw alike from one key,
    unlike an evaluation on the process's own times, and the run's published
    task draws the walk the pipeline's own task draws from the same noise. A
    flow grid ends at a clean sample, where a closing denoise moves nothing;
    the SD pipeline's DDIM grid ends short of one, where it would."""
    from dew.interop.pretrained import load_diffusion_source

    directory = pipelines / family / "pipeline"
    size = 16 if family in ("sd3", "flux") else 32
    settings = {"guidance": None, "unconditional_prob": 0.25, "ema_decay": None}
    configured = DiffusionRunConfig(pretrained=str(directory), preset=None, model=precision(),
                                    data=images_of("tfds://oxford_flowers102", size),
                                        val_metrics=(),
                                    objective=ObjectiveConfig("diffusion", {
                                        "solver": Euler(), "steps": 2, **settings})).build()
    source = load_diffusion_source(str(directory), dtype="float32", attention_impl="xla", size=(size, size))
    written = DiffusionObjective(source, solver=Euler(), steps=2, **settings)
    assert configured.model == written.model
    assert to_record(configured.process, Process) == to_record(written.process, Process)
    assert type(configured.autoencoder) is type(written.autoencoder)
    assert configured.inputs.sample == written.inputs.sample
    assert configured.inputs.conditions.keys() == written.inputs.conditions.keys()
    # Tokenizers are host objects without equality; they agree by the ids
    # they write, and the towers by what the loss below reads from them.
    for ours, theirs in zip(jax.tree.leaves(configured.inputs.tokenize(PROMPTS)),
                            jax.tree.leaves(written.inputs.tokenize(PROMPTS)), strict=True):
        np.testing.assert_array_equal(ours, theirs)
    states = [Trainer(objective, optax.sgd(1e-2), key=jax.random.PRNGKey(3)).initial_state()
              for objective in (configured, written)]
    leaves = [jax.tree.leaves(state.variables) for state in states]
    assert len(leaves[0]) == len(leaves[1])
    for ours, theirs in zip(*leaves, strict=True):
        np.testing.assert_array_equal(ours, theirs)
    batch = batch_for(configured, size)
    assert value(configured, states[0].variables, batch) == value(written, states[1].variables, batch)
    # Both sample by the pipeline's own grid and its solver's finish: from one key an evaluation of
    # either draws what the other's does, and not what the process's own times draw ...
    step = Step(jnp.asarray(0), jax.random.PRNGKey(5), None)
    drawn = [np.asarray(objective.evaluate(state.variables, batch, step).images)
             for objective, state in zip((configured, written), states, strict=True)]
    assert np.all(np.isfinite(drawn[0]))
    np.testing.assert_array_equal(drawn[0], drawn[1])
    on_times = DiffusionObjective(source, solver=Euler(), steps=2, grid=None, **settings)
    elsewhere = np.asarray(on_times.evaluate(states[1].variables, batch, step).images)
    assert not np.array_equal(drawn[0], elsewhere)
    # ... and the run's published task draws the walk the pipeline's own task draws from the same noise,
    # which the family's source tests hold to the source pipeline's own walk.
    prompts = list(PROMPTS)
    published = configured.build_task(states[0].variables)(prompts, key=7).host()
    pipeline = source.text_to_image()(prompts, steps=2, guidance=None, solver=Euler(), key=7).host()
    for name in ("latents", "images"):
        assert np.all(np.isfinite(getattr(published, name))), name
        np.testing.assert_array_equal(getattr(published, name), getattr(pipeline, name), err_msg=name)


def test_a_pretrained_run_refuses_a_preset_of_another_kind(pipelines):
    """Flux was trained as a velocity flow; the default EDM preset is not
    that convention, and a flow preset is."""
    common = {"pretrained": str(pipelines / "flux" / "pipeline"), "model": precision(),
              "data": images_of("tfds://oxford_flowers102", 16), "val_metrics": (),
              "objective": ObjectiveConfig("diffusion", {"solver": Euler(), "guidance": None, "steps": 2})}
    with pytest.raises(ValueError):
        DiffusionRunConfig(**common).build()
    assert DiffusionRunConfig(**common, preset=Flow(shift=3.0)).build() is not None


DRAWS = dict(np.load(FIXTURES / "flow" / "draws.npz"))


@pytest.mark.parametrize("family,preset", [
    ("sd3", Flow(shift=3.0)),
    ("flux", Flow(shift=1.0, density="uniform")),
], ids=["sd3", "flux"])
def test_a_flow_preset_reproduces_a_diffusers_scripts_default_draw_and_weighting(family, preset, pipelines):
    """tools/flow_draw_reference.py runs Diffusers 0.34.0's SD3 and Flux
    DreamBooth scripts' own draw-to-weighting statements, at their own
    defaults and over the published scheduler files, on this preset's
    draws (reflected, as the scripts' tables descend). The script's table
    index is a function of the draw, floor((1 - t) * 1000) in float32, and
    the preset's times give every index the script took; the scheduler's
    table at those indices is the script's noise level exactly, which the
    preset's continuous level reaches from below within one table step. The
    weights are equal, all one."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    schedule = preset().schedule
    # Drawn on the CPU the fixture's draws were made on: a GPU's normal draws and exponentials differ
    # from a CPU's in their last bits, and the script's table index is a function of the draw's bits.
    with jax.default_device(jax.devices("cpu")[0]):
        times = np.asarray(schedule.sample_t(jax.random.key(int(DRAWS["key"])),
                                             DRAWS[f"{family}/times"].size))
    np.testing.assert_array_equal(times, DRAWS[f"{family}/times"])
    indices = np.floor((np.float32(1) - times) * np.float32(1000)).astype(np.int64)
    np.testing.assert_array_equal(indices, DRAWS[f"{family}/indices"])
    config = json.loads((pipelines / family / "pipeline" / "scheduler" / "scheduler_config.json").read_text())
    table = FlowMatchEulerDiscreteScheduler.from_config(config).sigmas.numpy()
    np.testing.assert_array_equal(table[indices], DRAWS[f"{family}/sigmas"])

    wide = times.astype(np.float64)

    def shifted(t):
        return schedule.shift * t / (1 + (schedule.shift - 1) * t)

    above = DRAWS[f"{family}/sigmas"].astype(np.float64) - np.asarray(schedule.rates(times)[1], np.float64)
    step = shifted(np.minimum(wide + 1 / 1000, 1)) - shifted(wide)
    slack = 4 * np.finfo(np.float32).eps
    assert above.min() >= -slack and np.all(above <= step + slack), (above.min(), float((above - step).max()))
    np.testing.assert_array_equal(np.asarray(schedule.weight(times)), DRAWS[f"{family}/weighting"])


SHIFTS = dict(np.load(FIXTURES / "flow" / "resolution_shift.npz"))


def flux_shift(tokens: int, constants: str = "flux") -> float:
    """exp of Diffusers' Flux `calculate_shift` at `tokens` (tools/flux_shift_reference.py)."""
    return math.exp(SHIFTS[f"{constants}/mu"][list(SHIFTS["tokens"]).index(tokens)])


SHIFT_SEEDS = 16
"""Check 1's declared seeds for the shifted noise levels, sized by `seeds_for` (2026-10-21, on armada)."""


def shift_times(seed: int) -> tuple[np.ndarray]:
    """99 times in (0, 1), float32, as a schedule's grid holds them."""
    return (np.random.default_rng(seed).uniform(0.005, 0.995, size=99).astype(np.float32),)


@pytest.mark.parametrize("constants,fields", [
    ("flux", {}),
    ("long", {"max_tokens": 8192, "max_shift": 0.9}),
])
def test_the_resolution_shift_and_the_noise_levels_it_maps_to_are_diffusers(constants, fields):
    """At every token count, `ResolutionShift` and a loaded dynamic-shift
    scheduler file are exp(calculate_shift) to the bit, and the flow schedule
    at that shift maps each time to the noise level Diffusers'
    `FlowMatchEulerDiscreteScheduler.time_shift` gives it at mu, run here, on
    par with it over drawn times (check 1), for Flux's constants and a
    pipeline with another slope and span."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    top_shift, top = fields.get("max_shift", 1.15), fields.get("max_tokens", 4096)
    loaded = SourceSchedule.from_config({
        "_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000,
        "use_dynamic_shifting": True, "base_shift": 0.5, "max_shift": top_shift,
        "base_image_seq_len": 256, "max_image_seq_len": top})
    schedules, mus = {}, {}
    for index, tokens in enumerate(SHIFTS["tokens"]):
        resolution = ResolutionShift(tokens=int(tokens), **fields)
        shift = resolution.shift()
        assert shift == flux_shift(int(tokens), constants)
        schedule = loaded.training_process(int(tokens)).schedule
        assert schedule.shift == shift
        # A preset's resolution shift maps as the loaded file's does, to the bit.
        drawn, = shift_times(index)
        np.testing.assert_array_equal(Flow(resolution_shift=resolution)().schedule.rates(drawn)[1],
                                      schedule.rates(drawn)[1])
        schedules[f"{tokens} tokens"] = schedule
        mus[f"{tokens} tokens"] = float(SHIFTS[f"{constants}/mu"][index])
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)

    def dew(times):
        return {name: schedule.rates(times)[1] for name, schedule in schedules.items()}

    @as_given
    def diffusers(times):
        return {name: scheduler.time_shift(mu, 1.0, times) for name, mu in mus.items()}

    assert_as_exact(dew, diffusers, shift_times, f"{constants} noise levels", seeds=SHIFT_SEEDS)


def test_a_pretrained_run_stores_its_pipelines_weights_in_the_dtype_it_names(pipelines):
    """`pretrained_param_dtype` is the storage the pipeline's denoiser loads
    and trains in; with no pipeline it is refused rather than ignored."""
    objective = ObjectiveConfig("diffusion", {"guidance": None, "ema_decay": None})
    configured = DiffusionRunConfig(pretrained=str(pipelines / "flux" / "pipeline"), preset=None,
                                    model=precision(), pretrained_param_dtype="bfloat16",
                                    data=images_of("tfds://oxford_flowers102", 16),
                                        val_metrics=(),
                                    objective=objective).build()
    state = Trainer(configured, optax.sgd(1e-2), key=jax.random.PRNGKey(3)).initial_state()
    floating = [leaf for leaf in jax.tree.leaves(state.variables["params"])
                if jnp.issubdtype(leaf.dtype, jnp.floating)]
    assert floating and all(leaf.dtype == jnp.bfloat16 for leaf in floating)
    with pytest.raises(ValueError):
        DiffusionRunConfig(pretrained_param_dtype="bfloat16")


def test_a_pretrained_run_refuses_a_model_of_its_own():
    with pytest.raises(ValueError):
        DiffusionRunConfig(pretrained="some/pipeline", model=ModelConfig("simple_dit"),
                           text=TextCondition(encoder="t5"))


def test_a_published_family_trains_from_scratch_on_its_pipelines_text_towers(pipelines):
    """Flux from scratch in pixel space, conditioned by the Flux pipeline's
    CLIP and T5 towers: the conditioner's record goes in under `conditioning`,
    where the published family reads it, and ten steps on one batch lower
    its loss (one step draws its own noise, which a single step's loss on
    the evaluation's draw need not follow)."""
    directory = pipelines / "flux" / "pipeline"
    config = DiffusionRunConfig(
        model=ModelConfig("flux_transformer", {
            "in_channels": 12, "out_channels": 12, "num_layers": 1, "num_single_layers": 1,
            "heads": 2, "head_dim": 12, "joint_attention_dim": 16, "pooled_projection_dim": 10,
            "guidance_embeds": True, "axes_dims_rope": [4, 4, 4],
            "dtype": "float32", "attention_impl": "xla"}),
        data=images_of("tfds://oxford_flowers102", 8), preset=Flow(), val_metrics=(),
        text=TextCondition(encoder="diffusion_text", checkpoint=str(directory)),
        objective=ObjectiveConfig("diffusion",
                                  {"solver": Euler(), "guidance": None, "steps": 2, "ema_decay": None}))
    objective = config.build()
    assert set(objective.inputs.conditions) == {"conditioning"}
    batch = batch_for(objective, 8)
    trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.PRNGKey(0))
    initial = trainer.initial_state()
    before = value(objective, initial.variables, batch)
    step, state = trainer.compile(initial, batch), initial
    for _ in range(10):
        state, _, _, _, accepted, _ = step(state, batch)
        assert bool(accepted)
    assert value(objective, state.variables, batch) < before


def grpo_run(beta: float = 0.0, directory: str = "./checkpoints") -> DiffusionRunConfig:
    return DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 2, "emb_features": 16, "num_layers": 1,
                                         "num_heads": 2, "dtype": "float32", "attention_impl": "xla"}),
        data=images_of("tfds://oxford_flowers102", 4), preset=Flow(), val_metrics=(),
        trainer=TrainerConfig(checkpoint_dir=directory),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        objective=ObjectiveConfig("flow_grpo", {"clip_range": 0.2, "beta": beta, "solver": Euler(),
                                                "guidance": None, "steps": 2}),
        rollout=FlowRolloutConfig(reward="psnr", groups=2, steps=3))


def grpo_step(objective, rollout, **trainer):
    """One rollout over a batch and the policy step on its transitions."""
    trainer = Trainer(objective, optax.sgd(1e-2), key=jax.random.PRNGKey(1), rollout=rollout,
                      **trainer)
    initial = trainer.initial_state()
    prepared = rollout(initial, batch_for(objective, 4), jax.random.PRNGKey(2))
    started = [np.asarray(leaf) for leaf in jax.tree.leaves(initial.variables["params"])]
    state, _, _, _, accepted, _ = trainer.compile(initial, prepared)(initial, prepared)
    assert bool(accepted)
    return prepared, started, state


def test_flow_grpo_trains_from_the_run_config_on_a_registered_reward():
    """`rl` builds the Flow-GRPO objective and its rollout: groups of SDE
    samples per prompt, scored by PSNR against the prompt's own image, and
    one policy step on the transitions the rollout cut."""
    config = grpo_run()
    objective = config.build()
    assert isinstance(objective, FlowGRPOObjective) and config.rollout is not None
    prepared, started, state = grpo_step(objective, config.rollout.build(objective))
    assert prepared["latents"].shape[:2] == (jax.device_count() * 2, 2)
    assert np.all(np.isfinite(prepared["rewards"]))
    assert any(not np.array_equal(after, before) for after, before in zip(
        jax.tree.leaves(state.variables["params"]), started, strict=True))


def test_a_flow_grpo_runs_rollout_is_its_own_record_beside_the_objective():
    """A Flow-GRPO run's rollout is the run's `rollout` record, which the objective's record does
    not hold: a run naming none states the default, a rollout setting on the objective is no
    argument of it, and a rollout on a run whose objective trains on its data is refused."""
    stated = dataclasses.replace(grpo_run(), rollout=None)
    assert stated.rollout == FlowRolloutConfig()
    assert DiffusionRunConfig.from_dict(stated.to_dict()).rollout == FlowRolloutConfig()
    with pytest.raises(ValueError):
        dataclasses.replace(grpo_run(), objective=ObjectiveConfig("flow_grpo", {"reward": "psnr"}))
    with pytest.raises(ValueError):
        dataclasses.replace(grpo_run(), objective=ObjectiveConfig("diffusion", {}))
    with pytest.raises(ValueError):
        FlowRolloutConfig(reward="no_such_metric")
    parsed = DiffusionRunConfig.cli(["preset:flow", "--objective", "flow_grpo", "--rollout.reward", "psnr",
                                     "--rollout.groups", "2", "--rollout.train-steps", "3"])
    assert parsed.rollout == FlowRolloutConfig(reward="psnr", groups=2, train_steps=3)
    assert DiffusionRunConfig.from_dict(parsed.to_dict()).rollout == parsed.rollout


def test_a_saved_flow_grpo_run_restores_its_policy_and_not_its_kl_reference(tmp_path):
    """With a KL term the EMA slot holds the frozen initial model. The run's
    task restores the trained policy under `from_run`'s default `ema=None`,
    the weights the objective's own pipeline publishes."""
    from dew.checkpoints import Checkpoints
    from dew.inference import TextToImage

    config = grpo_run(beta=0.1, directory=str(tmp_path))
    objective = config.build()
    assert isinstance(objective, FlowGRPOObjective) and config.rollout is not None
    checkpoints = Checkpoints(str(tmp_path / "run"))
    _, started, state = grpo_step(objective, config.rollout.build(objective), checkpoints=checkpoints)
    checkpoints.save(1, state, None, {}, artifact=objective.inference_record())
    checkpoints.wait()
    config.save(str(tmp_path / "run"))

    restored = jax.tree.leaves(TextToImage.from_run(str(tmp_path / "run")).variables["params"])
    published = jax.tree.leaves(objective.pipeline(state).variables["params"])
    for got, want in zip(restored, published, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert any(not np.array_equal(np.asarray(got), before)
               for got, before in zip(restored, started, strict=True))


def test_flow_grpo_over_a_loaded_pipeline_samples_on_the_pipelines_grid(pipelines, tmp_path):
    """Flow-GRPO built over a published pipeline, its grid left out, evaluates on the pipeline's own
    grid as the denoising objective over it does, and its published task and the task a saved run
    restores draw what the pipeline's own task draws from the same noise. `grid=None` evaluates on
    the training process's times instead, as the denoising objective with that grid does, which
    the pipeline's grid does not draw. The SDE rollout walks the process's own times either way."""
    from dew.checkpoints import Checkpoints
    from dew.interop.pretrained import load_diffusion_source

    directory = pipelines / "sd3" / "pipeline"
    settings = {"solver": Euler(), "guidance": None, "steps": 2}
    rollout = FlowRolloutConfig(reward="psnr", groups=2, steps=3)
    source = load_diffusion_source(str(directory), dtype="float32", attention_impl="xla", size=(16, 16))
    step = Step(jnp.asarray(0), jax.random.PRNGKey(5), None)

    def built(**grid):
        grpo = FlowGRPOObjective(source, source.process, source.inputs, autoencoder=source.autoencoder,
                                 variables=source.variables, **settings, **grid)
        plain = DiffusionObjective(source, solver=Euler(), guidance=None, steps=2, **grid)
        return grpo, plain

    def started(objective):
        return Trainer(objective, optax.sgd(1e-2), key=jax.random.PRNGKey(3)).initial_state()

    def evaluated(objective):
        state = started(objective)
        images = np.asarray(objective.evaluate(state.variables, batch_for(objective, 16), step).images)
        assert np.all(np.isfinite(images))
        return images

    def rolled(objective):
        state = started(objective)
        drawn = rollout.build(objective)(state, batch_for(objective, 16), jax.random.PRNGKey(2))
        fields = {name: np.asarray(drawn[name]) for name in (
            "latents", "next_latents", "timesteps", "next_timesteps", OLD_LOG_PROBS_KEY, REWARDS_KEY)}
        assert all(np.all(np.isfinite(field)) for field in fields.values())
        return fields

    (inherited, plain), (unset, plain_unset) = built(), built(grid=None)
    on_grid, on_times = evaluated(inherited), evaluated(unset)
    np.testing.assert_array_equal(on_grid, evaluated(plain))
    np.testing.assert_array_equal(on_times, evaluated(plain_unset))
    assert not np.array_equal(on_grid, on_times)
    # The rollout walks the process's own times, whatever grid evaluation samples on.
    on_grid_rollout, on_times_rollout = rolled(inherited), rolled(unset)
    for name, field in on_grid_rollout.items():
        np.testing.assert_array_equal(field, on_times_rollout[name], err_msg=name)
    times = np.asarray(source.process.times(rollout.steps))
    np.testing.assert_array_equal(on_grid_rollout["timesteps"][0], times[:-1])
    np.testing.assert_array_equal(on_grid_rollout["next_timesteps"][0], times[1:])

    prompts = list(PROMPTS[:2])
    pipeline = source.text_to_image()
    reference = np.asarray(pipeline(prompts, steps=2, guidance=None, solver=Euler(), key=7).images)
    published = np.asarray(inherited.build_task(started(inherited).variables)(prompts, key=7).images)
    assert np.all(np.isfinite(reference)) and np.all(np.isfinite(published))
    np.testing.assert_array_equal(published, reference)

    config = DiffusionRunConfig(pretrained=str(directory), preset=None, model=precision(),
                                data=images_of("tfds://oxford_flowers102", 16),
                                val_metrics=(),
                                trainer=TrainerConfig(checkpoint_dir=str(tmp_path)),
                                objective=ObjectiveConfig("flow_grpo", settings), rollout=rollout)
    saved = config.build()
    initial = started(saved)
    # The run's step-0 state is the pipeline's weights, leaf for leaf.
    for (path, held), weight in zip(jax.tree_util.tree_leaves_with_path(initial.variables),
                                    jax.tree.leaves(source.variables), strict=True):
        assert np.all(np.isfinite(np.asarray(held))), jax.tree_util.keystr(path)
        np.testing.assert_array_equal(np.asarray(held), np.asarray(weight),
                                      err_msg=jax.tree_util.keystr(path))
    checkpoints = Checkpoints(str(tmp_path / "run"))
    checkpoints.save(0, initial, None, {}, artifact=saved.inference_record())
    checkpoints.wait()
    config.save(str(tmp_path / "run"))
    # The restored task runs where its weights were restored, over every device: the pipeline's task
    # over those weights, run there too, draws what it does.
    restored = TextToImage.from_run(str(tmp_path / "run"))
    published_variables = saved.pipeline(initial).variables
    for got, want in zip(jax.tree.leaves(restored.variables), jax.tree.leaves(published_variables),
                         strict=True):
        assert np.all(np.isfinite(np.asarray(got)))
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    drawn = np.asarray(restored(prompts, key=7).host().images)
    assert np.all(np.isfinite(drawn))
    np.testing.assert_array_equal(
        drawn,
        pipeline.bind(restored.variables)(prompts, steps=2, guidance=None, solver=Euler(),
                                          key=7).host().images)
