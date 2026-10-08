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
from reference_error import assert_as_exact_as_the_reference
from releases import released_sampling

from dew.config import ModelConfig, ObjectiveConfig, RunConfig, TrainerConfig
from dew.data import TFDSImages
from dew.diffusion.presets import Flow, ResolutionShift
from dew.diffusion.process import Process
from dew.diffusion.schedules import FlowMatchingScheduler
from dew.diffusion.schedules.source import SourceSchedule
from dew.diffusion.transforms import FlowMatchPredictionTransform, ScheduleWeighting
from dew.objectives import Step
from dew.objectives.diffusion import DiffusionObjective, DiffusionRunConfig, TextCondition
from dew.objectives.rl.flow import FlowGRPOObjective
from dew.registry import argument_records, to_record
from dew.sampling import CFG, Euler
from dew.training import Trainer

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def pipelines(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipelines")
    for family in ("sd3", "flux", "qwen_image"):
        with tarfile.open(FIXTURES / f"{family}_source.tar.xz") as archive:
            archive.extractall(root / family, filter="data")
    return root


def precision() -> ModelConfig:
    """The default model record at float32 on the XLA kernel."""
    model = DiffusionRunConfig().model
    return dataclasses.replace(model, fields={**model.fields, "dtype": "float32", "attention_impl": "xla"})


def value(objective, params, batch) -> float:
    loss, _ = objective.loss(params, batch, Step(jnp.asarray(0), jax.random.PRNGKey(5), None))
    return float(loss.total / loss.mass)


@pytest.mark.parametrize("family", ["sd3", "flux", "qwen_image"])
def test_a_run_fine_tunes_a_published_pipeline_and_rebinds_it_without_weights(
        family, pipelines, tmp_path):
    """The run trains the pipeline's own transformer under its own text
    conditioning, and a saved tree rebuilds over the directory's metadata
    alone: the rebuilt objective scores the trained weights exactly, from a
    copy of the pipeline with every weight file deleted."""
    directory = pipelines / family / "pipeline"
    size = 16 if family != "qwen_image" else 32
    config = DiffusionRunConfig(pretrained=str(directory), preset=None, model=precision(),
                                data=TFDSImages(image_size=size), val_metrics=(),
                                objective=ObjectiveConfig("diffusion",
                                                          {"solver": Euler(), "guidance": None, "steps": 2,
                                                           "unconditional_prob": 0.0, "ema_decay": None}))
    objective = config.build()
    assert set(objective.inputs.conditions) == {"conditioning"}
    batch = batch_for(objective, size)
    trainer = Trainer(objective, optax.sgd(1e-2), key=jax.random.PRNGKey(3))
    initial = trainer.initial_state()
    before = value(objective, initial.variables, batch)
    state, _, _, _, accepted = trainer.compile(initial, batch)(initial, batch)
    assert bool(accepted)
    trained = value(objective, state.variables, batch)
    assert trained < before

    metadata = tmp_path / "metadata"
    shutil.copytree(directory, metadata, ignore=shutil.ignore_patterns("*.safetensors"))
    rebuilt = dataclasses.replace(config, pretrained=str(metadata)).build(variables=state.variables)
    assert value(rebuilt, state.variables, batch) == trained


@pytest.mark.parametrize("family", ["sd3", "flux", "qwen_image"])
def test_a_run_config_builds_the_objective_the_python_api_builds(family, pipelines):
    """The wiring half of a configured run: `DiffusionRunConfig` over a
    published pipeline builds what `DiffusionObjective(Pretrained.load(...))`
    builds with the same settings at the data's resolution
    (`load_diffusion_source(size=)`): the same model, process (its
    resolution shift too), tokenized ids and autoencoder kind over the
    same initial variables, so both score a batch alike, through the VAE
    and the text towers. The loss the two share is held to Diffusers'
    train_dreambooth_flux.py statements on matched draws in
    tests/test_flow_matching.py (ad63cc6a)."""
    from dew.interop.pretrained import load_diffusion_source

    directory = pipelines / family / "pipeline"
    size = 16 if family != "qwen_image" else 32
    settings = {"guidance": None, "unconditional_prob": 0.25, "ema_decay": None}
    configured = DiffusionRunConfig(pretrained=str(directory), preset=None, model=precision(),
                                    data=TFDSImages(image_size=size), val_metrics=(),
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


def test_a_pretrained_run_refuses_a_preset_of_another_kind(pipelines):
    """Flux was trained as a velocity flow; the default EDM preset is not
    that convention, and a flow preset is."""
    common = {"pretrained": str(pipelines / "flux" / "pipeline"), "model": precision(),
              "data": TFDSImages(image_size=16), "val_metrics": (),
              "objective": ObjectiveConfig("diffusion", {"solver": Euler(), "guidance": None, "steps": 2})}
    with pytest.raises(ValueError, match="name a preset of its kind"):
        DiffusionRunConfig(**common).build()
    assert DiffusionRunConfig(**common, preset=Flow(shift=3.0)).build() is not None


def test_a_pretrained_run_records_its_pipelines_sampling_and_builds_it_after_a_release(pipelines,
                                                                                       monkeypatch):
    """A run over a published pipeline samples with the pipeline's own
    guidance and steps where it states none. Its record holds them, so a
    release that defaults a denoiser's sampling otherwise builds the run as
    it sampled."""
    config = DiffusionRunConfig(pretrained=str(pipelines / "sd3" / "pipeline"), preset=None,
                                model=precision(), data=TFDSImages(image_size=16), val_metrics=(),
                                objective=ObjectiveConfig("diffusion", {"solver": Euler(),
                                                                        "ema_decay": None}))
    objective = config.build()
    record = json.loads(json.dumps(config.recorded(objective).record()))
    sampled = {"guidance": objective.guidance, "steps": objective.steps}
    held = record["fields"]["objective"]["defaults"]
    assert {key: held[key] for key in sampled} == argument_records(DiffusionObjective, sampled)
    released_sampling(monkeypatch, CFG(scale=4.5), steps=7)
    rebuilt = RunConfig.read(record).build()
    assert {"guidance": rebuilt.guidance, "steps": rebuilt.steps} == sampled


def flow_run(family: str, size: int, pipelines) -> DiffusionRunConfig:
    return DiffusionRunConfig(pretrained=str(pipelines / family / "pipeline"), preset=None,
                              model=precision(), data=TFDSImages(image_size=size),
                              val_metrics=(),
                              objective=ObjectiveConfig("diffusion",
                                                        {"solver": Euler(), "guidance": None, "steps": 2}))


@pytest.mark.parametrize("size", [16, 64])
def test_sd3_trains_at_its_static_shift_at_any_size(size, pipelines):
    assert flow_run("sd3", size, pipelines).build().process.schedule.shift == 3.0


@pytest.mark.parametrize("family", ["sd3", "flux"])
def test_a_published_flow_pipeline_trains_on_the_sd3_papers_convention(family, pipelines):
    """`training_process` follows Esser et al. 2024 rather than the Diffusers
    scripts' command-line defaults: logit-normal times (mean 0, std 1) at
    the shift the pipeline's sampler walks, scored on the velocity at unit
    weight (docs/guides/diffusion.md says where the scripts differ)."""
    process = flow_run(family, 16, pipelines).build().process
    schedule = process.schedule
    assert (schedule.density, schedule.logit_mean, schedule.logit_std) == ("logit_normal", 0.0, 1.0)
    assert type(process.weighting) is ScheduleWeighting
    assert type(process.prediction) is FlowMatchPredictionTransform


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
    times = np.asarray(schedule.sample_t(jax.random.key(int(DRAWS["key"])), DRAWS[f"{family}/times"].size))
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


@pytest.mark.parametrize("size", [16, 32, 64])
def test_flux_trains_at_the_shift_its_pipeline_samples_the_runs_size_at(size, pipelines):
    """The shift is exp of Diffusers 0.34.0's Flux `calculate_shift` at the
    packed latent's token count, to the bit."""
    objective = flow_run("flux", size, pipelines).build()
    tokens = (size // (2 * objective.autoencoder.downscale_factor)) ** 2
    assert objective.process.schedule.shift == flux_shift(tokens)


@pytest.mark.parametrize("size", [128, 256, 512, 1024])
def test_a_scratch_flow_run_shifts_by_the_datas_resolution(size):
    """The preset's resolution shift at the data's 16-pixel token count is
    exp of Diffusers' Flux `calculate_shift` there, to the bit."""
    config = DiffusionRunConfig(preset=Flow(resolution_shift=ResolutionShift()),
                                data=TFDSImages(image_size=size), text=None,
                                model=ModelConfig("simple_dit"), val_metrics=())
    assert config.preset().schedule.shift == flux_shift((size // 16) ** 2)


@pytest.mark.parametrize("constants,fields", [
    ("flux", {}),
    ("long", {"max_tokens": 8192, "max_shift": 0.9}),
])
def test_the_resolution_shift_and_the_noise_levels_it_maps_to_are_diffusers(constants, fields):
    """At every token count, `ResolutionShift` and a loaded dynamic-shift
    scheduler file are exp(calculate_shift) to the bit, and the flow schedule
    at that shift maps each time to the noise level Diffusers'
    `_time_shift_exponential` gives it at mu, held to the float64 rule, for
    Flux's constants and a pipeline with another slope and span."""
    top_shift, top = fields.get("max_shift", 1.15), fields.get("max_tokens", 4096)
    loaded = SourceSchedule.from_config({
        "_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000,
        "use_dynamic_shifting": True, "base_shift": 0.5, "max_shift": top_shift,
        "base_image_seq_len": 256, "max_image_seq_len": top})
    for index, tokens in enumerate(SHIFTS["tokens"]):
        shift = ResolutionShift(tokens=int(tokens), **fields).shift()
        assert shift == flux_shift(int(tokens), constants)
        assert loaded.training_process(int(tokens)).schedule.shift == shift
        sigma = FlowMatchingScheduler(shift=shift).rates(SHIFTS["times"])[1]
        assert_as_exact_as_the_reference(sigma, SHIFTS[f"{constants}/sigmas"][index],
                                         SHIFTS[f"{constants}/sigmas_f64"][index], f"{constants} at {tokens}")


def test_a_pretrained_run_refuses_a_model_of_its_own():
    with pytest.raises(ValueError, match=r"leave model, text unset"):
        DiffusionRunConfig(pretrained="some/pipeline", model=ModelConfig("simple_dit"),
                           text=TextCondition(encoder="t5"))


def test_a_published_family_trains_from_scratch_on_its_pipelines_text_towers(pipelines):
    """Flux from scratch in pixel space, conditioned by the Flux pipeline's
    CLIP and T5 towers: the conditioner's record goes in under `conditioning`,
    where the published family reads it."""
    directory = pipelines / "flux" / "pipeline"
    config = DiffusionRunConfig(
        model=ModelConfig("flux_transformer", {
            "in_channels": 12, "out_channels": 12, "num_layers": 1, "num_single_layers": 1,
            "heads": 2, "head_dim": 12, "joint_attention_dim": 16, "pooled_projection_dim": 10,
            "guidance_embeds": True, "axes_dims_rope": [4, 4, 4],
            "dtype": "float32", "attention_impl": "xla"}),
        data=TFDSImages(image_size=8), preset=Flow(), val_metrics=(),
        text=TextCondition(encoder="diffusion_text", checkpoint=str(directory)),
        objective=ObjectiveConfig("diffusion",
                                  {"solver": Euler(), "guidance": None, "steps": 2, "ema_decay": None}))
    objective = config.build()
    assert set(objective.inputs.conditions) == {"conditioning"}
    batch = batch_for(objective, 8)
    trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.PRNGKey(0))
    initial = trainer.initial_state()
    before = value(objective, initial.variables, batch)
    state, _, _, _, accepted = trainer.compile(initial, batch)(initial, batch)
    assert bool(accepted)
    assert value(objective, state.variables, batch) < before


def grpo_run(beta: float = 0.0, directory: str = "./checkpoints") -> DiffusionRunConfig:
    return DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 2, "emb_features": 16, "num_layers": 1,
                                         "num_heads": 2, "dtype": "float32", "attention_impl": "xla"}),
        data=TFDSImages(image_size=4), preset=Flow(), val_metrics=(),
        trainer=TrainerConfig(checkpoint_dir=directory),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        objective=ObjectiveConfig("flow_grpo",
                                  {"reward": "psnr", "groups": 2, "rollout_steps": 3, "clip_range": 0.2,
                                   "beta": beta, "solver": Euler(), "guidance": None, "steps": 2}))


def grpo_step(objective, rollout, **trainer):
    """One rollout over a batch and the policy step on its transitions."""
    trainer = Trainer(objective, optax.sgd(1e-2), key=jax.random.PRNGKey(1), rollout=rollout,
                      **trainer)
    initial = trainer.initial_state()
    prepared = rollout(initial, batch_for(objective, 4), jax.random.PRNGKey(2))
    started = [np.asarray(leaf) for leaf in jax.tree.leaves(initial.variables["params"])]
    state, _, _, _, accepted = trainer.compile(initial, prepared)(initial, prepared)
    assert bool(accepted)
    return prepared, started, state


def test_flow_grpo_trains_from_the_run_config_on_a_registered_reward():
    """`rl` builds the Flow-GRPO objective and its rollout: groups of SDE
    samples per prompt, scored by PSNR against the prompt's own image, and
    one policy step on the transitions the rollout cut."""
    config = grpo_run()
    objective = config.build()
    assert isinstance(objective, FlowGRPOObjective)
    prepared, started, state = grpo_step(objective, objective.rollout())
    assert prepared["latents"].shape[:2] == (jax.device_count() * 2, 2)
    assert np.all(np.isfinite(prepared["rewards"]))
    assert any(not np.array_equal(after, before) for after, before in zip(
        jax.tree.leaves(state.variables["params"]), started, strict=True))


def test_a_saved_flow_grpo_run_restores_its_policy_and_not_its_kl_reference(tmp_path):
    """With a KL term the EMA slot holds the frozen initial model. The run's
    task restores the trained policy under `from_run`'s default `ema=None`,
    the weights the objective's own pipeline publishes."""
    from dew.checkpoints import Checkpoints
    from dew.sampling.pipelines import TextToImage

    config = grpo_run(beta=0.1, directory=str(tmp_path))
    objective = config.build()
    checkpoints = Checkpoints(str(tmp_path / "run"))
    _, started, state = grpo_step(objective, objective.rollout(), checkpoints=checkpoints)
    checkpoints.save(1, state, None, {}, artifact=objective.inference_record())
    checkpoints.wait()
    config.save(str(tmp_path / "run"))

    restored = jax.tree.leaves(TextToImage.from_run(str(tmp_path / "run")).variables["params"])
    published = jax.tree.leaves(objective.pipeline(state).variables["params"])
    for got, want in zip(restored, published, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert any(not np.array_equal(np.asarray(got), before)
               for got, before in zip(restored, started, strict=True))
