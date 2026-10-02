"""What `DiffusionRunConfig` reaches beyond a scratch model on CLIP text: a
published pipeline to fine-tune, a published family trained from scratch on
its own text towers, and Flow-GRPO on a reward.

The pipelines are the committed tiny SD3, Flux and Qwen-Image sources, whose
forwards the family tests match against diffusers.
"""

import dataclasses
import shutil
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.config import ModelConfig, TrainerConfig
from dew.data import TFDSImages
from dew.diffusion.presets import Flow, ResolutionShift
from dew.objectives import Step
from dew.objectives.diffusion import DiffusionRunConfig, FlowGRPO, TextCondition
from dew.objectives.rl.flow import FlowGRPOObjective
from dew.sampling import Euler
from dew.training import Trainer

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROMPTS = ["a red bird", "two cats"]


@pytest.fixture(scope="module")
def pipelines(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipelines")
    for family in ("sd3", "flux", "qwen_image"):
        with tarfile.open(FIXTURES / f"{family}_source.tar.xz") as archive:
            archive.extractall(root / family, filter="data")
    return root


def precision() -> ModelConfig:
    """The default model record at float32 on the XLA kernel."""
    return dataclasses.replace(DiffusionRunConfig().model, dtype="float32", attention_impl="xla")


def batch_for(objective, size: int) -> dict:
    """One row per device, the prompts in turn: a batch the data axis divides,
    in the channels the objective's sample field names."""
    rows, channels = jax.device_count(), objective.inputs.sample.shape[-1]
    pixels = np.tile(np.arange(size * size * channels, dtype=np.uint8).reshape(
        1, size, size, channels), (rows, 1, 1, 1))
    return {"image": pixels,
            **objective.inputs.tokenize([PROMPTS[row % len(PROMPTS)] for row in range(rows)])}


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
                                data=TFDSImages(image_size=size), solver=Euler(),
                                guidance=None, sampling_steps=2, unconditional_prob=0.0,
                                ema_decay=None, val_metrics=())
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


def test_a_pretrained_run_refuses_a_preset_of_another_kind(pipelines):
    """Flux was trained as a velocity flow; the default EDM preset is not
    that convention, and a flow preset is."""
    common = {"pretrained": str(pipelines / "flux" / "pipeline"), "model": precision(),
                  "data": TFDSImages(image_size=16), "solver": Euler(), "guidance": None,
                  "sampling_steps": 2, "val_metrics": ()}
    with pytest.raises(ValueError, match="name a preset of its kind"):
        DiffusionRunConfig(**common).build()
    assert DiffusionRunConfig(**common, preset=Flow(shift=3.0)).build() is not None


def flow_run(family: str, size: int, pipelines) -> DiffusionRunConfig:
    return DiffusionRunConfig(pretrained=str(pipelines / family / "pipeline"), preset=None,
                              model=precision(), data=TFDSImages(image_size=size),
                              solver=Euler(), guidance=None, sampling_steps=2,
                              val_metrics=())


@pytest.mark.parametrize("size", [16, 64])
def test_sd3_trains_at_its_static_shift_at_any_size(size, pipelines):
    assert flow_run("sd3", size, pipelines).build().process.schedule.shift == 3.0


@pytest.mark.parametrize("size", [16, 32, 64])
def test_flux_trains_at_the_shift_its_pipeline_samples_the_runs_size_at(size, pipelines):
    """Flux's `calculate_shift`: mu is linear in the packed latent's token
    count, 0.5 at 256 tokens and 1.15 at 4096, and the shift is exp(mu)."""
    objective = flow_run("flux", size, pipelines).build()
    tokens = (size // (2 * objective.autoencoder.downscale_factor)) ** 2
    mu = 0.5 + (tokens - 256) * (1.15 - 0.5) / (4096 - 256)
    assert objective.process.schedule.shift == pytest.approx(np.exp(mu), rel=1e-12)


@pytest.mark.parametrize("size", [128, 256, 512, 1024])
def test_a_scratch_flow_run_shifts_by_the_datas_resolution(size):
    """The preset's resolution shift at the data's 16-pixel token count, as
    the Flux pipeline's calculate_shift takes it."""
    config = DiffusionRunConfig(preset=Flow(resolution_shift=ResolutionShift()),
                                data=TFDSImages(image_size=size), text=None,
                                model=ModelConfig("simple_dit"), val_metrics=())
    tokens = (size // 16) ** 2
    mu = 0.5 + (tokens - 256) * (1.15 - 0.5) / (4096 - 256)
    assert config.preset().schedule.shift == pytest.approx(np.exp(mu), rel=1e-12)


def test_a_pretrained_run_refuses_a_model_of_its_own():
    with pytest.raises(ValueError, match=r"leave model.architecture, model.config, text unset"):
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
            "guidance_embeds": True, "axes_dims_rope": [4, 4, 4]},
            dtype="float32", attention_impl="xla"),
        data=TFDSImages(image_size=8), preset=Flow(), solver=Euler(),
        guidance=None, sampling_steps=2, ema_decay=None, val_metrics=(),
        text=TextCondition(encoder="diffusion_text", checkpoint=str(directory)))
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
                                         "num_heads": 2}, dtype="float32", attention_impl="xla"),
        data=TFDSImages(image_size=4), preset=Flow(), solver=Euler(),
        guidance=None, sampling_steps=2, val_metrics=(),
        trainer=TrainerConfig(checkpoint_dir=directory),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        rl=FlowGRPO(reward="psnr", groups=2, rollout_steps=3, clip_range=0.2, beta=beta))


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
    prepared, started, state = grpo_step(objective, config.rollout(objective))
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
    _, started, state = grpo_step(objective, config.rollout(objective), checkpoints=checkpoints)
    checkpoints.save(1, state, None, {})
    checkpoints.wait()
    config.save(str(tmp_path / "run"))

    restored = jax.tree.leaves(TextToImage.from_run(str(tmp_path / "run")).params["params"])
    published = jax.tree.leaves(objective.pipeline(state).params["params"])
    for got, want in zip(restored, published, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert any(not np.array_equal(np.asarray(got), before)
               for got, before in zip(restored, started, strict=True))
