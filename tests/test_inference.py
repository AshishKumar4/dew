"""Inference: a run directory through to a sample.

A run writes its resolved config as `run.json` next to its checkpoints;
`TextToImage.from_run` builds the objective from it the way the recipe did
and restores the weights, and `from_pretrained` is the same on a pulled hub
snapshot. Everything here drives the real pipeline from a checkpoint the
trainer has just written.
"""

import dataclasses
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import optax
import pytest

import dew.nn.backbones  # registers the models
from dew.artifacts import VideoGrid
from dew.config import ModelConfig, TrainerConfig
from dew.data import Dataset, OxfordFlowers
from dew.data.video import VideoDataset
from dew.diffusion import FlowMatchPredictionTransform
from dew.diffusion.schedules import FlowMatchingScheduler
from dew.inputs import Field, unit_range
from dew.objectives.base import merge
from dew.objectives.diffusion import DiffusionRunConfig, StableDiffusionAutoencoder, TextCondition
from dew.registry import presets, samplers
from test_diffusion_objective import StubText  # noqa: F401  registers "stub_text"
from dew.sampling import CFG, Heun, TextToImage
from dew.training import Checkpoints, Trainer

RES = 8
MODEL = dict(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1)


def run_config(directory, preset=presets.EDM()):
    """The resolved config of a tiny conditional DiT run in `directory`; the
    text condition names the registered stub encoder."""
    return DiffusionRunConfig(
        model=ModelConfig("simple_dit", dict(MODEL), dtype="float32", attention_impl="reference"),
        data=OxfordFlowers(image_size=RES),
        trainer=TrainerConfig(checkpoint_dir=str(directory), batch_size=8, steps=2, keep=1),
        preset=preset, sampler=samplers.Euler(), sampling_steps=3,
        text=TextCondition(encoder="stub_text", checkpoint="stub-clip"))


def make_run(directory, preset=presets.EDM()):
    """Two training steps of the tiny conditional DiT, its checkpoint and its
    `run.json` in `directory`, as the recipe leaves them: the objective is
    the config's own build."""
    config = run_config(directory, preset)
    objective = config.build()
    encoder = objective.inputs.conditions["textcontext"].encoder
    images = np.tile(np.linspace(0, 255, RES, dtype=np.float32)[None, :, None, None],
                     (8, 1, RES, 3)).astype(np.uint8)
    batch = {"image": images, "text": encoder.tokenize(["a", "b", "c", "d", "e", "f", "g", "h"])}

    class Stream:
        def __init__(self):
            self.position = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.position += 1
            return batch

        def get_state(self):
            return str(self.position).encode()

        def set_state(self, state):
            self.position = int(state.decode())

    checkpoints = Checkpoints(str(directory), keep=1)
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.PRNGKey(0),
                      checkpoints=checkpoints)
    state = trainer.fit(Dataset(train=Stream, val=None, records=None, batch=8),
                        steps=2, log_every=100, checkpoint_every=2)
    checkpoints.wait()
    config.save(str(directory))
    return objective, state


def test_pipeline_generates_from_a_run_directory(tmp_path):
    """The whole offline path: the run.json and checkpoint a run wrote, the
    model, process, inputs and weights rebuilt from them, and a sample out."""
    make_run(tmp_path)
    pipe = TextToImage.from_run(str(tmp_path))
    assert type(pipe.model).__name__ == "SimpleDiT" and pipe.model.emb_features == 16
    assert pipe.model.output_channels == 3
    assert pipe.inputs.sample == Field("image", (RES, RES, 3))
    assert pipe.inputs.conditions["textcontext"].encoder.checkpoint == "stub-clip"

    images = pipe(["a water lily", "a sunflower"], steps=3, guidance=2.0,
                  key=jax.random.PRNGKey(0))
    assert images.shape == (2, RES, RES, 3)
    assert np.all(np.isfinite(images))
    assert images.min() >= -1.0 and images.max() <= 1.0


def test_from_run_restores_the_averaged_weights_by_default(tmp_path):
    """The EMA copy is what a run publishes; `ema=False` reads the live ones."""
    objective, state = make_run(tmp_path)
    averaged = merge(state.params, state.ema)

    pipe = TextToImage.from_run(str(tmp_path))
    for expected, loaded in zip(jax.tree.leaves(averaged["params"]),
                                jax.tree.leaves(pipe.params["params"]), strict=True):
        np.testing.assert_allclose(np.asarray(loaded), np.asarray(expected))
    live = TextToImage.from_run(str(tmp_path), ema=False)
    for expected, loaded in zip(jax.tree.leaves(state.params["params"]),
                                jax.tree.leaves(live.params["params"]), strict=True):
        np.testing.assert_allclose(np.asarray(loaded), np.asarray(expected))
    assert not all(np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(
        jax.tree.leaves(pipe.params["params"]), jax.tree.leaves(live.params["params"])))
    # the frozen encoder's table is the run's, not something re-drawn
    np.testing.assert_array_equal(
        np.asarray(pipe.params["encoders"]["textcontext"]["table"]),
        np.asarray(objective.inputs.conditions["textcontext"].encoder.params["table"]))


def test_from_run_rebuilds_the_training_process_exactly(tmp_path):
    """run.json holds the preset's fields, so inference samples with the
    shift the run trained with and not the preset default."""
    make_run(tmp_path, preset=presets.Flow(shift=3.0, logit_mean=0.5))
    pipe = TextToImage.from_run(str(tmp_path))
    assert isinstance(pipe.process.schedule, FlowMatchingScheduler)
    assert pipe.process.schedule.shift == 3.0 and pipe.process.schedule.logit_mean == 0.5
    assert type(pipe.process.prediction) is FlowMatchPredictionTransform
    assert pipe.process.sampling is None


def test_from_pretrained_is_from_run_on_the_pulled_snapshot(tmp_path, monkeypatch):
    make_run(tmp_path)
    import dew.interop.hub as hub
    monkeypatch.setattr(hub, "pull_from_hub", lambda repo_id, revision=None: tmp_path)
    pipe = TextToImage.from_pretrained("user/flowers-dit")
    assert pipe.inputs.conditions["textcontext"].encoder.checkpoint == "stub-clip"


def test_sampler_and_guidance_are_call_arguments(tmp_path):
    """Two steps of training leave the zero-initialised head near zero, so
    the weights are nudged off it for the conditional and unconditional
    branches to differ; then guidance is visible in the sample."""
    make_run(tmp_path)
    loaded = TextToImage.from_run(str(tmp_path))
    pipe = dataclasses.replace(loaded, params=jax.tree.map(lambda leaf: leaf + 0.05, loaded.params))
    key = jax.random.PRNGKey(1)
    plain = pipe(["x"], steps=8, guidance=None, sampler=Heun(), key=key)
    guided = pipe(["x"], steps=8, guidance=CFG(4.0, interval=(0.2, 0.8)), sampler=Heun(), key=key)
    assert plain.shape == guided.shape == (1, RES, RES, 3)
    assert not np.allclose(plain, guided)
    assert np.array_equal(pipe(["x"], steps=8, guidance=None, sampler=Heun(), key=key), plain)
    assert np.array_equal(pipe(["x"], steps=8, guidance=4.0, sampler=Heun(), key=key),
                          pipe(["x"], steps=8, guidance=CFG(4.0), sampler=Heun(), key=key))


def test_the_run_record_refuses_a_field_it_does_not_know(tmp_path):
    """A run.json from another objective, or with a knob this class lacks,
    raises instead of building something other than what was trained."""
    config = run_config(tmp_path)
    record = config.to_dict()
    record["sampler_steps"] = 3
    with pytest.raises(ValueError, match="sampler_steps"):
        DiffusionRunConfig.from_dict(record)
    assert DiffusionRunConfig.from_dict(config.to_dict()) == config


def test_an_unconditional_run_builds_without_an_encoder(tmp_path):
    config = dataclasses.replace(run_config(tmp_path), text=None)
    objective = DiffusionRunConfig.from_dict(config.to_dict()).build()
    assert objective.inputs.conditions == {}
    assert set(objective.init(jax.random.PRNGKey(0))["encoders"]) == set()


def test_an_unconditional_default_model_takes_a_step():
    """text=None on the default unet builds, inits and takes one trainer
    step: with no text the cross-attention blocks fall back to
    self-attention."""
    from dew.data import Dataset
    from dew.objectives.base import Step
    from dew.training import Trainer
    config = DiffusionRunConfig(text=None)
    objective = config.build()
    images = np.zeros((8, 128, 128, 3), np.uint8)

    def batches():
        while True:
            yield {"image": images}

    state = Trainer(objective, optax.adam(1e-3),
                    key=jax.random.PRNGKey(0)).fit(
        Dataset(train=batches, val=None, records=None, batch=8), steps=1, log_every=100)

    assert int(state.step) == 1
    leaves = jax.tree.leaves(state.params["params"])
    assert leaves and all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)


def test_joint_stream_models_refuse_an_unconditional_run():
    """SimpleMMDiT and HierarchicalMMDiT run the text as a second stream
    through every block's joint attention, so with no text there is no
    sequence to project; the run is refused by name instead of failing in
    the first attention softmax over an empty slice."""
    from dew.config import ModelConfig

    base = DiffusionRunConfig(text=None)
    for architecture in ("simple_mmdit", "hierarchical_mmdit"):
        config = dataclasses.replace(base, model=ModelConfig(architecture, {}))
        with pytest.raises(ValueError, match="unconditional"):
            config.build()


def test_a_discrete_preset_is_refused_by_the_gaussian_objective():
    """`preset:mdlm` is one subcommand away on the diffusion recipe, and its
    process has no schedule the Gaussian objective can corrupt with, so the
    config names the preset and the objective that trains it instead of
    failing inside the loss."""
    from dew.diffusion.discrete import MDLM

    config = dataclasses.replace(DiffusionRunConfig(text=None), preset=MDLM(mask_id=0))
    with pytest.raises(ValueError, match="mdlm.*MaskedDiffusionObjective"):
        config.build()


def test_build_eval_metrics_follows_the_sample_field(tmp_path):
    """A video run scores its `VideoGrid` against its `video` field: the
    factories read that grid there, and the image-only metrics are refused
    by name instead of failing in the trainer."""
    video = dataclasses.replace(run_config(tmp_path),
                                data=VideoDataset(frame_size=8, frames=2),
                                val_metrics=["psnr"])
    (metric,) = video.build_eval_metrics()
    assert metric.reads is VideoGrid
    images = np.zeros((2, 2, 8, 8, 3), np.uint8)
    assert np.isinf(metric(VideoGrid(unit_range(images)), {"video": images}))
    with pytest.raises(ValueError, match="clip"):
        dataclasses.replace(video, val_metrics=["clip"]).build_eval_metrics()


def test_the_autoencoder_record_carries_its_revision(tmp_path):
    """A run trained with a non-default VAE revision rebuilds from its
    record; the revision used to fall out of the dataclass entirely."""
    config = dataclasses.replace(
        run_config(tmp_path),
        autoencoder=StableDiffusionAutoencoder(revision="flax", latent_scale=0.5))
    assert DiffusionRunConfig.from_dict(config.to_dict()) == config


def test_a_fresh_process_resolves_metrics_and_models_through_the_config():
    """The recipe runs in a process that imports nothing else first: the
    registries the config builds from fill on its import alone. `psnr` is
    pure, so resolving it proves the point without downloading weights."""
    root = Path(__file__).resolve().parents[1]
    code = ("from dew.objectives.diffusion.config import DiffusionRunConfig;"
            "from dew.registry import metrics, models;"
            "print(metrics['psnr']().name, 'simple_dit' in models)")
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={"PYTHONPATH": str(root / "src"), "JAX_PLATFORMS": "cpu",
             "PATH": os.environ.get("PATH", "")})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "psnr True"

def test_guidance_is_a_value_with_its_interval(tmp_path):
    """`guidance` was a bare scale whose 0 stood for "off", so `CFG.interval`
    could not be named from a run at all. It is the value now: a record
    builds one, None is the conditional prediction alone, and the interval
    survives the round trip."""
    config = dataclasses.replace(run_config(tmp_path), guidance=CFG(4.0, (0.2, 0.8)))
    assert DiffusionRunConfig.from_dict(config.to_dict()) == config
    assert config.to_dict()["guidance"] == {"scale": 4.0, "interval": [0.2, 0.8]}

    # The record a command line or a run.json carries builds the same value.
    from_record = DiffusionRunConfig.from_dict(
        {**config.to_dict(), "guidance": {"scale": 4.0, "interval": [0.2, 0.8]}})
    assert from_record.guidance == CFG(4.0, (0.2, 0.8))
    assert from_record.build().guidance == CFG(4.0, (0.2, 0.8))

    unguided = dataclasses.replace(config, guidance=None)
    assert unguided.build().guidance is None
    assert DiffusionRunConfig.from_dict(unguided.to_dict()).guidance is None

def test_the_text_condition_pins_a_revision(tmp_path, monkeypatch):
    """Both text loaders take a `revision` and the autoencoder's spec has
    always named one, so a run could not pin its text tower: a moved branch
    changed what a rerun conditioned on. The record carries it now, and only
    when set, since an encoder that takes no revision must still build."""
    from dew.registry import encoders as registry

    seen = {}
    original = registry["stub_text"].from_pretrained

    @classmethod
    def capture(cls, checkpoint, **fields):
        seen.update(checkpoint=checkpoint, **fields)
        return original(checkpoint, **{k: v for k, v in fields.items()
                                      if k not in ("revision", "max_length")})

    monkeypatch.setattr(registry["stub_text"], "from_pretrained", capture)

    pinned = dataclasses.replace(
        run_config(tmp_path),
        text=TextCondition(encoder="stub_text", checkpoint="stub-clip", revision="refs/pr/1"))
    assert DiffusionRunConfig.from_dict(pinned.to_dict()) == pinned
    pinned.text.build()
    assert seen["revision"] == "refs/pr/1"

    seen.clear()
    dataclasses.replace(pinned, text=TextCondition(encoder="stub_text",
                                                   checkpoint="stub-clip")).text.build()
    assert "revision" not in seen and "max_length" not in seen
