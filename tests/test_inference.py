"""Inference: a run directory through to a sample.

A run writes its resolved config as `run.json` next to its checkpoints;
`TextToImage.from_run` builds the objective from it the way the recipe did
and restores the weights, and `from_pretrained` is the same on a pulled hub
snapshot. Everything here drives the real pipeline from a checkpoint the
trainer has just written.
"""

import dataclasses
from dataclasses import dataclass

import dew
import jax
import numpy as np
import optax
import pytest

import dew.nn.backbones  # registers the models
from dew.artifacts import VideoGrid
from dew.config import ModelConfig, TrainerConfig
from dew.data import Dataset, OxfordFlowers, VideoDataset
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


def run_config(directory, preset=presets.EDM(), encoder="stub_text", checkpoint="stub-clip"):
    """The resolved config of a tiny conditional DiT run in `directory`; the
    text condition names the registered stub encoder by default."""
    return DiffusionRunConfig(
        model=ModelConfig("simple_dit", dict(MODEL), dtype="float32", attention_impl="reference"),
        data=OxfordFlowers(image_size=RES),
        trainer=TrainerConfig(checkpoint_dir=str(directory), batch_size=8, steps=2, keep=1),
        preset=preset, sampler=samplers.Euler(), sampling_steps=3,
        text=TextCondition(encoder=encoder, checkpoint=checkpoint))


def make_run(directory, preset=presets.EDM(), encoder="stub_text", checkpoint="stub-clip"):
    """Two training steps of the tiny conditional DiT, its checkpoint and its
    `run.json` in `directory`, as the recipe leaves them: the objective is
    the config's own build."""
    config = run_config(directory, preset, encoder, checkpoint)
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

    result = pipe(["a water lily", "a sunflower"], steps=3, guidance=2.0,
                  key=jax.random.PRNGKey(0))
    images = result.host().images
    assert images.shape == (2, RES, RES, 3)
    assert np.all(np.isfinite(images))
    assert images.min() >= -1.0 and images.max() <= 1.0
    # The front door names the same run and answers with the same task.
    front = dew.pipeline(str(tmp_path))
    assert isinstance(front, TextToImage) and front.steps == pipe.steps
    np.testing.assert_array_equal(
        front(["a water lily", "a sunflower"], steps=3, guidance=2.0, seed=0).host().images,
        pipe(["a water lily", "a sunflower"], steps=3, guidance=2.0, seed=0).host().images)


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
    plain = pipe(["x"], steps=8, guidance=None, sampler=Heun(), key=key).host().images
    guided = pipe(["x"], steps=8, guidance=CFG(4.0, interval=(0.2, 0.8)), sampler=Heun(), key=key).host().images
    assert plain.shape == guided.shape == (1, RES, RES, 3)
    assert not np.allclose(plain, guided)
    assert np.array_equal(pipe(["x"], steps=8, guidance=None, sampler=Heun(), key=key).host().images, plain)
    assert np.array_equal(pipe(["x"], steps=8, guidance=4.0, sampler=Heun(), key=key).host().images,
                          pipe(["x"], steps=8, guidance=CFG(4.0), sampler=Heun(), key=key).host().images)


def test_the_run_record_refuses_a_field_it_does_not_know(tmp_path):
    """A run.json from another objective, or with a knob this class lacks,
    raises a ValueError naming the field, and no model is built from it."""
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


def test_an_unconditional_unet_takes_a_step():
    """text=None on a unet builds, inits and takes one trainer step: with no
    text the cross-attention blocks fall back to self-attention. A tiny unet
    at 16 pixels; the default one at 128 pixels attends over 16k positions
    per stage and needs tens of GB on a CPU."""
    from dew.training import Trainer
    unet = {"emb_features": 32, "feature_depths": [8, 16], "num_res_blocks": 1,
            "norm_groups": 4, "attention_configs": [None, {"heads": 2}]}
    config = DiffusionRunConfig(
        model=ModelConfig("unet", unet, dtype="float32", attention_impl="reference"),
        data=OxfordFlowers(image_size=16), text=None)
    objective = config.build()
    images = np.zeros((8, 16, 16, 3), np.uint8)

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
    sequence to project; `build` raises a ValueError naming the architecture
    before the first attention softmax over an empty slice."""
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
    factories read that grid there, and an image-only metric raises a
    ValueError naming it in `build_eval_metrics`, before the trainer."""
    video = dataclasses.replace(run_config(tmp_path),
                                data=VideoDataset(frame_size=8, frames=2),
                                val_metrics=["psnr"])
    (metric,) = video.build_eval_metrics()
    images = np.zeros((2, 2, 8, 8, 3), np.uint8)
    assert np.isinf(metric.finalize(metric(VideoGrid(unit_range(images)), {"video": images})))
    with pytest.raises(ValueError, match="clip"):
        dataclasses.replace(video, val_metrics=["clip"]).build_eval_metrics()


def test_the_autoencoder_record_carries_its_revision(tmp_path):
    """A run trained with a non-default VAE revision rebuilds from its
    record with that revision."""
    config = dataclasses.replace(
        run_config(tmp_path),
        autoencoder=StableDiffusionAutoencoder(revision="flax", latent_scale=0.5))
    assert DiffusionRunConfig.from_dict(config.to_dict()) == config


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


def make_lm_run(directory, *, mesh=None):
    """Two training steps of a tiny byte-level decoder, its checkpoint and the
    `run.json` the LM recipe writes: the resolved model, tokenizer and budget."""
    import json

    from dew.objectives.lm import LMObjective, Samples
    from dew.sampling import Sampling
    from dew.training import MeshSpec

    fields = dict(vocab_size=256, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                  mlp_features=32, max_seq_len=16)
    model_config = ModelConfig("causal_transformer", fields, dtype="float32", attention_impl="reference")
    objective = LMObjective(model_config.build(), 8, ema_decay=0.9,
                            samples=Samples([1, 2, 3], 4, sampling=Sampling(temperature=0, eos_id=255)))
    rng = np.random.RandomState(0)
    batch = {"text": rng.randint(1, 250, (8, 9)).astype(np.int32)}

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
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.PRNGKey(0), checkpoints=checkpoints,
                      mesh=MeshSpec() if mesh is None else mesh)
    state = trainer.fit(Dataset(train=Stream, val=None, records=None, batch=8),
                        steps=2, log_every=100, checkpoint_every=2)
    checkpoints.wait()
    (directory / "run.json").write_text(json.dumps({
        "objective": "lm", "model": dataclasses.asdict(model_config), "tokenizer": "byte",
        "sample_tokens": 4, "ema_decay": 0.9, "data": {"seq_len": 8}}))
    return objective, state


def test_objective_pipeline_binds_the_trained_state_in_place(tmp_path):
    """A just-trained state is the pipeline: the averaged weights when the
    objective keeps them, the live ones on request, the objective's own
    sampling defaults, and the same images `from_run` restores."""
    objective, state = make_run(tmp_path)
    pipe = objective.pipeline(state)
    assert isinstance(pipe, TextToImage)
    assert (pipe.steps, pipe.guidance, pipe.sampler) == (objective.steps, objective.guidance, objective.sampler)
    for expected, bound in zip(jax.tree.leaves(state.averaged), jax.tree.leaves(pipe.params), strict=True):
        assert bound is expected
    live = objective.pipeline(state, ema=False)
    for expected, bound in zip(jax.tree.leaves(state.params), jax.tree.leaves(live.params), strict=True):
        assert bound is expected
    drawn = pipe(["a", "b"], seed=4).host().images
    assert drawn.shape == (2, RES, RES, 3)
    np.testing.assert_allclose(TextToImage.from_run(str(tmp_path))(["a", "b"], seed=4).host().images, drawn,
                               atol=2e-6, rtol=2e-6)


def test_pipeline_answers_an_lm_run_with_its_tokenizer_and_budget(tmp_path):
    """An LM run directory becomes a `TextGeneration` over the rebuilt model
    and restored weights, decoding through the run's tokenizer, budgeted by
    its sample setting, and drawing what the trained objective draws."""
    from dew.inference import TextGeneration
    from dew.sampling import Sampling

    objective, state = make_lm_run(tmp_path)
    task = dew.pipeline(str(tmp_path))
    assert isinstance(task, TextGeneration)
    assert task.max_new_tokens == 4 and task.sampling.eos_id == (255,)
    result = task("the ", seed=2, sampling=Sampling(temperature=0, eos_id=255))
    assert result.rows == 1 and result.host().tokens.shape == (1, 4 + 4)
    assert isinstance(result.text[0], str)
    trained = objective.pipeline(state, processor=task.processor)
    assert trained.max_new_tokens == 4
    np.testing.assert_array_equal(
        trained("the ", seed=2, sampling=Sampling(temperature=0, eos_id=255)).host().tokens,
        result.host().tokens)
    with pytest.raises(ValueError, match="exactly one of key and seed"):
        task("the ", seed=2, key=jax.random.key(2))
    with pytest.raises(ValueError, match="max_new_tokens is required"):
        dataclasses.replace(task, max_new_tokens=None)("the ", seed=2)


@pytest.mark.mesh
def test_pipeline_places_a_run_on_a_mesh_and_answers_the_same_images(tmp_path):
    """Placed under the trainer's layout, the weights shard over the mesh,
    prompts split over its batch axes, the result keeps that sharding, and
    `host()` reads back exactly what the single device draws."""
    from dew.nn.inputs import BATCH_AXES
    from dew.training import Layout, MeshSpec

    make_run(tmp_path)
    plain = TextToImage.from_run(str(tmp_path))
    placed = dew.pipeline(str(tmp_path), mesh=MeshSpec(fsdp=2), layout=Layout(min_shard=2 ** 6))
    specs = {leaf.sharding.spec for leaf in jax.tree.leaves(placed.params)}
    assert any("fsdp" in str(spec) for spec in specs)
    result = placed(["a", "b", "c"], steps=3, seed=7)
    assert result.images.sharding.spec == jax.sharding.PartitionSpec(BATCH_AXES)
    assert result.rows == 3
    np.testing.assert_allclose(result.host().images, plain(["a", "b", "c"], steps=3, seed=7).host().images,
                               atol=2e-5, rtol=2e-5)


def test_a_grid_prepares_the_process_and_times_and_final_denoise_ends_the_trajectory(tmp_path):
    """A source whose sampler pairs its own tables hands the task a `grid`:
    for a step count it answers the process and the explicit time grid the
    trajectory walks, and the prior the noise is drawn from. The process's
    own grid at the same points reproduces the plain task; a different grid
    changes the draw; the grid decides the length, so `steps + 1` points
    walk one more interval; `final_denoise=False` skips the closing clean
    prediction, and over a one-point grid hands the noise itself to the
    decode and clip that end every call."""
    import jax.numpy as jnp

    objective, state = make_run(tmp_path)
    plain = TextToImage.from_objective(objective, state.params)
    same = dataclasses.replace(plain, grid=lambda steps: (plain.process, plain.process.times(steps)))
    key = jax.random.key(3)
    reference = plain(["a"], steps=4, sampler=Heun(), key=key).host().images
    np.testing.assert_array_equal(same(["a"], steps=4, sampler=Heun(), key=key).host().images, reference)
    warped = dataclasses.replace(plain, grid=lambda steps: (plain.process, plain.process.times(steps) ** 2))
    assert not np.allclose(warped(["a"], steps=4, sampler=Heun(), key=key).host().images, reference)
    open_ended = dataclasses.replace(plain, final_denoise=False)
    assert not np.allclose(open_ended(["a"], steps=4, sampler=Heun(), key=key).host().images, reference)
    longer = dataclasses.replace(plain, grid=lambda steps: (plain.process, plain.process.times(steps + 1)))
    np.testing.assert_array_equal(longer(["a"], steps=3, sampler=Heun(), key=key).host().images,
                                  plain(["a"], steps=4, sampler=Heun(), key=key).host().images)
    start = dataclasses.replace(plain, grid=lambda steps: (plain.process, plain.process.times(steps)[:1]),
                                final_denoise=False)
    prepared = start.prepare(["a"], key=key, steps=4)
    np.testing.assert_array_equal(start(prepared, steps=4, sampler=Heun(), key=key).images,
                                  np.clip(np.asarray(prepared.noise), -1.0, 1.0))
    with pytest.raises(ValueError, match="different source grid"):
        start(prepared, steps=3, seed=3)
