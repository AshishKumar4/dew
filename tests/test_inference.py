"""Inference: a run directory through to a sample.

A run writes its resolved config as `run.json` next to its checkpoints;
`TextToImage.from_run` builds the objective from it the way the recipe did
and restores the weights, and `from_pretrained` is the same on a pulled hub
snapshot. Everything here drives the real pipeline from a checkpoint the
trainer has just written.
"""

import dataclasses
import json
import shutil
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from test_diffusion_objective import StubText  # noqa: F401  registers "stub_text"

import dew
import dew.nn.backbones  # registers the models
from dew.artifacts import VideoGrid
from dew.config import ModelConfig, RunConfig, TrainerConfig
from dew.data import Dataset, OxfordFlowers, VideoDataset
from dew.diffusion import FlowMatchPredictionTransform
from dew.diffusion.schedules import FlowMatchingScheduler
from dew.inputs import Field, unit_range
from dew.objectives.base import merge
from dew.objectives.diffusion import DiffusionRunConfig, StableDiffusionAutoencoder, TextCondition
from dew.registry import presets, samplers
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


def make_run(directory, preset=presets.EDM(), encoder="stub_text", checkpoint="stub-clip", steps=2):
    """`steps` training steps of the tiny conditional DiT, its checkpoint and
    its `run.json` in `directory`, as the recipe leaves them: the objective is
    the config's own build. A directory that already holds the run resumes
    it to `steps`."""
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
    state = trainer.fit(Dataset(train=lambda partition: Stream(), val=None, records=None, batch=8),
                        steps=steps, log_every=100, checkpoint_every=steps)
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


def test_a_run_saved_before_the_fourier_table_was_stored_samples_and_resumes_as_it_did(
        tmp_path, monkeypatch):
    """A checkpoint written before FourierEmbedding's table became a variable
    lacks it. Loading or resuming the run takes the table from the model's
    init, which draws the one the run trained against, so the run samples
    and trains on bit for bit as it did, on the suite's 8-device mesh too,
    where each device's batch is smaller than the table."""
    from dew.nn.blocks import FourierEmbedding

    tables = []

    def drawn_in_setup(self):  # FourierEmbedding.setup before the table was a variable
        drawn = np.random.RandomState(42).normal(size=(self.features // 2,))
        tables.append(drawn.astype(np.float32) * self.scale)
        self.frequencies = SimpleNamespace(value=jnp.asarray(drawn, dtype=jnp.float32) * self.scale)

    old, resumed_before = tmp_path / "old", tmp_path / "resumed-before"
    with monkeypatch.context() as before:
        before.setattr(FourierEmbedding, "setup", drawn_in_setup)
        make_run(old)
        shutil.copytree(old, resumed_before)
        sampled = TextToImage.from_run(str(old))(["a", "b"], seed=4).host().images
        _, trained = make_run(resumed_before, steps=3)
    assert "constants" not in Checkpoints(str(old)).stored()["params"]

    pipe = TextToImage.from_run(str(old))
    table = pipe.params["constants"]["conditioning"]["time_embed"]["layers_0"]["frequencies"]
    np.testing.assert_array_equal(np.asarray(table), tables[0])
    np.testing.assert_array_equal(pipe(["a", "b"], seed=4).host().images, sampled)

    _, resumed = make_run(old, steps=3)
    np.testing.assert_array_equal(
        np.asarray(resumed.params["constants"]["conditioning"]["time_embed"]["layers_0"]["frequencies"]),
        tables[0])
    for expected, actual in zip(jax.tree.leaves(trained.params["params"]),
                                jax.tree.leaves(resumed.params["params"]), strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


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


def test_the_text_encoder_follows_the_models_compute_dtype(tmp_path):
    """An unset text dtype is the run's own. The tower runs inside every
    step beside the model it conditions, so the precision the run trains in
    is the precision it runs in; the float32 a checkpoint is stored in is
    storage, and `param_dtype` still governs that. A set dtype is kept."""
    config = dataclasses.replace(
        run_config(tmp_path),
        model=ModelConfig("simple_dit", dict(MODEL), dtype="bfloat16", attention_impl="reference"),
        text=TextCondition(encoder="char_table", checkpoint="char_table"))
    encoder = config.build().inputs.conditions["textcontext"].encoder
    assert encoder.dtype == jnp.bfloat16
    assert encoder.param_dtype == "float32"
    assert encoder.params["table"].dtype == jnp.float32

    pinned = dataclasses.replace(config, text=dataclasses.replace(config.text, dtype="float32"))
    assert pinned.build().inputs.conditions["textcontext"].encoder.dtype == jnp.float32


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
        Dataset(train=lambda partition: batches(), val=None, records=None, batch=8), steps=1, log_every=100)

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
                                val_metrics=("psnr",))
    (metric,) = video.build_eval_metrics()
    images = np.zeros((2, 2, 8, 8, 3), np.uint8)
    assert np.isinf(metric.finalize(metric(VideoGrid(unit_range(images)), {"video": images})))
    with pytest.raises(ValueError, match="clip"):
        dataclasses.replace(video, val_metrics=("clip",)).build_eval_metrics()


def test_the_autoencoder_record_carries_its_revision(tmp_path):
    """A run trained with a non-default VAE revision rebuilds from its
    record with that revision."""
    config = dataclasses.replace(
        run_config(tmp_path),
        autoencoder=StableDiffusionAutoencoder(revision="flax", latent_scale=0.5))
    assert DiffusionRunConfig.from_dict(config.to_dict()) == config


def test_guidance_is_a_value_with_its_interval(tmp_path):
    """The run preserves interval and rescaling controls, or disables guidance."""
    config = dataclasses.replace(run_config(tmp_path), guidance=CFG(4.0, (0.2, 0.8), rescale=0.3))
    assert DiffusionRunConfig.from_dict(config.to_dict()) == config

    # The record a command line or a run.json carries builds the same value.
    from_record = DiffusionRunConfig.from_dict(
        {**config.to_dict(), "guidance": {"scale": 4.0, "interval": [0.2, 0.8], "rescale": 0.3}})
    assert from_record.guidance == CFG(4.0, (0.2, 0.8), rescale=0.3)
    assert from_record.build().guidance == CFG(4.0, (0.2, 0.8), rescale=0.3)

    unguided = dataclasses.replace(config, guidance=None)
    assert unguided.build().guidance is None
    assert DiffusionRunConfig.from_dict(unguided.to_dict()).guidance is None




def make_lm_run(directory, *, mesh=None, ema_decay=0.9, max_seq_len=16):
    """Two training steps of a tiny byte-level decoder, its checkpoint and the
    `run.json` the LM recipe writes: the resolved model, tokenizer and budget.
    `max_seq_len` is the model's window, which training at 9 ids does not
    reach."""
    import json

    from dew.objectives.lm import LMObjective, Samples
    from dew.sampling import Sampling
    from dew.training import MeshSpec

    fields = dict(vocab_size=256, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                  mlp_features=32, max_seq_len=max_seq_len)
    model_config = ModelConfig("causal_transformer", fields, dtype="float32", attention_impl="reference")
    objective = LMObjective(model_config.build(), 8, ema_decay=ema_decay,
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
    state = trainer.fit(Dataset(train=lambda partition: Stream(), val=None, records=None, batch=8),
                        steps=2, log_every=100, checkpoint_every=2)
    checkpoints.wait()
    (directory / "run.json").write_text(json.dumps({
        "objective": "lm", "model": dataclasses.asdict(model_config), "tokenizer": "byte",
        "sample_tokens": 4, "sampling": dataclasses.asdict(objective.samples.sampling),
        "ema_decay": ema_decay, "data": {"seq_len": 8}}))
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


def test_pipeline_points_xla_at_the_persistent_compilation_cache(tmp_path, monkeypatch):
    """Loading a task turns the on-disk executable cache on, in the directory
    the library picks, and a directory somebody already chose survives the
    next load: a serving process compiles a shape once, ever."""
    from pathlib import Path

    from dew.telemetry.instrumentation import default_compilation_cache_dir

    make_lm_run(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    previous = jax.config.jax_compilation_cache_dir
    try:
        jax.config.update("jax_compilation_cache_dir", None)
        dew.pipeline(str(tmp_path))
        chosen = default_compilation_cache_dir()
        assert jax.config.jax_compilation_cache_dir == chosen
        assert Path(chosen).is_dir() and str(tmp_path / "xdg") in chosen
        jax.config.update("jax_compilation_cache_dir", str(tmp_path / "mine"))
        dew.pipeline(str(tmp_path))
        assert jax.config.jax_compilation_cache_dir == str(tmp_path / "mine")
    finally:
        jax.config.update("jax_compilation_cache_dir", previous)


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


def test_a_quantized_runs_record_re_wraps_the_model_it_rebuilds(tmp_path):
    """The quantization knob is the trainer's, so `RunConfig.train` writes it
    under `trainer` in run.json and `dew.pipeline` reads it back there: the
    rebuilt model answers the quantized forward on the run's own weights,
    which the fp32 rebuild does not. Observed on CPU: 3e-02 on logits of
    order 1."""
    pytest.importorskip("qwix")
    from dew.objectives.lm import LMObjective
    from dew.training.quantization import Quantization

    batch, seq = 8, 8
    fields = dict(vocab_size=256, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                  mlp_features=32, max_seq_len=16)

    @dataclasses.dataclass(frozen=True)
    class LmRun(RunConfig):
        """The two fields the LM entry of `dew.pipeline` reads beside the model."""
        tokenizer: str = "byte"

    class Stream:
        def __init__(self):
            self.position = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.position += 1
            return {"text": np.random.RandomState(0).randint(1, 250, (batch, seq + 1)).astype(np.int32)}

        def get_state(self):
            return str(self.position).encode()

        def set_state(self, state):
            self.position = int(state.decode())

    config = LmRun(
        model=ModelConfig("causal_transformer", fields, dtype="float32",
                          attention_impl="reference"),
        trainer=TrainerConfig(checkpoint_dir=str(tmp_path), batch_size=batch, steps=1,
                              eval_every=None, checkpoint_every=1, compilation_cache_dir=None,
                              quantization=Quantization()))
    state = config.train(LMObjective(config.model.build(), seq, ema_decay=0.9),
                         Dataset(lambda partition: Stream(), None, None, batch), name="run")
    Checkpoints(str(tmp_path / "run")).wait()

    record = json.loads((tmp_path / "run" / "run.json").read_text())
    assert record["trainer"]["quantization"]["dtype"] == "int8"
    task = dew.pipeline(str(tmp_path / "run"))

    ids = jnp.asarray([[3, 4, 5, 6]], jnp.int32)
    plain = config.model.build()
    quantized = task.model.apply(task.variables, ids)
    assert float(jnp.max(jnp.abs(quantized - plain.apply(task.variables, ids)))) > 0.0
    assert int(state.step) == 1

    # A run.json from before the knob moved keeps the spec at the top level,
    # where the LM recipe's own flag wrote it, and still re-wraps.
    before = {**record, "quantization": record["trainer"]["quantization"],
              "trainer": {**record["trainer"], "quantization": None}}
    (tmp_path / "run" / "run.json").write_text(json.dumps(before))
    older = dew.pipeline(str(tmp_path / "run"))
    np.testing.assert_array_equal(older.model.apply(older.variables, ids), quantized)


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


def test_explicit_average_requests_do_not_substitute_live_weights(tmp_path):
    objective, state = make_lm_run(tmp_path, ema_decay=None)
    with pytest.raises(ValueError, match="no EMA"):
        objective.pipeline(state)
    with pytest.raises(ValueError, match="no EMA"):
        dew.pipeline(str(tmp_path))
    restored = dew.pipeline(str(tmp_path), ema=False)
    live = objective.pipeline(state, ema=False, processor=restored.processor)
    np.testing.assert_array_equal(restored("the ", seed=7).host().tokens,
                                  live("the ", seed=7).host().tokens)


def make_block_run(directory):
    """A block-diffusion run directory: the committed DiffusionGemma fixture's
    weights under a checkpoint, and the `run.json` a block run writes."""
    from pathlib import Path

    from dew.interop import load_pretrained
    from dew.objectives.diffusion.block import BlockDiffusionObjective

    fixture = Path(__file__).resolve().parent / "fixtures/hf/diffusion-gemma-workflow"
    bundle = load_pretrained(str(fixture), dtype="float32", attention_impl="xla", max_seq_len=32)
    objective = BlockDiffusionObjective(bundle.model, prompt_length=3, pretrained=bundle.variables)
    state = Trainer(objective, optax.sgd(0.01), key=jax.random.PRNGKey(2)).initial_state()
    checkpoints = Checkpoints(str(directory))
    checkpoints.save(0, state, None)
    checkpoints.wait()
    config = ModelConfig("diffusion_gemma",
                         {**bundle.config, "max_seq_len": bundle.model.max_seq_len},
                         dtype="float32", attention_impl="auto")
    (directory / "run.json").write_text(json.dumps({
        "objective": "block_diffusion", "model": dataclasses.asdict(config),
        "tokenizer": "byte", "pad_token_id": 0}))


def make_masked_run(directory):
    """A masked-diffusion run directory: the committed llada-tiny weights under
    a checkpoint, and the `run.json` a masked run writes."""
    from pathlib import Path

    from dew.diffusion.discrete import MDLM
    from dew.interop import load_pretrained
    from dew.objectives.diffusion.masked import MaskedDiffusionObjective

    fixture = Path(__file__).resolve().parent / "fixtures/hf/llada-tiny"
    source = load_pretrained(fixture, dtype="float32", attention_impl="xla")
    objective = MaskedDiffusionObjective(source.model, MDLM(mask_id=120)(), 8,
                                         pretrained=source.variables, ema_decay=None)
    state = Trainer(objective, optax.sgd(0.05), key=jax.random.PRNGKey(19)).initial_state()
    checkpoints = Checkpoints(str(directory))
    checkpoints.save(0, state, None)
    checkpoints.wait()
    fields = {name: value for name, value in source.model_config.items()
              if name not in ("dtype", "attention_impl")}
    config = ModelConfig("causal_transformer", fields, dtype="float32", attention_impl="xla")
    (directory / "run.json").write_text(json.dumps({
        "objective": "masked_diffusion", "model": dataclasses.asdict(config),
        "tokenizer": "byte", "sample_tokens": 8}))


@pytest.mark.parametrize("make,task_type,prompt,budget", [
    (make_lm_run, "TextGeneration", "the ", None),
    (make_block_run, "BlockGeneration", [[1, 5, 7]], 3),
    (make_masked_run, "MaskedGeneration", "ab", None)])
def test_every_saved_text_kind_constructs_through_its_own_task_class(
        tmp_path, make, task_type, prompt, budget):
    """`dew.pipeline` is the dispatch and nothing else: for each saved text
    kind the task class's own `from_run` builds the same task from the same
    run directory, and both draw the same text at the same seed."""
    from dew.inference import tasks

    make(tmp_path)
    front = dew.pipeline(str(tmp_path), ema=False)
    direct = getattr(tasks, task_type).from_run(str(tmp_path), ema=False)
    assert type(direct) is type(front) is getattr(tasks, task_type)
    drawn = direct(prompt, budget, seed=5)
    expected = front(prompt, budget, seed=5)
    np.testing.assert_array_equal(drawn.host().tokens, expected.host().tokens)
    assert direct.decode(drawn) == front.decode(expected)
    assert all(isinstance(row, str) for row in direct.decode(drawn))


@pytest.mark.parametrize("kind", ["jepa", "unregistered"])
def test_saved_non_generation_objectives_fail_at_the_front_door(tmp_path, kind):
    import json
    (tmp_path / "run.json").write_text(json.dumps({"objective": kind}))
    with pytest.raises(TypeError, match="no saved generation task") as refusal:
        dew.pipeline(str(tmp_path))
    named = str(refusal.value)
    assert kind in named
    for supported in ("diffusion", "lm", "dpo", "grpo", "ppo", "block_diffusion",
                      "masked_diffusion"):
        assert supported in named


def test_saved_sampling_policy_survives_a_disabled_preview_budget(tmp_path):
    import json

    from dew.sampling import Sampling

    objective, state = make_lm_run(tmp_path)
    policy = Sampling(temperature=0.37, top_k=3, eos_id=255)
    path = tmp_path / "run.json"
    record = json.loads(path.read_text())
    record.update(sample_tokens=0, sampling=dataclasses.asdict(policy))
    path.write_text(json.dumps(record))
    task = dew.pipeline(str(tmp_path))
    assert task.max_new_tokens is None
    with pytest.raises(ValueError, match="max_new_tokens is required"):
        task([[1, 2]], seed=4)
    expected = objective.policy(state.averaged, policy)([[1, 2]], 3, seed=4).host()
    actual = task([[1, 2]], 3, seed=4).host()
    np.testing.assert_array_equal(actual.tokens, expected.tokens)
    np.testing.assert_allclose(actual.behavior_log_probs, expected.behavior_log_probs, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("compute,storage", [(None, None), ("bfloat16", None),
                                               (None, "float32"), ("bfloat16", "float32")])
def test_saved_diffusion_precision_reconstructs_owners_without_source_weights(
        tmp_path, monkeypatch, compute, storage):
    import tarfile
    from pathlib import Path

    import jax.numpy as jnp
    from flax.core import unfreeze

    import dew.nn.autoencoders.vae as vae_loader
    import dew.nn.text_encoders as text_loader
    from dew.inference.pipeline import place
    from dew.inputs import CLIPText
    from dew.nn.autoencoders import StableDiffusionVAE
    from dew.objectives.base import FROZEN

    fixtures = Path(__file__).parent / "fixtures"
    with tarfile.open(fixtures / "tiny_diffusers.tar.xz") as archive:
        archive.extractall(tmp_path / "source", filter="data")
    directory = tmp_path / "run"
    config = dataclasses.replace(
        run_config(directory),
        model=ModelConfig("simple_dit", {**MODEL, "patch_size": 2},
                          dtype="float32", attention_impl="reference"),
        text=TextCondition(encoder="clip_text", checkpoint=str(fixtures / "clip/tiny"), dtype="float32"),
        autoencoder=StableDiffusionAutoencoder(modelname=str(tmp_path / "source/sd/vae"), dtype="float32"))
    objective = config.build()
    initial = Trainer(objective, optax.sgd(0.01), key=jax.random.PRNGKey(3)).initial_state()
    params = unfreeze(jax.tree.map(lambda leaf: (leaf + 0.015625).astype(jnp.bfloat16), initial.params))
    params = {**params, "constants": {**params["constants"], "scale": jnp.asarray([1.003], jnp.float32)}}
    params["params"]["packed"] = jnp.asarray([16777217], jnp.int32)
    params["encoders"]["textcontext"]["constants"] = {
        "scale": jnp.asarray([3.14159], jnp.float32), "ids": jnp.asarray([16777217], jnp.int32)}
    first = next(iter(params["params"]))
    averaged = {"params": {first: jax.tree.map(lambda leaf: leaf.astype(jnp.float32) + 0.03125,
                                             params["params"][first])},
                "constants": {"scale": jnp.asarray([1.007], jnp.float32)}}
    state = dataclasses.replace(initial, params=params, ema=averaged)
    checkpoints = Checkpoints(str(directory))
    checkpoints.save(0, state, None)
    checkpoints.wait()
    config.save(str(directory))

    def forbid_weights(*args, **kwargs):
        raise AssertionError("restoring a run opened source weight shards")

    monkeypatch.setattr(text_loader, "_read_tensors", forbid_weights)
    monkeypatch.setattr(vae_loader, "_read_vae_weights", forbid_weights)
    restored = dew.pipeline(str(directory), dtype=compute, param_dtype=storage)
    assert isinstance(restored, TextToImage)
    merged = merge(params, averaged)

    def expected_leaf(path, leaf):
        keys = tuple(entry.key for entry in path)
        parameter = (keys[0] in ("params", FROZEN, "autoencoder") or
                     keys[:3] in (("encoders", "textcontext", "params"),
                                  ("encoders", "textcontext", FROZEN)))
        return leaf.astype(storage) if storage and parameter and jnp.issubdtype(leaf.dtype, jnp.floating) else leaf

    expected_params = jax.tree_util.tree_map_with_path(expected_leaf, merged)
    for expected, actual in zip(jax.tree.leaves(expected_params), jax.tree.leaves(restored.params), strict=True):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    # Equal storage is insufficient for bitwise comparison: the unplaced
    # oracle otherwise compiles one row while the restored mesh pads to eight.
    expected_params = place(expected_params, None, None)
    encoder = objective.inputs.conditions["textcontext"].encoder
    assert isinstance(encoder, CLIPText)
    vae = objective.autoencoder
    assert isinstance(vae, StableDiffusionVAE)
    target = jnp.float32 if compute is None else jnp.bfloat16
    expected_encoder = dataclasses.replace(encoder, transformer=encoder.transformer.clone(dtype=target),
                                           dtype=target, params=expected_params["encoders"]["textcontext"])
    expected_vae = StableDiffusionVAE(model=vae.model.clone(dtype=target),
                                    params=expected_params["autoencoder"], dtype=target,
                                    latent_shift=vae.latent_shift, latent_scale=vae.latent_scale)
    condition = dataclasses.replace(objective.inputs.conditions["textcontext"], encoder=expected_encoder)
    inputs = dataclasses.replace(objective.inputs, conditions={"textcontext": condition})
    reference = dataclasses.replace(
        TextToImage.from_objective(objective, expected_params),
        model=objective.model.clone(dtype=target),
        inputs=inputs,
        autoencoder=expected_vae,
        # The unconditional branch is encoded from the weights the task is
        # built over, so the oracle encodes it from the restored ones too.
        blank=lambda given: jax.tree.map(
            lambda held, value: jnp.asarray(held, value.dtype),
            {"textcontext": expected_encoder.encode(
                expected_params["encoders"]["textcontext"],
                expected_encoder.tokenize([condition.unconditional]))}, given))

    np.testing.assert_array_equal(restored(["a red bird"], steps=2, seed=5).host().images,
                                  reference(["a red bird"], steps=2, seed=5).host().images)
    owned = restored.inputs.conditions["textcontext"].encoder.params
    for owner_leaf, bound_leaf in zip(jax.tree.leaves(owned),
                                      jax.tree.leaves(restored.params["encoders"]["textcontext"]), strict=True):
        assert owner_leaf.dtype == bound_leaf.dtype and owner_leaf.sharding == bound_leaf.sharding
        np.testing.assert_array_equal(owner_leaf, bound_leaf)
    assert isinstance(restored.autoencoder, StableDiffusionVAE)
    for owner_leaf, bound_leaf in zip(jax.tree.leaves(restored.autoencoder.params),
                                      jax.tree.leaves(restored.params["autoencoder"]), strict=True):
        assert owner_leaf.dtype == bound_leaf.dtype and owner_leaf.sharding == bound_leaf.sharding
        np.testing.assert_array_equal(owner_leaf, bound_leaf)


@pytest.mark.parametrize("compute,storage", [("bfloat16", None), (None, "bfloat16")])
def test_saved_decoder_compute_and_storage_overrides_generate_from_the_same_weights(tmp_path, compute, storage):
    import jax.numpy as jnp

    from dew.inference import TextGeneration
    from dew.objectives.base import FROZEN

    _, state = make_lm_run(tmp_path)
    baseline = dew.pipeline(str(tmp_path))
    assert isinstance(baseline, TextGeneration)
    first = next(iter(state.params["params"]))

    def move_to_frozen(variables):
        trainable = dict(variables["params"])
        frozen = {**variables.get(FROZEN, {}), first: trainable.pop(first)}
        return {**variables, "params": trainable, FROZEN: frozen}

    assert state.ema is not None
    step = int(state.step) + 1
    frozen = dataclasses.replace(state, step=jnp.asarray(step, state.step.dtype),
                                 params=move_to_frozen(state.params), ema=move_to_frozen(state.ema))
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(step, frozen, None)
    checkpoints.wait()
    params = baseline.variables
    if storage is not None:
        params = {**params, "params": jax.tree.map(lambda leaf: leaf.astype(storage), params["params"])}
    target = jnp.float32 if compute is None else jnp.bfloat16
    expected = dataclasses.replace(baseline, model=baseline.model.clone(dtype=target), variables=params)
    actual = dew.pipeline(str(tmp_path), dtype=compute, param_dtype=storage)
    assert isinstance(actual, TextGeneration)
    for reference, restored in zip(jax.tree.leaves(params), jax.tree.leaves(actual.variables), strict=True):
        assert restored.dtype == reference.dtype
        np.testing.assert_array_equal(restored, reference)
    wanted = expected([[1, 2]], 2, seed=7).host()
    result = actual([[1, 2]], 2, seed=7).host()
    np.testing.assert_array_equal(result.tokens, wanted.tokens)
    np.testing.assert_array_equal(result.raw_log_probs, wanted.raw_log_probs)


def test_saved_bare_encoder_weights_follow_storage_without_changing_compute(tmp_path):
    import jax.numpy as jnp

    from dew.inputs import CharTable

    objective, state = make_run(tmp_path, encoder="char_table", checkpoint="char_table")
    restored = dew.pipeline(str(tmp_path), dtype="float32", param_dtype="bfloat16")
    assert isinstance(restored, TextToImage)
    encoder = restored.inputs.conditions["textcontext"].encoder
    assert isinstance(encoder, CharTable)
    stored = merge(state.params, state.ema)
    expected_vars = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), stored)
    table = restored.params["encoders"]["textcontext"]["table"]
    assert table.dtype == jnp.bfloat16
    assert encoder.params["table"].dtype == table.dtype
    np.testing.assert_array_equal(encoder.params["table"], table)
    np.testing.assert_array_equal(table, expected_vars["encoders"]["textcontext"]["table"])
    tokens = encoder.tokenize(["ab"])
    hidden = encoder.encode(encoder.params, tokens).hidden
    assert hidden.dtype == jnp.float32
    np.testing.assert_array_equal(hidden, table[tokens["input_ids"]].astype(jnp.float32))
    original = objective.inputs.conditions["textcontext"]
    assert isinstance(original.encoder, CharTable)
    reference_encoder = dataclasses.replace(original.encoder, dtype=jnp.float32,
                                           params=expected_vars["encoders"]["textcontext"])
    expected = dataclasses.replace(TextToImage.from_objective(objective, expected_vars),
        inputs=dataclasses.replace(objective.inputs, conditions={
            "textcontext": dataclasses.replace(original, encoder=reference_encoder)}))
    np.testing.assert_array_equal(restored(["ab"], steps=2, seed=9).host().images,
                                  expected(["ab"], steps=2, seed=9).host().images)



