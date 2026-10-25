"""Inference: a run directory through to a sample.

A run writes its resolved config as `run.json` next to its checkpoints;
`TextToImage.from_run` builds the objective from it the way the recipe did
and restores the weights, and `from_pretrained` is the same on a pulled hub
snapshot. Everything here drives the real pipeline from a checkpoint the
trainer has just written.
"""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import Replaying
from diffusion_stubs import STUB_TEXT
from steady_state import steady_state

import dew
import dew.nn.backbones  # registers the models
from dew.artifacts import VideoGrid
from dew.config import MetricConfig, ModelConfig, ObjectiveConfig, TrainerConfig
from dew.config.diffusion import DiffusionRunConfig
from dew.data import ByteTokenizer, presets
from dew.data.dataset import Loader
from dew.data.sources import source
from dew.data.sources.files import Files
from dew.diffusion.presets import EDM, Flow
from dew.inference import RunProcessor, TextGeneration
from dew.inputs import Field, TextCondition, unit_range
from dew.objectives.base import merge
from dew.sampling import CFG, DPMSolverSDE, Euler, Heun, TextToImage
from dew.sampling.pipelines import Images
from dew.telemetry.instrumentation import Compiles
from dew.training import Checkpoints, LocalTracker, Trainer

RES = 8
MODEL = {"patch_size": 4, "emb_features": 16, "num_layers": 1, "num_heads": 2, "mlp_ratio": 1}


_DEFAULT_RUN_CONFIG_PRESET = EDM()


def run_config(directory, preset=_DEFAULT_RUN_CONFIG_PRESET, encoder=STUB_TEXT, checkpoint="stub-clip"):
    """The resolved config of a tiny conditional DiT run in `directory`; the
    text condition names the registered stub encoder by default."""
    return DiffusionRunConfig(
        model=ModelConfig("simple_dit", {**MODEL, "dtype": "float32", "attention_impl": "reference"}),
        data=source("tfds://oxford_flowers102").pipe(presets.images, RES, batch=8),
        trainer=TrainerConfig(checkpoint_dir=str(directory), steps=2, keep=1),
        preset=preset, text=TextCondition(encoder=encoder, checkpoint=checkpoint),
        objective=ObjectiveConfig("diffusion", {"solver": Euler(), "steps": 3}))


def stating(config, **fields):
    """`config` with its objective stating `fields` beside what it states."""
    return dataclasses.replace(config, objective=dataclasses.replace(
        config.objective, fields={**config.objective.fields, **fields}))


_DEFAULT_MAKE_RUN_PRESET = EDM()


def make_run(
    directory, preset=_DEFAULT_MAKE_RUN_PRESET, encoder=STUB_TEXT, checkpoint="stub-clip", steps=2, **stated
):
    """`steps` training steps of the tiny conditional DiT, its checkpoint and
    its `run.json` in `directory`, as the recipe leaves them: the objective is
    the config's own build, stating `stated` beside its own fields. A
    directory that already holds the run resumes it to `steps`."""
    config = stating(run_config(directory, preset, encoder, checkpoint), **stated)
    objective = config.build()
    encoder = objective.inputs.conditions["textcontext"].encoder
    images = np.tile(np.linspace(0, 255, RES, dtype=np.float32)[None, :, None, None],
                     (8, 1, RES, 3)).astype(np.uint8)
    batch = {"image": images, "text": encoder.tokenize(["a", "b", "c", "d", "e", "f", "g", "h"])}

    checkpoints = Checkpoints(str(directory), keep=1)
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.PRNGKey(0),
                      checkpoints=checkpoints)
    stream = Loader(train=lambda partition: Replaying(lambda position: batch), val=None, records=None,
                    batch=8)
    state = trainer.fit(stream,
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
        front(["a water lily", "a sunflower"], steps=3, guidance=2.0, key=0).host().images,
        pipe(["a water lily", "a sunflower"], steps=3, guidance=2.0, key=0).host().images)


def test_repeated_image_requests_reuse_their_programs_and_read_back_only_results(tmp_path):
    """Prompts of the same count and sampling settings after the first run
    the programs it compiled, and nothing comes back to the host but the
    images (`steady_state`). A request's prompts and key reach the device
    as it arrives, so only reads are held."""
    make_run(tmp_path)
    pipe = TextToImage.from_run(str(tmp_path))
    keys = [jax.random.key(index) for index in range(3)]
    pipe(["a water lily", "a sunflower"], steps=3, guidance=None, key=keys[0])
    with steady_state(allow=("host_to_device",)):
        prompts = [["a rose", "a tulip"], ["a daisy", "an iris"]]
        results = [pipe(rows, steps=3, guidance=None, key=key)
                   for rows, key in zip(prompts, keys[1:], strict=True)]
        jax.block_until_ready([result.images for result in results])
    assert [result.host().images.shape for result in results] == [(2, RES, RES, 3)] * 2


def test_pil_bytes_and_unit_range_are_diffusers_image_processor():
    """Diffusers 0.34.0's `VaeImageProcessor` is the reference both ways.
    `postprocess(..., output_type="pil")` writes the same bytes as `pil()` on
    every level's centre, every tie between two levels, the float32
    neighbours either side of each, and values past the range (both round
    half to even in float32, so both miss exact rounding on the same 191
    near-tie inputs). `unit_range` holds the float64 rule against
    `pil_to_numpy`'s division and `normalize` over all 256 bytes (3.7e-8
    against 2.7e-8 rms)."""
    import torch
    from diffusers.image_processor import VaeImageProcessor
    from reference_error import assert_as_exact_as_the_reference

    processor = VaeImageProcessor()
    levels = np.arange(256, dtype=np.float64)
    marks = np.concatenate([levels / 127.5 - 1, (levels[:-1] + 0.5) / 127.5 - 1]).astype(np.float32)
    neighbours = [np.nextafter(marks, np.float32(side)) for side in (-np.inf, np.inf)]
    values = np.concatenate([marks, *neighbours, np.asarray([-1.5, -1.0001, 1.0001, 3.0], np.float32)])
    pixels = np.repeat(values.reshape(1, 1, -1, 1), 3, axis=-1)
    ours = np.asarray(Images(jnp.asarray(pixels), rows=1).pil()[0])
    channels_first = torch.from_numpy(pixels).permute(0, 3, 1, 2)
    theirs = np.asarray(processor.postprocess(channels_first, output_type="pil")[0])
    np.testing.assert_array_equal(ours, theirs)

    data = np.arange(256, dtype=np.uint8)
    normalized = processor.normalize(torch.from_numpy(data.astype(np.float32) / 255.0)).numpy()
    assert_as_exact_as_the_reference(np.asarray(unit_range(data)), normalized, data / 127.5 - 1, "unit_range")

    padded = np.concatenate([pixels, pixels, np.full_like(pixels, -1)], axis=0)
    pictures = Images(jnp.asarray(padded), rows=2).pil()
    assert len(pictures) == 2
    for picture in pictures:
        assert picture.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(picture), theirs)
    gray = Images(jnp.ones((1, 2, 3, 1)), rows=1).pil()[0]
    assert gray.mode == "L" and gray.size == (3, 2) and np.all(np.asarray(gray) == 255)
    with pytest.raises(ValueError):
        Images(None, rows=1, latents=jnp.zeros((1, 2, 2, 4))).pil()
    with pytest.raises(ValueError):
        Images(jnp.zeros((1, 4, 2, 3, 3)), rows=1).pil()


def test_a_run_that_keeps_one_copy_of_its_weights_samples_it_by_default(tmp_path):
    """A published run may hold its weights once, with no average beside
    them; the default reads that copy, and asking for an average it does not
    keep is refused."""
    objective, state = make_run(tmp_path / "kept")
    single = tmp_path / "single"
    checkpoints = Checkpoints(str(single), keep=1)
    checkpoints.save(int(state.step), state.replace(ema=None), None, artifact=objective.inference_record())
    checkpoints.wait()
    stating(run_config(single), ema_decay=None).save(str(single))

    expected = (
        TextToImage.from_run(str(tmp_path / "kept"), ema=False)(["a lily"], steps=3, key=0).host().images
    )
    np.testing.assert_array_equal(TextToImage.from_run(str(single))(["a lily"], steps=3, key=0).host().images,
                                  expected)
    with pytest.raises(ValueError):
        TextToImage.from_run(str(single), ema=True)


def test_sampler_and_guidance_are_call_arguments(tmp_path):
    """Two steps of training leave the zero-initialised head near zero, so
    the weights are nudged off it for the conditional and unconditional
    branches to differ; then guidance is visible in the sample."""
    make_run(tmp_path)
    loaded = TextToImage.from_run(str(tmp_path))
    pipe = dataclasses.replace(loaded, variables=jax.tree.map(lambda leaf: leaf + 0.05, loaded.variables))
    key = jax.random.PRNGKey(1)
    plain = pipe(["x"], steps=8, guidance=None, solver=Heun(), key=key).host().images
    guided = (
        pipe(["x"], steps=8, guidance=CFG(4.0, interval=(0.2, 0.8)), solver=Heun(), key=key).host().images
    )
    assert plain.shape == guided.shape == (1, RES, RES, 3)
    assert not np.allclose(plain, guided)
    assert np.array_equal(pipe(["x"], steps=8, guidance=None, solver=Heun(), key=key).host().images, plain)
    assert np.array_equal(pipe(["x"], steps=8, guidance=4.0, solver=Heun(), key=key).host().images,
                          pipe(["x"], steps=8, guidance=CFG(4.0), solver=Heun(), key=key).host().images)


def test_another_guidance_scale_seed_or_grid_reuses_the_compiled_walk(tmp_path):
    """A request's CFG scale, a Brownian solver's seed and an explicit grid's values are data to the
    image walk: another value changes the sample and runs the program the first value compiled."""

    make_run(tmp_path)
    loaded = TextToImage.from_run(str(tmp_path))
    pipe = dataclasses.replace(loaded, variables=jax.tree.map(lambda leaf: leaf + 0.05, loaded.variables))
    grid = np.asarray(pipe.process.times(8), np.float32)
    nudged = grid.copy()
    nudged[4] = 0.5 * (grid[3] + grid[4])

    def draw(scale=4.0, seed=1, times=None):
        prepared = pipe.prepare(["x"], key=0, steps=8, times=times)
        return pipe(prepared, steps=8, guidance=CFG(scale), solver=DPMSolverSDE(seed=seed),
                    key=jax.random.PRNGKey(1)).host().images

    first = draw(times=grid)
    # The walk is `dew.sampling.pipelines._run`'s `run`.
    with Compiles("run") as compiled:
        for other in (draw(scale=2.0, times=grid), draw(seed=2, times=grid), draw(times=nudged)):
            assert not np.allclose(other, first)
        assert np.array_equal(draw(times=grid), first)
        assert np.array_equal(draw(), first)
    assert compiled.count == 0


def test_build_eval_metrics_follows_the_sample_field(tmp_path):
    """A video run scores its `VideoGrid` against its `video` field: a PSNR stating no field reads
    that grid there, and an image-only metric raises a ValueError naming it in
    `build_eval_metrics`, before the trainer."""
    video = dataclasses.replace(run_config(tmp_path),
                                data=Files("clips", format="files").pipe(presets.video, 2 / 25, frames=2,
                                                                          fps=25, size=8, batch=2),
                                val_metrics=(MetricConfig("psnr"),))
    (metric,) = video.build_eval_metrics()
    images = np.zeros((2, 2, 8, 8, 3), np.uint8)
    assert np.isinf(metric.finalize(metric(VideoGrid(unit_range(images)), {"video": images})))
    with pytest.raises(ValueError):
        dataclasses.replace(video, val_metrics=(MetricConfig("clip"),)).build_eval_metrics()


TINY_CLIP = str(Path(__file__).parent / "fixtures" / "clip" / "tiny")


def validated_run(directory, *, validates: bool, **fields):
    """Two steps of the tiny conditional DiT on a real tiny CLIP's text, `fields` replacing the run's
    own, scored by the metrics the run builds for data that `validates` or does not, and validated at
    its end on one held-out batch where it does: the scalars its tracker logged, and its last
    checkpointed step."""
    config = dataclasses.replace(run_config(directory, encoder="clip_text", checkpoint=TINY_CLIP), **fields)
    objective = config.build()
    encoder = objective.inputs.conditions["textcontext"].encoder
    images = np.tile(np.linspace(0, 255, RES, dtype=np.float32)[None, :, None, None],
                     (8, 1, RES, 3)).astype(np.uint8)
    batch = {"image": images, "text": encoder.tokenize(["a", "b", "c", "d", "e", "f", "g", "h"])}
    data = Loader(train=lambda partition: Replaying(lambda position: batch),
                  val=(lambda partition: iter([batch])) if validates else None, records=None, batch=8)
    checkpoints = Checkpoints(str(directory / "checkpoints"), keep=1)
    with LocalTracker(directory / "tracking") as tracker:
        trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.PRNGKey(0),
                          checkpoints=checkpoints, tracker=tracker)
        try:
            trainer.fit(data, steps=2, log_every=100, checkpoint_every=2,
                        metrics=config.build_eval_metrics(data.val is not None))
        finally:
            checkpoints.wait()
    with LocalTracker(directory / "tracking") as journal:
        return journal.scalars(), checkpoints.latest


def test_an_image_run_validates_on_clip_by_default_with_the_model_its_metric_names(tmp_path, monkeypatch):
    """An image run's default metric is CLIP's distance where its data validates: the run trains,
    validates and logs `val/clip_similarity` from the CLIP its metric names, here the committed tiny
    one, given on the command line (`MetricConfig("clip", {"modelname": ...})`, what the default
    holds with the published model, which only a download reads). Data with no validation split
    builds no default metric and trains."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    # Named as a command line names it: one JSON list of the metrics' records.
    named = DiffusionRunConfig.cli(["--val-metrics", json.dumps([{"name": "clip",
                                                                  "fields": {"modelname": TINY_CLIP}}])])
    assert named.val_metrics == (MetricConfig("clip", {"modelname": TINY_CLIP}),)
    scalars, _ = validated_run(tmp_path / "named", validates=True, val_metrics=named.val_metrics)
    final = {name: value for step, values in scalars if step == 2 for name, value in values.items()}
    assert 0.0 <= final["val/clip_similarity"] <= 2.0
    assert run_config(tmp_path).build_eval_metrics(validates=True)[0].name == "clip_similarity"
    scalars, latest = validated_run(tmp_path / "unvalidated", validates=False)
    assert latest == 2 and not any(name.startswith("val/") for _, values in scalars for name in values)


def test_a_metric_a_run_names_without_a_validation_split_refuses_before_an_update(tmp_path):
    """A run that names a metric asks for it whatever its data holds, so data with no validation
    split refuses the run in `Trainer.fit`, before its first update writes a checkpoint."""
    named = (MetricConfig("psnr"),)
    # The metric builds for data that does not validate; the run is what refuses it.
    run = dataclasses.replace(run_config(tmp_path), val_metrics=named)
    assert len(run.build_eval_metrics(validates=False)) == 1
    with pytest.raises(ValueError):
        validated_run(tmp_path, validates=False, val_metrics=named)
    assert Checkpoints(str(tmp_path / "checkpoints"), keep=1).latest is None


def make_lm_run(directory, *, mesh=None, ema_decay=0.9, max_seq_len=16, fields=None):
    """Two training steps of a tiny byte-level decoder, its checkpoints, whose
    record names the byte tokenizer it decodes through, and the `run.json` the
    LM recipe writes: the resolved model, tokenizer and budget.
    `max_seq_len` is the model's window, which training at 9 ids does not
    reach; `fields` override the decoder's shape."""
    import json

    from dew.objectives.lm import LMObjective, Samples
    from dew.sampling import Sampling
    from dew.training import MeshSpec

    fields = {"vocab_size": 256, "emb_features": 16, "num_layers": 1, "num_heads": 2, "head_dim": 8,
              "mlp_features": 32, "max_seq_len": max_seq_len, **(fields or {})}
    model_config = ModelConfig("causal_transformer", {**fields,
                                                    "dtype": "float32", "attention_impl": "reference"})
    objective = LMObjective(model_config.build(), 8, ema_decay=ema_decay,
                            samples=Samples([1, 2, 3], 4,
                                            sampling=Sampling(temperature=0, eos_token_ids=255)),
                            processor=RunProcessor(ByteTokenizer()))
    rng = np.random.RandomState(0)
    batch = {"text": rng.randint(1, 250, (8, 9)).astype(np.int32)}

    checkpoints = Checkpoints(str(directory), keep=1)
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.PRNGKey(0), checkpoints=checkpoints,
                      mesh=MeshSpec() if mesh is None else mesh)
    stream = Loader(train=lambda partition: Replaying(lambda position: batch), val=None, records=None,
                    batch=8)
    state = trainer.fit(stream,
                        steps=2, log_every=100, checkpoint_every=2)
    checkpoints.wait()
    (directory / "run.json").write_text(json.dumps({
        "objective": "lm", "model": dataclasses.asdict(model_config), "tokenizer": "byte",
        "max_new_tokens": 4, "sampling": dataclasses.asdict(objective.samples.sampling),
        "ema_decay": ema_decay, "data": {"seq_len": 8}}))
    return objective, state


def test_every_route_generates_through_the_same_packed_projections(tmp_path):
    """A run's task and a source's hold their query/key/value and gate/up packed, packed once at
    their load from the weights they own; a live objective's task keeps the trainer's own arrays,
    bound in place, and packs them inside each generation's compiled call. Every route draws, bit
    for bit, what a task over weights packed beforehand draws, and the run's and the objective's
    draws agree. The run task's tree is a serving layout: an objective built over it holds, and an
    export of it writes, the run's own tree, bit for bit."""
    from flax.core import unfreeze

    from dew.inference.projections import inference_projections, projection_groups
    from dew.interop import Pretrained
    from dew.objectives.lm import LMObjective
    from dew.quantization import Quantization
    from dew.sampling import Sampling

    run = tmp_path / "run"
    run.mkdir()
    objective, state = make_lm_run(run, ema_decay=None)
    Pretrained.from_run(str(run)).save(tmp_path / "source")
    routes = {"run": dew.pipeline(str(run)), "source": dew.pipeline(str(tmp_path / "source")),
              "objective": objective.pipeline(state)}
    for name, task in routes.items():
        attention = task.variables["params"]["layers_0"]["self_attn"]
        owned = name != "objective"
        assert ("qkv_proj" in attention) == owned, name
        assert ({"q_proj", "k_proj", "v_proj"} & set(attention) == set()) == owned, name
    for bound, live in zip(jax.tree.leaves(routes["objective"].variables), jax.tree.leaves(state.variables),
                           strict=True):
        assert bound is live
    greedy = Sampling(temperature=0, eos_token_ids=255)
    drawn = {}
    for name, task in routes.items():
        packed = inference_projections(task.model, task.variables)
        prepacked = TextGeneration(task.model, packed, task.processor)
        drawn[name] = task("the ", 4, key=2, sampling=greedy).host()
        expected = prepacked("the ", 4, key=2, sampling=greedy).host()
        np.testing.assert_array_equal(drawn[name].tokens, expected.tokens, err_msg=name)
        np.testing.assert_array_equal(drawn[name].raw_log_probs, expected.raw_log_probs, err_msg=name)
    np.testing.assert_array_equal(drawn["run"].raw_log_probs, drawn["objective"].raw_log_probs)
    trained = Pretrained.from_run(str(run))
    # A model quantized on the query alone would not pack the group anew; the tree still stores it packed.
    queried = Quantization(patterns=(".*/q_proj",)).apply(routes["run"].model)
    assert "qkv_proj" not in {group.packed for group in projection_groups(queried, trained.variables)}
    for model in (routes["run"].model, queried):
        held = LMObjective(model, 8, variables=routes["run"].variables).variables
        assert jax.tree.structure(unfreeze(held)) == jax.tree.structure(unfreeze(trained.variables))
        for ours, theirs in zip(jax.tree.leaves(held), jax.tree.leaves(trained.variables), strict=True):
            np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
    exported, canonical = trained.export(routes["run"].variables), trained.export()
    assert set(exported) == set(canonical)
    for name, tensor in canonical.items():
        np.testing.assert_array_equal(np.asarray(exported[name]), np.asarray(tensor), err_msg=name)


def test_a_generation_reads_its_models_groups_once_for_every_call(tmp_path):
    """A generate call finds the groups a borrowed tree packs in the call (`pending`) without
    binding the model again: the groups a model declares are a function of the model and its
    tree's shapes, which binding a deep decoder to read took tens of milliseconds of every call.
    A decoder that counts its declarations is asked once over every call on one tree, draws what
    a task over its tree packed beforehand draws, and is asked again for a tree of another shape."""
    from dew.inference.projections import inference_projections, projection_groups
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.sampling import Sampling

    asked = []

    class Counting(CausalTransformer):
        def inference_projection_groups(self, variables):
            asked.append(1)
            return super().inference_projection_groups(variables)

    run = tmp_path / "run"
    run.mkdir()
    objective, state = make_lm_run(run, ema_decay=None)
    model = objective.pipeline(state).model
    counting = Counting(**{field.name: getattr(model, field.name) for field in dataclasses.fields(model)
                           if field.init and field.name not in ("parent", "name")})
    task = TextGeneration(counting, state.variables, RunProcessor(ByteTokenizer()))
    greedy = Sampling(temperature=0, eos_token_ids=255)
    drawn = [task("the ", 2, key=key, sampling=greedy).host() for key in range(3)]
    assert len(asked) == 1
    # The kept plan is the plan: the draws are a prepacked task's, bit for bit.
    prepacked = TextGeneration(model, inference_projections(model, state.variables), task.processor)
    for key, generation in enumerate(drawn):
        expected = prepacked("the ", 2, key=key, sampling=greedy).host()
        np.testing.assert_array_equal(generation.raw_log_probs, expected.raw_log_probs)
    # A tree of another shape is asked again: the plan is kept per model and tree shape.
    widened = jax.tree.map(lambda leaf: leaf, dict(state.variables))
    widened["params"] = {**widened["params"], "spare": {"kernel": jnp.zeros((2, 2))}}
    projection_groups(counting, widened)
    assert len(asked) == 2


@pytest.mark.mesh(devices=3)
def test_a_run_on_a_mesh_packs_the_projections_its_layout_keeps_whole(tmp_path):
    """A packed kernel's columns are its members' outputs side by side, so a run's task packs a
    group only where the layout leaves each member's output axis whole, and the packed kernel
    keeps the members' placement. FSDP over three devices splits the inputs and both groups pack;
    a tensor axis of three splits three query heads' twelve output units, so the query, key and
    value stay apart, but not the MLP's 32, so the gate and up projections pack. On each mesh the
    task draws the tokens it draws on one device."""
    from dew.sampling import Sampling
    from dew.training import Layout, MeshSpec

    run = tmp_path / "run"
    run.mkdir()
    make_lm_run(run, ema_decay=None, fields={"emb_features": 12, "num_heads": 3, "num_kv_heads": 1,
                                             "head_dim": 4})
    devices = jax.devices()[:3]
    layout = Layout(min_shard=1, tolerance=1.0)
    greedy = Sampling(temperature=0, eos_token_ids=255)
    alone = dew.pipeline(str(run), mesh=MeshSpec().build(devices[:1]), layout=layout)
    here = alone("the ", 4, key=2, sampling=greedy).host()
    for spec, packed in ((MeshSpec(fsdp=3), {"qkv_proj", "gate_up_proj"}),
                         (MeshSpec(tensor=3), {"gate_up_proj"})):
        task = dew.pipeline(str(run), mesh=spec.build(devices), layout=layout)
        layer = task.variables["params"]["layers_0"]
        held = {*layer["self_attn"], *layer["mlp"]}
        assert held & {"qkv_proj", "gate_up_proj"} == packed, spec
        assert ({"q_proj", "k_proj", "v_proj"} <= held) == ("qkv_proj" not in packed), spec
        for name in packed:
            kernel = (layer["self_attn"] if name == "qkv_proj" else layer["mlp"])[name]["kernel"]
            assert kernel.sharding.shard_shape(kernel.shape)[-1] == kernel.shape[-1], (spec, name)
            assert kernel.sharding.shard_shape(kernel.shape)[0] == kernel.shape[0] // spec.fsdp, (spec, name)
        there = task("the ", 4, key=2, sampling=greedy).host()
        np.testing.assert_array_equal(there.tokens, here.tokens, err_msg=str(spec))


def test_objective_pipeline_binds_the_trained_state_in_place(tmp_path):
    """A just-trained state is the pipeline: the averaged weights when the
    objective keeps them, the live ones on request, the objective's own
    sampling defaults, and the same images `from_run` restores."""
    objective, state = make_run(tmp_path)
    pipe = objective.pipeline(state)
    assert isinstance(pipe, TextToImage)
    assert (pipe.steps, pipe.guidance, pipe.solver) == (objective.steps, objective.guidance, objective.solver)
    for expected, bound in zip(jax.tree.leaves(state.averaged), jax.tree.leaves(pipe.variables), strict=True):
        assert bound is expected
    live = objective.pipeline(state, ema=False)
    for expected, bound in zip(jax.tree.leaves(state.variables), jax.tree.leaves(live.variables),
                               strict=True):
        assert bound is expected
    drawn = pipe(["a", "b"], key=4).host().images
    assert drawn.shape == (2, RES, RES, 3)
    np.testing.assert_allclose(TextToImage.from_run(str(tmp_path))(["a", "b"], key=4).host().images, drawn,
                               atol=2e-6, rtol=2e-6)

    restored = TextToImage.from_run(str(tmp_path))
    restored_live = TextToImage.from_run(str(tmp_path), ema=False)
    for expected, loaded in zip(jax.tree.leaves(state.averaged["params"]),
                                jax.tree.leaves(restored.variables["params"]), strict=True):
        np.testing.assert_allclose(np.asarray(loaded), np.asarray(expected))
    for expected, loaded in zip(jax.tree.leaves(state.variables["params"]),
                                jax.tree.leaves(restored_live.variables["params"]), strict=True):
        np.testing.assert_allclose(np.asarray(loaded), np.asarray(expected))
    assert not all(np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(
        jax.tree.leaves(restored.variables["params"]),
        jax.tree.leaves(restored_live.variables["params"]), strict=True))
    np.testing.assert_array_equal(
        np.asarray(restored.variables["encoders"]["textcontext"]["table"]),
        np.asarray(objective.inputs.conditions["textcontext"].encoder.params["table"]))


def test_pipeline_answers_an_lm_run_with_its_tokenizer_and_budget(tmp_path):
    """An LM run directory becomes a `TextGeneration` over the rebuilt model
    and restored weights, decoding through the run's tokenizer, budgeted by
    its sample setting, and drawing what the trained objective draws."""
    from dew.inference import TextGeneration
    from dew.sampling import Sampling

    objective, state = make_lm_run(tmp_path)
    task = dew.pipeline(str(tmp_path))
    assert isinstance(task, TextGeneration)
    assert task.max_new_tokens == 4 and task.sampling.eos_token_ids == (255,)
    result = task("the ", key=2, sampling=Sampling(temperature=0, eos_token_ids=255))
    assert result.rows == 1 and result.host().tokens.shape == (1, 4 + 4)
    assert isinstance(result.text[0], str)
    trained = objective.pipeline(state, processor=task.processor)
    assert trained.max_new_tokens == 4
    np.testing.assert_array_equal(
        trained("the ", key=2, sampling=Sampling(temperature=0, eos_token_ids=255)).host().tokens,
        result.host().tokens)
    with pytest.raises(TypeError):
        task("the ", seed=2)
    with pytest.raises(ValueError):
        dataclasses.replace(task, max_new_tokens=None)("the ", key=2)


@pytest.mark.parametrize("earlier", [False, True])
def test_an_lm_run_over_a_sources_own_processor_reloads_that_processor(tmp_path, earlier):
    """A decoder fine-tuned from a checkpoint encodes text with the processor the
    checkpoint ships; its run records that processor's source, so the reloaded task
    encodes as the trained one does rather than loading as weights that take ids."""
    from model_support import tokenized_qwen3

    from dew.interop import Pretrained
    from dew.objectives.lm import LMObjective

    source = tokenized_qwen3(tmp_path / "source")
    loaded = Pretrained.load(source, dtype="float32", attention_impl="xla", max_seq_len=32)
    objective = LMObjective(loaded, 8)
    batch = {"text": np.random.RandomState(0).randint(1, 100, (8, 9)).astype(np.int32)}
    run = tmp_path / "run"
    checkpoints = Checkpoints(str(run), keep=1)
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.PRNGKey(0), checkpoints=checkpoints)
    state = trainer.fit(Loader(train=lambda partition: Replaying(lambda position: batch), val=None, records=None,
                               batch=8), steps=1, log_every=100, checkpoint_every=1)
    checkpoints.wait()
    if earlier:
        record = {**objective.inference_record(), "processor": str(source)}
        checkpoints.save(int(state.step) + 1, state, artifact=record)
        checkpoints.wait()

    task = dew.pipeline(str(run))
    assert loaded.text_processor is not None and task.processor is not None
    assert type(task.processor) is type(loaded.text_processor)
    np.testing.assert_array_equal(np.asarray(task.processor("the cat sat").tokens),
                                  np.asarray(loaded.text_processor("the cat sat").tokens))


def test_a_run_record_holding_a_field_its_objective_does_not_write_is_refused(tmp_path):
    """An LM run records its decoder's task settings (`TextSettings`) and nothing else beside
    its objective and model, and they read back as the objective's own; a record holding
    another field, as a sampling temperature no LM run writes, is refused rather than loaded
    with the field ignored."""
    from dew.inference.tasks import TextSettings, saved_settings, settings_fields

    objective, state = make_lm_run(tmp_path)
    checkpoints = Checkpoints(str(tmp_path), keep=2)
    record = checkpoints.artifact()
    assert isinstance(record, dict)
    assert set(record) == {"objective", "model", *settings_fields(TextSettings)}
    assert saved_settings(record, TextSettings) == objective.task_settings()
    dew.pipeline(str(tmp_path))
    checkpoints.save(int(state.step) + 1, state, artifact={**record, "temperature": 0.5})
    checkpoints.wait()
    with pytest.raises(ValueError):
        dew.pipeline(str(tmp_path))



def test_pipeline_loads_a_run_published_to_the_hub_at_the_commit_it_resolved(tmp_path, monkeypatch):
    """A Hub repository holding a run, as `HfApi().upload_folder` publishes
    it, loads as that run: pulled whole at the commit its metadata resolved
    (the name of the snapshot directory), with checkpoints selected by step."""
    import huggingface_hub

    from dew.sampling import Sampling

    snapshot = tmp_path / "3f2a9c"
    make_lm_run(snapshot)
    resolved, pulled = [], []

    def downloaded(repo_id, revision, allow_patterns=None):
        (pulled if allow_patterns is None else resolved).append(revision)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", downloaded)
    task = dew.pipeline("user/byte-lm", revision="main", step=2)
    assert isinstance(task, TextGeneration)
    assert (resolved, pulled) == (["main"], ["3f2a9c"])
    greedy = Sampling(temperature=0, eos_token_ids=255)
    local = dew.pipeline(str(snapshot), step=2)
    np.testing.assert_array_equal(task("the ", key=2, sampling=greedy).host().tokens,
                                  local("the ", key=2, sampling=greedy).host().tokens)


def test_an_lm_run_without_an_average_publishes_and_exports_its_live_weights(tmp_path):
    """An LM keeps no EMA unless asked, so each reader's default takes the
    live weights of such a run: the objective's pipeline, `dew.pipeline`
    and `Pretrained.from_run`."""
    from dew.interop import Pretrained

    run = tmp_path / "run"
    run.mkdir()
    objective, state = make_lm_run(run, ema_decay=None)
    published = objective.pipeline(state)
    for expected, bound in zip(
        jax.tree.leaves(state.variables), jax.tree.leaves(published.variables), strict=True
    ):
        assert bound is expected
    Pretrained.from_run(str(run)).save(tmp_path / "export")
    reloaded = Pretrained.load(tmp_path / "export", dtype="float32", attention_impl="reference")
    ids = jnp.asarray([[3, 4, 5, 6]], jnp.int32)
    np.testing.assert_array_equal(np.asarray(reloaded.model.apply(reloaded.variables, ids)),
                                  np.asarray(published.model.apply(published.variables, ids)))


def test_a_quantized_runs_record_re_wraps_the_model_it_rebuilds(tmp_path):
    """The quantization knob is the trainer's, so `RunConfig.train` writes it
    under `trainer` in run.json and `dew.pipeline` reads it back there: the
    rebuilt model answers the quantized forward on the run's own weights,
    which the fp32 rebuild does not. Observed on CPU: 3e-02 on logits of
    order 1."""
    pytest.importorskip("qwix")
    from dew.config.lm import LMRunConfig
    from dew.objectives.lm import LMObjective
    from dew.quantization import Quantization

    batch, seq = 8, 8
    fields = {"vocab_size": 256, "emb_features": 16, "num_layers": 1, "num_heads": 2, "head_dim": 8,
                  "mlp_features": 32, "max_seq_len": 16}

    ids = np.random.RandomState(0).randint(1, 250, (batch, seq + 1)).astype(np.int32)
    config = LMRunConfig(
        model=ModelConfig("causal_transformer", {**fields,
                                               "dtype": "float32", "attention_impl": "reference"}),
        trainer=TrainerConfig(checkpoint_dir=str(tmp_path), steps=1,
                              eval_every=None, checkpoint_every=1, compilation_cache_dir=None,
                              quantization=Quantization()))
    state = config.train(LMObjective(config.model.build(), seq, ema_decay=0.9),
                         Loader(lambda partition: Replaying(lambda position: {"text": ids}), None, None,
                                batch), name="run")
    Checkpoints(str(tmp_path / "run")).wait()

    record = json.loads((tmp_path / "run" / "run.json").read_text())["fields"]
    assert record["trainer"]["quantization"]["dtype"] == "int8"
    task = dew.pipeline(str(tmp_path / "run"))

    ids = jnp.asarray([[3, 4, 5, 6]], jnp.int32)
    plain = config.model.build()
    quantized = task.model.apply(task.variables, ids)
    assert float(jnp.max(jnp.abs(quantized - plain.apply(task.variables, ids)))) > 0.0
    assert int(state.step) == 1


@pytest.mark.mesh
def test_pipeline_places_a_run_on_a_mesh_and_answers_the_same_images(tmp_path):
    """Placed under the trainer's layout, the weights shard over the mesh,
    prompts split over its batch axes, the result keeps that sharding, and
    `host()` reads back exactly what the single device draws."""
    from dew.nn.inputs import request_axes
    from dew.training import Layout, MeshSpec

    make_run(tmp_path)
    plain = TextToImage.from_run(str(tmp_path))
    placed = dew.pipeline(str(tmp_path), mesh=MeshSpec(fsdp=2), layout=Layout(min_shard=2 ** 6))
    specs = {leaf.sharding.spec for leaf in jax.tree.leaves(placed.variables)}
    assert any("fsdp" in str(spec) for spec in specs)
    result = placed(["a", "b", "c"], steps=3, key=7)
    placed_on = result.images.sharding
    assert placed_on.spec == jax.sharding.PartitionSpec(request_axes(placed_on.mesh))
    assert result.rows == 3
    np.testing.assert_allclose(result.host().images, plain(["a", "b", "c"], steps=3, key=7).host().images,
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
    plain = TextToImage.from_objective(objective, state.variables)
    same = dataclasses.replace(plain, grid=lambda steps: (plain.process, plain.process.times(steps)))
    key = jax.random.key(3)
    reference = plain(["a"], steps=4, solver=Heun(), key=key).host().images
    np.testing.assert_array_equal(same(["a"], steps=4, solver=Heun(), key=key).host().images, reference)
    warped = dataclasses.replace(plain, grid=lambda steps: (plain.process, plain.process.times(steps) ** 2))
    assert not np.allclose(warped(["a"], steps=4, solver=Heun(), key=key).host().images, reference)
    open_ended = dataclasses.replace(plain, final_denoise=False)
    assert not np.allclose(open_ended(["a"], steps=4, solver=Heun(), key=key).host().images, reference)
    longer = dataclasses.replace(plain, grid=lambda steps: (plain.process, plain.process.times(steps + 1)))
    np.testing.assert_array_equal(longer(["a"], steps=3, solver=Heun(), key=key).host().images,
                                  plain(["a"], steps=4, solver=Heun(), key=key).host().images)
    start = dataclasses.replace(plain, grid=lambda steps: (plain.process, plain.process.times(steps)[:1]),
                                final_denoise=False)
    prepared = start.prepare(["a"], key=key, steps=4)
    np.testing.assert_array_equal(start(prepared, steps=4, solver=Heun(), key=key).images,
                                  np.clip(np.asarray(prepared.noise), -1.0, 1.0))
    with pytest.raises(ValueError):
        start(prepared, steps=3, key=3)


def test_explicit_average_requests_do_not_substitute_live_weights(tmp_path):
    objective, state = make_lm_run(tmp_path, ema_decay=None)
    with pytest.raises(ValueError):
        objective.pipeline(state, ema=True)
    with pytest.raises(ValueError):
        dew.pipeline(str(tmp_path), ema=True)
    restored = dew.pipeline(str(tmp_path))
    live = objective.pipeline(state, ema=False, processor=restored.processor)
    np.testing.assert_array_equal(restored("the ", key=7).host().tokens,
                                  live("the ", key=7).host().tokens)


def make_block_run(directory, fixture="diffusion-gemma-workflow", moved=0.0, tokenizer="byte",
                   pad_token_id=0):
    """A block-diffusion run directory: a committed DiffusionGemma fixture's
    weights under a checkpoint that records the objective, its `pad_token_id`
    and the tokenizer `tokenizer` names, each parameter moved by `moved` times
    a seeded normal draw (a stand-in for training); returns the objective and
    the saved state. The default fixture reads images and is the published
    layout's own; `diffusion-gemma-sft` is Google's dense text-only model,
    which transformers' class cannot build, so its export is refused."""
    from pathlib import Path

    from dew.data.text import tokenizer_for
    from dew.interop import Pretrained
    from dew.objectives.diffusion.block import BlockDiffusionObjective

    fixture = Path(__file__).resolve().parent / "fixtures/hf" / fixture
    bundle = Pretrained.load(str(fixture), dtype="float32", attention_impl="xla", max_seq_len=32)
    objective = BlockDiffusionObjective(bundle.model, prompt_length=3, variables=bundle.variables,
                                        pad_token_id=pad_token_id,
                                        processor=RunProcessor(tokenizer_for(tokenizer)))
    state = Trainer(objective, optax.sgd(0.01), key=jax.random.PRNGKey(2)).initial_state()
    if moved:
        leaves, tree = jax.tree.flatten(state.variables["params"])
        keys = jax.random.split(jax.random.key(5), len(leaves))
        params = jax.tree.unflatten(tree, [leaf + moved * jax.random.normal(key, leaf.shape, leaf.dtype)
                                           for leaf, key in zip(leaves, keys, strict=True)])
        state = state.replace(variables={**state.variables, "params": params})
    checkpoints = Checkpoints(str(directory))
    checkpoints.save(0, state, None, artifact=objective.inference_record())
    checkpoints.wait()
    return objective, state


def make_masked_run(directory):
    """A masked-diffusion run directory: the committed llada-tiny weights under
    a checkpoint that records the objective and its byte tokenizer."""
    from pathlib import Path

    from dew.diffusion.discrete import MDLM
    from dew.interop import Pretrained
    from dew.objectives.diffusion.masked import MaskedDiffusionObjective

    fixture = Path(__file__).resolve().parent / "fixtures/hf/llada-tiny"
    source = Pretrained.load(fixture, dtype="float32", attention_impl="xla")
    objective = MaskedDiffusionObjective(source.model, MDLM(mask_id=120)(), 8,
                                         variables=source.variables, ema_decay=None,
                                         processor=RunProcessor(ByteTokenizer()))
    state = Trainer(objective, optax.sgd(0.05), key=jax.random.PRNGKey(19)).initial_state()
    checkpoints = Checkpoints(str(directory))
    checkpoints.save(0, state, None, artifact=objective.inference_record())
    checkpoints.wait()


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
    drawn = direct(prompt, budget, key=5)
    expected = front(prompt, budget, key=5)
    np.testing.assert_array_equal(drawn.host().tokens, expected.host().tokens)
    assert direct.decode(drawn) == front.decode(expected)
    assert all(isinstance(row, str) for row in direct.decode(drawn))


def test_a_saved_block_run_pads_rows_as_its_live_task_does(tmp_path):
    """The block objective's pad id is a control of its task that a loader
    cannot read off the model, so the run records it: a restored task pads a
    row that ended at an EOS with it, as the live task does."""
    objective, state = make_block_run(tmp_path, pad_token_id=3)
    live = objective.pipeline(state, ema=False)
    # The row's first drawn token as the EOS ends it at once, so the rest of
    # its response is padding.
    eos = (int(live([[1, 5, 7]], 3, key=5).host().tokens[0, 3]),)
    expected = dataclasses.replace(live, eos_token_ids=eos)([[1, 5, 7]], 3, key=5).host()
    restored = dataclasses.replace(dew.pipeline(str(tmp_path), ema=False), eos_token_ids=eos)
    actual = restored([[1, 5, 7]], 3, key=5).host()
    np.testing.assert_array_equal(expected.lengths, [1])
    np.testing.assert_array_equal(actual.tokens, expected.tokens)


def test_saved_sampling_policy_survives_a_disabled_preview_budget(tmp_path):
    from dew.sampling import Sampling

    (tmp_path / "run").mkdir()
    objective, state = make_lm_run(tmp_path / "run")
    policy = Sampling(temperature=0.37, top_k=3, eos_token_ids=255)
    checkpoints = Checkpoints(str(tmp_path / "unbudgeted"))
    checkpoints.save(int(state.step), state, None, artifact={
        **objective.inference_record(), "max_new_tokens": 0, "sampling": dataclasses.asdict(policy)})
    checkpoints.wait()
    task = dew.pipeline(str(tmp_path / "unbudgeted"))
    assert task.max_new_tokens is None
    with pytest.raises(ValueError):
        task([[1, 2]], key=4)
    expected = objective.policy(state.averaged, policy)([[1, 2]], 3, key=4).host()
    actual = task([[1, 2]], 3, key=4).host()
    np.testing.assert_array_equal(actual.tokens, expected.tokens)
    np.testing.assert_allclose(actual.behavior_log_probs, expected.behavior_log_probs, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("compute,storage", [("bfloat16", None), (None, "bfloat16")])
def test_saved_decoder_compute_and_storage_overrides_generate_from_the_same_weights(
    tmp_path, compute, storage
):
    import jax.numpy as jnp

    from dew.inference import TextGeneration
    from dew.objectives.base import FROZEN

    objective, state = make_lm_run(tmp_path)
    baseline = dew.pipeline(str(tmp_path))
    assert isinstance(baseline, TextGeneration)
    first = next(iter(state.variables["params"]))

    def move_to_frozen(variables):
        trainable = dict(variables["params"])
        frozen = {**variables.get(FROZEN, {}), first: trainable.pop(first)}
        return {**variables, "params": trainable, FROZEN: frozen}

    assert state.ema is not None
    step = int(state.step) + 1
    frozen = dataclasses.replace(state, step=jnp.asarray(step, state.step.dtype),
                                 variables=move_to_frozen(state.variables), ema=move_to_frozen(state.ema))
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(step, frozen, None, artifact=objective.inference_record())
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
    wanted = expected([[1, 2]], 2, key=7).host()
    result = actual([[1, 2]], 2, key=7).host()
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
    from dew.interop.weights import checkpoint_dtype

    stored = merge(state.variables, state.ema)
    # Stored as the load stores them at bfloat16: every floating leaf but those the storage rule
    # keeps in float32, the `constants` a schedule or embedding holds among them.
    expected_vars = jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf.astype(checkpoint_dtype(
            leaf.dtype, "bfloat16", path=tuple(jax.tree_util.keystr((key,), simple=True) for key in path))),
        stored)
    table = restored.variables["encoders"]["textcontext"]["table"]
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
    np.testing.assert_array_equal(restored(["ab"], steps=2, key=9).host().images,
                                  expected(["ab"], steps=2, key=9).host().images)





def test_a_restored_task_reads_the_training_steps_blank_to_the_bit(tmp_path):
    """Fit, save, restore, sample: the restored task's unconditional branch is the one the training
    step encoded (`DiffusionObjective.blank_conditions` over the step's own towers), to the bit, with a
    bf16 CLIP as the published hybrid DiT has, and the task samples as the objective's own does."""
    from pathlib import Path

    config = stating(dataclasses.replace(
        run_config(tmp_path, preset=Flow()),
        text=TextCondition(encoder="clip_text", checkpoint=str(Path(__file__).parent / "fixtures/clip/tiny"),
                           dtype="bfloat16")), guidance=CFG(5.0), ema_decay=None)
    objective = config.build()
    trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.key(0))
    state = trainer.initial_state()
    batch = {"image": np.full((jax.device_count(), RES, RES, 3), 180, np.uint8),
             **objective.inputs.tokenize(["a bird"] * jax.device_count())}
    state, *_ = trainer.compile(state, batch)(state, batch)
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(1, state, None, artifact=objective.inference_record())
    checkpoints.wait()
    original = objective.pipeline(state, ema=False)
    restored = TextToImage.from_run(str(tmp_path), ema=False)

    prepared = restored.prepare(["a bird", "a cat"], key=0, steps=8)
    stepped = jax.jit(objective.blank_conditions)(state.variables, prepared.conditions)
    for actual, expected in zip(jax.tree.leaves(prepared.unconditional), jax.tree.leaves(stepped),
                                strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(restored(["a bird", "a cat"], steps=8, key=0).host().images,
                                  original(["a bird", "a cat"], steps=8, key=0).host().images)


def test_binding_new_encoder_weights_encodes_both_branches_with_them(tmp_path):
    """A restored task binds a new encoder tree: its conditional and unconditional features and
    samples match a fresh task over those weights, not the old ones."""
    from pathlib import Path

    config = stating(dataclasses.replace(
        run_config(tmp_path, preset=Flow()),
        text=TextCondition(encoder="clip_text", checkpoint=str(Path(__file__).parent / "fixtures/clip/tiny"),
                           dtype="bfloat16")), guidance=CFG(5.0), ema_decay=None)
    objective = config.build()
    trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.key(0))
    state = trainer.initial_state()
    batch = {"image": np.full((jax.device_count(), RES, RES, 3), 180, np.uint8),
             **objective.inputs.tokenize(["a bird"] * jax.device_count())}
    state, *_ = trainer.compile(state, batch)(state, batch)
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(1, state, None, artifact=objective.inference_record())
    checkpoints.wait()
    restored = TextToImage.from_run(str(tmp_path), ema=False)
    prompts = ["a bird", "a cat"]
    old_blank = restored.prepare(prompts, key=0, steps=8).unconditional
    encoders = jax.tree.map(lambda value: value + 0.03125, restored.variables["encoders"])
    changed = {**restored.variables, "encoders": encoders}
    rebound = restored.bind(changed)
    fresh_objective = config.build(variables=changed)
    fresh = TextToImage.from_objective(fresh_objective, changed)
    bound_inputs = rebound.prepare(prompts, key=0, steps=8)
    fresh_inputs = fresh.prepare(prompts, key=0, steps=8)
    new_blank, expected_blank = bound_inputs.unconditional, fresh_inputs.unconditional
    assert any(not np.array_equal(np.asarray(left), np.asarray(right))
               for left, right in zip(jax.tree.leaves(old_blank), jax.tree.leaves(new_blank), strict=True))
    for actual, expected in zip(jax.tree.leaves((bound_inputs.conditions, new_blank)),
                                jax.tree.leaves((fresh_inputs.conditions, expected_blank)), strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(rebound(prompts, steps=8, key=0).host().images,
                                  fresh(prompts, steps=8, key=0).host().images)
