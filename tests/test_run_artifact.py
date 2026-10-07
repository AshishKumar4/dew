"""Python training saves the model/task declaration with its own checkpoint."""
import dataclasses
import inspect

import flax.linen
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.checkpoints import Checkpoints
from dew.config import ModelConfig
from dew.data import Dataset, Loading
from dew.diffusion import schedules, transforms
from dew.diffusion.process import Process
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.training import Trainer


def model():
    return CausalTransformer(vocab_size=16, emb_features=16, num_layers=1, num_heads=2,
                             mlp_features=32, max_seq_len=16, dtype='float32', attention_impl='xla')


def test_live_model_record_rebuilds_exact_constructor_fields():
    original = model()
    rebuilt = ModelConfig.from_model(original).build()
    assert rebuilt == original.clone(dtype=jnp.float32)
    tokens = jnp.arange(1, 9)[None, :]
    variables = original.init(jax.random.key(0), tokens)
    np.testing.assert_array_equal(original.apply(variables, tokens), rebuilt.apply(variables, tokens))


def test_a_composite_model_record_nests_its_part_and_rebuilds_it():
    """DiffusionGemma holds a CausalTransformer: its record nests the part's own
    record, and building it gives the part's precision to the part alone."""
    from dew.nn.diffusion_gemma import DiffusionGemma

    original = DiffusionGemma(model().clone(layer_scalar="frozen"), canvas_length=4)
    record = ModelConfig.from_model(original)
    assert record.dtype is None and record.config["text"]["layer_scalar"] == "frozen"
    assert record.build() == original.clone(text=original.text.clone(dtype=jnp.float32))


def documented_block(page, after):
    """The first Python block of `page` that follows the sentence `after`."""
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / page).read_text()
    start = text.index("```python\n", text.index(after)) + len("```python\n")
    return text[start:text.index("```", start)]


def test_the_documented_model_records_itself_and_its_run_loads_back(tmp_path, monkeypatch):
    """docs/key-concepts.md's example, run as written in a module of its own:
    nothing registers it, its import path names it, its constructor fields
    are the record, and a run of the model loads back by that record."""
    import importlib

    from dew import Field, InputSpec
    from dew.diffusion.presets import Flow
    from dew.objectives.diffusion import DiffusionObjective
    from dew.sampling import TextToImage

    (tmp_path / "mymodels.py").write_text(
        documented_block("docs/key-concepts.md", "Your own model needs nothing to be recorded"))
    monkeypatch.syspath_prepend(str(tmp_path))
    mymodels = importlib.import_module("mymodels")
    model = mymodels.ResidualMLP(features=16)
    objective = DiffusionObjective(model, Flow(), InputSpec(Field("image", (4, 4, 3))),
                                   guidance=None, steps=2)
    rows = [{"image": np.full((4, 4, 3), 128, np.uint8)} for _ in range(8)]
    data = Dataset.from_grain(grain.MapDataset.source(rows), batch=8, loading=Loading(workers=0))
    checkpoints = Checkpoints(str(tmp_path / "run"))
    state = Trainer(objective, optax.sgd(.01), key=0, checkpoints=checkpoints).fit(
        data, steps=1, checkpoint_every=1)
    checkpoints.wait()

    recorded = checkpoints.artifact()["model"]
    assert (recorded["architecture"], recorded["config"]) == ("mymodels:ResidualMLP", {"features": 16})
    restored = TextToImage.from_run(str(tmp_path / "run"))
    assert type(restored.model) is mymodels.ResidualMLP and restored.model.features == 16
    np.testing.assert_array_equal(restored([""], key=3).host().images,
                                  objective.pipeline(state)([""], key=3).host().images)


def test_a_model_with_no_import_path_trains_and_saves_and_loading_says_why(tmp_path, caplog):
    """A class defined inside a function trains and checkpoints as any other;
    its record cannot name it, which the first checkpoint warns and loading
    the run raises."""
    from dew.inference import TextGeneration

    class Custom(type(model())):
        pass

    objective = LMObjective(Custom(**{field.name: getattr(model(), field.name)
                                      for field in dataclasses.fields(model())
                                      if field.init and field.name not in ("parent", "name")}),
                            seq_len=8, ema_decay=None)
    rows = [{'text': np.arange(9, dtype=np.int32)} for _ in range(8)]
    data = Dataset.from_grain(grain.MapDataset.source(rows), batch=8, loading=Loading(workers=0))
    checkpoints = Checkpoints(str(tmp_path / 'run'))
    with caplog.at_level("WARNING", logger="dew.training.trainer"):
        Trainer(objective, optax.sgd(.01), key=0, checkpoints=checkpoints).fit(data, steps=1,
                                                                               checkpoint_every=1)
    checkpoints.wait()
    assert "has no import path a record can name" in caplog.text
    with pytest.raises(ValueError, match="has no import path a record can name"):
        TextGeneration.from_run(str(tmp_path / 'run'))


class Opaque:
    pass


class Noted(flax.linen.Module):
    """A denoiser holding a field no record can carry."""

    note: object = Opaque()

    @flax.linen.compact
    def __call__(self, x, temb, textcontext=None, train=False):
        return flax.linen.Dense(x.shape[-1])(x)


def test_a_model_no_record_can_describe_still_checkpoints_and_loading_says_why(tmp_path, caplog):
    from dew import Field, InputSpec
    from dew.diffusion.presets import Flow
    from dew.objectives.diffusion import DiffusionObjective
    from dew.sampling import TextToImage

    objective = DiffusionObjective(Noted(), Flow(), InputSpec(Field("image", (4, 4, 3))), guidance=None,
                                   steps=2)
    rows = [{"image": np.zeros((4, 4, 3), np.uint8)} for _ in range(8)]
    data = Dataset.from_grain(grain.MapDataset.source(rows), batch=8, loading=Loading(workers=0))
    checkpoints = Checkpoints(str(tmp_path / 'run'))
    with caplog.at_level("WARNING", logger="dew.training.trainer"):
        Trainer(objective, optax.sgd(.01), key=0, checkpoints=checkpoints).fit(data, steps=1,
                                                                               checkpoint_every=1)
    checkpoints.wait()
    assert caplog.text.count("no loader can rebuild their model") == 1
    assert checkpoints.variables(ema=False)["params"]
    with pytest.raises(ValueError, match="no model to load: DiffusionObjective: Opaque is not something"):
        TextToImage.from_run(str(tmp_path / 'run'))


def test_an_adapted_run_records_its_base_and_adapter_and_loads_what_it_trained(tmp_path):
    """The record of a LoRA run is the base model's record with the adapter's
    rank, alpha, native modules and the name each binds under; the loaded
    task computes what the trained adapted model computes, with factors the
    training moved off zero."""
    from dew.inference import TextGeneration
    from dew.lora import LoRA

    base = model()
    adapter = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(
        base, base.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32)), key=1)
    objective = LMObjective(adapter.model, seq_len=8, ema_decay=None, variables=adapter.variables)
    rows = [{'text': np.arange(9, dtype=np.int32)} for _ in range(8)]
    data = Dataset.from_grain(grain.MapDataset.source(rows), batch=8, loading=Loading(workers=0))
    checkpoints = Checkpoints(str(tmp_path / 'run'))
    state = Trainer(objective, optax.sgd(1.0), key=0, checkpoints=checkpoints).fit(
        data, steps=2, checkpoint_every=1)
    checkpoints.wait()
    record = checkpoints.artifact()['model']
    assert record['architecture'] == 'dew.nn.backbones.causal_transformer:CausalTransformer'
    assert record['adapter'] == {'rank': 2, 'alpha': 4.0, 'rslora': False, 'dropout': 0.0, 'modules': [
        'params/layers_0/self_attn/q_proj', 'params/layers_0/self_attn/v_proj'], 'layouts': {
        f'params/layers_0/self_attn/{name}': {'name': f'layers_0.self_attn.{name}.weight', 'shape': [16, 16],
                                               'transpose': [1, 0]} for name in ('q_proj', 'v_proj')}}
    assert any(np.abs(np.asarray(leaf)).max() > 0 for path, leaf in jax.tree_util.tree_leaves_with_path(
        state.variables['params']) if 'lora_B' in jax.tree_util.keystr(path))
    task = TextGeneration.from_run(str(tmp_path / 'run'), ema=False)
    tokens = jnp.arange(1, 9)[None, :]
    np.testing.assert_array_equal(np.asarray(task.model.apply(task.variables, tokens)),
                                  np.asarray(objective.model.apply(state.variables, tokens)))


@pytest.mark.parametrize("kind", ["dpo", "ppo"])
def test_an_adapted_policy_run_loads_and_saves_the_policy_it_trained(kind, tmp_path):
    """A LoRA run of an objective whose average is its frozen reference
    (DPO), or that nests the policy beside a critic (PPO), loads the policy
    it trained: the adapters `Adapter.from_run` and `Pretrained.from_run`
    rebuild write the files the trained adapter writes, byte for byte, and
    the loaded task and bundle hold the base draw bitwise under `frozen`
    and compute what the trained policy computes."""
    from dew.inference import TextGeneration
    from dew.interop import Pretrained
    from dew.lora import Adapter, LoRA
    from dew.objectives.base import FROZEN, part
    from dew.objectives.rl import DPOObjective, PPOObjective, ValueHead

    base = model()
    adapter = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(
        base, base.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32)), key=1)
    count = max(2, jax.device_count())
    if kind == "dpo":
        objective = DPOObjective(adapter.model, seq_len=2, variables=adapter.variables)
        pairs = np.tile(np.asarray([[[1, 2, 3], [1, 2, 4]]], np.int32), (count, 1, 1))
        batch = {"input_ids": pairs, "completion_mask": np.ones_like(pairs, np.float32)}
    else:
        objective = PPOObjective(adapter.model, seq_len=2, critic=ValueHead(base.clone()), beta=0.1,
                                 variables=adapter.variables)
    trainer = Trainer(objective, optax.sgd(1.0), key=0)
    if kind == "ppo":
        # One packed chain per row, its last id sampled.
        ids = np.tile(np.asarray([[1, 2, 3]], np.int32), (count, 1))
        mask = np.tile(np.asarray([[0, 0, 1]], np.float32), (count, 1))
        batch = {"input_ids": ids, "text_segment_ids": np.ones_like(ids),
                 "text_positions": np.tile(np.arange(3, dtype=np.int32), (count, 1)),
                 "response_mask": mask, "advantages": mask}
        before = trainer.initial_state()
        batch["old_log_probs"] = np.asarray(objective.actor.packed_log_probs(
            part(before.variables, "policy"), batch))
        batch["behavior_log_probs"] = batch["old_log_probs"]
        values = np.asarray(objective.values(before.variables, batch))
        batch.update(old_values=values, returns=values + mask)
    data = Dataset(train=lambda partition: iter([batch, batch]), val=None, records=2 * count, batch=count)
    state = trainer.fit(data, steps=2, log_every=100, checkpoint_every=None)
    checkpoints = Checkpoints(str(tmp_path / "run"))
    checkpoints.save(int(state.step), state, None, artifact=objective.inference_record())
    checkpoints.wait()
    trained = part(state.variables, "policy") if kind == "ppo" else state.variables
    assert any(np.abs(np.asarray(leaf)).max() > 0 for path, leaf in jax.tree_util.tree_leaves_with_path(
        trained["params"]) if "lora_B" in jax.tree_util.keystr(path))
    adapter.save(trained, tmp_path / "in-process")
    rebuilt = Adapter.from_run(tmp_path / "run")
    rebuilt.save(rebuilt.variables, tmp_path / "from-run")
    bundle = Pretrained.from_run(tmp_path / "run")
    assert bundle.adapter is not None
    bundle.adapter.save(bundle.variables, tmp_path / "bundle")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        written = (tmp_path / "in-process" / name).read_bytes()
        assert (tmp_path / "from-run" / name).read_bytes() == written
        assert (tmp_path / "bundle" / name).read_bytes() == written
    tokens = jnp.arange(1, 9)[None, :]
    expected = np.asarray(adapter.model.apply(trained, tokens))
    task = TextGeneration.from_run(str(tmp_path / "run"))
    for loaded in (rebuilt, bundle):
        for got, want in zip(jax.tree.leaves(loaded.variables[FROZEN]),
                             jax.tree.leaves(adapter.variables[FROZEN]), strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    # The task folds `frozen` back into `params` as it binds.
    for loaded in (rebuilt, bundle, task):
        np.testing.assert_array_equal(np.asarray(loaded.model.apply(loaded.variables, tokens)), expected)


class TinyDenoiser(flax.linen.Module):
    features: int = 8

    @flax.linen.compact
    def __call__(self, x, temb, textcontext=None, train=False):
        hidden = flax.linen.Dense(self.features, name="hidden")(x)
        return x + flax.linen.Dense(x.shape[-1], name="out")(flax.linen.gelu(hidden))


def test_an_adapted_denoiser_run_loads_what_it_trained(tmp_path):
    from dew import Field, InputSpec
    from dew.diffusion.presets import Flow
    from dew.lora import LoRA
    from dew.objectives.diffusion import DiffusionObjective
    from dew.sampling import TextToImage

    base = TinyDenoiser()
    sample = jnp.zeros((1, 4, 4, 3))
    adapter = LoRA(rank=2, modules=("hidden", "out")).apply(
        base, base.init(jax.random.key(0), sample, jnp.zeros((1,))), key=1)
    objective = DiffusionObjective(adapter.model, Flow(), InputSpec(Field("image", (4, 4, 3))),
                                   guidance=None, steps=2, ema_decay=None,
                                   variables={**adapter.variables, "encoders": {}})
    rows = [{"image": np.full((4, 4, 3), 200, np.uint8)} for _ in range(8)]
    data = Dataset.from_grain(grain.MapDataset.source(rows), batch=8, loading=Loading(workers=0))
    checkpoints = Checkpoints(str(tmp_path / 'run'))
    state = Trainer(objective, optax.sgd(1.0), key=0, checkpoints=checkpoints).fit(
        data, steps=2, checkpoint_every=1)
    checkpoints.wait()
    assert checkpoints.artifact()['model']['adapter']['modules'] == ['params/hidden', 'params/out']
    pipe = TextToImage.from_run(str(tmp_path / 'run'), ema=False)
    x, t = jax.random.normal(jax.random.key(2), (2, 4, 4, 3)), jnp.zeros((2,))
    np.testing.assert_array_equal(np.asarray(pipe.model.apply(pipe.variables, x, t)),
                                  np.asarray(objective.model.apply(state.variables, x, t)))


def test_python_lm_run_saves_its_inference_record_without_run_json(tmp_path):
    objective = LMObjective(model(), seq_len=8, ema_decay=None)
    rows = [{'text': np.arange(9, dtype=np.int32)} for _ in range(16)]
    data = Dataset.from_grain(grain.MapDataset.source(rows), batch=8, loading=Loading(workers=0))
    checkpoints = Checkpoints(str(tmp_path / 'run'))
    trainer = Trainer(objective, optax.sgd(.01), key=0, checkpoints=checkpoints)
    state = trainer.fit(data, steps=2, log_every=2, checkpoint_every=1)
    checkpoints.wait()
    assert not (tmp_path / 'run' / 'run.json').exists()
    record = Checkpoints(str(tmp_path / 'run')).artifact(2)
    assert record['objective'] == 'dew.objectives.lm.objective:LMObjective'
    assert record['seq_len'] == 8
    from dew.interop import Pretrained, PretrainedDecoder, PretrainedMaskedDecoder
    bundle = Pretrained.from_run(tmp_path / 'run')
    assert isinstance(bundle, PretrainedDecoder)
    with pytest.raises(TypeError, match='PretrainedDecoder source, not a PretrainedMaskedDecoder'):
        PretrainedMaskedDecoder.from_run(tmp_path / 'run')
    rebuilt = ModelConfig.from_dict(record['model']).build()
    assert rebuilt == objective.model.clone(dtype=jnp.float32)
    assert record['tokenizer'] is None
    from dew.inference import TextGeneration
    task = TextGeneration.from_run(str(tmp_path / 'run'), ema=False)
    assert task.processor is None
    tokens = jnp.arange(1, 9)[None, :]
    expected = objective.model.apply(state.variables, tokens)
    actual = task.model.apply(task.variables, tokens)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("kind", ["text", "masked", "block"])
def test_task_from_pretrained_selects_the_requested_snapshot(tmp_path, monkeypatch, kind):
    from flax.core import freeze

    import dew.interop.hub as hub
    from dew.diffusion.discrete import MDLM
    from dew.inference import BlockGeneration, MaskedGeneration, TextGeneration
    from dew.nn.diffusion_gemma import DiffusionGemma
    from dew.objectives.diffusion import BlockDiffusionObjective, MaskedDiffusionObjective
    from dew.training import TrainState

    if kind == "text":
        objective, task_type = LMObjective(model(), 8, ema_decay=None), TextGeneration
    elif kind == "masked":
        masked = model().clone(causal=False, mask_token_id=0, qk_norm=False)
        objective, task_type = MaskedDiffusionObjective(masked, MDLM(mask_id=0)(), 8,
                                                       ema_decay=None), MaskedGeneration
    else:
        block = DiffusionGemma(model().clone(layer_scalar="frozen"), canvas_length=4)
        objective = BlockDiffusionObjective(block, prompt_length=4)
        task_type = BlockGeneration
    original = objective.init(jax.random.key(0))
    selected = jax.tree.map(lambda leaf: leaf + 2 if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf,
                           original)
    zero = jnp.asarray(0, jnp.int32)
    for name, variables in (("main", original), ("pinned", selected)):
        state = TrainState(step=zero, microstep=zero, updates=zero, variables=variables,
                           opt_state=(), ema=None, key=jax.random.key(0), scale=None,
                           window_size=jnp.asarray(1, jnp.int32))
        checkpoints = Checkpoints(str(tmp_path / name))
        checkpoints.save(0, state, None, artifact=objective.inference_record())
        checkpoints.wait()
    monkeypatch.setattr(hub, "snapshot_download", lambda repo_id, revision=None:
                        tmp_path / ("main" if revision is None else {"pinned": "pinned"}[revision]))
    task = task_type.from_pretrained("user/published-model", revision="pinned", ema=False)
    assert jax.tree.structure(task.variables) == jax.tree.structure(freeze(selected))
    for actual, expected in zip(jax.tree.leaves(task.variables), jax.tree.leaves(selected), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_builtin_process_records_preserve_noise_prediction_and_weights():
    from dew.diffusion.presets import EDM, Cosine, Flow
    for preset in (EDM(regime='pixel'), Flow(), Cosine()):
        original = preset()
        rebuilt = Process.from_json(original.to_json())
        time = jnp.linspace(.01, .99, 16)
        np.testing.assert_array_equal(original.schedule.rates(time)[0], rebuilt.schedule.rates(time)[0])
        np.testing.assert_array_equal(original.schedule.rates(time)[1], rebuilt.schedule.rates(time)[1])
        np.testing.assert_array_equal(original.weight(time), rebuilt.weight(time))
        rates = original.schedule.rates(time)
        clean, noise = jnp.ones((16, 1)), jnp.full((16, 1), .2)
        np.testing.assert_array_equal(original.prediction.get_target(clean, noise, rates),
                                      rebuilt.prediction.get_target(clean, noise, rates))
        np.testing.assert_array_equal(original.prediction.get_input_scale(rates),
                                      rebuilt.prediction.get_input_scale(rates))


def test_builtin_autoencoder_record_uses_the_saved_parameters():
    from dew.nn.autoencoders import AutoEncoder, AutoencoderKL, StableDiffusionVAE
    image = jnp.ones((1, 8, 8, 3))
    module = AutoencoderKL(channels=(4,), latent_channels=2, blocks_per_level=1, norm_groups=1,
                           dtype=jnp.float32)
    variables = module.init(jax.random.key(0), image)
    original = StableDiffusionVAE(model=module, params=variables['params'], dtype=jnp.float32)
    rebuilt = AutoEncoder.from_json(original.to_json(), params=original.params)
    np.testing.assert_array_equal(original.encode(original.params, image),
                                  rebuilt.encode(rebuilt.params, image))
    latent = original.encode(original.params, image)
    np.testing.assert_array_equal(original.decode(original.params, latent),
                                  rebuilt.decode(rebuilt.params, latent))


def test_masked_run_returns_its_own_bundle_kind(tmp_path):
    from dew.diffusion.discrete import MDLM
    from dew.interop import Pretrained, PretrainedMaskedDecoder
    from dew.objectives.diffusion.masked import MaskedDiffusionObjective

    masked = model().clone(causal=False, mask_token_id=0, qk_norm=False)
    objective = MaskedDiffusionObjective(masked, MDLM(mask_id=0)(), 8,
                                        head_chunks=1, ema_decay=None, steps=2)
    source = grain.MapDataset.source([{'text': np.arange(1, 9, dtype=np.int32)}] * 8)
    data = Dataset.from_grain(source, batch=8, loading=Loading(workers=0))
    trainer = Trainer(objective, optax.sgd(.01), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(tmp_path / 'run')))
    trainer.fit(data, steps=1, log_every=1, checkpoint_every=1)
    bundle = Pretrained.from_run(tmp_path / 'run')
    assert isinstance(bundle, PretrainedMaskedDecoder)
    assert bundle.model.mask_token_id == 0


def process_components():

    classes = [getattr(schedules, name) for name in schedules.__all__]
    classes += [value for name, value in vars(transforms).items()
                if not name.startswith('_') and inspect.isclass(value)
                and value.__module__ == transforms.__name__]
    return [cls for cls in classes if inspect.isclass(cls) and not inspect.isabstract(cls)
            and cls.__name__ != 'Weighting']


@pytest.mark.parametrize('component', process_components(), ids=lambda cls: cls.__name__)
def test_every_builtin_process_component_round_trips_nondefaults(component):

    arguments = {}
    for name, parameter in inspect.signature(component).parameters.items():
        default = parameter.default
        if name == 'betas':
            value = np.linspace(.003, .03, 37, dtype=np.float32)
        elif name == 'inner':
            value = transforms.DirectPredictionTransform(normalize_input=True)
        elif name == 'timesteps':
            value = 37
        elif name == 'gamma':
            value = 3.7
        elif name == 'threshold':
            value = (.7, 1.8)
        elif name == 'density':
            value = 'mode'
        elif default is None:
            value = 1.3
        elif isinstance(default, bool):
            value = not default
        elif isinstance(default, (int, float)):
            value = default * .8 if default else .2
        else:
            raise AssertionError(f'uncovered constructor parameter: {component.__name__}.{name}')
        arguments[name] = value
    value = component(**arguments)
    schedule = schedules.FlowMatchingScheduler(shift=1.7)
    prediction = transforms.DirectPredictionTransform()
    if isinstance(value, schedules.NoiseScheduler):
        schedule = value
    elif isinstance(value, transforms.PredictionTransform):
        prediction = value
    weighting = value if isinstance(value, (transforms.ScheduleWeighting, transforms.MinSNR,
                                            transforms.VelocityLoss)) else transforms.ScheduleWeighting()
    original = Process(schedule=schedule, prediction=prediction, weighting=weighting)
    rebuilt = Process.from_json(original.to_json())
    time = jnp.linspace(.01, .99, 16)
    for method in ('rates', 'weight', 'model_time'):
        left, right = getattr(original.schedule, method)(time), getattr(rebuilt.schedule, method)(time)
        jax.tree.map(np.testing.assert_array_equal, left, right)
    key = jax.random.key(3)
    np.testing.assert_array_equal(original.schedule.sample_t(key, 16), rebuilt.schedule.sample_t(key, 16))
    rates = original.schedule.rates(time)
    clean, noise = jnp.ones(16), jnp.full(16, .25)
    jax.tree.map(np.testing.assert_array_equal,
                 original.prediction.forward_diffusion(clean, noise, rates),
                 rebuilt.prediction.forward_diffusion(clean, noise, rates))
    np.testing.assert_array_equal(original.prediction.pred_transform(clean, noise, rates, time),
                                  rebuilt.prediction.pred_transform(clean, noise, rates, time))
    np.testing.assert_array_equal(original.weight(time), rebuilt.weight(time))
    if type(original.prediction) is not transforms.PredictionTransform:
        jax.tree.map(np.testing.assert_array_equal,
                     original.prediction.backward_diffusion(clean, noise, rates),
                     rebuilt.prediction.backward_diffusion(clean, noise, rates))


def test_python_diffusion_checkpoint_preserves_its_solver(tmp_path):
    from dew.diffusion.presets import Flow
    from dew.inference import TextToImage
    from dew.inputs import Field, InputSpec
    from dew.nn.backbones.dit import SimpleDiT
    from dew.objectives.diffusion import DiffusionObjective
    from dew.sampling.solvers import Euler

    model = SimpleDiT(patch_size=2, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1,
                      output_channels=1, dtype=jnp.float32)
    objective = DiffusionObjective(model, Flow(), InputSpec(Field('image', (4, 4, 1))),
                                   solver=Euler(), steps=2, ema_decay=None)
    source = grain.MapDataset.source([{'image': np.ones((4, 4, 1), np.float32)}] * 8)
    data = Dataset.from_grain(source, batch=8, loading=Loading(workers=0))
    trainer = Trainer(objective, optax.sgd(.01), key=0, checkpoints=Checkpoints(str(tmp_path / 'run')))
    trainer.fit(data, steps=1, log_every=1, checkpoint_every=1)
    trainer.checkpoints.wait()
    loaded = TextToImage.from_run(str(tmp_path / 'run'))
    assert isinstance(loaded.solver, Euler)
    assert loaded.steps == 2
