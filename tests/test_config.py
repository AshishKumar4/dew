"""RunConfig: the typed record of a run, its round trip, and what it builds."""

import dataclasses
import json
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import pytest
from affine_run import TimingRegression as Regression, batches
from diffusion_support import images_of
from flax import linen as nn

import dew.config
import dew.nn.backbones
from dew.config import RunConfig, TrainerConfig
from dew.config.diffusion import DiffusionRunConfig
from dew.data.dataset import Loader, Ramp
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.vision import GemmaProjector, SiglipVision
from dew.optim import OptimConfig
from dew.registry import Aliases, ModelConfig, projectors, towers
from dew.training import Checkpoints, Layout, MeshSpec, Wandb


@pytest.mark.parametrize("dataset", ["tfds", "arrayrecord_shards"])
def test_to_dict_and_from_dict_round_trip_a_run(dataset, tmp_path):
    """A run's record holds its data as its chain's record (`dew.data.records`), the preset it calls
    with its settings, which reads back the chain it was."""
    from dew.data import Files, presets
    from dew.data.sources import source

    data = {
        "tfds": source("tfds://oxford_flowers102").pipe(presets.images, 64, batch=32),
        "arrayrecord_shards": Files((str(tmp_path / "cc12m" / "*.array_record"),),
                                    format="array_record").pipe(presets.images, 64, batch=32),
    }[dataset]
    config = RunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 4, "emb_features": 64}),
        data=data,
        optim=OptimConfig(optimizer="lamb", learning_rate=1e-3, weight_decay=0.1, b1=0.8, b2=0.99),
        trainer=TrainerConfig(name="run", steps=10, mesh=MeshSpec(fsdp=2),
                              layout=Layout(rules={"mlp": "fsdp"}, min_shard=8)),
    )
    record = config.to_dict()
    assert record["trainer"]["layout"]["rules"] == [["mlp", "fsdp"]]
    assert RunConfig.from_dict(json.loads(json.dumps(record))).to_dict() == record
    assert record["data"]["stages"][-1]["pipe"] == "dew.data.presets:images"


def test_a_chain_the_record_cannot_describe_trains_from_python_and_refuses_to_rebuild():
    """A chain with a step only Python holds (a lambda) is recorded as unrecorded, with the reason,
    and rebuilding the run from that record is refused by name."""
    from dew.data import Memory

    data = Memory([{"x": 1.0, "y": 2.0}] * 8).map(lambda record: record).batch(8)
    record = RunConfig(data=data, trainer=TrainerConfig(steps=1)).to_dict()
    assert set(record["data"]) == {"unrecorded"}
    with pytest.raises(ValueError):
        RunConfig.from_dict(json.loads(json.dumps(record)))


def test_named_validation_splits_read_back_in_their_order(tmp_path):
    """A run's named validation splits are recorded in their order, which a saved run.json keeps: the
    first is the split a run ranks its checkpoints and calibrates on, so a save must not move another
    ahead of it."""
    from dew.data import presets
    from dew.data.sources.tokens import TokenFiles

    def windows(url):
        return TokenFiles(url).pipe(presets.token_windows, seq_len=8, batch=2)

    config = RunConfig(data=windows("tokens/train"), trainer=TrainerConfig(steps=1),
                       validation={"val": windows("tokens/val"), "test": windows("tokens/test")})
    loaded = RunConfig.load(config.save(str(tmp_path / "run")))
    assert isinstance(loaded.validation, Mapping) and list(loaded.validation) == ["val", "test"]
    assert [split.source.url for split in loaded.validation.values()] == ["tokens/val", "tokens/test"]


def test_a_record_with_an_unknown_field_or_without_a_required_one_is_refused():
    record = RunConfig().to_dict()
    with pytest.raises(ValueError):
        RunConfig.from_dict({**record, "trainer": {**record["trainer"], "epochs_per_eval": 1}})
    with pytest.raises(ValueError):
        RunConfig.from_dict({**record, "trainer": {**record["trainer"], "wandb": {"entity": "dew"}}})
    with pytest.raises(ValueError):
        RunConfig.from_dict({**record, "data": {"class": "flowers", "fields": {}}})


def test_a_token_budget_refuses_a_ramp_or_a_budget_under_one_token():
    with pytest.raises(ValueError):
        TrainerConfig(update_tokens=0)
    with pytest.raises(ValueError):
        TrainerConfig(update_tokens=96, batch_ramp=Ramp(start=8, increment=8, samples=8))


def test_a_token_budget_sets_the_accumulation_over_its_datas_token_batches():
    """The data's batch stage is the run's one batch: a budget of tokens an update sets the
    accumulation to that many of the data's token batches (`TrainerConfig.accumulating`), and refuses
    data batched in rows, a budget no whole number of its batches, or an accumulation set otherwise."""
    trainer = TrainerConfig(update_tokens=98304, steps=1)
    tokens = Loader(train=lambda partition: iter(()), val=None, records=None, batch=8, batch_tokens=24576)
    assert trainer.accumulating(tokens).accumulation == 4
    recorded = json.loads(json.dumps(RunConfig(trainer=trainer).to_dict()))
    assert RunConfig.from_dict(recorded).trainer == trainer
    for refused in (dataclasses.replace(tokens, batch_tokens=None),
                    dataclasses.replace(tokens, batch_tokens=24575)):
        with pytest.raises(ValueError):
            trainer.accumulating(refused)
    with pytest.raises(ValueError):
        dataclasses.replace(trainer, accumulation=3).accumulating(tokens)


@dataclasses.dataclass(frozen=True)
class KnobbedRun(RunConfig):
    """A kind of run with knobs of its own."""

    seq_len: int = 256
    pad_id: int | None = None


def test_save_and_load_carry_a_subclass_with_its_own_knobs(tmp_path):
    """The record names the run's class, so a reader that knows only the
    base builds the kind of run it records, and a reader of another kind
    refuses it."""
    config = KnobbedRun(trainer=TrainerConfig(steps=3), seq_len=64, pad_id=0)
    assert config.save(str(tmp_path)) == str(tmp_path / "run.json")
    assert KnobbedRun.load(str(tmp_path)) == config and RunConfig.load(str(tmp_path)) == config
    with pytest.raises(ValueError):
        DiffusionRunConfig.load(str(tmp_path))


def test_the_model_config_builds_with_its_own_dtype_and_attention():
    fields = {"vocab_size": 64, "emb_features": 32, "num_layers": 1, "num_heads": 2}
    config = ModelConfig("causal_transformer", {**fields, "dtype": "bfloat16", "attention_impl": "xla"})
    model = config.build()
    import jax
    ids = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
    params = model.init(jax.random.key(0), ids)
    logits = model.apply(params, ids)
    expected = CausalTransformer(**fields, dtype=jnp.bfloat16, attention_impl="xla").apply(
        params, ids)
    assert logits.dtype == jnp.float32
    assert jnp.array_equal(logits, expected)


@pytest.mark.parametrize("scaling", ["llama3", "longrope"])
def test_dtype_changes_preserve_union_held_rotary_policies(scaling):
    """Model metadata may hold either rotary policy. It survives a compute-dtype override while
    the model can still run on the same parameter tree."""
    import jax

    from dew.nn.rope import LongRopeScaling, RopeScaling

    policy = (RopeScaling(8.0, 1.0, 4.0, 16) if scaling == "llama3" else
              LongRopeScaling((1.0,) * 4, (2.0,) * 4, 16, factor=2.0))
    model = CausalTransformer(vocab_size=16, emb_features=16, num_layers=1, num_heads=2,
                              head_dim=8, mlp_features=32, max_seq_len=32, rope_scaling=policy,
                              dtype="float32", attention_impl="reference")
    configured = ModelConfig.from_model(model).with_dtype("bfloat16").build()
    ids = jnp.asarray([[1, 2, 3]], jnp.int32)
    variables = model.init(jax.random.key(0), ids)
    expected = model.clone(dtype=jnp.bfloat16).apply(variables, ids)
    actual = configured.apply(variables, ids)
    assert jnp.array_equal(actual, expected)


class Unprecise(nn.Module):
    """A model that declares none of the precision fields, as a plugin's may."""

    features: int = 4

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features)(x)


def test_a_model_without_dtype_builds_and_refuses_a_dtype_override():
    unprecise = f"{__name__}:Unprecise"
    config = ModelConfig(unprecise, {"features": 2})
    with pytest.raises(ValueError):
        config.with_dtype("float32")
    assert isinstance(config.build(), Unprecise)


def test_an_unknown_model_field_is_refused():
    with pytest.raises(ValueError):
        ModelConfig("simple_dit", {"depth": 3}).build()


def test_the_run_length_is_steps_or_epochs():
    with pytest.raises(ValueError):
        TrainerConfig(steps=10, epochs=1)

    streamed = Loader(train=lambda partition: iter(()), val=None, records=None, batch=4)
    sized = Loader(train=lambda partition: iter(()), val=None, records=100, batch=4)

    assert TrainerConfig(steps=10).total_steps(sized) == 10
    assert TrainerConfig(epochs=4).total_steps(sized) == 100
    with pytest.raises(ValueError):
        TrainerConfig(epochs=4).total_steps(streamed)
    with pytest.raises(ValueError):
        TrainerConfig().total_steps(sized)


def test_the_cli_parses_the_mesh_the_layout_and_a_setting_of_the_chain():
    config = RunConfig.cli([
        "--trainer.mesh.fsdp", "2", "--trainer.layout.min-shard", "8",
        "--trainer.steps", "5", "--model", "uvit", "--model.emb_features", "32",
        "--set", "data.images.size=64"], default=RunConfig(data=images_of("tfds://oxford_flowers102", 32)))
    assert config.trainer.mesh == MeshSpec(fsdp=2)
    assert config.trainer.layout.min_shard == 8
    assert config.model.fields == {"emb_features": 32}
    assert config.data is not None and config.data.facts()["fields"]["image"] == [64, 64, 3]
    assert RunConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()



# --------------------------------------------------------------------------
# A record builds a value
# --------------------------------------------------------------------------
if TYPE_CHECKING:
    class Unavailable:
        pass


@dataclasses.dataclass(frozen=True)
class PartialSpec:
    kernel: "Kind"
    extra: "Unavailable | None" = None
    Kind: int = 0

    def frequency(self):
        return self.kernel.rope_theta / self.kernel.window


def test_an_unresolved_dependency_type_does_not_hide_a_buildable_field():
    registry = Aliases("partial", {"partial": f"{__name__}:PartialSpec"})
    built = registry.build("partial", kernel={"window": 4, "rope_theta": 20.0})
    assert built.frequency() == 5.0


@dataclasses.dataclass(frozen=True)
class MixedSpec:
    kernel: "Kind | dict[str, object]"

    def frequency(self):
        if isinstance(self.kernel, dict):
            if self.kernel["dtype"] != "vendor_float":
                raise ValueError("the opaque kernel's dtype changed")
            return 2 * self.kernel["gain"]
        return self.kernel.rope_theta / self.kernel.window


def test_a_multi_union_leaves_the_selected_opaque_record_for_its_consumer():
    registry = Aliases("mixed", {"mixed": f"{__name__}:MixedSpec"})
    built = registry.build("mixed", kernel={"gain": 7, "dtype": "vendor_float"})
    assert built.frequency() == 14



@dataclasses.dataclass(frozen=True)
class Kind:
    window: int | None = None
    rope_theta: float = 10_000.0


@dataclasses.dataclass(frozen=True)
class Shape:
    width: int = 8
    mix: Kind | None = None
    kinds: Mapping[str, Kind] = dataclasses.field(default_factory=dict)
    layers: tuple[Kind, ...] = ()
    size: tuple[int, int] = (1, 1)
    stages: tuple = ()
    dtype: Any = None


def shapes() -> Aliases:
    return Aliases("shape", {"shape": f"{__name__}:Shape"})


def test_a_record_builds_the_value_its_field_declares():
    """A model config is a dict on the command line and in run.json, so a
    field whose type is a value takes that dict, and code that already holds
    the value passes it through."""
    built = shapes().build("shape", mix={"rope_theta": 1e6},
                           kinds={"sliding": {"window": 512}})
    assert built.mix == Kind(rope_theta=1e6)
    assert built.kinds == {"sliding": Kind(window=512)}
    assert shapes().build("shape", mix=Kind(window=1)).mix == Kind(window=1)
    built = shapes().build("shape", size=[4, 8], layers=[{"window": 2}, {}])
    assert built.size == (4, 8)
    assert built.layers == (Kind(window=2), Kind())


def test_a_record_with_the_wrong_number_of_entries_for_a_tuple_is_refused():
    with pytest.raises(ValueError):
        shapes().build("shape", size=[4])


def test_a_record_that_names_a_field_the_value_does_not_have_is_refused():
    with pytest.raises(ValueError):
        shapes().build("shape", mix={"theta": 1e6})


def _data(records=48, batch=8):
    """A Loader value with nothing behind it: the intervals read only its
    record count and its batch."""
    return Loader(train=lambda partition: iter(()), val=None, records=records, batch=batch)


def test_a_run_records_its_token_budget_and_final_checkpoint_policy(tmp_path):
    config = TrainerConfig(tokens=30, checkpoint_final=False,
                           checkpoint_dir=str(tmp_path), eval_every=None, checkpoint_every=None)
    run = RunConfig(trainer=config)
    assert RunConfig.from_dict(run.to_dict()).trainer == config
    assert config.total_steps(_data()) is None


def test_a_run_can_disable_final_checkpoint_saving(tmp_path):
    run = RunConfig(trainer=TrainerConfig(steps=1, checkpoint_final=False,
                    checkpoint_dir=str(tmp_path), eval_every=None, checkpoint_every=None))
    data = Loader(lambda partition: batches(), None, None, 8)
    assert int(run.train(Regression(), data, name="no-final").step) == 1
    assert Checkpoints(str(tmp_path / "no-final")).latest is None


def test_publishing_requires_a_final_checkpoint_policy():
    with pytest.raises(ValueError):
        TrainerConfig(publish=True, wandb=Wandb(project="test"), checkpoint_final=False)


def test_train_scores_every_named_validation_split(tmp_path):
    """#36 item 5: a held-out split and a test set, scored together at each
    evaluation and logged under their own names."""
    run = RunConfig(trainer=TrainerConfig(
        name="splits", checkpoint_dir=str(tmp_path / "runs"), steps=2,
        eval_every=1, checkpoint_every=None))
    held = next(batches())
    splits = {"val": lambda partition: iter([held]),
              "test": lambda partition: iter([{"x": held["x"], "y": held["y"] + 1}])}
    run.train(Regression(), Loader(lambda partition: batches(), None, None, 8), name="splits",
              validation=splits)
    rows = [json.loads(line) for line in
            (tmp_path / "runs" / "splits" / "tracking" / "scalars.jsonl").read_text().splitlines()]
    val, test = ({row["step"]: row["scalars"][f"{split}/loss"]
                  for row in rows if f"{split}/loss" in row["scalars"]} for split in ("val", "test"))
    assert sorted(val) == sorted(test) == [1, 2]
    assert all(test[step] > val[step] for step in val), "the shifted test targets score a higher loss"


@pytest.mark.mesh(devices=2)
def test_train_runs_the_trainer_its_config_describes(tmp_path):
    """Every field a trainer holds, off its default, shows in the run `train`
    makes of it: the parameters split over fsdp, a dynamic loss scale, an
    accumulator for two microbatches and the profiler window's trace."""
    import jax

    from dew.training import ProfileWindow

    trace = tmp_path / "trace"
    config = TrainerConfig(
        name="built", checkpoint_dir=str(tmp_path / "runs"), steps=3, accumulation=2,
        dynamic_scale=True, mesh=MeshSpec(fsdp=2), layout=Layout(min_shard=1, tolerance=1.0),
        profile=ProfileWindow(str(trace), steps=1, warmup=1), eval_every=None, checkpoint_every=None)

    state = RunConfig(trainer=config).train(Regression(), Loader(lambda partition: batches(), None, None, 8),
                                            name="built")

    assert any(not leaf.sharding.is_fully_replicated for leaf in jax.tree.leaves(state.variables))
    assert state.scale is not None
    assert state.accumulation is not None
    assert any(path.is_file() for path in trace.rglob("*"))


class _Bucket:
    """The three calls a run record makes on a path, recorded, and backed by a
    local directory: a real gs:// write needs credentials and a network, and
    what is under test is that a URI is never taken apart with os.path."""

    def __init__(self, uri, root, seen):
        self.uri, self.root, self.seen = str(uri), root, seen

    def __truediv__(self, name):
        return _Bucket(f"{self.uri}/{name}", self.root, self.seen)

    def _local(self):
        return self.root / self.uri.replace("gs://", "")

    def mkdir(self, parents=False, exist_ok=False):
        self.seen.append(("mkdir", self.uri))
        self._local().mkdir(parents=parents, exist_ok=exist_ok)

    def write_text(self, text):
        self.seen.append(("write", self.uri))
        self._local().write_text(text)

    def read_text(self):
        self.seen.append(("read", self.uri))
        return self._local().read_text()

    def __str__(self):
        return self.uri


def test_the_run_record_is_written_to_a_bucket(tmp_path, monkeypatch):
    """A gs:// checkpoint directory has no local form: the record is written
    through epath, and no local filesystem call sees the URI."""
    def refuse(*args, **kwargs):
        raise AssertionError("a URI must not reach the local filesystem calls")

    seen = []
    monkeypatch.setattr(dew.config.epath, "Path",
                        lambda uri: _Bucket(uri, tmp_path, seen))
    monkeypatch.setattr(os, "makedirs", refuse)
    config = RunConfig(trainer=TrainerConfig(steps=6))

    path = config.save("gs://dew-runs/flowers")

    assert path == "gs://dew-runs/flowers/run.json"
    assert seen == [("mkdir", "gs://dew-runs/flowers"),
                    ("write", "gs://dew-runs/flowers/run.json")]
    assert RunConfig.load("gs://dew-runs/flowers") == config
    assert seen[-1] == ("read", "gs://dew-runs/flowers/run.json")


def test_a_saved_model_retains_nested_mixer_behavior(tmp_path):
    import jax

    from dew.nn.backbones.layer_plan import LayerKind
    from dew.nn.mixers import AttentionMixer
    from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer

    mixer = GatedDeltaNetMixer(
        linear_num_key_heads=1, linear_num_value_heads=1,
        linear_key_head_dim=4, linear_value_head_dim=4, linear_conv_kernel_dim=2)
    run = RunConfig(model=ModelConfig("causal_transformer", {
        "vocab_size": 8, "emb_features": 4, "num_layers": 1, "num_heads": 1,
        "max_seq_len": 8, "mlp_features": 8, "mixer": AttentionMixer(),
        "kinds": {"full_attention": LayerKind(mixer=mixer)},
        "dtype": "float32",
    }))
    model = run.model.build()
    tokens = jnp.asarray([[0, 1, 2, 3]], jnp.int32)
    variables = model.init(jax.random.key(1), tokens)
    expected = model.apply(variables, tokens)
    run.save(str(tmp_path))
    restored = RunConfig.load(str(tmp_path)).model.build()
    assert jnp.array_equal(restored.apply(variables, tokens), expected)


@dataclasses.dataclass(frozen=True)
class VisionRun(RunConfig):
    """A run that records a vision tower and its projector."""

    vision: dict[str, object] = dataclasses.field(default_factory=lambda: {
        "tower": SiglipVision(hidden_size=4, intermediate_size=8, num_layers=1, num_heads=1, image_size=2,
                              patch_size=1),
        "projector": GemmaProjector(text_width=2, patches_per_side=2, tokens_per_side=1),
    })


def test_a_saved_run_retains_vision_tower_and_projector_outputs(tmp_path):
    import jax

    run = VisionRun()
    tower, projector = run.vision["tower"].build(), run.vision["projector"]
    pixels = jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 2, 2)
    tower_params = tower.init(jax.random.key(2), pixels)
    encoded = tower.apply(tower_params, pixels)
    projector_params = projector.init(jax.random.key(3), encoded)
    expected = projector.apply(projector_params, encoded)
    run.save(str(tmp_path))
    restored = VisionRun.load(str(tmp_path))
    actual = projectors.from_record(restored.vision["projector"]).apply(
        projector_params, towers.from_record(restored.vision["tower"]).build().apply(
            tower_params, pixels))
    assert jnp.array_equal(actual, expected)


def test_a_bare_tuple_record_builds_a_jitted_residual_block():
    import jax

    from dew.nn.blocks import ResidualBlock

    registry = Aliases("block", {"residual": "dew.nn.blocks:ResidualBlock"})
    model = registry.build("residual", features=2, norm_groups=0, kernel_size=[1, 3])
    pixels = jnp.arange(12, dtype=jnp.float32).reshape(1, 2, 3, 2)
    time = jnp.ones((1, 2))
    variables = model.init(jax.random.key(4), pixels, time)
    expected = ResidualBlock(features=2, norm_groups=0, kernel_size=(1, 3)).apply(
        variables, pixels, time)
    actual = jax.jit(lambda module, x: module.apply(variables, x, time), static_argnums=0)(
        model, pixels)
    assert jnp.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_deferred_annotations_keep_inherited_buildable_fields(monkeypatch):
    import sys
    import types

    module = types.ModuleType("dew_optional_annotation_case")
    module.__dict__.update(dataclasses=dataclasses, Kind=Kind)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    future = "from __future__ import annotations\n" if sys.version_info < (3, 14) else ""
    exec(future + """
@dataclasses.dataclass(frozen=True)
class Base:
    kernel: Kind

@dataclasses.dataclass(frozen=True)
class Spec(Base):
    extra: Unavailable | None = None

    def frequency(self):
        return self.kernel.rope_theta / self.kernel.window
""", module.__dict__)
    registry = Aliases("deferred", {"spec": f"{module.Spec.__module__}:Spec"})
    built = registry.build("spec", kernel={"window": 4, "rope_theta": 20.0})
    assert built.frequency() == 5.0


def test_a_learning_rate_schedule_is_a_typed_record_that_round_trips():
    """A run's record reads back the schedule it names, its tail and the muP
    groups included."""
    from dew.optim import ParamGroup, Power, PowerTail
    config = RunConfig(
        trainer=TrainerConfig(steps=1),
        optim=OptimConfig(learning_rate=Power(peak_value=0.01, warmup_steps=5, a=4.0, c=16.0,
                                         tail=PowerTail(start=8)),
                          param_groups=ParamGroup.mup(4.0)))
    loaded = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert loaded == config


def test_param_groups_with_schedules_momentum_and_bounds_round_trip():
    """A run's record reads back each group's schedules, `every` included,
    its bounds, and the config's own one-cycle schedule."""
    from dew.optim import Cosine, Exponential, OneCycle, ParamGroup
    momentum = OneCycle(peak_value=0.85, init_value=0.95, end_value=0.95, every=40)
    config = RunConfig(
        trainer=TrainerConfig(steps=1),
        optim=OptimConfig(
            optimizer="adam",
            learning_rate=Exponential(init_value=1e-3, end_value=1e-5, decay_steps=8, offset=1e-6),
            param_groups=(
                ParamGroup("delays", ("*/delay",), schedule=Cosine(peak_value=0.1, warmup_steps=0, every=40),
                           bounds=(0.0, 24.0)),
                ParamGroup("weights", ("*/kernel",), weight_decay=1e-5,
                           schedule=OneCycle(peak_value=5e-3, every=40), b1=momentum),
                ParamGroup("rest", ("*",), schedule=OneCycle(peak_value=5e-3, warmup_fraction=0.25),
                           b1=momentum),
            )))
    loaded = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert loaded == config
    assert loaded.optim.param_groups[0].bounds == (0.0, 24.0)


def test_every_kind_of_run_scores_its_own_metrics_unless_it_names_each_splits():
    """`val_metrics` is every run's (`RunConfig`). Where it names none, each split scores the kind's own
    (`default_metrics`): a language model its perplexity, a decision run its held-out suite, a JEPA run
    its probes. Where it names them, each split scores those that name it or no split
    (`MetricConfig.split`), a split none names scores none, and a metric of a split the run does not
    validate is refused; the record round trips through the command line and run.json alike."""
    import types

    from dew.config import MetricConfig
    from dew.config.jepa import JepaRunConfig
    from dew.config.lm import LMRunConfig
    from dew.decision.config import DecisionRunConfig
    from dew.decision.metrics import AURC, ECE, Accuracy
    from dew.decision.scoring import Brier, LogLoss
    from dew.objectives.jepa.probes import KnnProbe, LinearProbe
    from dew.objectives.lm.objective import Perplexity

    def kinds(metrics):
        return [type(metric) for metric in metrics]

    assert kinds(LMRunConfig().validation_metrics(validates=True)) == [Perplexity]
    assert LMRunConfig().validation_metrics(validates=False) == ()
    assert kinds(DecisionRunConfig().validation_metrics(validates=True)) == [Accuracy, ECE, AURC, LogLoss]
    assert kinds(JepaRunConfig(probe_classes=7).validation_metrics(validates=True)) == [LinearProbe, KnnProbe]
    linear, knn = JepaRunConfig(probe_classes=7, knn_k=5).validation_metrics(validates=False)
    assert (linear.num_classes, knn.num_classes, knn.k) == (7, 7, 5)
    splits = {"val": object(), "test": object()}
    stream = types.SimpleNamespace(val=None)
    assert LMRunConfig().suites(stream, splits) is None
    flag = '[{"name": "accuracy"}, {"name": "brier", "split": "test"}]'
    named = DecisionRunConfig.cli(["--val-metrics", flag])
    assert named.val_metrics == (MetricConfig("accuracy"), MetricConfig("brier", split="test"))
    suites = named.suites(stream, splits)
    assert {split: (suite.data, kinds(suite.metrics)) for split, suite in suites.items()} == {
        "val": (splits["val"], [Accuracy]), "test": (splits["test"], [Accuracy, Brier])}
    assert DecisionRunConfig.from_dict(named.to_dict()) == named
    only = LMRunConfig(val_metrics=[MetricConfig("perplexity", split="test")]).suites(stream, splits)
    assert {split: kinds(suite.metrics) for split, suite in only.items()} == {"val": [], "test": [Perplexity]}
    assert LMRunConfig(val_metrics=[]).suites(types.SimpleNamespace(val=None), None) == {}
    for stray in ([MetricConfig("brier", split="holdout")], [MetricConfig("brier")]):
        with pytest.raises(ValueError):
            LMRunConfig(val_metrics=stray).suites(stream, splits if stray[0].split else None)


@pytest.mark.parametrize("kind", ["lm", "decision", "diffusion"])
def test_every_run_kind_reads_its_source_options_from_the_command_line(kind):
    """`--source.text-only`, `--source.prediction-depths` and `--source.attention` are every run kind's,
    and its source's load reads them beside where it reads the source (`RunConfig.source_keywords`):
    the attention as the load's `modeling_overrides`."""
    from dew.config import SourceOptions
    from dew.config.diffusion import DiffusionRunConfig
    from dew.config.lm import LMRunConfig
    from dew.decision.config import DecisionRunConfig

    kinds = {"lm": LMRunConfig, "decision": DecisionRunConfig, "diffusion": DiffusionRunConfig}
    run = kinds[kind].cli(["--source.text-only", "--source.prediction-depths", "0",
                           "--source.attention", "full_attention", "block_bidirectional",
                           "--trainer.multi-host", "False"])
    assert run.source == SourceOptions(text_only=True, prediction_depths=0,
                                       attention={"full_attention": "block_bidirectional"})
    keywords = run.source_keywords(frozen=False)
    assert keywords["text_only"] is True and keywords["prediction_depths"] == 0 and keywords["on_host"]
    assert keywords["modeling_overrides"] == {"attention": {"full_attention": "block_bidirectional"}}
    assert type(run).from_dict(run.to_dict()).source == run.source


@pytest.mark.parametrize("options", [{"text_only": True},
                                     {"modeling_overrides": {"attention": {"full_attention": "causal"}}}],
                         ids=["text-only", "overrides"])
def test_a_diffusion_source_has_no_language_model_for_text_only_to_take(options):
    """A diffusion pipeline has no wrapper's language model for `text_only` to take, and no component
    an override reaches: each is refused before anything is read."""
    from dew.interop.pretrained import load_diffusion_source

    with pytest.raises(ValueError):
        load_diffusion_source("unused/pipeline", **options)


def test_a_trainers_cadences_may_be_shares_of_the_run_on_the_command_line():
    """`--trainer.log-every`, `--trainer.eval-every` and `--trainer.checkpoint-every` each take a share
    of the run such as 5%, which reaches fit as written for it to resolve against the run's length;
    a value that is neither steps, a share, epoch nor a duration is refused."""
    from dew.config.lm import LMRunConfig

    run = LMRunConfig.cli(["--trainer.log-every", "1%", "--trainer.eval-every", "5%",
                           "--trainer.checkpoint-every", "10%", "--trainer.multi-host", "False"])
    trainer = run.trainer
    assert (trainer.log_every, trainer.eval_every, trainer.checkpoint_every) == ("1%", "5%", "10%")
    for flags in (["--trainer.eval-every", "5x"], ["--trainer.log-every", "epoch"]):
        with pytest.raises((ValueError, SystemExit)):
            LMRunConfig.cli([*flags, "--trainer.multi-host", "False"])


def test_a_decision_run_chooses_its_head_by_its_subcommand():
    """`head:letters` and `head:joint-shape` choose a decision run's head on its command line, each with
    its own flags, and none keeps Laya's."""
    from dew.decision.config import DecisionRunConfig, JointShape, Letters

    run = DecisionRunConfig.cli(["--trainer.multi-host", "False", "head:letters", "--head.count", "4"])
    assert run.head == Letters(count=4)
    joint = DecisionRunConfig.cli(["--trainer.multi-host", "False", "head:joint-shape", "--head.layers", "2"])
    assert joint.head == JointShape(layers=2)
    assert DecisionRunConfig.cli(["--trainer.multi-host", "False"]).head is None


def test_a_decision_run_trains_its_backbone_beside_its_context_layers_on_the_command_line():
    """`--no-context-frozen` with `--context-layers` trains the backbone whole beside the blocks; by
    default it holds frozen."""
    from dew.decision.config import DecisionRunConfig

    flags = ["--trainer.multi-host", "False", "--context-layers", "0", "1"]
    assert DecisionRunConfig.cli(flags).context_frozen
    assert not DecisionRunConfig.cli([*flags, "--no-context-frozen"]).context_frozen


def test_a_run_branches_into_the_cooldowns_its_command_line_names():
    """`--trainer.cooldowns.at` and its length and shape choose a run's cooldown branches, which its
    record holds and reads back."""
    from dew.config.lm import LMRunConfig
    from dew.training.cooldown import Cooldowns

    run = LMRunConfig.cli(["--trainer.cooldowns.at", "20%", "40%", "--trainer.cooldowns.length", "10%",
                           "--trainer.cooldowns.shape", "linear", "--trainer.multi-host", "False"])
    assert run.trainer.cooldowns == Cooldowns(at=("20%", "40%"), length="10%", shape="linear")
    assert LMRunConfig.from_dict(run.to_dict()).trainer.cooldowns == run.trainer.cooldowns
    assert LMRunConfig.cli(["--trainer.multi-host", "False"]).trainer.cooldowns is None
