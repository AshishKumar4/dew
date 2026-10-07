"""RunConfig: the typed record of a run, its round trip, and what it builds."""

import dataclasses
import json
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import pytest
from flax import linen as nn
from test_instrumentation import Regression, batches

import dew.config
import dew.nn.backbones
from dew.config import ModelConfig, OptimConfig, RunConfig, TrainerConfig
from dew.data import Dataset
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.registry import Aliases, datasets, projectors, towers
from dew.training import Layout, MeshSpec


def test_to_dict_and_from_dict_round_trip_a_run():
    config = RunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 4, "emb_features": 64}),
        data=datasets["tfds_images"](image_size=64),
        optim=OptimConfig(optimizer="lamb", learning_rate=1e-3, weight_decay=0.1, b1=0.8, b2=0.99),
        trainer=TrainerConfig(name="run", steps=10, mesh=MeshSpec(fsdp=2),
                              layout=Layout(rules={"mlp": "fsdp"}, min_shard=8)),
    )
    record = config.to_dict()
    assert record["trainer"]["layout"]["rules"] == [["mlp", "fsdp"]]
    assert RunConfig.from_dict(record) == config


def test_a_tuple_field_comes_back_a_tuple_from_a_record():
    """JSON has no tuple, so a record holds a list where the class declares
    one. The class gets its tuple back: with a list in its place the loaded
    config compares unequal to the saved one and its spec is unhashable."""
    config = RunConfig(data=datasets["array_record_images"](image_size=64, shards=("cc12m",)),
                       trainer=TrainerConfig(steps=1))
    loaded = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    assert loaded == config


def test_a_record_with_an_unknown_field_or_without_a_required_one_is_refused():
    record = RunConfig().to_dict()
    with pytest.raises(ValueError, match="unknown fields \\['epochs_per_eval'\\]"):
        RunConfig.from_dict({**record, "trainer": {**record["trainer"], "epochs_per_eval": 1}})
    with pytest.raises(ValueError, match="missing fields \\['project'\\]"):
        RunConfig.from_dict({**record, "trainer": {**record["trainer"], "wandb": {"entity": "dew"}}})
    with pytest.raises(ValueError, match="no class is named 'flowers'"):
        RunConfig.from_dict({**record, "data": {"class": "flowers", "fields": {}}})


def test_a_field_a_record_lacks_takes_its_default():
    record = RunConfig(trainer=TrainerConfig(steps=7)).to_dict()
    trainer = dict(record["trainer"])
    del trainer["steps"]
    assert RunConfig.from_dict({**record, "trainer": trainer}).trainer.steps == TrainerConfig().steps


def test_a_registered_arrayrecord_spec_round_trips(tmp_path):
    """The unregistered-base bug: data=ArrayRecordImages used to write bare
    fields that from_dict could not rebuild. Registered, it records as
    {'name', 'fields'} like every other dataset and loads back."""
    from dew.data.images import ArrayRecordImages

    config = RunConfig(
        data=ArrayRecordImages(path=str(tmp_path), image_size=8),
        trainer=TrainerConfig(steps=1))
    loaded = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    assert loaded == config
    assert config.to_dict()["data"]["class"] == datasets.paths["array_record_images"]


def test_an_unregistered_member_fails_at_write_not_at_reload(tmp_path):
    """A spec the registry does not know cannot be written: to_dict raises
    now, instead of saving a record from_dict can never load back."""
    from dew.data.images import ImageDataset

    @dataclasses.dataclass(frozen=True)
    class Unregistered(ImageDataset):
        path: str | None = None

    config = RunConfig(
        data=Unregistered(path=str(tmp_path), image_size=8),
        trainer=TrainerConfig(steps=1))
    with pytest.raises(ValueError, match=r"Unregistered.*has no import path a record can name"):
        config.to_dict()


def test_save_and_load_carry_a_subclass_with_its_own_knobs(tmp_path):
    @dataclasses.dataclass(frozen=True)
    class LMRunConfig(RunConfig):
        seq_len: int = 256
        pad_id: int | None = None

    config = LMRunConfig(trainer=TrainerConfig(steps=3), seq_len=64, pad_id=0)
    assert config.save(str(tmp_path)) == str(tmp_path / "run.json")
    assert LMRunConfig.load(str(tmp_path)) == config
    with pytest.raises(ValueError, match="unknown fields \\['pad_id', 'seq_len'\\]"):
        RunConfig.load(str(tmp_path))


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


def test_a_model_builds_and_records_its_own_fields():
    fields = {"vocab_size": 64, "emb_features": 32, "num_layers": 1, "num_heads": 2,
              "dtype": "float32", "precision": "high"}
    config = ModelConfig("causal_transformer", fields)

    assert config.fields == fields
    assert config.build().precision == "high"
    assert RunConfig(model=config).to_dict()["model"] == {
        "name": config.name, "fields": fields, "adapter": None, "quantization": None}


class Unprecise(nn.Module):
    """A model that declares none of the precision fields, as a plugin's may."""

    features: int = 4

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features)(x)


def test_a_model_without_dtype_builds_and_refuses_a_dtype_override():
    unprecise = f"{__name__}:Unprecise"
    config = ModelConfig(unprecise, {"features": 2})
    with pytest.raises(ValueError, match=r"declares no dtype field and holds no part that does"):
        config.with_dtype("float32")
    assert isinstance(config.build(), Unprecise)


def test_a_composite_takes_a_dtype_override_through_its_parts():
    """A dtype override reaches DiffusionGemma's decoder without mutating the source."""
    text = {"class": "causal_transformer", "fields": {"vocab_size": 64, "emb_features": 32,
                                                     "num_layers": 1, "num_heads": 2}}
    fields = {"text": text, "canvas_length": 4}

    config = ModelConfig("diffusion_gemma", fields)
    written = config.with_dtype("float32").fields
    assert written["text"]["fields"]["dtype"] == "float32"
    assert "dtype" not in config.fields["text"]["fields"]
    with pytest.raises(ValueError, match=r"declares no dtype field and holds no part that does"):
        ModelConfig("diffusion_gemma", {"canvas_length": 4}).with_dtype("float32")


def test_an_unknown_model_field_is_refused():
    with pytest.raises(ValueError, match="unknown fields \\['depth'\\]"):
        ModelConfig("simple_dit", {"depth": 3}).build()


def test_the_run_length_is_steps_or_epochs():
    with pytest.raises(ValueError, match="set one"):
        TrainerConfig(steps=10, epochs=1)

    class Streamed:
        steps_per_epoch = None

    class Sized:
        steps_per_epoch = 25

    assert TrainerConfig(steps=10).total_steps(Sized()) == 10
    assert TrainerConfig(epochs=4).total_steps(Sized()) == 100
    with pytest.raises(ValueError, match="record count"):
        TrainerConfig(epochs=4).total_steps(Streamed())
    with pytest.raises(ValueError, match=r"--trainer.steps or --trainer.epochs"):
        TrainerConfig().total_steps(Sized())


def test_the_cli_parses_the_mesh_the_layout_and_a_dataset_subcommand():
    config = RunConfig.cli( [
        "--trainer.mesh.fsdp", "2", "--trainer.layout.min-shard", "8",
        "--trainer.steps", "5", "--model", "uvit", "--model.emb_features", "32",
        "data:token-windows", "--data.path", "tokens", "--data.seq-len", "8"])
    assert config.trainer.mesh == MeshSpec(fsdp=2)
    assert config.trainer.layout.min_shard == 8
    assert config.model.fields == {"emb_features": 32}
    assert type(config.data) is datasets["token_windows"]
    assert config.data.seq_len == 8
    assert RunConfig.from_dict(config.to_dict()) == config



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
    with pytest.raises(ValueError, match="2 entries"):
        shapes().build("shape", size=[4])


def test_a_dtype_is_a_dtype_wherever_it_is_named():
    """The rule is the field's name, at the top level and inside a record the
    unets keep their per-stage attention settings in."""
    built = shapes().build("shape", dtype="bfloat16", stages=({"dtype": "float32"},))
    assert built.dtype is jnp.bfloat16
    assert built.stages[0]["dtype"] is jnp.float32


def test_a_record_that_names_a_field_the_value_does_not_have_is_refused():
    with pytest.raises(ValueError, match=r"Kind does not match the record: unknown fields \['theta'\]"):
        shapes().build("shape", mix={"theta": 1e6})


def _data(records=48, batch=8):
    """A Dataset value with nothing behind it: the intervals read only its
    record count and its batch."""
    return Dataset(train=lambda partition: iter(()), val=None, records=records, batch=batch)


def test_an_interval_is_steps_a_pass_or_never():
    """The three answers a run needs from one field: a number of steps, one
    pass over the data, and never."""
    data = _data(records=48, batch=8)

    assert TrainerConfig().checkpoint_interval(data) == 6, "epoch is the default"
    assert TrainerConfig().eval_interval(data) == 6
    assert TrainerConfig(checkpoint_every=100).checkpoint_interval(data) == 100
    assert TrainerConfig(eval_every=100).eval_interval(data) == 100
    assert TrainerConfig(checkpoint_every=None).checkpoint_interval(data) is None
    assert TrainerConfig(eval_every=None).eval_interval(data) is None


def test_a_dataset_at_another_batch_than_the_run_is_refused(tmp_path, capsys):
    """The batch is one number: the recipes load the data at
    `trainer.batch_size` and `train` refuses a dataset that reads another,
    naming both, instead of training at one batch and reporting the other."""
    run = RunConfig(trainer=TrainerConfig(
        name="batch", checkpoint_dir=str(tmp_path / "runs"), steps=1, batch_size=8,
        eval_every=None, checkpoint_every=None))

    with pytest.raises(ValueError, match="batch-size is 8 and this dataset reads 16"):
        run.train(Regression(), Dataset(lambda partition: batches(), None, None, 16), name="batch")

    state = run.train(Regression(), Dataset(lambda partition: batches(), None, None, 8), name="batch")
    assert int(state.step) == 1
    output = capsys.readouterr().out
    assert "Experiment_Name: batch" in output
    assert f"Local tracking: {tmp_path / 'runs' / 'batch' / 'tracking'}" in output


def test_train_scores_every_named_validation_split(tmp_path):
    """#36 item 5: a held-out split and a test set, scored together at each
    evaluation and logged under their own names."""
    run = RunConfig(trainer=TrainerConfig(
        name="splits", checkpoint_dir=str(tmp_path / "runs"), steps=2, batch_size=8,
        eval_every=1, checkpoint_every=None))
    held = next(batches())
    splits = {"val": lambda partition: iter([held]),
              "test": lambda partition: iter([{"x": held["x"], "y": held["y"] + 1}])}
    run.train(Regression(), Dataset(lambda partition: batches(), None, None, 8), name="splits",
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
        name="built", checkpoint_dir=str(tmp_path / "runs"), steps=3, batch_size=8, accumulation=2,
        dynamic_scale=True, mesh=MeshSpec(fsdp=2), layout=Layout(min_shard=1, tolerance=1.0),
        profile=ProfileWindow(str(trace), steps=1, warmup=1), eval_every=None, checkpoint_every=None)

    state = RunConfig(trainer=config).train(Regression(), Dataset(lambda partition: batches(), None, None, 8),
                                            name="built")

    assert any(not leaf.sharding.is_fully_replicated for leaf in jax.tree.leaves(state.variables))
    assert state.scale is not None
    assert state.accumulation is not None
    assert any(path.is_file() for path in trace.rglob("*"))


def test_a_pass_over_the_data_needs_a_record_count():
    """A stream with no record count has no epoch, so "epoch" raises a
    ValueError that names the field."""
    streaming = _data(records=None)

    with pytest.raises(ValueError, match="checkpoint-every epoch needs a dataset"):
        TrainerConfig().checkpoint_interval(streaming)
    with pytest.raises(ValueError, match="eval-every epoch needs a dataset"):
        TrainerConfig().eval_interval(streaming)
    assert TrainerConfig(checkpoint_every=None).checkpoint_interval(streaming) is None
    assert TrainerConfig(checkpoint_every=5).checkpoint_interval(streaming) == 5


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
    config = RunConfig(trainer=TrainerConfig(batch_size=8, steps=6))

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


def test_a_saved_run_retains_vision_tower_and_projector_outputs(tmp_path):
    import jax

    from dew.nn.vision import GemmaProjector, SiglipVision

    @dataclasses.dataclass(frozen=True)
    class VisionRun(RunConfig):
        vision: dict[str, object] = dataclasses.field(default_factory=lambda: {
            "tower": SiglipVision(hidden_size=4, intermediate_size=8, num_layers=1,
                                   num_heads=1, image_size=2, patch_size=1),
            "projector": GemmaProjector(text_width=2,
                                         patches_per_side=2, tokens_per_side=1),
        })

    run = VisionRun()
    tower, projector = run.vision["tower"].build(), run.vision["projector"].build()
    pixels = jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 2, 2)
    tower_params = tower.init(jax.random.key(2), pixels)
    encoded = tower.apply(tower_params, pixels)
    projector_params = projector.init(jax.random.key(3), encoded)
    expected = projector.apply(projector_params, encoded)
    run.save(str(tmp_path))
    restored = VisionRun.load(str(tmp_path))
    actual = projectors.from_record(restored.vision["projector"]).build().apply(
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
    from dew.training.optim import ParamGroup, Power, PowerTail
    config = RunConfig(
        data=datasets["array_record_images"](image_size=64, shards=("cc12m",)),
        trainer=TrainerConfig(steps=1),
        optim=OptimConfig(schedule=Power(peak=0.01, warmup_steps=5, a=4.0, c=16.0,
                                         tail=PowerTail(start=8)),
                          param_groups=ParamGroup.mup(4.0)))
    loaded = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert loaded == config


def test_param_groups_with_schedules_momentum_and_bounds_round_trip():
    """A run's record reads back each group's schedules, `every` included,
    its bounds, and the config's own one-cycle schedule."""
    from dew.training.optim import Cosine, Exponential, OneCycle, ParamGroup
    momentum = OneCycle(peak=0.85, init=0.95, end=0.95, every=40)
    config = RunConfig(
        data=datasets["array_record_images"](image_size=64, shards=("cc12m",)),
        trainer=TrainerConfig(steps=1),
        optim=OptimConfig(
            optimizer="adam", schedule=Exponential(init=1e-3, end=1e-5, decay_steps=8, offset=1e-6),
            param_groups=(
                ParamGroup("delays", ("*/delay",), schedule=Cosine(peak=0.1, warmup_steps=0, every=40),
                           bounds=(0.0, 24.0)),
                ParamGroup("weights", ("*/kernel",), weight_decay=1e-5,
                           schedule=OneCycle(peak=5e-3, every=40), b1=momentum),
                ParamGroup("rest", ("*",), schedule=OneCycle(peak=5e-3, warmup_fraction=0.25), b1=momentum),
            )))
    loaded = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert loaded == config
    assert loaded.optim.param_groups[0].bounds == (0.0, 24.0)
