"""RunConfig: the typed record of a run, its round trip, and what it builds."""

import dataclasses
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import TimingRegression as Regression, batches
from diffusion_support import images_of
from flax import linen as nn
from model_support import attending

import dew.config
import dew.nn.backbones
from dew.checkpoints import RUN_FILE
from dew.config import RunConfig, TrainerConfig
from dew.config.diffusion import DiffusionRunConfig
from dew.data.dataset import Loader, Ramp
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.precision import Dtypes, Numerics
from dew.nn.vision import GemmaProjector, SiglipVision
from dew.optim import Adam, AdamW, Cosine, GaussNewton, Lamb, Muon, MuonClip, Optax, OptimConfig, ParamGroup
from dew.registry import Aliases, ModelConfig, projectors, towers
from dew.training import (
    Checkpoints,
    FitPolicy,
    Layout,
    LocalTracker,
    MeshSpec,
    TrainerAdvanced,
    UpdateBatch,
    Wandb,
)


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
        optim=OptimConfig(optimizer=Lamb(b1=0.8, b2=0.99, weight_decay=0.1), lr=1e-3),
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


def test_a_run_records_the_data_plan_it_reads_by_and_one_saved_before_names_eight(tmp_path):
    """A run leaves its data plan version to its build, which records this Dew's (`RunConfig.resumed`), so
    a resume reads its order as it did whatever version a later Dew reads by; a saved run whose record
    names none was written under version 8 (`earlier_data_plan`, row 3840), which this Dew reads none of:
    its data's compile refuses it, naming both ways on (tests/test_saved_runs.py)."""
    from dew.data import presets
    from dew.data.plan import PLAN_VERSIONS
    from dew.data.sources.tokens import TokenFiles

    config = RunConfig(data=TokenFiles("tokens/train").pipe(presets.token_windows, seq_len=8, batch=2),
                       trainer=TrainerConfig(steps=1))
    assert config.data_plan_version is None and config.resumed().data_plan_version == PLAN_VERSIONS[-1] == 17
    saved = config.resumed().record()
    assert RunConfig.read(saved).data_plan_version == 17
    earlier = {**saved, "fields": {name: value for name, value in saved["fields"].items()
                                   if name != "data_plan_version"}}
    directory = tmp_path / "planned"
    directory.mkdir()
    (directory / "run.json").write_text(json.dumps(earlier))
    assert RunConfig.load(str(directory)).data_plan_version == 8


def test_a_run_naming_the_version_before_this_dews_records_this_dews(tmp_path):
    """A run that names the data plan version before this Dew's, as one saved under it records, reads it as
    its own (`dew.data.plan.read_by`), and records this Dew's, the version its positions are saved under
    (train 801's F2)."""
    import numpy as np
    from affine_run import FEATURES

    from dew.data import Memory
    from dew.data.compile import compile
    from dew.data.plan import PLAN_VERSIONS

    rows = np.random.default_rng(0).normal(size=(16, FEATURES)).astype(np.float32)
    loader = compile(Memory([{"x": row, "y": 2 * row[:2]} for row in rows]).batch(8).repeat(), resume="exact")
    config = TrainerConfig(name="loaded", checkpoint_dir=str(tmp_path / "runs"), steps=1,
                           eval_every=None, checkpoint_every=None, data_resume="exact")
    RunConfig(trainer=config, data_plan_version=PLAN_VERSIONS[0]).train(Regression(), loader, name="loaded")
    assert RunConfig.load(str(tmp_path / "runs" / "loaded")).data_plan_version == PLAN_VERSIONS[-1]


def test_a_run_naming_a_version_this_dew_reads_none_of_is_refused_with_a_compiled_loader(tmp_path):
    """A run resumed exactly that names a data plan version this Dew reads none of is refused before it
    trains, handed a loader compiled already as a chain is (train 593's review); relaxed, it reads by this
    Dew's (`RunConfig.resumed`)."""
    import numpy as np
    from affine_run import FEATURES

    from dew.data import Memory
    from dew.data.compile import compile

    rows = np.random.default_rng(0).normal(size=(16, FEATURES)).astype(np.float32)
    chain = Memory([{"x": row, "y": 2 * row[:2]} for row in rows]).batch(8).repeat()
    loader = compile(chain)
    config = TrainerConfig(name="named", checkpoint_dir=str(tmp_path / "runs"), steps=1,
                           eval_every=None, checkpoint_every=None, data_resume="exact")
    with pytest.raises(ValueError):
        RunConfig(trainer=config, data_plan_version=13).train(Regression(), loader, name="named")


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
    accumulation to that many of the data's token batches (`TrainerConfig.accumulation`), and refuses
    data batched in rows, a budget no whole number of its batches, or phases of other token batches; an
    update of tokens and of batches together is refused as it is stated."""
    trainer = TrainerConfig(update_tokens=98304, steps=1)
    tokens = Loader(train=lambda partition: iter(()), val=None, records=None, batch=8, batch_tokens=24576)
    assert trainer.update_batches(tokens) == 4
    recorded = json.loads(json.dumps(RunConfig(trainer=trainer).to_dict()))
    assert RunConfig.from_dict(recorded).trainer == trainer
    for refused in (dataclasses.replace(tokens, batch_tokens=None),
                    dataclasses.replace(tokens, batch_tokens=24575)):
        with pytest.raises(ValueError):
            trainer.update_batches(refused)
    with pytest.raises(ValueError):
        UpdateBatch(tokens=98304, batches=3)
    # Phases of one token batch accumulate alike; phases of 24,576 then 12,288 tokens, which one
    # accumulation would pool into updates of 98,304 then 49,152 tokens, are refused (row 4885).
    alike = dataclasses.replace(tokens, batches=(8, 8), token_batches=(24576, 24576))
    assert trainer.update_batches(alike) == 4
    with pytest.raises(ValueError):
        trainer.update_batches(dataclasses.replace(alike, batches=(8, 4), token_batches=(24576, 12288)))


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


@attending("xla")
def test_the_model_config_builds_with_its_own_dtype_and_attention():
    fields = {"vocab_size": 64, "emb_features": 32, "num_layers": 1, "num_heads": 2}
    config = ModelConfig("causal_transformer", {**fields, "dtype": "bfloat16"})
    model = config.build()
    import jax
    ids = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
    params = model.init(jax.random.key(0), ids)
    logits = model.apply(params, ids)
    expected = CausalTransformer(**fields, dtype=jnp.bfloat16).apply(
        params, ids)
    assert logits.dtype == jnp.float32
    assert jnp.array_equal(logits, expected)


@attending("reference")
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
                              dtype="float32")
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


def test_a_setting_of_an_unset_layout_sets_it_from_its_defaults():
    """A run whose trainer states no layout (`TrainerConfig.layout` unset) takes a setting of one of its
    fields at its path (`RunConfig.override({"trainer.layout.min_shard": 8})`) over `Layout()`'s defaults,
    as the `--trainer.layout.*` flags do (row 9635: refused, naming no field of the run record)."""
    config = RunConfig(data=images_of("tfds://oxford_flowers102", 32))
    assert config.trainer.layout is None
    assert config.override({"trainer.layout.min_shard": 8}).trainer.layout == Layout(min_shard=8)


def test_the_cli_parses_the_mesh_and_the_layout_over_a_runs_model_and_chain():
    default = RunConfig(data=images_of("tfds://oxford_flowers102", 64),
                        model=ModelConfig("uvit", {"emb_features": 32}))
    config = RunConfig.cli(["--trainer.mesh.fsdp", "2", "--trainer.layout.min-shard", "8",
                            "--trainer.steps", "5"], default=default)
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
    config = TrainerConfig(tokens=30, checkpoint_final=False, checkpoint_dir=str(tmp_path),
                           eval_every=None, checkpoint_every=None)
    run = RunConfig(trainer=config)
    assert RunConfig.from_dict(run.to_dict()).trainer == config
    assert config.total_steps(_data()) is None


def test_a_run_can_disable_final_checkpoint_saving(tmp_path):
    run = RunConfig(trainer=TrainerConfig(steps=1, checkpoint_final=False,
                                          checkpoint_dir=str(tmp_path), eval_every=None,
                                          checkpoint_every=None))
    data = Loader(lambda partition: batches(), None, None, 8)
    assert int(run.train(Regression(), data, name="no-final").step) == 1
    assert Checkpoints(str(tmp_path / "no-final")).latest is None


def test_a_run_that_must_start_afresh_refuses_a_directory_that_holds_one_before_writing_its_record(tmp_path):
    """`trainer.resume="never"` trains a fresh run directory and refuses one that holds a checkpoint, on
    every rank, before the run's record is written over that run's; `resume="require"` refuses a
    directory that holds none to continue, so a job that must continue never trains from scratch."""
    def run(resume, steps: int = 1) -> RunConfig:
        return RunConfig(trainer=TrainerConfig(steps=steps, resume=resume,
                                               checkpoint_dir=str(tmp_path), eval_every=None,
                                               checkpoint_every=None))

    data = Loader(lambda partition: batches(), None, None, 8)
    with pytest.raises(ValueError):
        run("require").train(Regression(), data, name="kept")
    assert int(run("never").train(Regression(), data, name="kept").step) == 1
    record = (tmp_path / "kept" / RUN_FILE).read_bytes()
    with pytest.raises(ValueError):
        run("never", steps=2).train(Regression(), data, name="kept")
    assert (tmp_path / "kept" / RUN_FILE).read_bytes() == record


def test_publishing_requires_a_final_checkpoint_policy():
    with pytest.raises(ValueError):
        TrainerConfig(wandb=Wandb(project="test", publish=True), checkpoint_final=False)


def test_train_scores_every_named_validation_split(tmp_path):
    """#36 item 5: a held-out split and a test set, scored together at each
    evaluation and logged under their own names."""
    run = RunConfig(trainer=TrainerConfig(name="splits", checkpoint_dir=str(tmp_path / "runs"),
                                          steps=2, eval_every=1, checkpoint_every=None))
    held = next(batches())
    splits = {"val": lambda partition: iter([held]),
              "test": lambda partition: iter([{"x": held["x"], "y": held["y"] + 1}])}
    run.train(Regression(), Loader(lambda partition: batches(), None, None, 8), name="splits",
              validation=splits)
    reports = LocalTracker(tmp_path / "runs" / "splits" / "tracking").scalars()
    val, test = ({step: values[f"{split}/loss"] for step, values in reports if f"{split}/loss" in values}
                 for split in ("val", "test"))
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
    config = TrainerConfig(name="built", checkpoint_dir=str(tmp_path / "runs"), steps=3,
                           accumulation=2, precision=Numerics(Dtypes(compute="float16")),
                           mesh=MeshSpec(fsdp=2), layout=Layout(min_shard=1, tolerance=1.0),
                           profile=ProfileWindow(str(trace), steps=1, warmup=1), eval_every=None,
                           checkpoint_every=None)

    state = RunConfig(trainer=config).train(Regression(), Loader(lambda partition: batches(), None, None, 8),
                                            name="built")

    assert any(not leaf.sharding.is_fully_replicated for leaf in jax.tree.leaves(state.variables))
    assert state.scale is not None
    assert state.accumulation is not None
    assert any(path.is_file() for path in trace.rglob("*"))


def test_every_run_profiles_a_short_window_of_its_own_steps_kept_with_its_records(tmp_path):
    """A run profiles two of its steps once ten have run past its first compile, by default, through Dew's
    own profiler, its capture kept under the run's directory and reported beside its records
    (`ProfileTraced`); a run naming no window traces none; inside another capture (`dew.Profiler`), which
    holds the process's one profiling session, a run's own window yields where one naming its directory
    refuses (row 4112)."""
    import dew
    from dew.config.trainer import local_records
    from dew.training import ProfileWindow

    def trained(name: str, **fields) -> Path:
        config = TrainerConfig(name=name, checkpoint_dir=str(tmp_path / "runs"), steps=14,
                               eval_every=None, checkpoint_every=None, **fields)
        RunConfig(trainer=config).train(Regression(), Loader(lambda partition: batches(), None, None, 8),
                                        name=name)
        return tmp_path / "runs" / name

    # The run's window keeps its trace and manifest alone: it asks no XProf and converts no reports.
    from dew.telemetry import profile as telemetry_profile

    def unasked():
        raise AssertionError("a run's own window asked for XProf")

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(telemetry_profile, "require_profile_support", unasked)
        profiled = trained("profiled")
    assert any(path.is_file() for path in (profiled / "profile").rglob("*"))
    assert not any(path.name == "reports" for path in (profiled / "profile").rglob("*"))
    written = (profiled / "tracking" / "records.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in written]
    (traced,) = [row["value"] for row in records if row["type"] == "ProfileTraced"]
    assert traced["steps"] == 2
    # A run in a bucket keeps its capture with its other local records, in Dew's cache (`local_records`).
    remote = TrainerConfig(name="remote", checkpoint_dir="gs://bucket/runs").window("gs://bucket/runs/remote",
                                                                                   "remote")
    assert remote is not None and remote.directory == local_records("gs://bucket/runs/remote", "remote",
                                                                    "profile")
    assert "://" not in remote.directory
    plain = trained("plain", run_profile=None)
    assert not (plain / "profile").exists()
    with dew.Profiler(tmp_path / "outer"):
        inside = trained("inside")
        with pytest.raises(ValueError):
            trained("asked", profile=ProfileWindow(str(tmp_path / "asked"), 2))
    assert not (inside / "profile").exists()


def test_a_window_whose_manifest_is_not_written_leaves_the_run_training(caplog):
    """A profile window that reports nothing writes its manifest as it stops, a diagnostic of the run's:
    where it cannot be written (`dew.telemetry.profile.ExportFailed`, which only such a window's stop
    raises), the stop logs it and the run trains on (train 607's and 610's reviews)."""
    import logging

    from dew.telemetry.profile import ExportFailed
    from dew.training.profiling import stop_capture

    class Unwritten:
        """A profiler whose capture stopped and whose manifest could not be written."""

        def stop(self, *, wait: bool = True) -> None:
            raise ExportFailed("writing the capture's manifest failed", [OSError("read-only directory")])

    with caplog.at_level(logging.WARNING, logger="dew"):
        stop_capture(Unwritten(), None)
    assert any(record.name == "dew.training.profiling" and record.levelno == logging.WARNING
               for record in caplog.records)


@pytest.mark.parametrize(("saved", "expected"), [
    ({"directory": None, "steps": 2, "warmup": 10}, "default"),
    ({"directory": None, "steps": 25, "warmup": 50, "capture": {"host_tracer_level": 1}}, "customized"),
    (None, "none"),
])
def test_a_run_saved_before_its_own_window_was_a_field_traces_as_it_was_configured(tmp_path, saved, expected):
    """A run a Dew saved before a run's own window was its `run_profile` (row 4255) loads as it traced: a
    window naming no directory, the run's own, as its `run_profile` with its steps, warmup and capture,
    trace-only, so it resumes on a plain install asking no XProf; no window at all as none of its own."""
    from dew.telemetry.profile import CaptureOptions
    from dew.training.profiling import RunProfile

    record = RunConfig(trainer=TrainerConfig(name="saved", checkpoint_dir=str(tmp_path))).record()
    trainer = record["fields"]["trainer"]
    held = trainer.get("fields", trainer)
    held.pop("run_profile", None)
    held["profile"] = saved
    (tmp_path / "run.json").write_text(json.dumps(record))
    loaded = RunConfig.load(str(tmp_path / "run.json")).trainer
    window = loaded.window(str(tmp_path / "saved"), "saved")
    assert loaded.profile is None
    if expected == "none":
        assert loaded.run_profile is None and window is None
        return
    own = RunProfile() if expected == "default" else RunProfile(25, 50, CaptureOptions(host_tracer_level=1))
    assert loaded.run_profile == own
    assert window is not None and (window.steps, window.warmup, window.capture) == (own.steps, own.warmup,
                                                                                  own.capture)
    assert not window.reports


def test_a_runs_own_window_and_a_named_one_round_trip_through_its_record(tmp_path):
    """A run's own window, customized, and a window the run names each read back from the run's record as
    they were saved (row 4255's review)."""
    from dew.telemetry.profile import CaptureOptions
    from dew.training import ProfileWindow
    from dew.training.profiling import RunProfile

    for trainer in (TrainerConfig(run_profile=RunProfile(3, 4, CaptureOptions(host_tracer_level=1))),
                    TrainerConfig(profile=ProfileWindow(str(tmp_path / "named"), 5, warmup=6)),
                    TrainerConfig(run_profile=None)):
        saved = RunConfig(trainer=trainer).save(str(tmp_path / "saved"))
        assert RunConfig.load(saved).trainer == trainer


def test_a_saved_runs_profile_window_reads_its_capture_options_as_its_captures(tmp_path):
    """A run saved while its profile window held its capture's options as its own fields
    (`host_tracer_level`, `python_tracer_level`, `activity_cap`) loads with them its capture's
    (`ProfileWindow.capture`, row 2932), so it captures as it did."""
    from dew.telemetry.profile import CaptureOptions
    from dew.training import ProfileWindow

    capture = CaptureOptions(host_tracer_level=1, python_tracer_level=1, activity_cap=7)
    window = ProfileWindow(str(tmp_path / "trace"), 3, capture=capture)
    path = tmp_path / "run" / "run.json"
    assert RunConfig(trainer=TrainerConfig(profile=window)).save(str(tmp_path / "run")) == str(path)
    record = json.loads(path.read_text())
    # A field that declares its class exactly is the record of its fields; a union's is a class record.
    trainer = record["fields"]["trainer"]
    written = trainer.get("fields", trainer)["profile"]
    saved = written.get("fields", written)
    saved.update(saved.pop("capture"))
    path.write_text(json.dumps(record))
    assert RunConfig.load(str(path)).trainer.profile == window


class _Bucket:
    """The calls a run record makes on a path, recorded, and backed by a local
    directory: a real gs:// write needs credentials and a network, and what is
    under test is that a URI is never taken apart with os.path."""

    def __init__(self, uri, root, seen):
        self.uri, self.root, self.seen = str(uri), root, seen

    def __truediv__(self, name):
        return _Bucket(f"{self.uri}/{name}", self.root, self.seen)

    def _local(self):
        return self.root / self.uri.replace("gs://", "")

    def mkdir(self, parents=False, exist_ok=False):
        self.seen.append(("mkdir", self.uri))
        self._local().mkdir(parents=parents, exist_ok=exist_ok)

    def exists(self):
        self.seen.append(("exists", self.uri))
        return self._local().exists()

    def write_text(self, text):
        self.seen.append(("write", self.uri))
        self._local().write_text(text)

    def read_text(self):
        self.seen.append(("read", self.uri))
        return self._local().read_text()

    def unlink(self, missing_ok=False):
        self.seen.append(("unlink", self.uri))
        self._local().unlink(missing_ok=missing_ok)

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
                    ("write", "gs://dew-runs/flowers/run.json"),
                    ("unlink", "gs://dew-runs/flowers/earlier-masked-decay.json")]
    assert RunConfig.load("gs://dew-runs/flowers") == config
    assert ("read", "gs://dew-runs/flowers/run.json") in seen[3:], seen


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
        optim=OptimConfig(schedule=Power(peak_value=0.01, warmup_steps=5, a=4.0, c=16.0,
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
            optimizer=Adam(),
            schedule=Exponential(init_value=1e-3, end_value=1e-5, decay_steps=8, offset=1e-6),
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


def test_every_kind_of_run_scores_its_own_metrics_unless_it_names_each_splits(tmp_path):
    """`val_metrics` and `suites` are every run's (`RunConfig`). Where neither names any, each split
    scores the kind's own (`default_metrics`): a language model its perplexity, a decision run its
    held-out suite, a JEPA run its probes. `val_metrics` names what every split scores, a split's suite
    (`Suite`) what it scores in their place, a suite of no metrics none, and a suite of a split the run
    does not validate is refused, as are metrics named where no split is validated; the record round
    trips through the command line and run.json alike, and a run saved with a split named on a metric,
    its shared ones' null as an earlier serializer wrote it, loads with it as that split's suite (row
    11935)."""
    import types

    from dew.config import MetricConfig, Suite
    from dew.config.jepa import JepaRunConfig
    from dew.config.lm import LMRunConfig
    from dew.decision.config import DecisionRunConfig
    from dew.decision.metrics import AURC, ECE, Accuracy, MacroF1, ScoredQuestions
    from dew.decision.scoring import Brier, LogLoss
    from dew.objectives.jepa.probes import KnnProbe, LinearProbe
    from dew.objectives.lm.objective import Perplexity

    def kinds(metrics):
        return [type(metric) for metric in metrics]

    assert kinds(LMRunConfig().validation_metrics(validates=True)) == [Perplexity]
    assert LMRunConfig().validation_metrics(validates=False) == ()
    assert kinds(DecisionRunConfig().validation_metrics(validates=True)) == [
        Accuracy, MacroF1, ECE, AURC, LogLoss, Brier, ScoredQuestions]
    assert kinds(JepaRunConfig(probe_classes=7).validation_metrics(validates=True)) == [LinearProbe, KnnProbe]
    linear, knn = JepaRunConfig(probe_classes=7, knn_k=5).validation_metrics(validates=False)
    assert (linear.num_classes, knn.num_classes, knn.k) == (7, 7, 5)
    splits = {"val": object(), "test": object()}
    stream = types.SimpleNamespace(val=None)
    assert LMRunConfig().evaluation_suites(stream, splits) is None
    named = DecisionRunConfig.cli(["--val-metrics", '[{"name": "accuracy"}]', "--suites",
                                   '{"test": {"metrics": [{"name": "accuracy"}, {"name": "brier"}]}}'])
    assert named.val_metrics == (MetricConfig("accuracy"),)
    assert named.suites == {"test": Suite(metrics=(MetricConfig("accuracy"), MetricConfig("brier")))}
    suites = named.evaluation_suites(stream, splits)
    assert {split: (suite.data, kinds(suite.metrics)) for split, suite in suites.items()} == {
        "val": (splits["val"], [Accuracy]), "test": (splits["test"], [Accuracy, Brier])}
    assert DecisionRunConfig.from_dict(named.to_dict()) == named
    record = named.record()
    # As an earlier serializer wrote them: every metric's split, a shared one's null.
    saved = {**record["fields"], "val_metrics": [{"name": "accuracy", "split": None},
                                                 {"name": "brier", "split": "test"}]}
    del saved["suites"]
    (tmp_path / "run.json").write_text(json.dumps({**record, "fields": saved}))
    assert DecisionRunConfig.load(str(tmp_path)) == named
    only = LMRunConfig(val_metrics=[], suites={"test": Suite(metrics=(MetricConfig("perplexity"),))})
    scored = only.evaluation_suites(stream, splits)
    assert {split: kinds(suite.metrics) for split, suite in scored.items()} == {
        "val": [], "test": [Perplexity]}
    assert LMRunConfig(val_metrics=[]).evaluation_suites(types.SimpleNamespace(val=None), None) == {}
    # Metrics named where no split is validated would score nothing: refused.
    with pytest.raises(ValueError):
        LMRunConfig(val_metrics=[MetricConfig("perplexity")]).evaluation_suites(
            types.SimpleNamespace(val=None), None)
    with pytest.raises(ValueError):
        LMRunConfig(suites={"holdout": Suite(metrics=())}).evaluation_suites(stream, splits)


@pytest.mark.parametrize("kind", ["lm", "decision", "diffusion"])
def test_every_run_kind_reads_its_source_options_from_the_command_line(kind):
    """`--source.text-only`, `--source.prediction-depths` and `--source.attention` are every run kind's,
    and its source's load reads them beside where it reads the source (`RunConfig.source_keywords`): the
    attention as the load's `modeling_overrides`."""
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
    # A source the run trains is stored in its master weights' dtype, one it keeps frozen as stored.
    assert keywords["param_dtype"] == "float32" and run.source_keywords(frozen=True)["param_dtype"] == "auto"
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
    """`context:cross-context --context.layers` reads a decision run's context through those layers, the
    backbone frozen beside their blocks by default, trained whole with `--context.no-frozen`."""
    from dew.decision.config import DecisionRunConfig

    flags = ["--trainer.multi-host", "False", "context:cross-context", "--context.layers", "0", "1"]
    held = DecisionRunConfig.cli(flags).context
    assert held is not None and held.layers == (0, 1) and held.frozen
    trained = DecisionRunConfig.cli([*flags, "--context.no-frozen"]).context
    assert trained is not None and not trained.frozen


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


def test_a_run_whose_model_names_a_component_by_its_alias_reads_back_and_takes_an_override():
    """A run whose model names a component by an alias of another kind (a mixer's `attention`) reads
    back from its record as it was written, and takes an override, through the translation a record of
    an earlier main's fields reads through (`dew.inference.tasks.earlier_model`, ledger 2423)."""
    from dew.config.lm import LMRunConfig
    from dew.registry import ModelConfig

    fields = {"emb_features": 16, "num_layers": 1, "num_heads": 2,
              "mixer": {"class": "attention", "fields": {}}}
    run = LMRunConfig(model=ModelConfig("causal_transformer", fields),
                      trainer=TrainerConfig(steps=1))
    assert LMRunConfig.from_dict(json.loads(json.dumps(run.to_dict()))).model == run.model
    assert run.override({"trainer.steps": 3}).model == run.model


def test_a_saved_runs_encoder_by_its_earlier_path_reads_as_its_path_now_and_a_path_given_today_does_not():
    """A run saved before the encoders that bind a released checkpoint moved to dew.interop (row 5996) names
    one by its dew.inputs path, which its record reads as the path it has now (`earlier_run`); a path given
    today is read as given, and the earlier one names nothing."""
    from dew import registry
    from dew.config.earlier import earlier_run

    record = {"class": "dew.config:RunConfig",
              "fields": {"conditions": {"text": {"encoder": {"class": "dew.inputs.encoders:CLIPText"}}}}}
    read = earlier_run(record)
    assert read["fields"]["conditions"]["text"]["encoder"]["class"] == "dew.interop.encoders:CLIPText"
    with pytest.raises(ValueError):
        registry.imported("dew.inputs.encoders:CLIPText")


def test_a_saved_runs_default_layout_reads_as_unset_and_a_stated_one_as_it_is():
    """A run recorded before its trainer's layout could be unset recorded `Layout()` for its default, which
    reads back unset, its resume laid out as its checkpoint was (row 9534: read as a stated device layout);
    a layout of its own reads as it is."""
    from dew.config.earlier import earlier_trainer_layout
    from dew.distributed.placement import Layout
    from dew.registry import to_record

    default = {"class": "lm", "fields": {"trainer": {"layout": to_record(Layout(), Layout)}}}
    assert earlier_trainer_layout(default)["fields"]["trainer"]["layout"] is None
    stated = {"class": "lm", "fields": {"trainer": {"layout": to_record(Layout(min_shard=8), Layout)}}}
    assert earlier_trainer_layout(stated) == stated
    # Written while the trainer held `layout_unsettable` as a field (train 1109's review, F3): stated.
    fielded = {"class": "lm", "fields": {"trainer": {"layout": to_record(Layout(), Layout),
                                                     "layout_unsettable": True}}}
    assert earlier_trainer_layout(fielded)["fields"]["trainer"] == {"layout": to_record(Layout(), Layout)}


def test_a_run_stating_the_default_layout_saves_and_loads_it_stated(tmp_path):
    """A run whose trainer states `Layout()`, the device layout kept, saves and loads it as stated: a record
    written today says so beside its fields (`LAYOUT_UNSETTABLE`), so the migration of records written while
    `Layout()` was the default leaves it (row 9664: it loaded unset, and the chooser could offload)."""
    from dew.distributed.placement import Layout

    config = RunConfig(data=images_of("tfds://oxford_flowers102", 32))
    config = dataclasses.replace(config, trainer=dataclasses.replace(config.trainer, layout=Layout()))
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config.record()))
    assert RunConfig.load(str(path)).trainer.layout == Layout()
    unset = RunConfig(data=images_of("tfds://oxford_flowers102", 32))
    path.write_text(json.dumps(unset.record()))
    assert RunConfig.load(str(path)).trainer.layout is None


def test_a_run_saved_before_its_retired_and_unsettable_fields_loads_as_it_trained(tmp_path):
    """A decoder run saved while its trainer stated `export_model` (row 11636) and while its messages'
    additive term could not be unset, its true their default (row 11742), loads: the retired field taken off,
    its true read as unset, a false kept; a record written today keeps a stated true as stated. Each was
    refused or read as stated (the head review of row 11742: a saved run's default true raised with no
    messages, or overrode a checkpoint's choice)."""
    from dew.config.earlier import ADDITIVE_UNSETTABLE, earlier_run
    from dew.config.lm import LMRunConfig

    today = LMRunConfig().record()
    earlier = {key: value for key, value in today.items() if key != ADDITIVE_UNSETTABLE}
    trainer = {**today["fields"]["trainer"], "export_model": False}
    path = tmp_path / "run.json"
    path.write_text(json.dumps({**earlier, "fields": {**today["fields"], "trainer": trainer,
                                                      "loop_message_additive": True}}))
    assert LMRunConfig.load(str(path)).loop is None
    # A trainer recorded by its class, as a subclass's is: its retired field taken off inside its record.
    named = {"class": "dew.config.trainer:TrainerConfig", "fields": trainer}
    path.write_text(json.dumps({**earlier, "fields": {**today["fields"], "trainer": named}}))
    assert LMRunConfig.load(str(path)).trainer == LMRunConfig().trainer
    # Data a field holds that names a class and no fields passes as it is, migrating nothing.
    payload = {"class": "builtins:dict"}
    opaque = {**earlier, "fields": {**today["fields"], "trainer": trainer, "payload": payload}}
    assert earlier_run(opaque)["fields"]["payload"] == payload
    for record, additive in ((earlier, False), (today, True)):
        stated = {**record, "fields": {**record["fields"], "loop_message_additive": additive}}
        loop = earlier_run(stated)["fields"]["_earlier_loop"]
        assert loop["fields"]["message_additive"] is additive, (record is today, additive)

def test_a_run_saved_with_its_loops_choices_flat_loads_and_loops_as_it_did(tmp_path):
    """A decoder run saved while it changed its checkpoint's loop a choice at a time, by flat fields
    (`loop_steps`, `loop_layers`, `loop_exit_gate`, its sampled count's `loop_steps_*`; row 11937 states a
    run's loop whole and its passes apart), loads: its changes its loop's (`_LoopChange`), its count its
    passes, and it loops a checkpoint with no loop as it did, its gate's exit objective weighed as before."""
    from test_decision_recipe import QWEN

    from dew.config.earlier import _LoopChange
    from dew.config.lm import LMRunConfig
    from dew.interop import Pretrained
    from dew.nn.backbones.decoder_stack import Loop
    from dew.nn.backbones.looping import EXIT_ENTROPY
    from dew.training.phase_passes import SampledPasses

    today = LMRunConfig(pretrained=str(QWEN)).record()
    # A run of today records no saved run's translation.
    assert "_earlier_loop" not in today["fields"]
    path = tmp_path / "run.json"
    path.write_text(json.dumps({**today, "fields": {**today["fields"], "loop_steps": 2, "loop_layers": [1, 2],
                                                    "loop_exit_gate": True}}))
    loaded = LMRunConfig.load(str(path))
    assert loaded._earlier_loop == _LoopChange(layers=(1, 2), exit_gate=True) and loaded.passes == 2
    assert loaded.exit_settings() == {"exit_entropy": EXIT_ENTROPY}
    looped = loaded.looped(Pretrained.load(QWEN, dtype="float32"))
    assert looped.model.language_model.loop == Loop(2, step_norm=False, layers=(1, 2), exit_gate=True)
    sampled = {"loop_steps_sample": "lognormal-poisson", "loop_steps_mean": 2.0, "loop_steps_dispersion": 0.5,
               "loop_steps_max": 4}
    path.write_text(json.dumps({**today, "fields": {**today["fields"], **sampled}}))
    assert LMRunConfig.load(str(path)).passes == SampledPasses(2.0, 0.5, 4)


@pytest.mark.parametrize(("earlier", "built"), [
    ({"optimizer": "muon", "optimizer_opts": {"mu_dtype": "bfloat16", "nu_dtype": None}},
     Muon(state_dtype=("bfloat16", None))),
    ({"optimizer": "adamw", "optimizer_opts": {"mu_dtype": None, "nu_dtype": None}, "b2": 0.95},
     AdamW(b2=0.95, state_dtype="params")),
    ({"optimizer": "adam", "state_dtype": "int8", "weight_decay": 0.1},
     Adam(weight_decay=0.1, state_dtype="int8")),
    ({"optimizer": "dew.gauss_newton:gauss_newton", "optimizer_opts": {"damping": 1e-3}},
     GaussNewton(damping=1e-3)),
    ({"optimizer": "dew.optim:_muonclip_groups"}, MuonClip()),
    ({"optimizer": "optax:sgd", "optimizer_opts": {"momentum": 0.9}}, Optax("optax:sgd", {"momentum": 0.9})),
    # Its decay masked off, unspared: its decay kept, masked off by its own mask (`OptimConfig.decay_mask`,
    # `test_a_run_whose_decay_its_options_masked_off_resumes_its_optimizers_state_as_it_trained`); spared,
    # its own mask was written over it.
    ({"optimizer": "adamw", "optimizer_opts": {"mask": False}, "spared": False}, AdamW()),
    ({"optimizer": "adamw", "optimizer_opts": {"mask": False}}, AdamW()),
], ids=["muon-moments-apart", "adamw-params-moments", "adam-int8", "gauss-newton-path", "muonclip-path",
        "optax-path", "adamw-masked-unspared", "adamw-masked-spared"])
def test_a_run_saved_before_each_optimizer_held_its_settings_loads_its_optimizer_as_it_trained(
        tmp_path, earlier, built):
    """A run saved while its optimizer config held the optimizer's settings (row 11933) loads its optimizer
    as it trained: its moments' dtypes, apart where they were apart, its own fields, and a Dew optimizer named
    by its factory's import path as that optimizer, never a generic Optax factory over it (review of row
    11933); its trainer's pins, recorded flat or by its class, its diagnostics."""
    from dew.config.lm import LMRunConfig

    today = LMRunConfig().record()
    fields = today["fields"]
    assert isinstance(fields, dict)
    optim, trainer = fields["optim"], fields["trainer"]
    assert isinstance(optim, dict) and isinstance(trainer, dict)
    optim = {key: value for key, value in optim.items() if key != "optimizer"}
    trainer = {**trainer, "xla_flags": "--xla_dump_to=dumped", "tune_kernels": True}
    for held in (trainer, {"class": "dew.config.trainer:TrainerConfig", "fields": trainer}):
        path = tmp_path / "run.json"
        path.write_text(json.dumps({**today, "fields": {**fields, "optim": {**optim, **earlier},
                                                        "trainer": held}}))
        loaded = LMRunConfig.load(str(path))
        assert loaded.optim.optimizer == built
        pins = (loaded.diagnostics.xla_flags, loaded.diagnostics.tune_kernels)
        assert pins == ("--xla_dump_to=dumped", True)
        assert loaded.trainer == LMRunConfig().trainer


MASKED = json.loads((Path(__file__).parent / "fixtures" / "optimizers" / "masked-decay-65609490a.json")
                    .read_text())
"""Optimizers an earlier Dew (main 65609490a) built with their decay masked among their options, off
(`optimizer_opts={"mask": False}`) or by a tree of booleans (`{"mask": {"w": False, "b": True}}`), over
`OptimConfig(lr=1e-2)`'s other settings, each with the state its first step held, its tree's
structure (`treedef`, every node a checkpoint's metadata holds, empty ones among them) and each leaf's path,
shape and dtype, and that step's updates of two parameters with unit gradients, recorded by that Dew's own
`OptimConfig.build`."""


def _saved_optimizer(tmp_path, earlier):
    """The optimizer a language run saved with the optimizer config `earlier` (its earlier fields, over
    `OptimConfig(lr=1e-2)`'s others) loads, its record written in `tmp_path` (`run.json`)."""
    from dew.config.lm import LMRunConfig
    from dew.registry import to_record

    today = LMRunConfig().record()
    fields = today["fields"]
    held = to_record(OptimConfig(lr=1e-2), OptimConfig)
    assert isinstance(fields, dict) and isinstance(held, dict)
    optim = {key: value for key, value in held.items() if key != "optimizer"}
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "run.json"
    path.write_text(json.dumps({**today, "fields": {**fields, "optim": {**optim, **earlier}}}))
    return LMRunConfig.load(str(path)).optim


def _named(directory) -> TrainerConfig:
    """A trainer whose run is `directory`, its name and its checkpoints' directory, which a resume of it
    reads (`RunConfig.resumed`)."""
    return TrainerConfig(name=directory.name, checkpoint_dir=str(directory.parent))


def _typed(step) -> None:
    """Mark the saved step `step` as an intermediate Dew saved it, its optimizer state its typed optimizer's
    (`typed_optimizer`)."""
    written = step / "_CHECKPOINT_METADATA"
    metadata = json.loads(written.read_text())
    metadata["custom_metadata"] = {**(metadata.get("custom_metadata") or {}), "typed_optimizer": True}
    written.write_text(json.dumps(metadata))


def _marked_resume(run, rewritten: OptimConfig) -> OptimConfig:
    """The optimizer a whole resume of `run` reads (`RunConfig.resumed`, every field it resolves), its record
    `rewritten` by an intermediate Dew beside its mark of its whole config (`earlier-masked-decay.json`)."""
    from dew.config.lm import LMRunConfig

    LMRunConfig(optim=rewritten, trainer=_named(run)).save(str(run))
    (run / "earlier-masked-decay.json").write_text(json.dumps({"unmasked_all": True}))
    return LMRunConfig.load(str(run)).resumed().optim


def _checkpointed(directory, optimizer, steps, local=None, persistent: bool = True):
    """A tiny language run of `optimizer` fitted to `steps`, checkpointed each step in `directory` where
    `persistent`, and on this host in `local` where given (`local_every`), which resumes from its latest
    checkpoint in either."""
    from dew.data.dataset import Loader, replayed
    from dew.objectives.lm import LMObjective
    from dew.training import Trainer

    model = CausalTransformer(vocab_size=32, emb_features=8, num_layers=2, num_heads=1, mlp_features=16,
                              max_seq_len=8)
    batch = {"text": jnp.arange(40, dtype=jnp.int32).reshape(8, 5) % 32}
    tiers = {} if local is None else {"local_directory": str(local), "local_every": 1}
    trainer = Trainer(LMObjective(model, seq_len=4), optimizer, key=jax.random.key(0),
                      advanced=TrainerAdvanced(layout=Layout(min_shard=1, tolerance=1.0)),
                      checkpoints=Checkpoints(str(directory), **tiers))
    state = trainer.fit(Loader(train=replayed(lambda partition: iter([batch] * 4)), val=None, records=8,
                               batch=8), steps=steps, log_every=1, checkpoint_every=1 if persistent else None,
                        checkpoint_final=persistent)
    trainer.checkpoints.wait()
    return state


@pytest.mark.parametrize("case", sorted(MASKED))
def test_a_run_whose_decay_its_options_masked_off_resumes_its_optimizers_state_as_it_trained(tmp_path, case):
    """A run saved while its optimizer's options masked its decay (`optimizer_opts={"mask": False}`, or by a
    tree of booleans), unspared or in a param group that did not spare (its `spared`,
    `ParamGroup.decay_norms_and_biases` now), whatever the config spared, loads an optimizer whose decay that
    mask masks as it did (`OptimConfig.decay_mask`, `ParamGroup.decay_mask`), a spared decay's mask dropped as
    that Dew wrote its own over it: its state the earlier Dew's tree, node for node (`MASKED`'s `treedef`),
    each leaf at its path of its shape and dtype, and its step updating as that one's did, so a checkpoint of
    it restores as it was written
    (`test_a_checkpoint_whose_decay_its_options_masked_off_restores_into_the_optimizer_its_record_loads`;
    review of API01: a mask of False was read as a decay of 0, another state tree)."""
    held = MASKED[case]
    optimizer = _saved_optimizer(tmp_path, held["optim"]).build(None)
    params = {"w": jnp.arange(4.0).reshape(2, 2) / 10, "b": jnp.arange(2.0) / 10}
    state = optimizer.init(params)
    assert str(jax.tree.structure(state)) == held["treedef"]
    leaves = [[jax.tree_util.keystr(path), list(leaf.shape), str(leaf.dtype)]
              for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]]
    assert leaves == held["state"], leaves
    updates, _ = optimizer.update(jax.tree.map(jnp.ones_like, params), state, params)
    assert jax.tree.map(lambda leaf: leaf.tolist(), updates) == held["updates"]


@pytest.mark.parametrize("earlier", [
    {"optimizer": "adamw", "optimizer_opts": {"mask": {"w": "no"}}, "spared": False},
    {"optimizer": "adamw", "optimizer_opts": {"mask": "biases"}, "spared": False}],
    ids=["tree-of-other-leaves", "named-mask"])
def test_a_run_whose_masked_decay_this_dew_holds_no_state_of_is_refused_as_it_loads(tmp_path, earlier):
    """A run whose options masked its decay by anything but a tree of booleans (Optax's mask), which no
    record rebuilds, is refused as it loads, never loaded into another optimizer state."""
    with pytest.raises(ValueError):
        _saved_optimizer(tmp_path, earlier)


UNMASKED_GROUPS = [{"name": "kernels", "patterns": ["*kernel*"]},
                   {"name": "rest", "patterns": ["*"], "spared": True}]
"""An earlier run's param groups, its kernels' unspared and the rest's spared, so its mask reads in one
group."""


@pytest.mark.parametrize(("optimizer", "grouped"), [("adamw", False), ("adamw", True), ("adam", False),
                                                    ("adam", True)],
                         ids=["adamw-unspared", "adamw-grouped", "adam-coupled-unspared",
                              "adam-coupled-grouped"])
def test_a_checkpoint_whose_decay_its_options_masked_off_restores_into_the_optimizer_its_record_loads(
        tmp_path, optimizer, grouped):
    """A checkpoint an earlier Dew wrote of a run whose options masked its decay off where they were read,
    unspared or in an unspared param group, holds its decay step masked whole (`optax.MaskedState` over its
    empty state, Adam's coupled one ahead of its moments): the optimizer its record now loads masks it off
    alike (`decay_mask=False`), its state that tree node for node, so a persistent step and a newer local
    one restore through Orbax as written, every leaf, and the run steps on, its record rewritten as any
    resumed run's is (review of API01: a decay of 0 held another tree, which a restore shim read across).
    A group that spared its decay (`rest`) read no mask: a resume that drops its active decay is refused."""
    from dew.config.lm import LMRunConfig
    from dew.optim import ParamGroup

    earlier = {"optimizer": optimizer, "optimizer_opts": {"mask": False}, "weight_decay": 0.1,
               **({"param_groups": UNMASKED_GROUPS} if grouped else {"spared": False})}
    # Its earlier optimizer's state, a masked decay step's empty one where it read its mask, as a spared
    # decay's holds: the same tree the earlier Dew built.
    groups = tuple(ParamGroup(group["name"], tuple(group["patterns"]), decay_norms_and_biases=False)
                   for group in UNMASKED_GROUPS) if grouped else ()
    spec = AdamW(weight_decay=0.1) if optimizer == "adamw" else Adam(weight_decay=0.1)
    written = OptimConfig(optimizer=spec, lr=1e-2, param_groups=groups)
    run, local = tmp_path / "run", tmp_path / "local"
    _checkpointed(run, written, 1)
    saved = _checkpointed(run, written, 2, local=local, persistent=False)
    assert not (run / "2").exists() and list(local.rglob("2"))
    loaded = _saved_optimizer(run, earlier)
    resumed = _checkpointed(run, loaded, 2, local=local, persistent=False)
    assert jax.tree.structure(resumed.opt_state) == jax.tree.structure(saved.opt_state)
    for got, want in zip(jax.tree.leaves(resumed.opt_state), jax.tree.leaves(saved.opt_state), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    LMRunConfig(optim=loaded).save(str(run))
    if grouped:
        # The spared group's decay, active as it trained, dropped on resume: its state another tree.
        rest = [dataclasses.replace(group, weight_decay=0.0, decay_norms_and_biases=True)
                if group.name == "rest" else group for group in loaded.param_groups]
        with pytest.raises(ValueError):
            _checkpointed(run, dataclasses.replace(loaded, param_groups=tuple(rest)), 3, local=local)
    assert int(_checkpointed(run, loaded, 3, local=local).step) == 3


@pytest.mark.parametrize(("optimizer", "decay"), [("adamw", 0.1), ("adam", 0.1), ("adamw", 0.0)],
                         ids=["adamw-decayed", "adam-coupled-decayed", "adamw-undecayed"])
def test_a_run_an_intermediate_dew_rewrote_beside_its_mark_resumes_as_it_trained(tmp_path, optimizer, decay):
    """A run whose earlier record masked its decay off, resumed by an intermediate Dew (main f124c76b5),
    holds that Dew's rewrite of its record (its decay written as 0) beside its mark
    (`earlier-masked-decay.json`) and the earlier checkpoint it had not written again: its record loads its
    decay masked off by its own mask (`decay_mask=False`), a decay of 0 given an inert one, so a checkpoint
    holding the masked decay step restores as it was written, every leaf, and the run steps on; a run whose
    earlier decay was 0, which held no such step, is refused at its restore, its tree another, never read
    across (review of d22: the intermediate format was dropped with its translation)."""
    import json as json_

    from dew.config.lm import LMRunConfig

    spec = AdamW if optimizer == "adamw" else Adam
    # The earlier optimizer: a decay masked off, its step held where its decay was not 0.
    written = OptimConfig(optimizer=spec(weight_decay=decay), lr=1e-2, decay_norms_and_biases=decay == 0)
    run = tmp_path / "run"
    saved = _checkpointed(run, written, 1)
    rewritten = OptimConfig(optimizer=spec(weight_decay=0.0), lr=1e-2, decay_norms_and_biases=True)
    LMRunConfig(optim=rewritten, trainer=_named(run)).save(str(run))
    (run / "earlier-masked-decay.json").write_text(json_.dumps({"unmasked_all": True}))
    loaded = LMRunConfig.load(str(run)).resumed().optim
    assert type(loaded.optimizer) is spec, loaded
    assert loaded.decay_mask is False and loaded.decay_norms_and_biases, loaded
    assert isinstance(loaded.optimizer, Adam) and loaded.optimizer.weight_decay > 0, loaded
    if decay == 0:
        with pytest.raises(ValueError):
            _checkpointed(run, loaded, 2)
        return
    resumed = _checkpointed(run, loaded, 1)
    assert jax.tree.structure(resumed.opt_state) == jax.tree.structure(saved.opt_state)
    for got, want in zip(jax.tree.leaves(resumed.opt_state), jax.tree.leaves(saved.opt_state), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert int(_checkpointed(run, loaded, 2).step) == 2


def test_a_run_an_intermediate_dew_resumed_and_saved_after_its_mark_loads_its_rewritten_record(tmp_path):
    """A run an intermediate Dew (main f124c76b5) resumed under the record it rewrote beside its mark, and
    saved again, its newest step that Dew's own (its metadata's `typed_optimizer`), holds that record's
    optimizer state: its record loads as written, its decay 0 unmasked, never as the earlier run trained, so
    its newest checkpoint restores and the run steps on (review of d22: the mark was read whatever the
    run saved after it, refusing such a run's restore)."""


    rewritten = OptimConfig(optimizer=AdamW(weight_decay=0.0), lr=1e-2, decay_norms_and_biases=True)
    run = tmp_path / "run"
    saved = _checkpointed(run, rewritten, 2)
    _typed(run / "2")
    loaded = _marked_resume(run, rewritten)
    assert isinstance(loaded.optimizer, AdamW), loaded
    assert (loaded.decay_mask, loaded.optimizer.weight_decay) == (None, 0.0), loaded
    resumed = _checkpointed(run, loaded, 2)
    for got, want in zip(jax.tree.leaves(resumed.opt_state), jax.tree.leaves(saved.opt_state), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert int(_checkpointed(run, loaded, 3).step) == 3


def test_an_intermediate_dews_newer_local_step_decides_its_mark_once_and_for_good(tmp_path):
    """A run an intermediate Dew resumed under its rewritten record and saved again on the host's own disk
    alone (`local_directory`), its newest committed step that Dew's own, loads its record as written, as
    the step a resume restores is that one, not the persistent tier's earlier; and once a run writes its
    record again (`RunConfig.save`) its mark is decided for good, gone, so a later load never reads it
    across steps this Dew saved (review of d22: the persistent tier's newest step decided, and a second
    resume read the mark again)."""
    import json as json_

    from dew.config.lm import LMRunConfig

    rewritten = OptimConfig(optimizer=AdamW(weight_decay=0.0), lr=1e-2, decay_norms_and_biases=True)
    run, local = tmp_path / "run", tmp_path / "local"
    _checkpointed(run, rewritten, 1)
    # Its host's own tier under the run's name, as its trainer's `local_directory` places it.
    _checkpointed(run, rewritten, 2, local=local / run.name, persistent=False)
    [newest] = [path for path in local.rglob("_CHECKPOINT_METADATA") if path.parent.name == "2"]
    _typed(newest.parent)
    config = LMRunConfig(optim=rewritten, trainer=dataclasses.replace(
        _named(run), local_directory=str(local), local_every=1))
    config.save(str(run))
    mark = run / "earlier-masked-decay.json"
    mark.write_text(json_.dumps({"unmasked_all": True}))
    # Read as written, opening nothing: the mark is decided where the run resumes, its process joined.
    assert LMRunConfig.load(str(run)) == config
    loaded = LMRunConfig.load(str(run)).resumed()
    assert isinstance(loaded.optim.optimizer, AdamW), loaded.optim
    assert (loaded.optim.decay_mask, loaded.optim.optimizer.weight_decay) == (None, 0.0), loaded.optim
    loaded.save(str(run))
    assert not mark.exists()


def test_a_marked_runs_newest_save_no_resume_reads_leaves_its_mark_to_decide(tmp_path):
    """A run an intermediate Dew rewrote beside its mark whose newest save is one no resume reads, that Dew's
    model alone (a weights-only save of its typed optimizer), loads as its earlier run trained: the mark is
    decided by the step its resume restores (`Checkpoints.latest`), the untyped one before it, its decay
    masked off by its own mask (review of d22 r8: the newest numbered save decided, whatever a resume
    reads)."""
    import json as json_
    import shutil


    rewritten = OptimConfig(optimizer=AdamW(weight_decay=0.0), lr=1e-2, decay_norms_and_biases=True)
    run = tmp_path / "run"
    _checkpointed(run, rewritten, 1)
    shutil.copytree(run / "1", run / "2")
    # Its model alone, which no resume reads (`dew.checkpoints._payloads.PROJECTED`).
    (run / "2" / "projected.json").write_text(json_.dumps(["model"]))
    _typed(run / "2")
    loaded = _marked_resume(run, rewritten)
    assert loaded.decay_mask is False and isinstance(loaded.optimizer, AdamW), loaded
    assert loaded.optimizer.weight_decay > 0, loaded


@pytest.mark.parametrize("optimizer", ["adamw", "adam"], ids=["adamw", "adam-coupled"])
def test_a_checkpoint_whose_decay_its_options_masked_by_a_tree_restores_into_the_optimizer_its_record_loads(
        tmp_path, optimizer):
    """A checkpoint an earlier Dew wrote of a run whose options masked its decay by a tree of booleans
    (unspared: here its matrices alone) holds its decay step masked as a spared decay's is, the tree of
    `MASKED`'s `treedef` node for node: the optimizer its record now loads decays where that tree says
    (`OptimConfig.decay_mask`), so the checkpoint restores through Orbax as it was written, every leaf as
    written, and the run steps on (review of API01: the tree was refused, a run that trained refused its own
    record)."""
    import optax

    spec = AdamW if optimizer == "adamw" else Adam
    case = MASKED[f"{optimizer}-tree-unspared"]
    # The earlier optimizer's state, a masked decay step's: the tree a spared decay holds.
    written = OptimConfig(optimizer=spec(weight_decay=0.1), lr=1e-2)
    params = {"w": jnp.arange(4.0).reshape(2, 2) / 10, "b": jnp.arange(2.0) / 10}
    assert str(jax.tree.structure(written.build(None).init(params))) == case["treedef"]
    run = tmp_path / "run"
    saved = _checkpointed(run, written, 1)
    # The earlier run's mask, by the names of the parameters its optimizer stepped (its first moment's).
    mask = jax.tree.map(lambda moment: moment.ndim > 1, optax.tree_utils.tree_get(saved.opt_state, "mu"))
    assert {False, True} == set(jax.tree.leaves(mask)), mask
    loaded = _saved_optimizer(run, {**case["optim"], "optimizer_opts": {"mask": mask}})
    assert loaded.decay_mask == mask and loaded.decay_norms_and_biases
    resumed = _checkpointed(run, loaded, 1)
    assert jax.tree.structure(resumed.opt_state) == jax.tree.structure(saved.opt_state)
    for got, want in zip(jax.tree.leaves(resumed.opt_state), jax.tree.leaves(saved.opt_state), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert int(_checkpointed(run, loaded, 2).step) == 2


@pytest.mark.parametrize("optimizer", ["adamw", "adam"], ids=["adamw", "adam-coupled"])
@pytest.mark.parametrize("record", ["absent", "today"])
def test_an_active_decays_checkpoint_resumed_without_its_decay_is_refused(tmp_path, optimizer, record):
    """A checkpoint of a run decaying its parameters as its config spared them
    (`OptimConfig.decay_norms_and_biases` off, its
    decay step masked by the sparing: `optax.MaskedState` over an empty state, AdamW's and Adam's coupled
    one's alike) resumed by an optimizer that decays nothing (its decay 0, unspared) is refused by its
    restore, its run's record absent or today's: no restore reads one optimizer's state into another's
    tree."""
    from dew.config.lm import LMRunConfig

    spec = AdamW if optimizer == "adamw" else Adam
    active = OptimConfig(optimizer=spec(weight_decay=0.1), lr=1e-2)
    undecayed = OptimConfig(optimizer=spec(weight_decay=0.0), lr=1e-2, decay_norms_and_biases=True)
    run = tmp_path / "run"
    _checkpointed(run, active, 1)
    if record == "absent":
        (run / "run.json").unlink()
    else:
        LMRunConfig(optim=active).save(str(run))
    with pytest.raises(ValueError):
        _checkpointed(run, undecayed, 2)


@pytest.mark.parametrize("mask", [False, None], ids=["masked-off", "named-none"])
def test_an_optax_factory_takes_the_mask_its_earlier_options_gave_it(tmp_path, mask):
    """An Optax factory an earlier run named by its path takes its options as they gave them
    (`Optax.settings`), a mask among them, one named None too, which a factory whose own default masks
    otherwise reads apart from none given, the config's sparing written over it as before."""
    optimizer = _saved_optimizer(tmp_path, {"optimizer": "optax:adamw", "optimizer_opts": {"mask": mask},
                                            "spared": False}).optimizer
    assert optimizer == Optax("optax:adamw", {"mask": mask})


def test_an_earlier_muons_mask_and_gauss_newtons_cap_load_as_they_trained(tmp_path):
    """An earlier Muon run whose options masked its matrices' decay (`weight_decay_mask`) loads that mask as
    its decay's (`OptimConfig.decay_mask`, by Muon's own keyword); an earlier Gauss-Newton run whose options
    capped its direction beside the config's clip, which wrote over them, loads capped by the config's
    alone, as it trained (review of API01: both records were refused)."""
    from dew.optim import GaussNewton, Muon

    for name, mask in (("tree", {"w": False}), ("off", False), ("on", True)):
        muon = _saved_optimizer(tmp_path / name, {"optimizer": "muon", "weight_decay": 0.1,
                                                  "optimizer_opts": {"weight_decay_mask": mask}})
        assert isinstance(muon.optimizer, Muon) and muon.decay_mask == mask
    grouped = _saved_optimizer(tmp_path / "grouped", {
        "optimizer": "muon", "weight_decay": 0.1, "optimizer_opts": {"weight_decay_mask": {"w": False}},
        "param_groups": [{"name": "all", "patterns": ["*"], "spared": True}]})
    assert [(group.decay_mask, group.decay_norms_and_biases) for group in grouped.param_groups] == [
        ({"w": False}, True)]
    capped = _saved_optimizer(tmp_path / "gauss", {"optimizer": "gauss_newton", "clip_grads": 1.0,
                                                   "optimizer_opts": {"clip": 0.5}})
    assert capped.optimizer == GaussNewton() and capped.clip == 1.0


@pytest.mark.parametrize(("stage", "earlier", "fit"), [
    (2, {"split": 1, "microbatches": None}, FitPolicy(split=1)),
    (2, {"split": 1, "microbatches": 4}, FitPolicy(split=1, microbatches=4)),
    (1, {"split": 2, "microbatches": None}, FitPolicy(split=2)),
    (2, {"split": 2, "microbatches": 4}, FitPolicy(split=2, microbatches=4)),
    (2, {"split": None, "microbatches": None}, FitPolicy())],
    ids=["staged-outer-split", "staged-feeds", "unstaged", "staged-split-and-feeds", "planned"])
def test_a_run_saved_with_its_split_and_its_pipelines_microbatches_loads_both_and_trains_them(
        tmp_path, stage, earlier, fit):
    """A run saved while its trainer held a step's split and a pipeline's microbatches apart (row 11934)
    loads both as it trained them (`FitPolicy.split`, `FitPolicy.microbatches`), whatever its mesh: a
    staged run's outer split beside the microbatches its stages fed among them (review of API01: refused,
    a saved run this Dew would not resume), and its trainer holds both."""
    from dew.config.lm import LMRunConfig
    from dew.objectives.lm import LMObjective
    from dew.training import Trainer

    today = LMRunConfig(trainer=TrainerConfig(mesh=MeshSpec(stage=stage))).record()
    fields = today["fields"]
    assert isinstance(fields, dict) and isinstance(fields["trainer"], dict)
    trainer = {key: value for key, value in fields["trainer"].items() if key not in earlier}
    path = tmp_path / "run.json"
    path.write_text(json.dumps({**today, "fields": {**fields, "trainer": {**trainer, **earlier}}}))
    loaded = LMRunConfig.load(str(path)).trainer
    assert loaded.fit_policy == fit
    model = CausalTransformer(vocab_size=32, emb_features=8, num_layers=2, num_heads=1, mlp_features=16,
                              max_seq_len=8)
    run = Trainer(LMObjective(model, seq_len=4), optax.sgd(0.1), key=jax.random.key(0), split=loaded.split,
                  microbatches=loaded.microbatches, mesh=loaded.mesh)
    assert (run.split, run.fit_policy.microbatches) == (fit.stated_split, fit.microbatches)


def test_a_run_saved_with_its_experts_packed_loads_with_its_weights_packed(tmp_path):
    """A run saved while `SourceOptions.packed_weights` was `packed_experts` (row 8550) loads with its
    source's weights packed as it trained, its record read under today's name (`earlier_packed_experts`)."""
    from dew.data import presets
    from dew.data.sources.tokens import TokenFiles
    from dew.interop.source_options import SourceOptions

    record = RunConfig(data=TokenFiles("tokens/train").pipe(presets.token_windows, seq_len=8, batch=2),
                       trainer=TrainerConfig(steps=1),
                       source=SourceOptions(packed_weights=True)).record()
    source = dict(record["fields"]["source"])
    source["packed_experts"] = source.pop("packed_weights")
    directory = tmp_path / "packed"
    directory.mkdir()
    earlier = {**record, "fields": {**record["fields"], "source": source}}
    (directory / "run.json").write_text(json.dumps(earlier))
    assert RunConfig.load(str(directory)).source.packed_weights is True



def test_a_run_saved_before_its_surface_named_its_settings_loads_as_the_run_it_was(tmp_path):
    """A run saved while its optimizer held its rate as `learning_rate` (here a schedule's record), its
    sparing of norms and biases as `spared` (a group's too) and its clip as `clip_grads`, and its trainer
    held its length, its update and its fit in records of their own (`horizon`, `update`, `fit`), its kept
    checkpoints as a count, its ranking as a metric's name and its numerics as a preset's name, loads as
    the run it was (`dew.config.earlier.earlier_surface`)."""
    from dew.config.lm import LMRunConfig
    from dew.nn.precision import numerics_of
    from dew.training import Best, Keep

    expected = LMRunConfig(
        optim=OptimConfig(optimizer=AdamW(weight_decay=0.1), clip=1.0, decay_norms_and_biases=True,
                          schedule=Cosine(peak_value=3e-4, warmup_steps=10),
                          param_groups=(ParamGroup("head", ("head/*",), decay_norms_and_biases=False),
                                        ParamGroup("rest", ("*",)))),
        trainer=TrainerConfig(steps=100, max_tokens=10_000, accumulation=4, split=2, remat="full",
                              keep=Keep(latest=3), best=Best("val/loss"), precision=numerics_of("bf16-true")))
    # Read back as JSON, which the edits below rewrite as an earlier Dew wrote it.
    record = json.loads(json.dumps(expected.record()))

    def own(held):
        return held["fields"] if "class" in held and "fields" in held else held

    optim, trainer = own(record["fields"]["optim"]), own(record["fields"]["trainer"])
    optim.pop("lr", None)
    optim["learning_rate"] = optim.pop("schedule")
    optim["spared"] = not optim.pop("decay_norms_and_biases", False)
    optim["clip_grads"] = optim.pop("clip")
    for group in optim["param_groups"]:
        group["spared"] = not group.pop("decay_norms_and_biases", True)
    renamed = {"horizon": {"steps": "steps", "max_tokens": "max_tokens"},
               "update": {"accumulation": "batches"}, "fit": {"split": "split", "remat": "remat"}}
    for nested, names in renamed.items():
        trainer[nested] = {earlier: trainer.pop(today) for today, earlier in names.items()}
    trainer.update(keep=3, best="val/loss", precision="bf16-true")
    path = tmp_path / "run.json"
    path.write_text(json.dumps(record))
    assert LMRunConfig.load(str(path)) == expected
