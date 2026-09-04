"""RunConfig: the typed record of a run, its round trip, and what it builds."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional
import dataclasses

import jax.numpy as jnp
import pytest
import tyro

import dew.config
import dew.nn.backbones
from dew.config import ModelConfig, OptimConfig, RunConfig, TrainerConfig
from dew.data import Dataset
from dew.registry import Registry, datasets, models
from dew.training import Layout, MeshSpec


def test_to_dict_and_from_dict_round_trip_a_run():
    config = RunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 4, "emb_features": 64}),
        data=datasets["oxford_flowers102"](image_size=64),
        optim=OptimConfig(optimizer="muon", learning_rate=1e-3, weight_decay=0.1),
        trainer=TrainerConfig(name="run", steps=10, mesh=MeshSpec(fsdp=2),
                              layout=Layout(rules={"mlp": "fsdp"}, min_shard=8)),
    )
    record = config.to_dict()
    assert record["data"] == {"name": "oxford_flowers102",
                              "fields": dataclasses.asdict(config.data)}
    assert record["trainer"]["layout"]["rules"] == [["mlp", "fsdp"]]
    assert RunConfig.from_dict(record) == config


def test_a_record_with_an_unknown_or_a_missing_field_is_refused():
    record = RunConfig().to_dict()
    with pytest.raises(ValueError, match="unknown fields \\['epochs_per_eval'\\]"):
        RunConfig.from_dict({**record, "trainer": {**record["trainer"], "epochs_per_eval": 1}})
    trainer = dict(record["trainer"])
    del trainer["steps"]
    with pytest.raises(ValueError, match="missing fields \\['steps'\\]"):
        RunConfig.from_dict({**record, "trainer": trainer})
    with pytest.raises(KeyError, match="no dataset named 'flowers'"):
        RunConfig.from_dict({**record, "data": {"name": "flowers", "fields": {}}})


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


def test_the_model_config_builds_with_the_run_precision():
    config = ModelConfig("causal_transformer", {"vocab_size": 64, "emb_features": 32},
                         dtype="bfloat16", attention_impl="xla")
    model = config.build()
    assert isinstance(model, models["causal_transformer"])
    assert model.dtype is jnp.bfloat16 and model.attention_impl == "xla"
    assert config.fields()["dtype"] == "bfloat16"


def test_a_model_config_that_names_the_precision_twice_is_refused():
    with pytest.raises(ValueError, match="--model.dtype"):
        ModelConfig("simple_dit", {"dtype": "float32"}).fields()


@pytest.mark.parametrize("architecture", sorted(set(models) - {"jepa_predictor"}))
def test_every_architecture_builds_from_its_config(architecture):
    small = {"unet": {"emb_features": 32, "feature_depths": [16, 32], "num_res_blocks": 1,
                      "attention_configs": [None, None]},
             "unet_3d": {"emb_features": 32, "feature_depths": [16, 32], "num_res_blocks": 1,
                         "attention_configs": [None, None]},
             "causal_transformer": {"vocab_size": 64, "emb_features": 32, "num_layers": 1,
                                    "num_heads": 2}}
    fields = small.get(architecture, {"emb_features": 32, "num_layers": 1, "num_heads": 2}
                       if architecture != "hierarchical_mmdit" else {})
    model = ModelConfig(architecture, fields, dtype="float32").build()
    assert isinstance(model, models[architecture])
    assert model.dtype is jnp.float32


def test_an_unknown_model_field_is_refused():
    with pytest.raises(ValueError, match="no field for \\['depth'\\]"):
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
    with pytest.raises(ValueError, match="--trainer.steps or --trainer.epochs"):
        TrainerConfig().total_steps(Sized())


def test_the_cli_parses_the_mesh_the_layout_and_a_dataset_subcommand():
    config = tyro.cli(tyro.conf.CascadeSubcommandArgs[RunConfig], args=[
        "--trainer.mesh.fsdp", "2", "--trainer.layout.min-shard", "8",
        "--trainer.steps", "5", "--model.architecture", "uvit",
        "--model.config", '{"emb_features": 32}',
        "data:token-windows", "--data.path", "tokens", "--data.seq-len", "8"])
    assert config.trainer.mesh == MeshSpec(fsdp=2)
    assert config.trainer.layout.min_shard == 8 and config.trainer.layout.tolerance == 0.02
    assert config.model.config == {"emb_features": 32}
    assert type(config.data) is datasets["token_windows"]
    assert config.data.path == "tokens" and config.data.seq_len == 8
    assert RunConfig.from_dict(config.to_dict()) == config


def test_a_fresh_process_resolves_models_and_datasets_through_the_config():
    """The lm and jepa recipes run in a process that imports nothing else
    first, and both resolve a model and a dataset by name through this
    module, so importing it has to be enough to fill those registries."""
    root = Path(__file__).resolve().parents[1]
    code = ("from dew.config import ModelConfig;"
            "from dew.registry import datasets, models;"
            "print('causal_transformer' in models, 'token_windows' in datasets,"
            " ModelConfig(architecture='causal_transformer').fields()['dtype'])")
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={"PYTHONPATH": str(root / "src"), "JAX_PLATFORMS": "cpu",
             "PATH": os.environ.get("PATH", "")})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "True True bfloat16"


# --------------------------------------------------------------------------
# A record builds a value
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Kind:
    window: Optional[int] = None
    rope_theta: float = 10_000.0


@dataclasses.dataclass(frozen=True)
class Shape:
    width: int = 8
    mix: Optional[Kind] = None
    kinds: Mapping[str, Kind] = dataclasses.field(default_factory=dict)
    stages: tuple = ()
    dtype: Any = None


def shapes() -> Registry:
    registry = Registry("shape")
    registry("shape")(Shape)
    return registry


def test_a_record_builds_the_value_its_field_declares():
    """A model config is a dict on the command line and in run.json, so a
    field whose type is a value takes that dict, and code that already holds
    the value passes it through."""
    built = shapes().build("shape", mix={"rope_theta": 1e6},
                           kinds={"sliding": {"window": 512}})
    assert built.mix == Kind(rope_theta=1e6)
    assert built.kinds == {"sliding": Kind(window=512)}
    assert shapes().build("shape", mix=Kind(window=1)).mix == Kind(window=1)


def test_a_dtype_is_a_dtype_wherever_it_is_named():
    """The rule is the field's name, at the top level and inside a record the
    unets keep their per-stage attention settings in."""
    built = shapes().build("shape", dtype="bfloat16", stages=({"dtype": "float32"},))
    assert built.dtype is jnp.bfloat16
    assert built.stages[0]["dtype"] is jnp.float32


def test_a_record_that_names_a_field_the_value_does_not_have_is_refused():
    with pytest.raises(ValueError, match=r"Kind has no field for \['theta'\]"):
        shapes().build("shape", mix={"theta": 1e6})


def _data(records=48, batch=8, resumable=True):
    """A Dataset value with nothing behind it: the intervals only read its
    record count, its batch and whether it can report a position."""
    return Dataset(train=lambda: iter(()), val=None, records=records, batch=batch,
                   resumable=resumable)


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


def test_a_pass_over_the_data_needs_a_record_count():
    """A stream with no record count has no epoch, so "epoch" is refused by
    name instead of silently becoming every step or never."""
    streaming = _data(records=None)

    with pytest.raises(ValueError, match="checkpoint-every epoch needs a dataset"):
        TrainerConfig().checkpoint_interval(streaming)
    with pytest.raises(ValueError, match="eval-every epoch needs a dataset"):
        TrainerConfig().eval_interval(streaming)
    assert TrainerConfig(checkpoint_every=None).checkpoint_interval(streaming) is None
    assert TrainerConfig(checkpoint_every=5).checkpoint_interval(streaming) == 5


def test_a_dataset_that_cannot_report_its_position_takes_no_checkpoints():
    """The combination that used to be unreachable: an online stream keeps its
    record count, so "epoch" resolves, and the checkpoint would then carry no
    position. The refusal names the flag that makes the run possible."""
    stream = _data(records=48, resumable=False)

    with pytest.raises(ValueError, match=r"--trainer.checkpoint-every None"):
        TrainerConfig().checkpoint_interval(stream)
    with pytest.raises(ValueError, match="cannot report its read position"):
        TrainerConfig(checkpoint_every=10).checkpoint_interval(stream)

    assert TrainerConfig(checkpoint_every=None).checkpoint_interval(stream) is None
    assert TrainerConfig(checkpoint_every=None).eval_interval(stream) == 6, (
        "validation does not depend on a position")


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
    """A gs:// checkpoint directory has no local form: os.makedirs used to make
    a directory called `gs:` beside the run and the open then failed, after the
    training had already succeeded."""
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
