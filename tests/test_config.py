"""RunConfig: the typed record of a run, its round trip, and what it builds."""

import dataclasses

import jax.numpy as jnp
import pytest
import tyro

import dew.nn.backbones
from dew.config import ModelConfig, OptimConfig, RunConfig, TrainerConfig
from dew.registry import datasets, models
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
