"""The recipes' command lines: the registries as subcommands, and a run that
rebuilds from what it logged."""

import dataclasses
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tyro

from dew.config import ModelConfig, RunConfig
from dew.data import Dataset, OxfordFlowers, PackedTokens, TokenWindows
from dew.registry import datasets, encoders, presets, samplers
from dew.training import MeshSpec
from test_diffusion_objective import RES, TOKENS, StubText

# The manifest names the encoder through the registry.
encoders("stub_text")(StubText)

pytestmark = pytest.mark.mesh

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_recipe(name):
    path = REPO_ROOT / "recipes" / name / "train.py"
    spec = importlib.util.spec_from_file_location(f"recipe_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse(cls, args):
    return tyro.cli(tyro.conf.CascadeSubcommandArgs[cls], args=args)


def test_the_flags_pick_a_dataset_a_preset_and_a_sampler_from_the_registries():
    recipe = load_recipe("diffusion")
    config = parse(recipe.DiffusionRunConfig, [
        "--data.image-size", "64", "--data.augmentation", "flip_only",
        "preset:flow", "--preset.shift", "3.0", "sampler:heun",
        "--trainer.batch-size", "8", "--trainer.steps", "10", "--trainer.mesh.fsdp", "2",
        "--model.architecture", "simple_dit", "--model.config", '{"scan_order": "hilbert"}'])

    assert config.data == OxfordFlowers(image_size=64, augmentation="flip_only")
    assert config.preset == presets.Flow(shift=3.0) and config.sampler == samplers.Heun()
    assert config.trainer.batch_size == 8 and config.trainer.mesh == MeshSpec(fsdp=2)
    assert config.model.fields()["scan_order"] == "hilbert"


def test_the_default_dataset_takes_flags_without_naming_its_subcommand():
    config = parse(RunConfig, ["--data.image-size", "96", "--trainer.epochs", "3"])
    assert config.data == OxfordFlowers(image_size=96) and config.trainer.epochs == 3


def test_another_dataset_is_its_subcommand():
    recipe = load_recipe("lm")
    config = parse(recipe.LmRunConfig, ["data:packed-tokens", "--data.path", "d",
                                        "--data.seq-len", "64", "--data.packing-bins", "2"])
    assert config.data == PackedTokens(path="d", seq_len=64, packing_bins=2)
    with pytest.raises(ValueError, match="token-windows or data:packed-tokens"):
        recipe.LmRunConfig(data=OxfordFlowers())


def test_a_spec_field_the_dataset_lacks_is_a_command_line_error(capsys):
    with pytest.raises(SystemExit):
        parse(RunConfig, ["--data.seq-len", "8"])


@pytest.mark.parametrize("name", ["diffusion", "lm", "jepa"])
def test_a_recipe_config_round_trips_through_its_json_record(name):
    recipe = load_recipe(name)
    cls = {"diffusion": "DiffusionRunConfig", "lm": "LmRunConfig", "jepa": "JepaRunConfig"}[name]
    args = {"diffusion": ["preset:karras", "--preset.sigma-data", "0.6", "--guidance", "2.5"],
            "lm": ["--data.path", "d", "--sample-tokens", "4", "--ema-decay", "0.9"],
            "jepa": ["--probe-classes", "7", "--momentum", "0.9", "0.99"]}[name]
    config = parse(getattr(recipe, cls), [*args, "--trainer.steps", "5"])

    record = json.loads(json.dumps(config.to_dict()))

    assert getattr(recipe, cls).from_dict(record) == config
    assert record["data"]["name"] == datasets.name_of(type(config.data))


def test_steps_and_epochs_are_one_choice():
    with pytest.raises(SystemExit):
        parse(RunConfig, ["--trainer.steps", "5", "--trainer.epochs", "1"])
    config = parse(RunConfig, ["--trainer.epochs", "2"])
    data = Dataset(train=lambda: iter(()), val=None, records=100, batch=10)
    assert config.trainer.total_steps(data) == 20
    with pytest.raises(ValueError, match="--trainer.steps"):
        config.trainer.total_steps(Dataset(train=lambda: iter(()), val=None, records=None, batch=10))


class _Batches:
    """Endless captioned noise batches that report a position, like grain's."""

    def __init__(self, batch):
        self.batch, self.count = batch, 0

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        rng = np.random.RandomState(self.count)
        return {"image": rng.randint(0, 256, (self.batch, RES, RES, 3), np.uint8),
                "text": {"input_ids": np.ones((self.batch, TOKENS), np.int32),
                         "attention_mask": np.ones((self.batch, TOKENS), np.int32)}}

    def get_state(self):
        return json.dumps({"count": self.count}).encode()

    def set_state(self, state):
        self.count = json.loads(state)["count"]


def _batches(batch):
    return lambda: _Batches(batch)


def test_the_diffusion_entrypoint_runs_without_a_tracker_and_saves_its_run_spec(tmp_path, monkeypatch):
    recipe = load_recipe("diffusion")
    batch = 8
    data = Dataset(train=_batches(batch), val=lambda: itertools.islice(_batches(batch)(), 1),
                   records=4 * batch, batch=batch)
    monkeypatch.setattr(OxfordFlowers, "load", lambda self, *, batch: data)
    config = parse(recipe.DiffusionRunConfig, [
        "--text.encoder", "stub_text", "--text.checkpoint", "stub",
        "--data.image-size", str(RES), "--trainer.batch-size", str(batch), "--trainer.steps", "2",
        "--trainer.checkpoint-dir", str(tmp_path), "--trainer.name", "run",
        "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
        "--trainer.log-every", "1", "--model.architecture", "simple_dit", "--model.dtype", "float32",
        "--model.config", '{"patch_size": 4, "emb_features": 16, "num_layers": 1, "num_heads": 2}',
        "--sampling-steps", "2"])
    config = dataclasses.replace(config, val_metrics=[])

    state = recipe.main(config)

    assert int(state.step) == 2
    assert recipe.DiffusionRunConfig.load(str(tmp_path / "run")) == config
    assert config.to_dict()["preset"] == {"name": "edm", "fields": {
        "sigma_min": 0.002, "sigma_max": 80.0, "rho": 7.0, "sigma_data": 0.5,
        "P_mean": -0.4, "P_std": 1.0, "min_snr_gamma": None}}
    assert config.model_fields(None)["output_channels"] == 3
    assert (tmp_path / "run" / "2").is_dir()
