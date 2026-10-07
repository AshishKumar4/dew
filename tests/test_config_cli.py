"""The recipes' command lines: the registries as subcommands, and a run that
rebuilds from what it logged."""

import dataclasses
import importlib.util
import itertools
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import numpy as np
import pytest
from diffusion_stubs import RES, STUB_TEXT

from dew.config import RunConfig, ScheduleSpec
from dew.data import Dataset, OnlineImages, PackedTokens, TFDSImages
from dew.data.dataset import record_argument, tokenized
from dew.diffusion.presets import Flow
from dew.objectives.diffusion import DiffusionObjective
from dew.registry import datasets, import_path
from dew.sampling import Heun
from dew.training import MeshSpec

pytestmark = pytest.mark.mesh

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_recipe(name):
    """A recipe file, loaded once as `recipe_<name>`: a recipe registers its
    corpora on import, and a name maps to one class."""
    module = sys.modules.get(f"recipe_{name}")
    if module is None:
        path = REPO_ROOT / "recipes" / name / "train.py"
        spec = importlib.util.spec_from_file_location(f"recipe_{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


def parse(cls, args):
    return cls.cli(args)


def test_the_flags_pick_a_dataset_a_preset_and_a_solver_from_the_registries():
    recipe = load_recipe("diffusion")
    config = parse(recipe.DiffusionRunConfig, [
        "--data.image-size", "64", "--data.augmentation", "flip_only",
        "preset:flow", "--preset.shift", "3.0", "solver:heun",
        "--trainer.batch-size", "8", "--trainer.steps", "10", "--trainer.mesh.fsdp", "2",
        "--model", "simple_dit", "--model.scan_order", "hilbert"])

    assert config.data == TFDSImages(image_size=64, augmentation="flip_only")
    assert config.preset == Flow(shift=3.0) and config.solver == Heun()
    assert config.trainer.batch_size == 8 and config.trainer.mesh == MeshSpec(fsdp=2)
    assert config.model.fields["scan_order"] == "hilbert"


def test_the_default_dataset_takes_flags_without_naming_its_subcommand():
    config = parse(RunConfig, ["--data.image-size", "96", "--trainer.epochs", "3"])
    assert config.data == TFDSImages(image_size=96) and config.trainer.epochs == 3


def test_another_dataset_is_its_subcommand():
    recipe = load_recipe("lm")
    config = parse(recipe.LmRunConfig, ["data:packed-tokens", "--data.path", "d",
                                        "--data.seq-len", "64", "--data.packing-bins", "2"])
    assert config.data == PackedTokens(path="d", seq_len=64, packing_bins=2)
    with pytest.raises(ValueError, match="token-windows or data:packed-tokens"):
        recipe.LmRunConfig(data=TFDSImages())


def test_a_spec_field_the_dataset_lacks_is_a_command_line_error():
    with pytest.raises(SystemExit):
        parse(RunConfig, ["--data.seq-len", "8"])


@pytest.mark.parametrize("name", ["diffusion", "lm", "jepa"])
def test_a_recipe_config_round_trips_through_its_json_record(name):
    recipe = load_recipe(name)
    cls = {"diffusion": "DiffusionRunConfig", "lm": "LmRunConfig", "jepa": "JepaRunConfig"}[name]
    args = {"diffusion": ["preset:karras", "--preset.sigma-data", "0.6", "--guidance.scale", "2.5"],
            "lm": ["--data.path", "d", "--sample-tokens", "4", "--ema-decay", "0.9"],
            "jepa": ["--probe-classes", "7", "--momentum", "0.9", "0.99"]}[name]
    config = parse(getattr(recipe, cls), [*args, "--trainer.steps", "5"])

    record = json.loads(json.dumps(config.to_dict()))

    assert getattr(recipe, cls).from_dict(record) == config
    assert record["data"]["class"] == import_path(type(config.data))


def test_a_run_over_url_datasets_round_trips_through_its_json_record():
    """A diffusion run may train on any hub table of urls; its record has to
    name the sources, so the run and its solvers rebuild from run.json."""
    recipe = load_recipe("diffusion")
    config = recipe.DiffusionRunConfig(data=OnlineImages(sources=("user/urls",), image_size=64))

    record = json.loads(json.dumps(config.to_dict()))

    assert recipe.DiffusionRunConfig.from_dict(record) == config
    assert record["data"]["class"] == datasets.paths["online_images"]


def test_steps_and_epochs_are_one_choice():
    with pytest.raises(SystemExit):
        parse(RunConfig, ["--trainer.steps", "5", "--trainer.epochs", "1"])
    config = parse(RunConfig, ["--trainer.epochs", "2"])
    data = Dataset(train=lambda partition: iter(()), val=None, records=100, batch=10)
    assert config.trainer.total_steps(data) == 20
    with pytest.raises(ValueError, match=r"--trainer.steps"):
        config.trainer.total_steps(
            Dataset(train=lambda partition: iter(()), val=None, records=None, batch=10)
        )


class _Batches:
    """Endless captioned noise batches that report a position, like grain's.

    The captions leave as text, the way a dataset writes them; what turns
    them into tokens is the run's condition, through `tokenized`.
    """

    def __init__(self, batch):
        self.batch, self.count = batch, 0

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        rng = np.random.RandomState(self.count)
        return {"image": rng.randint(0, 256, (self.batch, RES, RES, 3), np.uint8),
                "caption": np.asarray(["a flower"] * self.batch)}

    def get_state(self):
        return json.dumps({"count": self.count}).encode()

    def set_state(self, state):
        self.count = json.loads(state)["count"]


def _batches(batch):
    return lambda partition: _Batches(batch)


def test_the_diffusion_entrypoint_runs_without_a_tracker_and_saves_its_run_spec(tmp_path, monkeypatch):
    recipe = load_recipe("diffusion")
    batch = 8
    def load(self, *, batch, tokenize=None):
        train = tokenized(_batches(batch), tokenize)
        return Dataset(train=train,
                       val=tokenized(lambda partition: itertools.islice(_batches(batch)(partition), 1),
                                     tokenize),
                       records=4 * batch, batch=batch)

    monkeypatch.setattr(TFDSImages, "load", load)
    config = parse(recipe.DiffusionRunConfig, [
        "--text.encoder", STUB_TEXT, "--text.checkpoint", "stub",
        "--data.image-size", str(RES), "--trainer.batch-size", str(batch), "--trainer.steps", "2",
        "--trainer.checkpoint-dir", str(tmp_path), "--trainer.name", "run",
        "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
        "--trainer.log-every", "1", "--model", "simple_dit", "--model.dtype", "float32",
        "--model.patch_size", "4", "--model.emb_features", "16",
        "--model.num_layers", "1", "--model.num_heads", "2",
        "--sampling-steps", "2"])
    # A validation pass needs a consumer; psnr scores samples against the
    # batch and downloads nothing, unlike the default clip metric.
    config = dataclasses.replace(config, val_metrics=("psnr",))

    state = recipe.main(config)

    assert int(state.step) == 2
    trained = dataclasses.replace(config, objective=import_path(DiffusionObjective))
    assert recipe.DiffusionRunConfig.load(str(tmp_path / "run")) == trained
    assert config.to_dict()["preset"] == {"class": "dew.diffusion.presets:EDM", "fields": {
        "sigma_min": 0.002, "sigma_max": 80.0, "rho": 7.0, "sigma_data": 0.5,
        "regime": "pixel", "P_mean": None, "P_std": None, "min_snr_gamma": None}}
    assert config.model_fields(None)["output_channels"] == 3
    assert (tmp_path / "run" / "2").is_dir()

def test_the_jepa_entrypoint_runs_without_a_tracker_and_saves_its_run_spec(tmp_path, monkeypatch):
    recipe = load_recipe("jepa")
    batch = 8
    size = 32

    class _Images:
        def __init__(self, count=None):
            self.count = 0
            self.remaining = count

        def __iter__(self):
            return self

        def __next__(self):
            if self.remaining is not None and self.remaining <= 0:
                raise StopIteration
            self.count += 1
            if self.remaining is not None:
                self.remaining -= 1
            rng = np.random.RandomState(self.count)
            return {
                "image": rng.randint(0, 256, (batch, size, size, 3)).astype(np.uint8),
                "label": np.zeros((batch,), np.int32),
            }

        def get_state(self):
            return json.dumps({"count": self.count}).encode()

        def set_state(self, state):
            self.count = json.loads(state)["count"]

    def load(self, *, batch, tokenize=None):
        return Dataset(train=lambda partition: _Images(), val=lambda partition: _Images(count=1),
                       records=4 * batch, batch=batch)

    monkeypatch.setattr(TFDSImages, "load", load)
    config = parse(
        recipe.JepaRunConfig,
        [
            "--data.image-size",
            str(size),
            "--trainer.batch-size",
            str(batch),
            "--trainer.steps",
            "2",
            "--trainer.checkpoint-dir",
            str(tmp_path),
            "--trainer.name",
            "run",
            "--trainer.compilation-cache-dir",
            "None",
            "--trainer.multi-host",
            "False",
            "--trainer.log-every",
            "1",
            "--trainer.eval-every",
            "None",  # no probes, so no validation pass
            "--model",
            "jepa_encoder",
            "--model.dtype",
            "float32",
            "--model.patch_size", "4", "--model.emb_features", "16",
            "--model.num_layers", "1", "--model.num_heads", "2", "--model.mlp_ratio", "2",
        ],
    )

    state = recipe.main(config)

    assert int(state.step) == 2
    assert recipe.JepaRunConfig.load(str(tmp_path / "run")) == config
    assert (tmp_path / "run" / "2").is_dir()


def test_a_jepa_run_without_probes_schedules_no_validation_pass():
    """A JEPA validation pass scores the frozen-encoder probes, so a run that
    names no probe classes yet schedules passes is refused where it is
    configured, not at its first pass."""
    recipe = load_recipe("jepa")
    with pytest.raises(ValueError, match="probe_classes"):
        recipe.JepaRunConfig()


def test_param_groups_on_the_command_line_read_their_schedules_as_records():
    """A group's schedule is the record that names it, as the run's record
    writes it, and the config's own schedule steps per epoch by its flag."""
    from dew.training.optim import Cosine, OneCycle, ParamGroup
    groups = [{"name": "delays", "patterns": ["*/delay"], "bounds": [0, 24],
               "schedule": {"class": "cosine", "fields": {"peak": 0.1, "warmup_steps": 0, "every": 40}}},
              {"name": "rest", "patterns": ["*"],
               "b1": {"class": "one_cycle", "fields": {"peak": 0.85, "init": 0.95, "end": 0.95}}}]
    config = parse(RunConfig, ["--optim.param-groups", json.dumps(groups), "optim.schedule:one-cycle",
                               "--optim.schedule.peak", "5e-3", "--optim.schedule.every", "40"])
    assert config.optim.schedule == OneCycle(peak=5e-3, every=40)
    assert config.optim.param_groups == (
        ParamGroup("delays", ("*/delay",), schedule=Cosine(peak=0.1, warmup_steps=0, every=40),
                   bounds=(0.0, 24.0)),
        ParamGroup("rest", ("*",), b1=OneCycle(peak=0.85, init=0.95, end=0.95)))


@dataclasses.dataclass(frozen=True)
class _Schedules(RunConfig):
    """A recipe with a named set of schedules, as sparx's surrogate and delay widths."""

    schedules: Annotated[Mapping[str, ScheduleSpec], record_argument(Mapping[str, ScheduleSpec])] = \
        dataclasses.field(default_factory=dict)


def test_a_mapping_of_class_records_reads_from_one_flag_and_round_trips_its_record():
    """#45: each schedule is the record that names it, read from one JSON
    object on the command line and written back by the run's record."""
    from dew.training.optim import Exponential, Linear

    given = {"sigma": {"class": "linear", "fields": {"peak": 0.5, "warmup_steps": 0, "end": 0.1}},
             "delay": {"class": "exponential", "fields": {"init": 10.0, "end": 1.0, "every": 4}}}
    config = parse(_Schedules, ["--schedules", json.dumps(given)])

    assert config.schedules == {"sigma": Linear(peak=0.5, warmup_steps=0, end=0.1),
                                "delay": Exponential(init=10.0, end=1.0, every=4)}
    assert _Schedules.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert parse(_Schedules, []).schedules == {}
