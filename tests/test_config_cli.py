"""The run classes' command lines: the registries as subcommands, and a run
that rebuilds from what it logged."""

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import pytest

from dew.config import RunConfig, ScheduleSpec
from dew.config.data import Augment, ImageData, TextData
from dew.data import Files
from dew.data.dataset import Loader, record_argument
from dew.diffusion.presets import Flow
from dew.objectives.diffusion import DiffusionRunConfig
from dew.objectives.jepa import JepaRunConfig
from dew.objectives.lm import LMRunConfig
from dew.registry import datasets, import_path
from dew.sampling import Heun
from dew.training import MeshSpec

pytestmark = pytest.mark.mesh

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse(cls, args):
    return cls.cli(args)


def test_the_flags_pick_a_dataset_a_preset_and_a_solver_from_the_registries():
    config = parse(DiffusionRunConfig, [
        "--data.image-size", "64", "--data.augment.crop-scale", "0.8", "1.0",
        "preset:flow", "--preset.shift", "3.0", "--objective.solver", '{"class": "heun"}',
        "--objective.guidance", '{"class": "dew.sampling.guidance:CFG", "fields": {"scale": 2.5}}',
        "--trainer.batch-size", "8", "--trainer.steps", "10", "--trainer.mesh.fsdp", "2",
        "--model", "simple_dit", "--model.scan_order", "hilbert"])

    assert config.data == dataclasses.replace(DiffusionRunConfig().data, image_size=64,
                                               augment=Augment(jitter=True, crop_scale=(0.8, 1.0)))
    assert config.preset == Flow(shift=3.0)
    assert config.objective.fields["solver"]["class"] == import_path(Heun)
    assert config.objective.fields["guidance"]["fields"]["scale"] == 2.5
    assert config.trainer.batch_size == 8 and config.trainer.mesh == MeshSpec(fsdp=2)
    assert config.model.fields["scan_order"] == "hilbert"


def test_another_dataset_is_its_subcommand():
    config = parse(LMRunConfig, ["data:text", "--data.source", "d", "--data.pack", "first-fit",
                                 "--data.seq-len", "64", "--data.packing-bins", "2"])
    assert config.data == TextData(source="d", seq_len=64, packing_bins=2, pack="first-fit")
    with pytest.raises(ValueError):
        LMRunConfig(data=ImageData(source="tfds://oxford_flowers102")).prepare()


def test_a_spec_field_the_dataset_lacks_is_a_command_line_error():
    with pytest.raises(SystemExit):
        parse(RunConfig, ["--data.seq-len", "8"])


@pytest.mark.parametrize("name", ["diffusion", "lm", "jepa"])
def test_a_run_config_round_trips_through_its_json_record(name):
    cls = {"diffusion": DiffusionRunConfig, "lm": LMRunConfig, "jepa": JepaRunConfig}[name]
    args = {"diffusion": ["preset:karras", "--preset.sigma-data", "0.6",
                          "--objective.unconditional-prob", "0.2"],
            "lm": ["--data.source", "d", "--max-new-tokens", "4", "--objective.ema-decay", "0.9"],
            "jepa": ["--probe-classes", "7", "--objective.momentum", "0.9", "0.99"]}[name]
    config = parse(cls, [*args, "--trainer.steps", "5"])

    record = json.loads(json.dumps(config.record()))

    assert RunConfig.read(record) == config
    assert record["fields"]["data"]["class"] == import_path(type(config.data))


def test_a_run_over_url_datasets_round_trips_through_its_json_record():
    """A diffusion run may train on any hub table of urls; its record has to
    name the sources, so the run and its solvers rebuild from run.json."""
    config = DiffusionRunConfig(data=ImageData(source=Files("hf://datasets/user/images/train.parquet",
        format="parquet"), image_size=64))

    record = json.loads(json.dumps(config.to_dict()))

    assert DiffusionRunConfig.from_dict(record) == config
    assert record["data"]["class"] == datasets.paths["images"]


def test_steps_and_epochs_are_one_choice():
    with pytest.raises(SystemExit):
        parse(RunConfig, ["--trainer.steps", "5", "--trainer.epochs", "1"])
    config = parse(RunConfig, ["--trainer.epochs", "2"])
    data = Loader(train=lambda partition: iter(()), val=None, records=100, batch=10)
    assert config.trainer.total_steps(data) == 20
    with pytest.raises(ValueError):
        config.trainer.total_steps(
            Loader(train=lambda partition: iter(()), val=None, records=None, batch=10)
        )


def test_the_cli_selects_a_token_budget_without_selecting_a_row_batch():
    config = parse(RunConfig, ["--trainer.batch-size", "None", "--trainer.batch-tokens", "24576",
                              "--trainer.update-tokens", "98304", "--trainer.steps", "1"])
    assert config.trainer.batch_size is None
    assert config.trainer.derived().accumulation == 4
    assert RunConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


def test_a_jepa_run_without_probes_schedules_no_validation_pass():
    """A JEPA validation pass scores the frozen-encoder probes, so a run that
    names no probe classes yet schedules passes is refused where it is
    configured, not at its first pass."""
    with pytest.raises(ValueError):
        JepaRunConfig()


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
