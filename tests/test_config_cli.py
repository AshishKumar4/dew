"""The run classes' command lines: the registries as subcommands, and a run
that rebuilds from what it logged."""

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import pytest
from diffusion_support import images_of

from dew.config import RunConfig, ScheduleSpec
from dew.data import Files
from dew.data.dataset import Loader, record_argument
from dew.diffusion.presets import Flow
from dew.objectives.diffusion import DiffusionRunConfig
from dew.objectives.jepa import JepaRunConfig
from dew.objectives.lm import LMRunConfig
from dew.registry import import_path
from dew.sampling import Heun
from dew.training import MeshSpec

pytestmark = pytest.mark.mesh

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse(cls, args):
    return cls.cli(args)


def test_the_flags_pick_a_preset_and_a_solver_and_set_sets_the_chains_settings():
    config = parse(DiffusionRunConfig, [
        "preset:flow", "--preset.shift", "3.0", "--objective.solver", '{"class": "heun"}',
        "--objective.guidance", '{"class": "dew.sampling.guidance:CFG", "fields": {"scale": 2.5}}',
        "--trainer.steps", "10", "--trainer.mesh.fsdp", "2",
        "--model", "simple_dit", "--model.scan_order", "hilbert"])

    assert config.preset == Flow(shift=3.0)
    assert config.objective.fields["solver"]["class"] == import_path(Heun)
    assert config.objective.fields["guidance"]["fields"]["scale"] == 2.5
    assert config.trainer.mesh == MeshSpec(fsdp=2)
    assert config.model.fields["scan_order"] == "hilbert"
    # The data's settings are its chain's calls', set by name (`dew.data.records.override`).
    chained = dataclasses.replace(config, data=images_of("tfds://oxford_flowers102", 32, batch=16))
    set_ = chained.assigned(["data.images.size=64", "data.images.batch=8"])
    assert set_.data.facts()["fields"]["image"] == [64, 64, 3] and set_.data.facts()["batch"] == 8


def test_an_lm_run_over_images_is_refused():
    with pytest.raises(ValueError):
        LMRunConfig(data=images_of("tfds://oxford_flowers102", 32)).prepare()


def test_a_spec_field_the_dataset_lacks_is_a_command_line_error():
    with pytest.raises(SystemExit):
        parse(RunConfig, ["--data.seq-len", "8"])


@pytest.mark.parametrize("name", ["diffusion", "lm", "jepa"])
def test_a_run_config_round_trips_through_its_json_record(name):
    from dew.data import presets
    from dew.data.sources.tokens import TokenFiles

    cls = {"diffusion": DiffusionRunConfig, "lm": LMRunConfig, "jepa": JepaRunConfig}[name]
    args = {"diffusion": ["preset:karras", "--preset.sigma-data", "0.6",
                          "--objective.unconditional-prob", "0.2"],
            "lm": ["--max-new-tokens", "4", "--objective.ema-decay", "0.9"],
            "jepa": ["--probe-classes", "7", "--objective.momentum", "0.9", "0.99"]}[name]
    data = (TokenFiles("d").pipe(presets.lm, None, seq_len=64, batch=8, pack=None) if name == "lm"
            else images_of("tfds://oxford_flowers102", 32))
    config = dataclasses.replace(parse(cls, [*args, "--trainer.steps", "5"]), data=data)

    record = json.loads(json.dumps(config.record()))

    assert RunConfig.read(record).record() == record
    preset = "lm" if name == "lm" else "images"
    assert record["fields"]["data"]["stages"][-1]["pipe"] == f"dew.data.presets:{preset}"


def test_a_run_over_url_datasets_round_trips_through_its_json_record():
    """A diffusion run may train on any hub table of urls; its record has to
    name the sources, so the run and its solvers rebuild from run.json."""
    from dew.data import presets

    config = DiffusionRunConfig(data=Files("hf://datasets/user/images/train.parquet", format="parquet").pipe(
        presets.images, 64, batch=32))

    record = json.loads(json.dumps(config.to_dict()))

    assert DiffusionRunConfig.from_dict(record).to_dict() == record
    assert record["data"]["source"]["fields"]["url"] == "hf://datasets/user/images/train.parquet"


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


def test_the_cli_sets_a_token_budget_the_datas_token_batches_divide():
    config = parse(RunConfig, ["--trainer.update-tokens", "98304", "--trainer.steps", "1"])
    assert config.trainer.update_tokens == 98304
    assert RunConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


def test_a_jepa_run_without_probes_trains_and_validates_its_loss(tmp_path):
    """A checkpointed JEPA recipe validates without class probes or a representation preview."""
    import numpy as np

    from dew.config import ModelConfig, TrainerConfig
    from dew.data import TFDS, Loading, presets

    images = TFDS("dew_images", path=str(REPO_ROOT / "tests" / "fixtures" / "tfds" / "dew_images" / "1.0.0"),
                  split="all")
    train, held = images.holdout(records=4)

    def prepared(chain):
        return chain.pipe(presets.images, 16, batch=8).read("memory").loading(Loading(workers=0))

    run = JepaRunConfig(
        model=ModelConfig("jepa_encoder", {"patch_size": 2, "emb_features": 16, "num_layers": 1,
                                           "num_heads": 2, "mlp_ratio": 2, "dtype": "float32",
                                           "attention_impl": "xla"}),
        predictor={"num_layers": 1, "predictor_features": 8}, num_target_blocks=2,
        data=prepared(train), validation=prepared(held),
        trainer=TrainerConfig(checkpoint_dir=str(tmp_path), steps=1, log_every=1,
                              eval_every=1, checkpoint_every=None))
    trained = run.prepare().train("jepa-loss")
    assert int(trained.updates) == 1
    reports = [json.loads(line)["scalars"] for line in
               (tmp_path / "jepa-loss" / "tracking" / "scalars.jsonl").read_text().splitlines()]
    assert any("val/loss" in report and np.isfinite(report["val/loss"]) for report in reports)
    assert not any("probe" in name for report in reports for name in report)


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
