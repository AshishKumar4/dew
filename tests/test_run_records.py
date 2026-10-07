"""A run config's record reads back as the config that wrote it.

`to_dict` writes every recorded field, `from_dict` builds the same value
back, a field the record lacks takes its declared default, and a field the
class does not declare is refused.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from dew.config import ObjectiveConfig, RunConfig, TrainerConfig
from dew.decision.config import DecisionRunConfig
from dew.objectives.diffusion import DiffusionRunConfig
from dew.objectives.jepa import JepaRunConfig
from dew.objectives.lm.config import LMRunConfig
from dew.training.quantization import Quantization

ROOT = Path(__file__).resolve().parents[1]


def written(config) -> dict:
    """The record `save` writes, as the JSON file holds it."""
    return json.loads(json.dumps(config.to_dict()))


@pytest.mark.parametrize("config", ["run", "diffusion", "lm", "jepa", "decision"])
def test_a_run_config_reads_back_as_its_class_from_its_own_record(config):
    """`run.json` names the run's class, so a reader that knows only
    `RunConfig` rebuilds a run of any kind as the kind it is."""
    # A JEPA run validates through probes it has no classes for by default.
    value = {"run": RunConfig(), "diffusion": DiffusionRunConfig(), "lm": LMRunConfig(),
             "jepa": JepaRunConfig(trainer=TrainerConfig(eval_every=None)),
             "decision": DecisionRunConfig()}[config]
    assert RunConfig.read(json.loads(json.dumps(value.record()))) == value


def test_a_record_naming_a_class_that_is_no_run_is_refused():
    with pytest.raises(ValueError, match="which is no DiffusionRunConfig"):
        DiffusionRunConfig.read(LMRunConfig().record())
    with pytest.raises(ValueError, match="is its class and its fields"):
        RunConfig.read(written(RunConfig()))


def test_a_record_keeps_what_it_states_and_defaults_what_it_lacks():
    run = DiffusionRunConfig()
    quantized = dataclasses.replace(run, trainer=dataclasses.replace(
        run.trainer, quantization=Quantization(dtype="fp8", patterns=(".*mlp.*",), weight_only=True)))
    assert DiffusionRunConfig.from_dict(written(quantized)) == quantized
    stated = LMRunConfig(objective=ObjectiveConfig("lm", {"ema_decay": 0.99}))
    assert LMRunConfig.from_dict({"objective": {"name": "lm", "fields": {"ema_decay": 0.99}}}) == stated
    assert LMRunConfig.from_dict({}).objective == ObjectiveConfig("lm")


def test_an_unknown_field_is_refused():
    record = written(DiffusionRunConfig())
    with pytest.raises(ValueError, match=r"unknown fields \['epochs'\]"):
        DiffusionRunConfig.from_dict({**record, "epochs": 3})
    with pytest.raises(ValueError, match=r"unknown fields \['warp'\]"):
        DiffusionRunConfig.from_dict({**record, "preset": {"class": "edm", "fields": {"warp": 1.0}}})
    with pytest.raises(ValueError, match=r"unknown fields \['seed'\]"):
        RunConfig.from_dict({"trainer": {"seed": 23}})


PUBLISHED = ROOT / "tests" / "fixtures" / "runs" / "hybrid-dit-176m" / "run.json"
LIVE_PIN = ROOT / "site" / "live" / "container" / "text-to-image"


def test_the_published_run_reads_back_as_it_was_written():
    """dewml/hybrid-dit-176m's run.json, as the revision the live image pins
    publishes it, is a current record: it loads and writes back unchanged.
    A record change that refuses it or rewrites it fails here, and the fix
    is to re-export the published run in the same change, since the site,
    the quick start and the live sampler all load it."""
    held = json.loads(PUBLISHED.read_text())
    assert json.loads(json.dumps(RunConfig.read(held).record())) == held


@pytest.mark.network
def test_the_fixture_is_the_record_the_live_image_pins():
    from huggingface_hub import hf_hub_download

    repo, revision = LIVE_PIN.read_text().strip().split("@")
    published = Path(hf_hub_download(repo, "run.json", revision=revision))
    assert json.loads(published.read_text()) == json.loads(PUBLISHED.read_text())
