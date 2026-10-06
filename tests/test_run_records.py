"""A run config's record reads back as the config that wrote it.

`to_dict` writes every recorded field, `from_dict` builds the same value
back, a field the record lacks takes its declared default, and a field the
class does not declare is refused.
"""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from dew.config import RunConfig, TrainerConfig
from dew.objectives.diffusion import DiffusionRunConfig
from dew.objectives.lm.config import LMRunConfig
from dew.training.quantization import Quantization

ROOT = Path(__file__).resolve().parents[1]


def recipe_config(name: str, cls: str) -> type:
    """A run config a recipe file declares, loaded as `recipe_<name>`."""
    module = sys.modules.get(f"recipe_{name}")
    if module is None:
        spec = importlib.util.spec_from_file_location(f"recipe_{name}", ROOT / "recipes" / name / "train.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return getattr(module, cls)


def written(config) -> dict:
    """The record `save` writes, as the JSON file holds it."""
    return json.loads(json.dumps(config.to_dict()))


@pytest.mark.parametrize("config", ["run", "diffusion", "lm", "recipe_lm", "recipe_jepa"])
def test_a_run_config_reads_back_from_its_own_record(config):
    if config == "recipe_jepa":
        # A JEPA run validates through probes it has no classes for by default.
        value = recipe_config("jepa", "JepaRunConfig")(trainer=TrainerConfig(eval_every=None))
    else:
        value = {"run": RunConfig, "diffusion": DiffusionRunConfig, "lm": LMRunConfig,
                 "recipe_lm": recipe_config("lm", "LmRunConfig")}[config]()
    assert type(value).from_dict(written(value)) == value


def test_a_record_keeps_what_it_states_and_defaults_what_it_lacks():
    run = DiffusionRunConfig()
    quantized = dataclasses.replace(run, trainer=dataclasses.replace(
        run.trainer, quantization=Quantization(dtype="fp8", patterns=(".*mlp.*",), weight_only=True)))
    assert DiffusionRunConfig.from_dict(written(quantized)) == quantized
    assert LMRunConfig.from_dict({"ema_decay": 0.99}) == LMRunConfig(ema_decay=0.99)
    assert LMRunConfig.from_dict({}).ema_decay is None


def test_an_unknown_field_is_refused():
    record = written(DiffusionRunConfig())
    with pytest.raises(ValueError, match=r"unknown fields \['epochs'\]"):
        DiffusionRunConfig.from_dict({**record, "epochs": 3})
    with pytest.raises(ValueError, match=r"unknown fields \['warp'\]"):
        DiffusionRunConfig.from_dict({**record, "preset": {"name": "edm", "fields": {"warp": 1.0}}})
    with pytest.raises(ValueError, match=r"unknown fields \['seed'\]"):
        RunConfig.from_dict({"trainer": {"seed": 23}})


PUBLISHED = ROOT / "tests" / "fixtures" / "runs" / "hybrid-dit-176m" / "run.json"
LIVE_PIN = ROOT / "site" / "live" / "container" / "text-to-image"


def test_the_published_run_reads_back_as_it_was_written():
    """dewml/hybrid-dit-176m's run.json, as the revision the live image pins
    publishes it, is a current record: it loads and writes back unchanged.
    A record change that refuses it or rewrites it fails here, and the fix
    is to re-export the published run in the same change, since the site,
    the quick start and the live sampler all load it. A field added since the
    export writes its default, and is listed here."""
    held = json.loads(PUBLISHED.read_text())
    added = {"optim": {**held["optim"], "b1": None, "b2": None}}
    assert json.loads(json.dumps(DiffusionRunConfig.from_dict(held).to_dict())) == {**held, **added}


@pytest.mark.network
def test_the_fixture_is_the_record_the_live_image_pins():
    from huggingface_hub import hf_hub_download

    repo, revision = LIVE_PIN.read_text().strip().split("@")
    published = Path(hf_hub_download(repo, "run.json", revision=revision))
    assert json.loads(published.read_text()) == json.loads(PUBLISHED.read_text())
