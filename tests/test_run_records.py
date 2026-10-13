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


def test_a_record_keeps_what_it_states_and_defaults_what_it_lacks():
    run = DiffusionRunConfig()
    quantized = dataclasses.replace(run, trainer=dataclasses.replace(
        run.trainer, quantization=Quantization(dtype="fp8", patterns=(".*mlp.*",), weight_only=True)))
    assert DiffusionRunConfig.from_dict(written(quantized)) == quantized
    stated = LMRunConfig(objective=ObjectiveConfig("lm", {"ema_decay": 0.99}))
    assert LMRunConfig.from_dict({"objective": {"name": "lm", "fields": {"ema_decay": 0.99}}}) == stated
    assert LMRunConfig.from_dict({}).objective == ObjectiveConfig("lm")


PUBLISHED = ROOT / "tests" / "fixtures" / "runs" / "hybrid-dit-176m" / "run.json"
LIVE_PIN = ROOT / "site" / "live" / "container" / "text-to-image"


def changed(written, held, at: str = "") -> list[str]:
    """Where `written` does not keep what `held` states: a key it lacks, a
    list of another length, or another value. What `held` does not state is
    no change."""
    if isinstance(held, dict) and isinstance(written, dict):
        return [place for key, value in held.items() for place in
                (changed(written[key], value, f"{at}/{key}") if key in written else [f"{at}/{key}"])]
    if isinstance(held, list) and isinstance(written, list) and len(written) == len(held):
        return [place for index, pair in enumerate(zip(written, held, strict=True))
                for place in changed(*pair, f"{at}/{index}")]
    return [] if written == held else [at]


@pytest.mark.parametrize("live", [False, pytest.param(True, marks=pytest.mark.network)])
def test_the_published_run_reads_back_as_it_was_written(live):
    """dewml/hybrid-dit-176m's run.json, as the revision the live image pins
    publishes it, is a current record: every field it states loads and writes
    back as it states it. A field added since is written beside them; a
    rename, a removal or a rewrite fails here, and the fix is to re-export the
    published run in the same change, since the site, the quick start and the
    live sampler all load it."""
    held = json.loads(PUBLISHED.read_text())
    if live:
        from huggingface_hub import hf_hub_download

        repo, revision = LIVE_PIN.read_text().strip().split("@")
        published = Path(hf_hub_download(repo, "run.json", revision=revision))
        assert json.loads(published.read_text()) == held
    assert changed(json.loads(json.dumps(RunConfig.read(held).record())), held) == []
    stated = held["fields"]["objective"]["fields"]
    built = RunConfig.read(held).objective.arguments
    assert (built["guidance"], built["steps"]) == (stated["guidance"], stated["steps"])
