"""A run config's record reads back as the config that wrote it.

`to_dict` writes every recorded field, `from_dict` builds the same value
back, a field the record lacks takes its declared default, and a field the
class does not declare is refused.
"""

import dataclasses
import inspect
import json
from pathlib import Path

import pytest

from dew.config import ModelConfig, ObjectiveConfig, RunConfig, TrainerConfig
from dew.decision.config import DecisionRunConfig
from dew.diffusion.discrete import MDLM
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.diffusion import DiffusionRunConfig, MaskedDiffusionObjective
from dew.objectives.jepa import JepaRunConfig
from dew.objectives.lm.config import LMRunConfig
from dew.registry import from_record, models, objectives, to_record
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
    with pytest.raises(ValueError, match=r"takes no \['warp'\], which the record defaults"):
        LMRunConfig.from_dict({"objective": {"name": "lm", "defaults": {"warp": 1.0}}})


@pytest.mark.parametrize("kind, name", [*((ModelConfig, name) for name in models),
                                        *((ObjectiveConfig, name) for name in objectives)],
                         ids=lambda value: value if isinstance(value, str) else value.__name__)
def test_a_config_records_its_class_defaults_and_builds_with_them_as_the_class_does(kind, name):
    """Every model and objective class records its defaults as JSON that reads
    back as written, and a config that has not outlived them builds with
    nothing beyond what it states: the class supplies its own defaults."""
    config = kind(name)
    read = from_record(kind, json.loads(json.dumps(to_record(config, kind))), dtypes=False)
    assert read == config
    assert read.arguments == read.fields == {}


def released(monkeypatch, member: type, name: str, value) -> None:
    """Default `member`'s `name` to `value` from here on, as a later release
    of the class would: in its constructor (the one a Flax module's wraps),
    and in the dataclass field a record reads a default from."""
    init = inspect.unwrap(member.__init__)
    parameters = inspect.signature(init).parameters
    if parameters[name].kind is inspect.Parameter.KEYWORD_ONLY:
        monkeypatch.setitem(init.__kwdefaults__, name, value)
    else:
        defaulted = [held for held, parameter in parameters.items()
                     if parameter.kind is parameter.POSITIONAL_OR_KEYWORD
                     and parameter.default is not parameter.empty]
        defaults = list(init.__defaults__)
        defaults[defaulted.index(name)] = value
        monkeypatch.setattr(init, "__defaults__", tuple(defaults))
    if dataclasses.is_dataclass(member):
        monkeypatch.setattr(member.__dataclass_fields__[name], "default", value)


def test_a_record_builds_the_defaults_it_was_written_with(monkeypatch):
    """A run records every argument its objective and its model default, so a
    later release that defaults them otherwise leaves `dew train run.json`
    and a resume building what the run trained: MDLM's sampling steps and
    the model's depth change here, the record still builds the old ones, and
    a new run takes the new ones."""
    def run():
        return LMRunConfig(objective=ObjectiveConfig("masked_diffusion"), model=ModelConfig(
            "causal_transformer", {"emb_features": 16, "num_heads": 2, "mlp_features": 32, "causal": False,
                                   "mask_token_id": 0, "qk_norm": False, "dtype": "float32"}))

    def built(config):
        model = config.model.build(vocab_size=16)
        objective = config.objective.build(model=model, process=MDLM(mask_id=0)(), seq_len=8)
        return model.num_layers, objective.steps

    record, (depth, steps) = json.loads(json.dumps(run().record())), built(run())
    released(monkeypatch, MaskedDiffusionObjective, "steps", steps + 1)
    released(monkeypatch, CausalTransformer, "num_layers", depth + 1)
    assert built(RunConfig.read(record)) == (depth, steps)
    assert built(run()) == (depth + 1, steps + 1)


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


def test_the_published_run_reads_back_as_it_was_written():
    """dewml/hybrid-dit-176m's run.json, as the revision the live image pins
    publishes it, is a current record: every field it states loads and writes
    back as it states it. A field added since is written beside them; a
    rename, a removal or a rewrite fails here, and the fix is to re-export the
    published run in the same change, since the site, the quick start and the
    live sampler all load it."""
    held = json.loads(PUBLISHED.read_text())
    assert changed(json.loads(json.dumps(RunConfig.read(held).record())), held) == []


@pytest.mark.network
def test_the_fixture_is_the_record_the_live_image_pins():
    from huggingface_hub import hf_hub_download

    repo, revision = LIVE_PIN.read_text().strip().split("@")
    published = Path(hf_hub_download(repo, "run.json", revision=revision))
    assert json.loads(published.read_text()) == json.loads(PUBLISHED.read_text())
