"""Saved run records load as the code that wrote them meant them.

A field a record lacks takes its declared default, so every default says
what runs recorded before the field existed did. `fixtures/record_defaults.json`
holds every recorded config class's fields and defaults; a field that
disappears or a default that changes fails here, and the fix is a new field
(or a rename the loader reads), never a silent default change.

The run fixtures are dewml/hybrid-dit-176m's run.json as published at two
commits: 3f11a17 predates EDM's `regime` and dfa94d6 carries it; both
predate `audio`.
"""

import dataclasses
import importlib.util
import json
import sys
import typing
from pathlib import Path

import pytest
from flax import linen as nn

from dew import registry
from dew.config import RunConfig, TrainerConfig, _FIELD_RENAMES, _recorded, _registry_for, _to_json
from dew.objectives.diffusion import DiffusionRunConfig, PretrainedAutoencoder
from dew.objectives.lm.config import LMRunConfig
from dew.training.quantization import Quantization

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "tests" / "fixtures" / "runs"
DEFAULTS = ROOT / "tests" / "fixtures" / "record_defaults.json"


def record(name: str) -> dict:
    held = json.loads((RUNS / name / "run.json").read_text())
    for old, new in _FIELD_RENAMES.get(TrainerConfig, {}).items():
        if old in held.get("trainer", {}):
            held["trainer"][new] = held["trainer"].pop(old)
    return held


def test_old_seed_record_reads_as_the_same_integer_root_key():
    loaded = RunConfig.from_dict({"trainer": {"seed": 23}})
    assert loaded.trainer.key == 23
    assert loaded.to_dict()["trainer"]["key"] == 23
    assert "seed" not in loaded.to_dict()["trainer"]
    with pytest.raises(ValueError, match="both seed and key"):
        RunConfig.from_dict({"trainer": {"seed": 23, "key": 23}})


def recipe_config(name: str, cls: str) -> type:
    """A run config a recipe file declares, loaded as `recipe_<name>`."""
    module = sys.modules.get(f"recipe_{name}")
    if module is None:
        spec = importlib.util.spec_from_file_location(f"recipe_{name}", ROOT / "recipes" / name / "train.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return getattr(module, cls)


def recorded_classes(annotation: registry.Annotation) -> list[type]:
    """The dataclasses a field of this annotation records: every member of a
    registry it names, the class itself, or those inside a union or a
    container."""
    annotation = registry.resolve_alias(annotation)
    held = _registry_for(annotation)
    if held is not None:
        return [member for member in held.values() if isinstance(member, type) and dataclasses.is_dataclass(member)]
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return [annotation]
    return [cls for argument in typing.get_args(annotation) for cls in recorded_classes(argument)]


def name_of(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def default_of(field: dataclasses.Field):
    """The field's default value, or MISSING for a required field."""
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return field.default


def described(value, annotation):
    """A default as the snapshot holds it. A dataclass is its class and the
    fields it sets away from that class's own defaults, which the class's
    own entry holds; anything else is what the record writes. A model's
    dtype or activation, which no record writes, is named by its qualified
    name, inside a sequence too. Such a name can sit under a library's
    private path (`jax._src...`), which a release may move without the
    default changing; the snapshot then fails, and the fix is the new name."""
    if not (dataclasses.is_dataclass(value) and not isinstance(value, type)):
        try:
            return json.loads(json.dumps(_to_json(value, annotation)))
        except TypeError:
            if isinstance(value, (list, tuple)):
                return [described(entry, kind) for entry, kind in
                        zip(value, registry.entry_types(annotation, len(value)), strict=True)]
            return {"value": f"{value.__module__}.{value.__qualname__}"}
    cls = type(value)
    return {"class": name_of(cls), "sets": {
        field.name: described(getattr(value, field.name), registry._declared_type(cls, field.name))
        for field in dataclasses.fields(cls)
        if _recorded(field) and getattr(value, field.name) != default_of(field)}}


def snapshot_default(field: dataclasses.Field, annotation):
    """What a record lacking `field` reads as. A factory other than a class
    or a lambda computes its default where it runs (the compilation cache is
    under the user's home), so the snapshot names the factory."""
    factory = field.default_factory
    if factory is not dataclasses.MISSING and not isinstance(factory, type) and factory.__name__ != "<lambda>":
        value = factory()
        if not dataclasses.is_dataclass(value):
            return {"factory": name_of(factory)}
    default = default_of(field)
    return "required" if default is dataclasses.MISSING else described(default, annotation)


def recorded_defaults() -> dict[str, dict[str, object]]:
    """Every config class a run record reaches, from each run config Dew and
    its recipes declare, and every registered model, tower and projector,
    whose fields a record holds only where the run set them, by module path:
    each recorded field's default as `snapshot_default` holds it. Flax's own
    `parent` and `name` are not a model's configuration."""
    import dew.nn.backbones  # noqa: F401  registers every model

    roots = [RunConfig, DiffusionRunConfig, LMRunConfig,
             recipe_config("lm", "LmRunConfig"), recipe_config("jepa", "JepaRunConfig"),
             *(model for model in registry.models.values() if isinstance(model, type)),
             *(tower for tower in registry.towers.values() if isinstance(tower, type)),
             *(projector for projector in registry.projectors.values() if isinstance(projector, type))]
    found: dict[str, dict[str, object]] = {}
    pending = list(roots)
    while pending:
        cls = pending.pop()
        if name_of(cls) in found:
            continue
        fields = {}
        flax_own = ("parent", "name") if issubclass(cls, nn.Module) else ()
        for field in dataclasses.fields(cls):
            if _recorded(field) and field.name not in flax_own:
                annotation = registry._declared_type(cls, field.name)
                pending.extend(recorded_classes(annotation))
                fields[field.name] = snapshot_default(field, annotation)
        found[name_of(cls)] = fields
    return dict(sorted(found.items()))


def test_every_recorded_default_is_the_one_older_runs_were_recorded_under():
    """A changed default would rebuild an older run as something it was not,
    and a removed field would refuse its records; a new field or class is
    added to the snapshot with the default that says what older runs did."""
    held = json.loads(DEFAULTS.read_text())
    today = recorded_defaults()
    gone = sorted(f"{cls}.{field}" for cls, fields in held.items() for field in fields
                  if field not in today.get(cls, {}))
    changed = sorted(f"{cls}.{field}: {fields[field]!r} -> {today[cls][field]!r}"
                     for cls, fields in held.items() for field in fields
                     if field in today.get(cls, {}) and today[cls][field] != fields[field])
    new = {cls: {field: default for field, default in fields.items() if field not in held.get(cls, {})}
           for cls, fields in today.items()}
    new = {cls: fields for cls, fields in new.items() if fields}
    assert not gone, f"recorded fields removed; keep them, or read the old name as a rename: {gone}"
    assert not changed, f"recorded defaults changed; add a new field instead: {changed}"
    assert not new, f"add these fields to {DEFAULTS.name}, with defaults meaning what older runs did:\n" + \
        json.dumps(new, indent=1, sort_keys=True)


def test_a_changed_default_fails_the_snapshot(monkeypatch):
    """The guard reads defaults off the classes themselves: moving one moves
    what the snapshot is compared against."""
    import dew.diffusion.presets as presets

    field = next(field for field in dataclasses.fields(presets.EDM) if field.name == "rho")
    monkeypatch.setattr(field, "default", 5.0)
    with pytest.raises(AssertionError, match=r"EDM.rho: 7.0 -> 5.0"):
        test_every_recorded_default_is_the_one_older_runs_were_recorded_under()


def test_a_changed_model_default_fails_the_snapshot(monkeypatch):
    """A run records only the model fields it set (`ModelConfig.config`), so
    the rest are the class's defaults and are held the same way."""
    from dew.nn.backbones.dit import SimpleDiT

    field = next(field for field in dataclasses.fields(SimpleDiT) if field.name == "mlp_ratio")
    monkeypatch.setattr(field, "default", field.default + 1)
    with pytest.raises(AssertionError, match=r"SimpleDiT.mlp_ratio"):
        test_every_recorded_default_is_the_one_older_runs_were_recorded_under()


def test_a_changed_tower_default_fails_the_snapshot(monkeypatch):
    from dew.nn.vision import Gemma4Vision

    field = next(field for field in dataclasses.fields(Gemma4Vision) if field.name == 'head_dim')
    monkeypatch.setattr(field, 'default', 64)
    with pytest.raises(AssertionError, match=r'Gemma4Vision.head_dim'):
        test_every_recorded_default_is_the_one_older_runs_were_recorded_under()


def contains(written, published) -> bool:
    """Whether every value `published` records is in `written` unchanged."""
    if isinstance(published, dict):
        return isinstance(written, dict) and all(
            name in written and contains(written[name], value) for name, value in published.items())
    return written == published


@pytest.mark.parametrize("name", ["hybrid-dit-176m-3f11a17", "hybrid-dit-176m-dfa94d6"])
def test_a_published_run_keeps_every_value_it_recorded(name):
    """What the record states comes back unchanged, and what it lacks takes
    today's default: `audio` None, a run conditioned on text alone."""
    run = DiffusionRunConfig.load(str(RUNS / name))
    written = json.loads(json.dumps(run.to_dict()))
    assert contains(written, record(name))
    assert run.audio is None
    assert DiffusionRunConfig.from_dict(written) == run


def test_a_run_from_before_the_regime_trains_on_the_sigmas_it_recorded():
    """The record states EDM2's P_mean -0.4 and P_std 1.0, which the code
    that wrote it drew from; they override the regime the run now fills."""
    run = DiffusionRunConfig.load(str(RUNS / "hybrid-dit-176m-3f11a17"))
    assert (run.preset.P_mean, run.preset.P_std) == (-0.4, 1.0)
    assert run.preset.lognormal() == (-0.4, 1.0)


def test_a_run_with_a_regime_keeps_it_and_its_autoencoder():
    """The autoencoder is recorded by its fields alone, so the SD VAE the run
    names reads back as the `PretrainedAutoencoder` it is today."""
    run = DiffusionRunConfig.load(str(RUNS / "hybrid-dit-176m-dfa94d6"))
    assert run.preset.regime == "latent" and run.preset.lognormal() == (-0.4, 1.0)
    assert run.autoencoder == PretrainedAutoencoder(modelname="pcuenq/sd-vae-ft-mse-flax", revision="main")


def test_an_unknown_field_is_refused():
    published = record("hybrid-dit-176m-dfa94d6")
    with pytest.raises(ValueError, match=r"unknown fields \['epochs'\]"):
        DiffusionRunConfig.from_dict({**published, "epochs": 3})
    with pytest.raises(ValueError, match=r"unknown fields \['warp'\]"):
        DiffusionRunConfig.from_dict({**published, "preset": {"name": "edm", "fields": {"warp": 1.0}}})


def quantized_before_weight_only(spec: Quantization) -> dict:
    """The dfa94d6 run record quantized as `spec`, written before
    `Quantization.weight_only` existed."""
    run = DiffusionRunConfig.from_dict(record("hybrid-dit-176m-dfa94d6"))
    written = json.loads(json.dumps(
        dataclasses.replace(run, trainer=dataclasses.replace(run.trainer, quantization=spec)).to_dict()))
    del written["trainer"]["quantization"]["weight_only"]
    return written


def test_a_quantized_run_from_before_weight_only_quantized_its_activations():
    spec = Quantization(dtype="fp8", patterns=(".*mlp.*",))
    run = DiffusionRunConfig.from_dict(quantized_before_weight_only(spec))
    assert run.trainer.quantization == spec and not run.trainer.quantization.weight_only


def test_a_task_reads_an_older_record_with_the_spec_at_the_top_level(tmp_path):
    """Tasks read `run.json` directly, including the spec at the top level
    where records from before the trainer held it carry it."""
    from dew.inference.tasks import _saved_quantization, run_record

    written = quantized_before_weight_only(Quantization())
    older = {**written, "quantization": written["trainer"]["quantization"],
             "trainer": {**written["trainer"], "quantization": None}}
    (tmp_path / "run.json").write_text(json.dumps(older))
    assert _saved_quantization(run_record(str(tmp_path))) == Quantization()

