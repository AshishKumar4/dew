"""A record naming a function by its import path rebuilds through the function's annotations."""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import typing
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from flax import linen as nn

from dew import registry
from dew._records import JSON
from dew.config import ObjectiveConfig
from dew.inputs import Field, InputSpec
from dew.nn import activations
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.backbones.jepa import JepaPredictor
from dew.objectives.supervised import Supervised
from dew.registry import Aliases, from_record

if TYPE_CHECKING:
    pass


@dataclasses.dataclass(frozen=True)
class Wiring:
    edges: tuple[tuple[int, int], ...]
    weight: float

    @classmethod
    def chain(cls, size: int, weight: float = 1.0) -> Wiring:
        return cls(tuple((index, index + 1) for index in range(size - 1)), weight)


class Circuit:
    """A plain class, as a network is: built by functions, not from its own fields."""

    def __init__(self, wirings: tuple[Wiring, ...], gain: float):
        self.wirings, self.gain = wirings, gain


def ring(size: int, weight: float = 1.0) -> Wiring:
    return Wiring(tuple((index, (index + 1) % size) for index in range(size)), weight)


def circuit(wiring: Wiring, *, gain: float = 1.0) -> Circuit:
    return Circuit((wiring,), gain)


def stacked(wirings: tuple[Wiring, ...], gain: float = 1.0) -> Circuit:
    return Circuit(wirings, gain)


@dataclasses.dataclass(frozen=True)
class Layer:
    wiring: Wiring
    depth: int = 1


def named(wiring: str) -> Circuit:
    return Circuit((), float(len(wiring)))


def path(name: str) -> str:
    return f"{__name__}:{name}"


# Kinds of functions, as sparx's networks and connectomes are: their own
# aliases, and records naming the functions by path.
wirings = Aliases("toy_wiring", {"ring": path("ring"), "chain": path("Wiring.chain")})
circuits = Aliases("toy_circuit", {"circuit": path("circuit"), "stacked": path("stacked"),
                                   "named": path("named"), "layer": path("Layer")})


def ring_record(size: int, **fields: float) -> dict:
    return {"class": path("ring"), "fields": {"size": size, **fields}}


def test_a_function_members_field_rebuilds_the_record_a_function_of_another_table_names():
    built = circuits.from_record({"class": "circuit", "fields": {"wiring": ring_record(3, weight=.5),
                                                                "gain": 2.0}})

    assert isinstance(built, Circuit) and built.gain == 2.0
    assert built.wirings == (ring(3, .5),)


def test_a_classmethod_member_builds_from_its_record():
    chain = {"class": path("Wiring.chain"), "fields": {"size": 3}}
    built = circuits.from_record({"class": "circuit", "fields": {"wiring": chain}})

    assert built.wirings == (Wiring(((0, 1), (1, 2)), 1.0),)


def test_records_inside_a_container_parameter_rebuild_each_entry():
    chain = {"class": path("Wiring.chain"), "fields": {"size": 2}}
    built = circuits.build("stacked", wirings=[ring_record(2), chain])

    assert built.wirings == (ring(2), Wiring.chain(2))


def test_a_dataclass_members_field_rebuilds_through_a_function_member():
    built = circuits.from_record({"class": "layer", "fields": {"wiring": ring_record(2), "depth": 3}})

    assert built == Layer(ring(2), 3)


def supervised(loss: dict) -> ObjectiveConfig:
    return ObjectiveConfig(functools.partial(Supervised, loss=loss,
                                             inputs=InputSpec(Field("x", (3,)))))


def test_a_model_records_wrappers_round_trip_by_their_class_and_refuse_what_wraps_nothing():
    """A model record's wrappers are any `registry.Wrapper`, each written with its class path and
    rebuilt as that class (`ModelConfig.wrappers`), so the record names no closed set of wrappers; a
    value that wraps nothing is refused where the record is made."""
    from dew.quantization import InputQuantization
    from dew.registry import ModelConfig, to_record

    quantized = InputQuantization(format="modelopt", sites=(("lm_head", "fp8"),))
    wrapped = ModelConfig(functools.partial(SimpleDiT), wrappers=(quantized,))
    rebuilt = ModelConfig.from_dict(to_record(wrapped, ModelConfig))
    assert rebuilt == wrapped and type(rebuilt.wrappers[0]) is InputQuantization
    with pytest.raises(TypeError):
        ModelConfig(functools.partial(SimpleDiT), wrappers=(Field("image", (4, 4, 3)),))


@dataclasses.dataclass(frozen=True)
class Loaded:
    """A component built around a value its record must not carry: it records `name` alone."""

    name: str
    weights: tuple[float, ...]

    def recorded_fields(self) -> dict:
        return {"name": self.name}


def test_a_recorded_dataclass_writes_its_view_alone_and_its_class_is_found_by_that_record():
    """A dataclass that states its recorded view (`registry.Recorded`) is written as that view, never
    its other fields, always naming its class; `component` finds the class back from the record, and
    refuses a protocol as the base, which no class need derive from."""
    from dew.registry import Recorded, component, to_record

    record = json.loads(json.dumps(to_record(Loaded("tiny", (1.0, 2.0)), object)))
    assert record == {"class": f"{Loaded.__module__}:Loaded", "fields": {"name": "tiny"}}
    assert component(Loaded, record) == (Loaded, {"name": "tiny"})
    with pytest.raises(TypeError):
        component(Recorded, record)


@dataclasses.dataclass(frozen=True)
class Settings:
    """A run's settings holding a component's record as JSON, bound by its owner later."""

    component: JSON


def test_a_class_record_in_a_json_field_stays_the_record_it_is():
    """A field typed JSON declares no class to build, so a component's class record there (a decision
    head's, an autoencoder's) is read back as written, for its owner to bind."""
    record = {"class": f"{Loaded.__module__}:Loaded", "fields": {"name": "tiny"}}
    assert from_record(Settings, {"component": record}) == Settings(record)


@dataclasses.dataclass(frozen=True)
class Called:
    """Settings holding JSON data and, beside it, a function."""

    data: JSON
    entries: dict[str, JSON]
    call: Callable[[], str] | None = None


def test_a_function_record_in_json_data_stays_the_data_it_is():
    """JSON data spelled like a function record, `{"function": name}`, is read back as written, whole or
    as a mapping's entry, where the walk from a record imported the function it names whatever its field
    declared (row 3957); a field that may hold a function still reads the record as the function."""
    spelled = {"function": "os:getcwd"}
    built = from_record(Called, {"data": spelled, "entries": {"f": spelled}, "call": spelled})
    assert built.data == spelled and built.entries == {"f": spelled}
    assert built.call is os.getcwd


class Pair(nn.Module):
    """A module whose two layers may be one, recording its view (`registry.Recorded`)."""

    first: nn.Module
    second: nn.Module

    def recorded_fields(self) -> dict:
        return {"first": self.first, "second": self.second}

    def __call__(self, x):
        return self.second(self.first(x))


def test_a_module_a_record_holds_twice_is_built_once_whether_or_not_it_records_its_view():
    """A layer two places hold, inside a recorded view or as a recorded view itself held twice, is
    written once and built once, so the rebuilt model keeps one set of parameters for it."""
    from dew.registry import read_arguments, to_record

    dense = nn.Dense(3)
    shared = Pair(dense, dense)
    record = json.loads(json.dumps(to_record(shared, object)))
    first, second = read_arguments(Pair, record["fields"]).values()
    assert first is second
    outer = json.loads(json.dumps(to_record(Pair(shared, shared), object)))
    inner, again = read_arguments(Pair, outer["fields"]).values()
    assert inner is again


def test_a_protocol_typed_field_reads_its_record_and_keeps_a_value_built_for_it():
    """A field typed by a protocol that is not runtime-checkable (a diffusion process's weighting)
    reads a record of its class, and a value the caller built for it is kept as it is."""
    from dew.diffusion import presets
    from dew.diffusion.process import Process
    from dew.registry import to_record

    process = presets.EDM(regime="pixel")()
    record = json.loads(json.dumps(to_record(process, Process)))
    assert type(from_record(Process, record).weighting) is type(process.weighting)
    assert from_record(Process, {**record, "weighting": process.weighting}).weighting is process.weighting


def test_a_linen_models_fields_are_written_against_the_annotations_its_own_module_resolves():
    """A linen model's constructor is written by Flax's dataclass transform, in Flax's namespace rather than
    the model's module: EDM2's U-Net's channel multipliers are written as records against `Sequence[int]`,
    which edm2's module imports under `from __future__ import annotations` (as a benchmark case's config
    and a parity cache key are written), and the U-Net builds from those records."""
    from dew.registry import argument_records, models

    records = argument_records(models["edm2_unet"], {"model_channels": 32, "channel_mult": [1, 2]})
    assert tuple(models.build("edm2_unet", **records).channel_mult) == (1, 2)


USER_MODELS = '''
import dew
from flax import linen as nn


@dew.register("users_dense")
class UsersDense(nn.Module):
    features: int = 4

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features)(x)
'''
"""A user's module of models, one registered by its decorator (`dew.register`) as the module is imported."""


def _ran(directory, script: str) -> str:
    """`script` run in a fresh Python beside the user's modules in `directory`, its printed last line."""
    import subprocess
    import sys

    paths = os.pathsep.join([str(directory), *filter(None, [os.environ.get("PYTHONPATH")])])
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=directory,
                         env={**os.environ, "PYTHONPATH": paths, "JAX_PLATFORMS": "cpu"})
    assert run.returncode == 0, run.stderr[-2000:]
    return run.stdout.strip().splitlines()[-1]


def test_a_users_decorated_model_builds_by_its_alias_from_a_config(tmp_path):
    """A user's model class named by `@dew.register(name)` as its module is imported builds by that alias
    from a config's record, as Dew's own models do (`ModelConfig`), and the record names it by its import
    path, which a process that never imported the user's module rebuilds (the owner, 2026-11-16: "if
    decorators could also be a supported way, I would like it"; row 6151), once it trusts the user's package
    (`import_trusted`), as a record of any class outside Dew."""
    (tmp_path / "users_models.py").write_text(USER_MODELS)
    record = json.loads(_ran(tmp_path, """
import json
import users_models
from dew.registry import ModelConfig, to_record
config = ModelConfig.from_dict({"name": "users_dense", "fields": {"features": 3}})
model = config.build()
assert type(model) is users_models.UsersDense and model.features == 3, model
print(json.dumps(to_record(config, ModelConfig)))
"""))
    assert record["name"] == "users_models:UsersDense", record
    rebuilt = _ran(tmp_path, f"""
import json
import sys
from dew.registry import ModelConfig, import_trusted
assert "users_models" not in sys.modules
record = json.loads({json.dumps(json.dumps(record))})
import_trusted(record, ("users_models",))
model = ModelConfig.from_dict(record).build()
print(type(model).__module__, type(model).__name__, model.features)
""")
    assert rebuilt.split() == ["users_models", "UsersDense", "3"]
    # A run that trusts the user's package (`dew train run.json --trust users_models`, which reads no record
    # yet) builds the alias its run.json names: trusting the package imports it, its decorators with it.
    aliased = _ran(tmp_path, """
from dew.registry import ModelConfig, import_trusted
import_trusted(None, ("users_models",))
model = ModelConfig.from_dict({"name": "users_dense", "fields": {"features": 5}}).build()
print(type(model).__name__, model.features)
""")
    assert aliased.split() == ["UsersDense", "5"]


class Defaulted(nn.Module):
    """A module whose default a record writes otherwise than it is held: a function."""

    activation: Callable = activations.silu

    @nn.compact
    def __call__(self, x):
        return self.activation(x)


def test_a_configs_defaults_read_back_as_the_records_it_wrote():
    """A model config's defaults (`Bound.defaults`) are records, and read back as those records
    (`JsonRecords`): written and read, or replaced, the config is the one it was, a function's default its
    `{"function": ...}`, never the value it names."""
    from dew.registry import ModelConfig, to_record

    config = ModelConfig(functools.partial(registry.models[f"{__name__}:Defaulted"]))
    assert config.defaults["activation"] == {"function": "dew.nn.activations:silu"}
    assert ModelConfig.from_dict(json.loads(json.dumps(to_record(config, ModelConfig)))) == config
    assert dataclasses.replace(config) == config


class _Held[T]:
    """A plain generic class no record builds: what a binder makes and hands a record's reader built."""


@dataclasses.dataclass(frozen=True)
class _HoldsSubscripted:
    held: _Held[int]
    named: str = "plain"


def test_a_field_of_a_subscripted_generic_keeps_the_instance_its_caller_built():
    """A dataclass field annotated with a subscripted generic of a nominal class keeps a built instance of
    the class as it is, as one annotated with the bare class does, its other fields read as before: a
    condition's encoder, `ConditionEncoder[Any, Any]`, bound around a user's plain encoder."""
    held = _Held[int]()
    built = from_record(_HoldsSubscripted, {"held": held, "named": "kept"})
    assert built.held is held and built.named == "kept"


@dataclasses.dataclass(frozen=True)
class _Boxed[T]:
    size: int = 1


@dataclasses.dataclass(frozen=True)
class _HoldsBoxed:
    boxed: _Boxed[int]


def test_a_record_of_a_field_of_a_subscripted_generic_builds_its_class():
    """A dataclass field annotated with a subscripted generic of a dataclass is built from its record, as one
    annotated with the bare class is, where the record was left a plain mapping: every reader of a field's
    class reads a generic's origin (`nominal_class`), the record's reader as the built value's (row 8387)."""
    built = from_record(_HoldsBoxed, {"boxed": {"size": 3}})
    assert built.boxed == _Boxed(size=3)


def test_a_field_of_a_subscripted_generic_records_its_value_as_a_field_of_its_class_does():
    """A field annotated with a subscripted generic of a dataclass writes a value of that class as its plain
    fields, as one annotated with the bare class does, not a class record naming an import path: the
    writer reads a generic's origin as the reader does (`nominal_class`, row 8403)."""
    from dew.registry import to_record

    assert to_record(_Boxed(size=3), _Boxed[int]) == to_record(_Boxed(size=3), _Boxed) == {"size": 3}


@dataclasses.dataclass(frozen=True)
class Halt:
    fraction: float = 1.0


@dataclasses.dataclass(frozen=True)
class Stall:
    evals: int = 1


@dataclasses.dataclass(frozen=True)
class Shared:
    either: Halt | Stall | None = None


type Stops = Halt | Stall
"""A union of classes through an alias, which a field's union names beside one of its classes again."""


@dataclasses.dataclass(frozen=True)
class Overlapping:
    either: Stops | Halt | None = None


@dataclasses.dataclass(frozen=True)
class Stopped:
    either: Halt | Stall | None = None
    alone: Halt | None = None
    sized: int | Halt = 0


def test_an_unnamed_record_of_a_union_of_classes_builds_the_one_whose_fields_hold_it():
    """A mapping naming no class (no `{"class": ...}`) for a field of a union of classes builds the one
    dataclass whose fields hold its every key, as a field of the class reads it (row 11980:
    `TrainerConfig.stop`, a Pause or a Plateau, given `{"fraction": 0.5}` held a dict and the run never
    paused); one several classes' fields hold is refused, as nothing says which it is; a class record still
    builds the one it names."""
    built = from_record(Stopped, {"either": {"fraction": 0.5}, "alone": {"fraction": 0.25},
                                  "sized": {"fraction": 0.75}})
    assert (built.either, built.alone, built.sized) == (Halt(0.5), Halt(0.25), Halt(0.75))
    assert from_record(Stopped, {"either": {"evals": 3}}).either == Stall(3)
    named = from_record(Stopped, {"either": {"class": f"{__name__}:Stall", "fields": {"evals": 3}}})
    assert named.either == Stall(3)
    with pytest.raises(ValueError):
        from_record(Shared, {"either": {}})


def test_a_saved_decoders_unnamed_yarn_scaling_builds_yarns_record():
    """A saved decoder's `rope_scaling` written as YaRN's fields alone, naming no class
    (`{"rope_type": "yarn", "factor": 4.0}`), reads as a `YarnScaling`, the one scaling of `Scaling` that
    takes it whole: Llama 3.1's and LongRoPE's lack their required fields, and each refuses another's
    `rope_type` (the review of API02b, P1: every class whose fields held the keys was ambiguous); a Llama
    3.1 ramp's fields alone read as its `RopeScaling`."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.rope import RopeScaling, YarnScaling

    record = {"vocab_size": 32, "emb_features": 16, "num_layers": 1, "num_heads": 2, "max_seq_len": 64,
              "rope_scaling": {"rope_type": "yarn", "factor": 4.0}}
    assert from_record(CausalTransformer, record).rope_scaling == YarnScaling(factor=4.0)
    llama = {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
             "original_max_position_embeddings": 8192}
    assert from_record(CausalTransformer, {**record, "rope_scaling": llama}).rope_scaling == RopeScaling(
        **llama)




@dataclasses.dataclass(frozen=True)
class Wide:
    size: int = 1


@dataclasses.dataclass(frozen=True)
class Tall:
    size: int = 1


@dataclasses.dataclass(frozen=True)
class OverWide:
    """A union member whose shared part is a `Wide`, refusing any other its record reads as it."""

    part: Wide

    def __post_init__(self) -> None:
        if not isinstance(self.part, Wide):
            raise ValueError("a part of another kind")


@dataclasses.dataclass(frozen=True)
class OverTall:
    """`OverWide`'s twin over a `Tall`."""

    part: Tall

    def __post_init__(self) -> None:
        if not isinstance(self.part, Tall):
            raise ValueError("a part of another kind")


@dataclasses.dataclass(frozen=True)
class Sharing:
    either: OverWide | OverTall | None = None
    again: Wide | None = None


def test_an_unnamed_union_records_winning_trial_keeps_the_shared_module_its_record_refers_to_again():
    """An unnamed record of a union whose part is a shared module (`{"share": n}`) the record refers to
    again (`{"shared": n}`): the one member that builds keeps the module it built, which the later
    reference reads, the same object; a trial that is refused keeps none (each trial reads the shared
    modules as they stood before any, the API02 review). A guard: no public read told the
    bookkeeping apart before."""
    part = {"class": f"{__name__}:Wide", "share": 1, "fields": {"size": 2}}
    built = from_record(Sharing, {"either": {"part": part}, "again": {"shared": 1}})
    assert built.either == OverWide(Wide(2)) and built.again is built.either.part


@dataclasses.dataclass(frozen=True)
class OverTallDeclared:
    """`OverTall` checking nothing itself: its declared `Tall` alone says what its part is."""

    part: Tall


@dataclasses.dataclass(frozen=True)
class SharedFirst:
    again: Wide | None = None
    either: OverWide | OverTallDeclared | None = None


def test_an_unnamed_union_records_trial_reads_a_module_shared_before_as_its_field_declares_it():
    """A module shared by a field read first (`{"share": n}`) that an unnamed union record refers to again
    (`{"shared": n}`): a trial whose field declares another class (`OverTallDeclared.part: Tall`) refuses the
    `Wide` as a module built for it would, so the one member it is builds, where the module read before
    was handed to every trial and two built, refused as ambiguous (the API02 review)."""
    part = {"class": f"{__name__}:Wide", "share": 1, "fields": {"size": 2}}
    built = from_record(SharedFirst, {"again": part, "either": {"part": {"shared": 1}}})
    assert built.either == OverWide(Wide(2)) and built.either.part is built.again


@dataclasses.dataclass(frozen=True)
class SharedLoosely:
    again: Wide | None = None
    anything: typing.Any = None


def test_a_module_shared_before_reads_again_under_a_field_declaring_any_value():
    """A module shared by a field read first (`{"share": n}`), referred to again (`{"shared": n}`) by a field
    declaring any value (`typing.Any`): it reads the one module, where `Any` reached `isinstance` and raised
    (the API02 review)."""
    part = {"class": f"{__name__}:Wide", "share": 1, "fields": {"size": 2}}
    built = from_record(SharedLoosely, {"again": part, "anything": {"shared": 1}})
    assert built.anything is built.again

def test_an_unnamed_record_of_a_union_naming_a_class_twice_builds_it_once():
    """A union naming one class twice, through an alias and again (`Stops | Halt`), holds it once: an
    unnamed record of its fields builds it, where two trials of the one class read as two that fit it (the
    review of API02b)."""
    assert from_record(Overlapping, {"either": {"fraction": 0.5}}).either == Halt(0.5)


def test_a_bound_member_reads_back_as_its_annotation_admits():
    """A member bound by keyword (`functools.partial`) reads back from its class record as the partial it
    was: under `functools.partial[Any]` and the bare partial whatever its class, under a class's partial only
    as a binding of that class, a built partial and its record alike (row 12896)."""
    import functools
    from typing import Any

    from dew.nn.backbones.jepa import JepaEncoder, JepaPredictor
    from dew.registry import from_record, to_record

    bound = functools.partial(JepaPredictor, num_layers=1, predictor_features=8)
    record = to_record(bound, functools.partial[JepaPredictor])
    for annotation in (functools.partial[Any], functools.partial, functools.partial[JepaPredictor]):
        again = from_record(annotation, record)
        assert isinstance(again, functools.partial)
        assert (again.func, again.keywords) == (JepaPredictor, {"num_layers": 1, "predictor_features": 8})
    other = functools.partial(JepaEncoder, num_layers=1)
    for given in (other, to_record(other, functools.partial)):
        with pytest.raises(ValueError):
            from_record(functools.partial[JepaPredictor], given)
    # A factory declared to return the class binds it too, given and as its record alike.
    from model_support import decoder

    from dew.nn.backbones.causal_transformer import CausalTransformer

    made = functools.partial(decoder, kind="attention")
    for given in (made, to_record(made, functools.partial)):
        read = from_record(functools.partial[CausalTransformer], given)
        assert isinstance(read, functools.partial) and read.func is decoder


USER_ENVS = """
import dataclasses
from typing import ClassVar


@dataclasses.dataclass(frozen=True)
class Opened:
    assets: str = "scene"
    built: ClassVar[list[str]] = []

    def __post_init__(self):
        self.built.append(self.assets)


@dataclasses.dataclass(frozen=True)
class Simulated:
    renderer: Opened
"""


def test_a_bound_record_holds_its_arguments_until_its_member_is_built(tmp_path):
    """A bound member holds its arguments as given until it is built (row 12896, train 1412's review): a
    model's views named by their entries alone, which its environment widens, held as named; and a record
    naming a part of a recorded kind (a trusted package's environment's renderer, one that opens assets)
    builds none of it, the record saved again as read, until the environment is built."""
    from dew.objectives.rl.dreamer import WorldModel
    from dew.registry import ModelConfig

    named = ModelConfig(functools.partial(WorldModel, reads="image")).with_fields(auxiliary=[["state", 256]])
    assert named.fields["auxiliary"] == [["state", 256]]
    (tmp_path / "users_envs.py").write_text(USER_ENVS)
    report = json.loads(_ran(tmp_path, """
import json
from dew.config.env import EnvConfig
from dew.registry import from_record, import_trusted, to_record
import_trusted(None, ("users_envs",))
import users_envs
renderer = {"class": "users_envs:Opened", "fields": {"assets": "track"}}
record = {"name": "users_envs:Simulated", "fields": {"renderer": renderer}}
env = from_record(EnvConfig, record)
read = list(users_envs.Opened.built)
written = to_record(env, EnvConfig)
built = env.build()
print(json.dumps({"read": read, "same": written["fields"] == record["fields"],
                  "built": built.renderer.assets, "opened": users_envs.Opened.built}))
"""))
    assert report == {"read": [], "same": True, "built": "track", "opened": ["track"]}, report


def test_a_record_naming_no_member_is_its_kinds_default_ones():
    """A record of a kind with a default member that names none, as an earlier Dew's could, binds that
    default (row 12896, train 1412's review): a model's the simple DiT, a rollout's Dreamer's."""
    from dew.config.env import RolloutConfig
    from dew.objectives.rl.dreamer import DreamerRollout
    from dew.registry import ModelConfig

    model = ModelConfig.from_dict({"fields": {"patch_size": 2}})
    assert (model.member, dict(model.fields)) == (SimpleDiT, {"patch_size": 2})
    assert from_record(RolloutConfig, {"fields": {"batch": 4}}).member is DreamerRollout


type _Predictor = functools.partial[JepaPredictor]


def test_a_bound_member_reads_back_through_its_alias():
    """A partial named by a type alias reads back as the partial it names (train 1412's review)."""
    from dew.registry import to_record

    bound = functools.partial(JepaPredictor, num_layers=4)
    again = from_record(_Predictor, to_record(bound, _Predictor))
    assert isinstance(again, functools.partial)
    assert (again.func, again.keywords) == (JepaPredictor, {"num_layers": 4})


def test_a_bound_records_fields_that_are_no_record_are_refused():
    """A bound member's `fields` that are no record (an empty list, an empty string, zero) are refused, not
    read as no fields; fields absent are none (train 1416's review)."""
    from dew.config.env import RolloutConfig
    from dew.registry import ModelConfig

    for fields in ([], "", 0):
        with pytest.raises(ValueError):
            ModelConfig.from_dict({"name": "simple_dit", "fields": fields})
        with pytest.raises(ValueError):
            from_record(RolloutConfig, {"fields": fields})
    assert dict(ModelConfig.from_dict({"name": "simple_dit"}).fields) == {}


def test_a_partial_is_recorded_only_where_its_field_reads_one_back():
    """A partial held where its field declares no `functools.partial[...]` (a function's or a class's under
    a callable or a class) is refused as it is written, not saved as a class record its reader builds the
    class from, or refuses as a function's (train 1416's review: a reward's partial saved, its run unread).
    """
    from dew.registry import Unrecordable, import_path, to_record

    for value, annotation in ((functools.partial(json.dumps, indent=2), Callable[..., str]),
                              (functools.partial(JepaPredictor, num_layers=1), JepaPredictor)):
        with pytest.raises(Unrecordable):
            to_record(value, annotation)
    # Under the partial its field declares, or its Optional, it is the class record that reads back.
    bound = functools.partial(JepaPredictor, num_layers=1)
    for annotation in (functools.partial[JepaPredictor], functools.partial[JepaPredictor] | None,
                       _Predictor | None):
        record = to_record(bound, annotation)
        assert isinstance(record, dict) and record["class"] == import_path(JepaPredictor)
        again = from_record(annotation, record)
        assert isinstance(again, functools.partial)
        assert (again.func, again.keywords) == (JepaPredictor, {"num_layers": 1})


def test_a_held_argument_partial_is_recorded_only_where_its_member_reads_one_back():
    """A model's held argument that is a partial of a function, where the member's field declares a
    callable (Unet's `activation`), is refused as its record is written: its reader refuses a function's
    class record (train 1429's review: a bound member's arguments were written against no annotation)."""
    from dew.nn.activations import cast_rounded
    from dew.nn.backbones.unet import Unet
    from dew.registry import ModelConfig, Unrecordable

    rounded = functools.partial(cast_rounded, dtype="bfloat16")
    for held in (ModelConfig(functools.partial(Unet, activation=rounded)),
                 ModelConfig(functools.partial(Unet, feature_depths=(64, rounded)))):
        with pytest.raises(Unrecordable):
            held.record()


type _MaybePredictor = _Predictor | None
type _TaggedPredictor = typing.Annotated[functools.partial[JepaPredictor] | None, "tag"]


def test_a_partial_under_a_nested_optional_alias_reads_back():
    """A partial under an Optional of an alias of an Optional of its partial alias reads back as the
    partial it was, as its reader flattens the nesting, an Annotated alias's among them (train 1429's
    review: the writer refused it; theme 73's: the reader kept an Annotated alias's record a dict)."""
    from dew.registry import to_record

    bound = functools.partial(JepaPredictor, num_layers=2)
    for annotation in (_MaybePredictor, _MaybePredictor | None, _TaggedPredictor):
        again = from_record(annotation, to_record(bound, annotation))
        assert isinstance(again, functools.partial)
        assert (again.func, again.keywords) == (JepaPredictor, {"num_layers": 2})


type _Predictors = list[functools.partial[JepaPredictor]]


def test_a_held_container_of_partials_is_checked_as_its_reader_types_its_entries():
    """A held argument's container of partials is typed as its reader types it, its alias and Optional
    flattened before its entries' types are read: a list of predictor partials under `_Predictors | None`
    reads back, and a function's partial in it does not (theme 73's review)."""
    from dew.nn.activations import cast_rounded

    predictors = [functools.partial(JepaPredictor, num_layers=1)]
    assert registry._unread_partial(predictors, _Predictors | None) is None
    stray = functools.partial(cast_rounded, dtype="bfloat16")
    assert registry._unread_partial([*predictors, stray], _Predictors | None) is stray



class Predicting(nn.Module):
    """A user's module holding a name or a tuple of predictor partials: a union whose one sequence member
    a record's list reads as."""

    predictors: str | tuple[functools.partial[JepaPredictor], ...] = "none"

    @nn.compact
    def __call__(self, x):
        return x


def test_a_tuple_of_partials_under_a_union_records_and_builds_as_its_sequence_member():
    """A bound member's tuple of partials under a union of a name and a tuple of partials is written as its
    reader reads it, the union's one sequence member, and builds back the partials it held (train 1430's
    review: the writer typed the entries by no member and refused them)."""
    from dew.registry import ModelConfig

    predictor = functools.partial(JepaPredictor, num_layers=2)
    config = ModelConfig(functools.partial(registry.models[f"{__name__}:Predicting"], predictors=(predictor,)))
    rebuilt = ModelConfig.from_dict(json.loads(json.dumps(config.record())))
    (built,) = rebuilt.build().predictors
    assert isinstance(built, functools.partial)
    assert (built.func, built.keywords) == (JepaPredictor, {"num_layers": 2})
