"""A record naming a function by its import path rebuilds through the function's annotations."""
from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import pytest
from flax import linen as nn

from dew._records import JSON
from dew.config import ObjectiveConfig
from dew.inputs import Field, InputSpec
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
    return ObjectiveConfig("supervised", {"loss": loss, "inputs": InputSpec(Field("x", (3,)))})


def test_a_model_records_wrappers_round_trip_by_their_class_and_refuse_what_wraps_nothing():
    """A model record's wrappers are any `registry.Wrapper`, each written with its class path and
    rebuilt as that class (`ModelConfig.wrappers`), so the record names no closed set of wrappers; a
    value that wraps nothing is refused where the record is made."""
    from dew.quantization import InputQuantization
    from dew.registry import ModelConfig, to_record

    quantized = InputQuantization(format="modelopt", sites=(("lm_head", "fp8"),))
    wrapped = ModelConfig("simple_dit", wrappers=(quantized,))
    rebuilt = ModelConfig.from_dict(to_record(wrapped, ModelConfig))
    assert rebuilt == wrapped and type(rebuilt.wrappers[0]) is InputQuantization
    with pytest.raises(TypeError):
        ModelConfig("simple_dit", wrappers=(Field("image", (4, 4, 3)),))


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
