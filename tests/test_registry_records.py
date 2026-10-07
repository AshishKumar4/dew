"""A record naming a function by its import path rebuilds through the function's annotations."""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest
from flax import linen as nn

from dew import registry
from dew.config import ModelConfig, ObjectiveConfig
from dew.inputs import Field, InputSpec
from dew.registry import Aliases, import_trusted

if TYPE_CHECKING:
    from decimal import Decimal


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


def test_a_value_already_built_passes_through():
    assert circuits.build("circuit", wiring=ring(4)).wirings == (ring(4),)


def test_a_function_member_refuses_a_field_it_does_not_take_or_lacks():
    with pytest.raises(ValueError, match=r"ring does not match the record: unknown fields \['radius'\]"):
        circuits.from_record({"class": "circuit", "fields": {"wiring": ring_record(3, radius=1.0)}})
    with pytest.raises(ValueError, match=r"missing fields \['size'\]"):
        wirings.from_record({"class": "ring", "fields": {}})


def test_a_record_naming_a_function_of_another_type_is_not_taken_for_the_field():
    """`named` returns a Circuit, not a Wiring, so a Wiring field refuses its record."""
    with pytest.raises(ValueError, match="not one Wiring"):
        circuits.build("circuit", wiring={"class": path("named"), "fields": {"wiring": "x"}})


def test_a_string_parameter_keeps_its_value():
    assert circuits.build("named", wiring="ring").gain == 4.0



def unresolved(size: int) -> Decimal:
    raise AssertionError("a record naming it is refused before it is called")


def test_an_annotation_its_module_imports_for_the_checker_alone_is_named():
    with pytest.raises(ValueError, match="unresolved's annotation of return names what"):
        circuits.build("circuit", wiring={"class": path("unresolved"), "fields": {"size": 1}})


@dataclasses.dataclass(frozen=True)
class Scaled:
    """A configured loss of a package outside Dew."""

    factor: float

    def __call__(self, outputs, batch):
        return outputs * self.factor


def supervised(loss: dict) -> ObjectiveConfig:
    return ObjectiveConfig("supervised", {"loss": loss, "inputs": InputSpec(Field("x", (3,)))})


def test_a_record_constructs_nothing_its_field_does_not_declare(monkeypatch):
    """A record may name anything imported. A callable field takes a class
    whose instances are called, of Dew's or a trusted package's, and a model
    or an objective names a class of that kind. `subprocess.Popen`, imported
    here, runs its command in its constructor, and each is refused before it
    is constructed."""
    import subprocess

    def constructed(self, *args, **kwargs):
        raise AssertionError("a record constructed subprocess.Popen")

    monkeypatch.setattr(subprocess.Popen, "__init__", constructed)
    popen = {"class": "subprocess:Popen", "fields": {"args": ["true"]}}
    with pytest.raises(ValueError, match="whose instances are called"):
        supervised(popen).build(model=nn.Dense(2))
    with pytest.raises(ValueError, match="which is no model"):
        ModelConfig("subprocess:Popen", {"args": ["true"]}).build()
    with pytest.raises(ValueError, match="which is no objective"):
        ObjectiveConfig("subprocess:Popen", {"args": ["true"]}).build()


def test_a_callable_class_outside_dew_is_built_once_its_package_is_trusted(monkeypatch):
    monkeypatch.setattr(registry, "_TRUSTED", set(registry._TRUSTED))
    objective = supervised({"class": path("Scaled"), "fields": {"factor": 2.0}})
    with pytest.raises(ValueError, match="trusted package"):
        objective.build(model=nn.Dense(2))
    import_trusted({}, (__name__.partition(".")[0],))
    assert objective.build(model=nn.Dense(2)).criterion == Scaled(2.0)
