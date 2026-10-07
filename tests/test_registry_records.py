"""A record naming a function by its import path rebuilds through the function's annotations."""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from dew.registry import Aliases

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
