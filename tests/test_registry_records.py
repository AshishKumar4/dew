"""A record naming a function by its import path rebuilds through the function's annotations."""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from dew.config import ObjectiveConfig
from dew.inputs import Field, InputSpec
from dew.registry import Aliases

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


def test_a_float64_model_records_its_dtype_and_the_name_needs_x64():
    """A model computing in float64 records its dtype by name and rebuilds from it where JAX
    computes in float64; without x64 the name is refused, since JAX would narrow it to float32."""
    import jax
    import jax.numpy as jnp
    import pytest

    from dew.config import ModelConfig
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.registry import resolve_dtype

    model = CausalTransformer(vocab_size=8, emb_features=8, num_layers=1, num_heads=1, mlp_features=8,
                              max_seq_len=4, dtype=jnp.float64)
    with jax.enable_x64(new_val=True):
        record = ModelConfig.from_model(model)
        assert record.fields["dtype"] == "float64"
        assert ModelConfig.from_dict(dataclasses.asdict(record)).build().dtype == jnp.float64
    with jax.enable_x64(new_val=False), pytest.raises(ValueError):
        resolve_dtype("float64")
