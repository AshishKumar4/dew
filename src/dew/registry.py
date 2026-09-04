"""Names for the things a run is made of.

One `Registry` per kind: `models`, `presets`, `samplers`, `datasets`,
`encoders`, `metrics`, `objectives`. A registry is a decorator, a mapping and
an attribute view over the same table, so `models["simple_dit"]`,
`models.SimpleDiT` and the class are one object. A name or a field the table
does not know raises; nothing is dropped or guessed.

The registries are empty at import. Each member registers itself where it is
defined, so importing a package fills its table and the registry module
imports none of them.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Iterator, Mapping
from typing import Any, Callable, Generic, TypeVar, Union

import jax.numpy as jnp

T = TypeVar("T")


class Registry(Mapping[str, T], Generic[T]):
    """Names for one kind of thing: a decorator, a mapping and an attribute view."""

    def __init__(self, kind: str):
        self.kind = kind
        self._members: dict[str, T] = {}

    def __call__(self, name: str, /) -> Callable[[T], T]:
        """`@models("simple_dit")` on the class it names."""
        if not isinstance(name, str) or not name:
            raise TypeError(f"a {self.kind} name is a non-empty string, not {name!r}")

        def register(member: T) -> T:
            held = self._members.get(name)
            if held is not None and held is not member:
                raise ValueError(
                    f"{self.kind} {name!r} is already {_describe(held)}; "
                    f"a name maps to one {self.kind}")
            self._members[name] = member
            return member

        return register

    def __getitem__(self, name: str) -> T:
        try:
            return self._members[name]
        except KeyError:
            raise KeyError(
                f"no {self.kind} named {name!r}; known: {', '.join(sorted(self._members))}"
            ) from None

    def __getattr__(self, attr: str) -> T:
        """`models.SimpleDiT`: the member whose class name is `attr`."""
        if attr.startswith("_"):
            raise AttributeError(attr)
        for member in self._members.values():
            if getattr(member, "__name__", None) == attr:
                return member
        raise AttributeError(
            f"no {self.kind} is called {attr!r}; known: "
            f"{', '.join(sorted(_describe(m) for m in self._members.values()))}")

    def __iter__(self) -> Iterator[str]:
        return iter(self._members)

    def __len__(self) -> int:
        return len(self._members)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {sorted(self._members)})"

    def name_of(self, member: T) -> str:
        """The name a member was registered under."""
        for name, held in self._members.items():
            if held is member:
                return name
        raise KeyError(f"{_describe(member)} is not a registered {self.kind}")

    def build(self, name: str, /, **fields: Any) -> Any:
        """Construct the member called `name` from keyword fields.

        A field the member has no declaration for is an error, since dropping
        it would build something other than what was asked for. Fields arrive
        from JSON as often as from code, so a field whose declared type is a
        value builds from a record here, at the one boundary where a logged
        config becomes an object: `models.build("m", attention={"heads": 8})`
        and `models.build("m", attention=Attention(heads=8))` agree.
        """
        member = self[name]
        if dataclasses.is_dataclass(member):
            declared = {f.name for f in dataclasses.fields(member) if f.init}
            unknown = sorted(set(fields) - declared)
            if unknown:
                raise ValueError(
                    f"{self.kind} {name!r} ({_describe(member)}) has no field for "
                    f"{unknown}; its fields are {sorted(declared)}")
            fields = {key: _field_value(member, key, value)
                      for key, value in fields.items()}
        return member(**fields)

    @property
    def union(self) -> Any:
        """`Union[...]` of the members, for a tyro subcommand over the table."""
        members = tuple(self._members.values())
        if not members:
            raise ValueError(f"the {self.kind} registry is empty")
        return Union[members] if len(members) > 1 else members[0]


def _describe(member: Any) -> str:
    return getattr(member, "__name__", repr(member))


def _declared_type(member: Any, field: str) -> Any:
    """The annotation of `member`'s `field`, or None when it cannot be read."""
    try:
        hints = typing.get_type_hints(member)
    except Exception:  # a forward reference to something not importable here
        return None
    return hints.get(field)


def _value_type(annotation: Any) -> Any:
    """The value class `annotation` asks for, looking through Optional only.

    A container of values is not itself a value, so `Mapping[str, LayerKind]`
    answers None and its entries are walked instead.
    """
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return annotation
    if typing.get_origin(annotation) not in (Union, types.UnionType):
        return None
    for argument in typing.get_args(annotation):
        if dataclasses.is_dataclass(argument) and isinstance(argument, type):
            return argument
    return None


def _entry_type(annotation: Any) -> Any:
    """What one entry of an annotated container holds."""
    arguments = [a for a in typing.get_args(annotation) if a is not Ellipsis]
    if typing.get_origin(annotation) in (Union, types.UnionType):
        arguments = [a for a in arguments if a is not type(None)]
        return _entry_type(arguments[0]) if len(arguments) == 1 else None
    return arguments[-1] if arguments else None


def from_record(annotation: Any, value: Any) -> Any:
    """`value` as its annotation asks for it: a record becomes the value it
    describes, and anything already built is left alone.

    Containers are walked, so a mapping of records and a tuple of records
    build their values too, which is what keeps a model config a dict from
    the command line all the way to the module.
    """
    if isinstance(value, Mapping):
        held = _value_type(annotation)
        if held is None:
            # A record with no value class behind it, such as one of the unets'
            # per-stage attention settings: entries are walked and the dtype
            # rule below still applies by name.
            return {key: resolve_dtype(item) if key == "dtype"
                    else from_record(_entry_type(annotation), item)
                    for key, item in value.items()}
        declared = sorted(f.name for f in dataclasses.fields(held) if f.init)
        unknown = sorted(set(value) - set(declared))
        if unknown:
            raise ValueError(f"{_describe(held)} has no field for {unknown}; its "
                             f"fields are {declared}")
        return held(**{key: _field_value(held, key, item)
                       for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return type(value)(from_record(_entry_type(annotation), item) for item in value)
    return value


def _field_value(member: Any, field: str, value: Any) -> Any:
    """One field on its way into `member`: a dtype by name, a value from a record."""
    if field == "dtype":
        return resolve_dtype(value)
    return from_record(_declared_type(member, field), value)


_DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}


def resolve_dtype(value: Any) -> Any:
    """A dtype as a module field: a jnp dtype, one of its names, or None."""
    if value is None or not isinstance(value, str):
        return value
    try:
        return _DTYPES[value]
    except KeyError:
        raise ValueError(
            f"dtype {value!r} is not one of {sorted(_DTYPES)}") from None


def dtype_name(value: Any) -> str | None:
    """The name `resolve_dtype` accepts for a dtype, for a logged config."""
    if value is None:
        return None
    for name, dtype in _DTYPES.items():
        if jnp.dtype(value) == jnp.dtype(dtype):
            return name
    raise ValueError(f"{value!r} is not a dtype a config can name")


def with_precision(name: str, config: Mapping[str, Any], *,
                   dtype: str, attention_impl: str) -> dict[str, Any]:
    """A model config with the run's compute dtype and attention kernel in it.

    Params stay float32 whatever `dtype` says; it is the compute dtype. The
    UNets keep per-stage attention settings in nested `attention_configs`,
    which do not inherit the model dtype and default `force_fp32_for_softmax`
    off, which no fused kernel can honour, so the knobs reach into them.
    """
    duplicate = sorted(set(config) & {"dtype", "attention_impl"})
    if duplicate:
        raise ValueError(
            f"the model config carries {duplicate}, which the run's precision "
            "settings own; set --model.dtype and --model.attention-impl instead")
    member = models[name]
    fields = {**config, "dtype": dtype,
              "attention_impl": None if attention_impl == "reference" else attention_impl}
    stages = {f.name: f for f in dataclasses.fields(member)}.get("attention_configs")
    if stages is not None:
        precise = {"dtype": dtype, "force_fp32_for_softmax": True}
        fields["attention_configs"] = [
            None if stage is None
            else {**stage, **precise} if isinstance(stage, Mapping)
            else dataclasses.replace(stage, **precise)
            for stage in config.get("attention_configs", stages.default)]
    return fields


models: Registry[type] = Registry("model")
presets: Registry[type] = Registry("preset")
samplers: Registry[type] = Registry("sampler")
datasets: Registry[type] = Registry("dataset")
encoders: Registry[type] = Registry("encoder")
metrics: Registry[Callable[..., Any]] = Registry("metric")
objectives: Registry[type] = Registry("objective")

__all__ = [
    "Registry", "models", "presets", "samplers", "datasets", "encoders", "metrics", "objectives",
    "resolve_dtype", "dtype_name", "with_precision",
]
