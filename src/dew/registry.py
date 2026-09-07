"""Names for the things a run is made of.

One Registry per kind, including model components such as mixers, towers and
projectors. A registry is a decorator, a mapping and
an attribute view over the same table, so `models["simple_dit"]`,
`models.SimpleDiT` and the class are one object. A name or a field the table
does not know raises.

The registries are empty at import. Each member registers itself where it is
defined, so importing a package fills its table and the registry module
imports none of them.
"""

from __future__ import annotations

import dataclasses
import sys
import types
import typing
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, Union

import jax.numpy as jnp
from typing_extensions import Format, get_annotations

if TYPE_CHECKING:
    from flax import linen as nn

    from dew.data.dataset import DatasetSpec
    from dew.diffusion.presets import Preset
    from dew.nn.mixers import MixerBase
    from dew.nn.vision import ProjectorBase, TowerBase
    from dew.inputs.encoders import ConditionEncoder
    from dew.objectives.base import Metric, Objective
    from dew.sampling.solvers import Solver

T = TypeVar("T", bound=Callable[..., Any])
M = TypeVar("M", bound=Callable[..., Any])


class Registry(Mapping[str, T], Generic[T]):
    """Names for one kind of thing: a decorator, a mapping and an attribute view."""

    def __init__(self, kind: str, *, record: Literal["name", "kind"] = "name"):
        self.kind = kind
        self.record = record
        # A decorator has no base class to test the member against, and it
        # hands back the class it decorated so a caller's checker keeps the
        # concrete type (`DiffusionObjective`, not `Objective`). The table is
        # untyped here and typed on the way out.
        self._members: dict[str, Any] = {}

    def __call__(self, name: str, /) -> Callable[[M], M]:
        """`@models("simple_dit")` on the class it names."""
        if not isinstance(name, str) or not name:
            raise TypeError(f"a {self.kind} name is a non-empty string, not {name!r}")

        def register(member: M) -> M:
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

    def name_of(self, member: object) -> str:
        """The name a member was registered under."""
        for name, held in self._members.items():
            if held is member:
                return name
        raise KeyError(f"{_describe(member)} is not a registered {self.kind}")

    def build(self, name: str, /, **fields: Any) -> Any:
        """Construct the member called `name` from keyword fields.

        A field the member does not declare is an error. Fields arrive from
        JSON as often as from code, so a field whose declared type is a value
        builds from a record here, where a logged config becomes an object:
        `models.build("m", attention={"heads": 8})` and
        `models.build("m", attention=Attention(heads=8))` agree.
        """
        member = self[name]
        return member(**self._declared_fields(name, member, fields))

    def _declared_fields(self, name: str, member: Any,
                         fields: dict[str, Any]) -> dict[str, Any]:
        """`fields` as the member declares them, or an error naming what it
        has no field for. A member that is not a dataclass takes them as given."""
        if not dataclasses.is_dataclass(member):
            return fields
        declared = {f.name for f in dataclasses.fields(member) if f.init}
        unknown = sorted(set(fields) - declared)
        if unknown:
            raise ValueError(
                f"{self.kind} {name!r} ({_describe(member)}) has no field for "
                f"{unknown}; its fields are {sorted(declared)}")
        return {key: _field_value(member, key, value) for key, value in fields.items()}

    @property
    def union(self) -> Any:
        """`Union[...]` of the members, for a tyro subcommand over the table."""
        members = list(self._members.values())
        if not members:
            raise ValueError(f"the {self.kind} registry is empty")
        return Union[tuple(members)] if len(members) > 1 else members[0]


def _describe(member: object) -> str:
    return getattr(member, "__name__", repr(member))


def _declared_type(member: type, field: str) -> object:
    """Resolve one field without evaluating unrelated dependency annotations."""
    for owner in member.__mro__:
        annotations = get_annotations(owner, format=Format.FORWARDREF)
        if field not in annotations:
            continue
        selected = types.SimpleNamespace(__annotations__={field: annotations[field]})
        try:
            return typing.get_type_hints(
                selected, globalns=dict(vars(owner)),
                localns=vars(sys.modules[owner.__module__]))[field]
        except NameError:
            # Only this field's unavailable dependency leaves its value opaque.
            return None
    return None


def _value_type(annotation: object) -> type | None:
    """A dataclass type behind an Optional, but not a multi-member union."""
    annotation = _unwrapped(annotation)
    return (annotation if isinstance(annotation, type) and dataclasses.is_dataclass(annotation)
            else None)


def _unwrapped(annotation: object) -> object:
    """`annotation` with an Optional looked through; a union of several
    members says nothing about its entries and answers None."""
    if typing.get_origin(annotation) not in (Union, types.UnionType):
        return annotation
    members = [a for a in typing.get_args(annotation) if a is not type(None)]
    return members[0] if len(members) == 1 else None


def entry_types(annotation: object, count: int) -> list[object]:
    """The annotation of each of the `count` entries of an annotated
    container: a fixed tuple's per-position types, otherwise its one element
    type repeated (a mapping's value type, a sequence's element). None is an
    unannotated entry, which takes its value as given."""
    annotation = _unwrapped(annotation)
    arguments = typing.get_args(annotation)
    if typing.get_origin(annotation) is tuple and Ellipsis not in arguments:
        if len(arguments) != count:
            raise ValueError(f"{annotation} takes {len(arguments)} entries, got {count}")
        return list(arguments)
    element = next((a for a in reversed(arguments) if a is not Ellipsis), None)
    return [element] * count


def wants_tuple(annotation: object) -> bool:
    """Whether a container annotation declares a tuple, which a record's
    list becomes; JSON has no tuple, and a frozen value with a list in a
    tuple field is unhashable and unequal to the one that was written."""
    annotation = _unwrapped(annotation)
    return annotation is tuple or typing.get_origin(annotation) is tuple


def from_record(annotation: object, value: Any) -> Any:
    """`value` as its annotation asks for it: a record becomes the value it
    describes, and anything already built is left alone.

    Containers are walked, so a mapping of records and a tuple of records
    build their values too, and a model config is a dict from the command
    line all the way to the module.
    """
    if typing.get_origin(annotation) in (Union, types.UnionType) and _unwrapped(annotation) is None:
        return value
    if isinstance(value, Mapping):
        held = _value_type(annotation)
        if held is None:
            # A record with no value class behind it, such as one of the unets'
            # per-stage attention settings: entries are walked and a "dtype"
            # entry resolves the same way as a dtype field.
            entries = entry_types(annotation, len(value))
            return {key: resolve_dtype(item) if key == "dtype" else from_record(entry, item)
                    for entry, (key, item) in zip(entries, value.items())}
        declared = sorted(f.name for f in dataclasses.fields(held) if f.init)
        unknown = sorted(set(value) - set(declared))
        if unknown:
            raise ValueError(f"{_describe(held)} has no field for {unknown}; its "
                             f"fields are {declared}")
        return held(**{key: _field_value(held, key, item)
                       for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        entries = entry_types(annotation, len(value))
        items = [from_record(entry, item) for entry, item in zip(entries, value)]
        return tuple(items) if wants_tuple(annotation) else type(value)(items)
    return value


def _field_value(member: Any, field: str, value: Any) -> Any:
    """One field on its way into `member`: a dtype from its name, a value from a record."""
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
    UNets keep per-stage attention settings in `attention_configs`, which do
    not inherit the model dtype and default `force_fp32_for_softmax` off,
    which no fused kernel can honour, so the knobs reach into them.

    A stage arrives either way: as a record, whose `dtype` name `build`
    resolves at the boundary with every other field, or as a built `Stage`,
    which nothing resolves afterwards, so its dtype is resolved here. The two
    agree once built, which `tests/test_models.py` asserts.
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
        fields["attention_configs"] = [
            None if stage is None
            else {**stage, "dtype": dtype, "force_fp32_for_softmax": True}
            if isinstance(stage, Mapping)
            else dataclasses.replace(stage, dtype=resolve_dtype(dtype),
                                     force_fp32_for_softmax=True)
            for stage in config.get("attention_configs", stages.default)]
    return fields


models: Registry[type[nn.Module]] = Registry("model")
presets: Registry[type[Preset]] = Registry("preset")
samplers: Registry[type[Solver[Any]]] = Registry("sampler")
datasets: Registry[type[DatasetSpec]] = Registry("dataset")
encoders: Registry[type[ConditionEncoder[Any]]] = Registry("encoder")
metrics: Registry[Callable[..., Metric]] = Registry("metric")
objectives: Registry[type[Objective]] = Registry("objective")
mixers: Registry[type[MixerBase]] = Registry("mixer", record="kind")
towers: Registry[type[TowerBase]] = Registry("tower", record="kind")
projectors: Registry[type[ProjectorBase]] = Registry("projector", record="kind")

# Core records nest their fields under a name; model component records inline
# their fields beside a kind discriminator read by the component's constructor.
REGISTRIES = (models, presets, samplers, datasets, encoders, metrics, objectives,
              mixers, towers, projectors)

__all__ = [
    "Registry", "models", "presets", "samplers", "datasets", "encoders", "metrics", "objectives",
    "mixers", "towers", "projectors", "REGISTRIES",
    "resolve_dtype", "dtype_name", "with_precision",
]
