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
import functools
import operator
import sys
import types
import typing
from collections.abc import Iterator, Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypedDict, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike
from typing_extensions import Format, get_annotations

from dew.records import JSON

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from flax import linen as nn

    from dew.data.dataset import DatasetSpec
    from dew.diffusion.presets import Preset
    from dew.inputs.encoders import ConditionEncoder
    from dew.nn.mixers import MixerBase
    from dew.nn.vision import ProjectorBase, TowerBase
    from dew.objectives.base import Metric, Objective
    from dew.sampling.solvers import Solver

T = TypeVar("T", bound=Callable[..., Any])
M = TypeVar("M", bound=Callable[..., Any])
Built = TypeVar("Built")
"""What calling a member builds: the module a model name builds, the spec a
dataset name builds. A registry is generic over both, since the table holds
the callable and `build` hands back what it returned."""

# A type annotation as a value: a class, a union of them, a subscripted
# generic, a PEP 695 alias, or the None a field with no resolvable annotation
# leaves behind. Every reader below takes one of these and asks it what it is.
type Annotation = type | types.UnionType | types.GenericAlias | typing.TypeAliasType | None

# One field of a member, as a caller writes it or a config file carries it:
# the JSON a file holds, a dtype, an enum member such as a matmul precision,
# an array a schedule was tabulated into, a callable a field takes as a hook,
# a value class already built (every value class here is a dataclass, a Flax
# module included), and the records and sequences of any of those. This is
# the width of what may arrive; `build` narrows each one again against the
# type its member declared before the member ever sees it.
type Configured = (JSON | DTypeLike | Enum | np.ndarray | np.generic
                   | types.FunctionType | types.BuiltinFunctionType
                   | DataclassInstance | Mapping[str, object]
                   | Mapping[str | tuple[str, ...], object] | Sequence[Configured])

# `build` called with no record at all, which is every caller that writes its
# fields as keywords. Shared because it is read and never written.
NO_RECORD: Mapping[str, object] = types.MappingProxyType({})


class Registry(Mapping[str, T], Generic[T, Built]):
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
        if type(name) is not str or not name:
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

    def name_of[Made](self, member: Callable[..., Made]) -> str:
        """The name a member was registered under. The table is scanned by
        identity, so this takes a member of any registry, whatever it makes."""
        for name, held in self._members.items():
            if held is member:
                return name
        raise KeyError(f"{_describe(member)} is not a registered {self.kind}")

    def build(self, name: str, record: Mapping[str, object] = NO_RECORD, /,
              **fields: Configured) -> Built:
        """Construct the member called `name` from a record, keyword fields, or both.

        A field the member does not declare is an error. Fields arrive from
        JSON as often as from code, so a field whose declared type is a value
        builds from a record here, where a logged config becomes an object:
        `models.build("m", attention={"heads": 8})` and
        `models.build("m", attention=Attention(heads=8))` agree.

        A whole parsed config is the positional `record`: its values are
        unnarrowed, and narrowing them against the member's declared types is
        this method's job, so the splat happens here rather than at a caller
        that would have to know the member's fields to write it.
        """
        member = self[name]
        return member(**self._declared_fields(name, member, {**record, **fields}))

    def _declared_fields(self, name: str, member: Callable[..., Built],
                         fields: Mapping[str, object]) -> Mapping[str, object]:
        """`fields` as the member declares them, or an error naming what it
        has no field for. A member that is not a dataclass takes them as given."""
        if not (isinstance(member, type) and dataclasses.is_dataclass(member)):
            return fields
        declared = {f.name for f in dataclasses.fields(member) if f.init}
        unknown = sorted(set(fields) - declared)
        if unknown:
            raise ValueError(
                f"{self.kind} {name!r} ({_describe(member)}) has no field for "
                f"{unknown}; its fields are {sorted(declared)}")
        return {key: resolve_dtype(value) if key == "dtype"
                else _rebuilt(_declared_type(member, key), value)
                for key, value in fields.items()}

    @property
    def union(self) -> type[Built] | types.UnionType:
        """`Union[...]` of the members, for a tyro subcommand over the table."""
        members = list(self._members.values())
        if not members:
            raise ValueError(f"the {self.kind} registry is empty")
        return functools.reduce(operator.or_, members)


def _describe[Made](member: Callable[..., Made]) -> str:
    return getattr(member, "__name__", repr(member))


def _declared_type(member: type, field: str) -> Annotation:
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
            break
    return None


def _value_type(annotation: Annotation) -> type | None:
    """A dataclass type behind an Optional, but not a multi-member union."""
    annotation = _unwrapped(annotation)
    return (annotation if isinstance(annotation, type) and dataclasses.is_dataclass(annotation)
            else None)


def resolve_alias(annotation: Annotation) -> Annotation:
    """A PEP 695 alias looked through to the type it declares. `get_origin`
    and `get_args` see nothing through one, so every reader goes through here
    before asking an annotation what it is."""
    while isinstance(annotation, typing.TypeAliasType):
        annotation = annotation.__value__
    return annotation


def _unwrapped(annotation: Annotation) -> Annotation:
    """`annotation` with an Optional looked through; a union of several
    members says nothing about its entries and answers None."""
    annotation = resolve_alias(annotation)
    if typing.get_origin(annotation) not in (Union, types.UnionType):
        return annotation
    members = [a for a in typing.get_args(annotation) if a is not type(None)]
    return members[0] if len(members) == 1 else None


def entry_types(annotation: Annotation, count: int) -> list[Annotation]:
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


def wants_tuple(annotation: Annotation) -> bool:
    """Whether the annotation declares an immutable sequence. A record's
    list is rebuilt as a tuple so the frozen value stays hashable; `list`
    and `MutableSequence` keep their list."""
    annotation = _unwrapped(annotation)
    return (annotation in (tuple, Sequence)
            or typing.get_origin(annotation) in (tuple, Sequence))


def from_record[ValueT](annotation: type[ValueT], value: Configured) -> ValueT:
    """`value` as the class `annotation` names, built from a record or already one.

    The class is the witness: what comes back is an instance of it or a
    `ValueError` naming what the record built instead, so a caller reads a
    value of the type it asked for rather than one it has to narrow again.
    `_rebuilt` is the same walk over an annotation that is not a class -- a
    union, a generic, an alias -- which only this module's own recursion has.
    """
    built = _rebuilt(annotation, value)
    if not isinstance(built, annotation):
        raise ValueError(f"{value!r} builds {_describe(type(built))}, "
                         f"not the {_describe(annotation)} the field declares")
    return built


def _rebuilt(annotation: Annotation, value: object) -> Configured:
    """`value` as its annotation asks for it: a record becomes the value it
    describes, and anything already built is left alone.

    Containers are walked, so a mapping of records and a tuple of records
    build their values too, and a model config is a dict from the command
    line all the way to the module.
    """
    annotation = resolve_alias(annotation)
    if typing.get_origin(annotation) in (Union, types.UnionType) and _unwrapped(annotation) is None:
        return configured(value)
    if isinstance(value, Mapping):
        held = _value_type(annotation)
        if held is None:
            # A record with no value class behind it, such as one of the unets'
            # per-stage attention settings: entries are walked and a "dtype"
            # entry resolves the same way as a dtype field.
            entries = entry_types(annotation, len(value))
            return {key: resolve_dtype(record) if key == "dtype" else _rebuilt(entry, record)
                    for entry, (key, record) in zip(entries, value.items(), strict=True)}
        declared = sorted(f.name for f in dataclasses.fields(held) if f.init)
        unknown = sorted(set(value) - set(declared))
        if unknown:
            raise ValueError(f"{_describe(held)} has no field for {unknown}; its "
                             f"fields are {declared}")
        return held(**{key: resolve_dtype(record) if key == "dtype"
                       else _rebuilt(_declared_type(held, key), record)
                       for key, record in value.items()})
    if isinstance(value, (list, tuple)):
        entries = entry_types(annotation, len(value))
        rebuilt = [_rebuilt(entry, record)
                   for entry, record in zip(entries, value, strict=True)]
        return tuple(rebuilt) if wants_tuple(annotation) else type(value)(rebuilt)
    return configured(value)


def configured(value: object) -> Configured:
    """One value a record carried, handed back as a field holds it.

    Nothing is converted here: this is the one place that says what a field
    can carry at all, so a value no member could take is refused where the
    record was read instead of inside the module that would have used it.
    """
    if value is None or isinstance(value, (bool, int, float, str, bytes, type, Enum, np.dtype)):
        return value
    if isinstance(value, (np.ndarray, np.generic, jax.Array)):
        return value
    if dataclasses.is_dataclass(value) or isinstance(value, (Mapping, Sequence)) or callable(value):
        return value
    raise ValueError(f"{value!r} is not a value a member field carries")


DtypeName = Literal["float32", "bfloat16", "float16"]
_DTYPES: dict[DtypeName, DTypeLike] = {
    "float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}


def resolve_dtype(value: object) -> DTypeLike | None:
    """A dtype as a module field: a jnp dtype, one of its names, or None.

    Every field named `dtype` is read here, wherever it arrives from, so a
    dtype passes through and a name becomes the dtype it names. Anything
    else is refused here rather than inside a module's first cast.
    """
    if value is None:
        return None
    if isinstance(value, str):
        for name, dtype in _DTYPES.items():
            if value == name:
                return dtype
        raise ValueError(f"dtype {value!r} is not one of {sorted(_DTYPES)}")
    if isinstance(value, (type, np.dtype)):
        return value
    raise ValueError(
        f"dtype {value!r} is not a dtype, nor one of {sorted(_DTYPES)}")


def dtype_name(value: DTypeLike | None) -> DtypeName | None:
    """The name `resolve_dtype` accepts for a dtype, for a logged config."""
    if value is None:
        return None
    for name, dtype in _DTYPES.items():
        if jnp.dtype(value) == jnp.dtype(dtype):
            return name
    raise ValueError(f"{value!r} is not a dtype a config can name")


# The flag each precision field is set by, for the error a model config that
# carries one of them raises.
_PRECISION_FLAGS = {"dtype": "--model.dtype", "attention_impl": "--model.attention-impl",
                    "param_dtype": "--model.param-dtype",
                    "precision": "--model.matmul-precision"}


class PrecisionFields(TypedDict, total=False):
    """What a run's precision settings write into a model config.

    Only the keys the named member declares are written, so the bag is
    partial by construction; `attention_configs` is the UNets' per-stage
    settings, which carry the dtype into each stage a config named. Its
    entries are a stage record, a built `Stage` or None, which is what a
    model config carries and what `build` narrows against the field.
    """

    dtype: str
    attention_impl: str
    param_dtype: str
    precision: str
    attention_configs: list[object]


def precision_fields(name: str, config: Mapping[str, object], *,
                     dtype: str, attention_impl: str, param_dtype: str | None = None,
                     matmul_precision: str | None = None) -> PrecisionFields:
    """The run's compute dtype and attention kernel as the fields a model takes.

    `with_precision` is the same settings merged into the config they belong
    to; this is them on their own, for a caller holding a typed field bag.

    `attention_impl` is an `AttentionImpl` and travels as it is written: the
    kernel reads 'reference' as the reference path, so nothing here rewrites
    the name a run recorded into the None a module field also accepts.

    `dtype` is the compute dtype; `param_dtype` is where the parameters are
    stored and `matmul_precision` what every matmul asks XLA for. Those two
    reach the model only where it declares the field (`param_dtype`,
    `precision`), so a model that declares neither takes neither and a run
    that names neither writes neither. Unset, parameters stay float32 and
    the model keeps its own precision, which is what every run did before
    the fields existed.

    The UNets keep per-stage attention settings in `attention_configs`,
    which do not inherit the model dtype and default `force_fp32_for_softmax`
    off, which no fused kernel can honour, so the knobs reach into them.

    A stage arrives either way: as a record, whose `dtype` name `build`
    resolves at the boundary with every other field, or as a built `Stage`,
    which nothing resolves afterwards, so its dtype is resolved here. The two
    agree once built, which `tests/test_models.py` asserts.
    """
    member = models[name]
    declared = {f.name for f in dataclasses.fields(member) if f.init}
    written: PrecisionFields = {"dtype": dtype, "attention_impl": attention_impl}
    if param_dtype is not None and "param_dtype" in declared:
        written["param_dtype"] = param_dtype
    if matmul_precision is not None and "precision" in declared:
        written["precision"] = matmul_precision
    duplicate = sorted(set(config) & set(written))
    if duplicate:
        raise ValueError(
            f"the model config carries {duplicate}, which the run's precision "
            f"settings own; set {', '.join(_PRECISION_FLAGS[held] for held in duplicate)} instead")
    fields: PrecisionFields = {**written}
    stages = {f.name: f for f in dataclasses.fields(member)}.get("attention_configs")
    if stages is not None:
        carried = config.get("attention_configs", stages.default)
        if not isinstance(carried, (list, tuple)):
            raise ValueError(
                f"attention_configs is {carried!r}; a unet takes one entry per "
                f"resolution stage, each a record, a Stage, or None for a stage "
                f"that does not attend")
        resolved: list[object] = []
        for stage in carried:
            if stage is None:
                resolved.append(None)
            elif isinstance(stage, Mapping):
                resolved.append({**stage, "dtype": dtype, "force_fp32_for_softmax": True})
            elif dataclasses.is_dataclass(stage) and not isinstance(stage, type):
                resolved.append(dataclasses.replace(
                    stage, dtype=resolve_dtype(dtype), force_fp32_for_softmax=True))
            else:
                raise ValueError(
                    f"attention_configs carries {stage!r}; a stage is a record or "
                    f"a Stage")
        fields["attention_configs"] = resolved
    return fields


def with_precision(name: str, config: Mapping[str, object], *,
                   dtype: str, attention_impl: str, param_dtype: str | None = None,
                   matmul_precision: str | None = None) -> Mapping[str, object]:
    """A model config with the run's compute dtype and attention kernel in it."""
    return {**config, **precision_fields(
        name, config, dtype=dtype, attention_impl=attention_impl,
        param_dtype=param_dtype, matmul_precision=matmul_precision)}


models: Registry[type[nn.Module], nn.Module] = Registry("model")
presets: Registry[type[Preset], Preset] = Registry("preset")
samplers: Registry[type[Solver[Any]], Solver[Any]] = Registry("sampler")
datasets: Registry[type[DatasetSpec], DatasetSpec] = Registry("dataset")
encoders: Registry[type[ConditionEncoder[Any]], ConditionEncoder[Any]] = Registry("encoder")
metrics: Registry[Callable[..., Metric], Metric] = Registry("metric")
objectives: Registry[type[Objective], Objective] = Registry("objective")
mixers: Registry[type[MixerBase], MixerBase] = Registry("mixer", record="kind")
towers: Registry[type[TowerBase], TowerBase] = Registry("tower", record="kind")
projectors: Registry[type[ProjectorBase], ProjectorBase] = Registry("projector", record="kind")

# Core records nest their fields under a name; model component records inline
# their fields beside a kind discriminator read by the component's constructor.
REGISTRIES = (models, presets, samplers, datasets, encoders, metrics, objectives,
              mixers, towers, projectors)

__all__ = ["REGISTRIES", "Registry", "datasets", "dtype_name", "encoders", "metrics", "mixers", "models",
           "objectives", "presets", "projectors", "resolve_dtype", "samplers", "towers", "with_precision"]
