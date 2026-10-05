"""Name the things a run is made of.

One Registry per kind, including model components such as mixers, towers and
projectors. A registry is the record layer: config files, the CLI and run
records name a member, and the registry maps that name to its class and back,
so `models["simple_dit"]` is `SimpleDiT`. Code builds the class itself. A
name or a field the table does not know raises.

The registries are empty at import. Each member registers itself where it is
defined, so importing a package fills its table and the registry module
imports none of them. A lookup of a name its table does not hold yet imports
the modules whose decorator registers that name, read off the sources
(`_registering_modules`), so a record loads in a process that imported
nothing beforehand, and nothing else is imported.

The sources read are Dew's own and those of every installed plugin: a
package that registers members into these tables names itself under the
`dew.plugins` entry-point group,

    [project.entry-points."dew.plugins"]
    sparx = "sparx"

and its modules are read the same way, without importing it, so a run
recorded with a plugin's model loads as a run of Dew's own does.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import importlib.metadata
import importlib.util
import operator
import re
import sys
import types
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, TypeVar, Union, overload

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
    from dew.training.optim import ScheduleBase

M = TypeVar("M", bound=Callable[..., Any])
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


_DECORATOR = re.compile(r"""^[ \t]*@(?:dew\.)?(?:registry\.)?(\w+)\(\s*["']([^"']+)["']\s*\)[ \t]*$""", re.M)
"""A registration as Dew's sources write it, `@models("simple_dit")` or
`@registry.objectives("lm")`: the registry's name in this module and the
member's."""


PLUGINS = "dew.plugins"
"""The entry-point group a package names itself under to register into Dew's tables."""


def _plugin_roots() -> list[Path]:
    """The package directory of each installed `dew.plugins` entry, found without importing it.

    An entry names a top-level package. One that is declared but cannot be
    found is a broken install, and raises with the entry that declared it.
    """
    roots = []
    for entry in sorted(importlib.metadata.entry_points(group=PLUGINS), key=lambda entry: entry.name):
        spec = importlib.util.find_spec(entry.value)
        if spec is None or not spec.submodule_search_locations:
            raise ImportError(f"the {PLUGINS} entry {entry.name!r} names {entry.value!r}, "
                              "which is not an importable package")
        roots.extend(Path(location) for location in spec.submodule_search_locations)
    return roots


@functools.cache
def _registering_modules() -> Mapping[tuple[str, str], tuple[str, ...]]:
    """Each `(registry, name)` Dew's and its plugins' sources register, and the modules that do.

    The decorators are the one statement of what registers where, so this
    reads them rather than keeping a second table: about 270 of Dew's files
    in 20 ms, once a process, and only when a lookup misses."""
    found: dict[tuple[str, str], list[str]] = {}
    for root in [Path(__file__).parent, *_plugin_roots()]:
        for path in sorted(root.rglob("*.py")):
            parts = path.relative_to(root.parent).with_suffix("").parts
            module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
            for attribute, name in _DECORATOR.findall(path.read_text()):
                found.setdefault((attribute, name), []).append(module)
    return {key: tuple(modules) for key, modules in found.items()}


class Registry[T: Callable[..., Any], Built](Mapping[str, T]):
    """Names one kind of thing: a decorator and a mapping from name to member.

    A table is held by a module attribute named for its kind plus "s", as
    `models` holds the `model` table, and that name is how its decorators read
    in sources. A plugin package makes a kind of its own the same way, in a
    module of its own, and passes it to `share` so records rebuild its members.
    """

    def __init__(self, kind: str, *, record: Literal["name", "kind"] = "name"):
        self.kind = kind
        self.attribute = f"{kind}s"
        self.record = record
        # A decorator has no base class to test the member against, and it
        # hands back the class it decorated so a caller's checker keeps the
        # concrete type (`DiffusionObjective`, not `Objective`). The table is
        # untyped here and typed on the way out.
        self._members: dict[str, Any] = {}

    def __call__(self, name: str, /) -> Callable[[M], M]:
        """Register the class it names, as `@models("simple_dit")`."""
        if type(name) is not str or not name:
            raise TypeError(f"a {self.kind} name is a non-empty string, not {name!r}")

        def register(member: M) -> M:
            held = self._members.get(name)
            if held is not None and held is not member:
                raise ValueError(
                    f"{self.kind} {name!r} is already {held.__name__}; "
                    f"a name maps to one {self.kind}")
            self._members[name] = member
            return member

        return register

    def __getitem__(self, name: str) -> T:
        if name not in self._members:
            for module in self._registering(name):
                importlib.import_module(module)
        try:
            return self._members[name]
        except KeyError:
            known = set(self._members) | {held for attribute, held in _registering_modules()
                                          if attribute == self.attribute}
            raise KeyError(f"no {self.kind} named {name!r}; known: {', '.join(sorted(known))}") from None

    def _registering(self, name: str) -> tuple[str, ...]:
        """The modules of Dew or a plugin whose decorator registers `name` in this table."""
        return tuple(module for (attribute, held), modules in _registering_modules().items()
                     if held == name and attribute == self.attribute
                     for module in modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._members)

    def __len__(self) -> int:
        return len(self._members)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {sorted(self._members)})"

    def name_of(self, member: Named) -> str:
        """Return the name a member was registered under. The table is scanned by
        identity, so this takes a member of any registry, whatever it makes."""
        for name, held in self._members.items():
            if held is member:
                return name
        raise KeyError(f"{member.__name__} is not a registered {self.kind}; register it once with "
                       f"`@dew.registry.{self.kind}s(\"{member.__name__.lower()}\")` above its class")

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

    def from_record(self, record: Mapping[str, object]) -> Built:
        """Construct the member a `{"kind": ..., **fields}` record names.

        A config writes a mixer, a tower or a projector this way where code
        passes the value `build` makes, so the two meet here. A record that
        names no registered kind raises ValueError, with the known ones.
        """
        fields = dict(record)
        kind = fields.pop("kind", None)
        if not isinstance(kind, str) or kind not in self:
            raise ValueError(
                f"a {self.kind} record names its kind, one of "
                f"{', '.join(sorted(self._members))}; got {kind!r}")
        return self.build(kind, fields)

    def _declared_fields(self, name: str, member: Callable[..., Built],
                         fields: Mapping[str, object]) -> Mapping[str, object]:
        """Return `fields` as the member declares them, or raise naming what it
        has no field for. A member that is not a dataclass takes them as given."""
        if not (isinstance(member, type) and dataclasses.is_dataclass(member)):
            return fields
        declared = {f.name for f in dataclasses.fields(member) if f.init}
        unknown = sorted(set(fields) - declared)
        if unknown:
            raise ValueError(
                f"{self.kind} {name!r} ({member.__name__}) has no field for "
                f"{unknown}; its fields are {sorted(declared)}")
        return {key: resolve_dtype(value) if key == "dtype"
                else _rebuilt(_declared_type(member, key), value)
                for key, value in fields.items()}

    @property
    def union(self) -> type[Built] | types.UnionType:
        """Return `Union[...]` of the members, for a tyro subcommand over the table."""
        return functools.reduce(operator.or_, self._members.values())


class Named(Protocol):
    """Declares the name a registry member carries of its own.

    A member is a class or a function -- the decorator takes both -- and each
    declares `__name__`, which is what an error names a member by. The registry's table holds members as the
    concrete type their decorator handed back, so this is the one thing read
    off them without the caller's own type.
    """

    @property
    def __name__(self) -> str: ...


def _declared_type(member: type, field: str) -> Annotation:
    """Resolve one field's annotation, without evaluating unrelated ones."""
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
    """Return a dataclass type behind an Optional, but not a multi-member union."""
    annotation = _unwrapped(annotation)
    return (annotation if isinstance(annotation, type) and dataclasses.is_dataclass(annotation)
            else None)


def resolve_alias(annotation: Annotation) -> Annotation:
    """Look a PEP 695 alias through to the type it declares. `get_origin`
    and `get_args` see nothing through one, so every reader goes through here
    before asking an annotation what it is."""
    while isinstance(annotation, typing.TypeAliasType):
        annotation = annotation.__value__
    return annotation


def _unwrapped(annotation: Annotation) -> Annotation:
    """Return `annotation` with an Optional looked through. A union of several
    members says nothing about its entries and answers None."""
    annotation = resolve_alias(annotation)
    if typing.get_origin(annotation) not in (Union, types.UnionType):
        return annotation
    members = [a for a in typing.get_args(annotation) if a is not type(None)]
    return members[0] if len(members) == 1 else None


def entry_types(annotation: Annotation, count: int) -> list[Annotation]:
    """Return the annotation of each of the `count` entries of an annotated
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
    """Return whether the annotation declares an immutable sequence. A record's
    list is rebuilt as a tuple so the frozen value stays hashable; `list`
    and `MutableSequence` keep their list."""
    annotation = _unwrapped(annotation)
    return (annotation in (tuple, Sequence)
            or typing.get_origin(annotation) in (tuple, Sequence))


def from_record[ValueT](annotation: type[ValueT], value: Configured) -> ValueT:
    """Return `value` as the class `annotation` names, from a record or already one.

    The class is the witness: what comes back is an instance of it or a
    `ValueError` naming what the record built instead, so a caller reads a
    value of the type it asked for rather than one it has to narrow again.
    `_rebuilt` is the same walk over an annotation that is not a class -- a
    union, a generic, an alias -- which only this module's own recursion has.
    """
    built = _rebuilt(annotation, value)
    if not isinstance(built, annotation):
        raise ValueError(f"{value!r} builds {type(built).__name__}, "
                         f"not the {annotation.__name__} the field declares")
    return built


def _rebuilt(annotation: Annotation, value: object) -> Configured:
    """Return `value` as its annotation asks for it: a record becomes the value it
    describes, and anything already built is left alone.

    Containers are walked, so a mapping of records and a tuple of records
    build their values too, and a model config is a dict from the command
    line all the way to the module.

    `dew.config._rebuild` is the sibling walk over a run record. It reads a
    registered member out of its `kind`/`name` record and honours the
    `record: False` field metadata, neither of which a module field has; this
    one resolves a `dtype` entry and walks a record with no value class.
    """
    annotation = resolve_alias(annotation)
    if typing.get_origin(annotation) in (Union, types.UnionType) and _unwrapped(annotation) is None:
        return configured(value)
    if isinstance(value, Mapping):
        held = _value_type(annotation)
        named = _unwrapped(annotation)
        if isinstance(named, type):
            member, fields = _nested_member(named, value)
            if fields is not value:
                held, value = member, fields
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
            raise ValueError(f"{held.__name__} has no field for {unknown}; its "
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


def _nested_member(held: type, value: Mapping[str, object]) -> tuple[type, Mapping[str, object]]:
    """The class and fields a field's record names. A registered part of a
    composite model is recorded as its registry writes it, `{"name": ...,
    "fields": {...}}` or `{"kind": ..., **fields}` (`dew.config._to_json`), so
    that record builds the member it names when that member is the field's
    class or a subclass; any other record is the field class's own fields."""
    for table in REGISTRIES:
        if table.record == "name":
            named, fields = value.get("name"), value.get("fields")
            if set(value) != {"name", "fields"} or not isinstance(fields, Mapping):
                continue
        else:
            named, fields = value.get("kind"), {key: field for key, field in value.items() if key != "kind"}
        member = table.get(named) if isinstance(named, str) else None
        if isinstance(member, type) and issubclass(member, held):
            return member, fields
    return held, value


def configured(value: object) -> Configured:
    """Return one value a record carried, as a field holds it.

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
    """Resolve a dtype for a module field: a jnp dtype, one of its names, or None.

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


@overload
def dtype_name(value: DTypeLike) -> DtypeName: ...
@overload
def dtype_name(value: None) -> None: ...
def dtype_name(value: DTypeLike | None) -> DtypeName | None:
    """Return the name `resolve_dtype` accepts for a dtype, for a logged config.

    A loader takes a dtype, `jnp.bfloat16`, or its name, and records the name."""
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
    """Names what a run's precision settings write into a model config.

    Only the keys the named member declares are written, so the bag is
    partial by construction; `attention_configs` is the UNets' per-stage
    settings, which carry the dtype into each stage a config named. Its
    entries are a stage record, a built `Stage` or None, which is what a
    model config carries and what `build` narrows against the field.
    """

    dtype: str | None
    attention_impl: str
    param_dtype: str
    precision: str
    attention_configs: list[object]


def precision_fields(name: str, config: Mapping[str, object], *,
                     dtype: str | None, attention_impl: str, param_dtype: str | None = None,
                     matmul_precision: str | None = None) -> PrecisionFields:
    """Return the run's compute dtype and attention kernel as the fields a model takes.

    `with_precision` is the same settings merged into the config they belong
    to; this is them on their own, for a caller holding a typed field bag.

    `attention_impl` is an `AttentionImpl` and travels as it is written; the
    kernel resolves 'auto' against each call, so the name a run recorded is
    the name the model holds.

    `dtype` is the compute dtype; `param_dtype` is where the parameters are
    stored and `matmul_precision` what every matmul asks XLA for. Those two
    reach the model only where it declares the field (`param_dtype`,
    `precision`), so a model that declares neither takes neither and a run
    that names neither writes neither. Unset, parameters stay float32 and the
    model keeps its own precision.

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
    written: PrecisionFields = {}
    if "dtype" in declared:
        written["dtype"] = dtype
    if "attention_impl" in declared:
        written["attention_impl"] = attention_impl
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


def float64_twin(config: Mapping[str, object]) -> Mapping[str, object]:
    """`config`, a model config `with_precision` wrote, computing in float64:
    its dtype and the dtype `precision_fields` wrote into each stage of the
    UNets' `attention_configs`, the only nested dtypes the policy writes.
    Under x64 the model computes in float64 throughout
    (`dew.nn.precision.at_least_fp32`): the twin a float64 reference runs. A
    run's dtype knob names no float64; only a reference computes in it."""
    twin: dict[str, object] = {**config, "dtype": jnp.float64}
    stages = config.get("attention_configs")
    if isinstance(stages, list | tuple):
        twin["attention_configs"] = [
            {**stage, "dtype": jnp.float64} if isinstance(stage, Mapping)
            else dataclasses.replace(stage, dtype=jnp.float64)
            if dataclasses.is_dataclass(stage) and not isinstance(stage, type) else stage
            for stage in stages]
    return twin


def with_precision(name: str, config: Mapping[str, object], *,
                   dtype: str | None, attention_impl: str, param_dtype: str | None = None,
                   matmul_precision: str | None = None) -> Mapping[str, object]:
    """Return a model config with the run's compute dtype and attention kernel in it.

    A composite that declares no `dtype` of its own, DiffusionGemma around its
    text decoder, passes the run's compute dtype to the registered model parts
    its record nests, which own it."""
    fields = {**config, **precision_fields(
        name, config, dtype=dtype, attention_impl=attention_impl,
        param_dtype=param_dtype, matmul_precision=matmul_precision)}
    if dtype is None or "dtype" in {field.name for field in dataclasses.fields(models[name])}:
        return fields
    return {key: _with_part_dtype(configured(value), dtype) for key, value in fields.items()}


def _with_part_dtype(value: Configured, dtype: str) -> Configured:
    """A composite's part with the run's compute dtype, where the part is a
    registered model, recorded or built, that declares one."""
    from flax import linen as nn

    if isinstance(value, nn.Module):
        owns = type(value) in models.values() and "dtype" in {f.name for f in dataclasses.fields(value)}
        return value.clone(dtype=resolve_dtype(dtype)) if owns else value
    if not isinstance(value, Mapping) or set(value) != {"name", "fields"}:
        return value
    name, fields = value["name"], value["fields"]
    if (isinstance(name, str) and name in models and isinstance(fields, Mapping)
            and "dtype" in {field.name for field in dataclasses.fields(models[name])}):
        return {"name": name, "fields": {**fields, "dtype": dtype}}
    return value


models: Registry[type[nn.Module], nn.Module] = Registry("model")
presets: Registry[type[Preset], Preset] = Registry("preset")
solvers: Registry[type[Solver[Any]], Solver[Any]] = Registry("solver")
datasets: Registry[type[DatasetSpec], DatasetSpec] = Registry("dataset")
encoders: Registry[type[ConditionEncoder[Any]], ConditionEncoder[Any]] = Registry("encoder")
metrics: Registry[Callable[..., Metric], Metric] = Registry("metric")
objectives: Registry[type[Objective], Objective] = Registry("objective")
mixers: Registry[type[MixerBase], MixerBase] = Registry("mixer", record="kind")
towers: Registry[type[TowerBase], TowerBase] = Registry("tower", record="kind")
projectors: Registry[type[ProjectorBase], ProjectorBase] = Registry("projector", record="kind")
schedules: Registry[type[ScheduleBase], ScheduleBase] = Registry("schedule", record="kind")

# Core records nest their fields under a name; model component records inline
# their fields beside the kind discriminator `Registry.from_record` reads.
REGISTRIES: list[Registry[Any, Any]] = [models, presets, solvers, datasets, encoders, metrics, objectives,
                                        mixers, towers, projectors, schedules]
"""The tables a record names members of: Dew's, then those its plugins `share`."""


def share[T: Registry[Any, Any]](table: T) -> T:
    """Add a plugin's table to `REGISTRIES` and return it, as
    `activations = share(Registry("activation", record="kind"))`.

    Records then rebuild its members wherever a field declares their base
    class, and write them back as their kind. A kind has one shared table;
    sharing the same table again is harmless.
    """
    for held in REGISTRIES:
        if held.kind == table.kind and held is not table:
            raise ValueError(f"a {table.kind} registry is already shared; a kind has one table")
    # By identity: a table is a Mapping, and `in` would call two empty
    # tables of different kinds equal.
    if not any(held is table for held in REGISTRIES):
        REGISTRIES.append(table)
    return table

__all__ = [
    "REGISTRIES",
    "Registry",
    "datasets",
    "encoders",
    "metrics",
    "mixers",
    "models",
    "objectives",
    "presets",
    "projectors",
    "schedules",
    "share",
    "solvers",
    "towers",
    "with_precision",
]
