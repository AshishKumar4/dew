"""A model's record: its class and the fields a run gives it (`ModelConfig`), and the base of every
registered class a run records the same way (`Registered`, which a run's objective and metrics take too).
It sits below the objectives, the loaders and the trainer, which write and read it, and imports nothing of
theirs: a model's wrappers are any `dew.registry.Wrapper`, each recorded by its class path."""

from __future__ import annotations

import dataclasses
import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import Annotated, ClassVar, Self

import tyro

from dew import registry
from dew._records import JSON
from dew.registry import (
    Annotation,
    Configured,
    Wrapper,
    declared_type,
    from_record,
    models,
    parameter_type,
    to_record,
)

JsonDict = Annotated[
    Mapping[str, object],
    tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="JSON",
        instance_from_str=lambda args: json.loads(args[0]),
        is_instance=lambda value: isinstance(value, dict),
        str_from_instance=lambda value: [json.dumps(value)],
    ),
]
"""The fields a model is built from, written as a single JSON string on the
command line. The registry knows which architecture takes which field and
narrows each one where it builds it, so the values are read there."""


@dataclasses.dataclass(frozen=True)
class Registered(ABC):
    """A registered class, named by an alias or an import path and held as the path, and the arguments a
    run gives it: `fields`, the ones the run states, and `defaults`, every other one the class
    defaults as it defaulted them when the run was first recorded, so the record builds the same
    thing after a default changes (`arguments`). Each kind names its registry (`_member`) and whether
    its stated fields are held as their records (`_recorded`)."""

    name: str
    fields: JsonDict = dataclasses.field(default_factory=dict)
    defaults: JsonDict = dataclasses.field(default_factory=dict)
    """Each argument the class defaults and `fields` leaves out, as a record holds it;
    one a record lacks, as an argument the class has since gained, takes the class's default now."""

    _recorded: ClassVar[bool] = True

    @staticmethod
    @abstractmethod
    def _member(name: str) -> type:
        """The class `name` names, from the kind's registry."""

    def __post_init__(self) -> None:
        member = self._member(self.name)
        object.__setattr__(self, "name", registry.import_path(member))
        if self._recorded:
            object.__setattr__(self, "fields", registry.argument_records(member, self.fields))
        object.__setattr__(self, "defaults", recorded_defaults(member, self.fields, self.defaults))

    @property
    def arguments(self) -> JsonDict:
        """The arguments the class is built from, before those the caller derives (`built_arguments`)."""
        return built_arguments(self._member(self.name), self.fields, self.defaults)


@dataclasses.dataclass(frozen=True)
class ModelConfig(Registered):
    """A model's class and the fields the run gives it.

    `name` is an alias or an import path, held as the path. `fields` are the
    class's own fields, its compute and storage dtypes, attention kernel and
    matmul precision included, which the command line sets one flag each
    (`RunConfig.cli`); a recipe fills the ones it derives from its data in
    `build`. `defaults` holds every other field the class defaults, as it
    defaulted them when the run was first recorded, so the record builds the
    same model after a default changes (`arguments`). `wrappers` are the
    classes the model was wrapped in, in the order they were applied.
    """

    name: str = "simple_dit"
    wrappers: tuple[Wrapper, ...] = ()
    """What wraps the built model, innermost first, which `build` applies in
    order: a checkpoint's input quantization, a bound LoRA (its factors are
    stored with the checkpoint variables), the quantized training a run
    wrapped it in."""

    _recorded: ClassVar[bool] = False

    def __post_init__(self) -> None:
        super().__post_init__()
        stray = [type(wrapper).__name__ for wrapper in self.wrappers if not isinstance(wrapper, Wrapper)]
        if stray:
            raise TypeError(f"a model record's wrappers are dew.registry.Wrapper records, not {stray}")

    @staticmethod
    def _member(name: str) -> type:
        return models[name]

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Read back the record `RunConfig.to_dict` writes for this field."""
        return from_record(cls, values, dtypes=False)

    @classmethod
    def from_model(cls, model) -> Self:
        """Return the module's class and constructor fields, and each wrapper
        its class was made in, in the order they were applied."""
        model_type, layers = registry.wrapper_layers(type(model))
        return cls(registry.import_path(model_type), registry.record_fields(model, model_type),
                   wrappers=tuple(layer._dew_wrapper() for layer in reversed(layers)))

    def with_dtype(self, dtype: str | None) -> Self:
        """This model computing in `dtype` (`dew.registry.with_dtype`); None keeps it."""
        return dataclasses.replace(self, fields=registry.with_dtype(self.name, self.fields, dtype))

    def build(self, **derived: Configured):
        """Build the model from its fields and the ones the recipe `derived`."""
        model = models.build(self.name, {**self.arguments, **derived})
        for wrapper in self.wrappers:
            model = wrapper.apply(model)
        return model


def member_fields(member: type) -> Iterator[tuple[str, Configured, Annotation]]:
    """Each field `_member_flags` flags: its name, its default
    (`dataclasses.MISSING` for none) and its annotation (None where its
    module cannot resolve it). A dataclass with a constructor of its own (a
    metric over its base's fields) has that constructor's arguments, as the
    registry builds it (`registry.record_class`)."""
    if registry.record_class(member) is not None:
        for field in registry.constructor_fields(member):
            default = (field.default_factory() if callable(field.default_factory) else field.default)
            yield field.name, default, declared_type(member, field.name)
        return
    for name, (parameter, owner) in registry.parameters(member)[0].items():
        if parameter.kind is parameter.POSITIONAL_OR_KEYWORD and parameter.default is parameter.empty:
            continue  # what the caller builds, the model first
        try:
            annotation = parameter_type(owner, name)
        except ValueError:
            annotation = None
        default = dataclasses.MISSING if parameter.default is parameter.empty else parameter.default
        yield name, default, annotation


def default_record(name: str, declared: Configured, annotation: Annotation) -> JSON:
    """A member's default `declared` as a record holds it: a dtype by its
    name, a matmul precision by its level, anything else as `to_record`
    writes it. TypeError or ValueError for a value no record carries."""
    if name in ("dtype", "param_dtype"):
        return registry.dtype_name(registry.resolve_dtype(declared))
    if name == "precision":
        return None if declared is None else str(declared).lower().removeprefix("precision.")
    # The record itself, not a copy: a module a later default shares turns
    # this one into its defining record.
    record = to_record(declared, annotation)
    json.dumps(record)
    return record


def recorded_defaults(member: type, stated: Mapping[str, object], recorded: Mapping[str, object]) -> JsonDict:
    """The record of each field or argument `member` defaults that `stated`
    leaves out: as `recorded` holds it, or else as the class defaults it now.
    A default no record carries (a callable, say) is left to the class, and
    a recorded one the class no longer takes is refused."""
    defaults, named = {}, set()
    # One record: a module two defaults share is written once.
    with registry.recording():
        for name, declared, annotation in member_fields(member):
            named.add(name)
            if name in stated or name in recorded or declared is dataclasses.MISSING:
                continue
            try:
                defaults[name] = default_record(name, declared, annotation)
            except (TypeError, ValueError):
                continue
    unknown = sorted(set(recorded) - named)
    if unknown:
        raise ValueError(f"{member.__name__} takes no {unknown}, which the record defaults")
    return {**defaults, **{name: value for name, value in recorded.items() if name not in stated}}


def built_arguments(member: type, fields: Mapping[str, object], defaults: Mapping[str, object]) -> JsonDict:
    """What a config builds `member` from beside what its caller derives:
    `fields`, over each of `defaults` the class no longer defaults the same.
    A default the class still has is left to it, so a class that treats an
    argument left out in its own way, as `DPOObjective` holds its reference
    at unit decay, still does."""
    now = recorded_defaults(member, {}, {})
    return {**{name: value for name, value in defaults.items() if name not in now or now[name] != value},
            **fields}


__all__ = ["JsonDict", "ModelConfig", "Registered", "built_arguments", "default_record", "member_fields",
           "recorded_defaults"]
