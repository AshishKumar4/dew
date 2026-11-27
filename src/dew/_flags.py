"""A registered kind's records as a command line chooses them (api03): each member of a `Bound` kind a
subcommand (`record_flags`), its arguments plain flags (`_flat`), and the subcommand a field's default
already is (`implied_subcommand`). Apart from `dew.registry`, whose records they read."""

from __future__ import annotations

import dataclasses
import functools
import operator
import types
import typing
from collections.abc import Callable
from typing import Annotated, Literal, Union

import tyro

from dew import registry
from dew.registry import Annotation, Bound, JsonRecords, member_fields

_PLAIN = (int, float, str, bool)
"""The values a command line sets a model's field to by a plain flag (`_flat`)."""


def _flat(annotation: Annotation) -> Annotation | None:
    """The part of a field's `annotation` a command line sets by a plain value (`_PLAIN`, a `Literal` of
    them, a tuple of them, None): a union's such members, its others (a record, a module) left to the
    record; None where no member but None is one."""
    origin = typing.get_origin(annotation)
    if origin is Annotated:
        return _flat(typing.get_args(annotation)[0])
    if annotation in _PLAIN or annotation is type(None):
        return annotation
    if origin is Literal:
        return annotation if all(isinstance(each, _PLAIN) for each in typing.get_args(annotation)) else None
    if origin is tuple:
        parts = typing.get_args(annotation)
        return annotation if parts and all(part is Ellipsis or part in _PLAIN for part in parts) else None
    if origin in (Union, types.UnionType):
        kept = [part for member in typing.get_args(annotation) if (part := _flat(member)) is not None]
        return functools.reduce(operator.or_, kept) if any(part is not type(None) for part in kept) else None
    return None


def _fits(value: registry.Configured, annotation: Annotation) -> bool:
    """Whether `value` is one a flag of the flat `annotation` (`_flat`) holds."""
    origin = typing.get_origin(annotation)
    if origin in (Union, types.UnionType):
        return any(_fits(value, member) for member in typing.get_args(annotation))
    if origin is Literal:
        return any(value == each and type(value) is type(each) for each in typing.get_args(annotation))
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        parts = typing.get_args(annotation)
        kinds = parts[:1] * len(value) if len(parts) == 2 and parts[1] is Ellipsis else parts
        return len(value) == len(kinds) and all(_fits(each, kind)
                                                for each, kind in zip(value, kinds, strict=True))
    if annotation is type(None):
        return value is None
    return type(value) is annotation or (annotation is float and type(value) is int)


@functools.cache
def record_flags[R: Bound](record: type[R], member: type | Callable[..., registry.Configured], path: str,
                           alias: str) -> type[R]:
    """`member`'s record (`record`, a `Bound` kind's) as a command line sets it (api03): a subclass whose
    own fields are the member's arguments a plain flag sets (`_flat`; one the run builds, an object, takes
    none), each defaulting to the member's default (None where it states none a flag holds, which leaves
    the member's), the record's `bound` and `defaults` hidden from the command line and its other settings
    flags as the record declares them; its `bound` the member, by the flags set apart from their defaults
    over any argument it was given. Its `_alias` names the subcommand it is (`implied_subcommand`)."""
    reserved = {field.name for field in dataclasses.fields(record)} | set(dir(record))
    # Each flag's annotation, `make_dataclass`'s field type, and its flat part (`_flat`), which `flagged`
    # reads a value by.
    flags = []
    flats: dict[str, Annotation] = {}
    stated_defaults: dict[str, registry.Configured] = {}
    for name, declared, annotation in member_fields(member):
        flat = None if annotation is None or name in reserved else _flat(annotation)
        if flat is None:
            continue
        held = declared is not dataclasses.MISSING and _fits(declared, flat)
        default = declared if held else None
        flats[name], stated_defaults[name] = flat, default
        flags.append((name, flat if held else flat | None, dataclasses.field(default=default)))
    suppressed = [("bound", tyro.conf.Suppress[functools.partial],
                   dataclasses.field(default_factory=lambda: functools.partial(member))),
                  ("defaults", tyro.conf.Suppress[JsonRecords], dataclasses.field(default_factory=dict))]

    def __post_init__(self) -> None:
        given = {name: vars(self)[name] for name in stated_defaults
                 if vars(self)[name] != stated_defaults[name]}
        object.__setattr__(self, "bound", functools.partial(member, **{**self.bound.keywords, **given}))
        record.__post_init__(self)

    namespace = {"__post_init__": __post_init__, "__module__": __name__, "_alias": alias.replace("_", "-"),
                 "_flags": flats, "__doc__": (member.__doc__ or "").split("\n\n")[0]}
    title = "".join(part.title() for part in alias.split("_"))
    made = dataclasses.make_dataclass(f"{record.__name__}{title}", [*suppressed, *flags], bases=(record,),
                                      frozen=True, namespace=namespace)
    if not issubclass(made, record):
        raise TypeError(f"{made.__name__} is no {record.__name__}")
    return made


def implied_subcommand(default: registry.Configured) -> str | None:
    """The subcommand a field's `default` is (`record_flags`'s `_alias`: `env:flight`), which a flag of the
    field given without one implies; None for any other default."""
    return vars(type(default)).get("_alias") if isinstance(default, Bound) else None
