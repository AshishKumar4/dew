"""A registered kind's records as a command line chooses them (api03): each member of a `Bound` kind a
subcommand (`record_flags`), its arguments plain flags (`_flat`), and the subcommand a field's default
already is (`implied_subcommand`). Apart from `dew.registry`, whose records they read."""

from __future__ import annotations

import dataclasses
import functools
import operator
import types
import typing
from collections.abc import Callable, Mapping
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


MERGED = "dew_flags_merged"
"""The names a subcommand's record's flags merged into its binding (`record_flags`), held on the binding
itself (a `functools.partial`'s attributes), no field of the record's, so a parse over it drops them."""


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
        # The flags the sole source of what they state: the arguments an earlier record's flags merged into
        # its `bound` (named on that binding, `MERGED`, which a parse over the record carries with it) are
        # dropped before this one's flags set apart from their defaults are merged, so a flag reset to its
        # default leaves the member's default, not a recipe's value it was set over (row 14127:
        # `--rollout.train-ratio 1024.0` over a recipe's 16.0 kept 16.0). An argument no flag merged stays,
        # a default-valued one stated outright among them (`Bound.flagged`); a binding made anew merged none.
        merged = vars(self.bound).get(MERGED, frozenset())
        kept = {name: value for name, value in self.bound.keywords.items() if name not in merged}
        given = {name: vars(self)[name] for name in stated_defaults
                 if vars(self)[name] != stated_defaults[name]}
        bound = functools.partial(member, **{**kept, **given})
        vars(bound)[MERGED] = frozenset(given)
        object.__setattr__(self, "bound", bound)
        record.__post_init__(self)

    namespace = {"__post_init__": __post_init__, "__module__": __name__, "_alias": alias.replace("_", "-"),
                 "_flags": flats, "__doc__": (member.__doc__ or "").split("\n\n")[0]}
    title = "".join(part.title() for part in alias.split("_"))
    made = dataclasses.make_dataclass(f"{record.__name__}{title}", [*suppressed, *flags], bases=(record,),
                                      frozen=True, namespace=namespace)
    if not issubclass(made, record):
        raise TypeError(f"{made.__name__} is no {record.__name__}")
    return made


def flag_held(made: type[Bound], fields: Mapping[str, registry.Configured]) -> dict[str, registry.Configured]:
    """The arguments among `fields` a subcommand's record (`made`, `record_flags`'s) holds on its flags
    (`Bound.flagged`): each a flag can state and set apart from the flag's default; one at the default stays
    an argument, stated as the record states it, which a flag at its default would leave unstated."""
    defaults = {field.name: field.default for field in dataclasses.fields(made)}
    return {name: value for name, value in fields.items()
            if name in made._flags and _fits(value, made._flags[name]) and value != defaults[name]}


def rebound[R: Bound](record: R, fields: Mapping[str, registry.Configured]) -> R:
    """`record` stating `fields` as its arguments (`Bound.with_fields`, `without_fields`): a command line's
    subcommand (`record_flags`) as its kind's own record so bound flags it (`Bound.flagged`), every flag and
    its `bound` taken from that, so an argument a flag held is the one stated now, not the flag's earlier
    value merged over it (row 14127); any other record so bound."""
    bound = functools.partial(record.member, **fields)
    if "_alias" not in vars(type(record)):
        return dataclasses.replace(record, bound=bound)
    flagged = dataclasses.replace(record.recorded(), bound=bound).flagged()
    return dataclasses.replace(record, **{field.name: vars(flagged)[field.name]
                                          for field in dataclasses.fields(record)})


def implied_subcommand(default: registry.Configured) -> str | None:
    """The subcommand a field's `default` is (`record_flags`'s `_alias`: `env:flight`), which a flag of the
    field given without one implies; None for any other default."""
    return vars(type(default)).get("_alias") if isinstance(default, Bound) else None
