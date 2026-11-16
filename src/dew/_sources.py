"""Where a process's code comes from: the file a loaded module is code of, whether a module is a user's own
code (loaded from a file no installed distribution versions and outside the interpreter's standard library)
or an installed distribution's, which its version states. One statement, which a traced program's identity
(`dew.training.identity`) and a data plan's step identity (`dew.data.plan`) both read."""

from __future__ import annotations

import abc
import dataclasses
import functools
import importlib.metadata
import inspect
import site
import sys
import sysconfig
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable


def source_file(held: Callable | type) -> Path | None:
    """The file `held`, a function or a class, is defined in, None where it has none: a class of a notebook,
    a REPL or `python -c` (`__main__` with no file), a builtin's or an extension's, or a file since removed.
    Read where its module names a file, where Python reads its source file without raising."""
    module = sys.modules.get(held.__module__)
    if module is None or vars(module).get("__file__") is None:
        return None
    place = inspect.getsourcefile(held)
    return Path(place) if place is not None and Path(place).is_file() else None


def recorded(name: str) -> Path | None:
    """The file whose source a process record holds for the loaded module `name` (`file_of`), None for one
    it holds none of: a module of no file, a frozen one (the interpreter's own, whatever file it names),
    Dew's (its package digest's) and `__main__` (stated where a call names it)."""
    module = sys.modules[name]
    spec = vars(module).get("__spec__")
    frozen = spec is not None and spec.origin == "frozen"
    if (vars(module).get("__file__") is None or frozen or name in ("dew", "__main__")
            or name.startswith("dew.")):
        return None
    return file_of(name)



def file_of(name: str) -> Path:
    """The file the loaded module `name` is code of: its own, or, where the file it names is none (a
    module an extension makes of itself), its nearest package's that is one; refused where none is, as
    nothing then states its code."""
    names = name.split(".")
    for parts in range(len(names), 0, -1):
        module = sys.modules.get(".".join(names[:parts]))
        file = None if module is None else vars(module).get("__file__")
        if file is not None and Path(file).is_file():
            return Path(file).resolve()
    raise ValueError(f"the loaded module {name} names no file of its code, so no record of this process "
                     "can state it")



def own_source(name: str | None) -> bool:
    """Whether the module `name` is a user's own code: loaded from a file no distribution versions and
    Dew's digest does not hold (`dew.training.identity.Process.sources`), or the script the process runs.
    Read once a loaded module (`_OWN`): a module loaded again is read again."""
    if name == "__main__":
        return vars(sys.modules[name]).get("__file__") is not None
    if name is None or name not in sys.modules:
        return False
    module = sys.modules[name]
    held = _OWN.get(name)
    if held is not None and held[0] is module:
        return held[1]
    path = recorded(name)
    placed = path is not None and within(path, installed())
    own = path is not None and not (placed and name.partition(".")[0] in distributed()) and (
        placed or not within(path, standard()))
    _OWN[name] = (module, own)
    return own


_OWN: dict[str, tuple[object, bool]] = {}
"""Each loaded module's verdict (`own_source`) beside the module it was read of: a program's identity
asks it of every function it states, its file's path resolved and placed each time (13% of a small fit)."""


@functools.cache
def distributed() -> Mapping[str, list[str]]:
    """The installed distributions each top-level package belongs to (`packages_distributions`), read
    once: a process installs none while it runs."""
    return importlib.metadata.packages_distributions()


@functools.cache
def installed() -> tuple[Path, ...]:
    """The directories this interpreter installs distributions into."""
    paths = sysconfig.get_paths()
    return tuple({Path(path).resolve() for path in (paths["purelib"], paths["platlib"],
                                                    *site.getsitepackages(), site.getusersitepackages())})


@functools.cache
def standard() -> tuple[Path, ...]:
    """The directories of this interpreter's standard library, whose code its version states."""
    paths = sysconfig.get_paths()
    return tuple({Path(paths["stdlib"]).resolve(), Path(paths["platstdlib"]).resolve()})


def within(path: Path, directories: Sequence[Path]) -> bool:
    return any(path.is_relative_to(directory) for directory in directories)


def names_read(code: types.CodeType) -> set[str]:
    """Every name `code` and the code nested in it (a lambda, a comprehension, a local function) reads."""
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names |= names_read(constant)
    return names



def wrapped(member):
    """The function a class's descriptor wraps: a static or class method's, a property's getter, a cached
    property's. Any other member is itself."""
    if isinstance(member, (staticmethod, classmethod)):
        return member.__func__
    if isinstance(member, property):
        return member.fget
    if isinstance(member, functools.cached_property):
        return member.func
    return member


@runtime_checkable
class Wrapping(Protocol):
    """A function that runs another, which it names (`functools.wraps`, `jax.jit`)."""

    __wrapped__: Callable


def hand_written(member) -> bool:
    """Whether `member` is, or wraps (`__wrapped__`), a function a file of the user's own code wrote: the
    file of the globals it was defined in, as against one made for the class (a dataclass's `__init__`,
    whose code is no file's, and its `__repr__`, which a wrapper of Python's own holds). By its globals,
    not the module it names: a decorator's wrapper names the module of the function it wraps."""
    seen: set[int] = set()
    while member is not None and id(member) not in seen:
        seen.add(id(member))
        if isinstance(member, types.FunctionType):
            defined = member.__globals__
            if (own_source(defined.get("__name__"))
                    and member.__code__.co_filename == defined.get("__file__")):
                return True
        member = member.__wrapped__ if isinstance(member, Wrapping) else None
    return False

def function_reads(function: types.FunctionType) -> tuple[dict, dict]:
    """What `function`'s body reads outside its arguments: the cells it closed over (None for one nothing
    has filled yet), and each name its code or the code nested in it reads (`names_read`) that its globals
    hold or that is a loaded module of the user's (one its body imports). A name is every use of it, an
    attribute's among them, so more is read than is used, never less. The one statement of what a function
    reads, which a traced program's identity (`dew.training.identity`) and a data step's held code
    (`dew.data.plan`) both walk."""
    code = function.__code__
    cells = {}
    for name, cell in zip(code.co_freevars, function.__closure__ or (), strict=True):
        try:
            cells[name] = cell.cell_contents
        except ValueError:
            cells[name] = None
    names = names_read(code)
    held = {name: function.__globals__[name] for name in names if name in function.__globals__}
    imported = {name: sys.modules[name] for name in names - held.keys()
                if name in sys.modules and own_source(name)}
    return cells, held | imported


def own_classes(kind: type) -> list[type]:
    """The classes of the user's own code `kind` is made of, itself and every base, in its MRO: what a call
    of its methods runs, an inherited method's class among them."""
    return [held for held in kind.__mro__ if own_source(held.__module__)]


_ABC_STATE = type(vars(abc.ABC)["_abc_impl"])
"""The type of what `abc.ABCMeta` keeps in each class it makes (its registry and caches), machinery no
statement of the user's code holds (`members`)."""


def members(held) -> list[tuple]:
    """What a class or a module of the user's holds, by name, in order: a method as the function its
    descriptor wraps (`wrapped`), anything else as itself. Of Python's own entries (`__dict__`, `__doc__`,
    `__annotations__`) only the methods the user wrote (`__call__`), which a call runs; the rest are its
    machinery's, the methods a dataclass makes among them, which the class's description states, the
    state `abc.ABCMeta` writes into every class it makes (`_ABC_STATE`), a Grain transform's among them, the
    descriptor Python makes for each of a class's own slots (mangled or not), and a generated field's
    default where the class attribute still holds it, which `generated_fields` states with its factory:
    a slotted class's and a child overriding a default hold it where their class attributes do not (train
    947's review), and a value set on the class at run time, or a class variable, stays a member (train
    950's review: skipping every field name left them unstated)."""
    defaults = {field.name: field.default for field in generated_fields(held)}
    found = []
    for name, member in sorted(vars(held).items()):
        member = wrapped(member)
        if ((name.startswith("__") and name.endswith("__") and not hand_written(member))
                or isinstance(member, _ABC_STATE)
                or (name in defaults and member is defaults[name])
                or (isinstance(member, types.MemberDescriptorType) and member.__objclass__ is held)):
            continue
        found.append((name, member))
    return found


MISSING = dataclasses.MISSING
"""What a `GeneratedField` holds where its field has no default or no factory."""


class GeneratedField(NamedTuple):
    """A field a class generator (`dataclasses`, attrs) recorded on a class: its name, its default and its
    default factory, each `MISSING` where it has none, and for an attrs field the rest of what attrs
    recorded of it (`parts`, `generated_fields`), each by its name; none for a dataclass's."""

    name: str
    default: Any
    factory: Any
    parts: tuple[tuple[str, Any], ...] = ()


def attrs_made(held: type) -> bool:
    """Whether attrs made `held` itself (its own `__attrs_attrs__`, which a subclass it did not decorate
    inherits)."""
    return "__attrs_attrs__" in vars(held)


def generated(held: type) -> bool:
    """Whether a class generator made `held`: a dataclass (one it derives from among them) or a class attrs
    made itself."""
    return dataclasses.is_dataclass(held) or attrs_made(held)


def field_names(held: type) -> list[str]:
    """The names of the fields a class generator recorded for `held` (`generated`), in order: what an
    instance of it holds, a record's parts."""
    if dataclasses.is_dataclass(held):
        return [field.name for field in dataclasses.fields(held)]
    return [field.name for field in vars(held)["__attrs_attrs__"]]


ATTRS_UNSTATED = frozenset({"name", "default", "counter", "type", "on_setattr"})
"""An attrs field's parts (its `Attribute`'s slots) `generated_fields` states apart or not at all: its name
and default (`GeneratedField`'s own), the order attrs counted it in, its annotation, which attrs runs nothing
of, and its own `on_setattr`, for which the hook the class's generated `__setattr__` runs on it stands."""


def _setattr_hooks(held: type) -> Mapping[str, Callable]:
    """The hook the `__setattr__` attrs generated for `held` runs on each field it hooks, by the field's name:
    the field's own `on_setattr` or the class's, held in that method's closure alone (its `sa_attrs`, each
    field's `Attribute` beside its hook). Empty for a class whose `__setattr__` attrs did not generate."""
    method = vars(held).get("__setattr__")
    if not isinstance(method, types.FunctionType) or hand_written(method):
        return {}
    for cell in method.__closure__ or ():
        held_hooks = cell.cell_contents
        if isinstance(held_hooks, dict):
            return {name: hook for name, (_, hook) in held_hooks.items()}
    return {}


def generated_fields(held: type) -> list[GeneratedField]:
    """The fields a class generator recorded on `held` itself, in order, its bases' among them and an
    override's default its own: a dataclass's (`dataclasses.fields`, its class and init variables, which
    are no fields, apart) or an attrs class's (`__attrs_attrs__`, a default `attrs.Factory`'s factory its
    factory, the rest of what attrs recorded of it its parts). Empty for a class no generator made
    itself."""
    if "__dataclass_fields__" in vars(held):
        return [GeneratedField(field.name, field.default, field.default_factory)
                for field in dataclasses.fields(held)]
    if attrs_made(held):
        # attrs is loaded where a class of it exists; its sentinel and factory type are its own.
        attr = sys.modules["attr"]
        converter_type, hooks = vars(attr).get("Converter"), _setattr_hooks(held)
        found = []
        for field in vars(held)["__attrs_attrs__"]:
            default, factory = field.default, MISSING
            if isinstance(default, attr.Factory):
                default, factory = MISSING, default.factory
            # Every other slot of its `Attribute` (converter, validator, eq and order flags and keys, hash,
            # init, repr, metadata, keyword-only, alias, whatever this attrs records) but `ATTRS_UNSTATED`,
            # an `attrs.Converter` as its converter and how it is called (where this attrs has one), and the
            # hook the class's generated `__setattr__` runs on it, the class's own where the field sets none
            # (train 955's review: a class-level hook and an `eq=False` went unstated).
            parts = []
            for name in vars(type(field))["__slots__"]:
                if name in ATTRS_UNSTATED:
                    continue
                value = getattr(field, name)
                if converter_type is not None and isinstance(value, converter_type):
                    value = (value.converter, value.takes_self, value.takes_field)
                parts.append((name, value))
            if (hook := hooks.get(field.name)) is not None:
                parts.append(("on_setattr", hook))
            found.append(GeneratedField(field.name, MISSING if default is attr.NOTHING else default, factory,
                                        tuple(parts)))
        return found
    return []
