"""Where a process's code comes from: the file a loaded module is code of, whether a module is a user's own
code (loaded from a file no installed distribution versions and outside the interpreter's standard library)
or an installed distribution's, which its version states. One statement, which a traced program's identity
(`dew.training.identity`) and a data plan's step identity (`dew.data.plan`) both read."""

from __future__ import annotations

import abc
import functools
import importlib.metadata
import inspect
import site
import sys
import sysconfig
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable


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
    state `abc.ABCMeta` writes into every class it makes (`_ABC_STATE`), a Grain transform's among them, and
    the descriptor Python makes for each of a class's own slots (a slotted dataclass's fields), whose values
    an instance's fields state (`structured`): read as code, a frozen slotted step was inexact, and its
    unchanged exact resume refused (the base review of 826882bbf)."""
    slots = vars(held).get("__slots__", ())
    slots = (slots,) if isinstance(slots, str) else tuple(slots)
    found = []
    for name, member in sorted(vars(held).items()):
        member = wrapped(member)
        if ((name.startswith("__") and name.endswith("__") and not hand_written(member))
                or isinstance(member, _ABC_STATE)
                or (isinstance(member, types.MemberDescriptorType) and name in slots)):
            continue
        found.append((name, member))
    return found
