"""What a call of a program is, stated without tracing it.

A process that kept the executable of a program it compiled (`dew.training.ahead.keeping`) hands it to
a later one (`dew.training.ahead.kept`) only for the same call: the same function, arguments, settings,
code and host. JAX finds that out by tracing the call and reading the program; the whole point of a kept
executable is to skip the trace, so the call has to be stated from what stands before one. Two records
do it:

- `call_identity`, a call's own: its function and how it is jitted, its static arguments by value, the
  others' structure, shapes, dtypes and placements, JAX's trace context with Dew's contexts in it, the
  mesh (all of `OuterProgram.key_of`), every other setting of JAX's, Dew's environment variables and the
  facts of the process and host that Dew's code declares its traces read
  (`dew.nn.kernels.generation.traced_fact`).
- `Process`, the code those traces run: Python, Dew's source, the version of each installed distribution a
  loaded module belongs to and the source of every other loaded module.

Nothing is stated by a printed address or assumed equal: a value with no form here (`stated`) refuses
its call by name (`Unestablished`), and that call is compiled as without a file.

What the two cover, and what they do not. Code is covered by its source. Data is covered by the values
a call reaches through its arguments and through what its function reads outside them: the cells it
closed over, its defaults and every name its code reads that its globals hold, and the same of every
function, class and module of a user's own code it reaches that way (code no distribution versions and
Dew's digest does not hold, which only a walk can state): a class's data with its members' reads, a
module named as an object by everything it holds. Dew's and an installed library's module state is
theirs to state: JAX's through its settings, Dew's through the facts its readers declare
(`traced_fact`). Beyond that the identity
carries JAX's own purity contract across processes: code that changes what a trace computes by assigning
module state at run time is outside the identity, as it is outside `jax.jit`'s cache, which does not
retrace when a global a traced function reads is assigned. `dew.training.ahead.kept(path, verify=True)`
is the proof for a file and a process, at the cost of the trace: it lowers each handed call and holds it
to the kept program's digest.

This module is the boundary where Dew reads values it has no type for: any object a call names, any
module a process loaded. Its private readers ask each what Python's object model says of it (its
module, its name, its file, its slots) and refuse what answers nothing.
"""

from __future__ import annotations

import contextvars
import dataclasses
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import platform
import site
import sys
import sysconfig
import threading
import types
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path

import jax
import numpy as np
from jax._src import config as jax_config
from jax.experimental.layout import Format, Layout

from dew import _cache
from dew._records import JSON
from dew.algebra import terms
from dew.nn.kernels.generation import TRACED_FACTS
from dew.nn.kernels.options import COMPILE_POINT, OuterProgram


class Unestablished(ValueError):
    """A value a call's identity cannot state without tracing the call (`stated`): what a save leaves
    out, naming it (`dew.training.ahead.Keeping.refused`), and a handing compiles as without its file."""


def _coded(module: str | None, at: str) -> str:
    """`module`, the module a class or function at `at` is defined in, where its code is a file a process
    record digests or a version states (`Process`): refused where it has none (a notebook's or a shell's
    `__main__`), as its code could change with nothing to show it."""
    # A module with no file of its own is its nearest package's (an extension module's submodule).
    names = (module or "").split(".")
    for parts in range(len(names), 0, -1):
        found = sys.modules.get(".".join(names[:parts]))
        origin = getattr(getattr(found, "__spec__", None), "origin", None)
        if found is not None and (getattr(found, "__file__", None) is not None
                                  or origin in ("built-in", "frozen")):
            return module or ""
    raise Unestablished(f"{at}: defined in {module!r}, which has no source file")


def _main(module: str) -> list:
    """What states `__main__`'s code where a class or function of it is part of an identity: its file's
    digest, as no process record holds it (a process that saves and one that serves run other scripts)."""
    return [_source(Path(sys.modules[module].__file__))] if module == "__main__" else []


def _named(value, at: str) -> list | None:
    """`value` as the name its module holds it under (a class, a module's function, a jitted one, a
    singleton), where it has one: its definition is its module's code, the process record's. None for a
    value its module does not hold under its own name (a local function, an instance)."""
    module, name = getattr(value, "__module__", None), getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(name, str) or module not in sys.modules:
        return None
    held = sys.modules[module]
    for part in name.split("."):
        held = getattr(held, part, None)
    return ["named", _coded(module, at), name, *_main(module)] if held is value else None


def _code(code: types.CodeType) -> str:
    """A function's code as its digest: its instructions with the constants, names and nested code they
    read, so two functions written on one line that differ in a constant are two."""
    constants = [_code(constant) if isinstance(constant, types.CodeType)
                 else sorted(map(repr, constant)) if isinstance(constant, frozenset) else repr(constant)
                 for constant in code.co_consts]
    read = (code.co_code.hex(), constants, code.co_names, code.co_varnames, code.co_freevars,
            code.co_cellvars, code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_flags)
    return hashlib.sha256(repr(read).encode()).hexdigest()


_NONE_KNOWN: Mapping[int, JSON] = types.MappingProxyType({})
"""No value known by its `id` (`stated`'s default)."""


def stated(value, at: str, known: Mapping[int, JSON] = _NONE_KNOWN) -> JSON:
    """`value` as an identity states it: a JSON value equal in every process that holds an equal value.

    A class, a module's function or any object its module holds under its own name is that name (`_named`:
    its code is the process record's, `Process`), and one of a user's own code with what it reads (`_own`);
    a function no module holds (a lambda, a local one) is its code (`_code`) with what it reads (`_read`); a
    dataclass its type and fields; a number its type and bits, an array of NumPy's its dtype, shape and
    bytes; a mapping's pairs and a set's members in the order it holds them, which a trace may read; a
    module of the user's everything it holds (`_module_state`); an enumeration's member,
    a mesh, a sharding and a partition as the algebra's structural key states them
    (`dew.algebra.terms.parameter_key`, the one owner of a placement as plain data); JAX's other values of
    a call's key (an abstract value, a tree's structure node by node, a format, a device) by their parts. A
    value among `known` (by its `id`) is what its caller says of it (`call_identity`).

    Refused, naming `at` and the value's type, for anything else, an array of JAX's among them (a constant
    of the program, which its executable cannot serialize): an object with no such form could differ
    between two processes with nothing in the identity to show it."""
    return _stated(value, at, known, frozenset())


def _stated(value, at: str, known: Mapping[int, JSON], walking: frozenset[int]) -> JSON:
    """`stated`, within the functions `walking` whose reads are being stated: one of them met again (a
    function that names itself) is that function, stated once."""
    again = functools.partial(_stated, known=known, walking=walking)
    if id(value) in known:
        return known[id(value)]
    if (plain := _data(value)) is not None:
        return plain
    # A value of exactly Python's own type; a subclass (an enumeration's member) is stated as what it is.
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, jax.Array):
        raise Unestablished(f"{at}: an array of JAX's, a constant of the program")
    if isinstance(value, types.ModuleType):
        # A module of the user's is everything it holds: code that names it may read any of it.
        held = _module_state(value, at, known, walking) if _own_source(value.__name__) else None
        return ["module", _coded(value.__name__, at), held]
    if isinstance(value, logging.Logger):
        return ["logger", value.name]
    if isinstance(value, (types.FunctionType, type)) and _own_source(value.__module__):
        return _own(value, at, known, walking)
    if _hand_written(getattr(value, "__wrapped__", None)):
        # A callable that is no function and wraps one of the user's (a cache, a jitted function): what
        # it holds beside the function (its cache, its options) is its type's, which nothing here states.
        raise Unestablished(f"{at}: a {type(value).__module__}.{type(value).__qualname__} wrapping a "
                            "function of the user's own code")
    if (named := _named(value, at)) is not None:
        return named
    if isinstance(value, type):
        # A class its module holds under no name of its own (`NoneType`, an extension's) is still its
        # module's, by its name; one made inside a function is one of however many that function made.
        if "<locals>" in value.__qualname__:
            raise Unestablished(f"{at}: the class {value.__module__}.{value.__qualname__}, made inside a "
                                "function")
        module = _coded(value.__module__, at)
        return ["class", module, value.__qualname__, *_main(module)]
    if isinstance(value, (tuple, list)):
        # A named tuple's type names its fields, which its parts take in order.
        return ["sequence", again(type(value), at=at),
                [again(each, at=f"{at}[{index}]") for index, each in enumerate(value)]]
    if isinstance(value, (set, frozenset)):
        # In the order it holds them, as a mapping's: a trace that iterates it reads that order, which
        # for strings is another in each process unless its hash seed is set.
        return ["set", again(type(value), at=at), [again(each, at=at) for each in value]]
    if isinstance(value, Mapping):
        # In the order it holds them: a trace that iterates it reads that order.
        return ["map", again(type(value), at=at),
                [[again(key, at=at), again(each, at=f"{at}[{key!r}]")] for key, each in value.items()]]
    if isinstance(value, functools.partial):
        return ["partial", again(value.func, at=at), again(value.args, at=at), again(value.keywords, at=at)]
    if isinstance(value, types.MethodType):
        return ["method", again(value.__self__, at=at), value.__name__]
    if isinstance(value, types.FunctionType):
        if id(value) in walking:
            return ["itself", value.__qualname__]
        module = _coded(value.__module__, at)
        return ["function", module, value.__qualname__, _code(value.__code__), *_main(module),
                _read(value, at, known, walking)]
    if dataclasses.is_dataclass(value):
        return ["record", again(type(value), at=at),
                [[field.name, again(getattr(value, field.name, None), at=f"{at}.{field.name}")]
                 for field in dataclasses.fields(value)]]
    key = terms.parameter_key(value)
    if key[0] != "object":
        return ["native", _plain(key, at)]
    if (parts := _parts(value, at, again)) is not None:
        return parts
    raise Unestablished(f"{at}: a {type(value).__module__}.{type(value).__qualname__} has no form an "
                        "identity can state")


def _data(value) -> JSON | None:
    """A scalar of NumPy's, a float, a complex, bytes, a dtype or an array of NumPy's by its type and
    bits: a NumPy scalar's type is part of it (a trace makes an array of its dtype), a float is its bits.
    None for a value that is none of them."""
    # NumPy's scalars first: several are subclasses of Python's own (a float64 is a float).
    if isinstance(value, np.generic):
        return ["scalar", str(value.dtype), value.tobytes().hex()]
    if type(value) is float:
        return ["float", value.hex()]
    if type(value) is complex:
        return ["complex", value.real.hex(), value.imag.hex()]
    if type(value) is bytes:
        return ["bytes", hashlib.sha256(value).hexdigest()]
    if isinstance(value, np.dtype):
        return ["dtype", str(value)]
    if isinstance(value, np.ndarray):
        return ["array", str(value.dtype), list(value.shape),
                hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()]
    return None


def _parts(value, at: str, again: Callable[..., JSON]) -> JSON | None:
    """One of JAX's values that a call's key holds and the algebra's key does not state, by its parts
    (each through `again`, `stated` as its caller walks): an abstract value, a tree's structure, a format,
    one device's sharding, a device, and any other of JAX's by its attributes where it holds them plainly
    (an axis environment's sizes and names), the layout XLA chooses (`Layout.AUTO`). None for a value that
    is none of them."""
    if isinstance(value, jax.core.ShapedArray):
        return ["abstract", list(value.shape), str(value.dtype), bool(value.weak_type)]
    if isinstance(value, type(jax.tree.structure(()))):
        return _tree(value, at, again)
    if isinstance(value, Format):
        return ["format", again(value.layout, at=f"{at}.layout"), again(value.sharding, at=f"{at}.sharding")]
    if value is Layout.AUTO:
        # The layout XLA chooses, one singleton: what a format asks of the compile, not a layout it holds.
        return ["layout", "auto"]
    if isinstance(value, jax.sharding.SingleDeviceSharding):
        [device] = value.device_set
        return ["device sharding", again(device, at=at), value.memory_kind]
    if isinstance(value, jax.Device):
        return ["device", value.platform, value.id, value.device_kind]
    if type(value).__module__.partition(".")[0] not in ("jax", "jaxlib"):
        return None
    slots = [slot for held in type(value).__mro__ for slot in getattr(held, "__slots__", ())
             if slot not in ("__weakref__", "__dict__")]
    names = [*slots, *vars(value)] if hasattr(value, "__dict__") else slots
    if not names:
        return None
    return ["object", again(type(value), at=at),
            [[name, again(getattr(value, name), at=f"{at}.{name}")] for name in sorted(names)]]


def _plain(key: Hashable, at: str) -> JSON:
    """The algebra's structural key (`terms.parameter_key`) as JSON, refused where it holds an object it
    keys by itself (its `object` kind), which no other process reads."""
    if isinstance(key, tuple):
        if len(key) == 3 and key[0] == "object":
            raise Unestablished(f"{at}: holds a {key[1]}, which has no form an identity can state")
        return [_plain(part, at) for part in key]
    if key is None or isinstance(key, (bool, int, float, str)):
        return key
    raise Unestablished(f"{at}: holds a {type(key).__module__}.{type(key).__qualname__}, which has no "
                        "form an identity can state")


def _names(code: types.CodeType) -> set[str]:
    """Every name `code` and the code nested in it (a lambda, a comprehension, a local function) reads."""
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names |= _names(constant)
    return names


def _read(function: types.FunctionType, at: str, known: Mapping[int, JSON] = _NONE_KNOWN,
          walking: frozenset[int] = frozenset()) -> JSON:
    """What `function`'s body reads outside its arguments, as an identity states it (`stated`): the cells
    it closed over, its defaults, and each name its code or the code nested in it reads (`_names`) that
    its globals hold or that is a loaded module of the user's (one its body imports). A name is every use
    of it, an attribute's among them, so more is stated than is read, never less. A trace bakes each such
    value into its program, so one that differs is another call."""
    code = function.__code__
    within = walking | {id(function)}
    named = function.__qualname__
    cells = {}
    for name, cell in zip(code.co_freevars, function.__closure__ or (), strict=True):
        try:
            cells[name] = cell.cell_contents
        except ValueError:
            # A cell nothing has filled yet: the function reads no value through it.
            cells[name] = None
    names = _names(code)
    held = {name: function.__globals__[name] for name in names if name in function.__globals__}
    imported = {name: sys.modules[name] for name in names - held.keys()
                if name in sys.modules and _own_source(name)}
    return {"closure": {name: _stated(value, f"{at} ({named}'s closure, {name})", known, within)
                        for name, value in sorted(cells.items())},
            "globals": {name: _stated(value, f"{at} ({named}'s global {name})", known, within)
                        for name, value in sorted((held | imported).items())},
            "defaults": _stated((function.__defaults__, function.__kwdefaults__),
                                f"{at} ({named}'s defaults)", known, within)}


def _own(value, at: str, known: Mapping[int, JSON], walking: frozenset[int]) -> JSON:
    """A function or a class of a user's own code (`_own_source`) by its name there, with what it reads: a
    function's reads (`_read`); a class's data, each attribute its body or other code set on it, and its
    members' reads (`_members`), its bases' of the user's code among them. Such code reaches the program
    through a name alone, and what it reads at run time (a module's setting another module assigned) is
    nothing its source states, so it is walked, through every function, class and module of the user's
    it names in turn; one met again within its own walk is itself, stated once."""
    module = _coded(value.__module__, at)
    stated_name = ["own", module, value.__qualname__, *_main(module)]
    if id(value) in walking:
        return [*stated_name, "itself"]
    if "<locals>" in value.__qualname__:
        raise Unestablished(f"{at}: {value.__module__}.{value.__qualname__}, made inside a function")
    if isinstance(value, types.FunctionType):
        return [*stated_name, _code(value.__code__), _read(value, at, known, walking)]
    within = walking | {id(value)}
    return [*stated_name, {held.__qualname__: _members(held, at, known, within)
                           for held in value.__mro__ if _own_source(held.__module__)}]


def _members(held, at: str, known: Mapping[int, JSON], walking: frozenset[int]) -> JSON:
    """What a class or a module of the user's holds, each under its name, as an identity states it
    (`stated`): a method by the function its descriptor wraps (`_wrapped`), anything else by its value, a
    partial or a bound method whole. Of Python's own entries (`__dict__`, `__doc__`, `__annotations__`) it
    holds the methods the user wrote (`__call__`), which a trace runs; the rest are its machinery's, the
    methods a dataclass makes among them, which the source states."""
    stated_members = {}
    for name, member in sorted(vars(held).items()):
        member = _wrapped(member)
        if name.startswith("__") and name.endswith("__") and not _hand_written(member):
            continue
        stated_members[name] = _stated(member, f"{at} ({held.__name__}.{name})", known, walking)
    return stated_members


def _wrapped(member):
    """The function a class's descriptor wraps: a static or class method's, a property's getter, a cached
    property's. Any other member is itself."""
    if isinstance(member, (staticmethod, classmethod)):
        return member.__func__
    if isinstance(member, property):
        return member.fget
    if isinstance(member, functools.cached_property):
        return member.func
    return member


def _hand_written(member) -> bool:
    """Whether `member` is, or wraps (`__wrapped__`), a function a file of the user's own code wrote: the
    file of the globals it was defined in, as against one made for the class (a dataclass's `__init__`,
    whose code is no file's, and its `__repr__`, which a wrapper of Python's own holds). By its globals,
    not the module it names: a decorator's wrapper names the module of the function it wraps."""
    seen: set[int] = set()
    while member is not None and id(member) not in seen:
        seen.add(id(member))
        if isinstance(member, types.FunctionType):
            defined = member.__globals__
            if (_own_source(defined.get("__name__"))
                    and member.__code__.co_filename == defined.get("__file__")):
                return True
        member = getattr(member, "__wrapped__", None)
    return False


def _module_state(module: types.ModuleType, at: str, known: Mapping[int, JSON],
                  walking: frozenset[int]) -> JSON:
    """Everything a module of the user's holds (`_members`), for code that names the module itself and so
    may read any of it; a module met again within its own walk is itself."""
    if id(module) in walking:
        return "itself"
    return _members(module, at, known, walking | {id(module)})


def _tree(tree, at: str, again: Callable[..., JSON]) -> JSON:
    """A tree's structure node by node: a leaf, or a node's type, the data it holds beside its children (a
    mapping's keys, a custom node's static fields) and its children. Never as JAX prints it, which holds a
    node's data by its printed form."""
    node = tree.node_data()
    if node is None:
        return "leaf" if tree.num_leaves else "none"
    kind, held = node
    return ["node", again(kind, at=at), again(held, at=f"{at} (a {kind.__qualname__}'s node data)"),
            [_tree(child, at, again) for child in tree.children()]]


_UNSET = "<unset>"
"""What a context variable with neither a value nor a default reads as (`_outside`)."""


def _outside(function: str) -> None:
    """Hold a call of `function` to standing outside every context of Dew's own: each context variable
    of Dew's modules at its default on this thread. One off its default may steer the trace by more than
    its value states (an opt-in tuning run's reading takes picks from disk), so the call's identity is
    refused, naming the variable, until its module states what a trace reads through it. The compile
    point's own blocks decide how a program is compiled and kept, not what it computes
    (`dew.nn.kernels.options.COMPILE_POINT`). Found by reading the modules, so a variable added later is
    held to the same rule with nothing to list."""
    for name, module in sorted(sys.modules.items()):
        if module is None or not (name == "dew" or name.startswith("dew.")):
            continue
        for attribute, variable in vars(module).items():
            if (isinstance(variable, contextvars.ContextVar) and variable not in COMPILE_POINT
                    and variable.get(_UNSET) is not contextvars.Context().run(variable.get, _UNSET)):
                raise Unestablished(f"{function} is called within {name}.{attribute}'s context, which its "
                                    "trace may read")


def _global_context() -> tuple:
    """JAX's trace context as this process holds it outside every block: read on a thread of its own,
    which no block's value follows."""
    read: list[tuple] = []
    reader = threading.Thread(target=lambda: read.append(jax_config.trace_context()))
    reader.start()
    reader.join()
    return read[0]


def settings() -> JSON:
    """Every setting of JAX's as this thread holds it, by name: its trace context is in a call's key, and
    a trace reads others it does not key on (the default PRNG implementation). All but the persistent
    cache's switch, which a save turns off for its own compiles and no program reads
    (`dew._cache.UNCACHED`)."""
    # A state reads as this thread holds it; a flag that is none has its one value.
    held = jax.config.values
    return {name: stated(getattr(jax.config, name, held[name]), f"JAX's {name}")
            for name in sorted(held) if name not in _cache.UNCACHED}


def environment() -> JSON:
    """Dew's environment variables, which its code reads as it traces (`DEW_WIDE_CPU_ARITHMETIC`)."""
    return {name: value for name, value in sorted(os.environ.items()) if name.startswith("DEW_")}


def facts() -> JSON:
    """Each fact of the process and its host that Dew's code declares its traces read
    (`dew.nn.kernels.generation.traced_fact`), by where its reader is defined, as it reads now."""
    return {name: stated(reader(), name) for name, reader in sorted(TRACED_FACTS.items())}


def call_identity(program: OuterProgram, key: Hashable) -> JSON:
    """What a call of `program` under `key` (`OuterProgram.key_of`) is, stated without tracing it
    (`stated`), for a call outside every other context of Dew's (`_outside`): the function with what its
    body reads outside its arguments (`_read`), how it is jitted and which arguments it holds settled,
    the key (its static arguments by value, the others' structure, shapes, dtypes and placements, JAX's
    trace context with Dew's own user contexts in it, the mesh), JAX's other settings, Dew's environment
    and the declared facts (`settings`, `environment`, `facts`). With its process's code (`Process`) it is
    what the call's trace reads, so two calls of one identity in two processes of one record trace one
    program. Refused where any of it has no stated form (`Unestablished`)."""
    # A context a trace records into is no part of its program (`dew._cache.RECORDING`): the collection
    # it holds while one is open, never its empty default, which is a value like any other. A setting
    # whose value is a bare object is unset: the mark every process holds of its own, where it is the one
    # the process holds outside any block (`_global_context`).
    known: dict[int, JSON] = {id(setting): ["unset"] for setting in _global_context()
                              if type(setting) is object}
    for context in _cache.RECORDING:
        recorded = context.value
        if recorded is not None and recorded != ():
            known[id(recorded)] = ["recording"]
    function = program.function
    named = called(program)
    _outside(named)
    # A bound method's body is its function's; its object is stated with the method.
    body = function.__func__ if isinstance(function, types.MethodType) else function
    return {"function": stated(function, "the program's function", known),
            "reads": (_read(body, "the program's function", known)
                      if isinstance(body, types.FunctionType) else None),
            "jitted": stated(program.arguments, f"{named}'s jit arguments", known),
            "settled": stated(program.settled, f"{named}'s settled arguments", known),
            "call": stated(key, f"{named}'s call", known),
            "settings": settings(), "environment": environment(), "facts": facts()}


def digest(identity: JSON) -> str:
    """An identity (`call_identity`) as the key its program is kept by."""
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class Process:
    """The code a process's traces run, beside each call's own identity (`call_identity`): its Python,
    Dew's source (`dew._cache.package_digest`), the version of each installed distribution a loaded module
    belongs to, and the source of every other loaded module (a checkout installed in place, a user's own
    package or script beside it; `__main__` is stated where a call names it). With them, the settings,
    environment and facts as the process held them when it saved (`settings`, `environment`, `facts`):
    each call's identity holds its own, and these name what differs before any call. A process holds
    programs another kept only where all of it is the same (`held`); the JAX stack, XLA's flags and the
    devices are the target's (`dew.training.ahead.BuildTarget`)."""

    python: str
    dew: str
    distributions: tuple[tuple[str, str], ...]
    sources: tuple[tuple[str, str], ...]
    settings: str
    environment: str
    facts: str

    @staticmethod
    def here() -> Process:
        """This process's record, read now: after its programs traced, every module a trace imported is
        loaded and so among them."""
        installed, standard, packages = _installed(), _standard(), _distributed()
        distributions: dict[str, str] = {}
        sources: dict[str, str] = {}
        for name in sorted(sys.modules):
            path = _recorded(name)
            if path is None:
                continue
            placed = _within(path, installed)
            owners = packages.get(name.partition(".")[0], ()) if placed else ()
            if owners:
                distributions.update((owner, importlib.metadata.version(owner)) for owner in owners)
            elif placed or not _within(path, standard):
                sources[name] = _source(path)
        return Process(platform.python_version(), _cache.package_digest(),
                       tuple(sorted(distributions.items())), tuple(sorted(sources.items())),
                       *(json.dumps(read(), sort_keys=True) for read in (settings, environment, facts)))

    def differences(self) -> dict[str, str]:
        """What this process holds otherwise than the record, by what it is (`python`, `dew`, a
        distribution's or a module's name, a setting's, an environment variable's or a fact's), each said
        as the record's and its own: a distribution or a module the record names is read here whether or
        not this process loaded it."""
        here = Process.here()
        differing = {name: f"{then}, not {now}" for name, then, now in (
            ("python", self.python, here.python), ("dew", self.dew, here.dew)) if then != now}
        for kind, then, now in (("setting", self.settings, here.settings),
                                ("environment", self.environment, here.environment),
                                ("fact", self.facts, here.facts)):
            recorded, held = json.loads(then), json.loads(now)
            differing.update({f"{kind} {name}": f"{recorded.get(name)!r}, not {held.get(name)!r}"
                              for name in sorted(recorded.keys() | held.keys())
                              if recorded.get(name) != held.get(name)})
        installed = {owner for owners in _distributed().values() for owner in owners}
        for distribution, version in self.distributions:
            found = importlib.metadata.version(distribution) if distribution in installed else None
            if found != version:
                differing[f"distribution {distribution}"] = f"{version}, not {found}"
        for name, source in self.sources:
            file = _found(name)
            if file is None or _source(file) != source:
                differing[f"source {name}"] = "another source, or none here"
        return differing

    def held(self) -> None:
        """Refused, naming what differs, where this process is not the record's (`differences`)."""
        if differing := self.differences():
            raise ValueError("these programs were traced by another process's code or settings than this "
                             f"one's: {'; '.join(f'{name} {said}' for name, said in differing.items())}")


def _recorded(name: str) -> Path | None:
    """The file whose source a process record holds for the loaded module `name` (`_file`), None for one
    it holds none of: a module of no file, a frozen one (the interpreter's own, whatever file it names),
    Dew's (its package digest's) and `__main__` (stated where a call names it)."""
    module = sys.modules[name]
    frozen = getattr(getattr(module, "__spec__", None), "origin", None) == "frozen"
    if (getattr(module, "__file__", None) is None or frozen or name in ("dew", "__main__")
            or name.startswith("dew.")):
        return None
    return _file(name)


def _found(name: str) -> Path | None:
    """The file of the module `name` as this process would import it, loaded or not; None where it finds
    none."""
    loaded = sys.modules.get(name)
    file = (getattr(loaded, "__file__", None) if loaded is not None
            else getattr(importlib.util.find_spec(name), "origin", None))
    return Path(file) if file is not None and Path(file).is_file() else None


def _qualified(function) -> str:
    """A program's function by where it is defined and its name there, as far as it states them."""
    return (f"{getattr(function, '__module__', None)}."
            f"{getattr(function, '__qualname__', type(function).__qualname__)}")


def called(program: OuterProgram) -> str:
    """`program`'s function by its module and name, as a refusal names it."""
    return _qualified(program.function)


def _file(name: str) -> Path:
    """The file the loaded module `name` is code of: its own, or, where the file it names is none (a
    module an extension makes of itself), its nearest package's that is one; refused where none is, as
    nothing then states its code."""
    names = name.split(".")
    for parts in range(len(names), 0, -1):
        file = getattr(sys.modules.get(".".join(names[:parts])), "__file__", None)
        if file is not None and Path(file).is_file():
            return Path(file).resolve()
    raise ValueError(f"the loaded module {name} names no file of its code, so no record of this process "
                     "can state it")


def _own_source(name: str | None) -> bool:
    """Whether the module `name` is a user's own code: loaded from a file no distribution versions and
    Dew's digest does not hold (`Process.sources`), or the script the process runs."""
    if name == "__main__":
        return getattr(sys.modules[name], "__file__", None) is not None
    if name is None or name not in sys.modules:
        return False
    path = _recorded(name)
    if path is None:
        return False
    placed = _within(path, _installed())
    owned = placed and name.partition(".")[0] in _distributed()
    return not owned and (placed or not _within(path, _standard()))


@functools.cache
def _distributed() -> Mapping[str, list[str]]:
    """The installed distributions each top-level package belongs to (`packages_distributions`), read
    once: a process installs none while it runs."""
    return importlib.metadata.packages_distributions()


@functools.cache
def _installed() -> tuple[Path, ...]:
    """The directories this interpreter installs distributions into."""
    paths = sysconfig.get_paths()
    return tuple({Path(path).resolve() for path in (paths["purelib"], paths["platlib"],
                                                    *site.getsitepackages(), site.getusersitepackages())})


@functools.cache
def _standard() -> tuple[Path, ...]:
    """The directories of this interpreter's standard library, whose code its version states."""
    paths = sysconfig.get_paths()
    return tuple({Path(paths["stdlib"]).resolve(), Path(paths["platstdlib"]).resolve()})


def _within(path: Path, directories: Sequence[Path]) -> bool:
    return any(path.is_relative_to(directory) for directory in directories)


def _source(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
