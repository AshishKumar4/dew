"""One statement of a user's code and the values it reads, as plain data equal in every process that holds an
equal value (`stated`): what a traced program's identity holds of its function and call
(`dew.training.identity`) and what holds a data step's own code (`dew.data.plan.StepIdentity`), which reads no
training module. A function or a class of the user's own code (`dew._sources.own_source`) is its name with
what it reads, walked through every function, class and module of the user's it reaches; anything else by
its type and value, or refused (`Unestablished`)."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import logging
import sys
import types
from collections.abc import Callable, Hashable, Mapping
from pathlib import Path

import jax
import numpy as np
from jax.experimental.layout import Format, Layout

from dew import _hashing, _sources
from dew._records import JSON
from dew.algebra import terms


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
    if module != "__main__":
        return []
    file = sys.modules[module].__file__
    if file is None:
        raise Unestablished("`__main__` is no file (an interactive session), so no record states its code")
    return [source_digest(Path(file))]


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
    a function no module holds (a lambda, a local one) is its code (`_code`) with what it reads (`reads`); a
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
    if (plain := _data(value, at)) is not None:
        return plain
    # A value of exactly Python's own type; a subclass (an enumeration's member) is stated as what it is.
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, jax.Array):
        raise Unestablished(f"{at}: an array of JAX's, a constant of the program")
    if isinstance(value, types.ModuleType):
        # A module of the user's is everything it holds: code that names it may read any of it.
        held = _module_state(value, at, known, walking) if _sources.own_source(value.__name__) else None
        return ["module", _coded(value.__name__, at), held]
    if isinstance(value, logging.Logger):
        return ["logger", value.name]
    if isinstance(value, (types.FunctionType, type)) and _sources.own_source(value.__module__):
        return _own(value, at, known, walking)
    if _sources.hand_written(getattr(value, "__wrapped__", None)):
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
                reads(value, at, known, walking)]
    if dataclasses.is_dataclass(value):
        return ["record", again(type(value), at=at),
                [[field.name, again(getattr(value, field.name, None), at=f"{at}.{field.name}")]
                 for field in dataclasses.fields(value)]]
    key = terms.parameter_key(value)
    if isinstance(key, tuple) and key[0] != "object":
        return ["native", _plain(key, at)]
    if (parts := _parts(value, at, again)) is not None:
        return parts
    raise Unestablished(f"{at}: a {type(value).__module__}.{type(value).__qualname__} has no form an "
                        "identity can state")


def _data(value, at: str) -> JSON | None:
    """An array or a scalar of NumPy's, a float, a complex, bytes or a dtype by its type and bits: an
    array or a NumPy scalar by its kind and its fingerprint (`_hashing.array_fingerprint`: its dtype, shape
    and bytes), a float by its bits. A scalar and a 0-d array of one value fingerprint alike, and a trace
    may branch on which it holds, so each keeps its own tag: nested in a record no type field beside it
    tells them apart. One whose storage holds pointers (an array of objects, a structured scalar with an
    object field) is refused as that fingerprint refuses it: its bytes stay the same when what it points to
    changes. None for a value that is none of them."""
    # NumPy's scalars first: several are subclasses of Python's own (a float64 is a float).
    if isinstance(value, np.generic | np.ndarray):
        hasher = hashlib.sha256()
        try:
            _hashing.array_fingerprint(hasher, np.asarray(value))
        except TypeError as refusal:
            raise Unestablished(f"{at}: {refusal}") from None
        return ["scalar" if isinstance(value, np.generic) else "array", hasher.hexdigest()]
    if type(value) is float:
        return ["float", value.hex()]
    if type(value) is complex:
        return ["complex", value.real.hex(), value.imag.hex()]
    if type(value) is bytes:
        return ["bytes", hashlib.sha256(value).hexdigest()]
    if isinstance(value, np.dtype):
        return ["dtype", str(value)]
    return None


def _parts(value, at: str, again: Callable[..., JSON]) -> JSON | None:
    """One of JAX's values that a call's key holds and the algebra's key does not state, by its parts
    (each through `again`, `stated` as its caller walks): a tree's structure, a format,
    one device's sharding, a device, and any other of JAX's by its attributes where it holds them plainly
    (an axis environment's sizes and names), the layout XLA chooses (`Layout.AUTO`). None for a value that
    is none of them."""
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


def reads(function: types.FunctionType, at: str, known: Mapping[int, JSON] = _NONE_KNOWN,
          walking: frozenset[int] = frozenset()) -> JSON:
    """What `function`'s body reads outside its arguments (`dew._sources.function_reads`), as an identity
    states it (`stated`): the cells it closed over, its defaults, and each name its code or the code nested
    in it reads that its globals hold or that is a loaded module of the user's. A trace bakes each such
    value into its program, so one that differs is another call."""
    within = walking | {id(function)}
    named = function.__qualname__
    cells, read = _sources.function_reads(function)
    return {"closure": {name: _stated(value, f"{at} ({named}'s closure, {name})", known, within)
                        for name, value in sorted(cells.items())},
            "globals": {name: _stated(value, f"{at} ({named}'s global {name})", known, within)
                        for name, value in sorted(read.items())},
            "defaults": _stated((function.__defaults__, function.__kwdefaults__),
                                f"{at} ({named}'s defaults)", known, within)}


def _own(value, at: str, known: Mapping[int, JSON], walking: frozenset[int]) -> JSON:
    """A function or a class of a user's own code (`dew._sources.own_source`) by its name there, with what
    it reads: a function's reads (`reads`); a class's data, each attribute its body or other code set on
    it, and its members' reads (`_members`), its bases' of the user's code among them. Such code reaches
    the program through a name alone, and what it reads at run time (a module's setting another module
    assigned) is nothing its source states, so it is walked, through every function, class and module of
    the user's it names in turn; one met again within its own walk is itself, stated once."""
    module = _coded(value.__module__, at)
    stated_name = ["own", module, value.__qualname__, *_main(module)]
    if id(value) in walking:
        return [*stated_name, "itself"]
    if "<locals>" in value.__qualname__:
        raise Unestablished(f"{at}: {value.__module__}.{value.__qualname__}, made inside a function")
    if isinstance(value, types.FunctionType):
        return [*stated_name, _code(value.__code__), reads(value, at, known, walking)]
    within = walking | {id(value)}
    return [*stated_name, {held.__qualname__: _members(held, at, known, within)
                           for held in _sources.own_classes(value)}]


def _members(held, at: str, known: Mapping[int, JSON], walking: frozenset[int]) -> JSON:
    """What a class or a module of the user's holds, each under its name (`dew._sources.members`), as an
    identity states it (`stated`): a method by the function its descriptor wraps, anything else by its
    value, a partial or a bound method whole."""
    return {name: _stated(member, f"{at} ({held.__name__}.{name})", known, walking)
            for name, member in _sources.members(held)}


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


def source_digest(path: Path) -> str:
    """The digest of the file at `path`'s bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
