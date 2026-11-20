"""One statement of a user's code and the values it reads, as plain data equal in every process that holds an
equal value (`stated`): what a rung record's key holds of a model (`dew.training.rungs`), a program's trees as
its digest states them (`dew.nn.kernels.options.program_digest`) and what holds a data step's own code
(`dew.data.plan.StepIdentity`), which reads no training module. A function or a class of the user's own code
(`dew._sources.own_source`) is its name with what it reads, walked through every function, class and module of
the user's it reaches; anything else by its type and value, or refused (`Unestablished`).

This module is the boundary where Dew reads values it has no type for: any object a call names, any module a
process loaded. Its private readers ask each what Python's object model says of it (its module, its name, its
file, its slots) and refuse what answers nothing."""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import sys
import types
from collections.abc import Callable, Hashable, Mapping
from pathlib import Path

import jax
import numpy as np
from jax._src.named_sharding import UNSPECIFIED
from jax.experimental.layout import Format, Layout
from jaxlib import xla_client
from jaxlib._jax import PjitFunction

from dew import _hashing, _sources
from dew._records import JSON


class Unestablished(ValueError):
    """A value with no stated form (`stated`): a program whose trees hold one is one a save leaves out,
    naming it (`dew.training.ahead.Keeping.refused`), and a handing compiles as without its file."""


def _coded(module: str | None, at: str) -> str:
    """`module`, the module a class or function at `at` is defined in, where its code is a file or a version
    states it: refused where it has none (a notebook's or a shell's
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


def _named(value, at: str) -> list | None:
    """`value` as the name its module holds it under (a class, a module's function, a library's jitted one,
    a singleton), where it has one: its definition is its module's code, the process record's. None for a
    value its module does not hold under its own name (a local function, an instance), and for one of
    `__main__`, whose code no process record holds: it is stated by what it is instead, as the user's own
    code is (`_own`)."""
    module, name = getattr(value, "__module__", None), getattr(value, "__qualname__", None)
    if (not isinstance(module, str) or not isinstance(name, str) or module not in sys.modules
            or module == "__main__"):
        return None
    held = sys.modules[module]
    for part in name.split("."):
        held = getattr(held, part, None)
    return ["named", _coded(module, at), name] if held is value else None


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
    its code named by where it is defined), and one of a user's own code with what it reads (`_own`);
    a function no module holds (a lambda, a local one) is its code (`_code`) with what it reads (`reads`); a
    dataclass its type and fields; a number its type and bits, an array of NumPy's its dtype, shape and
    bytes; a mapping's pairs and a set's members in the order it holds them, which a trace may read; a
    module of the user's everything it holds (`_module_state`); an enumeration's member,
    a mesh, a sharding and a partition as the algebra's structural key states them
    (`dew.algebra.terms.parameter_key`, the one owner of a placement as plain data); JAX's other values of
    a call's key (an abstract value, a tree's structure node by node, a format, a device) by their parts. A
    value among `known` (by its `id`) is what its caller says of it.

    An array of JAX's is its values and sharding (`_jax_array`); a function `jax.jit` wraps, what it wraps
    and its options (`_jitted`); a program Dew's compile point compiles (`dew.jit`'s), its function and its
    arguments (`_outer_program`). Refused, naming `at` and the value's type, for anything else: an object
    with no such form could differ between two processes with nothing in the identity to show it."""
    return _stated(value, at, known, ())


def _stated(value, at: str, known: Mapping[int, JSON], walking: tuple[int, ...]) -> JSON:
    """`stated`, within the functions and containers `walking` whose reads or parts are being stated, the
    path to `value` in order: a function among them met again (one that names itself) is that function,
    stated once; a container among them met again is a reference back to it, by how far up the path it
    stands (`met_again`)."""
    again = functools.partial(_stated, known=known, walking=walking)
    if id(value) in known:
        return known[id(value)]
    if (plain := _data(value, at)) is not None:
        return plain
    # A value of exactly Python's own type; a subclass (an enumeration's member) is stated as what it is.
    if value is None or type(value) in (bool, int, str):
        return value
    if (computed := _computed(value, at, again)) is not None:
        return computed
    if isinstance(value, types.ModuleType):
        # A module of the user's is everything it holds: code that names it may read any of it.
        held = _module_state(value, at, known, walking) if _sources.own_source(value.__name__) else None
        return ["module", _coded(value.__name__, at), held]
    if isinstance(value, logging.Logger):
        return ["logger", value.name]
    if isinstance(value, (types.FunctionType, type)) and _sources.own_source(value.__module__):
        return _own(value, at, known, walking)
    if _sources.hand_written(getattr(value, "__wrapped__", None)):
        return _wrapping_users(value, at, again)
    if (named := _named(value, at)) is not None:
        return named
    if isinstance(value, type):
        # A class its module holds under no name of its own (`NoneType`, an extension's) is still its
        # module's, by its name; one made inside a function is one of however many that function made.
        if "<locals>" in value.__qualname__:
            raise Unestablished(f"{at}: the class {value.__module__}.{value.__qualname__}, made inside a "
                                "function")
        module = _coded(value.__module__, at)
        return ["class", module, value.__qualname__]
    if isinstance(value, (tuple, list, set, frozenset, Mapping, functools.partial, types.MethodType)) or (
            _sources.generated_instance(value)):
        # A container met again within itself (a mapping one of whose values is that mapping, a registry
        # holding a function that reads it) is a reference back to where the path met it, as a function
        # met again is itself: finite, where walking it recursed past Python's limit and crashed the read
        # (the base review of b4cabd6c0), and stated, where refusing it refused an unchanged step's exact
        # resume (train 947's review). One met twice apart, in two branches, is stated twice: `walking` is
        # the path's alone.
        if id(value) in walking:
            return met_again(value, walking)
        again = functools.partial(_stated, known=known, walking=(*walking, id(value)))
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
        return ["function", module, value.__qualname__, _code(value.__code__),
                reads(value, at, known, walking)]
    if _sources.generated_instance(value):
        # An instance of a class a generator made (a dataclass's, attrs', its validators' among them) by its
        # fields: one unset (an attrs field with no init) is None.
        return ["record", again(type(value), at=at),
                [[name, again(getattr(value, name, None), at=f"{at}.{name}")]
                 for name in _sources.field_names(type(value))]]
    # The algebra read where a value calls for it, never at load: importing any dew.algebra module runs the
    # package's __init__, which imports cost, program and scan, and they reach dew.nn.kernels, whose compile
    # point and coordination read this module (`Unestablished`, `stated`), a cycle at load.
    from dew.algebra import terms

    key = terms.parameter_key(value)
    if isinstance(key, tuple) and key[0] != "object":
        return ["native", _plain(key, at)]
    if (parts := _parts(value, at, again)) is not None:
        return parts
    if (converted := _converter(value, at, again)) is not None:
        return converted
    raise Unestablished(f"{at}: a {type(value).__module__}.{type(value).__qualname__} has no form an "
                        "identity can state")




def _wrapping_users(value, at: str, again: Callable[..., JSON]) -> JSON:
    """A callable that is no function and wraps one of the user's, a jitted one aside (`_computed`): a
    function with JAX's custom derivative is its function, its rules and the options a trace reads, as the
    one module naming them reads it (`dew.nn.derivatives.custom_rule`, lint_slop's SLOP026; row 10087: an
    arm holding a test's own rule ran unprepared). Refused for any other: what it holds beside the function
    (a cache's entries) is its type's, which nothing here states."""
    from dew.nn.derivatives import custom_rule

    if (rule := custom_rule(value)) is not None:
        kind, parts = rule
        return [kind, [[name, again(part, at=f"{at} (its {name})")] for name, part in parts.items()]]
    raise Unestablished(f"{at}: a {type(value).__module__}.{type(value).__qualname__} wrapping a "
                        "function of the user's own code")


def _computed(value, at: str, again: Callable[..., JSON]) -> JSON | None:
    """One of JAX's values a program computes from or with: an array (`_jax_array`), a function `jax.jit`
    wraps, by its module's name for it where a library's (`_named`), else by what it wraps (`_jitted`), a
    program Dew's compile point compiles, by its function and arguments (`_outer_program`); None for any
    other."""
    if isinstance(value, jax.Array):
        return _jax_array(value, at, again)
    if (program := _outer_program(value, at, again)) is not None:
        return program
    if isinstance(value, PjitFunction):
        # A library's jitted function its module names (`jax.nn.silu`) is that name, as the library's code
        # is its version's; one wrapping the user's code is what it wraps, with its options.
        if not _sources.hand_written(value.__wrapped__) and (named := _named(value, at)) is not None:
            return named
        return _jitted(value, at, again)
    return None


def _outer_program(value, at: str, again: Callable[..., JSON]) -> JSON | None:
    """A program Dew's compile point compiles (`dew.nn.kernels.options.OuterProgram`: `dew.jit`'s,
    `program`'s) by its kind, the function it compiles and the jit arguments it compiles it under (its
    compiler options among them), each stated as any value is, and refused naming that part where it has no
    form: never what it holds as it runs (its jit, its executables, their cache and its lock), which a process
    holds of its own and which two equal programs hold apart. None for any other value."""
    # The kernels' options read where a value calls for them, never at load (the algebra's account below).
    from dew.nn.kernels.options import OuterProgram

    if not isinstance(value, OuterProgram):
        return None
    return ["outer program", again(value.function, at=f"{at} (its function)"),
            again(value.arguments, at=f"{at} (its arguments)")]


def _jitted(value, at: str, again: Callable[..., JSON]) -> JSON:
    """A function `jax.jit` wraps by what it wraps (`__wrapped__`, stated as any function is: a user's own by
    its code and reads) and every option its wrapping records (`PjitInfo`: static and donated arguments,
    shardings and layouts, their trees, `keep_unused`, `inline`, a backend or device, compiler options),
    each stated as its value is: two wrappings of one function under other options trace other
    programs."""
    recorded = value._jit_info
    options = {name: _jit_option(getattr(recorded, name), f"{at} (jit option {name})", again)
               for name in sorted(_JIT_OPTIONS)}
    return ["jit", again(value.__wrapped__, at=at), options]


def _jit_option(value, at: str, again: Callable[..., JSON]) -> JSON:
    """One option a jit records, a tuple of shardings or layouts entry by entry, JAX's mark of a sharding
    the call left to JAX (`UNSPECIFIED`) as that."""
    if value is UNSPECIFIED:
        return "unspecified"
    if isinstance(value, tuple):
        return [_jit_option(each, f"{at}[{index}]", again) for index, each in enumerate(value)]
    return again(value, at=at)


_JIT_OPTIONS = frozenset({"static_argnums", "static_argnames", "donate_argnums", "donate_argnames",
                          "in_shardings_treedef", "in_shardings_leaves", "out_shardings_treedef",
                          "out_shardings_leaves", "in_layouts_treedef", "in_layouts_leaves",
                          "out_layouts_treedef", "out_layouts_leaves", "keep_unused", "inline", "backend",
                          "device", "compiler_options_kvs", "user_specified_in_shardings"})
"""The parts of a jit's recorded options (`PjitInfo`) that decide what it traces and compiles; its source
line, signature and counters, which a function's own statement holds or which decide nothing, apart."""


def _jax_array(value: jax.Array, at: str, again: Callable[..., JSON]) -> JSON:
    """An array of JAX's by its values, fetched to the host, as a NumPy array's (`_data`: its dtype, shape
    and bytes; a typed PRNG key's its key type and key data, `terms.held_bits`), its weak type and its
    sharding: a trace that closes over it bakes all of them into its program, and may branch on its weak
    type. A typed key's dtype is stated as any value is, by its implementation's parts, which its printed
    tag alone does not tell apart (two implementations may print one). Refused where this process holds
    only part of it, whose bytes no one process reads."""
    if not value.is_fully_addressable:
        raise Unestablished(f"{at}: an array of JAX's this process holds only part of")
    # A typed PRNG key's bits are its key data, its dtype the key type, which NumPy cannot read whole.
    from dew.algebra import terms  # read where called, as `_stated` reads it

    dtype, bits = terms.held_bits(value)
    return ["jax array", again(dtype, at=f"{at}.dtype"), jax.typeof(value).weak_type, _data(bits, at),
            again(value.sharding, at=f"{at}.sharding")]


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


def _traceback(value) -> JSON | None:
    """A traceback of JAX's by its frames (`traceback_frames`): a checkified program's error tree holds each
    check's in its node data, which tells where a check that fails was; None for any other value."""
    if not isinstance(value, xla_client.Traceback):
        return None
    frames: list[JSON] = list(traceback_frames(value))
    return ["traceback", frames]


def traceback_frames(traceback) -> list[str]:
    """Each frame of a JAX `traceback` as its module, line and function (`dotted.module:12 (function)`), where
    its code is wherever the run is installed: so two programs whose checks sit on other lines are two, and
    one of the same code is one across installs. The one statement of a traceback, a program file's too
    (`dew.training.ahead.HeldTraceback`)."""
    return [f"{_frame_module(frame.file_name)}:{frame.line_num} ({frame.function_name})"
            for frame in traceback.frames]


def _frame_module(file: str) -> str:
    """The module a frame's `file` is, by the longest import path entry it lies under (`a/b.py` under it is
    `a.b`), each entry as Python reads it (`''` the working directory, as under `-c`, a REPL or a notebook),
    else its file's name: no directory where the run is installed, and one module however Python was
    launched."""
    real = os.path.realpath(file)
    roots = [root for entry in sys.path
             if real.startswith(root := os.path.join(os.path.realpath(entry or os.curdir), ""))]
    if not roots:
        return os.path.basename(file)
    relative = os.path.splitext(real[len(max(roots, key=len)):])[0]
    return relative.replace(os.sep, ".").removesuffix(".__init__")


def _parts(value, at: str, again: Callable[..., JSON]) -> JSON | None:
    """One of JAX's values that a call's key holds and the algebra's key does not state, by its parts
    (each through `again`, `stated` as its caller walks): a tree's structure, a format,
    one device's sharding, a device, and any other of JAX's by its attributes where it holds them plainly
    (an axis environment's sizes and names), the layout XLA chooses (`Layout.AUTO`), a traceback
    (`_traceback`). None for a value that is none of them."""
    if (frames := _traceback(value)) is not None:
        return frames
    if isinstance(value, type(jax.tree.structure(()))):
        return tree_shape(value, at, again)
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
    unset = [name for name in names if not hasattr(value, name)]
    if unset:
        # A slot its object leaves unset (a trace's equation context) holds no value to state.
        raise Unestablished(f"{at}: a {type(value).__module__}.{type(value).__qualname__} leaves "
                            f"{', '.join(sorted(unset))} unset, which has no form an identity can state")
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
          walking: tuple[int, ...] = ()) -> JSON:
    """What `function`'s body reads outside its arguments (`dew._sources.function_reads`), as an identity
    states it (`stated`): the cells it closed over, its defaults, and each name its code or the code nested
    in it reads that its globals hold or that is a loaded module of the user's. A trace bakes each such
    value into its program, so one that differs is another call. A library's function (a local one
    `_named` cannot name: `jax.checkpoint`'s, `jax.grad`'s) reads its module's globals as its installed
    version holds them, which its module's record states (`_coded`, as a module's function `_named`
    names), so they are not walked (`library_function`): its closure and defaults are, a user's function
    among them (rows 10064 on: a library's caches among its globals, no arm holding Dew's SwiGLU had a
    statement)."""
    within = (*walking, id(function))
    named = function.__qualname__
    cells, read = _sources.function_reads(function)
    return {"closure": {name: _stated(value, f"{at} ({named}'s closure, {name})", known, within)
                        for name, value in sorted(cells.items())},
            "globals": {name: _stated(value, f"{at} ({named}'s global {name})", known, within)
                        for name, value in sorted(read.items())} if not library_function(function) else {},
            "defaults": _stated((function.__defaults__, function.__kwdefaults__),
                                f"{at} ({named}'s defaults)", known, within)}


def library_function(function: types.FunctionType) -> bool:
    """Whether `function` is a library's: its module one of the user's own code is not (`own_source`: an
    installed distribution's, the standard library's). A script's, a notebook's (`__main__` with no file)
    and one no module holds are the user's."""
    return function.__module__ not in (None, "__main__") and not _sources.own_source(function.__module__)


def _own(value, at: str, known: Mapping[int, JSON], walking: tuple[int, ...]) -> JSON:
    """A function or a class of a user's own code (`dew._sources.own_source`) by its name there, with what
    it reads: a function's reads (`reads`); a class's data, each attribute its body or other code set on
    it, and its members' reads (`_members`), its bases' of the user's code among them. Such code reaches
    the program through a name alone, and what it reads at run time (a module's setting another module
    assigned) is nothing its source states, so it is walked, through every function, class and module of
    the user's it names in turn; one met again within its own walk is itself, stated once. A class's
    classes are its MRO's in order, each whole (`_class`: its members, its dataclass fields with their
    defaults and factories, a metaclass of the user's; keyed by name alone, a base's members replaced its
    child's of the same name, the child's edit unseen, the base review of 201a04c58). No file's digest is
    part of it: its code and its reads state it completely, a script's (`__main__`) as any module's, and a
    script's other bytes are no step's or program's (an unrelated edit of `__main__` refused a class step's
    exact resume, the same review; a class stated without its fields' defaults and factories handed a
    kept program back over an edit of them, train 947's review)."""
    module = _coded(value.__module__, at)
    stated_name = ["own", module, value.__qualname__]
    if id(value) in walking:
        return [*stated_name, "itself"]
    if isinstance(value, types.FunctionType):
        # A function made inside another (`<locals>`) among them: its code and what it reads, its closure's
        # cells too, are all it is, however many its maker made.
        return [*stated_name, _code(value.__code__), reads(value, at, known, walking)]
    if "<locals>" in value.__qualname__:
        raise Unestablished(f"{at}: the class {value.__module__}.{value.__qualname__}, made inside a "
                            "function")
    within = (*walking, id(value))
    return [*stated_name, [_class(held, at, known, within) for held in _sources.own_classes(value)]]


def _class(held: type, at: str, known: Mapping[int, JSON], walking: tuple[int, ...]) -> JSON:
    """One class of a user's own class's MRO, completely: its module and name, its members (`_members`), its
    generated fields (`_field`: a dataclass's or an attrs class's), and its
    metaclass where that is the user's own code too (which runs as the class is made and may set what it
    holds)."""
    fields: list[JSON] = [_field(held, field, at, known, walking)
                          for field in _sources.generated_fields(held)]
    maker = type(held)
    made = (_stated(maker, f"{at} ({held.__name__}'s metaclass)", known, walking)
            if _sources.own_source(maker.__module__) else None)
    entry: list[JSON] = [_coded(held.__module__, at), held.__qualname__, _members(held, at, known, walking),
                         fields, made]
    # How its generator made it, as it records its options (`generation_options`).
    options = _sources.generation_options(held)
    return [*entry, dict(options)] if options else entry


def _converter(value, at: str, again: Callable[..., JSON]) -> JSON | None:
    """An `attrs.Converter` (attrs 24.1 and later) by its converter and how it is called, wherever the walk
    meets one: a field's own, or one nested in `attrs.converters.pipe` or `optional`'s closure; None for any
    other value. Unwrapped in a field's own slot alone, a nested one left its step inexact (train 957's
    review)."""
    attr = sys.modules.get("attr")
    converter_type = None if attr is None else vars(attr).get("Converter")
    if converter_type is None or not isinstance(value, converter_type):
        return None
    return ["converter", again(value.converter, at=f"{at}'s converter"), value.takes_self, value.takes_field]


def _field(held: type, field: _sources.GeneratedField, at: str, known: Mapping[int, JSON],
           walking: tuple[int, ...]) -> JSON:
    """A generated field (`dew._sources.generated_fields`) as an identity states it: its name, default and
    default factory, and every other part its generator recorded of it, each by its part's name."""
    stated: list[JSON] = [field.name,
                          _field_part(field.default, f"{at} ({held.__name__}.{field.name})", known, walking),
                          _field_part(field.factory, f"{at} ({held.__name__}.{field.name} factory)", known,
                                      walking)]
    if field.parts:
        stated.append({part: _stated(value, f"{at} ({held.__name__}.{field.name} {part})", known, walking)
                       for part, value in field.parts})
    return stated


def _field_part(value, at: str, known: Mapping[int, JSON], walking: tuple[int, ...]) -> JSON:
    """A generated field's default or default factory as an identity states it, `"missing"` where it has
    none (`dew._sources.MISSING`)."""
    return "missing" if value is _sources.MISSING else _stated(value, at, known, walking)


def met_again(value, walking: tuple[int, ...]) -> JSON:
    """A container met again on its own path (`walking`), as a reference back to it: its type and how many
    steps up the path it stands, the same wherever the path began."""
    return ["again", f"{type(value).__module__}.{type(value).__qualname__}",
            len(walking) - walking.index(id(value))]


def _members(held, at: str, known: Mapping[int, JSON], walking: tuple[int, ...]) -> JSON:
    """What a class or a module of the user's holds, each under its name (`dew._sources.members`), as an
    identity states it (`stated`): a method by the function its descriptor wraps, anything else by its
    value, a partial or a bound method whole."""
    return {name: _stated(member, f"{at} ({held.__name__}.{name})", known, walking)
            for name, member in _sources.members(held)}


def _module_state(module: types.ModuleType, at: str, known: Mapping[int, JSON],
                  walking: tuple[int, ...]) -> JSON:
    """Everything a module of the user's holds (`_members`), for code that names the module itself and so
    may read any of it; a module met again within its own walk is itself."""
    if id(module) in walking:
        return "itself"
    return _members(module, at, known, (*walking, id(module)))


def tree_shape(tree, at: str, again: Callable[..., JSON]) -> JSON:
    """A tree's structure node by node: a leaf, or a node's type, the data it holds beside its children (a
    mapping's keys, a custom node's static fields) and its children, each type and datum as `again` states
    it (`stated`'s, or a kept draw's key's, `dew.objectives.drawn.held_signature`). Never as JAX prints it,
    which holds a node's data by its printed form."""
    node = tree.node_data()
    if node is None:
        return "leaf" if tree.num_leaves else "none"
    kind, held = node
    return ["node", again(kind, at=at), again(held, at=f"{at} (a {kind.__qualname__}'s node data)"),
            [tree_shape(child, at, again) for child in tree.children()]]


def source_digest(path: Path) -> str:
    """The digest of the file at `path`'s bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
