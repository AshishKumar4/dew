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
retrace when a global a traced function reads is assigned. `dew.training.ahead.kept` proves each handed
call by default, at the cost of the trace: it lowers it and holds it to the kept program's digest; its
`verify=False` trusts this identity alone.

This module is the boundary where Dew reads values it has no type for: any object a call names, any
module a process loaded. Its private readers ask each what Python's object model says of it (its
module, its name, its file, its slots) and refuse what answers nothing.
"""

from __future__ import annotations

import contextvars
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import threading
import types
from collections.abc import Hashable
from pathlib import Path

import jax
from jax._src import config as jax_config

from dew import _cache, _sources
from dew._records import JSON
from dew._stated import Unestablished, reads, source_digest, stated
from dew.nn.kernels.generation import TRACED_FACTS
from dew.nn.kernels.options import COMPILE_POINT, OuterProgram

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
    body reads outside its arguments (`_read`), how it is jitted, the key (its static arguments by value,
    the others' structure, shapes, dtypes and placements, JAX's trace context with Dew's own user contexts
    in it, the mesh), JAX's other settings, Dew's environment and the declared facts (`settings`,
    `environment`, `facts`). With its process's code (`Process`) it is what the call's trace reads, so two
    calls of one identity in two processes of one record trace one program. Refused where any of it has no
    stated form (`Unestablished`)."""
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
            "reads": (reads(body, "the program's function", known)
                      if isinstance(body, types.FunctionType) else None),
            "jitted": stated(program.arguments, f"{named}'s jit arguments", known),
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
    package or script beside it; a function or class of `__main__` a call reaches is stated by its code
    and what it reads, as any of the user's own code is, `dew._stated._own`, the script's other lines by
    none). With them, the settings, environment and facts as the process held them when it saved
    (`settings`, `environment`, `facts`):
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
        installed, standard, packages = _sources.installed(), _sources.standard(), _sources.distributed()
        distributions: dict[str, str] = {}
        sources: dict[str, str] = {}
        for name in sorted(sys.modules):
            path = _sources.recorded(name)
            if path is None:
                continue
            placed = _sources.within(path, installed)
            owners = packages.get(name.partition(".")[0], ()) if placed else ()
            if owners:
                distributions.update((owner, importlib.metadata.version(owner)) for owner in owners)
            elif placed or not _sources.within(path, standard):
                sources[name] = source_digest(path)
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
        installed = {owner for owners in _sources.distributed().values() for owner in owners}
        for distribution, version in self.distributions:
            found = importlib.metadata.version(distribution) if distribution in installed else None
            if found != version:
                differing[f"distribution {distribution}"] = f"{version}, not {found}"
        for name, source in self.sources:
            file = _found(name)
            if file is None or source_digest(file) != source:
                differing[f"source {name}"] = "another source, or none here"
        return differing

    def held(self) -> None:
        """Refused, naming what differs, where this process is not the record's (`differences`)."""
        if differing := self.differences():
            raise ValueError("these programs were traced by another process's code or settings than this "
                             f"one's: {'; '.join(f'{name} {said}' for name, said in differing.items())}")


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
