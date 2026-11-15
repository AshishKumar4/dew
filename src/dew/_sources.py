"""Where a process's code comes from: the file a loaded module is code of, whether a module is a user's own
code (loaded from a file no installed distribution versions and outside the interpreter's standard library)
or an installed distribution's, which its version states. One statement, which a traced program's identity
(`dew.training.identity`) and a data plan's step identity (`dew.data.plan`) both read."""

from __future__ import annotations

import functools
import importlib.metadata
import inspect
import site
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from pathlib import Path


def source_file(held: object) -> Path | None:
    """The file `held`, a function or a class, is defined in, None where it has none: a class of a notebook,
    a REPL or `python -c` (`__main__` with no file), a builtin's, or a file since removed."""
    try:
        place = inspect.getsourcefile(held)  # type: ignore[arg-type]
    except (OSError, TypeError):
        return None
    return Path(place) if place is not None and Path(place).is_file() else None


def recorded(name: str) -> Path | None:
    """The file whose source a process record holds for the loaded module `name` (`file_of`), None for one
    it holds none of: a module of no file, a frozen one (the interpreter's own, whatever file it names),
    Dew's (its package digest's) and `__main__` (stated where a call names it)."""
    module = sys.modules[name]
    frozen = getattr(getattr(module, "__spec__", None), "origin", None) == "frozen"
    if (getattr(module, "__file__", None) is None or frozen or name in ("dew", "__main__")
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
