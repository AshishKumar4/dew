"""One compilation of a data chain (`compiling`): what it reads once of what may change between two,
the checkout and each source's fingerprint."""

from __future__ import annotations

import contextvars
import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import Protocol

from dew import _checkout


class Fingerprinted(Protocol):
    """A store that states its content (`dew.data.sources.base.Source.fingerprint`)."""

    def fingerprint(self) -> Mapping[str, object]: ...


@dataclasses.dataclass
class _Compilation:
    """What one compilation reads once of what may change between compilations: the checkout's
    revision and cleanness, and each source's fingerprint (`once`)."""

    checkout: tuple[str, bool] | None = None
    fingerprints: dict[int, tuple[Fingerprinted, Mapping[str, object]]] = dataclasses.field(
        default_factory=dict)


_COMPILING: contextvars.ContextVar[_Compilation | None] = contextvars.ContextVar("dew_data_compiling",
                                                                                 default=None)


def compiling[**Arguments, Result](function: Callable[Arguments, Result]) -> Callable[Arguments, Result]:
    """`function`, an entry point of the compiler, run as one compilation: every manifest it makes
    names one reading of the checkout, and each source is fingerprinted once (`once`); a compilation
    inside it is part of it."""
    @functools.wraps(function)
    def compiled(*arguments: Arguments.args, **options: Arguments.kwargs) -> Result:
        if _COMPILING.get() is not None:
            return function(*arguments, **options)
        token = _COMPILING.set(_Compilation())
        try:
            return function(*arguments, **options)
        finally:
            _COMPILING.reset(token)

    return compiled


def once(source: Fingerprinted, fingerprinted: Callable[[], Mapping[str, object]]) -> Mapping[str, object]:
    """`source`'s fingerprint, `fingerprinted()`, computed once a compilation (`compiling`): a plan,
    its caches' keys and descriptions each read it, and its records may change between two."""
    compilation = _COMPILING.get()
    if compilation is None:
        return fingerprinted()
    held = compilation.fingerprints.get(id(source))
    if held is None or held[0] is not source:
        held = (source, fingerprinted())
        compilation.fingerprints[id(source)] = held
    return held[1]


def checkout_now() -> tuple[str, bool]:
    """The checkout's revision and whether it is clean, read once a compilation (`compiling`)."""
    compilation = _COMPILING.get()
    if compilation is not None and compilation.checkout is not None:
        return compilation.checkout
    read = _checkout.revision() or "unversioned", _checkout.clean()
    if compilation is not None:
        compilation.checkout = read
    return read
