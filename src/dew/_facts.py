"""The facts a trace reads of the process or its host (`traced_fact`), recorded as they are read
(`recording`), so a value made from them holds where each answers as it did (`still`) and is read again
wherever it is taken (`replay`): what `dew._cache.holding` holds a value with. Imports no JAX as it loads,
as the device module that declares most facts (`dew.nn.kernels.generation`) does not."""

import contextlib
import contextvars
import dataclasses
import functools
from collections.abc import Callable, Hashable, Iterator, Mapping

type Facts = dict[Hashable, tuple[Callable[[], Hashable], Hashable]]
"""The facts a making read (`recording`): each by its reader and question, with a reader asking it again
and the answer it got."""

type Held[Value] = tuple[Value, Facts]
"""A held value with the facts its making read (`holding`)."""

_RECORDINGS: contextvars.ContextVar[tuple[Facts, ...]] = contextvars.ContextVar("fact_recordings", default=())
"""The facts read within each open `recording`, innermost last: what a trace records into, no trace reads,
so a trace held by its state (`dew.nn.kernels.options.trace_context`) is held in any recording's scope
(`recording` registers it `dew._cache.untraced` as it first sets it: this module imports no JAX as it
loads)."""


def stated(value) -> Hashable:
    """`value` as a fact's answer or question is held and compared: by its contents, equal wherever they
    are however often its holder is built anew (a reader returning a fresh mapping): a hashable value by
    itself, a mapping, sequence or set by its members, a dataclass by its type and fields; an array, which a
    program reading it closes over, by itself (`Same`), as an object that states none of these is."""
    import jax

    from dew._cache import Same

    if isinstance(value, jax.Array):
        return Same(value)
    if isinstance(value, Mapping):
        return tuple(sorted(((key, stated(answer)) for key, answer in value.items()), key=repr))
    if isinstance(value, list | tuple):
        return tuple(stated(element) for element in value)
    if isinstance(value, set | frozenset):
        return frozenset(stated(member) for member in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return type(value), tuple(stated(getattr(value, field.name)) for field in dataclasses.fields(value))
    return value if isinstance(value, Hashable) else Same(value)


@contextlib.contextmanager
def recording() -> Iterator[Facts]:
    """Each fact read within the block (`traced_fact`), by its reader and question, with the answer it got
    (`stated`): what a trace kept for a later one read, which holds where each answers as it did (`still`).
    Recordings nest; a read records into each open one."""
    from dew import _cache

    _cache.untraced(_RECORDINGS)
    reads: Facts = {}
    token = _RECORDINGS.set((*_RECORDINGS.get(), reads))
    try:
        yield reads
    finally:
        _RECORDINGS.reset(token)


def still(reads: Facts) -> bool:
    """Whether each fact of `reads` (a `recording`'s) answers now as it did when it was read."""
    return all(asked() == answer for asked, answer in reads.values())


def replay(reads: Facts) -> None:
    """`reads`, a held value's (`recording`'s), recorded into each open recording, as if read again: a trace
    taking a held value read what its making read."""
    for recorded in _RECORDINGS.get():
        recorded.update(reads)


def traced_fact[**Question, Answer](reader: Callable[Question, Answer]) -> Callable[Question, Answer]:
    """Declare `reader` a fact a trace reads of the process or its host (the memory a device holds, a wheel's
    presence, a launch config a compile refused), where it is defined: a program compiled under one answer
    is another under the next (a route taken, a kernel refused). Its reader is the fact's one accessor: a
    read records its question and answer into each open `recording`, so a held value holds where each answer
    its making read holds (`still`). A fact read but by its reader is an input no recording holds, a stale
    value's."""
    name = f"{reader.__module__}.{reader.__qualname__}"

    @functools.wraps(reader)
    def read(*question: Question.args, **named: Question.kwargs) -> Answer:
        answer = reader(*question, **named)
        if recordings := _RECORDINGS.get():
            held = stated((question, named))
            for recorded in recordings:
                recorded[name, held] = (lambda: stated(reader(*question, **named))), stated(answer)
        return answer

    return read
