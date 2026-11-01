"""A value an instance computes once, under a lock of its own, for the readers whose threads first read it at
once: an index's lease, a file's table, a cache's writer, each read by a reader its worker unpickled."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Self, overload

LOCK = "_once"
"""The key of the lock an instance's values are computed under, in its `__dict__`, which a state it pickles
leaves out, as it leaves out the values themselves."""


class locked_cached_property[Owner, Value]:
    """`functools.cached_property` computed under a lock of its instance's own, which Python 3.12's no longer
    takes: of the threads that first read it at once, one computes it and the rest read what it computed,
    and another instance's first read waits on none of it. The lock is made in the instance's `__dict__`
    (`LOCK`) as it is first needed, so an instance a worker unpickled makes its own; it is reentrant, so one
    such value's computation may read another of the instance's."""

    def __init__(self, compute: Callable[[Owner], Value]) -> None:
        self.compute = compute
        self.__doc__ = compute.__doc__

    def __set_name__(self, owner: type[Owner], name: str) -> None:
        self.name = name

    @overload
    def __get__(self, instance: None, owner: type[Owner]) -> Self: ...

    @overload
    def __get__(self, instance: Owner, owner: type[Owner] | None = None) -> Value: ...

    def __get__(self, instance: Owner | None, owner: type[Owner] | None = None) -> Self | Value:
        if instance is None:
            return self
        held = instance.__dict__
        with held.setdefault(LOCK, threading.RLock()):
            if self.name not in held:
                held[self.name] = self.compute(instance)
            return held[self.name]

    def taken(self, instance: Owner) -> Value | None:
        """`instance`'s value, where it computed one, no longer held: taken under the lock its first read
        takes, so a close never takes a value a thread is still computing and leaves the new one held."""
        with instance.__dict__.setdefault(LOCK, threading.RLock()):
            return instance.__dict__.pop(self.name, None)
