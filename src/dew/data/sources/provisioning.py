"""The store that prepares a read a source cannot take yet: one capability walker, beneath the views whose
rows are their store's and into a concatenation's stores, for `source(..., provision=True)`, the analysis
of a read, and a holdout's placing (row 5755)."""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

from dew.data.sources.base import SAME_ROWS, Read, Source, View, beneath
from dew.data.sources.concatenate import Concatenate


@runtime_checkable
class Provisionable(Protocol):
    """A store that prepares a read it cannot take yet (an index, a local mirror)."""

    def can_provision(self, read: Read) -> bool: ...
    def provision(self) -> Source: ...
    def preparation_bytes(self) -> int | None:
        """The bytes preparing reads, as the store states them before any read, None where it states none
        (row 4118)."""
        ...


@dataclasses.dataclass(frozen=True)
class _ConcatenationPreparers:
    """A concatenation's stores that cannot take a read, each prepared by its own preparer
    (`provisioner`), so the concatenation takes it once they have."""

    concatenation: Concatenate
    preparers: tuple[Provisionable, ...]

    def can_provision(self, read: Read) -> bool:
        return True

    def provision(self) -> Source:
        for preparer in self.preparers:
            preparer.provision()
        return self.concatenation

    def preparation_bytes(self) -> int | None:
        sizes = [preparer.preparation_bytes() for preparer in self.preparers]
        return None if None in sizes else sum(size for size in sizes if size is not None)


def provisioner(source: Source, read: Read) -> Provisionable | None:
    """The source that can prepare `read`, through the views whose rows are their source's
    (`SAME_ROWS`) over it, unless one takes no such read; for a concatenation, each of its stores
    that cannot take `read` yet, all of which must prepare it."""
    preparing = source
    for preparing in beneath(source, SAME_ROWS):
        if isinstance(preparing, Provisionable):
            break
        if isinstance(preparing, View) and read not in preparing.reads:
            return None
    source = preparing
    if isinstance(source, Concatenate):
        preparers = [provisioner(member, read) for member in source.sources if not member.supports(read)]
        if not preparers or any(preparer is None for preparer in preparers):
            return None
        ready = tuple(preparer for preparer in preparers if preparer is not None)
        return _ConcatenationPreparers(source, ready)
    return source if isinstance(source, Provisionable) and source.can_provision(read) else None
