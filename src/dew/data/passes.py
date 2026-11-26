"""A stream a repeat reads pass after pass (`dew.data.lowering._repeated`), each pass going on counting its
corpus, a pass that yields no row refused where that is certain to repeat."""

from __future__ import annotations

import logging

import grain.python as pygrain

from dew.data.handoff import _Barren
from dew.data.mixing import rewound

_log = logging.getLogger("dew.data.lowering")


class Passes(pygrain.IterDataset):
    """A stream a repeat reads pass after pass, each going on counting its corpus. Endlessly read
    (`endless`), a pass that yields no row is refused where that is certain to repeat (`certain`),
    and said at each empty pass after the first otherwise, naming the steps that may yield another row
    on a later pass (`varying`)."""

    def __init__(self, parent: pygrain.IterDataset, name: str, *, endless: bool, certain: bool,
                 barren: bool = False, varying: tuple[str, ...] = ()):
        super().__init__(parent)
        self._rows, self._name, self._endless, self._certain = parent, name, endless, certain
        self._barren, self._varying = barren, varying

    def __iter__(self) -> pygrain.DatasetIterator:
        return PassesIterator(self._rows.__iter__(), self._name, self._endless, self._certain, self._barren,
                               self._varying)


class PassesIterator(pygrain.DatasetIterator):
    """Its parent's rows and position. The repeat restarts a pass by restoring, right after one
    ends, the very position it took as the first pass's start (Grain's `_parent_starting_state`,
    the first this iterator gives); any other restore, a checkpoint's even once the last pass
    ended, lands where its position says, with the counts it holds."""

    def __init__(self, parent: pygrain.DatasetIterator, name: str, endless: bool, certain: bool,
                 barren: bool = False, varying: tuple[str, ...] = ()):
        super().__init__(parent)
        self._rows, self._name, self._endless, self._certain = parent, name, endless, certain
        self._barren, self._varying = barren, varying
        self._yielded = self._ended = False
        self._empty = 0
        self._opening: object | None = None

    def __next__(self):
        try:
            value = next(self._rows)
        except StopIteration:
            if not self._yielded and self._barren:
                raise _Barren(self._name) from None
            if not self._yielded and self._endless:
                if self._certain:
                    raise ValueError(f"a pass of {self._name} yields no rows, so repeating it reads "
                                     "nothing forever") from None
                self._empty += 1
                # Every empty pass after the first, loudly: a read that is not moving says why it reads on.
                if self._empty > 1:
                    _log.warning("%s yielded no row in its last %d passes; it reads on, as %s may make "
                                 "a row on a later pass", self._name, self._empty,
                                 ", ".join(self._varying) or "a step before its repeat")
            self._ended = True
            raise
        self._yielded, self._empty = True, 0
        return value

    def get_state(self):
        state = self._rows.get_state()
        if self._opening is None:
            self._opening = state
        return state

    def set_state(self, state) -> None:
        # The repeat rewinding the pass that just ended keeps the corpus's counts.
        rewinding = self._ended and state is self._opening
        self._rows.set_state(rewound(state, self._rows.get_state()) if rewinding else state)
        self._yielded, self._ended = not rewinding, False
