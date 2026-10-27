"""Accepted-row accounting and mixtures measured in valid target tokens."""

from __future__ import annotations

import dataclasses
import itertools
import logging
import math
import sys
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Protocol, TypeGuard, overload, runtime_checkable

import grain.experimental as gx
import grain.python as pygrain
import numpy as np

from dew._records import boolean, integer, record, text
from dew.batch import Batch
from dew.data.dataset import LONG_RUN, logged_run
from dew.data.pipeline import Record, Rejected

_log = logging.getLogger(__name__)


@dataclasses.dataclass
class CorpusCounts:
    """What one corpus's read counted, under its identity (`name`, which plans and saved positions
    key it by) and the short name its metrics go by (`label`). `accepted` counts the rows it
    delivered and `truncated` those of them cut short; a rejection counts what the step that drops
    it reads, a record or a piece or window a step made of one. `attempted` counts what its last
    filter read, for its own warning; it describes no one unit of the corpus, so no metric shows it."""

    name: str
    label: str
    attempted: int = 0
    accepted: int = 0
    truncated: int = 0
    chunks: int = 0
    chunk_records: int = 0
    rejected: Counter[str] = dataclasses.field(default_factory=Counter)

    def metrics(self, label: str) -> dict[str, float]:
        """The metrics a run logs of this corpus under `label`, each exact in one unit: the rows it
        delivered and its rejections by reason as integers, the share of those rows cut short and
        the chunks per chunked record."""
        return {
            f"data/rows/{label}": self.accepted,
            f"data/truncated/{label}": self.truncated / max(1, self.accepted),
            f"data/chunks_per_record/{label}": self.chunks / max(1, self.chunk_records),
            **{f"data/rejected/{label}/{reason}": count for reason, count in self.rejected.items()},
        }

    def state(self) -> Mapping[str, object]:
        return {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "truncated": self.truncated,
            "chunks": self.chunks,
            "chunk_records": self.chunk_records,
            "rejected": dict(self.rejected),
        }

    def restore(self, state: Mapping[str, object]) -> None:
        self.attempted = integer(state["attempted"], "attempted")
        self.accepted = integer(state["accepted"], "accepted")
        self.truncated = integer(state["truncated"], "truncated")
        self.chunks = integer(state["chunks"], "chunks")
        self.chunk_records = integer(state["chunk_records"], "chunk_records")
        self.rejected = Counter(
            {name: integer(value, name) for name, value in record(state["rejected"], "rejected").items()}
        )

    def followed(self, reader: CorpusCounts) -> CorpusCounts:
        """These counts of the prefix a memory read ran, followed by `reader`'s of the stages it
        read after them: the reader delivered the rows and counted those cut short, and the two
        dropped and chunked apart."""
        return dataclasses.replace(
            reader, attempted=max(self.attempted, reader.attempted), chunks=self.chunks + reader.chunks,
            chunk_records=self.chunk_records + reader.chunk_records, rejected=self.rejected + reader.rejected)

    def expanded(self, outputs: Sequence[Record]) -> None:
        """Count one chunked document and its fan-out before later stages can reject chunks."""
        self.chunk_records += 1
        self.chunks += sum(value is not None and not isinstance(value, Rejected) for value in outputs)


_WARN_OVER = 0.01
"""The share of a corpus's attempted records it may reject before a run warns, once: a filter that
drops more than one record in a hundred is worth a look."""

class _KeepAccepted:
    def __init__(self, counts: CorpusCounts, tokens: Callable[[Batch], int] | None,
                 certain_after: int | None = None):
        self.counts, self.tokens, self.certain_after = counts, tokens, certain_after
        self.consecutive = 0
        self.warned = False

    def __call__(self, element: Batch | Rejected) -> TypeGuard[Batch]:
        self.counts.attempted += 1
        reason = element.reason if isinstance(element, Rejected) else None
        if reason is None and self.tokens is not None and not isinstance(element, Rejected):
            count = self.tokens(element)
            if count < 0:
                raise ValueError(f"corpus {self.counts.name} has a negative valid-target count: {count}")
            if count == 0:
                reason = "zero_targets"
        if reason is None:
            self.counts.accepted += 1
            if isinstance(element, Mapping):
                self.counts.truncated += int(bool(element.get("_dew_truncated", False)))
            self.consecutive = 0
            return True
        self.counts.rejected[reason] += 1
        self.consecutive += 1
        if not self.warned and sum(self.counts.rejected.values()) / self.counts.attempted > _WARN_OVER:
            _log.warning("corpus %s rejected more than %.1f%% of attempted records (%s)",
                         self.counts.label, _WARN_OVER * 100, reason)
            self.warned = True
        if self.certain_after is not None and self.consecutive >= self.certain_after:
            reasons = ", ".join(f"{name} {count}" for name, count in self.counts.rejected.most_common(3))
            raise ValueError(
                f"corpus {self.counts.label} rejected its last {self.consecutive} records, every record of "
                f"two whole passes of what it reads ({reasons}), and its steps reject the same records each "
                "pass, so the read would read nothing forever. A read per share filters each share's own "
                "records: read it with fewer shares, or elastically (`.elastic()`), every reader filtering "
                "every record")
        if logged_run(self.consecutive, LONG_RUN):
            reasons = ", ".join(f"{name} {count}" for name, count in self.counts.rejected.most_common(3))
            _log.warning("corpus %s rejected its last %d records and has kept %d; the read goes on until "
                         "one is kept (most rejected: %s)", self.counts.label, self.consecutive,
                         self.counts.accepted, reasons)
        return False


_COUNTER = "dew.corpus_counts"
"""Where a counting iterator's position keeps its corpus's counts, beside its parent's position:
a key of Dew's own, which `state_counts` and `rewound` find and no Grain position holds."""


class _CountedIterator[Element](pygrain.DatasetIterator[Element]):
    """Checkpoint a corpus counter with its parent, including replay during parent restoration."""

    def __init__(self, parent: pygrain.DatasetIterator, counts: CorpusCounts):
        super().__init__(parent)
        self._read, self._counts = parent, counts

    def __next__(self) -> Element:
        return next(self._read)

    def get_state(self) -> Mapping[str, object]:
        return {"parent": self._read.get_state(), _COUNTER: self._counted()}

    def _counted(self) -> Mapping[str, object]:
        return {"corpus": self._counts.name, "label": self._counts.label, "counts": self._counts.state()}

    def set_state(self, state: Mapping[str, object]) -> None:
        self._read.set_state(record(state["parent"], "counted parent"))
        self._restore(record(state[_COUNTER], "corpus counter"))

    def _restore(self, counted: Mapping[str, object]) -> None:
        self._counts.restore(record(counted["counts"], "corpus counts"))

    def close(self) -> None:
        self._read.close()


class Chunked(pygrain.IterDataset[Record]):
    """A chunk expansion counted at its fan-out, with counters in the iterator position."""

    def __init__(self, parent: pygrain.IterDataset[Record], name: str, label: str,
                 expand: Callable[[CorpusCounts], gx.FlatMapTransform]):
        super().__init__(parent)
        self._records, self._name, self._label, self._expand = parent, name, label, expand

    def __iter__(self) -> pygrain.DatasetIterator[Record]:
        counts = CorpusCounts(self._name, self._label)
        return _CountedIterator(iter(gx.FlatMapIterDataset(self._records, self._expand(counts))), counts)


class _AcceptedIterator(_CountedIterator[Batch]):
    _read: pygrain.DatasetIterator[Batch]

    def __init__(self, parent: pygrain.DatasetIterator[Batch], predicate: _KeepAccepted):
        super().__init__(parent, predicate.counts)
        self._predicate = predicate

    def _counted(self) -> Mapping[str, object]:
        return {**super()._counted(), "consecutive": self._predicate.consecutive,
                "warned": self._predicate.warned}

    def _restore(self, counted: Mapping[str, object]) -> None:
        super()._restore(counted)
        self._predicate.consecutive = integer(counted["consecutive"], "consecutive")
        self._predicate.warned = boolean(counted["warned"], "warned")


class Accepted(pygrain.IterDataset[Batch]):
    """Grain's filter, with checkpointed counts. A filter of an endless repeat whose passes it
    rejects alike refuses a run of `certain_after` rejections, two whole passes, as reading nothing
    forever; any other run of rejections it logs and reads on, a fetch that failed for now say."""

    def __init__(
        self,
        parent: pygrain.IterDataset[Batch | Rejected] | pygrain.IterDataset[Batch],
        name: str,
        label: str,
        *,
        tokens: Callable[[Batch], int] | None = None,
        certain_after: int | None = None,
    ):
        super().__init__(parent)
        self._records, self._name, self._label = parent, name, label
        self._tokens, self._certain_after = tokens, certain_after

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        counts = CorpusCounts(self._name, self._label)
        predicate = _KeepAccepted(counts, self._tokens, self._certain_after)
        return _AcceptedIterator(iter(self._records.filter(predicate)), predicate)


def token_weights(
    names: Sequence[str], weights: Sequence[float], windows: Sequence[int], targets: Sequence[int]
) -> tuple[float, ...]:
    """Window shares for authoritative per-corpus valid-target counts."""
    adjusted = []
    for name, weight, rows, tokens in zip(names, weights, windows, targets, strict=True):
        if tokens <= 0:
            raise ValueError(
                f"corpus {name} has zero valid target tokens"
                if tokens == 0
                else f"corpus {name} has a negative valid-target count"
            )
        if rows <= 0:
            raise ValueError(f"corpus {name} has no accepted windows")
        adjusted.append(weight * rows / tokens)
    return tuple(adjusted)


class _DeficitIterator(pygrain.DatasetIterator[Batch]):
    """Each row from the corpus furthest below its share of the targets delivered, each corpus
    read through its first row ahead (`_PrimeReads`)."""

    def __init__(
        self,
        reads: Sequence[_PrimedRead],
        weights: tuple[int, ...],
        tokens: Callable[[Batch], int],
        keys: tuple[str, ...],
    ):
        super().__init__(reads)
        self._reads, self._weights, self._tokens, self._keys = tuple(reads), weights, tokens, keys
        self._delivered = [0] * len(reads)

    def __next__(self) -> Batch:
        total = sum(self._delivered)
        scale = sum(self._weights)
        chosen = max(
            range(len(self._reads)),
            key=lambda index: self._weights[index] * total - self._delivered[index] * scale,
        )
        value = next(self._reads[chosen])
        tokens = self._tokens(value)
        if tokens <= 0:
            raise RuntimeError(
                "token deficit mix received a window without valid targets; filter it before mixing"
            )
        self._delivered[chosen] += tokens
        return value

    def get_state(self) -> Mapping[str, object]:
        return {
            "parents": {key: read.get_state() for key, read in zip(self._keys, self._reads, strict=True)},
            "delivered": dict(zip(self._keys, self._delivered, strict=True)),
        }

    def set_state(self, state: Mapping[str, object]) -> None:
        """Restore each corpus the state keeps by its key, and start one it keeps none of where its
        weight's share of the targets the others delivered stands, so it joins at its share rather
        than read alone until it catches up. A corpus the state keeps and this mix has not is
        dropped."""
        parents = record(state["parents"], "token deficit corpora")
        delivered = {key: integer(count, "delivered targets")
                     for key, count in record(state["delivered"], "delivered targets").items()}
        if set(delivered) != set(parents):
            raise ValueError("token deficit state keeps positions and deliveries of other corpora")
        if any(count < 0 for count in delivered.values()):
            raise ValueError("token deficit state has negative delivered targets")
        kept = [key in parents for key in self._keys]
        for key, read in zip(self._keys, self._reads, strict=True):
            if key in parents:
                read.set_state(record(parents[key], "corpus iterator"))
        restored = [delivered[key] for key in self._keys if key in parents]
        total, weight = sum(restored), sum(w for w, held in zip(self._weights, kept, strict=True) if held)
        self._delivered = [delivered[key] if held else (w * total // weight if weight else 0)
                           for key, w, held in zip(self._keys, self._weights, kept, strict=True)]

    def close(self) -> None:
        for read in self._reads:
            read.close()


@dataclasses.dataclass(frozen=True)
class Proportions:
    """A weighted mix's selection policy, Grain's (`MapDataset.mix`): the weights scaled so the least
    is 100 and truncated to whole proportions, and position k of the mix read from the corpus whose
    count of records a prefix holds grows from k to k + 1. A phase counts its prefix by it
    (`counts`) as the mix reads by it (`ProportionalSelection`), so the two never disagree."""

    proportions: tuple[int, ...]

    @classmethod
    def of(cls, weights: Sequence[float]) -> Proportions:
        scale = 100 / min(weights)
        return cls(tuple(int(weight * scale) for weight in weights))

    def counts(self, records: int) -> tuple[int, ...]:
        """How many records of each corpus the first `records` positions of the mix read."""
        counts, remaining, left = [], sum(self.proportions), records
        for proportion in self.proportions:
            rest = left * (remaining - proportion) // remaining
            counts.append(left - rest)
            left, remaining = rest, remaining - proportion
        return tuple(counts)

    def covering(self, sizes: Sequence[int]) -> int:
        """The fewest positions of the mix in which every corpus reads `sizes` of its records, its
        size: the pass in which each repeated corpus has come round once."""
        def covers(positions: int) -> bool:
            return all(count >= size for count, size in zip(self.counts(positions), sizes, strict=True))

        high = 1
        while not covers(high):
            high *= 2
        low = high // 2
        while low < high:
            middle = (low + high) // 2
            low, high = (middle + 1, high) if not covers(middle) else (low, middle)
        return low

    def at(self, position: int) -> tuple[int, int]:
        """The corpus position `position` of the mix reads, the one whose count grows with it, and
        which of that corpus's records: the next it hasn't read."""
        before, after = self.counts(position), self.counts(position + 1)
        grown = (corpus for corpus, (was, now) in enumerate(zip(before, after, strict=True)) if now != was)
        corpus = next(grown)
        return corpus, before[corpus]

    def run(self, start: int, positions: int) -> tuple[np.ndarray, np.ndarray] | None:
        """`at` of `positions` positions in a row from `start`, as two arrays, corpus and record:
        `counts`' split taken for every position at once, a corpus at a time, so a run costs what one
        position does in Python and a corpus's arithmetic in its vector. None where a position's
        products would leave 64-bit integers, which `at` reads exactly."""
        remaining = sum(self.proportions)
        if (start + positions + 1) * remaining >= 2**62:
            return None
        left = np.arange(start, start + positions + 1, dtype=np.int64)
        corpus = np.full(positions, -1, np.int64)
        record = np.zeros(positions, np.int64)
        for index, proportion in enumerate(self.proportions):
            rest = left * (remaining - proportion) // remaining
            # The position's corpus is the first whose count grows from it to the next position.
            grows = (corpus < 0) & (np.diff(left - rest) > 0)
            corpus[grows], record[grows] = index, (left - rest)[:-1][grows]
            if (corpus >= 0).all():
                break
            left, remaining = rest, remaining - proportion
        return corpus, record


class ProportionalSelection(pygrain.DatasetSelectionMap):
    """Grain's weighted mix of corpora of `lengths` records as a selection map
    (`MapDataset.select_from_datasets`), by `Proportions`: as long as the mix reads no record of
    any corpus twice."""

    RUN = 4096
    """The positions read ahead together once a reader reads one after another (`Proportions.run`)."""

    def __init__(self, lengths: Sequence[int], weights: Sequence[float]):
        self._proportions = Proportions.of(weights)
        total = sum(self._proportions.proportions)
        spans = [length / (proportion / total)
                 for length, proportion in zip(lengths, self._proportions.proportions, strict=True)]
        if any(span == 0 for span in spans):
            raise ValueError("every corpus of a mix needs a record")
        self._length = min(sys.maxsize, int(min(spans)))
        self._run: tuple[int, np.ndarray, np.ndarray] | None = None
        self._last = -2
        self._lock = threading.Lock()

    def __getstate__(self) -> dict:
        """What a worker process the selection is sent to reads: no run and no lock, its own."""
        return {"_proportions": self._proportions, "_length": self._length}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state, _run=None, _last=-2, _lock=threading.Lock())

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[int, int]:
        """`Proportions.at` of `index`: read from a run of positions where a reader reads them one
        after another, as a loader does, at a corpus's Python cost a run rather than a position; a
        position reached otherwise (a resume, a seek, a shuffled read) is read alone."""
        with self._lock:
            held = self._run
            if held is not None and held[0] <= index < held[0] + len(held[1]):
                start, corpora, records = held
            elif index == self._last + 1 and (run := self._proportions.run(index, self.RUN)) is not None:
                start, (corpora, records) = index, run
                self._run = start, corpora, records
            else:
                self._last = index
                return self._proportions.at(index)
            self._last = index
            return int(corpora[index - start]), int(records[index - start])

    def __repr__(self) -> str:
        return f"ProportionalSelection(proportions={self._proportions.proportions}, length={self._length})"


class TokenDeficit(pygrain.IterDataset[Batch]):
    """Pick the corpus furthest below its token share, with index-order ties."""

    def __init__(
        self,
        parents: Sequence[pygrain.IterDataset[Batch]],
        weights: Sequence[float],
        keys: Sequence[str],
        *,
        tokens: Callable[[Batch], int],
        names: Sequence[str],
        labels: Sequence[str],
    ):
        filtered = tuple(
            Accepted(parent, name, label, tokens=tokens)
            for parent, name, label in zip(parents, names, labels, strict=True)
        )
        super().__init__(filtered)
        ratios = tuple(Fraction(weight) for weight in weights)
        scale = math.lcm(*(ratio.denominator for ratio in ratios))
        self._corpora, self._weights, self._tokens, self._keys = (
            filtered,
            tuple(ratio.numerator * (scale // ratio.denominator) for ratio in ratios),
            tokens,
            tuple(keys),
        )

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        group = _PrimeReads([iter(parent) for parent in self._corpora])
        reads = [_PrimedRead(group, index) for index in range(len(self._corpora))]
        return _DeficitIterator(reads, self._weights, self._tokens, self._keys)


class _PrimeMap:
    def __init__(self, parents: Sequence[pygrain.MapDataset[Batch]]):
        self.parents = tuple(parents)
        self.lock = threading.Lock()
        self.done = False
        self.pending: dict[int, Batch] = {}

    def read(self, parent: int, index: int) -> Batch | None:
        with self.lock:
            if not self.done:
                for corpus, dataset in enumerate(self.parents):
                    row = dataset[0]
                    if row is not None:
                        self.pending[corpus] = row
                self.done = True
            if index == 0 and parent in self.pending:
                return self.pending.pop(parent)
        return self.parents[parent][index]

    def __getstate__(self):
        return {name: value for name, value in self.__dict__.items() if name != "lock"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()


class PrimedMap(pygrain.MapDataset[Batch]):
    def __init__(self, parent: pygrain.MapDataset[Batch], group: _PrimeMap, index: int):
        super().__init__(parent)
        self._records, self._group, self._index = parent, group, index

    def __len__(self) -> int:
        return len(self._records)

    @overload
    def __getitem__(self, index: slice) -> pygrain.MapDataset[Batch]: ...
    @overload
    def __getitem__(self, index: int) -> Batch | None: ...
    def __getitem__(self, index: int | slice) -> Batch | pygrain.MapDataset[Batch] | None:
        return self.slice(index) if isinstance(index, slice) else self._group.read(self._index, index)


def primed_maps(parents: Sequence[pygrain.MapDataset[Batch]]) -> tuple[pygrain.MapDataset[Batch], ...]:
    group = _PrimeMap(parents)
    return tuple(PrimedMap(parent, group, index) for index, parent in enumerate(parents))


class _PrimeReads:
    """Every corpus of a mix, each read through its first row ahead. A mix
    captures every corpus's position for each record it gives (a packer
    captures it before each record), and only the corpora the group read
    moved, so a corpus's position is captured again only after the group
    reads it or it is restored (`state`)."""

    def __init__(self, parents: Sequence[pygrain.DatasetIterator[Batch]]):
        self.parents = tuple(parents)
        self.pending: dict[int, Batch] = {}
        # Each corpus's captured position, before the row it holds read ahead while it holds one.
        self.states: dict[int, Mapping[str, object]] = {}
        # The corpora whose first row was read ahead: every corpus at the start, and one that joins
        # a resumed mix, before the mix gives a row.
        self.primed: set[int] = set()

    def read(self, index: int) -> Batch:
        for corpus, parent in enumerate(self.parents):
            if corpus not in self.primed:
                # Kept only once the row is read: a corpus refused as it is read ahead keeps the
                # position, and the counts, its refusal left.
                self.states.pop(corpus, None)
                before = parent.get_state()
                self.pending[corpus] = next(parent)
                self.states[corpus] = before
                self.primed.add(corpus)
        self.states.pop(index, None)
        if index in self.pending:
            return self.pending.pop(index)
        return next(self.parents[index])

    def state(self, index: int) -> Mapping[str, object]:
        """The position of corpus `index`: before its row read ahead while it holds one."""
        if index not in self.states:
            self.states[index] = self.parents[index].get_state()
        return self.states[index]

    def restore(self, index: int, state: Mapping[str, object], primed: bool) -> None:
        self.parents[index].set_state(state)
        self.pending.pop(index, None)
        self.states.pop(index, None)
        if primed:
            self.primed.add(index)
        else:
            self.primed.discard(index)


class _PrimedRead(pygrain.DatasetIterator[Batch]):
    """One corpus of a mix (`_PrimeReads`). The position records whether the
    mix had read the corpus's first row ahead, so a restored mix reads each
    corpus on from it and starts none again, but for one that joined, which it
    reads ahead before it gives a row."""

    def __init__(self, group: _PrimeReads, index: int):
        super().__init__(group.parents[index])
        self._group, self._index = group, index

    def __next__(self) -> Batch:
        return self._group.read(self._index)

    def get_state(self) -> Mapping[str, object]:
        return {"parent": self._group.state(self._index), "primed": self._index in self._group.primed}

    def set_state(self, state: Mapping[str, object]) -> None:
        self._group.restore(self._index, record(state["parent"], "primed corpus"),
                            boolean(state["primed"], "primed"))

    def close(self) -> None:
        self._group.parents[self._index].close()


class _ReadDataset(pygrain.IterDataset[Batch]):
    def __init__(self, parent: pygrain.IterDataset[Batch], group: _PrimeReads, index: int):
        super().__init__(parent)
        self._group, self._index = group, index

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        return _PrimedRead(self._group, self._index)


class PrimedMix(pygrain.IterDataset[Batch]):
    """Grain's weighted mix of corpora, each read through its first row ahead (`_PrimeReads`), and
    each keyed by its member's key (`keys`): Grain saves a keyed mix's positions by key and
    restores those it finds, so a member that joins starts at its first row and one that left is
    dropped while the others resume where they were."""

    def __init__(self, parents: Sequence[pygrain.IterDataset[Batch]], weights: Sequence[float],
                 keys: Sequence[str]):
        super().__init__(parents)
        self._corpora, self._weights, self._keys = tuple(parents), tuple(weights), tuple(keys)

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        group = _PrimeReads([iter(parent) for parent in self._corpora])
        parents = {key: _ReadDataset(parent, group, index)
                   for index, (key, parent) in enumerate(zip(self._keys, self._corpora, strict=True))}
        weights = dict(zip(self._keys, self._weights, strict=True))
        return iter(pygrain.IterDataset.mix(parents, weights))


@runtime_checkable
class _NamedTuple(Protocol):
    """A named tuple, which its documented `_make` rebuilds from its fields' values."""

    _fields: tuple[str, ...]

    def _make(self, iterable: Sequence[object]) -> Sequence[object]: ...


def rewound(start: Mapping[str, object], current: Mapping[str, object]) -> Mapping[str, object]:
    """`start`, the position a repeat rewinds its pass to, with each counter's counts as `current`
    holds them (`_COUNTER`: its counts, run of rejections and warning): a pass read again goes on
    counting the corpus, where a restored position replaces its counts. Every other part of a
    position is the iterator's own and is left as `start` has it."""
    shared = [key for key in start if key in current]
    parts = _rewound_parts([start[key] for key in shared], [current[key] for key in shared])
    kept = {**start, **dict(zip(shared, parts, strict=True))}
    if _COUNTER in start and _COUNTER in current:
        kept[_COUNTER] = current[_COUNTER]
    return kept


def _rewound_parts(start: Sequence[object], current: Sequence[object]) -> Sequence[object]:
    """Each part of a position rewound against its place in `current` (`rewound`): the mappings and
    the lists and tuples of one length are walked, rebuilt as what they were, and anything else
    is the iterator's own and kept."""
    parts: list[object] = []
    for value, now in zip(start, current, strict=True):
        if isinstance(value, Mapping) and isinstance(now, Mapping):
            parts.append(rewound(value, now))
        elif isinstance(value, (list, tuple)) and isinstance(now, (list, tuple)) and len(value) == len(now):
            inner = _rewound_parts(value, now)
            parts.append(list(inner) if isinstance(value, list)
                         else value._make(inner) if isinstance(value, _NamedTuple) else tuple(inner))
        else:
            parts.append(value)
    return parts


def state_counts(state: Mapping[str, object]) -> dict[str, CorpusCounts]:
    """Read per-corpus counters through Grain's nested iterator checkpoints.

    Within one iterator a corpus's counters are its stages': the outermost
    counts the rows it delivered and those of them cut short, and each its
    own rejections. The iterators Grain interleaves, its worker processes or a
    corpus's shards, each read records of their own, so their counts add,
    a shard waiting past the cycle (a future state) included."""

    def visit(branch) -> dict[str, CorpusCounts]:
        counted: dict[str, CorpusCounts] = {}
        if isinstance(branch, Mapping):
            node = branch.get(_COUNTER)
            if isinstance(node, Mapping):
                corpus = text(node["corpus"], "corpus")
                counted[corpus] = CorpusCounts(corpus, text(node["label"], "label"))
                counted[corpus].restore(record(node["counts"], "corpus counts"))
            interleaved = branch.get("iterators_in_use_states")
            if isinstance(interleaved, list):
                # A restored interleave keeps the positions of shards past its cycle as future states.
                future = branch.get("future_states") or {}
                independent: dict[str, CorpusCounts] = {}
                for child in (*interleaved, *(future.values() if isinstance(future, Mapping) else ())):
                    for counts in visit(child).values():
                        _added(independent, counts)
                _nested(counted, independent)
            for name, child in branch.items():
                if name not in ("iterators_in_use_states", "future_states", _COUNTER):
                    _nested(counted, visit(child))
        elif isinstance(branch, (tuple, list)):
            for child in branch:
                _nested(counted, visit(child))
        return counted

    return visit(state)


def data_metrics(counts: Iterable[CorpusCounts]) -> dict[str, float]:
    """The metrics a run logs of its corpora (`CorpusCounts.metrics`), each under its label; corpora
    that share one are numbered in the order of their identities, past every label a corpus has."""
    corpora = sorted(counts, key=lambda counted: counted.name)
    shared = Counter(counted.label for counted in corpora)
    taken = set(shared)
    metrics: dict[str, float] = {}
    for counted in corpora:
        label = counted.label
        if shared[label] > 1:
            label = next(numbered for place in itertools.count(1)
                         if (numbered := f"{label}-{place}") not in taken)
            taken.add(label)
        metrics.update(counted.metrics(label))
    return metrics


def _nested(counted: dict[str, CorpusCounts], inner: Mapping[str, CorpusCounts]) -> None:
    """Fold the counters of a corpus's inner stages into its outer ones', in one iterator: the
    outermost delivered the rows and counted those cut short."""
    for corpus, counts in inner.items():
        if corpus not in counted:
            counted[corpus] = counts
            continue
        combined = counted[corpus]
        combined.attempted = max(combined.attempted, counts.attempted)
        combined.chunks = max(combined.chunks, counts.chunks)
        combined.chunk_records = max(combined.chunk_records, counts.chunk_records)
        combined.rejected.update(counts.rejected)


def _added(counted: dict[str, CorpusCounts], counts: CorpusCounts) -> None:
    """Add a corpus's counts from an iterator that read records of its own."""
    if counts.name not in counted:
        counted[counts.name] = CorpusCounts(counts.name, counts.label)
    combined = counted[counts.name]
    combined.attempted += counts.attempted
    combined.accepted += counts.accepted
    combined.truncated += counts.truncated
    combined.chunks += counts.chunks
    combined.chunk_records += counts.chunk_records
    combined.rejected.update(counts.rejected)
