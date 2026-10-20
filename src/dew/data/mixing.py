"""Accepted-row accounting and mixtures measured in valid target tokens."""

from __future__ import annotations

import dataclasses
import itertools
import logging
import math
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from typing import TypeGuard, overload

import grain.experimental as gx
import grain.python as pygrain

from dew.data.dataset import Batch
from dew.data.pipeline import Record, Rejected
from dew.records import boolean, integer, record

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


@dataclasses.dataclass(frozen=True)
class SearchBudget:
    consecutive: int = 10_000
    warn_over: float = 0.01

    def __post_init__(self) -> None:
        if self.consecutive < 1:
            raise ValueError("rejection search budgets must be positive")
        if not 0 <= self.warn_over <= 1:
            raise ValueError("warn_over is a share between zero and one")


_DEFAULT_BUDGET = SearchBudget()


class _KeepAccepted:
    def __init__(self, counts: CorpusCounts, budget: SearchBudget, tokens: Callable[[Batch], int] | None):
        self.counts, self.budget, self.tokens = counts, budget, tokens
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
            if not isinstance(element, Rejected):
                self.counts.truncated += int(bool(element.get("_dew_truncated", False)))
            self._probe()
            self.consecutive = 0
            return True
        self.counts.rejected[reason] += 1
        self.consecutive += 1
        self._probe()
        if (
            not self.warned
            and sum(self.counts.rejected.values()) / self.counts.attempted > self.budget.warn_over
        ):
            _log.warning(
                "corpus %s rejected more than %.1f%% of attempted records (%s)",
                self.counts.name,
                self.budget.warn_over * 100,
                reason,
            )
            self.warned = True
        if self.consecutive >= self.budget.consecutive:
            raise ValueError(
                f"corpus {self.counts.name}: rejection search budget exhausted after "
                f"{self.consecutive} consecutive rejections ({reason}); "
                "this is not a proof that the corpus is empty"
            )
        return False

    def _probe(self) -> None:
        if self.counts.attempted == 16:
            _log.info(
                "corpus %s: estimated rejection share over first 16 attempted records %.4f",
                self.counts.name,
                sum(self.counts.rejected.values()) / 16,
            )


class _CountedIterator[Element](pygrain.DatasetIterator[Element]):
    """Checkpoint a corpus counter with its parent, including replay during parent restoration."""

    def __init__(self, parent: pygrain.DatasetIterator, counts: CorpusCounts):
        super().__init__(parent)
        self._read, self._counts = parent, counts

    def __next__(self) -> Element:
        return next(self._read)

    def get_state(self) -> Mapping[str, object]:
        return {"parent": self._read.get_state(), "corpus": self._counts.name, "label": self._counts.label,
                "counts": self._counts.state()}

    def set_state(self, state: Mapping[str, object]) -> None:
        self._read.set_state(record(state["parent"], "counted parent"))
        self._counts.restore(record(state["counts"], "corpus counts"))

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

    def get_state(self) -> Mapping[str, object]:
        return {
            **super().get_state(),
            "consecutive": self._predicate.consecutive,
            "warned": self._predicate.warned,
        }

    def set_state(self, state: Mapping[str, object]) -> None:
        super().set_state(state)
        self._predicate.consecutive = integer(state["consecutive"], "consecutive")
        self._predicate.warned = bool(state["warned"])


class Accepted(pygrain.IterDataset[Batch]):
    """Grain's filter, with a named search budget and checkpointed counts."""

    def __init__(
        self,
        parent: pygrain.IterDataset[Batch | Rejected] | pygrain.IterDataset[Batch],
        name: str,
        label: str,
        *,
        budget: SearchBudget = _DEFAULT_BUDGET,
        tokens: Callable[[Batch], int] | None = None,
    ):
        super().__init__(parent)
        self._records, self._name, self._label = parent, name, label
        self._budget, self._tokens = budget, tokens

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        counts = CorpusCounts(self._name, self._label)
        predicate = _KeepAccepted(counts, self._budget, self._tokens)
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
    def __init__(
        self,
        parents: Sequence[pygrain.DatasetIterator[Batch]],
        weights: tuple[int, ...],
        tokens: Callable[[Batch], int],
        keys: tuple[str, ...],
    ):
        super().__init__(parents)
        self._reads, self._weights, self._tokens, self._keys = tuple(parents), weights, tokens, keys
        self._delivered = [0] * len(parents)
        self._pending: list[Batch | None] = [None] * len(parents)
        self._pending_states: list[Mapping[str, object] | None] = [None] * len(parents)
        # The corpora whose first row was read ahead, so a rejection budget a corpus exhausts is
        # met before any row: every corpus at the start, and one that joins a resumed mix.
        self._primed = [False] * len(parents)

    def __next__(self) -> Batch:
        for index, parent in enumerate(self._reads):
            if not self._primed[index]:
                self._pending_states[index] = parent.get_state()
                self._pending[index] = next(parent)
                self._primed[index] = True
        total = sum(self._delivered)
        scale = sum(self._weights)
        chosen = max(
            range(len(self._reads)),
            key=lambda index: self._weights[index] * total - self._delivered[index] * scale,
        )
        value = self._pending[chosen]
        if value is None:
            value = next(self._reads[chosen])
        self._pending[chosen], self._pending_states[chosen] = None, None
        tokens = self._tokens(value)
        if tokens <= 0:
            raise RuntimeError(
                "token deficit mix received a window without valid targets; filter it before mixing"
            )
        self._delivered[chosen] += tokens
        return value

    def get_state(self) -> Mapping[str, object]:
        return {
            "parents": {
                key: pending if value is not None and pending is not None else parent.get_state()
                for key, parent, value, pending in zip(
                    self._keys, self._reads, self._pending, self._pending_states, strict=True
                )
            },
            "delivered": dict(zip(self._keys, self._delivered, strict=True)),
            "primed": dict(zip(self._keys, self._primed, strict=True)),
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
        for key, parent in zip(self._keys, self._reads, strict=True):
            if key in parents:
                parent.set_state(record(parents[key], "corpus iterator"))
        restored = [delivered[key] for key in self._keys if key in parents]
        total, weight = sum(restored), sum(w for w, held in zip(self._weights, kept, strict=True) if held)
        self._delivered = [delivered[key] if held else (w * total // weight if weight else 0)
                           for key, w, held in zip(self._keys, self._weights, kept, strict=True)]
        self._pending = [None for _ in self._reads]
        self._pending_states = [None for _ in self._reads]
        # A corpus whose first row was read ahead is restored to before it, and reads it when chosen;
        # one that joined is read ahead as the mix reads on.
        primed = record(state["primed"], "primed corpora")
        self._primed = [key in primed and boolean(primed[key], "primed") for key in self._keys]

    def close(self) -> None:
        for parent in self._reads:
            parent.close()


class TokenDeficit(pygrain.IterDataset[Batch]):
    """Pick the corpus furthest below its token share, with index-order ties."""

    def __init__(
        self,
        parents: Sequence[pygrain.IterDataset[Batch]],
        weights: Sequence[float],
        *,
        tokens: Callable[[Batch], int],
        names: Sequence[str],
        labels: Sequence[str],
        keys: Sequence[str],
        budget: SearchBudget = _DEFAULT_BUDGET,
    ):
        if not parents or len(parents) != len(weights) or len(names) != len(parents):
            raise ValueError("token deficit mix needs one weight and name per corpus")
        keys = tuple(keys)
        if len(keys) != len(parents) or len(set(keys)) != len(keys):
            raise ValueError("token deficit mix needs one distinct key per corpus")
        if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError("token deficit weights must be finite and positive")
        filtered = tuple(
            Accepted(parent, name, label, budget=budget, tokens=tokens)
            for parent, name, label in zip(parents, names, labels, strict=True)
        )
        super().__init__(filtered)
        ratios = tuple(Fraction(weight) for weight in weights)
        scale = math.lcm(*(ratio.denominator for ratio in ratios))
        self._corpora, self._weights, self._tokens, self._keys = (
            filtered,
            tuple(ratio.numerator * (scale // ratio.denominator) for ratio in ratios),
            tokens,
            keys,
        )

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        return _DeficitIterator([iter(parent) for parent in self._corpora], self._weights, self._tokens,
                                self._keys)


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
    def __init__(self, parents: Sequence[pygrain.DatasetIterator[Batch]]):
        self.parents = tuple(parents)
        self.pending: dict[int, Batch] = {}
        self.before: dict[int, Mapping[str, object]] = {}
        # The corpora whose first row was read ahead: every corpus at the start, and one that joins
        # a resumed mix, before the mix gives a row.
        self.primed: set[int] = set()

    def read(self, index: int) -> Batch:
        for corpus, parent in enumerate(self.parents):
            if corpus not in self.primed:
                self.before[corpus] = parent.get_state()
                self.pending[corpus] = next(parent)
                self.primed.add(corpus)
        if index in self.pending:
            self.before.pop(index, None)
            return self.pending.pop(index)
        return next(self.parents[index])


class _PrimedRead(pygrain.DatasetIterator[Batch]):
    """One corpus of a mix. The mix captures every corpus's position for each
    record it gives (a packer captures it before each record), and only the
    corpus that gave the record moved, so a corpus captures its position again
    only after it reads or is restored. The position records whether the mix
    had read the corpus's first row ahead, so a restored mix reads each
    corpus on from it and starts none again, but for one that joined, which it
    reads ahead before it gives a row."""

    def __init__(self, group: _PrimeReads, index: int):
        super().__init__(group.parents[index])
        self._group, self._index = group, index
        self._state: Mapping[str, object] | None = None

    def __next__(self) -> Batch:
        self._state = None
        return self._group.read(self._index)

    def get_state(self) -> Mapping[str, object]:
        if self._state is None:
            before = self._group.before
            self._state = (before[self._index] if self._index in before
                           else self._group.parents[self._index].get_state())
        return {"parent": self._state, "primed": self._index in self._group.primed}

    def set_state(self, state: Mapping[str, object]) -> None:
        self._state = None
        self._group.parents[self._index].set_state(record(state["parent"], "primed corpus"))
        self._group.pending.pop(self._index, None)
        self._group.before.pop(self._index, None)
        if boolean(state["primed"], "primed"):
            self._group.primed.add(self._index)
        else:
            self._group.primed.discard(self._index)

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
        if len(keys) != len(parents) or len(set(keys)) != len(keys):
            raise ValueError("a mix needs one distinct key per corpus")
        super().__init__(parents)
        self._corpora, self._weights, self._keys = tuple(parents), tuple(weights), tuple(keys)

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        group = _PrimeReads([iter(parent) for parent in self._corpora])
        parents = {key: _ReadDataset(parent, group, index)
                   for index, (key, parent) in enumerate(zip(self._keys, self._corpora, strict=True))}
        weights = dict(zip(self._keys, self._weights, strict=True))
        return iter(pygrain.IterDataset.mix(parents, weights))


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
            corpus, label, stored = branch.get("corpus"), branch.get("label"), branch.get("counts")
            if isinstance(corpus, str) and isinstance(label, str) and isinstance(stored, Mapping):
                counted[corpus] = CorpusCounts(corpus, label)
                counted[corpus].restore(stored)
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
                if name not in ("iterators_in_use_states", "future_states"):
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
