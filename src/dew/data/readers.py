"""Checkpointed Grain readers, host stages and canonical finite shares."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Literal

import grain.python as pygrain
import numpy as np

from dew import _position
from dew._records import json_value, record
from dew.batch import EXAMPLES_CONSUMED, VALID_ROWS, Batch
from dew.data.dataset import (
    Closeable,
    DataPartition,
    Ends,
    FilledBatch,
    Forwarding,
    GlobalReader,
    Position,
    Reader,
    _CheckpointableMapping,
    rows_of,
    take_batch,
)
from dew.data.mixing import CorpusCounts, data_metrics, state_counts
from dew.data.pipeline import HostStep, Map
from dew.data.plan import COUNTERS, Plan, earlier_position
from dew.data.storage import mirror_metrics

READER_FIELDS = (EXAMPLES_CONSUMED, VALID_ROWS)
"""A batch's fields its reader owns, which a host step after the batch neither reads nor writes
(`_HostSteps`): the examples it consumes and which of its rows are real."""


@dataclasses.dataclass(frozen=True)
class _HostSteps:
    steps: tuple[HostStep, ...]

    def __call__(self, value: Batch) -> Batch:
        from dew.data.lowering import _outputs

        # The examples the batch consumes and which of its rows are real are the reader's accounting,
        # which no step of the batch's own sees or changes: a projection or a fresh model-input dict
        # keeps them (row 3937: a projection dropped an evaluation's real-row mask).
        kept = {name: value[name] for name in READER_FIELDS if name in value}
        held = {name: column for name, column in value.items() if name not in kept}
        for step in self.steps:
            held = record(_outputs(Map(step), None, held)[0], "host map result")
        return {**held, **kept}


class _Host(_CheckpointableMapping):
    def __init__(self, source: Iterator[Batch], steps: tuple[HostStep, ...]):
        super().__init__(source, _HostSteps(steps))
        # The global reader this reads, kept through close: it states its order
        # and position, and reads on from a position set again, as a phased run
        # does of a finished phase. Closed, this reads nothing until then.
        self._global = source if isinstance(source, GlobalReader) else None

    def get_state(self) -> Position:
        return super().get_state() if self._global is None else self._global.get_state()

    def set_state(self, state: Position) -> None:
        source = self._source
        self._source = self._global or source
        try:
            super().set_state(state)
        except BaseException:
            self._source = source
            raise

    @property
    def records(self) -> int:
        if self._global is None:
            raise TypeError("host-stage parent has no global record count")
        return self._global.records

    @property
    def order(self) -> str:
        if self._global is None:
            raise TypeError("host-stage parent has no global order")
        return self._global.order


class _CanonicalPass:
    def __init__(self, reader: Reader, partition: DataPartition, batch: int):
        import jax

        self._partition = partition
        self._batch = batch
        self._source = reader(DataPartition()) if jax.process_index() == 0 else None

    def __iter__(self) -> Iterator[Batch]:
        return self

    def __next__(self) -> Batch:
        import jax
        from jax.experimental import multihost_utils

        from dew._coordination import agreed, broadcast_from_process_zero
        from dew.data.sources.codec import decode, encode

        root = jax.process_index() == 0
        held = agreed(
            "canonical validation read",
            lambda: next(self._source, None) if self._source is not None else None,
        )
        payload = b"" if held is None else encode(held)
        size = broadcast_from_process_zero(len(payload))
        if not size:
            raise StopIteration
        if jax.process_count() > 1:
            buffer = np.frombuffer(payload, np.uint8).copy() if root else np.zeros(size, np.uint8)
            payload = np.asarray(multihost_utils.broadcast_one_to_all(buffer, is_source=root)).tobytes()
        whole = record(decode(payload), "canonical validation batch")
        return _FillShare(self._partition, self._batch)(whole)

    def close(self) -> None:
        if isinstance(self._source, Closeable):
            self._source.close()
        self._source = None


@dataclasses.dataclass
class _Published:
    """A position a reader published, and its counts (`state_counts`) once they are asked."""

    state: Mapping[str, object] | None
    read: Mapping[str, CorpusCounts] | None = None


@dataclasses.dataclass
class _PublishedCounts:
    """Reader-owned snapshots retained without retaining its Grain iterator: the position last published,
    whose counts are read from it as they are asked, a run logging them every few steps, not read from
    each batch's position (row 4079). A publication replaces the whole `_Published`, so counts read off
    one, on another thread than the reader's, are that one's and kept on it alone."""

    published: _Published = dataclasses.field(default_factory=lambda: _Published(None))

    @property
    def counts(self) -> Mapping[str, CorpusCounts]:
        published = self.published
        if published.read is None:
            state = published.state
            published.read = MappingProxyType({} if state is None else state_counts(state))
        return published.read

    def publish(self, state: Mapping[str, object] | None) -> None:
        self.published = _Published(state)


class _PerShare(Forwarding):
    """Lazy Grain reader with owner-published positions and counters."""

    @property
    def ends(self) -> Ends:
        """Never where its order repeats without end, else on any share apart: its shares cut the
        order each its own."""
        return "never" if self._endless else "apart"

    @property
    def stop_seconds(self) -> float:
        """Its Loading's stop budget: the Grain iterator it wraps reports none, so the reader that
        owns the iterator states it, as the global reader does."""
        return self._stop_seconds

    def __init__(
        self,
        open_reader: Callable[[], pygrain.DatasetIterator[Batch]],
        plan: Plan,
        resume: Literal["exact", "non-exact"],
        *,
        stop_seconds: float,
        endless: bool = True,
        replicated: bool = False,
    ):
        self._open, self._plan = open_reader, plan
        self._resume: Literal["exact", "non-exact"] = resume
        self._stop_seconds = stop_seconds
        self._endless = endless
        # An elastic read's position (`position.REPLICATED`), the same on every reader.
        self._replicated = replicated
        self._read: pygrain.DatasetIterator[Batch] | None = None
        self._saved: Mapping[str, object] | None = None
        self._state: Mapping[str, object] | None = None
        self._statistics = _PublishedCounts()

    def _forwarded(self) -> pygrain.DatasetIterator[Batch] | None:
        return self._read

    def _reader(self) -> pygrain.DatasetIterator[Batch]:
        """Open and restore the reader on its iteration thread, every loading worker's process
        started at once (`start_prefetch`). Grain's interleave of its workers makes their iterators
        unstarted where it is asked its position before its first read, as a flat map over it asks
        at its start (a pooled read's, `_Build.handed`), and where it is restored, and each then
        starts at its first turn, after the one before it has read: a spawn and an import apart,
        3.5 s each on a 4-CPU host, so 30 workers read their first two minutes at 73 rows/s on an
        A100 host (ledger row 1846)."""
        if self._read is None:
            self._read = self._open()
            if self._saved is not None:
                self._read.set_state(self._saved)
            self._read.start_prefetch()
        return self._read

    def __iter__(self) -> Iterator[Batch]:
        return self

    def __next__(self) -> Batch:
        reader = self._reader()
        try:
            value = next(reader)
        except StopIteration:
            self._publish(reader.get_state())
            raise
        self._publish(reader.get_state())
        return value

    def _publish(self, state: Mapping[str, object]) -> None:
        self._state = state
        self._statistics.publish(state)

    def get_state(self) -> bytes:
        state = self._state
        if state is None:
            state = self._reader().get_state()
            self._publish(state)
        saved = {"dew_data_plan": self._plan.record(), "counters": COUNTERS, "iterator": state}
        # A mix's captured positions are shared mappings (`SharedMapping`), written as the records they
        # read as.
        return json.dumps(json_value({_position.REPLICATED: saved} if self._replicated else saved,
                                     "data position")).encode()

    def set_state(self, state: bytes) -> None:
        held = _position.replicated(state)
        if self._replicated != (held is not None):
            raise ValueError(
                "this read is elastic, and the saved position is one share's own offset; resume it with the "
                "chain that saved it (without .elastic())" if self._replicated else
                "the saved position is an elastic read's, and this read is per share; resume it with "
                ".elastic() on the chain")
        decoded = record(json.loads(state), "saved per-share position") if held is None else held
        # The saved position's one translation (`earlier_position`), then its check.
        written, iterator = earlier_position(decoded, self._plan)
        self._plan.check(written, resume=self._resume)
        saved = record(iterator, "saved Grain iterator")
        self.close()
        self._saved = saved
        self._statistics.publish(None)

    def identity(self) -> str:
        """What every share of the pool reads, which the pool agrees before it
        reads (`Plan.identity`): the plan without this share's index, which
        only the position it saves keeps, so a share resumes only its own rows."""
        return self._plan.identity()

    def statistics(self) -> Mapping[str, float]:
        """The metrics of the counters last published by the reading thread, and what this process's
        mirrors did (`mirror_metrics`)."""
        return {**data_metrics(self._statistics.counts.values()), **mirror_metrics()}

    def close(self) -> None:
        if self._read is not None:
            self._read.close()
            self._read = None
        self._state = None


class _Statistics:
    def __init__(self, counts: Mapping[str, CorpusCounts]):
        self._counts = counts
        self._published: list[_PublishedCounts] = []

    def observe(self, stream: _PerShare) -> None:
        self._published.append(stream._statistics)

    def __call__(self) -> Mapping[str, float]:
        """The metrics of every corpus: the counts of the prefix a memory read ran, followed by
        those the latest reader published of the rest (`CorpusCounts.followed`), which a reader
        restored from another's position carries on from."""
        read = {name: counts for published in self._published for name, counts in published.counts.items()}
        merged = {name: counts.followed(read.pop(name)) if name in read else counts
                  for name, counts in self._counts.items()}
        return data_metrics([*merged.values(), *read.values()])


@dataclasses.dataclass(frozen=True)
class _FillShare:
    partition: DataPartition
    batch: int

    def __call__(self, batch: Batch) -> Batch:
        held, real = self.partition.held(rows_of(batch))
        return FilledBatch(self.partition.rows(self.batch), real=real).map(take_batch(batch, held))
