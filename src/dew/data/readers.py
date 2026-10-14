"""Checkpointed Grain readers, host stages and canonical finite shares."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Literal

import grain.python as pygrain
import numpy as np

from dew.data.dataset import (
    Batch,
    Closeable,
    DataPartition,
    FilledBatch,
    Forwarding,
    GlobalReader,
    Reader,
    _CheckpointableMapping,
    rows_of,
    take_batch,
)
from dew.data.mixing import CorpusCounts, state_counts
from dew.data.pipeline import HostStep, Map
from dew.data.plan import Plan
from dew.records import record


@dataclasses.dataclass(frozen=True)
class _HostSteps:
    steps: tuple[HostStep, ...]

    def __call__(self, value: Batch) -> Batch:
        from dew.data.lowering import _outputs

        for step in self.steps:
            value = record(_outputs(Map(step), None, value)[0], "host map result")
        return value


class _Host(_CheckpointableMapping):
    def __init__(self, source: Iterator[Batch], steps: tuple[HostStep, ...]):
        super().__init__(source, _HostSteps(steps))

    @property
    def records(self) -> int:
        if not isinstance(self._source, GlobalReader):
            raise TypeError("host-stage parent has no global record count")
        return self._source.records

    @property
    def order(self) -> str:
        if not isinstance(self._source, GlobalReader):
            raise TypeError("host-stage parent has no global order")
        return self._source.order


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

        from dew.coordination import agreed, broadcast_from_process_zero
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
class _PublishedCounts:
    """Reader-owned snapshots retained without retaining its Grain iterator."""

    metrics: Mapping[str, float] = dataclasses.field(default_factory=lambda: MappingProxyType({}))


class _PerShare(Forwarding):
    """Lazy Grain reader with owner-published positions and counters."""

    @property
    def endless(self) -> bool:
        return self._endless

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
    ):
        self._open, self._plan = open_reader, plan
        self._resume: Literal["exact", "non-exact"] = resume
        self._stop_seconds = stop_seconds
        self._endless = endless
        self._read: pygrain.DatasetIterator[Batch] | None = None
        self._saved: Mapping[str, object] | None = None
        self._state: Mapping[str, object] | None = None
        self._statistics = _PublishedCounts()

    def _forwarded(self) -> pygrain.DatasetIterator[Batch] | None:
        return self._read

    def _reader(self) -> pygrain.DatasetIterator[Batch]:
        """Open and restore the reader on its iteration thread."""
        if self._read is None:
            self._read = self._open()
            if self._saved is not None:
                self._read.set_state(self._saved)
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
        self._statistics.metrics = MappingProxyType({
            key: value
            for counts in state_counts(state).values()
            for key, value in counts.metrics().items()
        })

    def get_state(self) -> bytes:
        state = self._state
        if state is None:
            state = self._reader().get_state()
            self._publish(state)
        return json.dumps({"dew_data_plan": self._plan.record(), "iterator": state}).encode()

    def set_state(self, state: bytes) -> None:
        decoded = record(json.loads(state), "saved per-share position")
        self._plan.check(record(decoded.get("dew_data_plan"), "saved data plan"), resume=self._resume)
        saved = record(decoded.get("iterator"), "saved Grain iterator")
        self.close()
        self._saved = saved
        self._statistics.metrics = MappingProxyType({})

    def identity(self) -> str:
        """What every share of the pool reads, which the pool agrees before it
        reads (`Plan.identity`): the plan without this share's index, which
        only the position it saves keeps, so a share resumes only its own rows."""
        return self._plan.identity()

    def statistics(self) -> Mapping[str, float]:
        """The immutable counters last published by the reading thread."""
        return self._statistics.metrics

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
        scalars = {key: value for counts in self._counts.values() for key, value in counts.metrics().items()}
        for published in self._published:
            for name, value in published.metrics.items():
                scalars[name] = max(scalars.get(name, 0.0), value)
        return scalars


@dataclasses.dataclass(frozen=True)
class _FillShare:
    partition: DataPartition
    batch: int

    def __call__(self, batch: Batch) -> Batch:
        mine = take_batch(batch, slice(self.partition.index, None, self.partition.count))
        empty = rows_of(mine) == 0
        if empty:
            mine = take_batch(batch, slice(1))
        return FilledBatch(self.partition.rows(self.batch), real=not empty).map(mine)
