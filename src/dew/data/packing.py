"""First-fit windows by a plan made from document lengths alone.

`plan_windows` makes the plan, which chunks of which documents each window
holds and which windows hold a target. A published generation stores it
(`PackingPlan`), and a source that states its documents' lengths makes it
when it is first read. `PackedSource` reads windows by either plan, by
index in one global order, the same at any process count, without a pass
over the tokens.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from typing import ClassVar

import grain.python as grain
import numpy as np

from dew.data.cache import Generation
from dew.data.dataset import Batch, Records
from dew.data.pipeline import Pack
from dew.data.plan import fingerprint
from dew.data.sources.base import Read, Shard, ShardManifest, Source, _consumed
from dew.data.tokens import first_fit, packed_window
from dew.records import record


def plan_name(pack: Pack, field: str) -> str:
    return "packing_" + fingerprint(dataclasses.asdict(pack))[:20] + "_" + field


def plan_windows(lengths: np.ndarray, pack: Pack) -> Mapping[str, np.ndarray]:
    """The first-fit plan of documents of `lengths` into `pack.seq_len`-token
    windows: each chunk's document, offset and size, the chunks in packing
    order and where each window starts in it, each window's targets, and the
    windows that hold one (`accepted`), with their total and the count left
    out."""
    if pack.policy != "first-fit":
        raise ValueError("indexed packing currently needs policy='first-fit'")
    chunk_sizes = np.asarray(lengths, np.int64)
    documents = np.arange(len(chunk_sizes), dtype=np.int64)
    offsets = np.zeros(len(chunk_sizes), np.int64)
    order, starts = first_fit(chunk_sizes, pack.seq_len, pack.bins)
    targets = np.asarray(
        [
            sum(max(0, int(chunk_sizes[index]) - 1) for index in order[starts[window] : starts[window + 1]])
            for window in range(len(starts) - 1)
        ],
        np.int64,
    )
    kept = np.flatnonzero(targets > 0)
    return {
        "documents": documents,
        "offsets": offsets,
        "sizes": chunk_sizes,
        "order": order,
        "starts": starts,
        "targets": targets,
        "accepted": kept,
        "totals": np.asarray([targets[kept].sum(), np.count_nonzero(targets == 0)], np.int64),
    }


@dataclasses.dataclass(frozen=True)
class PackingPlan:
    pack: Pack

    def __call__(self, generation: Generation) -> Mapping[str, np.ndarray]:
        lengths = generation.source().lengths()
        if lengths is None:
            raise ValueError("published packing needs certified per-document lengths")
        planned = plan_windows(lengths, self.pack)
        return {plan_name(self.pack, field): value for field, value in planned.items()}


_FIELDS = ("documents", "offsets", "sizes", "order", "starts", "targets", "accepted")
"""The plan arrays that fix which chunks each window holds, which a saved
position's order fingerprints."""


@dataclasses.dataclass(frozen=True)
class _Published:
    """A plan as a generation stores it, read a page at a time."""

    generation: Generation
    pack: Pack
    kind: ClassVar[str] = "published"

    @property
    def location(self) -> str:
        return self.generation.url

    def at(self, field: str, index: int) -> int:
        return int(self.generation.plan_at(plan_name(self.pack, field), index).item())

    def __len__(self) -> int:
        return self.generation.plan_length(plan_name(self.pack, "accepted"))

    def fingerprints(self) -> Mapping[str, str]:
        return {field: self.generation.plan_hash(plan_name(self.pack, field)) for field in _FIELDS}


@dataclasses.dataclass(frozen=True)
class _Held:
    """A plan made from the lengths `source` states, when it is first read,
    and held in memory."""

    source: Source
    pack: Pack
    kind: ClassVar[str] = "planned"

    @functools.cached_property
    def arrays(self) -> Mapping[str, np.ndarray]:
        lengths = self.source.lengths()
        if lengths is None:
            raise ValueError(f"{self.source.describe()} states no document lengths to plan its packing by")
        return plan_windows(lengths, self.pack)

    @property
    def location(self) -> str:
        return self.source.describe()

    def at(self, field: str, index: int) -> int:
        return int(self.arrays[field][index])

    def __len__(self) -> int:
        return len(self.arrays["accepted"])

    def fingerprints(self) -> Mapping[str, str]:
        return {field: fingerprint(self.arrays[field]) for field in _FIELDS}


class _PackedRows:
    """The windows a plan makes of `documents`, by index, each with its targets."""

    def __init__(self, documents: Records, plan: _Published | _Held):
        self._documents, self._plan, self._pack = documents, plan, plan.pack

    def _at(self, field: str, index: int) -> int:
        return self._plan.at(field, index)

    def __len__(self) -> int:
        return len(self._plan)

    def __getitem__(self, index: int) -> Batch:
        if not 0 <= index < len(self):
            raise IndexError(index)
        window = self._at("accepted", index)
        start, stop = self._at("starts", window), self._at("starts", window + 1)
        chunks = []
        for position in range(start, stop):
            chunk = self._at("order", position)
            document = record(self._documents[self._at("documents", chunk)], "packed document")
            offset, size = self._at("offsets", chunk), self._at("sizes", chunk)
            chunks.append(
                {name: np.asarray(document[name])[offset : offset + size] for name in self._pack.fields}
            )
        return {
            **packed_window(chunks, self._pack.seq_len),
            "_dew_valid_targets": self._at("targets", window),
        }


@dataclasses.dataclass(frozen=True)
class PackedSource:
    """A source's documents packed first-fit into windows by a plan of their
    lengths (`plan_windows`): the plan a published generation stores
    (`published`), or the one made from the lengths a source states when it
    is first read (`planned`). Either way the windows are read by index in
    one global order. It reads windows, not its documents' records, so it
    names them `documents` rather than wrapping them as a source."""

    documents: Source
    plan: _Published | _Held

    @classmethod
    def published(cls, source: Source, generation: Generation, pack: Pack) -> PackedSource:
        return cls(source, _Published(generation, pack))

    @classmethod
    def planned(cls, source: Source, pack: Pack) -> PackedSource:
        return cls(source, _Held(source, pack))

    def describe(self) -> str:
        return f"{self.documents.describe()} {self.plan.kind} packing={self.plan.pack!r}"

    def consumed(self, read: Read) -> PackedSource:
        documents = _consumed(self.documents, "indexed" if read == "streamed" else read)
        plan = dataclasses.replace(self.plan, source=documents) if isinstance(self.plan, _Held) else self.plan
        return dataclasses.replace(self, documents=documents, plan=plan)

    def fingerprint(self) -> Mapping[str, object]:
        return {"source": self.documents.fingerprint(), "packing": self.order_fingerprints()}

    def order_fingerprints(self) -> Mapping[str, str]:
        return self.plan.fingerprints()

    def supports(self, read: Read) -> bool:
        return self.documents.supports("indexed" if read == "streamed" else read)

    def records(self, read: Read) -> _PackedRows:
        return _PackedRows(self.documents.records(read), self.plan)

    def metadata(self) -> Mapping[str, object]:
        return {
            **self.documents.metadata(),
            "windows": len(self.plan),
            "rows": len(self.plan),
            "valid_targets": self.plan.at("totals", 0),
            "zero_target_windows": 0,
        }

    def lengths(self) -> np.ndarray | None:
        return None

    def manifest(self) -> ShardManifest:
        size, location = len(self.plan), self.plan.location
        return ShardManifest(
            tuple(
                Shard(
                    f"{location}:{start}", location, start, min(start + 1024, size), min(1024, size - start)
                )
                for start in range(0, size, 1024)
            ),
            self._reader,
        )

    def _reader(self, shard: Shard) -> grain.IterDataset[Batch]:
        return grain.MapDataset.source(self.records("indexed"))[shard.start : shard.stop].to_iter_dataset()
