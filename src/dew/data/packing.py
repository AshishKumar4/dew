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

import grain.python as pygrain
import numpy as np

from dew._records import integer, record
from dew.batch import Batch
from dew.data.dataset import Records
from dew.data.generation import Generation
from dew.data.pipeline import ExpandStep, Pack
from dew.data.plan import fingerprint
from dew.data.sources.base import Read, Rows, ShardManifest, Source, View, _consumed, _indexed_manifest
from dew.data.tokens import FirstFit, chunk_lengths, chunk_table, first_fit, packed_window


def plan_name(pack: Pack, field: str) -> str:
    return "packing_" + fingerprint(dataclasses.asdict(pack))[:20] + "_" + field


def refuse_overlong(lengths, pack: Pack, described: str) -> None:
    """Refuse documents of `lengths` that a first-fit or best-fit `pack` would
    have to cut to fit its windows: a pack places documents whole, and a run
    that means to cut them says so with a chunk step before it
    (`dew.data.text.Chunks`). A concatenating pack splits by design."""
    longest = int(np.max(lengths, initial=0))
    if pack.policy != "concat" and longest > pack.seq_len:
        raise ValueError(
            f"{described} holds a document of {longest} tokens, longer than the {pack.seq_len}-token "
            f"windows it packs into; cut documents first with flat_map(dew.data.text.Chunks({pack.seq_len}))"
        )


def chunks_for(step: ExpandStep, pack: Pack) -> int | None:
    """How far apart the pieces overlap where `step`, just before `pack`, cuts
    each document as a plan of lengths cuts it (`chunk_table`): pieces of the
    window's length, the same cut in every field the pack places; None where
    it cuts otherwise or is no chunk step."""
    from dew.data.text import Chunks

    if isinstance(step, Chunks) and step.length == pack.seq_len and set(pack.fields) <= set(step.fields):
        return step.overlap
    return None


def most_documents(starts: np.ndarray, accepted: np.ndarray) -> int:
    """The most documents, or a cut document's pieces, one of a plan's `accepted` windows holds,
    from where each window starts in its packing order (`plan_windows`)."""
    return int(np.diff(np.asarray(starts))[np.asarray(accepted)].max(initial=0))


_FIELDS = ("documents", "offsets", "sizes", "order", "starts", "targets", "accepted")
"""The plan arrays that fix which chunks each window holds, which a saved
position's order fingerprints."""


def plan_windows(lengths: np.ndarray, pack: Pack, overlap: int = 0) -> Mapping[str, np.ndarray]:
    """The first-fit plan of documents of `lengths` into `pack.seq_len`-token
    windows, a document longer than a window cut into window-long pieces
    `overlap` tokens apart: each chunk's document, offset and size, the chunks
    in packing order and where each window starts in it, each window's
    targets, and the windows that hold one (`accepted`), with their total and
    the count left out."""
    if pack.policy != "first-fit":
        raise ValueError("indexed packing currently needs policy='first-fit'")
    if pack.along:
        raise ValueError("a plan of document lengths packs tokens alone; fields along other axes "
                         "pack as their rows arrive")
    chunk_sizes = chunk_lengths(lengths, pack.seq_len, overlap)
    documents, offsets = chunk_table(lengths, pack.seq_len, overlap)
    order, starts = first_fit(chunk_sizes, pack.seq_len, pack.bins, pack.segments)
    targets = np.asarray(
        [
            sum(max(0, int(chunk_sizes[index]) - 1) for index in order[starts[window] : starts[window + 1]])
            for window in range(len(starts) - 1)
        ],
        np.int64,
    )
    kept = np.flatnonzero(targets > 0)
    placed = dict(zip(_FIELDS, (documents, offsets, chunk_sizes, order, starts, targets, kept), strict=True))
    return {**placed, "totals": np.asarray([targets[kept].sum(), np.count_nonzero(targets == 0)], np.int64)}


class FirstFitPacked(pygrain.IterDataset[Batch]):
    """`documents`, each as `pack` reads one (`Pack.document`), packed first-fit into windows as a
    stream of them arrives: `pack.bins` windows
    open at a time, placed as the plan of lengths places them (`FirstFit`) and laid out as it lays
    them (`packed_window`), the windows of grain's `FirstFitPackIterDataset` with the stage's
    fields, so a stream and an indexed read of one corpus pack one way. Its position is grain's
    packer's: where the documents of the windows it emits now began, how many of those windows it
    emitted and how many in all, so a position either saved resumes on the other."""

    def __init__(self, documents: pygrain.IterDataset[Batch], pack: Pack):
        super().__init__(documents)
        self._documents, self._pack = documents, pack

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        return _FirstFitPacking(self._documents.__iter__(), self._pack)


class _FirstFitPacking(pygrain.DatasetIterator[Batch]):
    def __init__(self, documents: pygrain.DatasetIterator[Batch], pack: Pack):
        super().__init__(documents)
        self._documents, self._pack = documents, pack
        self._along = dict(pack.along)
        self._sized = (pack.fields[0], *self._along)
        self._placement = FirstFit((pack.seq_len, *self._along.values()), pack.bins, pack.segments)
        self._restart(documents.get_state())
        self._counter = 0

    def _restart(self, began: Mapping[str, object]) -> None:
        """No window open or emitted, the next document read where `began` stands."""
        self._began, self._ending = began, None
        self._placement.restart()
        self._held: list[list[Batch]] = [[] for _ in range(self._pack.bins)]
        self._emitting: list[Batch] | None = None
        self._next = 0

    def _emitted(self, ending: Mapping[str, object]) -> list[Batch]:
        """Every window that holds a document laid out, to emit in order, the windows after them
        beginning where `ending` stands."""
        self._ending = ending
        self._emitting = [packed_window(held, self._pack.seq_len, self._along)
                          for held in self._held[:self._placement.holding()]]
        self._placement.restart()
        self._held = [[] for _ in range(self._pack.bins)]
        return self._emitting

    def __next__(self) -> Batch:
        emitting = self._emitting
        while emitting is None:
            before = self._documents.get_state()
            try:
                document = next(self._documents)
            except StopIteration:
                if not self._placement.holding():
                    raise
                emitting = self._emitted(before)
                break
            sizes = [len(document[field]) for field in self._sized]
            chosen = self._placement.place(sizes)
            if chosen is None:
                emitting = self._emitted(before)
                chosen = self._placement.place(sizes)
                assert chosen == 0
            self._held[chosen].append(document)
        window = emitting[self._next]
        self._next += 1
        self._counter += 1
        if self._next == len(emitting):
            # The windows after these begin with the document that did not fit them.
            self._began, self._emitting, self._next = self._ending, None, 0
        return window

    def get_state(self) -> Mapping[str, object]:
        return {"parent": self._began, "next_row": self._next, "counter": self._counter}

    def set_state(self, state) -> None:
        saved = record(state, "first-fit packer position")
        self._documents.set_state(record(saved["parent"], "first-fit packer's documents"))
        self._restart(self._documents.get_state())
        self._next = integer(saved["next_row"], "first-fit packer's next window")
        self._counter = integer(saved["counter"], "first-fit packer's windows emitted")

    def close(self) -> None:
        self._documents.close()


@dataclasses.dataclass(frozen=True)
class PackingPlan:
    """The plan a generation publishes for `pack`, which places its documents
    whole: a chunk step in the cached prefix cuts them first."""

    pack: Pack

    def __call__(self, generation: Generation) -> Mapping[str, np.ndarray]:
        lengths = generation.source().lengths()
        if lengths is None:
            raise ValueError("published packing needs certified per-document lengths")
        refuse_overlong(lengths, self.pack, generation.url)
        planned = plan_windows(lengths, self.pack)
        return {plan_name(self.pack, field): value for field, value in planned.items()}


@dataclasses.dataclass(frozen=True)
class _Published:
    """A plan as a generation stores it, read a page at a time."""

    generation: Generation
    pack: Pack
    kind: ClassVar[str] = "published"

    @property
    def location(self) -> str:
        return self.generation.url

    @functools.cached_property
    def most_documents(self) -> int:
        """The most documents a window this plan reads holds (`most_documents`)."""
        return most_documents(self.generation.plan(plan_name(self.pack, "starts")),
                              self.generation.plan(plan_name(self.pack, "accepted")))

    def refuse_cuts(self) -> None:
        """Refuse a plan that cut a document, as one published before packs placed
        documents whole may have: an unchunked pack does not cut."""
        if self.generation.plan(plan_name(self.pack, "offsets")).any():
            raise ValueError(
                f"{self.generation.url} published windows that cut its documents, which its pack "
                f"places whole; read it with flat_map(dew.data.text.Chunks({self.pack.seq_len})) "
                f"before the pack, as it was published"
            )

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
    overlap: int = 0
    """How many tokens the pieces of a cut document share (`Chunks.overlap`)."""
    kind: ClassVar[str] = "planned"

    @functools.cached_property
    def arrays(self) -> Mapping[str, np.ndarray]:
        lengths = self.source.lengths()
        if lengths is None:
            raise ValueError(f"{self.source.describe()} states no document lengths to plan its packing by")
        return plan_windows(lengths, self.pack, self.overlap)

    @property
    def location(self) -> str:
        return self.source.describe()

    @property
    def most_documents(self) -> int:
        """The most documents a window this plan reads holds (`most_documents`)."""
        return most_documents(self.arrays["starts"], self.arrays["accepted"])

    def refuse_cuts(self) -> None:
        """Refuse documents the plan would cut, which an unchunked pack places whole."""
        lengths = self.source.lengths()
        if lengths is not None:
            refuse_overlong(lengths, self.pack, self.source.describe())

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
            document = self._pack.document(record(self._documents[self._at("documents", chunk)],
                                                  "packed document"))
            offset, size = self._at("offsets", chunk), self._at("sizes", chunk)
            chunks.append(
                {name: np.asarray(document[name])[offset : offset + size] for name in self._pack.fields}
            )
        return {
            **packed_window(chunks, self._pack.seq_len),
            "_dew_valid_targets": self._at("targets", window),
        }


@dataclasses.dataclass(frozen=True)
class PackedSource(View[Source]):
    """A source's documents packed first-fit into windows by a plan of their
    lengths (`plan_windows`): the plan a published generation stores
    (`published`), or the one made from the lengths a source states when it
    is first read (`planned`). Either way the windows are read by index in
    one global order. Its rows are windows of its documents (`source`), not
    their records (`View.rows`)."""

    rows: ClassVar[Rows] = "derived"
    source: Source
    plan: _Published | _Held

    @classmethod
    def published(cls, source: Source, generation: Generation, pack: Pack, *,
                  chunked: bool = False) -> PackedSource:
        """`source` read by the plan `generation` published, cutting documents
        only where a chunk step before the pack says to (`chunked`)."""
        plan = _Published(generation, pack)
        if not chunked:
            plan.refuse_cuts()
        return cls(source, plan)

    @classmethod
    def planned(cls, source: Source, pack: Pack, *, overlap: int | None = None) -> PackedSource:
        """`source` read by a plan of the lengths it states, cutting documents
        only where a chunk step before the pack says to, its pieces `overlap`
        tokens apart; with None no step does."""
        plan = _Held(source, pack, 0 if overlap is None else overlap)
        if overlap is None:
            plan.refuse_cuts()
        return cls(source, plan)

    def describe(self) -> str:
        return f"{self.source.describe()} {self.plan.kind} packing={self.plan.pack!r}"

    @property
    def max_length(self) -> int:
        """The inputs each window of the pack holds (`Pack.max_length`), which a token budget divides
        into rows (`dew.data.pipeline.TokenGeometry`)."""
        return self.plan.pack.max_length

    def consumed(self, read: Read) -> PackedSource:
        documents = _consumed(self.source, "indexed" if read == "streamed" else read)
        plan = dataclasses.replace(self.plan, source=documents) if isinstance(self.plan, _Held) else self.plan
        return dataclasses.replace(self, source=documents, plan=plan)

    def fingerprint(self) -> Mapping[str, object]:
        return {"source": self.source.fingerprint(), "packing": self.order_fingerprints()}

    def order_fingerprints(self) -> Mapping[str, str]:
        return self.plan.fingerprints()

    def supports(self, read: Read) -> bool:
        return self.source.supports("indexed" if read == "streamed" else read)

    def records(self, read: Read) -> _PackedRows:
        return _PackedRows(self.source.records(read), self.plan)

    def metadata(self) -> Mapping[str, object]:
        return {
            **super().metadata(),
            "rows": len(self.plan),
            "valid_targets": self.plan.at("totals", 0),
            "zero_target_windows": 0,
        }

    def lengths(self) -> np.ndarray | None:
        return None

    def manifest(self) -> ShardManifest:
        return _indexed_manifest(self, self.plan.location, 1024)
