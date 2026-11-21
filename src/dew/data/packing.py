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
from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar

import grain.experimental as gx
import grain.python as pygrain
import numpy as np

from dew._records import integer, integers, record
from dew.batch import Batch
from dew.data.dataset import Records
from dew.data.generation import Generation
from dew.data.pipeline import ACCOUNTS, DOCUMENT, ExpandStep, Pack
from dew.data.plan import fingerprint
from dew.data.sources.base import Read, Rows, ShardManifest, Source, View, _consumed, _indexed_manifest
from dew.data.tokens import (
    FirstFit,
    PackedWindows,
    chunk_lengths,
    chunk_table,
    entries,
    first_fit,
    packed_window,
)


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


def plan_windows(lengths: np.ndarray, pack: Pack, overlap: int | None = 0) -> Mapping[str, np.ndarray]:
    """The first-fit plan of documents of `lengths` into `pack.seq_len`-token
    windows, a document longer than a window cut into window-long pieces
    `overlap` tokens apart: each chunk's document, offset and size, the chunks
    in packing order and where each window starts in it, each window's
    targets, and the windows that hold one (`accepted`), with their total and
    the count left out. With `overlap` None no chunk step cuts them: each document is one entry, an
    empty one too, as a stream of them places it (`FirstFitPacked`), counted against the windows'
    segments (row 6277)."""
    if pack.policy != "first-fit":
        raise ValueError("indexed packing currently needs policy='first-fit'")
    if pack.along:
        raise ValueError("a plan of document lengths packs tokens alone; fields along other axes "
                         "pack as their rows arrive")
    if overlap is None:
        chunk_sizes = np.asarray(lengths, np.int64)
        documents, offsets = np.arange(len(chunk_sizes), dtype=np.int64), np.zeros(len(chunk_sizes), np.int64)
    else:
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

    def __init__(self, documents: pygrain.IterDataset[Batch], pack: Pack, settled: Settled | None = None):
        super().__init__(documents)
        self._documents, self._pack, self._settled = documents, pack, settled

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        return _FirstFitPacking(self._documents.__iter__(), self._pack, self._settled)


type Laid = list[tuple[Batch, tuple[int, ...]]]
"""Windows a first-fit laying emits together, in order, each with the places of the documents it holds among
those laid (`_FirstFitLaying.emitted`)."""

type Settled = Callable[[Laid], Laid]
"""What a reader makes of each set of windows a first-fit laying emits together before it hands them on: the
counts a set's windows carry moved to where its documents' examples complete (`dew.data.lowering`)."""


def window_fields(pack: Pack) -> tuple[dict[str, int], tuple[str, ...]]:
    """Each field a window of `pack` holds and its length, and the fields a document's sizes are counted
    along, each with its segment ids and positions: the first token field, and each along an axis of
    its own (`PackedWindows`)."""
    return (dict.fromkeys(pack.fields, pack.seq_len) | dict(pack.along),
            (pack.fields[0], *(name for name, _ in pack.along)))


@dataclasses.dataclass(frozen=True)
class LastPieces:
    """The windows a first-fit laying emits together (`Settled`), each source example's
    counts (`ACCOUNTS`) and, of a mixture's, its document's end (`DOCUMENT`) moved to the window emitted last
    of those holding a piece of it, so it completes where its every piece is delivered (row 5713). An
    expansion's pieces before its last carry no counts (`dew.data.pipeline.consumed_by`), and first-fit can
    lay the last piece in an earlier window than another: width 5 in two windows, pieces of 2, 5 and 3
    tokens lay the first and last in the first window, the 5 in the second. A piece laid in a set emitted
    before is delivered already. The documents' tokens say which each is by `field`'s segment ids."""

    field: str

    def __call__(self, laid: Laid) -> Laid:
        segments = f"{self.field}_segment_ids"
        # Each document of the set by its place among those laid: its window and its last token's place there.
        ends: dict[int, tuple[int, int]] = {}
        for at, (window, held) in enumerate(laid):
            ids = np.asarray(window[segments])
            for segment, document in enumerate(held, start=1):
                # A document of no tokens holds no place in its window, nor any count to move.
                places = np.flatnonzero(ids == segment)
                if places.size:
                    ends[document] = (at, int(places[-1]))
        pieces: list[int] = []
        for document in sorted(ends):
            at, token = ends[document]
            window = laid[at][0]
            if not any(int(window[name][token]) for name in ACCOUNTS):
                pieces.append(document)
                continue
            last = max((*pieces, document), key=lambda piece: (ends[piece][0], piece))
            pieces = []
            into, end = ends[last]
            if into == at:
                continue
            target = laid[into][0]
            for name in ACCOUNTS:
                target[name][end], window[name][token] = window[name][token], 0
            if DOCUMENT in window:
                # Its document ends where its last piece does, and the piece's where the document did.
                ended, held = window[DOCUMENT], target[DOCUMENT]
                held[end], ended[token] = ended[token], held[end]
        return laid


class _FirstFitLaying:
    """Documents, each as `pack` reads one (`Pack.document`), laid first-fit into windows as they come:
    `pack.bins` windows open at a time, a document that fits none emitting every one that holds a
    document and starting afresh. Each window it emits is its fields by name, as grain hands a window
    on, with the places of the documents it holds among those laid, in the order it laid them, which
    its segment ids number from 1."""

    def __init__(self, pack: Pack, settled: Settled | None = None):
        self._pack, self._settled = pack, settled
        self._lengths, self._paired = window_fields(pack)
        self._placement = FirstFit([self._lengths[name] for name in self._paired], pack.bins, pack.segments)
        self._count = 0
        self.restart()

    def restart(self) -> None:
        """No window open."""
        self._placement.restart()
        self._laid: PackedWindows | None = None
        self._held: list[list[int]] = [[] for _ in range(self._pack.bins)]

    def holding(self) -> bool:
        """Whether an open window holds a document."""
        return self._placement.holding() > 0

    def lay(self, document: Batch) -> list[tuple[Batch, tuple[int, ...]]]:
        """`document` laid first-fit: the windows it emitted, where it fit no open one, else none. The
        first of a set lays out the set's windows at its fields' dtypes and shapes, as grain lays them."""
        sizes = [entries(document[name]) for name in self._paired]
        chosen = self._placement.place(sizes)
        emitted: list[tuple[Batch, tuple[int, ...]]] = []
        if chosen is None:
            emitted = self.emitted()
            chosen = self._placement.place(sizes)
            assert chosen == 0
        if self._laid is None:
            self._laid = PackedWindows(document, self._lengths, self._paired, self._pack.bins)
        self._laid.add(chosen, document)
        self._held[chosen].append(self._count)
        self._count += 1
        return emitted

    def emitted(self) -> list[tuple[Batch, tuple[int, ...]]]:
        """Every window that holds a document, in order, none open after."""
        assert self._laid is not None
        windows: Laid = [(dict(sorted(window.items())), tuple(self._held[index]))
                         for index, window in enumerate(self._laid.windows())]
        self.restart()
        return windows if self._settled is None else self._settled(windows)


def first_fit_windows(documents: Sequence[Batch], pack: Pack) -> list[tuple[Batch, tuple[int, ...]]]:
    """`documents`, each as `pack` reads one (`Pack.document`), packed first-fit into windows in the
    order given, as the stream's packer packs them (`FirstFitPacked`, `_FirstFitLaying`). Each window,
    its fields by name, with the indices of the documents it holds in the order it laid them, which its
    segment ids number from 1."""
    laying = _FirstFitLaying(pack)
    windows = [window for document in documents for window in laying.lay(document)]
    return windows + laying.emitted() if laying.holding() else windows


class _FirstFitPacking(pygrain.DatasetIterator[Batch]):
    def __init__(self, documents: pygrain.DatasetIterator[Batch], pack: Pack, settled: Settled | None):
        super().__init__(documents)
        self._documents = documents
        self._laying = _FirstFitLaying(pack, settled)
        self._restart(documents.get_state())
        self._counter = 0

    def _restart(self, began: Mapping[str, object]) -> None:
        """No window open or emitted, the next document read where `began` stands."""
        self._began, self._ending = began, None
        self._laying.restart()
        self._emitting: list[Batch] | None = None
        self._next = 0

    def _emitted(self, windows: list[tuple[Batch, tuple[int, ...]]],
                 ending: Mapping[str, object]) -> list[Batch]:
        """`windows`, to emit in order, the windows after them beginning where `ending` stands."""
        self._ending = ending
        self._emitting = [window for window, _ in windows]
        return self._emitting

    def __next__(self) -> Batch:
        emitting = self._emitting
        while emitting is None:
            before = self._documents.get_state()
            try:
                document = next(self._documents)
            except StopIteration:
                if not self._laying.holding():
                    raise
                emitting = self._emitted(self._laying.emitted(), before)
                break
            emitted = self._laying.lay(document)
            if emitted:
                emitting = self._emitted(emitted, before)
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
        planned = plan_windows(lengths, self.pack, None)
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
    overlap: int | None = None
    """How many tokens the pieces of a cut document share (`Chunks.overlap`); None where no chunk step cuts
    them, each document placed whole, an empty one too (`plan_windows`)."""
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
        plan = _Held(source, pack, overlap)
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


class ConcatPacked(pygrain.IterDataset[Batch]):
    """`documents`, each as `pack` reads one (`Pack.document`), laid end to end in windows of `pack.seq_len`
    tokens and cut where a window ends, the next window starting again `pack.overlap` tokens before the cut,
    as Grain's concatenating packer lays them (`ConcatThenSplitIterDataset`, `PackedWindows`). A window
    holds at most `pack.segments` documents or pieces of one. Each piece holds its document's entries along
    an axis of its own (`pack.along`) whole, as Grain's packer holds a feature it never cuts (its
    `meta_features`), so every window a document reaches holds them; a window closes where the next
    document's do not fit what is left of theirs, and a document whose entries outgrow an empty window is
    refused. Its position is where the next window begins: the document read then, and how far into it."""

    def __init__(self, documents: pygrain.IterDataset[Batch], pack: Pack):
        super().__init__(documents)
        self._documents, self._pack = documents, pack

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        return _ConcatPacking(self._documents.__iter__(), self._pack)


type _Begun = tuple[Mapping[str, object], Batch, int]
"""A document a window begins with: the documents' position before it, the document, and where in it."""


class _ConcatPacking(pygrain.DatasetIterator[Batch]):
    def __init__(self, documents: pygrain.DatasetIterator[Batch], pack: Pack):
        super().__init__(documents)
        self._documents, self._pack = documents, pack
        self._lengths, self._paired = window_fields(pack)
        self._pending: _Begun | None = None

    def _begun(self) -> _Begun:
        if (pending := self._pending) is not None:
            self._pending = None
            return pending
        before = self._documents.get_state()
        return before, next(self._documents), 0

    def __next__(self) -> Batch:
        pack, ids = self._pack, self._pack.fields[0]
        room, laid, held = dict(self._lengths), None, 0
        while pack.segments is None or held < pack.segments:
            try:
                before, document, start = self._begun()
            except StopIteration:
                break
            along = {name: entries(document[name]) for name, _ in pack.along}
            if any(size > self._lengths[name] for name, size in along.items()):
                raise ValueError(f"a document of {along} entries along its own axes is longer than the "
                                 f"windows of {dict(pack.along)} it packs into")
            count = entries(document[ids])
            # A full window takes a document of no tokens left, as Grain's packer does; one with tokens
            # begins the next window, as does one whose entries along their own axes no longer fit.
            if (start < count and not room[ids]) or any(size > room[name] for name, size in along.items()):
                self._pending = (before, document, start)
                break
            end = min(count, start + room[ids])
            piece = {name: value[start:end] if name in pack.fields else value
                     for name, value in document.items()}
            if laid is None:
                laid = PackedWindows(piece, self._lengths, self._paired, 1)
            laid.add(0, piece)
            held += 1
            room = {name: left - (end - start if name in pack.fields else along[name])
                    for name, left in room.items()}
            if end < count:
                self._pending = (before, document, max(end - pack.overlap, 0))
                break
        if laid is None:
            raise StopIteration
        (window,) = laid.windows()
        return window

    def get_state(self) -> Mapping[str, object]:
        if self._pending is None:
            return {"parent": self._documents.get_state(), "offset": 0}
        before, _, start = self._pending
        return {"parent": before, "offset": start}

    def set_state(self, state: Mapping[str, object]) -> None:
        saved = record(state, "concatenating packer position")
        if "remainder_slices" in saved:
            self._grains(saved)
            return
        self._documents.set_state(record(saved["parent"], "concatenating packer's documents"))
        self._begin(integer(saved["offset"], "concatenating packer's offset"))

    def _begin(self, offset: int) -> None:
        """The next window begun `offset` tokens into the next document, or with it whole."""
        self._pending = None
        if offset:
            before = self._documents.get_state()
            self._pending = (before, next(self._documents), offset)

    def _grains(self, saved: Mapping[str, object]) -> None:
        """A position Grain's concatenating packer saved (`ConcatThenSplitIterDataset`'s, before row 9059):
        its documents' position before the remainder it held, where the remainder starts in its document,
        and the windows emitted since, which a restore emits again. Where the pack overlaps its cuts, its
        documents were those cuts written out (`_OverlappedCuts` then, saving the room its window had left
        beside their position), and the remainder starts in that document, at a cut: past the pieces before
        it, the first that room long and each after a window's, each next one starting `overlap` tokens
        before the last one's end in the document itself."""
        pack = self._pack
        parent = record(saved["parent_state"], "Grain packer's documents")
        room = pack.seq_len
        if pack.overlap:
            room = integer(parent["room"], "packed window room")
            parent = record(parent["documents"], "packed documents")
        self._documents.set_state(parent)
        start, _ = integers(record(saved["remainder_slices"], "packed remainder")[pack.fields[0]],
                            "packed remainder")
        offset = max(start, 0)
        if pack.overlap and start > 0:
            written, offset = start, 0
            while written > 0:
                written, offset, room = written - room, max(offset + room - pack.overlap, 0), pack.seq_len
            if written:
                raise ValueError(f"a saved remainder {start} tokens into its overlapped document starts at "
                                 "no cut of it")
        self._begin(offset)
        for _ in range(integer(saved["elements_from_buffer_after_checkpoint"], "windows emitted")):
            next(self)

    def close(self) -> None:
        self._documents.close()


def packed_documents(documents: pygrain.IterDataset, stage: Pack, *, consumed: bool = False
                     ) -> pygrain.IterDataset:
    """`documents` packed for the stage's policy: first-fit as an indexed read's plan places them
    (`FirstFitPacked`), each example's counts where its last piece is emitted where the documents carry them
    (`consumed`, `LastPieces`); concatenated as they come (`ConcatPacked`); best-fit by Grain's packer."""
    if stage.policy == "first-fit":
        return FirstFitPacked(documents, stage, LastPieces(stage.fields[0]) if consumed else None)
    if stage.policy == "concat":
        return ConcatPacked(documents, stage)
    sizes, _ = window_fields(stage)
    # Fields cut alike with the first share its segments and positions, as the indexed planner's
    # windows do (`packed_window`), so Grain writes no copy of them for the rest.
    return gx.BestFitPackIterDataset(documents, length_struct=sizes, num_packing_bins=stage.bins,
                                     shuffle_bins=False, max_sequences_per_bin=stage.segments,
                                     meta_features=stage.fields[1:])
