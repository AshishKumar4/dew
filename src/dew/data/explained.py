"""What `Loader.explain` renders: how a compiled read reads its training batches, as its compile chose
and sized them (`TrainingRead`, `Sized`) and as its plan records them (`dew.data.plan.Plan`). Every line is a
fact the compile recorded; nothing here reads, measures or decides."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from dew._records import record

if TYPE_CHECKING:
    from dew.data.dataset import Loader, Loading
    from dew.data.sources.base import Read

type Chooser = Literal["compile", "chain", "sources"]
"""Who chose a read: `compile(read=)`, the chain's `.read(mode)`, or its sources (`_preferred_read`)."""


@dataclasses.dataclass(frozen=True)
class Sized:
    """What automatic loading measured of a read as it opened, which it sized its workers and buffers by
    (`Loading.resolved`)."""

    sampled: int
    """The batches it read in turn, the first, which opens the read, among them (`SAMPLED_BATCHES`)."""
    python: float
    """The mean seconds a sampled batch after the first took to make."""
    slowest: float
    """The seconds the slowest of them took."""
    batch_bytes: int
    """A sampled batch's mean bytes."""
    cpus: int
    budget: int
    """The bytes the read's workers and buffers may hold: half the memory the process may still take, or
    `ram_budget_mb`."""
    batches: int | None
    """The batches the run reads, its stated length; None where nothing states it."""


@dataclasses.dataclass(frozen=True)
class TrainingRead:
    """How a compile reads its training batches: the read and who chose it, the loading its readers run
    with as automatic loading sized it, the worker processes the read runs in, each corpus's own where a
    read that drops records reads in workers of each corpus's (`dew.data.lowering._Build.pools`), and
    a read that would read faster, which the compile leaves to be asked for."""

    read: Read
    chooser: Chooser
    loading: Loading
    workers: int
    pools: Mapping[str, int] = dataclasses.field(default_factory=dict)
    faster: Read | None = None
    """`memory`, where another read was taken and every source can be read into memory, its chain lowering
    there: a memory read takes each record from RAM after one load of the corpus, with no read of its store
    per record. Whether the corpus fits is not known: its sources' bytes before its steps are not measured.
    A position is saved under its read, so another resumes none of it: it is named, never taken."""


_CHOSEN: Mapping[Chooser, str] = {
    "compile": "as compile(read=) asked",
    "chain": "as the chain's .read() asked",
    "sources": "as its sources prefer",
}


def explained(loader: Loader) -> str:
    """`loader`'s training read as its compile recorded it, a line a fact (`Loader.explain`)."""
    tokens = "" if loader.batch_tokens is None else f", {loader.batch_tokens} tokens"
    records = "" if loader.records is None else f", {loader.records} records an epoch"
    lines = [f"batch {loader.batch}{tokens}{records}"]
    if loader.batches:
        lines.append(f"{len(loader.batches)} phases, batches {list(loader.batches)}; the read below is the "
                     "first's")
    if (reading := loader.reading) is not None:
        lines.append(f"read: {reading.read}, {_CHOSEN[reading.chooser]}")
        lines.extend(_loading_lines(reading))
        if reading.faster is not None:
            lines.append(f"faster where the corpus fits in host memory: a {reading.faster} read, each record "
                         f"from RAM after one load of it; ask for it with .read({reading.faster!r}): a "
                         f"position saved under the {reading.read} read does not resume under it")
            lines.extend(_rows_held(loader, reading))
    if (plan := loader.plan) is not None:
        lines.append(f"order: {'one global order' if plan.lowering == 'global' else 'cut per share'}")
        lines.extend(f"  {name}: {json.dumps(value, sort_keys=True, separators=(',', ':'))}"
                     for name, value in record(plan.spelled_order(), "data plan order").items())
        lines.extend(f"step: {step.name} {step.config}" for step in plan.manifest.steps)
        lines.extend(f"derived: {name} {value}" for name, value in plan.derived.items())
        lines.extend(f"cache: {reference}" for reference in plan.cache)
        if plan.volatile:
            lines.append("volatile: a source or step may read differently each time")
    return "\n".join(lines)


def _loading_lines(reading: TrainingRead) -> list[str]:
    """The workers and buffers `reading` reads in, and what automatic loading measured to size them."""
    loading = reading.loading
    held = (f"each holding {loading.ahead} batches ready" if reading.workers else "in the training process")
    lines = [f"workers: {reading.workers}, {held}; {loading.threads} threads a reader, "
             f"{loading.read_buffer} reads ahead"]
    lines.extend(f"  {corpus}: {count} workers" for corpus, count in reading.pools.items())
    if (sized := loading.sized) is not None:
        run = "an unstated count of" if sized.batches is None else str(sized.batches)
        lines.append(f"sized for {run} batches from {sized.sampled} sampled: {sized.python * 1e3:.2f} ms of "
                     f"Python a batch, the slowest {sized.slowest * 1e3:.2f} ms, {sized.batch_bytes} bytes; "
                     f"{sized.cpus} CPUs, {sized.budget / 2**30:.2f} GiB to hold")
    return lines


def _rows_held(loader: Loader, reading: TrainingRead) -> list[str]:
    """The bytes of an epoch's rows as the read's sampled batches hold them, where they were sampled and the
    records counted: the rows its steps make, not the source records a memory read holds first."""
    sized = reading.loading.sized
    if sized is None or loader.records is None:
        return []
    rows, budget = loader.records * sized.batch_bytes / loader.batch / 2**30, sized.budget / 2**30
    return [f"  an epoch's rows hold {rows:.2f} GiB as batched, against {budget:.2f} GiB this read may "
            "hold; the sources' own bytes are not measured"]
