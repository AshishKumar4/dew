"""Where a data stream stopped, as the bytes a checkpoint stores.

A stream reports one of two kinds of position. A *global* one counts the
records the whole run has read: one number, the same on every process, with
no shard in it, over an order that is the same order at any process count.
A position two processes wrote is then where one process or four resume. A
*per-process* one is where one process's own shard stopped, which is what
grain reports for a pipeline whose windows are built out of a shard's own
records; only the count that wrote it can read it back.

`dew.checkpoints` decides whether a resume may change the process count, and
it holds the saved bytes without the iterator that wrote them, so which kind
a position is has to be readable from the bytes. A global one is dew's
envelope under one namespaced key; anything else, grain's own states
included, is a shard offset. This module is where the two layers agree on
that: `dew.data` writes the envelope and `dew.checkpoints` reads it.
"""

from __future__ import annotations

import base64
import dataclasses
import json
from collections.abc import Mapping

from dew._records import JSON, integer, json_value, listed, record, text

ENVELOPE = "dew_global_position"
"""The key a global position is written under.

Namespaced because a stream's state is any JSON object grain or a custom
source cares to report, and one of those must not be mistaken for dew's.
"""


PER_SHARE = ("a per-share read resumes only on the shares, workers and cycle length that saved it, so "
             "resume this checkpoint there. A new run resumes on any partition when it reads a global "
             "order: publish its filtered or packed prefix (with a terminal pack's plan) once, by "
             "`dew data publish` or `publish(chain, Cache(url))`, and read it by index with "
             "`.cache(generation)`; provisioning a source's index (`dew data provision`) lets it be read by "
             "index, which is global only where every later step keeps a global order (no filter, flat "
             "map, windows, online packing or workers before the batch); or read the chain elastically "
             "(`.elastic()`), which every reader reads whole and keeps its share of")
"""What a refusal of a per-share position on another topology tells its reader (docs/design/data.md §10,
the provisioned lowering)."""


@dataclasses.dataclass(frozen=True)
class Global:
    """`records` records of `order` have been read, by every process together.

    `order` describes the record order that count indexes into, as JSON, its
    sources among it (`unplaced`). The same
    offset into another corpus, another record count or another shuffle seed
    is another place, and nothing in the offset itself says which order it
    came from: grain's DataLoader compared the repr of its sampler and its
    source to catch that, and an elastic iterator's state is a bare offset.
    """

    records: int
    order: str
    completed: tuple[tuple[str, int], ...] = ()
    """The orders a phased run read before `order`, each with the global
    record count it ended at, oldest first; empty for a run of one order.
    `records` counts every record of the run, the completed phases' too."""

    plan: Mapping[str, object] | None = None
    tokens: tuple[int, ...] | None = ()
    """The committed tokens each phase of `completed` ended at, for a run on the committed-token clock
    (`dew.data.pipeline.schedule`), which no record count of a phase states; empty on the record clock,
    None in a position written before positions held them."""


def encode(place: Global) -> bytes:
    """`place` as the bytes a checkpoint stores for it."""
    held = dataclasses.asdict(place)
    if place.tokens is None:
        # A position read from before positions held stage tokens records none, as it was written.
        del held["tokens"]
    return json.dumps({ENVELOPE: held}).encode()


def decode(position: bytes) -> Global | None:
    """The global position `position` holds, or None when it is a shard offset.

    Anything that is not dew's envelope is a shard offset, including bytes
    that are not JSON at all: a stream may report its state as any bytes it
    likes, and only this envelope carries the promise that no shard is in it.
    """
    envelope = _enveloped(position, ENVELOPE)
    if envelope is None:
        return None
    fields = {"records", "order", "completed"}
    if fields - set(envelope):
        raise ValueError(
            f"a saved global data position is missing {sorted(fields - set(envelope))}; "
            f"the checkpoint's position was written by dew and is damaged")
    completed = tuple((str(order), int(end)) for order, end in envelope["completed"])
    plan = envelope.get("plan")
    return Global(records=int(envelope["records"]), order=str(envelope["order"]),
                  completed=completed, plan=None if plan is None else record(plan, "data plan"),
                  tokens=None if "tokens" not in envelope else tuple(int(end) for end in envelope["tokens"]))


def read(position: bytes) -> Global:
    """The global position `position` holds, for a stream that resumes from
    one; a shard offset is refused, since there is no conversion."""
    place = decode(position)
    if place is None:
        raise ValueError(
            "this training stream resumes from a global record count, and the "
            "saved position is one process's own offset into its shard, from a "
            "stream that batches its own records. There is no conversion; resume "
            "the run that wrote it with the dataset that wrote it")
    return place


REPLICATED = "dew_replicated_position"
"""The key an elastic read's position is written under (`dew.data.Pipeline.elastic`): where the order
one reader reads stopped, which every reader reads whole and keeps its share of, so every process
writes the same one and any process count reads it back."""


PLAN = "dew_data_plan"
"""The key an elastic read's position holds its data plan's record under (`dew.data.plan.Plan.record`)."""


def replicated(position: bytes) -> Mapping[str, object] | None:
    """The elastic read's position `position` holds (`REPLICATED`), or None for any other kind."""
    return _enveloped(position, REPLICATED)


RAMPED = "dew_ramped_position"
"""The key a batch ramp over an elastic read writes its position under (`dew.data.dataset.RampedStream`):
the read's own position where the ramp last read (`source`, an elastic read's, `REPLICATED`), the
records of that read its steps had handed over (`skip`), and every record they had (`records`), all
counted over the one order every reader reads, so every process writes the same one and any process
count reads it back."""


@dataclasses.dataclass(frozen=True)
class Ramped:
    """A batch ramp's place over an elastic read (`RAMPED`): its read restored at `source`, and its next
    read's first `skip` records skipped, it has handed over `records`."""

    source: bytes
    skip: int
    records: int


def encode_ramped(place: Ramped) -> bytes:
    """`place` as the bytes a checkpoint stores for it."""
    return json.dumps({RAMPED: {"source": json.loads(place.source), "skip": place.skip,
                                "records": place.records}}).encode()


def ramped(position: bytes) -> Ramped | None:
    """The ramp's place `position` holds (`RAMPED`), or None for any other kind."""
    held = _enveloped(position, RAMPED)
    if held is None:
        return None
    return Ramped(json.dumps(record(held["source"], "ramped read's position")).encode(),
                  integer(held["skip"], "records skipped"), integer(held["records"], "records handed over"))


STEP_PHASES = "dew_step_phases"
"""The key a read of phases each its own stream writes its position under (`dew.data.dataset.StepPhases`):
the phase it stands in, the steps it has read, the phases it finished and how, and that phase's own
reader's position (`position`, base64), which is what a process count reads."""


def phase_position(position: bytes) -> bytes | None:
    """The position of the phase's own reader inside a read of phases' `position` (`STEP_PHASES`), or None
    for any other kind."""
    held = _enveloped(position, STEP_PHASES)
    if held is None:
        return None
    return base64.b64decode(text(held["position"], "the saved phase's position"))


def unplaced(order: dict[str, JSON] | list[JSON] | str | int | float | bool | None) -> JSON:
    """`order` without its sources' descriptions, which name where a host
    mounted them, its corpora's and phases' included."""
    if isinstance(order, dict):
        return {name: unplaced(value) for name, value in order.items() if name != "source"}
    if isinstance(order, list):
        return [unplaced(value) for value in order]
    return order


def place(position: bytes) -> Mapping[str, object]:
    """The place a position any process count reads (`translates`) names, as every process that wrote it
    holds it: the position with each order in it, a global position's, its finished phases' and its
    data plan's, an elastic read's plan's, without its sources' descriptions (`unplaced`). Processes
    that mount one dataset at different paths read one order (`dew.data.plan.Plan.identity`) to one
    place, and save positions whose bytes differ only there; a ramp's read and a read of phases' own
    reader's position are read the same way."""
    if (phased := phase_position(position)) is not None:
        return {STEP_PHASES: {**_held(position, STEP_PHASES), "position": place(phased)}}
    if (globally := decode(position)) is not None:
        completed = [[unplaced(json.loads(order)), end] for order, end in globally.completed]
        return {ENVELOPE: {**dataclasses.asdict(globally), "order": unplaced(json.loads(globally.order)),
                           "completed": completed,
                           "plan": None if globally.plan is None else _unplaced_plan(globally.plan)}}
    if (read := ramped(position)) is not None:
        return {RAMPED: {"source": place(read.source), "skip": read.skip, "records": read.records}}
    return {REPLICATED: {name: _unplaced_plan(value) if name == PLAN else value
                         for name, value in _held(position, REPLICATED).items()}}


def _unplaced_plan(plan: Mapping[str, object]) -> Mapping[str, object]:
    """A data plan's record (`dew.data.plan.Plan.record`) with its order unplaced (`unplaced`), or a
    read of phases' plans (`dew.data.dataset.PhasedStream`: each phase's, None where it states none),
    each so."""
    held = record(plan, "saved data plan")
    if "phases" in held:
        def phased(phase: JSON, key: str) -> Mapping[str, object] | None:
            return None if phase is None else _unplaced_plan(record(phase, key))

        return {"phases": listed(held["phases"], "saved phase plans", read=phased, what="phases' data plans")}
    return {**held, "order": unplaced(json_value(held["order"], "saved plan order"))}


def _held(position: bytes, key: str) -> dict:
    """The object `position` holds under dew's `key`, which its kind was read by."""
    held = _enveloped(position, key)
    if held is None:
        raise ValueError(f"a saved data position holds no {key}; it was written by dew and is damaged")
    return held


def _enveloped(position: bytes, key: str) -> dict | None:
    """The object `position` holds under dew's `key`, None where it is not JSON or holds no object
    there."""
    try:
        stored = json.loads(position)
    except (UnicodeDecodeError, json.JSONDecodeError):
        # Bytes that are not JSON take the same path as JSON that is not this
        # envelope, which is the one `decode` promises.
        stored = None
    held = stored.get(key) if isinstance(stored, dict) else None
    return held if isinstance(held, dict) else None


def translates(position: bytes) -> bool:
    """Whether a process count other than the one that wrote `position` can
    read it: true for a global position, an elastic read's and a ramp's over
    one, false for one shard's own offset; a read of phases' as its phase's
    own reader's (`phase_position`)."""
    if (phased := phase_position(position)) is not None:
        return translates(phased)
    return decode(position) is not None or replicated(position) is not None or ramped(position) is not None
