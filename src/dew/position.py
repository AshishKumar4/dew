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

import dataclasses
import json
from collections.abc import Mapping

from dew._records import record

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

    `order` describes the record order that count indexes into. The same
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


def encode(place: Global) -> bytes:
    """`place` as the bytes a checkpoint stores for it."""
    return json.dumps({ENVELOPE: dataclasses.asdict(place)}).encode()


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
                  completed=completed, plan=None if plan is None else record(plan, "data plan"))


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


def replicated(position: bytes) -> Mapping[str, object] | None:
    """The elastic read's position `position` holds (`REPLICATED`), or None for any other kind."""
    return _enveloped(position, REPLICATED)


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
    read it: true for a global position and an elastic read's, false for one
    shard's own offset."""
    return decode(position) is not None or replicated(position) is not None
