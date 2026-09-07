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

ENVELOPE = "dew_global_position"
"""The key a global position is written under.

Namespaced because a stream's state is any JSON object grain or a custom
source cares to report, and one of those must not be mistaken for dew's.
"""


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


def encode(place: Global) -> bytes:
    """`place` as the bytes a checkpoint stores for it."""
    return json.dumps({ENVELOPE: dataclasses.asdict(place)}).encode()


def decode(position: bytes) -> Global | None:
    """The global position `position` holds, or None when it is a shard offset.

    Anything that is not dew's envelope is a shard offset, including bytes
    that are not JSON at all: a stream may report its state as any bytes it
    likes, and only this envelope carries the promise that no shard is in it.
    """
    try:
        stored = json.loads(position)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    envelope = stored.get(ENVELOPE) if isinstance(stored, dict) else None
    if not isinstance(envelope, dict):
        return None
    if not {"records", "order"} <= set(envelope):
        raise ValueError(
            f"a saved global data position is missing {sorted({'records', 'order'} - set(envelope))}; "
            f"the checkpoint's position was written by dew and is damaged")
    return Global(records=int(envelope["records"]), order=str(envelope["order"]))


def translates(position: bytes) -> bool:
    """Whether a process count other than the one that wrote `position` can
    read it: true for a global position, false for one shard's own offset."""
    return decode(position) is not None
