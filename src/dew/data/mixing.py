"""Accepted-row accounting and mixtures measured in valid target tokens."""

from __future__ import annotations

import dataclasses
import functools
import itertools
import logging
import math
import sys
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Protocol, TypeGuard, runtime_checkable

import grain.experimental as gx
import grain.python as pygrain
import numpy as np

from dew._records import boolean, integer, record, text
from dew.batch import Batch
from dew.data.dataset import LONG_RUN, logged_run
from dew.data.pipeline import Owed, Record, Rejected, accounts, member_of, members_of
from dew.data.shared import Shared

_log = logging.getLogger(__name__)


@dataclasses.dataclass
class CorpusCounts:
    """What one corpus's read counted, under its identity (`name`, which plans and saved positions
    key it by) and the short name its metrics go by (`label`). `accepted` counts the rows it
    delivered and `truncated` those of them cut short, its source examples where an objective makes
    several rows of each (`FlatMap.examples`): in a run bounded in examples, those whose last rows it
    delivered, as each row says (`examples`), an example a step refused or made no row of counted as
    refused; in any other, those it read before the objective's step made rows of them; `cut_examples`,
    `cut_rows` and
    `cut_removed` count, by what each cut (a decision row's instructions, options, state, or a context it
    did not read), the delivered rows a cut shortened, the rows it shortened of them, and the tokens it
    removed (characters, for a kind that names them: `state_characters`), as the delivered rows report
    them (`delivered`); a rejection counts what the step that
    drops it reads, a record or a piece or window a step made of one, a member a delivered row left out
    among them. `declined` counts, by the reason its layout refuses it, the delivered rows whose
    augmentation a training read declines where it draws it (a decision row's "none of the above",
    `_dew_declined`), each then read as it was admitted. `attempted` counts what its last
    filter read, for its own warning; it describes no one unit of the corpus, so no metric shows it.
    A counter past an objective's step that makes rows of the source examples it counted before
    (`rows` false) counts the rows it delivers, cut short and cut, and what it rejects of them, but no
    example: a corpus none of whose counters counts its examples shows no `data/rows`. `delivered_rows`
    counts the rows it delivered, which the share of them cut short is of: `accepted`'s count where each
    row is one, not where a row stands for the examples it completes (`examples`) or for none. A
    mixture's members count their records as the mix draws them; one a run bounds in examples counts each
    member's where their rows are delivered, and what a step after the mix refuses of a member's rows
    under that member (`MEMBER`, `MEMBERS`): a step of Grain's own after the mix, which no row's member
    goes through, refuses under the mixture's name."""

    name: str
    label: str
    rows: bool = True
    examples: bool = False
    """Whether the rows it counts delivered stand for the corpus's source examples their last rows
    complete, as each says (`CONSUMED`, less those it carries for examples a step refused or made no row
    of, `REFUSED`): a run bounded in examples counts them so where its rows are delivered, in the reading
    process (`dew.data.lowering._Build.records`)."""
    attempted: int = 0
    accepted: int = 0
    truncated: int = 0
    delivered_rows: int = 0
    chunks: int = 0
    chunk_records: int = 0
    rejected: Counter[str] = dataclasses.field(default_factory=Counter)
    cut_examples: Counter[str] = dataclasses.field(default_factory=Counter)
    cut_rows: Counter[str] = dataclasses.field(default_factory=Counter)
    cut_removed: Counter[str] = dataclasses.field(default_factory=Counter)
    left_out: Counter[str] = dataclasses.field(default_factory=Counter)
    """The members delivered rows left out, by reason (`_dew_rejected`), counted apart from the records
    the corpus rejected whole, so a recount of the delivered rows (`counted_anew`) starts them over."""
    rejected_cut: Counter[str] = dataclasses.field(default_factory=Counter)
    """The tokens the cuts of the records the corpus rejected whole removed, by what each cut
    (`Rejected.cut`), counted with their rejections."""
    left_out_cut: Counter[str] = dataclasses.field(default_factory=Counter)
    """The tokens the cuts of the members delivered rows left out removed, by what each cut
    (`_dew_left_out_cut`), counted with them."""
    declined: Counter[str] = dataclasses.field(default_factory=Counter)

    def delivered(self, element: Batch | bytes, examples: int | None = None) -> None:
        """Count one row the corpus delivered: accepted, cut short where it says it was
        (`_dew_truncated`), each cut it reports (`_dew_cut`: by what it cut, the rows it shortened and
        the tokens it removed), the members it left out by reason (`_dew_rejected`) and the tokens their
        cuts removed (`_dew_left_out_cut`), and the augmentation a read declines for it, by reason
        (`_dew_declined`). A mixture's counter counts the `examples` of this corpus the row completes
        (`MEMBERS`)."""
        self.delivered_rows += 1
        if examples is not None:
            self.accepted += examples
        elif not isinstance(element, Mapping):
            self.accepted += int(self.rows)
        elif self.examples:
            consumed, refused = accounts(element)
            self.accepted += consumed - refused
        else:
            self.accepted += int(self.rows)
        if not isinstance(element, Mapping):
            return
        self.truncated += int(bool(element.get("_dew_truncated", False)))
        for kind, (rows, tokens) in element.get("_dew_cut", {}).items():
            self.cut_examples[kind] += 1
            self.cut_rows[kind] += int(rows)
            self.cut_removed[kind] += int(tokens)
        for reason, count in element.get("_dew_rejected", {}).items():
            self.left_out[reason] += int(count)
        self.left_out_cut.update(element.get("_dew_left_out_cut", {}))
        for reason, count in element.get("_dew_declined", {}).items():
            self.declined[reason] += int(count)

    def dropped(self, rejected: Rejected) -> None:
        """Count one record the corpus rejected whole, under its reason, with the tokens its cuts
        removed (`Rejected.cut`)."""
        self.rejected[rejected.reason] += 1
        self.rejected_cut.update(rejected.cut)

    def counted_anew(self) -> None:
        """Start over the counts of delivered rows, for a read that counts them again whole (a memory
        read's rows, read once more as it holds them)."""
        self.accepted, self.truncated, self.delivered_rows = 0, 0, 0
        for counted in (self.cut_examples, self.cut_rows, self.cut_removed, self.left_out, self.left_out_cut,
                        self.declined):
            counted.clear()

    @property
    def refused(self) -> Counter[str]:
        """What the corpus left out, by reason: the records it rejected whole and the members its
        delivered rows left out (`left_out`)."""
        return self.rejected + self.left_out

    @property
    def refused_cut(self) -> Counter[str]:
        """The tokens the cuts of what the corpus left out removed, by what each cut: of the records it
        rejected whole and of the members its delivered rows left out."""
        return self.rejected_cut + self.left_out_cut

    def metrics(self, label: str) -> dict[str, float]:
        """The metrics a run logs of this corpus under `label`, each exact in one unit: the rows it
        delivered, its rejections and the augmentation it declined by reason as integers, the share
        of those rows cut short and the chunks per chunked record."""
        examples = {f"data/rows/{label}": self.accepted} if self.rows else {}
        return {
            **examples,
            f"data/truncated/{label}": self.truncated / max(1, self.delivered_rows),
            f"data/chunks_per_record/{label}": self.chunks / max(1, self.chunk_records),
            **{f"data/rejected/{label}/{reason}": count for reason, count in self.refused.items()},
            **{f"data/cut_examples/{label}/{kind}": count for kind, count in self.cut_examples.items()},
            **{f"data/cut/{label}/{kind}": count for kind, count in self.cut_rows.items()},
            **{f"data/cut_removed/{label}/{kind}": count for kind, count in self.cut_removed.items()},
            **{f"data/cut_refused/{label}/{kind}": count for kind, count in self.refused_cut.items()},
            **{f"data/declined/{label}/{reason}": count for reason, count in self.declined.items()},
        }

    def state(self) -> Mapping[str, object]:
        return {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "truncated": self.truncated,
            "delivered_rows": self.delivered_rows,
            "chunks": self.chunks,
            "chunk_records": self.chunk_records,
            "rejected": dict(self.rejected),
            "left_out": dict(self.left_out),
            "cut_examples": dict(self.cut_examples),
            "cut_rows": dict(self.cut_rows),
            "cut_removed": dict(self.cut_removed),
            "rejected_cut": dict(self.rejected_cut),
            "left_out_cut": dict(self.left_out_cut),
            "declined": dict(self.declined),
        }

    def restore(self, state: Mapping[str, object]) -> None:
        self.attempted = integer(state["attempted"], "attempted")
        self.accepted = integer(state["accepted"], "accepted")
        self.truncated = integer(state["truncated"], "truncated")
        self.delivered_rows = integer(state["delivered_rows"], "delivered_rows")
        self.chunks = integer(state["chunks"], "chunks")
        self.chunk_records = integer(state["chunk_records"], "chunk_records")
        self.rejected = Counter(
            {name: integer(value, name) for name, value in record(state["rejected"], "rejected").items()}
        )
        for name in CUT_COUNTS:
            setattr(self, name, Counter({kind: integer(value, kind)
                                         for kind, value in record(state[name], name).items()}))

    def followed(self, reader: CorpusCounts) -> CorpusCounts:
        """These counts of the prefix a memory read ran, followed by `reader`'s of the stages it
        read after them: the reader delivered the rows and counted those cut short, and its examples
        where it counts them (`rows`), else the prefix did, and the two dropped and chunked apart."""
        examples = {} if reader.rows else {
            "rows": self.rows, "examples": self.examples, "accepted": self.accepted}
        return dataclasses.replace(
            reader, **examples, attempted=max(self.attempted, reader.attempted),
            chunks=self.chunks + reader.chunks, chunk_records=self.chunk_records + reader.chunk_records,
            rejected=self.rejected + reader.rejected, rejected_cut=self.rejected_cut + reader.rejected_cut)

    def expanded(self, outputs: Sequence[Record]) -> None:
        """Count one chunked document and its fan-out before later stages can reject chunks."""
        self.chunk_records += 1
        self.chunks += sum(value is not None and not isinstance(value, Rejected) for value in outputs)


_WARN_OVER = 0.01
"""The share of a corpus's attempted records it may reject before a run warns, once: a filter that
drops more than one record in a hundred is worth a look."""

class _KeepAccepted:
    def __init__(self, counts: CorpusCounts, tokens: Callable[[Batch], int] | None,
                 certain_after: int | None = None, members: Mapping[str, CorpusCounts] | None = None,
                 credits: bool = False):
        self.counts, self.tokens, self.certain_after = counts, tokens, certain_after
        self.members, self.credits = dict(members or {}), credits
        """A mixture's members' counts, under which it counts what it rejects of each member's rows
        (`Rejected.member`, `MEMBER`), and, where it `credits` them (its last counter), the examples each
        member's rows complete as it delivers them (`MEMBERS`)."""
        self.consecutive = 0
        self.warned = False
        self.owed = Owed()
        """The source examples the rows it rejected since the last it kept were the last of (`CONSUMED`),
        which that next row carries on (`_AcceptedIterator`)."""
        self.rejections = 0
        """How many records it rejected, its own and its members', counted as it rejects them."""
        self._states: Shared[Mapping[str, object]] | None = None
        self._touched: set[str] = set(self.members)
        """The members whose counts moved since `_states` was made, which alone a capture makes again."""

    def _of(self, member: str | None) -> CorpusCounts:
        """The counts a row of `member` (`MEMBER`) is counted under: its member's, else the counter's own."""
        if member is None or not self.members:
            return self.counts
        self._touched.add(member)
        return self.members[member]

    def member_states(self) -> Shared[Mapping[str, object]]:
        """Each member's counts as they stand, by member key (`MEMBERS`): the states captured before, the
        members' that moved since made again, the rest shared with them (`Shared`), so a capture before
        each record costs it the members it counted, not every member (row 4079)."""
        touched = {key: {"corpus": self.members[key].name, "label": self.members[key].label,
                         "counts": self.members[key].state()} for key in self._touched}
        self._states = Shared.of(touched) if self._states is None else self._states.replaced(touched)
        self._touched.clear()
        return self._states

    def restored(self) -> None:
        """Every member's counts and the rejections as a restore left them: each member's state made again
        at its next capture, the rejections counted afresh."""
        self._touched = set(self.members)
        self.rejections = sum(sum(counts.rejected.values())
                              for counts in (self.counts, *self.members.values()))

    def __call__(self, element: Batch | Rejected) -> TypeGuard[Batch]:
        self.counts.attempted += 1
        if isinstance(element, Rejected):
            self.rejections += 1
            reason = element.reason
            self.owed.owe(element.consumed)
            self._of(element.member).dropped(element)
        else:
            count = None if self.tokens is None else self.tokens(element)
            if count is not None and count < 0:
                raise ValueError(f"corpus {self.counts.name} has a negative valid-target count: {count}")
            if count != 0:
                self._delivered(element)
                self.consecutive = 0
                return True
            reason = "zero_targets"
            self.rejections += 1
            self.owed.owe(accounts(element)[0])
            self._of(member_of(element)).rejected[reason] += 1
        self.consecutive += 1
        if not self.warned and self.rejections / self.counts.attempted > _WARN_OVER:
            _log.warning("corpus %s rejected more than %.1f%% of attempted records (%s)",
                         self.counts.label, _WARN_OVER * 100, reason)
            self.warned = True
        if self.certain_after is not None and self.consecutive >= self.certain_after:
            reasons = ", ".join(f"{name} {count}" for name, count in self._rejected().most_common(3))
            raise ValueError(
                f"corpus {self.counts.label} rejected its last {self.consecutive} records, every record of "
                f"two whole passes of what it reads ({reasons}), and its steps reject the same records each "
                "pass, so the read would read nothing forever. A read per share filters each share's own "
                "records: read it with fewer shares, or elastically (`.elastic()`), every reader filtering "
                "every record")
        if logged_run(self.consecutive, LONG_RUN):
            rejected = self._rejected()
            reasons = ", ".join(f"{name} {count}" for name, count in rejected.most_common(3))
            # What it kept, which a counter past an objective's expansion reads but counts as no rows.
            kept = self.counts.attempted - sum(rejected.values())
            _log.warning("corpus %s rejected its last %d records and has kept %d; the read goes on until "
                         "one is kept (most rejected: %s)", self.counts.label, self.consecutive, kept,
                         reasons)
        return False


    def _rejected(self) -> Counter:
        """What it rejected by reason, its own and its members' (a mixture's counter past the mix): merged
        only where a raise or a log names the reasons, never a row at a time (row 3531)."""
        return sum((counts.rejected for counts in self.members.values()), self.counts.rejected.copy())

    def _delivered(self, element: Batch) -> None:
        """Count a row delivered: under its member where a mixture's last counter credits its members,
        each member the examples its rows complete (`MEMBERS`), the row's own counts its member's."""
        if not self.credits:
            self.counts.delivered(element)
            return
        own = member_of(element)
        completed = members_of(element)
        owner = self._of(own)
        for member, examples in completed.items():
            if member != own:
                self._touched.add(member)
                self.members[member].accepted += examples
        owner.delivered(element, 0 if own is None else completed.get(own, 0))


COUNTER = "dew.corpus_counts"

CUT_COUNTS = ("cut_examples", "cut_rows", "cut_removed", "left_out", "rejected_cut", "left_out_cut",
              "declined")
"""A corpus's counts by kind (`CorpusCounts`), each a record of its own in a saved position."""
"""Where a counting iterator's position keeps its corpus's counts, beside its parent's position:
a key of Dew's own, which `state_counts` and `rewound` find and no Grain position holds."""


class _CountedIterator[Element](pygrain.DatasetIterator[Element]):
    """Checkpoint a corpus counter with its parent, including replay during parent restoration."""

    def __init__(self, parent: pygrain.DatasetIterator, counts: CorpusCounts):
        super().__init__(parent)
        self._read, self._counts = parent, counts

    def __next__(self) -> Element:
        return next(self._read)

    def get_state(self) -> Mapping[str, object]:
        return {"parent": self._read.get_state(), COUNTER: self._counted()}

    def _counted(self) -> Mapping[str, object]:
        # A counter that counts no rows says so (`CorpusCounts.rows`); any other counts them.
        rowless = {} if self._counts.rows else {"rows": False}
        return {"corpus": self._counts.name, "label": self._counts.label, "counts": self._counts.state(),
                **rowless}

    def set_state(self, state: Mapping[str, object]) -> None:
        self._read.set_state(record(state["parent"], "counted parent"))
        self._restore(record(state[COUNTER], "corpus counter"))

    def _restore(self, counted: Mapping[str, object]) -> None:
        self._counts.restore(record(counted["counts"], "corpus counts"))

    def close(self) -> None:
        self._read.close()


class Chunked(pygrain.IterDataset[Record]):
    """A chunk expansion counted at its fan-out, with counters in the iterator position: its documents
    and their chunks, and no row (`CorpusCounts.rows`), which the counters about it count."""

    def __init__(self, parent: pygrain.IterDataset[Record], name: str, label: str,
                 expand: Callable[[CorpusCounts], gx.FlatMapTransform]):
        super().__init__(parent)
        self._records, self._name, self._label, self._expand = parent, name, label, expand

    def __iter__(self) -> pygrain.DatasetIterator[Record]:
        counts = CorpusCounts(self._name, self._label, rows=False)
        return _CountedIterator(iter(gx.FlatMapIterDataset(self._records, self._expand(counts))), counts)


class _AcceptedIterator(_CountedIterator[Batch]):
    _read: pygrain.DatasetIterator[Batch]

    def __init__(self, parent: pygrain.DatasetIterator[Batch], predicate: _KeepAccepted):
        super().__init__(parent, predicate.counts)
        self._predicate = predicate

    def __next__(self) -> Batch:
        # The examples the rows rejected before it were the last of are consumed where this row is.
        return self._predicate.owed.paid(next(self._read))

    def _counted(self) -> Mapping[str, object]:
        # A mixture's counter keeps its members' counts beside its own, by member key (`MEMBERS`).
        return {**super()._counted(), "consecutive": self._predicate.consecutive,
                "warned": self._predicate.warned, "consumed": self._predicate.owed.examples,
                "members": self._predicate.member_states()}

    def _restore(self, counted: Mapping[str, object]) -> None:
        super()._restore(counted)
        self._predicate.consecutive = integer(counted["consecutive"], "consecutive")
        self._predicate.warned = boolean(counted["warned"], "warned")
        self._predicate.owed = Owed(integer(counted["consumed"], "consumed examples pending"))
        # A member that joined since starts its counts; one that left takes its counts with it.
        members = record(counted["members"], "a mixture counter's members")
        for key, counts in self._predicate.members.items():
            if key in members:
                counts.restore(record(record(members[key], "member counter")["counts"], "member counts"))
        self._predicate.restored()


class Carried(pygrain.IterDataset[Batch]):
    """`parent`'s rows, each marker's source examples (`Rejected.consumed`) carried on to the next row and
    the markers dropped, counting nothing: the carrier after a step whose markers no corpus's count
    follows, a batched map's group it makes no rows of (`dew.data.lowering._Groups`). The examples it
    owes are checkpointed with its parent's position; those the stream ends owing are its end's
    (`dew.training._fit.drained`)."""

    def __init__(self, parent: pygrain.IterDataset[Batch | Rejected]):
        super().__init__(parent)
        self._records = parent

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        return _CarriedIterator(iter(self._records))


class _CarriedIterator(pygrain.DatasetIterator[Batch]):
    def __init__(self, parent: pygrain.DatasetIterator):
        super().__init__(parent)
        self._read, self._owed = parent, Owed()

    def __next__(self) -> Batch:
        while True:
            element = next(self._read)
            if isinstance(element, Rejected):
                self._owed.owe(element.consumed)
                continue
            return self._owed.paid(element)

    def get_state(self) -> Mapping[str, object]:
        return {"parent": self._read.get_state(), "owed": self._owed.examples}

    def set_state(self, state: Mapping[str, object]) -> None:
        self._read.set_state(record(state["parent"], "carried parent"))
        self._owed = Owed(integer(state["owed"], "examples owed"))

    def close(self) -> None:
        self._read.close()


class Accepted(pygrain.IterDataset[Batch]):
    """Grain's filter, with checkpointed counts. A filter of an endless repeat whose passes it
    rejects alike refuses a run of `certain_after` rejections, two whole passes, as reading nothing
    forever; any other run of rejections it logs and reads on, a fetch that failed for now say. One
    past an objective's step that makes rows of source examples counts none of the rows it keeps
    (`rows` false, `CorpusCounts.rows`). A mixture's counter past the mix counts what it rejects of each
    member's rows under that member (`members`, each's name and label), and its last counter, which
    `credits` them, the examples each member's rows complete as it delivers them (`MEMBERS`); each
    member is its key, name and label."""

    def __init__(
        self,
        parent: pygrain.IterDataset[Batch | Rejected] | pygrain.IterDataset[Batch],
        name: str,
        label: str,
        *,
        tokens: Callable[[Batch], int] | None = None,
        certain_after: int | None = None,
        rows: bool = True,
        examples: bool = False,
        members: Sequence[tuple[str, str, str]] = (),
        credits: bool = False,
    ):
        super().__init__(parent)
        self._records, self._name, self._label = parent, name, label
        self._tokens, self._certain_after, self._rows, self._examples = tokens, certain_after, rows, examples
        self._members, self._credits = tuple(members), credits

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        counts = CorpusCounts(self._name, self._label, rows=self._rows, examples=self._examples)
        members = {key: CorpusCounts(name, label, examples=True) for key, name, label in self._members}
        predicate = _KeepAccepted(counts, self._tokens, self._certain_after, members, self._credits)
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
    """Each row from the corpus furthest below its share of the targets delivered, each corpus read
    as it is drawn (`_MixReads`)."""

    def __init__(
        self,
        reads: Sequence[_MixRead],
        weights: tuple[int, ...],
        tokens: Callable[[Batch], int],
        keys: tuple[str, ...],
    ):
        super().__init__(reads)
        self._reads, self._weights, self._tokens, self._keys = tuple(reads), weights, tokens, keys
        self._delivered = [0] * len(reads)
        self._parents = _Captured(dict(zip(keys, self._reads, strict=True)))

    def __next__(self) -> Batch:
        total = sum(self._delivered)
        scale = sum(self._weights)
        chosen = max(
            range(len(self._reads)),
            key=lambda index: self._weights[index] * total - self._delivered[index] * scale,
        )
        self._parents.moved(self._keys[chosen])
        value = next(self._reads[chosen])
        tokens = self._tokens(value)
        if tokens <= 0:
            raise RuntimeError(
                "token deficit mix received a window without valid targets; filter it before mixing"
            )
        self._delivered[chosen] += tokens
        return value

    def get_state(self) -> Mapping[str, object]:
        return {"parents": self._parents.captured(),
                "delivered": dict(zip(self._keys, self._delivered, strict=True))}

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
        self._parents.restored()
        for key, read in zip(self._keys, self._reads, strict=True):
            if key in parents:
                read.set_state(record(parents[key], "corpus iterator"))
        restored = [delivered[key] for key in self._keys if key in parents]
        total, weight = sum(restored), sum(w for w, held in zip(self._weights, kept, strict=True) if held)
        self._delivered = [delivered[key] if held else (w * total // weight if weight else 0)
                           for key, w, held in zip(self._keys, self._weights, kept, strict=True)]

    def close(self) -> None:
        for read in self._reads:
            read.close()


@dataclasses.dataclass(frozen=True)
class Proportions:
    """A weighted mix's selection policy, Grain's (`MapDataset.mix`): the weights scaled so the least
    is 100 and truncated to whole proportions, and position k of the mix read from the corpus whose
    count of records a prefix holds grows from k to k + 1. That schedule repeats every window of the
    mix, its proportions' sum of positions, in which each corpus reads its proportion; a `seed`
    permutes the corpus choices within each window by a permutation of its own (`window`), so the
    mix reads its corpora in no fixed order and every window still holds each at its share. A phase
    counts its prefix by it (`counts`) as the mix reads by it (`ProportionalSelection`), so the two
    never disagree."""

    proportions: tuple[int, ...]
    seed: int | None = None

    @classmethod
    def of(cls, weights: Sequence[float], seed: int | None = None) -> Proportions:
        scale = 100 / min(weights)
        return cls(tuple(int(weight * scale) for weight in weights), seed)

    def window(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """The `index`th window of the mix: the corpus each of its positions reads, and how many of that
        corpus's records the window read before it. Unseeded, every window is Grain's schedule; seeded,
        window `index` is it permuted by `np.random.default_rng([seed, index])`, as a shard order's
        epochs are (`dew.data.ordering`)."""
        return _window(self.proportions, self.seed, index)

    def counts(self, records: int) -> tuple[int, ...]:
        """How many records of each corpus the first `records` positions of the mix read."""
        if self.seed is not None:
            windows, offset = divmod(records, sum(self.proportions))
            read = np.bincount(self.window(windows)[0][:offset], minlength=len(self.proportions))
            return tuple(windows * proportion + int(count)
                         for proportion, count in zip(self.proportions, read, strict=True))
        counts, remaining, left = [], sum(self.proportions), records
        for proportion in self.proportions:
            rest = left * (remaining - proportion) // remaining
            counts.append(left - rest)
            left, remaining = rest, remaining - proportion
        return tuple(counts)

    def covering(self, sizes: Sequence[int]) -> int:
        """The fewest positions of the mix in which every corpus reads `sizes` of its records, its
        size: the pass in which each repeated corpus has come round once."""
        def covers(positions: int) -> bool:
            return all(count >= size for count, size in zip(self.counts(positions), sizes, strict=True))

        high = 1
        while not covers(high):
            high *= 2
        low = high // 2
        while low < high:
            middle = (low + high) // 2
            low, high = (middle + 1, high) if not covers(middle) else (low, middle)
        return low

    def at(self, position: int) -> tuple[int, int]:
        """The corpus position `position` of the mix reads, the one whose count grows with it, and
        which of that corpus's records: the next it hasn't read."""
        if self.seed is not None:
            windows, offset = divmod(position, sum(self.proportions))
            corpora, before = self.window(windows)
            corpus = int(corpora[offset])
            return corpus, windows * self.proportions[corpus] + int(before[offset])
        before, after = self.counts(position), self.counts(position + 1)
        grown = (corpus for corpus, (was, now) in enumerate(zip(before, after, strict=True)) if now != was)
        corpus = next(grown)
        return corpus, before[corpus]

    def run(self, start: int, positions: int) -> tuple[np.ndarray, np.ndarray] | None:
        """`at` of `positions` positions in a row from `start`, as two arrays, corpus and record:
        `counts`' split taken for every position at once, a corpus at a time, so a run costs what one
        position does in Python and a corpus's arithmetic in its vector. None where a position's
        products would leave 64-bit integers, which `at` reads exactly."""
        remaining = sum(self.proportions)
        if (start + positions + 1) * remaining >= 2**62:
            return None
        if self.seed is not None:
            place = np.arange(start, start + positions, dtype=np.int64)
            windows, offsets = np.divmod(place, remaining)
            corpus, record = np.empty(positions, np.int64), np.empty(positions, np.int64)
            for index in np.unique(windows):
                held = windows == index
                corpora, before = self.window(int(index))
                corpus[held] = corpora[offsets[held]]
                record[held] = (int(index) * np.asarray(self.proportions, np.int64)[corpus[held]]
                                + before[offsets[held]])
            return corpus, record
        left = np.arange(start, start + positions + 1, dtype=np.int64)
        corpus = np.full(positions, -1, np.int64)
        record = np.zeros(positions, np.int64)
        for index, proportion in enumerate(self.proportions):
            rest = left * (remaining - proportion) // remaining
            # The position's corpus is the first whose count grows from it to the next position.
            grows = (corpus < 0) & (np.diff(left - rest) > 0)
            corpus[grows], record[grows] = index, (left - rest)[:-1][grows]
            if (corpus >= 0).all():
                break
            left, remaining = rest, remaining - proportion
        return corpus, record


@functools.lru_cache(maxsize=8)
def _window(proportions: tuple[int, ...], seed: int | None, index: int) -> tuple[np.ndarray, np.ndarray]:
    """`Proportions.window`: the corpus each position of window `index` reads, and how many of its
    corpus's records the window read before it."""
    size = sum(proportions)
    run = Proportions(proportions).run(0, size)
    if run is None:
        raise ValueError(f"a mix of proportions {proportions} counts past 64-bit integers in one window")
    corpora = run[0] if seed is None else run[0][np.random.default_rng([seed, index]).permutation(size)]
    before = np.empty(size, np.int64)
    for corpus in range(len(proportions)):
        places = np.flatnonzero(corpora == corpus)
        before[places] = np.arange(len(places))
    return corpora, before


class ProportionalSelection(pygrain.DatasetSelectionMap):
    """Grain's weighted mix of corpora of `lengths` records as a selection map
    (`MapDataset.select_from_datasets`), by `Proportions`, its corpus choices permuted within each
    window by `seed` where one is given: as long as the mix reads no record of any corpus twice."""

    RUN = 4096
    """The positions read ahead together once a reader reads one after another (`Proportions.run`)."""

    def __init__(self, lengths: Sequence[int], weights: Sequence[float], seed: int | None = None):
        self._proportions = Proportions.of(weights, seed)
        total = sum(self._proportions.proportions)
        spans = [length / (proportion / total)
                 for length, proportion in zip(lengths, self._proportions.proportions, strict=True)]
        if any(span == 0 for span in spans):
            raise ValueError("every corpus of a mix needs a record")
        self._length = min(sys.maxsize, int(min(spans)))
        if seed is not None:
            # A seeded window may draw a corpus's records sooner than its share: the mix ends before
            # the first position that would read past a corpus, the most positions whose counts every
            # corpus holds (`Proportions.counts`, monotone in its positions).
            low, high = 0, min(sys.maxsize, self._length + total)
            while low < high:
                middle = (low + high + 1) // 2
                held = all(count <= length for count, length
                           in zip(self._proportions.counts(middle), lengths, strict=True))
                low, high = (middle, high) if held else (low, middle - 1)
            self._length = low
        self._run: tuple[int, np.ndarray, np.ndarray] | None = None
        self._last = -2
        self._lock = threading.Lock()

    def __getstate__(self) -> dict:
        """What a worker process the selection is sent to reads: no run and no lock, its own."""
        return {"_proportions": self._proportions, "_length": self._length}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state, _run=None, _last=-2, _lock=threading.Lock())

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[int, int]:
        """`Proportions.at` of `index`: read from a run of positions where a reader reads them one
        after another, as a loader does, at a corpus's Python cost a run rather than a position; a
        position reached otherwise (a resume, a seek, a shuffled read) is read alone."""
        with self._lock:
            held = self._run
            if held is not None and held[0] <= index < held[0] + len(held[1]):
                start, corpora, records = held
            elif index == self._last + 1 and (run := self._proportions.run(index, self.RUN)) is not None:
                start, (corpora, records) = index, run
                self._run = start, corpora, records
            else:
                self._last = index
                return self._proportions.at(index)
            self._last = index
            return int(corpora[index - start]), int(records[index - start])

    def __repr__(self) -> str:
        return (f"ProportionalSelection(proportions={self._proportions.proportions}, "
                f"seed={self._proportions.seed}, length={self._length})")


class TokenDeficit(pygrain.IterDataset[Batch]):
    """Pick the corpus furthest below its token share, with index-order ties."""

    def __init__(
        self,
        parents: Sequence[pygrain.IterDataset[Batch]],
        weights: Sequence[float],
        keys: Sequence[str],
        *,
        tokens: Callable[[Batch], int],
        names: Sequence[str],
        labels: Sequence[str],
    ):
        filtered = tuple(
            Accepted(parent, name, label, tokens=tokens)
            for parent, name, label in zip(parents, names, labels, strict=True)
        )
        super().__init__(filtered)
        ratios = tuple(Fraction(weight) for weight in weights)
        scale = math.lcm(*(ratio.denominator for ratio in ratios))
        self._corpora, self._weights, self._tokens, self._keys = (
            filtered,
            tuple(ratio.numerator * (scale // ratio.denominator) for ratio in ratios),
            tokens,
            tuple(keys),
        )

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        group = _MixReads(self._corpora)
        reads = [_MixRead(group, index) for index in range(len(self._corpora))]
        return _DeficitIterator(reads, self._weights, self._tokens, self._keys)


_UNOPENED = "unopened"
"""The key of a mix's corpus's position where the mix has not opened it (`_MixRead`)."""


class _MixReads:
    """Every corpus of a mix, each opened and read only as the mix draws from it: a corpus the mix has
    not drawn holds no iterator, so no reader's threads, buffers or files, and no position but its
    start and the counts it carries there (`state`). A mix captures every corpus's position for each
    record it gives (a packer captures it before each record), and only the corpus it read moved, so
    a corpus's position is captured again only after the mix reads it or it is restored."""

    def __init__(self, corpora: Sequence[pygrain.IterDataset[Batch]]):
        self.corpora = tuple(corpora)
        self.opened: dict[int, pygrain.DatasetIterator[Batch]] = {}
        # Each opened corpus's captured position, until a read or a restore moves it.
        self.states: dict[int, Mapping[str, object]] = {}
        # A position whose counters a corpus back at its start, unopened, takes on as it opens, every
        # counter its stages keep (`rewound`): a pass a repeat reads again goes on counting each corpus.
        self.counted: dict[int, Mapping[str, object]] = {}

    def parent(self, index: int) -> pygrain.DatasetIterator[Batch]:
        """Corpus `index`'s iterator, opened as the mix first draws from or restores it. It runs
        under its own Grain options: the only ones Dew sets above a mix are its loading workers'
        shared-memory floor (`min_shm_size`), which the worker boundary reads, not a corpus."""
        if index not in self.opened:
            opened = iter(self.corpora[index])
            if index in self.counted:
                opened.set_state(rewound(opened.get_state(), self.counted.pop(index)))
            self.opened[index] = opened
        return self.opened[index]

    def read(self, index: int) -> Batch:
        self.states.pop(index, None)
        return next(self.parent(index))

    def state(self, index: int) -> Mapping[str, object] | None:
        """The position of corpus `index`, None where it is unopened (`counts`)."""
        if index not in self.opened:
            return None
        if index not in self.states:
            self.states[index] = self.opened[index].get_state()
        return self.states[index]

    def counts(self, index: int) -> Mapping[str, object] | None:
        """The position whose counters corpus `index`, unopened, carries (None before it counted)."""
        return self.counted.get(index)

    def restore(self, index: int, state: Mapping[str, object]) -> None:
        """Corpus `index` at `state`."""
        self.states.pop(index, None)
        self.parent(index).set_state(state)

    def started(self, index: int, counted: Mapping[str, object] | None) -> None:
        """Corpus `index` back at its start, unopened, carrying the counters of `counted`, a position
        it held (none where it is None): a repeat's rewind hands it the corpus's live position to go on
        counting from (`rewound`), a restore the counts it saved."""
        self.states.pop(index, None)
        self.close(index)
        if counted is not None:
            self.counted[index] = counted

    def close(self, index: int) -> None:
        self.counted.pop(index, None)
        opened = self.opened.pop(index, None)
        if opened is not None:
            opened.close()


class _MixRead(pygrain.DatasetIterator[Batch]):
    """One corpus of a mix (`_MixReads`): its parent's position, or, unopened, the counters it carries
    (`_UNOPENED`). A position a mix saved while it read each corpus's first row ahead holds the
    corpus before that row, which a restore reads next; one that held a corpus's counts alone in its
    parent's place is a corpus at its start carrying them."""

    def __init__(self, group: _MixReads, index: int):
        super().__init__()
        self._group, self._index = group, index

    def __next__(self) -> Batch:
        return self._group.read(self._index)

    def get_state(self) -> Mapping[str, object]:
        state = self._group.state(self._index)
        if state is None:
            return {_UNOPENED: self._group.counts(self._index)}
        return {"parent": state}

    def set_state(self, state: Mapping[str, object]) -> None:
        if _UNOPENED in state:
            counted = state[_UNOPENED]
            self._group.started(self._index, None if counted is None else record(counted, "mixed corpus"))
            return
        parent = record(state["parent"], "mixed corpus")
        if set(parent) <= {COUNTER}:
            self._group.started(self._index, parent if parent.get(COUNTER) is not None else None)
        else:
            self._group.restore(self._index, parent)

    def close(self) -> None:
        self._group.close(self._index)


class _ReadDataset(pygrain.IterDataset[Batch]):
    def __init__(self, parent: pygrain.IterDataset[Batch], group: _MixReads, index: int):
        super().__init__(parent)
        self._group, self._index = group, index

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        return _MixRead(self._group, self._index)


class KeyedMix(pygrain.IterDataset[Batch]):
    """Grain's weighted mix of corpora, each read as it is drawn (`_MixReads`), and
    each keyed by its member's key (`keys`): Grain saves a keyed mix's positions by key and
    restores those it finds, so a member that joins starts at its first row and one that left is
    dropped while the others resume where they were. With a `seed`, the corpus each position reads is
    `Proportions(seed=...)`'s, its windows' choices permuted (`_SeededMix`), its position saved alike."""

    def __init__(self, parents: Sequence[pygrain.IterDataset[Batch]], weights: Sequence[float],
                 keys: Sequence[str], seed: int | None = None):
        super().__init__(parents)
        self._corpora, self._weights, self._keys = tuple(parents), tuple(weights), tuple(keys)
        self._seed = seed

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        group = _MixReads(self._corpora)
        if self._seed is not None:
            reads = {key: _MixRead(group, index) for index, key in enumerate(self._keys)}
            return _SeededMix(reads, Proportions.of(self._weights, self._seed))
        parents = {key: _ReadDataset(parent, group, index)
                   for index, (key, parent) in enumerate(zip(self._keys, self._corpora, strict=True))}
        weights = dict(zip(self._keys, self._weights, strict=True))
        return iter(pygrain.IterDataset.mix(parents, weights))


class _Captured:
    """A mix's corpora's positions by key, captured before each record the mix gives: the corpus it read
    since the last capture captured again, every other's shared with it (`Shared`), so a capture costs
    a record the corpus it read, not every corpus (row 4079)."""

    def __init__(self, reads: Mapping[str, pygrain.DatasetIterator[Batch]]):
        self._reads = reads
        self._held: Shared[Mapping[str, object]] | None = None
        self._moved: set[str] = set(reads)

    def moved(self, key: str) -> None:
        """Corpus `key` read, its position captured again at the next capture."""
        self._moved.add(key)

    def restored(self) -> None:
        """Every corpus restored, each captured again at the next capture."""
        self._moved = set(self._reads)

    def captured(self) -> Shared[Mapping[str, object]]:
        changed = {key: self._reads[key].get_state() for key in self._moved}
        self._held = Shared.of(changed) if self._held is None else self._held.replaced(changed)
        self._moved.clear()
        return self._held


class _SeededMix(pygrain.DatasetIterator[Batch]):
    """A keyed mix's reads (`KeyedMix`) in the corpus order a seeded `Proportions` gives each position,
    its position saved as Grain saves its own mix's: each corpus's under its key, the positions read
    and whether a corpus ended the mix, so a restore finds the corpora it holds and starts the rest."""

    def __init__(self, reads: Mapping[str, _MixRead], proportions: Proportions):
        super().__init__()
        self._reads, self._keys, self._proportions = dict(reads), tuple(reads), proportions
        self._index, self._stop = 0, False
        self._parents = _Captured(self._reads)

    def __next__(self) -> Batch:
        if self._stop:
            raise StopIteration
        corpus, _ = self._proportions.at(self._index)
        self._index += 1
        key = self._keys[corpus]
        self._parents.moved(key)
        try:
            return next(self._reads[key])
        except StopIteration:
            self._stop = True
            raise

    def get_state(self) -> Mapping[str, object]:
        return {"parents": self._parents.captured(), "index": self._index, "stop": self._stop}

    def set_state(self, state: Mapping[str, object]) -> None:
        parents = record(state["parents"], "seeded mix")
        self._parents.restored()
        for key, read in self._reads.items():
            if (held := parents.get(key)) is not None:
                read.set_state(record(held, "mixed corpus"))
        self._index = integer(state["index"], "seeded mix index")
        self._stop = boolean(state["stop"], "seeded mix stop")

    def close(self) -> None:
        for read in self._reads.values():
            read.close()


@runtime_checkable
class _NamedTuple(Protocol):
    """A named tuple, which its documented `_make` rebuilds from its fields' values."""

    _fields: tuple[str, ...]

    def _make(self, iterable: Sequence[object]) -> Sequence[object]: ...


def rewound(start: Mapping[str, object], current: Mapping[str, object]) -> Mapping[str, object]:
    """`start`, the position a repeat rewinds its pass to, with each counter's counts as `current`
    holds them (`COUNTER`: its counts, run of rejections and warning): a pass read again goes on
    counting the corpus, where a restored position replaces its counts. Every other part of a
    position is the iterator's own and is left as `start` has it. A mix's corpus the pass had not
    opened at its start (`_MixRead`'s `_UNOPENED`) is rewound to that, carrying the position it holds
    now, open or not, whose counters, every one its stages keep, it takes on as it opens again."""
    if _UNOPENED in start:
        return {_UNOPENED: current.get("parent", current.get(_UNOPENED))}
    shared = [key for key in start if key in current]
    parts = _rewound_parts([start[key] for key in shared], [current[key] for key in shared])
    kept = {**start, **dict(zip(shared, parts, strict=True))}
    if COUNTER in start and COUNTER in current:
        kept[COUNTER] = current[COUNTER]
    return kept


def _rewound_parts(start: Sequence[object], current: Sequence[object]) -> Sequence[object]:
    """Each part of a position rewound against its place in `current` (`rewound`): the mappings and
    the lists and tuples of one length are walked, rebuilt as what they were, and anything else
    is the iterator's own and kept."""
    parts: list[object] = []
    for value, now in zip(start, current, strict=True):
        if isinstance(value, Mapping) and isinstance(now, Mapping):
            parts.append(rewound(value, now))
        elif isinstance(value, (list, tuple)) and isinstance(now, (list, tuple)) and len(value) == len(now):
            inner = _rewound_parts(value, now)
            parts.append(list(inner) if isinstance(value, list)
                         else value._make(inner) if isinstance(value, _NamedTuple) else tuple(inner))
        else:
            parts.append(value)
    return parts


def state_counts(state: Mapping[str, object]) -> dict[str, CorpusCounts]:
    """Read per-corpus counters through Grain's nested iterator checkpoints.

    Within one iterator a corpus's counters are its stages': the outermost
    counts the rows it delivered and those of them cut short, and each its
    own rejections. The iterators Grain interleaves, its worker processes or a
    corpus's shards, each read records of their own, so their counts add,
    a shard waiting past the cycle (a future state) included."""

    def visit(branch) -> dict[str, CorpusCounts]:
        counted: dict[str, CorpusCounts] = {}
        rowless: CorpusCounts | None = None
        if isinstance(branch, Mapping):
            node = branch.get(COUNTER)
            if isinstance(node, Mapping):
                corpus = text(node["corpus"], "corpus")
                counts = CorpusCounts(corpus, text(node["label"], "label"),
                                      rows=boolean(node.get("rows", True), "counter rows"))
                counts.restore(record(node["counts"], "corpus counts"))
                if counts.rows:
                    counted[corpus] = counts
                else:
                    rowless = counts
                # A mixture's counter past the mix holds its members' counts beside its own (a chunk
                # counter holds none).
                for member in record(node.get("members", {}), "a mixture counter's members").values():
                    saved = record(member, "member counter")
                    held = CorpusCounts(text(saved["corpus"], "member corpus"),
                                        text(saved["label"], "member label"), examples=True)
                    held.restore(record(saved["counts"], "member counts"))
                    _nested(counted, {held.name: held})
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
                if name not in ("iterators_in_use_states", "future_states", COUNTER):
                    _nested(counted, visit(child))
        elif isinstance(branch, (tuple, list)):
            for child in branch:
                _nested(counted, visit(child))
        if rowless is not None:
            _past(counted, rowless)
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
        # Members a row left out travel with the row, as its cuts do: the outermost counts them.
        combined.rejected.update(counts.rejected)
        combined.rejected_cut.update(counts.rejected_cut)


def _past(counted: dict[str, CorpusCounts], outer: CorpusCounts) -> None:
    """Fold a counter past an objective's step that makes rows of its corpus's source examples (`rows`
    false) over the counters inside it: they counted the examples, it the rows made of them, which it
    delivered, cut short and cut, and the rejections of both."""
    inner = counted.get(outer.name)
    counted[outer.name] = outer if inner is None else dataclasses.replace(
        outer, rows=inner.rows, examples=inner.examples, accepted=inner.accepted,
        attempted=max(inner.attempted, outer.attempted), chunks=max(inner.chunks, outer.chunks),
        chunk_records=max(inner.chunk_records, outer.chunk_records), rejected=inner.rejected + outer.rejected,
        rejected_cut=inner.rejected_cut + outer.rejected_cut)


def _added(counted: dict[str, CorpusCounts], counts: CorpusCounts) -> None:
    """Add a corpus's counts from an iterator that read records of its own."""
    if counts.name not in counted:
        counted[counts.name] = CorpusCounts(counts.name, counts.label, rows=counts.rows)
    combined = counted[counts.name]
    combined.attempted += counts.attempted
    combined.accepted += counts.accepted
    combined.delivered_rows += counts.delivered_rows
    combined.truncated += counts.truncated
    combined.chunks += counts.chunks
    combined.chunk_records += counts.chunk_records
    combined.rejected.update(counts.rejected)
    combined.cut_examples.update(counts.cut_examples)
    combined.cut_rows.update(counts.cut_rows)
    combined.cut_removed.update(counts.cut_removed)
    combined.left_out.update(counts.left_out)
    combined.rejected_cut.update(counts.rejected_cut)
    combined.left_out_cut.update(counts.left_out_cut)
    combined.declined.update(counts.declined)
