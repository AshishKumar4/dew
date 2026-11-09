"""Preference pairs for DPO: chosen and rejected completions packed into windows, a shared prompt once.

A `PreferencePair` step reads records of `chosen` and `rejected` token ids, lists or one-dimensional
arrays, with `chosen_mask` and `rejected_mask` marking the completion tokens, and lays each pair out as
one document: with `share`, the prompt both halves begin with (their longest common prefix before either's
completion) once, then each half's rest continuing it (`dew.batch.Forks`), each token at its own half's
place; without, the two halves whole. `preferences` (`dew.data.presets`) packs the documents first-fit
into windows of two halves' tokens, several pairs a window (`PreferenceWindows`), and the DPO objective
reduces each pair's halves apart. A mask must run as long as its ids; anything else fails naming the row.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import grain.python as grain
import numpy as np

from dew._records import record
from dew.batch import UNIT_ROWS, token_ids
from dew.data.facts import shaped, unshaped
from dew.data.forked_rows import window_forks
from dew.data.pipeline import PACK_BINS
from dew.data.tokens import first_fit

if TYPE_CHECKING:
    from dew.batch import Batch

IDS_KEY = "input_ids"
"""Batch key holding each window's `[B, W]` ids, its pairs' documents end to end."""

MASK_KEY = "completion_mask"
"""Batch key holding the `[B, W]` completion marks, 1 on completion tokens."""

REAL_KEY = "real_mask"
"""Batch key holding the `[B, W]` marks of each window's ids, 1 on its pairs' tokens and 0 on padding."""

PAIRS_KEY = "pair_ids"
"""Batch key holding each token's pair `[B, W]`, from 1, 0 on padding: a window's own numbering as the
packer writes it, every pair's own across its batch once `PairIds` numbers them, which the DPO loss reads
(`dew.objectives.rl.preference.pair_numbers`)."""

CONTINUED_KEY = "continued"
"""Batch key holding whether each window `[B]` continues the pair the window before it in the stream
opened: a window of one half's (`segments=1`) holding a rejected half, which `PairIds` reads."""

HALVES_KEY = "halves"
"""Batch key holding each token's half `[B, W]`: 1 the chosen's, 2 the rejected's, 0 a prompt both share
and padding."""

POSITIONS_KEY = "pair_positions"
"""Batch key holding each token's place in its own half `[B, W]`."""

DOCUMENTS_KEY = "fork_documents"
"""Batch key holding each token's document in its window `[B, W]` (`dew.batch.Forks`), from 1, 0 on
padding."""

PARENTS_KEY = "fork_parents"
"""Batch key holding the document each of a window's documents continues `[B, D]`, 0 for none."""

FIELDS = ("chosen", "rejected", "chosen_mask", "rejected_mask")

TOKEN_FIELDS = ("text", "pair_documents", "pair_parents", MASK_KEY, HALVES_KEY, POSITIONS_KEY)
"""A pair document's per-token fields, which a window packs: its ids, each token's document in the pair
and that document's parent there, its completion marks, its half and its place in its half."""


def window_width(seq_len: int, segments: int | None = None) -> int:
    """The tokens a window holds: both halves of a pair whose halves take `seq_len` each, or one half where
    a window holds one document (`segments=1`, a model that reads one document a row)."""
    return seq_len if segments == 1 else 2 * seq_len


def window_documents(seq_len: int, segments: int | None = None) -> int:
    """The most documents a window holds: `segments` where given, else three a pair (a shared prompt and two
    rests), a pair at least two tokens."""
    return segments if segments is not None else 3 * (window_width(seq_len) // 2)


def _mask(values: object, name: str, length: int, where: str) -> list[int]:
    """One 0/1 mask as long as its ids, a list, a tuple or a one-dimensional array as its ids are."""
    marks = values.tolist() if isinstance(values, np.ndarray) and values.ndim == 1 else values
    if not isinstance(marks, (list, tuple)) or len(marks) != length:
        raise ValueError(
            f"{where}: {name} runs as long as its ids ({length}), "
            f"got {values!r}")
    mask: list[int] = []
    for mark in marks:
        if mark not in (0, 1):
            raise ValueError(f"{where}: {name} marks completion tokens 0/1, got {mark!r}")
        mask.append(int(mark))
    return mask


def pair_document(chosen: Sequence[int], rejected: Sequence[int], chosen_mask: Sequence[int],
                  rejected_mask: Sequence[int], *, share: bool) -> dict[str, np.ndarray]:
    """One pair as the document a window packs (`TOKEN_FIELDS`): with `share`, the halves' common prefix
    before either's completion once (document 1), then each half's rest (documents 2 and 3) continuing
    it; without, the chosen half (1) and the rejected half (2) whole. Each token keeps its half's place."""
    prefix = 0
    if share:
        for one, other, mark, again in zip(chosen, rejected, chosen_mask, rejected_mask, strict=False):
            if one != other or mark or again:
                break
            prefix += 1
    halves = ((1, chosen, chosen_mask), (2, rejected, rejected_mask))
    pieces = []
    if prefix:
        pieces.append((list(chosen[:prefix]), 1, 0, [0] * prefix, 0, range(prefix)))
    for half, ids, mask in halves:
        number = len(pieces) + 1
        pieces.append((list(ids[prefix:]), number, 1 if prefix else 0, list(mask[prefix:]), half,
                       range(prefix, len(ids))))
    return {
        "text": np.asarray([token for ids, *_ in pieces for token in ids], np.int32),
        "pair_documents": np.concatenate([np.full(len(ids), number, np.int32)
                                          for ids, number, *_ in pieces]),
        "pair_parents": np.concatenate([np.full(len(ids), parent, np.int32)
                                        for ids, _, parent, *_ in pieces]),
        MASK_KEY: np.asarray([mark for _, _, _, marks, _, _ in pieces for mark in marks], np.int32),
        HALVES_KEY: np.concatenate([np.full(len(ids), half, np.int32) for ids, _, _, _, half, _ in pieces]),
        POSITIONS_KEY: np.asarray([place for *_, places in pieces for place in places], np.int32),
    }


@dataclasses.dataclass(frozen=True)
class PreferencePair(grain.MapTransform):
    """A chosen and a rejected completion's ids and loss masks laid out as one document a window packs
    (`pair_document`), each half at most `seq_len` ids, which the facts state (`seq_len`, a run's DPO row
    width); with `share`, the prompt they begin with once."""

    seq_len: int = 256
    _: dataclasses.KW_ONLY
    share: bool = True

    def __post_init__(self) -> None:
        if self.seq_len < 1:
            raise ValueError("preference seq_len must be positive")

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """A pair's document in place of its completions, one entry a token of its fields."""
        rows = unshaped(facts, *FIELDS)
        for key in TOKEN_FIELDS:
            rows = shaped(rows, key, [None])
        return {**rows, "seq_len": self.seq_len}

    def map(self, element: Batch) -> Batch:
        element = record(element, "preference row")
        if "chosen" not in element or "rejected" not in element:
            raise ValueError("a preference pair needs both chosen and rejected ids")
        halves = {}
        for name in ("chosen", "rejected"):
            tokens = token_ids(element.get(name), where=f"preference: {name}")
            if len(tokens) > self.seq_len:
                raise ValueError(f"preference {name} has {len(tokens)} tokens for seq_len={self.seq_len}")
            completion = _mask(
                element.get(f"{name}_mask", [1] * len(tokens)), f"{name}_mask", len(tokens), "preference"
            )
            halves[name] = (tokens, completion)
        kept = {key: value for key, value in element.items() if key not in FIELDS}
        (chosen, chosen_mask), (rejected, rejected_mask) = halves["chosen"], halves["rejected"]
        return {**kept, **pair_document(chosen, rejected, chosen_mask, rejected_mask, share=self.share)}


@dataclasses.dataclass(frozen=True)
class PairHalves:
    """A pair's document (`PreferencePair`, its halves whole) as one record a half, in order, the chosen
    first: what a window of one document holds (`segments=1`), a model that reads one document a row."""

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """Halves in place of pairs, their token fields of no fixed length."""
        return facts

    def flat_map(self, element: Batch) -> list[Batch]:
        documents = np.asarray(element["pair_documents"])
        if np.any(np.asarray(element["pair_parents"]) > 0):
            raise ValueError("a pair whose halves share their prompt is one document; lay its halves whole "
                             "(share=False) to read them a row each")
        kept = {key: value for key, value in element.items() if key not in TOKEN_FIELDS}
        return [{**kept, **{name: np.asarray(element[name])[documents == number] for name in TOKEN_FIELDS}}
                for number in (1, 2)]


@dataclasses.dataclass(frozen=True)
class PreferenceWindows(grain.MapTransform):
    """A window of packed pair documents (`PreferencePair`, packed along `TOKEN_FIELDS`) as DPO reads it
    (`window`): `width` ids, its pairs numbered, its documents and, where its pairs share their prompts
    (`share`), the documents they continue, at most `documents` a window."""

    width: int
    documents: int
    share: bool = True

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """A window's keys in place of its packed fields."""
        rows = unshaped(facts, *TOKEN_FIELDS, *(f"{name}_segment_ids" for name in TOKEN_FIELDS),
                        *(f"{name}_positions" for name in TOKEN_FIELDS))
        for key in (IDS_KEY, MASK_KEY, REAL_KEY, PAIRS_KEY, HALVES_KEY, POSITIONS_KEY,
                    *((DOCUMENTS_KEY,) if self.documents > 1 else ())):
            rows = shaped(rows, key, [self.width])
        # A half's longest, the run's DPO row width, as a pair a row states it (`PreferencePair`).
        rows = {**rows, "seq_len": self.width if self.documents == 1 else self.width // 2}
        if self.documents == 1:
            rows = shaped(rows, CONTINUED_KEY, [])
        return shaped(rows, PARENTS_KEY, [self.documents]) if self.share else rows

    def map(self, element: Batch) -> Batch:
        laid = window({name: np.asarray(element[name]) for name in TOKEN_FIELDS},
                      np.asarray(element["text_segment_ids"]), self.documents, share=self.share)
        kept = {key: value for key, value in element.items()
                if key not in TOKEN_FIELDS and not key.endswith(("_segment_ids", "_positions"))}
        return {**kept, **laid}


def window(fields: Mapping[str, np.ndarray], pairs: np.ndarray, documents: int, *,
           share: bool = True) -> dict[str, np.ndarray]:
    """A window's batch keys from its pairs' packed `fields` (`TOKEN_FIELDS`, one entry a slot) and each
    slot's pair (`pairs`, from 1, 0 on padding): the one layout `PreferenceWindows` and `windows` write,
    its documents (`DOCUMENTS_KEY`) only where a window holds several, and their parents (`PARENTS_KEY`)
    only where the pairs share their prompts."""
    pairs = np.asarray(pairs, np.int32)
    real = pairs > 0
    laid, parents = window_forks(pairs, np.where(real, fields["pair_documents"], 0),
                                 np.where(real, fields["pair_parents"], 0), documents)
    halves = np.where(real, fields[HALVES_KEY], 0)
    keys = {IDS_KEY: np.where(real, fields["text"], 0).astype(np.int32),
            # A window whose first half is a rejected one continues the pair the window before it opened.
            CONTINUED_KEY: np.asarray(bool(real[0]) and int(halves[0]) == 2),
            MASK_KEY: np.where(real, fields[MASK_KEY], 0).astype(np.int32),
            REAL_KEY: real.astype(np.int32), PAIRS_KEY: pairs,
            HALVES_KEY: np.where(real, fields[HALVES_KEY], 0).astype(np.int32),
            POSITIONS_KEY: np.where(real, fields[POSITIONS_KEY], 0).astype(np.int32)}
    if documents > 1:
        keys[DOCUMENTS_KEY] = laid
        del keys[CONTINUED_KEY]
    return {**keys, PARENTS_KEY: parents} if share else keys


def windows(pairs: Sequence[Mapping[str, object]], seq_len: int, *, share: bool = True,
            segments: int | None = None, count: int | None = None,
            bins: int = PACK_BINS) -> Batch:
    """Preference records (`PreferencePair`'s) packed first-fit with `bins` open windows, in their order,
    into `count` windows (as many as they need where None) of `window_width(seq_len, segments)` tokens, at
    most `segments` documents a window (`preferences`), stacked as a batch: the windows the `preferences`
    preset packs, built on the host for a caller holding its pairs."""
    share = share and segments != 1
    step = PreferencePair(seq_len, share=share)
    documents = [step.map(dict(pair)) for pair in pairs]
    if segments == 1:
        documents = [half for document in documents for half in PairHalves().flat_map(document)]
    width = window_width(seq_len, segments)
    held_documents = window_documents(seq_len, segments)
    grouped, starts = first_fit(np.asarray([len(document["text"]) for document in documents], np.int64),
                                width, bins, segments or width // 2)
    held = len(starts) - 1 if count is None else count
    if held < len(starts) - 1:
        raise ValueError(f"the pairs need {len(starts) - 1} windows, more than the {count} asked for")
    laid = []
    for at in range(held):
        members = grouped[starts[at]:starts[at + 1]] if at < len(starts) - 1 else []
        fields = {name: np.zeros(width, np.int32) for name in TOKEN_FIELDS}
        pairs_of = np.zeros(width, np.int32)
        slot = 0
        for number, member in enumerate(members, start=1):
            document = documents[member]
            span = slice(slot, slot + len(document["text"]))
            for name in TOKEN_FIELDS:
                fields[name][span] = document[name]
            pairs_of[span] = number
            slot = span.stop
        laid.append(window(fields, pairs_of, held_documents, share=share))
    return PairIds()({key: np.stack([each[key] for each in laid]) for key in laid[0]})


def pair_windows(ids, mask, *, real=None, share: bool = True, segments: int | None = None,
                 count: int | None = None) -> Batch:
    """Pairs laid out a pair a row, `ids` and `mask` `[B, 2, S]` (chosen at index 0, each half's real ids
    those `real` marks, every one where None), as the windows `windows` packs of them: the one-pair-a-row
    form a caller holds, laid out as a DPO run's batches are."""
    ids, mask = np.asarray(ids), np.asarray(mask)
    held = np.ones(ids.shape, bool) if real is None else np.asarray(real, bool)
    pairs = [{name: values[pair, half][held[pair, half]].tolist()
              for half, side in enumerate(("chosen", "rejected"))
              for name, values in ((side, ids), (f"{side}_mask", mask))}
             for pair in range(ids.shape[0])]
    return windows(pairs, ids.shape[2], share=share, segments=segments, count=count)


@dataclasses.dataclass(frozen=True)
class PairIds:
    """A batch of windows (`PreferenceWindows`, rows in the stream's order) with each pair numbered across
    the batch (`PAIRS_KEY`, from 1 in the batch's order) and each window's unit (`UNIT_ROWS`): a window
    of one half (`CONTINUED_KEY`) is its pair's, so a step reads a pair's two windows in one of its parts
    however it cuts its rows (`dew.training.planning.unit_order`). The DPO objective's batch step
    (`DPOObjective.data`); numbering a numbered batch again leaves it as it is."""

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """The batch's fields with each window's unit beside them."""
        return shaped(facts, UNIT_ROWS, [])

    def __call__(self, batch: Batch) -> Batch:
        pairs = np.asarray(batch[PAIRS_KEY], np.int64)
        rows = pairs.shape[0]
        continued = (np.zeros(rows, bool) if CONTINUED_KEY not in batch
                     else np.asarray(batch[CONTINUED_KEY], bool).reshape(rows))
        if rows and continued[0]:
            raise ValueError("a batch opens on the second window of a pair, whose first the batch before it "
                             "holds: a batch of windows of one half holds an even count of them")
        numbered = np.zeros_like(pairs)
        units = np.zeros(rows, np.int32)
        opened = 0
        for row in range(rows):
            # A window's own pairs in their order, its first the one before it where it continues it.
            local = list(dict.fromkeys(pairs[row][pairs[row] > 0].tolist()))
            first = opened if continued[row] else opened + 1
            mapping = {pair: first + at for at, pair in enumerate(local)}
            numbered[row] = [mapping.get(pair, 0) for pair in pairs[row].tolist()]
            opened = max(opened, first + len(local) - 1)
            units[row] = first if local else opened + rows + row
        return {**batch, PAIRS_KEY: numbered.astype(np.int32), UNIT_ROWS: units}
