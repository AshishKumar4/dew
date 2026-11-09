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
from dew.batch import token_ids
from dew.data.facts import shaped, unshaped
from dew.data.forked_rows import window_forks
from dew.data.packing import first_fit_windows
from dew.data.pipeline import PACK_BINS, Pack

if TYPE_CHECKING:
    from dew.batch import Batch

IDS_KEY = "input_ids"
"""Batch key holding each window's `[B, W]` ids, its pairs' documents end to end."""

MASK_KEY = "completion_mask"
"""Batch key holding the `[B, W]` completion marks, 1 on completion tokens."""

REAL_KEY = "real_mask"
"""Batch key holding the `[B, W]` marks of each window's ids, 1 on its pairs' tokens and 0 on padding."""

PAIRS_KEY = "pair_ids"
"""Batch key holding each token's pair `[B, W]`, from 1 in the window's order, 0 on padding: a window's own
numbering as the packer writes it, which the DPO loss numbers across the windows a step reads
(`dew.objectives.rl.preference.pair_numbers`)."""

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


def window_shape(seq_len: int, *, pairs: int = 1, segments: int | None = None) -> tuple[int, int]:
    """A window's inputs and its finals, the tail of its halves' final ids (`dew.batch.Coordinates.finals`),
    for `pairs` pairs a window whose halves take at most `seq_len` ids: room for the pairs' ids, less the
    two finals the fewest pairs hold (one pair's two halves of `seq_len - 1` inputs each, a pair a window),
    and two finals a pair; with `segments=1` (a model that reads one document a row), one half's, its ids
    shifting by one (`PairWindows`)."""
    if segments == 1:
        return seq_len - 1, 1
    return 2 * pairs * seq_len - 2, 2 * pairs


def window_documents(*, pairs: int = 1, segments: int | None = None) -> int:
    """The most documents a window holds: one with `segments=1`, else three a pair (a shared prompt and
    each half's rest)."""
    return 1 if segments == 1 else 3 * pairs


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
    before either's completion once (document 1), then each half's rest (documents 2 and 3) continuing it,
    each rest at least its half's final id; without, the chosen half (1) and the rejected half (2) whole.
    Each token keeps its half's place."""
    if not chosen or not rejected:
        raise ValueError("a preference pair's halves each hold at least one id")
    prefix = 0
    if share:
        for one, other, mark, again in zip(chosen[:-1], rejected[:-1], chosen_mask, rejected_mask,
                                           strict=False):
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
class PairWindows(grain.MapTransform):
    """A pair's document (`PreferencePair`, its halves whole) as its two halves each in a window of `width`
    of its own, the chosen first, on an axis of the pair's (`[2, width]` each key, `window`): what a model
    that reads one document a row takes (`segments=1`), every row dealer keeping a pair whole."""

    width: int

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """A pair's two windows' keys in place of its document."""
        rows = unshaped(facts, *TOKEN_FIELDS)
        for key in (IDS_KEY, MASK_KEY, REAL_KEY, PAIRS_KEY, HALVES_KEY, POSITIONS_KEY):
            rows = shaped(rows, key, [2, self.width])
        return {**rows, "seq_len": self.width}

    def map(self, element: Batch) -> Batch:
        if np.any(np.asarray(element["pair_parents"]) > 0):
            raise ValueError("a pair whose halves share their prompt is one document; lay its halves whole "
                             "(share=False) to read them a row each")
        whole = {name: np.asarray(element[name]) for name in TOKEN_FIELDS}
        documents = whole["pair_documents"]
        halves = []
        for number in (1, 2):
            taken = documents == number
            size = int(taken.sum())
            if size > self.width:
                raise ValueError(f"a half of {size} tokens past its window of {self.width}")
            fields = {name: np.zeros(self.width, values.dtype) for name, values in whole.items()}
            for name, values in whole.items():
                fields[name][:size] = values[taken]
            halves.append(window(fields, (np.arange(self.width) < size).astype(np.int32), 1, share=False))
        kept = {key: value for key, value in element.items() if key not in TOKEN_FIELDS}
        return {**kept, **{key: np.stack([half[key] for half in halves]) for key in halves[0]}}


@dataclasses.dataclass(frozen=True)
class PreferenceWindows(grain.MapTransform):
    """A window of packed pair documents (`PreferencePair`, packed along `TOKEN_FIELDS` into `inputs + 2`
    slots, `pack_of`) as DPO reads it (`window`): each half's final id moved to a tail of `finals`
    (`dew.batch.Coordinates.finals`), the rest the inputs in their order, its pairs numbered, its documents
    and, where its pairs share their prompts (`share`), the documents they continue, at most `documents` a
    window."""

    inputs: int
    finals: int
    documents: int
    share: bool = True

    @property
    def width(self) -> int:
        """The window's ids: its inputs and its finals."""
        return self.inputs + self.finals

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """A window's keys in place of its packed fields."""
        rows = unshaped(facts, *TOKEN_FIELDS, *(f"{name}_segment_ids" for name in TOKEN_FIELDS),
                        *(f"{name}_positions" for name in TOKEN_FIELDS))
        for key in (IDS_KEY, MASK_KEY, REAL_KEY, PAIRS_KEY, HALVES_KEY, POSITIONS_KEY,
                    *((DOCUMENTS_KEY,) if self.documents > 1 else ())):
            rows = shaped(rows, key, [self.width])
        # A half's longest, the run's DPO row width, as a pair a row states it (`PreferencePair`).
        rows = {**rows, "seq_len": (self.inputs + 2) // self.finals}
        return shaped(rows, PARENTS_KEY, [self.documents]) if self.share else rows

    def map(self, element: Batch) -> Batch:
        packed = {name: np.asarray(element[name]) for name in TOKEN_FIELDS}
        pairs = np.asarray(element["text_segment_ids"])
        documents = np.where(pairs > 0, packed["pair_documents"], 0)
        parents = np.where(pairs > 0, packed["pair_parents"], 0)
        # Each pair's leaf documents, those no other continues, and the last slot of each: its final.
        width = int(documents.max(initial=0)) + 1
        keys = pairs * width + documents
        continued = set((pairs * width + parents)[(pairs > 0) & (parents > 0)].tolist())
        last = len(keys) - 1 - np.unique(keys[::-1], return_index=True)[1]
        finals = np.sort([slot for slot in last if pairs[slot] > 0 and int(keys[slot]) not in continued])
        if len(finals) > self.finals:
            raise ValueError(f"a window holds {len(finals)} halves' finals, past its {self.finals}")
        body = np.ones(len(keys), bool)
        body[finals] = False
        body &= pairs > 0
        order = np.concatenate([np.flatnonzero(body), finals])
        spans = (int(body.sum()), len(finals))
        if spans[0] > self.inputs:
            raise ValueError(f"a window's inputs hold {spans[0]} ids, past its {self.inputs}")
        slots = np.concatenate([np.arange(spans[0]), self.inputs + np.arange(spans[1])])
        fields = {name: np.zeros(self.width, values.dtype) for name, values in packed.items()}
        held = np.zeros(self.width, np.int32)
        for name, values in packed.items():
            fields[name][slots] = values[order]
        held[slots] = pairs[order]
        laid = window(fields, held, self.documents, share=self.share)
        kept = {key: value for key, value in element.items()
                if key not in TOKEN_FIELDS and not key.endswith(("_segment_ids", "_positions"))}
        return {**kept, **laid}


def pack_of(inputs: int, finals: int, *, bins: int = PACK_BINS) -> Pack:
    """The pack that lays pair documents into windows of `inputs` and `finals` (`PreferenceWindows`):
    first-fit along their ids into `inputs + 2`, a pair's two finals beyond its inputs, at most a pair a
    final's two slots."""
    return Pack(inputs + 2, TOKEN_FIELDS, bins=bins, policy="first-fit", segments=finals // 2)


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
    keys = {IDS_KEY: np.where(real, fields["text"], 0).astype(np.int32),
            MASK_KEY: np.where(real, fields[MASK_KEY], 0).astype(np.int32),
            REAL_KEY: real.astype(np.int32), PAIRS_KEY: pairs,
            HALVES_KEY: np.where(real, fields[HALVES_KEY], 0).astype(np.int32),
            POSITIONS_KEY: np.where(real, fields[POSITIONS_KEY], 0).astype(np.int32)}
    if documents > 1:
        keys[DOCUMENTS_KEY] = laid
    return {**keys, PARENTS_KEY: parents} if share else keys


def windows(pairs: Sequence[Mapping[str, object]], seq_len: int, *, share: bool = True,
            segments: int | None = None, count: int | None = None, pairs_a_window: int = 1,
            inputs: int | None = None, bins: int = PACK_BINS) -> Batch:
    """Preference records (`PreferencePair`'s, halves of at most `seq_len` ids) laid out as the
    `preferences` preset lays them, run on the host for a caller holding its pairs, stacked as a batch:
    packed first-fit with `bins` open windows, in their order, `pairs_a_window` at most a window, into
    `count` windows (as many as they need where None, the rest empty), each `inputs` wide, or where None
    with room for `pairs_a_window` of its longest pair's ids, its shared prompt once and its halves' rests,
    less the two finals in its tail
    (`window_shape`, the stream's packer, `first_fit_windows`); with `segments=1`, a pair a record, its
    halves each in a window of `seq_len` (`PairWindows`), `count` pairs."""
    share = share and segments != 1
    step = PreferencePair(seq_len, share=share)
    documents = [step.map(dict(pair)) for pair in pairs]
    if segments == 1:
        laid = [PairWindows(seq_len).map(document) for document in documents]
        held = len(laid) if count is None else count
        if held < len(laid):
            raise ValueError(f"{len(laid)} pairs, more than the {count} asked for")
        laid += [{key: np.zeros_like(value) for key, value in laid[0].items()}] * (held - len(laid))
    else:
        _, finals = window_shape(seq_len, pairs=pairs_a_window)
        if inputs is None:
            # Room for a window's pairs of the longest, their prompts once: their ids, less one pair's finals.
            inputs = pairs_a_window * max(len(document["text"]) for document in documents) - 2
        pack = pack_of(inputs, finals, bins=bins)
        packed = [window for window, _ in first_fit_windows([pack.document(each) for each in documents],
                                                            pack)]
        held = len(packed) if count is None else count
        if held < len(packed):
            raise ValueError(f"the pairs need {len(packed)} windows, more than the {count} asked for")
        empty = {name: np.zeros_like(value) for name, value in packed[0].items()}
        laying = PreferenceWindows(inputs, finals, window_documents(pairs=pairs_a_window), share)
        laid = [laying.map(each) for each in packed + [empty] * (held - len(packed))]
    return {key: np.stack([each[key] for each in laid]) for key in laid[0]}


def pair_windows(ids, mask, *, real=None, share: bool = True, segments: int | None = None,
                 count: int | None = None, pairs_a_window: int = 1) -> Batch:
    """Pairs laid out a pair a row, `ids` and `mask` `[B, 2, S]` (chosen at index 0, each half's real ids
    those `real` marks, every one where None), as the windows `windows` packs of them: the one-pair-a-row
    form a caller holds, laid out as a DPO run's batches are."""
    ids, mask = np.asarray(ids), np.asarray(mask)
    held = np.ones(ids.shape, bool) if real is None else np.asarray(real, bool)
    pairs = [{name: values[pair, half][held[pair, half]].tolist()
              for half, side in enumerate(("chosen", "rejected"))
              for name, values in ((side, ids), (f"{side}_mask", mask))}
             for pair in range(ids.shape[0])]
    return windows(pairs, ids.shape[2], share=share, segments=segments, count=count,
                   pairs_a_window=pairs_a_window)
