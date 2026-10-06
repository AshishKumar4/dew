"""How a state and one question become the token row a decision model reads.

`MarkerLayout` is Laya's (laya/common.py `build_sequence`, `build_head` and
`parallel_layout` at NandhaKishorM/laya a4a8921): the question, then one
marker token before each option's text, then the state,

    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]

and the model scores each option at its marker. The option texts share a
token budget, `head_max_len`, and the state takes what is left of `max_len`.
`StateFirstLayout` puts the state first and each marker after its option,
for a causal backbone.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from flax import struct

from dew import records
from dew.data.text import HFTokenizer, Tokenizer
from dew.decision.questions import Choice, Noul, Question, Score
from dew.nn.inputs import pad_token_rows
from dew.records import JSON


@dataclass(frozen=True)
class Specials:
    """The special tokens a layout places.

    They are the row's start token (None for a vocabulary without one), the
    separator, the option marker (and its text, which is removed from anything a
    caller writes), and the padding id.
    """

    begin: int | None
    separator: int
    marker: int
    marker_text: str
    pad: int

    @classmethod
    def of(cls, tokenizer: Tokenizer) -> "Specials":
        """Return a Hugging Face tokenizer's own special tokens.

        That is [CLS] or its start token, [SEP] or its end token, [MASK] (or else its
        end token) as the marker, and its padding token or else its end token.
        """
        if not isinstance(tokenizer, HFTokenizer):
            raise ValueError("Specials.of reads a Hugging Face tokenizer's special tokens; "
                             "give a ByteTokenizer's Specials explicitly")
        hf = tokenizer.tokenizer

        def first(*names: str) -> int | None:
            ids = [getattr(hf, name) for name in names]
            return next((records.integer(value, name) for value, name in zip(ids, names, strict=True)
                         if value is not None), None)

        end = first("sep_token_id", "eos_token_id")
        if end is None:
            raise ValueError("the tokenizer has neither a separator nor an end token to lay rows out with")
        marker = first("mask_token_id") or end
        return cls(begin=first("cls_token_id", "bos_token_id"), separator=end, marker=marker,
                   marker_text=records.text(hf.convert_ids_to_tokens(marker), "marker"),
                   pad=first("pad_token_id") or end)


@dataclass(frozen=True)
class Encoded:
    """One question's row: its tokens and its option markers in slot order.

    In the parallel layout it also holds each token's position and option slot
    (0 for the tokens every option shares).
    """

    tokens: tuple[int, ...]
    markers: tuple[int, ...]
    positions: tuple[int, ...] | None
    slots: tuple[int, ...] | None
    state_tokens: int
    """The state's length in tokens, before the layout cut it to fit."""
    state_kept: int


def render(value: JSON) -> str:
    """Return a value as the model reads it: text as it is, and anything structured as JSON."""
    match value:
        case str():
            return value
        case _:
            return json.dumps(value, ensure_ascii=False)


@dataclass(frozen=True)
class Layout:
    """How one question and its state become a row, scored at one marker per option.

    Each option is rendered as text (`key: description` for a choice,
    `level i: description` for a score, `false: ...` and `true: ...` for a noul)
    and cut to `option_tokens`. When the options exceed `head_max_len` minus 16
    tokens, every option is cut evenly to fit, and the instructions get what the
    options leave, at least 8 tokens. The state fills the rest of `max_len`; a
    list state (a conversation) keeps its end, and any other state keeps its
    start. A marker past `max_len` is dropped.

    `parallel` starts every option at the same position and lets no option attend
    to another, so the options' order does not affect their scores; whatever
    follows the options continues after the longest one.
    """

    max_len: int = 512
    head_max_len: int = 192
    option_tokens: int = 48
    parallel: bool = False

    def state(self, encoder: Tokenizer, specials: Specials, state: JSON) -> tuple[int, ...]:
        """Return the state's tokens, which every question of a request shares."""
        text = render(state).replace(specials.marker_text, " ")
        return tuple(encoder.encode(text, add_special_tokens=False))

    def encode(self, encoder: Tokenizer, specials: Specials, question: Question,
               state: Sequence[int], *, conversation: bool = False,
               order: Sequence[int] | None = None) -> Encoded:
        """Lay out `question` over the state's tokens `state`, with its options in `order`.

        Slot s shows option order[s]; None keeps the question's order. A
        `conversation` keeps the end of the state instead of its start.
        """
        texts = self.options(question)
        slots = list(range(len(texts))) if order is None else list(order)
        if sorted(slots) != list(range(len(texts))):
            raise ValueError(f"an option order is a permutation of the {len(texts)} options, got {slots}")
        instructions = render(question.instructions).replace(specials.marker_text, " ")
        head = encoder.encode(f"{question.kind} question: {instructions}", add_special_tokens=False)
        options = [encoder.encode(" " + texts[slot].replace(specials.marker_text, " "),
                                  add_special_tokens=False)[:self.option_tokens] for slot in slots]
        # Each option's budget counts its marker.
        budget = self.head_max_len - sum(len(option) + 1 for option in options)
        if budget < 16:
            each = max(4, (self.head_max_len - 16) // max(1, len(options)))
            options = [option[:each - 1] for option in options]
            budget = self.head_max_len - sum(len(option) + 1 for option in options)
        return self._row(specials, head[:max(8, budget)], options, state, conversation)

    def _row(self, specials: Specials, head: list[int], options: list[list[int]], state: Sequence[int],
             conversation: bool) -> Encoded:
        raise NotImplementedError

    def _kept(self, state: Sequence[int], room: int, conversation: bool) -> list[int]:
        room = max(0, room)
        return list(state[max(0, len(state) - room):] if conversation else state[:room])

    def _finished(self, tokens: list[int], markers: list[int], spans: list[tuple[int, int]],
                  state: Sequence[int], kept: int) -> Encoded:
        positions, option_slots = (_parallel(spans, len(tokens)) if self.parallel else (None, None))
        return Encoded(
            tuple(tokens[:self.max_len]), tuple(marker for marker in markers if marker < self.max_len),
            None if positions is None else positions[:self.max_len],
            None if option_slots is None else option_slots[:self.max_len], len(state), kept)

    @staticmethod
    def options(question: Question) -> tuple[str, ...]:
        """Return each option's text, in the question's order."""
        if isinstance(question, Choice):
            return tuple(key if description is None or description == "" else f"{key}: {render(description)}"
                         for key, description in zip(question.options, question.descriptions, strict=True))
        if isinstance(question, Score):
            return tuple(f"level {level}: {render(description)}"
                         for level, description in enumerate(question.descriptions))
        assert isinstance(question, Noul)
        false, true = question.descriptions
        return (f"false: {'no, the statement does not hold' if false in (None, '') else render(false)}",
                f"true: {'yes, the statement holds' if true in (None, '') else render(true)}")


@dataclass(frozen=True)
class MarkerLayout(Layout):
    """Laya's layout, for a bidirectional backbone: the question, a marker before each option, then the state.

    `parallel` is Laya's `option_layout="parallel"`; the published checkpoints
    are sequential.
    """

    def _row(self, specials, head, options, state, conversation):
        tokens = [*_begin(specials), *head, specials.separator]
        markers = []
        for option in options:
            markers.append(len(tokens))
            tokens.extend([specials.marker, *option])
        tokens.append(specials.separator)
        spans = list(zip(markers, [*markers[1:], len(tokens) - 1], strict=True))
        kept = self._kept(state, self.max_len - len(tokens) - 1, conversation)
        return self._finished([*tokens, *kept, specials.separator], markers, spans, state, len(kept))


@dataclass(frozen=True)
class StateFirstLayout(Layout):
    """The layout for a causal backbone: the state, the question, then each option followed by its marker.

    A causal backbone reads each token only after the tokens before it:

        [start] state [SEP] <type> question: <instructions> [SEP] opt0 [MARK] opt1 [MARK] ... [SEP]

    so the hidden state at each marker has read the state, the question and that
    option. In the parallel layout an option reads no other option.
    """

    def _row(self, specials, head, options, state, conversation):
        tail = [*head, specials.separator]
        width = sum(len(option) + 1 for option in options) + 1
        begin = _begin(specials)
        kept = self._kept(state, self.max_len - len(begin) - 1 - len(tail) - width, conversation)
        tokens = [*begin, *kept, specials.separator, *tail]
        starts, markers = [], []
        for option in options:
            starts.append(len(tokens))
            tokens.extend(option)
            markers.append(len(tokens))
            tokens.append(specials.marker)
        tokens.append(specials.separator)
        spans = list(zip(starts, [*starts[1:], len(tokens) - 1], strict=True))
        return self._finished(tokens, markers, spans, state, len(kept))


def _begin(specials: Specials) -> list[int]:
    return [] if specials.begin is None else [specials.begin]


def _parallel(spans: list[tuple[int, int]], length: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Laya's `parallel_layout`, over each option's span of the row: every
    span starts at the first one's position, and what follows the options
    continues after the longest."""
    if not spans:
        return tuple(range(length)), (0,) * length
    start = spans[0][0]
    positions, slots = list(range(start)), [0] * start
    for slot, (first, end) in enumerate(spans):
        positions.extend(range(start, start + end - first))
        slots.extend([slot + 1] * (end - first))
    after = start + max(end - first for first, end in spans)
    positions.extend(range(after, after + length - len(positions)))
    slots.extend([0] * (length - len(slots)))
    return tuple(positions), tuple(slots)


@struct.dataclass
class DecisionInputs:
    """A batch of laid-out questions, one row each, padded to its longest.

    - `tokens` `[B, L]` ids, `valid` `[B, L]` the real tokens;
    - `markers` `[B, K]` each option slot's position, `options` `[B, K]` the
      real slots;
    - `kinds` `[B]` each row's question type, as the head embeds it;
    - in the parallel layout, `positions` `[B, L]` and `slots` `[B, L]`.
    """

    tokens: jnp.ndarray
    valid: jnp.ndarray
    markers: jnp.ndarray
    options: jnp.ndarray
    kinds: jnp.ndarray
    positions: jnp.ndarray | None = None
    slots: jnp.ndarray | None = None

    @classmethod
    def collate(cls, rows: Sequence[Encoded], kinds: Sequence[int], pad: int) -> "DecisionInputs":
        """Right-pad `rows` into one `[B, L]` batch (`pad_token_rows`), with `kinds` as their question types.

        The option slots are padded to the widest row.
        """
        parallel = rows[0].positions is not None
        if any((row.positions is not None) != parallel for row in rows):
            raise ValueError("one batch lays its rows out one way, sequential or parallel")
        fields = ({"positions": [row.positions or () for row in rows],
                   "slots": [row.slots or () for row in rows]} if parallel else None)
        tokens, padded = pad_token_rows([row.tokens for row in rows], pad_id=pad, padding_side="right",
                                        fields=fields)
        valid = padded.get("attention_mask", np.ones(tokens.shape, bool))
        width = max(len(row.markers) for row in rows)
        markers = np.zeros((len(rows), width), np.int32)
        options = np.zeros((len(rows), width), bool)
        for index, row in enumerate(rows):
            markers[index, :len(row.markers)] = row.markers
            options[index, :len(row.markers)] = True
        return cls(jnp.asarray(tokens), jnp.asarray(valid), jnp.asarray(markers), jnp.asarray(options),
                   jnp.asarray(np.asarray(kinds, np.int32)),
                   jnp.asarray(padded["positions"], jnp.int32) if parallel else None,
                   jnp.asarray(padded["slots"], jnp.int32) if parallel else None)
