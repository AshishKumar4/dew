"""How a state and its questions become the token rows a decision model reads.

A layout places each question's instructions and each of its options at a
span of tokens, which the head reads. `MarkerLayout` is Laya's (laya/common.py
`build_sequence`, `build_head` and `parallel_layout` at NandhaKishorM/laya
a4a8921), one row per question: the question, then one marker token before
each option's text, then the state,

    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]

and the model scores each option at its marker. The option texts share a
token budget, `head_max_len`, and the state takes what is left of `max_len`.
`StateFirstLayout` puts the state first and each marker after its option,
for a causal backbone. `JointLayout` is Clef's, one row for every question
of a request, each option a span of its own.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
from flax import struct

from dew import records
from dew.data.text import HFTokenizer, Tokenizer
from dew.decision.questions import Choice, Noul, Question, Score, kind_of
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
class Laid:
    """One question as a row lays it out.

    `span` holds the question's own tokens, its instructions, and `options` each
    option slot's tokens, as (start, end) positions in the row. Slot s shows the
    question's option `order[s]`.
    """

    name: str
    kind: int
    """The question's type, numbered as `KINDS` numbers it."""
    span: tuple[int, int]
    options: tuple[tuple[int, int], ...]
    order: tuple[int, ...]


@dataclass(frozen=True)
class Encoded:
    """One row: its tokens and the questions laid out in it.

    In the parallel layout it also holds each token's position and option slot
    (0 for the tokens every option shares).
    """

    tokens: tuple[int, ...]
    questions: tuple[Laid, ...]
    positions: tuple[int, ...] | None
    slots: tuple[int, ...] | None
    state_tokens: int
    """The state's length in tokens, before the layout cut it to fit."""
    state_kept: int
    media: tuple[int, int] | None = None
    """Where the row holds a request's images, as a [start, end) span of its tokens."""


def render(value: JSON) -> str:
    """Return a value as the model reads it: text as it is, and anything structured as JSON."""
    match value:
        case str():
            return value
        case _:
            return json.dumps(value, ensure_ascii=False)


def _conversation(state: JSON) -> bool:
    """Whether `state` is a conversation, a list of turns, which keeps its newest turns (Laya's agent)."""
    match state:
        case list():
            return True
        case _:
            return False


def _slots(count: int, order: Sequence[int] | None) -> list[int]:
    slots = list(range(count)) if order is None else [int(slot) for slot in order]
    if sorted(slots) != list(range(count)):
        raise ValueError(f"an option order is a permutation of the {count} options, got {slots}")
    return slots


@dataclass(frozen=True)
class Layout:
    """How a state and its questions become rows of at most `max_len` tokens."""

    max_len: int = 512
    joint: ClassVar[bool] = False
    """Whether every question of a request shares one row, rather than one row each."""
    image: ClassVar[str | None] = None
    """The text that stands for one image where a row holds a request's images,
    which the backbone's processor expands; None where a row holds none."""

    def rows(self, encoder: Tokenizer, specials: Specials, state: JSON, questions: Mapping[str, Question],
             *, orders: Mapping[str, Sequence[int]] | None = None,
             media: Sequence[int] = ()) -> list[Encoded]:
        """Lay out `questions` about `state`, each question's options in its `orders` entry.

        Slot s of a question shows its option order[s]; a question `orders` does not
        name keeps its own order (or the layout's). `media` are the tokens the
        backbone's processor gave a request's images, which a layout with an
        `image` text places in the row.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class QuestionLayout(Layout):
    """A layout of one row per question, scored at one marker token per option.

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

    head_max_len: int = 192
    option_tokens: int = 48
    parallel: bool = False

    def state(self, encoder: Tokenizer, specials: Specials, state: JSON) -> tuple[int, ...]:
        """Return the state's tokens, which every question of a request shares."""
        text = render(state).replace(specials.marker_text, " ")
        return tuple(encoder.encode(text, add_special_tokens=False))

    def rows(self, encoder: Tokenizer, specials: Specials, state: JSON, questions: Mapping[str, Question],
             *, orders: Mapping[str, Sequence[int]] | None = None,
             media: Sequence[int] = ()) -> list[Encoded]:
        if media:
            raise ValueError(f"a {type(self).__name__} row has no place for images")
        tokens = self.state(encoder, specials, state)
        return [self.encode(encoder, specials, question, tokens, conversation=_conversation(state),
                            order=(orders or {}).get(name), name=name)
                for name, question in questions.items()]

    def encode(self, encoder: Tokenizer, specials: Specials, question: Question,
               state: Sequence[int], *, conversation: bool = False,
               order: Sequence[int] | None = None, name: str = "") -> Encoded:
        """Lay out `question` over the state's tokens `state`, with its options in `order`.

        Slot s shows option order[s]; None keeps the question's order. A
        `conversation` keeps the end of the state instead of its start.
        """
        texts = self.options(question)
        slots = _slots(len(texts), order)
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
        tokens, span, markers, spans, kept = self._row(specials, head[:max(8, budget)], options, state,
                                                       conversation)
        positions, option_slots = (_parallel(spans, len(tokens)) if self.parallel else (None, None))
        laid = Laid(name, kind_of(question), span,
                    tuple((marker, marker + 1) for marker in markers if marker < self.max_len), tuple(slots))
        return Encoded(
            tuple(tokens[:self.max_len]), (laid,),
            None if positions is None else positions[:self.max_len],
            None if option_slots is None else option_slots[:self.max_len], len(state), kept)

    def _row(self, specials: Specials, head: list[int], options: list[list[int]], state: Sequence[int],
             conversation: bool) -> tuple[list[int], tuple[int, int], list[int], list[tuple[int, int]], int]:
        """The row's tokens, the instructions' span, each option's marker, each
        option's whole span (which the parallel layout moves), and how much of
        the state it kept."""
        raise NotImplementedError

    def _kept(self, state: Sequence[int], room: int, conversation: bool) -> list[int]:
        room = max(0, room)
        return list(state[max(0, len(state) - room):] if conversation else state[:room])

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
class MarkerLayout(QuestionLayout):
    """Laya's layout, for a bidirectional backbone: the question, a marker before each option, then the state.

    `parallel` is Laya's `option_layout="parallel"`; the published checkpoints
    are sequential.
    """

    def _row(self, specials, head, options, state, conversation):
        tokens = [*_begin(specials), *head, specials.separator]
        span = (len(_begin(specials)), len(_begin(specials)) + len(head))
        markers = []
        for option in options:
            markers.append(len(tokens))
            tokens.extend([specials.marker, *option])
        tokens.append(specials.separator)
        spans = list(zip(markers, [*markers[1:], len(tokens) - 1], strict=True))
        kept = self._kept(state, self.max_len - len(tokens) - 1, conversation)
        return [*tokens, *kept, specials.separator], span, markers, spans, len(kept)


@dataclass(frozen=True)
class StateFirstLayout(QuestionLayout):
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
        tokens = [*begin, *kept, specials.separator]
        span = (len(tokens), len(tokens) + len(head))
        tokens.extend(tail)
        starts, markers = [], []
        for option in options:
            starts.append(len(tokens))
            tokens.extend(option)
            markers.append(len(tokens))
            tokens.append(specials.marker)
        tokens.append(specials.separator)
        spans = list(zip(starts, [*starts[1:], len(tokens) - 1], strict=True))
        return tokens, span, markers, spans, len(kept)


CLEF_SYSTEM = ("Read the complete state and schema. Decide every field jointly. Each answer "
               "must be exactly one of that field's allowed options.")
CLEF_NOUL = {"true": "The proposition is true or the answer is yes.",
             "false": "The proposition is false or the answer is no."}


@dataclass(frozen=True)
class JointLayout(Layout):
    """Clef's layout: every question of a request in one row, each option a span of its own.

    The row is a chat prompt (joint_schema_model.py `encode_record` at
    Cloudflare/clef 2f3de3dd): Clef's system prompt, the state, then a schema of
    numbered fields, each with its id, type, instructions and allowed options,
    and the opening of the assistant's turn. A request's images come before the
    state, each laid out as `image` and expanded by the backbone's processor.
    Each option is rendered as compact
    JSON with sorted keys, `{"description": ..., "option_id": ...}`, and the
    head pools the tokens of each option and of each question's instructions.
    A choice's options come in the order of their keys, and a noul's as true,
    then false, with Clef's descriptions where the question gives none. The
    state is cut to `max_state_tokens`, and to what the schema leaves of
    `max_len`; the schema itself is never cut.
    """

    max_len: int = 16384
    max_state_tokens: int | None = None
    joint: ClassVar[bool] = True
    image: ClassVar[str | None] = "<|vision_start|><|image_pad|><|vision_end|>"

    def rows(self, encoder: Tokenizer, specials: Specials, state: JSON, questions: Mapping[str, Question],
             *, orders: Mapping[str, Sequence[int]] | None = None,
             media: Sequence[int] = ()) -> list[Encoded]:
        def tokens(text: str) -> list[int]:
            return encoder.encode(text, add_special_tokens=False)

        schema = tokens("\n\nSCHEMA FIELDS:\n")
        laid = []
        for index, (name, question) in enumerate(questions.items()):
            schema.extend(tokens(f"\nFIELD {index + 1}\nID: {name}\nTYPE: {question.kind}\nINSTRUCTION: "))
            start = len(schema)
            schema.extend(tokens(_compact(question.instructions)))
            span = (start, len(schema))
            schema.extend(tokens("\nALLOWED OPTIONS:\n"))
            described = self.options(question)
            order = (orders or {}).get(name)
            slots = (_slots(len(described), order) if order is not None
                     else self.order(question))
            spans = []
            for number, slot in enumerate(slots):
                key, description = described[slot]
                schema.extend(tokens(f"OPTION {number + 1}: "))
                start = len(schema)
                semantics: dict[str, JSON] = {"option_id": key}
                if description is not None:
                    semantics["description"] = description
                schema.extend(tokens(_compact(semantics)))
                spans.append((start, len(schema)))
                schema.extend(tokens("\n"))
            schema.extend(tokens("END FIELD\n"))
            laid.append(Laid(name, kind_of(question), span, tuple(spans), tuple(slots)))
        opening = tokens(f"<|im_start|>system\n{CLEF_SYSTEM}<|im_end|>\n<|im_start|>user\nSTATE:\n")
        prefix = opening + list(media)
        suffix = tokens("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
        state_ids = tokens(_compact(state))
        whole = len(state_ids)
        if self.max_state_tokens is not None:
            state_ids = state_ids[:self.max_state_tokens]
        fixed = len(prefix) + len(schema) + len(suffix)
        if fixed > self.max_len:
            raise ValueError(f"the schema needs {fixed} tokens before the state, "
                             f"and max_len is {self.max_len}")
        state_ids = state_ids[:self.max_len - fixed]
        offset = len(prefix) + len(state_ids)

        def moved(span: tuple[int, int]) -> tuple[int, int]:
            return span[0] + offset, span[1] + offset

        shifted = tuple(Laid(question.name, question.kind, moved(question.span),
                             tuple(moved(option) for option in question.options), question.order)
                        for question in laid)
        return [Encoded(tuple(prefix + state_ids + schema + suffix), shifted, None, None, whole,
                        len(state_ids), (len(opening), len(prefix)) if media else None)]

    @staticmethod
    def options(question: Question) -> tuple[tuple[str, JSON], ...]:
        """Return each option's key and description, in the question's order."""
        if isinstance(question, Noul):
            criteria = {**CLEF_NOUL, **question.criteria}
            return tuple((key, criteria[key]) for key in question.options)
        return tuple(zip(question.options, question.descriptions, strict=True))

    @staticmethod
    def order(question: Question) -> list[int]:
        """Return the question's options in Clef's order: a choice's by key, a noul's true first."""
        keys = question.options
        if isinstance(question, Choice):
            return sorted(range(len(keys)), key=lambda index: keys[index])
        if isinstance(question, Noul):
            return [keys.index("true"), keys.index("false")]
        return list(range(len(keys)))


def _compact(value: JSON) -> str:
    """A value as Clef renders it: text as it is, anything else as compact JSON with sorted keys."""
    match value:
        case str():
            return value
        case _:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


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
    """A batch of laid-out rows, padded to the longest row, the most questions and the most options.

    - `tokens` `[B, L]` ids, `valid` `[B, L]` the real tokens;
    - `kinds` `[B, Q]` each question's type (`KINDS`), `questions` `[B, Q]` the
      real questions, `spans` `[B, Q, 2]` their instructions' (start, end);
    - `option_spans` `[B, Q, K, 2]` each option slot's (start, end), `options`
      `[B, Q, K]` the real slots;
    - in the parallel layout, `positions` `[B, L]` and `slots` `[B, L]`.

    A row of one question with one marker per option, Laya's, has Q of 1 and
    spans one token long.
    """

    tokens: jnp.ndarray
    valid: jnp.ndarray
    kinds: jnp.ndarray
    questions: jnp.ndarray
    spans: jnp.ndarray
    option_spans: jnp.ndarray
    options: jnp.ndarray
    positions: jnp.ndarray | None = None
    slots: jnp.ndarray | None = None

    @property
    def markers(self) -> jnp.ndarray:
        """Return the `[B, Q, K]` first token of each option slot, where a marker layout scores it."""
        return self.option_spans[..., 0]

    @classmethod
    def collate(cls, rows: Sequence[Encoded], pad: int, *, questions: int | None = None,
                options: int | None = None) -> "DecisionInputs":
        """Right-pad `rows` into one batch (`pad_token_rows`), its question and option slots to the widest.

        `questions` and `options` widen those slots further, to shapes a caller
        keeps fixed.
        """
        parallel = rows[0].positions is not None
        if any((row.positions is not None) != parallel for row in rows):
            raise ValueError("one batch lays its rows out one way, sequential or parallel")
        fields = ({"positions": [row.positions or () for row in rows],
                   "slots": [row.slots or () for row in rows]} if parallel else None)
        tokens, padded = pad_token_rows([row.tokens for row in rows], pad_id=pad, padding_side="right",
                                        fields=fields)
        valid = padded.get("attention_mask", np.ones(tokens.shape, bool))
        count = max(questions or 1, *(len(row.questions) for row in rows))
        width = max(options or 1, *(len(laid.options) for row in rows for laid in row.questions))
        kinds = np.zeros((len(rows), count), np.int32)
        real = np.zeros((len(rows), count), bool)
        spans = np.zeros((len(rows), count, 2), np.int32)
        option_spans = np.zeros((len(rows), count, width, 2), np.int32)
        real_options = np.zeros((len(rows), count, width), bool)
        for index, row in enumerate(rows):
            for slot, laid in enumerate(row.questions):
                kinds[index, slot] = laid.kind
                real[index, slot] = True
                spans[index, slot] = laid.span
                option_spans[index, slot, :len(laid.options)] = laid.options
                real_options[index, slot, :len(laid.options)] = True
        return cls(jnp.asarray(tokens), jnp.asarray(valid), jnp.asarray(kinds), jnp.asarray(real),
                   jnp.asarray(spans), jnp.asarray(option_spans), jnp.asarray(real_options),
                   jnp.asarray(padded["positions"], jnp.int32) if parallel else None,
                   jnp.asarray(padded["slots"], jnp.int32) if parallel else None)
