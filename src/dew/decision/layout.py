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
for a causal backbone. `JointLayout` lays every question of a request in
one row, each option a span of its own.
"""

import dataclasses
import functools
import json
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeGuard

import jax.numpy as jnp
import numpy as np
from flax import struct
from typing_extensions import get_protocol_members

from dew import records
from dew.data.text import HFTokenizer, Tokenizer
from dew.decision.questions import Choice, Noul, Question, Score, kind_of
from dew.nn.inputs import Coordinates, coordinates, pad_token_rows
from dew.records import JSON

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class TokenizerSpecials(Protocol):
    """What `Specials.of` reads off a tokenizer, as a transformers tokenizer
    holds it: its special tokens' ids, each None where it has none, and an id's
    token text."""

    @property
    def cls_token_id(self) -> int | None: ...
    @property
    def sep_token_id(self) -> int | None: ...
    @property
    def eos_token_id(self) -> int | None: ...
    @property
    def bos_token_id(self) -> int | None: ...
    @property
    def mask_token_id(self) -> int | None: ...
    @property
    def pad_token_id(self) -> int | None: ...

    def convert_ids_to_tokens(self, ids: int, /) -> str | list[str]: ...


def _holds_specials(tokenizer: "Tokenizer | TokenizerSpecials | PreTrainedTokenizerBase"
                    ) -> TypeGuard[TokenizerSpecials]:
    """Whether `tokenizer` answers what `TokenizerSpecials` declares. Each member is read, not
    looked up statically, since a transformers tokenizer answers its ids through `__getattr__`."""
    return all(hasattr(tokenizer, name) for name in get_protocol_members(TokenizerSpecials))


class _Same:
    """An encoder as a cache key (`_same`), so a tokenizer need not hash."""

    __slots__ = ("__weakref__", "encoder")

    def __init__(self, encoder: Tokenizer):
        self.encoder = encoder


_keys: weakref.WeakValueDictionary[int, _Same] = weakref.WeakValueDictionary()


def _same(encoder: Tokenizer) -> _Same:
    """`encoder`'s key. A key holds its encoder, so while the key lives no
    other object takes the encoder's id."""
    key = _keys.get(id(encoder))
    if key is None:
        key = _keys[id(encoder)] = _Same(encoder)
    return key


@functools.lru_cache(maxsize=1 << 14)
def _piece(encoder: _Same, text: str) -> tuple[int, ...]:
    """`text`'s tokens under `encoder`, without specials. A layout writes the
    same templates, instructions and options into every row of a corpus, so
    the last 16,384 texts it wrote are kept, and a text the corpus repeats is
    read back while it is among them."""
    return tuple(encoder.encoder.encode(text, add_special_tokens=False))


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
    def of(cls, tokenizer: Tokenizer | TokenizerSpecials) -> "Specials":
        """Return a tokenizer's own special tokens, read off any tokenizer that
        holds them as `TokenizerSpecials` declares, a transformers tokenizer or
        the one an `HFTokenizer` loads among them.

        That is [CLS] or its start token, [SEP] or its end token, [MASK] (or else its
        end token) as the marker, and its padding token or else its end token.
        """
        hf = tokenizer.tokenizer if isinstance(tokenizer, HFTokenizer) else tokenizer
        if not _holds_specials(hf):
            raise ValueError(f"Specials.of reads a tokenizer's special tokens, which this "
                             f"{type(hf).__name__} does not hold; give its Specials explicitly")

        def first(*names: str) -> int | None:
            ids = [getattr(hf, name) for name in names]
            return next((records.integer(value, name) for value, name in zip(ids, names, strict=True)
                         if value is not None), None)

        end = first("sep_token_id", "eos_token_id")
        if end is None:
            raise ValueError("the tokenizer has neither a separator nor an end token to lay rows out with")

        def or_end(name: str) -> int:
            found = first(name)
            return end if found is None else found

        marker = or_end("mask_token_id")
        return cls(begin=first("cls_token_id", "bos_token_id"), separator=end, marker=marker,
                   marker_text=records.text(hf.convert_ids_to_tokens(marker), "marker"),
                   pad=or_end("pad_token_id"))


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
    trimmed: bool = False
    """Whether the layout shortened the question's instructions or options to fit its budgets."""
    context: tuple[int, ...] | None = None
    """The tokens of the context the row reads beside its state, through a backbone's
    cross-attention, or None for none."""


def question_fields(row: Encoded, count: int, width: int) -> dict[str, np.ndarray]:
    """`row`'s questions as fixed-shape arrays, `count` question slots of
    `width` option slots each: `kinds` [Q], `questions` [Q] (the real ones),
    `spans` [Q, 2], `option_spans` [Q, K, 2] and `options` [Q, K] (the real
    ones). A training row (`Encoding`) and a served batch
    (`DecisionInputs.collate`) both read a row through this."""
    fields = {"kinds": np.zeros(count, np.int32), "questions": np.zeros(count, bool),
              "spans": np.zeros((count, 2), np.int32), "option_spans": np.zeros((count, width, 2), np.int32),
              "options": np.zeros((count, width), bool)}
    for slot, laid in enumerate(row.questions):
        fields["kinds"][slot], fields["questions"][slot], fields["spans"][slot] = laid.kind, True, laid.span
        fields["option_spans"][slot, : len(laid.options)] = laid.options
        fields["options"][slot, : len(laid.options)] = True
    return fields


def render(value: JSON) -> str:
    """Return a value as the model reads it: text as it is, and anything structured as JSON."""
    match value:
        case str():
            return value
        case _:
            return json.dumps(value, ensure_ascii=False)


def context_tokens(tokenizer: Tokenizer, context: JSON | None, length: int) -> np.ndarray:
    """A context's tokens as a row carries them: its text's (`render`), without specials, the first
    `length` of them, and none for no context."""
    if context is None:
        return np.zeros(0, np.int32)
    return np.asarray(tokenizer.encode(render(context), add_special_tokens=False)[:length], np.int32)

def _conversation(state: JSON) -> bool:
    """Whether `state` is a conversation, a list of turns, which keeps its newest turns (Laya's agent)."""
    match state:
        case list():
            return True
        case _:
            return False


def _least(name: str, value: int, least: int) -> None:
    """Refuse a layout budget `name` that is not an integer of at least `least` tokens."""
    if type(value) is not int or value < least:
        raise ValueError(f"a layout's {name} is a whole number of at least {least} tokens, got {value!r}")


def _slots(count: int, order: Sequence[int] | None) -> list[int]:
    slots = list(range(count)) if order is None else [int(slot) for slot in order]
    if sorted(slots) != list(range(count)):
        raise ValueError(f"an option order is a permutation of the {count} options, got {slots}")
    return slots


@dataclass(frozen=True)
class Bounds:
    """The most rows (`segments`), questions and options one packed window holds;
    `options` is None where the layout trims a question's options to share a
    budget rather than giving each tokens of its own (`QuestionLayout`)."""

    segments: int
    questions: int
    options: int | None


@dataclass(frozen=True)
class Layout:
    """How a state and its questions become rows of tokens that fit `max_len` (`fits`)."""

    max_len: int = 512
    joint: ClassVar[bool] = False
    """Whether every question of a request shares one row, rather than one row each."""

    def __post_init__(self) -> None:
        _least("max_len", self.max_len, 1)
    images: ClassVar[bool] = False
    """Whether a row holds a request's images, each as the backbone's processor
    writes and expands one (`Processor.image_text`)."""

    def state(self, encoder: Tokenizer, specials: Specials, state: JSON) -> tuple[int, ...]:
        """Return the state's tokens, which every question of a request shares."""
        raise NotImplementedError

    def rows(self, encoder: Tokenizer, specials: Specials, state: JSON, questions: Mapping[str, Question],
             *, orders: Mapping[str, Sequence[int]] | None = None, media: Sequence[int] = (),
             tokens: Sequence[int] | None = None) -> list[Encoded]:
        """Lay out `questions` about `state`, each question's options in its `orders` entry.

        Slot s of a question shows its option order[s]; a question `orders` does not
        name keeps its own order (or the layout's). `media` are the tokens the
        backbone's processor gave a request's images, which a layout with an
        `image` text places in the row. `tokens` are the state's (`state`),
        where the caller has them already.
        """
        raise NotImplementedError

    def fits(self, row: Encoded) -> bool:
        """Whether `row` is within `max_len`. A layout cuts the state to fit
        and keeps a schema whole, so a row whose schema alone passes
        `max_len` does not fit: training leaves its example out (`Fit`),
        counted as unfit, and a task refuses it."""
        return len(row.tokens) <= self.max_len

    def bounds(self, encoder: Tokenizer, specials: Specials) -> Bounds:
        """The most rows, questions and options a window of `max_len` tokens
        packs, from the fewest tokens each takes in this layout: a row of one
        one-option question about an empty state, and what one more option or
        one more one-option question adds to it, laid out with room to spare
        so that nothing is cut. A window of m rows then holds at most m
        questions and m options in m rows' tokens, and each further question
        or option takes its own. The packer holds every window to these
        bounds, so a row shorter still starts a new window rather than
        outgrowing one."""
        roomy = dataclasses.replace(self, max_len=1 << 30)
        empty = self.state(encoder, specials, "")

        def size(*options: int) -> int:
            asked = {f"q{index}": Choice("x", [str(option) for option in range(count)])
                     for index, count in enumerate(options)}
            return sum(len(row.tokens) for row in roomy.rows(encoder, specials, "", asked, tokens=empty))

        row = size(1)
        question, option = size(1, 1) - row, size(2) - row
        if row > self.max_len:
            raise ValueError(f"max_len={self.max_len} holds no row of a {type(self).__name__}, whose "
                             f"shortest is {row} tokens")
        rows = range(1, self.max_len // row + 1)
        each = min(option, question)
        options = None if each <= 0 else max(count + int((self.max_len - count * row) // each)
                                             for count in rows)
        return Bounds(len(rows), max(count + (self.max_len - count * row) // question for count in rows),
                      options)


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

    def __post_init__(self) -> None:
        super().__post_init__()
        _least("head_max_len", self.head_max_len, 1)
        _least("option_tokens", self.option_tokens, 1)

    def state(self, encoder: Tokenizer, specials: Specials, state: JSON) -> tuple[int, ...]:
        """Return the state's tokens, which every question of a request shares."""
        text = render(state).replace(specials.marker_text, " ")
        return tuple(encoder.encode(text, add_special_tokens=False))

    def rows(self, encoder: Tokenizer, specials: Specials, state: JSON, questions: Mapping[str, Question],
             *, orders: Mapping[str, Sequence[int]] | None = None, media: Sequence[int] = (),
             tokens: Sequence[int] | None = None) -> list[Encoded]:
        if media:
            raise ValueError(f"a {type(self).__name__} row has no place for images")
        if tokens is None:
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
        same = _same(encoder)
        head = list(_piece(same, f"{question.kind} question: {instructions}"))
        whole = [list(_piece(same, " " + texts[slot].replace(specials.marker_text, " "))) for slot in slots]
        options = [option[:self.option_tokens] for option in whole]
        # Each option's budget counts its marker.
        budget = self.head_max_len - sum(len(option) + 1 for option in options)
        if budget < 16:
            each = max(4, (self.head_max_len - 16) // max(1, len(options)))
            options = [option[:each - 1] for option in options]
            budget = self.head_max_len - sum(len(option) + 1 for option in options)
        kept_head = head[:max(8, budget)]
        trimmed = len(kept_head) < len(head) or any(
            len(kept) < len(full) for kept, full in zip(options, whole, strict=True))
        tokens, span, markers, spans, kept = self._row(specials, kept_head, options, state, conversation)
        positions, option_slots = (_parallel(spans, len(tokens)) if self.parallel else (None, None))
        laid = Laid(name, kind_of(question), span,
                    tuple((marker, marker + 1) for marker in markers if marker < self.max_len), tuple(slots))
        return Encoded(
            tuple(tokens[:self.max_len]), (laid,),
            None if positions is None else positions[:self.max_len],
            None if option_slots is None else option_slots[:self.max_len], len(state), kept,
            trimmed=trimmed)

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


JOINT_SYSTEM = ("Read the complete state and schema. Decide every field jointly. Each answer "
                "must be exactly one of that field's allowed options.")
"""The joint layout's system prompt."""
JOINT_PREFILL = "JOINT SCHEMA DECISIONS:"
"""The joint layout's opening of the assistant's turn, after the turn's own markup."""


@dataclass(frozen=True)
class Framing:
    """How a joint row frames its text as a conversation: a `system` message, the user's turn that
    holds the state and the schema, and the assistant's turn opened with `prefill`.

    The turns' markup is the backbone's own: where the tokenizer has a chat template and `template`
    holds, its template renders the system message, the user's turn around its content and the
    generation prompt (`dew.data.chat.framed`), `thinking` its reasoning switch (False: a reasoning
    model's empty thought, None: the template's own default), so every token of the markup is the
    tokenizer's own special token, never ChatML spelled out for a vocabulary without it.
    `chat_template`, a Jinja chat template, renders the turns in place of the tokenizer's own (a base
    model's, which has none, or a framing of an experiment's), and `options` are the other variables
    a template reads beside the messages; both are recorded with the layout, so a run reloads its
    framing. A tokenizer with no chat template and none given, or `template` False, frames the turns
    as plain text: the system message, then the user's content, each after a blank line."""

    system: str = JOINT_SYSTEM
    prefill: str = JOINT_PREFILL
    thinking: bool | None = False
    template: bool = True
    chat_template: str | None = None
    options: Mapping[str, JSON] = dataclasses.field(default_factory=dict)

    def around(self, encoder: Tokenizer) -> tuple[str, str]:
        """The text before the user's turn's content and after it, to the assistant's prefill."""
        from dew.registry import to_record

        return _framed(_same(encoder), json.dumps(to_record(self, Framing), sort_keys=True))


@functools.lru_cache(maxsize=64)
def _framed(encoder: _Same, recorded: str) -> tuple[str, str]:
    """`Framing.around`, rendered once a tokenizer and framing, the framing by its record (`_same`)."""
    from dew.registry import from_record

    framing = from_record(Framing, json.loads(recorded))
    tokenizer = encoder.encoder
    templated = isinstance(tokenizer, HFTokenizer) and (
        framing.chat_template is not None or tokenizer.tokenizer.chat_template)
    if framing.template and templated and isinstance(tokenizer, HFTokenizer):
        from dew.data.chat import framed

        before, after = framed(tokenizer.tokenizer, framing.system, where=tokenizer.name,
                               thinking=framing.thinking, chat_template=framing.chat_template,
                               options=framing.options)
        return before, after + framing.prefill
    return f"{framing.system}\n\n", f"\n\n{framing.prefill}"
JOINT_NOUL = {"true": "The proposition is true or the answer is yes.",
              "false": "The proposition is false or the answer is no."}
"""A noul option's description where its question gives none."""


@dataclass(frozen=True)
class JointLayout(Layout):
    """Every question of a request in one row, each option a span of its own.

    The row is a chat prompt: the system prompt (`JOINT_SYSTEM`), the state,
    then a schema of numbered fields, each with its id, type, instructions and allowed options,
    and the opening of the assistant's turn. A request's images come before the
    state, each as the backbone's processor writes one (`Processor.image_text`), and expanded by it.
    Each option is rendered as compact
    JSON with sorted keys, `{"description": ..., "option_id": ...}`, and the
    head pools the tokens of each option and of each question's instructions.
    A choice's options come in the order of their keys, and a noul's as true,
    then false, with `JOINT_NOUL`'s descriptions where the question gives none. The
    state is cut to `max_state_tokens`, and to what the schema leaves of
    `max_len`; the schema itself is never cut, so a schema longer than
    `max_len` leaves no state and a row that does not fit (`fits`). The row's
    conversation markup is the backbone's own chat template's (`framing`).
    """

    max_len: int = 16384
    max_state_tokens: int | None = None
    framing: Framing = Framing()
    joint: ClassVar[bool] = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_state_tokens is not None:
            _least("max_state_tokens", self.max_state_tokens, 0)
    images: ClassVar[bool] = True

    def state(self, encoder: Tokenizer, specials: Specials, state: JSON) -> tuple[int, ...]:
        return tuple(encoder.encode(_compact(state), add_special_tokens=False))

    def rows(self, encoder: Tokenizer, specials: Specials, state: JSON, questions: Mapping[str, Question],
             *, orders: Mapping[str, Sequence[int]] | None = None, media: Sequence[int] = (),
             tokens: Sequence[int] | None = None) -> list[Encoded]:
        same = _same(encoder)

        def piece(text: str) -> list[int]:
            return list(_piece(same, text))

        schema = piece("\n\nSCHEMA FIELDS:\n")
        laid = []
        for index, (name, question) in enumerate(questions.items()):
            schema.extend(piece(f"\nFIELD {index + 1}\nID: {name}\nTYPE: {question.kind}\nINSTRUCTION: "))
            start = len(schema)
            schema.extend(piece(_compact(question.instructions)))
            span = (start, len(schema))
            schema.extend(piece("\nALLOWED OPTIONS:\n"))
            described = self.options(question)
            order = (orders or {}).get(name)
            slots = (_slots(len(described), order) if order is not None
                     else self.order(question))
            spans = []
            for number, slot in enumerate(slots):
                key, description = described[slot]
                schema.extend(piece(f"OPTION {number + 1}: "))
                start = len(schema)
                semantics: dict[str, JSON] = {"option_id": key}
                if description is not None:
                    semantics["description"] = description
                schema.extend(piece(_compact(semantics)))
                spans.append((start, len(schema)))
                schema.extend(piece("\n"))
            schema.extend(piece("END FIELD\n"))
            laid.append(Laid(name, kind_of(question), span, tuple(spans), tuple(slots)))
        # The user's turn holds the state and the schema, framed as the backbone's template writes a turn.
        before, after = self.framing.around(encoder)
        opening = piece(f"{before}STATE:\n")
        prefix = opening + list(media)
        suffix = piece(f"\n{after}")
        state_ids = list(self.state(encoder, specials, state) if tokens is None else tokens)
        whole = len(state_ids)
        if self.max_state_tokens is not None:
            state_ids = state_ids[:self.max_state_tokens]
        state_ids = state_ids[:max(0, self.max_len - len(prefix) - len(schema) - len(suffix))]
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
            criteria = {**JOINT_NOUL, **question.criteria}
            return tuple((key, criteria[key]) for key in question.options)
        return tuple(zip(question.options, question.descriptions, strict=True))

    @staticmethod
    def order(question: Question) -> list[int]:
        """Return the question's options in the joint order: a choice's by key, a noul's true first."""
        keys = question.options
        if isinstance(question, Choice):
            return sorted(range(len(keys)), key=lambda index: keys[index])
        if isinstance(question, Noul):
            return [keys.index("true"), keys.index("false")]
        return list(range(len(keys)))


def default_layout(causal: bool, one_question: bool) -> Layout:
    """The layout a head reads where none is given: every question of a
    request in one row (`JointLayout`) for a head that reads several a row
    (`Head.one_question` false), else the one the backbone's order asks for,
    the state first for a causal backbone (`StateFirstLayout`) and Laya's
    markers for a bidirectional one (`MarkerLayout`)."""
    if not one_question:
        return JointLayout()
    return StateFirstLayout() if causal else MarkerLayout()


def _compact(value: JSON) -> str:
    """A value as the joint layout renders it: text as it is, anything else as compact JSON with
    sorted keys."""
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
    """A batch of laid-out rows: packed windows of whole rows, or rows padded to the longest.

    - `tokens` `[B, L]` ids, `segments` `[B, L]` each token's row in its
      window, numbered from 1 in window order and 0 on the padding after the
      last (`Pack`); a padded batch is windows of one row each (`collate`);
    - `positions` `[B, L]` each token's place in its row (None: counted from
      0 along it), and in the parallel layout `slots` `[B, L]` each token's
      option slot, 0 for the tokens every option shares;
    - `kinds` `[B, Q]` each question's type (`KINDS`), `questions` `[B, Q]` the
      real questions, in their rows' window order, `spans` `[B, Q, 2]` their
      instructions' (start, end) in the window;
    - `option_spans` `[B, Q, K, 2]` each option slot's (start, end), `options`
      `[B, Q, K]` the real slots;
    - `option_slots` `[B, P]` each window's real options as flat indices
      `q * K + k` into its `[Q, K]` grid, in that order (`compacted`), and
      `slotted` `[B, P]` which of them are real, or None for the grid itself;
    - `context` `[B, C]` each window row's context tokens, which a backbone's
      cross-attention reads, `context_lengths` `[B]` how many are real, the
      padding after them, or None for rows that carry none.

    A row of one question with one marker per option, Laya's, has spans one
    token long. A head that reads `option_slots` computes the real options
    alone, P rather than Q x K of them: a window holds a few of its grid's.
    `max_segments` bounds the rows a window holds (`Layout.bounds`).
    """

    tokens: jnp.ndarray
    segments: jnp.ndarray
    kinds: jnp.ndarray
    questions: jnp.ndarray
    spans: jnp.ndarray
    option_spans: jnp.ndarray
    options: jnp.ndarray
    positions: jnp.ndarray | None = None
    slots: jnp.ndarray | None = None
    option_slots: jnp.ndarray | None = None
    slotted: jnp.ndarray | None = None
    context: jnp.ndarray | None = None
    context_lengths: jnp.ndarray | None = None
    max_segments: int = struct.field(pytree_node=False, default=1)

    @property
    def valid(self) -> jnp.ndarray:
        """Return the `[B, L]` real tokens."""
        return self.segments != 0

    def coordinates(self) -> Coordinates:
        """The tokens' `Coordinates`: each row a document, real tokens first, at
        most `max_segments` a window. The padding after them runs as one more
        sequence, which a sequence kernel adds to the documents itself
        (`dew.nn.attention.flash_attention`)."""
        return coordinates(segments=self.segments, positions=self.positions,
                           max_segments=self.max_segments, right_padded=True)

    @property
    def question_segments(self) -> jnp.ndarray:
        """Return the `[B, Q]` row each question belongs to, 0 for a padding slot."""
        return jnp.where(self.questions, jnp.take_along_axis(self.segments, self.spans[..., 0], axis=1), 0)

    @property
    def markers(self) -> jnp.ndarray:
        """Return the `[B, Q, K]` first token of each option slot, where a marker layout scores it."""
        return self.option_spans[..., 0]

    @classmethod
    def collate(cls, rows: Sequence[Encoded], pad: int, *, questions: int | None = None,
                options: int | None = None, option_slots: int | None = None,
                context_length: int | None = None) -> "DecisionInputs":
        """Right-pad `rows` into one batch (`pad_token_rows`), one row a window, its question
        and option slots to the widest.

        `questions` and `options` widen those slots further, to shapes a caller
        keeps fixed. `option_slots` is the real options a row may hold, a shape
        a caller keeps fixed too (`compacted`); without it, the batch's most,
        rounded up to `OPTION_TILE`, where that is fewer than the grid's slots.
        With a `context_length`, each row's context (`Encoded.context`, none
        where it holds none) is padded to that many tokens.
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
        laid = [question_fields(row, count, width) for row in rows]
        kinds, real, spans, option_spans, real_options = (
            np.stack([fields[name] for fields in laid])
            for name in ("kinds", "questions", "spans", "option_spans", "options"))
        if option_slots is None:
            most = int(real_options.reshape(len(rows), -1).sum(axis=1).max())
            option_slots = -(-max(most, 1) // OPTION_TILE) * OPTION_TILE
            compact = option_slots < count * width
        else:
            compact = True
        indexed = [compacted(row, option_slots) for row in real_options] if compact else None
        context = lengths = None
        if context_length is not None:
            held = [row.context or () for row in rows]
            if max(map(len, held)) > context_length:
                raise ValueError(f"a row's context holds {max(map(len, held))} tokens, more than the "
                                 f"{context_length} the backbone's cross-attention reads")
            context = jnp.asarray([[*tokens, *[pad] * (context_length - len(tokens))] for tokens in held],
                                  jnp.int32)
            lengths = jnp.asarray([len(tokens) for tokens in held], jnp.int32)
        return cls(jnp.asarray(tokens), jnp.asarray(valid, jnp.int32), jnp.asarray(kinds), jnp.asarray(real),
                   jnp.asarray(spans), jnp.asarray(option_spans), jnp.asarray(real_options),
                   jnp.asarray(padded["positions"], jnp.int32) if parallel else None,
                   jnp.asarray(padded["slots"], jnp.int32) if parallel else None,
                   None if indexed is None else jnp.asarray(np.stack([slots for slots, _ in indexed])),
                   None if indexed is None else jnp.asarray(np.stack([real for _, real in indexed])),
                   context, lengths)


OPTION_TILE = 128
"""What a batch's option slots round up to when its caller fixes none (`DecisionInputs.collate`)."""


def compacted(options: np.ndarray, slots: int) -> tuple[np.ndarray, np.ndarray]:
    """A row's real options, `options` `[Q, K]`, as `slots` flat indices into
    its grid in row order, padded with 0, and which of them are real.

    Raises ValueError for a row of more real options than `slots`: its shape
    is the caller's to keep, and widening it would compile the step again.
    """
    real = np.flatnonzero(np.asarray(options, bool).reshape(-1)).astype(np.int32)
    if len(real) > slots:
        raise ValueError(f"a row of {len(real)} options outgrows its {slots} option slots; "
                         "raise option_slots to the most options a row holds")
    indices = np.zeros(slots, np.int32)
    indices[:len(real)] = real
    return indices, np.arange(slots) < len(real)
