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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeGuard, get_args

import jax.numpy as jnp
import numpy as np
from flax import struct
from typing_extensions import get_protocol_members

from dew import _records
from dew._records import JSON
from dew.batch import Coordinates, coordinates
from dew.data.text import HFTokenizer, Offsets, Tokenizer
from dew.data.tokens import share_tokens
from dew.decision.questions import Choice, Noul, Question, Score, kind_of
from dew.decision.renderer import Chunk, Rendered, Unit, render as rendered_chunks
from dew.nn.inputs import pad_token_rows

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
            return next((_records.integer(value, name) for value, name in zip(ids, names, strict=True)
                         if value is not None), None)

        end = first("sep_token_id", "eos_token_id")
        if end is None:
            raise ValueError("the tokenizer has neither a separator nor an end token to lay rows out with")

        def or_end(name: str) -> int:
            found = first(name)
            return end if found is None else found

        marker = or_end("mask_token_id")
        return cls(begin=first("cls_token_id", "bos_token_id"), separator=end, marker=marker,
                   marker_text=_records.text(hf.convert_ids_to_tokens(marker), "marker"),
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
    readout: int | None = None
    """Where a readout of labels reads the question's answer (`Labels`): the position of the row's
    answer cue's last token, or None where the options are read at their own tokens."""
    label_ids: tuple[tuple[int, ...], ...] | None = None
    """Each option slot's label as the token ids a readout reads at `readout`, one a variant
    (`Labels.variants`), or None where no label stands for the options."""


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
    spans_in_context: bool = False
    """Whether the questions' spans are positions in the context, which holds the schema
    (`Schema.stream`), rather than in the row."""
    blocks: tuple[int, ...] | None = None
    """Each token's block (`QuestionLayout.blocks`): 0 on the row before its question, 1 from the
    question to the row's end, which a backbone that reads blocks attends to whole within each
    (`AttentionMetadata.block_groups`); None for a row read in order."""


def _blocked(row: Encoded, boundary: int) -> Encoded:
    """`row` with its tokens before `boundary` block 0 and the rest block 1 (`Encoded.blocks`)."""
    return dataclasses.replace(row, blocks=(0,) * boundary + (1,) * (len(row.tokens) - boundary))


OPTIONS: Literal["options"] = "options"
"""A question field's axis of one entry per option slot (`QuestionField.shape`)."""


@dataclasses.dataclass(frozen=True)
class QuestionField:
    """One array a row holds along its questions: its name, the shape each question slot holds
    (`OPTIONS` for an axis of one entry per option slot), its dtype, its value in a slot that holds
    no question, and when a row carries it: always, where its options stand under labels
    (`labelled`), in training (`trained`), or where its questions stand in consistency groups
    (`grouped`)."""

    name: str
    shape: tuple[int | Literal["options"], ...]
    dtype: type
    fill: int = 0
    when: Literal["always", "labelled", "trained", "grouped"] = "always"

    def empty(self, questions: int, width: int) -> np.ndarray:
        """The field over `questions` slots of `width` options each, every slot at its fill."""
        shape = tuple(width if isinstance(axis, str) else axis for axis in self.shape)
        return np.full((questions, *shape), self.fill, self.dtype)


QUESTION_FIELDS = (QuestionField("kinds", (), np.int32), QuestionField("questions", (), np.bool_),
                   QuestionField("spans", (2,), np.int32),
                   QuestionField("option_spans", (OPTIONS, 2), np.int32),
                   QuestionField("options", (OPTIONS,), np.bool_),
                   QuestionField("readout", (), np.int32, when="labelled"))
"""The fields along a row's questions a model reads (`DecisionInputs`): `kinds`, `questions` (the
real ones), `spans`, `option_spans`, `options` (the real ones), and where its options stand under
labels (`Labels`) `readout`, each question's answer cue. Every producer, packer and consumer of a
row's questions reads their names, shapes and fills here; a training row extends them
(`dew.decision.rows.TRAINING_FIELDS`)."""


def question_fields(row: Encoded, count: int, width: int, labelled: bool = False) -> dict[str, np.ndarray]:
    """`row`'s questions as fixed-shape arrays (`QUESTION_FIELDS`), `count` question slots of `width`
    option slots each, `readout` where its options stand under labels (`labelled`). A training row
    (`Encoding`) and a served batch (`DecisionInputs.collate`) both read a row through this."""
    fields = {field.name: field.empty(count, width) for field in QUESTION_FIELDS
              if field.when == "always" or (field.when == "labelled" and labelled)}
    for slot, laid in enumerate(row.questions):
        fields["kinds"][slot], fields["questions"][slot], fields["spans"][slot] = laid.kind, True, laid.span
        fields["option_spans"][slot, : len(laid.options)] = laid.options
        fields["options"][slot, : len(laid.options)] = True
        if labelled:
            assert laid.readout is not None, "a labelled row says where its answer is read"
            fields["readout"][slot] = laid.readout
    return fields


class _Pieces:
    """An encoder whose texts the renderer reads through the layout's cache of pieces (`_piece`):
    a corpus's rows repeat their templates, instructions and options."""

    def __init__(self, encoder: Tokenizer):
        self.same = _same(encoder)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(_piece(self.same, text))


MEDIA_MARK = ("media",)
"""The mark of a row's images (`Encoded.media`), which no question's mark, a `(name, part)` pair, is."""

INSTRUCTIONS_MARK = "instructions"
"""The part of a question's mark that its instructions take, beside its options' slots."""


def render(value: JSON) -> str:
    """Return a value as the model reads it: text as it is, and anything structured as JSON."""
    match value:
        case str():
            return value
        case _:
            return json.dumps(value, ensure_ascii=False)


def context_tokens(tokenizer: Tokenizer, context: JSON | None, length: int) -> np.ndarray:
    """A context's tokens as a row carries them: its text's (`render`), without specials, and none for
    no context. A context of more than `length` tokens, which the cross-attention reads no more of,
    is refused (ValueError), never cut: a cut would answer from part of what the caller gave."""
    if context is None:
        return np.zeros(0, np.int32)
    tokens = rendered_chunks([Chunk(render(context))], tokenizer, "pieces").tokens
    if len(tokens) > length:
        raise ValueError(f"the context is {len(tokens)} tokens, more than the {length} its cross-attention "
                         "reads (context_length)")
    return np.asarray(tokens, np.int32)


def with_context(rows: Sequence[Encoded], tokenizer: Tokenizer, context: JSON | None,
                 length: int | None) -> list[Encoded]:
    """`rows` with the context a backbone's cross-attention of `length` context tokens reads beside
    each (None: a backbone with none): a schema laid out as the rows' context (`Schema.stream`), read
    whole, or else `context`, a request's own, its first `length` tokens (`context_tokens`). The one
    resolution of a row's context, which a run's encoding and a task's admission both read. Refused
    (ValueError): a request's context beside a schema read as the context, a context with no
    cross-attention to read it, and a schema longer than `length`."""
    if any(row.spans_in_context for row in rows):
        if context is not None:
            raise ValueError("a request's context and a schema read as the row's context are one context "
                             "too many; a layout that reads the schema in context reads no other context")
        if length is None:
            raise ValueError("a schema read as the row's context needs a backbone with cross-attention "
                             "layers to read it (Pretrained.with_cross_attention)")
        for row in rows:
            if row.context is None:
                raise ValueError("a row whose questions are laid out in its context carries no context")
            if len(row.context) > length:
                raise ValueError(f"the schema needs {len(row.context)} context tokens, more than the "
                                 f"{length} the backbone's cross-attention reads")
        return list(rows)
    if context is None:
        return list(rows)
    if length is None:
        raise ValueError("a request's context is read by a backbone's cross-attention, which this "
                         "backbone does not have")
    tokens = tuple(context_tokens(tokenizer, context, length).tolist())
    return [dataclasses.replace(row, context=tokens) for row in rows]

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

    def bounds(self, encoder: Tokenizer, specials: Specials, context_length: int | None = None) -> Bounds:
        """The most rows, questions and options a window of `max_len` tokens
        packs, from the fewest tokens each takes in this layout: a row of one
        one-option question about an empty state, and what one more option or
        one more one-option question adds to it, laid out with room to spare
        so that nothing is cut. A window of m rows then holds at most m
        questions and m options in m rows' tokens, and each further question
        or option takes its own. The packer holds every window to these
        bounds, so a row shorter still starts a new window rather than
        outgrowing one. Where the questions are laid out in the row's context
        (`Encoded.spans_in_context`), a window holds one row, and the questions
        and options are those a context of `context_length` tokens holds."""
        roomy = dataclasses.replace(self, max_len=1 << 30)
        empty = self.state(encoder, specials, "")
        probe = roomy.rows(encoder, specials, "", {"q": Choice("x", ["0"])}, tokens=empty)
        in_context = probe[0].spans_in_context
        if in_context and context_length is None:
            raise ValueError("a layout that reads its questions in the row's context is bounded by the "
                             "context's length; give its context_length")
        budget = context_length if in_context and context_length is not None else self.max_len

        def size(*options: int) -> int:
            asked = {f"q{index}": Choice("x", [str(option) for option in range(count)])
                     for index, count in enumerate(options)}
            laid = roomy.rows(encoder, specials, "", asked, tokens=empty)
            return sum(len(row.context or ()) if in_context else len(row.tokens) for row in laid)

        row = size(1)
        question, option = size(1, 1) - row, size(2) - row
        if row > budget:
            raise ValueError(f"{'context_length' if in_context else 'max_len'}={budget} holds no row of a "
                             f"{type(self).__name__}, whose shortest is {row} tokens")
        if in_context:
            # One row a window: as many questions and options as its context holds.
            each = min(option, question)
            left = budget - row
            return Bounds(1, 1 + left // question, None if each <= 0 else 1 + left // each)
        rows = range(1, self.max_len // row + 1)
        each = min(option, question)
        options = None if each <= 0 else max(count + int((self.max_len - count * row) // each)
                                             for count in rows)
        return Bounds(len(rows), max(count + (self.max_len - count * row) // question for count in rows),
                      options)


LETTERS = tuple(chr(ord("A") + index) for index in range(26))


LABEL_NAMES = (*LETTERS, *(first + second for first in LETTERS for second in LETTERS))
"""Every label, in order: `A` to `Z`, then `AA`, `AB`, ..., `ZZ`, as a release's slots name them."""


def label_names(count: int) -> tuple[str, ...]:
    """The first `count` labels of `LABEL_NAMES`."""
    if count > len(LABEL_NAMES):
        raise ValueError(f"{count} options outnumber the {len(LABEL_NAMES)} labels")
    return LABEL_NAMES[:count]


@dataclass(frozen=True)
class Labels:
    """Options rendered as lettered lines, read at the row's answer cue by a readout of the labels'
    logits (`dew.decision.head.LabelReadout`, `SlotReadout`): slot s as `{name}. {text}` on a line of
    its own, its span the whole line, and after the options the `cue`, whose last token the readout
    reads. Each label is read as the token id of each of `variants`, a template of its name (`{}`): a
    release's slot is one variant, its name after a space; a readout groups several as one. A variant
    that spells as other than one token is refused.

    `names` are the labels in slot order, a release's own list where it names its slots otherwise
    (one that leaves out the pairs its tokenizer spells as two tokens), or `label_names`' where unset;
    the lines and the ids both read them (`names_for`), and a question with more options than they
    name is refused."""

    cue: str = "\nAnswer:"
    variants: tuple[str, ...] = (" {}",)
    names: tuple[str, ...] | None = None

    def names_for(self, count: int) -> tuple[str, ...]:
        """The first `count` labels."""
        if self.names is None:
            return label_names(count)
        if count > len(self.names):
            raise ValueError(f"{count} options outnumber the layout's {len(self.names)} labels")
        return self.names[:count]

    def spelled(self, encoder: Tokenizer, count: int) -> "Labels":
        """These labels named for `encoder`: themselves where they name their own, else the first `count` of
        `LABEL_NAMES` whose every variant `encoder` spells as one token, skipping those it spells as
        two (Qwen3.5 spells " BQ" as two, and 570 of the 702 as one), so a run letters as many options
        as its tokenizer can. Fewer than `count` such names is refused."""
        if self.names is not None:
            return self
        same = _same(encoder)
        named = [name for name in LABEL_NAMES
                 if all(len(_piece(same, variant.format(name))) == 1 for variant in self.variants)]
        if count > len(named):
            raise ValueError(f"{count} options outnumber the {len(named)} labels this tokenizer spells as "
                             f"one token each (of {len(LABEL_NAMES)})")
        return dataclasses.replace(self, names=tuple(named[:count]))

    def ids(self, encoder: Tokenizer, count: int) -> tuple[tuple[int, ...], ...]:
        """Each of the first `count` labels' ids, one a variant."""
        same = _same(encoder)
        found = []
        for name in self.names_for(count):
            ids = []
            for variant in self.variants:
                spelled = variant.format(name)
                pieces = _piece(same, spelled)
                if len(pieces) != 1:
                    raise ValueError(f"the label {spelled!r} spells as {len(pieces)} tokens, not one")
                ids.append(pieces[0])
            found.append(tuple(ids))
        return tuple(found)


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
    labels: Labels | None = None
    """Options rendered as lettered lines and read at an answer cue (`Labels`), or None for a marker
    after each option, read at the marker."""
    segments: "tuple[QuestionSegment, ...] | None" = None
    """A row's text as segments of its own (`dew.decision.renderer`), read at its labels' readout at the
    row's last token, or None for the layout's own text."""
    unit: Unit = "pieces"
    """How `segments`' text is encoded: piece by piece, or whole (`dew.decision.renderer`)."""
    framing: "Framing | None" = None
    """The conversation `segments`' text is the user's turn of, or None for none."""
    blocks: bool = False
    """Whether each row splits into two blocks (`Encoded.blocks`), its parts as `segments` order them:
    a state-first row's state, and its question from its header to the row's end, a published slot
    release's prefix and suffix (serve.patch `block_ids`); a question-first row's question, and its
    state from the state's first token to the row's end (the `Text` between them keeps to the part it
    follows). A backbone that reads blocks attends to each whole, and the second to the first in
    order. Off, every row is read in order."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _least("head_max_len", self.head_max_len, 1)
        _least("option_tokens", self.option_tokens, 1)
        if self.segments is not None:
            kinds = [type(segment) for segment in self.segments]
            others = sorted({type(segment).__name__ for segment in self.segments
                             if not isinstance(segment, QUESTION_KINDS)})
            if others:
                raise ValueError(f"a question row holds no schema and no images, so it has no place for "
                                 f"{others}; its parts are {[kind.__name__ for kind in QUESTION_KINDS]}")
            if kinds.count(Instructions) != 1 or kinds.count(OptionLines) != 1 or kinds.count(AnswerCue) > 1:
                raise ValueError("a question row's segments hold its instructions and its options once, and "
                                 f"its cue at most once, not {[kind.__name__ for kind in kinds]}")
            if self.labels is None or self.parallel:
                raise ValueError("a question row of segments reads its options under labels (labels=), one "
                                 "line after another")
        if self.blocks and self.parallel:
            raise ValueError("blocks and parallel options each say what a row's tokens read; a row takes one")

    def state(self, encoder: Tokenizer, specials: Specials, state: JSON) -> tuple[int, ...]:
        """Return the state's tokens, which every question of a request shares."""
        text = render(state).replace(specials.marker_text, " ")
        return tuple(encoder.encode(text, add_special_tokens=False))

    def rows(self, encoder: Tokenizer, specials: Specials, state: JSON, questions: Mapping[str, Question],
             *, orders: Mapping[str, Sequence[int]] | None = None, media: Sequence[int] = (),
             tokens: Sequence[int] | None = None) -> list[Encoded]:
        if media:
            raise ValueError(f"a {type(self).__name__} row has no place for images")
        if self.segments is not None:
            return [self._segmented(encoder, specials, state, question, (orders or {}).get(name), name)
                    for name, question in questions.items()]
        if tokens is None:
            tokens = self.state(encoder, specials, state)
        return [self.encode(encoder, specials, question, tokens, conversation=_conversation(state),
                            order=(orders or {}).get(name), name=name)
                for name, question in questions.items()]

    def _segmented(self, encoder: Tokenizer, specials: Specials, state: JSON, question: Question,
                   order: Sequence[int] | None, name: str) -> Encoded:
        """A row of `question` written by `self.segments`, framed by `self.framing` and encoded by
        `self.unit` (`dew.decision.renderer`), read at its last token. A whole row encodes its state
        with the rest of its text, never the request's state ids alone.

        A row past `max_len` cuts its state as its `State` segment keeps it (`_cut`), or none, and a
        row that no cut fits is left to `fits`."""
        assert self.segments is not None and self.labels is not None
        described = list(zip(question.options, question.descriptions, strict=True))
        slots = (_slots(len(described), order) if order is not None
                 else [1, 0] if isinstance(question, Noul) else list(range(len(described))))
        names = self.labels.names_for(len(slots))
        # The second block opens where the row's second part begins (`blocks`): the question's header
        # after a state, or the state's first copy after a question.
        kinds = [segment for segment in self.segments if isinstance(segment, (State, Instructions))]
        question_first = bool(kinds) and isinstance(kinds[0], Instructions) and any(
            isinstance(segment, State) for segment in kinds)
        chunks: list[Chunk | State] = []
        for segment in self.segments:
            if isinstance(segment, Text):
                chunks.append(Chunk(segment.text))
            elif isinstance(segment, State):
                chunks.append(segment)
            elif isinstance(segment, Instructions):
                instructions = (rendered(question.instructions, segment.render, specials)
                                or segment.defaults.get(question.kind, ""))
                head, tail = segment.headers[question.kind].split("{}")
                # The question starts at its header: the suffix a release's blocks read from
                # (serve.patch `suffix_start`), where it follows the state.
                opening = (("suffix", 0),) if self.blocks and not question_first else ()
                chunks.extend([Chunk(head, points=opening),
                               Chunk(instructions, marks=(("question", 0, len(instructions)),)), Chunk(tail)])
            elif isinstance(segment, OptionLines):
                for index, (label, slot) in enumerate(zip(names, slots, strict=True)):
                    key, description = described[slot]
                    option = (segment.level.format(key) if isinstance(question, Score)
                              else segment.noul[0 if key == "true" else 1] if isinstance(question, Noul)
                              else key)
                    given = (segment.criteria.get(key) if isinstance(question, Noul) and description is None
                             else description)
                    text = rendered(given, segment.render, specials)
                    line = (segment.line if text else segment.bare).format(label=label, name=option,
                                                                           text=text)
                    chunks.extend([*([Chunk(segment.join)] if index else []),
                                   Chunk(line, marks=((("option", index), 0, len(line)),))])
            elif isinstance(segment, AnswerCue):
                template = segment.template if len(names) <= segment.at else segment.many
                chunks.append(Chunk(template.format(labels=segment.join.join(names))))
        copies = [segment for segment in self.segments if isinstance(segment, State)]
        texts = [rendered(state, copy.render, specials) for copy in copies]

        def built(kept: Sequence[str | tuple[int, ...]]) -> Rendered:
            # Under pieces each copy is its own piece, its ids; whole, its text read with the rest.
            given = iter(kept)
            # The state's first copy opens a question-first row's second block.
            first = next((at for at, part in enumerate(chunks) if isinstance(part, State)), None)
            opening = (("suffix", 0),) if self.blocks and question_first else ()

            def piece(at: int, own: str | tuple[int, ...]) -> Chunk:
                points = opening if at == first else ()
                return Chunk(ids=own, points=points) if isinstance(own, tuple) else Chunk(own, points=points)

            pieces = [piece(at, next(given)) if isinstance(part, State) else part
                      for at, part in enumerate(chunks)]
            return rendered_chunks(self._framed(encoder, pieces), encoder, self.unit)

        laid, (whole, state_kept) = (self._cut_ids(built, texts, copies, encoder) if self.unit == "pieces"
                                     else self._cut(built, texts, copies, encoder))
        tokens = laid.tokens
        row = Laid(name, kind_of(question), laid.marks["question"],
                   tuple(laid.marks[("option", index)] for index in range(len(slots))), tuple(slots),
                   readout=len(tokens) - 1, label_ids=self.labels.ids(encoder, len(slots)))
        encoded = Encoded(tokens, (row,), None, None, whole, state_kept)
        # The prefix is the tokens that end at or before the suffix's first character (serve.patch
        # `token_layout`).
        return _blocked(encoded, laid.points["suffix"]) if self.blocks else encoded

    def _cut_ids(self, built: Callable[[Sequence[tuple[int, ...]]], Rendered], texts: Sequence[str],
                 copies: "Sequence[State]", encoder: Tokenizer) -> tuple[Rendered, tuple[int, int]]:
        """The row `built` over each copy of the state, its own text (`texts`) its own piece of ids,
        each cut to its share of what the rest of the row leaves of `max_len` and to its own budget
        (`allocated`), keeping its `keep`; and the least-kept copy's tokens and how many it keeps
        (`_least_kept`): a piece's ids add to the row as they are, so the shares are exact. A copy kept
        whole is never cut, and a row it leaves too long is `fits`' to refuse."""
        ids = [tuple(encoder.encode(text, add_special_tokens=False)) if text else () for text in texts]
        if not copies:
            return built([]), (0, 0)
        rest = len(built([()] * len(copies)).tokens)
        counts = allocated([len(own) for own in ids], copies, self.max_len - rest)
        kept = [_kept(own, count, copy.keep) for own, count, copy in zip(ids, counts, copies, strict=True)]
        return built(kept), _least_kept([(len(own), len(held)) for own, held in zip(ids, kept, strict=True)])

    def _cut(self, built: Callable[[Sequence[str]], Rendered], texts: Sequence[str],
             copies: "Sequence[State]", encoder: Tokenizer) -> tuple[Rendered, tuple[int, int]]:
        """The row `built` over each copy of the state, its own text (`texts`), encoded whole with the
        rest, each copy cut to its share of what the rest of the row leaves of `max_len` and its own
        budget (`allocated`, counted in its own tokens) at one of its token boundaries
        (`Offsets`), keeping its `keep`; and the least-kept copy's tokens and how many it keeps,
        encoded alone (`_least_kept`). A merge across a cut can lengthen the row, so while it does not
        fit the copy that keeps the most steps back one boundary, the row encoded whole again each
        time, until it fits or no copy keeps any, which `fits` then refuses. A copy kept whole is
        never cut."""
        if not copies:
            return built([]), (0, 0)
        counts = [len(encoder.encode(text, add_special_tokens=False)) if text else 0 for text in texts]
        full = built(list(texts))
        # A copy kept whole is never cut, so its cap sends no row down the cutting path.
        capped = any(copy.keep != "all" and copy.max_tokens is not None and copy.max_tokens < count
                     for copy, count in zip(copies, counts, strict=True))
        if len(full.tokens) <= self.max_len and not capped:
            return full, _least_kept([(count, count) for count in counts])
        if not isinstance(encoder, Offsets):
            raise TypeError(f"a row encoded whole cuts its state at the tokenizer's character offsets, "
                            f"which a {type(encoder).__name__} does not keep")
        offsets = [encoder.offsets(text) if text else ([], [], []) for text in texts]
        rest = len(built([""] * len(copies)).tokens)
        lengths = [len(own) for own, _, _ in offsets]
        kept = allocated(lengths, copies, self.max_len - rest)

        def piece(at: int, count: int | None = None) -> str:
            # A copy's first (or last) `count` tokens, at the character boundary their offsets give.
            text, (own, starts, ends), copy = texts[at], offsets[at], copies[at]
            count = kept[at] if count is None else count
            if count >= len(own):
                return text
            if count <= 0:
                return ""
            return text[:ends[count - 1]] if copy.keep != "end" else text[starts[len(own) - count]:]

        def own_fits(at: int, count: int) -> bool:
            # A capped copy's slice, re-encoded alone, holds no more than its cap: a slice at a
            # character boundary may encode to more tokens than it was counted by.
            cap = copies[at].max_tokens
            sliced = piece(at, count)
            return cap is None or not sliced or len(encoder.encode(sliced, add_special_tokens=False)) <= cap

        for at, copy in enumerate(copies):
            while copy.keep != "all" and kept[at] > 0 and not own_fits(at, kept[at]):
                kept[at] -= 1
        laid = built([piece(at) for at in range(len(copies))])
        while len(laid.tokens) > self.max_len:
            movable = [at for at, copy in enumerate(copies) if copy.keep != "all" and kept[at] > 0]
            if not movable:
                break
            kept[max(movable, key=lambda at: kept[at])] -= 1
            laid = built([piece(at) for at in range(len(copies))])
        # A merge across a cut may let a longer cut fit than the share counted: each copy takes the
        # longest that still fits, its own cap and the row's length both held.
        for at, copy in enumerate(copies):
            while (len(laid.tokens) <= self.max_len and copy.keep != "all" and kept[at] < lengths[at]
                   and own_fits(at, kept[at] + 1)):
                longer = built([piece(other, kept[other] + (other == at)) for other in range(len(copies))])
                if len(longer.tokens) > self.max_len:
                    break
                kept[at] += 1
                laid = longer
        alone = [len(encoder.encode(own, add_special_tokens=False)) if (own := piece(at)) else 0
                 for at in range(len(copies))]
        return laid, _least_kept(list(zip(counts, alone, strict=True)))

    def _framed(self, encoder: Tokenizer, chunks: list[Chunk]) -> list[Chunk]:
        """`chunks` as the user's turn of `self.framing`, none without one: under a whole unit the turn
        rendered with its own content (`dew.data.chat.framed_turn`), its marks carried to where the
        template holds the content, as written or trimmed; under pieces the content between the turn's
        two halves (`Framing.around`)."""
        if self.framing is None:
            return chunks
        if self.unit == "pieces":
            before, after = self.framing.around(encoder)
            return [Chunk(before), *chunks, Chunk(after)]
        content, marks, points, at = "", [], [], 0
        for chunk in chunks:
            marks.extend((key, at + start, at + end) for key, start, end in chunk.marks)
            points.extend((key, at + point) for key, point in chunk.points)
            content += chunk.text
            at += len(chunk.text)
        templated = isinstance(encoder, HFTokenizer) and (self.framing.chat_template is not None
                                                          or bool(encoder.tokenizer.chat_template))
        if self.framing.template and isinstance(encoder, HFTokenizer) and templated:
            from dew.data.chat import framed_turn

            text, start, dropped, kept = framed_turn(encoder.tokenizer, self.framing.system, content,
                                                     where=encoder.name, thinking=self.framing.thinking,
                                                     chat_template=self.framing.chat_template,
                                                     options=self.framing.options)
            moved = tuple((key, start + min(max(first - dropped, 0), kept),
                           start + min(max(last - dropped, 0), kept)) for key, first, last in marks)
            # A point the template trims away stands where the content it preceded now starts.
            placed = tuple((key, start + min(max(point - dropped, 0), kept)) for key, point in points)
            return [Chunk(text + self.framing.prefill, marks=moved, points=placed)]
        before, after = self.framing.around(encoder)
        shifted = tuple((key, len(before) + first, len(before) + last) for key, first, last in marks)
        return [Chunk(before + content + after, marks=shifted,
                      points=tuple((key, len(before) + point) for key, point in points))]

    def encode(self, encoder: Tokenizer, specials: Specials, question: Question,
               state: Sequence[int], *, conversation: bool = False,
               order: Sequence[int] | None = None, name: str = "") -> Encoded:
        """Lay out `question` over the state's tokens `state`, with its options in `order`.

        Slot s shows option order[s]; None keeps the question's order. A
        `conversation` keeps the end of the state instead of its start. A
        layout in blocks splits the row at its question's head (`blocks`).
        """
        row = self._encoded(encoder, specials, question, state, conversation=conversation, order=order,
                            name=name)
        return _blocked(row, row.questions[0].span[0]) if self.blocks else row

    def _encoded(self, encoder: Tokenizer, specials: Specials, question: Question,
                 state: Sequence[int], *, conversation: bool, order: Sequence[int] | None,
                 name: str) -> Encoded:
        """`encode`'s row, read in order."""
        texts = self.options(question)
        slots = _slots(len(texts), order)
        instructions = render(question.instructions).replace(specials.marker_text, " ")
        same = _same(encoder)
        head = list(_piece(same, f"{question.kind} question: {instructions}"))
        if self.labels is not None:
            return self._labelled(encoder, specials, question, head, texts, slots, state, conversation, name)
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

    def _labelled(self, encoder: Tokenizer, specials: Specials, question: Question, head: list[int],
                  texts: tuple[str, ...], slots: Sequence[int], state: Sequence[int], conversation: bool,
                  name: str) -> Encoded:
        """A row of `question` whose options stand under labels (`self.labels`): the state and the
        question, ordered as the layout orders them (`_state_first`), each option slot a line
        `{label}. {text}`, its span the whole line, and the answer cue last, at which the labels'
        readout reads the answer. The lines and the cue share `head_max_len` with the instructions as
        a marker row's options do: each line cut to `option_tokens`, and every line cut evenly where
        they leave the instructions fewer than 16 tokens. In the parallel layout every line starts at
        the first one's position and none attends to another, and the cue, after the longest, reads
        them all."""
        assert self.labels is not None
        same = _same(encoder)
        names = self.labels.names_for(len(slots))
        whole = [list(_piece(same, f"\n{label}. " + texts[slot].replace(specials.marker_text, " ")))
                 for label, slot in zip(names, slots, strict=True)]
        cue = list(_piece(same, self.labels.cue))
        lines = [line[:self.option_tokens] for line in whole]
        budget = self.head_max_len - sum(map(len, lines)) - len(cue)
        if budget < 16:
            each = max(4, (self.head_max_len - 16 - len(cue)) // max(1, len(lines)))
            lines = [line[:each] for line in lines]
            budget = self.head_max_len - sum(map(len, lines)) - len(cue)
        kept_head = head[:max(8, budget)]
        trimmed = len(kept_head) < len(head) or any(
            len(kept) < len(full) for kept, full in zip(lines, whole, strict=True))
        begin = _begin(specials)
        fixed = len(begin) + len(kept_head) + sum(map(len, lines)) + len(cue) + 2
        kept = self._kept(state, self.max_len - fixed, conversation)
        if self._state_first:
            tokens = [*begin, *kept, specials.separator]
            span = (len(tokens), len(tokens) + len(kept_head))
            tokens.extend([*kept_head, specials.separator])
        else:
            tokens = [*begin, *kept_head, specials.separator]
            span = (len(begin), len(begin) + len(kept_head))
        spans = []
        for line in lines:
            spans.append((len(tokens), len(tokens) + len(line)))
            tokens.extend(line)
        if not self._state_first:
            tokens.extend([specials.separator, *kept])
        tokens.extend(cue)
        # A row past max_len does not fit (`fits`): training leaves it out and a task refuses it.
        positions, option_slots = (_parallel(spans, len(tokens)) if self.parallel else (None, None))
        laid = Laid(name, kind_of(question), span, tuple(spans), tuple(slots), readout=len(tokens) - 1,
                    label_ids=self.labels.ids(encoder, len(slots)))
        return Encoded(tuple(tokens), (laid,), positions, option_slots, len(state), len(kept),
                       trimmed=trimmed)

    _state_first: ClassVar[bool] = True
    """Whether a row starts with the state, before the question; a marker layout puts it after."""

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

    _state_first: ClassVar[bool] = False

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

    def __post_init__(self) -> None:
        from dew.data.chat import template_options

        template_options(self.options, where="Framing.options")
        if self.chat_template is not None and not self.template:
            raise ValueError("a framing with its own chat_template frames through it; template=False frames "
                             "as plain text, so the two are not one framing")

    def around(self, encoder: Tokenizer) -> tuple[str, str]:
        """The text before the user's turn's content and after it, to the assistant's prefill."""
        from dew.registry import to_record

        # The record keeps the options in their own order, which a template that walks them reads.
        return _framed(_same(encoder), json.dumps(to_record(self, Framing)))


@functools.lru_cache(maxsize=64)
def _framed(encoder: _Same, recorded: str) -> tuple[str, str]:
    """`Framing.around`, rendered once a tokenizer and framing, the framing by its record (`_same`).
    A chat template of the framing's own renders through a tokenizer that renders templates
    (`HFTokenizer`), and is refused (ValueError) over one that does not, rather than left unread."""
    from dew.registry import from_record

    framing = from_record(Framing, json.loads(recorded), dtypes=False)
    tokenizer = encoder.encoder
    if framing.chat_template is not None and not isinstance(tokenizer, HFTokenizer):
        raise ValueError(f"a framing's own chat template renders through a tokenizer with a chat "
                         f"template renderer, which {type(tokenizer).__name__} is not")
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
class Text:
    """Text a joint row writes as given, at its place in the user's turn. Text beside the turn's
    markup, or beside other text, is read with it as one piece, as the turn writes it."""

    text: str


@dataclass(frozen=True)
class Media:
    """Where a joint row places a request's images, each as the backbone's processor writes one
    (`Processor.image_text`) and expands it; a request without images places nothing here."""


type Render = Literal["question", "compact", "entry", "state"]
"""How a segment writes a value as text (`rendered`)."""


@dataclass(frozen=True)
class State:
    """The state, as compact JSON with sorted keys, its start kept: cut to its own budget
    (`max_tokens`, or a joint layout's `max_state_tokens` where it gives none) and to its share of
    what the rest of the row leaves of its `max_len` (`dew.data.tokens.share_tokens`). A row may hold
    the state more than once, each copy its own segment, render, budget and `keep`. A
    question row's segments say how it renders (`render`, `rendered`) and what a cut keeps of it
    (`keep`: its start, its end, or all of it, a row that it leaves too long then unfit); a joint row
    renders it compact."""

    render: Render = "compact"
    keep: Literal["start", "end", "all"] = "start"
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens is not None:
            _least("max_tokens", self.max_tokens, 0)


def allocated(lengths: Sequence[int], copies: "Sequence[State]", room: int,
              default_cap: int | None = None) -> list[int]:
    """How many of its own tokens each copy of the state keeps of `room` (`dew.data.tokens.share_tokens`):
    a copy kept whole (`keep="all"`) all of its `lengths`, whatever its cap, and the others what is
    left, each at most its own cap (`max_tokens`, or `default_cap` where it gives none)."""
    return share_tokens(lengths, room, caps=[default_cap if copy.max_tokens is None else copy.max_tokens
                                             for copy in copies],
                        whole=[copy.keep == "all" for copy in copies])


def _least_kept(pairs: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """Of each copy's tokens and how many it keeps, the copy that keeps the least share of its own
    (the first of equals), so `Encoded.state_kept < state_tokens` says whether any copy was cut;
    `(0, 0)` for a row of none."""
    return min(pairs, key=lambda pair: pair[1] / pair[0] if pair[0] else 1.0, default=(0, 0))


def _kept(ids: Sequence[int], count: int, keep: Literal["start", "end", "all"]) -> tuple[int, ...]:
    """`count` of the state's `ids`, its first (or, with `keep` the end, its last)."""
    return tuple(ids[:count] if keep != "end" else ids[len(ids) - count:])


@dataclass(frozen=True)
class Schema:
    """Every question of the request as a field of the schema, never cut: its id, type and
    instructions, then each allowed option, each rendered as compact JSON with sorted keys,
    `{"description": ..., "option_id": ...}`. The head pools each option's tokens and each
    question's instructions'.

    `stream` is where the schema is read: in the row (`"row"`), or as the row's context
    (`"context"`), which the backbone reads in a pass of its own, with its own weights, and its
    cross-attention layers read beside the row (`DecisionModel.states`), the head pooling the
    questions and options from that pass."""

    stream: Literal["row", "context"] = "row"


type Segment = Text | Media | State | Schema
"""A part of a joint row's user turn (`JointLayout.segments`)."""


@dataclass(frozen=True)
class Instructions:
    """A question's instructions under a header of its kind (`headers`, `{}` the rendered
    instructions), rendered by `render` (`rendered`); `defaults` are a kind's instructions where its
    question's render empty, as instructions of whitespace alone do under a release's `entry`. The
    question's span is its instructions'."""

    headers: Mapping[str, str]
    render: Render = "question"
    defaults: Mapping[str, str] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class OptionLines:
    """A question's options, one line each under its label (`QuestionLayout.labels`), joined by
    `join`: `line` filled with the label, the option's name and its description rendered by `render`,
    `bare` where that renders empty; each option's span is its whole line. A choice's names are its
    keys, a score's levels `level`, and a noul's `noul`, its true option first, `criteria` its options'
    descriptions where its question gives none."""

    line: str = "{label}. {name}: {text}"
    bare: str = "{label}. {name}"
    join: str = "\n"
    render: Render = "entry"
    level: str = "level {}"
    noul: tuple[str, str] = ("yes", "no")
    criteria: Mapping[str, str] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class AnswerCue:
    """The cue a readout answers after: `template` filled with the options' labels joined by `join`,
    `many` past `at` options."""

    template: str = "Answer with one letter: {labels}."
    many: str = "Answer with one label: {labels}."
    at: int = 26
    join: str = ", "


type QuestionSegment = Text | State | Instructions | OptionLines | AnswerCue
"""A part of a question row's text (`QuestionLayout.segments`)."""

QUESTION_KINDS: tuple[type, ...] = get_args(QuestionSegment.__value__)
"""The parts a question row renders (`QuestionSegment`), which its layout refuses any other of."""
JOINT_KINDS: tuple[type, ...] = get_args(Segment.__value__)
"""The parts a joint row renders (`Segment`), which its layout refuses any other of: its questions,
their instructions and options, are its schema's, and it reads no cue."""

type RowSegment = Text | Media | State | Schema | Instructions | OptionLines | AnswerCue
"""A part of either layout's row (`Segment`, `QuestionSegment`), as a run names its row's parts
(`DecisionRunConfig.segments`); each layout refuses the parts it has none of."""



def rendered(value: JSON, policy: Render, specials: Specials) -> str:
    """`value` as a segment writes it: `question`, a question layout's text (`render`) without the
    marker's text; `compact`, a joint row's sorted compact JSON; `entry`, a release's entry (a string
    stripped, any other value indented JSON, None nothing); `state`, a release's state (a string as
    it is, any other value indented JSON, None nothing)."""
    if policy == "question":
        return render(value).replace(specials.marker_text, " ")
    if policy == "compact":
        return _compact(value)
    match value:
        case None:
            return ""
        case str():
            return value.strip() if policy == "entry" else value
        case _:
            return json.dumps(value, ensure_ascii=False, indent=2)


JOINT_SEGMENTS: tuple[Segment, ...] = (Text("STATE:\n"), Media(), State(), Schema(), Text("\n"))
"""The joint row's user turn as the Laya joint release lays it out: the state's heading, the
images, the state, the schema, and a line break before the turn's end."""


@dataclass(frozen=True)
class JointLayout(Layout):
    """Every question of a request in one row, each option a span of its own.

    The row is a chat prompt: the system prompt (`JOINT_SYSTEM`), then the user's turn, its
    `segments` in order (`JOINT_SEGMENTS`: the state's heading, a request's images, the state,
    then a schema of numbered fields, each with its id, type, instructions and allowed options),
    and the opening of the assistant's turn. A layout of `segments` of its own writes the turn
    otherwise: the schema first, or text of its own between the parts; it holds the schema once,
    the state and the images at most once each. A choice's options come in the order of their
    keys, and a noul's as true, then false, with `JOINT_NOUL`'s descriptions where the question
    gives none. The state is cut to `max_state_tokens`, and to what the rest of the row leaves of
    `max_len`; the schema itself is never cut, so a schema longer than `max_len` leaves no state
    and a row that does not fit (`fits`). The row's conversation markup is the backbone's own chat
    template's (`framing`).
    """

    max_len: int = 16384
    max_state_tokens: int | None = None
    framing: Framing = Framing()
    segments: tuple[Segment, ...] = JOINT_SEGMENTS
    joint: ClassVar[bool] = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_state_tokens is not None:
            _least("max_state_tokens", self.max_state_tokens, 0)
        kinds = [type(segment) for segment in self.segments]
        others = sorted({type(segment).__name__ for segment in self.segments
                     if not isinstance(segment, JOINT_KINDS)})
        if others:
            raise ValueError(f"a joint row writes its questions in its schema and reads no cue, so it has no "
                             f"place for {others}; its parts are {[kind.__name__ for kind in JOINT_KINDS]}")
        if kinds.count(Schema) != 1 or kinds.count(Media) > 1:
            raise ValueError("a joint row's segments hold the schema once, and the images "
                             f"at most once each, not {[kind.__name__ for kind in kinds]}")
        if any(isinstance(segment, State) and segment.render != "compact" for segment in self.segments):
            raise ValueError("a joint row writes its state as sorted compact JSON (State(render='compact'))")

    @property
    def schema_in_context(self) -> bool:
        """Whether the row's schema is its context (`Schema.stream`), its spans the context's."""
        return any(isinstance(segment, Schema) and segment.stream == "context" for segment in self.segments)
    images: ClassVar[bool] = True

    def state(self, encoder: Tokenizer, specials: Specials, state: JSON) -> tuple[int, ...]:
        return tuple(encoder.encode(_compact(state), add_special_tokens=False))

    def rows(self, encoder: Tokenizer, specials: Specials, state: JSON, questions: Mapping[str, Question],
             *, orders: Mapping[str, Sequence[int]] | None = None, media: Sequence[int] = (),
             tokens: Sequence[int] | None = None) -> list[Encoded]:
        """The request's one row, rendered as chunks (`dew.decision.renderer`, piece by piece as the
        joint release encodes it): each run of text between the turn's other parts one chunk, the
        images, each copy of the state and the schema chunks of ids, and each question's and option's
        tokens the marks its chunks carry."""
        pieces = _Pieces(encoder)
        if media and not any(isinstance(segment, Media) for segment in self.segments):
            raise ValueError("this joint layout's segments have no place for images")
        schema = rendered_chunks(self._schema(questions, orders), pieces, "pieces")
        state_ids = tuple(self.state(encoder, specials, state) if tokens is None else tokens)
        whole = len(state_ids)
        # The turn's markup and the text beside it are read as one piece each, as the turn writes them.
        before, after = self.framing.around(encoder)
        parts: list[str | Segment] = [before, *self.segments, after]
        laid_out: list[tuple[Segment | None, Chunk]] = []
        for part in parts:
            if isinstance(part, (str, Text)):
                text = part if isinstance(part, str) else part.text
                if laid_out and laid_out[-1][0] is None and laid_out[-1][1].ids is None:
                    laid_out[-1] = (None, Chunk(laid_out[-1][1].text + text))
                else:
                    laid_out.append((None, Chunk(text)))
                continue
            if isinstance(part, Schema) and not self.schema_in_context:
                chunk = Chunk(ids=schema.tokens,
                              marks=tuple((key, *span) for key, span in schema.marks.items()))
            elif isinstance(part, Media) and media:
                chunk = Chunk(ids=tuple(media), marks=((MEDIA_MARK, 0, len(media)),))
            else:
                # A state's place, filled once its copies share the room left (below); the schema read as
                # the row's context; the images' place in a request that has none.
                assert isinstance(part, (State, Schema, Media)), part
                chunk = Chunk(ids=())
            laid_out.append((part, chunk))
        # The row without its state, each copy a chunk of no ids, rendered as the row is.
        room = self.max_len - len(rendered_chunks([chunk for _, chunk in laid_out], pieces, "pieces").tokens)
        copies = [segment for segment, _ in laid_out if isinstance(segment, State)]
        counts = iter(allocated([whole] * len(copies), copies, room, self.max_state_tokens))
        chunks = [Chunk(ids=_kept(state_ids, next(counts), segment.keep)) if isinstance(segment, State)
                  else chunk for segment, chunk in laid_out]
        # A row without a state segment keeps none of the state.
        kept_count = min((len(chunk.ids or ()) for (segment, _), chunk in zip(laid_out, chunks, strict=True)
                          if isinstance(segment, State)), default=0)
        row = rendered_chunks(chunks, pieces, "pieces")
        marks = schema.marks if self.schema_in_context else row.marks
        laid = tuple(Laid(name, kind_of(question), marks[(name, INSTRUCTIONS_MARK)],
                          tuple(marks[(name, number)] for number in range(len(slots))), tuple(slots))
                     for (name, question), slots in zip(questions.items(), self._slots(questions, orders),
                                                        strict=True))
        if self.schema_in_context:
            # The schema is the row's context, its spans the context's own positions.
            return [Encoded(row.tokens, laid, None, None, whole, kept_count, row.marks.get(MEDIA_MARK),
                            context=schema.tokens, spans_in_context=True)]
        return [Encoded(row.tokens, laid, None, None, whole, kept_count, row.marks.get(MEDIA_MARK))]

    def _slots(self, questions: Mapping[str, Question],
               orders: Mapping[str, Sequence[int]] | None) -> list[list[int]]:
        """Each question's option slots: the order a request gives (`orders`), else the joint order."""
        return [_slots(len(self.options(question)), (orders or {})[name]) if name in (orders or {})
                else self.order(question) for name, question in questions.items()]

    def _schema(self, questions: Mapping[str, Question],
                orders: Mapping[str, Sequence[int]] | None) -> list[Chunk]:
        """The schema's chunks (`Schema`), each a piece the joint release encodes alone, each
        question's instructions marked `(name, INSTRUCTIONS_MARK)` and the option it shows n-th
        `(name, n)`."""
        chunks = [Chunk("\n\nSCHEMA FIELDS:\n")]
        slotted = zip(questions.items(), self._slots(questions, orders), strict=True)
        for index, ((name, question), slots) in enumerate(slotted):
            instructions = _compact(question.instructions)
            chunks += [Chunk(f"\nFIELD {index + 1}\nID: {name}\nTYPE: {question.kind}\nINSTRUCTION: "),
                       Chunk(instructions, marks=(((name, INSTRUCTIONS_MARK), 0, len(instructions)),)),
                       Chunk("\nALLOWED OPTIONS:\n")]
            described = self.options(question)
            for number, slot in enumerate(slots):
                key, description = described[slot]
                semantics: dict[str, JSON] = {"option_id": key}
                if description is not None:
                    semantics["description"] = description
                text = _compact(semantics)
                chunks += [Chunk(f"OPTION {number + 1}: "),
                           Chunk(text, marks=(((name, number), 0, len(text)),)), Chunk("\n")]
            chunks.append(Chunk("END FIELD\n"))
        return chunks

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


def question_owners(segments, spans, questions, spans_in_context: bool, xp=np):
    """The `[B, Q]` row each question of packed windows belongs to, 0 for a padding slot: the row its
    span starts in, or, where the spans are the context's (`DecisionInputs.spans_in_context`), the
    window's one row. `xp` is the array namespace, NumPy for host geometry and `jax.numpy` in a trace,
    so the model's question grouping and the counted FLOPs read one owner."""
    if spans_in_context:
        return xp.asarray(questions).astype(segments.dtype)
    starts = xp.asarray(spans)[..., 0]
    return xp.where(questions, xp.take_along_axis(segments, starts, axis=1), 0)


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
    - `context` `[B, C]` each window's rows' context tokens end to end, which a
      backbone's cross-attention reads, `context_lengths` `[B]` how many are
      real, the padding after them, and `context_segments` `[B, C]` each one's
      row in the window as `segments` numbers it (0 for padding), which its
      row's tokens alone read; or None for rows that carry none.

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
    context_segments: jnp.ndarray | None = None
    readout: jnp.ndarray | None = None
    """`[B, Q]` where a readout of labels reads each question's answer (`Laid.readout`), or None; the
    labels' ids are the readout head's own (`LabelReadout.label_ids`)."""
    blocks: jnp.ndarray | None = None
    """`[B, L]` each token's block in its row (`Encoded.blocks`), -1 on the padding, or None for rows
    read in order."""
    max_segments: int = struct.field(pytree_node=False, default=1)
    spans_in_context: bool = struct.field(pytree_node=False, default=False)
    """Whether the questions' spans are positions in `context`, which holds the schema
    (`Schema.stream`), rather than in `tokens`."""

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
        """Return the `[B, Q]` row each question belongs to, 0 for a padding slot (`question_owners`)."""
        return question_owners(self.segments, self.spans, self.questions, self.spans_in_context, jnp)

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
        in_context = rows[0].spans_in_context
        if any(row.spans_in_context != in_context for row in rows):
            raise ValueError("one batch reads its rows' schemas one way, in the row or as its context")
        blocked = rows[0].blocks is not None
        if any((row.blocks is not None) != blocked for row in rows):
            raise ValueError("one batch reads its rows one way, in blocks or in order")
        fields = ({"positions": [row.positions or () for row in rows],
                   "slots": [row.slots or () for row in rows]} if parallel else {})
        if blocked:
            fields["blocks"] = [row.blocks or () for row in rows]
        tokens, padded = pad_token_rows([row.tokens for row in rows], pad_id=pad, padding_side="right",
                                        fields=fields or None)
        valid = padded.get("attention_mask", np.ones(tokens.shape, bool))
        count = max(questions or 1, *(len(row.questions) for row in rows))
        width = max(options or 1, *(len(laid.options) for row in rows for laid in row.questions))
        labelled = labelled_rows(rows)
        laid = [question_fields(row, count, width, labelled) for row in rows]
        stacked = {name: np.stack([fields[name] for fields in laid]) for name in laid[0]}
        along: dict[str, Any] = {name: jnp.asarray(value) for name, value in stacked.items()}
        real_options = stacked["options"]
        if option_slots is None:
            most = int(real_options.reshape(len(rows), -1).sum(axis=1).max())
            option_slots = -(-max(most, 1) // OPTION_TILE) * OPTION_TILE
            compact = option_slots < count * width
        else:
            compact = True
        indexed = [compacted(row, option_slots) for row in real_options] if compact else None
        slots_of, slotted = ((None, None) if indexed is None else
                             (jnp.asarray(np.stack([slot for slot, _ in indexed])),
                              jnp.asarray(np.stack([real for _, real in indexed]))))
        context = lengths = None
        if context_length is not None:
            held = [row.context or () for row in rows]
            if max(map(len, held)) > context_length:
                raise ValueError(f"a row's context holds {max(map(len, held))} tokens, more than the "
                                 f"{context_length} the backbone's cross-attention reads")
            context = jnp.asarray([[*tokens, *[pad] * (context_length - len(tokens))] for tokens in held],
                                  jnp.int32)
            lengths = jnp.asarray([len(tokens) for tokens in held], jnp.int32)
        return cls(tokens=jnp.asarray(tokens), segments=jnp.asarray(valid, jnp.int32),
                   **along,
                   positions=jnp.asarray(padded["positions"], jnp.int32) if parallel else None,
                   slots=jnp.asarray(padded["slots"], jnp.int32) if parallel else None,
                   option_slots=slots_of, slotted=slotted,
                   context=context, context_lengths=lengths,
                   # The padding is in no block.
                   blocks=jnp.asarray(np.where(valid, padded["blocks"], -1), jnp.int32) if blocked else None,
                   spans_in_context=in_context)


def labelled_rows(rows: Sequence[Encoded]) -> bool:
    """Whether `rows`' options stand under labels (`Labels`), read at an answer cue; one batch reads
    its rows' options one way."""
    labelled = {laid.readout is not None for row in rows for laid in row.questions}
    if len(labelled) > 1:
        raise ValueError("one batch reads its rows' options one way, under labels or at their markers")
    return bool(labelled) and labelled.pop()


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
