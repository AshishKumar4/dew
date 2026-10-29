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
from dew.decision.renderer import Chunk, Point, Rendered, Unit, render as rendered_chunks
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
    instructions_cut: int = 0
    """The question's instructions' tokens its layout's declared budget cut (`QuestionLayout.head_max_len`);
    none where they are kept whole."""
    options_cut: int = 0
    """The options' tokens its layout's declared budgets cut (`QuestionLayout.head_max_len`,
    `option_tokens`), over every option; none where each is kept whole."""
    state_cut: int = 0
    """The state's tokens the row's cut removed, over every copy of it the row holds, the whole state
    where the row holds none of it."""
    state_characters_cut: int = 0
    """The state's characters a cut removed where the row's text is encoded whole, over every copy: a
    cut text encodes anew, so its removed tokens are not a count to subtract."""
    options_alike: bool = False
    """Whether a cut left two of a question's options as the same tokens where their whole texts
    differ, which no reading tells apart: such a row is refused (`Layout.refusal`)."""
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

    @property
    def trimmed(self) -> bool:
        """Whether its layout's budgets cut the question's instructions or options."""
        return self.instructions_cut > 0 or self.options_cut > 0

    @property
    def cuts(self) -> dict[str, int]:
        """The tokens each cut removed from the row, by what it cut (`instructions`, `options`,
        `state`), or the characters of a state its row encodes whole (`state_characters`), and
        `options_alike` where its options' cut left two alike; empty where the row holds its state and
        its questions whole. Training counts them in
        its data counters and a server in its own (`dew.data.mixing.CorpusCounts`), alike."""
        kinds = (("instructions", self.instructions_cut), ("options", self.options_cut),
                 ("state", self.state_cut), ("state_characters", self.state_characters_cut))
        found = {kind: tokens for kind, tokens in kinds if tokens}
        # A row whose cut left two options alike counts once, an event that removes nothing beside the
        # tokens its options lost (`cut_unit`).
        return {**found, "options_alike": 0} if self.options_alike else found


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


def context_tokens(tokenizer: Tokenizer, context: JSON | None, length: int | None) -> np.ndarray:
    """A context's tokens as a row carries them: its text's (`render`), without specials, and none for
    no context. A context of more than `length` tokens, which the cross-attention reads no more of,
    is refused (ValueError), never cut: a cut would answer from part of what the caller gave; None
    reads it whole, as a count of a context no cross-attention reads does."""
    if context is None:
        return np.zeros(0, np.int32)
    tokens = rendered_chunks([Chunk(render(context))], tokenizer, "pieces").tokens
    if length is not None and len(tokens) > length:
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

    def refusal(self, row: Encoded) -> str | None:
        """Why training leaves `row` out, or None where it trains: `unfit`, past `max_len` (`fits`),
        or `options_alike`, where a declared cut left two options the same tokens, which no reading
        tells apart. Training counts it under the reason (`dew.data.mixing.CorpusCounts`); a server
        refuses an unfit request under the same reason, and answers one whose options its release's
        budgets cut alike as the release does, the cut in its usage."""
        if not self.fits(row):
            return "unfit"
        return "options_alike" if row.options_alike else None

    def bounds(self, encoder: Tokenizer, specials: Specials, context_length: int | None = None,
               segments: int | None = None, context_window: int | None = None) -> Bounds:
        """The most rows, at most `segments`, questions and options a window of `max_len` tokens
        packs, from the fewest tokens each takes in this layout: a row of one
        one-option question about an empty state, and what one more option or
        one more one-option question adds to it, laid out with room to spare
        so that nothing is cut. A window of m rows then holds at most m
        questions and m options in m rows' tokens, and each further question
        or option takes its own. The packer holds every window to these
        bounds, so a row shorter still starts a new window rather than
        outgrowing one. Where the questions are laid out in the row's context
        (`Encoded.spans_in_context`), each row reads a context of its own of at most `context_length`
        tokens (`DecisionInputs.context_segments`), the window's contexts at most `context_window` (one
        row's where none is given): a window holds as many rows as its tokens hold, and as many questions
        and options as its contexts do, a context's schema of one question taking the place a row's
        tokens take along them."""
        roomy = dataclasses.replace(self, max_len=1 << 30)
        empty = self.state(encoder, specials, "")
        probe = roomy.rows(encoder, specials, "", {"q": Choice("x", ["0"])}, tokens=empty)
        in_context = probe[0].spans_in_context
        if in_context and context_length is None:
            raise ValueError("a layout that reads its questions in the row's context is bounded by the "
                             "context's length; give its context_length")

        def size(*options: int) -> tuple[int, int]:
            # A row's tokens and its context's, of a one-option question about an empty state for each.
            asked = {f"q{index}": Choice("x", [str(option) for option in range(count)])
                     for index, count in enumerate(options)}
            laid = roomy.rows(encoder, specials, "", asked, tokens=empty)
            return sum(len(row.tokens) for row in laid), sum(len(row.context or ()) for row in laid)

        row, held = size(1)
        if row > self.max_len:
            raise ValueError(f"max_len={self.max_len} holds no row of a {type(self).__name__}, whose "
                             f"shortest is {row} tokens")
        most = self.max_len // row if segments is None else min(segments, self.max_len // row)
        if in_context:
            assert context_length is not None
            if held > context_length:
                raise ValueError(f"context_length={context_length} holds no row's schema of a "
                                 f"{type(self).__name__}, whose shortest is {held} tokens")
            # Its questions and options stand along the window's contexts, each row's schema its own.
            budget, row = context_window or context_length, held
            question, option = size(1, 1)[1] - held, size(2)[1] - held
        else:
            budget = self.max_len
            question, option = size(1, 1)[0] - row, size(2)[0] - row
        rows = range(1, min(most, budget // row) + 1)
        each = min(option, question)
        options = None if each <= 0 else max(count + int((budget - count * row) // each) for count in rows)
        # Its rows are as many as its tokens hold; their questions and options, those their stream holds.
        return Bounds(most, max(count + (budget - count * row) // question for count in rows), options)


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

    def capacity(self, encoder: Tokenizer) -> int:
        """How many options these labels letter under `encoder`: their own names, or every name of
        `LABEL_NAMES` whose every variant it spells as one token (`spelled`)."""
        if self.names is not None:
            return len(self.names)
        same = _same(encoder)
        return sum(all(len(_piece(same, variant.format(name))) == 1 for variant in self.variants)
                   for name in LABEL_NAMES)

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


class _Unread:
    """The state of a row written from its tokens alone (`QuestionLayout.encode`), which gives no text
    to render, apart from a state of JSON null, which renders as any state does."""


_UNREAD = _Unread()


@dataclass(frozen=True)
class QuestionLayout(Layout):
    """A layout of one row per question, scored at one marker token per option.

    Each option is rendered as text (`key: description` for a choice,
    `level i: description` for a score, `false: ...` and `true: ...` for a noul).
    The question, its instructions and options, is kept whole, and the state
    fills the rest of `max_len`; a list state (a conversation) keeps its end, and
    any other state keeps its start. A question that alone passes `max_len` is a
    row that does not fit, which training leaves out and a task refuses, each
    counting it (`Layout.refusal`); its text is never cut silently.

    A layout may declare budgets for the question, as Laya's release does: each
    option cut to `option_tokens`, and where the options exceed `head_max_len`
    minus 16 tokens every option cut evenly to fit and the instructions given
    what the options leave, at least 8 tokens. Each cut is counted
    (`Encoded.instructions_cut`, `options_cut`), and a row whose cut leaves two options alike is
    refused (`Encoded.options_alike`).

    `parallel` starts every option at the same position and lets no option attend
    to another, so the options' order does not affect their scores; whatever
    follows the options continues after the longest one.
    """

    head_max_len: int | None = None
    """The tokens the question, its instructions and its options, may take, as a release declares it
    (Laya's 192), or None to keep the question whole."""
    option_tokens: int | None = None
    """The most tokens one option keeps, as a release declares it (Laya's 48), or None to keep each
    whole."""
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

    split: Literal["part", "question", "whole"] | None = None
    """Where a row in `blocks` splits: at its second part (`part`, a row of segments' own law: a
    state-first row's question from its header on, a question-first row's state from its first token
    on), at its question's first token (`question`, the built-in rows'), or nowhere, the whole row one
    block its backbone reads both ways (`whole`); None takes the layout's own law. A row read in order
    (`blocks` off) splits nowhere, so it states none."""

    order: "RowOrder | None" = None
    """The order the layout's own row writes its parts in (`built_in`), each part as the built-in row
    writes it, its begin token, separators, header, budgets and a conversation's kept end alike: the
    state first, the question first, or the state, the question and the state again (`re_read`, each
    copy its share of the room); None takes the layout's own (`StateFirstLayout`'s state first,
    `MarkerLayout`'s question first). A row of its own `segments` states its own order."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.head_max_len is not None:
            _least("head_max_len", self.head_max_len, 1)
        if self.option_tokens is not None:
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
        lines = self._lines
        if lines.marker == "none" and self.labels is None:
            raise ValueError("a question row whose options carry no marker reads them under labels (labels=)")
        if lines.marker != "none" and self.labels is not None:
            raise ValueError("a question row reads its options at their markers or under labels, not both")
        if self.parallel and lines.join:
            raise ValueError("parallel options start at one position, so no text stands between them; "
                             "their lines join with nothing")
        if (self.head_max_len is not None or self.option_tokens is not None) and self._unit != "pieces":
            raise ValueError("a question's budgets cut its pieces' tokens, which a row encoded whole has "
                             "none of")
        if self.blocks and self.parallel:
            raise ValueError("blocks and parallel options each say what a row's tokens read; a row takes one")
        if self.split is not None and not self.blocks:
            raise ValueError(f"split={self.split!r} says where a row in blocks splits; this layout's rows "
                             "are read in order, as a backbone whose attention reads no blocks reads them "
                             "(a block_bidirectional --source.attention lays a run's rows in blocks)")
        if self.order is not None and self.segments is not None:
            raise ValueError("order= writes the layout's own row in an order; segments= write a row of their "
                             "own, in the order they are given")

    @property
    def written(self) -> "tuple[QuestionSegment, ...]":
        """The row's parts in order: its own `segments`, or the layout's built-in row (`built_in`)."""
        if self.segments is not None:
            return self.segments
        return built_in(self.labels, self.order or ("state_first" if self._state_first else "question_first"))

    @property
    def _unit(self) -> Unit:
        """How the row's text is encoded: `unit` for a row of segments, the built-in row piece by piece."""
        return self.unit if self.segments is not None else "pieces"

    @property
    def _lines(self) -> "OptionLines":
        return next(segment for segment in self.written if isinstance(segment, OptionLines))

    def state(self, encoder: Tokenizer, specials: Specials, state: JSON) -> tuple[int, ...]:
        """Return the state's tokens, which every question of a request shares."""
        text = render(state).replace(specials.marker_text, " ")
        return tuple(encoder.encode(text, add_special_tokens=False))

    def rows(self, encoder: Tokenizer, specials: Specials, state: JSON, questions: Mapping[str, Question],
             *, orders: Mapping[str, Sequence[int]] | None = None, media: Sequence[int] = (),
             tokens: Sequence[int] | None = None) -> list[Encoded]:
        if media:
            raise ValueError(f"a {type(self).__name__} row has no place for images")
        if tokens is None and self._unit == "pieces":
            # The request's questions share its state, encoded once for them all.
            tokens = self.state(encoder, specials, state)
        return [self._written(encoder, specials, question, (orders or {}).get(name), name, state=state,
                              tokens=tokens, conversation=_conversation(state))
                for name, question in questions.items()]

    def encode(self, encoder: Tokenizer, specials: Specials, question: Question,
               state: Sequence[int], *, conversation: bool = False,
               order: Sequence[int] | None = None, name: str = "") -> Encoded:
        """Lay out `question` over the state's tokens `state` (`state`, a row of one copy of the state
        rendered as the layout renders it), with its options in `order` (`rows`).

        Slot s shows option order[s]; None keeps the question's order. A `conversation` keeps the end of
        the state instead of its start."""
        return self._written(encoder, specials, question, order, name, state=_UNREAD, tokens=state,
                             conversation=conversation)

    def _written(self, encoder: Tokenizer, specials: Specials, question: Question,
                 order: Sequence[int] | None, name: str, *, state: JSON | _Unread,
                 tokens: Sequence[int] | None, conversation: bool) -> Encoded:
        """A row of `question` written by its parts (`written`), framed by `framing` and encoded by
        `unit` (`dew.decision.renderer`), read at its markers or at its last token under labels.

        The question's declared budgets cut its pieces as Laya's release does (`_budgeted`); the state
        takes what the rest of the row leaves of `max_len`, each copy cut as its `State` keeps it, and a
        row that no cut fits is left to `fits`. A single copy rendered as the layout's own state reads
        the request's precomputed `tokens` where it gives them. The built-in row is written piece by
        piece, unframed, as a release writes it; `unit` and `framing` are a row of segments' own."""
        segments, labels, lines, unit = self.written, self.labels, self._lines, self._unit
        described = list(zip(question.options, question.descriptions, strict=True))
        slots = (_slots(len(described), order) if order is not None
                 else [1, 0] if isinstance(question, Noul) and lines.noul_first == "true"
                 else list(range(len(described))))
        names = labels.names_for(len(slots)) if labels is not None else ("",) * len(slots)
        texts = self.options(question)
        split = self.split or ("question" if self.segments is None else "part")
        parts = [segment for segment in segments if isinstance(segment, (State, Instructions))]
        question_first = bool(parts) and isinstance(parts[0], Instructions) and any(
            isinstance(segment, State) for segment in parts)
        opening = (("suffix", 0),) if self.blocks and split == "part" else ()
        chunks: list[Chunk | State] = []
        head_at, cue_at, line_at, labelled = None, None, [], []
        for segment in segments:
            if isinstance(segment, Text):
                chunks.append(Chunk(segment.text))
            elif isinstance(segment, Special):
                ids = tuple(_begin(specials) if segment.token == "begin" else (specials.separator,))
                # A begin token the tokenizer has none of writes nothing, so text around it reads as one.
                if ids:
                    chunks.append(Chunk(ids=ids))
            elif isinstance(segment, State):
                chunks.append(segment)
            elif isinstance(segment, Instructions):
                # The question's header opens the second part, where it follows the state.
                head_at = _instruction_chunks(chunks, segment, question, specials,
                                              opening if not question_first else ())
            elif isinstance(segment, OptionLines):
                line_at, labelled = self._option_chunks(chunks, segment, question, described, slots, names,
                                                        texts, specials)
            elif isinstance(segment, AnswerCue):
                template = segment.template if len(names) <= segment.at else segment.many
                cue_at = len(chunks)
                chunks.append(Chunk(template.format(labels=segment.join.join(names))))
        instructions_cut, options_cut, alike = self._question_cut(encoder, chunks, head_at, line_at, cue_at,
                                                                  labelled)
        copies = [_kept_by(segment, conversation) for segment in segments if isinstance(segment, State)]
        # A copy rendered as the layout renders its state reads the request's precomputed tokens where it
        # gives them, under pieces; any other renders the state's text.
        reads = [tokens is not None and unit == "pieces" and copy.render == "question" for copy in copies]
        if isinstance(state, _Unread) and not all(reads):
            raise ValueError("a row of its state's tokens alone holds copies of it rendered as the layout "
                             "renders its state, encoded piece by piece")
        state_texts = ["" if read or isinstance(state, _Unread) else rendered(state, copy.render, specials)
                       for copy, read in zip(copies, reads, strict=True)]

        def built(kept: Sequence[str | tuple[int, ...]]) -> Rendered:
            # Under pieces each copy is its own piece, its ids; whole, its text read with the rest.
            given_copies = iter(kept)
            # The state's first copy opens a question-first row's second part.
            first = next((at for at, part in enumerate(chunks) if isinstance(part, State)), None)

            def piece(at: int, own: str | tuple[int, ...]) -> Chunk:
                points = opening if at == first and question_first else ()
                return Chunk(ids=own, points=points) if isinstance(own, tuple) else Chunk(own, points=points)

            pieces = [piece(at, next(given_copies)) if isinstance(part, State) else part
                      for at, part in enumerate(chunks)]
            # Pieces read each text through the layout's cache, as a corpus repeats its templates,
            # instructions and options; a row encoded whole reads its run's offsets.
            return rendered_chunks(self._framed(encoder, pieces),
                                   _Pieces(encoder) if unit == "pieces" else encoder, unit)

        if unit == "pieces":
            ids = [tuple(tokens) if read and tokens is not None
                   else tuple(encoder.encode(text, add_special_tokens=False)) if text else ()
                   for text, read in zip(state_texts, reads, strict=True)]
            laid, held = self._cut_ids(built, ids, copies)
            characters = 0
        else:
            laid, held, characters = self._cut(built, state_texts, copies, encoder)
        whole, state_kept = _least_kept(held)
        if not held:
            # A row of no State segment holds none of the state: all of it is cut.
            if tokens is None:
                assert not isinstance(state, _Unread), "a row of the state's tokens alone is given them"
                tokens = self.state(encoder, specials, state)
            whole = len(tokens)
            held = [(whole, 0)]
        row_tokens = laid.tokens
        spans = [laid.marks[("option", index)] for index in range(len(slots))]
        row = Laid(name, kind_of(question), laid.marks["question"], tuple(spans), tuple(slots),
                   readout=len(row_tokens) - 1 if labels is not None else None,
                   label_ids=labels.ids(encoder, len(slots)) if labels is not None else None)
        groups = [laid.points[("group", index)] for index in range(len(slots))]
        positions, option_slots = (_parallel(list(zip(groups, [*groups[1:], laid.points["options_end"]],
                                                      strict=True)), len(row_tokens))
                                   if self.parallel else (None, None))
        # A cut of a row encoded whole is its characters (`_cut`); of a row of pieces, its own tokens.
        retokenized = unit == "whole" and bool(copies)
        encoded = Encoded(row_tokens, (row,), positions, option_slots, whole, state_kept,
                          state_cut=0 if retokenized else sum(own - kept for own, kept in held),
                          state_characters_cut=characters, instructions_cut=instructions_cut,
                          options_cut=options_cut, options_alike=alike)
        if not self.blocks:
            return encoded
        # The prefix is the tokens that end at or before the suffix's first character (serve.patch
        # `token_layout`).
        boundary = (laid.points["suffix"] if split == "part" else row.span[0] if split == "question"
                    else len(row_tokens))
        return _blocked(encoded, boundary)

    @staticmethod
    def _option_chunks(chunks: "list[Chunk | State]", segment: "OptionLines", question: Question,
                       described: Sequence[tuple[str, JSON]], slots: Sequence[int], names: Sequence[str],
                       texts: Sequence[str], specials: Specials) -> tuple[list[int], list[tuple[str, str]]]:
        """Append each option slot's chunks to `chunks`: its line, under its marker where the options
        carry one, joined by `segment.join`; each slot's group opening at its first chunk (`group`),
        the last closing the options (`options_end`), its span its marker's or its line's. Return where
        each line stands in `chunks`, and the text it writes before its label and through it
        (`_label_text`), as the template it was written by (`line`, or `bare`) writes them."""
        line_at, labelled = [], []
        for index, (label, slot) in enumerate(zip(names, slots, strict=True)):
            key, description = described[slot]
            option = (segment.level.format(key) if isinstance(question, Score)
                      else segment.noul[0 if key == "true" else 1] if isinstance(question, Noul) else key)
            given = (segment.criteria.get(key) if isinstance(question, Noul) and description is None
                     else description)
            text = rendered(given, segment.render, specials)
            template = segment.line if text else segment.bare
            fields = {"label": label, "name": option, "text": text,
                      "option": texts[slot].replace(specials.marker_text, " ")}
            labelled.append(_label_text(template, fields))
            line = template.format(**fields)
            if index and segment.join:
                chunks.append(Chunk(segment.join))
            final = index == len(slots) - 1
            opened = ((("group", index), 0),)
            marker = Chunk(ids=(specials.marker,), marks=((("option", index), 0, 1),),
                           points=(opened if segment.marker == "before" else ())
                           + ((("options_end", 1),) if final and segment.marker == "after" else ()))
            if segment.marker == "before":
                chunks.append(marker)
            line_at.append(len(chunks))
            chunks.append(Chunk(line, marks=((("option", index), 0, len(line)),) if segment.marker == "none"
                                else (),
                                points=(opened if segment.marker != "before" else ())
                                + ((("options_end", len(line)),) if final and segment.marker != "after"
                                    else ())))
            if segment.marker == "after":
                chunks.append(marker)
        return line_at, labelled

    def _question_cut(self, encoder: Tokenizer, chunks: "list[Chunk | State]", head_at: int | None,
                      line_at: Sequence[int], cue_at: int | None,
                      labelled: Sequence[tuple[str, str]]) -> tuple[int, int, bool]:
        """Cut the question's pieces in `chunks` as its declared budgets keep them (`_budgeted`), in
        place, and return what the cut removed: the instructions' tokens and the options', and whether
        it left two options alike past their labels. Nothing is cut where no budget is declared."""
        if self.head_max_len is None and self.option_tokens is None:
            return 0, 0, False
        assert head_at is not None
        same = _same(encoder)
        head = list(_piece(same, _text(chunks[head_at])))
        whole = [list(_piece(same, _text(chunks[at]))) for at in line_at]
        fixed = 0 if cue_at is None else len(_piece(same, _text(chunks[cue_at])))
        kept_head, kept = self._budgeted(head, whole, each=int(self._lines.marker != "none"), fixed=fixed)
        chunks[head_at] = _as_ids(chunks[head_at], kept_head)
        for at, ids in zip(line_at, kept, strict=True):
            chunks[at] = _as_ids(chunks[at], ids)
        # A line's label tells it from the others whatever its cut leaves: the options it holds are its
        # tokens without the label's own, wherever its template writes it, where the label encodes
        # apart from them.
        labels = [tuple(list(_piece(same, text)) if text else [] for text in own) for own in labelled]
        return (len(head) - len(kept_head), sum(map(len, whole)) - sum(map(len, kept)),
                _alike([_unlabelled(line, *own) for line, own in zip(whole, labels, strict=True)],
                       [_unlabelled(line, *own) for line, own in zip(kept, labels, strict=True)]))

    def _cut_ids(self, built: Callable[[Sequence[tuple[int, ...]]], Rendered], ids: Sequence[tuple[int, ...]],
                 copies: "Sequence[State]") -> tuple[Rendered, list[tuple[int, int]]]:
        """The row `built` over each copy of the state, its own piece of ids (`ids`), each cut to its
        share of what the rest of the row leaves of `max_len` and to its own budget (`allocated`),
        keeping its `keep`; and each copy's tokens and how many it keeps: a piece's ids add to the row
        as they are, so the shares are exact. A copy kept whole is never cut, and a row it leaves too
        long is `fits`' to refuse."""
        if not copies:
            return built([]), []
        rest = len(built([()] * len(copies)).tokens)
        counts = allocated([len(own) for own in ids], copies, self.max_len - rest)
        kept = [_kept(own, count, copy.keep) for own, count, copy in zip(ids, counts, copies, strict=True)]
        return built(kept), [(len(own), len(held)) for own, held in zip(ids, kept, strict=True)]

    def _cut(self, built: Callable[[Sequence[str]], Rendered], texts: Sequence[str],
             copies: "Sequence[State]", encoder: Tokenizer) -> tuple[Rendered, list[tuple[int, int]], int]:
        """The row `built` over each copy of the state, its own text (`texts`), encoded whole with the
        rest, each copy cut to its share of what the rest of the row leaves of `max_len` and its own
        budget (`allocated`, counted in its own tokens) at one of its token boundaries
        (`Offsets`), keeping its `keep`; each copy's tokens and how many it keeps, encoded alone; and
        the characters the cut removed of every copy.
        A merge across a cut can lengthen the row, so while it does not
        fit the copy that keeps the most steps back one boundary, the row encoded whole again each
        time, until it fits or no copy keeps any, which `fits` then refuses. A copy kept whole is
        never cut."""
        if not copies:
            return built([]), [], 0
        counts = [len(encoder.encode(text, add_special_tokens=False)) if text else 0 for text in texts]
        full = built(list(texts))
        # A copy kept whole is never cut, so its cap sends no row down the cutting path.
        capped = any(copy.keep != "all" and copy.max_tokens is not None and copy.max_tokens < count
                     for copy, count in zip(copies, counts, strict=True))
        if len(full.tokens) <= self.max_len and not capped:
            return full, [(count, count) for count in counts], 0
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
        # What the cut removed is characters: a cut text encodes anew, so no count of its own tokens
        # subtracts from the whole copy's.
        removed = sum(len(text) - len(piece(at)) for at, text in enumerate(texts))
        return laid, list(zip(counts, alone, strict=True)), removed

    def _framed(self, encoder: Tokenizer, chunks: list[Chunk]) -> list[Chunk]:
        """`chunks` as the user's turn of `self.framing`, none without one or for the built-in row: under
        a whole unit and a chat template the turn rendered with its own content
        (`dew.data.chat.framed_turn`), its marks carried to where the template holds the content, as
        written or trimmed; otherwise the content between the turn's two halves (`Framing.around`). Its
        special tokens and markers stand as their ids between the halves the template writes around
        the content it holds as written; a template that trims the content around them has no place
        for them, and is refused."""
        if self.framing is None or self.segments is None:
            return chunks
        templated = (self.unit == "whole" and self.framing.template and isinstance(encoder, HFTokenizer)
                     and (self.framing.chat_template is not None or bool(encoder.tokenizer.chat_template)))
        if not templated:
            before, after = self.framing.around(encoder)
            return [Chunk(before), *chunks, Chunk(after)]
        content, marks, points, at = "", [], [], 0
        for chunk in chunks:
            marks.extend((key, at + start, at + end) for key, start, end in chunk.marks)
            points.extend((key, at + point) for key, point in chunk.points)
            content += chunk.text
            at += len(chunk.text)
        assert isinstance(encoder, HFTokenizer)
        from dew.data.chat import framed_turn

        text, start, dropped, kept = framed_turn(encoder.tokenizer, self.framing.system, content,
                                                 where=encoder.name, thinking=self.framing.thinking,
                                                 chat_template=self.framing.chat_template,
                                                 options=self.framing.options)
        if any(chunk.ids for chunk in chunks):
            # The template trims the text at the content's ends, which special tokens and markers keep
            # their places between.
            held = _trimmed(chunks, dropped, len(content) - dropped - kept)
            return [Chunk(text[:start]), *held, Chunk(text[start + kept:] + self.framing.prefill)]
        moved = tuple((key, start + min(max(first - dropped, 0), kept),
                       start + min(max(last - dropped, 0), kept)) for key, first, last in marks)
        # A point the template trims away stands where the content it preceded now starts.
        placed = tuple((key, start + min(max(point - dropped, 0), kept)) for key, point in points)
        return [Chunk(text + self.framing.prefill, marks=moved, points=placed)]

    _state_first: ClassVar[bool] = True
    """Whether a row starts with the state, before the question; a marker layout puts it after."""

    def _budgeted(self, head: list[int], options: list[list[int]], *, each: int, fixed: int
                  ) -> tuple[list[int], list[list[int]]]:
        """The question's `head` and `options` as the layout's declared budgets keep them: each option
        cut to `option_tokens`, and where they and the `fixed` tokens (a cue) leave the head fewer than
        16 tokens of `head_max_len`, every option cut evenly to fit and the head given what they leave,
        at least 8, each option counting `each` tokens more (its marker); whole where none is declared.
        This is Laya's reference's cut, its release's budgets declared."""
        cap = self.option_tokens
        options = [option[:cap] for option in options] if cap is not None else options
        if self.head_max_len is None:
            return head, options
        budget = self.head_max_len - fixed - sum(len(option) + each for option in options)
        if budget < 16:
            share = max(4, (self.head_max_len - 16 - fixed) // max(1, len(options)))
            options = [option[:share - each] for option in options]
            budget = self.head_max_len - fixed - sum(len(option) + each for option in options)
        return head[:max(8, budget)], options

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


@dataclass(frozen=True)
class StateFirstLayout(QuestionLayout):
    """The layout for a causal backbone: the state, the question, then each option followed by its marker.

    A causal backbone reads each token only after the tokens before it:

        [start] state [SEP] <type> question: <instructions> [SEP] opt0 [MARK] opt1 [MARK] ... [SEP]

    so the hidden state at each marker has read the state, the question and that
    option. In the parallel layout an option reads no other option.
    """


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
    keep: Literal["start", "end", "all", "end_if_conversation"] = "start"
    """What a cut keeps of the state: its start, its end, all of it, or its end where it is a
    conversation (a list state, its last turns) and its start otherwise, as the built-in rows keep it."""
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


def _alike(whole: Sequence[Sequence[int]], kept: Sequence[Sequence[int]]) -> bool:
    """Whether a cut left two options the same tokens (`kept`) that were not the same whole."""
    return len({tuple(option) for option in kept}) < len({tuple(option) for option in whole})


def _unlabelled(line: Sequence[int], before: Sequence[int], through: Sequence[int]) -> tuple[int, ...]:
    """An option's tokens on its lettered `line` without its label's, where the label encodes apart
    from its neighbours (`through`, the text through the label, starts with `before`'s tokens, the text
    before it): the tokens before the label and those past `through`; a line cut inside the label's
    text, the tokens before the label alone; the whole line where the label merged with the text
    before it or after it."""
    if tuple(through[:len(before)]) != tuple(before):
        return tuple(line)
    if tuple(line[:len(through)]) == tuple(through):
        return (*before, *line[len(through):])
    if tuple(through[:len(line)]) == tuple(line):
        # A cut inside the label's text holds nothing of the option past the label.
        return tuple(line[:len(before)])
    return tuple(line)


def _kept_by(copy: State, conversation: bool) -> State:
    """`copy` with what its cut keeps resolved for a state that is or is not a conversation
    (`State.keep`'s `end_if_conversation`)."""
    if copy.keep != "end_if_conversation":
        return copy
    return dataclasses.replace(copy, keep="end" if conversation else "start")


def _text(chunk: "Chunk | State") -> str:
    assert isinstance(chunk, Chunk) and chunk.ids is None, "a question's budget cuts its text pieces"
    return chunk.text


def _as_ids(chunk: "Chunk | State", ids: Sequence[int]) -> Chunk:
    """A text piece `chunk` as the ids a cut kept of it: its marks over the whole piece and its points
    at either end carried to the ids'."""
    assert isinstance(chunk, Chunk)
    marks = tuple((key, 0, len(ids)) for key, _, _ in chunk.marks)
    points = tuple((key, 0 if at == 0 else len(ids)) for key, at in chunk.points)
    return Chunk(ids=tuple(ids), marks=marks, points=points)


def _instruction_chunks(chunks: "list[Chunk | State]", segment: "Instructions", question: Question,
                        specials: Specials, points: Sequence[Point]) -> int:
    """Append `question`'s instructions under its kind's header to `chunks`, the header opening at
    `points`: one piece of both where the question's span is the header's (`Instructions.span`), else
    the header's halves around the instructions, the span the instructions'. Return where the piece the
    question's budget cuts stands in `chunks`."""
    instructions = (rendered(question.instructions, segment.render, specials)
                    or segment.defaults.get(question.kind, ""))
    head, tail = segment.headers[question.kind].split("{}")
    if segment.span == "header":
        text = head + instructions + tail
        chunks.append(Chunk(text, marks=(("question", 0, len(text)),), points=tuple(points)))
        return len(chunks) - 1
    chunks.extend([Chunk(head, points=tuple(points)),
                   Chunk(instructions, marks=(("question", 0, len(instructions)),)), Chunk(tail)])
    return len(chunks) - 2


def _trimmed(chunks: Sequence[Chunk], leading: int, trailing: int) -> list[Chunk]:
    """`chunks` without the `leading` characters of the text before their first ids and the `trailing`
    after their last, as a chat template trims its turn's content around them, each mark and point in
    what is left where it stood; a trim that reaches past an id has no place to keep it, and is
    refused."""
    opening = next(at for at, chunk in enumerate(chunks) if chunk.ids)
    closing = max(at for at, chunk in enumerate(chunks) if chunk.ids)
    if (leading > sum(len(chunk.text) for chunk in chunks[:opening])
            or trailing > sum(len(chunk.text) for chunk in chunks[closing + 1:])):
        raise ValueError("the chat template trims its turn's content past a special token or marker, which "
                         "it has no place for as text; encode the row piece by piece (unit='pieces')")

    def cut(chunk: Chunk, start: int, end: int) -> Chunk:
        # The chunk's characters from `start` to `end`, its marks and points moved and held within them.
        def held(at: int) -> int:
            return min(max(at - start, 0), end - start)
        return Chunk(chunk.text[start:end], marks=tuple((key, held(first), held(last))
                                                        for key, first, last in chunk.marks),
                     points=tuple((key, held(at)) for key, at in chunk.points))

    trimmed = list(chunks)
    for at in range(opening):
        taken = min(leading, len(trimmed[at].text))
        trimmed[at] = cut(trimmed[at], taken, len(trimmed[at].text))
        leading -= taken
    for at in reversed(range(closing + 1, len(chunks))):
        taken = min(trailing, len(trimmed[at].text))
        trimmed[at] = cut(trimmed[at], 0, len(trimmed[at].text) - taken)
        trailing -= taken
    return trimmed


def _label_text(line: str, fields: Mapping[str, str]) -> tuple[str, str]:
    """The text an option's `line` writes before its label, and through it: its template up to
    `{label}`, and on to the field after it without that field's leading space (the built-in row's
    `\n{label}.`), each filled with the option's `fields`; none where the line writes no label."""
    if "{label}" not in line:
        return "", ""
    at = line.index("{label}")
    following = line.find("{", at + len("{label}"))
    through = line[:following if following >= 0 else len(line)].rstrip(" ")
    return line[:at].format(**fields), through.format(**fields)


type RowOrder = Literal["state_first", "question_first", "re_read"]
"""The order a question layout's own row writes its parts in (`QuestionLayout.order`)."""


def built_in(labels: "Labels | None", order: RowOrder) -> "tuple[QuestionSegment, ...]":
    """The built-in question row as segments (`QuestionLayout.written`): the begin token, the question's
    header and instructions as one piece (`"{kind} question: {instructions}"`), the option lines and the
    state, each part after a separator, in `order`: the state first, the question first, or the state,
    the question and the state again; Laya's markers before each option where the question comes first,
    a causal row's after, or lettered lines read at the labels' cue, which ends the row in place of its
    last separator. A conversation keeps its end in every copy."""
    head = Instructions({kind: f"{kind} question: {{}}" for kind in ("choice", "noul", "score")},
                        render="question", span="header")
    state = State(render="question", keep="end_if_conversation")
    begin, separator = Special("begin"), Special("separator")
    if labels is None:
        option = " {option}"
        lines = OptionLines(line=option, bare=option, join="", noul_first="false",
                            marker="before" if order == "question_first" else "after")
        end: QuestionSegment = separator
    else:
        line = "\n{label}. {option}"
        lines = OptionLines(line=line, bare=line, join="", noul_first="false")
        cue = labels.cue.replace("{", "{{").replace("}", "}}")
        end = AnswerCue(template=cue, many=cue)
    question = (head, separator, lines)
    if order == "question_first":
        return begin, *question, separator, state, end
    if order == "re_read":
        return begin, state, separator, *question, separator, state, end
    return begin, state, separator, *question, end


def _kept(ids: Sequence[int], count: int, keep: Literal["start", "end", "all", "end_if_conversation"]
          ) -> tuple[int, ...]:
    """`count` of the state's `ids`, its first (or, with `keep` the end, its last), `keep` resolved for
    the state it cuts (`_kept_by`)."""
    assert keep != "end_if_conversation", "a copy's keep is resolved for its state before it cuts"
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
class Special:
    """A special token the row writes as its id: the tokenizer's `begin` token (none where it has
    none) or the layout's `separator` (`Specials`)."""

    token: Literal["begin", "separator"]


@dataclass(frozen=True)
class Instructions:
    """A question's instructions under a header of its kind (`headers`, `{}` the rendered
    instructions), rendered by `render` (`rendered`); `defaults` are a kind's instructions where its
    question's render empty, as instructions of whitespace alone do under a release's `entry`. The
    question's span is its instructions' (`span="instructions"`), or the header and its instructions
    written as one piece (`span="header"`, the built-in rows' `"{kind} question: {instructions}"`)."""

    headers: Mapping[str, str]
    render: Render = "question"
    defaults: Mapping[str, str] = dataclasses.field(default_factory=dict)
    span: Literal["instructions", "header"] = "instructions"


@dataclass(frozen=True)
class OptionLines:
    """A question's options, one line each, joined by `join`: `line` filled with the option's label
    (`QuestionLayout.labels`), its name and its description rendered by `render`, `bare` where that
    renders empty, and `{option}` its whole text as the built-in rows write it (`QuestionLayout.options`:
    `key: description`, `level i: description`, `false: ...`); each option's span is its whole line. A
    choice's names are its keys, a score's levels `level`, and a noul's `noul`, its true option first
    (`noul_first`), `criteria` its options' descriptions where its question gives none.

    `marker` writes the layout's marker token before or after each line, where the option is read at
    its marker (Laya's markers before, a causal row's after) and its span is the marker's; `none` reads
    the options under labels at the answer cue."""

    line: str = "{label}. {name}: {text}"
    bare: str = "{label}. {name}"
    join: str = "\n"
    render: Render = "entry"
    level: str = "level {}"
    noul: tuple[str, str] = ("yes", "no")
    criteria: Mapping[str, str] = dataclasses.field(default_factory=dict)
    marker: Literal["none", "before", "after"] = "none"
    noul_first: Literal["true", "false"] = "true"


@dataclass(frozen=True)
class AnswerCue:
    """The cue a readout answers after: `template` filled with the options' labels joined by `join`,
    `many` past `at` options."""

    template: str = "Answer with one letter: {labels}."
    many: str = "Answer with one label: {labels}."
    at: int = 26
    join: str = ", "


type QuestionSegment = Text | Special | State | Instructions | OptionLines | AnswerCue
"""A part of a question row's text (`QuestionLayout.segments`)."""

QUESTION_KINDS: tuple[type, ...] = get_args(QuestionSegment.__value__)
"""The parts a question row renders (`QuestionSegment`), which its layout refuses any other of."""
JOINT_KINDS: tuple[type, ...] = get_args(Segment.__value__)
"""The parts a joint row renders (`Segment`), which its layout refuses any other of: its questions,
their instructions and options, are its schema's, and it reads no cue."""

type RowSegment = Text | Special | Media | State | Schema | Instructions | OptionLines | AnswerCue
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
        conversation = _conversation(state)
        chunks = [Chunk(ids=_kept(state_ids, next(counts), _kept_by(segment, conversation).keep))
                  if isinstance(segment, State) else chunk for segment, chunk in laid_out]
        # A row without a state segment keeps none of the state.
        kept_count = min((len(chunk.ids or ()) for (segment, _), chunk in zip(laid_out, chunks, strict=True)
                          if isinstance(segment, State)), default=0)
        row = rendered_chunks(chunks, pieces, "pieces")
        marks = schema.marks if self.schema_in_context else row.marks
        laid = tuple(Laid(name, kind_of(question), marks[(name, INSTRUCTIONS_MARK)],
                          tuple(marks[(name, number)] for number in range(len(slots))), tuple(slots))
                     for (name, question), slots in zip(questions.items(), self._slots(questions, orders),
                                                        strict=True))
        held = [len(chunk.ids or ()) for (segment, _), chunk in zip(laid_out, chunks, strict=True)
                if isinstance(segment, State)]
        # A row of no State segment holds none of the state: all of it is cut.
        state_cut = sum(whole - kept for kept in held) if held else whole
        if self.schema_in_context:
            # The schema is the row's context, its spans the context's own positions.
            return [Encoded(row.tokens, laid, None, None, whole, kept_count, row.marks.get(MEDIA_MARK),
                            context=schema.tokens, spans_in_context=True, state_cut=state_cut)]
        return [Encoded(row.tokens, laid, None, None, whole, kept_count, row.marks.get(MEDIA_MARK),
                        state_cut=state_cut)]

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


def question_owners(segments, spans, questions, spans_in_context: bool, xp=np, rows=None):
    """The `[B, Q]` row each question of packed windows belongs to, 0 for a padding slot: the row the
    packer placed it in (`rows`, `PackedRows`' `question_rows`), or where none is given the row its
    span starts in, and where the spans are the context's (`DecisionInputs.spans_in_context`) the
    window's one row (`DecisionInputs.collate`). `xp` is the array namespace, NumPy for host geometry
    and `jax.numpy` in a trace, so the model's question grouping and the counted FLOPs read one owner."""
    if rows is not None:
        return xp.where(questions, xp.asarray(rows), 0).astype(segments.dtype)
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
      row's tokens alone read; or None for rows that carry none;
    - `question_rows` `[B, Q]` each question's row in its window as the packer
      placed it, numbered as `segments` numbers its tokens (0 for a padding
      slot), or None where a question's row is the one its span starts in.

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
    question_rows: jnp.ndarray | None = None
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
    whole_blocks: bool = struct.field(pytree_node=False, default=False)
    """Whether each row's one block is the whole row (`QuestionLayout.split` 'whole'), which a backbone
    reads as non-causal attention by rows (`AttentionMetadata.whole_blocks`)."""

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
        return question_owners(self.segments, self.spans, self.questions, self.spans_in_context, jnp,
                               self.question_rows)

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
