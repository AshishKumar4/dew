"""One renderer of decision rows (`dew.decision.renderer`, docs/design/decision-render.md): a row's
chunks encoded piece by piece or whole, each mark's tokens the ones that overlap its characters, a
token across two marks both's; and a published slot release's prompt as a layout value
(`QuestionLayout(segments=...)`), its rows the release's own text and ids (qev/prompt.py at OneJev
81ce62f, rendered as its engine renders them), its options' spans the lines its runtime pools, its
answer at its last token, and its state cut only where its segments say."""

import dataclasses
import json
from pathlib import Path
from typing import Literal

import pytest

from dew.data.text import ByteTokenizer, HFTokenizer
from dew.decision import Choice, Noul, Question, Score
from dew.decision.layout import (
    LETTERS,
    AnswerCue,
    Framing,
    Instructions,
    Labels,
    OptionLines,
    QuestionLayout,
    Specials,
    State,
    Text,
)
from dew.decision.renderer import Chunk, render
from dew.records import JSON

QWEN = Path(__file__).parent / "fixtures" / "hf" / "qwen38-dense-tiny"
"""Qwen3.5's ChatML template, with its thinking switch, over a tiny vocabulary of bytes and Qwen's
special tokens: no merges, so its ids are the text's bytes and the template's own tokens."""
LAYA = Path(__file__).parent / "fixtures" / "laya" / "tiny" / "tokenizer"
"""Laya's tiny BPE tokenizer, which merges."""
ELEPHANT, ELEPHANT_REVISION = "baselabs/elephant-1", "b3f8a93462fb1051a9de031e94a7ff380e254072"
"""The slot release whose tokenizer spells every one of its labels as one token."""

SLOT_SYSTEM = ("Apply the question to the state. Choose exactly one of the listed options. Respond with only "
               "its uppercase letter, with no explanation or reasoning.")
NOT_ONE_TOKEN = frozenset(["BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ",
                           "GJ", "GK", "GQ", "GZ", "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ"])
SLOT_LABELS = (*LETTERS, *[first + second for first in LETTERS for second in LETTERS
                           if first + second not in NOT_ONE_TOKEN][:256 - 26])
"""The release's 256 labels (qev/prompt.py `SLOT_LABELS`): A to Z, then the pairs its tokenizer spells
as one token."""


RELEASE_LABELS = Labels(variants=("{}",), names=SLOT_LABELS)
"""The release's labels, each read as its bare self (OneJev engine.py `_verify_slots`)."""


def release(*, max_len: int = 32768, keep: Literal["start", "end", "all"] = "all") -> QuestionLayout:
    """The slot release's prompt, as values: its system message, a tagged state, the question under its
    kind's header, its options as lettered lines, and its cue, framed with thinking off, encoded whole."""
    return QuestionLayout(
        max_len=max_len, labels=RELEASE_LABELS, unit="whole",
        framing=Framing(system=SLOT_SYSTEM, prefill="", thinking=False),
        segments=(Text("<state>\n"), State(render="state", keep=keep), Text("\n</state>\n\n"),
                  Instructions({"choice": "Question: {}", "noul": "Question: {}",
                                "score": "Rate the state: {}"}, render="entry",
                               defaults={"choice": "Which option applies to the state?",
                                         "score": "Which level describes the state?",
                                         "noul": ("Is the statement true, or is the answer to the question "
                                                  "yes?")}),
                  Text("\n\nOptions:\n"),
                  OptionLines(criteria={"true": "the statement is true / the answer is yes",
                                        "false": "the statement is false / the answer is no"}),
                  Text("\n\n"), AnswerCue()))


SPECIALS = Specials(begin=None, separator=0, marker=1, marker_text="\x01", pad=0)
STATE = "I was charged twice this month, please refund one of them."
TEAM = Choice("Which team should handle this?", {"billing": "Charges, refunds, invoices",
                                                  "technical": "App or site faults",
                                                  "fraud": "Suspected unauthorised use"})
URGENCY = Score("How urgent is this?", ["Can wait", "Today", "Blocking the customer now"])
CHOICE_PROMPT = (
    "<|im_start|>system\nApply the question to the state. Choose exactly one of the listed options. Respond "
    "with only its uppercase letter, with no explanation or reasoning.<|im_end|>\n<|im_start|>user\n<state>\n"
    "I was charged twice this month, please refund one of them.\n</state>\n\nQuestion: Which team should "
    "handle this?\n\nOptions:\nA. billing: Charges, refunds, invoices\nB. technical: App or site faults\nC. "
    "fraud: Suspected unauthorised use\n\nAnswer with one letter: A, B, C.<|im_end|>\n<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n")
"""The release's own render of a three-option choice (its `_render_prompt`, Qwen3.5's template)."""
SCORE_PROMPT = CHOICE_PROMPT.replace(
    "Question: Which team should handle this?\n\nOptions:\nA. billing: Charges, refunds, invoices\nB. "
    "technical: App or site faults\nC. fraud: Suspected unauthorised use",
    "Rate the state: How urgent is this?\n\nOptions:\nA. level 0: Can wait\nB. level 1: Today\nC. level 2: "
    "Blocking the customer now")
"""The release's own render of a three-level score."""


@pytest.fixture(scope="module")
def qwen() -> HFTokenizer:
    return HFTokenizer(str(QWEN), local_files_only=True)


def lines_of(prompt: str, count: int) -> list[tuple[int, int]]:
    """Each option line's characters in `prompt`, as the release's runtime spans them (serve.patch
    `render_question`): its whole line, the line break after it left out."""
    start = prompt.index("Options:\n") + len("Options:\n")
    spans = []
    for _ in range(count):
        end = prompt.index("\n", start)
        spans.append((start, end))
        start = end + 1
    return spans


def overlapping(tokenizer: HFTokenizer, prompt: str, span: tuple[int, int]) -> tuple[int, int]:
    """The tokens of `prompt` that overlap `span`'s characters, as serve.patch's `token_layout` finds
    them, from the tokenizer's own offsets."""
    _, starts, ends = tokenizer.offsets(prompt)
    first = next(index for index, end in enumerate(ends) if end > span[0])
    last = max(index for index, start in enumerate(starts) if start < span[1]) + 1
    return first, last


@pytest.mark.parametrize(("question", "prompt"), [(TEAM, CHOICE_PROMPT), (URGENCY, SCORE_PROMPT)])
def test_a_slot_release_row_is_the_releases_own_prompt(qwen, question, prompt):
    """A choice and a score lay out as the release renders them: their ids the tokenizer's of the
    release's whole prompt, each option's span the tokens overlapping its line, the answer the last
    token, and each label the bare letter's one id (over a vocabulary that merges nothing; the
    release's own tokenizer, which merges, under `network`)."""
    [row] = release().rows(qwen, SPECIALS, STATE, {"asked": question})
    assert list(row.tokens) == qwen.encode(prompt, add_special_tokens=False)
    [laid] = row.questions
    assert list(laid.options) == [overlapping(qwen, prompt, line) for line in lines_of(prompt, 3)]
    assert laid.readout == len(row.tokens) - 1
    assert laid.label_ids == tuple((qwen.encode(label, add_special_tokens=False)[0],) for label in "ABC")


def test_a_noul_asks_yes_first_with_the_releases_descriptions(qwen):
    """A noul's options are yes, then no, its true option first (`Laid.order`), each with the
    release's own description where its question gives none; a noul whose instructions are whitespace
    alone, which its protocol admits from the wire and the release's entry renders empty, asks the
    release's default instructions, its span theirs."""
    [row] = release().rows(qwen, SPECIALS, STATE, {"holds": Noul("Was the customer charged twice?")})
    [laid] = row.questions
    text = qwen.decode(list(row.tokens))
    assert laid.order == (1, 0)
    assert ("A. yes: the statement is true / the answer is yes\nB. no: the statement is false / the answer "
            "is no\n\nAnswer with one letter: A, B.") in text
    blank = Question.from_wire({"type": "noul", "instructions": "   "})
    [row] = release().rows(qwen, SPECIALS, STATE, {"holds": blank})
    start, end = row.questions[0].span
    assert qwen.decode(list(row.tokens[start:end])) == ("Is the statement true, or is the answer to the "
                                                        "question yes?")


def test_more_options_than_letters_ask_for_one_label_their_structures_indented(qwen):
    """A choice of more options than letters asks for one label, by the cue past them, each line under
    the layout's own label; a structured description and a structured state are indented JSON."""
    labels = Labels(variants=("{}",), names=(*LETTERS, "a", "b", "c", "d"))
    layout = release()
    many = QuestionLayout(max_len=layout.max_len, labels=labels, unit="whole", framing=layout.framing,
                          segments=layout.segments)
    queues = Choice("Which queue?", {f"queue{index:02d}": ({"team": index} if index == 0 else None)
                                     for index in range(30)})
    [row] = many.rows(qwen, SPECIALS, {"ticket": "charged twice"}, {"queue": queues})
    text = qwen.decode(list(row.tokens))
    assert "A. queue00: " + json.dumps({"team": 0}, indent=2) + "\nB. queue01\n" in text
    assert "a. queue26\n" in text and "Answer with one label: " + ", ".join(labels.names or ()) + "." in text
    assert "<state>\n" + json.dumps({"ticket": "charged twice"}, indent=2) + "\n</state>" in text
    [laid] = row.questions
    assert len(laid.options) == 30 and laid.label_ids == labels.ids(qwen, 30)


@pytest.mark.network
def test_more_options_than_letters_answer_with_one_of_the_releases_labels():
    """Under the release's own tokenizer (elephant-1 at its pinned commit), which merges, its prompts'
    ids and spans are its own; every one of its 256 labels spells as one token, as its runtime
    verifies them; and a choice of more options than letters names
    them by those labels and asks for one label, a structured description indented JSON and a
    structured state the same."""
    pinned = HFTokenizer(f"{ELEPHANT}@{ELEPHANT_REVISION}")
    for question, prompt in ((TEAM, CHOICE_PROMPT), (URGENCY, SCORE_PROMPT)):
        [row] = release().rows(pinned, SPECIALS, STATE, {"asked": question})
        assert list(row.tokens) == pinned.encode(prompt, add_special_tokens=False)
        [laid] = row.questions
        assert list(laid.options) == [overlapping(pinned, prompt, line) for line in lines_of(prompt, 3)]
    ids = RELEASE_LABELS.ids(pinned, len(SLOT_LABELS))
    assert all(pinned.decode(list(label)) == name for label, name in zip(ids, SLOT_LABELS, strict=True))
    queues = Choice("Which queue?", {f"queue{index:02d}": ({"team": index} if index == 0 else None)
                                     for index in range(30)})
    [row] = release().rows(pinned, SPECIALS, {"ticket": "charged twice"}, {"queue": queues})
    text = pinned.decode(list(row.tokens))
    assert "A. queue00: " + json.dumps({"team": 0}, indent=2) + "\nB. queue01\n" in text
    assert "AA. queue26\n" in text and text.endswith(
        "Answer with one label: " + ", ".join(SLOT_LABELS[:30]) + ".<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n")
    assert "<state>\n" + json.dumps({"ticket": "charged twice"}, indent=2) + "\n</state>" in text


def test_the_two_units_agree_where_the_vocabulary_never_merges():
    """Over bytes, which never merge, a row encoded whole is the row encoded piece by piece: tokens,
    spans and answer."""
    whole = release(max_len=1024)
    pieces = QuestionLayout(max_len=1024, labels=whole.labels, segments=whole.segments, unit="pieces",
                            framing=whole.framing)
    one, other = (layout.rows(ByteTokenizer(), SPECIALS, STATE, {"team": TEAM})[0]
                  for layout in (whole, pieces))
    assert one.tokens == other.tokens and one.questions == other.questions


def test_a_token_across_two_marks_belongs_to_both():
    """Encoded whole, a token that overlaps two marks' characters is in both marks' spans, as the
    release's runtime counts a token straddling two lines in both; encoded piece by piece, each mark
    holds its own piece's tokens. Laya's tiny tokenizer merges `n` and `t` into one token."""
    laya = HFTokenizer(str(LAYA), local_files_only=True)
    first, second = "n", "t"
    assert len(laya.encode(first + second, add_special_tokens=False)) == 1
    chunks = [Chunk(first, marks=(("one", 0, 1),)), Chunk(second, marks=(("two", 0, 1),))]
    merged = render(chunks, laya, "whole")
    assert len(merged.tokens) == 1 and merged.marks == {"one": (0, 1), "two": (0, 1)}
    apart = render(chunks, laya, "pieces")
    assert apart.marks == {"one": (0, 1), "two": (1, 2)}


def cutting(keep: Literal["start", "end"], unit: Literal["pieces", "whole"], room: int) -> QuestionLayout:
    layout = release()
    segments = tuple(State(render="state", keep=keep) if isinstance(segment, State) else segment
                     for segment in layout.segments or ())
    return QuestionLayout(max_len=room, labels=layout.labels, unit=unit, framing=layout.framing,
                          segments=segments)


@pytest.mark.parametrize("unit", ["whole", "pieces"])
@pytest.mark.parametrize("keep", ["start", "end"])
def test_a_row_past_its_length_cuts_its_state_where_its_segments_say(qwen, keep, unit):
    """The release never cuts its state: a row past its length does not fit (`fits`). A layout whose
    state keeps its start (or its end) keeps the longest prefix (or suffix) that fits, its options and
    cue whole. Over a vocabulary of bytes with room for 40 more tokens and a two-byte character
    across the 40th: encoded whole, the cut falls at the state's own token boundaries, so it keeps the
    39 characters before the character, never half of it; piece by piece the state is its own piece,
    cut at its ids, its first (or last) 40 kept."""
    refunds = " ".join(["refund"] * 200)
    long = "a" * 39 + "é" + refunds if keep == "start" else refunds + "é" + "b" * 39
    [empty] = release().rows(qwen, SPECIALS, "", {"team": TEAM})
    room = len(empty.tokens) + 40
    [whole] = release(max_len=room).rows(qwen, SPECIALS, long, {"team": TEAM})
    assert not release(max_len=room).fits(whole)
    layout = cutting(keep, unit, room)
    [cut] = layout.rows(qwen, SPECIALS, long, {"team": TEAM})
    assert layout.fits(cut) and cut.state_tokens == len(qwen.encode(long, add_special_tokens=False))
    text = qwen.decode(list(cut.tokens))
    assert "Answer with one letter: A, B, C." in text
    if unit == "whole":
        kept = text[text.index("<state>\n") + len("<state>\n"):text.index("\n</state>")]
        assert kept == ("a" * 39 if keep == "start" else "b" * 39) and cut.state_kept == 39
    else:
        ids = qwen.encode(long, add_special_tokens=False)
        own = ids[:40] if keep == "start" else ids[-40:]
        tokens = list(cut.tokens)
        assert cut.state_kept == 40 and any(tokens[at:at + 40] == own for at in range(len(tokens)))


def test_a_framing_carries_its_marks_through_a_trimming_template_and_refuses_one_that_rewrites(qwen):
    """A template that trims the user's turn holds its content trimmed: the marks move with it, each
    option's span still its line's tokens, a state of whitespace alone trimmed away with no span of
    its own; one that keeps the content as written keeps every span its full length, the last
    option's too. One that rewrites the content has no character map to it and is refused."""
    trimming = ("{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content | trim }}"
                "<|im_end|>\n{% endfor %}<|im_start|>assistant\n")
    layout = release()
    spaced = QuestionLayout(max_len=layout.max_len, labels=layout.labels, unit="whole",
                            framing=Framing(system=SLOT_SYSTEM, prefill="", thinking=None,
                                            chat_template=trimming),
                            segments=(Text("  "), *(layout.segments or ())))
    [row] = spaced.rows(qwen, SPECIALS, STATE, {"team": TEAM})
    prompt = qwen.decode(list(row.tokens))
    assert "<|im_start|>user\n<state>" in prompt
    [laid] = row.questions
    assert list(laid.options) == [overlapping(qwen, prompt, line) for line in lines_of(prompt, 3)]
    question_first = tuple(segment for segment in layout.segments or ()
                           if not isinstance(segment, (State, Text)))
    trailing = QuestionLayout(max_len=layout.max_len, labels=layout.labels, unit="whole",
                              framing=spaced.framing,
                              segments=(question_first[0], Text("\n\nOptions:\n"), question_first[1],
                                        Text("\n\n"), question_first[2], Text("\n"), State(render="state")))
    [blank] = trailing.rows(qwen, SPECIALS, "   ", {"team": TEAM})
    shown = qwen.decode(list(blank.tokens))
    assert "Answer with one letter: A, B, C.<|im_end|>" in shown
    assert list(blank.questions[0].options) == [overlapping(qwen, shown, line) for line in lines_of(shown, 3)]
    verbatim = trimming.replace("message.content | trim", "message.content")
    options_last = tuple(segment for segment in layout.segments or () if not isinstance(segment, AnswerCue))
    padded = QuestionLayout(max_len=layout.max_len, labels=layout.labels, unit="whole",
                            framing=Framing(system=SLOT_SYSTEM, prefill="", thinking=None,
                                            chat_template=verbatim),
                            segments=(Text("   "), *options_last[:-1]))
    [row] = padded.rows(qwen, SPECIALS, STATE, {"team": TEAM})
    prompt = qwen.decode(list(row.tokens))
    last = prompt.index("C. fraud: Suspected unauthorised use")
    expected = overlapping(qwen, prompt, (last, last + len("C. fraud: Suspected unauthorised use")))
    assert row.questions[0].options[-1] == expected
    rewriting = trimming.replace("message.content | trim", "message.content | upper")
    with pytest.raises(ValueError):
        QuestionLayout(max_len=layout.max_len, labels=layout.labels, unit="whole", segments=layout.segments,
                       framing=Framing(system=SLOT_SYSTEM, prefill="", thinking=None,
                                       chat_template=rewriting)).rows(qwen, SPECIALS, STATE, {"team": TEAM})


class Marked:
    """An encoder whose every text, the empty one too, opens with a marker id: a vocabulary whose
    encoding of nothing is not nothing."""

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return [7, *text.encode()]


def test_every_piece_is_encoded_an_empty_one_too():
    """Under `pieces` each text chunk is encoded alone, an empty one as the encoder encodes nothing,
    and an absent segment is a chunk of no ids: a context of no text is the encoder's own empty
    encoding, as a release that encodes each piece it writes reads it."""
    from dew.decision.layout import context_tokens

    assert render([Chunk(""), Chunk(ids=()), Chunk("a")], Marked(), "pieces").tokens == (7, 7, 97)
    assert list(context_tokens(Marked(), "", 8)) == [7]


def test_the_renderer_is_its_own_module_beside_the_value_renderer():
    """`dew.decision.renderer` is the row renderer, and `dew.decision.render` still renders a value."""
    import dew.decision
    import dew.decision.renderer as renderer

    assert renderer.render is render and callable(dew.decision.render)
    assert dew.decision.render({"a": 1}) != renderer.__name__


def test_a_whole_row_over_a_vocabulary_without_offsets_is_refused():
    """A vocabulary that keeps no character offsets cannot place a whole row's spans."""

    class Plain:
        bos_id = None

        def encode(self, text, *, add_special_tokens=True):
            return list(text.encode())

        def decode(self, ids):
            return bytes(ids).decode()

    with pytest.raises(TypeError):
        release(max_len=1024).rows(Plain(), SPECIALS, STATE, {"team": TEAM})


def test_a_whole_copy_over_its_cap_lays_out_whole():
    """A copy kept whole is never cut, its cap aside: a row it fits lays it out whole (row 1616)."""
    layout = QuestionLayout(max_len=1 << 12, labels=Labels(variants=("{}",)), unit="whole",
                            segments=(Text("S:"), State(render="state", keep="all", max_tokens=1),
                                      Text("\n"), *(release().segments or ())[-5:]))
    [row] = layout.rows(ByteTokenizer(), SPECIALS, "whole", {"team": TEAM})
    assert "S:whole\n" in ByteTokenizer().decode(list(row.tokens))
    assert (row.state_tokens, row.state_kept) == (5, 5)


@pytest.mark.parametrize("unit", ["pieces", "whole"])
def test_a_question_row_holds_the_state_twice_each_copy_its_own_render_and_budget(qwen, unit):
    """A question row may hold the state more than once, each copy rendered and cut on its own: one its
    text's first 12 tokens, one its compact JSON's last 9, the row's kept count its least-kept copy's
    share, under either unit."""
    layout = release()
    state: JSON = {"note": "charged twice this month"}
    segments = (Text("<state>\n"), State(render="state", max_tokens=12), Text("\n</state>\n<tail>"),
                State(render="compact", keep="end", max_tokens=9), Text("</tail>\n\n"),
                *(segment for segment in layout.segments or () if not isinstance(segment, (State, Text))))
    segments = (*segments[:5], segments[5], Text("\n\nOptions:\n"), segments[6], Text("\n\n"), segments[7])
    twice = QuestionLayout(max_len=1 << 16, labels=layout.labels, unit=unit, framing=layout.framing,
                           segments=segments)
    [row] = twice.rows(qwen, SPECIALS, state, {"team": TEAM})
    text = qwen.decode(list(row.tokens))
    indented = json.dumps(state, ensure_ascii=False, indent=2)
    compact = json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    assert "<state>\n" + indented[:12] + "\n</state>" in text
    assert "<tail>" + compact[-9:] + "</tail>" in text
    shares = [(len(indented), 12), (len(compact), 9)]
    assert (row.state_tokens, row.state_kept) == min(shares, key=lambda pair: pair[1] / pair[0])


@pytest.mark.parametrize("keep", ["start", "end"])
def test_a_capped_copy_keeps_whole_characters_within_its_cap(keep):
    """Encoded whole, a copy cut inside a character of several tokens keeps the character whole or not
    at all, its own encoding within its cap: a cap of one byte keeps no two-byte character."""
    layout = QuestionLayout(max_len=1 << 12, labels=Labels(variants=("{}",)), unit="whole",
                            segments=(Text("S:"), State(render="state", keep=keep, max_tokens=1),
                                      Text("\n"), *(release().segments or ())[-5:]))
    [row] = layout.rows(ByteTokenizer(), SPECIALS, "éé", {"team": TEAM})
    text = ByteTokenizer().decode(list(row.tokens))
    assert "S:\n" in text and (row.state_tokens, row.state_kept) == (4, 0)


def test_a_cut_takes_the_longest_state_a_merge_lets_fit():
    """Encoded whole, a cut keeps the longest state that fits, past the share counted: where a state
    token merges with the text before it, keeping it costs the row no token, and the cut keeps it."""
    laya = HFTokenizer(str(LAYA), local_files_only=True)
    segments = (Text("n"), State(render="state"), Text("\n"), *(release().segments or ())[-5:])
    empty = QuestionLayout(max_len=1 << 12, labels=Labels(variants=("{}",)), unit="whole", segments=segments)
    [bare] = empty.rows(laya, SPECIALS, "", {"team": TEAM})
    tight = dataclasses.replace(empty, max_len=len(bare.tokens))
    [row] = tight.rows(laya, SPECIALS, "ta", {"team": TEAM})
    assert tight.fits(row) and row.state_kept >= 1
