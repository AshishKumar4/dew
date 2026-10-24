"""One renderer of decision rows (`dew.decision.render`, docs/design/decision-render.md): a row's
chunks encoded piece by piece or whole, each mark's tokens the ones that overlap its characters, a
token across two marks both's; and a published slot release's prompt as a layout value
(`QuestionLayout(segments=...)`), its rows the release's own text and ids (qev/prompt.py at OneJev
81ce62f, rendered as its engine renders them), its options' spans the lines its runtime pools, its
answer at its last token, and its state cut only where its segments say."""

import json
from pathlib import Path
from typing import Literal

import pytest

from dew.data.text import ByteTokenizer, HFTokenizer
from dew.decision import Choice, Noul, Score
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
from dew.decision.render import Chunk, render

QWEN = Path(__file__).parent / "fixtures" / "hf" / "qwen38-dense-tiny"
"""Qwen3.5's tokenizer and ChatML template, tiny, with its thinking switch."""

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
    token, and each label the bare letter's one id."""
    [row] = release().rows(qwen, SPECIALS, STATE, {"asked": question})
    assert list(row.tokens) == qwen.encode(prompt, add_special_tokens=False)
    [laid] = row.questions
    assert list(laid.options) == [overlapping(qwen, prompt, line) for line in lines_of(prompt, 3)]
    assert laid.readout == len(row.tokens) - 1
    assert laid.label_ids == tuple((qwen.encode(label, add_special_tokens=False)[0],) for label in "ABC")


def test_a_noul_asks_yes_first_and_more_options_than_letters_answer_with_one_label(qwen):
    """A noul's options are yes, then no, its true option first (`Laid.order`), each with the
    release's own description where its question gives none; a choice of more options than letters
    names them by the release's labels and asks for one label, its lines' structured descriptions
    indented JSON."""
    [row] = release().rows(qwen, SPECIALS, STATE, {"holds": Noul("Was the customer charged twice?")})
    [laid] = row.questions
    text = qwen.decode(list(row.tokens))
    assert laid.order == (1, 0)
    assert ("A. yes: the statement is true / the answer is yes\nB. no: the statement is false / the answer "
            "is no\n\nAnswer with one letter: A, B.") in text
    queues = Choice("Which queue?", {f"queue{index:02d}": ({"team": index} if index == 0 else None)
                                     for index in range(30)})
    [row] = release().rows(qwen, SPECIALS, {"ticket": "charged twice"}, {"queue": queues})
    text = qwen.decode(list(row.tokens))
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


def test_a_token_across_two_marks_belongs_to_both(qwen):
    """Encoded whole, a token that overlaps two marks' characters is in both marks' spans, as the
    release's runtime counts a token straddling two lines in both; encoded piece by piece, each mark
    holds its own piece's tokens."""
    merges = json.loads((QWEN / "tokenizer.json").read_text())["model"]["merges"]
    pairs = (merge.split(" ") if isinstance(merge, str) else merge for merge in merges)
    first, second = next((a, b) for a, b in pairs if len(qwen.encode(a + b, add_special_tokens=False)) == 1
                         and len(a) == 1 and len(b) == 1)
    chunks = [Chunk(first, marks=(("one", 0, 1),)), Chunk(second, marks=(("two", 0, 1),))]
    merged = render(chunks, qwen, "whole")
    assert len(merged.tokens) == 1 and merged.marks == {"one": (0, 1), "two": (0, 1)}
    apart = render(chunks, qwen, "pieces")
    assert apart.marks == {"one": (0, 1), "two": (1, 2)}


def test_a_row_past_its_length_cuts_its_state_where_its_segments_say(qwen):
    """The release never cuts its state: a row past its length does not fit (`fits`). A layout whose
    state keeps its start cuts the state's end at its own token boundaries to the longest that fits,
    the row encoded whole, and its options and cue whole."""
    long = " ".join(["refund"] * 200)
    [whole] = release(max_len=200).rows(qwen, SPECIALS, long, {"team": TEAM})
    assert not release(max_len=200).fits(whole)
    cutting = release(max_len=200, keep="start")
    [cut] = cutting.rows(qwen, SPECIALS, long, {"team": TEAM})
    assert cutting.fits(cut) and cut.state_kept < cut.state_tokens
    text = qwen.decode(list(cut.tokens))
    assert "<state>\nrefund refund" in text and "Answer with one letter: A, B, C." in text
    [longer] = release(max_len=len(cut.tokens) + 8, keep="start").rows(qwen, SPECIALS, long, {"team": TEAM})
    assert longer.state_kept >= cut.state_kept


def test_a_framing_carries_its_marks_through_a_trimming_template_and_refuses_one_that_rewrites(qwen):
    """A template that trims the user's turn holds its content trimmed: the marks move with it, each
    option's span still its line's tokens. One that rewrites the content has no character map to it
    and is refused."""
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
    rewriting = trimming.replace("message.content | trim", "message.content | upper")
    with pytest.raises(ValueError):
        QuestionLayout(max_len=layout.max_len, labels=layout.labels, unit="whole", segments=layout.segments,
                       framing=Framing(system=SLOT_SYSTEM, prefill="", thinking=None,
                                       chat_template=rewriting)).rows(qwen, SPECIALS, STATE, {"team": TEAM})


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
