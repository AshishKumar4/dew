"""A question layout's one writer in each order its own row takes (`QuestionLayout.order`): the state first,
the question first, and the state, the question and the state again, each part as the built-in row writes
it, and the row split into blocks, or kept one, as `split` says. Rows are read under GPT-2's tokenizer,
each part's tokens its own piece's."""

import dataclasses
import json

import pytest
from real_tokenizers import real_tokenizer

from dew.data.text import ByteTokenizer
from dew.data.tokens import share_tokens
from dew.decision import Choice
from dew.decision.layout import (
    AnswerCue,
    Framing,
    Instructions,
    Labels,
    MarkerLayout,
    OptionLines,
    QuestionLayout,
    Special,
    Specials,
    State,
    StateFirstLayout,
    Text,
)
from dew.registry import from_record, to_record

GPT2 = real_tokenizer("gpt2")
SPECIALS = Specials.of(GPT2)
LABELS = Labels().spelled(GPT2, 3)
QUESTION = Choice("Which team handles a lost card?", {
    "cards": "Card issues and replacements", "loans": "Loan questions", "fraud": "Suspicious use"})
CONVERSATION = [{"role": "user" if turn % 2 == 0 else "assistant",
                 "content": f"turn {turn}: where is my card?"} for turn in range(30)]


def piece(text: str) -> list[int]:
    """`text`'s tokens as the writer encodes one piece of a row."""
    return list(GPT2.encode(text, add_special_tokens=False))


def question_pieces(layout: QuestionLayout) -> tuple[list[int], list[int], list[int]]:
    """The built-in row's question as its pieces: the header and instructions, the lettered option lines,
    and the answer cue."""
    head = piece(f"choice question: {QUESTION.instructions}")
    lines = [token for name, text in zip(LABELS.names_for(3), layout.options(QUESTION), strict=True)
             for token in piece(f"\n{name}. {text}")]
    return head, lines, piece(LABELS.cue)


@pytest.mark.parametrize("order", ["state_first", "question_first", "re_read"])
def test_each_order_writes_the_built_in_rows_parts_in_that_order(order):
    """Each order's row is the begin token, the built-in question's header, lettered lines and cue, and the
    state, a separator after each part but the last, in its order; the state the request's whole, as the
    built-in row writes it."""
    layout = StateFirstLayout(labels=LABELS, order=order)
    [row] = layout.rows(GPT2, SPECIALS, "my card is lost", {"q": QUESTION})
    head, lines, cue = question_pieces(layout)
    begin, separator, state = [SPECIALS.begin], [SPECIALS.separator], piece("my card is lost")
    question = head + separator + lines
    expected = {"state_first": begin + state + separator + question + cue,
                "question_first": begin + question + separator + state + cue,
                "re_read": begin + state + separator + question + separator + state + cue}[order]
    assert list(row.tokens) == expected
    [laid] = row.questions
    start = len(begin) + (0 if order == "question_first" else len(state) + len(separator))
    assert laid.span == (start, start + len(head)) and laid.readout == len(expected) - 1
    assert row.state_kept == row.state_tokens == len(state) and not row.cuts


def test_a_run_of_either_layout_writes_the_other_layouts_own_row_in_its_order():
    """The order reaches the row a layout's own order writes: a state-first layout's question-first row is
    Laya's layout's own, and Laya's layout's state-first row a state-first layout's own, under markers and
    under labels alike."""
    for labels in (None, LABELS):
        for state in ("my card is lost", CONVERSATION):
            question_first = StateFirstLayout(labels=labels, order="question_first", max_len=96)
            state_first = MarkerLayout(labels=labels, order="state_first", max_len=96)
            assert (question_first.rows(GPT2, SPECIALS, state, {"q": QUESTION})
                    == MarkerLayout(labels=labels, max_len=96).rows(GPT2, SPECIALS, state, {"q": QUESTION}))
            own = StateFirstLayout(labels=labels, max_len=96)
            assert (state_first.rows(GPT2, SPECIALS, state, {"q": QUESTION})
                    == own.rows(GPT2, SPECIALS, state, {"q": QUESTION}))


def test_a_re_read_row_keeps_a_conversations_end_in_each_copy_of_its_share():
    """A re-read row too long for its window cuts each copy of a conversation to its share of the room the
    rest leaves (`share_tokens`), each keeping the conversation's end, the row filling its window; a
    record state keeps its start in each."""
    layout = StateFirstLayout(labels=LABELS, order="re_read", max_len=120)
    head, lines, cue = question_pieces(layout)
    begin, separator = [SPECIALS.begin], [SPECIALS.separator]
    for state, end in ((CONVERSATION, True), ({"customer": "A", "history": list(range(80))}, False)):
        ids = list(layout.state(GPT2, SPECIALS, state))
        rest = len(begin) + 3 * len(separator) + len(head) + len(lines) + len(cue)
        first, second = share_tokens([len(ids), len(ids)], layout.max_len - rest)
        assert first < len(ids)

        def kept(count: int, ids=ids, end=end) -> list[int]:
            return ids[len(ids) - count:] if end else ids[:count]

        [row] = layout.rows(GPT2, SPECIALS, state, {"q": QUESTION})
        assert list(row.tokens) == (begin + kept(first) + separator + head + separator + lines + separator
                                    + kept(second) + cue)
        assert len(row.tokens) == layout.max_len
        assert row.state_cut == 2 * len(ids) - first - second


@pytest.mark.parametrize("order", ["state_first", "question_first", "re_read"])
def test_each_order_trims_the_question_as_its_budgets_declare(order):
    """The question's declared budgets trim its instructions and options alike in every order: each row's
    question is the state-first row's own, its cuts the same, counted."""
    budgets = {"labels": LABELS, "head_max_len": 30, "option_tokens": 4}
    long = Choice("Which of the teams below handles a customer who reports that their card was lost or "
                  "stolen while they were travelling abroad last week?",
                  dict(zip(QUESTION.options, QUESTION.descriptions, strict=True)))
    [own] = StateFirstLayout(**budgets).rows(GPT2, SPECIALS, "my card is lost", {"q": long})
    [row] = StateFirstLayout(**budgets, order=order).rows(GPT2, SPECIALS, "my card is lost", {"q": long})
    assert own.options_cut > 0 and own.instructions_cut > 0
    assert row.cuts == own.cuts

    def question(encoded) -> list[int]:
        [laid] = encoded.questions
        return list(encoded.tokens[laid.span[0]:laid.options[-1][1]])

    assert question(row) == question(own)


def test_a_row_in_blocks_splits_where_its_split_says():
    """In blocks, a built-in row splits at its question's first token; `part` splits a question-first row
    at its state's, the question one block and the state the other; and `whole` keeps the row one block
    its backbone reads both ways, its tokens those of the row read in order."""
    state = "my card is lost"
    plain = StateFirstLayout(labels=LABELS).rows(GPT2, SPECIALS, state, {"q": QUESTION})[0]
    [whole] = StateFirstLayout(labels=LABELS, blocks=True, split="whole").rows(GPT2, SPECIALS, state,
                                                                              {"q": QUESTION})
    assert whole.tokens == plain.tokens and whole.blocks == (0,) * len(plain.tokens)
    [split] = StateFirstLayout(labels=LABELS, blocks=True).rows(GPT2, SPECIALS, state, {"q": QUESTION})
    start = split.questions[0].span[0]
    assert split.blocks == (0,) * start + (1,) * (len(split.tokens) - start)
    layout = StateFirstLayout(labels=LABELS, blocks=True, split="part", order="question_first")
    [part] = layout.rows(GPT2, SPECIALS, state, {"q": QUESTION})
    head, lines, _ = question_pieces(layout)
    question = 1 + len(head) + 1 + len(lines) + 1
    assert part.blocks == (0,) * question + (1,) * (len(part.tokens) - question)
    assert list(part.tokens[question:question + len(piece(state))]) == piece(state)


def test_a_layout_refuses_a_split_of_rows_read_in_order_and_an_order_beside_its_own_segments():
    """A split says where a row in blocks splits, so a layout whose rows are read in order states none;
    and an order writes the layout's own row, which segments of a row's own replace."""
    with pytest.raises(ValueError):
        StateFirstLayout(labels=LABELS, split="whole")
    segments = (Instructions({"choice": "{}"}), OptionLines(), State(), AnswerCue())
    with pytest.raises(ValueError):
        StateFirstLayout(labels=LABELS, segments=segments, order="re_read")


def test_a_layout_of_an_order_reloads_its_order_and_an_earlier_record_its_own():
    """A layout's order and split are recorded, so it reloads them; an earlier run's record, which states
    neither, reloads to the layout's own row."""
    layout = StateFirstLayout(labels=LABELS, order="re_read", blocks=True, split="whole")
    recorded = json.loads(json.dumps(to_record(layout, StateFirstLayout)))
    assert from_record(StateFirstLayout, recorded, dtypes=False) == layout
    earlier = {key: value for key, value in recorded.items() if key not in ("order", "split", "blocks")}
    assert from_record(StateFirstLayout, earlier, dtypes=False) == StateFirstLayout(labels=LABELS)


def test_a_state_of_json_null_is_written_as_any_state_is():
    """A request whose state is JSON null writes it as `null`, as any state renders, in every order, by
    rows and by its tokens alike."""
    for order in ("state_first", "question_first", "re_read"):
        layout = StateFirstLayout(labels=LABELS, order=order)
        [row] = layout.rows(GPT2, SPECIALS, None, {"q": QUESTION})
        assert list(layout.state(GPT2, SPECIALS, None)) == piece("null")
        assert row.state_tokens == len(piece("null"))
        assert layout.encode(GPT2, SPECIALS, QUESTION, piece("null"), name="q") == row


@pytest.mark.parametrize("order", ["state_first", "question_first", "re_read"])
def test_a_row_of_its_states_tokens_is_the_row_of_its_state(order):
    """A row written from the state's tokens alone (`encode`) is the row its state writes, a
    conversation's end kept in every copy under a window that cuts them."""
    layout = StateFirstLayout(labels=LABELS, order=order, max_len=120)
    [row] = layout.rows(GPT2, SPECIALS, CONVERSATION, {"q": QUESTION})
    assert row.state_cut > 0
    tokens = layout.state(GPT2, SPECIALS, CONVERSATION)
    assert layout.encode(GPT2, SPECIALS, QUESTION, tokens, conversation=True, name="q") == row


def test_a_row_of_no_state_counts_the_state_it_leaves_out_from_its_tokens_too():
    """A row whose segments write no state leaves all of it out, counted as cut, whether the row is
    written from the state or from its tokens alone."""
    layout = StateFirstLayout(labels=LABELS, segments=(Instructions({"choice": "{}"}), OptionLines(),
                                                       AnswerCue()))
    [row] = layout.rows(GPT2, SPECIALS, "my card is lost", {"q": QUESTION})
    assert row.state_tokens == row.state_cut == len(piece("my card is lost")) and row.state_kept == 0
    assert layout.encode(GPT2, SPECIALS, QUESTION, piece("my card is lost"), name="q") == row


def test_a_cut_that_leaves_options_their_labels_alone_finds_them_alike_by_the_template_each_took():
    """Options of no description are written by `bare`, and a cut to their labels leaves them alike, found
    by that template's label (`[A]`), not `line`'s (`A.`); a cut that keeps a letter of each name leaves
    them apart (the control)."""
    labels = Labels(names=("A", "B"), variants=("{}",))
    lines = OptionLines(line="{label}. {name}: {text}", bare="[{label}] {name}")
    segments = (Instructions({"choice": "{}"}), lines, State(), AnswerCue())
    bytes_specials = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
    question = {"q": Choice("Which?", ["alpha", "beta"])}

    def row(option_tokens: int):
        layout = StateFirstLayout(labels=labels, segments=segments, option_tokens=option_tokens)
        [encoded] = layout.rows(ByteTokenizer(), bytes_specials, "x", question)
        return layout, encoded

    layout, alike = row(4)
    assert alike.options_alike and layout.refusal(alike) == "options_alike"
    layout, apart = row(5)
    assert apart.options_cut > 0 and not apart.options_alike and layout.refusal(apart) is None


def test_a_framed_row_keeps_its_special_tokens_between_the_halves_of_its_turn():
    """A row of segments that writes special tokens, framed and encoded whole, holds each special token
    as its id between the halves of its turn: the framing's own, or the halves a chat template writes
    around the content it holds, as written or trimmed at its ends, the ids in their places; a begin
    token the tokenizer has none of writes nothing, as without it."""
    segments = (Special("begin"), Instructions({"choice": "{}"}), Special("separator"), OptionLines(),
                Text("\n"), State(render="state"), AnswerCue())
    plain = StateFirstLayout(labels=LABELS, segments=segments, unit="whole")
    [own] = plain.rows(GPT2, SPECIALS, "my card is lost", {"q": QUESTION})
    verbatim = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    for framing in (Framing(template=False), Framing(chat_template=verbatim)):
        framed = StateFirstLayout(labels=LABELS, segments=segments, unit="whole", framing=framing)
        [row] = framed.rows(GPT2, SPECIALS, "my card is lost", {"q": QUESTION})
        before = framing.system + ("\n\n" if not framing.template else "")
        assert row.tokens.count(SPECIALS.separator) == own.tokens.count(SPECIALS.separator) == 2
        assert list(row.tokens[:len(piece(before)) + 1]) == [*piece(before), SPECIALS.begin]
    # A template that trims the newline before the begin token writes the row it writes without it.
    trimming = "{% for message in messages %}{{ message['content'] | trim }}{% endfor %}"
    leading = StateFirstLayout(labels=LABELS, segments=(Text("\n"), *segments), unit="whole",
                               framing=Framing(chat_template=trimming))
    held = StateFirstLayout(labels=LABELS, segments=segments, unit="whole",
                            framing=Framing(chat_template=verbatim))
    assert (leading.rows(GPT2, SPECIALS, "my card is lost", {"q": QUESTION})
            == held.rows(GPT2, SPECIALS, "my card is lost", {"q": QUESTION}))
    unbegun = dataclasses.replace(SPECIALS, begin=None)
    templated = Framing(chat_template=verbatim)
    rows = [StateFirstLayout(labels=LABELS, segments=written, unit="whole", framing=templated)
            .rows(GPT2, unbegun, "my card", {"q": QUESTION}) for written in (segments, segments[1:])]
    assert rows[0] == rows[1]


def lettered_row(lines: OptionLines, question, option_tokens: int | None):
    """The byte-encoded row of `question` about "x" whose options `lines` writes under the labels A and
    B, cut to `option_tokens`."""
    segments = (Instructions({"choice": "{}"}), lines, State(), AnswerCue())
    layout = StateFirstLayout(labels=Labels(names=("A", "B"), variants=("{}",)), segments=segments,
                              option_tokens=option_tokens)
    bytes_specials = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
    [encoded] = layout.rows(ByteTokenizer(), bytes_specials, "x", question)
    return encoded


def test_a_template_that_writes_a_field_before_its_label_finds_options_alike_without_it():
    """A line whose template writes a field before its label (`{text}[{label}] {name}`) holds its options
    without the label wherever it stands: a cut that leaves two options their shared text and labels
    alone leaves them alike, one that keeps a letter of each name leaves them apart, and under a budget
    that cuts nothing the row is the row of no budget."""
    lines = OptionLines(line="{text}[{label}] {name}", bare="{name} [{label}]")
    alike = {"q": Choice("Which?", {"alpha": "same", "beta": "same"})}
    assert lettered_row(lines, alike, 8).options_alike
    assert lettered_row(lines, alike, 9).options_cut > 0 and not lettered_row(lines, alike, 9).options_alike
    bare = {"q": Choice("Which?", ["alpha", "beta"])}
    assert lettered_row(lines, bare, 64) == lettered_row(lines, bare, None)
    assert not lettered_row(lines, bare, 64).cuts


def test_a_cut_inside_a_label_leaves_options_alike():
    """A budget that ends inside an option's label (`A.` cut to `A`) holds nothing of the option past it,
    so two such options are alike; one that keeps a letter of each name past the label leaves them apart
    (the control)."""
    question = {"q": Choice("Which?", ["alpha", "beta"])}
    assert lettered_row(OptionLines(), question, 1).options_alike
    cut = lettered_row(OptionLines(), question, 4)
    assert cut.options_cut > 0 and not cut.options_alike


def test_a_requests_questions_share_one_encoding_of_its_state():
    """A request of several questions encodes its state once for them all, each row holding it as the
    request's one question would."""
    state = "my card was lost while I was travelling abroad"
    encoded: list[str] = []

    class Counting(ByteTokenizer):
        def encode(self, text, *, add_special_tokens=True):
            encoded.append(text)
            return super().encode(text, add_special_tokens=add_special_tokens)

    bytes_specials = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
    layout = StateFirstLayout()
    questions = {name: Choice(f"Which {name}?", ["a", "b"]) for name in ("first", "second", "third")}
    rows = layout.rows(Counting(), bytes_specials, state, questions)
    assert encoded.count(state) == 1
    for name, row in zip(questions, rows, strict=True):
        assert row == layout.rows(ByteTokenizer(), bytes_specials, state, {name: questions[name]})[0]
