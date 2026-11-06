"""A released decision model's prompt as a declaration (row 4482, theme DR01): the segments and value
policies a release's rows need beyond the slot release's, each one more part or policy of the one renderer
(docs/design/decision-inputs.md §1 rejects a layout class a release): a summary of the request in each
question's row, a question written again where the state is long, a conversation written as turns, and a
long list indexed, an empty value substituted and option names normalized."""

import dataclasses
import json

import pytest
from test_decision_render import QWEN, SPECIALS, STATE, TEAM, URGENCY

from dew.data.text import HFTokenizer
from dew.decision import Choice
from dew.decision.layout import Instructions, OptionLines, QuestionLayout, State, Text
from dew.decision.slot_source import SlotSource
from dew.registry import from_record, to_record

release = SlotSource.layout


@pytest.fixture(scope="module")
def qwen() -> HFTokenizer:
    return HFTokenizer(str(QWEN), local_files_only=True)


def layout_of(segments, *, unit="whole", max_len=None) -> QuestionLayout:
    """The slot release's layout with `segments` for its own: its framing, labels and length."""
    base = release()
    return QuestionLayout(max_len=base.max_len if max_len is None else max_len, labels=base.labels, unit=unit,
                          framing=base.framing, segments=tuple(segments))


def text_of(qwen, row) -> str:
    return qwen.decode(list(row.tokens))


def test_a_summary_of_the_request_is_written_in_each_questions_row(qwen):
    """Each question's row holds a line for every question the request asks, in its order, the row's own by
    `current` where that is given; the question's spans are its own parts' alone, as the summary is text no
    span reads, and a layout with a summary lays no question alone (`encode` has one)."""
    from dew.decision.layout import RequestSummary

    first, state, closing, *question = release().segments
    summary = RequestSummary(line="{index}. {name} ({kind}): {instructions}",
                             current="> {index}. {name} ({kind}): {instructions}")
    layout = layout_of((first, state, closing, summary, Text("\n\n"), *question))
    rows = layout.rows(qwen, SPECIALS, STATE, {"team": TEAM, "urgency": URGENCY})
    lines = ("1. team (choice): Which team should handle this?",
             "2. urgency (score): How urgent is this?")
    own = [(f"> {lines[0]}\n{lines[1]}"), (f"{lines[0]}\n> {lines[1]}")]
    assert [expected in text_of(qwen, row) for row, expected in zip(rows, own, strict=True)] == [True, True]
    team = rows[0]
    [laid] = team.questions
    assert [qwen.decode(list(team.tokens[start:end])) for start, end in laid.options] == [
        "A. billing: Charges, refunds, invoices", "B. technical: App or site faults",
        "C. fraud: Suspected unauthorised use"]
    [alone] = layout.rows(qwen, SPECIALS, STATE, {"team": TEAM})
    assert "1. team (choice)" in text_of(qwen, alone) and "2." not in text_of(qwen, alone)
    with pytest.raises(ValueError):
        layout.encode(qwen, SPECIALS, TEAM, qwen.encode(STATE, add_special_tokens=False), name="team")


def question_first(over: int) -> QuestionLayout:
    """The question, the state, then the question once more where the state holds `over` tokens or more."""
    from dew.decision.layout import Again

    _, state, _, *question = release().segments
    asked, options_text, options, _, cue = question
    return layout_of((asked, options_text, options, Text("\n\n"), state, Text("\n\n"),
                      Again(over=over), Text("\n\n"), cue))


def test_a_question_is_written_again_where_the_state_is_long_enough_to_forget_it(qwen):
    """A repetition of the question's instructions and options, after a state of at least `over` tokens
    (its longest copy, before any cut), is text beside the row's own: the spans, the readout and the
    option groups are those of the question asked once, and a state a token short of `over` asks it once."""
    long_enough = len(qwen.encode(STATE, add_special_tokens=False))
    [again] = question_first(long_enough).rows(qwen, SPECIALS, STATE, {"team": TEAM})
    [once] = question_first(long_enough + 1).rows(qwen, SPECIALS, STATE, {"team": TEAM})
    twice, single = text_of(qwen, again), text_of(qwen, once)
    for part in ("Question: Which team should handle this?", "A. billing: Charges, refunds, invoices"):
        assert (twice.count(part), single.count(part)) == (2, 1)
    [laid] = again.questions
    first_options = [qwen.decode(list(again.tokens[start:end])) for start, end in laid.options]
    assert first_options == ["A. billing: Charges, refunds, invoices", "B. technical: App or site faults",
                             "C. fraud: Suspected unauthorised use"]
    assert qwen.decode(list(again.tokens[laid.span[0]:laid.span[1]])) == "Which team should handle this?"
    assert laid.readout == len(again.tokens) - 1
    assert again.tokens[laid.options[0][0]:laid.options[0][1]] == once.tokens[
        once.questions[0].options[0][0]:once.questions[0].options[0][1]]


CONVERSATION = [{"role": "user", "content": "I was charged twice."},
                {"role": "assistant", "content": "Which card?"},
                {"role": "user", "content": "The visa."}]


def turned(**given) -> QuestionLayout:
    from dew.decision.layout import Turns

    turns = Turns(open="<|{role}|>\n", close="\n", roles={"assistant": "bot"})
    segments = [State(turns=turns, **given) if isinstance(segment, State) else segment
                for segment in release().segments]
    return layout_of(segments)


def test_a_conversation_state_is_written_turn_by_turn_in_the_releases_markup(qwen):
    """A state written as turns is each turn's markup around its content, a role renamed where the release
    names it so; an empty conversation is nothing, or its substitute (`Values.empty`), and a state that is
    no list of turns is refused."""
    from dew.decision.layout import Values

    [row] = turned().rows(qwen, SPECIALS, CONVERSATION, {"team": TEAM})
    assert ("<state>\n<|user|>\nI was charged twice.\n<|bot|>\nWhich card?\n<|user|>\nThe visa.\n\n</state>"
            in text_of(qwen, row))
    [bare] = turned().rows(qwen, SPECIALS, [], {"team": TEAM})
    assert "<state>\n\n</state>" in text_of(qwen, bare)
    [filled] = turned(values=Values(empty="(no turns yet)")).rows(qwen, SPECIALS, [], {"team": TEAM})
    assert "<state>\n(no turns yet)\n</state>" in text_of(qwen, filled)
    with pytest.raises(ValueError):
        turned().rows(qwen, SPECIALS, "a plain state", {"team": TEAM})


def test_a_long_list_is_written_indexed_and_an_empty_state_is_substituted(qwen):
    """A state of more entries than `Values.indexed` is one line each, `item` filled with the entry's index
    from `start`; one of fewer, or any other state, as its `render` writes it; an empty state, an empty
    string among them, its substitute."""
    from dew.decision.layout import Values

    values = Values(indexed=2, start=1, empty="(none)")
    layout = layout_of([State(render="state", values=values) if isinstance(segment, State) else segment
                        for segment in release().segments])

    def written(state) -> str:
        [row] = layout.rows(qwen, SPECIALS, state, {"team": TEAM})
        return text_of(qwen, row)

    assert "<state>\n[1] alpha\n[2] beta\n[3] gamma\n</state>" in written(["alpha", "beta", "gamma"])
    assert json.dumps(["alpha", "beta"], indent=2) in written(["alpha", "beta"])
    assert "<state>\n(none)\n</state>" in written([]) and "<state>\n(none)\n</state>" in written("  ")


@pytest.mark.parametrize(("policy", "lines"), [
    ("as_is", ["A. billing_team: Charges", "B. TechSupport: Faults", "C. fraud-desk: Misuse"]),
    ("words", ["A. billing team: Charges", "B. Tech Support: Faults", "C. fraud desk: Misuse"]),
    ("lower", ["A. billing team: Charges", "B. tech support: Faults", "C. fraud desk: Misuse"])])
def test_option_names_are_written_as_the_release_writes_them(qwen, policy, lines):
    """An option named `billing_team`, `TechSupport` or `fraud-desk` is written as it is, as words (split at
    its separators and camel-case humps), or those in lower case (`Values.names`); its key, which the
    answer names it by, is its own."""
    from dew.decision.layout import Values

    queue = Choice("Which queue?",
                   {"billing_team": "Charges", "TechSupport": "Faults", "fraud-desk": "Misuse"})
    layout = layout_of([dataclasses.replace(segment, values=Values(names=policy))
                        if isinstance(segment, OptionLines) else segment for segment in release().segments])
    [row] = layout.rows(qwen, SPECIALS, STATE, {"queue": queue})
    [laid] = row.questions
    assert [qwen.decode(list(row.tokens[start:end])) for start, end in laid.options] == lines
    assert queue.options == ("billing_team", "TechSupport", "fraud-desk")


def test_a_row_writes_its_question_once_unless_a_segment_says_again():
    """A question row holds its instructions and its options once each, its cue at most once, and writes
    one more of them as an `Again`; the parts of a segment that its text cannot fill are refused."""
    from dew.decision.layout import Again, RequestSummary, Turns, Values

    base = release().segments
    instructions = next(segment for segment in base if isinstance(segment, Instructions))
    with pytest.raises(ValueError):
        layout_of((*base, instructions))
    for refused in (lambda: Again(parts=()), lambda: Again(over=-1), lambda: Values(indexed=0),
                    lambda: Values(item="{label}"), lambda: RequestSummary(line="{label}"),
                    lambda: Turns(open="{who}"), lambda: State(turns=Turns(), render="state")):
        with pytest.raises(ValueError):
            refused()


def test_the_new_segments_are_records_and_a_layout_without_them_records_none_of_them():
    """A layout with each new segment and policy reads back from its record equal; a record written
    before them, which holds none of their fields, reads as the layout it was (the slot release's)."""
    from dew.decision.layout import Again, RequestSummary, Turns, Values

    *_, instructions, options_text, options, _, cue = release().segments
    layout = layout_of((RequestSummary(values=Values(names="words")), Text("\n"),
                        State(turns=Turns(roles={"assistant": "bot"}), values=Values(empty="(none)")),
                        instructions, options_text, dataclasses.replace(options, values=Values(indexed=3)),
                        Again(over=100), cue))
    assert from_record(QuestionLayout, to_record(layout, QuestionLayout)) == layout
    def earlier(record):
        """`record` as a run saved before the new fields wrote it: none of them."""
        if isinstance(record, dict):
            return {key: earlier(value) for key, value in record.items() if key not in ("values", "turns")}
        return [earlier(value) for value in record] if isinstance(record, list) else record

    assert from_record(QuestionLayout, earlier(to_record(release(), QuestionLayout))) == release()
