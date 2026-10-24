"""A joint row's user turn is composed of segments (`dew.decision.layout.JointLayout.segments`): the
default composition writes what the Laya joint release lays out
(tests/test_decision_joint_checkpoint.py::test_the_joint_layout_lays_out_what_the_release_does), and
another writes the schema before the state, or text of its own between the parts, its spans and its
cut where its parts are. A layout and a head are recorded by import path (`dew.registry`), so one of a
package of a run's own reloads, and a record of an earlier run, which names its class alone, reloads
too."""

import dataclasses

import pytest

from dew.data.text import ByteTokenizer
from dew.decision import Choice, JointLayout, Layout, Media, Noul, Schema, State, Text
from dew.decision.head import Head, JointSchemaHead
from dew.decision.layout import JOINT_SEGMENTS, Specials
from dew.records import JSON
from dew.registry import from_record, to_record

QUESTIONS = {"pick": Choice("Which one?", ["a", "b", "c"]), "holds": Noul("Does it hold?")}
SPECIALS = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
TOKENIZER = ByteTokenizer()
SCHEMA_FIRST = (Schema(), Text("\n\nSTATE:\n"), Media(), State(), Text("\n"))


def text(tokens) -> str:
    return TOKENIZER.decode(list(tokens))


def test_the_default_segments_lay_out_the_release_rows_and_another_order_moves_the_schema_first():
    """The default segments are the release's turn, so a layout naming them lays out what the default
    does. Schema first, the schema's spans are where its questions are in the row, before the state,
    the state after it, the parts in the order the segments name them."""
    state: JSON = {"text": "one two three"}
    [default] = JointLayout().rows(TOKENIZER, SPECIALS, state, QUESTIONS)
    assert JointLayout(segments=JOINT_SEGMENTS).rows(TOKENIZER, SPECIALS, state, QUESTIONS) == [default]
    [first] = JointLayout(segments=SCHEMA_FIRST).rows(TOKENIZER, SPECIALS, state, QUESTIONS)
    written = text(first.tokens)
    assert written.index("SCHEMA FIELDS") < written.index("STATE:") < written.index("one two three")
    for laid, question in zip(first.questions, QUESTIONS.values(), strict=True):
        start, end = laid.span
        assert text(first.tokens[start:end]) == question.instructions
        assert end < written.index("STATE:")
    def options(row) -> list[str]:
        return sorted(text(row.tokens[start:end]) for laid in row.questions for start, end in laid.options)

    assert options(first) == options(default)
    assert (first.state_tokens, first.state_kept) == (default.state_tokens, default.state_kept)


def test_the_state_takes_what_the_rest_of_the_row_leaves_wherever_it_stands():
    """The schema is never cut and the state takes what the other parts leave of `max_len`,
    whichever order the segments name: a row one token too short for the whole state keeps all but
    its last token, the schema first as after."""
    state = "one two three four five six"
    for segments in (JOINT_SEGMENTS, SCHEMA_FIRST):
        roomy = JointLayout(max_len=1 << 20, segments=segments)
        [whole] = roomy.rows(TOKENIZER, SPECIALS, state, QUESTIONS)
        [cut] = dataclasses.replace(roomy, max_len=len(whole.tokens) - 1).rows(TOKENIZER, SPECIALS, state,
                                                                             QUESTIONS)
        assert cut.state_kept == whole.state_kept - 1 and len(cut.tokens) == len(whole.tokens) - 1
        assert [laid.span[1] - laid.span[0] for laid in cut.questions] == [
            laid.span[1] - laid.span[0] for laid in whole.questions]


def test_a_row_holds_the_schema_once_and_the_state_and_images_at_most_once():
    """Segments without the schema, or with two states, or two places for images, are refused when
    the layout is built; images given to segments with no place for them are refused when a row is
    laid out."""
    refused = ((State(),), (Schema(), Schema()), (Schema(), State(), State()), (Schema(), Media(), Media()))
    for segments in refused:
        with pytest.raises(ValueError):
            JointLayout(segments=segments)
    with pytest.raises(ValueError):
        JointLayout(segments=(Text("STATE:\n"), State(), Schema())).rows(TOKENIZER, SPECIALS, "s", QUESTIONS,
                                                                         media=(1, 2))


@dataclasses.dataclass(frozen=True)
class FramedLayout(JointLayout):
    """A layout of a run's own package: a joint layout under another name."""


class WiderHead(JointSchemaHead):
    """A head of a run's own package: a joint head under another name."""


def test_a_layout_and_a_head_of_a_runs_own_reload_by_import_path_and_an_earlier_runs_record_too():
    """A layout's record names its class by import path, its segments with it, so a layout class of a
    package of the run's own reloads, as does a head's; a record of an earlier run names a class of
    Dew's by name alone, and reloads as that class."""
    layout = FramedLayout(segments=SCHEMA_FIRST, max_state_tokens=7)
    assert from_record(Layout, to_record(layout, Layout), dtypes=False) == layout
    head = WiderHead(hidden_size=8)
    assert Head.from_record(head.record(), 8) == head
    recorded = JointSchemaHead(hidden_size=8).record()
    assert isinstance(recorded, dict) and recorded["name"] == "dew.decision.head:JointSchemaHead"
    earlier = {"name": "JointSchemaHead", "fields": recorded["fields"]}
    assert Head.from_record(earlier, 8) == JointSchemaHead(hidden_size=8)
