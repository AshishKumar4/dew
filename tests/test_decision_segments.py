"""A joint row's user turn is composed of segments (`dew.decision.layout.JointLayout.segments`): the
default composition writes what the Laya joint release lays out
(tests/test_decision_joint_checkpoint.py::test_the_joint_layout_lays_out_what_the_release_does), and
another writes the schema before the state, or text of its own between the parts, its spans and its
cut where its parts are. A layout and a head are recorded by import path (`dew.registry`), so one of a
package of a run's own reloads, and a record of an earlier run, which names its class alone, reloads
too."""

import dataclasses

import pytest

from dew._records import JSON
from dew.data.text import ByteTokenizer
from dew.decision import Choice, JointLayout, Layout, Media, Noul, Schema, State, Text
from dew.decision.head import Head, JointSchemaHead
from dew.decision.layout import (
    JOINT_SEGMENTS,
    AnswerCue,
    Instructions,
    Labels,
    OptionLines,
    Specials,
    StateFirstLayout,
)
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


def test_a_row_holds_the_schema_once_and_the_images_at_most_once():
    """Segments without the schema, or with two places for images, are refused when the layout is
    built, and two states are not; images given to segments with no place for them are refused when a
    row is laid out."""
    refused = ((State(),), (Schema(), Schema()), (Schema(), Media(), Media()))
    JointLayout(segments=(Schema(), State(), State()))
    for segments in refused:
        with pytest.raises(ValueError):
            JointLayout(segments=segments)
    with pytest.raises(ValueError):
        JointLayout(segments=(Text("STATE:\n"), State(), Schema())).rows(TOKENIZER, SPECIALS, "s", QUESTIONS,
                                                                         media=(1, 2))


HEADERS = dict.fromkeys(("choice", "noul", "score"), "Q: {}\n")
"""A question row's header of each kind, its instructions in its place."""


@pytest.mark.parametrize("part", [Instructions(HEADERS), OptionLines(), AnswerCue()],
                         ids=lambda part: type(part).__name__)
def test_a_joint_row_refuses_a_part_it_cannot_render(part):
    """A joint row writes its questions, their instructions and options, in its schema and reads no cue:
    a question row's part among its segments is refused when the layout is built, never rendered as
    nothing (row 1943)."""
    with pytest.raises(ValueError):
        JointLayout(segments=(Schema(), State(), part))


@pytest.mark.parametrize("part", [Schema(), Media()], ids=lambda part: type(part).__name__)
def test_a_question_row_refuses_a_part_it_cannot_render(part):
    """A question row holds no schema and no images: a joint row's part among its segments is refused
    when the layout is built, its own parts alone built (row 1943)."""
    own = (Instructions(HEADERS), OptionLines(), State(render="state"), AnswerCue())
    labels = Labels(variants=("{}",))
    StateFirstLayout(max_len=256, head_max_len=64, option_tokens=8, labels=labels, segments=own)
    with pytest.raises(ValueError):
        StateFirstLayout(max_len=256, head_max_len=64, option_tokens=8, labels=labels, segments=(*own, part))


@dataclasses.dataclass(frozen=True)
class FramedLayout(JointLayout):
    """A layout of a run's own package: a joint layout under another name."""


class WiderHead(JointSchemaHead):
    """A head of a run's own package: a joint head under another name."""


def test_a_layout_and_a_head_of_a_runs_own_reload_by_import_path():
    """A layout's record names its class by import path, its segments with it, so a layout class of a
    package of the run's own reloads, as does a head's; an earlier run's record, which names a head of
    Dew's by class name alone under `name`, reloads as that class once read as a run is
    (`earlier_shapes`)."""
    from dew.inference.tasks import earlier_shapes

    layout = FramedLayout(segments=SCHEMA_FIRST, max_state_tokens=7)
    assert from_record(Layout, to_record(layout, Layout), dtypes=False) == layout
    head = WiderHead(hidden_size=8)
    assert Head.from_record(head.record(), 8) == head
    recorded = JointSchemaHead(hidden_size=8).record()
    assert isinstance(recorded, dict) and recorded["class"] == "dew.decision.head:JointSchemaHead"
    for name in ("JointSchemaHead", "dew.decision.head:JointSchemaHead"):
        earlier = earlier_shapes({"head": {"name": name, "fields": recorded["fields"]}})
        assert Head.from_record(earlier["head"], 8) == JointSchemaHead(hidden_size=8)


def test_a_row_without_a_state_segment_keeps_none_of_the_state():
    """Segments without the state write none of it, and the row says so: none of the state kept, so
    a request whose state the row leaves out reads as cut."""
    [row] = JointLayout(segments=(Schema(),)).rows(TOKENIZER, SPECIALS, "abc", QUESTIONS)
    assert (row.state_tokens, row.state_kept) == (3, 0)
    assert "abc" not in text(row.tokens)


def test_a_saved_run_reloads_its_segments_and_an_earlier_runs_record_reads_as_before(tmp_path):
    """A run of a joint layout of segments of its own reloads them from its record, its task answering
    as the live one does; a run saved as an earlier Dew saved it, its record naming its layout and
    head under "name" by class name alone and holding no segments, loads through the pinned run
    boundary (`run_record`) as the default composition and the head it named, answering as the run
    of today's record does. A head record of another shape there is refused."""
    import jax
    import numpy as np
    import optax
    from decision_support import decision_rows
    from test_decision_objective import EXAMPLES, INTENT, saved_and_reloaded, tiny_backbone

    from dew.checkpoints import Checkpoints
    from dew.decision import Decide, DecisionObjective
    from dew.training import Trainer

    layout = JointLayout(max_len=1024, segments=SCHEMA_FIRST)
    head = JointSchemaHead(32, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)
    base = DecisionObjective(tiny_backbone().clone(max_seq_len=layout.max_len), tokenizer=TOKENIZER,
                             specials=SPECIALS, layout=layout, head=head)
    task = Decide(base.model, base.init(jax.random.key(0)), layout, TOKENIZER, SPECIALS)
    _, live, reloaded = saved_and_reloaded(task, tmp_path / "run")
    assert reloaded.layout == layout
    asked = {"team": INTENT}
    np.testing.assert_array_equal(reloaded(EXAMPLES[0].state, asked)["team"].probabilities,
                                  live(EXAMPLES[0].state, asked)["team"].probabilities)
    default = dataclasses.replace(task, layout=JointLayout(max_len=1024))
    objective = DecisionObjective(default)
    state = Trainer(objective, optax.adam(1e-3), key=0).fit(
        decision_rows(EXAMPLES, 8, objective=objective.data()), steps=1, log_every=100)
    record = objective.inference_record()
    assert isinstance(record, dict)
    fields = to_record(JointLayout(max_len=1024), JointLayout)
    assert isinstance(fields, dict)
    laid = {name: value for name, value in fields.items() if name not in ("segments", "framing")}
    earlier = {**record, "head": {"name": "JointSchemaHead", "fields": record["head"]["fields"]},
               "layout": {"name": "JointLayout", "fields": laid}}
    for name, artifact in (("today", record), ("earlier", earlier),
                           ("unread", {**earlier, "head": {**earlier["head"], "kind": "joint"}})):
        with Checkpoints(str(tmp_path / name)) as checkpoints:
            checkpoints.save(1, state, artifact=artifact)
            checkpoints.wait()
    today, loaded = (Decide.from_run(str(tmp_path / name)) for name in ("today", "earlier"))
    assert loaded.layout == JointLayout(max_len=1024) and loaded.model.head == today.model.head
    np.testing.assert_array_equal(loaded(EXAMPLES[0].state, asked)["team"].probabilities,
                                  today(EXAMPLES[0].state, asked)["team"].probabilities)
    with pytest.raises(ValueError):
        Decide.from_run(str(tmp_path / "unread"))


def test_a_whole_request_whose_row_leaves_its_state_out_is_refused():
    """A layout whose segments write no state keeps none of it, so a scoring task that answers only
    whole requests refuses a request with a state, as it refuses one whose state the row cut."""
    import jax
    from test_decision_objective import tiny_backbone

    from dew.decision import Decide, DecisionObjective

    layout = JointLayout(max_len=1024, segments=(Schema(),))
    head = JointSchemaHead(32, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)
    base = DecisionObjective(tiny_backbone().clone(max_seq_len=layout.max_len), tokenizer=TOKENIZER,
                             specials=SPECIALS, layout=layout, head=head)
    task = Decide(base.model, base.init(jax.random.key(0)), layout, TOKENIZER, SPECIALS)
    scoring, _ = task.scoring(whole=True)
    asked = {name: question.wire() for name, question in QUESTIONS.items()}
    body = {"state": "one two three", "questions": asked}
    with pytest.raises(ValueError):
        scoring.admit(body)


def test_copies_of_the_state_share_the_room_as_any_rows_segments_do():
    """A row's copies of the state share its room as any row's segments do (`share_tokens`): each its
    cap, its own or the joint layout's, a whole copy all it holds (row 1616)."""
    from dew.data.tokens import share_tokens
    from dew.decision.layout import allocated

    copies = [State(max_tokens=4), State(keep="all", max_tokens=2), State()]
    assert allocated([10, 6, 10], copies, 12, default_cap=5) == share_tokens(
        [10, 6, 10], 12, caps=[4, 2, 5], whole=[False, True, False]) == [3, 6, 3]


def test_a_joint_row_holds_the_state_twice_each_copy_cut_to_its_own_budget():
    """A joint row may hold the state more than once: each copy cut from the whole state to its own
    budget and keeping its own end, the row's kept count its least-kept copy's; a layout of one state
    lays its rows out as it did, and its record reloads it."""
    state = "one two three four five six seven"
    twice = (Text("FIRST:"), State(max_tokens=5), Text("\nLAST:"), State(keep="end", max_tokens=9), Schema())
    layout = JointLayout(max_len=1 << 20, segments=twice)
    [row] = layout.rows(TOKENIZER, SPECIALS, state, QUESTIONS)
    written = text(row.tokens)
    # A string state is its own compact text (`_compact`), each byte a token of the byte vocabulary.
    assert ("FIRST:" + state[:5] + "\nLAST:" + state[-9:]) in written
    assert (row.state_tokens, row.state_kept) == (len(state), 5)
    assert from_record(Layout, to_record(layout, Layout), dtypes=False) == layout
    once = JointLayout(max_len=1 << 20, segments=(State(), Schema()))
    again = JointLayout(max_len=1 << 20, segments=(State(max_tokens=None), Schema()))
    assert (once.rows(TOKENIZER, SPECIALS, state, QUESTIONS)
            == again.rows(TOKENIZER, SPECIALS, state, QUESTIONS))
    with pytest.raises(ValueError):
        JointLayout(segments=(State(render="state"), Schema()))
