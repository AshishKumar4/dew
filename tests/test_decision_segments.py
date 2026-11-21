"""A joint row's user turn is composed of segments (`dew.decision.layout.SegmentLayout.segments`): the
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
from dew.decision import Choice, Layout, Media, Noul, Schema, State, Text
from dew.decision.head import Head, JointSchemaHead
from dew.decision.layout import (
    JOINT_SEGMENTS,
    AnswerCue,
    Instructions,
    Labels,
    OptionLines,
    SegmentLayout,
    Specials,
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
    [default] = SegmentLayout.joint_row().rows(TOKENIZER, SPECIALS, state, QUESTIONS)
    assert SegmentLayout.joint_row(segments=JOINT_SEGMENTS).rows(TOKENIZER, SPECIALS, state,
                                                                 QUESTIONS) == [default]
    [first] = SegmentLayout.joint_row(segments=SCHEMA_FIRST).rows(TOKENIZER, SPECIALS, state, QUESTIONS)
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
        roomy = SegmentLayout.joint_row(max_len=1 << 20, segments=segments)
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
    SegmentLayout.joint_row(segments=(Schema(), State(), State()))
    for segments in refused:
        with pytest.raises(ValueError):
            SegmentLayout.joint_row(segments=segments)
    with pytest.raises(ValueError):
        SegmentLayout.joint_row(segments=(Text("STATE:\n"), State(), Schema())).rows(TOKENIZER, SPECIALS, "s",
                                                                                     QUESTIONS,
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
        SegmentLayout.joint_row(segments=(Schema(), State(), part))


@dataclasses.dataclass(frozen=True)
class KeptState(State):
    """A state part of a kind of its own, which a joint row renders as the state it is."""


def test_a_part_whose_kind_a_layout_renders_is_rendered_whatever_its_subclass():
    """A joint row's part of a kind of its own, a state's subclass, is built and rendered as the state
    itself, its row the same tokens."""
    [own] = SegmentLayout.joint_row(segments=(Schema(), KeptState())).rows(TOKENIZER, SPECIALS, "abc",
                                                                           QUESTIONS)
    [plain] = SegmentLayout.joint_row(segments=(Schema(), State())).rows(TOKENIZER, SPECIALS, "abc",
                                                                         QUESTIONS)
    assert own.tokens == plain.tokens


@pytest.mark.parametrize("part", [Schema(), Media()], ids=lambda part: type(part).__name__)
def test_a_question_row_refuses_a_part_it_cannot_render(part):
    """A question row holds no schema and no images: a joint row's part among its segments is refused
    when the layout is built, its own parts alone built (row 1943)."""
    own = (Instructions(HEADERS), OptionLines(), State(render="state"), AnswerCue())
    labels = Labels(variants=("{}",))
    SegmentLayout.state_first_row(max_len=256, labels=labels, segments=own)
    with pytest.raises(ValueError):
        SegmentLayout.state_first_row(max_len=256, labels=labels, segments=(*own, part))


@dataclasses.dataclass(frozen=True)
class FramedLayout(SegmentLayout):
    """A layout of a run's own package: the segment layout under another name."""


class WiderHead(JointSchemaHead):
    """A head of a run's own package: a joint head under another name."""


def test_a_layout_and_a_head_of_a_runs_own_reload_by_import_path():
    """A layout's record names its class by import path, its segments with it, so a layout class of a
    package of the run's own reloads, as does a head's; an earlier run's record, which names a head of
    Dew's by class name alone under `name`, reloads as that class once read as a run is
    (`earlier_shapes`)."""
    from dew.inference.tasks import earlier_shapes

    layout = FramedLayout(form="joint", segments=SCHEMA_FIRST, max_state_tokens=7)
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
    [row] = SegmentLayout.joint_row(segments=(Schema(),)).rows(TOKENIZER, SPECIALS, "abc", QUESTIONS)
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

    layout = SegmentLayout.joint_row(max_len=1024, segments=SCHEMA_FIRST)
    head = JointSchemaHead(32, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)
    base = DecisionObjective(tiny_backbone().clone(max_seq_len=layout.max_len), tokenizer=TOKENIZER,
                             specials=SPECIALS, layout=layout, head=head)
    task = Decide(base.model, base.init(jax.random.key(0)), layout, TOKENIZER, SPECIALS)
    _, live, reloaded = saved_and_reloaded(task, tmp_path / "run")
    assert reloaded.layout == layout
    asked = {"team": INTENT}
    np.testing.assert_array_equal(reloaded(EXAMPLES[0].state, asked)["team"].probabilities,
                                  live(EXAMPLES[0].state, asked)["team"].probabilities)
    default = dataclasses.replace(task, layout=SegmentLayout.joint_row(max_len=1024))
    objective = DecisionObjective(default)
    state = Trainer(objective, optax.adam(1e-3), key=0).fit(
        decision_rows(EXAMPLES, 8, objective=objective.data()), steps=1, log_every=100)
    record = objective.inference_record()
    assert isinstance(record, dict)
    fields = to_record(SegmentLayout.joint_row(max_len=1024), SegmentLayout)
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
    assert loaded.layout == SegmentLayout.joint_row(max_len=1024) and loaded.model.head == today.model.head
    np.testing.assert_array_equal(loaded(EXAMPLES[0].state, asked)["team"].probabilities,
                                  today(EXAMPLES[0].state, asked)["team"].probabilities)
    with pytest.raises(ValueError):
        Decide.from_run(str(tmp_path / "unread"))


def test_an_earlier_record_of_a_segmented_question_layout_loads_without_the_budgets_it_ignored(tmp_path):
    """A run of a lettered question layout of segments, saved as an earlier Dew saved it, its record
    holding the question budgets it wrote at their defaults (192 and 48) and its rows ignored, and not
    yet the row's order, loads through the pinned run boundary (`earlier_shapes`) and answers as the run
    of today's record does (row 2006); a record of today, which states the order, keeps the budgets it
    states, which a row of segments now takes (row 2007)."""
    import numpy as np
    import optax
    from decision_support import decision_rows
    from test_decision_objective import EXAMPLES, INTENT, tiny_backbone

    from dew.checkpoints import Checkpoints
    from dew.decision import Decide, DecisionObjective
    from dew.decision.head import LabelReadout
    from dew.training import Trainer

    labels = Labels(names=("A", "B", "C"), variants=("{}",))
    segments = (Instructions(HEADERS), OptionLines(), State(render="state"), AnswerCue())
    layout = SegmentLayout.state_first_row(max_len=256, labels=labels, segments=segments)
    head = LabelReadout(32, labels.ids(TOKENIZER, 3))
    objective = DecisionObjective(tiny_backbone().clone(max_seq_len=layout.max_len), tokenizer=TOKENIZER,
                                  specials=SPECIALS, layout=layout, head=head)
    state = Trainer(objective, optax.adam(1e-3), key=0).fit(
        decision_rows(EXAMPLES, 8, objective=objective.data()), steps=1, log_every=100)
    record = objective.inference_record()
    assert isinstance(record, dict)
    fields = record["layout"]["fields"]
    budgets = {"head_max_len": 192, "option_tokens": 48}
    written = {**{name: value for name, value in fields.items() if name not in ("order", "split")}, **budgets}
    earlier = {**record, "layout": {**record["layout"], "fields": written}}
    budgeted = {**record, "layout": {**record["layout"], "fields": {**fields, **budgets}}}
    for name, artifact in (("today", record), ("earlier", earlier), ("budgeted", budgeted)):
        with Checkpoints(str(tmp_path / name)) as checkpoints:
            checkpoints.save(1, state, artifact=artifact)
            checkpoints.wait()
    today, loaded, kept = (Decide.from_run(str(tmp_path / name)) for name in ("today", "earlier", "budgeted"))
    assert loaded.layout == today.layout
    assert kept.layout == dataclasses.replace(today.layout, **budgets)
    asked = {"team": INTENT}
    np.testing.assert_array_equal(loaded(EXAMPLES[0].state, asked)["team"].probabilities,
                                  today(EXAMPLES[0].state, asked)["team"].probabilities)


def test_a_whole_request_whose_row_leaves_its_state_out_is_refused():
    """A layout whose segments write no state keeps none of it, so a scoring task that answers only
    whole requests refuses a request with a state, as it refuses one whose state the row cut."""
    import jax
    from test_decision_objective import tiny_backbone

    from dew.decision import Decide, DecisionObjective

    layout = SegmentLayout.joint_row(max_len=1024, segments=(Schema(),))
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
    layout = SegmentLayout.joint_row(max_len=1 << 20, segments=twice)
    [row] = layout.rows(TOKENIZER, SPECIALS, state, QUESTIONS)
    written = text(row.tokens)
    # A string state is its own compact text (`_compact`), each byte a token of the byte vocabulary.
    assert ("FIRST:" + state[:5] + "\nLAST:" + state[-9:]) in written
    assert (row.state_tokens, row.state_kept) == (len(state), 5)
    assert from_record(Layout, to_record(layout, Layout), dtypes=False) == layout
    once = SegmentLayout.joint_row(max_len=1 << 20, segments=(State(), Schema()))
    again = SegmentLayout.joint_row(max_len=1 << 20, segments=(State(max_tokens=None), Schema()))
    assert (once.rows(TOKENIZER, SPECIALS, state, QUESTIONS)
            == again.rows(TOKENIZER, SPECIALS, state, QUESTIONS))
    with pytest.raises(ValueError):
        SegmentLayout.joint_row(segments=(State(render="state"), Schema()))


def test_a_joint_row_keeps_a_conversations_end_where_its_state_keeps_a_conversations_end():
    """A joint row's state that keeps a conversation's end (`end_if_conversation`) keeps a list state's
    last turns, as a copy kept at its end does, and a record state's start, as a copy kept at its start
    does."""
    conversation = [{"role": "user", "content": f"turn {turn}"} for turn in range(20)]
    record = {"history": list(range(40))}

    def row(keep, state):
        layout = SegmentLayout.joint_row(max_len=1 << 20, segments=(State(keep=keep, max_tokens=12),
                                                                    Schema()))
        return layout.rows(TOKENIZER, SPECIALS, state, QUESTIONS)

    assert row("end_if_conversation", conversation) == row("end", conversation) != row("start", conversation)
    assert row("end_if_conversation", record) == row("start", record) != row("end", record)


@pytest.mark.parametrize(("earlier", "fields", "now"), [
    ("MarkerLayout", {"max_len": 96, "parallel": True}, SegmentLayout.marker_row(max_len=96, parallel=True)),
    ("StateFirstLayout", {"max_len": 96}, SegmentLayout.state_first_row(max_len=96)),
    ("QuestionLayout", {"order": "re_read"}, SegmentLayout.state_first_row(order="re_read")),
    ("JointLayout", {"max_state_tokens": 7}, SegmentLayout.joint_row(max_state_tokens=7)),
    ("JointLayout", {"max_len": 2048, "segments": [{"class": "dew.decision.layout:Schema"}]},
     SegmentLayout.joint_row(max_len=2048, segments=(Schema(),)))],
    ids=["marker", "state first", "their base", "joint by default", "joint of its own"])
def test_a_record_of_an_earlier_layout_class_reads_as_the_segment_layout_it_is_now(earlier, fields, now):
    """A step's declaration that records its layout by one of the three classes an earlier Dew had, or
    their base, which wrote its rows state first, reads through the pinned boundary (`earlier_shapes`) as
    the segment layout of that row, its own fields over the defaults its class held (a joint row's 16384
    tokens, its turn's segments and framing)."""
    from dew.inference.tasks import earlier_shapes

    record = earlier_shapes({"layout": {"class": f"dew.decision.layout:{earlier}", "fields": fields}})
    assert from_record(Layout, record["layout"], dtypes=False) == now


def test_a_run_record_holding_an_earlier_layout_class_anywhere_reads_as_the_segment_layout():
    """A run's record (`run.json`) that names an earlier layout class inside a component, an objective's
    layout or a recorded stage's, reads through the run's boundary (`earlier_run`) as the segment layout of
    that row, and builds; the record's other fields, an example's data among them, as they were. Only the
    top-level declaration was read, and the nested record named a class no longer importable (the head review
    of theme 17, F2)."""
    from dew.config.earlier import earlier_run

    earlier = {"class": "dew.decision.layout:JointLayout", "fields": {"max_state_tokens": 7}}
    record = {"objective": {"class": "dew.decision.objective:DecisionObjective",
                            "fields": {"layout": earlier, "stages": [{"layout": earlier}]}},
              "examples": [{"class": "dew.decision.layout:JointLayout", "text": "an example's data"}]}
    read = earlier_run(record)
    fields = read["objective"]["fields"]
    for laid in (fields["layout"], fields["stages"][0]["layout"]):
        assert from_record(Layout, laid, dtypes=False) == SegmentLayout.joint_row(max_state_tokens=7)
    assert read["examples"] == record["examples"]


def test_the_joint_row_asks_every_question_and_refuses_a_question_rows_option_text():
    """The joint row asks every question of a request in its one row (`row_questions`), and refuses the
    described option text a row a question writes, which its schema writes otherwise: it took it and wrote
    option ids beside descriptions (the head review of theme 17, F3, F4)."""
    assert SegmentLayout.joint_row().row_questions(3) == 3
    assert SegmentLayout.state_first_row().row_questions(3) == 1
    with pytest.raises(ValueError):
        SegmentLayout.joint_row(option_text="described")


class Merging(ByteTokenizer):
    """The byte vocabulary, but for "xyz", which a row's encoding takes as one token: a longer cut of a
    state that ends in it encodes to fewer tokens than the cut before it."""

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ids = super().encode(text, add_special_tokens=add_special_tokens)
        merged, at = [], 0
        while at < len(ids):
            if ids[at:at + 3] == list(b"xyz"):
                merged.append(1)
                at += 3
            else:
                merged.append(ids[at])
                at += 1
        return merged


def test_a_cut_state_takes_a_longer_cut_whose_row_encodes_shorter_past_one_that_overflows():
    """A row whose room holds the state "aaxy" whole only as its encoding merges the state's "xy" with the
    text after it ("z"): the cut its share counted ("a", the room left by the text after it) grows past
    "aa" and "aax", which overflow, to the whole state, whose row is as short as the first cut's (row 9060:
    the growth stopped at the first longer cut that overflowed)."""
    tokenizer, state = Merging(), "aaxy"
    segments = (Text("S:"), State(), Text("z"), Schema())
    # Encoded whole, as a row merges across its segments (`unit='whole'`).
    [whole] = SegmentLayout.joint_row(max_len=1 << 20, segments=segments, unit="whole").rows(
        tokenizer, SPECIALS, state, QUESTIONS)
    [row] = SegmentLayout.joint_row(max_len=len(whole.tokens), segments=segments, unit="whole").rows(
        tokenizer, SPECIALS, state, QUESTIONS)
    # The whole state, its "xy" merged with the "z" after it: the row the room past all lays out.
    assert list(row.tokens) == list(whole.tokens) and 1 in list(row.tokens)
