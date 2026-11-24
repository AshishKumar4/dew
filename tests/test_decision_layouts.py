"""What every segment layout's row holds (decision-inputs.md §1), over the three forms
(`SegmentLayout.marker_row`, `state_first_row`, `joint_row`) and a grid of their configurations, states
and questions: each question's span its instructions, each option's span its marker or its line, the state
cut to what the row leaves and counted, the question's declared budgets kept, and a position main saved
under a layout class resuming exactly under the segment layout it is now."""

import itertools
import json
from pathlib import Path

import pytest

from dew._records import JSON
from dew.data.text import ByteTokenizer
from dew.decision import Choice, Media, Noul, Schema, State, Text
from dew.decision.layout import Labels, SegmentLayout, Specials

TOKENIZER = ByteTokenizer()
SPECIALS = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)

STATES: dict[str, JSON] = {
    "short": "one two three",
    "long": " ".join(f"word{index}" for index in range(160)),
    "record": {"name": "a tool", "steps": [1, 2, 3], "done": False},
    "conversation": [{"role": "user", "content": "find the file"},
                     {"role": "assistant", "content": "which one?"},
                     {"role": "user", "content": "the newest"}],
}
QUESTIONS = {
    "one": {"pick": Choice("Which one?", ["a", "b", "c"])},
    "two": {"pick": Choice("Which one?", ["alpha", "beta"]), "holds": Noul("Does it hold?")},
    "many": {"pick": Choice("Which tool?", [f"tool {index}" for index in range(9)])},
}
# One byte each: a label is one token (`ByteTokenizer`).
LABELS = Labels(names=("A", "B", "C", "D", "E", "F", "G", "H", "I"), variants=("{}",))


def question_rows() -> dict[str, SegmentLayout]:
    """Each per-question layout configuration, by name."""
    configured: dict[str, SegmentLayout] = {}
    for kind, layout in (("marker", SegmentLayout.marker_row),
                         ("state_first", SegmentLayout.state_first_row)):
        for max_len, parallel, labels, unit in itertools.product((512, 72), (False, True), (None, LABELS),
                                                                ("pieces", "whole")):
            name = (f"{kind} {max_len} {'parallel' if parallel else 'sequential'} "
                    f"{'labels' if labels else 'markers'} {unit}")
            configured[name] = layout(max_len=max_len, parallel=parallel, labels=labels, unit=unit)
        for order in ("state_first", "question_first", "re_read"):
            configured[f"{kind} order {order}"] = layout(labels=LABELS, order=order, max_len=160)
        configured[f"{kind} option tokens"] = layout(option_tokens=3, max_len=160)
        configured[f"{kind} head max len"] = layout(head_max_len=40, max_len=160)
        configured[f"{kind} blocks"] = layout(blocks=True, max_len=160)
    return configured


def joint() -> dict[str, SegmentLayout]:
    """Each joint layout configuration, by name."""
    schema_first = (Schema(), Text("\n\nSTATE:\n"), Media(), State(render="compact"), Text("\n"))
    return {f"joint {max_len} {name}{' state ' + str(cap) if cap else ''}": SegmentLayout.joint_row(
        max_len=max_len, max_state_tokens=cap, **({} if segments is None else {"segments": segments}))
        for max_len, (name, segments), cap in itertools.product(
            (16384, 220), (("release", None), ("schema first", schema_first)), (None, 24))}


LAYOUTS = {**question_rows(), **joint()}


def text(ids) -> str:
    """The bytes `ids` spell, the tokenizer's specials and markers aside."""
    return bytes(token for token in ids if 0 < token < 253).decode("utf-8", "replace")


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_each_row_holds_its_questions_at_their_spans_and_its_state_as_far_as_it_fits(name):
    """Every row a layout writes over every state and set of questions: it fits its `max_len` or is
    refused (`refusal`); a joint row holds every question, a question row one; each question's span is
    its instructions and each option's its marker (`marker_text`, one token) or under labels its line,
    which spells its label and its option, as far as a declared budget keeps them (`head_max_len`,
    `option_tokens`), the options in the order of their slots; the state's tokens it keeps are no more than
    it holds, a cut counted where any copy kept less (`state_cut`, `state_characters_cut`), none where it
    fits whole, and a joint row's no more than its cap (`max_state_tokens`)."""
    layout = LAYOUTS[name]
    for (state_name, state), questions in itertools.product(STATES.items(), QUESTIONS.values()):
        rows = layout.rows(TOKENIZER, SPECIALS, state, questions)
        assert len(rows) == (1 if layout.joint else len(questions))
        for row in rows:
            if layout.refusal(row) is not None:
                continue
            assert len(row.tokens) <= layout.max_len
            assert row.state_kept <= row.state_tokens
            # A cut is counted, its tokens or a row encoded whole its characters, where a copy kept less.
            assert (row.state_kept < row.state_tokens) == bool(row.state_cut or row.state_characters_cut)
            if layout.joint and layout.max_state_tokens is not None:
                assert row.state_kept <= layout.max_state_tokens
            if state_name == "short" and layout.max_len >= 512:
                assert row.state_kept == row.state_tokens
            written = row.context if row.spans_in_context else row.tokens
            for laid in row.questions:
                question = questions[laid.name]
                start, end = laid.span
                assert 0 <= start < end <= len(written)
                if layout.joint:
                    assert text(written[start:end]) == question.instructions
                elif layout.head_max_len is None:
                    assert question.instructions in text(written[start:end])
                spans = list(laid.options)
                assert [span[0] for span in spans] == sorted(span[0] for span in spans)
                assert all(0 <= low < high <= len(written) for low, high in spans)
                if layout.joint:
                    named = [json.loads(text(written[low:high]))["option_id"] for low, high in spans]
                    assert len(set(named)) == len(named) == len(question.options)
                    continue
                shown = [question.options[slot] for slot in laid.order]
                for (low, high), option in zip(spans, shown, strict=True):
                    if layout.labels is None:
                        assert (high - low, written[low]) == (1, SPECIALS.marker)
                    elif layout.option_tokens is None:
                        assert option in text(written[low:high])


def test_a_budgeted_row_keeps_its_question_and_options_to_their_budgets():
    """A question row's declared budgets cut its pieces and count the cut: its options' tokens
    (`option_tokens`) and its instructions with them (`head_max_len`), the rest of the row as written."""
    questions = QUESTIONS["many"]
    for layout, field in ((SegmentLayout.marker_row(option_tokens=3, max_len=160), "options_cut"),
                          (SegmentLayout.marker_row(head_max_len=40, max_len=160), "instructions_cut")):
        whole = SegmentLayout.marker_row(max_len=160).rows(TOKENIZER, SPECIALS, STATES["short"], questions)[0]
        cut = layout.rows(TOKENIZER, SPECIALS, STATES["short"], questions)[0]
        assert getattr(cut, field) > 0
        assert len(whole.tokens) - len(cut.tokens) == cut.instructions_cut + cut.options_cut


PLANS = Path(__file__).with_name("decision_layout_plans.json")
"""The data plans main wrote, as a position holds them, for a decision chain under each of the three
layout classes (written at 54416de4b0e by `Plan.record`)."""


@pytest.mark.parametrize(("form", "built"), [
    ("marker", lambda: SegmentLayout.marker_row(max_len=96, parallel=True)),
    ("state_first", lambda: SegmentLayout.state_first_row(max_len=96)),
    ("joint", lambda: SegmentLayout.joint_row(max_state_tokens=7))])
def test_a_position_main_saved_under_a_layout_class_resumes_exactly_under_its_segment_layout(form, built):
    """A data position main saved for a decision chain under one of the three layout classes resumes
    exactly under the segment layout of that row: the plan reads its steps' layout descriptions as the
    segment layout they are now (`earlier_described`), and finds nothing changed. The steps' descriptions
    named the classes, and an exact resume of every saved position was refused (the head review of theme 17,
    F1)."""
    from dew.data.compile import compile
    from dew.data.dataset import Loading
    from dew.data.pipeline import Pipeline
    from dew.data.sources import Memory
    from dew.decision.head import DecisionHead
    from dew.decision.rows import Fit

    specials = Specials(begin=254, separator=253, marker=0, marker_text="\x00", pad=255)
    chain = Pipeline(Memory([{"id": index} for index in range(4)])).filter_map(
        Fit(built(), TOKENIZER, specials, DecisionHead(64))).batch(2)
    plan = compile(chain, read="streamed", loading=Loading(workers=0)).plan
    assert plan is not None
    assert plan.check(json.loads(PLANS.read_text())[form], resume="exact") == ()


def test_a_saved_heads_description_drops_only_the_fields_its_class_retired():
    """A position main saved describes its decision head with the attention kernel it held
    (`attention_impl`, retired by row 11938), which the plan's reading of saved descriptions leaves out
    (`dew._records.earlier_described_fields`), so it resumes exactly (above); a field its class never held
    and never retired stays, so a description naming one is refused, never read as a different step."""
    from dew._position import PlanChanged
    from dew.data.compile import compile
    from dew.data.dataset import Loading
    from dew.data.pipeline import Pipeline
    from dew.data.sources import Memory
    from dew.decision.head import DecisionHead
    from dew.decision.rows import Fit

    specials = Specials(begin=254, separator=253, marker=0, marker_text="\x00", pad=255)
    chain = Pipeline(Memory([{"id": index} for index in range(4)])).filter_map(
        Fit(SegmentLayout.joint_row(max_state_tokens=7), TOKENIZER, specials, DecisionHead(64))).batch(2)
    plan = compile(chain, read="streamed", loading=Loading(workers=0)).plan
    assert plan is not None
    saved = json.loads(PLANS.read_text())["joint"]
    held = [step for step in saved["manifest"]["steps"] if '"attention_impl"' in step["config"]]
    assert held
    for step in held:
        config = json.loads(step["config"])
        config["head"]["queries"] = 3
        step["config"] = json.dumps(config, sort_keys=True)
    with pytest.raises(PlanChanged):
        plan.check(saved, resume="exact")


def test_a_saved_plans_strings_are_read_as_json_only_where_a_step_states_its_configuration():
    """A saved plan's descriptions read without retired fields (`dew._records.earlier_described_fields`)
    decode a plan's step's configuration (`StepIdentity.config`, an entry's of its `steps`) alone: a string
    a description holds that mentions a retired field and is no JSON (a caption's template,
    `{{"stream":"{label}"}}`), a step's own field named `config` among them, and a step's declared identity
    (its configuration then too), are kept as written, where a string under any `config` was decoded and
    those raised (the API02 review)."""
    from dew._records import earlier_described_fields

    template = '{{"stream":"{label}"}}'
    head = {"class": "dew.decision.head.DecisionHead", "attention_impl": "xla", "width": 8}
    config = json.dumps({"head": head, "caption": template, "config": template}, sort_keys=True)
    # A step declaring its own identity (`Pipeline.map(identity=...)`), its configuration that identity.
    declared = '{{"attention_impl":"{label}"}}'
    saved = {"manifest": {"steps": [{"config": config, "name": "fit"},
                                    {"config": declared, "declared": declared, "name": "named"}]},
             "caption": template}
    read = earlier_described_fields(saved)
    assert read["manifest"]["steps"][1] == saved["manifest"]["steps"][1]
    assert read["caption"] == template
    head_now = {"class": "dew.decision.head.DecisionHead", "width": 8}
    assert json.loads(read["manifest"]["steps"][0]["config"]) == {
        "caption": template, "config": template, "head": head_now}
