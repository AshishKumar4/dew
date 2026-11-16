"""Byte parity of the decision layouts (decision-inputs.md §1): every row each of the three layouts writes
(`SegmentLayout.marker_row`, `state_first_row`, `joint_row`) over a grid of their configurations, states and
questions, its tokens, spans, positions, option slots, parallel positions and cuts and its refusal, digested
and held
against the digests main's renderers wrote (decision_layout_parity.json), so one segment layout configured
as each of them writes the same bytes."""

import dataclasses
import hashlib
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
DIGESTS = Path(__file__).with_name("decision_layout_parity.json")

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


def question_rows() -> dict[str, object]:
    """Each per-question layout configuration the grid writes, by name."""
    configured: dict[str, object] = {}
    for kind, layout in (("marker", SegmentLayout.marker_row), ("state_first",
                                                                SegmentLayout.state_first_row)):
        for max_len, parallel, labels, unit in itertools.product((512, 72), (False, True), (None, LABELS),
                                                                ("pieces", "whole")):
            name = (f"{kind} {max_len} {'parallel' if parallel else 'sequential'} "
                    f"{'labels' if labels else 'markers'} {unit}")
            configured[name] = (layout, {"max_len": max_len, "parallel": parallel, "labels": labels,
                                         "unit": unit})
        for order in ("state_first", "question_first", "re_read"):
            configured[f"{kind} order {order}"] = (layout, {"labels": LABELS, "order": order, "max_len": 160})
        configured[f"{kind} option tokens"] = (layout, {"option_tokens": 3, "max_len": 160})
        configured[f"{kind} head max len"] = (layout, {"head_max_len": 40, "max_len": 160})
        configured[f"{kind} blocks"] = (layout, {"blocks": True, "max_len": 160})
    return configured


def joint() -> dict[str, object]:
    """Each joint layout configuration the grid writes, by name."""
    schema_first = (Schema(), Text("\n\nSTATE:\n"), Media(), State(), Text("\n"))
    return {f"joint {max_len} {name}{' state ' + str(cap) if cap else ''}": (
        SegmentLayout.joint_row, {"max_len": max_len, **({} if segments is None else {"segments": segments}),
                      "max_state_tokens": cap})
            for max_len, (name, segments), cap in itertools.product(
                (16384, 220), (("release", None), ("schema first", schema_first)), (None, 24))}


def digest(layout_type, settings: dict) -> str:
    """Every row `layout_type(**settings)` writes over each state and question set, digested: each row's
    every field and its refusal, or the error that refused the layout or a set's rows."""
    try:
        layout = layout_type(**settings)
    except (TypeError, ValueError) as error:
        return f"refused {type(error).__name__}"
    cells = []
    for state, questions in itertools.product(STATES.values(), QUESTIONS.values()):
        try:
            rows = layout.rows(TOKENIZER, SPECIALS, state, questions)
            cells.append([(dataclasses.astuple(row), layout.refusal(row)) for row in rows])
        except (TypeError, ValueError) as error:
            cells.append(f"refused {type(error).__name__}")
    return hashlib.sha256(repr(cells).encode()).hexdigest()[:24]


def written() -> dict[str, str]:
    """The grid's digests, by configuration."""
    return {name: digest(*configured) for name, configured in {**question_rows(), **joint()}.items()}


@pytest.mark.parametrize("name", sorted({**question_rows(), **joint()}))
def test_each_layout_writes_the_bytes_mains_renderers_wrote(name):
    """Each configuration's rows, over every state and question set, digest as main's renderers' did."""
    configured = {**question_rows(), **joint()}[name]
    assert digest(*configured) == json.loads(DIGESTS.read_text())[name]


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
