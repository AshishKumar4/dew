"""Byte parity of the decision layouts (decision-inputs.md §1): every row each of the three layouts writes
(`MarkerLayout`, `StateFirstLayout`, `JointLayout`) over a grid of their configurations, states and questions,
its tokens, spans, positions, option slots, parallel positions and cuts and its refusal, digested and held
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
from dew.decision import Choice, JointLayout, Media, Noul, Schema, State, Text
from dew.decision.layout import Labels, MarkerLayout, Specials, StateFirstLayout

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
LABELS = Labels(names=("A", "B", "C", "D", "E", "F", "G", "H", "I"))


def per_question() -> dict[str, object]:
    """Each per-question layout configuration the grid writes, by name."""
    configured: dict[str, object] = {}
    for kind, layout in (("marker", MarkerLayout), ("state_first", StateFirstLayout)):
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
        JointLayout, {"max_len": max_len, **({} if segments is None else {"segments": segments}),
                      "max_state_tokens": cap})
            for max_len, (name, segments), cap in itertools.product(
                (16384, 220), (("release", None), ("schema first", schema_first)), (None, 24))}


def digest(layout_type, settings: dict) -> dict[str, str]:
    """Each state and question set's rows under `layout_type(**settings)`, digested: their every field and
    each one's refusal, or the error that refused the layout or its rows."""
    try:
        layout = layout_type(**settings)
    except (TypeError, ValueError) as error:
        return {"layout": f"refused {type(error).__name__}"}
    digests = {}
    for (state_name, state), (questions_name, questions) in itertools.product(STATES.items(),
                                                                              QUESTIONS.items()):
        try:
            rows = layout.rows(TOKENIZER, SPECIALS, state, questions)
            written = repr([(dataclasses.astuple(row), layout.refusal(row)) for row in rows])
        except (TypeError, ValueError) as error:
            written = f"refused {type(error).__name__}"
        digests[f"{state_name} {questions_name}"] = hashlib.sha256(written.encode()).hexdigest()[:20]
    return digests


def written() -> dict[str, dict[str, str]]:
    """The grid's digests, by configuration."""
    return {name: digest(*configured) for name, configured in {**per_question(), **joint()}.items()}


@pytest.mark.parametrize("name", sorted({**per_question(), **joint()}))
def test_each_layout_writes_the_bytes_mains_renderers_wrote(name):
    """Each configuration's rows, over every state and question set, digest as main's renderers' did."""
    configured = {**per_question(), **joint()}[name]
    assert digest(*configured) == json.loads(DIGESTS.read_text())[name]
