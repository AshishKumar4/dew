"""Pinned decision sources, their held-out splits, and their training requests."""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from dew.decision import Choice, Example

RECIPES = Path(__file__).parents[1] / "recipes" / "decision"


@pytest.fixture(scope="module")
def sources():
    sys.path.insert(0, str(RECIPES))
    spec = importlib.util.spec_from_file_location("decision_sources", RECIPES / "sources.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sources_hold_out_a_seeded_two_percent_with_bounds(sources, monkeypatch, tmp_path):
    question = Choice("Which?", ["a", "b"])

    def examples(source):
        count = 120 if isinstance(source, sources.Intents) else 3000
        return [Example(str(index), {"q": question}, {"q": "a"}) for index in range(count)]

    monkeypatch.setattr(sources.Intents, "examples", examples)
    monkeypatch.setattr(sources.Clinc, "examples", examples)
    default = sources.Mixture()
    off = {entry.name: dataclasses.replace(getattr(default, entry.name), weight=0.0)
           for entry in dataclasses.fields(default)}
    mixture = sources.Mixture(**{**off, "banking77": sources.Intents(seed=17), "clinc150": sources.Clinc()})
    made = sources.write(mixture, tmp_path)
    assert made["rows"] == {"banking77": {"train": 70, "held": 50},
                            "clinc150": {"train": 2940, "held": 60}}
    for name, source in mixture.sources():
        train, held = source.split()
        assert (train, held) == source.split()
        assert {row.state for row in train}.isdisjoint(row.state for row in held)
        assert held != dataclasses.replace(source, seed=source.seed + 1).split()[1]
        written = [json.loads(line) for line in (tmp_path / f"{name}.held.jsonl").read_text().splitlines()]
        assert written == [sources.row(row) for row in held]
    monkeypatch.setattr(sources.Rows, "examples", lambda self: [Example("x", {"q": question})] * 110_000)
    assert len(sources.Rows().split()[1]) == 2000
    assert len(sources.Rows(held=0, limit=3).split()[0]) == 3
    assert sources.Rows(held=0).split()[1] == []
    monkeypatch.setattr(sources.TypedDecisions, "examples", lambda self: examples(sources.Intents()))
    assert len(sources.TypedDecisions().split()[1]) == 100
    calibration = [Example("calibration", {"q": question})]
    monkeypatch.setattr(sources.OpenJev, "read", lambda self, split: calibration if split == "calibration"
                        else examples(sources.Intents()))
    assert sources.OpenJev().split()[1] == calibration
