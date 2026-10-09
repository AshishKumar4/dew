"""Pinned decision sources, their held-out splits, and their training requests."""

import csv
import dataclasses
import importlib.util
import json
import sys
from collections import Counter
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


@pytest.mark.parametrize("label", ["E", "S", "C", "I"])
def test_esci_rows_reproduce_the_kits_request(sources, label):
    example = sources.Esci.example({"query": "red shoes", "esci_label": label, "product_title": "blue shoes",
                                    "product_description": "Running shoes", "product_bullet_point": None,
                                    "product_brand": "Example", "product_color": "blue"})
    assert example.state == {"search_query": "red shoes", "product": {
        "title": "blue shoes", "description": "Running shoes", "brand": "Example", "color": "blue"}}
    assert example.questions["answer"].wire() == {
        "type": "choice",
        "instructions": "Classify the relevance of this product to the search query using the ESCI categories.",
        "criteria": {"E": "Exact: the product satisfies the search query.",
                     "S": "Substitute: a product that could substitute for the requested product.",
                     "C": "Complement: a product that complements the requested product.",
                     "I": "Irrelevant: the product does not address the requested product need."}}
    assert example.answers == {"answer": label}


def test_isarcasm_reads_csv_and_uses_the_binary_target(sources, monkeypatch, tmp_path):
    path = tmp_path / "train.En.csv"
    tweets = ['Wonderful, another "perfect" day.\nReally.', "An ordinary afternoon."]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["tweet", "sarcastic", "sarcasm"])
        writer.writeheader()
        writer.writerows([{"tweet": tweets[0], "sarcastic": "1", "sarcasm": "0"},
                          {"tweet": tweets[1], "sarcastic": "0", "sarcasm": "1"}])
    monkeypatch.setattr(sources, "_github", lambda *args: path)
    examples = sources.ISarcasm().examples()
    assert [row.state for row in examples] == tweets
    assert [row.answers for row in examples] == [{"sarcastic": "yes"}, {"sarcastic": "no"}]
    assert examples[0].questions["sarcastic"].wire() == {
        "type": "choice", "instructions": "Is this text intended to be sarcastic?",
        "criteria": {"no": "No", "yes": "Yes"}}


def test_headroom_is_opt_in_and_decontaminated_like_other_sources(sources, monkeypatch, tmp_path):
    from contamination import Evaluated, Overlaps

    default = sources.Conversion().selected()
    assert not {"esci", "isarcasm"} & dict(default.sources()).keys()
    selected = sources.Conversion(headroom=True).selected()
    assert (selected.esci.weight, selected.isarcasm.weight) == (0.02, 0.01)
    examples = [sources.ISarcasm.example({"tweet": text, "sarcastic": "1"})
                for text in ("What a wonderful day!", "Just what I needed.", "How surprising.")]
    monkeypatch.setattr(sources.Esci, "examples", lambda self: list(examples))
    monkeypatch.setattr(sources.ISarcasm, "examples", lambda self: list(examples))
    off = {entry.name: dataclasses.replace(getattr(default, entry.name), weight=0.0)
           for entry in dataclasses.fields(default)}
    mixture = sources.Mixture(**{**off, "esci": dataclasses.replace(selected.esci, held=1),
                                  "isarcasm": dataclasses.replace(selected.isarcasm, held=1)})
    overlaps = Overlaps.of(Evaluated.request(row.state, {name: q.wire() for name, q in row.questions.items()})
                           for row in examples)
    made = sources.write(mixture, tmp_path, overlaps)
    assert made["rows"] == {"esci": {"train": 0, "held": 0}, "isarcasm": {"train": 0, "held": 0}}
    assert sum(entry["dropped"] for entry in made["decontamination"].values()) == 6


@pytest.mark.network
@pytest.mark.parametrize("name", ["esci", "isarcasm"])
def test_pinned_headroom_files_have_the_released_train_counts(sources, name):
    if name == "isarcasm":
        examples = sources.ISarcasm().examples()
        assert len(examples) == 3468
        assert Counter(row.answers["sarcastic"] for row in examples) == {"yes": 867, "no": 2601}
    else:
        locales = Counter()
        for row in sources.Esci().rows():
            assert row["split"] == "train" and row["small_version"] == 1
            assert sources.Esci.example(row).answers["answer"] in ("E", "S", "C", "I")
            locales[row["product_locale"]] += 1
        assert locales == {"us": 419653, "es": 152891, "jp": 209094}
        assert sum(locales.values()) == 781638
