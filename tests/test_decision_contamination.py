"""recipes/decision/contamination.py: what makes a training example a repeat of an evaluation item."""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

from dew.decision import Choice, DecisionTable, Example

RECIPES = Path(__file__).parents[1] / "recipes" / "decision"


@pytest.fixture(scope="module")
def contamination():
    sys.path.insert(0, str(RECIPES))
    spec = importlib.util.spec_from_file_location("decision_contamination", RECIPES / "contamination.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


INSTRUCTIONS = ("Read the customer's message carefully and decide which team of the support desk "
                "should take this ticket first")
"""One question every row of a table is asked: long enough to be compared, were it compared."""


def _table(path: Path, texts: list[tuple[str, str]]) -> list[Example]:
    """`texts` as a DecisionTable reads them, one schema asked of every row."""
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        writer.writerows(texts)
    examples, _ = DecisionTable(path=str(path), question="team", instructions=INSTRUCTIONS,
                                options=("billing", "technical", "sales"), held_out=0.0).examples()
    return examples


def test_a_shared_question_is_not_a_repeat_but_a_shared_decision_is(contamination, tmp_path):
    """A table asks every row the same question over the same options, so a
    training split and its test split share them by design: rows of other
    states are all kept. A training row that repeats a test row's state with
    its question and options is the same decision, and is dropped however
    short its state, in either option order or option spelling."""
    test = _table(tmp_path / "test.csv", [("refund please", "billing"), ("site is down", "technical"),
                                         ("how much is the pro plan this year", "sales")])
    evaluated = contamination.Overlaps.of(
        contamination.Evaluated.request(example.state, {name: question.wire()
                                                        for name, question in example.questions.items()})
        for example in test)
    train = _table(tmp_path / "train.csv", [("charged twice", "billing"), ("error 500 on login", "technical"),
                                           ("can I upgrade", "sales"), ("my card was declined", "billing")])
    assert evaluated.keep("table", train, natural=True) == train

    repeated = _table(tmp_path / "repeat.csv", [("refund please", "billing")])
    assert evaluated.keep("repeat", repeated, natural=True) == []
    listed = Example("Refund, please!", {"team": Choice(INSTRUCTIONS, ["sales", "billing", "technical"])},
                     {"team": "billing"})
    assert evaluated.hits(listed)
    # The same short state asked another question is another decision.
    other = Example("refund please", {"team": Choice("Which team?", ["billing", "technical", "sales"])})
    assert not evaluated.hits(other)
    assert evaluated.report == {"table": {"checked": 4, "dropped": 0}, "repeat": {"checked": 1, "dropped": 1}}


def test_content_shared_with_an_evaluation_item_is_a_repeat(contamination):
    """Beyond exact decisions, a training state that is an evaluation state, or
    a line of one, or shares 13 words running with one, is dropped whatever it
    is asked; a synthetic set's state only when it is the whole state."""
    passage = ("The committee met on Tuesday and agreed that the new library would open in the spring "
               "after the last of the shelves had been delivered")
    evaluated = contamination.Overlaps.of([contamination.Evaluated.request(passage, {})])
    asked = {"q": Choice("Which season?", ["spring", "autumn"])}
    assert evaluated.hits(Example(passage.upper() + "!", asked))
    assert evaluated.hits(Example("Background.\n" + passage, asked))
    assert evaluated.hits(Example("Notes: " + passage[40:], asked))
    assert not evaluated.hits(Example("Notes: " + passage[40:], asked), natural=False)
    assert not evaluated.hits(Example("The committee met on Tuesday", asked))
