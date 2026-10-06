"""Questions, answers and confidence, against Jev's published definitions (docs.typesafe.ai)."""

import math

import numpy as np
import pytest

from dew.decision import Choice, EntropyConfidence, JevConfidence, Noul, Question, Score, TopProbability

DEPARTMENT = Choice("Which team should handle this?",
                    criteria={"billing": "Payments", "technical": "Bugs", "sales": None})
FRUSTRATION = Score("How frustrated is the customer?", criteria=["Calm", "Frustrated", "Very angry"])
URGENT = Noul("Does this convey urgency?", criteria={"true": "Explicitly time-sensitive"})


@pytest.mark.parametrize("probabilities, expected", [
    ((0.6, 0.3, 0.1), 0.4), ((0.6, 0.2, 0.2), 0.4), ((1.0, 0.0, 0.0), 1.0), ((1 / 3, 1 / 3, 1 / 3), 0.0),
])
def test_choice_confidence_is_the_top_probability_above_an_even_split(probabilities, expected):
    """Jev's Choice formula, (p_max - 1/n) / (1 - 1/n): only the top probability counts."""
    assert JevConfidence()(DEPARTMENT, np.array(probabilities)) == pytest.approx(expected)


@pytest.mark.parametrize("probabilities, expected", [
    ((0.0, 0.5, 0.5), 0.25), ((0.5, 0.0, 0.5), 0.0), ((0.0, 0.57, 0.43), 1 - 0.43 / (2 / 3)),
    ((0.0, 1.0, 0.0), 1.0),
])
def test_score_confidence_weighs_distance_from_the_most_likely_level(probabilities, expected):
    """Jev's Score formula and its page's worked examples: torn between
    neighbours costs less than torn between the ends."""
    assert JevConfidence()(FRUSTRATION, np.array(probabilities)) == pytest.approx(expected)


def test_noul_confidence_is_the_choice_formula_over_yes_and_no():
    assert JevConfidence()(URGENT, np.array([0.1, 0.9])) == pytest.approx(0.8)
    assert JevConfidence()(URGENT, np.array([0.5, 0.5])) == pytest.approx(0.0)


def test_entropy_confidence_is_one_less_normalized_entropy():
    """Laya's `confidence_from_probs`, 1 - H(p) / log n, and its noul's top probability."""
    p = np.array([0.7, 0.2, 0.1])
    entropy = -sum(value * math.log(value) for value in p)
    assert EntropyConfidence()(DEPARTMENT, p) == pytest.approx(1 - entropy / math.log(3))
    assert EntropyConfidence()(URGENT, np.array([0.3, 0.7])) == pytest.approx(0.7)
    assert TopProbability()(DEPARTMENT, p) == pytest.approx(0.7)


def test_answers_read_their_fields_off_the_distribution():
    choice = DEPARTMENT.answer(np.array([0.4, 0.4, 0.2]), JevConfidence())
    assert choice.choice == "billing"  # the first of two equally likely options
    score = FRUSTRATION.answer(np.array([0.0, 0.95, 0.05]), JevConfidence())
    assert score.score == pytest.approx(1.05)
    assert URGENT.answer(np.array([0.05, 0.95]), JevConfidence()).noul == pytest.approx(0.95)


@pytest.mark.parametrize("question", [DEPARTMENT, FRUSTRATION, URGENT,
                                      Choice({"question": "Same person?", "candidate": {"name": "J"}},
                                             criteria=["yes", "no"])])
def test_a_question_reads_back_from_its_wire_form(question):
    assert Question.from_wire(question.wire()) == question


@pytest.mark.parametrize("build, message", [
    (lambda: Choice("pick", criteria={}), "1 to 255"),
    (lambda: Choice("pick", criteria=[f"o{index}" for index in range(256)]), "1 to 255"),
    (lambda: Choice("pick", criteria=["a", "a"]), "distinct"),
    (lambda: Score("rate", criteria=["only"]), "2 to 10"),
    (lambda: Score("rate", criteria=[str(index) for index in range(11)]), "2 to 10"),
    (lambda: Noul("yes?", criteria={"maybe": "x"}), "true and false"),
    (lambda: Noul(""), "instructions"),
    (lambda: Question.from_wire({"type": "rank", "instructions": "x"}), "noul, choice or score"),
])
def test_a_question_jev_would_refuse_is_refused(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_an_answer_label_names_an_option_by_key_or_index():
    assert DEPARTMENT.index("sales") == 2
    assert DEPARTMENT.index(1) == 1
    with pytest.raises(ValueError, match="none of"):
        DEPARTMENT.index("legal")
