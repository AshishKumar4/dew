"""Questions, answers and confidence, against Jev's published definitions (docs.typesafe.ai)."""


import numpy as np
import pytest

from dew.decision import Choice, JevConfidence, Noul, Question, Score

DEPARTMENT = Choice("Which team should handle this?",
                    criteria={"billing": "Payments", "technical": "Bugs", "sales": None})
FRUSTRATION = Score("How frustrated is the customer?", criteria=["Calm", "Frustrated", "Very angry"])
URGENT = Noul("Does this convey urgency?", criteria={"true": "Explicitly time-sensitive"})


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
    with pytest.raises(ValueError):
        build()
