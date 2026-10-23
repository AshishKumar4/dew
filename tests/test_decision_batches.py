"""`Decide.systemone` over a list of request bodies: each answered as it is alone, its rows scored with
the others', a refusal its own body's."""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image
from reference_error import assert_as_exact, widened

from dew.decision import Decide
from dew.decision.model import DecisionModel
from dew.decision.questions import ChoiceAnswer, Question

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
LAYA = FIXTURES / "laya" / "tiny"
CASES = json.loads((FIXTURES / "laya" / "cases.json").read_text())

DISCRETE = ("type", "choice", "legend", "abstained")
"""A wire answer's fields a padded batch cannot move: its kind, its argmax option, its legend and its
abstention. Its probabilities, confidence, noul and score are continuous, held by check 1 at the
answers they are rounded from (`test_a_batchs_probabilities_are_each_bodys_alone`)."""

MARGIN = 1e-3
"""The least gap a case admits between an answer's two likeliest options where its choice is held
exactly: far past float32's rounding of a probability, so a batch and a lone body agree on it."""

BATCH_SEEDS = 16
"""The batched probabilities' declared draws (`seeds_for`, paired): each draw is four bodies over the
cases' questions, their states drawn from the seed."""


@pytest.fixture(scope="module")
def decide() -> Decide:
    return Decide.from_pretrained(LAYA, attention_impl="xla")


def discrete(response: dict) -> dict:
    """A response's answers by their `DISCRETE` fields, and its usage."""
    return {"answers": {name: {field: answer[field] for field in DISCRETE if field in answer}
                        for name, answer in response["answers"].items()},
            "usage": response["usage"], "model": response["model"]}


def clear(decide: Decide, request: dict) -> bool:
    """Whether every choice of `request`, alone, holds its two likeliest options `MARGIN` apart."""
    questions = {name: Question.from_wire(wire) for name, wire in request["questions"].items()}
    [(answers, _)] = decide.batch([(request["state"], questions)])
    gaps = [np.diff(np.sort(answer.probabilities)[-2:])[0] for answer in answers.values()
            if isinstance(answer, ChoiceAnswer) and len(answer.probabilities) > 1]
    return all(float(gap) > MARGIN for gap in gaps)


def test_a_list_of_bodies_is_answered_as_each_alone_and_a_refusal_is_its_own(decide):
    """Every case in one list, beside two bodies Jev refuses: each case's response is its lone
    response in every field a batch cannot move, its usage included; each refused body takes the
    ValueError it raises alone, and the others are answered."""
    cases = list(CASES.values())
    assert all(clear(decide, request) for request in cases)
    refused = [{"questions": cases[0]["questions"]}, {**cases[0], "unknown": 1}]
    bodies = [refused[0], *cases[:2], refused[1], *cases[2:]]
    answered = decide.systemone(bodies, details=True)
    assert len(answered) == len(bodies)
    for body, response in zip(bodies, answered, strict=True):
        if body in refused:
            assert isinstance(response, ValueError)
            with pytest.raises(ValueError):
                decide.systemone(body, details=True)
        else:
            assert isinstance(response, dict)
            assert discrete(response) == discrete(decide.systemone(body, details=True))
    assert all(isinstance(response, ValueError) for response in decide.systemone(refused))


def test_a_batchs_probabilities_are_each_bodys_alone(decide):
    """Each body's answers in a batch of four, every option's probability, on par from float64 with its
    answers alone (check 1): a padded batch is another program, which rounds its sums otherwise."""
    words = ["billing", "invoice", "charged", "refund", "twice", "login", "error", "password", "plan",
             "upgrade", "price", "urgent"]
    cases = list(CASES.values())
    wide = {}

    def bodies_of(seed) -> list[dict]:
        """Four bodies over the cases' questions, their states drawn from `seed`."""
        rng = np.random.default_rng([0xBA7C, int(seed)])
        return [{**cases[index % len(cases)], "state": " ".join(rng.choice(words, 12))} for index in range(4)]

    def scored(task: Decide, seed, *, together: bool) -> np.ndarray:
        bodies = bodies_of(seed)
        asked = [(body["state"], {name: Question.from_wire(wire)
                                  for name, wire in body["questions"].items()}) for body in bodies]
        answered = task.batch(asked) if together else [task.batch([one])[0] for one in asked]
        return np.concatenate([answer.probabilities for answers, _ in answered
                               for answer in answers.values()])

    def truth(seed):
        with jax.enable_x64(new_val=True):
            if "task" not in wide:
                # The release in float64: its backbone and head computing in it, over its weights widened.
                model = decide.model
                wide["task"] = dataclasses.replace(
                    decide, model=DecisionModel(model.backbone.clone(dtype=jnp.float64),
                                                model.head.clone(dtype=jnp.float64)),
                    variables=widened(decide.variables))
            return scored(wide["task"], seed, together=False)

    def draw(seed):
        # The draw is its seed; the bodies are made from it, their text no array the check widens.
        return (np.int64(seed),)

    assert_as_exact(lambda seed: scored(decide, seed, together=True),
                    lambda seed: scored(decide, seed, together=False), draw,
                    "batched probabilities", seeds=BATCH_SEEDS, truth_fn=truth)


def test_a_mixed_list_answers_its_images_alone_and_its_text_together(tmp_path):
    """A joint release, whose backbone reads images: in one list, a body with images is answered by
    its own pass, as it is alone, to the same response, and the text bodies beside it as each is alone
    in every field a batch cannot move."""
    release = FIXTURES / "joint_checkpoint" / "tiny"
    backbone = FIXTURES / "hf" / "qwen38-dense-tiny"
    for source in [*backbone.iterdir(), release / "joint_head_config.json",
                   release / "joint_head.safetensors"]:
        (tmp_path / source.name).symlink_to(source.resolve())
    decide = Decide.from_pretrained(tmp_path, attention_impl="xla")
    pictures = [Image.fromarray(array) for array in np.load(backbone / "images.npy")]
    seen = {**CASES["quickstart"], "images": pictures}
    text = [CASES["structured"], CASES["conversation"]]
    answered = decide.systemone([text[0], seen, text[1]])
    assert answered[1] == decide.systemone(seen)
    for body, response in zip(text, (answered[0], answered[2]), strict=True):
        assert isinstance(response, dict)
        assert discrete(response) == discrete(decide.systemone(body))


def test_a_mixed_list_with_a_backbone_that_reads_no_images_refuses_the_images_alone(decide):
    """A text backbone refuses a body's images in a list as it does alone, and answers the rest."""
    answered = decide.systemone([{**CASES["quickstart"], "images": ["aGVsbG8="]}, CASES["structured"]])
    assert isinstance(answered[0], ValueError)
    assert isinstance(answered[1], dict)
    assert discrete(answered[1]) == discrete(decide.systemone(CASES["structured"]))



def test_one_engine_serves_generation_and_scoring_from_one_thread(decide):
    """A generation server and a scoring server under one engine (`dew.inference.engine.Engine`), their
    requests submitted together from the caller's thread: each generation draws what it draws alone,
    and each scoring answer is its lone answer, the two kinds stepped in turn."""
    from model_support import serving_task

    from dew.inference import Engine, ScoringServer, Server

    bound = serving_task()
    prompts, budgets = ["12", "5", "1234567"], [3, 5, 2]
    alone = [bound(prompt, budget, key=index) for index, (prompt, budget) in enumerate(zip(prompts, budgets,
                                                                                          strict=True))]
    generating = Server.from_task(bound, slots=2, capacity=64, admission=2)
    scoring = ScoringServer.from_task(*decide.scoring(), rows=2)
    engine = Engine(generating, scoring)
    engine.start()
    try:
        drawn = [generating.submit(prompt, budget, key=index)
                 for index, (prompt, budget) in enumerate(zip(prompts, budgets, strict=True))]
        answered = [scoring.submit(body) for body in CASES.values()]
        generations = [ticket.result() for ticket in drawn]
        answers = [future.result() for future in answered]
    finally:
        engine.close()
    assert [generation.text for generation in generations] == [generation.text for generation in alone]
    for body, answer in zip(CASES.values(), answers, strict=True):
        assert discrete(answer) == discrete(decide.systemone(body))


def test_a_long_scoring_backlog_does_not_hold_generation_back(decide):
    """One engine serving generation and scoring, a pass a scoring turn: with a scoring backlog of
    one row a pass, a generation submitted beside it finishes while the backlog still has passes to
    run, since each turn steps each kind once, a scoring turn one pass, however much scoring waits."""
    from model_support import serving_task

    from dew.decision.task import Budget
    from dew.inference import Engine, ScoringServer, Server

    bound = serving_task()
    generating = Server.from_task(bound, slots=1, capacity=64, admission=1)
    scoring = ScoringServer.from_task(*dataclasses.replace(decide, budget=Budget(rows=1)).scoring())
    finished: list[str] = []
    engine = Engine(generating, scoring)
    bodies = [*CASES.values()] * 3
    answered = [scoring.submit(body) for body in bodies]
    for future in answered:
        future.add_done_callback(lambda _: finished.append("scoring"))
    drawn = generating.submit("12", 3, key=0)
    drawn.add_done_callback(lambda _: finished.append("generation"))
    engine.start()
    try:
        drawn.result()
        for future in answered:
            future.result()
    finally:
        engine.close()
    assert finished.index("generation") < len(bodies) - 1
    assert drawn.result().text == bound("12", 3, key=0).text
