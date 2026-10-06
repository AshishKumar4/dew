"""Clef's layout and head against Clef's own code (Cloudflare/clef 2f3de3dd), under the float64 rule."""

import json
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import optax
import pytest
from reference_error import assert_as_exact_as_the_reference
from safetensors.numpy import load_file

from dew.checkpoints import Checkpoints
from dew.data.text import HFTokenizer
from dew.decision import (
    Choice,
    Decide,
    DecisionInputs,
    DecisionObjective,
    Encoded,
    Example,
    JointLayout,
    JointSchemaHead,
    Noul,
    Question,
    Specials,
    TopProbability,
)
from dew.decision.calibration import softmax
from dew.decision.clef import ClefHead
from dew.training.trainer import Trainer

FIXTURES = Path(__file__).parent / "fixtures"
TINY = FIXTURES / "clef" / "tiny"
BACKBONE = FIXTURES / "hf" / "qwen38-dense-tiny"
CASES = json.loads((FIXTURES / "laya" / "cases.json").read_text())


@pytest.fixture(scope="module")
def tokenizer() -> HFTokenizer:
    return HFTokenizer(str(BACKBONE), local_files_only=True)


def laid_out(tokenizer: HFTokenizer) -> dict[str, tuple[dict[str, Question], Encoded]]:
    """Each case's questions and its one joint row."""
    rows = {}
    for case, request in CASES.items():
        questions = {name: Question.from_wire(wire) for name, wire in request["questions"].items()}
        [row] = JointLayout().rows(tokenizer, Specials.of(tokenizer), request["state"], questions)
        rows[case] = questions, row
    return rows


def test_the_joint_layout_lays_out_what_clef_does(tokenizer):
    """Token for token and span for span: Clef's prompt, a choice's options by
    key, a noul's true first with Clef's descriptions, structured values as
    compact sorted JSON, and every option's span."""
    expected = json.loads((TINY / "layouts.json").read_text())
    for case, (questions, row) in laid_out(tokenizer).items():
        assert list(row.tokens) == expected[case]["ids"], case
        for laid, reference in zip(row.questions, expected[case]["questions"], strict=True):
            assert laid.name == reference["id"]
            assert list(laid.span) == reference["span"], (case, laid.name)
            assert [list(span) for span in laid.options] == reference["options"], (case, laid.name)
            assert [questions[laid.name].options[slot] for slot in laid.order] == reference["option_ids"]


def head_logits(tokenizer: HFTokenizer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dew's head, Clef's fp32 head, and Clef's float64 head, all on the same
    fp32 backbone states, over every question of every case."""
    loaded = ClefHead.load(TINY, attention_impl="xla")
    table = load_file(BACKBONE / "model.safetensors")["lm_head.weight"]
    found, reference, truth = [], [], []
    with np.load(TINY / "states.npz") as states, np.load(TINY / "logits.npz") as logits:
        for case, (_, row) in laid_out(tokenizer).items():
            inputs = DecisionInputs.collate([row], 0)
            scores = loaded.head.apply({"params": loaded.params}, states[case][None], inputs, table)
            for slot, laid in enumerate(row.questions):
                key = f"{case}/{laid.name}"
                found.append(np.asarray(scores)[0, slot, :len(laid.options)])
                reference.append(logits[key])
                truth.append(logits[f"{key}/head64"])
    return np.concatenate(found), np.concatenate(reference), np.concatenate(truth)


def test_clefs_head_matches_clef_under_the_float64_rule(tokenizer):
    """Every question of every case, read off the same fp32 backbone states,
    held to twice Clef's own head's fp32 distance from its float64 run
    (measured: 1.42 times)."""
    assert_as_exact_as_the_reference(*head_logits(tokenizer), "Clef head logits")


def test_the_rule_catches_clefs_types_in_layas_order(tokenizer, monkeypatch):
    """Clef numbers its type embedding noul, choice, score; reading it in
    Laya's order moves the logits far past the rule."""
    from dew.decision import head

    monkeypatch.setattr(head, "_CLEF_ROWS", (0, 1, 2))
    with pytest.raises(AssertionError, match=r"allowed 2\.0"):
        assert_as_exact_as_the_reference(*head_logits(tokenizer), "Clef head logits, Laya's type order")


def test_a_padded_batch_scores_every_row_as_it_scores_alone(tokenizer):
    """All five cases in one batch, padded to the longest row and the most
    questions and options: padding reaches none of the head's pooling or
    attention, so each question scores as it does alone."""
    loaded = ClefHead.load(TINY, attention_impl="xla")
    table = load_file(BACKBONE / "model.safetensors")["lm_head.weight"]
    rows = laid_out(tokenizer)
    inputs = DecisionInputs.collate([row for _, row in rows.values()], 0)
    with np.load(TINY / "states.npz") as states:
        length = inputs.tokens.shape[1]
        padded = np.stack([np.pad(states[case], ((0, length - len(states[case])), (0, 0))) for case in rows])
        batched = np.asarray(loaded.head.apply({"params": loaded.params}, padded, inputs, table))
        for index, (case, (_, row)) in enumerate(rows.items()):
            alone = np.asarray(loaded.head.apply({"params": loaded.params}, states[case][None],
                                                 DecisionInputs.collate([row], 0), table))
            for slot, laid in enumerate(row.questions):
                width = len(laid.options)
                np.testing.assert_allclose(batched[index, slot, :width], alone[0, slot, :width], atol=2e-5)


def test_a_head_record_rebuilds_the_head():
    head = JointSchemaHead(hidden_size=32, width=24, routing_layers=2, layers=2, heads=2, feedforward=40)
    record = head.record()
    assert record == {"name": "JointSchemaHead", "fields": {"width": 24, "routing_layers": 2, "layers": 2,
                                                              "heads": 2, "feedforward": 40}}
    assert JointSchemaHead.from_record(record, 32) == head


def test_a_clef_release_answers_as_clefs_own_model(tmp_path):
    """`Decide.from_pretrained` reads a Clef release, the Qwen 3.5 backbone
    with the head beside it. Dew's backbone and head together are held to
    twice Clef's own fp32 model's distance from its float64 run, and
    `systemone`, with Clef's confidence, gives each option the probability
    Clef's fp32 logits give it, within the fourth decimal both round to."""
    for source in [*BACKBONE.iterdir(), TINY / "joint_head_config.json", TINY / "joint_head.safetensors"]:
        (tmp_path / source.name).symlink_to(source.resolve())
    decide = replace(Decide.from_pretrained(tmp_path, attention_impl="xla"), confidence=TopProbability())
    layouts = json.loads((TINY / "layouts.json").read_text())
    found, reference, truth = [], [], []
    with np.load(TINY / "logits.npz") as logits:
        for case, request in CASES.items():
            questions = {name: Question.from_wire(wire) for name, wire in request["questions"].items()}
            [row] = decide.layout.rows(decide.tokenizer, decide.specials, request["state"], questions)
            scores = np.asarray(decide.model.logits(decide.variables, DecisionInputs.collate([row], 0)))
            answers = decide.systemone(request)["answers"]
            for slot, (laid, clef) in enumerate(zip(row.questions, layouts[case]["questions"], strict=True)):
                key = f"{case}/{laid.name}"
                found.append(scores[0, slot, :len(laid.options)])
                reference.append(logits[key])
                truth.append(logits[f"{key}/f64"])
                probabilities = softmax(logits[key].astype(np.float64), 1.0)
                expected = dict(zip(clef["option_ids"], probabilities, strict=True))
                answer = answers[laid.name]
                given = ({"true": answer["noul"]} if answer["type"] == "noul" else answer["probabilities"])
                for option, value in given.items():
                    assert abs(value - expected[option]) <= 1e-4, (case, laid.name, option)
                if answer["type"] != "noul":
                    assert answer["confidence"] == max(given.values())
    assert_as_exact_as_the_reference(np.concatenate(found), np.concatenate(reference), np.concatenate(truth),
                                     "Clef over Dew's Qwen 3.5")


def test_a_clef_release_fine_tunes_and_reloads_from_the_run(tmp_path):
    """A Clef release is a whole task to fine-tune: its Qwen 3.5 backbone,
    joint head and layout train together on rows of several questions, and
    the saved run loads back as a task with identical probabilities."""
    release = tmp_path / "release"
    release.mkdir()
    for source in [*BACKBONE.iterdir(), TINY / "joint_head_config.json", TINY / "joint_head.safetensors"]:
        (release / source.name).symlink_to(source.resolve())
    team = Choice("Which team?", {"billing": "payments", "technical": "outages", "sales": "pricing"})
    urgent = Noul("Is it urgent?")
    answered = (("charged twice", "billing", "true"), ("site is down", "technical", "true"),
                ("how much is pro", "sales", "false"), ("refund please", "billing", "false"))
    examples = [Example(state, {"team": team, "urgent": urgent}, {"team": label, "urgent": hurry})
                for state, label, hurry in answered * 2]
    # Training pads every row to the layout's max_len; Clef serves up to 16384.
    objective = DecisionObjective(Decide.from_pretrained(release, attention_impl="xla"),
                                  layout=JointLayout(max_len=1024))
    data = objective.dataset(examples, batch=8, validation=examples)
    run = tmp_path / "run"
    checkpoints = Checkpoints(str(run), keep=1)
    start = objective.init(jax.random.key(0))
    state = Trainer(objective, optax.adam(1e-3), key=0, checkpoints=checkpoints).fit(
        data, steps=2, log_every=100, checkpoint_every=2)
    checkpoints.wait()
    before = jax.tree.leaves(start["params"]["head"])
    after = jax.tree.leaves(state.variables["params"]["head"])
    assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(before, after, strict=True))

    live, reloaded = objective.pipeline(state), Decide.from_run(str(run))
    asked = {"team": team, "urgent": urgent}
    first, second = live("charged twice!", asked), reloaded("charged twice!", asked)
    for answer, again in zip(first.values(), second.values(), strict=True):
        np.testing.assert_array_equal(answer.probabilities, again.probabilities)
