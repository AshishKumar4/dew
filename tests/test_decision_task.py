"""`Decide`: batching, tournaments, calibration and Jev's wire form, against Laya's own agent."""

import base64
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.decision import (
    Abstention,
    Binning,
    Budget,
    Calibration,
    Choice,
    Decide,
    EntropyConfidence,
    Example,
    Question,
    Scored,
    Temperatures,
)
from dew.decision.calibration import softmax
from dew.decision.task import decoded_images

FIXTURES = Path(__file__).parent / "fixtures" / "laya"
TINY = FIXTURES / "tiny"
CASES = json.loads((FIXTURES / "cases.json").read_text())


@pytest.fixture(scope="module")
def decide() -> Decide:
    return Decide.from_pretrained(TINY, attention_impl="xla")


def test_systemone_answers_what_layas_agent_answers(decide):
    """Laya's `Agent.system_one` on every case (answers.json), with Laya's
    entropy confidence selected: the same choice, and every reported number
    within one unit of its fourth decimal, the rounding both apply to logits
    that differ by fp32 rounding."""
    laya = replace(decide, confidence=EntropyConfidence())
    expected = json.loads((TINY / "answers.json").read_text())
    for case, request in CASES.items():
        found = laya.systemone(request, details=True)
        assert found["usage"]["input_tokens"] == expected[case]["usage"]["input_tokens"]
        for name, answer in found["answers"].items():
            reference = expected[case]["answers"][name]
            assert answer["type"] == reference["type"]
            if answer["type"] == "choice":
                assert answer["choice"] == reference["choice"]
            for field in ("noul", "score", "confidence"):
                if field in reference and field in answer:
                    assert abs(answer[field] - reference[field]) <= 1e-4, (case, name, field)
            for option, value in answer.get("probabilities", {}).items():
                assert abs(value - reference["probabilities"][option]) <= 1e-4, (case, name, option)


def test_systemone_speaks_jev_and_nothing_more(decide):
    """Jev's response fields exactly: a noul answers `noul` alone, a choice
    and a score their documented fields, usage two counts."""
    found = decide.systemone(CASES["quickstart"])
    assert set(found) == {"model", "answers", "usage"}
    assert set(found["usage"]) == {"input_tokens", "output_tokens"}
    assert set(found["answers"]["churn"]) == {"type", "noul"}
    assert set(found["answers"]["department"]) == {"type", "choice", "probabilities", "confidence"}
    assert set(found["answers"]["urgency"]) == {"type", "score", "legend", "probabilities", "confidence"}
    assert found["answers"]["urgency"]["legend"] == {"0": "not urgent", "1": "soon", "2": "today",
                                                     "3": "blocking"}


@pytest.mark.parametrize("request_body, message", [
    ({"questions": CASES["quickstart"]["questions"]}, "state"),
    ({"state": "x", "questions": {}}, "at least one question"),
    ({"state": "x", "questions": {"q": {"type": "rank", "instructions": "x"}}}, "noul, choice or score"),
    ({"state": "x", "questions": CASES["quickstart"]["questions"], "stream": True}, "not \\['stream'\\]"),
])
def test_systemone_refuses_a_request_jev_would_refuse(decide, request_body, message):
    with pytest.raises(ValueError, match=message):
        decide.systemone(request_body)


def png(color: tuple[int, int, int]) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_images_decode_from_every_form_a_request_carries():
    """Clef's `images` extension: PIL images, encoded bytes, and base64,
    bare or as a data URL, all read as the same pixels."""
    from PIL import Image

    raw = png((200, 10, 30))
    encoded = base64.b64encode(raw).decode()
    images = decoded_images([Image.open(io.BytesIO(raw)), raw, encoded, f"data:image/png;base64,{encoded}"])
    assert [image.size for image in images] == [(4, 3)] * 4
    assert all(image.convert("RGB").getpixel((0, 0)) == (200, 10, 30) for image in images)
    assert decoded_images(None) == []
    for bad, message in ((["%%%"], "not base64"), ([base64.b64encode(b"text").decode()], "Pillow"),
                         ([3], "PIL image"), ("x", "a list")):
        with pytest.raises(ValueError, match=message):
            decoded_images(bad)


def test_strict_answers_drop_the_extensions_and_hold_jevs_fields(decide):
    """A text backbone refuses images it cannot read; a strict answer drops
    them, as Jev's endpoint knows no images, and holds Jev's fields alone."""
    with_images = {**CASES["quickstart"], "images": [base64.b64encode(png((0, 0, 0))).decode()]}
    with pytest.raises(ValueError, match="vision encoder"):
        decide.systemone(with_images)
    assert decide.systemone(with_images, strict=True) == decide.systemone(CASES["quickstart"])
    with pytest.raises(ValueError, match="no details"):
        decide.systemone(CASES["quickstart"], strict=True, details=True)


def test_a_small_budget_splits_passes_without_moving_a_logit(decide):
    """Every question of every case at a budget of two rows a pass, held to
    Laya's one-row logits under the float64 rule."""
    small = replace(decide, budget=Budget(tokens=256, rows=2))
    rows, keys = [], []
    for case, request in CASES.items():
        laid = small._encoded(request["state"], {name: Question.from_wire(wire)
                                                 for name, wire in request["questions"].items()})
        rows.extend(laid)
        keys.extend(f"sequential/{case}/{encoded.questions[0].name}" for encoded in laid)
    found = [logits[0, :len(encoded.questions[0].options)]
             for encoded, logits in zip(rows, small.logits(rows), strict=True)]
    with np.load(TINY / "logits.npz") as reference:
        assert_as_exact_as_the_reference(np.concatenate(found),
                                         np.concatenate([reference[key] for key in keys]),
                                         np.concatenate([reference[f"{key}/f64"] for key in keys]),
                                         "budgeted logits")


def test_a_tournament_takes_each_group_winner_to_the_final(decide):
    """Fourteen options in groups of at most four: four groups, their
    winners the final's options, and each winner its group's own answer."""
    request = CASES["many_options"]
    question = Question.from_wire(request["questions"]["intent"])
    assert isinstance(question, Choice)
    final = decide.tournament(request["state"], {"intent": question}, group=4)["intent"]
    options = list(question.options)
    groups = [options[part * 14 // 4:(part + 1) * 14 // 4] for part in range(4)]
    described = dict(zip(question.options, question.descriptions, strict=True))
    winners = [decide(request["state"], {"g": Choice(question.instructions,
                                                     {option: described[option] for option in group})})["g"]
               for group in groups]
    assert final.options == tuple(answer.choice for answer in winners)
    assert final.probabilities.shape == (4,)


def synthetic(temperature: float, count: int, options: int, seed: int) -> list[Scored]:
    """Answers whose labels are drawn from softmax(z / temperature)."""
    rng = np.random.default_rng(seed)
    question = Choice("pick", [f"o{index}" for index in range(options)])
    scored = []
    for _ in range(count):
        logits = rng.normal(0.0, 3.0, options)
        scored.append(Scored(question.kind, logits, int(rng.choice(options, p=softmax(logits, temperature)))))
    return scored


def log_loss(scored: list[Scored], temperature: float) -> float:
    return float(np.mean([-np.log(softmax(held.logits, temperature)[held.label]) for held in scored]))


def test_a_fitted_temperature_is_the_log_loss_minimum():
    """The loss is convex in the inverse temperature, so the fit is its
    minimum: a step either way raises the held-out loss."""
    scored = synthetic(2.0, 3000, 4, seed=0)
    fitted = Temperatures.fit(scored, bucket_minimum=100).of("choice", 4)
    assert log_loss(scored, fitted) < log_loss(scored, fitted * 1.01)
    assert log_loss(scored, fitted) < log_loss(scored, fitted / 1.01)


def test_temperatures_fall_back_from_bucket_to_type_and_hold_their_bounds():
    """A bucket below its floor takes its type's temperature; a fit that
    would sharpen past 0.5 is held at 0.5, as Laya's agent holds it."""
    scored = synthetic(0.2, 400, 4, seed=1)
    temperatures = Temperatures.fit(scored, bucket_minimum=2000)
    assert scored[0].bucket not in temperatures.buckets
    assert "choice" in temperatures.types
    assert temperatures.of("choice", 4) == 0.5


def test_abstention_keeps_the_accepted_answers_within_the_target_error():
    scored = synthetic(1.0, 2000, 4, seed=2)
    temperatures = Temperatures.fit(scored, bucket_minimum=100)
    gate = Abstention.fit(scored, temperatures, target_error=0.1)
    threshold = gate.threshold("choice", 4)
    assert threshold is not None
    tops = [round(float(softmax(held.logits, temperatures.of("choice", 4)).max()), 4) for held in scored]
    accepted = [held for held, top in zip(scored, tops, strict=True) if top >= threshold]
    errors = sum(int(np.argmax(held.logits)) != held.label for held in accepted)
    assert (errors + 1) / (len(accepted) + 1) <= 0.1


def test_binning_maps_each_bin_to_its_accuracy():
    scored = synthetic(1.0, 1000, 2, seed=3)
    temperatures = Temperatures()
    binning = Binning.fit(scored, temperatures, bins=5, bucket_minimum=100)
    tops = np.array([softmax(held.logits, 1.0).max() for held in scored])
    correct = np.array([int(np.argmax(held.logits)) == held.label for held in scored])
    index = np.clip((tops * 5).astype(int), 0, 4)
    for bin_index in range(5):
        if np.any(index == bin_index):
            assert binning.maps[scored[0].bucket][bin_index] == pytest.approx(
                correct[index == bin_index].mean())


def test_calibrated_fits_on_labelled_examples_and_gates_answers(decide):
    """Examples with answers by key or index; the fitted task divides by its
    temperatures and marks an answer below its threshold."""
    examples = [Example.of({"state": request["state"], "questions": request["questions"],
                            "answers": dict.fromkeys(request["questions"], 0)})
                for request in CASES.values()] * 4
    calibrated = decide.calibrated(examples, type_minimum=4, bucket_minimum=4, target_error=0.0)
    assert calibrated.calibration.temperatures.types
    assert calibrated.calibration.abstention is not None
    answered = calibrated.systemone(CASES["quickstart"], details=True)["answers"]
    assert all("abstained" in answer for answer in answered.values())


def answered():
    """Every case's questions, each answered with its first option."""
    return [Example.of({"state": request["state"], "questions": request["questions"],
                        "answers": dict.fromkeys(request["questions"], 0)})
            for request in CASES.values()]


def passed(objective, variables, examples):
    """Each held-out row's laid-out batch fields, raw logits and evaluated
    probabilities, over its real option slots, in the pass's order."""
    import jax
    import jax.numpy as jnp

    from dew.data.dataset import DataPartition
    from dew.decision.task import laid_out
    from dew.objectives.base import VALID_ROWS, Step

    step = Step(jnp.asarray(0), jax.random.key(0), None)
    rows = []
    for batch in objective.held_out(examples, batch=8)(DataPartition()):
        probabilities = np.asarray(objective.evaluate(variables, batch, step).probabilities)
        logits = np.asarray(objective._logits(variables, laid_out(batch)))
        # A batch the pass filled with repeats marks its real rows.
        for row in np.flatnonzero(batch.get(VALID_ROWS, np.ones(len(batch["kinds"]), bool))):
            count = int(batch["options"][row, 0].sum())
            rows.append((int(batch["kinds"][row, 0]), logits[row, 0, :count], probabilities[row, 0, :count]))
    return rows


def float32_softmax(logits, temperature):
    """The tempered softmax computed in float32 throughout."""
    scaled = logits.astype(np.float32) / np.float32(temperature)
    shifted = np.exp(scaled - scaled.max())
    return shifted / shifted.sum()


def test_score_is_a_validation_pass_through_the_task_s_temperatures(decide):
    """The pass divides each question's logits by the temperature its answer
    uses, which the released checkpoint sets away from 1: every row chooses
    what the task answers, its probabilities are as exact as float32's own
    tempered softmax of the same logits, and score reports that pass."""
    from dew.decision import KINDS, DecisionObjective

    temperatures = decide.calibration.temperatures
    assert temperatures.of("choice", 3) != 1
    examples = answered()
    answers = [answer for example in examples for answer in decide(example.state, example.questions).values()]
    rows = passed(DecisionObjective(decide, temperatures=temperatures), decide.variables, examples)
    assert [int(np.argmax(found)) for _, _, found in rows] == [
        int(np.argmax(answer.probabilities)) for answer in answers]
    divided = [(logits, temperatures.of(KINDS[kind].kind, len(logits))) for kind, logits, _ in rows]
    assert_as_exact_as_the_reference(
        np.concatenate([found for _, _, found in rows]),
        np.concatenate([float32_softmax(logits, scale) for logits, scale in divided]),
        np.concatenate([softmax(logits.astype(np.float64), scale) for logits, scale in divided]),
        "tempered probabilities")
    scores = decide.score(examples)
    assert set(scores) == {"accuracy", "ece", "aurc", "log_loss"}
    assert scores["accuracy"] == pytest.approx(np.mean([int(np.argmax(answer.probabilities)) == 0
                                                        for answer in answers]))


def test_an_objective_from_a_calibrated_task_validates_without_its_temperatures(decide):
    """The temperatures fit the released logits, which training moves, so a
    run's validation divides by none: exactly what identity temperatures
    give, and not what the task's own give."""
    from dew.decision import DecisionObjective

    examples = answered()
    untempered = DecisionObjective(decide)
    assert untempered.temperatures is None
    plain = [found for _, _, found in passed(untempered, decide.variables, examples)]
    identity = passed(DecisionObjective(decide, temperatures=Temperatures()), decide.variables, examples)
    tempered = passed(DecisionObjective(decide, temperatures=decide.calibration.temperatures),
                      decide.variables, examples)
    np.testing.assert_array_equal(np.concatenate(plain), np.concatenate([found for _, _, found in identity]))
    assert not np.array_equal(np.concatenate(plain), np.concatenate([found for _, _, found in tempered]))


def test_a_gate_abstains_below_its_threshold(decide):
    """One threshold for every answer: above the top probability abstains,
    at or below it passes."""
    question = Question.from_wire(CASES["quickstart"]["questions"]["department"])
    top = float(decide(CASES["quickstart"]["state"], {"q": question})["q"].probabilities.max())
    assert decide.gated(top + 1e-3)(CASES["quickstart"]["state"], {"q": question})["q"].abstained
    assert not decide.gated(top - 1e-3)(CASES["quickstart"]["state"], {"q": question})["q"].abstained


def test_a_calibration_is_a_run_record_value():
    """A fitted calibration writes as its fields and reads back equal through
    the one record walk a run config uses."""
    from dew.registry import from_record, to_record

    scored = synthetic(1.5, 600, 3, seed=4)
    temperatures = Temperatures.fit(scored, bucket_minimum=100)
    binning = Binning.fit(scored, temperatures, bins=5, bucket_minimum=100)
    calibration = Calibration(temperatures, binning, Abstention.fit(scored, temperatures, binning))
    written = json.loads(json.dumps(to_record(calibration, Calibration)))
    assert from_record(Calibration, written) == calibration


def test_an_example_refuses_an_answer_its_question_cannot_give():
    with pytest.raises(ValueError, match="none of"):
        Example.of({"state": "x", "questions": CASES["quickstart"]["questions"],
                    "answers": {"department": "legal"}})


def test_a_task_saves_in_layas_layout_and_reads_back_the_same(decide, tmp_path):
    """`save_pretrained` writes what `from_pretrained` reads: Laya's tensors
    under Laya's names (all but the action head and the unused temperature
    buffer, which no answer reads), the encoder's config, the tokenizer and
    the temperatures, so the reloaded task answers every case identically."""
    from safetensors.numpy import load_file

    from dew.interop.hf_decoders import translate_config

    decide.save_pretrained(tmp_path)
    written, original = load_file(tmp_path / "model.safetensors"), load_file(TINY / "model.safetensors")
    kept = {name for name in original if not name.startswith("act_head.") and name != "temperature"}
    assert set(written) == kept
    for name in kept:
        np.testing.assert_array_equal(written[name], original[name], err_msg=name)
    # The published encoder config names no architecture, which reads as the
    # masked LM; the written one names the bare encoder its tensors are.
    configs = [replace(translate_config(json.loads((root / "encoder" / "config.json").read_text())).value,
                       head_transform=None, head_bias=False) for root in (tmp_path, TINY)]
    assert configs[0] == configs[1]
    # The temperatures go out as Laya's agent applies them, held within [0.5, 5],
    # since llama.cpp's server applies what it reads: the published 0.3 for
    # eleven options and more is written as the 0.5 both answer with.
    agent = json.loads((tmp_path / "rl_agent_config.json").read_text())
    published = json.loads((TINY / "rl_agent_config.json").read_text())
    assert agent["temperature"] == published["temperature"]
    assert agent["head_layers"] == published["head_layers"]
    assert agent["temperature_by_options"] == {**published["temperature_by_options"], "choice:11+": 0.5}
    again = replace(Decide.from_pretrained(tmp_path, attention_impl="xla"), name=decide.name)
    for request in CASES.values():
        assert again.systemone(request) == decide.systemone(request)


def test_a_released_checkpoint_refuses_what_its_layout_cannot_hold(decide, tmp_path):
    gated = decide.gated(0.5)
    with pytest.raises(ValueError, match="temperatures alone"):
        gated.save_pretrained(tmp_path)
