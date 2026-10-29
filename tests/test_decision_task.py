"""`Decide`: batching, tournaments, calibration and Jev's wire form, against Laya's own agent."""

import base64
import io
import json
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import numpy as np
import pytest
from decision_support import decision_rows
from reference_error import as_given, assert_as_exact, assert_as_exact_as_the_reference
from steady_state import steady_state

from dew.artifacts import Decisions
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
from dew.decision.metrics import Answered
from dew.decision.task import decoded_images

FIXTURES = Path(__file__).parent / "fixtures" / "laya"
TINY = FIXTURES / "tiny"
CASES = json.loads((FIXTURES / "cases.json").read_text())


@pytest.fixture(scope="module")
def decide() -> Decide:
    return Decide.from_pretrained(TINY, attention_impl="xla")


def test_a_decision_source_loads_through_pretrained_and_its_declared_pipeline(decide, tmp_path):
    """A Laya component release loads through the public source loader, including a subfolder,
    and its declared task retains the same answers through export."""
    from shutil import copytree

    from dew import pipeline
    from dew.interop import Pretrained

    copytree(TINY, tmp_path / "release")
    bundle = Pretrained.load(tmp_path, subfolder="release", dtype="float32", attention_impl="xla")
    task = replace(Decide.of(bundle), name=decide.name)
    request = CASES["quickstart"]
    assert task.systemone(request) == decide.systemone(request)
    original = jax.tree.map(lambda value: np.asarray(value).copy(), bundle.variables)
    for _ in range(2):
        assert_same_rounded_answers(replace(bundle.pipeline(), name=decide.name).systemone(request),
                                    decide.systemone(request), "borrowed bundle task")
    for held, expected in zip(jax.tree.leaves(bundle.variables), jax.tree.leaves(original), strict=True):
        np.testing.assert_array_equal(held, expected)
    bundle.save(tmp_path / "export", variables=bundle.variables, max_shard_size="10KB")
    served = pipeline(str(tmp_path / "export"), dtype="float32")
    assert_same_rounded_answers(replace(served, name=decide.name).systemone(request),
                                decide.systemone(request), "public source pipeline")


_ANSWERED = """
import json
import sys
from dew.decision import Decide
task = Decide.from_pretrained("dew/tiny-laya", subfolder="release", attention_impl="xla")
print(json.dumps(task.systemone(json.loads(sys.argv[1]))))
"""


def test_a_hub_folder_release_answers_under_the_name_it_was_asked_by(tmp_path):
    """A release in a folder of a Hub repo answers under the repo and folder it was loaded by, not
    the cache directory it resolved to, as Laya's server names the model it serves."""
    import os
    import subprocess
    import sys
    from shutil import copytree

    from interop_support import hub_cache

    copytree(TINY, tmp_path / "repo" / "release")
    hub_cache(tmp_path / "hub", "dew/tiny-laya", tmp_path / "repo")
    env = {**os.environ, "HF_HUB_CACHE": str(tmp_path / "hub"), "HF_HUB_OFFLINE": "1", "JAX_PLATFORMS": "cpu"}
    run = subprocess.run([sys.executable, "-c", _ANSWERED, json.dumps(CASES["quickstart"])],
                         capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr[-2000:]
    assert json.loads(run.stdout.splitlines()[-1])["model"] == "dew/tiny-laya/release"


def assert_same_rounded_answers(found: dict, expected: dict, case: str) -> None:
    """Two `systemone(details=True)` answers to `case` from logits that differ
    by fp32 rounding: the same choices, and every reported number within one
    unit of its fourth decimal, the rounding both apply."""
    assert found["usage"]["input_tokens"] == expected["usage"]["input_tokens"]
    for name, answer in found["answers"].items():
        reference = expected["answers"][name]
        assert answer["type"] == reference["type"]
        if answer["type"] == "choice":
            assert answer["choice"] == reference["choice"]
        for field in ("noul", "score", "confidence"):
            if field in reference and field in answer:
                assert abs(answer[field] - reference[field]) <= 1e-4, (case, name, field)
        for option, value in answer.get("probabilities", {}).items():
            assert abs(value - reference["probabilities"][option]) <= 1e-4, (case, name, option)


def test_systemone_answers_what_layas_agent_answers(decide):
    """Laya's `Agent.system_one` on every case (answers.json), with Laya's
    entropy confidence selected (`assert_same_rounded_answers`)."""
    laya = replace(decide, confidence=EntropyConfidence())
    expected = json.loads((TINY / "answers.json").read_text())
    for case, request in CASES.items():
        assert_same_rounded_answers(laya.systemone(request, details=True), expected[case], case)


@pytest.mark.parametrize("request_body, message", [
    ({"questions": CASES["quickstart"]["questions"]}, "state"),
    ({"state": "x", "questions": {}}, "at least one question"),
    ({"state": "x", "questions": {"q": {"type": "rank", "instructions": "x"}}}, "noul, choice or score"),
    ({"state": "x", "questions": CASES["quickstart"]["questions"], "stream": True}, "not \\['stream'\\]"),
])
def test_systemone_refuses_a_request_jev_would_refuse(decide, request_body, message):
    with pytest.raises(ValueError):
        decide.systemone(request_body)


def png(color: tuple[int, int, int]) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_images_decode_from_every_form_a_request_carries():
    """The `images` extension: PIL images, encoded bytes, and base64,
    bare or as a data URL, all read as the same pixels."""
    from PIL import Image

    raw = png((200, 10, 30))
    encoded = base64.b64encode(raw).decode()
    images = decoded_images([Image.open(io.BytesIO(raw)), raw, encoded, f"data:image/png;base64,{encoded}"])
    assert [image.size for image in images] == [(4, 3)] * 4
    assert all(image.convert("RGB").getpixel((0, 0)) == (200, 10, 30) for image in images)
    assert decoded_images(None) == []
    for bad, _message in ((["%%%"], "not base64"), ([base64.b64encode(b"text").decode()], "Pillow"),
                         ([3], "PIL image"), ("x", "a list")):
        with pytest.raises(ValueError):
            decoded_images(bad)


def test_strict_answers_drop_the_extensions_and_hold_jevs_fields(decide):
    """A text backbone refuses images it cannot read; a strict answer drops
    them, as Jev's endpoint knows no images, and holds Jev's fields alone."""
    with_images = {**CASES["quickstart"], "images": [base64.b64encode(png((0, 0, 0))).decode()]}
    with pytest.raises(ValueError):
        decide.systemone(with_images)
    assert decide.systemone(with_images, strict=True) == decide.systemone(CASES["quickstart"])
    with pytest.raises(ValueError):
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


def test_a_questions_tournament_is_its_own_beside_a_longer_one(decide):
    """Each question plays its own rounds: beside a question of fourteen options, which takes two rounds
    of groups of three, one of five options plays one round and waits for the final with its two
    winners, the finalists it reaches alone, not split again to one; its answer's probabilities cover
    those finalists."""
    request = CASES["many_options"]
    question = Question.from_wire(request["questions"]["intent"])
    assert isinstance(question, Choice)
    described = dict(zip(question.options, question.descriptions, strict=True))
    short = Choice(question.instructions, {option: described[option] for option in question.options[:5]})
    alone = decide.tournament(request["state"], {"short": short}, group=3)["short"]
    beside = decide.tournament(request["state"], {"short": short, "intent": question}, group=3)
    assert len(alone.options) == 2
    assert beside["short"].options == alone.options
    assert len(beside["short"].probabilities) == len(alone.options)
    assert len(beside["intent"].options) <= 3


def synthetic(temperature: float, count: int, options: int, seed: int) -> list[Scored]:
    """Answers whose labels are drawn from softmax(z / temperature)."""
    rng = np.random.default_rng(seed)
    question = Choice("pick", [f"o{index}" for index in range(options)])
    scored = []
    for _ in range(count):
        logits = rng.normal(0.0, 3.0, options)
        label = int(rng.choice(options, p=softmax(logits, temperature)))
        scored.append(Scored(question.kind, logits, np.eye(options)[label]))
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


def test_calibrating_on_the_distributions_the_model_gives_keeps_its_temperatures(decide):
    """Held-out answers whose gold is a distribution rather than one option:
    where every gold is the distribution the uncalibrated model gives, log
    loss is least at temperature 1, so each question type's fitted
    temperature is 1. Fitting to each gold's most likely option alone would
    sharpen it instead."""
    from dew.decision import Calibration

    raw = replace(decide, calibration=Calibration())
    held = []
    for request in CASES.values():
        example = Example.of({"state": request["state"], "questions": request["questions"],
                              "answers": dict.fromkeys(request["questions"], 0)})
        answers = raw(example.state, example.questions)
        held.append({"state": request["state"], "questions": request["questions"],
                     "targets": {name: answer.probabilities.tolist() for name, answer in answers.items()}})
    fitted = raw.calibrated(held, type_minimum=1).calibration.temperatures
    assert set(fitted.types) == {"choice", "score", "noul"}
    for temperature in fitted.types.values():
        assert temperature == pytest.approx(1.0, rel=1e-3)


def test_a_temperature_fit_does_not_depend_on_a_common_logit_shift():
    """Gold distributions packed in float32, as a batch packs them, hold a
    little less than unit mass; read as they are, the fit's slope would grow
    with every logit's common offset. Read as distributions, three answers
    whose gold is their own softmax fit temperature 1 at any offset."""
    from dew.decision.calibration import _temperature

    logits = np.asarray([[0.0, 1.0, 2.0], [0.5, -1.0, 0.0], [2.0, 2.5, -0.5]])
    for shift in (0.0, 1e7):
        for widened in (False, True):
            golds = [softmax(row, 1.0).astype(np.float32) for row in logits]
            held = [Scored("choice", row + shift, gold.astype(np.float64) if widened else gold)
                    for row, gold in zip(logits, golds, strict=True)]
            assert _temperature(held) == pytest.approx(1.0, rel=1e-6)


def served(calibration, question, scored):
    """The answers `calibration` serves rather than abstains on."""
    return [held for held in scored
            if not calibration.abstains(question, calibration.probabilities(question, held.logits))]


def test_abstention_keeps_the_answers_it_serves_within_the_target_error():
    scored = synthetic(1.0, 2000, 4, seed=2)
    temperatures = Temperatures.fit(scored, bucket_minimum=100)
    calibration = Calibration(temperatures, abstention=Abstention.fit(scored, temperatures, target_error=0.1))
    accepted = served(calibration, Choice("pick", [f"o{index}" for index in range(4)]), scored)
    assert accepted
    errors = sum(int(np.argmax(held.logits)) != held.label for held in accepted)
    assert (errors + 1) / (len(accepted) + 1) <= 0.1


def test_abstention_serves_every_answer_its_fit_accepted():
    """Ninety-nine right answers at a confidence of 0.89997 and one wrong at 0.90002: the fit accepts
    all hundred within a 10% target, and serving accepts the same hundred, not the wrong one alone."""
    question = Choice("pick", ["a", "b"])

    def answered(top, right):
        return Scored(question.kind, np.asarray([np.log(top / (1 - top)), 0.0]), np.eye(2)[0 if right else 1])

    scored = [answered(0.89997, right=True)] * 99 + [answered(0.90002, right=False)]
    calibration = Calibration(abstention=Abstention.fit(scored, Temperatures(), target_error=0.1))
    assert len(served(calibration, question, scored)) == 100


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

    from dew.batch import VALID_ROWS
    from dew.data.dataset import DataPartition
    from dew.decision.task import laid_out
    from dew.objectives.base import Step

    step = Step(jnp.asarray(0), jax.random.key(0), None)
    found = []
    from dew.data import Memory
    from dew.data.compile import held
    from dew.decision import decisions

    rows = Memory([{"example": example} for example in examples]).pipe(decisions).read("memory")
    reader = held(rows, objective=objective.data(), batch=8)
    for batch in reader(DataPartition()):
        probabilities = np.asarray(objective.evaluate(variables, batch, step).probabilities)
        logits = np.asarray(objective._logits(variables, laid_out(batch, objective.bounds.segments)))
        # A batch the pass filled with repeats marks its real windows.
        for window in np.flatnonzero(batch.get(VALID_ROWS, np.ones(len(batch["kinds"]), bool))):
            for slot in np.flatnonzero(batch["questions"][window]):
                count = int(batch["options"][window, slot].sum())
                found.append((int(batch["kinds"][window, slot]), logits[window, slot, :count],
                              probabilities[window, slot, :count]))
    return found


def float32_softmax(logits, temperature):
    """The tempered softmax computed in float32 throughout."""
    scaled = logits.astype(np.float32) / np.float32(temperature)
    shifted = np.exp(scaled - scaled.max())
    return shifted / shifted.sum()


TEMPERED_SEEDS = 16
"""Check 1's declared seeds for the tempered probabilities, sized by `seeds_for` (2026-10-21, on armada)."""


def test_score_is_a_validation_pass_through_the_task_s_temperatures(decide):
    """The pass divides each question's logits by the temperature its answer
    uses, which the released checkpoint sets away from 1: every row chooses
    what the task answers, its probabilities are on par with float32's own
    tempered softmax of the same logits over drawn weights (check 1), and
    score reports that pass."""
    import jax

    from dew.decision import KINDS, DecisionObjective

    temperatures = decide.calibration.temperatures
    assert temperatures.of("choice", 3) != 1
    examples = answered()
    answers = [answer for example in examples for answer in decide(example.state, example.questions).values()]
    rows = passed(DecisionObjective(decide, temperatures=temperatures), decide.variables, examples)
    assert [int(np.argmax(found)) for _, _, found in rows] == [
        int(np.argmax(answer.probabilities)) for answer in answers]
    objective = DecisionObjective(decide, temperatures=temperatures)

    def draw(seed):
        # The checkpoint's weights, each moved by a relative 1e-2 of its own: logits near its own.
        keys = iter(jax.random.split(jax.random.key(seed), len(jax.tree.leaves(decide.variables))))

        def moved(leaf):
            if not np.issubdtype(leaf.dtype, np.floating):
                return np.asarray(leaf)
            return np.asarray(leaf * (1 + 1e-2 * jax.random.normal(next(keys), leaf.shape)), leaf.dtype)

        return (jax.tree.map(moved, decide.variables),)

    def tempered(variables, softmax_of):
        # The pass runs the checkpoint's model on weights of its dtypes, which a float64 truth's widened.
        variables = jax.tree.map(lambda leaf, like: np.asarray(leaf).astype(like.dtype), variables,
                                 decide.variables)
        return np.concatenate([softmax_of(logits, temperatures.of(KINDS[kind].kind, len(logits)))
                               for kind, logits, _ in passed(objective, variables, examples)])

    assert_as_exact(as_given(lambda variables: np.concatenate(
                        [found for _, _, found in passed(objective, variables, examples)])),
                    as_given(lambda variables: tempered(variables, float32_softmax)), draw,
                    "tempered probabilities", seeds=TEMPERED_SEEDS,
                    truth_fn=lambda variables: tempered(variables, lambda logits, scale: softmax(
                        logits.astype(np.float64), scale)))
    scores = decide.score(examples)
    assert set(scores) == {"accuracy", "macro_f1", "ece", "aurc", "log_loss", "brier", "scored_questions"}
    assert scores["accuracy"] == pytest.approx(np.mean([int(np.argmax(answer.probabilities)) == 0
                                                        for answer in answers]))
    # A second score runs the forward the first compiled.
    with steady_state(allow=("host_to_device", "device_to_host")):
        assert decide.score(examples) == scores


def test_an_evaluation_runs_the_modules_the_objective_holds_now(decide):
    """A substitution at the same shapes that changes the arithmetic, as a
    narrowed or quantized head does, is what the next evaluation runs, as an
    objective built holding it runs it."""
    import jax.numpy as jnp

    from dew.decision import DecisionObjective

    def logits(objective):
        return np.concatenate([found for _, found, _ in passed(objective, decide.variables, examples)])

    examples = answered()
    objective = DecisionObjective(decide)
    before = logits(objective)
    backbone, head = (entry.module for entry in objective.program_key())
    narrowed = [backbone, head.clone(dtype=jnp.bfloat16)]
    objective.substitute(narrowed)
    built = DecisionObjective(decide)
    built.substitute(narrowed)
    expected = logits(built)
    assert not np.array_equal(before, expected)
    np.testing.assert_array_equal(logits(objective), expected)


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


@pytest.mark.parametrize("kind, question, gold", [
    ("answers", "urgency", "legal"),
    ("answers", "urgency", 4),
    ("answers", "urgency", 2.0),
    ("answers", "urgency", True),
    ("targets", "unasked", [0.5, 0.5]),
    ("answers", "department", "billing"),
    ("targets", "department", [0.5, 0.5]),
    ("targets", "department", [0.9, 0.3, -0.2]),
    ("targets", "department", [0.6, 0.3, 0.3]),
    ("targets", "department", {"billing": 0.6, "legal": 0.4}),
    ("targets", "department", 0.6),
], ids=["an-option-it-does-not-offer", "an-index-past-its-options", "a-float", "a-bool",
        "a-target-for-a-question-it-does-not-ask", "an-answer-beside-a-target", "a-target-of-another-length",
        "a-negative-chance", "chances-that-do-not-sum-to-one", "a-target-keyed-by-options-it-does-not-offer",
        "a-target-of-one-number"])
def test_an_example_refuses_gold_that_scores_no_option_of_its_question(kind, question, gold):
    """A row's gold names an option of a question the row asks: an answer by its key or its index, a
    target a distribution over every option, in the question's order or keyed by option, and one or the
    other for a question. The row trains on that gold, and the same row with one answer or target
    changed so it scores no option of its question, or a question it does not ask, is refused rather
    than trained with its gold dropped or scored against an option it does not offer."""
    questions = CASES["quickstart"]["questions"]
    row = {"state": "x", "questions": questions, "answers": {"urgency": 2},
           "targets": {"department": {"technical": 0.25, "billing": 0.5, "sales": 0.25}}}
    valid = Example.of(row)
    assert valid.labels() == {"department": 0, "urgency": 2}
    np.testing.assert_array_equal(valid.distribution("department"), [0.5, 0.25, 0.25])
    with pytest.raises(ValueError):
        Example.of({**row, kind: {**row[kind], question: gold}})


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


def test_a_task_under_lora_saves_its_factors_merged_and_reads_back_the_same(decide, tmp_path):
    """A ModernBERT task whose backbone runs under LoRA, its factors moved off
    zero, saves in Laya's layout with the factors merged into their kernels
    (`Pretrained.bound`): the reloaded plain checkpoint answers every case as
    the adapted task does, the merged sum rounded once into each kernel
    (`assert_same_rounded_answers`), and differently from the task without the
    adapter."""
    from dew.batch import part
    from dew.decision.model import DecisionModel
    from dew.lora import LoRA
    from dew.objectives.base import joined

    backbone = part(decide.variables, "backbone")
    adapter = LoRA(rank=4, modules=("q_proj", "v_proj")).apply(decide.model.backbone, backbone, key=1)
    moved = jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf + 0.05 if "lora_B" in jax.tree_util.keystr(path) else leaf, adapter.variables)
    adapted = replace(decide, model=DecisionModel(adapter.model, decide.model.head),
                      variables=joined({"backbone": moved, "head": part(decide.variables, "head")}))

    adapted.save_pretrained(tmp_path)
    again = replace(Decide.from_pretrained(tmp_path, attention_impl="xla"), name=decide.name)
    changed = False
    for case, request in CASES.items():
        expected = adapted.systemone(request, details=True)
        assert_same_rounded_answers(again.systemone(request, details=True), expected, case)
        changed |= decide.systemone(request, details=True)["answers"] != expected["answers"]
    assert changed


def test_dew_export_writes_a_decision_run_in_the_tasks_own_release(decide, tmp_path):
    """`dew export` writes a decision run through its task's release writer
    (`Decide.save_pretrained`): the run of a step's training over Laya's tiny
    checkpoint exports the files that writer writes for it, byte for byte."""
    import optax

    from dew.checkpoints import Checkpoints
    from dew.cli.main import Export
    from dew.decision import DecisionObjective
    from dew.training.trainer import Trainer

    objective = DecisionObjective(decide)
    data = decision_rows(answered() * 2, 8, answered(), objective=objective.data())
    checkpoints = Checkpoints(str(tmp_path / "run"), keep=1)
    Trainer(objective, optax.adam(1e-3), key=0, checkpoints=checkpoints).fit(
        data, steps=1, log_every=100, checkpoint_every=1, eval_every=100)
    checkpoints.wait()
    assert Export(str(tmp_path / "run"), str(tmp_path / "export")).run_command() == 0
    Decide.from_run(str(tmp_path / "run")).save_pretrained(tmp_path / "release")
    written = sorted(path.relative_to(tmp_path / "export") for path in (tmp_path / "export").rglob("*")
                     if path.is_file())
    assert written == sorted(path.relative_to(tmp_path / "release")
                             for path in (tmp_path / "release").rglob("*") if path.is_file())
    for name in written:
        assert (tmp_path / "export" / name).read_bytes() == (tmp_path / "release" / name).read_bytes(), name


def test_a_released_checkpoint_keeps_its_option_budget(decide, tmp_path):
    """A task whose options keep 4 tokens each, not the layout's 48, saves
    that budget and reads it back: the reloaded task answers a question of
    long options as the task does, and the 48-token task answers otherwise."""
    from dew.decision import Choice

    tight = replace(decide, layout=replace(decide.layout, option_tokens=4))
    tight.save_pretrained(tmp_path)
    again = Decide.from_pretrained(tmp_path, attention_impl="xla")
    assert again.layout.option_tokens == 4
    question = {"q": Choice("Which team handles it?", {
        "billing": "invoices, refunds, chargebacks, double charges and any dispute about money taken",
        "technical": "outages, errors, failed logins, slow pages and anything that does not work"})}
    state = "I was charged twice and the second charge bounced my rent."
    expected = tight(state, question)["q"].probabilities
    np.testing.assert_array_equal(again(state, question)["q"].probabilities, expected)
    assert not np.array_equal(decide(state, question)["q"].probabilities, expected)


def test_a_released_checkpoint_refuses_what_its_layout_cannot_hold(decide, tmp_path):
    """A gated task, and a task whose row order or blocks Laya's config cannot state, are refused before
    anything is written; Laya's own order, named, saves as the release's."""
    gated = decide.gated(0.5)
    with pytest.raises(ValueError):
        gated.save_pretrained(tmp_path)
    for chosen in ({"order": "state_first"}, {"order": "re_read"}, {"blocks": True}):
        with pytest.raises(ValueError):
            replace(decide, layout=replace(decide.layout, **chosen)).save_pretrained(tmp_path / "chosen")
        assert not (tmp_path / "chosen").exists()
    replace(decide, layout=replace(decide.layout, order="question_first")).save_pretrained(tmp_path / "own")
    assert Decide.from_pretrained(tmp_path / "own", attention_impl="xla").layout == decide.layout


MESH_SEEDS = 8
"""Check 1's declared seeds for a task's logits on a mesh against one device's."""


@dataclass(frozen=True)
class Recorded:
    """A metric that records each answered question's probabilities over its real options as a pass
    evaluates them (`Answered`), in the pass's order, and scores nothing."""

    found: list
    name = "recorded"
    reads = Decisions

    def __call__(self, artifact, batch, /) -> float:
        answered = Answered.of(artifact)
        self.found.extend(row[real] for row, real in zip(answered.probabilities, answered.options,
                                                         strict=True))
        return 0.0

    def merge(self, accumulated: float, contribution: float, /) -> float:
        return accumulated

    def finalize(self, accumulated: float, /) -> float:
        return accumulated


@pytest.mark.mesh
def test_a_task_whose_weights_span_a_mesh_scores_on_that_mesh(decide):
    """A task whose weights are placed on a mesh of every device scores its examples there, its rows
    split over the mesh's row axes, as a run's validation splits them, rather than handing one device's
    inputs to weights on several: a pass of three rows, the task's budget, is rounded up to a multiple
    of the devices the mesh splits rows over. Each answered question's probabilities as its score
    evaluates them (`Evaluation.run`), and its logits as its calibration reads them (`_scored`), are on
    par from float64 with the task's on one device (check 1), and its metrics are the ones it scores
    there."""
    from jax.sharding import NamedSharding, PartitionSpec

    from dew.training import MeshSpec

    examples = answered()
    mesh = MeshSpec().build(jax.devices())
    task = replace(decide, budget=Budget(rows=3))
    wide = replace(Decide.from_pretrained(TINY, dtype="float64", param_dtype="float64", attention_impl="xla"),
                   budget=task.budget)

    def evaluated(on: Decide) -> np.ndarray:
        found: list = []
        on.score(examples, [Recorded(found)])
        return np.concatenate(found)

    def held_logits(on: Decide) -> np.ndarray:
        return np.concatenate([scored.logits for scored in on._scored(on._held(examples, on._mesh))])

    def spread(variables) -> Decide:
        return replace(task, variables=jax.device_put(variables, NamedSharding(mesh, PartitionSpec())))

    def draw(seed):
        key = jax.random.key(seed)
        return (jax.tree.map(lambda leaf: leaf * (1 + 0.05 * jax.random.normal(key, np.shape(leaf))),
                             decide.variables),)

    @as_given
    def alone(variables) -> np.ndarray:
        return held_logits(replace(task, variables=variables))

    def truth(variables) -> np.ndarray:
        return held_logits(replace(wide, variables=variables))

    assert_as_exact(as_given(lambda variables: held_logits(spread(variables))), alone, draw,
                    "scored on a mesh",
                    seeds=MESH_SEEDS, truth_fn=truth)
    assert_as_exact(as_given(lambda variables: evaluated(spread(variables))),
                    as_given(lambda variables: evaluated(replace(task, variables=variables))), draw,
                    "evaluated on a mesh", seeds=MESH_SEEDS,
                    truth_fn=lambda variables: evaluated(replace(wide, variables=variables)))
    alone, scores = task.score(examples), spread(decide.variables).score(examples)
    assert set(scores) == set(alone) and all(np.isfinite(value) for value in scores.values())
    assert scores["accuracy"] == alone["accuracy"]


def trained_runs(decide, tmp_path, count: int, *, name: str = "run") -> list[str]:
    """`count` runs of the task trained apart, one more step each, from one head (the task's own, held):
    each a checkpoint of its own, as averaging reads them."""
    import optax

    from dew.checkpoints import Checkpoints
    from dew.decision import DecisionObjective
    from dew.training.trainer import Trainer

    runs = []
    for index in range(count):
        objective = DecisionObjective(decide)
        data = decision_rows(answered() * 2, 8, answered(), objective=objective.data())
        directory = str(tmp_path / f"{name}{index}")
        checkpoints = Checkpoints(directory, keep=1)
        Trainer(objective, optax.adam(1e-3), key=index, checkpoints=checkpoints).fit(
            data, steps=index + 1, log_every=100, checkpoint_every=index + 1, eval_every=100)
        checkpoints.wait()
        runs.append(directory)
    return runs


def test_runs_that_share_a_head_average_as_the_checkpoints_average_them_and_refit_their_temperatures(
        decide, tmp_path):
    """Runs trained apart from one held head average: the task's weights are the runs' fp32 mean as
    any run's average takes it (`dew.checkpoints.averaged`), each floating leaf float64's mean of the
    two rounded to fp32; the task's weights record every run's step and read, so no one run's
    directory may hold its calibration (`save` refuses), and `calibrate_on` refits it as the average's
    own calibration."""
    from etils import epath

    from dew.checkpoints import averaged

    runs = trained_runs(decide, tmp_path, 2)
    tasks = [Decide.from_run(run) for run in runs]
    two = Decide.averaged(runs)
    trees = (two.variables, averaged(runs).variables, tasks[0].variables, tasks[1].variables)
    for leaf, general, one, other in zip(*(jax.tree.leaves(tree) for tree in trees), strict=True):
        assert np.array_equal(np.asarray(leaf), np.asarray(general))
        if np.issubdtype(np.asarray(leaf).dtype, np.floating):
            halves = (np.asarray(one, np.float64) + np.asarray(other, np.float64)) / 2
            assert np.array_equal(np.asarray(leaf), halves.astype(np.float32))
    assert two.weights is not None
    assert [entry[0] for entry in two.weights.averaged] == [str(epath.Path(run)) for run in runs]
    assert two.calibration == Calibration()
    with pytest.raises(ValueError):
        two.save(runs[0])
    # Enough held-out answers that every question kind clears its calibration minimum (TYPE_MINIMUM).
    held_out = answered() * 10
    calibrated = Decide.averaged(runs, calibrate_on=held_out)
    assert calibrated.calibration != Calibration()
    assert calibrated.calibration == two.calibrated(held_out).calibration


def test_runs_of_other_task_settings_or_a_single_run_do_not_average(decide, tmp_path):
    """Runs average where their recorded task settings are one task's: a run laid out otherwise does
    not average with the others, nor does a single run."""
    runs = trained_runs(decide, tmp_path, 1)
    with pytest.raises(ValueError):
        Decide.averaged(runs)
    other = replace(decide, layout=replace(decide.layout, head_max_len=decide.layout.head_max_len // 2))
    runs += trained_runs(other, tmp_path, 1, name="other")
    with pytest.raises(ValueError):
        Decide.averaged(runs)


@pytest.mark.parametrize("kind", ["marker", "state-first"])
def test_an_averaged_task_of_any_layout_saves_reloads_and_serves_as_it_was(decide, tmp_path, kind):
    """An average of runs, its temperatures refit on held-out answers, saves to a directory of its own
    as a run (`save`), which `from_run` and `dew.pipeline` read back with its weights, calibration and
    provenance, answering every case as the task it was: under Laya's markers and a layout of the state
    first (the joint release's in test_decision_joint_checkpoint). Saving it there again keeps it; a
    run's own directory, which holds other weights, refuses it."""
    import dataclasses

    from dew.decision import StateFirstLayout
    from dew.inference.loading import pipeline

    if kind == "state-first":
        decide = replace(decide, layout=StateFirstLayout(max_len=decide.layout.max_len,
                                                         head_max_len=decide.layout.head_max_len))
    runs = trained_runs(decide, tmp_path, 2)
    held_out = answered() * 10
    task = Decide.averaged(runs, calibrate_on=held_out)
    assert task.calibration != Calibration()
    directory = str(tmp_path / "averaged")
    task.save(directory)
    for reloaded in (Decide.from_run(directory), pipeline(directory)):
        assert isinstance(reloaded, Decide)
        assert reloaded.layout == task.layout and reloaded.weights == task.weights
        assert reloaded.calibration == task.calibration
        for leaf, own in zip(jax.tree.leaves(reloaded.variables), jax.tree.leaves(task.variables),
                             strict=True):
            assert np.array_equal(np.asarray(leaf), np.asarray(own))
        for request in CASES.values():
            questions = {name: Question.from_wire(wire) for name, wire in request["questions"].items()}
            found, expected = reloaded(request["state"], questions), task(request["state"], questions)
            assert found.keys() == expected.keys()
            for name in expected:
                np.testing.assert_equal(dataclasses.asdict(found[name]), dataclasses.asdict(expected[name]))
    Decide.from_run(directory).save(directory)
    with pytest.raises(ValueError):
        task.save(runs[0])
