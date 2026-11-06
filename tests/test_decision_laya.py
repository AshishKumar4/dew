"""dew.decision against Laya's own code (NandhaKishorM/laya a4a8921), under the float64 rule."""

import functools
import json
import os
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import GateFailure, Inconclusive, as_given, assert_as_exact, indexed_recording, recording
from reference_runs import assert_as_exact_as_recorded

from dew.decision import Decide, DecisionInputs, Encoded, Question
from dew.decision.model import DecisionModel

FIXTURES = Path(__file__).parent / "fixtures" / "laya"
TINY = FIXTURES / "tiny"


def cases() -> dict:
    return json.loads((FIXTURES / "cases.json").read_text())


def rows(checkpoint: Decide, *, parallel: bool, requests: dict | None = None):
    """Every question of every case (of `requests`, the fixture's by default), laid out by Dew, keyed as
    the fixture keys them."""
    # The release's layout, its budgets as it states them, in each option order.
    layout = replace(checkpoint.layout, parallel=parallel)
    name = "parallel" if parallel else "sequential"
    for case, request in (cases() if requests is None else requests).items():
        state = layout.state(checkpoint.tokenizer, checkpoint.specials, request["state"])
        for qid, wire in request["questions"].items():
            question = Question.from_wire(wire)
            encoded = layout.encode(checkpoint.tokenizer, checkpoint.specials, question, state,
                                    conversation=isinstance(request["state"], list))
            yield f"{name}/{case}/{qid}", question, encoded


@pytest.mark.parametrize("parallel", [False, True])
def test_the_marker_layout_lays_out_what_laya_does(parallel):
    """Token for token, marker for marker: instructions cut to the option
    budget, options cut evenly past it, a conversation keeping its end, the
    marker text cut out of what the caller wrote, and the parallel positions."""
    checkpoint = Decide.from_pretrained(TINY)
    expected = json.loads((TINY / "layouts.json").read_text())
    for key, _, encoded in rows(checkpoint, parallel=parallel):
        assert list(encoded.tokens) == expected[key]["ids"], key
        assert markers(encoded) == expected[key]["markers"], key
        if parallel:
            assert list(encoded.positions or ()) == expected[key]["positions"], key
            assert list(encoded.slots or ()) == expected[key]["slots"], key


def markers(encoded: Encoded) -> list[int]:
    """The marker of each option of the row's one question."""
    return [start for start, _ in encoded.questions[0].options]


FIXTURE = TINY / "seeds.npz"
LAYA_SEEDS = 64
"""The declared seeds, sized by `seeds_for` paired as the recording confirmed them
(tools/venv_reference.py's `laya` site, tools/laya_worker.py)."""
OUTPUTS = ("sequential", "parallel")
"""The leaves `dew` gives: every question's option logits of every case in key order, by layout."""


@functools.cache
def tiny() -> Decide:
    return Decide.from_pretrained(TINY, attention_impl="xla")


@functools.cache
def laid(parallel: bool) -> tuple[DecisionInputs, ...]:
    """Each question of every case collated alone, as Laya runs them, in key order, on the host: a
    traced `dew` reads them as constants."""
    checkpoint = tiny()
    with jax.ensure_compile_time_eval():
        return tuple(jax.tree.map(np.asarray, DecisionInputs.collate([encoded], checkpoint.specials.pad))
                     for _, _, encoded in sorted(rows(checkpoint, parallel=parallel), key=lambda row: row[0]))


def draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's weights for the tiny checkpoint's model: kernels and embeddings at their initializer's
    scale, norms' scales about one and biases off zero, under Dew's paths (`dew.`) and Laya's names
    (`param.`, `dew.decision.laya.laya_tensors`), the requests the fixture's cases."""
    from flax.traverse_util import flatten_dict, unflatten_dict

    from dew.decision.laya import laya_tensors

    rng = np.random.default_rng([1557, 7, seed])
    checkpoint = tiny()
    drawn = {}
    for path, value in flatten_dict(checkpoint.variables).items():
        shape = np.shape(value)
        if path[-1] == "scale":
            leaf = rng.normal(1.0, 0.3, shape)
        elif path[-1] == "bias":
            leaf = rng.normal(0.0, 0.3, shape)
        else:
            leaf = rng.normal(0.0, 1.0, shape) / np.sqrt(np.prod(shape[:-1]))
        drawn[path] = leaf.astype(np.float32)
    tensors, _ = laya_tensors(checkpoint.model, unflatten_dict(drawn))
    return {**{f"dew.{'/'.join(path)}": value for path, value in drawn.items()},
            **{f"param.{name}": np.asarray(tensors[name], np.float32) for name in tensors}}


def model_at(dtype) -> DecisionModel:
    """The tiny checkpoint's model computing in `dtype`, its backbone's and head's."""
    model = tiny().model
    if dtype == jnp.float32:
        return model
    return DecisionModel(model.backbone.clone(dtype=dtype), model.head.clone(dtype=dtype))


def variables_of(dtype, inputs) -> dict:
    from flax.traverse_util import unflatten_dict

    return unflatten_dict({tuple(key.removeprefix("dew.").split("/")): jnp.asarray(value, dtype)
                           for key, value in inputs.items() if key.startswith("dew.")})


def dew(dtype, inputs, *, typed: bool = True) -> dict:
    """Dew's option logits at `dtype` under a draw's weights, each question alone (`laid`); `typed`
    False drops the question type, the one place a choice and a noul differ in the head."""
    model, variables = model_at(dtype), variables_of(dtype, inputs)

    def scored(row):
        row = row if typed else replace(row, kinds=jnp.zeros_like(row.kinds))
        return model.logits(variables, row)[0, 0]
    return {layout: jnp.concatenate([scored(row) for row in laid(layout == "parallel")])
            for layout in OUTPUTS}


def test_laya_logits_match_laya_under_the_float64_rule():
    """Every case's option logits, sequential and parallel, over drawn weights, as exact as Laya's
    float32 run against its float64 run (check 1)."""
    assert_as_exact_as_recorded(FIXTURE, "laya", LAYA_SEEDS, draw, dew, "Laya's option logits")


def test_the_rule_catches_a_head_that_ignores_the_question_type():
    """The type embedding is the only place a choice and a noul differ in
    the head; dropping it moves the logits past the rule: check 1 decides
    against it, the truth the typed head's own float64 run."""
    drawn, reference, truth = indexed_recording(recording(FIXTURE), "laya", LAYA_SEEDS, inputs_of=draw,
                                                truth_fn=lambda inputs: dew(jnp.float64, inputs))
    untyped = jax.jit(lambda inputs: dew(jnp.float32, inputs, typed=False))
    with pytest.raises(GateFailure) as failed:
        assert_as_exact(as_given(lambda index, inputs: untyped(inputs)), reference, drawn,
                        "logits without the question type", seeds=LAYA_SEEDS, truth_fn=truth)
    assert not isinstance(failed.value, Inconclusive)


def test_padded_rows_score_as_they_do_alone():
    """Every question of every case in one batch: padding reaches neither
    the encoder's nor the head's attention, so the batch is held to Laya's
    one-row runs under the same rule, and a short row's missing options
    score -inf."""
    checkpoint = tiny()
    ordered = sorted(rows(checkpoint, parallel=False), key=lambda row: row[0])
    batch = DecisionInputs.collate([encoded for _, _, encoded in ordered], checkpoint.specials.pad)
    widths = [len(markers(encoded)) for _, _, encoded in ordered]
    batched = np.asarray(checkpoint.model.logits(checkpoint.variables, batch))[:, 0]
    assert all(np.all(np.isneginf(batched[row, width:])) for row, width in enumerate(widths))

    def in_one_batch(dtype, inputs):
        scores = model_at(dtype).logits(variables_of(dtype, inputs), batch)[:, 0]
        return {**dew(dtype, inputs), "sequential": jnp.concatenate(
            [scores[row, :width] for row, width in enumerate(widths)])}
    assert_as_exact_as_recorded(FIXTURE, "laya", LAYA_SEEDS, draw, in_one_batch, "batched Laya logits")


REAL_FIXTURE = FIXTURES / "real" / "seeds.npz"
REAL_SEEDS = 8
"""The released checkpoint's declared seeds, sized by `seeds_for` paired as the recording confirmed them
(tools/venv_reference.py's `laya-real` site)."""
REAL_OUTPUTS = OUTPUTS
WORDS = ("account", "billing", "charge", "refund", "march", "payment", "rent", "today", "cancel", "issue",
         "please", "fix", "error", "login", "password", "upgrade", "plan", "price", "invoice", "card",
         "bank", "late", "again", "twice", "need", "help", "now", "team", "support", "reply", "order",
         "shipping", "broken", "missing", "slow", "angry", "calm", "thanks", "week", "month", "service",
         "contract", "renewal", "discount", "trial", "email", "phone", "address", "delivery", "package")
"""The words a drawn request's text is made of (`drawn_requests`)."""


def drawn_requests(seed: int) -> dict:
    """The fixture's cases with each state's text drawn (`WORDS`, each sentence as long as the case's
    own, give or take a quarter), every question as it is: the structure the released checkpoint's site
    holds fixed, the values a seed draws."""
    rng = np.random.default_rng([1557, 8, seed])

    def text(written: str) -> str:
        length = max(1, int(len(written.split()) * rng.uniform(0.75, 1.25)))
        return " ".join(rng.choice(WORDS, length))

    def state(value):
        if isinstance(value, str):
            return value if value == "[MASK]" else text(value)
        if isinstance(value, list):
            return [state(item) for item in value]
        if isinstance(value, dict):
            return {key: state(item) for key, item in value.items()}
        return value
    return {case: {**request, "state": state(request["state"])} for case, request in cases().items()}


def REAL_draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's requests (`drawn_requests`), as the JSON text the worker reads."""
    return {"requests": np.asarray(json.dumps(drawn_requests(seed)))}


@functools.cache
def released() -> Decide:
    source = json.loads((FIXTURES / "real" / "source.json").read_text())
    return Decide.from_pretrained(source["repo"], revision=source["revision"], attention_impl="xla")


def REAL_dew(dtype, inputs) -> dict[str, np.ndarray]:
    """The released checkpoint's option logits at `dtype` over a draw's requests, each question alone,
    by layout, every question's concatenated in key order: on the host, its requests text."""
    checkpoint = released()
    model = checkpoint.model if dtype == jnp.float32 else DecisionModel(
        checkpoint.model.backbone.clone(dtype=dtype), checkpoint.model.head.clone(dtype=dtype))
    variables = jax.tree.map(lambda value: jnp.asarray(value, dtype), checkpoint.variables)
    requests = json.loads(str(inputs["requests"]))
    found = {}
    # HIGHEST: XLA:GPU's fp32 dot is otherwise TF32's, where Laya's torch dot is fp32's.
    with jax.default_matmul_precision("highest"):
        for layout in OUTPUTS:
            laid = sorted(rows(checkpoint, parallel=layout == "parallel", requests=requests),
                          key=lambda row: row[0])
            found[layout] = np.concatenate([np.asarray(model.logits(variables, DecisionInputs.collate(
                [encoded], checkpoint.specials.pad)))[0, 0] for _, _, encoded in laid])
    return found


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("DEW_NETWORK_TESTS"), reason="downloads the 1.7 GB English checkpoint")
@pytest.mark.gpu(reason="XLA:CPU's fp32 dot rounds more than MKL's")
def test_the_released_english_checkpoint_matches_laya():
    """convaiinnovations/laya at the pinned revision: its layouts token for token on the fixture's cases
    (tools/laya_reference.py --real), and its option logits over drawn requests as exact as Laya's float32
    run against its float64 run (check 1, tools/venv_reference.py's `laya-real` recording, Laya's runs on
    the CUDA GPU the test runs Dew on).

    It runs on a GPU at HIGHEST precision (`REAL_dew`): XLA:CPU's fp32 dot accumulates with 2.3 times the
    RMS error of the MKL dot torch runs (a [96, 1024] x [1024, 1024] product, 5.7e-7 against 2.5e-7, equal
    at a reduction of 64), which ModernBERT-large's 1024- and 2624-long reductions carry through 28
    layers."""
    checkpoint = released()
    expected = json.loads((FIXTURES / "real" / "layouts.json").read_text())
    for parallel in (False, True):
        for key, _, encoded in rows(checkpoint, parallel=parallel):
            assert list(encoded.tokens) == expected[key]["ids"], key
    drawn, reference, truth = indexed_recording(recording(REAL_FIXTURE), "laya-real", REAL_SEEDS,
                                                inputs_of=REAL_draw,
                                                truth_fn=lambda inputs: REAL_dew(jnp.float64, inputs))
    assert_as_exact(as_given(lambda index, inputs: REAL_dew(jnp.float32, inputs)), reference, drawn,
                    "released Laya logits", seeds=REAL_SEEDS, truth_fn=truth)


