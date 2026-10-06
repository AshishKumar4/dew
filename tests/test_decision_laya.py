"""dew.decision against Laya's own code (NandhaKishorM/laya a4a8921), under the float64 rule."""

import json
import os
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.decision import DecisionInputs, Encoded, LayaCheckpoint, MarkerLayout, Question

FIXTURES = Path(__file__).parent / "fixtures" / "laya"
TINY = FIXTURES / "tiny"


def cases() -> dict:
    return json.loads((FIXTURES / "cases.json").read_text())


def rows(checkpoint: LayaCheckpoint, *, parallel: bool):
    """Every question of every case, laid out by Dew, keyed as the fixture keys them."""
    layout = MarkerLayout(checkpoint.layout.max_len, checkpoint.layout.head_max_len, parallel=parallel)
    name = "parallel" if parallel else "sequential"
    for case, request in cases().items():
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
    checkpoint = LayaCheckpoint.load(TINY)
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


def logits(checkpoint: LayaCheckpoint, *, parallel: bool) -> dict[str, np.ndarray]:
    """Each question's option logits, one row per question as Laya runs them."""
    found = {}
    for key, _, encoded in rows(checkpoint, parallel=parallel):
        inputs = DecisionInputs.collate([encoded], checkpoint.specials.pad)
        found[key] = np.asarray(checkpoint.model.logits(checkpoint.variables, inputs))[0, 0]
    return found


@pytest.mark.parametrize("parallel", [False, True])
def test_laya_logits_match_laya_under_the_float64_rule(parallel):
    """Every case's option logits, sequential and parallel, held to twice
    Laya's own fp32 distance from its float64 run."""
    checkpoint = LayaCheckpoint.load(TINY, attention_impl="xla")
    found = logits(checkpoint, parallel=parallel)
    with np.load(TINY / "logits.npz") as reference:
        keys = sorted(found)
        assert_as_exact_as_the_reference(
            np.concatenate([found[key] for key in keys]),
            np.concatenate([reference[key] for key in keys]),
            np.concatenate([reference[f"{key}/f64"] for key in keys]), "Laya logits")


def test_the_rule_catches_a_head_that_ignores_the_question_type():
    """The type embedding is the only place a choice and a noul differ in
    the head; dropping it moves the logits past the rule."""
    checkpoint = LayaCheckpoint.load(TINY, attention_impl="xla")
    with np.load(TINY / "logits.npz") as reference:
        found = {}
        for key, _, encoded in rows(checkpoint, parallel=False):
            inputs = DecisionInputs.collate([encoded], checkpoint.specials.pad)
            one_type = replace(inputs, kinds=jnp.zeros_like(inputs.kinds))
            found[key] = np.asarray(checkpoint.model.logits(checkpoint.variables, one_type))[0, 0]
        keys = sorted(found)
        with pytest.raises(AssertionError, match=r"allowed 2\.0"):
            assert_as_exact_as_the_reference(
                np.concatenate([found[key] for key in keys]),
                np.concatenate([reference[key] for key in keys]),
                np.concatenate([reference[f"{key}/f64"] for key in keys]), "Laya logits, one type")


def test_padded_rows_score_as_they_do_alone():
    """Every question of every case in one batch: padding reaches neither
    the encoder's nor the head's attention, so the batch is held to Laya's
    one-row runs under the same rule, and a short row's missing options
    score -inf."""
    checkpoint = LayaCheckpoint.load(TINY, attention_impl="xla")
    laid = list(rows(checkpoint, parallel=False))
    inputs = DecisionInputs.collate([encoded for _, _, encoded in laid], checkpoint.specials.pad)
    batched = np.asarray(checkpoint.model.logits(checkpoint.variables, inputs))[:, 0]
    widths = [len(markers(encoded)) for _, _, encoded in laid]
    assert all(np.all(np.isneginf(batched[row, width:])) for row, width in enumerate(widths))
    with np.load(TINY / "logits.npz") as reference:
        assert_as_exact_as_the_reference(
            np.concatenate([batched[row, :width] for row, width in enumerate(widths)]),
            np.concatenate([reference[key] for key, _, _ in laid]),
            np.concatenate([reference[f"{key}/f64"] for key, _, _ in laid]), "batched Laya logits")


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("DEW_NETWORK_TESTS"), reason="downloads the 1.7 GB English checkpoint")
@pytest.mark.skipif(jax.default_backend() != "gpu", reason="XLA:CPU's fp32 dot rounds more than MKL's")
def test_the_released_english_checkpoint_matches_laya():
    """convaiinnovations/laya at the pinned revision against tools/laya_reference.py --real.

    It runs on a GPU at HIGHEST precision, where Dew's RMS distance from
    float64 measured 0.89 (sequential) and 1.25 (parallel) times Laya's on
    the RTX 4080. On the CPU it measures 2.20 and 1.70: XLA:CPU's fp32 dot
    accumulates with 2.3 times the RMS error of the MKL dot torch runs (a
    [96, 1024] x [1024, 1024] product, 5.7e-7 against 2.5e-7, equal at a
    reduction of 64), which ModernBERT-large's 1024- and 2624-long
    reductions carry through 28 layers."""
    source = json.loads((FIXTURES / "real" / "source.json").read_text())
    with jax.default_matmul_precision("highest"):
        checkpoint = LayaCheckpoint.load(source["repo"], revision=source["revision"], attention_impl="xla")
        compare_released(checkpoint)


def compare_released(checkpoint: LayaCheckpoint) -> None:
    expected = json.loads((FIXTURES / "real" / "layouts.json").read_text())
    for parallel in (False, True):
        for key, _, encoded in rows(checkpoint, parallel=parallel):
            assert list(encoded.tokens) == expected[key]["ids"], key
        found = logits(checkpoint, parallel=parallel)
        keys = sorted(found)
        with np.load(FIXTURES / "real" / "logits.npz") as reference:
            assert_as_exact_as_the_reference(
                np.concatenate([found[key] for key in keys]),
                np.concatenate([reference[key] for key in keys]),
                np.concatenate([reference[f"{key}/f64"] for key in keys]), "released Laya logits")
