"""Fixed-weight decision copies in two option orders, alone and sharing a packed window."""

import functools
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import as_given, assert_as_exact, widened

from dew import algebra
from dew.decision import Budget, Choice, Decide
from dew.decision.model import DecisionModel
from dew.interop import Pretrained


FIXTURES = Path(__file__).parent / "fixtures"
QUESTION = Choice("team", {"pay": "payments", "tech": "outages", "sales": "pricing"})
ORDERS = ((0, 1, 2), (1, 2, 0))
COMPANION = Choice("which department", ["billing", "technical", "sales", "support", "account", "other"])
SEEDS = 16


@pytest.fixture(scope="module", params=("float32", "bfloat16"))
def copies(request, tmp_path_factory):
    source = tmp_path_factory.mktemp("rotated-release")
    backbone = FIXTURES / "hf" / "qwen38-dense-tiny"
    joint = FIXTURES / "joint_checkpoint" / "tiny"
    for item in [*backbone.iterdir(), joint / "joint_head_config.json",
                 joint / "joint_head.safetensors"]:
        (source / item.name).symlink_to(item.resolve())
    dtype = request.param
    task = Decide.of(Pretrained.load(source, attention_impl="xla", dtype=dtype, param_dtype=dtype,
                                    max_seq_len=2048))
    task = replace(task, layout=replace(task.layout, max_len=2048),
                   budget=Budget(rows=1, tokens=2048))
    with jax.enable_x64(new_val=True):
        model = task.model
        wide = replace(task, model=DecisionModel(model.backbone.clone(dtype=jnp.float64),
                                                 model.head.clone(dtype=jnp.float64)),
                       variables=widened(task.variables))

    @functools.lru_cache(maxsize=128)
    def rows_of(seed):
        rng = np.random.default_rng(int(seed))
        state = " ".join(rng.choice(["pay", "bug", "vip", "new", "old", "net", "app", "pro"], 4))
        rows = [task.layout.rows(task.tokenizer, task.specials, state, {"q": QUESTION},
                                 orders={"q": order})[0] for order in ORDERS]
        rows += task.layout.rows(task.tokenizer, task.specials, "invoice " * 20, {"other": COMPANION})
        return rows

    def probabilities(held, seed, together):
        rows = rows_of(int(seed))
        scores = held.scored(rows)[:2] if together else [held.scored([row])[0] for row in rows[:2]]
        found = []
        for row, logits in zip(rows[:2], scores, strict=True):
            canonical = np.empty(3, np.asarray(logits).dtype)
            canonical[list(row.questions[0].order)] = logits[0, :3]
            found.append(held.calibration.probabilities(QUESTION, canonical))
        return np.stack(found)

    rows = rows_of(0)
    plan = task.planned()
    assert all(plan.add(row) for row in rows)
    assert len(set(plan.windows())) == 1
    assert tuple(row.questions[0].order for row in rows[:2]) == ORDERS
    assert rows[0].tokens != rows[1].tokens
    print("ROTATED_SETUP", dtype, [len(row.tokens) for row in rows], plan.windows(), flush=True)
    print("ROTATED_NATIVE_ALONE", dtype, probabilities(task, 0, False).tolist(), flush=True)
    print("ROTATED_NATIVE_PACKED", dtype, probabilities(task, 0, True).tolist(), flush=True)
    with jax.enable_x64(new_val=True), algebra.passes(False):
        reference = probabilities(wide, 0, False)
        packed = probabilities(wide, 0, True)
    print("ROTATED_FLOAT64_ALONE", dtype, reference.tolist(), flush=True)
    print("ROTATED_FLOAT64_PACKED", dtype, packed.tolist(), flush=True)
    print("ROTATED_FLOAT64_ORDER_BIAS", dtype, (reference[1] - reference[0]).tolist(), flush=True)
    return task, wide, probabilities


def draw(seed):
    """Only the request changes; every draw uses the same loaded weights."""
    return (np.int64(seed),)


def test_each_rotated_copy_and_their_average_agree_alone_and_packed(copies):
    task, wide, probabilities = copies

    def scored(held, seed, together):
        found = probabilities(held, seed, together)
        return {"copies": found, "average": found.mean(axis=0)}

    def truth(seed):
        with jax.enable_x64(new_val=True), algebra.passes(False):
            return scored(wide, seed, False)

    assert_as_exact(as_given(lambda seed: scored(task, seed, True)),
                    as_given(lambda seed: scored(task, seed, False)), draw,
                    "each rotated copy alone versus packed", seeds=SEEDS, truth_fn=truth)


def test_the_assumed_option_order_equivalence(copies):
    task, wide, probabilities = copies

    def truth(seed):
        with jax.enable_x64(new_val=True), algebra.passes(False):
            return probabilities(wide, seed, False)[0]

    assert_as_exact(as_given(lambda seed: probabilities(task, seed, False)[1]),
                    as_given(lambda seed: probabilities(task, seed, False)[0]), draw,
                    "assumed option-order equivalence", seeds=SEEDS, truth_fn=truth)
