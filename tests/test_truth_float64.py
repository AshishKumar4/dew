"""Check 1's truth is float64's (`reference_error._of_float64`): a null control of truths that are, and each
way a truth has been a second fp32 run, injected."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arm_arithmetic import float64_held, stored_as
from reference_error import STAGES_LOG, NarrowTruth, as_given, assert_as_exact

WEIGHT = np.random.default_rng(5430).normal(size=(64, 8)).astype(np.float32)
"""A parameter the arms close over, as a model's variables are."""


def held(seed):
    """Rows alone, the weight closed over."""
    return (jnp.asarray(np.random.default_rng([5430, seed]).normal(size=(16, 64)), jnp.float32),)


def drawn(seed):
    """Rows and the weight, both drawn."""
    return (*held(seed), jnp.asarray(WEIGHT))


def product(x, weight):
    return x @ weight


def closing(x):
    """The product over the closed-over float32 weight, at its rows' dtype where they are wider."""
    return x @ WEIGHT


def rounded(x, weight):
    """A product that states its rounding: stored bfloat16, read at its operands' dtype."""
    return (x @ weight).astype(jnp.bfloat16).astype(x.dtype)


def selected(x, weight):
    """The two largest sizes of a product stored bfloat16, negated: its rounding stated once, what is
    selected and negated of it exact."""
    stored = jnp.abs((x @ weight).astype(jnp.bfloat16))
    return (-jax.lax.top_k(stored, 2)[0]).astype(x.dtype)


def dequantized(x, weight):
    """A product over rows their format defines by a division in its own arithmetic."""
    return (x / jnp.asarray(3, x.dtype)) @ weight


def dequantized_declared(x, weight):
    """`dequantized` as its truth states it: the division in float64, declared float32's."""
    return stored_as(x / 3, jnp.float32) @ weight


def dropped_out(x, weight):
    """The product with the entries a float32 uniform draw keeps, as a dropout draws its mask in float32 in
    every mode: the draw's arithmetic decides, and rounds no value an output reads."""
    kept = jax.random.uniform(jax.random.key(5430), (16, 8), jnp.float32) < 0.75
    return jnp.where(kept, x @ weight, 0)


def with_a_dropped_value(x, weight):
    """The float64 product beside an fp32 one nothing reads."""
    jnp.tanh(x.astype(jnp.float32) @ weight.astype(jnp.float32))
    return x @ weight


def fixed_float32(x, weight):
    """A model of a fixed float32 dtype: whatever it is given, it computes in fp32."""
    return jnp.asarray(x, jnp.float32) @ jnp.asarray(weight, jnp.float32)


def computed_narrow(x, weight):
    """An fp32 product widened at the end: float64 by its dtype, fp32 by its values."""
    return fixed_float32(x, weight).astype(jnp.float64)


def with_its_place(x, weight):
    """The product and the place of each row's largest value, an integer leaf."""
    return {"sum": x @ weight, "most": jnp.argmax(x @ weight, -1)}


def with_no_history(x, weight):
    """The product beside a leaf of no entries, float32 by its declaration."""
    return {"sum": x @ weight, "history": jnp.zeros((16, 0, 8), jnp.float32)}


def in_numpy(dtype):
    """The product computed on the host in `dtype`."""
    return lambda x, weight: np.asarray(x, dtype) @ np.asarray(weight, dtype)


def two_leaves(narrow):
    """A wide leaf beside one computed as `narrow` computes."""
    return lambda x, weight: {"wide": x @ weight, "other": narrow(x, weight)}


NULL = {
    "the reference on widened inputs": (product, drawn, None),
    "a float64 formula of its own": (product, drawn, lambda x, weight: jnp.einsum("rd,dw->rw", x, weight)),
    "parameters held in float64": (closing, held, lambda x: x @ float64_held(WEIGHT)),
    "parameters left float32, read exactly": (closing, held, None),
    "a leaf of no entries in float32": (with_no_history, drawn, None),
    "a stated rounding": (rounded, drawn, None),
    "a stated rounding's entries selected and negated": (selected, drawn, None),
    "a format's own division, declared": (dequantized, drawn, dequantized_declared),
    "a decision drawn in float32": (dropped_out, drawn, None),
    "an fp32 value no output reads": (product, drawn, with_a_dropped_value),
    "an integer leaf beside": (with_its_place, drawn, None),
    "a NumPy truth in float64": (product, drawn, as_given(in_numpy(np.float64))),
}
"""Truths that are float64's: each site's arm (Dew's and the reference alike), its draw and its truth (None:
the reference on the inputs widened)."""

INJECTED = {
    "a model of a fixed float32 dtype as its own truth": (fixed_float32, drawn, None, ""),
    "an fp32 product widened at the end": (product, drawn, computed_narrow, ""),
    "one leaf of two computed in fp32": (two_leaves(product), drawn, two_leaves(computed_narrow),
                                         "['other']"),
    "one leaf of two left float32": (two_leaves(product), drawn, two_leaves(fixed_float32), "['other']"),
    "a weight drawn and applied in fp32": (product, drawn, lambda x, weight: (x @ weight) * (
        1 + jax.random.uniform(jax.random.key(5430), (16, 8), jnp.float32) * 0).astype(jnp.float64), ""),
    "a NumPy truth in float32": (product, drawn, as_given(in_numpy(np.float32)), ""),
}
"""Truths that are a second fp32 run, each with the leaf its refusal names."""


@pytest.mark.parametrize("name", NULL)
def test_a_truth_that_computes_in_float64_is_judged(name):
    """The null control: a site whose truth is float64's (the reference on widened inputs, a formula of its
    own, parameters held in float64 or left float32 and read exactly, a stated rounding and what is selected
    of it, a format's own division declared (`stored_as`), a float32 draw read only by a comparison, a dead
    fp32 value, an integer leaf, a float32 leaf of no entries, a float64 NumPy truth) is judged as before,
    its arm on par with itself."""
    arm, draw, truth = NULL[name]
    assert_as_exact(arm, arm, draw, name, seeds=8, truth_fn=truth)


@pytest.mark.parametrize("name", INJECTED)
def test_a_truth_that_is_a_second_fp32_run_is_refused_by_its_leaf(name):
    """The injected faults: a truth that computes in fp32 (a model of a fixed dtype, an fp32 product
    widened at the end, a float32 leaf, the same float32 draw read as a value, a float32 NumPy truth) is
    refused before any verdict, naming the leaf, where an arm on par with itself would otherwise pass
    against it."""
    arm, draw, truth, leaf = INJECTED[name]
    with pytest.raises(NarrowTruth) as refused:
        assert_as_exact(arm, arm, draw, name, seeds=8, truth_fn=truth)
    assert refused.value.leaf == leaf


@pytest.mark.parametrize("operation", [jnp.add, jnp.subtract, jnp.multiply, jnp.divide,
                                       lambda a, b: jnp.sqrt(jnp.abs(a))],
                         ids=["sum", "difference", "product", "quotient", "square root"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=str)
def test_one_operation_declared_a_formats_is_the_formats_own_result(operation, dtype):
    """What a truth declares a format's own (`stored_as`) is that format's result bit for bit: one sum,
    difference, product, quotient or square root of the format's values, computed in float64 and rounded
    once to the format, is the operation run in the format, over a million pairs across the format's
    range."""
    rng = np.random.default_rng(5430)
    a, b = (jnp.asarray(rng.standard_normal(1 << 20) * 2.0 ** rng.integers(-40, 40, 1 << 20), dtype)
            for _ in range(2))
    own = operation(a, b)
    with jax.enable_x64(new_val=True):
        declared = stored_as(operation(a.astype(jnp.float64), b.astype(jnp.float64)), dtype)
        np.testing.assert_array_equal(np.asarray(declared), np.asarray(own, np.float64))


def test_a_judgement_logs_whether_its_reference_equalled_the_truth(tmp_path, monkeypatch):
    """A site's stage line says whether its reference was the truth bit for bit on every first-stage seed:
    true of sums fp32 holds exactly (whole numbers), false of a product that rounds."""
    log = tmp_path / "stages.jsonl"
    monkeypatch.setenv(STAGES_LOG, str(log))

    def whole(seed):
        rng = np.random.default_rng(seed)
        return tuple(jnp.asarray(rng.integers(-8, 9, shape), jnp.float32) for shape in ((16, 64), (64, 8)))

    assert_as_exact(product, product, whole, "whole numbers", seeds=8)
    assert_as_exact(product, product, drawn, "a product that rounds", seeds=8)
    logged = {entry["label"]: entry["exact"] for entry in map(json.loads, log.read_text().splitlines())}
    assert logged == {"whole numbers": True, "a product that rounds": False}
