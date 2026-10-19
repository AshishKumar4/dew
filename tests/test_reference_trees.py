"""The summation trees' interpreter (tests/reference_trees.py `evaluated`) against the math it
reorders: the same function, each reduction under a declared tree."""

import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference, widened
from reference_trees import evaluated, tree_contract, tree_sum

from dew.interop import Pretrained

TINY = Path(__file__).resolve().parent / "fixtures" / "hf" / "dory-tiny"
TREES = ("pairwise", "blocked-4")
"""The trees the interpreter is checked under: pairwise, and pairwise blocks of 4 summed in turn."""


def every_reduction(x, w, scan_x, flag):
    """A function holding each reduction the interpreter rewrites, inside each construct it walks: a
    batched dot under jit, a sum over two axes, a scan, a while loop and a cond."""
    @jax.jit
    def projected(x, w):
        return jnp.einsum("btk,kn->btn", x, w)

    def step(total, row):
        return total + jnp.sum(row * row), jnp.dot(row, w[:, 0])

    hidden = projected(x, w)
    summed = jnp.sum(hidden, axis=(0, 2))
    total, scanned = jax.lax.scan(step, jnp.zeros((), x.dtype), scan_x)
    looped = jax.lax.while_loop(lambda carry: carry[0] < 3,
                                lambda carry: (carry[0] + 1, carry[1] + jnp.sum(carry[1])), (0, summed))[1]
    branched = jax.lax.cond(flag, lambda: jnp.dot(summed, summed), lambda: jnp.sum(summed))
    return hidden, looped, total, scanned, branched


def operands(dtype, integers: bool, seed: int = 0):
    keys = jax.random.split(jax.random.key(seed), 3)
    draw = ((lambda key, shape: jax.random.randint(key, shape, -3, 4).astype(dtype)) if integers
            else (lambda key, shape: jax.random.normal(key, shape, dtype)))
    flag = jnp.asarray(1, bool)
    return draw(keys[0], (2, 5, 37)), draw(keys[1], (37, 11)), draw(keys[2], (6, 37)), flag


@pytest.mark.parametrize("tree", TREES)
def test_sums_that_round_nowhere_come_out_exactly_under_every_tree(tree):
    """Small integers sum exactly in any order: under each tree the interpreted function gives the
    function's own values to the bit, through jit, scan, while and cond."""
    args = operands(jnp.float32, integers=True)
    found = evaluated(every_reduction, tree)(*args)
    for mine, theirs in zip(found, every_reduction(*args), strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


@pytest.mark.parametrize("tree", TREES)
def test_each_reduction_is_summed_as_its_tree_sums_it(tree):
    """A dot and a sum over random terms come out as `tree_contract` and `tree_sum` give them to the
    bit, not as XLA's own order does: each reduction runs under the tree."""
    x, w = jax.random.normal(jax.random.key(1), (3, 37)), jax.random.normal(jax.random.key(2), (37, 5))
    dot = evaluated(jnp.dot, tree)(x, w)
    np.testing.assert_array_equal(np.asarray(dot), np.asarray(
        tree_contract(x[:, None, :], w.T[None, :, :], -1, tree)))
    summed = evaluated(functools.partial(jnp.sum, axis=1), tree)(x)
    np.testing.assert_array_equal(np.asarray(summed), np.asarray(tree_sum(x, 1, tree)))


def test_a_contraction_too_large_for_one_piece_is_taken_a_chunk_of_columns_at_a_time(monkeypatch):
    """A contraction past `PRODUCTS` products is summed in chunks of the right operand's columns, the
    last padded, to the same bits as in one piece."""
    import reference_trees

    x, w = jax.random.normal(jax.random.key(3), (4, 64)), jax.random.normal(jax.random.key(4), (64, 23))
    whole = evaluated(jnp.dot, "pairwise")(x, w)
    monkeypatch.setattr(reference_trees, "PRODUCTS", 4 * 64 * 5)
    chunked = evaluated(jnp.dot, "pairwise")(x, w)
    np.testing.assert_array_equal(np.asarray(chunked), np.asarray(whole))


def test_a_primitive_whose_order_the_tree_does_not_set_is_refused():
    with pytest.raises(NotImplementedError, match="cumsum"):
        evaluated(jnp.cumsum, "pairwise")(jnp.ones(8))


@pytest.mark.parametrize("tree", TREES)
def test_the_tiny_dory_under_each_tree_is_as_exact_as_under_xlas(tree):
    """The tiny Dory's float32 logits, every one of its sums under the tree, within the rule of
    XLA's own float32 forward's distance from float64: an equally exact reference."""
    config = json.loads((TINY / "config.json").read_text())
    loaded = Pretrained.load(str(TINY), dtype="float32", attention_impl="xla")
    ids = jnp.asarray(np.random.default_rng(0).integers(0, config["vocab_size"], (1, 24)), jnp.int32)
    mine = evaluated(loaded.model.apply, tree)(loaded.variables, ids)
    plain = loaded.model.apply(loaded.variables, ids)
    with jax.enable_x64(new_val=True):
        truth = np.asarray(loaded.model.clone(dtype=jnp.float64).apply(widened(loaded.variables), ids))
    assert not np.array_equal(np.asarray(mine), np.asarray(plain))
    assert_as_exact_as_the_reference(np.asarray(mine), np.asarray(plain), truth,
                                     f"the tiny Dory under {tree}")
