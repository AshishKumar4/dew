"""A vocabulary head's logits at a fixed set of columns (`dew.nn.protocols.OutputTable.columns`): the
head's own logits restricted to those ids, on par from float64 with restricting the full logits
(check 1), scale, bias and softcap included, in fp32 and bf16, and through its gradient, whose
transpose scatters back into the table; `exact` computes the product in fp32 whatever the table's
compute; ids outside the vocabulary are refused."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, with_gradients

from dew.nn.protocols import OutputTable

VOCAB, FEATURES = 97, 16
IDS = (3, 40, 41, 96, 0)
COLUMN_SEEDS = 8
"""Check 1's declared seeds for the restricted columns against the full logits restricted."""


def table(rng, dtype, vocab_major: bool) -> OutputTable:
    matrix = rng.standard_normal((VOCAB, FEATURES) if vocab_major else (FEATURES, VOCAB)) * 0.5
    return OutputTable(jnp.asarray(matrix, dtype), vocab_major, bias=jnp.asarray(rng.standard_normal(VOCAB),
                                                                               jnp.float32),
                       softcap=3.0, compute=dtype, scale=jnp.asarray(1 + rng.random(VOCAB), jnp.float32))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("vocab_major", [True, False])
def test_a_tables_columns_are_its_logits_restricted_to_them(dtype, vocab_major):
    def draw(seed):
        rng = np.random.default_rng([0xC01, seed])
        head = table(rng, dtype, vocab_major)
        hidden = jnp.asarray(rng.standard_normal((3, 5, FEATURES)), dtype)
        cotangent = jnp.asarray(rng.standard_normal((3, 5, len(IDS))), jnp.float32)
        return head.matrix, hidden, cotangent, head

    def columns(matrix, hidden, cotangent, head):
        return with_gradients(lambda matrix, hidden: OutputTable(
            matrix, vocab_major, head.bias, head.softcap, compute=head.compute, scale=head.scale).columns(
            hidden, IDS))(matrix, hidden, cotangent)

    def restricted(matrix, hidden, cotangent, head):
        return with_gradients(lambda matrix, hidden: OutputTable(
            matrix, vocab_major, head.bias, head.softcap, compute=head.compute, scale=head.scale).logits(
            hidden)[..., jnp.asarray(IDS)])(matrix, hidden, cotangent)

    def truth(matrix, hidden, cotangent, head):
        # The float64 truth computes in float64, the table's own compute widened with its leaves.
        return restricted(matrix, hidden, cotangent, dataclasses.replace(head, compute=None))

    assert_as_exact(columns, restricted, draw, f"columns {jnp.dtype(dtype).name}", seeds=COLUMN_SEEDS,
                    truth_fn=truth)


def test_exact_columns_compute_in_fp32_whatever_the_tables_compute():
    """A bf16 table's exact columns are the columns of the same table held in fp32 and computed in it
    at fp32 precision, bit for bit: the product is not rounded to bf16 as its logits' is."""
    rng = np.random.default_rng(0)
    head = table(rng, jnp.bfloat16, vocab_major=True)
    hidden = jnp.asarray(rng.standard_normal((4, FEATURES)), jnp.bfloat16)
    wide = OutputTable(jnp.asarray(head.matrix, jnp.float32), vocab_major=True, bias=head.bias,
                       softcap=head.softcap, precision=jax.lax.Precision.HIGHEST, compute=jnp.float32,
                       scale=head.scale)
    np.testing.assert_array_equal(np.asarray(head.columns(hidden, IDS, exact=True)),
                                  np.asarray(wide.columns(jnp.asarray(hidden, jnp.float32), IDS)))
    assert not np.array_equal(np.asarray(head.columns(hidden, IDS)), np.asarray(wide.columns(
        jnp.asarray(hidden, jnp.float32), IDS)))


def test_columns_outside_the_vocabulary_are_refused():
    head = table(np.random.default_rng(0), jnp.float32, vocab_major=True)
    with pytest.raises(IndexError):
        head.columns(jnp.zeros((2, FEATURES)), [0, VOCAB])
