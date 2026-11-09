"""Oja's rule with a gate a value dimension (`recurrence_chunk.chunk_gated_oja_rule`, fla's gated_oja_rule)
on the derived chunk forms of its transposed memory, read along its values: its output, final memory and
gradients on par with the rule a token at a time from float64 (check 1)."""

import functools

import jax
import numpy as np
import pytest
from test_recurrence_chunk import assert_packed_on_par, hosted, sigmoid, unit

from dew.nn.kernels.recurrence_chunk import chunk_gated_oja_rule, recurrent_gated_oja_rule

ROWS, LENGTH, KEY_HEADS, VALUE_HEADS, KEY_WIDTH, VALUE_WIDTH = 2, 70, 1, 2, 16, 16
CHUNK = 16
DOCUMENTS = ((0, 23, 40), (9, 48))
"""Each row's packed documents' first tokens: the first row's memory zeroed at token 0, the second's entering
its first document, 48 a chunk's first token."""
SEEDS = {False: 16, True: 32}
"""Each site's declared seeds (one sequence, packed), sized by `seeds_for` on armada."""

CHUNKED = functools.partial(chunk_gated_oja_rule, chunk_size=CHUNK)
"""The chunked rule at the test's chunk."""


def draw(seed: int):
    """The rule's fp32 operands from `seed` (`hosted`), as fla's test draws them, and a cotangent of its
    output and final memory: fresh queries and keys, values of unit length, a head's write gate a sigmoid's, a
    log decay a value dimension a log sigmoid's, a memory entering."""
    normal = np.random.default_rng([seed, 1]).standard_normal
    query, key = (normal((ROWS, LENGTH, KEY_HEADS, KEY_WIDTH)) for _ in range(2))
    value = unit(normal((ROWS, LENGTH, VALUE_HEADS, VALUE_WIDTH)))
    gv = -np.logaddexp(0.0, -normal((ROWS, LENGTH, VALUE_HEADS, VALUE_WIDTH)))
    beta = sigmoid(normal((ROWS, LENGTH, VALUE_HEADS)))
    state = 0.3 * normal((ROWS, VALUE_HEADS, KEY_WIDTH, VALUE_WIDTH))
    cotangent = (normal((ROWS, LENGTH, VALUE_HEADS, VALUE_WIDTH)),
                 normal((ROWS, VALUE_HEADS, KEY_WIDTH, VALUE_WIDTH)))
    return *(hosted(t) for t in (query, key, value, gv, beta, state)), jax.tree.map(hosted, cotangent)


@pytest.mark.parametrize('packed', (False, True))
def test_the_chunked_rule_is_as_exact_as_the_rule_a_token_at_a_time(packed):
    """The chunked rule's output, final memory and every operand's gradient on par with the rule a token at a
    time from float64, one sequence and packed documents (check 1): the delta rule's derived forms on the
    transposed memory, its readout along the memory's values."""
    assert_packed_on_par(CHUNKED, recurrent_gated_oja_rule, draw, 'gated oja', packed=packed,
                         documents=DOCUMENTS, length=LENGTH, seeds=SEEDS[packed])
