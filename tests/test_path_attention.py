"""PaTH attention (`dew.nn.kernels.path_attention`, fla's parallel_path_attn) through its chunks' UT forms:
its output and gradients on par with its definition a lag at a time from float64, one sequence and packed
documents (check 1)."""

import functools

import numpy as np
import pytest
from test_recurrence_chunk import assert_packed_on_par, hosted, sigmoid

from dew.nn.kernels.path_attention import path_attention, recurrent_path_attention

ROWS, LENGTH, QUERY_HEADS, HEADS, WIDTH, CHUNK = 2, 40, 4, 2, 16, 16
"""Three chunks of 16 tokens, the last ragged, two query heads a key head."""
DOCUMENTS = ((0, 13, 22), (0, 9, 33))
"""Each row's packed documents' first tokens, on and off the chunks' grid."""
SEEDS = {False: 8, True: 16}
"""Each site's declared seeds (one sequence, packed), sized by `seeds_for` on armada."""

CHUNKED = functools.partial(path_attention, chunk=CHUNK)
"""The chunked form at the test's chunk of 16 tokens."""


def draw(seed: int):
    """PaTH's fp32 operands from `seed` (`hosted`), as fla's layer forms them, and a cotangent of its output:
    queries, keys and values, the transforms' directions of unit length and strengths `2 sigmoid` (a
    reflection at two), a forget gate's log decay `log sigmoid`."""
    normal = np.random.default_rng([seed, 1]).standard_normal
    q = normal((ROWS, LENGTH, QUERY_HEADS, WIDTH))
    k, v, w = (normal((ROWS, LENGTH, HEADS, WIDTH)) for _ in range(3))
    w = w / np.linalg.norm(w, axis=-1, keepdims=True)
    beta = 2.0 * sigmoid(normal((ROWS, LENGTH, HEADS)))
    gate = np.log(sigmoid(normal((ROWS, LENGTH, QUERY_HEADS)) + 2.0))
    cotangent = normal((ROWS, LENGTH, QUERY_HEADS, WIDTH))
    return *(hosted(t) for t in (q, k, v, w, beta, gate)), hosted(cotangent)


@pytest.mark.parametrize('packed', (False, True))
def test_the_chunks_are_as_exact_as_the_definition_a_lag_at_a_time(packed):
    """The chunked UT form's output and every operand's gradient on par with the scores formed by carrying
    each query back through one token's transform a step, from float64, one sequence and packed documents
    (check 1): each chunk's transforms one solve, a query chunk walking the key chunks before it, each softmax
    within its document."""
    assert_packed_on_par(CHUNKED, recurrent_path_attention, draw, 'path attention', packed=packed,
                         documents=DOCUMENTS, length=LENGTH, seeds=SEEDS[packed])
