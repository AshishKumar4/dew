"""DeltaFormer's attention (`dew.nn.kernels.deltaformer`, fla's deltaformer_attn) as one unit lower-triangular
solve and a softmax attention: its output and gradients on par with fla's reference a token at a time from
float64, one sequence and packed documents (check 1)."""


import numpy as np
import pytest
from test_recurrence_chunk import assert_packed_on_par, hosted, sigmoid

from dew.nn.kernels.deltaformer import deltaformer, recurrent_deltaformer

ROWS, LENGTH, HEADS, WIDTH = 2, 32, 2, 16
"""32 tokens: the float64 terms check 1 traces for its cancelling rows took 120 s at 70 (jsf-dfmprof1)."""
DOCUMENTS = ((0, 13, 22), (0, 9, 20))
"""Each row's packed documents' first tokens."""
SEEDS = {False: 32, True: 8}
"""Each site's declared seeds (one sequence, packed), sized by `seeds_for` on armada."""


def draw(seed: int):
    """DeltaFormer's fp32 operands from `seed` (`hosted`), as fla's test draws them, and a cotangent of its
    output: queries, keys and values, a write strength a head a sigmoid's."""
    normal = np.random.default_rng([seed, 1]).standard_normal
    q, k, v = (normal((ROWS, LENGTH, HEADS, WIDTH)) for _ in range(3))
    beta = sigmoid(normal((ROWS, LENGTH, HEADS)))
    return *(hosted(t) for t in (q, k, v, beta)), hosted(normal((ROWS, LENGTH, HEADS, WIDTH)))


@pytest.mark.parametrize('packed', (False, True))
def test_the_solve_is_as_exact_as_the_recurrence_a_token_at_a_time(packed):
    """The solve's output and every operand's gradient on par with fla's reference a token at a time from
    float64, one sequence and packed documents (check 1): each token's value corrected by its earlier tokens'
    corrected values, each softmax and the solve within its document."""
    assert_packed_on_par(deltaformer, recurrent_deltaformer, draw, 'deltaformer', packed=packed,
                         documents=DOCUMENTS, length=LENGTH, seeds=SEEDS[packed])
