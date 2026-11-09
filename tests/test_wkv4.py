"""RWKV4's WKV (`dew.nn.kernels.wkv4`, fla's fused_recurrent_rwkv4) as one associative scan of its spans'
carries: its output, final carry and gradients on par with fla's recurrence a token at a time from float64,
one sequence and packed documents (check 1)."""

import functools

import jax
import numpy as np
import pytest
from reference_error import assert_as_exact, with_gradients
from test_recurrence_chunk import hosted, marked

from dew.nn.kernels.wkv4 import recurrent_wkv4, wkv4

ROWS, LENGTH, CHANNELS = 2, 70, 16
DOCUMENTS = ((0, 23, 40), (9, 48))
"""Each row's packed documents' first tokens: the first row's carry dropped at token 0, the second's
entering its first document."""
SEEDS = {False: 256, True: 512}
"""Each site's declared seeds (one sequence, packed), sized by `seeds_for` on armada."""


def draw(seed: int):
    """The WKV's fp32 operands from `seed` (`hosted`) and a cotangent of its output and final carry: a
    channel's decay and bonus, keys spread over a few units (the running maximum's range), values, and a
    carry entering whose denominator is positive and numerator its values' mean's scale."""
    normal = np.random.default_rng([seed, 1]).standard_normal
    time_decay, time_first = normal((CHANNELS,)), normal((CHANNELS,))
    key, value = 2.0 * normal((ROWS, LENGTH, CHANNELS)), normal((ROWS, LENGTH, CHANNELS))
    beta = np.exp(normal((ROWS, CHANNELS)))
    state = np.stack([beta * normal((ROWS, CHANNELS)), beta, normal((ROWS, CHANNELS))], axis=1)
    cotangent = (normal((ROWS, LENGTH, CHANNELS)), normal((ROWS, 3, CHANNELS)))
    return *(hosted(t) for t in (time_decay, time_first, key, value, state)), jax.tree.map(hosted, cotangent)


@pytest.mark.parametrize('packed', (False, True))
def test_the_scan_is_as_exact_as_the_recurrence_a_token_at_a_time(packed):
    """The scan's output, final carry and every operand's gradient on par with fla's recurrence a token at a
    time from float64, one sequence and packed documents (check 1): each span's carry rescaled to the larger
    running maximum as it joins, a document's first token reading an empty past."""
    starts = marked(DOCUMENTS, LENGTH) if packed else None
    dew = jax.jit(with_gradients(functools.partial(wkv4, starts=starts)))
    reference = jax.jit(with_gradients(functools.partial(recurrent_wkv4, starts=starts)))
    # Packed, the first row's carry entering is dropped at its token 0, so its gradient is exactly zero in
    # both arms: held to the reference's residue there, zero, no float64 terms traced (the token loop's terms
    # took 11.5 GB and were killed, jsf-wkv4t1); one sequence holds no zero row, so none are traced either.
    assert_as_exact(dew, reference, draw, f'wkv4 packed={packed}', seeds=SEEDS[packed],
                    terms=False if packed else None)
