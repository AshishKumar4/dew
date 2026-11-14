"""Log-linear attention (`dew.nn.kernels.log_linear`, fla's log_linear_attn): its output and gradients on par
with fla's reference, the Fenwick tree's level masks built level by level (fla/ops/log_linear_attn/naive.py),
from float64, one sequence and packed documents (check 1)."""

import functools
import math

import jax.numpy as jnp
import numpy as np
import pytest
from test_recurrence_chunk import assert_packed_on_par, hosted, sigmoid

from dew.nn.kernels.log_linear import levels, log_linear_attention

ROWS, LENGTH, GROUPS, HEADS, WIDTH, VALUES = 2, 40, 1, 2, 16, 8
"""Two rows of 40 tokens (six levels), one key group serving two value heads."""
DOCUMENTS = ((0, 13, 22), (0, 9, 33))
"""Each row's packed documents' first tokens."""
SEEDS = {False: 32, True: 32}
"""Each site's declared seeds (one sequence, packed), sized by `seeds_for` on armada."""


def level_mask(level: int, length: int) -> np.ndarray:
    """fla's `construct_level_mask`'s pattern `[T, T]`: the keys a query reaches at `level`."""
    rows, columns = np.indices((length, length))
    if level == 0:
        return rows == columns
    half = 1 << (level - 1)
    base = rows - rows % half
    return (rows % (1 << level) >= half) & (columns + half >= base) & (columns < base)


def naive(query, key, value, g, level_scales, *, starts=None):
    """fla's `naive_log_linear_attn` (scale `N ** -0.5`), each packed document a sequence of its own from
    its first token (`DOCUMENTS`' where `starts` is given)."""
    if starts is not None:
        pieces = []
        for row, begins in enumerate(DOCUMENTS):
            ends = (*begins[1:], LENGTH)
            pieces.append(jnp.concatenate([
                naive(*(x[row:row + 1, begin:end] for x in (query, key, value, g)),
                      level_scales[row:row + 1, begin:end, :, :levels(end - begin)])
                for begin, end in zip(begins, ends, strict=True)], axis=1))
        return jnp.concatenate(pieces)
    length = query.shape[1]
    work = jnp.promote_types(value.dtype, jnp.float32)
    q, k = (jnp.repeat(x.astype(work), HEADS // GROUPS, axis=2) for x in (query, key))
    running = jnp.cumsum(g.astype(work), axis=1).transpose(0, 2, 1)
    causal = np.tril(np.ones((length, length), bool))
    decay = jnp.exp(jnp.where(causal, running[..., :, None] - running[..., None, :], -jnp.inf))
    weights = level_scales.astype(work).transpose(0, 2, 1, 3)
    mixed = sum(decay * level_mask(level, length) * weights[..., level, None]
                for level in range(math.ceil(math.log2(length)) + 1))
    scores = jnp.einsum('bhlc,blhn,bchn->bhlc', mixed, q, k)
    return (jnp.einsum('bhlc,bchp->blhp', scores, value.astype(work)) * WIDTH ** -0.5).astype(value.dtype)


def draw(seed: int):
    """fp32 operands from `seed`: unit queries and keys, a log decay `log sigmoid`, level weights about
    one, and a cotangent of the output."""
    normal = np.random.default_rng([seed, 3]).standard_normal
    q, k = (normal((ROWS, LENGTH, GROUPS, WIDTH)) for _ in range(2))
    q, k = (x / np.linalg.norm(x, axis=-1, keepdims=True) for x in (q, k))
    v = normal((ROWS, LENGTH, HEADS, VALUES))
    g = np.log(sigmoid(normal((ROWS, LENGTH, HEADS)) + 2.0))
    weights = 1.0 + 0.3 * normal((ROWS, LENGTH, HEADS, levels(LENGTH)))
    return *(hosted(x) for x in (q, k, v, g, weights)), hosted(normal((ROWS, LENGTH, HEADS, VALUES)))


@pytest.mark.parametrize('packed', (False, True))
def test_the_masked_form_is_as_exact_as_flas_level_by_level(packed):
    """The output and every operand's gradient, the level weights' included, of the masked form (each key's
    level its position's highest bit apart from the query's) on par with fla's reference building each
    level's mask, from float64, one sequence and packed documents counting positions from their own first
    tokens (check 1)."""
    # Packed, each row's documents bound its levels' blocks (`max_segments`).
    dew = functools.partial(log_linear_attention, max_segments=max(map(len, DOCUMENTS))) if packed else \
        log_linear_attention
    assert_packed_on_par(dew, naive, draw, 'log-linear attention', packed=packed, documents=DOCUMENTS,
                         length=LENGTH, seeds=SEEDS[packed])
