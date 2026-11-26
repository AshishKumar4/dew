"""A sub-chunk's own pairs under a decay a key dimension as anchored products (`recurrence_algebra`'s
`own_anchors`, `own_anchored`, `own_compact`; the forward kernel's `anchored.step`): where the guard holds,
the pairs each its own decay (check 1); the guard failing at each edge of its range, where the kernel takes
the pairs each its own decay instead; and a query's gradient through the template, interpreted, of two score
cotangents that cancel to their 24th bit, bit for bit. (Kimi's template under its declared bound on par with
the portable math, either side of the guard, waits on row 13271: before any anchored product its final
memory is far off that math's at the bound.)"""


import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import COMPILES_ON_CPU
from reference_error import FEWEST_SEEDS, assert_as_exact, widened, with_gradients
from test_recurrence_chunk import chunk_on, token_by_token
from test_recurrence_stability import SITES, cotangents
from test_recurrence_template_stability import bf16

from dew.nn.kernels import recurrence_algebra as algebra

CHUNK, KEYS, SUB = 64, 32, algebra.SUB
BOUND = -5.0
"""Kimi delta attention's decay bound, a token's least natural log decay (its `lower_bound`)."""


def _drawn(seed: int, *, steep: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A key block's rows and keys `[C, keys]` and its decays' base-2 prefix within the chunk, each token's
    log decay within the bound, one token's at -100 where `steep` (past `OWN_REACH` of its anchor)."""
    rng = np.random.default_rng(seed)
    rows, keys = (rng.standard_normal((CHUNK, KEYS)).astype(np.float32) for _ in range(2))
    decays = rng.uniform(BOUND, 0.0, (CHUNK, KEYS))
    if steep:
        decays[SUB + SUB // 2 + 1] = -100.0
    return rows, keys, np.cumsum(decays * algebra.LOG2_E, axis=0).astype(np.float32)


def _anchored(rows, keys, prefix) -> tuple:
    """The own pairs compact `[C, SUB]` as one anchored product at each sub-chunk's middle token
    (`own_anchored`, `own_compact`), its separated prefix and its rows' anchors."""
    whole = algebra.separated(jnp.asarray(prefix))
    anchors = algebra.own_anchors(lambda token: tuple(half[token] for half in whole), (CHUNK, KEYS))
    down, up = algebra.own_anchored(whole, anchors)
    product = jnp.dot(jnp.asarray(rows) * down, (jnp.asarray(keys) * up).T, precision="highest")
    return algebra.own_compact(product), whole, anchors, (down, up)


def _each_its_own(rows, keys, prefix):
    """The own pairs compact `[C, SUB]` each its own decay as the exact path forms them
    (`algebra.key_diagonal_bands`: each pair's decay the product of its tokens' own, `token_decays`), the
    keys and decays read `e` rows back within the sub-chunk at offset `e`."""
    whole = algebra.separated(prefix)
    decays = algebra.token_decays(whole, tuple(jnp.roll(half, 1, axis=0) for half in whole))

    def shifted(e):
        return jnp.roll(keys, e, axis=0), jnp.roll(decays, e, axis=0)

    return algebra.key_diagonal_bands([rows], shifted)[0]


def _truth(rows, keys, prefix):
    """The own pairs compact `[C, SUB]`, `sum_k r_ik k_jk exp2(g_ik - g_jk)` for `j` at or before `i` in its
    sub-chunk, zero past it: check 1's float64 truth."""
    tokens = jnp.arange(CHUNK)
    earlier = tokens[:, None] // SUB * SUB + jnp.arange(SUB)[None, :]
    terms = rows[:, None, :] * keys[earlier] * jnp.exp2(prefix[:, None, :] - prefix[earlier])
    return jnp.where(earlier <= tokens[:, None], terms.sum(-1), 0.0)


def test_the_anchored_own_pairs_are_as_exact_as_each_pair_its_own_decay():
    """Within the bound, the anchored product's compact blocks (`own_anchored`, `own_compact`) on par from
    float64 with the pairs each its own decay as the exact path forms them (check 1): each row's pairs at or
    before it in its sub-chunk, zero past it."""
    def draw(seed):
        return tuple(jnp.asarray(x) for x in _drawn(seed))

    assert_as_exact(lambda *x: _anchored(*x)[0], _each_its_own, draw, "a sub-chunk's own pairs, anchored",
                    seeds=FEWEST_SEEDS, truth_fn=_truth)


@pytest.mark.parametrize(("edge", "fails"), [
    ("within its range", False), ("a token's prefix past its anchor's reach", True),
    ("a faint channel scaled below the band", True), ("a channel scaled past the band", True),
    ("a zero channel", False), ("a channel not finite", True)])
def test_the_guard_fails_past_each_edge_of_its_range(edge, fails):
    """The anchored products' guard (`own_reach_exceeded`, `own_out_of_band`, `own_rows_failed`) fails, so the
    kernels take the pairs each its own decay, where a token's prefix lies past `OWN_REACH` of its anchor,
    where any element's scaled value lies past `2^OWN_BAND` or, given nonzero, below `2^-OWN_BAND` (a key
    channel of 2^-80 beside a unit one: its row's largest in band, its own scaled to 2^-136, below bf16's
    least, the head review's P0 of d93018eb2), or is not finite; a zero channel and draws within the bound
    hold it."""
    rows, keys, prefix = _drawn(1, steep=edge == "a token's prefix past its anchor's reach")
    _, whole, anchors, (down, up) = _anchored(rows, keys, prefix)
    # Row 16, the first token of the second sub-chunk, its key factor the least a token takes there.
    least = int(np.argmin(np.asarray(up)[SUB]))
    if edge == "a faint channel scaled below the band":
        keys[SUB, least] = np.float32(2.0 ** -(algebra.OWN_BAND + 1) / float(up[SUB, least]))
    elif edge == "a channel scaled past the band":
        rows[SUB, 0] = np.float32(2.0 ** (algebra.OWN_BAND + 1) / float(down[SUB, 0]))
    elif edge == "a zero channel":
        keys[SUB, least] = 0.0
    elif edge == "a channel not finite":
        rows[5, 3] = np.nan
    given = [(jnp.asarray(value), jnp.asarray(value) * factor)
             for value, factor in ((rows, down), (keys, up))]
    failed = algebra.own_rows_failed(algebra.own_reach_exceeded(whole, anchors), given)
    assert float(jnp.max(failed)) == float(fails)


UNBOUND_QUERIES = dataclasses.replace(SITES['vector delta'], decay_bound=BOUND, unit_queries=False)
"""Kimi delta attention's hooks under its declared bound, its queries free to be zero: the template takes
its own pairs as anchored products (`anchored.anchoring`)."""


@COMPILES_ON_CPU
def test_a_query_gradient_of_cancelling_score_cotangents_keeps_their_last_bits():
    """A query's gradient of two score cotangents that cancel to their 24th bit, through the template and its
    own pairs' backward (`own_pairs.key_pairs_own`), interpreted, the token-by-token recurrence's in float64
    bit for bit: zero queries and decays, keys `e0` and `-e0` written whole, values `(1, 2^-9, 2^-20)` and
    `(0, 0, -2^-20)`, so the corrected values are `(1, 2^-9, 2^-20)` and `(1, 2^-9, 0)`, and a unit cotangent
    on the third token's three value columns: its score cotangents `1 + 2^-9 + 2^-20` and `1 + 2^-9`, its
    query's first channel `scale (2^-20)` a value head: an anchored backward staging the cotangents in two
    bf16 parts took both as `(1, 2^-9)`, the gradient zero (a head review's P0 of 28c4e9f0f)."""
    operands, (out, final) = cotangents(UNBOUND_QUERIES, 'the decay bound', 0)
    writes = jnp.zeros_like(operands.beta).at[:, :2].set(1.0)
    key = jnp.zeros_like(operands.key).at[:, 0, :, 0].set(1.0).at[:, 1, :, 0].set(-1.0)
    value = (jnp.zeros_like(operands.value).at[:, 0, :, :3].set(jnp.asarray([1.0, 2.0 ** -9, 2.0 ** -20]))
             .at[:, 1, :, 2].set(-2.0 ** -20))
    zeroed = {name: jnp.zeros_like(getattr(operands, name)) for name in ('query', 'log_decay', 'state')
              if getattr(operands, name) is not None}
    operands = bf16(dataclasses.replace(operands, key=key, value=value, beta=writes, **zeroed),
                    UNBOUND_QUERIES)
    cotangent = (jnp.zeros_like(out).at[:, 2, :, :3].set(1.0).astype(jnp.bfloat16), jnp.zeros_like(final))
    found = with_gradients(lambda given: chunk_on('mosaic_gpu', UNBOUND_QUERIES, given))(operands, cotangent)
    with jax.enable_x64(new_val=True):
        truth = with_gradients(lambda given: token_by_token(UNBOUND_QUERIES, given))(
            *widened((operands, cotangent)))
    ours, theirs = (np.asarray(grads[1].query[:, 2, :, 0], np.float64) for grads in (found, truth))
    assert np.all(theirs != 0) and np.array_equal(ours, theirs), (ours, theirs)
