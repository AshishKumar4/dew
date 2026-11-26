"""A sub-chunk's own pairs under a decay a key dimension as anchored products (`recurrence_algebra`'s
`own_anchors`, `own_anchored`, `own_compact`, `own_columns`; the kernels' `anchored.step`, `anchored.back`):
where the guard holds, the pairs each its own decay (check 1); the guard failing at each edge of its range,
where the kernels take the pairs each its own decay instead; the backward's compact blocks' layout; and a
query's gradient through the template, interpreted, of two score cotangents that cancel to their 24th bit,
bit for bit. (Kimi's template under its declared bound on par with
the portable math, either side of the guard, waits on row 13271: before any anchored product its final
memory is far off that math's at the bound.)"""


import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import COMPILES_ON_CPU
from reference_error import FEWEST_SEEDS, assert_as_exact, with_gradients
from test_recurrence_chunk import chunk_on, token_by_token
from test_recurrence_stability import SITES, cotangents
from test_recurrence_template_stability import bf16

from dew.nn.kernels import generation, recurrence_algebra as algebra

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


EXACT_PAIRS = dataclasses.replace(UNBOUND_QUERIES, decay_bound=None)
"""The same hooks with no bound declared: the template's own pairs each their own decay
(`anchored.anchoring`), the reference the anchored backward is held to."""


def _cancelling(seed):
    """A draw whose third token's score cotangents cancel to their 24th bit (`(operands, cotangents)`): zero
    queries, decays and memory, keys `e0` and `-e0` written whole, values `(a, 2^-9 b, 2^-20 m)` and
    `(0, 0, -2^-20 m)` (`a`, `b`, `m` drawn in [1, 2) on bf16's grid), so the corrected values are
    `(a, 2^-9 b, 2^-20 m)` and `(a, 2^-9 b, 0)`, and a unit cotangent on the third token's three value
    columns: its score cotangents `a + 2^-9 b + 2^-20 m` and `a + 2^-9 b`, its query's first channel
    `scale 2^-20 m` a value head."""
    operands, (out, final) = cotangents(UNBOUND_QUERIES, 'the decay bound', 0)
    a, b, m = (1.0 + np.random.default_rng([13497, seed]).integers(0, 128, 3) / 128.0)
    writes = jnp.zeros_like(operands.beta).at[:, :2].set(1.0)
    key = jnp.zeros_like(operands.key).at[:, 0, :, 0].set(1.0).at[:, 1, :, 0].set(-1.0)
    first = jnp.asarray([a, 2.0 ** -9 * b, 2.0 ** -20 * m])
    value = jnp.zeros_like(operands.value).at[:, 0, :, :3].set(first).at[:, 1, :, 2].set(-2.0 ** -20 * m)
    zeroed = {name: jnp.zeros_like(getattr(operands, name)) for name in ('query', 'log_decay', 'state')
              if getattr(operands, name) is not None}
    operands = bf16(dataclasses.replace(operands, key=key, value=value, beta=writes, **zeroed),
                    UNBOUND_QUERIES)
    cotangent = jnp.zeros_like(out).at[:, 2, :, :3].set(1.0).astype(jnp.bfloat16)
    return operands, (cotangent, jnp.zeros_like(final))


def _through(hooks):
    """The template's output, final memory and gradients for `hooks` (`chunk_on('mosaic_gpu')`)."""
    return with_gradients(lambda given: chunk_on('mosaic_gpu', hooks, given))


def _token_by_token(operands, cotangent):
    return with_gradients(lambda given: token_by_token(UNBOUND_QUERIES, given))(operands, cotangent)


@COMPILES_ON_CPU
@pytest.mark.check1_shards(seeds=8, per_task=8)
def test_a_query_gradient_of_cancelling_score_cotangents_is_as_exact_as_the_exact_pairs():
    """The cancelling draws' gradients through the template's anchored backward (`anchored.back`),
    interpreted, on par from float64 with its pairs each their own decay (check 1, `EXACT_PAIRS`): staged in
    two bf16 parts, both score cotangents took `(a, 2^-9 b)` and the query's gradient was zero (a head
    review's P0 of 28c4e9f0f); in `anchored.BLOCK_PARTS`, kept (and check 1 rejects that zero,
    `test_check_1_rejects_a_zeroed_query_gradient_of_the_cancelling_draws`)."""
    assert_as_exact(_through(UNBOUND_QUERIES), _through(EXACT_PAIRS), _cancelling,
                    "the cancelling draws' anchored gradients", seeds=8, truth_fn=_token_by_token)


def test_check_1_rejects_a_zeroed_query_gradient_of_the_cancelling_draws():
    """Check 1 over the cancelling draws rejects their query's gradient zeroed, the anchored backward's loss
    in two parts, against the portable math's (`chunk_on('xla')`, every gradient its own and the query's
    zeroed): the draws resolve the `scale 2^-20 m` element the loss would drop, so the template's check
    above holds it."""
    portable = with_gradients(lambda given: chunk_on('xla', EXACT_PAIRS, given))

    def zeroed(operands, cotangent):
        found, grads = portable(operands, cotangent)
        return found, dataclasses.replace(grads, query=jnp.zeros_like(grads.query))

    with pytest.raises(AssertionError):
        assert_as_exact(zeroed, portable, _cancelling, "a zeroed query gradient", seeds=8,
                        truth_fn=_token_by_token)


@COMPILES_ON_CPU
def test_the_cancelling_draws_reach_the_anchored_backward():
    """The cancelling draws take the anchored backward: their call's plan on the interpreted target
    anchors `key_pairs_own` (`mainloop.Launch.reducing`), and its guard holds on them: zero decays, so every
    factor is one and no prefix leaves `OWN_REACH`; every operand element zero or within `2^+-OWN_BAND`
    (`own_rows_failed` clear); and the pairs' gradients at most `a + 2^-9 b + 2^-20 m` below 4, the
    scaled operands at most 2, their product far within `2^OWN_PRODUCT`."""
    from dew.nn.kernels.recurrence_chunk import template_bind
    from dew.nn.kernels.templates.recurrence import template

    operands, _ = _cancelling(0)
    facts = generation.interpreted_target()
    with generation.assumed(facts, interpreted=True):
        call, _ = template_bind(operands.query, operands.key, operands.value, operands.log_decay,
                                operands.beta, operands.state, rounded=True, decay_bound=BOUND)
        _, bound = call.bound['mosaic_gpu']
        tiles = template.MOSAIC.derived(bound, facts)
        planned = template.MOSAIC.launches(bound, tiles, facts, backward=True)
    assert {launch.name: launch.reducing for launch in planned}["recurrence_key_pairs_own"] == 1
    rows = [jnp.asarray(x, jnp.float32).reshape(-1, x.shape[-1]) for x in (operands.query, operands.key)]
    assert all(np.all(np.asarray(algebra.own_rows_failed(None, [(x, x)])) == 0.0) for x in rows)
@pytest.mark.parametrize("transposed", [False, True])
def test_a_sub_chunks_compact_columns_are_its_square_strictly_before_or_after(transposed):
    """`own_columns` of a call's pairs `[R, H * C]`: row `i`'s column `c` the pair `(i, f + c)` strictly
    before `i` (`f` its sub-chunk's first token), and `transposed` row `j`'s column `r` the pair `(f + r, j)`
    strictly after `j`, zero elsewhere: each sub-chunk's own square, the anchored backward's operands."""
    rows, heads = 2 * CHUNK, 3
    pairs = np.random.default_rng(3).standard_normal((rows, heads * CHUNK)).astype(np.float32)
    found = np.asarray(algebra.own_columns(jnp.asarray(pairs), chunk=CHUNK, transposed=transposed))
    expected = np.zeros((rows, heads * SUB), np.float32)
    for token in range(rows):
        start, place = token - token % CHUNK, token % CHUNK
        first = place - place % SUB
        for h in range(heads):
            for c in range(SUB):
                other = first + c
                if transposed and other > place:
                    expected[token, h * SUB + c] = pairs[start + other, h * CHUNK + place]
                elif not transposed and other < place:
                    expected[token, h * SUB + c] = pairs[token, h * CHUNK + other]
    assert np.array_equal(found, expected)


@pytest.mark.parametrize("target", ["sm86", "sm120"])
def test_kimis_training_plan_anchors_its_own_pairs(target):
    """Kimi delta attention's bf16 training step as the template plans it on `target`: its forward's own pairs
    anchored (`intra_own` reducing its guard across warps, `mainloop.Launch.reducing`), and its backward's
    (`key_pairs_own`), an anchored block of it holding a program: an RTX A6000's at 32 keys, an RTX PRO
    6000's at 16."""
    from test_recurrence_mosaic import RESOURCE_SHAPES, resource_operands

    from dew.nn.kda import KimiDeltaAttention
    from dew.nn.kernels import generation
    from dew.nn.kernels.recurrence_chunk import template_bind
    from dew.nn.kernels.templates.recurrence import template

    assert RESOURCE_SHAPES['kimi-linear-kda'][-1], "Kimi's decay a key dimension's"
    operands = [jnp.zeros(shape.shape, shape.dtype) for shape in
                resource_operands('kimi-linear-kda', jnp.bfloat16, 1, 512)]
    facts = generation.gpu_target(target)
    with generation.assumed(facts):
        call, _ = template_bind(*operands, rounded=True, decay_bound=KimiDeltaAttention.lower_bound)
        _, bound = call.bound['mosaic_gpu']
        tiles = template.MOSAIC.derived(bound, facts)
        launches = {launch.name: launch for launch in template.MOSAIC.launches(bound, tiles, facts,
                                                                             backward=True)}
    reducing = {name: launches[name].reducing
                for name in ("recurrence_intra_own", "recurrence_key_pairs_own")}
    assert reducing == {"recurrence_intra_own": 1, "recurrence_key_pairs_own": 1}, reducing


@pytest.mark.parametrize("target", ["sm86", "sm120"])
def test_kimis_own_pairs_launches_fit_each_budget_they_are_planned_at(target):
    """Kimi's bf16 training step planned at its device's shared budget and at budgets a block of 4 KiB and two
    below it, down to the last the template derives at: each own pairs launch (`intra_own`,
    `intra_own_scores`, `key_pairs_own`) fits the budget it is planned at with the scratch it launches with
    (`mainloop.Launch.shared`, its `reducing` warpgroups' counted), as its fit priced it
    (`own_pairs.anchored_plan`: fitted without the forward's scratch, its launch took more than it priced,
    train 1337's review, row 13495). No budget the template derives at holds no anchored forward block (its
    other launches need twice the forward's), so each stays anchored."""
    from test_recurrence_mosaic import resource_operands

    from dew.nn.kda import KimiDeltaAttention
    from dew.nn.kernels.recurrence_chunk import template_bind
    from dew.nn.kernels.templates.recurrence import template

    operands = [jnp.zeros(shape.shape, shape.dtype) for shape in
                resource_operands('kimi-linear-kda', jnp.bfloat16, 1, 512)]
    base = generation.gpu_target(target)
    for budget in (generation.shared_budget(base) - 4096 * step for step in range(3)):
        facts = dataclasses.replace(base, shared_limit=budget)
        with generation.assumed(facts):
            call, _ = template_bind(*operands, rounded=True, decay_bound=KimiDeltaAttention.lower_bound)
            _, bound = call.bound['mosaic_gpu']
            tiles = template.MOSAIC.derived(bound, facts)
            if tiles is None:
                break
            own = [launch for launch in template.MOSAIC.launches(bound, tiles, facts, backward=True)
                   if "_own" in launch.name]
        assert all(launch.shared <= budget and launch.reducing == 1 for launch in own), (
            budget, [(launch.name, launch.shared, launch.reducing) for launch in own])
