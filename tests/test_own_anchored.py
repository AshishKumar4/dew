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
its forward's own pairs as anchored products (`anchored.anchoring`), its backward's each its own decay."""


EXACT_PAIRS = dataclasses.replace(UNBOUND_QUERIES, decay_bound=None)
"""The same hooks with no bound declared: the template's own pairs each their own decay, forward and back
(`anchored.anchoring`), the reference the bounded call's gradients are held to."""


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
    """The cancelling draws' gradients through the template under Kimi's bound (its forward's own pairs
    anchored), interpreted, on par from float64 with its pairs each their own decay (check 1,
    `EXACT_PAIRS`): a backward staging its score cotangents in two bf16 parts took both as `(a, 2^-9 b)` and
    the query's gradient zero (a head review's P0 of 28c4e9f0f), which check 1 rejects
    (`test_check_1_rejects_a_zeroed_query_gradient_of_the_cancelling_draws`)."""
    assert_as_exact(_through(UNBOUND_QUERIES), _through(EXACT_PAIRS), _cancelling,
                    "the cancelling draws' gradients under the bound", seeds=8, truth_fn=_token_by_token)


def test_check_1_rejects_a_zeroed_query_gradient_of_the_cancelling_draws():
    """Check 1 over the cancelling draws rejects their query's gradient zeroed, a two-part backward's loss,
    against the portable math's (`chunk_on('xla')`, every gradient its own and the query's zeroed): the
    draws resolve the `scale 2^-20 m` element the loss would drop, so the template's check above holds
    it."""
    portable = with_gradients(lambda given: chunk_on('xla', EXACT_PAIRS, given))

    def zeroed(operands, cotangent):
        found, grads = portable(operands, cotangent)
        return found, dataclasses.replace(grads, query=jnp.zeros_like(grads.query))

    with pytest.raises(AssertionError):
        assert_as_exact(zeroed, portable, _cancelling, "a zeroed query gradient", seeds=8,
                        truth_fn=_token_by_token)


@pytest.mark.parametrize(("target", "anchors"),
                         [("sm80", 0), ("sm86", 0), ("sm90", 1), ("sm100", 1), ("sm120", 0)])
def test_kimis_training_plan_anchors_its_forward_own_pairs_alone_where_products_read_shared_memory(
        target, anchors):
    """Kimi delta attention's bf16 training step as the template plans it on `target`: its forward's own pairs
    anchored (`intra_own` reducing its guard across warps, `mainloop.Launch.reducing`) where its family's
    products read shared memory (`anchored.anchors_on`: an A100's anchored intra_own 10.9% slower, a
    GH200's 12.1% faster, M-JS-291c and M-JS-291b), its backward's (`key_pairs_own`) each its own decay:
    anchored, a GH200's took 899.7 us where these took 501.8, its training step 2912.7 against 2445.1
    (M-JS-293), its block holding half the programs a multiprocessor these do on an H100, a quarter on an
    A100."""
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
    assert reducing == {"recurrence_intra_own": anchors, "recurrence_key_pairs_own": 0}, reducing


OWN_LAUNCHES = {"recurrence_intra_own", "recurrence_intra_own_scores", "recurrence_key_pairs_own"}
"""Kimi's training step's own pairs launches: the forward's (scores and system, then scores) and the
backward's."""


@pytest.mark.parametrize("target", ["sm86", "sm90", "sm120"])
def test_kimis_own_pairs_launches_fit_each_budget_they_are_planned_at(target):
    """Kimi's bf16 training step planned at its device's shared budget and at a budget 4 KiB below it (an
    RTX PRO 6000's 8 KiB below derives none): the template derives at each, and each own pairs launch
    (`OWN_LAUNCHES`, every one of them planned) fits the budget it is planned at with the scratch it
    launches with (`mainloop.Launch.shared`, its `reducing` warpgroups' counted), as its fit priced it
    (`own_pairs.anchored_plan`: fitted without the forward's scratch, its launch took more than it priced,
    train 1337's review, row 13495)."""
    from test_recurrence_mosaic import resource_operands

    from dew.nn.kda import KimiDeltaAttention
    from dew.nn.kernels.recurrence_chunk import template_bind
    from dew.nn.kernels.templates.recurrence import template

    operands = [jnp.zeros(shape.shape, shape.dtype) for shape in
                resource_operands('kimi-linear-kda', jnp.bfloat16, 1, 512)]
    base = generation.gpu_target(target)
    for budget in (generation.shared_budget(base) - 4096 * step for step in range(2)):
        facts = dataclasses.replace(base, shared_limit=budget)
        with generation.assumed(facts):
            call, _ = template_bind(*operands, rounded=True, decay_bound=KimiDeltaAttention.lower_bound)
            _, bound = call.bound['mosaic_gpu']
            tiles = template.MOSAIC.derived(bound, facts)
            assert tiles is not None, budget
            own = {launch.name: launch for launch in template.MOSAIC.launches(bound, tiles, facts,
                                                                             backward=True)
                   if launch.name in OWN_LAUNCHES}
        assert set(own) == OWN_LAUNCHES, (budget, sorted(own))
        assert all(launch.shared <= budget for launch in own.values()), (
            budget, {name: (launch.shared, launch.reducing) for name, launch in own.items()})


def test_kimis_forward_own_pairs_fall_back_where_no_anchored_block_fits():
    """Kimi's forward own pairs (`own_pairs.own_plan`, the scores' and the system's) planned directly on an
    H100 whose shared budget is the least its anchored products take, and a byte below it: there
    anchored, here each pair its own decay at a key block that holds a program (`mainloop.key_block_holds`
    of the exact plan), the anchored scratch and staging no longer fitting. (The template's other launches
    need about twice the forward's own pairs', so no budget the whole template derives at reaches this
    boundary.)"""
    from test_recurrence_chunk import chunk_on
    from test_recurrence_mosaic import resource_operands

    from dew.nn.kernels.recurrence_chunk import Chunk
    from dew.nn.kernels.templates.recurrence import chunk_forward, mainloop, own_pairs

    q, k, v, g, beta, h0 = (jnp.zeros(shape.shape, shape.dtype) for shape in
                            resource_operands('kimi-linear-kda', jnp.bfloat16, 1, 512))
    base = generation.gpu_target("sm90")
    with generation.assumed(base), chunk_forward.observed() as calls:
        jax.eval_shape(lambda *x: chunk_on('mosaic_gpu', UNBOUND_QUERIES, Chunk(
            x[0], x[1], x[2], log_decay=x[3], beta=x[4], state=x[5])), q, k, v, g, beta, h0)
    products = ("scores", "system")

    def planned(budget):
        call = dataclasses.replace(calls[0], target=dataclasses.replace(base, shared_limit=budget))
        return call, own_pairs.own_plan(call, products)

    low, high = 0, generation.shared_budget(base)
    assert planned(high)[1][0], "anchored at the device's budget"
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (low, middle) if planned(middle)[1][0] else (middle, high)
    call, (anchors, keys) = planned(low)
    plan = own_pairs.own_plans(call, products)
    assert not anchors and mainloop.key_block_holds(call, *plan(anchors=False), contracted=True), (low, keys)
