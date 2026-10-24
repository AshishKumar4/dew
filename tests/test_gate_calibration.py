"""The float64 gate's calibration (tests/reference_error.py, tests/reference_trees.py): computations
as exact as their reference, by another equally exact tree, pass it, and faults injected into Dew
alone, the reference's ensemble frozen, fail it.

Each comparison draws its orders from one orbit, the same input with the units of one axis
permuted (the axis each test names), and holds every element of every output over ORDERS orders
a batch, its flagged elements confirmed over fresh batches. The ensemble is declared before Dew
runs: the reference itself and its math under each predeclared tree assignment, each admitted at a
leaf only where no less exact than the reference there.

The ensemble's false-failure rate where a reduction is composed whole is not calibrated: dense
attention over keys in a held-out fixed order, against the reference, four tree assignments and
sixteen fixed key orders, still flags 2 of 8192 output elements at 64 keys in chunks of 16 at every
stage. That null is measured beside the suite, not asserted in it, until the rate is calibrated at
each test's element count; the source-aware graph gate holds attention's key sums node by node
instead (tests/test_reference_graph.py)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as splash
from reference_error import ORDERS, REFERENCE, assert_leaves_as_exact_over_orders, widened
from reference_trees import assignments, tree_contract, tree_sum

from dew.nn.attention import scaled_dot_product_attention

EPSILON = 1e-6


def orbit(units: int, order: int) -> np.ndarray:
    """Order `order`'s permutation of `units` units: order 0 the identity."""
    return np.arange(units) if order == 0 else np.random.default_rng(order).permutation(units)


def rms_norm(summed, x, scale):
    """x over the root mean of its squares over the last axis, times `scale`, the squares summed by
    `summed`."""
    inverse = jax.lax.rsqrt(summed(jnp.square(x)) / x.shape[-1] + EPSILON)
    return x * inverse * scale


def xla_sum(squares):
    return jnp.sum(squares, axis=-1, keepdims=True)


def halves_sum(squares):
    """The same sum as its two halves added: another tree, as exact."""
    left, right = jnp.split(squares, [squares.shape[-1] // 2], axis=-1)
    return jnp.sum(left, axis=-1, keepdims=True) + jnp.sum(right, axis=-1, keepdims=True)


def tree_of(tree):
    return lambda squares: tree_sum(squares, -1, tree)[..., None]


NORM_TREES = ("pairwise", "blocked-64", "blocked-128")
"""The norm's members: the trees its one reduction class, the sum of squares over the width, declares;
blocks of 8 or 32 summed in turn are less exact than XLA's own sum at 256 terms, and admission would
leave them out."""


def norm_draw(dew_sum, shape, *, seed: int = 0):
    """A draw of an RMSNorm's output, Dew's sum of squares `dew_sum` against XLA's and the members'
    (NORM_TREES), over orders that permute the width, the axis the sum runs over."""
    keys = jax.random.split(jax.random.key(seed), 2)
    x = jax.random.normal(keys[0], shape, jnp.float32)
    scale = 0.7 + 0.2 * jax.random.normal(keys[1], shape[-1:], jnp.float32)
    dew = jax.jit(functools.partial(rms_norm, dew_sum))
    members = {REFERENCE: jax.jit(functools.partial(rms_norm, xla_sum))} | {
        tree: jax.jit(functools.partial(rms_norm, tree_of(tree))) for tree in NORM_TREES}
    with jax.enable_x64(new_val=True):
        truth = np.asarray(jax.jit(functools.partial(rms_norm, xla_sum))(*widened((x, scale))))

    def draw(order):
        units = orbit(shape[-1], order)
        back = np.argsort(units)
        moved = (x[..., units], scale[units])

        def found(fn):
            return {"output": np.asarray(fn(*moved), np.float64)[..., back]}
        return found(dew), {name: found(fn) for name, fn in members.items()}, {"output": truth}
    return draw


@pytest.mark.parametrize("shape", [(32, 256), (32, 100)], ids=["whole-blocks", "ragged"])
def test_an_rms_norm_summed_in_halves_is_as_exact_as_the_ensemble(shape):
    """The null of the half-sum norm: its sum of squares summed as two halves added is as exact as
    XLA's one sum, so it passes against the ensemble, whose members are each admitted at the output;
    a width that is no whole number of blocks (100) passes too."""
    report = assert_leaves_as_exact_over_orders(norm_draw(halves_sum, shape), f"halves {shape}",
                                                rows=False)
    assert REFERENCE in report["['output']"]["admitted"]


def reversed_sum(squares):
    """XLA's own sum over the terms in reverse: another tree, as exact, that the ensemble does not hold."""
    return jnp.sum(squares[..., ::-1], axis=-1, keepdims=True)


def test_a_tree_the_ensemble_does_not_hold_passes():
    """A held-out tree, XLA's sum over the terms reversed, is as exact as the reference and passes."""
    assert_leaves_as_exact_over_orders(norm_draw(reversed_sum, (32, 256)), "held-out tree", rows=False)


def lost_term(squares):
    """The sum of squares with its last term dropped: a defect."""
    return jnp.sum(squares[..., :-1], axis=-1, keepdims=True)


def test_a_norm_that_drops_a_term_fails():
    with pytest.raises(AssertionError):
        assert_leaves_as_exact_over_orders(norm_draw(lost_term, (32, 256)), "lost term", rows=False)


def test_a_hundredfold_error_on_a_thousandth_of_a_row_fails():
    """Dew rounds as XLA does but at 1 of a row's 1024 elements, where it is 100 times as far from
    float64: the leaf and row rules average it away, the element rule does not."""
    base = norm_draw(xla_sum, (8, 1024))

    def draw(order):
        dew, members, truth = base(order)
        output = dew["output"].copy()
        output[3, 517] = truth["output"][3, 517] + 100 * (output[3, 517] - truth["output"][3, 517])
        return {"output": output}, members, truth

    with pytest.raises(AssertionError):
        assert_leaves_as_exact_over_orders(draw, "one element", rows=False)


def test_a_hundredfold_error_where_the_ensemble_raises_the_bar_most_fails():
    """A fault at the element whose bar the members raise most over the reference's own, measured over
    the first batch before Dew is injected: a hundred times the widest member's error there fails. The
    fault is the widest member's, not the reference's: where the native sum rounds an element nearly
    exactly, the members raise its bar furthest (436 times on an A100's sum), and a hundred times the
    native error there is a rounding the widest member makes."""
    shape = (8, 1024)
    base = norm_draw(xla_sum, shape)
    sums = {}
    for order in range(ORDERS):
        _, members, truth = base(order)
        for name, found in members.items():
            sums[name] = sums.get(name, 0) + np.abs(found["output"] - truth["output"])
    native, widest = sums[REFERENCE], np.max(np.stack(list(sums.values())), axis=0)
    raised = np.where(native > 0, widest / np.where(native > 0, native, 1), 0)
    at = np.unravel_index(int(np.argmax(raised)), shape)
    member = max(sums, key=lambda name: sums[name][at])

    def draw(order):
        dew, members, truth = base(order)
        output = dew["output"].copy()
        output[at] = truth["output"][at] + 100 * (members[member]["output"][at] - truth["output"][at])
        return {"output": output}, members, truth

    with pytest.raises(AssertionError):
        assert_leaves_as_exact_over_orders(draw, "the widest element", rows=False)


def test_an_ulp_off_where_every_member_is_exact_fails():
    """Integer squares sum exactly under every tree, so every member is exactly float64's sum; a Dew
    one ulp off them fails, though one ulp is all it is."""
    x = np.random.default_rng(0).integers(-8, 9, (16, 64)).astype(np.float32)

    def summed(tree):
        return jax.jit(lambda values: tree_sum(values, -1, tree) if tree else jnp.sum(values, axis=-1))

    members = {REFERENCE: summed(None)} | {tree: summed(tree) for tree in NORM_TREES}
    truth = x.astype(np.float64).sum(axis=-1)

    def draw(order):
        units = orbit(x.shape[-1], order)
        outputs = {name: {"sum": np.asarray(fn(x[:, units]), np.float64)} for name, fn in members.items()}
        assert all(np.array_equal(found["sum"], truth) for found in outputs.values())
        dew = np.nextafter(np.float32(truth), np.float32(np.inf)).astype(np.float64)
        return {"sum": dew}, outputs, {"sum": truth}

    with pytest.raises(AssertionError):
        assert_leaves_as_exact_over_orders(draw, "one ulp", rows=False)


def tree_attention(q, k, v, mask, trees):
    """Dense attention with its scores summed over the head width under `trees["width"]` and its
    softmax's denominator and the probabilities times the values summed over keys under
    `trees["keys"]`: q [B, S, H, D], k and v [B, T, Hk, D], `mask` [B, 1, S, T]."""
    groups = q.shape[2] // k.shape[2]
    k, v = (jnp.repeat(x, groups, axis=2) for x in (k, v))
    scores = tree_contract(q.transpose(0, 2, 1, 3)[:, :, :, None], k.transpose(0, 2, 1, 3)[:, :, None],
                           -1, trees["width"]) / np.sqrt(q.shape[-1]).astype(q.dtype)
    scores = jnp.where(mask, scores, -jnp.inf)
    weights = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
    total = tree_sum(weights, -1, trees["keys"])[..., None]
    output = tree_contract(weights[..., None], v.transpose(0, 2, 1, 3)[:, :, None], -2, trees["keys"])
    return (output / total).transpose(0, 2, 1, 3)


ATTENTION_TREES = {"width": ("pairwise", "blocked-8"), "keys": ("pairwise", "blocked-8")}
"""Attention's reduction classes and the trees each declares; their four assignments are members."""

KEY_ORDERS = tuple(range(100, 116))
"""The seeds of the fixed key orders the reference's own attention is a member under: an exact symmetry
of its math, the keys, the values and the mask's columns moved alike, so each sums its keys in an
order of its own that no width order moves."""


def attention_draw(dew, length: int, chunk: int):
    """A draw of chunked causal attention, its output and its gradients in q, k and v, Dew's `dew`
    against XLA's reference attention and the members (ATTENTION_TREES' assignments and the reference
    over each of KEY_ORDERS), over orders that permute the head width, which the scores sum over, and
    the cotangent with the output; the sums over keys no order moves, and the members are what
    reorder them."""
    keys = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(keys[0], (2, length, 4, 16), jnp.float32)
    k, v = (jax.random.normal(key, (2, length, 2, 16), jnp.float32) for key in keys[1:3])
    cotangent = jax.random.normal(keys[3], q.shape, jnp.float32)
    keep = np.broadcast_to(np.asarray(splash.ChunkedCausalMask((length, length), chunk)[:, :]),
                           (2, length, length))
    mask = jnp.asarray(keep)[:, None]

    def reference(q, k, v):
        return scaled_dot_product_attention(q, k, v, implementation="reference", mask=mask)

    def results(fn, cot, q, k, v):
        output, pullback = jax.vjp(fn, q, k, v)
        return (output, *pullback(cot.astype(output.dtype)))

    sides = {"dew": dew(mask), REFERENCE: reference} | {
        f"width {chosen['width']}, keys {chosen['keys']}":
            functools.partial(tree_attention, mask=mask, trees=chosen)
        for chosen in assignments(ATTENTION_TREES)} | {
        f"keys in order {seed}": keys_in_order(mask, seed) for seed in KEY_ORDERS}
    compiled = {name: jax.jit(functools.partial(results, fn)) for name, fn in sides.items()}
    with jax.enable_x64(new_val=True):
        truth = [np.asarray(leaf) for leaf in jax.jit(functools.partial(results, reference))(
            *widened((cotangent, q, k, v)))]
    names = ("output", "query", "key", "value")

    def draw(order):
        units = orbit(q.shape[-1], order)
        back = np.argsort(units)
        moved = [x[..., units] for x in (cotangent, q, k, v)]
        found = {name: dict(zip(names, (np.asarray(leaf, np.float64)[..., back] for leaf in fn(*moved)),
                                strict=True)) for name, fn in compiled.items()}
        dew_found = found.pop("dew")
        return dew_found, found, dict(zip(names, truth, strict=True))
    return draw


def keys_in_order(mask, seed: int):
    """Dense reference attention over the keys in the fixed order `seed` draws, the values and the
    mask's columns with them: the same function, another tree over keys."""
    def attend(q, k, v):
        order = np.random.default_rng(seed).permutation(k.shape[1])
        return scaled_dot_product_attention(q, k[:, order], v[:, order], implementation="reference",
                                            mask=mask[..., order])
    return attend


def test_attention_that_loses_a_key_fails():
    """Dense attention that drops each row's last visible key: a defect."""
    def dropping(mask):
        last = jnp.cumsum(mask, axis=-1) == jnp.sum(mask, axis=-1, keepdims=True)
        kept = mask & ~(last & mask) | (jnp.sum(mask, axis=-1, keepdims=True) == 1) & mask

        def attend(q, k, v):
            return scaled_dot_product_attention(q, k, v, implementation="reference", mask=kept)
        return attend

    with pytest.raises(AssertionError):
        assert_leaves_as_exact_over_orders(attention_draw(dropping, 37, 8), "lost key", rows=False)
