"""The source-aware gate (tests/reference_error.py `assert_graph_as_exact`, src/dew/nn/kernels/graph.py):
an RMS norm and dense attention as graphs of the values they round and hold, their backward from the
common forward's saved values and a fixed cotangent, held node by node on common parents.

Each graph's nodes are jitted alone and its witness composes them, so the witness is the arithmetic
the nodes run; a witness that rounds otherwise fails check 1. The truth graph is the reference's
graph at float64, and its backward is checked against the float64 chain rule (`jax.vjp`).

Every Sum node, the norms' and attention's, is held against the canonical Sum family
(reference_trees.sum_family) as members beside the reference: two trees equally exact on average
differ element by element, and against the reference's one tree equally exact code was flagged on
some CPUs and not others (ledger 758: a pairwise key sum at 2 of 768 elements on an AMD AVX2 worker
and none on AVX-512; the norm backward's dscale halves at 1 of 256 on an Intel Xeon). With the
family, held-out trees as exact as the reference (recursive halving, pairwise over the keys
reversed, XLA's sum reversed, strided-32) passed at 48, 128 and 256 keys on both ISAs, a tree less
exact on long positive sums (blocked-12) failed there, and an eight-ulp fault at one element failed
everywhere (24 armada workers a shape); the rule's false-failure rate at real element counts is not calibrated by these counts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import REFERENCE, assert_graph_as_exact
from reference_trees import sum_family, tree_sum

from dew.nn.kernels.graph import Graph, Node, Pointwise, Sum

EPSILON = 1e-6


def composed(inputs, nodes, outputs, held=()):
    """A graph whose witness runs its own nodes in turn and returns its outputs and the values `held`."""
    def fused(**given):
        values = dict(given)
        for node in nodes:
            values[node.name] = node.fn(**{parent: values[parent] for parent in node.parents})
        return {name: values[name] for name in (*outputs, *held)}
    return Graph(inputs, tuple(nodes), tuple(outputs), fused)


SUM_TREES = sum_family(16)
"""Every Sum node's members in these graphs: the canonical Sum family at a 16-term tile
(reference_trees.sum_family)."""


def tree_members(arithmetic, trees):
    """The reference's node `arithmetic(sum)` under each of `trees` (reference_trees.tree_sum) as members."""
    return {tree: jax.jit(arithmetic(lambda terms, tree=tree: tree_sum(terms, -1, tree))) for tree in trees}


def norm_graph(summed, trees=()):
    """An RMS norm over the width: the sum of squares `summed` (a Sum over the width, with the
    reference's math under each of `trees` as members), then the norm."""
    def squared(total):
        return lambda x: total(jnp.square(x))
    squares = Node("squares", ("x",), Sum({"x": -1}), jax.jit(squared(summed)),
                   members=tree_members(squared, trees))
    output = Node("output", ("x", "squares", "scale"), Pointwise(),
                  jax.jit(lambda x, squares, scale:
                          x * jax.lax.rsqrt(squares[..., None] / x.shape[-1] + EPSILON) * scale))
    return composed(("x", "scale"), (squares, output), ("output",), held=("squares",))


def width_orbit(inputs):
    """Orders that permute the width of `x` and `scale`, and the map taking `output` back."""
    def orbit(order):
        units = np.arange(inputs["x"].shape[-1]) if order == 0 else \
            np.random.default_rng(order).permutation(inputs["x"].shape[-1])
        back = np.argsort(units)
        moved = {"x": inputs["x"][..., units], "scale": inputs["scale"][units]}
        return moved, lambda outputs: {name: value[..., back] for name, value in outputs.items()}
    return orbit


def norm_inputs():
    keys = jax.random.split(jax.random.key(0), 2)
    return {"x": jax.random.normal(keys[0], (16, 256), jnp.float32),
            "scale": 0.7 + 0.2 * jax.random.normal(keys[1], (256,), jnp.float32)}


def xla(squares):
    return jnp.sum(squares, axis=-1)


def halves(squares):
    left, right = jnp.split(squares, 2, axis=-1)
    return jnp.sum(left, axis=-1) + jnp.sum(right, axis=-1)


def test_a_norm_that_sums_in_halves_is_held_where_it_rounds_and_passes():
    """The half-sum norm's sum of squares, held over its own width orders on common parents, and its
    pointwise tail, held once on the reference's sums and on Dew's, are on par: it passes."""
    inputs = norm_inputs()
    reference = norm_graph(xla, SUM_TREES)
    report = assert_graph_as_exact(norm_graph(halves), reference, reference, inputs,
                                   width_orbit(inputs), "halves")
    assert set(report) == {(name, bundle) for name in ("squares", "output")
                           for bundle in ("reference's parents", "dew's parents")}


def test_a_norm_that_drops_a_square_fails_at_its_sum():
    inputs = norm_inputs()
    with pytest.raises(AssertionError):
        assert_graph_as_exact(norm_graph(lambda squares: jnp.sum(squares[..., 1:], axis=-1)), norm_graph(xla),
                              norm_graph(xla), inputs, width_orbit(inputs), "lost square")


def test_a_pointwise_tail_an_ulp_off_where_the_reference_is_exact_fails():
    """Dew's tail one ulp past the reference's at every element: a pointwise node is held element by
    element on common parents, so even an ulp fails where the reference rounds correctly."""
    inputs = norm_inputs()
    dew = norm_graph(xla)
    off = Node("output", ("x", "squares", "scale"), Pointwise(),
               jax.jit(lambda x, squares, scale:
                       jnp.nextafter(dew.node("output").fn(x, squares, scale), jnp.inf)))
    with pytest.raises(AssertionError):
        assert_graph_as_exact(composed(dew.inputs, (dew.node("squares"), off), ("output",), ("squares",)),
                              norm_graph(xla), norm_graph(xla), inputs, width_orbit(inputs), "an ulp")


def test_a_witness_that_rounds_otherwise_than_its_graph_fails():
    inputs = norm_inputs()
    dew = norm_graph(xla)
    lying = Graph(dew.inputs, dew.nodes, dew.outputs,
                  lambda **given: {"output": dew.fused(**given)["output"] * np.float32(1 + 2 ** -20),
                                   "squares": dew.fused(**given)["squares"]})
    with pytest.raises(AssertionError):
        assert_graph_as_exact(lying, norm_graph(xla), norm_graph(xla), inputs, width_orbit(inputs), "witness")


def test_a_nan_at_one_element_fails():
    inputs = norm_inputs()
    dew = norm_graph(xla)
    broken = Node("output", ("x", "squares", "scale"), Pointwise(),
                  jax.jit(lambda x, squares, scale:
                          dew.node("output").fn(x, squares, scale).at[3, 5].set(jnp.nan)))
    with pytest.raises(AssertionError):
        assert_graph_as_exact(composed(dew.inputs, (dew.node("squares"), broken), ("output",), ("squares",)),
                              norm_graph(xla), norm_graph(xla), inputs, width_orbit(inputs), "nan")


def norm_backward(summed, trees=()):
    """The norm's backward from the common forward's saved `squares` and a fixed cotangent `g`: the sum
    over the width of g·x·scale (a Sum), then dx and dscale's per-row terms (Pointwise), and dscale's sum
    over the rows (a Sum); each Sum with the reference's math under each of `trees` as members."""
    def inverse(x, squares):
        return jax.lax.rsqrt(squares[..., None] / x.shape[-1] + EPSILON)

    def projection(total):
        return lambda g, x, scale: total(g * scale * x)

    def scale_gradient(total):
        return lambda g, x, squares: total(jnp.moveaxis(g * x * inverse(x, squares), 0, -1))

    projected = Node("projected", ("g", "x", "scale"), Sum({"g": -1, "x": -1, "scale": -1}),
                     jax.jit(projection(summed)), members=tree_members(projection, trees))
    dx = Node("dx", ("g", "x", "scale", "squares", "projected"), Pointwise(),
              jax.jit(lambda g, x, scale, squares, projected: inverse(x, squares) * scale * g
                      - x * inverse(x, squares) ** 3 * projected[..., None] / x.shape[-1]))
    dscale = Node("dscale", ("g", "x", "squares"), Sum({"g": 0, "x": 0, "squares": 0}),
                  jax.jit(scale_gradient(summed)), members=tree_members(scale_gradient, trees))
    return composed(("g", "x", "scale", "squares"), (projected, dx, dscale), ("dx", "dscale"))


def backward_inputs():
    forward = norm_inputs()
    squares = norm_graph(xla).fused(**forward)["squares"]
    return {"g": jax.random.normal(jax.random.key(5), forward["x"].shape, jnp.float32), **forward,
            "squares": squares}


def backward_orbit(inputs):
    """Width orders of the backward's inputs, the saved sums unmoved, and dx taken back."""
    def orbit(order):
        width = inputs["x"].shape[-1]
        units = np.arange(width) if order == 0 else np.random.default_rng(order).permutation(width)
        back = np.argsort(units)
        moved = {"g": inputs["g"][..., units], "x": inputs["x"][..., units],
                 "scale": inputs["scale"][units], "squares": inputs["squares"]}
        return moved, lambda outputs: {"dx": outputs["dx"][..., back], "dscale": outputs["dscale"][..., back]}
    return orbit


def test_the_float64_backward_graph_is_the_chain_rule():
    """The truth backward graph, at float64 on the forward's widened values, is `jax.vjp` of the
    float64 norm: the backward's oracle is the chain rule, not a hand derivation."""
    with jax.enable_x64(new_val=True):
        given = {name: jnp.asarray(value, jnp.float64) for name, value in backward_inputs().items()}
        given["squares"] = jnp.sum(jnp.square(given["x"]), axis=-1)
        found = norm_backward(xla).fused(**given)
        _, pullback = jax.vjp(lambda x, scale: norm_graph(xla).fused(x=x, scale=scale)["output"], given["x"],
                              given["scale"])
        dx, dscale = pullback(given["g"])
    np.testing.assert_allclose(found["dx"], dx, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(found["dscale"], dscale, rtol=1e-12, atol=1e-14)


def test_a_backward_that_sums_in_halves_passes_and_one_that_drops_a_term_fails():
    inputs = backward_inputs()
    reference = norm_backward(xla, SUM_TREES)
    assert_graph_as_exact(norm_backward(halves), reference, reference, inputs, backward_orbit(inputs),
                          "backward halves")
    with pytest.raises(AssertionError):
        assert_graph_as_exact(norm_backward(lambda terms: jnp.sum(terms[..., 1:], axis=-1)),
                              norm_backward(xla), norm_backward(xla), inputs, backward_orbit(inputs),
                              "backward lost term")


def key_sums(keys_sum):
    """The denominator's and the accumulator's arithmetic, each a sum over keys by `keys_sum`."""
    return {"total": jax.jit(lambda weights: keys_sum(weights)),
            "accumulated": jax.jit(lambda weights, v: keys_sum(weights[:, None, :] * v.T[None]))}


def attention_graph(scores_of, keys_sum, trees=()):
    """Dense causal attention for one head: its scores (a Sum over the head width), each row's max and
    its exponentials (Pointwise), the denominator and the accumulator (Sums over keys), and the output.
    Every Sum takes the reference's math under each of `trees` as members."""
    sums = key_sums(keys_sum)
    members = {tree: key_sums(lambda terms, tree=tree: tree_sum(terms, -1, tree)) for tree in trees}
    scores = Node("scores", ("q", "k"), Sum({"q": -1, "k": -1}), jax.jit(scores_of),
                  members=tree_members(masked_scores, trees))
    weights = Node("weights", ("scores",), Pointwise(), jax.jit(
        lambda scores: jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))))
    total = Node("total", ("weights",), Sum({"weights": -1}), sums["total"],
                 members={tree: fns["total"] for tree, fns in members.items()})
    accumulated = Node("accumulated", ("weights", "v"), Sum({"weights": -1, "v": 0}), sums["accumulated"],
                       members={tree: fns["accumulated"] for tree, fns in members.items()})
    output = Node("output", ("accumulated", "total"), Pointwise(),
                  jax.jit(lambda accumulated, total: accumulated / total[:, None]))
    return composed(("q", "k", "v"), (scores, weights, total, accumulated, output), ("output",),
                    held=("total", "accumulated"))


def masked_scores(summed):
    def scores(q, k):
        product = summed(q[:, None, :] * k[None, :, :]) / np.sqrt(q.shape[-1]).astype(q.dtype)
        causal = jnp.arange(q.shape[0])[:, None] >= jnp.arange(k.shape[0])[None, :]
        return jnp.where(causal, product, -1e30)
    return scores


def attention_inputs():
    keys = jax.random.split(jax.random.key(1), 3)
    return {name: jax.random.normal(key, (48, 16), jnp.float32)
            for name, key in zip("qkv", keys, strict=True)}


def head_orbit(inputs):
    def orbit(order):
        units = np.arange(16) if order == 0 else np.random.default_rng(order).permutation(16)
        back = np.argsort(units)
        moved = {name: value[..., units] for name, value in inputs.items()}
        return moved, lambda outputs: {"output": outputs["output"][..., back]}
    return orbit


def plain(terms):
    return jnp.sum(terms, axis=-1)


def halving(terms):
    """Recursive halving at n // 2: a tree of pairwise's accuracy that the key sums' family does not hold."""
    n = terms.shape[-1]
    return terms[..., 0] if n == 1 else halving(terms[..., : n // 2]) + halving(terms[..., n // 2:])


@pytest.mark.parametrize("held_out", ["halving", "reversed"])
def test_attention_summing_its_keys_by_a_held_out_tree_passes_against_the_family(held_out):
    """Two trees equally exact on average differ element by element in expected error, so a key sum is
    held against the worst admitted member of the canonical Sum family, not the reference's one tree
    (ledger 758). A held-out tree as exact as the reference passes on every worker: recursive halving,
    and XLA's own sum over the keys reversed."""
    inputs = attention_inputs()
    keys_sum = halving if held_out == "halving" else (lambda terms: jnp.sum(terms[..., ::-1], axis=-1))
    reference = attention_graph(masked_scores(plain), plain, SUM_TREES)
    report = assert_graph_as_exact(attention_graph(masked_scores(plain), keys_sum), reference, reference,
                                   inputs, head_orbit(inputs), held_out)
    assert all(REFERENCE in report[node, "reference's parents"]["['node']"]["admitted"]
               for node in ("total", "accumulated"))


def test_a_key_sum_eight_ulps_off_at_one_element_fails_against_the_family():
    """The family widens the bar element by element and no further: a key sum eight ulps past its own
    rounding at one element of 768 fails."""
    def off(terms):
        found = tree_sum(terms, -1, "pairwise")
        at = (3,) if found.ndim == 1 else (3, 5)
        for _ in range(8):
            found = found.at[at].set(jnp.nextafter(found[at], jnp.inf))
        return found

    inputs = attention_inputs()
    reference = attention_graph(masked_scores(plain), plain, SUM_TREES)
    with pytest.raises(AssertionError):
        assert_graph_as_exact(attention_graph(masked_scores(plain), off), reference, reference, inputs,
                              head_orbit(inputs), "eight ulps")


def test_attention_that_loses_a_key_fails_at_its_denominator():
    inputs = attention_inputs()
    reference = attention_graph(masked_scores(plain), plain)
    dew = attention_graph(masked_scores(plain), lambda terms: jnp.sum(terms[..., 1:], axis=-1))
    with pytest.raises(AssertionError):
        assert_graph_as_exact(dew, reference, reference, inputs, head_orbit(inputs), "lost key")
