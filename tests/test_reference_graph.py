"""The source-aware gate (tests/reference_ensemble.py `assert_graph_as_exact`, src/dew/nn/kernels/graph.py):
an RMS norm and dense attention as graphs of the values they round and hold, their backward from the
common forward's saved values and a fixed cotangent, held node by node on common parents.

Each graph's nodes are jitted alone and its witness composes them, so the witness is the arithmetic
the nodes run; a witness that rounds otherwise fails check 1. The truth graph is the reference's
graph at float64, and its backward is checked against the float64 chain rule (`jax.vjp`).

The element rule's false-failure rate on equally exact sides is not calibrated, so a null that sums
attention's keys by another tree than the reference's is measured beside the suite, not asserted in
it: with both sides' key orders redrawn at the node it flags 0 to about 1e-3 of the accumulator's
elements at every stage (13 to 24 of 8192 to 32768 at 128 and 256 keys; 2 of 768 at 48 keys on an
x86-64-v3 worker, none on x86-64-v4), a rate that grows with the element count."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_ensemble import assert_graph_as_exact

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


def norm_graph(summed):
    """An RMS norm over the width: the sum of squares `summed` (a Sum over the width), then the norm."""
    squares = Node("squares", ("x",), Sum({"x": -1}), jax.jit(lambda x: summed(jnp.square(x))))
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
    report = assert_graph_as_exact(norm_graph(halves), norm_graph(xla), norm_graph(xla), inputs,
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


def norm_backward(summed):
    """The norm's backward from the common forward's saved `squares` and a fixed cotangent `g`: the sum
    over the width of g·x·scale (a Sum), then dx and dscale's per-row terms (Pointwise), and dscale's sum
    over the rows (a Sum)."""
    def inverse(x, squares):
        return jax.lax.rsqrt(squares[..., None] / x.shape[-1] + EPSILON)

    projected = Node("projected", ("g", "x", "scale"), Sum({"g": -1, "x": -1, "scale": -1}),
                     jax.jit(lambda g, x, scale: summed(g * scale * x)))
    dx = Node("dx", ("g", "x", "scale", "squares", "projected"), Pointwise(),
              jax.jit(lambda g, x, scale, squares, projected: inverse(x, squares) * scale * g
                      - x * inverse(x, squares) ** 3 * projected[..., None] / x.shape[-1]))
    dscale = Node("dscale", ("g", "x", "squares"), Sum({"g": 0, "x": 0, "squares": 0}),
                  jax.jit(lambda g, x, squares: summed(jnp.moveaxis(g * x * inverse(x, squares), 0, -1))))
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
    assert_graph_as_exact(norm_backward(halves), norm_backward(xla), norm_backward(xla), inputs,
                          backward_orbit(inputs), "backward halves")
    with pytest.raises(AssertionError):
        assert_graph_as_exact(norm_backward(lambda terms: jnp.sum(terms[..., 1:], axis=-1)),
                              norm_backward(xla), norm_backward(xla), inputs, backward_orbit(inputs),
                              "backward lost term")


def attention_graph(scores_of, keys_sum):
    """Dense causal attention for one head: its scores (a Sum over the head width), each row's max and
    its exponentials (Pointwise), the denominator and the accumulator (Sums over keys), and the output."""
    scores = Node("scores", ("q", "k"), Sum({"q": -1, "k": -1}), jax.jit(scores_of))
    weights = Node("weights", ("scores",), Pointwise(), jax.jit(
        lambda scores: jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))))
    total = Node("total", ("weights",), Sum({"weights": -1}), jax.jit(lambda weights: keys_sum(weights)))
    accumulated = Node("accumulated", ("weights", "v"), Sum({"weights": -1, "v": 0}), jax.jit(
        lambda weights, v: keys_sum(weights[:, None, :] * v.T[None])))
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


def test_attention_that_loses_a_key_fails_at_its_denominator():
    inputs = attention_inputs()
    reference = attention_graph(masked_scores(plain), plain)
    dew = attention_graph(masked_scores(plain), lambda terms: jnp.sum(terms[..., 1:], axis=-1))
    with pytest.raises(AssertionError):
        assert_graph_as_exact(dew, reference, reference, inputs, head_orbit(inputs), "lost key")
