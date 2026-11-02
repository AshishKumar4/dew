"""The algebra fuzzed (docs/design/testing.md, property tests): programs drawn from a grammar of what the pass
rewrites (sums, products, contractions, reductions, softmax and log-sum-exp walks) each rewritten on par with
itself as written against the float64 truth (check 1); declarations drawn from a grammar of softmax walks
(their logits' hooks, their outputs and their exponential's base) each derived, its walk over the key tiles in
order or in two halves merged on par with the declared function. Hypothesis draws the structures and shapes,
`assert_as_exact`'s seeds the values; every value is finite, as the gate holds it."""

from datetime import timedelta

import hypothesis
import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st
from reference_error import assert_as_exact
from test_algebra_derive import WIDTH, declared, draw, merged_whole, walked_whole, whole

from dew import algebra
from dew.algebra import SATURATION, derive

LEAVES = ("x", "y", "a")
"""A program's operands: `x` and `y` `[rows, columns]`, `a` `[columns]` broadcast along the rows; a
contraction's other operand `w` `[columns, columns]`."""

UNARY = ("tanh", "square", "abs", "contract", "sum", "max", "softmax", "logsumexp")
BINARY = ("add", "sub", "mul")
SCALES = (0.125, 0.5, 2.0, -3.0)


@st.composite
def programs(draw_from, depth: int = 3):
    """A program's tree: an operand at depth 0 (or where the draw stops early), else an operation of one or
    two drawn trees, or a power of two's or another constant's scaling of one. Every exponential is of a
    value less its row's maximum, every logarithm of a sum of such exponentials, so a finite input gives a
    finite value."""
    if depth == 0 or (depth < 3 and draw_from(st.booleans())):
        return ("leaf", draw_from(st.sampled_from(LEAVES)))
    kind = draw_from(st.sampled_from((*UNARY, *BINARY, "scale")))
    if kind in BINARY:
        return (kind, draw_from(programs(depth - 1)), draw_from(programs(depth - 1)))
    if kind == "scale":
        return (kind, draw_from(st.sampled_from(SCALES)), draw_from(programs(depth - 1)))
    return (kind, draw_from(programs(depth - 1)))


def _program(tree):
    """`tree` as a function of `(x, y, a, w)`."""
    def run(x, y, a, w):
        def value(node):
            kind = node[0]
            if kind == "leaf":
                return {"x": x, "y": y, "a": jnp.broadcast_to(a, x.shape)}[node[1]]
            if kind == "scale":
                return node[1] * value(node[2])
            if kind in BINARY:
                left, right = value(node[1]), value(node[2])
                return {"add": left + right, "sub": left - right, "mul": left * right}[kind]
            v = value(node[1])
            if kind in ("tanh", "abs"):
                return getattr(jnp, kind)(v)
            if kind == "square":
                return v * v
            if kind == "contract":
                return v @ w
            if kind in ("sum", "max"):
                return jnp.broadcast_to(getattr(jnp, kind)(v, axis=-1, keepdims=True), v.shape)
            m = jnp.max(v, axis=-1, keepdims=True)
            e = jnp.exp(v - m)
            if kind == "softmax":
                return e / jnp.sum(e, axis=-1, keepdims=True)
            return jnp.broadcast_to(m + jnp.log(jnp.sum(e, axis=-1, keepdims=True)), v.shape)

        return value(tree)

    return run


# About 0.2 s an example on armada's CPU (at most 0.4 s), 40 s in all.
@hypothesis.settings(max_examples=200, deadline=timedelta(seconds=2))
@hypothesis.given(tree=programs(), rows=st.sampled_from((4, 8)), columns=st.sampled_from((8, 16)),
                  dtype=st.sampled_from((jnp.float32, jnp.bfloat16)))
def test_a_rewritten_program_is_on_par_with_the_program(tree, rows, columns, dtype):
    """Any program the grammar draws, rewritten by the pass, is on par with the program as written against
    the float64 truth, its every value finite."""
    def operands(seed):
        rng = np.random.default_rng(seed)
        return tuple(jnp.asarray(rng.standard_normal(shape), dtype)
                     for shape in ((rows, columns), (rows, columns), (columns,), (columns, columns)))

    run = _program(tree)
    assert_as_exact(jax.jit(algebra.rewritten(run)), jax.jit(run), operands, f"fuzzed {tree}", seeds=16)


HOOKS = ("none", "softcap", "bias", "causal")
OUTPUTS = ("values", "lse", "peak", "total")
LOG2E = float(np.log2(np.e))


def _declared(hook: str, outputs: tuple[str, ...], base):
    """A softmax walk over the keys of its logits under `hook` (a softcap's, a distance bias's, a causal
    mask's, or none), in base `base` (`e`, or 2 with log2(e) folded into the scale), its `outputs` among the
    weighted values, the log-sum-exp, the maximum and the total."""
    def function(q, k, v, rows, keys):
        s = (q @ k.T) * (WIDTH ** -0.5 * (LOG2E if base == 2 else 1.0))
        if hook == "softcap":
            s = 30.0 * jnp.tanh(s / 30.0)
        elif hook == "bias":
            s = s - 0.125 * jnp.abs(rows[:, None] - keys[None, :]).astype(s.dtype)
        seen = keys[None, :] <= rows[:, None] if hook == "causal" else None
        if seen is not None:
            s = jnp.where(seen, s, -jnp.inf)
        m = jnp.max(s, axis=-1)
        p = (jnp.exp2 if base == 2 else jnp.exp)(s - m[:, None])
        if seen is not None:
            p = jnp.where(seen, p, 0.0)
        total = jnp.sum(p, axis=-1)
        made = {"values": (p @ v) / total[:, None], "lse": m + (jnp.log2 if base == 2 else jnp.log)(total),
                "peak": m, "total": total}
        return tuple(made[name] for name in outputs) if len(outputs) > 1 else made[outputs[0]]

    return function


# About 0.7 s an example on armada's CPU (at most 1 s), 42 s in all.
@hypothesis.settings(max_examples=60, deadline=timedelta(seconds=5))
@hypothesis.given(hook=st.sampled_from(HOOKS), outputs=st.sets(st.sampled_from(OUTPUTS), min_size=1),
                  base=st.sampled_from(("e", 2)), merged=st.booleans())
# A base-2 log-sum-exp under a bias: its `log2` was outside the language, kept as an opaque value the finish
# evaluated from the whole operands.
@hypothesis.example(hook="bias", outputs={"lse"}, base=2, merged=False)
def test_a_derived_walk_is_on_par_with_its_declaration(hook, outputs, base, merged):
    """Any softmax walk the grammar draws derives, and its derived walk, over the key tiles in order or in two
    halves merged, is on par with the declared function against the float64 truth."""
    function = _declared(hook, tuple(sorted(outputs)), base)
    forms = derive.derive(declared(function, base), accurate=False, saturation=SATURATION)
    run = merged_whole(forms) if merged else walked_whole(forms)
    label = f"derived {hook} {sorted(outputs)} base {base} {'merged' if merged else 'walked'}"
    assert_as_exact(jax.jit(run), jax.jit(whole(function)), draw, label, seeds=16)
