"""The derived forms' stability (bible §6; docs/design/testing.md, a template's stability suite): every
derived form under the stability families that broke derived forms before, generated over the template
suites' own families and the edges each pass found:

- attention's: the template suite's every family (fp32 and bf16 as drawn, and in fp16, the dtype the derived
  forward walks), on the plain-JAX mainloop, forward and backward, its output and gradients finite wherever
  the float64 truth's rounded to their dtype are and a row that sees nothing zero
  (`test_attention_stability.assert_finite`, `_held_empty`); and the edges as their own draws: a query a score
  hook empties beside a value product past fp32's largest, a query no span reaches beside one, keys past a
  whole tile, a hook infinite or NaN at the keys it masks, values at fp32's and fp16's largest and least, each
  in every dtype the derived forms walk, finite wherever the reference is;
- the recurrence's: the template suite's every site and regime (`test_recurrence_stability`), its output and
  final memory finite wherever float64's are and on par with the token-by-token recurrence's (check 1), and
  the decays' edges both ways (none, every token vanishing, a live decay then a vanishing one, the reverse),
  each through its value, its gradient and its gradient's tangent (every order the derived forms hold).

A drawn input's check 1 is each derived form's own suite's (tests/test_attention_derived.py,
tests/test_algebra_derive.py, tests/test_algebra_scan.py); the families here are adversarial, where the fast
class owes finiteness and exact zeros (docs/design/testing.md)."""

import contextlib
import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_attention_stability as attention_suite
import test_recurrence_stability as recurrence_suite
from reference_error import widened, with_gradients
from test_recurrence_chunk import chunk_on, token_by_token

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template, attention_portable


@contextlib.contextmanager
def derived(forms: str = "fast"):
    """Within the block, the run's `forms` (the fast ones, or the accurate ones: natural logits, the anchor's
    split, the keyed sums) with the template's walks the derived ones."""
    numerics = dataclasses.replace(generation.numerics(), forms=forms)
    with generation.run_precision(numerics):
        yield


FORMS = ["fast", "accurate"]
"""The forms each derived walk is held under."""


ATTENTION_FAMILIES = [(name, dtype) for name in attention_suite.TEMPLATE_FAMILIES
                      for dtype in (None, "float16")]
"""Each template family as drawn (None) and in fp16, the dtype the derived forward walks."""


@pytest.mark.parametrize("forms", FORMS)
@pytest.mark.parametrize(("name", "dtype"), ATTENTION_FAMILIES,
                         ids=[f"{name}-{dtype or 'as drawn'}" for name, dtype in ATTENTION_FAMILIES])
def test_the_derived_attention_is_finite_where_the_truth_is_on_every_family(name, dtype, forms):
    """The plain-JAX mainloop's derived forms under each of its `forms`, forward and backward, finite
    wherever the float64 truth rounded to their dtype is (`assert_finite`) over the family's draws (an
    anchor's, one key's and a masked query's among them), a row that sees nothing zero."""
    family = attention_suite.FAMILIES[name]
    family = family if dtype is None else dataclasses.replace(family, dtype=dtype)
    draw = attention_suite._cast(family, family.draw)
    truth = with_gradients(attention_suite._reference(family, anchored=True))
    label = f"portable derived {forms} {family.dtype} {name}"
    with derived(forms):
        differentiated = jax.jit(with_gradients(attention_suite._mainloops(family, {})["portable"]))
        attention_suite.assert_finite(differentiated, truth, draw, label, seeds=attention_suite.SEEDS[name])
        attention_suite._held_empty(family, differentiated(*draw(0)), label)


def _rows(rows, dtype, width: int = 8):
    """A call's operand of one batch row and head, `rows` its first components (each row's rest zero)."""
    rows = np.asarray(rows, np.float64).reshape(len(rows), -1)
    return jnp.asarray(np.pad(rows, ((0, 0), (0, width - rows.shape[1])))[None, :, None, :], dtype)


def _emptied(s, q_pos, k_pos):
    """A score hook hiding every key from every query but the first."""
    del k_pos
    return jnp.where(q_pos == 0, s, -jnp.inf)


def _poisoned(value: float):
    """A score hook writing `value` (infinite or NaN) at each key past a query's own, the causal mask's."""
    def hook(s, q_pos, k_pos):
        return jnp.where(k_pos > q_pos, value, s)
    return hook


def _edges(dtype) -> dict:
    """The edges each pass found, as calls of `dtype`: `(query, key, value, dout, spans, hooks)`."""
    largest = float(jnp.finfo(dtype).max)
    big = 2.0 ** (jnp.finfo(dtype).maxexp - 1)
    spans_two = template.spans(2, 1, 1)
    empty = template.KeySpans(jnp.asarray([[0, 1]], jnp.int32), jnp.asarray([[1, 1]], jnp.int32), None, None,
                              keys=1)
    causal = template.spans(16, 16, 1, causal=True)
    rng = np.random.default_rng(0)
    drawn = [rng.standard_normal((16, 1)) for _ in range(4)]
    # Two queries over one key, the value's first component a large power of two and the second query's
    # cotangent 4 there: its `dO Vᵀ` past the dtype's largest beside a weight of 0.
    pair = (_rows([[0.0], [0.0]], dtype), _rows([[0.0]], dtype), _rows([[big]], dtype),
            _rows([[0.0], [4.0]], dtype))
    return {
        "a query a hook empties": (*pair, spans_two, template.Hooks(score_mod=_emptied)),
        "a query no span reaches": (*pair, empty, None),
        "keys past a whole tile": (_rows(drawn[0][:12], dtype), _rows(drawn[1][:12], dtype),
                                   _rows(drawn[2][:12], dtype), _rows(drawn[3][:12], dtype),
                                   template.spans(12, 12, 1), None),
        "an infinite hook where masked": (*(_rows(x, dtype) for x in drawn), causal,
                                          template.Hooks(score_mod=_poisoned(jnp.inf))),
        "a NaN hook where masked": (*(_rows(x, dtype) for x in drawn), causal,
                                    template.Hooks(score_mod=_poisoned(jnp.nan))),
        "values at the largest": (_rows(drawn[0], dtype), _rows(drawn[1], dtype),
                                  _rows([[largest]] * 16, dtype), _rows(drawn[3], dtype), causal, None),
        "values at the least": (_rows(drawn[0], dtype), _rows(drawn[1], dtype),
                                _rows([[float(jnp.finfo(dtype).smallest_subnormal)]] * 16, dtype),
                                _rows(drawn[3], dtype), causal, None),
    }


EDGES = [(name, dtype) for dtype in ("float16", "bfloat16", "float32") for name in _edges(jnp.float32)]


@pytest.mark.parametrize("forms", FORMS)
@pytest.mark.parametrize(("name", "dtype"), EDGES, ids=[f"{name}-{dtype}" for name, dtype in EDGES])
def test_the_derived_attention_is_finite_where_the_reference_is_at_each_edge(name, dtype, forms):
    """The plain-JAX mainloop's derived forms under each of its `forms` at the edge `name` in `dtype`: its
    output and gradients finite wherever the reference's are."""
    query, key, value, dout, spans, hooks = _edges(jnp.dtype(dtype))[name]
    tiles = {"block_q": 8, "block_k": 8, "pad": 8}

    def reference(q, k, v):
        return template.reference(q, k, v, spans, hooks)

    out, pull = jax.vjp(reference, query, key, value)
    theirs = (out, *pull(dout))
    with derived(forms):
        made = attention_portable.backward(query, key, value, spans, hooks, dout, **tiles)
        mine = (attention_portable.attend(query, key, value, spans, hooks, **tiles), made.query, made.key,
                made.value)
    for leaf, ours, exact in zip(("output", "dq", "dk", "dv"), mine, theirs, strict=True):
        finite = np.isfinite(np.asarray(exact, np.float32))
        assert np.isfinite(np.asarray(ours, np.float32))[finite].all(), f"{name} {dtype} {leaf}"


RECURRENCE_CASES = [pytest.param(name, regime, id=f"{name}-{regime}")
                    for name in recurrence_suite.SITES for regime in recurrence_suite.REGIMES]


@pytest.mark.parametrize(("name", "regime"), RECURRENCE_CASES)
def test_the_derived_chunk_stays_finite_and_exact_on_every_regime(name, regime):
    """The portable math's derived chunk forms (`recurrence_chunk.derived_rule`) under the template suite's
    regime: output and final memory finite wherever float64's are, and on par with the token-by-token
    recurrence's (check 1)."""
    hooks = recurrence_suite.SITES[name]
    seeds = recurrence_suite.SEEDS[name, regime]
    with derived():
        recurrence_suite.assert_finite_and_on_par(
            functools.partial(chunk_on, "derived", hooks, chunk_size=recurrence_suite.CHUNK),
            functools.partial(token_by_token, hooks),
            functools.partial(recurrence_suite.stressed, hooks, regime),
            f"the derived chunk, {name} under {regime}", seeds=seeds, finite_seeds=seeds)


DECAY_EDGES = {"no decay": [0.0] * 4, "every token vanishing": [-(2.0 ** 127)] * 4,
               "live then vanishing": [-100.0, -(2.0 ** 127), -1.0, -(2.0 ** 127)],
               "vanishing then live": [-(2.0 ** 127), -(2.0 ** 70), -1.0, -0.5]}
"""Each token's log decay at the edges the derived decays' prefix met (a whole chunk's pattern repeated)."""


@pytest.mark.parametrize("name", list(recurrence_suite.SITES))
@pytest.mark.parametrize("edge", list(DECAY_EDGES))
def test_the_derived_chunk_holds_every_order_at_the_decays_edges(edge, name):
    """The derived chunk under the decays' edge `edge`: its output and final memory, their gradient along
    the log decays and that gradient's tangent, each finite wherever the token-by-token recurrence's are."""
    hooks = recurrence_suite.SITES[name]
    (operands,) = recurrence_suite.stressed(hooks, "the decay bound", 0)
    tokens = operands.value.shape[1]
    pattern = jnp.resize(jnp.asarray(DECAY_EDGES[edge], jnp.float32), (tokens,))
    target = jnp.broadcast_to(pattern[None, :, None], operands.value.shape[:3])
    if hooks.rated:
        # A rated decay's dt, the target over its rate, scales its writes too: each head's rate scaled by a
        # power of two, which rounds no decay, so a vanishing decay's writes stay in fp32's range (at the
        # draws' rates its float64 output was near 1e39), and its values by another, so the cotangent of dt
        # times its rate (`log_gates`' pullback, before the fixture divides it back) does too (-6.2e38 at the
        # token after a vanishing one, its composed gradient 1.2e24).
        operands = dataclasses.replace(operands, rate=operands.rate * 2.0 ** 48,
                                       value=operands.value * 2.0 ** -16)
    operands = recurrence_suite._formed(operands, hooks, target)

    def orders(fn, given, decays):
        def total(at):
            out, memory = fn(recurrence_suite._formed(given, hooks, at))
            return jnp.sum(out) + jnp.sum(memory)

        value, gradient = jax.value_and_grad(total)(decays)
        _, tangent = jax.jvp(jax.grad(total), (decays,), (jnp.ones_like(decays),))
        return value, gradient, tangent

    with jax.enable_x64(new_val=True):
        exact = orders(functools.partial(token_by_token, hooks), *widened((operands, target)))
    with derived():
        mine = orders(functools.partial(chunk_on, "derived", hooks, chunk_size=recurrence_suite.CHUNK),
                      operands, target)
    for leaf, ours, theirs in zip(("value", "gradient", "tangent"), mine, exact, strict=True):
        # Finite where float64's is within the call's float32: a value past its range overflows as it must.
        with np.errstate(over="ignore"):
            finite = np.isfinite(np.asarray(theirs).astype(np.float32))
        assert np.isfinite(np.asarray(ours))[finite].all(), f"{name} {edge} {leaf}"
