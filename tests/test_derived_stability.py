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
- the recurrence's: the decays' edges both ways (none, every token vanishing, a live decay then a vanishing
  one, the reverse), each through its value, its gradient and its gradient's tangent (every order the derived
  forms hold); the template suite's every site and regime is the portable route's own suite's
  (`test_recurrence_stability`, the plain-JAX mainloop's).

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
from reference_error import widened
from test_recurrence_chunk import chunk_on, token_by_token

from dew.nn.kernels import generation
from dew.nn.kernels.templates import (
    attention as template,
    attention_mosaic,
    attention_portable,
    attention_route,
    attention_triton,
)


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
"""Each template family as drawn (None) and in fp16, the dtype the derived forward walks (`Family.retyped`:
at fp16's own extremes where the family's values are a dtype's)."""


@pytest.mark.parametrize("forms", FORMS)
@pytest.mark.parametrize(("name", "dtype"), ATTENTION_FAMILIES,
                         ids=[f"{name}-{dtype or 'as drawn'}" for name, dtype in ATTENTION_FAMILIES])
def test_the_derived_attention_is_finite_where_the_truth_is_on_every_family(name, dtype, forms):
    """The plain-JAX mainloop's derived forms under each of its `forms`, forward and backward, finite
    wherever the float64 truth rounded to their dtype is (`assert_finite`) over the family's draws (an
    anchor's, one key's and a masked query's among them), a row that sees nothing zero."""
    family = attention_suite.FAMILIES[name]
    family = family if dtype is None else family.retyped(dtype)
    draw = attention_suite._cast(family, family.draw)
    truth = attention_suite._cached_reference_fn(*attention_suite._reference_key(family), anchored=True)
    label = f"portable derived {forms} {family.dtype} {name}"
    with derived(forms):
        natural, accurate = attention_route.run_forms()
        differentiated = attention_suite._jitted_mainloop(
            "portable", attention_suite._spans_key(family.spans),
            (("accurate", accurate), ("natural", natural)), dropping=False)
        attention_suite.assert_finite(differentiated, truth, draw, label, seeds=attention_suite.SEEDS[name],
                                      empty=family.empty)

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


@functools.cache
def _edge_reference(name: str):
    """Cached jitted reference forward and VJP for edge `name` (row 12050)."""
    _, _, _, _, spans, hooks = _edges(jnp.float32)[name]

    @jax.jit
    def run(query, key, value, dout):
        out, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans, hooks), query, key, value)
        return (out, *pull(dout))

    return run


@functools.cache
def _edge_derived(name: str, forms: str):
    """Cached jitted derived portable forward and backward for `(name, forms)` (row 12050)."""
    _, _, _, _, spans, hooks = _edges(jnp.float32)[name]
    tiles = attention_portable.Tiles(block_q=8, block_k=8, pad=8)

    @jax.jit
    def run(query, key, value, dout):
        natural, accurate = attention_route.run_forms()
        made = attention_portable.backward(tiles, query, key, value, spans, hooks, dout,
                                           natural=natural, accurate=accurate)
        out = attention_portable.xla_mainloop(tiles, query, key, value, spans, hooks, natural=natural)
        return (out, made.query, made.key, made.value)

    return run

@pytest.mark.parametrize("forms", FORMS)
@pytest.mark.parametrize(("name", "dtype"), EDGES, ids=[f"{name}-{dtype}" for name, dtype in EDGES])
def test_the_derived_attention_is_finite_where_the_reference_is_at_each_edge(name, dtype, forms):
    """The plain-JAX mainloop's derived forms under each of its `forms` at the edge `name` in `dtype`: its
    output and gradients finite wherever the reference's are."""
    query, key, value, dout, _, _ = _edges(jnp.dtype(dtype))[name]
    theirs = _edge_reference(name)(query, key, value, dout)
    with derived(forms):
        mine = _edge_derived(name, forms)(query, key, value, dout)
    for leaf, ours, exact in zip(("output", "dq", "dk", "dv"), mine, theirs, strict=True):
        finite = np.isfinite(np.asarray(exact, np.float32))
        assert np.isfinite(np.asarray(ours, np.float32))[finite].all(), f"{name} {dtype} {leaf}"

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
        mine = orders(functools.partial(chunk_on, "xla", hooks, chunk_size=recurrence_suite.CHUNK),
                      operands, target)
    for leaf, ours, theirs in zip(("value", "gradient", "tangent"), mine, exact, strict=True):
        # Finite where float64's is within the call's float32: a value past its range overflows as it must.
        with np.errstate(over="ignore"):
            finite = np.isfinite(np.asarray(theirs).astype(np.float32))
        assert np.isfinite(np.asarray(ours))[finite].all(), f"{name} {edge} {leaf}"


WIDTH = 64
"""The witnesses' head width, every mainloop's 16-bit one: each value and cotangent in its first channel."""


def _channel(rows: list[list[float]]) -> jax.Array:
    """`[B, N, 1, WIDTH]` bf16 of each batch row's `N` entries in the first channel, zeros past it."""
    first = jnp.asarray(rows, jnp.float32)[:, :, None, None]
    return jnp.pad(first, ((0, 0), (0, 0), (0, 0), (0, WIDTH - 1))).astype(jnp.bfloat16)


def _held(rows: list) -> np.ndarray:
    """`rows` as float32, an entry None (one no truth holds: an `inf` cotangent's own) NaN, unchecked."""
    return np.asarray([[np.nan if entry is None else entry for entry in row] for row in rows], np.float32)


def _witness(keys: list[float], values: list[list[float]], douts: list[list[float]], bias: list[list[float]],
             gradients: dict[str, list], *, shared: bool, bounds: dict | None = None) -> tuple:
    """A call of queries of zeros over keys whose first channel is `keys` (every row's), its `values` and
    output cotangents `douts` each row's, each row's keys `bounds`' (`template.spans`' `lengths` and
    `starts`) or all, under a tile bias `[B or 1, 1, Q, K]` (`bias` its queries' rows, one table for every
    row where `shared`), and its exact gradients at a logit scale of 1, each leaf's first channel
    (`gradients`, the bias's whole; `_held`'s None unchecked, in every channel), every other channel's 0."""
    batch, queries, count = len(values), len(douts[0]), len(keys)
    query = jnp.zeros((batch, queries, 1, WIDTH), jnp.bfloat16)
    key = jnp.broadcast_to(_channel([keys]), (batch, count, 1, WIDTH))
    table = jnp.asarray(bias, jnp.float32)[None, None]
    hooks = template.Hooks(lambda s, q_pos, k_pos, bias: s + bias.astype(s.dtype), (template.Param(
        table if shared else jnp.broadcast_to(table, (batch, 1, queries, count)), template.Kind.TILE),))
    # An unchecked entry's every channel unchecked: an inf cotangent's NaN reaches them all.
    expected = {name: np.where(np.isnan(first), np.nan, np.pad(first, ((0, 0),) * 3 + ((0, WIDTH - 1),)))
                for name, rows in gradients.items() if name != "bias"
                for first in (_held(rows)[:, :, None, None],)}
    expected["bias"] = _held(gradients["bias"]).reshape(hooks.score_params[0].value.shape)
    spans = template.spans(queries, count, batch, **{name: jnp.asarray(bound, jnp.int32)
                                                      for name, bound in (bounds or {}).items()})
    return (query, key, _channel(values), _channel(douts), hooks, spans), expected


def _witnesses(name: str) -> list[tuple]:
    """`name`'s calls (`WITNESSES`), each with and without a bias the rows share where both are finite."""
    sharing = (False, True)
    match name:
        case "a row beside one at fp32's largest":
            # Row A's values at fp32's largest take its values beneath a shift (`attention.value_shift`),
            # row B's `[0, 2^-120]` none: its weights 1/2, `dS = [-2^5, 2^5]`, its dQ `2^5` (keys `[0, 1]`),
            # where one shift for the call took its 2^-120 beneath 2^-248 and gave 0.
            return [_witness([0.0, 1.0], [[2.0 ** 127, 2.0 ** 127], [0.0, 2.0 ** -120]],
                             [[2.0 ** 127], [2.0 ** 127]], [[0.0, 0.0]],
                             {"query": [[0.0], [32.0]], "key": [[0.0, 0.0]] * 2,
                              "value": [[2.0 ** 126, 2.0 ** 126]] * 2,
                              "bias": [[-32.0, 32.0]] if shared else [[0.0, 0.0], [-32.0, 32.0]]},
                             shared=shared) for shared in sharing]
        case "rows at their own scales":
            # Row A's `dO V` past fp32's largest by one binade (a shift of 1), row B's within it (none):
            # each row's bias gradient `[-2^119, 2^119]` at its own scale, the shared one their sum.
            return [_witness([0.0, 0.0], [[2.0 ** 126, 2.0 ** 126 + 2.0 ** 120], [0.0, 1.0]],
                             [[2.0], [2.0 ** 121]], [[0.0, 0.0]],
                             {"query": [[0.0]] * 2, "key": [[0.0, 0.0]] * 2,
                              "value": [[1.0, 1.0], [2.0 ** 120, 2.0 ** 120]],
                              "bias": [[-(2.0 ** 120), 2.0 ** 120]] if shared
                                      else [[-(2.0 ** 119), 2.0 ** 119]] * 2},
                             shared=shared) for shared in sharing]
        case "rows past fp32 that cancel":
            # Each row's bias gradient `±2^252` past fp32's largest, opposite: the shared one's sum 0,
            # where each row's raised first was `±inf` and their sum NaN.
            return [_witness([0.0, 0.0], [[0.0, 2.0 ** 127], [0.0, -(2.0 ** 127)]],
                             [[2.0 ** 127], [2.0 ** 127]], [[0.0, 0.0]],
                             {"query": [[0.0]] * 2, "key": [[0.0, 0.0]] * 2,
                              "value": [[2.0 ** 126, 2.0 ** 126]] * 2, "bias": [[0.0, 0.0]]}, shared=True)]
        case "an empty query of cotangent inf":
            # A query the bias empties, its cotangent inf, beside a visited one of cotangent 0, values at
            # fp32's largest: the launch's cotangent there 0 (`attention.launch_cotangent`) before its
            # shift, so every gradient 0, where its inf took the row beneath a shift of 257 and its
            # restoring factors inf.
            return [_witness([0.0, 0.0], [[2.0 ** 127, 2.0 ** 127]] * 2, [[0.0, np.inf]] * 2,
                             [[0.0, 0.0], [-np.inf, -np.inf]],
                             {"query": [[0.0, 0.0]] * 2, "key": [[0.0, 0.0]] * 2,
                              "value": [[0.0, 0.0]] * 2, "bias": [[0.0, 0.0]] * 2}, shared=True)]
        case "rows that cancel to a subnormal":
            # Each row's bias gradient normal and unshifted, `±2^-125` and `±(2^-125 + 2^-132)`, opposite:
            # the shared one's sum `±2^-132`, a subnormal, which fp32's sum flushed on XLA's CPU.
            return [_witness([0.0, 0.0], [[0.0, 2.0 ** -123], [2.0 ** -123 + 2.0 ** -130, 0.0]],
                             [[1.0], [1.0]], [[0.0, 0.0]],
                             {"query": [[0.0]] * 2, "key": [[0.0, 0.0]] * 2, "value": [[0.5, 0.5]] * 2,
                              "bias": [[2.0 ** -132, -(2.0 ** -132)]] if shared
                                      else [[-(2.0 ** -125), 2.0 ** -125],
                                            [2.0 ** -125 + 2.0 ** -132, -(2.0 ** -125 + 2.0 ** -132)]]},
                             shared=shared) for shared in sharing]
        case "keys beside an inf cotangent's":
            # Row A sees key 0 alone under a cotangent of inf (a shift of 257), row B keys 1 and 2 (`[0, 1]`
            # under 1, its bias gradient there `[-1/4, 1/4]`): row A's 0 at the keys it does not see raised
            # is 0, so the shared bias's sum there is row B's, where `0 * 2^128 * 2^129` was NaN. Row A's own
            # gradients, an inf cotangent's, are checked by none.
            return [_witness([0.0, 0.0, 0.0], [[2.0 ** 127, 0.0, 0.0], [0.0, 0.0, 1.0]], [[np.inf], [1.0]],
                             [[0.0, 0.0, 0.0]],
                             {"query": [[None], [0.0]], "key": [[None] * 3, [0.0] * 3],
                              "value": [[None] * 3, [0.0, 0.5, 0.5]], "bias": [[None, -0.25, 0.25]]},
                             shared=True, bounds={"lengths": [1, 3], "starts": [0, 1]})]
    raise KeyError(name)


WITNESSES = ["a row beside one at fp32's largest", "rows at their own scales", "rows past fp32 that cancel",
             "an empty query of cotangent inf", "rows that cancel to a subnormal",
             "keys beside an inf cotangent's"]
"""Calls whose rows' values the backward takes beneath shifts of their own (`attention.value_shift`), their
gradients exact (`_witnesses`): a shared bias's summed across its rows at each one's scale
(`attention.hook_gradient`)."""


def _mainloops(query, key, value, dout, hooks, spans) -> dict[str, template.Gradients]:
    """The call's gradients at a logit scale of 1 on each mainloop: the plain-JAX one's, and the
    Pallas-Triton one's and the Mosaic one's as an H100's, interpreted."""
    natural, accurate = attention_route.run_forms()
    form = {"scale": 1.0, "natural": natural, "accurate": accurate}
    made = {"plain-JAX": attention_portable.backward(attention_portable.Tiles(block_q=8, block_k=8, pad=8),
                                                     query, key, value, spans, hooks, dout, **form)}
    with generation.assumed("sm90", interpreted=True):
        made["Pallas-Triton"] = attention_triton.backward(query, key, value, spans, hooks, dout,
                                                          interpret=True, **form)
        residuals = attention_mosaic.forward(query, key, value, spans, hooks, scale=1.0, natural=natural,
                                             residuals=template.Kept.SPLIT, interpret=True)[1:]
        made["Mosaic"] = attention_mosaic.backward(query, key, value, spans, hooks, dout, residuals=residuals,
                                                   interpret=True, **form)
    return made


@pytest.mark.parametrize("witness", WITNESSES)
def test_a_rows_gradients_are_at_its_own_shift_and_a_shared_bias_sums_them(witness):
    """Each witness's every gradient its exact one on every mainloop (`WITNESSES`): a row's own taken at its
    own shift, whatever the rows share, and a bias the rows share summed across them at each one's, rounded
    once (`attention.hook_gradient`)."""
    for call, expected in _witnesses(witness):
        for mainloop, made in _mainloops(*call).items():
            for name, want in expected.items():
                got = np.asarray(made.score_params[0] if name == "bias" else getattr(made, name), np.float32)
                held = ~np.isnan(want)
                np.testing.assert_array_equal(got[held], want[held], err_msg=f"{witness} {mainloop} {name}")


def test_a_shared_bias_summed_a_slab_at_a_time_is_its_exact_sum(monkeypatch):
    """A tile bias's gradient summed across 4 rows (`template.raised_sum`) a slab of its columns at a time
    where its digits would hold more entries than `RAISED_ELEMENTS` (row 10694: 5.47 GiB at once on an
    A100), each column still its exact sum rounded once: rows of `±2^100` under shifts of 150 that cancel to
    0 beside a 2^-149 part kept, and a row's own scale per column, the same bits as the whole sum's."""
    rng = np.random.default_rng(10694)
    grad = rng.standard_normal((4, 2, 3, 64)).astype(np.float32)
    grad[:2, :, :, :8] = np.float32(2.0 ** 100) * np.array([1, -1], np.float32)[:, None, None, None]
    grad[2:, :, :, :8] = np.float32(2.0 ** -149) * np.array([1, 0], np.float32)[:, None, None, None]
    shift = jnp.asarray(np.array([150, 150, 0, 3], np.int32)).reshape(4, 1, 1, 1)
    whole = np.asarray(template.raised_sum(jnp.asarray(grad), shift, (0,)))
    monkeypatch.setattr(template, "RAISED_ELEMENTS", 2 * 3 * 4 * 16)
    slabbed = np.asarray(template.raised_sum(jnp.asarray(grad), shift, (0,)))
    assert slabbed.shape == whole.shape == (1, 2, 3, 64)
    np.testing.assert_array_equal(slabbed.view(np.uint32), whole.view(np.uint32))
    truth = sum(grad[row].astype(np.float64) * 2.0 ** int(np.asarray(shift).ravel()[row]) for row in range(4))
    np.testing.assert_array_equal(slabbed[0, :, :, :8], np.float32(truth[:, :, :8]))
    # A negative axis the sum's own: rows of 64 summed whole, never slabbed along themselves (at a cap of 64
    # entries a slab, two slabs of the kept axis).
    monkeypatch.setattr(template, "RAISED_ELEMENTS", 64)
    ones = jnp.ones((2, 64), jnp.float32)
    np.testing.assert_array_equal(np.asarray(template.raised_sum(ones, jnp.zeros((2, 1), jnp.int32), (-1,))),
                                  np.full((2, 1), 64.0, np.float32))


def test_the_references_tangent_keeps_a_part_beneath_values_that_cancel():
    """One fp32 query over four keys of zero logits, values `2^127` in one channel (their sums past the
    format): the reference's tangent along the values `[2^127, -2^127, 2^-30, 0]` moves the mean by `2^-32`,
    its mean's contraction a band at a time (`scaled_contraction._banded`), where one exponent for the values
    flushed `2^-30` before the large ones cancelled and gave 0 (row 12452)."""
    query = jnp.zeros((1, 1, 1, 1), jnp.float32)
    key = jnp.zeros((1, 4, 1, 1), jnp.float32)
    spans = template.spans(1, 4, 1)
    direction = jnp.asarray([2.0 ** 127, -(2.0 ** 127), 2.0 ** -30, 0.0], jnp.float32).reshape(1, 4, 1, 1)
    _, tangent = jax.jvp(lambda value: template.reference(query, key, value, spans),
                         (jnp.full((1, 4, 1, 1), 2.0 ** 127, jnp.float32),), (direction,))
    np.testing.assert_array_equal(np.asarray(tangent).ravel(), [2.0 ** -32])

