"""The template's decode (`attention_decode`): a decode step's attention through the template's forward over a
built work list, each query head of a key head's group a packed row, each row's keys cut into slices, merged.

Each case is held to the reference path's attention over the same keys (check 1): a step of one query position
and a verify block of four, grouped heads, rows of their own lengths and starts, sinks and a softcap, and the
builders' inverse (`unpacked` of `packed`) and the work list's slices exactly covering each row's keys.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS
from reference_error import assert_as_exact

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template, attention_decode

KEYS = 256
SEEDS = 8
CALLS = {
    "one-query": (1, 8, 2),
    "verify-four": (4, 8, 2),
    "mha": (1, 4, 4),
}
"""Decode steps by query positions, query heads and key heads (grouped 4 over 1 and 2, and not)."""
HOOKS = ("plain", "sinks", "softcap")
LENGTHS, STARTS = (200, 37), (0, 16)
"""Two rows' keys: the first from 0 to 200, the second from 16 to 37, a window's."""


def _spans(positions: int) -> template.KeySpans:
    return template.spans(positions, KEYS, len(LENGTHS), lengths=jnp.asarray(LENGTHS, jnp.int32),
                          starts=jnp.asarray(STARTS, jnp.int32))


def _hooks(hook: str, heads: int, positions: int, sinks: jax.Array) -> template.Hooks | None:
    if hook == "sinks":
        return template.Hooks(initial=template.sinks(sinks, len(LENGTHS), positions, 64))
    if hook == "softcap":
        return template.Hooks(template.Capped(2.0))
    return None


def _drawn(positions: int, heads: int, kv_heads: int, dtype):
    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 4)
        query = jax.random.normal(streams[0], (len(LENGTHS), positions, heads, 64), dtype)
        key, value = (jax.random.normal(stream, (len(LENGTHS), KEYS, kv_heads, 64), dtype)
                      for stream in streams[1:3])
        sinks = jax.random.normal(streams[3], (heads,), jnp.float32)
        return query, key, value, sinks

    return draw


@INTERPRETS
@pytest.mark.parametrize("hook", HOOKS)
@pytest.mark.parametrize("call", CALLS)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
def test_a_decode_step_through_the_template_is_on_par_interpreted(call, hook, dtype):
    """A decode step through the template (`attention_decode.attend`) under the interpreter as an H100 runs
    it, on par with the reference path over the same keys (check 1): packed, sliced and merged."""
    positions, heads, kv_heads = CALLS[call]
    spans = _spans(positions)

    def ours(query, key, value, sinks):
        with generation.assumed("sm90"):
            return attention_decode.attend(query, key, value, spans, _hooks(hook, heads, positions, sinks),
                                           interpret=True)

    def reference(query, key, value, sinks):
        return template.reference(query, key, value, spans, _hooks(hook, heads, positions, sinks))

    assert_as_exact(jax.jit(ours), jax.jit(reference), _drawn(positions, heads, kv_heads, dtype),
                    f"template decode {call} {hook} {jnp.dtype(dtype).name}", seeds=SEEDS)


def test_packing_a_calls_heads_into_its_rows_is_undone_by_unpacking():
    """A key head's group of query heads as its rows (`packed`) and back (`unpacked`): the query as it was,
    each packed row's keys its query's."""
    query = jax.random.normal(jax.random.key(0), (2, 3, 8, 16))
    spans = _spans(3)
    rows, row_spans, _ = attention_decode.packed(query, spans, 2)
    assert rows.shape == (2, 12, 2, 16)
    np.testing.assert_array_equal(attention_decode.unpacked(rows, 8), query)
    np.testing.assert_array_equal(row_spans.start, np.repeat(np.asarray(spans.start), 4, axis=1))


@pytest.mark.parametrize("slices", [1, 3, 7])
def test_merging_a_rows_slices_and_its_initial_state_is_its_whole_softmax(slices):
    """Slices of a row's keys, each its output normalized, its maximum and its denominator, and a sink's
    initial state, merged (`merged`): the row's softmax over every key and the sink at once, a slice of no
    key adding nothing."""
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((2, 3, 1, 40)) * 4.0
    values = rng.standard_normal((2, 3, 1, 40, 8))
    sink = rng.standard_normal((2, 1, 3))
    cuts = np.linspace(0, 40, slices + 1).astype(int)
    outs, maxima, totals = [], [], []
    for lo, hi in itertools.pairwise(cuts):
        part = logits[..., lo:hi]
        peak = part.max(axis=-1) if hi > lo else np.full(part.shape[:-1], -np.inf)
        weights = np.exp(part - peak[..., None]) if hi > lo else part
        total = weights.sum(axis=-1)
        summed = np.einsum("bqhk,bqhkd->bqhd", weights, values[..., lo:hi, :])
        outs.append(summed / np.where(total > 0, total, 1)[..., None])
        maxima.append(peak)
        totals.append(total)
    out = jnp.asarray(np.stack(outs, axis=1), jnp.float32)
    maximum = jnp.asarray(np.transpose(np.stack(maxima, axis=1), (0, 3, 1, 2)), jnp.float32)
    total = jnp.asarray(np.transpose(np.stack(totals, axis=1), (0, 3, 1, 2)), jnp.float32)
    initial = template.Initial(jnp.asarray(sink, jnp.float32), jnp.ones(sink.shape, jnp.float32),
                               jnp.zeros((*sink.shape, 8), jnp.float32))
    found = attention_decode.merged(out, maximum, total, initial)
    scores = np.concatenate([logits, np.transpose(sink, (0, 2, 1))[..., None]], axis=-1)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    expected = np.einsum("bqhk,bqhkd->bqhd", weights[..., :40], values)
    np.testing.assert_allclose(found, expected, rtol=1e-5, atol=1e-6)
