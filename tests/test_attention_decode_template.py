"""The template's decode (`attention_decode`): a decode step's attention through the template's forward over a
built work list, each query head of a key head's group a packed row, each row's keys cut into slices, merged.

Each case is held to the reference path's attention over the same keys (check 1): a step of one query position
and a verify block of four, grouped heads, rows of their own lengths and starts, sinks and a softcap, and the
builders' inverse (`unpacked` of `packed`) and the work list's slices exactly covering each row's keys.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS
from reference_error import assert_as_exact

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template, attention_decode, attention_mosaic

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
def test_a_work_lists_slices_cover_each_rows_keys_once(slices):
    """Each row's slices (`sliced`) are contiguous and cover its keys exactly once, every inner boundary a key
    tile's, the padding rows seeing none."""
    spans = _spans(1)
    query = jnp.zeros((2, 1, 1, 16))
    plan = attention_decode.Plan(attention_mosaic.Tiles(64, 64, 2, 1), rows=4, slices=slices)
    _, item_spans, _ = attention_decode.sliced(query, spans, None, plan)
    start = np.asarray(item_spans.start).reshape(2, slices, 4)
    end = np.asarray(item_spans.end).reshape(2, slices, 4)
    for row, (lo, hi) in enumerate(zip(STARTS, LENGTHS, strict=True)):
        bounds = [(start[row, s, 0], end[row, s, 0]) for s in range(slices)]
        assert bounds[0][0] == lo and bounds[-1][1] == hi
        assert all(bounds[s][1] == bounds[s + 1][0] for s in range(slices - 1))
        assert all(b % 64 == 0 for _, b in bounds[:-1] if lo < b < hi)
        assert (start[row, :, 1:] == end[row, :, 1:]).all()

