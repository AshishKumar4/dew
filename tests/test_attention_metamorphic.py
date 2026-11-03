"""Metamorphic properties of the attention template (row 3710): transformations of a call that leave its
output unchanged, or move it predictably, judged by check 1 (tests/reference_error.py) of the transformed call
against the reference path over the untransformed one, its float64 truth the reference's: never bitwise, so a
property holds of whatever form the route runs. Hypothesis draws each transformation (the suite's derandomized
profile, docs/design/testing.md): a full call's keys and values permuted together, a causal call against its
own first queries, query heads permuted with the key heads they read, a decode step's keys cut into every
slice count its work list takes, and a paged cache's pages permuted with its table. The plain-JAX mainloop
runs the dense properties on this CPU; the decode and paged ones run the Mosaic mainloop interpreted, as an
H100 runs it."""

import hypothesis
import jax
import jax.numpy as jnp
import numpy as np
from generation_support import INTERPRETS
from hypothesis import strategies as st
from reference_error import assert_as_exact

from dew.nn.kernels import generation
from dew.nn.kernels.templates import (
    attention as template,
    attention_decode,
    attention_portable,
    attention_triton,
)

SEEDS = 8
ROWS, QUERIES, KEYS, HEADS, KV_HEADS, WIDTH = 2, 48, 48, 4, 2, 32
"""A dense call's geometry: a grouped call whose queries and keys are no whole tile, so every walk pads."""

DENSE = hypothesis.settings(max_examples=6, deadline=None)
"""Each dense property's draws: the plain-JAX mainloop's check 1 a draw, about 3 s on armada's CPU."""

INTERPRETED = hypothesis.settings(max_examples=3, deadline=None)
"""Each interpreted property's draws: the Mosaic mainloop interpreted, about 15 s a draw."""


def _dense_draw(seed: int, queries: int = QUERIES):
    streams = jax.random.split(jax.random.key(seed), 3)
    query = jax.random.normal(streams[0], (ROWS, queries, HEADS, WIDTH), jnp.bfloat16)
    key, value = (jax.random.normal(stream, (ROWS, KEYS, KV_HEADS, WIDTH), jnp.bfloat16)
                  for stream in streams[1:])
    return query, key, value


def _portable(query, key, value, spans):
    # The plain-JAX mainloop at 32-query and 32-key tiles, padded as the route pads a call.
    return attention_portable.attend(query, key, value, spans, block_q=32, block_k=32,
                                     pad=attention_triton.BLOCK)


@DENSE
@hypothesis.given(order=st.permutations(range(KEYS)))
def test_a_full_calls_output_is_its_keys_permuted_together_with_their_values(order):
    """A call that sees every key: permuting its keys and values together leaves each query's output."""
    order = jnp.asarray(order, jnp.int32)
    spans = template.spans(QUERIES, KEYS, ROWS)

    def permuted(query, key, value):
        return _portable(query, key[:, order], value[:, order], spans)

    assert_as_exact(jax.jit(permuted), jax.jit(lambda q, k, v: template.reference(q, k, v, spans)),
                    _dense_draw, "full call, keys permuted", seeds=SEEDS)


@DENSE
@hypothesis.given(prefix=st.integers(1, QUERIES - 1))
def test_a_causal_calls_first_queries_are_the_call_cut_to_them(prefix):
    """A causal call's first `prefix` queries see only the first `prefix` keys: the whole call's outputs there
    are the call cut to those queries and keys."""
    spans = template.spans(QUERIES, KEYS, ROWS, causal=True)
    cut = template.spans(prefix, prefix, ROWS, causal=True)

    def whole(query, key, value):
        return _portable(query, key, value, spans)[:, :prefix]

    def reference(query, key, value):
        return template.reference(query[:, :prefix], key[:, :prefix], value[:, :prefix], cut)

    assert_as_exact(jax.jit(whole), jax.jit(reference), _dense_draw, "causal call's first queries",
                    seeds=SEEDS)


@DENSE
@hypothesis.given(heads=st.permutations(range(KV_HEADS)), within=st.permutations(range(HEADS // KV_HEADS)))
def test_a_calls_heads_permuted_with_the_key_heads_they_read_permute_its_output(heads, within):
    """Query heads permuted with the key heads their group reads (and within each group): each query head's
    output moves with it."""
    group = HEADS // KV_HEADS
    order = jnp.asarray([kv * group + member for kv in heads for member in within], jnp.int32)
    kv_order = jnp.asarray(heads, jnp.int32)
    spans = template.spans(QUERIES, KEYS, ROWS, causal=True)
    back = jnp.argsort(order)

    def permuted(query, key, value):
        out = _portable(query[:, :, order], key[:, :, kv_order], value[:, :, kv_order], spans)
        return out[:, :, back]

    assert_as_exact(jax.jit(permuted), jax.jit(lambda q, k, v: template.reference(q, k, v, spans)),
                    _dense_draw, "heads permuted with their key heads", seeds=SEEDS)


DECODE_ROWS, DECODE_KEYS, DECODE_HEADS, DECODE_KV = 2, 512, 8, 2
PAGE = 64
"""A decode step's geometry on the interpreter: rows of their own lengths over 512 keys, pages of 64."""


def _decode_draw(seed: int):
    streams = jax.random.split(jax.random.key(seed), 3)
    query = jax.random.normal(streams[0], (DECODE_ROWS, 1, DECODE_HEADS, 64), jnp.bfloat16)
    key, value = (jax.random.normal(stream, (DECODE_ROWS, DECODE_KEYS, DECODE_KV, 64), jnp.bfloat16)
                  for stream in streams[1:])
    return query, key, value


def _decode_spans(lengths):
    return template.spans(1, DECODE_KEYS, DECODE_ROWS, lengths=jnp.asarray(lengths, jnp.int32))


@INTERPRETS
@INTERPRETED
@hypothesis.given(slices=st.sampled_from((1, 2, 4, 8)),
                  lengths=st.tuples(st.integers(1, DECODE_KEYS), st.integers(1, DECODE_KEYS)))
def test_a_decode_steps_keys_cut_into_any_slice_count_merge_to_its_attention_interpreted(slices, lengths):
    """A decode step's keys cut into `slices` slices a row, each walked by a program of its own and merged
    (`attention_decode`'s work list at a pinned slice count, rows of their own lengths): on par with the
    step walked whole."""
    spans = _decode_spans(lengths)
    derived = attention_decode.plan

    def pinned(*args, **kwargs):
        work = derived(*args, **kwargs)
        return attention_decode.Plan(work.config, slices)

    def sliced(query, key, value):
        attention_decode.plan = pinned
        try:
            with generation.assumed("sm90"):
                return attention_decode.attend(query, key, value, spans, interpret=True)
        finally:
            attention_decode.plan = derived

    assert_as_exact(jax.jit(sliced), jax.jit(lambda q, k, v: template.reference(q, k, v, spans)),
                    _decode_draw, f"decode step cut into {slices} slices", seeds=SEEDS)


@INTERPRETS
@INTERPRETED
@hypothesis.given(order=st.permutations(range(2 * DECODE_KEYS // PAGE)))
def test_a_paged_caches_pages_permuted_with_its_table_attend_as_before_interpreted(order):
    """A paged cache's pages permuted in its pool, its table mapping each row's slots to where they moved:
    each row reads the same keys, and its step is on par with the dense cache's."""
    spans = _decode_spans((DECODE_KEYS, DECODE_KEYS - 37))
    per_row = DECODE_KEYS // PAGE
    moved = np.asarray(order)
    # Page p of the row-major pool lands at moved[p]; the table names each row's pages where they now are.
    table = jnp.asarray(moved.reshape(DECODE_ROWS, per_row), jnp.int32)
    inverse = jnp.asarray(np.argsort(moved), jnp.int32)

    def paged(query, key, value):
        pools = [x.reshape(DECODE_ROWS * per_row, PAGE, DECODE_KV, 64).transpose(2, 0, 1, 3)[:, inverse]
                 for x in (key, value)]
        with generation.assumed("sm90"):
            return attention_decode.attend(query, *pools, spans, interpret=True, table=table)

    assert_as_exact(jax.jit(paged), jax.jit(lambda q, k, v: template.reference(q, k, v, spans)),
                    _decode_draw, "paged cache, pages permuted", seeds=SEEDS)
