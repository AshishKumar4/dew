"""Metamorphic properties of the attention template (row 3710): transformations of a call that leave its
output unchanged, or move it predictably, judged by check 1 (tests/reference_error.py) of the transformed call
against the reference path over the untransformed one, its float64 truth the reference's: never bitwise, so a
property holds of whatever form the route runs. Hypothesis draws each transformation (the suite's derandomized
profile, docs/design/testing.md): a full call's keys and values permuted together, a causal call against its
own first queries, query heads permuted with the key heads they read, and a decode step's keys cut into each
slice count its work list takes (every count a case, Hypothesis drawing the rows' lengths; a paged cache's
pages permuted with its table are `test_attention_decode_template`'s paged cells). The plain-JAX mainloop runs
the dense properties on this CPU; the decode one runs the Mosaic mainloop interpreted, as an H100 runs it."""


import hypothesis
import jax
import jax.numpy as jnp
import pytest
from generation_support import INTERPRETS
from hypothesis import strategies as st
from reference_error import assert_as_exact

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import (
    attention as template,
    attention_decode,
    attention_mosaic,
    attention_portable,
    attention_triton,
)
from dew.nn.kernels.templates.attention import Kept

SEEDS = 8
ROWS, QUERIES, KEYS, HEADS, KV_HEADS, WIDTH = 2, 48, 48, 4, 2, 32
"""A dense call's geometry: a grouped call whose queries and keys are no whole tile, so every walk pads."""

DENSE = hypothesis.settings(max_examples=30, deadline=None)
"""Each dense property's draws: the plain-JAX mainloop's check 1 a draw, about 0.5 s on armada's CPU (six
draws' tests took 3.1 s in CI), about 15 s a property."""

INTERPRETED = hypothesis.settings(max_examples=8, deadline=None)
"""Each interpreted case's draws: the Mosaic mainloop interpreted, about 2.7 s a draw (three draws' test took
7.95 s in CI), about 22 s a slice count."""


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
"""A decode step's geometry on the interpreter: rows of their own lengths over 512 keys."""


def _decode_draw(seed: int):
    streams = jax.random.split(jax.random.key(seed), 3)
    query = jax.random.normal(streams[0], (DECODE_ROWS, 1, DECODE_HEADS, 64), jnp.bfloat16)
    key, value = (jax.random.normal(stream, (DECODE_ROWS, DECODE_KEYS, DECODE_KV, 64), jnp.bfloat16)
                  for stream in streams[1:])
    return query, key, value


def _decode_spans(lengths):
    return template.spans(1, DECODE_KEYS, DECODE_ROWS, lengths=jnp.asarray(lengths, jnp.int32))


@INTERPRETS
def test_a_far_short_rows_slices_merge_through_its_masked_rewalk_interpreted():
    """A decode step cut into 8 slices a row, its first row's query 2^16 times a drawn one (past the fold's
    reach, so the masked rewalk walks it) and 70 keys long, two key tiles: its six slices of no key store
    their empty walks' start in the rewalk for the merge to read (row 3778). On par with the step walked
    whole (check 1)."""
    spans = _decode_spans((70, DECODE_KEYS))

    def far(seed: int):
        query, key, value = _decode_draw(seed)
        return query.at[0].multiply(2.0 ** 16), key, value

    def sliced(query, key, value):
        with generation.assumed("sm90"):
            rows, row_spans, _ = attention_decode.packed(query, spans, DECODE_KV)
            config = attention_decode.plan(rows, key, value, row_spans, None, scale=None,
                                           target=generation.gpu_target()).config
            with tuning.pinned({attention_mosaic.MosaicForward: config}):
                out, maximum, total = attention_mosaic.forward(
                    rows, key, value, row_spans, template.Hooks(), scale=None, natural=False,
                    residuals=Kept.OUTPUT, interpret=True, out_dtype=jnp.float32, masked=True, splits=8)[:3]
            spec = attention_decode._spec(rows, row_spans, template.Hooks(), scale=None)
            merged = attention_decode.merged(spec, out, maximum, total)
            return attention_decode.unpacked(merged, DECODE_HEADS).astype(query.dtype)

    assert_as_exact(jax.jit(sliced), jax.jit(lambda q, k, v: template.reference(q, k, v, spans)), far,
                    "a far short row's slices through its masked rewalk", seeds=SEEDS)


@INTERPRETS
@INTERPRETED
@pytest.mark.parametrize("slices", [1, 2, 4, 8])
@hypothesis.given(lengths=st.tuples(st.integers(1, DECODE_KEYS), st.integers(1, DECODE_KEYS)))
def test_a_decode_steps_keys_cut_into_any_slice_count_merge_to_its_attention_interpreted(slices, lengths):
    """A decode step's keys cut into `slices` slices a row (`attention_mosaic.forward`'s `splits` at the
    decode plan's config, a program a slice), the slices merged (`attention_decode.merged`), rows of their own
    lengths: on par with the step walked whole."""
    spans = _decode_spans(lengths)

    def sliced(query, key, value):
        with generation.assumed("sm90"):
            rows, row_spans, _ = attention_decode.packed(query, spans, DECODE_KV)
            config = attention_decode.plan(rows, key, value, row_spans, None, scale=None,
                                           target=generation.gpu_target()).config
            with tuning.pinned({attention_mosaic.MosaicForward: config}):
                out, maximum, total = attention_mosaic.forward(
                    rows, key, value, row_spans, template.Hooks(), scale=None, natural=False,
                    residuals=Kept.OUTPUT, interpret=True, out_dtype=jnp.float32, masked=True,
                    splits=slices)[:3]
            if slices == 1:
                # One slice's launch gives its rows without a slices axis.
                out, maximum, total = out[:, None], maximum[:, :, None], total[:, :, None]
            spec = attention_decode._spec(rows, row_spans, template.Hooks(), scale=None)
            merged = attention_decode.merged(spec, out, maximum, total)
            return attention_decode.unpacked(merged, DECODE_HEADS).astype(query.dtype)

    assert_as_exact(jax.jit(sliced), jax.jit(lambda q, k, v: template.reference(q, k, v, spans)),
                    _decode_draw, f"decode step cut into {slices} slices", seeds=SEEDS)

