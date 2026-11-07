"""Metamorphic properties of the attention template (row 3710): transformations of a call that leave its
output unchanged, or move it predictably, judged by check 1 (tests/reference_error.py) of the transformed call
against the reference path over the untransformed one, its float64 truth the reference's: never bitwise, so a
property holds of whatever form the route runs. Hypothesis draws each transformation (the suite's derandomized
profile, docs/design/testing.md): a full call's keys and values permuted together, a causal call against its
own first queries, query heads permuted with the key heads they read, and a decode step's keys cut into each
slice count its work list takes (every count a case, each draw's rows of lengths of their own, both ends
among them; a paged cache's pages permuted with its table are `test_attention_decode_template`'s paged
cells). The plain-JAX mainloop runs
the dense properties on this CPU; the decode one runs the Mosaic mainloop interpreted, as an H100 runs it."""

import functools

import hypothesis
import jax
import jax.numpy as jnp
import pytest
from generation_support import INTERPRETS
from hypothesis import strategies as st
from reference_error import as_given, assert_as_exact

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import (
    attention as template,
    attention_algebra as algebra,
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

def _dense_draw(seed: int, queries: int = QUERIES):
    streams = jax.random.split(jax.random.key(seed), 3)
    query = jax.random.normal(streams[0], (ROWS, queries, HEADS, WIDTH), jnp.bfloat16)
    key, value = (jax.random.normal(stream, (ROWS, KEYS, KV_HEADS, WIDTH), jnp.bfloat16)
                  for stream in streams[1:])
    return query, key, value


def _portable(query, key, value, spans):
    # The plain-JAX mainloop at 32-query and 32-key tiles, padded as the route pads a call.
    return attention_portable.xla_mainloop(attention_portable.Tiles(32, 32, attention_triton.BLOCK), query,
                                           key, value, spans)


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


def _slices_merged(query, key, value, spans, *, merges: bool):
    """A folded decode step (`natural=False`) cut into 8 slices a row at the decode plan's config, merged in
    the launch (`merges`, `MosaicForward.merges`) or after it (`algebra.merged_slices`), interpreted."""
    with generation.assumed("sm90"):
        rows, row_spans, _ = attention_decode.packed(query, spans, DECODE_KV)
        config = attention_decode.plan(rows, key, value, row_spans, None, scale=None,
                                       target=generation.gpu_target()).config
        with tuning.pinned({attention_mosaic.MosaicForward: config}):
            if merges:
                merged = attention_mosaic.forward(
                    rows, key, value, row_spans, template.Hooks(), scale=None, natural=False,
                    interpret=True, out_dtype=jnp.float32, masked=True, splits=8, merges=True)
            else:
                out, maximum, total = attention_mosaic.forward(
                    rows, key, value, row_spans, template.Hooks(), scale=None, natural=False,
                    residuals=Kept.OUTPUT, interpret=True, out_dtype=jnp.float32, masked=True, splits=8)[:3]
                spec = attention_decode._spec(rows, row_spans, template.Hooks(), scale=None)
                merged = algebra.merged_slices(spec, out, maximum, total)
        return template.ungrouped_rows(merged, DECODE_HEADS).astype(query.dtype)


@INTERPRETS
@pytest.mark.parametrize("merges", [False, True], ids=["after", "in-launch"])
def test_a_far_short_rows_slices_merge_through_its_masked_rewalk_interpreted(merges):
    """A decode step cut into 8 slices a row, its first row's query 2^16 times a drawn one (past the fold's
    reach, so the masked rewalk walks it) and 70 keys long, two key tiles: its six slices of no key store
    their empty walks' start in the rewalk for the merge to read (row 3778), the slices merged after the
    launch or in it, each in the units of its finished rows (`algebra.finished_form`). On par with the step
    walked whole (check 1)."""
    spans = _decode_spans((70, DECODE_KEYS))

    def far(seed: int):
        query, key, value = _decode_draw(seed)
        return query.at[0].multiply(2.0 ** 16), key, value

    assert_as_exact(jax.jit(functools.partial(_slices_merged, spans=spans, merges=merges)),
                    jax.jit(lambda q, k, v: template.reference(q, k, v, spans)), far,
                    "a far short row's slices through its masked rewalk", seeds=SEEDS)


@INTERPRETS
def test_a_decode_steps_slices_merged_in_the_launch_rewalk_rows_whose_sums_pass_fp32s_largest_interpreted():
    """A decode step cut into 8 slices a row and merged in the launch (`MosaicForward.merges`), its values
    near bf16's largest under equal logits, so its rows' weighted values pass fp32's largest and its guarded
    rewalk walks each row's keys whole, merging nothing (`_inference_walk`): on par with jax's own attention,
    which normalizes its weights before their product (check 1)."""
    lengths = (300, DECODE_KEYS)
    spans = _decode_spans(lengths)

    def large(seed: int):
        query, key, _ = _decode_draw(seed)
        value = jax.random.uniform(jax.random.key(seed + 1), (DECODE_ROWS, DECODE_KEYS, DECODE_KV, 64),
                                   jnp.float32, 2.9e38, 3.3e38).astype(jnp.bfloat16)
        return jnp.zeros_like(query), key, value

    def reference(query, key, value):
        return jax.nn.dot_product_attention(query, key, value, implementation="xla",
                                            key_value_seq_lengths=jnp.asarray(lengths, jnp.int32))

    assert_as_exact(jax.jit(functools.partial(_slices_merged, spans=spans, merges=True)),
                    as_given(jax.jit(reference)), large,
                    "a decode step's slices merged in the launch past fp32's largest", seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("merges", [False, True], ids=["after", "in-launch"])
def test_slices_whose_outputs_sum_past_fp32s_largest_merge_to_their_attention_interpreted(merges):
    """A decode step cut into 8 slices a row, each slice's 64 keys one whose logit is 30 and whose values near
    bf16's largest: no slice's sums pass fp32's largest, but their outputs, each near 3e38, sum past it, so
    their merge weighs each output by its share of the masses before its product (row 4526,
    `algebra.finished_merged`), after the launch or in it: on par with jax's own attention, which normalizes
    its weights before their product (check 1)."""
    lengths = (DECODE_KEYS, DECODE_KEYS)
    spans = _decode_spans(lengths)
    dominant = jnp.arange(DECODE_KEYS) % 64 == 0

    def outputs_near_the_largest(seed: int):
        query, key, value = _decode_draw(seed)
        query = jnp.zeros_like(query).at[..., 0].set(1.0)
        key = jnp.where(dominant[None, :, None, None] & (jnp.arange(64) == 0), 240.0, key).astype(key.dtype)
        large = jax.random.uniform(jax.random.key(seed + 1), value.shape, jnp.float32, 2.9e38, 3.3e38)
        value = jnp.where(dominant[None, :, None, None], large, value).astype(value.dtype)
        return query, key, value

    def reference(query, key, value):
        return jax.nn.dot_product_attention(query, key, value, implementation="xla",
                                            key_value_seq_lengths=jnp.asarray(lengths, jnp.int32))

    assert_as_exact(jax.jit(functools.partial(_slices_merged, spans=spans, merges=merges)),
                    as_given(jax.jit(reference)), outputs_near_the_largest,
                    "slices whose outputs sum past fp32's largest", seeds=SEEDS)


@functools.partial(jax.jit, static_argnames="slices")
def _sliced_step(query, key, value, lengths, *, slices: int):
    """A decode step of rows `lengths` long, its keys cut into `slices` slices a row at the decode plan's
    config, merged in the launch (`MosaicForward.merges`: the last slice's program merges the others'
    states, the interpreter's programs in turn standing in for its waits), the fold's raw products
    (`subtracted`) as `attention_decode.attend` walks them; lengths an operand, so every drawn length
    compiles one program."""
    spans = _decode_spans(lengths)
    with generation.assumed("sm90"):
        rows, row_spans, _ = attention_decode.packed(query, spans, DECODE_KV)
        config = attention_decode.plan(rows, key, value, row_spans, None, scale=None,
                                       target=generation.gpu_target()).config
        with tuning.pinned({attention_mosaic.MosaicForward: config}):
            merged = attention_mosaic.forward(
                rows, key, value, row_spans, template.Hooks(), scale=None, natural=False, interpret=True,
                out_dtype=jnp.float32, masked=True, splits=slices, subtracted=True, merges=True)
        return template.ungrouped_rows(merged, DECODE_HEADS).astype(query.dtype)


@jax.jit
def _whole_step(query, key, value, lengths):
    return template.reference(query, key, value, _decode_spans(lengths))


def _lengths_draw(seed: int):
    """`_decode_draw`'s step and its rows' lengths: one row of a single key beside one of every key, then the
    other way round, then lengths drawn from 1 to every key, so a slice of no key, a short row's first and a
    full row's last are each among a case's draws."""
    if seed < 2:
        lengths = jnp.asarray((1, DECODE_KEYS) if seed == 0 else (DECODE_KEYS, 1), jnp.int32)
    else:
        lengths = jax.random.randint(jax.random.key(seed + 1000), (DECODE_ROWS,), 1, DECODE_KEYS + 1)
    return (*_decode_draw(seed), lengths)


@INTERPRETS
@pytest.mark.parametrize("slices", [1, 2, 4, 8])
def test_a_decode_steps_keys_cut_into_any_slice_count_merge_to_its_attention_interpreted(slices):
    """A decode step's keys cut into `slices` slices a row (`attention_mosaic.forward`'s `splits` at the
    decode plan's config, a program a slice), the slices merged in the launch (`MosaicForward.merges`), rows
    of their own lengths, each draw's (`_lengths_draw`): on par with the step walked whole. The lengths an
    operand, one program judges every draw (row 4640: Hypothesis's eight lengths a case, each over eight
    draws, took the eight-slice case 124 s)."""
    assert_as_exact(functools.partial(_sliced_step, slices=slices), _whole_step, _lengths_draw,
                    f"decode step cut into {slices} slices", seeds=SEEDS)

