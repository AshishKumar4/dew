"""The template's decode (`attention_decode`): a decode step's attention through the template's forward over a
built work list, each query head of a key head's group a packed row, each row's keys cut into slices, merged.

Each case is held to the reference path's attention over the same keys (check 1): a step of one query position
and a verify block of four, grouped heads, rows of their own lengths and starts, sinks and a softcap, a row
past the fold's reach, and the builders' inverse (`unpacked` of `packed`) and the work list's slices exactly
covering each row's keys.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS
from reference_error import assert_as_exact

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template, attention_algebra as algebra, attention_decode

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


def _drawn(positions: int, heads: int, kv_heads: int, dtype, width: int = 64):
    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 4)
        query = jax.random.normal(streams[0], (len(LENGTHS), positions, heads, width), dtype)
        key, value = (jax.random.normal(stream, (len(LENGTHS), KEYS, kv_heads, width), dtype)
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



@INTERPRETS
def test_a_256_wide_decode_step_whose_split_launch_does_not_fit_is_on_par_interpreted():
    """A 256-wide bf16 decode step (Qwen3.5-0.8B's heads: 8 query heads over 2 key heads) on an H100, whose
    split launch's fp32 slices and the query's stage pass its shared memory, runs a slice a row through the
    template (`attention_decode.plan`), on par with the reference path (check 1)."""
    spans = _spans(1)

    def ours(query, key, value, sinks):
        del sinks
        with generation.assumed("sm90"):
            return attention_decode.attend(query, key, value, spans, None, interpret=True)

    def reference(query, key, value, sinks):
        del sinks
        return template.reference(query, key, value, spans, None)

    assert_as_exact(jax.jit(ours), jax.jit(reference), _drawn(1, 8, 2, jnp.bfloat16, 256),
                    "template decode 256 wide bfloat16", seeds=SEEDS)

@INTERPRETS
@pytest.mark.parametrize("call", ["one-query", "verify-four"])
def test_a_decode_rows_logits_past_the_folds_reach_walk_again_natural_interpreted(call):
    """A decode step whose first row's query is 2^16 times a drawn one (its folded logits past
    `algebra.FOLD_REACH`, where the fold's rounding moves its weights) and whose second row's is drawn: the
    first row walked again in the natural form by the masked launch (`algebra.walked_launches`), the second
    kept folded, each on par with the reference path (check 1)."""
    positions, heads, kv_heads = CALLS[call]
    spans = _spans(positions)
    drawn = _drawn(positions, heads, kv_heads, jnp.bfloat16)

    def draw(seed: int):
        query, key, value, _ = drawn(seed)
        return query.at[0].multiply(2.0 ** 16), key, value

    def ours(query, key, value):
        with generation.assumed("sm90"):
            return attention_decode.attend(query, key, value, spans, interpret=True)

    assert_as_exact(jax.jit(ours), jax.jit(lambda *operands: template.reference(*operands, spans)), draw,
                    f"template decode past the fold's reach {call}", seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("hook", ["plain", "sinks", "far"])
def test_a_decode_step_whose_programs_fill_the_gpu_finishes_each_row_in_its_walk_interpreted(hook):
    """A decode step of 17 rows over 8 key heads (136 programs, past an H100's 132 multiprocessors), so its
    work list takes one slice a row (`plan`) and the walk finishes each row itself from the initial state (a
    sink's), in the query's dtype; `far` its first row's query 2^16 times a drawn one, past the fold's reach,
    so the masked rewalk walks that row's query block alone: on par with the reference path
    (check 1)."""
    batch, heads, keys = 17, 8, 128
    lengths = jnp.asarray([keys - 7 * row for row in range(batch)], jnp.int32)
    spans = template.spans(1, keys, batch, lengths=lengths)

    def hooked(sinks):
        return template.Hooks(initial=template.sinks(sinks, batch, 1, 64)) if hook == "sinks" else None

    def ours(query, key, value, sinks):
        with generation.assumed("sm90"):
            rows, row_spans, _ = attention_decode.packed(query, spans, heads)
            work = attention_decode.plan(rows, key, value, row_spans, None, scale=None,
                                         target=generation.gpu_target())
            assert work.slices == 1, work
            return attention_decode.attend(query, key, value, spans, hooked(sinks), interpret=True)

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 4)
        query = jax.random.normal(streams[0], (batch, 1, heads, 64), jnp.bfloat16)
        if hook == "far":
            query = query.at[0].multiply(2.0 ** 16)
        key, value = (jax.random.normal(stream, (batch, keys, heads, 64), jnp.bfloat16)
                      for stream in streams[1:3])
        return query, key, value, jax.random.normal(streams[3], (heads,), jnp.float32)

    assert_as_exact(jax.jit(ours), jax.jit(lambda q, k, v, s: template.reference(q, k, v, spans, hooked(s))),
                    draw, f"template decode of one slice a row {hook}", seeds=SEEDS)


@INTERPRETS
def test_a_work_lists_rows_read_their_cache_rows_in_place_interpreted():
    """A serving step's work list over a cache of two rows: three batch rows, a decode row's one query and
    two pieces of four prompt queries each, reading cache rows 1, 0 and 1 where they lie (`attend`'s
    `sources`), each query over its own span (a piece's causal over its row's earlier keys, past its prefix):
    on par with the reference path over the rows gathered (check 1)."""
    sources = jnp.asarray([1, 0, 1], jnp.int32)
    starts = jnp.zeros((3, 4), jnp.int32)
    ends = jnp.asarray([[200, 0, 0, 0], [97, 98, 99, 100], [181, 182, 183, 184]], jnp.int32)
    spans = template.KeySpans(starts, ends, None, None, keys=KEYS)

    def ours(query, key, value):
        with generation.assumed("sm90"):
            return attention_decode.attend(query, key, value, spans, interpret=True, sources=sources)

    def reference(query, key, value):
        return template.reference(query, key[sources], value[sources], spans)

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 3)
        query = jax.random.normal(streams[0], (3, 4, 8, 64), jnp.bfloat16)
        key, value = (jax.random.normal(stream, (2, KEYS, 2, 64), jnp.bfloat16) for stream in streams[1:])
        return query, key, value

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, "template decode over a work list's rows",
                    seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("drafts", [4, 8, 16])
def test_a_verify_blocks_drafts_read_their_prefix_causally_interpreted(drafts):
    """A speculative verify block (`drafts` tokens a row written after the row's cache, `slots` theirs), each
    draft over its row's prefix and the drafts before it (causal over the block's slots) through the template
    decode, packed over 32 query heads on 8 key heads: on par with the reference path (check 1)."""
    held = jnp.asarray([200 - drafts, 37 - drafts], jnp.int32)
    slots = held[:, None] + jnp.arange(drafts, dtype=jnp.int32)[None]
    spans = template.spans(drafts, KEYS, 2, causal=True, lengths=held + drafts, slots=slots)
    drawn = _drawn(drafts, 32, 8, jnp.bfloat16)

    def ours(query, key, value):
        with generation.assumed("sm90"):
            return attention_decode.attend(query, key, value, spans, interpret=True)

    assert_as_exact(jax.jit(ours), jax.jit(lambda q, k, v: template.reference(q, k, v, spans)),
                    lambda seed: drawn(seed)[:3], f"template verify block of {drafts}", seeds=SEEDS)


def a_verify_block_through_the_route_is_on_par(drafts: int) -> None:
    """A GPU cell (tests/test_gpu_attention.py): a speculative verify block as the layer's cached call states
    it (`CrossDocuments` of its drafts over each row's held keys, causal), through the attention template's
    route, which runs a memory-bound inference call as the decode's work list (`attention_decode.serves`), 32
    query heads over 8 key heads, rows of their own lengths over 4096 keys: on par with the reference path
    over the same documents (check 1)."""
    from dew.nn.attention import scaled_dot_product_attention
    from dew.nn.documents import CrossDocuments

    rows, keys = 2, 4096
    held = jnp.asarray([keys - drafts, 1000], jnp.int32)
    documents = CrossDocuments(jnp.ones((rows, drafts), jnp.int32),
                               (jnp.arange(keys)[None] < (held + drafts)[:, None]).astype(jnp.int32))

    def attended(implementation):
        def attend(query, key, value):
            return scaled_dot_product_attention(query, key, value, causal=True, segment_ids=documents,
                                                implementation=implementation)
        return attend

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 3)
        query = jax.random.normal(streams[0], (rows, drafts, 32, 128), jnp.bfloat16)
        key, value = (jax.random.normal(stream, (rows, keys, 8, 128), jnp.bfloat16) for stream in streams[1:])
        return query, key, value

    assert_as_exact(jax.jit(attended("template")), jax.jit(attended("reference")), draw,
                    f"verify block of {drafts} through the route", seeds=SEEDS)


PAGE = 128
"""A paged cache's keys a page: whole key tiles of the derived decode launch's."""


@INTERPRETS
@pytest.mark.parametrize("page", [PAGE, 16])
@pytest.mark.parametrize("call", ["one-query", "verify-four"])
def test_a_paged_decode_step_through_the_template_is_on_par_interpreted(call, page):
    """A decode step over a paged cache (pools `[KV, P, page, D]` and their page table, each row's pages in an
    order of its own, some the other row's neighbours), read by the template where its table maps each key
    tile, on par with the reference path over the cache gathered dense (check 1): pages of whole key tiles,
    and pages of 16 keys, the docs' (docs/concepts/inference.md), several to a key tile, each loaded on its
    own (`mosaic.Pieces`, row 3648)."""
    positions, heads, kv_heads = CALLS[call]
    spans = _spans(positions)
    per_row = KEYS // page
    table = jnp.asarray(np.random.default_rng(page).permutation(2 * per_row).reshape(2, per_row), jnp.int32)

    def gathered(pool):
        # [KV, P, page, D] at each row's pages: [B, K, KV, D].
        rows = pool[:, table]
        return jnp.transpose(rows, (1, 2, 3, 0, 4)).reshape(len(LENGTHS), KEYS, kv_heads, pool.shape[-1])

    def ours(query, key_pool, value_pool):
        with generation.assumed("sm90"):
            return attention_decode.attend(query, key_pool, value_pool, spans, interpret=True, table=table)

    def reference(query, key_pool, value_pool):
        return template.reference(query, gathered(key_pool), gathered(value_pool), spans)

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 3)
        query = jax.random.normal(streams[0], (len(LENGTHS), positions, heads, 64), jnp.bfloat16)
        pools = (jax.random.normal(stream, (kv_heads, 2 * per_row, page, 64), jnp.bfloat16)
                 for stream in streams[1:])
        return query, *pools

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, f"template paged decode {call} {page}",
                    seeds=SEEDS)


def test_packing_a_calls_heads_into_its_rows_is_undone_by_unpacking():
    """A key head's group of query heads as its rows (`packed`) and back (`template.ungrouped_rows`): the
    query as it was, each packed row's keys its query's."""
    query = jax.random.normal(jax.random.key(0), (2, 3, 8, 16))
    spans = _spans(3)
    rows, row_spans, _ = attention_decode.packed(query, spans, 2)
    assert rows.shape == (2, 12, 2, 16)
    np.testing.assert_array_equal(template.ungrouped_rows(rows, 8), query)
    np.testing.assert_array_equal(row_spans.per_query().start, np.repeat(np.asarray(spans.start), 4, axis=1))


def _states(logits: jax.Array, values: jax.Array, cuts) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Each slice's softmax state over its keys `cuts[s]` to `cuts[s + 1]`, and one more of no key: its output
    normalized `[B, S, Q, KV, Dv]`, its maximum and denominator `[B, KV, S, Q]`."""
    outs, maxima, totals = [], [], []
    for lo, hi in itertools.pairwise(cuts):
        part = logits[..., lo:hi]
        peak = jnp.max(part, axis=-1)
        weights = jnp.exp(part - peak[..., None])
        total = jnp.sum(weights, axis=-1)
        outs.append(jnp.einsum("bqhk,bqhkd->bqhd", weights, values[..., lo:hi, :]) / total[..., None])
        maxima.append(peak)
        totals.append(total)
    outs.append(jnp.zeros_like(outs[0]))
    maxima.append(jnp.full_like(maxima[0], -jnp.inf))
    totals.append(jnp.zeros_like(totals[0]))
    maximum, total = (jnp.transpose(jnp.stack(x, axis=1), (0, 3, 1, 2)) for x in (maxima, totals))
    return jnp.stack(outs, axis=1), maximum, total


@pytest.mark.parametrize("slices", [1, 3, 7])
def test_merging_a_rows_slices_and_its_initial_state_is_its_whole_softmax(slices):
    """Slices of a row's keys, each its output normalized, its maximum and its denominator, a slice of no
    key, and a sink's initial state, merged (`merged`): on par with the row's softmax over every key and the
    sink at once (check 1)."""
    cuts = np.linspace(0, 40, slices + 1).astype(int)

    def ours(logits, values, sink):
        initial = template.Initial(sink, jnp.ones_like(sink), jnp.zeros((*sink.shape, values.shape[-1])))
        spec = algebra.Forward("spans", 40, ids=False, scale=1.0, natural=True, initial=True)
        out, maximum, total = _states(logits, values, cuts)
        return attention_decode.merged(spec, out, maximum, total, initial)

    def whole(logits, values, sink):
        scores = jnp.concatenate([logits, jnp.transpose(sink, (0, 2, 1))[..., None]], axis=-1)
        weights = jax.nn.softmax(scores, axis=-1)[..., :-1]
        return jnp.einsum("bqhk,bqhkd->bqhd", weights, values)

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 3)
        return (4.0 * jax.random.normal(streams[0], (2, 3, 1, 40)),
                jax.random.normal(streams[1], (2, 3, 1, 40, 8)), jax.random.normal(streams[2], (2, 1, 3)))

    assert_as_exact(jax.jit(ours), jax.jit(whole), draw, f"decode merge over {slices} slices", seeds=SEEDS)

