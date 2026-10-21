"""dew.nn.kernels.templates.attention: the attention template's shared core. Its visibility (`KeySpans`)
keeps the keys Dew's masks keep for every call form; its block map visits every visible entry and calls
a tile whole only where every entry is visible; and its XLA arithmetic (`reference`), hooks and all, is
on par with Dew's reference attention from float64 (check 1)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact
from test_attention_splash import exact_attention

from dew.nn.attention import (
    alibi_bias,
    chunk_mask,
    combined_attention_mask,
    scaled_dot_product_attention,
    with_key_lengths,
    with_key_starts,
)
from dew.nn.documents import CrossDocuments, cross_document_mask, with_documents
from dew.nn.kernels.templates import attention as template

QUERIES, KEYS = 100, 96
"""A call's query and key counts, neither a multiple of a tile, so the block map's edges pad; a call of
its own documents attends over its own tokens, `QUERIES` of each (`geometry`)."""

ROWS = 2


def own_documents():
    """A call's own packed documents `[2, 100]`: three and two documents, then padding (0)."""
    return jnp.asarray([[1] * 30 + [2] * 40 + [3] * 20 + [0] * 10, [4] * 55 + [5] * 45], jnp.int32)


def cross_documents():
    """Another sequence's documents: queries `[2, 100]` in any order, keys `[2, 96]` ascending, padding
    after them, the second row's second document holding no query."""
    queries = jnp.asarray([[2] * 20 + [1] * 50 + [0] * 30, [7] * 100], jnp.int32)
    keys = jnp.asarray([[1] * 60 + [2] * 30 + [0] * 6, [7] * 70 + [8] * 26], jnp.int32)
    return CrossDocuments(queries, keys)


LENGTHS = jnp.asarray([96, 61], jnp.int32)
STARTS = jnp.asarray([0, 17], jnp.int32)

CASES = {
    "full": {},
    "causal": {"causal": True},
    "causal-window": {"causal": True, "sliding_window": 24},
    "two-sided-window": {"sliding_window": 24},
    "chunk": {"chunk": 32},
    "causal-chunk": {"causal": True, "chunk": 32},
    "documents": {"segment_ids": "own"},
    "causal-documents-window": {"causal": True, "sliding_window": 24, "segment_ids": "own"},
    "cross-documents": {"segment_ids": "cross"},
    "cross-causal": {"causal": True, "segment_ids": "cross"},
    "cross-window": {"sliding_window": 24, "segment_ids": "cross"},
    "cross-causal-window": {"causal": True, "sliding_window": 24, "segment_ids": "cross"},
    "lengths": {"lengths": True},
    "lengths-starts": {"lengths": True, "starts": True},
    "documents-lengths": {"segment_ids": "own", "lengths": True},
    "cross-causal-lengths-starts": {"causal": True, "segment_ids": "cross", "lengths": True, "starts": True},
}


def geometry(case: dict) -> tuple[int, int]:
    """`case`'s query and key counts: a call's own documents are its keys, so it has as many of each."""
    return (QUERIES, QUERIES) if case.get("segment_ids") == "own" else (QUERIES, KEYS)


def arguments(case: dict) -> dict:
    """`case`'s `template.spans` keywords, its documents and key bounds materialized."""
    ids = {"own": own_documents, "cross": cross_documents}
    keywords = {name: value for name, value in case.items() if name in ("causal", "sliding_window", "chunk")}
    if "segment_ids" in case:
        keywords["segment_ids"] = ids[case["segment_ids"]]()
    if case.get("lengths"):
        keywords["lengths"] = LENGTHS
    if case.get("starts"):
        keywords["starts"] = STARTS
    return keywords


def dews_mask(case: dict) -> np.ndarray:
    """The keys Dew's reference path keeps for `case`, `[B, Q, K]`, composed as
    `scaled_dot_product_attention` composes them: another sequence's documents under causality or a
    window aligned to their last keys (`cross_document_mask`), else causality and a window on the
    indices (`combined_attention_mask`) narrowed to the documents (`with_documents`); a chunk as
    `chunk_mask` keeps it; then key lengths and starts."""
    keywords = arguments(case)
    queries, keys = geometry(case)
    ids = keywords.get("segment_ids")
    causal, window = keywords.get("causal", False), keywords.get("sliding_window")
    if isinstance(ids, CrossDocuments) and (causal or window is not None):
        mask = cross_document_mask(ids, causal=causal, sliding_window=window)
    else:
        mask = combined_attention_mask(queries, keys, causal, window, None)
        if ids is not None:
            mask = with_documents(mask, ids)
    if "chunk" in keywords:
        chunked = chunk_mask(jnp.arange(queries), jnp.arange(keys), keywords["chunk"])
        mask = chunked if mask is None else jnp.logical_and(mask, chunked)
    if "lengths" in keywords:
        mask = with_key_lengths(mask, keywords["lengths"], keys)
    mask = with_key_starts(mask, keywords.get("starts"), keys)
    full = np.ones((ROWS, queries, keys), bool)
    return full if mask is None else np.broadcast_to(np.asarray(mask)[:, 0], full.shape)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES)
def test_key_spans_keep_the_keys_dews_masks_keep(case):
    """Each call form's spans (`template.spans`, `template.dense`) keep exactly the keys Dew's reference
    path keeps, under `jax.jit` too, the bounds traced."""
    expected = dews_mask(case)
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    np.testing.assert_array_equal(np.asarray(template.dense(spans)), expected)
    static = {name: case[name] for name in ("causal", "sliding_window", "chunk") if name in case}
    traced = jax.jit(lambda keywords: template.dense(
        template.spans(queries, keys, ROWS, **keywords, **static)))
    dynamic = {name: value for name, value in arguments(case).items()
               if name not in ("causal", "sliding_window", "chunk")}
    np.testing.assert_array_equal(np.asarray(traced(dynamic)), expected)


BLOCKS = [(16, 16), (32, 16), (16, 64), (64, 32)]


@pytest.mark.parametrize("block_q, block_k", BLOCKS)
@pytest.mark.parametrize("case", CASES.values(), ids=CASES)
def test_a_block_map_visits_every_visible_entry_and_calls_a_tile_whole_only_where_it_is(case, block_q,
                                                                                         block_k):
    """At each tile shape, every visible entry lies in a tile its query block visits (`lo` to `hi`), and
    its key block's query range (`q_lo` to `q_hi`) holds that query block; every tile in a query block's
    whole range (`full_lo` to `full_hi`) whose blocks carry one document, or which has no documents, has
    every entry visible, the padding past the call's queries and keys excluded from no tile; with no
    narrowing, every tile inside the call is whole."""
    queries_, keys_ = geometry(case)
    spans = template.spans(queries_, keys_, ROWS, **arguments(case))
    blocks = template.block_map(spans, block_q, block_k)
    seen = np.asarray(template.dense(spans))
    lo, hi, full_lo, full_hi, q_lo, q_hi = (np.asarray(x) for x in (
        blocks.lo, blocks.hi, blocks.full_lo, blocks.full_hi, blocks.q_lo, blocks.q_hi))
    rows, queries, keys = np.nonzero(seen)
    i, j = queries // block_q, keys // block_k
    assert np.all((lo[rows, i] <= j) & (j < hi[rows, i]))
    assert np.all((q_lo[rows, j] <= i) & (i < q_hi[rows, j]))
    query_doc = None if blocks.query_doc is None else np.asarray(blocks.query_doc)
    key_doc = None if blocks.key_doc is None else np.asarray(blocks.key_doc)
    padded = np.zeros((ROWS, -(-queries_ // block_q) * block_q, -(-keys_ // block_k) * block_k), bool)
    padded[:, :queries_, :keys_] = seen
    whole = 0
    for row in range(ROWS):
        for block in range(lo.shape[1]):
            assert lo[row, block] <= full_lo[row, block] <= full_hi[row, block] <= hi[row, block]
            for tile in range(full_lo[row, block], full_hi[row, block]):
                if query_doc is not None and not (query_doc[row, block] == key_doc[row, tile] > 0):
                    continue
                entries = padded[row, block * block_q:(block + 1) * block_q,
                                 tile * block_k:(tile + 1) * block_k]
                assert entries.all(), (row, block, tile)
                whole += 1
    if case is CASES["full"]:
        assert whole == ROWS * (queries_ // block_q) * (keys_ // block_k)


SEEDS = 8
"""The reference arithmetic's declared draws (check 1), sized by `seeds_for`."""


def drawing(queries: int, keys: int):
    """A draw of queries `[2, queries, 4, 32]` over keys and values `[2, keys, 2, 32]`, two query heads
    a key head."""
    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 3)
        query = jax.random.normal(streams[0], (ROWS, queries, 4, 32), jnp.float32)
        key = jax.random.normal(streams[1], (ROWS, keys, 2, 32), jnp.float32)
        return query, key, jax.random.normal(streams[2], (ROWS, keys, 2, 32), jnp.float32)

    return draw


SINKS = jnp.asarray([0.5, -1.0, 2.0, -3.0], jnp.float32)
SOFTCAP = 0.5


HOOKS = {
    "none": {},
    "softcap": {"softcap": True},
    "sinks": {"sinks": True},
    "alibi": {"alibi": True},
    "softcap-sinks-alibi": {"softcap": True, "sinks": True, "alibi": True},
}


def hooks_of(hooks: dict, queries: int) -> template.Hooks:
    """The template's hooks for `hooks`: a softcap and ALiBi as one score_mod, sinks as the softmax's
    initial state."""
    slopes = alibi_bias(jnp.ones((1, 1)), 4, dtype=jnp.float32).reshape(4)
    cap = SOFTCAP if hooks.get("softcap") else None

    def score_mod(s, q_pos, k_pos, *params):
        if cap is not None:
            s = cap * jnp.tanh(s / cap)
        if hooks.get("alibi"):
            s = s + params[0].astype(s.dtype) * k_pos.astype(s.dtype)
        return s

    params = (template.Param(slopes, template.Kind.HEAD),) if hooks.get("alibi") else ()
    initial = template.sinks(SINKS, ROWS, queries, 32) if hooks.get("sinks") else None
    used = cap is not None or hooks.get("alibi")
    return template.Hooks(score_mod if used else None, params, initial)


ARITHMETIC = ("full", "causal-window", "causal-documents-window", "cross-causal-lengths-starts")


@pytest.mark.parametrize("case", [CASES[name] for name in ARITHMETIC], ids=ARITHMETIC)
def test_the_scores_hide_exactly_the_keys_their_spans_hide(case):
    """`template.scores`, which `reference`'s softmax takes, under every score hook: -inf where the
    call's spans hide the key, finite where they keep it."""
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    query, key, _ = drawing(queries, keys)(0)
    logits = template.scores(query, key, spans, hooks_of(HOOKS["softcap-sinks-alibi"], queries))
    seen = np.broadcast_to(np.asarray(template.dense(spans))[:, None], logits.shape)
    np.testing.assert_array_equal(np.isfinite(np.asarray(logits)), seen)


@pytest.mark.parametrize("hooks", HOOKS.values(), ids=HOOKS)
@pytest.mark.parametrize("case", [CASES[name] for name in ARITHMETIC], ids=ARITHMETIC)
def test_the_templates_arithmetic_is_on_par_with_dews_reference_attention(case, hooks):
    """`template.reference` over each call form's spans and each hook, on par with Dew's reference path
    (`scaled_dot_product_attention`, 'reference', the same keys as a mask) over the site's draws, both
    measured from float64 (`exact_attention` under that mask); a query that keeps no key reads zero on
    both."""
    mask = jnp.asarray(dews_mask(case))[:, None]
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    sinks = SINKS if hooks.get("sinks") else None
    softcap = SOFTCAP if hooks.get("softcap") else None
    alibi = alibi_bias(jnp.arange(keys)[None], 4, dtype=jnp.float32) if hooks.get("alibi") else None

    def ours(query, key, value):
        return template.reference(query, key, value, spans, hooks_of(hooks, queries))

    def dews(query, key, value):
        bias = None if alibi is None else alibi.astype(query.dtype)
        return scaled_dot_product_attention(query, key, value, mask=mask, bias=bias, sinks=sinks,
                                            softcap=softcap, implementation="reference")

    def exact(query, key, value):
        return exact_attention(query, key, value, None if sinks is None else sinks.astype(query.dtype),
                               softcap=softcap, mask=np.asarray(mask))

    # `exact_attention` takes no bias: under ALiBi the truth is Dew's reference path in float64.
    assert_as_exact(jax.jit(ours), jax.jit(dews), drawing(queries, keys),
                    f"template arithmetic {case} {hooks}",
                    seeds=SEEDS, truth_fn=None if alibi is not None else exact)
