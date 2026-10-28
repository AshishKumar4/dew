"""dew.nn.kernels.templates.attention: the attention template's shared core. Its visibility (`KeySpans`)
keeps the keys Dew's masks keep for every call form; its block map visits every visible entry and calls
a tile whole only where every entry is visible; and its XLA arithmetic (`reference`), hooks and all, is
on par with Dew's reference attention from float64 (check 1)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import exact_attention
from reference_error import assert_as_exact

from dew.batch import coordinates
from dew.nn.attention import (
    alibi_bias,
    chunk_mask,
    combined_attention_mask,
    scaled_dot_product_attention,
    with_key_lengths,
    with_key_starts,
)
from dew.nn.documents import (
    CrossDocuments,
    aligned_places,
    cross_document_mask,
    document_ids,
    document_mask,
    with_documents,
)
from dew.nn.inputs import counted_places
from dew.nn.kernels.templates import attention as template
from dew.nn.precision import FP32_SUMS
from dew.nn.visibility import windowed

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


def key_validity(keys: int) -> jax.Array:
    """Which of a call's first `keys` keys are real `[2, keys]`: holes of a key and of five in the first row,
    one across `own_documents`' boundary at 30; every seventh key in the second, and a run of thirty from
    60, longer than the cases' window, so the queries late in it whose window it covers read no key."""
    index = np.arange(QUERIES)
    first = ~((index == 5) | ((index >= 28) & (index < 33)) | ((index >= 40) & (index < 45)))
    second = (index % 7 != 3) & ~((index >= 60) & (index < 90))
    return jnp.asarray(np.stack([first, second])[:, :keys])


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
    "documents-causal-chunk-places": {"causal": True, "chunk": 16, "segment_ids": "own", "positions": True},
    "cross-causal-chunk": {"causal": True, "chunk": 32, "segment_ids": "cross"},
    "widest-window": {"sliding_window": 2 ** 31 - 1},
    "causal-widest-window": {"causal": True, "sliding_window": 2 ** 31 - 1},
    "block-causal": {"causal": True, "block_groups": True},
    "documents-block-causal-window": {"causal": True, "sliding_window": 24, "segment_ids": "own",
                                      "block_groups": True},
    "cross-block-causal-window": {"causal": True, "sliding_window": 24, "segment_ids": "cross",
                                  "block_groups": True},
    "valid": {"valid": True},
    "causal-window-valid": {"causal": True, "sliding_window": 24, "valid": True},
    "documents-causal-valid": {"causal": True, "segment_ids": "own", "valid": True},
    "cross-causal-window-valid": {"causal": True, "sliding_window": 24, "segment_ids": "cross",
                                  "valid": True},
    "documents-window-valid-places": {"causal": True, "sliding_window": 24, "segment_ids": "own",
                                      "valid": True, "positions": True},
}


OPTIONS = ((20, 4, 10), (10, 3, 12))
"""Each row's decision layout over its call's 100 tokens (`decision_runs`): its prefix's length, then its
options' count and width, the second row's options leaving a fragment empty."""


def decision_runs() -> tuple[np.ndarray, np.ndarray]:
    """A parallel decision layout's runs `[2, 100, 6, 2]` and places `[2, 100]` (`dew.decision.layout`): a
    shared token reads every fragment of its row, the prefix, each option and the suffix, each a run of its
    own; an option's token reads the prefix, its own option and the suffix, the rest empty. The places
    restart at each option, the suffix continuing after the widest, so they never decrease within a run."""
    runs = np.zeros((ROWS, QUERIES, 6, 2), np.int32)
    places = np.zeros((ROWS, QUERIES), np.int32)
    for row, (prefix, count, width) in enumerate(OPTIONS):
        after = prefix + count * width
        fragments = [(0, prefix), *((prefix + k * width, prefix + (k + 1) * width) for k in range(count)),
                     (after, QUERIES)]
        places[row] = np.concatenate([np.arange(prefix), *(np.arange(prefix, prefix + width),) * count,
                                      np.arange(prefix + width, prefix + width + QUERIES - after)])
        for query in range(QUERIES):
            option = (query - prefix) // width if prefix <= query < after else None
            kept = fragments if option is None else [fragments[0], fragments[1 + option], fragments[-1]]
            runs[row, query, :len(kept)] = kept
    return runs, places


RUN_CASES = {
    "runs": {"runs": True},
    "runs-window-places": {"runs": True, "sliding_window": 6, "positions": True},
    "runs-causal": {"runs": True, "causal": True},
    "runs-documents-valid": {"runs": True, "segment_ids": "own", "valid": True},
    "runs-chunk-places": {"runs": True, "chunk": 8, "positions": True},
}
"""Call forms over runs (`template.spans`'s `runs`): a decision row's, under a two-sided window over its
places, under causality, beside packed documents and key validity, and in chunks of its places, which
restart at each option, so a suffix's chunk holds keys of every option."""


def band_runs(width: int) -> np.ndarray:
    """One run a query `[2, 100, 1, 2]`, the `width` keys up to its own, as a chunked or banded local-attention
    layer passes its runs (`local_attention.chunk_runs`)."""
    index = np.arange(QUERIES, dtype=np.int32)
    run = np.stack([np.maximum(index + 1 - width, 0), index + 1], axis=-1)
    return np.broadcast_to(run[None, :, None], (ROWS, QUERIES, 1, 2))


ONE_RUN = {"band": 12}
"""A call over one run a query (`band_runs`), which `template.spans` keeps as one key interval a query, the
form every mainloop walks (ledger 1991)."""


def geometry(case: dict) -> tuple[int, int]:
    """`case`'s query and key counts: a call's own documents, or blocks of its own tokens, are its keys, so
    it has as many of each; a cache call's blocks are its keys'."""
    own = (case.get("segment_ids") == "own" or case.get("runs") or case.get("band")
           or (case.get("block_groups") and case.get("segment_ids") is None))
    return (QUERIES, QUERIES) if own else (QUERIES, KEYS)


def block_groups(keys: int = QUERIES) -> tuple[np.ndarray, np.ndarray]:
    """Two layouts of contiguous blocks over a call's first `keys` keys `[2, keys]`, -1 outside any, read as
    whole blocks and as images: blocks of a token, blocks across `own_documents`' boundaries (at 30 and 70 in
    the first row, 55 in the second) and `cross_documents`' (at 60 in the first, 70 in the second), a block
    of padding, blocks ending the row, blocks longer than the cases' window; and the images over the
    blocks, as a kind reading both does."""
    first = np.full((ROWS, QUERIES), -1, np.int32)
    for row, runs in enumerate((((3, 4), (10, 22), (25, 35), (50, 60), (58, 64), (68, 72), (90, 100)),
                                ((0, 1), (40, 50), (52, 58), (60, 61), (66, 99)))):
        for group, (lo, hi) in enumerate(runs):
            first[row, lo:hi] = group
    second = np.full((ROWS, QUERIES), -1, np.int32)
    second[:, 15:40] = 7
    second[:, 41:43] = 8
    return first[:, :keys], second[:, :keys]


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
    if case.get("block_groups"):
        keywords["block_groups"], keywords["image_groups"] = block_groups(geometry(case)[1])
    if case.get("valid"):
        keywords["valid"] = key_validity(geometry(case)[1])
    if case.get("band"):
        keywords["runs"] = jnp.asarray(band_runs(case["band"]))
    if case.get("runs"):
        runs, places = decision_runs()
        keywords["runs"] = jnp.asarray(runs)
        if case.get("positions"):
            keywords["positions"] = (jnp.asarray(places), jnp.asarray(places))
            return keywords
    if case.get("positions"):
        # Each token's place in its own document, as a packed row's coordinates count it, over its real
        # tokens where only some are (`counted_places`).
        places = (coordinates(segments=keywords["segment_ids"]).positions if "valid" not in keywords
                  else counted_places(keywords["segment_ids"], keywords["valid"]))
        keywords["positions"] = (places, places)
    return keywords


def dews_mask(case: dict) -> np.ndarray:
    """The keys Dew's reference path keeps for `case`, `[B, Q, K]`, composed as
    `scaled_dot_product_attention` composes them: another sequence's documents under causality or a
    window aligned to their last keys (`cross_document_mask`), else causality and a window on the
    indices (`combined_attention_mask`) narrowed to the documents (`with_documents`); a chunk as
    `chunk_mask` keeps it over the call's places (the attention layer's `_chunked`), its indices where it
    names none, as a window does where the call names places and real keys (the layer's counted places);
    then key lengths and starts, and the real keys."""
    keywords = arguments(case)
    queries, keys = geometry(case)
    ids = keywords.get("segment_ids")
    causal, window = keywords.get("causal", False), keywords.get("sliding_window")
    # Each query's own key: its index, or a cache call's slot, its document's first key plus its aligned
    # place, none where that lies before its document's keys.
    slots = np.broadcast_to(np.arange(queries), (ROWS, queries))
    cross = False
    if isinstance(ids, CrossDocuments) and (causal or window is not None):
        cross = True
        mask = cross_document_mask(ids, causal=causal, sliding_window=window)
        query_ids, key_ids = (np.asarray(x) for x in document_ids(ids))
        first = np.argmax(key_ids[:, None, :] == query_ids[:, :, None], axis=-1)
        ranks = np.asarray(aligned_places(ids))
        slots = np.where(ranks >= 0, first + ranks, -1)
    elif window is not None and "positions" in keywords:
        query_places, key_places = (np.asarray(x) for x in keywords["positions"])
        counted = windowed(query_places[:, :, None], key_places[:, None, :], window, causal=causal)[:, None]
        base = combined_attention_mask(queries, keys, causal, None, None)
        mask = counted if base is None else base & counted
    else:
        mask = combined_attention_mask(queries, keys, causal, window, None)
    held = (slots >= 0) & (slots < keys)
    for groups, whole in ((keywords.get("block_groups"), True), (keywords.get("image_groups"), False)):
        if groups is None or not causal:
            continue
        # A block's keys are seen by every query of its document whose own key is in it: a whole
        # block's whatever the window (transformers' `block_sequence_ids`), an image's within the
        # window, which bounds the keys behind (Gemma's).
        own = np.where(held, np.take_along_axis(groups, np.clip(slots, 0, keys - 1), axis=1), -1)
        same = (own[:, :, None] == groups[:, None, :]) & (own[:, :, None] >= 0)
        if window is not None and not whole:
            same = same & windowed(slots[:, :, None], np.arange(keys)[None, None, :], window, causal=causal)
        if ids is not None:
            same = same & np.asarray(document_mask(ids))
        mask = same[:, None] if mask is None else mask | same[:, None]
    if ids is not None and not cross:
        mask = with_documents(mask, ids)
    if "chunk" in keywords:
        places = keywords.get("positions", (jnp.arange(queries), jnp.arange(keys)))
        chunked = chunk_mask(*places, keywords["chunk"])
        mask = chunked if mask is None else jnp.logical_and(mask, chunked)
    if "lengths" in keywords:
        mask = with_key_lengths(mask, keywords["lengths"], keys)
    mask = with_key_starts(mask, keywords.get("starts"), keys)
    if "valid" in keywords:
        real = jnp.asarray(keywords["valid"])[:, None, None, :]
        mask = real if mask is None else jnp.logical_and(mask, real)
    if "runs" in keywords:
        # A query keeps a key one of its runs holds.
        runs = np.asarray(keywords["runs"])
        index = np.arange(keys)[None, None, :, None]
        held = np.any((index >= runs[:, :, None, :, 0]) & (index < runs[:, :, None, :, 1]), axis=-1)
        mask = held[:, None] if mask is None else jnp.logical_and(mask, held[:, None])
    full = np.ones((ROWS, queries, keys), bool)
    return full if mask is None else np.broadcast_to(np.asarray(mask)[:, 0], full.shape)


@pytest.mark.parametrize("case", [*CASES.values(), *RUN_CASES.values(), ONE_RUN],
                         ids=[*CASES, *RUN_CASES, "one-run"])
def test_key_spans_keep_the_keys_dews_masks_keep(case):
    """Each call form's spans (`template.spans`, `template.dense`) keep exactly the keys Dew's reference
    path keeps, under `jax.jit` too, the bounds traced."""
    expected = dews_mask(case)
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    np.testing.assert_array_equal(np.asarray(template.dense(spans)), expected)
    static = {name: case[name] for name in ("causal", "sliding_window", "chunk") if name in case}
    if case is CASES["block-causal"]:
        # A block runs to its end, past the query's own place.
        assert expected[0, 10, 21] and not expected[0, 9, 10]
    if case is CASES["documents-block-causal-window"]:
        # The second row's image of 15 to 40 sees ahead and, behind, only within the window of 24, and its
        # whole block of 66 to 99 past the window too, but no further; the first row's image stops at the
        # document's end at 30.
        assert expected[1, 16, 39] and expected[1, 39, 16] and not expected[1, 39, 15]
        assert expected[1, 98, 66] and not expected[1, 98, 65]
        assert expected[0, 16, 29] and not expected[0, 16, 30]
    if case is CASES["causal-window-valid"]:
        # The second row's queries 84 to 89, whose window of 24 behind holds only the run of invalid keys
        # from 60, read none; query 90 reads itself.
        assert not expected[1, 84:90].any() and expected[1, 90, 90]
    if case is CASES["documents-window-valid-places"]:
        # The window counts real tokens: query 27, across the hole at 5, keeps key 3, 23 real tokens behind
        # it and 24 slots, and not key 2.
        assert expected[0, 27, 3] and not expected[0, 27, 2]
    if case is CASES["cross-block-causal-window"]:
        # A cache call's query 68 sits at slot 58, whose block runs to 64 past its document's last key
        # at 59: it sees 59 and not 60. The second row's first 30 queries hold no slot, so see nothing.
        assert expected[0, 68, 59] and not expected[0, 68, 60] and not expected[1, :30].any()
    if case is CASES["documents-causal-chunk-places"]:
        # The second document opens at token 30, so its chunk 0 holds tokens 30 to 45: chunks counted
        # from the row's start would part tokens 31 and 32.
        assert expected[0, 32, 31]
    traced = jax.jit(lambda keywords: template.dense(
        template.spans(queries, keys, ROWS, **keywords, **static)))
    dynamic = {name: value for name, value in arguments(case).items()
               if name not in ("causal", "sliding_window", "chunk")}
    np.testing.assert_array_equal(np.asarray(traced(dynamic)), expected)


BLOCKS = [(16, 16), (32, 16), (16, 64), (64, 32)]


def uniform_documents(ids, block: int) -> np.ndarray:
    """Each block's document id `[B, N // block]` where every one of its tokens carries that one id,
    else 0, the padding id, from the call's ids themselves; a block running past them is 0 unless its
    tokens are all padding too."""
    ids = np.asarray(ids)
    padded = np.zeros((ids.shape[0], -(-ids.shape[1] // block) * block), ids.dtype)
    padded[:, :ids.shape[1]] = ids
    blocks = padded.reshape(ids.shape[0], -1, block)
    return np.where((blocks == blocks[..., :1]).all(axis=-1), blocks[..., 0], 0)


@pytest.mark.parametrize("block_q, block_k", BLOCKS)
@pytest.mark.parametrize("case", [*CASES.values(), *RUN_CASES.values()], ids=[*CASES, *RUN_CASES])
def test_a_block_map_visits_every_visible_entry_and_calls_a_tile_whole_only_where_it_is(case, block_q,
                                                                                         block_k):
    """At each tile shape, every visible entry lies in a tile its query block visits (`lo` to `hi`), and
    its key block's query range (`q_lo` to `q_hi`) holds that query block; each block's one document
    (`query_doc`, `key_doc`) is the one its tokens carry in the spans' ids, its documents' and its invalid
    keys' padding id (`uniform_documents`); every tile in a query
    block's whole range (`full_lo` to `full_hi`) whose blocks carry one document, or which has no
    documents, has every entry visible, the padding past the call's queries and keys excluded from no
    tile; with no narrowing, every tile inside the call is whole, and a call of its own documents has
    whole tiles inside them. Over runs the block visits only its ranges (`ranges`), each tile once, and every
    tile of its whole range, its widest run's tiles every query sees through that run, has every entry
    visible; a tile its queries see whole through other runs is not called whole, so the whole range over
    runs is sound, not exact."""
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
    if blocks.ranges is not None:
        ranges = np.asarray(blocks.ranges)
        # The ranges are sorted, disjoint, nonempty first: a walk of them visits each tile once.
        lengths = ranges[..., 1] - ranges[..., 0]
        assert np.all(lengths >= 0)
        assert np.all(ranges[..., 1:, 0][lengths[..., 1:] > 0] >= ranges[..., :-1, 1][lengths[..., 1:] > 0])
        assert np.all(np.diff((lengths > 0).astype(int), axis=-1) <= 0)
        covered = np.any((ranges[rows, i, :, 0] <= j[:, None]) & (j[:, None] < ranges[rows, i, :, 1]), -1)
        assert covered.all()
    query_doc = key_doc = None
    if spans.query_ids is not None and spans.key_ids is not None:
        query_doc = uniform_documents(spans.query_ids, block_q)
        key_doc = uniform_documents(spans.key_ids, block_k)
        np.testing.assert_array_equal(np.asarray(blocks.query_doc), query_doc)
        np.testing.assert_array_equal(np.asarray(blocks.key_doc), key_doc)
    padded = np.zeros((ROWS, -(-queries_ // block_q) * block_q, -(-keys_ // block_k) * block_k), bool)
    padded[:, :queries_, :keys_] = seen
    whole = 0
    for row in range(ROWS):
        for block in range(lo.shape[1]):
            assert lo[row, block] <= full_lo[row, block] <= full_hi[row, block] <= hi[row, block]
            for tile in range(full_lo[row, block], full_hi[row, block]):
                if query_doc is not None and not (query_doc[row, block] == key_doc[row, tile] != 0):
                    continue
                entries = padded[row, block * block_q:(block + 1) * block_q,
                                 tile * block_k:(tile + 1) * block_k]
                assert entries.all(), (row, block, tile)
                whole += 1
    # Every tile whose entries are all visible, of one document where there are any, is in a whole range.
    tiles = padded.reshape(ROWS, padded.shape[1] // block_q, block_q, -1, block_k).all(axis=(2, 4))
    if query_doc is not None:
        tiles &= (query_doc[:, :, None] == key_doc[:, None, :]) & (query_doc[:, :, None] != 0)
    assert whole == tiles.sum() if blocks.ranges is None else whole <= tiles.sum()
    if (case is CASES["documents"] or case is RUN_CASES["runs"]) and (block_q, block_k) == (16, 16):
        assert whole > 0


SEEDS = 8
"""The reference arithmetic's declared draws (check 1), sized by `seeds_for`."""


def drawing(queries: int, keys: int, dtype=jnp.float32):
    """A draw of queries `[2, queries, 4, 32]` over keys and values `[2, keys, 2, 32]` in `dtype`, two
    query heads a key head."""
    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 3)
        return tuple(jax.random.normal(stream, (ROWS, count, heads, 32), jnp.float32).astype(dtype)
                     for stream, count, heads in zip(streams, (queries, keys, keys), (4, 2, 2), strict=True))

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


DTYPES = [jnp.float32, jnp.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: jnp.dtype(dtype).name)
@pytest.mark.parametrize("hooks", HOOKS.values(), ids=HOOKS)
@pytest.mark.parametrize("case", [CASES[name] for name in ARITHMETIC], ids=ARITHMETIC)
def test_the_templates_arithmetic_is_on_par_with_dews_reference_attention(case, hooks, dtype):
    """`template.reference` over each call form's spans and each hook, on par with Dew's reference path
    (`scaled_dot_product_attention`, 'reference', the same keys as a mask) over the site's draws, both
    measured from float64 (`exact_attention` under that mask); a query that keeps no key reads zero on
    both. The template sums its fp32 products as the kernels it stands for do, so the reference path
    sums its own as an accelerator does (`FP32_SUMS`)."""
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

    def summed(query, key, value):
        with jax.default_matmul_precision(FP32_SUMS):
            return dews(query, key, value)

    def exact(query, key, value):
        return exact_attention(query, key, value, None if sinks is None else sinks.astype(query.dtype),
                               softcap=softcap, mask=np.asarray(mask))

    # `exact_attention` takes no bias: under ALiBi the truth is Dew's reference path in float64.
    assert_as_exact(jax.jit(ours), jax.jit(summed), drawing(queries, keys, dtype),
                    f"template arithmetic {case} {hooks} {jnp.dtype(dtype).name}",
                    seeds=SEEDS, truth_fn=dews if alibi is not None else exact)


LAYOUTS = ("causal-documents-window", "cross-causal-lengths-starts")


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: jnp.dtype(dtype).name)
@pytest.mark.parametrize("case", [CASES[name] for name in LAYOUTS], ids=LAYOUTS)
def test_every_parameter_layout_and_the_output_hook_are_on_par(case, dtype):
    """A score hook reading a parameter of each score layout, a query's temperature, a key's bias and a
    dense bias, `s * t_q + b_k + B_qk`, and an output gate `o * sigmoid(g)`, on par with Dew's reference
    path taking them as it can: the temperature in the queries, the biases as one dense bias, the gate
    on its output; both measured from float64."""
    mask = jnp.asarray(dews_mask(case))[:, None]
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    streams = jax.random.split(jax.random.key(5), 4)
    temperature = jax.random.uniform(streams[0], (ROWS, 4, queries), jnp.float32, 0.5, 1.5)
    key_bias = jax.random.normal(streams[1], (ROWS, 4, keys), jnp.float32)
    dense_bias = 0.5 * jax.random.normal(streams[2], (ROWS, 4, queries, keys), jnp.float32)
    gate = jax.random.normal(streams[3], (ROWS, queries, 4, 32), jnp.float32)

    def score_mod(s, q_pos, k_pos, t, b, bias):
        return s * t.astype(s.dtype) + b.astype(s.dtype) + bias.astype(s.dtype)

    def output_mod(o, g):
        return o * jax.nn.sigmoid(g.astype(o.dtype))

    hooks = template.Hooks(score_mod, (template.Param(temperature, template.Kind.QUERY),
                                       template.Param(key_bias, template.Kind.KEY),
                                       template.Param(dense_bias, template.Kind.TILE)),
                           None, output_mod, (template.Param(gate, template.Kind.OUTPUT),))

    def ours(query, key, value):
        return template.reference(query, key, value, spans, hooks)

    def dews(query, key, value):
        wide = jnp.promote_types(query.dtype, jnp.float32)
        scaled = (query.astype(wide) * jnp.swapaxes(temperature, 1, 2)[..., None]).astype(query.dtype)
        # Each bias widened before the sum, as the template widens each parameter: a float64 truth sums
        # them in float64, not their fp32 sum widened after its rounding.
        bias = (key_bias[:, :, None, :].astype(wide) + dense_bias.astype(wide)).astype(query.dtype)
        out = scaled_dot_product_attention(scaled, key, value, mask=mask, bias=bias,
                                           implementation="reference")
        return (out.astype(wide) * jax.nn.sigmoid(gate.astype(wide))).astype(query.dtype)

    assert_as_exact(jax.jit(ours), jax.jit(dews), drawing(queries, keys, dtype),
                    f"template layouts {case} {jnp.dtype(dtype).name}", seeds=SEEDS)


def halves(spans: template.KeySpans, half: int) -> tuple[template.KeySpans, template.KeySpans]:
    """`spans` over the keys before `half` and over those from it, each in its own keys' indices."""
    ids = spans.key_ids
    return (template.KeySpans(jnp.minimum(spans.start, half), jnp.minimum(spans.end, half), spans.query_ids,
                              None if ids is None else ids[:, :half], half),
            template.KeySpans(jnp.maximum(spans.start - half, 0), jnp.maximum(spans.end - half, 0),
                              spans.query_ids, None if ids is None else ids[:, half:], spans.keys - half))


@pytest.mark.parametrize("split", ["sliced-at-a-tile-edge", "spanned-at-the-half"])
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: jnp.dtype(dtype).name)
@pytest.mark.parametrize("hooks", [HOOKS["none"], HOOKS["sinks"]], ids=["none", "sinks"])
@pytest.mark.parametrize("case", [CASES[name] for name in ARITHMETIC], ids=ARITHMETIC)
def test_a_resumed_partial_is_on_par_with_one_pass_over_its_keys(case, hooks, dtype, split):
    """The softmax over a call's first keys, its output, maxima and denominators resumed as the initial
    state of one over the rest (`template.Initial`, its numerator the output times the denominator, a
    query that saw nothing resuming from no state), gives the output, maxima and denominators of one
    pass over every key, on par from float64 over the site's draws: the keys parted at a tile's edge
    into two arrays, as a split-KV launch parts them, or parted at their half by the spans over the
    whole arrays. The resume's contract is at the cuts launches make, tile edges. (Sliced into arrays of
    50 or 64 of these 100 keys, XLA:CPU's fp32 product of the queries with the keys rounds the logits
    1.555 times as far from float64 as the whole 100-key product does on the same entries, over 20
    draws; sliced at 48 or 100, alike: the reference's own dot rounding by its shape, which the
    spanned half keeps the whole call's. Recorded 2026-10-21.)"""
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    start = hooks_of(hooks, queries)
    if split == "sliced-at-a-tile-edge":
        edge = keys // 2 // 32 * 32
        first, rest = halves(spans, edge)

        def parted(array):
            return array[:, :edge], array[:, edge:]
    else:
        edge = keys // 2
        first = template.KeySpans(jnp.minimum(spans.start, edge), jnp.minimum(spans.end, edge),
                                  spans.query_ids, spans.key_ids, keys)
        rest = template.KeySpans(jnp.maximum(spans.start, edge), spans.end, spans.query_ids, spans.key_ids,
                                 keys)

        def parted(array):
            return array, array

    def resumed(query, key, value):
        (key_first, key_rest), (value_first, value_rest) = parted(key), parted(value)
        out, m, total = template.reference(query, key_first, value_first, first, start, residuals=True)
        acc = jnp.swapaxes(out.astype(jnp.float32), 1, 2) * total[..., None]
        state = template.Initial(jnp.where(total > 0, m, -jnp.inf), total, acc)
        return template.reference(query, key_rest, value_rest, rest, template.Hooks(initial=state),
                                  residuals=True)

    def whole(query, key, value):
        return template.reference(query, key, value, spans, start, residuals=True)

    assert_as_exact(jax.jit(resumed), jax.jit(whole), drawing(queries, keys, dtype),
                    f"template resumed {case} {hooks} {jnp.dtype(dtype).name} {split}", seeds=SEEDS)


@pytest.mark.parametrize("case", [CASES[name] for name in CASES], ids=list(CASES))
def test_the_spans_form_says_what_their_intervals_hold(case):
    """`KeySpans.form`, which a kernel resolves at trace time, agrees with the spans' own visibility:
    'full' every key for every query, 'causal' keys 0 to i for query i, 'spans' anything narrower."""
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    seen = np.asarray(template.dense(spans))
    if spans.form == "full":
        assert seen.all()
    elif spans.form == "causal":
        np.testing.assert_array_equal(seen, np.broadcast_to(np.tri(queries, keys, dtype=bool), seen.shape))
    else:
        assert spans.form == "spans"
    plain = {name: value for name, value in case.items() if name != "causal"}
    assert (spans.form != "spans") == (not plain and case.get("causal") in (None, True))


@pytest.mark.parametrize("case", [*CASES.values(), *RUN_CASES.values()], ids=[*CASES, *RUN_CASES])
def test_a_query_reads_a_key_exactly_where_its_spans_hold_one(case):
    """`template.reads`, the queries a call's spans leave a key, is where their visibility holds one."""
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    np.testing.assert_array_equal(np.asarray(template.reads(spans)),
                                  np.asarray(template.dense(spans)).any(-1))


def test_key_validity_hides_keys_by_their_ids_and_keeps_a_cross_calls_alignment():
    """Key validity is the documents' padding id on the invalid keys (`template.spans`): a cross call's
    spans, which its alignment to the documents' last keys bounds, are the same with it and without it, its
    invalid keys id 0, its queries' ids its own; a call of no documents gives its queries and valid keys 1."""
    case = CASES["cross-causal-window-valid"]
    queries, keys = geometry(case)
    given = arguments(case)
    valid = given.pop("valid")
    plain = template.spans(queries, keys, ROWS, **given)
    hidden = template.spans(queries, keys, ROWS, **given, valid=valid)
    for field in ("start", "end", "query_ids"):
        np.testing.assert_array_equal(np.asarray(getattr(hidden, field)), np.asarray(getattr(plain, field)))
    assert plain.key_ids is not None
    np.testing.assert_array_equal(np.asarray(hidden.key_ids), np.where(valid, plain.key_ids, 0))
    bare = template.spans(queries, keys, ROWS, causal=True, valid=valid)
    np.testing.assert_array_equal(np.asarray(bare.query_ids), 1)
    np.testing.assert_array_equal(np.asarray(bare.key_ids), np.asarray(valid, np.int32))


@pytest.mark.parametrize("ids, keywords", [
    ([[1, 2, 1, 2, 2, 3, 1, 0]], {"causal": True, "starts": [3]}),
    ([[1, 2, 1, 2, 2, 3, 1, 0]], {"sliding_window": 2}),
    ([[2, 1, 2]], {"causal": True, "starts": [1], "lengths": [2]}),
], ids=["causal-starts", "two-sided-window", "interval-of-another-document"])
def test_reads_count_the_keys_of_interleaved_documents(ids, keywords):
    """Interleaved documents, whose keys no interval holds alone: a query reads a key where its document
    holds one inside its interval, so a query whose interval holds only another document's keys (the
    third row's last, over key 1 alone) reads none."""
    ids = jnp.asarray(ids, jnp.int32)
    starts, lengths = (None if name not in keywords else jnp.asarray(keywords[name], jnp.int32)
                       for name in ("starts", "lengths"))
    spans = template.spans(ids.shape[1], ids.shape[1], 1, segment_ids=ids,
                           causal=keywords.get("causal", False),
                           sliding_window=keywords.get("sliding_window"), starts=starts, lengths=lengths)
    np.testing.assert_array_equal(np.asarray(template.reads(spans)),
                                  np.asarray(template.dense(spans)).any(-1))


@pytest.mark.parametrize("valid, ids", [
    ([[1, 0, 0, 1, 1, 1, 0, 1]], None),
    ([[1, 0, 0, 1, 1, 1, 0, 1], [1, 1, 0, 0, 0, 1, 1, 1]],
     [[1, 1, 1, 1, 2, 2, 2, 2], [3, 3, 3, 4, 4, 4, 4, 4]]),
], ids=["padding-inside", "documents"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "two-sided"])
def test_a_window_counts_the_places_it_is_given(valid, ids, causal):
    """A window over a call's places (`counted_places`, each padding slot at the place before it) keeps the
    real keys within w - 1 places of the query, in its document: behind query 3 of the first row, padding
    inside, a window of two keeps key 0, where one over slots would drop it. The second row's second
    document opens with padding that took the first document's place, which no real key's window reads."""
    valid = jnp.asarray(valid, bool)
    ids = None if ids is None else jnp.asarray(ids, jnp.int32)
    places = counted_places(ids, valid)
    assert places is not None
    real = np.asarray(valid)[:, None, :]
    for window in (1, 2, 3):
        spans = template.spans(8, 8, valid.shape[0], causal=causal, sliding_window=window,
                               positions=(places, places), segment_ids=ids)
        expected = windowed(places[:, :, None], places[:, None, :], window, causal=causal)
        if causal:
            expected = expected & np.tril(np.ones((8, 8), bool))
        if ids is not None:
            expected = expected & document_mask(ids)
        np.testing.assert_array_equal(np.asarray(template.dense(spans)) & real, np.asarray(expected) & real)
        if window == 2 and causal:
            assert np.asarray(template.dense(spans))[0, 3, 0]


def test_a_window_holding_no_key_keeps_its_lower_bound_against_an_image():
    """A cache call's query at place 100 under a causal window of two over keys at places 0 and 1: its
    window holds no key, so neither is seen, though its own key (its document's last, aligned) begins an
    image, which reads only within the window; a whole block reads past it."""
    ids = CrossDocuments(jnp.asarray([[1]], jnp.int32), jnp.asarray([[1, 1]], jnp.int32))
    places = (jnp.asarray([[100]], jnp.int32), jnp.asarray([[0, 1]], jnp.int32))
    groups = jnp.asarray([[-1, 0]], jnp.int32)
    images = template.spans(1, 2, 1, causal=True, sliding_window=2, positions=places, segment_ids=ids,
                            image_groups=groups)
    np.testing.assert_array_equal(np.asarray(template.dense(images)), [[[False, False]]])
    blocks = template.spans(1, 2, 1, causal=True, sliding_window=2, positions=places, segment_ids=ids,
                            block_groups=groups)
    np.testing.assert_array_equal(np.asarray(template.dense(blocks)), [[[False, True]]])


@pytest.mark.parametrize(("query", "key"), [(jnp.int64, jnp.uint64), (jnp.int32, jnp.uint32)])
def test_places_no_integer_dtype_holds_both_of_are_refused(query, key):
    """Signed and unsigned places no enabled integer dtype holds both of (JAX would promote int64 beside
    uint64 to float64, and canonicalize int32 beside uint32 to int32 without x64) are refused for one dtype
    rather than moved in floats or a narrowed one; int32 beside uint32 widens to int64 under x64."""
    with jax.enable_x64(query == jnp.int64):
        places = (jnp.asarray([[0]], query), jnp.asarray([[0]], key))
        with pytest.raises(ValueError):
            template.spans(1, 1, 1, sliding_window=1, positions=places)
    with jax.enable_x64(new_val=True):
        mixed = template.spans(1, 2, 1, sliding_window=2, positions=(
            jnp.asarray([[0]], jnp.int32), jnp.asarray([[0, 1]], jnp.uint32)))
        signed = template.spans(1, 2, 1, sliding_window=2, positions=(
            jnp.asarray([[0]], jnp.int64), jnp.asarray([[0, 1]], jnp.int64)))
        np.testing.assert_array_equal(np.asarray(template.dense(mixed)), np.asarray(template.dense(signed)))


def test_a_window_moves_its_bound_in_a_dtype_that_holds_both_places():
    """Queries' places at the top of int8, keys' in int16 past it: a two-sided window of two around 127
    keeps the key at 128, its bound moved in the wider dtype rather than held at int8's largest."""
    places = (jnp.asarray([[127]], jnp.int8), jnp.asarray([[127, 128]], jnp.int16))
    spans = template.spans(1, 2, 1, sliding_window=2, positions=places)
    np.testing.assert_array_equal(np.asarray(template.dense(spans)), [[[True, True]]])


@pytest.mark.parametrize("window", [2 ** 31, 2 ** 32 - 1, 2 ** 32 + 7], ids=["half", "span", "past"])
def test_a_window_over_the_widest_places_neither_wraps_nor_aliases(window):
    """Places at both ends of int32, under windows as wide as their whole range and past it, keep the keys
    the window keeps counted exactly (in int64), causal and two-sided: no bound wraps around the dtype."""
    places = jnp.asarray([[-2 ** 31, -2 ** 31 + 5, -1, 0, 2 ** 31 - 6, 2 ** 31 - 1]], jnp.int32)
    wide = np.asarray(places, np.int64)
    queries, keys = wide[:, :, None], wide[:, None, :]
    for causal in (True, False):
        spans = template.spans(6, 6, 1, causal=causal, sliding_window=window, positions=(places, places))
        after = 0 if causal else window - 1
        expected = (keys >= queries - (window - 1)) & (keys <= queries + after)
        np.testing.assert_array_equal(np.asarray(template.dense(spans)), expected)


def test_hooks_are_a_pytree_whose_values_a_compiled_call_takes():
    """A compiled call takes the hooks as its argument: new parameter and state values run the one
    program it traced, and their gradients flow back into a `Hooks` of the same structure."""
    queries, keys = geometry(CASES["causal-window"])
    spans = template.spans(queries, keys, ROWS, **arguments(CASES["causal-window"]))
    query, key, value = drawing(queries, keys)(0)
    traces = []

    def attend(hooks):
        traces.append(hooks)
        return template.reference(query, key, value, spans, hooks)

    compiled = jax.jit(attend)
    alibi = hooks_of(HOOKS["softcap-sinks-alibi"], queries)
    steeper = template.Hooks(alibi.score_mod, (template.Param(2 * alibi.score_params[0].value,
                                                              template.Kind.HEAD),),
                             template.Initial(alibi.initial.m + 1, alibi.initial.total, alibi.initial.acc))
    outputs = compiled(alibi), compiled(steeper)
    assert len(traces) == 1
    assert not np.array_equal(np.asarray(outputs[0]), np.asarray(outputs[1]))
    grads = jax.grad(lambda hooks: jnp.sum(attend(hooks)))(alibi)
    assert jax.tree.structure(grads) == jax.tree.structure(alibi)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(grads))
    assert np.abs(np.asarray(grads.score_params[0].value)).sum() > 0


def test_a_call_of_no_keys_reads_its_initial_state_or_zero():
    """With no keys, every call form leaves each query none; the output is zero from no state or sinks,
    and a resumed partial's own `acc / total`."""
    cross = CrossDocuments(jnp.asarray([[1, 1, 0]], jnp.int32), jnp.zeros((1, 0), jnp.int32))
    for segment_ids in (None, cross):
        spans = template.spans(3, 0, 1, causal=True, sliding_window=4, chunk=2, segment_ids=segment_ids,
                               lengths=jnp.zeros((1,), jnp.int32))
        assert template.dense(spans).shape == (1, 3, 0)
        blocks = template.block_map(spans, 16, 16)
        assert not np.asarray(blocks.hi - blocks.lo).any()
        query = jnp.ones((1, 3, 2, 4), jnp.float32)
        empty = jnp.zeros((1, 0, 1, 4), jnp.float32)
        np.testing.assert_array_equal(np.asarray(template.reference(query, empty, empty, spans)), 0)
        sinks = template.Hooks(initial=template.sinks(jnp.asarray([0.5, -1.0]), 1, 3, 4))
        np.testing.assert_array_equal(np.asarray(template.reference(query, empty, empty, spans, sinks)), 0)
        state = template.Initial(jnp.zeros((1, 2, 3)), jnp.full((1, 2, 3), 2.0), jnp.full((1, 2, 3, 4), 3.0))
        resumed = template.reference(query, empty, empty, spans, template.Hooks(initial=state))
        np.testing.assert_array_equal(np.asarray(resumed), 1.5)


def test_document_ids_wider_than_int32_keep_their_documents_apart():
    """Ids past int32, under x64, keep their documents apart: `2 ** 32 + 1` is not document 1, and
    `2 ** 32` is no padding."""
    with jax.enable_x64(new_val=True):
        ids = jnp.asarray([[1, 1, 2 ** 32 + 1, 2 ** 32 + 1, 2 ** 32]], jnp.int64)
        spans = template.spans(5, 5, 1, segment_ids=ids)
        np.testing.assert_array_equal(np.asarray(template.dense(spans)), np.asarray(document_mask(ids)))


@pytest.mark.parametrize("ids, places, chunk", [
    ([[1, 1, 0, 1, 1]], [[0, 1, 0, 2, 3]], 2),
    ([[1, 2, 1, 2, 1, 2]], [[0, 0, 1, 1, 2, 2]], 2),
    ([[1, 2, 2]], [[0, 0, 1]], 2 ** 31 - 1),
], ids=["padding-inside", "interleaved", "widest"])
def test_chunks_keep_the_keys_dews_chunk_mask_keeps_over_any_document_layout(ids, places, chunk):
    """A call's own documents, padding inside one, two interleaved, or a chunk as wide as int32 allows:
    the spans keep exactly the keys `document_mask & chunk_mask` keeps over the call's places, causal or
    not."""
    ids, places = jnp.asarray(ids, jnp.int32), jnp.asarray(places, jnp.int32)
    tokens = ids.shape[1]
    for causal in (False, True):
        spans = template.spans(tokens, tokens, 1, causal=causal, chunk=chunk, positions=(places, places),
                               segment_ids=ids)
        expected = document_mask(ids) & chunk_mask(places, places, chunk)[:, 0]
        if causal:
            expected &= np.tril(np.ones((tokens, tokens), bool))
        np.testing.assert_array_equal(np.asarray(template.dense(spans)), np.asarray(expected))


def test_cross_documents_chunk_on_their_given_places():
    """Another sequence's documents with their own places: a query's chunk holds its document's keys
    whose places share it, as `cross_document_mask & chunk_mask` keeps them."""
    cross = CrossDocuments(jnp.asarray([[2, 2, 1, 1, 0]], jnp.int32),
                           jnp.asarray([[1, 1, 1, 2, 2, 2]], jnp.int32))
    query_places = jnp.asarray([[1, 2, 0, 2, 0]], jnp.int32)
    key_places = jnp.asarray([[0, 1, 2, 0, 1, 2]], jnp.int32)
    spans = template.spans(5, 6, 1, chunk=2, positions=(query_places, key_places), segment_ids=cross)
    expected = document_mask(cross) & chunk_mask(query_places, key_places, 2)[:, 0]
    np.testing.assert_array_equal(np.asarray(template.dense(spans)), np.asarray(expected))


def test_a_call_of_no_queries_maps_no_tiles():
    """With no queries, the block map has no query blocks and reads no key block."""
    blocks = template.block_map(template.spans(0, 40, 1, causal=True), 16, 16)
    assert blocks.lo.shape == (1, 0)
    np.testing.assert_array_equal(np.asarray(blocks.q_hi), 0)


def test_chunks_over_the_widest_places_neither_wrap_nor_alias():
    """Places at int32's least, and int64 places past int32, chunk as `chunk_mask` chunks them."""
    ids = jnp.asarray([[1, 2]], jnp.int32)
    places = jnp.asarray([[-2 ** 31, -2 ** 31 + 1]], jnp.int32)
    spans = template.spans(2, 2, 1, chunk=3, positions=(places, places), segment_ids=ids)
    expected = document_mask(ids) & chunk_mask(places, places, 3)[:, 0]
    np.testing.assert_array_equal(np.asarray(template.dense(spans)), np.asarray(expected))
    with jax.enable_x64(new_val=True):
        ids = jnp.asarray([[1, 1, 1]], jnp.int32)
        places = jnp.asarray([[2 ** 32, 2 ** 32 + 1, 2 ** 32 + 2]], jnp.int64)
        spans = template.spans(3, 3, 1, chunk=2, positions=(places, places), segment_ids=ids)
        expected = document_mask(ids) & chunk_mask(places, places, 2)[:, 0]
        np.testing.assert_array_equal(np.asarray(template.dense(spans)), np.asarray(expected))


def test_cross_documents_of_two_id_widths_align_as_one():
    """Another sequence's documents whose queries' and keys' ids differ in width: each query aligns
    within its own document, by an independent causal document mask: the int8 queries' 127 is not the
    int16 keys' 383, which narrowed to int8 would be."""
    for query_ids, key_ids in (([[127]], [[127]]), ([[127, 3]], [[3, 127, 383, 383]])):
        cross = CrossDocuments(jnp.asarray(query_ids, jnp.int8), jnp.asarray(key_ids, jnp.int16))
        spans = template.spans(len(query_ids[0]), len(key_ids[0]), 1, causal=True, segment_ids=cross)
        queries, keys = np.asarray(query_ids[0]), np.asarray(key_ids[0])
        places = [np.flatnonzero(keys == document)[-1] - (np.sum(queries[index + 1:] == document))
                  for index, document in enumerate(queries)]
        expected = (queries[:, None] == keys[None, :]) & (np.arange(len(keys))[None, :]
                                                            <= np.asarray(places)[:, None])
        np.testing.assert_array_equal(np.asarray(template.dense(spans))[0], expected)


def test_cross_documents_of_ids_wider_than_int32_align_apart():
    """Another sequence's documents with ids past int32, causal: each query aligns within its own
    document, as `cross_document_mask` aligns it."""
    with jax.enable_x64(new_val=True):
        ids = jnp.asarray([[1, 1, 2 ** 32 + 1, 2 ** 32 + 1]], jnp.int64)
        cross = CrossDocuments(ids, ids)
        spans = template.spans(4, 4, 1, causal=True, segment_ids=cross)
        expected = cross_document_mask(cross, causal=True, sliding_window=None)[:, 0]
        np.testing.assert_array_equal(np.asarray(template.dense(spans)), np.asarray(expected))
        np.testing.assert_array_equal(np.asarray(expected)[0],
                                      np.tril(np.ones((4, 4), bool)) & np.asarray(document_mask(ids)[0]))


def test_a_chunk_wider_than_its_places_dtype_holds_every_place_of_one_sign():
    """A chunk wider than int32 counts, over int32 places, holds the largest places together and the
    negative ones apart, as `chunk_mask` does (over int64 places, an ordinary chunk)."""
    ids = jnp.asarray([[1, 1, 1]], jnp.int32)
    places = jnp.asarray([[-1, 2 ** 31 - 2, 2 ** 31 - 1]], jnp.int32)
    spans = template.spans(3, 3, 1, chunk=2 ** 31, positions=(places, places), segment_ids=ids)
    with jax.enable_x64(new_val=True):
        wide = places.astype(jnp.int64)
        expected = document_mask(ids) & chunk_mask(wide, wide, 2 ** 31)[:, 0]
    np.testing.assert_array_equal(np.asarray(template.dense(spans)), np.asarray(expected))


def test_a_block_of_unsigned_ids_is_whole_only_within_one_document():
    """Unsigned document ids: a block whose tokens carry two documents is tagged as none, so a block map
    over interleaved documents calls no tile whole."""
    ids = jnp.asarray([[1, 2, 1, 2, 1, 2, 1, 2]], jnp.uint32)
    blocks = template.block_map(template.spans(8, 8, 1, segment_ids=ids), 2, 2)
    np.testing.assert_array_equal(np.asarray(blocks.query_doc), uniform_documents(ids, 2))
    np.testing.assert_array_equal(np.asarray(blocks.key_doc), uniform_documents(ids, 2))
    seen = np.asarray(template.dense(template.spans(8, 8, 1, segment_ids=ids)))
    for block in range(4):
        for tile in range(int(blocks.full_lo[0, block]), int(blocks.full_hi[0, block])):
            if blocks.query_doc[0, block] == blocks.key_doc[0, tile] != 0:
                assert seen[0, 2 * block:2 * block + 2, 2 * tile:2 * tile + 2].all()


def test_a_wider_initial_state_is_read_in_its_own_dtype():
    """Under x64, a state whose denominator and numerator are float64 and maximum float32 starts a
    float32 query's softmax in float64: a call of no keys reads `acc / total` unrounded."""
    with jax.enable_x64(new_val=True):
        state = template.Initial(jnp.zeros((1, 1, 1), jnp.float32), jnp.full((1, 1, 1), 1 + 2 ** -24),
                                 jnp.ones((1, 1, 1, 1), jnp.float64))
        query = jnp.ones((1, 1, 1, 1), jnp.float32)
        empty = jnp.zeros((1, 0, 1, 1), jnp.float32)
        spans = template.spans(1, 0, 1)
        out, _, total = template.reference(query, empty, empty, spans, template.Hooks(initial=state),
                                           residuals=True)
        assert float(total[0, 0, 0]) == 1 + 2 ** -24
        assert float(out[0, 0, 0, 0]) == np.float32(1 / (1 + 2 ** -24))


def test_sinks_keep_their_own_wider_dtype():
    """Float64 sinks start the softmax in float64, unrounded."""
    with jax.enable_x64(new_val=True):
        state = template.sinks(jnp.asarray([1 + 2 ** -30], jnp.float64), 1, 1, 1)
        assert state.m.dtype == state.total.dtype == state.acc.dtype == jnp.float64
        assert float(state.m[0, 0, 0]) == 1 + 2 ** -30
