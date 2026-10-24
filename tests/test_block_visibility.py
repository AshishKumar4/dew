"""Block-causal attention layers (`AttentionMixer.bidirectional_blocks`, `bidirectional_images`): each token
of a block (`AttentionMetadata.block_groups`, `image_groups`) sees the block's later tokens too, within its
document, on the layers that read them, and every other layer stays causal. A layer reads exactly the keys
the rule keeps, written out here apart from Dew (`block_causal`), its QK-Clip maxima taken over them: over
packed documents, over a window with padding inside the row, and on a layer reading both groups. A cached
call over its blocks decodes on par with the full pass (check 1), the cache keeping each slot's groups.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import TINY_DECODER, model_draw
from reference_error import assert_as_exact

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mixers import AttentionMixer

VOCAB = 37
LENGTH = 8


def model(**kinds):
    """A two-layer float32 decoder of the layer kinds `kinds` names, in turn, each with its fields."""
    types = tuple(kinds) * (TINY_DECODER["num_layers"] // len(kinds))
    return CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, "dtype": "float32", "layer_types": types,
                                "kinds": kinds})


def block_causal(*groups, segments=None, valid=None, window=None) -> np.ndarray:
    """The keys each query keeps, `[B, Q, K]`, as transformers' `create_causal_mask(block_sequence_ids)` says:
    a key at or before the query, or of a block of the query's (equal ids, -1 none), inside the query's
    document (0 padding, which sees nothing) and among the real keys `valid`; a causal window of w keeps the
    keys fewer than w real tokens behind the query, and every one ahead of it."""
    rows, length = np.shape(groups[0])
    keep = np.broadcast_to(np.tril(np.ones((length, length), bool)), (rows, length, length)).copy()
    for blocks in groups:
        blocks = np.asarray(blocks)
        keep |= (blocks[:, :, None] == blocks[:, None, :]) & (blocks[:, :, None] >= 0)
    real = np.ones((rows, length), bool) if valid is None else np.asarray(valid, bool)
    if window is not None:
        places = np.maximum(np.cumsum(real, axis=1) - 1, 0)
        keep &= (places[:, :, None] - places[:, None, :]) < window
    if segments is not None:
        segments = np.asarray(segments)
        keep &= (segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] != 0)
    return keep & real[:, None, :]


BLOCKS = np.asarray([[-1, 0, 0, 0, -1, 1, 1, -1], [2, 2, -1, 3, 3, 3, 3, -1]], np.int32)
IMAGES = np.asarray([[4, 4, 4, 4, 4, -1, -1, -1], [-1, -1, -1, -1, -1, 5, 5, 5]], np.int32)
SEGMENTS = np.asarray([[1, 1, 1, 2, 2, 2, 2, 0], [1, 1, 1, 1, 2, 2, 2, 2]], np.int32)
GAPS = np.asarray([[1, 0, 0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1]], bool)
WINDOWED = np.asarray([[-1, -1, -1, 0, 0, -1, -1, -1], [0, 0, 0, -1, 1, 1, -1, -1]], np.int32)

LAYOUTS = {
    "blocks": ({"full_attention": {"mixer": AttentionMixer(bidirectional_blocks=True)}},
               {"block_groups": BLOCKS}, block_causal(BLOCKS)),
    "blocks-across-documents": ({"full_attention": {"mixer": AttentionMixer(bidirectional_blocks=True)}},
                                {"block_groups": BLOCKS, "segment_ids": SEGMENTS},
                                block_causal(BLOCKS, segments=SEGMENTS)),
    "images-and-blocks": ({"full_attention": {"mixer": AttentionMixer(bidirectional_images=True,
                                                                      bidirectional_blocks=True)}},
                          {"block_groups": BLOCKS, "image_groups": IMAGES}, block_causal(BLOCKS, IMAGES)),
    "window-over-padding": ({"sliding_attention": {"window": 2,
                                                   "mixer": AttentionMixer(bidirectional_blocks=True)}},
                            {"block_groups": WINDOWED, "attention_mask": GAPS},
                            block_causal(WINDOWED, valid=GAPS, window=2)),
}
"""Each layout's kind, call fields and the keys the rule keeps. The window's first row holds padding between
query 3 and key 0, which its window of two real tokens keeps, where one of two slots would drop it."""


@pytest.mark.parametrize("layout", LAYOUTS.values(), ids=list(LAYOUTS))
def test_a_block_layer_reads_the_keys_the_block_causal_rule_keeps(layout):
    """A layer reading block groups computes what it computes over an explicit mask of the rule's keys
    (`attention_pairwise_mask`), its logits and its QK-Clip maxima alike: both run the reference path over
    the same keys, so they agree bitwise, and a key added or dropped moves them. The rule keeps keys a
    causal mask does not, so the groups are read."""
    kinds, fields, rule = layout
    decoder = model(**kinds)
    ids = jax.random.randint(jax.random.key(0), (2, LENGTH), 0, VOCAB)
    params = decoder.init(jax.random.key(1), ids)
    segment_ids, attention_mask, block_groups, image_groups = (
        None if name not in fields else jnp.asarray(fields[name])
        for name in ("segment_ids", "attention_mask", "block_groups", "image_groups"))
    blocked, sown = decoder.apply(params, ids, segment_ids=segment_ids, attention_mask=attention_mask,
                                  block_groups=block_groups, image_groups=image_groups, mutable=["qk"])
    masked, maxima = decoder.apply(params, ids, segment_ids=segment_ids, attention_mask=attention_mask,
                                   attention_pairwise_mask=jnp.asarray(rule), mutable=["qk"])
    assert (rule & ~np.tril(np.ones((LENGTH, LENGTH), bool))).any()
    np.testing.assert_array_equal(np.asarray(blocked), np.asarray(masked))
    jax.tree.map(lambda mine, theirs: np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs)),
                 sown, maxima)


def test_a_layer_that_reads_no_groups_stays_causal():
    """Block and image groups reach every layer, and a layer whose kind reads neither computes what it
    computes without them, bitwise."""
    decoder = model(full_attention={})
    ids = jax.random.randint(jax.random.key(0), (2, LENGTH), 0, VOCAB)
    params = decoder.init(jax.random.key(1), ids)
    grouped = decoder.apply(params, ids, block_groups=jnp.asarray(BLOCKS), image_groups=jnp.asarray(IMAGES))
    np.testing.assert_array_equal(np.asarray(grouped), np.asarray(decoder.apply(params, ids)))


CALLS = ((0, 4), (4, 8), (8, 9), (9, 10), (10, 11), (11, 12))
"""A cached rollout's calls over twelve tokens: a prefill of four, a cached call of four, then single
steps."""
CACHED_BLOCKS = np.asarray([[-1, 0, 0, -1, 1, 1, 1, -1, -1, -1, -1, -1],
                            [2, 2, 2, 2, -1, 3, 3, 3, -1, 4, -1, -1]], np.int32)
CACHED_IMAGES = np.asarray([[-1, -1, 5, 5, -1, -1, 6, 6, -1, -1, -1, -1],
                            [-1, 7, 7, -1, 8, 8, 8, 8, -1, -1, -1, -1]], np.int32)
"""Blocks and images inside each call, as a cached rollout reads them: a block a call splits sees, in its
earlier call, none of its later tokens, which that call does not hold."""
CACHED_SEEDS = 16
"""Check 1's declared seeds for the cached rollout, sized by `seeds_for`."""


def test_a_cached_rollout_reads_its_blocks_as_the_full_pass_does():
    """Gemma 4's layout, images on its sliding layers and blocks on its full ones: a prefill, a cached call of
    several tokens and single steps, each call's groups written to the cache at its slots, on par with the
    full pass over the same groups from the cached rollout's first prediction on (check 1, from float64)."""
    decoder = model(sliding_attention={"window": 3, "mixer": AttentionMixer(bidirectional_images=True)},
                    full_attention={"mixer": AttentionMixer(bidirectional_blocks=True)})
    blocks, images = jnp.asarray(CACHED_BLOCKS), jnp.asarray(CACHED_IMAGES)

    def rolled(decoder):
        def run(params, ids):
            _, made = decoder.apply(params, ids.shape[0], method=CausalTransformer.init_cache,
                                    mutable=["cache"])
            cache, logits = made["cache"], []
            for lo, hi in CALLS:
                out, mutated = decoder.apply({**params, "cache": cache}, ids[:, lo:hi], decode=True,
                                             block_groups=blocks[:, lo:hi], image_groups=images[:, lo:hi],
                                             mutable=["cache"])
                cache = mutated["cache"]
                logits.append(out)
            return jnp.concatenate(logits, axis=1)[:, 3:]
        return jax.jit(run)

    def full(decoder):
        return jax.jit(lambda params, ids: decoder.apply(params, ids, block_groups=blocks,
                                                         image_groups=images)[:, 3:])

    wide = decoder.clone(dtype=jnp.float64, attention_impl="reference")
    assert_as_exact(rolled(decoder), full(decoder), model_draw(decoder, (2, 12), VOCAB), "cached blocks",
                    seeds=CACHED_SEEDS, truth_fn=full(wide))
