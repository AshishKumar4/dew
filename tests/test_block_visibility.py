"""Block-causal attention layers (`AttentionMixer.bidirectional_blocks`, `bidirectional_images`): each token
of a block (`AttentionMetadata.block_groups`, `image_groups`) sees the block's later tokens too, within its
document, on the layers that read them, and every other layer stays causal; a window bounds an image's keys
behind the query and not a block's. A layer reads exactly the keys the rule keeps, written out here apart
from Dew (`block_causal`), its QK-Clip maxima taken over them: over packed documents, over windows shorter
than a block or an image, with padding inside the row, and on a layer reading both groups. A cached rollout
decodes on par with the full pass (check 1), its cache keeping each slot's groups, so a block's token reads
the block's earlier keys past the window from any earlier call. A non-causal kind keeps its kernel, a
server keeps a block layer's prefill apart, and a prediction depth reads the groups of the tokens it reads.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import TINY_DECODER, model_draw
from reference_error import assert_as_exact

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.kernels import generation
from dew.nn.mixers import AttentionMixer

VOCAB = 37
LENGTH = 8


def model(layers: int = TINY_DECODER["num_layers"], **kinds):
    """A float32 decoder of `layers` layers of the kinds `kinds` names, in turn, each with its fields."""
    types = tuple(kinds) * (layers // len(kinds))
    return CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, "dtype": "float32", "num_layers": layers,
                                "layer_types": types, "kinds": kinds})


def block_causal(blocks=None, images=None, *, segments=None, valid=None, window=None) -> np.ndarray:
    """The keys each query keeps, `[B, Q, K]`: a key at or before the query, or of an image of the query's
    (equal ids, -1 none), within a causal window of w, which keeps the keys fewer than w real tokens behind
    the query and every one ahead (Gemma's `AND(sliding_window, OR(causal, blockwise))`); or of a block of
    the query's, whatever the window (transformers' `block_sequence_ids`); inside the query's document (0
    padding, which sees nothing) and among the real keys `valid`."""
    rows, length = np.shape(next(groups for groups in (blocks, images) if groups is not None))

    def same(groups):
        if groups is None:
            return np.zeros((rows, length, length), bool)
        groups = np.asarray(groups)
        return (groups[:, :, None] == groups[:, None, :]) & (groups[:, :, None] >= 0)

    keep = np.tril(np.ones((length, length), bool))[None] | same(images)
    real = np.ones((rows, length), bool) if valid is None else np.asarray(valid, bool)
    if window is not None:
        places = np.maximum(np.cumsum(real, axis=1) - 1, 0)
        keep = keep & ((places[:, :, None] - places[:, None, :]) < window)
    keep = keep | same(blocks)
    if segments is not None:
        segments = np.asarray(segments)
        keep = keep & (segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] != 0)
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
    "images-and-blocks-past-a-window": (
        {"sliding_attention": {"window": 2, "mixer": AttentionMixer(bidirectional_images=True,
                                                                    bidirectional_blocks=True)}},
        {"block_groups": BLOCKS, "image_groups": IMAGES}, block_causal(BLOCKS, IMAGES, window=2)),
    "window-over-padding": ({"sliding_attention": {"window": 2,
                                                   "mixer": AttentionMixer(bidirectional_blocks=True)}},
                            {"block_groups": WINDOWED, "attention_mask": GAPS},
                            block_causal(WINDOWED, valid=GAPS, window=2)),
    "images-over-padding": ({"sliding_attention": {"window": 2,
                                                   "mixer": AttentionMixer(bidirectional_images=True)}},
                            {"image_groups": WINDOWED, "attention_mask": GAPS},
                            block_causal(images=WINDOWED, valid=GAPS, window=2)),
}
"""Each layout's kind, call fields and the keys the rule keeps. Past a window of two, the first row's image
of five tokens sees only its keys ahead and the one behind, and its second row's block of four sees all of
itself. Over padding, the first row holds padding between query 3 and key 0, which a window of two real
tokens keeps, where one of two slots would drop it; the second row's three-token run is a block read whole
or an image its window cuts."""


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
    if layout is LAYOUTS["images-and-blocks-past-a-window"]:
        assert not rule[0, 4, 1] and rule[0, 1, 4] and rule[1, 6, 3]
    np.testing.assert_array_equal(np.asarray(blocked), np.asarray(masked))
    jax.tree.map(lambda mine, theirs: np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs)),
                 sown, maxima)


HF = Path(__file__).parent / "fixtures" / "hf"
BLOCKED = {"attention": {"full_attention": "block_bidirectional"}}


def test_a_loaded_model_reads_blocks_on_the_kinds_an_override_names(tmp_path):
    """`Pretrained.load(..., modeling_overrides=)` turns block reading on for the layer kinds it names,
    on a family whose release states none, through the reader a release's own overrides take: a
    Llama's attention reads its blocks as the block-causal rule keeps them, bitwise, and without
    groups computes the plain release's logits, bitwise; a wrapped hybrid's named kind reads blocks and its
    other kind is the release's; the saved model states the override and reloads reading blocks;
    and a recurrent kind refuses to read blocks."""
    from dew.interop import Pretrained

    plain = Pretrained.load(HF / "llama-tiny", dtype="float32", attention_impl="xla")
    blocked = Pretrained.load(HF / "llama-tiny", dtype="float32", attention_impl="xla",
                              modeling_overrides=BLOCKED)
    assert blocked.model.kind_of("full_attention").mixer.bidirectional_blocks
    ids = jax.random.randint(jax.random.key(0), (2, LENGTH), 0, blocked.model.vocab_size)
    groups = jnp.asarray(BLOCKS)
    read = blocked.model.apply(blocked.variables, ids, block_groups=groups)
    rule = block_causal(BLOCKS)
    assert (rule & ~np.tril(np.ones((LENGTH, LENGTH), bool))).any()
    np.testing.assert_array_equal(np.asarray(read), np.asarray(blocked.model.apply(
        blocked.variables, ids, attention_pairwise_mask=jnp.asarray(rule))))
    np.testing.assert_array_equal(np.asarray(blocked.model.apply(blocked.variables, ids)),
                                  np.asarray(plain.model.apply(plain.variables, ids)))

    hybrid = Pretrained.load(HF / "qwen38-dense-tiny", dtype="float32", attention_impl="xla",
                             modeling_overrides=BLOCKED)
    release = Pretrained.load(HF / "qwen38-dense-tiny", dtype="float32", attention_impl="xla")
    assert hybrid.model.language_model.kind_of("full_attention").mixer.bidirectional_blocks
    assert hybrid.model.language_model.kind_of("linear_attention") == release.model.language_model.kind_of(
        "linear_attention")

    blocked.save(tmp_path / "saved")
    stated = json.loads((tmp_path / "saved" / "config.json").read_text())["modeling_overrides"]
    assert stated == BLOCKED
    reloaded = Pretrained.load(tmp_path / "saved", dtype="float32", attention_impl="xla")
    again = reloaded.model.apply(reloaded.variables, ids, block_groups=groups)
    np.testing.assert_array_equal(np.asarray(again), np.asarray(read))
    with pytest.raises(ValueError):
        Pretrained.load(HF / "qwen38-dense-tiny", dtype="float32", attention_impl="xla",
                        modeling_overrides={"attention": {"linear_attention": "block_bidirectional"}})


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


@pytest.mark.parametrize("shared", [False, True], ids=["own-keys", "shared-keys"])
def test_a_cached_rollout_reads_its_blocks_as_the_full_pass_does(shared):
    """Gemma 4's layout, images on its sliding layers and blocks on its full ones: a prefill, a cached call of
    several tokens and single steps, each call's groups at its tokens' slots, on par with the full pass over
    the same groups from the cached rollout's first prediction on (check 1, from float64). With shared keys
    the last two layers read the first two's (Gemma 4's trailing run), so a layer reading groups over keys
    another layer caches holds no cache of its own."""
    kinds = {"sliding_attention": {"window": 3, "mixer": AttentionMixer(bidirectional_images=True)},
             "full_attention": {"mixer": AttentionMixer(bidirectional_blocks=True)}}
    decoder = (model(4, **kinds).clone(kv_shared_layers=(2, 3)) if shared else model(**kinds))
    blocks, images = jnp.asarray(CACHED_BLOCKS), jnp.asarray(CACHED_IMAGES)

    def full(decoder):
        return jax.jit(lambda params, ids: decoder.apply(params, ids, block_groups=blocks,
                                                         image_groups=images)[:, 3:])

    rolled = rollout(decoder, CALLS, block_groups=blocks, image_groups=images)
    wide = decoder.clone(dtype=jnp.float64, attention_impl="reference")
    assert_as_exact(lambda params, ids: rolled(params, ids)[:, 3:], full(decoder),
                    model_draw(decoder, (2, 12), VOCAB), "cached blocks", seeds=CACHED_SEEDS,
                    truth_fn=full(wide))


def rollout(decoder, calls, **groups):
    """`decoder`'s logits over a row's tokens fed through its cache in `calls`, each a slice of the tokens
    with its slice of `groups`, as a function of its variables and tokens, compiled."""
    def run(params, ids):
        _, made = decoder.apply(params, ids.shape[0], method="init_cache", mutable=["cache"])
        cache, logits = made["cache"], []
        for lo, hi in calls:
            out, mutated = decoder.apply({**params, "cache": cache}, ids[:, lo:hi], decode=True,
                                         mutable=["cache"], **{name: value[:, lo:hi]
                                                               for name, value in groups.items()})
            cache = mutated["cache"]
            logits.append(out)
        return jnp.concatenate(logits, axis=1)
    return jax.jit(run)


PAST_THE_WINDOW = {
    "single-step": (((0, 3), (3, 4)), np.asarray([[0, 0, 0, 0], [-1, 1, 1, 1]], np.int32), 3),
    "cached-call": (((0, 3), (3, 6)), np.asarray([[0, 0, 0, 0, 0, 0], [-1, 2, 2, 2, 2, 2]], np.int32), 3),
}
"""Calls over a row, its blocks, which run past the window of two from earlier calls into the last, and
where the last call begins."""


@pytest.mark.parametrize("shared", [False, True], ids=["own-keys", "shared-keys"])
@pytest.mark.parametrize("case", PAST_THE_WINDOW.values(), ids=list(PAST_THE_WINDOW))
def test_a_blocks_last_call_reads_its_earlier_keys_past_the_window_from_the_cache(case, shared):
    """A block read whole that a rollout's calls part, under a window of two: its tokens in the last call,
    a single step or a cached call of several, read its keys in earlier calls past the window, by the groups
    the cache keeps beside them, on par with the full pass from float64 (check 1). One layer, or a second
    that shares the first's keys, so those tokens' outputs read only keys the cache holds as the full pass
    computes them; a block's tokens in an earlier call cannot see its later ones, which no cache gives."""
    calls, blocks, last = case
    kinds = {"sliding_attention": {"window": 2, "mixer": AttentionMixer(bidirectional_blocks=True)}}
    decoder = model(2, **kinds).clone(kv_shared_layers=(1,)) if shared else model(1, **kinds)
    groups = jnp.asarray(blocks)

    def full(decoder):
        return jax.jit(lambda params, ids: decoder.apply(params, ids, block_groups=groups)[:, last:])

    rolled = rollout(decoder, calls, block_groups=groups)
    wide = decoder.clone(dtype=jnp.float64, attention_impl="reference")
    assert_as_exact(lambda params, ids: rolled(params, ids)[:, last:], full(decoder),
                    model_draw(decoder, blocks.shape, VOCAB), "blocks past the window", seeds=CACHED_SEEDS,
                    truth_fn=full(wide))


def test_a_non_causal_kind_reading_blocks_keeps_its_kernel(without_deterministic_ops):
    """A non-causal kind sees its whole row already, so the groups it would read add nothing: a call with
    and without them reports the same attention routes on an A100 (`generation.routes`), cuDNN named, with
    no mask of their own. Only routes are traced, no program compiled, so on the cuda lane, whose
    --xla_gpu_deterministic_ops keeps cuDNN away from a run (`without_deterministic_ops`), the flag is
    taken out for the routes the claim is about."""
    decoder = model(full_attention={"mixer": AttentionMixer(bidirectional_blocks=True)}).clone(
        causal=False, attention_impl="cudnn", dtype=jnp.bfloat16)
    ids = jax.random.randint(jax.random.key(0), (2, LENGTH), 0, VOCAB)
    # The same parameters as this host's kernel makes them, where a named cuDNN does not run.
    params = decoder.clone(attention_impl="xla").init(jax.random.key(1), ids)

    def routes(**groups):
        with generation.assumed("sm80"), generation.routes() as found:
            jax.eval_shape(lambda params, ids: decoder.apply(params, ids, **groups), params, ids)
        return [(route.choice, route.kernel) for route in found]

    plain = routes()
    assert ("attention", "cudnn") in plain
    assert routes(block_groups=jnp.asarray(BLOCKS)) == plain


def test_a_server_keeps_a_block_layers_prefill_apart():
    """A server's mixed step reads each row's cache as a decode step does, which carries no groups, so a
    model with a kind reading blocks or images keeps its prompts' prefill in a forward of its own."""
    assert model(full_attention={}).mixed_admission_refusal() is None
    for mixer in (AttentionMixer(bidirectional_blocks=True), AttentionMixer(bidirectional_images=True)):
        assert model(full_attention={"mixer": mixer}).mixed_admission_refusal() is not None


def test_a_prediction_depth_reads_the_groups_of_the_tokens_it_reads():
    """Prediction depth 1 reads the tokens from the second on (`mtp_hidden_states`), so its layers read the
    groups from there: a block of the first two tokens is one token to it, which adds no key, and its
    states are those of no block, bitwise; a block of the second and third is two, which moves them."""
    decoder = model(full_attention={"mixer": AttentionMixer(bidirectional_blocks=True)}).clone(
        num_nextn_predict_layers=1)
    ids = jax.random.randint(jax.random.key(0), (2, LENGTH), 0, VOCAB)
    params = decoder.init(jax.random.key(1), ids)
    hidden = decoder.apply(params, ids, method=CausalTransformer.hidden_states)

    def depth(lo: int, hi: int) -> np.ndarray:
        groups = np.full((2, LENGTH), -1, np.int32)
        groups[:, lo:hi] = 0
        return np.asarray(decoder.apply(params, hidden, ids, block_groups=jnp.asarray(groups),
                                        method=CausalTransformer.mtp_hidden_states)[0])

    none = depth(0, 0)
    np.testing.assert_array_equal(depth(0, 2), none)
    assert not np.array_equal(depth(1, 3), none)


def test_a_multimodal_wrappers_cached_rollout_reads_its_blocks_as_its_full_pass_does():
    """A multimodal wrapper hands its language model the call's groups (`MultimodalTransformer`): text alone
    through its cache, a prefill, a cached call and single steps, with blocks inside each call, on par with
    its full pass over the same groups from float64 (check 1)."""
    from dew.nn.multimodal import MultimodalTransformer
    from dew.nn.vision import GemmaProjector, SiglipVision

    def wrapped(decoder):
        return MultimodalTransformer(
            decoder, SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2,
                                  image_size=8, patch_size=4),
            GemmaProjector(text_width=TINY_DECODER["emb_features"], patches_per_side=2, tokens_per_side=1),
            family="gemma3", image_token_id=1)

    decoder = model(full_attention={"mixer": AttentionMixer(bidirectional_blocks=True)})
    wrapper = wrapped(decoder)
    groups = jnp.asarray(CACHED_BLOCKS)

    def full(wrapper):
        return jax.jit(lambda params, ids: wrapper.apply(params, ids, block_groups=groups)[:, 3:])

    rolled = rollout(wrapper, CALLS, block_groups=groups)
    wide = wrapped(decoder.clone(dtype=jnp.float64, attention_impl="reference"))
    assert_as_exact(lambda params, ids: rolled(params, ids)[:, 3:], full(wrapper),
                    model_draw(wrapper, (2, 12), VOCAB), "multimodal cached blocks", seeds=CACHED_SEEDS,
                    truth_fn=full(wide))
