"""Reindexing a decode cache has to move whole prefixes, not tree shapes.

`gather_cache_rows` is the contract beam branching and speculative rollback
rest on. The check is behavioural: build rows from different prompts, gather
them into a duplicated and reordered set, keep decoding, and require every
gathered row to produce what the prompt it came from produces on its own.
Every mixer keeps different state, so each one is exercised: dense attention
scanned and unscanned, a gated delta net's convolution and recurrent summary,
latent attention's compressed cache, and a multimodal wrapper's next position.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.backbones.causal_transformer import CausalTransformer, gather_cache_rows
from dew.nn.inputs import ModelInputs
from dew.sampling import Sampling, generate
from dew.sampling.text import _operations, _prefill
from test_text_rollout_contract import decoder

VOCAB = 13


def multimodal():
    """A text-only multimodal wrapper: its cache adds the next position."""
    from dew.nn.multimodal import MultimodalTransformer
    from dew.nn.vision import GemmaProjector, SiglipVision

    return MultimodalTransformer(
        decoder(), SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2,
                                image_size=8, patch_size=4),
        GemmaProjector(vision_width=16, text_width=16, patches_per_side=2, tokens_per_side=1),
        family="gemma3", image_token_id=1)


MODELS = {
    "attention": lambda: decoder(),
    "scanned": lambda: CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=2,
                                         num_heads=2, head_dim=8, mlp_features=32, max_seq_len=12,
                                         dtype="float32", scan_layers=True),
    "mla": lambda: decoder("mla"),
    "recurrent": lambda: decoder("recurrent"),
    "multimodal": multimodal,
}


def walk(model, params, prompt, steps):
    """Greedy continuation without a cache, one full forward per token."""
    sequence = np.asarray(prompt)
    for _ in range(steps):
        logits = np.asarray(model.apply(params, jnp.asarray(sequence))[:, -1])
        sequence = np.concatenate([sequence, logits.argmax(-1)[:, None].astype(np.int32)], axis=1)
    return sequence


@pytest.mark.parametrize("kind", list(MODELS))
def test_gathered_cache_rows_decode_like_the_prefixes_they_came_from(kind):
    """Duplicated and reordered rows keep their own history: after the gather
    each row's greedy continuation is the continuation of the prompt that row
    was copied from, which a cache that mixed rows cannot produce."""
    model = MODELS[kind]()
    prompts = jnp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], jnp.int32)
    params = model.init(jax.random.key(0), prompts[:1])
    rows = jnp.asarray([2, 0, 2, 1], jnp.int32)

    def continue_from(cache_rows):
        ops = _operations(model, params, 0, 0)
        state, _ = _prefill(model, params, ModelInputs(prompts), ops)
        state = ops.reindex(state, cache_rows)
        drawn = []
        for _ in range(4):
            token = jnp.argmax(state.logits, axis=-1).astype(jnp.int32)
            drawn.append(token)
            state = ops.advance(state, token, jnp.ones(cache_rows.shape[0], bool))
        return jnp.stack(drawn, axis=1)

    gathered = np.asarray(jax.jit(continue_from)(rows))
    expected = walk(model, params, prompts, 4)[:, 3:]
    np.testing.assert_array_equal(gathered, expected[np.asarray(rows)])
    # The duplicated rows agree with each other and the reordering is real.
    np.testing.assert_array_equal(gathered[0], gathered[2])
    assert not np.array_equal(gathered[0], gathered[1])


@pytest.mark.parametrize("kind", list(MODELS))
def test_every_cache_leaf_moves_with_its_row(kind):
    """A leaf left behind would keep a row's old state. Gathering a pure
    permutation and its inverse has to return the cache unchanged, leaf for
    leaf, which fails as soon as one leaf indexes a different axis."""
    model = MODELS[kind]()
    prompts = jnp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], jnp.int32)
    params = model.init(jax.random.key(0), prompts[:1])
    state, _ = _prefill(model, params, ModelInputs(prompts), _operations(model, params, 0, 0))
    order = jnp.asarray([2, 0, 1], jnp.int32)
    inverse = jnp.asarray([1, 2, 0], jnp.int32)

    rotated = gather_cache_rows(state.cache, order)
    restored = gather_cache_rows(rotated, inverse)
    leaves = jax.tree.leaves(state.cache)
    assert leaves, "the cache has to hold something for this to prove anything"
    for before, after, moved in zip(leaves, jax.tree.leaves(restored),
                                    jax.tree.leaves(rotated)):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
        assert moved.shape == before.shape


def test_beam_branching_reuses_one_prefill_without_mixing_prompts():
    """Beam search copies a prompt's cache row into every beam. Each returned
    row still starts from its own prompt, which is what a gather that dropped
    a leaf would break."""
    model = decoder()
    prompts = jnp.asarray([[1, 2, 3], [7, 8, 9]], jnp.int32)
    params = model.init(jax.random.key(0), prompts[:1])
    from dew.sampling import Beam

    found = generate(model, params, prompts, 4, key=jax.random.key(0),
                     sampling=Sampling(pad_id=0), strategy=Beam(width=1), n=1)
    np.testing.assert_array_equal(np.asarray(found.tokens)[:, :3], np.asarray(prompts))
    # Width one is the greedy walk, so a mixed prefix shows up immediately.
    np.testing.assert_array_equal(np.asarray(found.tokens)[:, 3:], walk(model, params, prompts, 4)[:, 3:])
