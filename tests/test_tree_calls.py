"""A cached call over a tree of candidate tokens (`dew.nn.inputs.AttentionMetadata.tree`), tree speculative
decoding's verify block: each token's parent among the call's, its attention masked by its key runs. A
chain is the tree of one path, read as the plain cached call reads its tokens; a mixer that reads no tree
refuses one, and a tree the call cannot carry is refused.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import TINY_DECODER

from dew.nn.backbones.causal_transformer import CausalTransformer

VOCAB = 37
PROMPT, BLOCK = 5, 3


def cached(decoder, params, prompt):
    """`decoder`'s cache after `prompt` `[B, P]`, prefilled in one cached call."""
    _, made = decoder.apply(params, prompt.shape[0], method=CausalTransformer.init_cache, mutable=["cache"])
    _, state = decoder.apply({**params, "cache": made["cache"]}, prompt, decode=True, mutable=["cache"])
    return state["cache"]


def chain_runs(rows: int, prompt: int, block: int) -> jax.Array:
    """The key runs `[rows, block, 1, 2]` of a chain of `block` tokens after `prompt` cached ones: token `i`
    reads the cache and the chain through itself."""
    ends = prompt + jnp.arange(block) + 1
    return jnp.broadcast_to(jnp.stack([jnp.zeros_like(ends), ends], -1)[None, :, None], (rows, block, 1, 2))


def test_a_chain_called_as_a_tree_reads_what_the_plain_cached_call_reads():
    """A chain is a tree of one path (each token's parent the one before it, the first's the cache's last):
    called as a tree, its attention masked by the chain's key runs, an attention decoder computes the plain
    cached call's logits bit for bit, and leaves the cache it leaves."""
    decoder = CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, "dtype": "float32"})
    ids = jax.random.randint(jax.random.key(0), (2, PROMPT + BLOCK), 0, VOCAB)
    params = decoder.init(jax.random.key(1), ids)
    cache = cached(decoder, params, ids[:, :PROMPT])
    block = ids[:, PROMPT:]
    plain, left = decoder.apply({**params, "cache": cache}, block, decode=True, mutable=["cache"])
    tree, held = decoder.apply({**params, "cache": cache}, block, decode=True, mutable=["cache"],
                               attention_tree=(-1, 0, 1), attention_key_runs=chain_runs(2, PROMPT, BLOCK))
    np.testing.assert_array_equal(np.asarray(tree), np.asarray(plain))
    jax.tree.map(lambda mine, theirs: np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs)),
                 held, left)


def test_a_tree_is_refused_where_no_mixer_reads_it_or_the_call_cannot_carry_it():
    """A gated delta net reads no tree (it would run the tokens as a chain), so a hybrid refuses a tree
    call by name; a tree outside a cached call, without the key runs that mask its attention, of another
    length than the call's tokens, or with a parent not before its child is refused."""
    hybrid = CausalTransformer(
        vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, head_dim=8, mlp_features=32,
        max_seq_len=16, dtype="float32", layer_types=("linear_attention", "full_attention"),
        kinds={"linear_attention": {"mixer": {"class": "gated_delta_net", "fields": {
            "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 6,
            "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4}}}})
    decoder = CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, "dtype": "float32"})
    ids = jax.random.randint(jax.random.key(0), (2, PROMPT + BLOCK), 0, VOCAB)
    block, runs = ids[:, PROMPT:], chain_runs(2, PROMPT, BLOCK)
    for model in (hybrid, decoder):
        params = model.init(jax.random.key(1), ids)
        cache = {**params, "cache": cached(model, params, ids[:, :PROMPT])}
        refused = [{"attention_tree": (-1, 0, 0), "attention_key_runs": runs}] if model is hybrid else [
            {"decode": False, "attention_tree": (-1, 0, 0), "attention_key_runs": runs},
            {"attention_tree": (-1, 0, 0)},
            {"attention_tree": (-1, 0), "attention_key_runs": runs},
            {"attention_tree": (-1, 2, 0), "attention_key_runs": runs}]
        for options in refused:
            with pytest.raises(ValueError):
                model.apply(cache, block, mutable=["cache"], **{"decode": True, **options})
