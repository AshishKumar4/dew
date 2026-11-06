"""A cached call over a tree of candidate tokens (`dew.nn.inputs.AttentionMetadata.tree`), tree speculative
decoding's verify block: each token's parent among the call's, its attention masked by its key runs. A
chain is the tree of one path, read as the plain cached call reads its tokens; a mixer that reads no tree
refuses one, and a tree the call cannot carry is refused.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import TINY_DECODER, float64_twin
from reference_error import assert_as_exact

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


def test_a_tree_is_refused_where_a_layer_cannot_read_it_or_the_call_cannot_carry_it():
    """A mamba mixer reads no tree (it would run the tokens as a chain), and an n-gram lookup reads the
    call's tokens in their order, so each model refuses a tree call; a tree outside a cached call,
    without the key runs that mask its attention, of another length than the call's tokens, with a parent
    not before its child, or branching without its tokens' positions is refused."""
    from dew.nn.engram import Engram

    mamba = CausalTransformer(
        vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, head_dim=8, mlp_features=32,
        max_seq_len=16, dtype="float32", layer_types=("mamba", "full_attention"),
        kinds={"mamba": {"mixer": {"class": "mamba2", "fields": {
            "num_heads": 4, "head_dim": 8, "state_size": 8, "n_groups": 1, "chunk_size": 4}}}})
    engram = CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, "dtype": "float32"}, engram=Engram(
        layer_ids=(0,), num_embeddings=(168,), max_ngram_size=3, vocab_size=VOCAB, n_heads=2, head_dim=4,
        compressed_vocab_size=VOCAB))
    decoder = CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, "dtype": "float32"})
    ids = jax.random.randint(jax.random.key(0), (2, PROMPT + BLOCK), 0, VOCAB)
    block, runs = ids[:, PROMPT:], chain_runs(2, PROMPT, BLOCK)
    placed = PROMPT + jnp.asarray([[0, 1, 1]] * 2)
    for model in (mamba, engram, decoder):
        params = model.init(jax.random.key(1), ids)
        cache = {**params, "cache": cached(model, params, ids[:, :PROMPT])}
        tree = {"attention_tree": (-1, 0, 0), "attention_key_runs": runs, "rotary_positions": placed}
        refused = [tree] if model is not decoder else [
            {**tree, "decode": False},
            {"attention_tree": (-1, 0, 0), "rotary_positions": placed},
            {**tree, "attention_tree": (-1, 0)},
            {**tree, "attention_tree": (-1, 2, 0)},
            {"attention_tree": (-1, 0, 0), "attention_key_runs": runs}]
        for options in refused:
            with pytest.raises(ValueError):
                model.apply(cache, block, mutable=["cache"], **{"decode": True, **options})


TREE = (-1, 0, 0, 1, 2)
"""A root and two children, each with a child of its own: two paths of three tokens past the cache."""


def hybrid_model() -> CausalTransformer:
    """Qwen3.5's stack at toy size: a gated delta net, whose recurrent memory and conv tail are each row's
    own state, then full attention."""
    return CausalTransformer(
        vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, head_dim=8, mlp_features=32,
        max_seq_len=16, dtype="float32", layer_types=("linear_attention", "full_attention"),
        kinds={"linear_attention": {"mixer": {"class": "gated_delta_net", "fields": {
            "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 6,
            "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4}}}})


def paths(tree: tuple[int, ...]) -> list[list[int]]:
    """Each tree token's path from the root through it, oldest first."""
    held = []
    for token in range(len(tree)):
        path = [token]
        while tree[path[-1]] >= 0:
            path.append(tree[path[-1]])
        held.append(path[::-1])
    return held


def depths(tree: tuple[int, ...]) -> list[int]:
    """Each tree token's depth, its root's 0."""
    return [len(path) - 1 for path in paths(tree)]


def tree_runs(rows: int, prompt: int, tree: tuple[int, ...]) -> jax.Array:
    """The key runs `[rows, S, R, 2]` of a tree call after `prompt` cached tokens: each token reads the
    cache and its path's slots, one run each, an empty run filling the deepest path's count."""
    count = 1 + max(map(len, paths(tree)))
    runs = np.zeros((len(tree), count, 2), np.int32)
    for token, path in enumerate(paths(tree)):
        runs[token, 0] = (0, prompt)
        for at, held in enumerate(path, start=1):
            runs[token, at] = (prompt + held, prompt + held + 1)
    return jnp.broadcast_to(jnp.asarray(runs)[None], (rows, *runs.shape))


def test_a_hybrids_tree_call_reads_each_token_as_its_path_decoded_alone():
    """A gated delta net's tree call (`dew.nn.linear.tree_conv`, `tree_rule`) beside attention's, each
    token at its depth's position: each token's logits are on par with its root-to-token path decoded as
    a plain cached call from the same cache, from the float64 run's (check 1), its conv over its
    ancestors and past the root the cache's history, its recurrence from its parent's state."""
    model = hybrid_model()
    wide = float64_twin(model)
    rows = 2

    def draw(seed):
        key = jax.random.fold_in(jax.random.key(4527), seed)
        ids = jax.random.randint(key, (rows, PROMPT + len(TREE)), 0, VOCAB)
        params = nn.unbox(model.init(jax.random.fold_in(jax.random.key(1), seed), ids))
        return params, ids[:, :PROMPT], ids[:, PROMPT:]

    @jax.jit
    def tree_logits(params, prompt, block):
        cache = cached(model, params, prompt)
        placed = jnp.broadcast_to(PROMPT + jnp.asarray(depths(TREE)), (rows, len(TREE)))
        logits, _ = model.apply({**params, "cache": cache}, block, decode=True, mutable=["cache"],
                                attention_tree=TREE, attention_key_runs=tree_runs(rows, PROMPT, TREE),
                                rotary_positions=placed)
        return logits

    def decoded(decoder):
        @jax.jit
        def each_path(params, prompt, block):
            cache = cached(decoder, params, prompt)
            ends = []
            for path in paths(TREE):
                logits, _ = decoder.apply({**params, "cache": cache}, block[:, jnp.asarray(path)],
                                          decode=True, mutable=["cache"])
                ends.append(logits[:, -1])
            return jnp.stack(ends, axis=1)
        return each_path

    def truth(params, prompt, block):
        with jax.enable_x64(new_val=True):
            held = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), params)
            return decoded(wide)(held, prompt, block)

    assert_as_exact(tree_logits, decoded(model), draw, "gated delta net tree", seeds=64, truth_fn=truth)
