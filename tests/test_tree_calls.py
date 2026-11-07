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
    """Lightning attention reads no tree (it would run the tokens as a chain), an n-gram lookup reads the
    call's tokens in their order and an indexed latent attention selects among every key before each query,
    so each model refuses a tree call; a tree outside a cached call,
    without the key runs that mask its attention, of another length than the call's tokens, with a parent
    not before its child, or branching without its tokens' positions (coordinates of validity alone
    stating none) is refused."""
    from dew.nn.engram import Engram
    from dew.nn.hyper_connections import HyperConnections

    lightning = CausalTransformer(
        vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, head_dim=8, mlp_features=32,
        max_seq_len=16, dtype="float32", layer_types=("linear_attention", "full_attention"),
        kinds={"linear_attention": {"mixer": {"class": "lightning_attention", "fields": {"block_size": 4}}}})
    engram = CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, "dtype": "float32"}, engram=Engram(
        layer_ids=(0,), num_embeddings=(168,), max_ngram_size=3, vocab_size=VOCAB, n_heads=2, head_dim=4,
        compressed_vocab_size=VOCAB), hyper_connections=HyperConnections(hc_mult=2, head="weighted"))
    indexed = CausalTransformer(
        vocab_size=VOCAB, emb_features=32, num_layers=1, num_heads=2, head_dim=16, mlp_features=64,
        max_seq_len=16, dtype="float32", mixer={"class": "mla", "fields": {
            **MLA, "index_topk": 4, "index_n_heads": 2, "index_head_dim": 16}})
    decoder = CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, "dtype": "float32"})
    ids = jax.random.randint(jax.random.key(0), (2, PROMPT + BLOCK), 0, VOCAB)
    block, runs = ids[:, PROMPT:], chain_runs(2, PROMPT, BLOCK)
    placed = PROMPT + jnp.asarray([[0, 1, 1]] * 2)
    for model in (lightning, engram, indexed, decoder):
        params = model.init(jax.random.key(1), ids)
        cache = {**params, "cache": cached(model, params, ids[:, :PROMPT])}
        tree = {"attention_tree": (-1, 0, 0), "attention_key_runs": runs, "rotary_positions": placed}
        refused = [tree] if model is not decoder else [
            {**tree, "decode": False},
            {"attention_tree": (-1, 0, 0), "rotary_positions": placed},
            {**tree, "attention_tree": (-1, 0)},
            {**tree, "attention_tree": (-1, 2, 0)},
            {"attention_tree": (-1, 0, 0), "attention_key_runs": runs},
            # Coordinates of validity alone state no position.
            {"attention_tree": (-1, 0, 0), "attention_key_runs": runs,
             "attention_mask": jnp.ones((2, BLOCK))}]
        for options in refused:
            with pytest.raises(ValueError):
                model.apply(cache, block, mutable=["cache"], **{"decode": True, **options})


TREE = (-1, 0, 0, 1, 2)
"""A root and two children, each with a child of its own: two paths of three tokens past the cache."""


RECURRENT = {
    "gated_delta_net": {"linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 6,
                        "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4},
    "kimi_delta_attention": {"linear_num_heads": 2, "linear_head_dim": 8, "linear_conv_kernel_dim": 4},
    "mamba": {"intermediate_size": 32, "state_size": 4, "time_step_rank": 2},
    "mamba2": {"num_heads": 4, "head_dim": 8, "state_size": 8, "n_groups": 1, "chunk_size": 4},
    "short_conv": {"conv_L_cache": 3}}
"""The mixers holding row state that a tree call reads, at toy size: Qwen3.5's gated delta net, Kimi's
delta attention, Mamba-1 and -2, and LFM2's gated short convolution."""


MLA = {"q_lora_rank": 8, "kv_lora_rank": 8, "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 8}
"""DeepSeek's latent attention at toy size, dense (no indexer), as Kimi Linear's full-attention layers
hold it."""


def hybrid_model(mixer: str, attention: dict | None = None) -> CausalTransformer:
    """A hybrid stack at toy size: a recurrent `mixer`, whose memory and conv tail are each row's own
    state, then full attention, the standard layer's or the `attention` mixer's."""
    return CausalTransformer(
        vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, head_dim=8, mlp_features=32,
        max_seq_len=16, dtype="float32", layer_types=("linear_attention", "full_attention"),
        kinds={"linear_attention": {"mixer": {"class": mixer, "fields": RECURRENT[mixer]}},
               **({} if attention is None else {"full_attention": {"mixer": attention}})})


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


@pytest.mark.parametrize("mixer", sorted(RECURRENT))
@pytest.mark.parametrize("tree", [TREE, (-1,)], ids=["branching", "one-token"])
def test_a_hybrids_tree_call_reads_each_token_as_its_path_decoded_alone(tree, mixer):
    """A recurrent mixer's tree call (`dew.nn.linear.tree_scan`, a convolution's `tree_conv`) beside
    attention's, each token at its depth's position and its attention's keys at theirs: each token's
    logits are on par with its root-to-token path decoded as a plain cached call from the same cache,
    from the float64 run's (check 1), its conv over its ancestors and past the root the cache's history,
    its recurrence from its parent's state; a one-token tree too, which takes the tree's path where a
    step would take a single token's."""
    assert_reads_paths(hybrid_model(mixer), tree, f"{mixer} tree")


@pytest.mark.parametrize("tree", [TREE, (-1,)], ids=["branching", "one-token"])
def test_a_latent_attention_tree_call_reads_each_token_as_its_path_decoded_alone(tree):
    """Kimi Linear's stack at toy size, delta attention then DeepSeek's latent attention (`MLAMixer`): its
    tree call, the latent attention masked over its cache by the call's key runs and its rope heads turned
    at each token's depth, reads each token as its path decoded alone, from the float64 run's (check 1)."""
    model = hybrid_model("kimi_delta_attention", {"class": "mla", "fields": MLA})
    assert_reads_paths(model, tree, "kimi linear tree")


def test_a_latent_attention_cached_call_reads_by_its_runs_past_its_queries_validity():
    """A latent attention's cached call that states its key runs reads by them alone, its queries' validity
    replaced as the standard attention's is (`call_visibility`): a padded query reads the keys its runs
    name that the cache holds, bit for bit as a real query at the same position whose runs name those
    alone, rather than none."""
    model = CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, head_dim=8,
                              mlp_features=32, max_seq_len=16, dtype="float32",
                              mixer={"class": "mla", "fields": MLA})
    ids = jax.random.randint(jax.random.key(0), (2, PROMPT + BLOCK), 0, VOCAB)
    params = model.init(jax.random.key(1), ids)
    cache = {**params, "cache": cached(model, params, ids[:, :PROMPT])}
    block, runs = ids[:, PROMPT:], chain_runs(2, PROMPT, BLOCK)
    placed = jnp.broadcast_to(PROMPT + jnp.arange(BLOCK), (2, BLOCK))
    padded, _ = model.apply(cache, block, decode=True, mutable=["cache"], attention_key_runs=runs,
                            rotary_positions=placed, attention_mask=jnp.asarray([[1, 1, 0]] * 2, bool))
    # The last query's runs without its own slot, which a padded token's call leaves unwritten.
    narrowed = runs.at[:, BLOCK - 1, 0, 1].set(PROMPT + BLOCK - 1)
    real, _ = model.apply(cache, block, decode=True, mutable=["cache"], attention_key_runs=narrowed,
                          rotary_positions=placed)
    np.testing.assert_array_equal(np.asarray(padded[:, -1]), np.asarray(real[:, -1]))


@pytest.mark.parametrize("mixer", sorted(RECURRENT))
def test_a_tree_token_a_row_does_not_hold_is_skipped_as_a_plain_cached_call_skips_it(mixer):
    """A tree call whose rows hold different tokens (a verify's rows drafting trees of fewer tokens, the
    rest padding): a recurrent stack reads each token as its path decoded alone with the same tokens held,
    a token a row does not hold leaving its descendants' conv history and state as a plain cached call
    leaves them, its real ancestors before it (`dew.nn.linear.tree_windows`, `tree_rule`), from the float64
    run's (check 1). Here the first row holds no first child, so its child convolves the root and the
    cache's history alone."""
    model = CausalTransformer(
        vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, head_dim=8, mlp_features=32,
        max_seq_len=16, dtype="float32", layer_types=("linear_attention",) * 2,
        kinds={"linear_attention": {"mixer": {"class": mixer, "fields": RECURRENT[mixer]}}})
    held = np.ones((2, len(TREE)), bool)
    held[0, 1] = False
    assert_reads_paths(model, TREE, f"{mixer} tree past a token not held", valid=jnp.asarray(held))


def assert_reads_paths(model: CausalTransformer, tree: tuple[int, ...], label: str, *,
                       valid: jax.Array | None = None) -> None:
    """`model`'s tree call over `tree` past a cached prompt, each token at its depth's position and its
    attention's keys at theirs, on par with each token's root-to-token path decoded as a plain cached call
    from the same cache, from the float64 run's (check 1), over drawn weights and ids; the tokens each row
    holds `valid` (`[2, S]`, every one where None), each path's held alike, for a stack with no attention,
    whose keys a tree's runs name whether held or not."""
    wide = float64_twin(model)
    rows = 2

    def draw(seed):
        key = jax.random.fold_in(jax.random.key(4527), seed)
        ids = jax.random.randint(key, (rows, PROMPT + len(tree)), 0, VOCAB)
        params = nn.unbox(model.init(jax.random.fold_in(jax.random.key(1), seed), ids))
        return params, ids[:, :PROMPT], ids[:, PROMPT:]

    @jax.jit
    def tree_logits(params, prompt, block):
        cache = cached(model, params, prompt)
        placed = jnp.broadcast_to(PROMPT + jnp.asarray(depths(tree)), (rows, len(tree)))
        # The keys' places as a tree verify states them, each cache slot's: the prompt's, then each tree
        # token's its depth's.
        slots = jnp.arange(model.max_seq_len).at[PROMPT:PROMPT + len(tree)].set(
            PROMPT + jnp.asarray(depths(tree)))
        keys = jnp.broadcast_to(slots, (rows, model.max_seq_len))
        if valid is not None:
            logits, _ = model.apply({**params, "cache": cache}, block, decode=True, mutable=["cache"],
                                    attention_tree=tree, rotary_positions=placed, attention_mask=valid)
            return logits
        logits, _ = model.apply({**params, "cache": cache}, block, decode=True, mutable=["cache"],
                                attention_tree=tree, attention_key_runs=tree_runs(rows, PROMPT, tree),
                                attention_key_positions=keys, rotary_positions=placed)
        return logits

    def decoded(decoder):
        @jax.jit
        def each_path(params, prompt, block):
            cache = cached(decoder, params, prompt)
            ends = []
            for path in paths(tree):
                held = {} if valid is None else {"attention_mask": valid[:, jnp.asarray(path)]}
                logits, _ = decoder.apply({**params, "cache": cache}, block[:, jnp.asarray(path)],
                                          decode=True, mutable=["cache"], **held)
                ends.append(logits[:, -1])
            return jnp.stack(ends, axis=1)
        return each_path

    def truth(params, prompt, block):
        with jax.enable_x64(new_val=True):
            held = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), params)
            return decoded(wide)(held, prompt, block)

    assert_as_exact(tree_logits, decoded(model), draw, label, seeds=64, truth_fn=truth)
