"""A shared-prefix fork in the backbone call (`dew.batch.Forks`): documents that continue another, each
reading its ancestors as its own beginning, so a prefix several rows share is computed once. Rows packed
over their shared prefixes (`dew.data.forked_rows.ForkedRows`) score and train as the rows laid out whole
on every mixer that reads forks; a cached call over a tree of candidates (tree speculative decoding's
verify) reads each token as its path decoded alone; a model that would compute forks otherwise lays the
rows whole.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import TINY_DECODER, float64_truth, float64_twin
from reference_error import assert_as_exact

from dew.batch import Forks, coordinates
from dew.data.forked_rows import ForkedRows
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.hyper_connections import HyperConnections
from dew.objectives.lm.objective import LMObjective

VOCAB = 37
PROMPT, BLOCK = 5, 3
WIDTH = 24
"""The windows' slots: one holds a group of the shared rows."""

ROWS = [[3, 4, 5, 6, 7, 8, 9, 10, 11], [3, 4, 5, 6, 7, 8, 9, 12], [3, 4, 5, 6, 7, 8, 9, 13, 14, 15],
        [3, 4, 5, 6, 7, 8, 9, 10, 16], [20, 21, 22, 23, 24]]
"""Four rows over one prompt of seven tokens, two of them sharing an eighth, and a row of its own."""

FORK_SEEDS = 64
"""Check 1's declared seeds for rows over forks against the rows laid out whole: `seeds_for`, paired, sized
the Mamba-2 hybrid's site, its A_log gradient a sum over every token of each head's, at 64 with sixteen
heads (at four it resolves within no count up to MOST_SEEDS, at thirty-two at 256; armada, 11-09)."""

RECURRENT = {
    "gated_delta_net": {"linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 6,
                        "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4},
    "kimi_delta_attention": {"linear_num_heads": 2, "linear_head_dim": 8, "linear_conv_kernel_dim": 4},
    "mamba": {"intermediate_size": 32, "state_size": 4, "time_step_rank": 2},
    "mamba2": {"num_heads": 16, "head_dim": 8, "state_size": 8, "n_groups": 1, "chunk_size": 4},
    "short_conv": {"conv_L_cache": 3},
    "lightning_attention": {"block_size": 4},
    "mamba3": {"head_dim": 8, "state_size": 8, "chunk_size": 4}}
"""The mixers holding row state that read forks, at toy size: Qwen3.5's gated delta net, Kimi's delta
attention, Mamba-1, -2 and -3, LFM2's gated short convolution and MiniMax's lightning attention."""

MLA = {"q_lora_rank": 8, "kv_lora_rank": 8, "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 8}
"""DeepSeek's latent attention at toy size, dense (no indexer), as Kimi Linear's full-attention layers
hold it."""


def hybrid_model(mixer: str, attention: dict | None = None) -> CausalTransformer:
    """A hybrid stack at toy size: a recurrent `mixer`, whose memory and conv tail are each row's own
    state, then full attention, the standard layer's or the `attention` mixer's."""
    return CausalTransformer(
        vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, head_dim=8, mlp_features=32,
        max_seq_len=WIDTH, dtype="float32", layer_types=("linear_attention", "full_attention"),
        kinds={"linear_attention": {"mixer": {"class": mixer, "fields": RECURRENT[mixer]}},
               **({} if attention is None else {"full_attention": {"mixer": attention}})})


def attention_model(**fields) -> CausalTransformer:
    return CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, "max_seq_len": WIDTH, "dtype": "float32",
                                **fields})


MODELS = {
    "attention": lambda: attention_model(),
    "windowed attention": lambda: attention_model(layer_types=("windowed", "windowed"),
                                                  kinds={"windowed": LayerKind(window=4)}),
    **{f"{mixer} hybrid": lambda mixer=mixer: hybrid_model(mixer) for mixer in RECURRENT},
    "kimi linear": lambda: hybrid_model("kimi_delta_attention", {"class": "mla", "fields": MLA}),
    "mixture": lambda: attention_model(mixture=Mixture(experts=4, top_k=2, expert_features=16)),
    "looped": lambda: attention_model(loop=Loop(2, exit_gate=True)),
}
"""The decoders whose every layer reads forks: attention, windowed attention, each recurrent mixer beside
attention, a latent attention beside Kimi's delta attention, a routed feed-forward, and a looped stack whose
every pass reads the forks."""


@pytest.mark.parametrize("kind", sorted(MODELS))
def test_rows_over_forks_score_and_train_as_the_rows_laid_out_whole(kind):
    """Rows packed over their shared prefixes (`ForkedRows`, a prompt computed once) give each row's
    next-token losses, and the gradient of their weighted sum in every parameter, what the rows laid out
    alone give (check 1, over drawn weights, float64 truth): each document over its ancestors, each token
    scored from its predecessor, a document's first from its parent's last (`Forks.predecessors`), and the
    gradient of a shared token summed over the rows that read it."""
    model = MODELS[kind]()
    objective = LMObjective(model, WIDTH - 1)
    assert objective.shares_prefixes(train=True)
    shared = ForkedRows.pack(ROWS, WIDTH)
    whole = ForkedRows.pack(ROWS, WIDTH, share=False, documents=1)
    assert shared.coordinates.forks is not None and whole.coordinates.forks is None
    assert shared.tokens.shape[0] < whole.tokens.shape[0]
    weights = jnp.asarray(np.random.default_rng(7).uniform(0.5, 1.5, (len(ROWS), max(map(len, ROWS)) - 1)),
                          jnp.float32)
    initialized = jax.jit(objective.init)

    def scored(objective, forks):
        def run(variables):
            def loss(params):
                losses = forks.targets(objective.token_scores({**variables, "params": params}, forks.tokens,
                                                              coordinates=forks.coordinates).losses)
                return jnp.sum(losses * weights), losses

            grads, losses = jax.grad(loss, has_aux=True)(variables["params"])
            return {"losses": losses, "grads": grads}
        return jax.jit(run)

    assert_as_exact(scored(objective, shared), scored(objective, whole),
                    lambda seed: (nn.unbox(initialized(jax.random.key(seed))),), f"{kind} over forks",
                    seeds=FORK_SEEDS,
                    truth_fn=float64_truth(scored(LMObjective(float64_twin(model), WIDTH - 1), whole)))


def test_forked_rows_lay_each_prefix_once_and_each_row_back_at_its_tokens():
    """`ForkedRows.pack` lays each group's longest shared prefix once and each row's rest after it as a
    document continuing it, every token at its own row's place, and maps each row's tokens back to their
    slots; a token whose carried field differs is not shared, and `share=False` lays every row whole."""
    forks = ForkedRows.pack(ROWS, WIDTH)
    tokens = np.asarray(forks.tokens).reshape(-1)
    positions = np.asarray(forks.coordinates.positions).reshape(-1)
    for number, row in enumerate(ROWS):
        at = np.asarray(forks.index[number, :len(row)])
        assert list(tokens[at]) == row and list(positions[at]) == list(range(len(row)))
        assert np.all(np.asarray(forks.index[number, len(row):]) == -1)
    laid = int(np.sum(np.asarray(forks.coordinates.segments) > 0))
    # The four rows over their seven shared tokens and each one's rest, the fifth whole.
    assert laid == 7 + 2 + 1 + 3 + 2 + 5
    sampled = [np.full(len(row), number) for number, row in enumerate(ROWS)]
    apart = ForkedRows.pack(ROWS, WIDTH, fields={"sampled": sampled})
    assert int(np.sum(np.asarray(apart.coordinates.segments) > 0)) == sum(map(len, ROWS))
    assert apart.coordinates.forks is None
    # Each token's tree, the group its document's chain of ancestors roots in (`Forks.trees`): the four
    # rows over the prompt one tree, the fifth another.
    trees = np.asarray(forks.coordinates.forks.trees(forks.coordinates.segments)).reshape(-1)
    owned = [{(int(slot) // WIDTH, int(trees[slot])) for slot in np.asarray(forks.index[number, :len(row)])}
             for number, row in enumerate(ROWS)]
    assert all(len(held) == 1 and next(iter(held))[1] > 0 for held in owned)
    assert owned[0] == owned[1] == owned[2] == owned[3] != owned[4]


def test_a_model_that_would_read_forks_otherwise_lays_its_rows_whole():
    """A decoder that reads its tokens in their row's order beyond its mixers (engram's n-gram hashes, a
    block router, an expert capacity), whose mixer reads no forks (DeepSeek V4's compressed windows, which
    pool a row's tokens in its order), that reads each token's later ones (a bidirectional decoder, whose
    prefix would read its branches), or that draws dropout in a training step, says why a call over forks
    would not compute each row as its own (`ForkedCalls`): its objective lays its rows whole
    (`shares_prefixes`), and a call over forks is refused, where the same decoder without the cause runs
    it. A joint tree (`Forks.joint`), one row read together, draws its dropout once and runs in training;
    its other causes refuse it all the same."""
    from dew.nn.engram import Engram

    forks = ForkedRows.pack(ROWS, WIDTH)
    refusing = {
        "engram": attention_model(engram=Engram(layer_ids=(0,), num_embeddings=(168,), max_ngram_size=3,
                                                vocab_size=VOCAB, n_heads=2, head_dim=4,
                                                compressed_vocab_size=VOCAB),
                                  hyper_connections=HyperConnections(hc_mult=2, head="weighted")),
        "block routing": attention_model(mixture=Mixture(experts=4, top_k=2, expert_features=16,
                                                         routing_block=2, block_experts=2)),
        "capacity": attention_model(mixture=Mixture(experts=4, top_k=2, expert_features=16,
                                                    capacity_factor=1.0)),
        "deepseek_v4": attention_model(mixer={"class": "deepseek_v4", "fields": {"rope_head_dim": 8}},
                                       kinds={"full_attention": LayerKind(window=8)}),
    }
    for name, model in refusing.items():
        assert not LMObjective(model, WIDTH - 1).shares_prefixes(), name
        params = model.init(jax.random.key(0), jnp.ones((1, WIDTH), jnp.int32))
        with pytest.raises(ValueError):
            model.apply(params, forks.tokens, coordinates=forks.coordinates)
    # A bidirectional decoder scores no next tokens; its call over forks is refused all the same.
    bidirectional = attention_model(causal=False)
    params = bidirectional.init(jax.random.key(0), jnp.ones((1, WIDTH), jnp.int32))
    assert bidirectional.fork_refusal() is not None
    with pytest.raises(ValueError):
        bidirectional.apply(params, forks.tokens, coordinates=forks.coordinates)
    dropping = attention_model(dropout_rate=0.1)
    assert LMObjective(dropping, WIDTH - 1).shares_prefixes()
    assert not LMObjective(dropping, WIDTH - 1).shares_prefixes(train=True)
    params = dropping.init(jax.random.key(0), jnp.ones((1, WIDTH), jnp.int32))
    dropping.apply(params, forks.tokens, coordinates=forks.coordinates)
    with pytest.raises(ValueError):
        dropping.apply(params, forks.tokens, coordinates=forks.coordinates, train=True,
                       rngs={"dropout": jax.random.key(1)})
    # A joint tree is one row read together (`Forks.joint`, a decision row's siblings): it draws its dropout
    # once, as that row does, so its training call runs; every other refusal holds.
    assert forks.coordinates.forks is not None
    joint = dataclasses.replace(forks.coordinates,
                                forks=dataclasses.replace(forks.coordinates.forks, joint=True))
    dropping.apply(params, forks.tokens, coordinates=joint, train=True,
                   rngs={"dropout": jax.random.key(1)})
    for model in refusing.values():
        held = model.init(jax.random.key(0), jnp.ones((1, WIDTH), jnp.int32))
        with pytest.raises(ValueError):
            model.apply(held, forks.tokens, coordinates=joint)


def test_a_mask_beside_forks_narrows_what_they_read():
    """A call's pairwise mask beside its forks reads a key where both let it (`call_visibility`): the same
    as the call whose mask is already the forks' visibility narrowed by it, over drawn weights (check 1)."""
    from dew.nn.attention_masks import visibility_mask

    model = attention_model(attention_impl="xla")
    forks = ForkedRows.pack(ROWS, WIDTH)
    laid = forks.coordinates
    windows = forks.tokens.shape
    drawn = np.random.default_rng(3).uniform(size=(*windows, WIDTH)) < 0.8
    mask = jnp.asarray(drawn) | jnp.eye(WIDTH, dtype=bool)
    assert laid.forks is not None and laid.segments is not None
    seen = visibility_mask(WIDTH, WIDTH, causal=False, sliding_window=None,
                           key_runs=laid.forks.runs(laid.segments))[:, 0]
    # The same tokens as one document a row, at the same places: the mask alone says what each reads.
    plain = coordinates(valid=laid.valid, positions=laid.positions)
    initialized = jax.jit(model.init)

    # A padding slot's output is no token's: each arm's real tokens alone are compared.
    real = jnp.asarray(laid.valid)[..., None]

    def beside(variables):
        return jnp.where(real, model.apply(variables, forks.tokens, coordinates=laid,
                                           attention_pairwise_mask=mask), 0)

    def narrowed(variables):
        return jnp.where(real, model.apply(variables, forks.tokens, coordinates=plain,
                                           attention_pairwise_mask=mask & seen), 0)

    assert_as_exact(jax.jit(beside), jax.jit(narrowed),
                    lambda seed: (initialized(jax.random.key(seed), jnp.ones((1, WIDTH), jnp.int32)),),
                    "a mask beside forks", seeds=FORK_SEEDS)


# ---------------------------------------------------------------- a cached call over a tree of candidates

TREE = (-1, 0, 0, 1, 2)
"""A root and two children, each with a child of its own: two paths of three tokens past the cache."""


def cached(decoder, params, prompt, segments=None):
    """`decoder`'s cache after `prompt` `[B, P]`, prefilled in one cached call, packed by `segments`."""
    _, made = decoder.apply(params, prompt.shape[0], method=CausalTransformer.init_cache, mutable=["cache"])
    laid = {} if segments is None else {"segment_ids": segments}
    _, state = decoder.apply({**params, "cache": made["cache"]}, prompt, decode=True, mutable=["cache"],
                             **laid)
    return state["cache"]


def paths(tree: tuple[int, ...]) -> list[list[int]]:
    """Each tree token's path from the root through it, oldest first."""
    held = []
    for token in range(len(tree)):
        path = [token]
        while tree[path[-1]] >= 0:
            path.append(tree[path[-1]])
        held.append(path[::-1])
    return held


def tree_coordinates(tree: tuple[int, ...], rows: int, base: int, valid=None):
    """A tree call's layout: a document a token continuing its parent's, the root's the row's cache, each
    at the root's `base` plus its depth (`Forks.positions`)."""
    segments = jnp.broadcast_to(jnp.arange(1, len(tree) + 1, dtype=jnp.int32), (rows, len(tree)))
    parents = np.asarray([[parent + 1 for parent in tree]] * rows, np.int32)
    forks = Forks(jnp.asarray(parents), depth=Forks.depth_of(parents))
    laid = coordinates(segments=segments, valid=valid, max_segments=len(tree),
                       positions=forks.positions(segments, jnp.full((rows,), base, jnp.int32), valid))
    return dataclasses.replace(laid, forks=forks)


@pytest.mark.parametrize("model", [*sorted(RECURRENT), "kimi linear"])
@pytest.mark.parametrize("tree", [TREE, (-1,), (-1, 0, 1)], ids=["branching", "one-token", "chain"])
def test_a_tree_call_reads_each_token_as_its_path_decoded_alone(tree, model):
    """A hybrid's cached call over a tree (a document a token, each continuing its parent's, the root's the
    row's cache): each token's logits are on par with its root-to-token path decoded as a plain cached call
    from the same cache, from the float64 run's (check 1), its conv over its ancestors and past the root the
    cache's history (`forked_conv`), its recurrence from its parent's state (`forked_scan`), its attention
    over the cache and its ancestors (`fork_runs`); a one-token tree and a chain too."""
    built = (hybrid_model("kimi_delta_attention", {"class": "mla", "fields": MLA}) if model == "kimi linear"
             else hybrid_model(model))
    assert_reads_paths(built, tree, f"{model} tree")


@pytest.mark.parametrize("tree", [TREE, (-1, 0, 1)], ids=["branching", "chain"])
def test_a_windowed_tree_call_reads_its_cached_keys_at_the_places_forks_give_them(tree):
    """A windowed decoder's cached call over a tree reads each cached key at its place (`fork_key_places`:
    the prefix counted back from the root's place, each node at its depth's), not its slot: each token's
    logits are on par with its path decoded alone, a window of two keys reaching past the tree into the
    prefix (check 1)."""
    model = attention_model(max_seq_len=16, layer_types=("windowed", "windowed"),
                            kinds={"windowed": LayerKind(window=2)})
    assert_reads_paths(model, tree, "windowed tree")


@pytest.mark.parametrize("mixer", sorted(RECURRENT))
def test_a_tree_token_a_row_does_not_hold_passes_its_parents_state_on(mixer):
    """A tree call whose rows hold different tokens (a verify's rows drafting trees of fewer tokens, the
    rest padding): a hybrid stack reads each token as its path decoded alone with the same tokens held,
    a document holding no real token passing its parent's conv history and state on, as a plain cached
    call passes a padded token's, and its attention reading the real tokens where the cache stores them,
    compacted past the one it does not hold (`fork_runs`) (check 1)."""
    model = hybrid_model(mixer)
    held = np.ones((2, len(TREE)), bool)
    held[0, 1] = False
    assert_reads_paths(model, TREE, f"{mixer} tree past a token not held", valid=jnp.asarray(held))


def test_a_tree_after_a_packed_prompt_reads_its_last_document_alone():
    """A tree verified after a prompt prefilled packed (two documents in a row) reads the row's last
    document and its own ancestors alone (`fork_anchor`: the cache's run from that document's first slot):
    each token's logits are on par with the tree after the last document prefilled alone (check 1), so the
    earlier document never leaks into the verify, as a packed prompt's continuation reads its last document
    alone."""
    model = attention_model(max_seq_len=16)
    rows = 2
    initialized = jax.jit(model.init)
    segments = jnp.asarray([[1, 1, 1, 2, 2, 2]] * rows, jnp.int32)

    def draw(seed):
        key = jax.random.fold_in(jax.random.key(4527), seed)
        ids = jax.random.randint(key, (rows, 6 + len(TREE)), 0, VOCAB)
        return nn.unbox(initialized(jax.random.fold_in(key, 1), ids)), ids[:, :6], ids[:, 6:]

    def after(*, packed: bool):
        @jax.jit
        def run(params, prompt, block):
            cache = (cached(model, params, prompt, segments) if packed
                     else cached(model, params, prompt[:, 3:]))
            # The tree continues the row's last document, three tokens in.
            laid = tree_coordinates(TREE, rows, 3)
            logits, _ = model.apply({**params, "cache": cache}, block, decode=True, mutable=["cache"],
                                    coordinates=laid)
            return logits
        return run

    assert_as_exact(after(packed=True), after(packed=False), draw, "a tree after a packed prompt",
                    seeds=FORK_SEEDS)


def assert_reads_paths(model: CausalTransformer, tree: tuple[int, ...], label: str, *,
                       valid: jax.Array | None = None) -> None:
    """`model`'s cached call over `tree` past a cached prompt (`tree_coordinates`), on par with each token's
    root-to-token path decoded as a plain cached call from the same cache, from the float64 run's (check 1),
    over drawn weights and ids; the tokens each row holds `valid` (`[2, S]`, every one where None), each
    path's held alike, for a stack with no attention, whose keys a tree's runs name whether held or not."""
    wide = float64_twin(model)
    rows = 2
    initialized = jax.jit(model.init)

    def draw(seed):
        key = jax.random.fold_in(jax.random.key(4527), seed)
        ids = jax.random.randint(key, (rows, PROMPT + len(tree)), 0, VOCAB)
        params = nn.unbox(initialized(jax.random.fold_in(jax.random.key(1), seed), ids))
        return params, ids[:, :PROMPT], ids[:, PROMPT:]

    @jax.jit
    def tree_logits(params, prompt, block):
        cache = cached(model, params, prompt)
        laid = tree_coordinates(tree, rows, PROMPT, valid)
        logits, _ = model.apply({**params, "cache": cache}, block, decode=True, mutable=["cache"],
                                coordinates=laid)
        # A token a row does not hold scores nothing: each arm's held tokens alone are compared.
        return logits if valid is None else jnp.where(valid[..., None], logits, 0)

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
            found = jnp.stack(ends, axis=1)
            return found if valid is None else jnp.where(valid[..., None], found, 0)
        return each_path

    assert_as_exact(tree_logits, decoded(model), draw, label, seeds=64, truth_fn=float64_truth(decoded(wide)))


@pytest.mark.parametrize("kind", ["attention", "gated_delta_net hybrid", "mixture"])
def test_a_groups_chains_score_over_their_shared_prompt_as_laid_whole(kind):
    """A GRPO pack that lays each group's prompt once (`pack`'s `share`, a chain's rest continuing it) scores
    every sampled id, and the gradient of their weighted sum in every parameter, as the chains packed whole
    (check 1, over drawn weights, float64 truth): the importance ratio reads each id against what collection
    sampled, token for token, in fewer rows."""
    from dew.objectives.rl.grpo import GRPOObjective
    from dew.objectives.rl.sessions import RESPONSE_MASK_KEY, SESSION_INDEX_KEY, Call, Session, Status, pack

    rng = np.random.default_rng(11)
    sessions = []
    for group in range(2):
        prompt = tuple(int(token) for token in rng.integers(1, VOCAB, 9))
        for sample in range(3):
            sampled = tuple(int(token) for token in rng.integers(1, VOCAB, int(rng.integers(2, 5))))
            call = Call(prompt, sampled, tuple(-0.5 for _ in sampled), "stop", 0)
            sessions.append(Session("t", str(group), sample, 0, (call,), Status.COMPLETED, float(sample)))
    shared, whole = pack(sessions, WIDTH, share=True), pack(sessions, WIDTH)
    assert shared[RESPONSE_MASK_KEY].shape[0] < whole[RESPONSE_MASK_KEY].shape[0]
    model = MODELS[kind]()
    grpo = GRPOObjective(model, WIDTH - 1)

    def sampled_at(batch):
        # Each sampled id's slot, by its session and its place in the row.
        rows, columns = np.nonzero(batch[RESPONSE_MASK_KEY] > 0)
        order = np.lexsort((columns, rows, batch[SESSION_INDEX_KEY][rows, columns]))
        return rows[order], columns[order]

    weights = jnp.asarray(rng.uniform(0.5, 1.5, len(sampled_at(whole)[0])), jnp.float32)

    def scored(objective, batch):
        rows, columns = sampled_at(batch)

        def run(variables):
            def loss(params):
                found = objective.packed_log_probs({**variables, "params": params}, batch)[rows, columns]
                return jnp.sum(found * weights), found

            grads, found = jax.grad(loss, has_aux=True)(variables["params"])
            return {"log_probs": found, "grads": grads}
        return jax.jit(run)

    truth = float64_truth(scored(GRPOObjective(float64_twin(model), WIDTH - 1), whole))
    initialized = jax.jit(grpo.init)
    assert_as_exact(scored(grpo, shared), scored(grpo, whole),
                    lambda seed: (nn.unbox(initialized(jax.random.key(seed))),),
                    f"{kind} group over its prompt", seeds=FORK_SEEDS, truth_fn=truth)


def test_two_chains_of_a_session_keep_each_sampled_id_apart():
    """Two calls of one session over one prompt that sample alike at first split into two chains: a pack
    that shares their prompt shares no sampled id, so each call's sampled ids, likelihoods and advantages
    stand on their own slots, as the chains packed whole hold them (review of the shared prompt)."""
    from dew.objectives.rl.sessions import (
        BEHAVIOR_LOG_PROBS_KEY,
        CALL_INDEX_KEY,
        RESPONSE_MASK_KEY,
        Call,
        Session,
        Status,
        pack,
    )

    calls = (Call((1,), (2, 3), (-0.1, -0.2), "stop", 0), Call((1,), (2, 4), (-0.3, -0.4), "stop", 0))
    sessions = [Session("t", "g", sample, 0, calls, Status.COMPLETED, float(sample)) for sample in range(2)]
    shared, whole = pack(sessions, 16, share=True), pack(sessions, 16)
    for batch in (shared, whole):
        drawn = batch[RESPONSE_MASK_KEY] > 0
        assert int(drawn.sum()) == 8
        np.testing.assert_array_equal(np.sort(batch[BEHAVIOR_LOG_PROBS_KEY][drawn]),
                                      np.sort(np.asarray([-0.1, -0.2, -0.3, -0.4] * 2, np.float32)))
        assert sorted(batch[CALL_INDEX_KEY][drawn].tolist()) == [0] * 4 + [1] * 4


FORKED_ROWS, FORKED_LENGTH = 8, 2048
"""The forked recurrence's cost check's call: a decision window's rows at their length."""


def _compiled_cost(function, *shapes) -> tuple[int, float]:
    """The temporaries and FLOPs of the compiled gradient of `function`'s summed output in every operand."""
    compiled = jax.jit(jax.grad(lambda *operands: function(*operands).astype(jnp.float32).sum(),
                                argnums=tuple(range(len(shapes))))).lower(*shapes).compile()
    cost = compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    return int(compiled.memory_analysis().temp_size_in_bytes), float(cost["flops"])


def _recurrence_call(layer: str, length: int, documents: int):
    """`layer`'s chunked recurrence over `FORKED_ROWS` rows of `length` tokens, three documents a row, and
    with `documents` past zero each entering a memory of its own (`Chunk.entered`), two of them the first's
    (a state and two questions over it): the function, its operands' shapes, its chunk and its memories'
    elements."""
    from dew.nn.kernels.recurrence_chunk import CHUNK_SIZE, chunk_gated_delta_rule
    from dew.nn.mixers.mamba2 import CHUNK_SIZE as SSD_CHUNK, _chunked

    def shaped(*dims):
        return jax.ShapeDtypeStruct((FORKED_ROWS, *dims), jnp.float32)

    starts = np.zeros((FORKED_ROWS, length), bool)
    starts[:, [0, length // 13, length // 2]] = True
    starts = jnp.asarray(starts)
    entered = None
    if documents:
        held = np.zeros((FORKED_ROWS, documents), np.int32)
        held[:, 1:3] = 1
        entered = jnp.asarray(held)
    if layer == "gated delta net":
        memory = (3, 8, 8)
        operands = [shaped(length, 1, 8), shaped(length, 1, 8), shaped(length, 3, 8), shaped(length, 3),
                    shaped(length, 3)]

        def call(query, key, value, gate, beta, *state):
            return chunk_gated_delta_rule(query, key, value, gate, beta, *state, starts=starts,
                                          max_segments=documents or 3, write_bound=1.0, entered=entered)[0]
        chunk = CHUNK_SIZE
    else:
        memory = (4, 8, 8)
        operands = [shaped(length, 4, 8), shaped(length, 4), shaped(length, 1, 8), shaped(length, 1, 8)]
        rate, skip = -jnp.ones((4,)), jnp.ones((4,))

        def call(x, dt, keys, queries, *state):
            return _chunked(x, jax.nn.softplus(dt), rate, keys, queries, skip, state[0] if state else None,
                            SSD_CHUNK, starts=starts, entered=entered)[0]
        chunk = SSD_CHUNK
    if documents:
        operands.append(shaped(documents, *memory))
    return call, operands, chunk, FORKED_ROWS * documents * math.prod(memory)


@pytest.mark.parametrize("documents", [6, 173], ids=["the windows' own", "a loose bound"])
@pytest.mark.parametrize("layer", ["gated delta net", "SSD"])
def test_a_forked_recurrence_costs_its_tokens_and_a_chunk_a_document(layer, documents):
    """A recurrence whose documents each enter a memory of their own (a call over forks) costs, in its
    gradient's temporaries and FLOPs, what its packed row costs laid one chunk longer a document
    (`AlignedDocuments`: each document from a chunk's edge), with each document's memory, its cotangent,
    its final and that final's cotangent beside it, fp32, each an operation an element: a decision window's
    rows at a documents bound of its own windows and at a loose one."""
    call, operands, chunk, memories = _recurrence_call(layer, FORKED_LENGTH, documents)
    temporaries, flops = _compiled_cost(call, *operands)
    packed = _recurrence_call(layer, FORKED_LENGTH + documents * chunk, 0)
    bound, work = _compiled_cost(packed[0], *packed[1])
    assert temporaries <= bound + 4 * 4 * memories, (temporaries, bound, memories)
    assert flops <= work + 4 * memories, (flops, work, memories)
