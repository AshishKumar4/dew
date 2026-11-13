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
from forks_support import LAYOUTS, assert_forks_as_own_rows
from model_support import TINY_DECODER, float64_twin
from reference_error import assert_as_exact

from dew.batch import Forks, coordinates
from dew.data.forked_rows import ForkedRows
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.engram import Engram, PLELayer
from dew.nn.hyper_connections import HyperConnections
from dew.nn.inputs import MixerCall
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


def engram_model() -> CausalTransformer:
    """An attention decoder with an engram lookup at its first layer, its n-grams three ids long, over two
    residual streams."""
    lookup = Engram(layer_ids=(0,), num_embeddings=(168,), max_ngram_size=3, vocab_size=VOCAB, n_heads=2,
                    head_dim=4, compressed_vocab_size=VOCAB)
    return attention_model(engram=lookup, hyper_connections=HyperConnections(hc_mult=2, head="weighted"))


MODELS = {
    "attention": lambda: attention_model(),
    "windowed attention": lambda: attention_model(layer_types=("windowed", "windowed"),
                                                  kinds={"windowed": LayerKind(window=4)}),
    **{f"{mixer} hybrid": lambda mixer=mixer: hybrid_model(mixer) for mixer in RECURRENT},
    "kimi linear": lambda: hybrid_model("kimi_delta_attention", {"class": "mla", "fields": MLA}),
    "mixture": lambda: attention_model(mixture=Mixture(experts=4, top_k=2, expert_features=16)),
    "looped": lambda: attention_model(loop=Loop(2, exit_gate=True)),
    "engram": engram_model,
}
"""The decoders whose every layer reads forks: attention, windowed attention, each recurrent mixer beside
attention, a latent attention beside Kimi's delta attention, a routed feed-forward, a looped stack whose
every pass reads the forks, and engram's n-gram lookups, each token's n-grams its chain's."""


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
                    truth_fn=scored(LMObjective(float64_twin(model), WIDTH - 1), whole))


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
    """A decoder that reads its tokens in their row's order beyond its mixers (a block router, an expert
    capacity), whose mixer reads no forks (DeepSeek V4's compressed windows, which
    pool a row's tokens in its order), that reads each token's later ones (a bidirectional decoder, whose
    prefix would read its branches), or that draws dropout in a training step, says why a call over forks
    would not compute each row as its own (`ForkedCalls`): its objective lays its rows whole
    (`shares_prefixes`), and a call over forks is refused, where the same decoder without the cause runs
    it. A joint tree (`Forks.joint`), one row read together, draws its dropout once and runs in training;
    its other causes refuse it all the same."""
    forks = ForkedRows.pack(ROWS, WIDTH)
    refusing = {
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
    as the call whose mask is already the forks' visibility narrowed by it, over drawn weights (check 1),
    each held to the float64 twin's narrowed call: a model of a stated `dtype` computes in it whatever its
    inputs, so the reference widened would be the reference again, its rounding no truth's."""
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

    def narrowed(model):
        def run(variables):
            return jnp.where(real, model.apply(variables, forks.tokens, coordinates=plain,
                                               attention_pairwise_mask=mask & seen), 0)
        return run

    assert_as_exact(jax.jit(beside), jax.jit(narrowed(model)),
                    lambda seed: (initialized(jax.random.key(seed), jnp.ones((1, WIDTH), jnp.int32)),),
                    "a mask beside forks", seeds=FORK_SEEDS, truth_fn=narrowed(float64_twin(model)))


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
    forks = Forks(jnp.asarray(parents), depth=Forks.depth_of(parents), held=Forks.held_of(parents))
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
    alone; each held to the float64 twin's tree after the document alone, as a model of a stated `dtype`
    computes in it whatever its inputs."""
    model = attention_model(max_seq_len=16)
    rows = 2
    initialized = jax.jit(model.init)
    segments = jnp.asarray([[1, 1, 1, 2, 2, 2]] * rows, jnp.int32)

    def draw(seed):
        key = jax.random.fold_in(jax.random.key(4527), seed)
        ids = jax.random.randint(key, (rows, 6 + len(TREE)), 0, VOCAB)
        return nn.unbox(initialized(jax.random.fold_in(key, 1), ids)), ids[:, :6], ids[:, 6:]

    def after(model, *, packed: bool):
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

    assert_as_exact(after(model, packed=True), after(model, packed=False), draw,
                    "a tree after a packed prompt", seeds=FORK_SEEDS,
                    truth_fn=after(float64_twin(model), packed=False))


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

    assert_as_exact(tree_logits, decoded(model), draw, label, seeds=64, truth_fn=decoded(wide))


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

    truth = scored(GRPOObjective(float64_twin(model), WIDTH - 1), whole)
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


def _summed_gradient(function):
    """The gradient of `function`'s summed output in every operand but its documents' layout, its first
    argument, which a run supplies as an operand of the program too: a cost no constant layout
    specializes."""
    def gradient(layout, *operands):
        return jax.grad(lambda *operands: function(layout, *operands).astype(jnp.float32).sum(),
                        argnums=tuple(range(len(operands))))(*operands)
    return gradient


def _compiled_cost(function, layout, shapes, counter: str) -> tuple[int, float]:
    """The temporaries and FLOPs of `function`'s summed gradient (`_summed_gradient`, its documents' `layout`
    an operand) as `counter` counts them where its count is whole: 'cpu', XLA's cost analysis of the program
    compiled for this cpu, which counts every operation it runs; a CUDA generation ('sm80'), the program
    compiled for that GPU with no device (`kernel_compile.lowered_for`), every matrix product and each
    kernel launch at the work it states (`instrumentation.step_flops`), refused where a launch states none,
    whose work a count would leave out."""
    gradient = _summed_gradient(function)
    if counter == "cpu":
        compiled = jax.jit(gradient).lower(layout, *shapes).compile()
        cost = compiled.cost_analysis()
        flops = float((cost[0] if isinstance(cost, list) else cost)["flops"])
    else:
        from kernel_compile import lowered_for

        from dew.telemetry.instrumentation import step_flops

        compiled = lowered_for(gradient, (layout, *shapes), counter).compile()
        counted = step_flops(compiled)
        assert counted is not None, "a loop of no stated trips: the program's count is its body's"
        assert counted.unsized == 0, (f"{counted.unsized:.0f} launches state no work, which no count may "
                                      "leave out")
        flops = counted.hardware
    return int(compiled.memory_analysis().temp_size_in_bytes), flops


def _recurrence_call(layer: str, length: int, documents: int, depth: int = 0):
    """`layer`'s chunked recurrence over `FORKED_ROWS` rows of `length` tokens, `documents` a row of about
    equal lengths (three where zero), and with `depth` past zero a forked call through `forked_scan`, its
    first `depth` documents a chain and every later one continuing the chain's last (a state and questions
    over it, `depth` one): the function of its layout (its coordinates) and operands, the layout as the call
    supplies it, its operands' shapes, its chunk and its table's stored elements a row (`Forks.held`
    memories)."""
    from dew.nn.kernels.recurrence_chunk import CHUNK_SIZE, chunk_gated_delta_rule
    from dew.nn.linear import fork_start, forked_scan
    from dew.nn.mixers.mamba2 import CHUNK_SIZE as SSD_CHUNK, _chunked

    def shaped(*dims):
        return jax.ShapeDtypeStruct((FORKED_ROWS, *dims), jnp.float32)

    count = documents or 3
    numbers = np.minimum(np.arange(length) * count // length + 1, count).astype(np.int32)
    segments = jnp.asarray(np.broadcast_to(numbers, (FORKED_ROWS, length)))
    layout = coordinates(segments=segments, valid=segments > 0, max_segments=count)
    if depth:
        parents = np.asarray([0, *range(1, depth), *([depth] * (count - depth))], np.int32)
        held = np.broadcast_to(parents, (FORKED_ROWS, count))
        layout = dataclasses.replace(layout, forks=Forks(jnp.asarray(held), depth=Forks.depth_of(held),
                                                         held=Forks.held_of(held)))
    if layer == "gated delta net":
        memory, chunk = (3, 8, 8), CHUNK_SIZE
        operands = [shaped(length, 1, 8), shaped(length, 1, 8), shaped(length, 3, 8), shaped(length, 3),
                    shaped(length, 3)]

        def run(query, key, value, gate, beta, *, valid, starts, state=None, **table):
            kept = valid[:, :, None]
            return chunk_gated_delta_rule(query, key, value, jnp.where(kept, gate, 0.0),
                                          jnp.where(kept, beta, 0.0), state, starts=starts,
                                          max_segments=count, write_bound=1.0, **table)
    else:
        memory, chunk = (4, 8, 8), SSD_CHUNK
        operands = [shaped(length, 4, 8), shaped(length, 4), shaped(length, 1, 8), shaped(length, 1, 8)]
        rate, skip = -jnp.ones((4,)), jnp.ones((4,))

        def run(x, dt, keys, queries, *, valid, starts, state=None, **table):
            step = jnp.where(valid[:, :, None], jax.nn.softplus(dt), 0.0)
            forked = None if not table else (table['loads'], table['stores'], table['held'])
            return _chunked(x, step, rate, keys, queries, skip, state, SSD_CHUNK, starts=starts,
                            forked=forked)

    def call(laid, *operands):
        if not depth:
            return run(*operands, valid=laid.valid, starts=laid.starts)[0]

        def rule(convolved, *, valid, starts, state, loads, stores, held):
            return run(*convolved, valid=valid, starts=starts, state=state, loads=loads, stores=stores,
                       held=held)

        return forked_scan(operands, rule, laid, fork_start(None, (FORKED_ROWS, *memory), jnp.float32))

    stored = 0 if not depth else layout.forks.held * math.prod(memory)
    return call, layout, operands, chunk, FORKED_ROWS * stored


@pytest.mark.parametrize("counter", [
    pytest.param("cpu", marks=pytest.mark.skipif(
        jax.default_backend() != "cpu",
        reason="XLA's cost analysis counts no GPU kernel launch's work, so a native program's FLOPs are a "
               "part reported as the whole (ledger 5372)")),
    "sm80", "sm90"], ids=["the portable walk on a cpu", "an A100's routes", "an H100's routes"])
@pytest.mark.parametrize("depth", [1, 3], ids=["a state and its questions", "a chain three deep"])
@pytest.mark.parametrize("documents", [6, 173], ids=["the windows' own", "a loose bound"])
@pytest.mark.parametrize("layer", ["gated delta net", "SSD"])
def test_a_forked_recurrence_costs_its_tokens_and_a_chunk_a_document(layer, documents, depth, counter):
    """A forked call's recurrence (`forked_scan`) costs, in its gradient's temporaries and FLOPs, at most
    twice what its packed row costs laid one chunk longer a document (`AlignedDocuments`: each document
    from a chunk's edge), whatever its depth: one walk over the row, the call's own operands and their
    cotangents laid onto the grid and back at most once more, beside its table of stored finals and their
    cotangents, fp32, an operation an element (`Forks.held` memories a row, never the documents bound): a
    decision window's rows at a documents bound of its own windows and at a loose one, a state and its
    questions and a chain three deep, its layout an operand of the compiled program as a run supplies it.
    Counted where each count is whole (`_compiled_cost`): the portable walk on a cpu, and the routes an
    A100 and an H100 take, compiled with no device, every launch at the work it states. Before the one walk
    the recurrence ran the row once a level of the forks' chains, 2.4 to 3.1 times the packed row at depth
    one (armada, 11-10)."""
    call, layout, operands, chunk, stored = _recurrence_call(layer, FORKED_LENGTH, documents, depth)
    temporaries, flops = _compiled_cost(call, layout, operands, counter)
    packed, packed_layout, packed_operands, _, _ = _recurrence_call(layer, FORKED_LENGTH + documents * chunk,
                                                                    documents)
    bound, work = _compiled_cost(packed, packed_layout, packed_operands, counter)
    assert temporaries <= 2 * bound + 4 * 4 * stored, (temporaries, bound, flops, work, stored)
    assert flops <= 2 * work + 4 * stored, (temporaries, bound, flops, work, stored)


def test_a_forked_conv_reads_each_documents_real_tokens_past_its_holes():
    """A forked call whose parent and child each hold invalid slots inside their runs (paused tokens, a
    caller's own Coordinates) convolves each real token after the real tokens before it in its chain, as the
    same call laid with the holes taken out does (`forked_conv`, the paused-token contract of
    `_masked_conv1d`); a hole reads and advances nothing and is zero."""
    from dew.nn.linear import forked_conv

    channels, taps = 4, 3
    segments = np.asarray([[1, 1, 1, 1, 2, 2, 2, 3, 3, 0]], np.int32)
    held = np.asarray([[True, False, True, True, True, False, True, True, False, False]])
    parents = jnp.asarray([[0, 1, 1]], jnp.int32)
    forks = Forks(parents, depth=1)

    def laid(segments, valid):
        return dataclasses.replace(coordinates(segments=jnp.asarray(segments), valid=jnp.asarray(valid),
                                               max_segments=3), forks=forks)

    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.standard_normal((1, segments.shape[1], channels)), jnp.float32)
    kernel = jnp.asarray(rng.standard_normal((channels, taps)), jnp.float32)
    found = np.asarray(forked_conv(x, kernel, laid(segments, held)))
    order = np.flatnonzero(held[0])
    compact = np.zeros_like(segments)
    compact[0, :len(order)] = segments[0, order]
    moved = jnp.zeros_like(x).at[0, :len(order)].set(x[0, order])
    expected = np.asarray(forked_conv(moved, kernel, laid(compact, compact > 0)))
    np.testing.assert_array_equal(found[0, order], expected[0, :len(order)])
    assert np.all(found[0, ~held[0]] == 0)


def test_a_cached_call_stating_no_validity_places_its_keys_as_real_tokens():
    """A cached call over forks whose coordinates state no validity (a one-document call, which the
    backbone fills none for) places its keys as one whose every token is real (`fork_key_places`)."""
    from dew.nn.inputs import AttentionMetadata
    from dew.nn.mixers.attention import fork_key_places

    rows, queries, keys = 2, 3, 8
    plain = dataclasses.replace(coordinates((rows, queries)), forks=Forks(jnp.zeros((rows, 1), jnp.int32),
                                                                          depth=0))
    positions = jnp.broadcast_to(jnp.arange(5, 5 + queries), (rows, queries))
    first = jnp.asarray([5, 5], jnp.int32)

    def placed(laid):
        return np.asarray(fork_key_places(AttentionMetadata(coordinates=laid), positions, first, rows,
                                          queries, keys))

    real = dataclasses.replace(plain, valid=jnp.ones((rows, queries), bool))
    np.testing.assert_array_equal(placed(plain), placed(real))


class _PerLayerOverHidden(nn.Module):
    """A per-layer embedding (`dew.nn.engram.PLELayer`) over a layer's `[B, S, 2W]` input as two streams of
    `W`, each token's bucket ids its own input's (the same token's in a forked call and its own row), giving
    what it adds to the streams: zero on padding."""

    layer: PLELayer

    @nn.compact
    def __call__(self, hidden, call):
        batch, length, width = hidden.shape
        streams = hidden.reshape(batch, length, 2, width // 2)
        ids = (jnp.floor(jnp.abs(hidden[..., :2]) * 13).astype(jnp.int32) % self.layer.rows)
        return (self.layer(streams, ids, call) - streams).reshape(batch, length, width)


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_the_per_layer_embedding_reads_each_documents_chain_over_forks(layout):
    """Qwen4-Exp's per-layer embedding over documents that continue another (`dew.batch.Forks`) gives each
    document's tokens, and the gradients through them, what its own row of its chain and itself gives (check
    1): its dilated conv reads its chain's last gated values (`forked_conv`), across a short parent into its
    grandparent."""
    module = _PerLayerOverHidden(PLELayer(rows=23, head_dim=4, hc_mult=2, emb_features=8, norm_eps=1e-6,
                                          kernel_size=3, dilation=2, use_bias=True))
    hidden = jnp.ones((1, 4, 16), jnp.float32)
    variables = module.init(jax.random.key(0), hidden, MixerCall())
    assert_forks_as_own_rows(module, variables, 16, layout, "the per-layer embedding")


def test_a_block_over_runs_reads_no_key_past_the_calls_lengths():
    """A causal call over key runs with block groups and key lengths (`spans`) reads a query's block whole
    within its run and never a key past its row's length or before its start: the dense rule's
    `(causal or block) and run and starts <= key < length`, query by query (the block's widening took the
    run's end past the length, 9d3e1d9a1)."""
    from dew.nn.attention_masks import visibility_mask

    keys = 4
    runs = jnp.broadcast_to(jnp.asarray([0, keys], jnp.int32), (2, keys, 1, 2))
    blocks = jnp.zeros((2, keys), jnp.int32)
    lengths, starts = jnp.asarray([2, 4], jnp.int32), jnp.asarray([0, 1], jnp.int32)
    found = np.asarray(visibility_mask(keys, keys, causal=True, sliding_window=None,
                                       key_value_seq_lengths=lengths, key_value_seq_starts=starts,
                                       key_runs=runs, block_groups=blocks))[:, 0]
    for row in range(2):
        for query in range(keys):
            for key in range(keys):
                bounded = int(starts[row]) <= key < int(lengths[row])
                assert found[row, query, key] == bounded, (row, query, key)


def test_a_windowed_block_over_forks_reads_its_block_whole_and_its_window_else():
    """A windowed layer reading a prompt's block over forks (`fork_runs` with the block groups it reads, the
    spans' `_whole_blocks`) sees what the dense rule gives within each token's chain: a key of its own
    block whatever the window, else a key at or before it within the window of its place, `(window and
    causal) or block`, and nothing past its chain."""
    from dew.nn.attention_masks import visibility_mask
    from dew.nn.inputs import AttentionMetadata, fork_runs

    forks = ForkedRows.pack(ROWS, WIDTH)
    laid = forks.coordinates
    assert laid.forks is not None and laid.segments is not None and laid.positions is not None
    segments, places = np.asarray(laid.segments), np.asarray(laid.positions)
    windows, width = segments.shape
    slots = np.arange(width)
    # A prompt block: each window's first document's first five tokens.
    blocks = np.where((segments == 1) & (slots < 5), 1, -1).astype(np.int32)
    window = 2
    runs = fork_runs(AttentionMetadata(coordinates=laid, block_groups=jnp.asarray(blocks)), windows, width,
                     groups=("block_groups",))
    found = np.asarray(visibility_mask(width, width, causal=False, sliding_window=window, key_runs=runs,
                                       block_groups=jnp.asarray(blocks), places=jnp.asarray(places)))[:, 0]
    parents = np.asarray(laid.forks.parents)
    for row in range(windows):
        for query in np.flatnonzero(segments[row] > 0):
            chain, document = set(), segments[row, query]
            while document:
                chain.add(document)
                document = parents[row, document - 1]
            for key in np.flatnonzero(segments[row] > 0):
                mine = segments[row, key] == segments[row, query]
                before = (segments[row, key] in chain) and (key <= query or not mine)
                windowed = before and 0 <= places[row, query] - places[row, key] < window
                blocked = mine and blocks[row, query] >= 0 and blocks[row, key] == blocks[row, query]
                assert found[row, query, key] == (windowed or blocked), (row, query, key)


def _cached_fork_block():
    """A windowed layer's cached call over forks reading a block (row 5437): a row's cache holds three keys,
    the call a block of three tokens continuing it, a padding token, and two tokens of a fork of the block's
    document, stored from slot three (`fork_anchor`'s `first`) past the padding (`fork_slots`). Its spans'
    arguments under a window of one, and the keys the dense rule gives each query: a block's token its
    block's every key (`(window and causal) or block`), the fork's tokens their own key alone, the padding
    none."""
    from dew.nn.inputs import AttentionMetadata, fork_runs, fork_slots
    from dew.nn.mixers.attention import fork_key_places

    segments = jnp.asarray([[1, 1, 1, 0, 2, 2]], jnp.int32)
    valid = segments > 0
    parents = np.asarray([[0, 1]], np.int32)
    forks = Forks(jnp.asarray(parents), depth=Forks.depth_of(parents), held=Forks.held_of(parents))
    first, start, keys = jnp.asarray([3]), jnp.asarray([0]), 8
    positions = forks.positions(segments, jnp.asarray([3], jnp.int32), valid)
    laid = dataclasses.replace(
        coordinates(segments=segments, valid=valid, max_segments=2, positions=positions), forks=forks)
    metadata = AttentionMetadata(coordinates=laid,
                                 block_groups=jnp.asarray([[1, 1, 1, -1, -1, -1]], jnp.int32))
    spanned = {"key_runs": fork_runs(metadata, 1, 6, (first, start), groups=("block_groups",)),
               "query_slots": fork_slots(metadata, 1, 6, first),
               "places": (positions, fork_key_places(metadata, positions, first, 1, 6, keys)),
               "block_groups": jnp.asarray([[-1, -1, -1, 1, 1, 1, -1, -1]], jnp.int32)}
    seen = np.zeros((6, keys), bool)
    seen[:3, 3:6] = True
    seen[4, 6] = seen[5, 7] = True
    return spanned, seen


def test_a_cached_windowed_block_over_forks_reads_its_block_whole_and_its_window_else():
    """The spans of a cached call over forks reading a block (`_cached_fork_block`) are the dense rule's:
    each of the block's tokens sees its block's every key past its window, from the slots its queries are
    stored at."""
    from dew.nn.attention_masks import visibility_mask

    spanned, seen = _cached_fork_block()
    np.testing.assert_array_equal(np.asarray(spanned["query_slots"]), [[3, 4, 5, -1, 6, 7]])
    found = np.asarray(visibility_mask(6, 8, causal=False, sliding_window=1, **spanned))[0, 0]
    np.testing.assert_array_equal(found, seen)


def test_a_cached_windowed_block_over_forks_attends_as_its_dense_rule():
    """A cached call over forks reading a block (`_cached_fork_block`) through the public entry, its runs,
    groups, places and slots, attends exactly as the same call under its dense rule's mask, the reference
    path computing both: its groups kept past the call's own causality, which its runs hold."""
    from dew.nn.attention import scaled_dot_product_attention

    spanned, seen = _cached_fork_block()
    query, key, value = (jax.random.normal(jax.random.key(index), (1, length, 2, 16), jnp.float32)
                         for index, length in enumerate((6, 8, 8)))
    spans = scaled_dot_product_attention(query, key, value, sliding_window=1, implementation="reference",
                                         **spanned)
    dense = scaled_dot_product_attention(query, key, value, mask=jnp.asarray(seen)[None, None],
                                         implementation="reference")
    np.testing.assert_array_equal(np.asarray(spans), np.asarray(dense))


def test_rollouts_share_their_prompts_where_their_scorer_does():
    """A rollout scheduler packs a task's samples over one copy of their prompt where the objective that
    trains on them shares, read at each call (`RolloutScheduler.scorer`, its `shares_prefixes(train=True)`):
    a GRPO policy over a forked decoder, and a PPO objective whose critic reads forks too; it packs every
    sample whole where the critic's backbone reads no forks or no scorer is named."""
    import optax
    from model_support import TINY_DECODER
    from test_rl_methods import Scripted, session

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.rl.grpo import GRPOObjective
    from dew.objectives.rl.ppo import PPOObjective, ValueHead
    from dew.objectives.rl.scheduler import RolloutScheduler, Synchronous
    from dew.objectives.rl.sessions import FORK_PARENTS_KEY
    from dew.training import Trainer

    model = CausalTransformer(vocab_size=VOCAB, **TINY_DECODER)
    sessions = {"0": [session(0, sample, float(sample), (2,)) for sample in range(2)]}
    batch = {"task_id": np.asarray([0], np.int32)}

    def packed(scorer, objective):
        state = Trainer(objective, optax.sgd(0.1), key=jax.random.key(3)).initial_state()
        scheduler = RolloutScheduler(Synchronous(Scripted(sessions)), groups=2, scorer=scorer)
        return lambda: FORK_PARENTS_KEY in scheduler(state, batch, jax.random.key(0), objective=objective)

    class Plain(nn.Module):
        """A backbone that states no reading of a call over forks."""

        def hidden_states(self, tokens, **_):
            return jnp.zeros((*tokens.shape, 4))

    policy = GRPOObjective(model, Scripted.width - 1)
    critic = PPOObjective(model, Scripted.width - 1, critic=ValueHead(model))
    for objective in (policy, critic):
        assert packed(objective, policy)()
    assert not packed(PPOObjective(model, Scripted.width - 1, critic=ValueHead(Plain())), policy)()
    assert not packed(None, policy)()
