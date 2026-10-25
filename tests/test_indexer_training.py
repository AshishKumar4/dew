"""DeepSeek-V3.2 sparse-indexer training: the KL objective and its two phases.

The arithmetic is checked against MaxText 0.2.4's `calculate_indexer_loss`
and exact top-k mask, fetched and run as published by
tools/indexer_reference.py: the loss and its gradient on par with the
published loss's from its float64 run over recorded seeds (check 1,
tests/reference_error.py), the selection MaxText's mask exactly, as on the
transcription's tiny tensors (tests/fixtures/indexer). The phases are
checked through the LM objective on a toy DeepSeek stack: the warm-up moves
the indexer and nothing else, the sparse phase keeps the indexer's gradient
and the main model's apart, and a packed batch scores its documents as it
would alone. Everything runs at fp32 on CPU.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from recording import RecordingTracker
from reference_error import (
    assert_as_exact,
    assert_moves,
    declared_seeds,
    indexed_recording,
    recording,
    with_gradients,
)

from dew.data.dataset import Loader
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import ModelInputs, coordinates
from dew.nn.mla import INDEXER, indexer_kl
from dew.nn.sparse_selection import top_k_keys
from dew.objectives.base import FROZEN, Step
from dew.objectives.lm import IndexerTraining, LMObjective
from dew.training import Layout, MeshSpec, Trainer
from tools import indexer_reference as transcription
from tools.indexer_reference import published_inputs, published_keep

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "indexer"
META = json.loads((FIXTURES / "meta.json").read_text())
PUBLISHED = np.load(FIXTURES / "maxtext.npz")
MLA_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mla"
MLA_CONFIG = json.loads((MLA_FIXTURES / "config.json").read_text())

SEQ = 16
VOCAB = 64
TINY_SHARD = 256
SEEDS = {'covering': 256, 'covered positions': 512, 'dropped': 256, 'dropped positions': 256, 'packed': 64,
         'packed alone': 128, 'total': 512, 'documents alone': 512, 'split tree': 8, 'per query': 64}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (on armada):
the most any of a site's cases took."""


def fixture() -> dict:
    with np.load(FIXTURES / "indexer_loss.npz") as data:
        return {key: jnp.asarray(value) for key, value in data.items()}


def causal(length: int) -> jax.Array:
    return jnp.tril(jnp.ones((length, length), bool))[None]


def packed_keep(segments: jax.Array) -> jax.Array:
    inside = (segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] != 0)
    return inside & causal(segments.shape[1])


# --------------------------------------------------------------------------
# The loss against the reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["causal_dense", "packed_dense", "causal_sparse"])
def test_the_loss_its_gradient_and_the_selection_are_maxtexts(name):
    """MaxText 0.2.4's own `calculate_indexer_loss` and `generate_mask`,
    run as published by tools/indexer_reference.py behind the masking and
    `jax.lax.top_k` of `Indexer.__call__`, on 2 x 24 x 24 scores with
    exact-zero ties: the exact top-k selection is MaxText's mask; the dense
    warm-up's loss over the causal or packed keys and the sparse loss over
    the selection, and their gradients in the indexer's scores, what trains
    the indexer, are on par with the published loss's from its float64 run
    over the recorded seeds (check 1), the truth Dew's own loss in float64,
    held at every seed to the recording's sketch."""
    mask, phase = name.split("_")
    data = {key: jnp.asarray(PUBLISHED[key]) for key in ("query", "key", "scores", "segments")}
    length, scale = data["scores"].shape[-1], float(PUBLISHED["scale"])
    keep = causal(length) if mask == "causal" else packed_keep(data["segments"])
    np.testing.assert_array_equal(np.broadcast_to(keep, published_keep(mask).shape), published_keep(mask))
    selected = top_k_keys(data["scores"], keep, 6)
    np.testing.assert_array_equal(np.asarray(selected), PUBLISHED[f"{mask}/selected"])

    @jax.jit
    def kl(inputs, cotangent):
        """The loss and its gradient in the scores under `cotangent` (`with_gradients`), the selection
        read off the scores as given."""
        region = top_k_keys(inputs["scores"], keep, 6) if phase == "sparse" else keep
        value, gradient = with_gradients(
            lambda scores: jnp.mean(indexer_kl(scores, inputs["query"], inputs["key"], region, scale)))(
            inputs["scores"], cotangent)
        return {"loss": value, "gradient": gradient}

    path = FIXTURES / f"recorded_{name}.npz"
    seeds = declared_seeds(path)[name]
    # The recording holds the loss's gradient, a scalar loss's under the cotangent 1, which each draw
    # carries for Dew's arm.
    recorded_draw, reference, truth = indexed_recording(
        recording(path), name, seeds, truth_fn=lambda inputs: kl(inputs, np.float64(1)),
        inputs_of=published_inputs)
    assert_as_exact(lambda index, inputs, cotangent: kl(inputs, cotangent),
                    lambda index, inputs, cotangent: reference(index, inputs),
                    lambda seed: (*recorded_draw(seed), np.float32(1)), f"{name} indexer loss", seeds=seeds,
                    truth_fn=lambda index, inputs, cotangent: truth(index, inputs))

    legacy = fixture()
    legacy_keep = causal(META["length"]) if mask == "causal" else packed_keep(legacy["segments"])
    legacy_selected = top_k_keys(legacy["scores"], legacy_keep, META["top_k"])
    np.testing.assert_array_equal(legacy_selected, legacy[f"{mask}_selected"])
    if mask == "causal":
        counts = jnp.sum(legacy_selected, axis=-1)
        assert bool(jnp.all(counts == jnp.minimum(jnp.arange(META["length"]) + 1, META["top_k"])))
    else:
        per_query = indexer_kl(legacy["scores"], legacy["query"], legacy["key"], legacy_keep,
                               float(legacy["scale"]))
        assert bool(jnp.all(per_query[1, 5:] == 0))
        assert bool(jnp.all(jnp.isfinite(jax.grad(
            lambda scores: jnp.sum(indexer_kl(scores, legacy["query"], legacy["key"],
                                              legacy_keep, float(legacy["scale"]))))(legacy["scores"]))))

    # Each query's KL, which the published mean does not show, on par with the transcription's
    # (tools/indexer_reference.py, the same steps in NumPy) from its float64 run.
    scale_ = np.float32(transcription.HEAD_DIM ** -0.5)
    additive = (transcription.causal_mask() if mask == "causal"
                else transcription.packed_mask(np.asarray(legacy["segments"])))

    def draw(seed):
        rng = np.random.default_rng([23, seed])
        shape = (transcription.BATCH, transcription.LENGTH, transcription.HEADS, transcription.HEAD_DIM)
        query, key = (rng.normal(size=shape).astype(np.float32) for _ in range(2))
        length = transcription.LENGTH
        scores = np.maximum(rng.normal(scale=1.5, size=(transcription.BATCH, length, length)), 0)
        return scores.astype(np.float32), query, key

    def transcribed(scores, query, key):
        dtype = np.asarray(scores).dtype
        indexer_mask, masked_scores = transcription.indexer_call(np.asarray(scores), additive,
                                                                 transcription.TOP_K)
        scaled = np.asarray(query) * np.asarray(scale_, dtype)
        per_query, _ = transcription.calculate_indexer_loss(
            masked_scores, scaled, np.asarray(key), additive, indexer_mask, sparse_loss=phase == "sparse",
            dtype=dtype)
        return {"per query": per_query}

    def ours(scores, query, key):
        region = top_k_keys(scores, legacy_keep, transcription.TOP_K) if phase == "sparse" else legacy_keep
        return {"per query": indexer_kl(scores, query, key, region, float(scale_))}

    assert_as_exact(jax.jit(ours), transcribed, draw, f"{name} per query", seeds=SEEDS['per query'])


# --------------------------------------------------------------------------
# The mixer with and without a top-k
# --------------------------------------------------------------------------


def v32_block(**overrides):
    from test_mla import block_variables, fixture as mla_fixture, mla_module

    tensors = mla_fixture("mla_v32")
    module = mla_module(dict(MLA_CONFIG["v32"], **overrides))
    return module, block_variables(tensors), jnp.asarray(tensors["hidden"])


def test_an_indexer_without_a_top_k_leaves_the_attention_dense():
    """The warm-up's layer: the indexer's weights in the tree and its KL
    sown, the output the dense block's to the bit."""
    dense, variables, hidden = v32_block(index_topk=None, index_n_heads=None, index_head_dim=None)
    indexed, _, _ = v32_block(index_topk=None)
    dense_out = dense.apply({"params": {name: value for name, value in variables["params"].items()
                                        if name != INDEXER}}, hidden)
    indexed_out, sown = indexed.apply(variables, hidden, mutable=["indexer"])
    assert bool(jnp.all(indexed_out == dense_out))
    assert sown["indexer"]["kl"][0].shape == hidden.shape[:2]


def test_a_top_k_covering_the_keys_selects_the_dense_mask():
    """With the sequence inside the top-k the sparse layer attends every
    causal key, so its output and its KL are on par with the dense-indexed
    layer's over drawn inputs (check 1). Below the sequence the selection
    drops keys and both move past rounding (`assert_moves`), but at the
    positions the top-4 still covers, whose KL stays on par."""
    indexed, variables, hidden = v32_block(index_topk=None)
    covering, _, _ = v32_block(index_topk=hidden.shape[1])
    selecting, _, _ = v32_block(index_topk=4)

    def draw(seed):
        return (jax.random.normal(jax.random.fold_in(jax.random.key(7), seed), hidden.shape, hidden.dtype),)

    def run(module):
        def apply(hidden):
            out, sown = module.apply(variables, hidden, mutable=["indexer"])
            return {"output": out, "kl": sown["indexer"]["kl"][0]}
        return jax.jit(apply)

    def kl_at(module, positions):
        whole = run(module)
        return lambda hidden: {"kl": whole(hidden)["kl"][:, positions]}

    assert_as_exact(run(covering), run(indexed), draw, "covering top-k", seeds=SEEDS['covering'])
    assert_as_exact(kl_at(selecting, slice(None, 4)), kl_at(indexed, slice(None, 4)), draw,
                    "covered positions", seeds=SEEDS['covered positions'])
    assert_moves(run(indexed), run(selecting), draw, "dropped keys", seeds=SEEDS['dropped'])
    assert_moves(kl_at(indexed, slice(4, None)), kl_at(selecting, slice(4, None)), draw, "dropped positions",
                 seeds=SEEDS['dropped positions'])


def test_a_packed_batch_selects_inside_its_documents():
    """Packed positions restart per document, so the selection has to
    read the rows' causal order and the segment mask, not the positions:
    with a top-k covering every key the sparse layer attends on par with
    the dense-indexed one, and each document's tokens come out on par with
    them alone, over drawn models and tokens (check 1). Before the fix the
    mask read the positions as row order and the covering top-k differed
    from dense by 2.8."""
    segments = jnp.asarray(np.array([[1] * 3 + [2] * 4 + [0] * (SEQ - 7)]))
    positions = jnp.asarray(np.array([list(range(3)) + list(range(4)) + [0] * (SEQ - 7)]))
    covering, dense = deepseek_stack(SEQ), deepseek_stack(None)
    init = jax.jit(covering.init)

    def draw(seed):
        rng = np.random.default_rng([4, seed])
        first, second = rng.integers(1, VOCAB, size=(3,)), rng.integers(1, VOCAB, size=(4,))
        tokens = jnp.asarray(np.concatenate([first, second, [0] * (SEQ - 7)])[None], jnp.int32)
        return init(jax.random.fold_in(jax.random.key(0), seed), tokens), tokens

    def packed(model):
        return jax.jit(lambda params, tokens: model.apply(params, tokens, positions=positions,
                                                          segment_ids=segments)[:, :7])

    def second_document(model):
        return jax.jit(lambda params, tokens: model.apply(params, tokens, positions=positions,
                                                          segment_ids=segments)[:, 3:7])

    def alone(model):
        def run(params, tokens):
            shifted = jnp.concatenate([tokens[:, 3:7], jnp.zeros((1, SEQ - 4), tokens.dtype)], axis=1)
            return model.apply(params, shifted)[:, :4]
        return jax.jit(run)

    wide = covering.clone(dtype=jnp.float64)
    assert_as_exact(packed(covering), packed(dense), draw, "packed covering top-k", seeds=SEEDS['packed'],
                    truth_fn=packed(dense.clone(dtype=jnp.float64)))
    assert_as_exact(second_document(covering), alone(covering), draw, "packed document alone",
                    seeds=SEEDS['packed alone'], truth_fn=alone(wide))


# --------------------------------------------------------------------------
# The phases through the objective
# --------------------------------------------------------------------------


def deepseek_stack(index_topk, **overrides) -> CausalTransformer:
    return CausalTransformer(**{
        "vocab_size": VOCAB, "emb_features": 32, "num_layers": 2, "num_heads": 2,
        "head_dim": 16, "mlp_features": 64, "max_seq_len": SEQ,
        "mixer": {"class": "mla", "fields": {"q_lora_rank": 8, "kv_lora_rank": 8,
               "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 8,
               "index_topk": index_topk, "index_n_heads": 2, "index_head_dim": 16}},
        **overrides})


def token_batch(batch=4, seed=0):
    ids = np.random.default_rng(seed).integers(0, VOCAB, size=(batch, SEQ + 1))
    return {"text": jnp.asarray(ids, jnp.int32)}


def step_at(index=0, key=1):
    return Step(step=jnp.asarray(index), key=jax.random.key(key), ema=None)


def is_indexer(path) -> bool:
    return any(getattr(entry, "key", None) == INDEXER for entry in path)


def split_norms(tree) -> tuple[float, float]:
    """Global norms of the indexer's leaves and of every other leaf."""
    leaves = jax.tree_util.tree_leaves_with_path(tree)
    indexer = [leaf for path, leaf in leaves if is_indexer(path)]
    rest = [leaf for path, leaf in leaves if not is_indexer(path)]
    return float(optax.tree.norm(indexer)), float(optax.tree.norm(rest))


def test_the_sparse_phase_keeps_the_indexer_and_the_model_apart():
    """The main weights see the plain cross entropy gradient whatever the
    KL weighs, the indexer's gradient is the KL's alone and scales with
    its weight, to the bit at weights a power of two apart, and without the
    term the indexer moves nowhere; the loss is the cross entropy plus the
    weighted KL, on par with that sum of its reports over drawn batches
    (check 1)."""
    batch = token_batch()
    plain = LMObjective(deepseek_stack(4), SEQ)
    params = plain.init(jax.random.key(0))

    def gradient(objective):
        (value, aux), grads = jax.value_and_grad(
            lambda p: objective.scalar_loss(p, batch, step_at()), has_aux=True)(params)
        return value, aux, grads["params"]

    ce, _, plain_grads = gradient(plain)
    assert split_norms(plain_grads)[0] == 0
    seen = {}
    for weight in (0.5, 2.0):
        objective = LMObjective(
            deepseek_stack(4), SEQ, indexer=IndexerTraining("sparse", weight=weight))
        _, aux, grads = gradient(objective)
        assert float(aux.metrics["ce"]) == float(ce)
        seen[weight] = split_norms(grads)[0]
        assert seen[weight] > 0
        kl_grads = jax.jit(
            jax.grad(
                lambda p, objective=objective: objective.loss(p, batch, step_at())[1].metrics["indexer_kl"]
            )
        )(params)["params"]
        for path, leaf in jax.tree_util.tree_leaves_with_path(kl_grads):
            if not is_indexer(path):
                assert bool(jnp.all(leaf == 0)), jax.tree_util.keystr(path)
        assert float(aux.metrics["indexer_kl"]) > 0
    assert seen[2.0] == 4 * seen[0.5]

    objective = LMObjective(deepseek_stack(4), SEQ, indexer=IndexerTraining("sparse", weight=0.5))
    wide = LMObjective(deepseek_stack(4, dtype=jnp.float64), SEQ,
                       indexer=IndexerTraining("sparse", weight=0.5))

    def total(objective, reported):
        def run(params, text):
            value, aux = objective.scalar_loss(params, {"text": text}, step_at())
            return value if not reported else aux.metrics["ce"] + 0.5 * aux.metrics["indexer_kl"]
        return jax.jit(run)

    def draw(seed):
        return params, token_batch(seed=1 + seed)["text"]

    assert_as_exact(total(objective, reported=False), total(objective, reported=True), draw, "sparse total",
                    seeds=SEEDS['total'], truth_fn=total(wide, reported=True))

    data = fixture()
    keep = causal(META["length"])
    scale = float(data["scale"])
    grads = jax.grad(
        lambda scores, query, key: jnp.sum(indexer_kl(scores, query, key, keep, scale)),
        argnums=(0, 1, 2))(data["scores"], data["query"], data["key"])
    assert bool(jnp.all(grads[1] == 0)) and bool(jnp.all(grads[2] == 0))


def packed_row(*documents):
    """One `[SEQ + 1]` row: the documents back to back, then padding (segment 0)."""
    ids = np.zeros((SEQ + 1,), np.int32)
    segments = np.zeros((SEQ + 1,), np.int32)
    positions = np.zeros((SEQ + 1,), np.int32)
    cursor = 0
    for index, document in enumerate(documents, start=1):
        ids[cursor:cursor + len(document)] = document
        segments[cursor:cursor + len(document)] = index
        positions[cursor:cursor + len(document)] = np.arange(len(document))
        cursor += len(document)
    return ids, segments, positions


def packed_batch(rows):
    ids, segments, positions = (np.stack(column) for column in zip(*rows, strict=True))
    return {"text": jnp.asarray(ids), "text_coordinates": coordinates(
        segments=segments, positions=positions, max_segments=ids.shape[1], right_padded=True, xp=np)}


def test_a_packed_batch_scores_the_kl_of_its_documents_alone():
    """Two documents packed into one row, then each alone with padding:
    the KL per counted query is on par over drawn documents (check 1), so
    the packed mask isolates the documents inside the indexer and its
    target, and the padding queries count nothing. A row of padding alone
    contributes no query."""
    def draw(seed):
        tokens = np.random.default_rng([3, seed]).integers(1, VOCAB, size=(SEQ + 1,))
        first, second = tokens[:7], tokens[7:13]
        rows = [packed_row(first, second)], [packed_row(first), packed_row(second), packed_row()]
        return tuple(packed_batch(row) for row in rows)

    for phase, topk in (("warmup", None), ("sparse", 4)):
        objective, wide = (LMObjective(deepseek_stack(topk, dtype=dtype), SEQ, indexer=IndexerTraining(phase))
                           for dtype in (jnp.float32, jnp.float64))
        params = objective.init(jax.random.key(0))

        def kl(objective, at, params=params):
            return jax.jit(lambda *batches: objective.scalar_loss(params, batches[at], step_at())[1]
                           .metrics["indexer_kl"])

        assert np.isfinite(float(kl(objective, 0)(*draw(0))))
        assert_as_exact(kl(objective, 0), kl(objective, 1), draw, f"{phase} documents alone",
                        seeds=SEEDS['documents alone'], truth_fn=kl(wide, 1))


def test_a_float64_kl_is_summed_in_float64():
    """The objective sums its layers' sown indexer KLs at their own width, at least fp32: a float64
    model's per-query KLs give a float64 total, the float64 sum over the counted queries, where an
    fp32 one stays fp32."""
    import math

    from dew.objectives.lm.objective import batch_text

    objective = LMObjective(deepseek_stack(4), SEQ, indexer=IndexerTraining("sparse"))
    batch = token_batch()
    prepared = batch_text(batch)
    with jax.enable_x64(new_val=True):
        kl = jnp.asarray(np.random.default_rng(7).random((4, SEQ)) * 1e-3 + 1.0, jnp.float64)
        total, mass = objective._indexer_term({"layer": {"kl": (kl,)}}, prepared, batch)
        assert total.dtype == jnp.float64
        assert float(total) == math.fsum(np.asarray(kl).ravel()) and float(mass) == 4 * SEQ
        narrow, _ = objective._indexer_term({"layer": {"kl": (kl.astype(jnp.float32),)}}, prepared, batch)
        assert narrow.dtype == jnp.float32


@pytest.mark.parametrize("phase, topk", [("warmup", None), ("sparse", 4)])
def test_packing_carried_by_the_model_inputs_scores_the_same_kl(phase, topk):
    """Either phase reads a packed row's documents off its `ModelInputs` as
    it reads them off the batch's coordinates, to the bit, padding
    queries included, which count nothing either way."""
    tokens = np.random.default_rng(3).integers(1, VOCAB, size=(SEQ + 1,))
    columns = packed_batch([packed_row(tokens[:7], tokens[7:13])])
    carried = {"text": ModelInputs(columns["text"], {"coordinates": columns["text_coordinates"]})}
    objective = LMObjective(deepseek_stack(topk), SEQ, indexer=IndexerTraining(phase))
    params = objective.init(jax.random.key(0))

    _, by_columns = objective.scalar_loss(params, columns, step_at())
    _, by_inputs = objective.scalar_loss(params, carried, step_at())

    assert float(by_inputs.metrics["indexer_kl"]) == float(by_columns.metrics["indexer_kl"])


@pytest.mark.parametrize("phase, topk", [("warmup", None), ("sparse", 4)])
def test_padding_the_model_inputs_mask_leaves_out_counts_no_query(phase, topk):
    """One document, then padding whose ids are real tokens. The attention
    mask of the `ModelInputs` marks the padding as the segment column's zeros
    do, and the KL per counted query is the same number either way."""
    tokens = np.random.default_rng(3).integers(1, VOCAB, size=(SEQ + 1,))
    ids, segments, positions = packed_row(tokens[:7])
    ids[7:] = np.random.default_rng(5).integers(1, VOCAB, size=ids.shape[0] - 7)
    by_segments = packed_batch([(ids, segments, positions)])
    by_mask = {"text": ModelInputs(jnp.asarray(ids[None]),
                                   {"attention_mask": jnp.asarray(segments[None] != 0)})}
    objective = LMObjective(deepseek_stack(topk), SEQ, indexer=IndexerTraining(phase))
    params = objective.init(jax.random.key(0))

    _, masked_by_segments = objective.scalar_loss(params, by_segments, step_at())
    _, masked_by_inputs = objective.scalar_loss(params, by_mask, step_at())

    assert (float(masked_by_inputs.metrics["indexer_kl"])
            == float(masked_by_segments.metrics["indexer_kl"]))


def test_the_warmup_starts_a_fresh_indexer_beside_a_dense_checkpoint():
    """A dense tree (no indexer) is what the warm-up begins from: its
    leaves land frozen, unchanged, and the indexer comes from the init.
    A tree missing anything else is refused by name."""
    dense = deepseek_stack(None, mixer={"class": "mla", "fields": {"q_lora_rank": 8, "kv_lora_rank": 8,
                                        "qk_nope_head_dim": 8, "qk_rope_head_dim": 8,
                                        "v_head_dim": 8}})
    checkpoint = dense.init(jax.random.key(5), jnp.zeros((1, SEQ), jnp.int32))
    objective = LMObjective(deepseek_stack(None), SEQ, indexer=IndexerTraining("warmup"),
                            variables=checkpoint)
    params = objective.init(jax.random.key(0))
    frozen = dict(jax.tree_util.tree_leaves_with_path(params[FROZEN]))
    given = dict(jax.tree_util.tree_leaves_with_path(checkpoint["params"]))
    assert frozen.keys() == given.keys()
    for path, leaf in given.items():
        assert bool(jnp.all(frozen[path] == leaf)), jax.tree_util.keystr(path)
    assert jax.tree.structure(params["params"]) == jax.tree.structure(
        LMObjective(deepseek_stack(None), SEQ, indexer=IndexerTraining("warmup"))
        .init(jax.random.key(0))["params"])
    incomplete = {"params": {name: value for name, value in checkpoint["params"].items()
                             if name != "embed_tokens"}}
    with pytest.raises(ValueError):
        LMObjective(deepseek_stack(None), SEQ, indexer=IndexerTraining("warmup"),
                    variables=incomplete).init(jax.random.key(0))


def test_the_sparse_phase_reads_the_warmup_tree():
    """The warm-up's split tree hands over to the sparse phase, which trains
    the whole tree, as one params collection, leaf for leaf. A plain
    objective keeps the split it is given, so it trains the indexer alone."""
    warmup = LMObjective(deepseek_stack(None), SEQ, indexer=IndexerTraining("warmup"))
    split = warmup.init(jax.random.key(0))
    params = LMObjective(deepseek_stack(4), SEQ, indexer=IndexerTraining("sparse"),
                         variables=split).init(jax.random.key(1))
    assert sorted(params) == ["params"]
    leaves = dict(jax.tree_util.tree_leaves_with_path(params["params"]))
    expected = dict(jax.tree_util.tree_leaves_with_path(split["params"]))
    expected.update(jax.tree_util.tree_leaves_with_path(split[FROZEN]))
    assert leaves.keys() == expected.keys()
    assert all(bool(jnp.all(leaves[path] == expected[path])) for path in leaves)

    kept = LMObjective(deepseek_stack(4), SEQ, variables=split).init(jax.random.key(1))
    assert jax.tree.structure(kept) == jax.tree.structure(split)
    assert all(bool(jnp.all(ours == theirs))
               for ours, theirs in zip(jax.tree.leaves(kept), jax.tree.leaves(split), strict=True))


def test_evaluation_and_scoring_read_the_split_tree():
    """Teacher-forced scores under the warm-up split are on par with the
    dense model's on the merged tree over drawn batches (check 1)."""
    objective = LMObjective(deepseek_stack(None), SEQ, indexer=IndexerTraining("warmup"), ema_decay=0.999)
    params = objective.init(jax.random.key(0))
    merged = LMObjective(deepseek_stack(None), SEQ, variables=params).init(jax.random.key(0))
    dense, wide = (LMObjective(deepseek_stack(None, dtype=dtype), SEQ)
                   for dtype in (jnp.float32, jnp.float64))
    scored = jax.jit(lambda text: objective.evaluate(params, {"text": text}, step_at()).losses)

    def scores(objective):
        return jax.jit(lambda merged, text: objective.token_scores(merged, text).losses)

    def draw(seed):
        return merged, token_batch(seed=seed)["text"]

    assert_as_exact(lambda merged, text: scored(text), scores(dense), draw, "split tree scores",
                    seeds=SEEDS['split tree'], truth_fn=scores(wide))
    assert not objective.ema.select((FROZEN, "layers_0"))
    assert objective.ema.select(("params", "layers_0"))



def token_batches():
    rng = np.random.default_rng(0)
    while True:
        yield {"text": rng.integers(0, VOCAB, size=(8, SEQ + 1)).astype(np.int32)}


@pytest.mark.mesh
def test_the_warmup_trains_through_the_trainer():
    """Three steps on the simulated mesh, the parameters two ways over
    fsdp: the optimizer state covers the indexer alone, the frozen
    collection comes back bit for bit, the indexer moves, the reported KL
    falls, and the placement check passes over both collections."""
    model = deepseek_stack(None)
    tracker = RecordingTracker()
    trainer = Trainer(
        LMObjective(model, SEQ, indexer=IndexerTraining("warmup")), optax.adam(1e-2),
        key=jax.random.key(0), mesh=MeshSpec(fsdp=2),
        layout=Layout(min_shard=TINY_SHARD), tracker=tracker)
    # The state fit starts from: the jitted init, whose initializers fuse
    # differently from an eager one at the last bit.
    initial, _, _ = trainer.place()

    state = trainer.fit(Loader(lambda partition: token_batches(), None, batch=8, records=None),
                        steps=3, log_every=1)

    assert jax.tree.structure(state.opt_state) == jax.tree.structure(
        optax.adam(1e-2).init(initial.variables["params"]))
    assert all(is_indexer(path) for path, _ in jax.tree_util.tree_leaves_with_path(state.variables["params"]))
    frozen_before = jax.tree.leaves(initial.variables[FROZEN])
    frozen_after = jax.tree.leaves(state.variables[FROZEN])
    assert all(bool(jnp.all(a == b)) for a, b in zip(frozen_before, frozen_after, strict=True))
    moved = jax.tree.leaves(jax.tree.map(lambda a, b: jnp.any(a != b),
                                         initial.variables["params"], state.variables["params"]))
    assert all(bool(x) for x in moved)
    kls = [entry["train/indexer_kl"] for _, entry in tracker.scalars if "train/indexer_kl" in entry]
    assert len(kls) == 3 and all(np.isfinite(kls)) and kls[-1] < kls[0], kls
    assert state.variables[FROZEN]["layers_0"]["self_attn"]["kv_b_proj"]["kernel"].sharding.spec == (
        state.variables["params"]["layers_0"]["self_attn"][INDEXER]["wq_b"]["kernel"].sharding.spec)
    abstract = jax.eval_shape(trainer.initial_state)
    trainer.layout.check(abstract.variables, trainer.shardings(abstract).variables, trainer.device_mesh)

    (before, aux), grads = jax.value_and_grad(
        lambda params: trainer.objective.scalar_loss(params, token_batch(), step_at()),
        has_aux=True)(initial.variables)
    assert float(before) == float(aux.metrics["indexer_kl"])
    assert float(optax.tree.norm(grads[FROZEN])) == 0
    assert float(optax.tree.norm(grads["params"])) > 0
