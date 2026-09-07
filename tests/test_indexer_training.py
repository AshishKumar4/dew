"""DeepSeek-V3.2 sparse-indexer training: the KL objective and its two phases.

The arithmetic is checked against MaxText 0.2.4's `calculate_indexer_loss`
and exact top-k mask, transcribed in tools/indexer_reference.py and run on
fixed-seed tensors into tests/fixtures/indexer. The phases are checked
through the LM objective on a toy DeepSeek stack: the warm-up moves the
indexer and nothing else, the sparse phase keeps the indexer's gradient
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

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mla import INDEXER, indexer_kl, top_k_keys
from dew.objectives.base import Step, scalar_loss
from dew.objectives.lm import FROZEN, IndexerTraining, LMObjective
from dew.training import Layout, MeshSpec, Trainer

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "indexer"
META = json.loads((FIXTURES / "meta.json").read_text())
MLA_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mla"
MLA_CONFIG = json.loads((MLA_FIXTURES / "config.json").read_text())

SEQ = 16
VOCAB = 64
TINY_SHARD = 256


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


def test_the_dense_kl_matches_the_reference_warmup_loss():
    """`indexer_sparse_training=False`: the KL over every causal key.

    Tolerance 1e-6 on the per-query values and the mean; observed 1.8e-7.
    """
    data = fixture()
    per_query = indexer_kl(data["scores"], data["query"], data["key"],
                           causal(META["length"]), float(data["scale"]))
    np.testing.assert_allclose(per_query, data["causal_dense_kl"], atol=1e-6)
    assert float(jnp.mean(per_query)) == pytest.approx(float(data["causal_dense_loss"]), abs=1e-6)


def test_the_selection_is_the_reference_exact_top_k():
    """Exactly k keys per query where k are allowed, ties to the earliest
    key, and every allowed key before that; the fixture's scores tie at
    exact zero on nearly half the entries, so the rule decides the mask."""
    data = fixture()
    keep = causal(META["length"])
    selected = top_k_keys(data["scores"], keep, META["top_k"])
    assert bool(jnp.all(selected == data["causal_selected"]))
    counts = jnp.sum(selected, axis=-1)
    assert bool(jnp.all(counts == jnp.minimum(jnp.arange(META["length"]) + 1, META["top_k"])))
    assert bool(jnp.all(top_k_keys(data["scores"], packed_keep(data["segments"]), META["top_k"])
                        == data["packed_selected"]))


def test_the_sparse_kl_matches_the_reference_sparse_loss():
    """`indexer_sparse_training=True`: both distributions over the top-k.

    Tolerance 1e-6; observed 1.8e-7.
    """
    data = fixture()
    selected = top_k_keys(data["scores"], causal(META["length"]), META["top_k"])
    per_query = indexer_kl(data["scores"], data["query"], data["key"], selected, float(data["scale"]))
    np.testing.assert_allclose(per_query, data["causal_sparse_kl"], atol=1e-6)
    assert float(jnp.mean(per_query)) == pytest.approx(float(data["causal_sparse_loss"]), abs=1e-6)


def test_a_packed_batch_keeps_its_documents_apart_and_its_padding_at_zero():
    """The packed mask reaches both distributions, and a padding query
    that may attend nothing scores exactly zero rather than nan, as the
    reference's additive mask leaves it. Tolerance 1e-6; observed 1.9e-7."""
    data = fixture()
    per_query = indexer_kl(data["scores"], data["query"], data["key"],
                           packed_keep(data["segments"]), float(data["scale"]))
    np.testing.assert_allclose(per_query, data["packed_dense_kl"], atol=1e-6)
    assert bool(jnp.all(per_query[1, 5:] == 0))
    assert bool(jnp.all(jnp.isfinite(jax.grad(
        lambda s: jnp.sum(indexer_kl(s, data["query"], data["key"],
                                     packed_keep(data["segments"]), float(data["scale"]))))(
        data["scores"]))))


def test_the_kl_moves_the_indexer_scores_and_not_the_attention():
    """The target is a constant of the loss: no gradient reaches the
    query or key, and the scores' gradient is the softmax difference."""
    data = fixture()
    keep = causal(META["length"])
    scale = float(data["scale"])
    grads = jax.grad(
        lambda s, q, k: jnp.sum(indexer_kl(s, q, k, keep, scale)), argnums=(0, 1, 2))(
        data["scores"], data["query"], data["key"])
    assert bool(jnp.all(grads[1] == 0)) and bool(jnp.all(grads[2] == 0))
    masked = jnp.where(keep, data["scores"], -jnp.inf)
    indexer = jax.nn.softmax(masked, axis=-1)
    logits = jnp.einsum("bshd,bthd->bhst", data["query"] * scale, data["key"])
    target = jnp.sum(jax.nn.softmax(jnp.where(keep, logits, -jnp.inf), axis=-1), axis=1)
    target = target / jnp.sum(target, axis=-1, keepdims=True)
    np.testing.assert_allclose(grads[0], indexer - target, atol=1e-6)


# --------------------------------------------------------------------------
# The mixer with and without a top-k
# --------------------------------------------------------------------------


def v32_block(**overrides):
    from test_mla import block_variables, mla_module, fixture as mla_fixture

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
    causal key, so its output and its KL are the dense-indexed layer's.
    Below the sequence the selection drops keys and both differ."""
    indexed, variables, hidden = v32_block(index_topk=None)
    covering, _, _ = v32_block(index_topk=hidden.shape[1])
    selecting, _, _ = v32_block(index_topk=4)
    reference, dense_sown = indexed.apply(variables, hidden, mutable=["indexer"])
    covered, covered_sown = covering.apply(variables, hidden, mutable=["indexer"])
    selected, selected_sown = selecting.apply(variables, hidden, mutable=["indexer"])
    np.testing.assert_allclose(covered, reference, atol=1e-6)
    np.testing.assert_allclose(covered_sown["indexer"]["kl"][0], dense_sown["indexer"]["kl"][0], atol=1e-6)
    assert float(jnp.max(jnp.abs(selected - reference))) > 0.1
    # Positions the top-4 covers agree, later ones do not.
    np.testing.assert_allclose(selected_sown["indexer"]["kl"][0][:, :4],
                               dense_sown["indexer"]["kl"][0][:, :4], atol=1e-6)
    assert float(jnp.max(jnp.abs(selected_sown["indexer"]["kl"][0][:, 4:]
                                 - dense_sown["indexer"]["kl"][0][:, 4:]))) > 0.1


def test_the_kl_is_sown_only_when_its_collection_is_open():
    module, variables, hidden = v32_block()
    _, sown = module.apply(variables, hidden, mutable=["qk"])
    assert "indexer" not in sown


def test_a_packed_batch_selects_inside_its_documents():
    """Packed positions restart per document, so the selection has to
    read the rows' causal order and the segment mask, not the positions:
    with a top-k covering every key the sparse layer attends as the
    dense-indexed one, and each document's tokens come out as they do
    alone. Before the fix the mask read the positions as row order and
    the covering top-k differed from dense by 2.8."""
    rng = np.random.default_rng(4)
    first, second = rng.integers(1, VOCAB, size=(3,)), rng.integers(1, VOCAB, size=(4,))
    tokens = jnp.asarray(np.concatenate([first, second, [0] * (SEQ - 7)])[None], jnp.int32)
    segments = jnp.asarray(np.array([[1] * 3 + [2] * 4 + [0] * (SEQ - 7)]))
    positions = jnp.asarray(np.array([list(range(3)) + list(range(4)) + [0] * (SEQ - 7)]))
    covering, dense = deepseek_stack(SEQ), deepseek_stack(None)
    params = covering.init(jax.random.key(0), tokens)
    packed = covering.apply(params, tokens, positions=positions, segment_ids=segments)
    np.testing.assert_allclose(
        packed[:, :7], dense.apply(params, tokens, positions=positions, segment_ids=segments)[:, :7],
        atol=1e-6)
    alone = jnp.asarray(np.concatenate([second, [0] * (SEQ - 4)])[None], jnp.int32)
    np.testing.assert_allclose(packed[:, 3:7], covering.apply(params, alone)[:, :4], atol=1e-5)


# --------------------------------------------------------------------------
# The phases through the objective
# --------------------------------------------------------------------------


def deepseek_stack(index_topk, **overrides) -> CausalTransformer:
    return CausalTransformer(**{**dict(
        vocab_size=VOCAB, emb_features=32, num_layers=2, num_heads=2,
        head_dim=16, mlp_features=64, max_seq_len=SEQ,
        mixer={"kind": "mla", "q_lora_rank": 8, "kv_lora_rank": 8,
               "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 8,
               "index_topk": index_topk, "index_n_heads": 2, "index_head_dim": 16}),
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


def test_the_warmup_trains_the_indexer_and_nothing_else():
    """The params collection holds the indexer alone and the frozen one the
    rest, the loss reaches only the former, and thirty Adam steps halve the
    KL: 0.32 to 0.09 observed."""
    objective = LMObjective(deepseek_stack(None), SEQ, indexer=IndexerTraining("warmup"))
    params = objective.init(jax.random.key(0))
    batch = token_batch()
    assert sorted(params) == [FROZEN, "params"]
    assert all(is_indexer(path) for path, _ in jax.tree_util.tree_leaves_with_path(params["params"]))
    assert not any(is_indexer(path) for path, _ in jax.tree_util.tree_leaves_with_path(params[FROZEN]))
    loss = jax.jit(jax.value_and_grad(
        lambda p: scalar_loss(objective, p, batch, step_at()), has_aux=True))
    (before, aux), grads = loss(params)
    assert set(aux.metrics) == {"indexer_kl"}
    assert float(before) == pytest.approx(float(aux.metrics["indexer_kl"]))
    assert float(optax.tree.norm(grads[FROZEN])) == 0
    assert float(optax.tree.norm(grads["params"])) > 0
    optimizer = optax.adam(1e-2)
    state = optimizer.init(params["params"])
    for _ in range(30):
        (after, aux), grads = loss(params)
        updates, state = optimizer.update(grads["params"], state, params["params"])
        params = {**params, "params": optax.apply_updates(params["params"], updates)}
    assert float(after) < float(before) / 2, (float(before), float(after))


def test_the_sparse_phase_keeps_the_indexer_and_the_model_apart():
    """The main weights see the plain cross entropy gradient whatever the
    KL weighs, the indexer's gradient is the KL's alone and scales with
    its weight, and without the term the indexer moves nowhere."""
    batch = token_batch()
    plain = LMObjective(deepseek_stack(4), SEQ)
    params = plain.init(jax.random.key(0))

    def gradient(objective):
        (_, aux), grads = jax.value_and_grad(
            lambda p: scalar_loss(objective, p, batch, step_at()), has_aux=True)(params)
        return aux, grads["params"]

    _, plain_grads = gradient(plain)
    assert split_norms(plain_grads)[0] == 0
    seen = {}
    for weight in (0.5, 2.0):
        aux, grads = gradient(LMObjective(
            deepseek_stack(4), SEQ, indexer=IndexerTraining("sparse", weight=weight)))
        seen[weight] = split_norms(grads)[0]
        main = [(path, leaf) for path, leaf in jax.tree_util.tree_leaves_with_path(grads)
                if not is_indexer(path)]
        reference = dict(jax.tree_util.tree_leaves_with_path(plain_grads))
        for path, leaf in main:
            assert bool(jnp.all(leaf == reference[path])), jax.tree_util.keystr(path)
        assert float(aux.metrics["indexer_kl"]) > 0
    assert seen[2.0] == pytest.approx(4 * seen[0.5], rel=1e-5)


def test_the_loss_is_the_cross_entropy_plus_the_weighted_kl():
    """One report, one sum: the sparse phase's value is the plain
    objective's cross entropy plus `weight` times the reported KL."""
    batch = token_batch()
    plain = LMObjective(deepseek_stack(4), SEQ)
    params = plain.init(jax.random.key(0))
    ce, _ = scalar_loss(plain, params, batch, step_at())
    sparse = LMObjective(deepseek_stack(4), SEQ, indexer=IndexerTraining("sparse", weight=0.25))
    value, aux = scalar_loss(sparse, params, batch, step_at())
    assert float(aux.metrics["ce"]) == pytest.approx(float(ce), rel=1e-6)
    assert float(value) == pytest.approx(float(ce) + 0.25 * float(aux.metrics["indexer_kl"]), rel=1e-5)


def test_a_packed_batch_scores_the_kl_of_its_documents_alone():
    """Two documents packed into one row, then each alone with padding:
    the KL per counted query is the same, so the packed mask isolates the
    documents inside the indexer and its target, and the padding queries
    count nothing. A row of padding alone contributes no query."""
    tokens = np.random.default_rng(3).integers(1, VOCAB, size=(SEQ + 1,))
    first, second = tokens[:7], tokens[7:13]

    def row(*documents):
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

    together = row(first, second)
    apart = [row(first), row(second), row()]

    def batch(rows):
        ids, segments, positions = (np.stack(column) for column in zip(*rows))
        return {"text": jnp.asarray(ids), "text_segment_ids": jnp.asarray(segments),
                "text_positions": jnp.asarray(positions)}

    for phase, topk in (("warmup", None), ("sparse", 4)):
        objective = LMObjective(deepseek_stack(topk), SEQ, indexer=IndexerTraining(phase))
        params = objective.init(jax.random.key(0))
        _, packed = scalar_loss(objective, params, batch([together]), step_at())
        _, alone = scalar_loss(objective, params, batch(apart), step_at())
        assert np.isfinite(float(packed.metrics["indexer_kl"]))
        assert float(packed.metrics["indexer_kl"]) == pytest.approx(
            float(alone.metrics["indexer_kl"]), rel=1e-5), phase


def test_the_warmup_starts_a_fresh_indexer_beside_a_dense_checkpoint():
    """A dense tree (no indexer) is what the warm-up begins from: its
    leaves land frozen, unchanged, and the indexer comes from the init.
    A tree missing anything else is refused by name."""
    dense = deepseek_stack(None, mixer={"kind": "mla", "q_lora_rank": 8, "kv_lora_rank": 8,
                                        "qk_nope_head_dim": 8, "qk_rope_head_dim": 8,
                                        "v_head_dim": 8})
    checkpoint = dense.init(jax.random.key(5), jnp.zeros((1, SEQ), jnp.int32))
    objective = LMObjective(deepseek_stack(None), SEQ, indexer=IndexerTraining("warmup"),
                            pretrained=checkpoint)
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
    with pytest.raises(ValueError, match="embed_tokens"):
        LMObjective(deepseek_stack(None), SEQ, indexer=IndexerTraining("warmup"),
                    pretrained=incomplete).init(jax.random.key(0))


def test_the_sparse_phase_reads_the_warmup_tree():
    """The warm-up's split tree hands over as one params collection, leaf
    for leaf, and the plain objective reads it the same way."""
    warmup = LMObjective(deepseek_stack(None), SEQ, indexer=IndexerTraining("warmup"))
    split = warmup.init(jax.random.key(0))
    for objective in (LMObjective(deepseek_stack(4), SEQ, indexer=IndexerTraining("sparse"),
                                  pretrained=split),
                      LMObjective(deepseek_stack(4), SEQ, pretrained=split)):
        params = objective.init(jax.random.key(1))
        assert sorted(params) == ["params"]
        leaves = dict(jax.tree_util.tree_leaves_with_path(params["params"]))
        expected = dict(jax.tree_util.tree_leaves_with_path(split["params"]))
        expected.update(jax.tree_util.tree_leaves_with_path(split[FROZEN]))
        assert leaves.keys() == expected.keys()
        assert all(bool(jnp.all(leaves[path] == expected[path])) for path in leaves)


def test_evaluation_and_scoring_read_the_split_tree():
    """Teacher-forced scores under the warm-up split are the dense model's
    on the merged tree."""
    objective = LMObjective(deepseek_stack(None), SEQ, indexer=IndexerTraining("warmup"))
    params = objective.init(jax.random.key(0))
    batch = token_batch()
    scored = objective.evaluate(params, batch, step_at())
    merged = LMObjective(deepseek_stack(None), SEQ, pretrained=params).init(jax.random.key(0))
    expected = objective.token_scores(merged, batch["text"])
    np.testing.assert_allclose(scored.losses, expected.losses, rtol=1e-6, atol=1e-6)
    assert not objective.ema.select((FROZEN, "layers_0"))
    assert objective.ema.select(("params", "layers_0"))


@pytest.mark.parametrize("phase, topk, terms", [
    ("warmup", 4, {}),
    ("sparse", None, {}),
    ("warmup", None, {"balance_rate": 0.01}),
    ("warmup", None, {"mtp_weight": 0.1}),
])
def test_a_phase_that_disagrees_with_the_model_is_refused(phase, topk, terms):
    """The warm-up wants dense attention, the sparse phase a top-k, and
    the warm-up moves nothing a main-loss term could train."""
    model = deepseek_stack(topk, num_nextn_predict_layers=1 if "mtp_weight" in terms else 0,
                           **({"mixture": {"experts": 4, "top_k": 2, "layers": (1,), "bias": True,
                                          "expert_features": 16}}
                              if "balance_rate" in terms else {}))
    with pytest.raises(ValueError):
        LMObjective(model, SEQ, indexer=IndexerTraining(phase), **terms)


def test_a_model_without_an_indexer_is_refused():
    plain = deepseek_stack(None, mixer={"kind": "mla", "q_lora_rank": 8, "kv_lora_rank": 8,
                                        "qk_nope_head_dim": 8, "qk_rope_head_dim": 8,
                                        "v_head_dim": 8})
    with pytest.raises(ValueError, match="index_n_heads"):
        LMObjective(plain, SEQ, indexer=IndexerTraining("sparse"))
    with pytest.raises(ValueError, match="positive"):
        IndexerTraining("sparse", weight=0.0)
    with pytest.raises(ValueError, match="warmup"):
        IndexerTraining("dense")


class RecordingTracker:
    def __init__(self):
        self.scalars = []

    def log(self, scalars, step):
        self.scalars.append(dict(scalars))

    def artifact(self, value, step):
        pass


class Data:
    def __init__(self, batches):
        self._batches = batches

    def train(self):
        return self._batches()

    val, batch, records, steps_per_epoch = None, 8, None, None


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

    state = trainer.fit(Data(token_batches), steps=3, log_every=1)

    assert jax.tree.structure(state.opt_state) == jax.tree.structure(
        optax.adam(1e-2).init(initial.params["params"]))
    assert all(is_indexer(path) for path, _ in jax.tree_util.tree_leaves_with_path(state.params["params"]))
    frozen_before = jax.tree.leaves(initial.params[FROZEN])
    frozen_after = jax.tree.leaves(state.params[FROZEN])
    assert all(bool(jnp.all(a == b)) for a, b in zip(frozen_before, frozen_after, strict=True))
    moved = jax.tree.leaves(jax.tree.map(lambda a, b: jnp.any(a != b),
                                         initial.params["params"], state.params["params"]))
    assert all(bool(x) for x in moved)
    kls = [entry["train/indexer_kl"] for entry in tracker.scalars if "train/indexer_kl" in entry]
    assert len(kls) == 3 and all(np.isfinite(kls)) and kls[-1] < kls[0], kls
    assert state.params[FROZEN]["layers_0"]["self_attn"]["kv_b_proj"]["kernel"].sharding.spec == (
        state.params["params"]["layers_0"]["self_attn"][INDEXER]["wq_b"]["kernel"].sharding.spec)
    abstract = jax.eval_shape(trainer.initial_state)
    trainer.layout.check(abstract.params, trainer.shardings(abstract).params, trainer.device_mesh)
