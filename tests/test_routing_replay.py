"""Routing replay: a trainer forward uses the experts the rollout engine used.

R3 (arXiv 2510.11370) and verl's `router_replay_patch.py` replace the
router's top-k selection with the recorded ids and still gather the gate
weights from the trainer's scores. These tests pin that contract on Dew's
decoder: the replayed choice equals the rollout's exactly even where the
trainer's own top-k would differ, however the stack runs (plain, scanned,
banked on the device or the host, pipelined over stages); the gate keeps its
gradient; and an engine's `[tokens, layers, top_k]` record lands on the
right layers.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.inference.banks import HeldBanks, host_banked
from dew.nn.backbones.causal_transformer import CausalTransformer, Mixture
from dew.nn.sharding import pipeline_microbatches
from dew.objectives.lm import LMObjective
from dew.objectives.lm.objective import router_counts
from dew.objectives.rl.sessions import ROUTED_EXPERTS_KEY, ROUTED_KEY, Call, Session, Status, pack
from dew.training import Layout, MeshSpec, build_mesh

VOCAB, SEQ_LEN, LAYERS, EXPERTS, TOP_K = 64, 16, 4, 8, 2


def objective(scan: bool = False, dtype=None, every: int | None = None, **fields) -> LMObjective:
    model = CausalTransformer(
        vocab_size=VOCAB, emb_features=32, num_layers=LAYERS, num_heads=2, num_kv_heads=1,
        mlp_features=64, max_seq_len=SEQ_LEN, scan_layers=scan, dtype=dtype,
        mixture=Mixture(experts=EXPERTS, top_k=TOP_K, every=every), **fields)
    return LMObjective(model, SEQ_LEN)


def choices(obj: LMObjective, params, tokens, routes=None, key="indices"):
    """Every sparse layer's sown choice, by layer index."""
    sown = obj.token_scores(params, tokens, routing=True, routes=routes).routing
    return {int(name.split('_')[1]): np.asarray(sown[name]['mlp']['gate'][key][0]) for name in sown}


def engine_layout(chosen: dict, batch: int) -> np.ndarray:
    """A record as vLLM returns it: every decoder layer indexed, the dense
    ones zero, and no row for the last id."""
    routed = np.zeros((batch, SEQ_LEN + 1, LAYERS, TOP_K), np.uint8)
    for layer, indices in chosen.items():
        routed[:, :-1, layer] = indices
    return routed


def tokens_of(batch: int, seed: int = 1):
    return jax.random.randint(jax.random.key(seed), (batch, SEQ_LEN + 1), 0, VOCAB)


def replayed_log_probs(obj, params, tokens, routes):
    return -obj.token_scores(params, tokens, routes=routes).losses


@pytest.mark.parametrize("scan", [False, True])
def test_replay_reproduces_the_rollout_routing_where_the_trainer_would_flip(scan):
    """The rollout ran an older policy in bf16; the trainer runs the update in
    fp32. Its own top-k disagrees with the rollout's on some tokens, and the
    replayed forward uses the rollout's experts on every token of every layer."""
    trainer, engine = objective(scan), objective(scan, dtype=jnp.bfloat16)
    params = trainer.init(jax.random.key(0))
    stale = jax.tree.map(lambda leaf: leaf + 0.05 * jax.random.normal(jax.random.key(3), leaf.shape, leaf.dtype),
                         params)
    tokens = tokens_of(4)
    recorded = choices(engine, stale, tokens)
    native = choices(trainer, params, tokens)
    replayed = choices(trainer, params, tokens, routes=(engine_layout(recorded, 4), None))
    assert sum(int(np.any(native[layer] != recorded[layer], -1).sum()) for layer in native) > 0, \
        "the fixture must make the trainer's own routing disagree"
    for layer in native:
        np.testing.assert_array_equal(replayed[layer], recorded[layer])


def test_replaying_a_forwards_own_routing_changes_nothing():
    obj = objective(scan=True)
    params = obj.init(jax.random.key(0))
    tokens = tokens_of(2)
    routed = engine_layout(choices(obj, params, tokens), 2)
    np.testing.assert_array_equal(replayed_log_probs(obj, params, tokens, (routed, None)),
                                  obj.per_token_log_probs(params, tokens))


def test_the_gate_keeps_its_gradient_under_replay():
    """Only the selection is fixed: the replayed loss is smooth in the router
    kernel, and its gradient matches a central difference along a direction."""
    obj = objective(every=2)
    params = obj.init(jax.random.key(0))
    tokens = tokens_of(2)
    routed = jax.random.randint(jax.random.key(2), (2, SEQ_LEN + 1, LAYERS, 1), 0, EXPERTS)
    routed = jnp.concatenate([routed, (routed + 1) % EXPERTS], axis=-1)

    def loss(kernel):
        layer = params['params']['layers_1']
        tree = {**params, 'params': {**params['params'], 'layers_1': {
            **layer, 'mlp': {**layer['mlp'], 'gate': {'kernel': kernel}}}}}
        return jnp.sum(replayed_log_probs(obj, tree, tokens, (routed, None)))

    kernel = params['params']['layers_1']['mlp']['gate']['kernel']
    gradient = jax.grad(loss)(kernel)
    direction = jax.random.normal(jax.random.key(4), kernel.shape)
    step = 1e-3
    numeric = (loss(kernel + step * direction) - loss(kernel - step * direction)) / (2 * step)
    assert float(jnp.abs(gradient).sum()) > 0
    np.testing.assert_allclose(float(jnp.vdot(gradient, direction)), float(numeric), rtol=2e-2)


def test_tokens_without_a_record_keep_the_routers_own_choice():
    obj = objective()
    params = obj.init(jax.random.key(0))
    tokens = tokens_of(2)
    native = choices(obj, params, tokens)
    routed = (engine_layout(native, 2) + 3) % EXPERTS
    covered = np.zeros((2, SEQ_LEN + 1), bool)
    covered[:, :5] = True
    replayed = choices(obj, params, tokens, routes=(routed, covered))
    for layer, choice in replayed.items():
        np.testing.assert_array_equal(choice[:, :5], routed[:, :5, layer])
    # Replay upstream changes every later layer's input, so the first
    # router alone still sees the states its own choice was made on.
    np.testing.assert_array_equal(replayed[0][:, 5:], native[0][:, 5:])


@pytest.mark.parametrize("layout", [Layout(min_shard=1, tolerance=1.0),
                                    Layout(min_shard=1, tolerance=1.0, host_parameters=("params/layers_*",))],
                         ids=["device", "host"])
def test_replay_on_banked_weights_scores_as_on_the_plain_tree(layout):
    """A store banked by run, resident or in pinned host memory, is read by
    the prefetch loop one layer at a time; the replay reaches each layer the
    same way and scores what the plain tree scores."""
    plain = objective()
    banked = objective(scan=True, bank_layers=2)
    params = plain.init(jax.random.key(0))
    tokens = tokens_of(2)
    routed = (engine_layout(choices(plain, params, tokens), 2) + 1) % EXPERTS
    store = host_banked(banked.model, HeldBanks(params), layout=layout)
    expected = replayed_log_probs(plain, params, tokens, (routed, None))
    np.testing.assert_allclose(replayed_log_probs(banked, store, tokens, (routed, None)), expected, atol=1e-5)
    assert float(jnp.max(jnp.abs(expected - plain.per_token_log_probs(params, tokens)))) > 1e-3


@pytest.mark.mesh(devices=4)
def test_replay_under_a_pipeline_scores_as_on_one_stage():
    """Four layers over two stages fed four microbatches: every stage's
    routers read their own microbatch's slice of the record."""
    obj = objective()
    params = obj.init(jax.random.key(0))
    tokens = tokens_of(8)
    routed = (engine_layout(choices(obj, params, tokens), 8) + 1) % EXPERTS

    def scored(spec):
        mesh = build_mesh(spec)
        with jax.set_mesh(mesh), pipeline_microbatches(spec.microbatches):
            return np.asarray(jax.jit(lambda p: replayed_log_probs(obj, p, tokens, (routed, None)))(params))

    flat = scored(MeshSpec(fsdp=4))
    np.testing.assert_allclose(scored(MeshSpec(fsdp=2, stage=2, microbatches=2)), flat, atol=1e-5)


def test_under_replay_the_bias_counts_the_replayed_experts():
    """Megatron's R3 path as verl patches it: the balancing bias counts the
    experts the tokens were sent to, the replayed ones."""
    obj = objective()
    params = obj.init(jax.random.key(0))
    tokens = tokens_of(2)
    routed = (engine_layout(choices(obj, params, tokens), 2) + 1) % EXPERTS
    sown = obj.token_scores(params, tokens, routing=True, routes=(routed, None)).routing
    counts = router_counts({"layers_0": {"mlp": {"gate": {"e_score_correction_bias": jnp.zeros(EXPERTS)}}}}, sown)
    np.testing.assert_array_equal(counts["layers_0"]["mlp"]["gate"]["e_score_correction_bias"],
                                  np.bincount(routed[:, :-1, 0].ravel(), minlength=EXPERTS))


def test_a_record_that_does_not_fit_the_model_is_refused():
    obj = objective(every=2)
    params = obj.init(jax.random.key(0))
    tokens = jnp.zeros((1, SEQ_LEN + 1), jnp.int32)
    with pytest.raises(ValueError, match="layers, top_k"):
        obj.token_scores(params, tokens, routes=(jnp.zeros((1, SEQ_LEN + 1, 2, TOP_K), jnp.int32), None))
    with pytest.raises(ValueError, match="choosing"):
        obj.token_scores(params, tokens, routes=(jnp.zeros((1, SEQ_LEN + 1, LAYERS, TOP_K + 1), jnp.int32), None))


def test_pack_takes_each_ids_routing_from_the_latest_call_that_forwarded_it():
    """Call 2 merges onto call 1; its record re-covers the whole chain except
    its own last sampled id, which no call forwarded. The engine's uint8 ids
    stay uint8 through the call and the batch."""
    first = Call((1, 2), (3,), (-0.1,), "tool_calls", 0,
                 routed_experts=np.full((2, LAYERS, TOP_K), 1, np.uint8))
    second = Call((1, 2, 3, 4), (5, 6), (-0.2, -0.3), "stop", 0,
                  routed_experts=np.arange(5 * LAYERS * TOP_K, dtype=np.uint8).reshape(5, LAYERS, TOP_K))
    unrecorded = Call((9,), (8,), (-0.4,), "stop", 0)
    rollouts = [Session("t", "g", 0, 0, (first, second), Status.COMPLETED, 1.0),
                Session("t", "g", 1, 0, (unrecorded,), Status.COMPLETED, 0.0)]
    batch = pack(rollouts, 8)
    assert second.routed_experts is not None and second.routed_experts.dtype == np.uint8
    assert batch[ROUTED_EXPERTS_KEY].dtype == np.uint8
    np.testing.assert_array_equal(batch[ROUTED_EXPERTS_KEY][0, :5], second.routed_experts)
    # The unrecorded rollout's chain shares the row, at ids 6 and 7.
    assert batch["input_ids"][0, 6:].tolist() == [9, 8]
    assert batch[ROUTED_KEY][0].tolist() == [True] * 5 + [False] * 3
    assert ROUTED_EXPERTS_KEY not in pack(rollouts[1:], 8)


def test_a_call_record_must_cover_every_forwarded_id():
    with pytest.raises(ValueError, match="3 rows for this call"):
        Call((1, 2), (3, 4), (-0.1, -0.1), "stop", 0, routed_experts=np.zeros((4, 2, 2), np.int32))
    with pytest.raises(ValueError, match="nonnegative"):
        Call((1,), (3,), (-0.1,), "stop", 0, routed_experts=np.full((1, 2, 2), -1, np.int32))


def test_packed_grpo_replays_the_routing_its_batch_carries():
    """`pack` lays call records out as `routed_experts`; the GRPO objective's
    packed scoring replays them, and scores the batch as a direct replay does."""
    from dew.objectives.rl.grpo import GRPOObjective

    obj = objective(scan=True)
    grpo = GRPOObjective(obj.model, SEQ_LEN)
    params = obj.init(jax.random.key(0))
    tokens = tokens_of(1)
    routed = (engine_layout(choices(obj, params, tokens), 1) + 1) % EXPERTS
    ids = [int(token) for token in np.asarray(tokens[0])]
    call = Call(tuple(ids[:5]), tuple(ids[5:]), (-1.0,) * (SEQ_LEN - 4), "length", 0,
                routed_experts=routed[0, :SEQ_LEN])
    batch = pack([Session("t", "g", 0, 0, (call,), Status.COMPLETED, 1.0)], SEQ_LEN + 1)
    scored = np.asarray(grpo.packed_log_probs(params, batch))[0, 5:]
    expected = np.asarray(replayed_log_probs(obj, params, tokens, (routed, None)))[0, 4:]
    native = np.asarray(obj.per_token_log_probs(params, tokens))[0, 4:]
    np.testing.assert_allclose(scored, expected, atol=1e-6)
    assert np.abs(scored - native).max() > 1e-3


def test_a_multimodal_mixture_replays_through_its_language_model():
    """A vision-conditioned MoE decoder (Gemma 3's wrapper here, Gemma 4
    26B-A4B's in the wild) takes the replay beside the media and hands it to
    its language model's routers."""
    from dew.nn.multimodal import MultimodalTransformer
    from dew.nn.vision import GemmaProjector, SiglipVision

    text = CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, num_kv_heads=2,
                             head_dim=8, mlp_features=32, max_seq_len=SEQ_LEN, dtype=jnp.float32,
                             attention_impl="reference", mixture=Mixture(experts=4, top_k=2))
    vision = SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2, image_size=8, patch_size=4)
    projection = GemmaProjector(vision_width=16, text_width=16, patches_per_side=2, tokens_per_side=1)
    model = MultimodalTransformer(text, vision, projection, family="gemma3", image_token_id=1, dtype=jnp.float32)
    tokens = jnp.asarray([[2, 1, 3, 4, 5, 6]], jnp.int32)
    indices = jnp.asarray([[-1, 0, -1, -1, -1, -1]], jnp.int32)
    media = {"pixel_values": jnp.linspace(-0.5, 0.5, 3 * 8 * 8).reshape(1, 1, 3, 8, 8)}
    variables = model.init(jax.random.key(0), tokens, image_indices=indices, conditioning=media)
    routed = np.stack([np.stack([(np.arange(6) + layer) % 4, (np.arange(6) + layer + 1) % 4], -1)
                       for layer in range(2)], 1)[None]

    def run(**replay):
        _, sown = model.apply(variables, tokens, image_indices=indices, conditioning=media,
                              method=type(model).hidden_states, mutable=["router"], **replay)
        return {name: np.asarray(tree["mlp"]["gate"]["indices"][0]) for name, tree in sown["router"]["language_model"].items()}

    replayed = run(routed_experts=routed)
    for layer in range(2):
        np.testing.assert_array_equal(replayed[f"layers_{layer}"], routed[:, :, layer])
    assert any(not np.array_equal(own, replayed[name]) for name, own in run().items())
