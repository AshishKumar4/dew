"""Routing replay: a trainer forward uses the experts the rollout engine used.

R3 (arXiv 2510.11370) and verl's `router_replay_patch.py` replace the
router's top-k selection with the recorded ids and still gather the gate
weights from the trainer's scores. These tests pin that contract on Dew's
decoder: the replayed choice equals the rollout's exactly even where the
trainer's own top-k would differ, however the stack runs (plain, scanned,
banked on the device or the host, pipelined over stages); the gate keeps its
gradient; and an engine's `[tokens, layers, top_k]` record lands on the
right layers. Where two ways of running the stack score the same replay,
they are on par from the stack's float64 run over drawn models, tokens and
records (check 1, tests/reference_error.py).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import as_given, assert_as_exact, with_gradients

from dew.inference.banks import HeldBanks
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.mixers.mlp import MLPMixer
from dew.nn.sharding import pipeline_microbatches
from dew.objectives.lm import LMObjective
from dew.objectives.rl.sessions import ROUTED_EXPERTS_KEY, ROUTED_KEY, Call, Session, Status, pack
from dew.training import Layout, MeshSpec

VOCAB, SEQ_LEN, LAYERS, EXPERTS, TOP_K = 64, 16, 4, 8, 2
SEEDS = {'gate': 128, 'banked': 32, 'pipeline': 8, 'conflicting prefix': 128, 'packed': 64}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (on armada):
the most any of a site's cases took."""


def objective(scan: bool = False, dtype=None, layers: tuple[int, ...] | None = None,
              parallel: bool = False, **fields) -> LMObjective:
    model = CausalTransformer(
        vocab_size=VOCAB, emb_features=32, num_layers=LAYERS, num_heads=2, num_kv_heads=1,
        mlp_features=64, max_seq_len=SEQ_LEN, scan_layers=scan, dtype=dtype,
        mixture=Mixture(experts=EXPERTS, top_k=TOP_K, layers=layers, parallel=parallel), **fields)
    return LMObjective(model, SEQ_LEN)


def choices(obj: LMObjective, params, tokens, routes=None, key="indices"):
    """Every sparse layer's sown choice, by layer index."""
    sown = obj.token_scores(params, tokens, routing=True, routes=routes).routing
    assert sown is not None
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


def replay_draw(obj, batch: int):
    """Check 1's draw of a replay: a seed's parameters and tokens, and a record one expert off the
    router's own choice on every layer, so the replay moves each token."""
    init = jax.jit(obj.init)
    sown = jax.jit(lambda params, tokens: obj.token_scores(params, tokens, routing=True).routing)

    def draw(seed):
        params = init(jax.random.fold_in(jax.random.key(0), seed))
        tokens = tokens_of(batch, seed=1000 + seed)
        chosen = {int(name.split('_')[1]): np.asarray(tree['mlp']['gate']['indices'][0])
                  for name, tree in sown(params, tokens).items()}
        return params, tokens, (engine_layout(chosen, batch) + 1) % EXPERTS
    return draw


def replayed(obj):
    """`obj`'s replayed log-probabilities as a function of a draw (`replay_draw`)."""
    return jax.jit(lambda params, tokens, routed: replayed_log_probs(obj, params, tokens, (routed, None)))


@pytest.mark.parametrize("scan", [False, True])
def test_replay_reproduces_the_rollout_routing_where_the_trainer_would_flip(scan):
    """The rollout ran an older policy in bf16; the trainer runs the update in
    fp32. Its own top-k disagrees with the rollout's on some tokens, and the
    replayed forward uses the rollout's experts on every token of every layer."""
    trainer, engine = objective(scan), objective(scan, dtype=jnp.bfloat16)
    params = trainer.init(jax.random.key(0))
    stale = jax.tree.map(
        lambda leaf: leaf + 0.05 * jax.random.normal(jax.random.key(3), leaf.shape, leaf.dtype), params
    )
    tokens = tokens_of(4)
    recorded = choices(engine, stale, tokens)
    native = choices(trainer, params, tokens)
    replayed = choices(trainer, params, tokens, routes=(engine_layout(recorded, 4), None))
    assert sum(int(np.any(native[layer] != recorded[layer], -1).sum()) for layer in native) > 0, \
        "the fixture must make the trainer's own routing disagree"
    for layer in native:
        np.testing.assert_array_equal(replayed[layer], recorded[layer])


def test_a_routed_mixer_replays_the_record_as_a_routed_feedforward_does():
    """Nemotron-H routes in its mixer's slot (an `MLPMixer` with a mixture),
    not its feed-forward's; a replayed record reaches that site too, so the
    layer runs the recorded experts rather than its own."""
    model = CausalTransformer(
        vocab_size=VOCAB, emb_features=32, num_layers=2, num_heads=2, num_kv_heads=1, mlp_features=0,
        max_seq_len=SEQ_LEN, layer_types=("moe", "full_attention"),
        kinds={"moe": LayerKind(mixer=MLPMixer(intermediate_size=32,
                                               mixture=Mixture(experts=EXPERTS, top_k=TOP_K)))})
    obj = LMObjective(model, SEQ_LEN)
    params = obj.init(jax.random.key(0))
    tokens = tokens_of(2)

    def chosen(routes=None):
        sown = obj.token_scores(params, tokens, routing=True, routes=routes).routing
        (indices,) = [leaf for path, leaf in jax.tree_util.tree_leaves_with_path(sown)
                      if path[0].key == "layers_0" and path[-2].key == "indices"]
        return np.asarray(indices)

    native = chosen()
    recorded = (native + 1) % EXPERTS
    replayed = chosen((engine_layout({0: recorded}, 2)[:, :, :2], None))
    np.testing.assert_array_equal(replayed, recorded)


def test_gpt_oss_experts_in_a_mixer_slot_replay_the_record():
    """GPT OSS's experts, which a mixture builds under swigluoai (`Mixture.build`), replay a record
    in a mixer's slot as in a feed-forward's: two records score the tokens differently, where a
    slot that dropped the record would run the router's own experts for both."""
    model = CausalTransformer(
        vocab_size=VOCAB, emb_features=32, num_layers=2, num_heads=2, num_kv_heads=1, mlp_features=0,
        max_seq_len=SEQ_LEN, layer_types=("moe", "full_attention"),
        kinds={"moe": LayerKind(mixer=MLPMixer(intermediate_size=32, activation="swigluoai",
                                               mixture=Mixture(experts=EXPERTS, top_k=TOP_K)))})
    obj = LMObjective(model, SEQ_LEN)
    params = obj.init(jax.random.key(0))
    tokens = tokens_of(2)
    first, second = (np.broadcast_to(np.asarray(experts, np.uint8), (2, SEQ_LEN, TOP_K))
                     for experts in ((0, 1), (2, 3)))
    first_scores, second_scores = (
        np.asarray(replayed_log_probs(obj, params, tokens, (engine_layout({0: record}, 2)[:, :, :2], None)))
        for record in (first, second))
    assert not np.array_equal(first_scores, second_scores)


def test_replaying_a_forwards_own_routing_changes_nothing():
    obj = objective(scan=True)
    params = obj.init(jax.random.key(0))
    tokens = tokens_of(2)
    routed = engine_layout(choices(obj, params, tokens), 2)
    np.testing.assert_array_equal(replayed_log_probs(obj, params, tokens, (routed, None)),
                                  obj.per_token_log_probs(params, tokens))


def test_the_gate_keeps_its_gradient_under_replay():
    """Only the selection is fixed: the replayed loss is smooth in the router
    kernel, and its gradient, taken backward, is on par with the one taken
    forward a kernel entry at a time, over drawn models, tokens and records
    (check 1), the truth the stack's forward derivatives in float64. The loss
    is scored from the whole logits, whose head forward mode passes; the tiled
    head's custom backward is another test's."""
    obj, wide = objective(layers=(1, 3)), objective(layers=(1, 3), dtype=jnp.float64)
    init = jax.jit(obj.init)

    def draw(seed):
        params = init(jax.random.fold_in(jax.random.key(0), seed))
        routed = jax.random.randint(jax.random.fold_in(jax.random.key(2), seed), (2, SEQ_LEN + 1, LAYERS, 1),
                                    0, EXPERTS)
        pair = jnp.concatenate([routed, (routed + 1) % EXPERTS], axis=-1)
        # The loss's cotangent, one, held in the draw as the gate takes it.
        return params, tokens_of(2, seed=1000 + seed), pair, jnp.float32(1)

    def backward_of(loss, one):
        return lambda kernel: with_gradients(loss)(kernel, one)[1]

    def forward(loss, one):
        # A directional derivative a kernel entry at a time, looped: the experts' ragged product has
        # no batching rule past its first dimension.
        def run(kernel):
            basis = jnp.eye(kernel.size, dtype=kernel.dtype).reshape(-1, *kernel.shape)
            return jax.lax.map(lambda tangent: jax.jvp(loss, (kernel,), (tangent,))[1], basis).reshape(
                kernel.shape)
        return run

    def gradient(obj, differentiate):
        def run(params, tokens, routed, one):
            def loss(kernel):
                layer = params['params']['layers_1']
                tree = {**params, 'params': {**params['params'], 'layers_1': {
                    **layer, 'mlp': {**layer['mlp'], 'gate': {'kernel': kernel}}}}}
                return -jnp.sum(obj.token_scores(tree, tokens, routes=(routed, None), logits=True).losses)
            return differentiate(loss, one)(params['params']['layers_1']['mlp']['gate']['kernel'])
        return jax.jit(run)

    backward = gradient(obj, backward_of)
    assert float(jnp.abs(backward(*draw(0))).sum()) > 0
    assert_as_exact(backward, gradient(obj, forward), draw, "gate gradient", seeds=SEEDS['gate'],
                    truth_fn=gradient(wide, forward))


@pytest.mark.parametrize("parallel", [False, True], ids=["router", "gemma4"])
def test_a_replayed_pair_weighs_alike_however_far_below_the_routers_own_choice(parallel):
    """Two replayed experts whose logits are equal share each token's weight
    equally, whether every other expert's logit sits thousands above theirs
    or all are zero: their weights are the softmax of their own two logits.
    The full softmax's probabilities underflow to zero at that distance,
    which Gemma 4's renormalization turned into NaN and the shared router's
    into a dropped branch. The two experts are copies, so either router
    scores the same losses and gradients, bit for bit, the router's own
    gradient zero and the pair's not."""
    obj = objective(parallel=parallel)
    params = obj.init(jax.random.key(0))
    tokens = tokens_of(2)
    direction = jax.random.normal(jax.random.key(5), (32, 1))
    pair = np.isin(np.arange(EXPERTS), (1, 2))
    apart = 1e4 * direction / jnp.linalg.norm(direction) * np.where(pair, -1.0, 1.0)

    def router(path) -> bool:
        names = [entry.key for entry in path]
        return names[-2:] == ["gate", "kernel"] or names[-3:] == ["router", "proj", "kernel"]

    def expert(path) -> bool:
        return "experts" in [entry.key for entry in path]

    def routers(kernel):
        # The pair's experts are copies of one another.
        return jax.tree_util.tree_map_with_path(
            lambda path, leaf: jnp.broadcast_to(kernel, leaf.shape) if router(path)
            else leaf.at[2].set(leaf[1]) if expert(path) else leaf, params)

    routed = np.broadcast_to(np.asarray([1, 2], np.uint8), (2, SEQ_LEN + 1, LAYERS, TOP_K))

    def scored(tree):
        def loss(variables):
            log_probs = replayed_log_probs(obj, variables, tokens, (routed, None))
            return jnp.sum(log_probs), log_probs
        (_, log_probs), gradient = jax.value_and_grad(loss, has_aux=True)(tree)
        return log_probs, gradient

    far, near = scored(routers(apart)), scored(routers(jnp.zeros((32, EXPERTS))))
    for got, want in zip(jax.tree.leaves(far), jax.tree.leaves(near), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    gradients = jax.tree_util.tree_leaves_with_path(far[1])
    assert all(not np.any(np.asarray(leaf)) for path, leaf in gradients if router(path))
    assert all(np.any(np.asarray(leaf)[1:3]) for path, leaf in gradients if expert(path))


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


@pytest.mark.parametrize("where", ["device", "host"])
def test_replay_on_banked_weights_scores_as_on_the_plain_tree(where):
    """A store banked by run, resident or in pinned host memory, is read by
    the prefetch loop one layer at a time; the replay reaches each layer the
    same way and scores what the plain tree scores."""
    layout = Layout(min_shard=1, tolerance=1.0,
                    host_parameters=("params/layers_*",) if where == "host" else ())
    plain = objective()
    banked = objective(scan=True, bank_layers=2)
    draw = replay_draw(plain, 2)
    scored = jax.jit(lambda store, tokens, routed: replayed_log_probs(banked, store, tokens, (routed, None)))

    @as_given
    def from_banks(params, tokens, routed):
        return np.asarray(scored(HeldBanks(params).place(banked.model, layout=layout), tokens, routed))

    assert_as_exact(from_banks, replayed(plain), draw, f"{where} banked replay", seeds=SEEDS['banked'],
                    truth_fn=replayed(objective(dtype=jnp.float64)))
    params, tokens, routed = draw(0)
    native = plain.per_token_log_probs(params, tokens)
    assert not np.array_equal(replayed(plain)(params, tokens, routed), native)


@pytest.mark.mesh(devices=4)
def test_replay_under_a_pipeline_scores_as_on_one_stage():
    """Four layers over two stages fed four microbatches: every stage's
    routers read their own microbatch's slice of the record, on par with the
    stack on one stage (check 1)."""
    obj = objective()
    run = replayed(obj)

    def scored(spec, microbatches=None):
        mesh = spec.build()

        def score(params, tokens, routed):
            with jax.set_mesh(mesh), pipeline_microbatches(microbatches):
                return np.asarray(run(params, tokens, routed))
        return score

    assert_as_exact(scored(MeshSpec(fsdp=2, stage=2), 2), scored(MeshSpec(fsdp=4)), replay_draw(obj, 8),
                    "pipelined replay", seeds=SEEDS['pipeline'],
                    truth_fn=replayed(objective(dtype=jnp.float64)))


def test_pack_extends_recorded_routing_when_the_shared_prefix_agrees():
    """Call 2 merges onto call 1; its record re-covers the whole chain except
    its own last sampled id, which no call forwarded. The engine's uint8 ids
    stay uint8 through the call and the batch."""
    first = Call((1, 2), (3,), (-0.1,), "tool_calls", 0,
                 routed_experts=np.full((2, LAYERS, TOP_K), 1, np.uint8))
    assert first.routed_experts is not None
    routes = np.arange(5 * LAYERS * TOP_K, dtype=np.uint8).reshape(5, LAYERS, TOP_K)
    routes[:2] = first.routed_experts
    second = Call((1, 2, 3, 4), (5, 6), (-0.2, -0.3), "stop", 0, routed_experts=routes)
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


@pytest.mark.parametrize("conflict", [0, 2])
def test_packed_grpo_preserves_each_calls_conflicting_prefix_routing(conflict):
    from dew.objectives.rl.grpo import GRPOObjective

    obj, wide = objective(scan=True), objective(scan=True, dtype=jnp.float64)
    grpo = GRPOObjective(obj.model, SEQ_LEN)
    init = jax.jit(obj.init)
    first_routes = np.broadcast_to(np.arange(TOP_K, dtype=np.uint8), (4, LAYERS, TOP_K)).copy()
    later_routes = np.broadcast_to(first_routes[0], (7, LAYERS, TOP_K)).copy()
    later_routes[conflict] += TOP_K
    first = Call((1, 2, 3), (4, 5), (-.1, -.2), "tool_calls", 0, routed_experts=first_routes)
    second = Call((1, 2, 3, 4, 5, 6), (7, 8), (-.3, -.4), "stop", 1, routed_experts=later_routes)
    batch = pack([Session("t", "g", 0, 0, (first, second), Status.COMPLETED, 1.)], SEQ_LEN + 1)
    sampled = batch["response_mask"] != 0
    order = np.argsort(batch["call_index"][sampled], kind="stable")
    packed = jax.jit(lambda params: grpo.packed_log_probs(params, batch))

    def calls(obj):
        run = replayed(obj)

        def direct(params):
            found = []
            for call in (first, second):
                assert call.routed_experts is not None
                ids = np.zeros((1, SEQ_LEN + 1), np.int32)
                size = len(call.prompt) + len(call.tokens)
                ids[0, :size] = call.prompt + call.tokens
                routes = np.pad(call.routed_experts, ((0, SEQ_LEN + 2 - size), (0, 0), (0, 0)))[None]
                found.append(run(params, jnp.asarray(ids), routes)[0, len(call.prompt) - 1:size - 1])
            return jnp.concatenate(found)
        return direct

    def draw(seed):
        return (init(jax.random.fold_in(jax.random.key(0), seed)),)

    assert_as_exact(as_given(lambda params: np.asarray(packed(params))[sampled][order]), calls(obj), draw,
                    f"{conflict} conflicting prefix", seeds=SEEDS['conflicting prefix'], truth_fn=calls(wide))
    np.testing.assert_array_equal(batch["behavior_log_probs"][sampled][order],
                                  np.array([-.1, -.2, -.3, -.4], np.float32))
    assert len(np.unique(batch["text_segment_ids"][sampled])) == 2


def test_packed_grpo_replays_the_routing_its_batch_carries():
    """`pack` lays call records out as `routed_experts`; the GRPO objective's
    packed scoring replays them, and scores the batch as a direct replay does."""
    from dew.objectives.rl.grpo import GRPOObjective

    obj = objective(scan=True)
    grpo = GRPOObjective(obj.model, SEQ_LEN)
    packed = jax.jit(grpo.packed_log_probs)

    @as_given
    def through_pack(params, tokens, routed):
        ids = [int(token) for token in np.asarray(tokens[0])]
        call = Call(tuple(ids[:5]), tuple(ids[5:]), (-1.0,) * (SEQ_LEN - 4), "length", 0,
                    routed_experts=np.asarray(routed)[0, :SEQ_LEN])
        batch = pack([Session("t", "g", 0, 0, (call,), Status.COMPLETED, 1.0)], SEQ_LEN + 1)
        return np.asarray(packed(params, batch))[0, 5:]

    def direct(obj):
        run = replayed(obj)
        return lambda params, tokens, routed: run(params, tokens, routed)[0, 4:]

    draw = replay_draw(obj, 1)
    assert_as_exact(through_pack, direct(obj), draw, "packed replay", seeds=SEEDS['packed'],
                    truth_fn=direct(objective(scan=True, dtype=jnp.float64)))
    params, tokens, routed = draw(0)
    native = np.asarray(obj.per_token_log_probs(params, tokens))[0, 4:]
    assert not np.array_equal(through_pack(params, tokens, routed), native)


def test_a_multimodal_mixture_replays_through_its_language_model():
    """A vision-conditioned MoE decoder (Gemma 3's wrapper here, Gemma 4
    26B-A4B's in the wild) takes the replay beside the media and hands it to
    its language model's routers."""
    from dew.nn.multimodal import MultimodalTransformer
    from dew.nn.vision import GemmaProjector, SiglipVision

    text = CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, num_kv_heads=2,
                             head_dim=8, mlp_features=32, max_seq_len=SEQ_LEN, dtype=jnp.float32,
                             attention_impl="reference", mixture=Mixture(experts=4, top_k=2))
    vision = SiglipVision(
        hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2, image_size=8, patch_size=4
    )
    projection = GemmaProjector(text_width=16, patches_per_side=2, tokens_per_side=1)
    model = MultimodalTransformer(
        text, vision, projection, family="gemma3", image_token_id=1, dtype=jnp.float32
    )
    tokens = jnp.asarray([[2, 1, 3, 4, 5, 6]], jnp.int32)
    indices = jnp.asarray([[-1, 0, -1, -1, -1, -1]], jnp.int32)
    media = {"pixel_values": jnp.linspace(-0.5, 0.5, 3 * 8 * 8).reshape(1, 1, 3, 8, 8)}
    variables = model.init(jax.random.key(0), tokens, image_indices=indices, conditioning=media)
    routed = np.stack([np.stack([(np.arange(6) + layer) % 4, (np.arange(6) + layer + 1) % 4], -1)
                       for layer in range(2)], 1)[None]

    def run(**replay):
        _, sown = model.apply(variables, tokens, image_indices=indices, conditioning=media,
                              method=type(model).hidden_states, mutable=["router"], **replay)
        return {
            name: np.asarray(tree["mlp"]["gate"]["indices"][0])
            for name, tree in sown["router"]["language_model"].items()
        }

    replayed = run(routed_experts=routed)
    for layer in range(2):
        np.testing.assert_array_equal(replayed[f"layers_{layer}"], routed[:, :, layer])
    assert any(not np.array_equal(own, replayed[name]) for name, own in run().items())
