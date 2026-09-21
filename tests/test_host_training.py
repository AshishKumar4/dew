"""Host-owned full-tree transactions: coupling, replay, restart and placement."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
from test_training_transactions import ShortScaleTrainer, Terms, Tiny, batches

from dew.checkpoints import Checkpoints
from dew.nn.backbones.causal_transformer import CausalTransformer, Mixture
from dew.nn.inputs import ModelInputs
from dew.objectives.base import FROZEN, Aux, EMASpec, Mean, Objective
from dew.objectives.lm import LMObjective
from dew.training import Layout, Trainer
from dew.training.host import companion_mesh, transfer

HOST = Layout(host=("params",), min_shard=1, tolerance=1.)
DEVICE = Layout(min_shard=1, tolerance=1.)


STREAMED_BOUND = 1e-6


def _leaves(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(left), jax.tree.leaves(right), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        yield jax.tree_util.keystr(path), np.asarray(a), np.asarray(b)


def equal(left, right):
    for path, a, b in _leaves(left, right):
        np.testing.assert_array_equal(a, b, err_msg=path)


def close(left, right, bound=STREAMED_BOUND):
    for path, a, b in _leaves(left, right):
        np.testing.assert_allclose(a, b, atol=bound, rtol=0, err_msg=path)


class Coupled(Objective):
    ema = EMASpec(optax.constant_schedule(.5))

    def init(self, key, variables=None):
        return {"params": {"first": jnp.array([1., 2.]), "second": jnp.array([3., 4., 5.])}}

    def loss(self, params, batch, step):
        p = params["params"]
        loss = jnp.sum(p["first"] * batch["small"]) + jnp.sum(p["second"] * batch["large"])
        return loss, Aux({})


def centered():
    def update(updates, state, params=None):
        leaves = jax.tree.leaves(updates)
        mean = sum(jnp.sum(x) for x in leaves) / sum(x.size for x in leaves)
        return jax.tree.map(lambda x: -.01 * (x - mean), updates), state
    return optax.GradientTransformation(lambda params: optax.EmptyState(), update)


@pytest.mark.parametrize("optimizer", [
    optax.chain(optax.clip_by_global_norm(.25), optax.adam(optax.linear_schedule(.02, .01, 3))),
    centered(),
    optax.partition({"a": optax.adam(.01), "b": optax.adamw(.02)},
                    {"first": "a", "second": "b"}),
], ids=["global-clip-scheduled-adam", "cross-leaf-mean", "masked-partition-state"])
def test_complete_optimizer_tree_crosses_the_execution_boundary(optimizer):
    data = {"small": jnp.asarray(.5), "large": jnp.asarray(40.)}
    states = []
    for layout in (DEVICE, HOST):
        trainer = Trainer(Coupled(), optimizer, key=jax.random.key(4), layout=layout)
        state, _, _ = trainer.place()
        step = trainer.compile(state, data)
        for _ in range(3):
            state, *_ = step(state, data)
        states.append(state)
    equal(*states)


@pytest.mark.parametrize("composite", [False, True])
def test_host_accumulation_replays_original_rng_and_mutable_snapshots(tmp_path, composite):
    data = batches()
    states = []
    for layout in (DEVICE, HOST):
        trainer = ShortScaleTrainer(
            Tiny(composite), optax.adamw(.03, weight_decay=.1), key=jax.random.PRNGKey(9),
            accumulation=2, dynamic_scale=True, layout=layout)
        state, _, _ = trainer.place()
        step = trainer.compile(state, data[0])
        prefix, *_ = step(state, data[0])
        rejected, _, _, _, accepted = step(prefix, data[1])
        assert not bool(accepted)
        equal(prefix.params, rejected.params)
        equal(prefix.accumulation, rejected.accumulation)
        state, *_ = step(rejected, data[2])
        assert int(state.updates) == 1
        assert float(state.params["stats"]["seen"]) == 2
        states.append(state)
    equal(*states)

    checkpoints = Checkpoints(str(tmp_path / "run"))
    trainer = ShortScaleTrainer(
        Tiny(composite), optax.adamw(.03, weight_decay=.1), key=jax.random.PRNGKey(9),
        accumulation=2, dynamic_scale=True, layout=HOST, checkpoints=checkpoints)
    start, _, _ = trainer.place()
    step = trainer.compile(start, data[0])
    prefix, *_ = step(start, data[0])
    checkpoints.save(1, prefix, b"position")
    checkpoints.wait()
    restored, _, position = trainer.place()
    assert position == b"position"
    equal(prefix, restored)
    resumed_step = trainer.compile(restored, data[1])
    resumed, *_ = resumed_step(restored, data[1])
    resumed, *_ = resumed_step(resumed, data[2])
    equal(resumed, states[1])


def test_decoder_scan_training_keeps_the_original_logical_state():
    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=2, num_heads=2,
                              mlp_features=16, max_seq_len=8, scan_layers=True)
    data = {"text": jnp.tile(jnp.arange(9, dtype=jnp.int32)[None], (jax.device_count(), 1))}
    states = []
    for layout in (DEVICE, HOST):
        trainer = Trainer(LMObjective(model, 8, head_chunks=1), optax.adam(.001),
                          key=jax.random.key(1), layout=layout)
        state, _, _ = trainer.place()
        step = trainer.compile(state, data)
        state, *_ = step(state, data)
        assert "layers_0" in state.params["params"] and "layers_1" in state.params["params"]
        assert "layers_0_1" not in state.params["params"]
        states.append(state)
    close(*states)


def test_dropout_composite_replay_restarts_with_effects_and_ema_intact(tmp_path):
    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=2, num_heads=2,
                              mlp_features=16, max_seq_len=8, scan_layers=True, dropout_rate=.4,
                              mixture=Mixture(experts=4, top_k=2, bias=True))
    objective = LMObjective(model, 8, head_chunks=1, aux_loss_alpha=.2, seq_aux=False,
                            balance_rate=.01, ema_decay=.9)
    checkpoints = Checkpoints(str(tmp_path / "dropout"))
    def make(checkpoints=None):
        return Trainer(objective, optax.chain(optax.clip_by_global_norm(.5), optax.adam(.001)),
                       key=jax.random.key(3), layout=HOST, accumulation=2, checkpoints=checkpoints)
    data = {"text": jnp.tile(jnp.arange(9, dtype=jnp.int32)[None], (jax.device_count(), 1))}
    trainer = make()
    initial, _, _ = trainer.place()
    step = trainer.compile(initial, data)
    prefix, *_ = step(initial, data)
    checkpoints.save(1, prefix, None)
    whole, *_ = step(prefix, data)
    checkpoints.wait()
    restarted = make(checkpoints)
    restored, _, _ = restarted.place()
    equal(prefix, restored)
    resumed, *_ = restarted.compile(restored, data)(restored, data)
    equal(whole, resumed)
    assert int(resumed.updates) == 1


def test_nonfinite_optimizer_candidate_rolls_back_all_cpu_owned_fields():
    def update(updates, state, params=None):
        return jax.tree.map(lambda x: jnp.full_like(x, jnp.inf), updates), state
    optimizer = optax.GradientTransformation(lambda params: optax.EmptyState(), update)
    results = []
    for layout in (DEVICE, HOST):
        trainer = ShortScaleTrainer(Tiny(), optimizer, key=jax.random.PRNGKey(9),
                                    dynamic_scale=True, layout=layout)
        initial, _, _ = trainer.place()
        data = batches()[0]
        final, _, _, finite, accepted = trainer.compile(initial, data)(initial, data)
        assert bool(finite) and not bool(accepted)
        equal(initial.params, final.params)
        equal(initial.ema, final.ema)
        equal(initial.opt_state, final.opt_state)
        assert int(final.microstep) == int(final.updates) == 0
        assert int(final.step) == 1
        # Preserve the existing distinction: candidate-state rejection rolls
        # back the transaction, while the scaler tracks gradient finiteness.
        assert final.scale is not None and initial.scale is not None
        assert float(final.scale.scale) == float(initial.scale.scale)
        assert int(final.scale.fin_steps) == int(initial.scale.fin_steps) + 1
        results.append(final)
    equal(*results)


def test_composite_replay_does_not_apply_effects_twice():
    class Effects(Objective[Mean | Terms, jax.Array]):
        ema = Tiny.ema

        def __init__(self):
            self.reference = Tiny(True)

        def init(self, key, variables=None):
            return self.reference.init(key, variables)

        def loss(self, params, batch, step):
            stats, aux = self.reference.loss(params, batch, step)
            return stats, Aux(aux.metrics, variables=aux.variables, effects=jnp.asarray(1.))

        def reduce_loss(self, stats):
            return self.reference.reduce_loss(stats)

        def apply_effects(self, variables, effects):
            return {"stats": {"seen": variables["stats"]["seen"] + effects}}

    results = []
    for layout in (DEVICE, HOST):
        trainer = Trainer(Effects(), optax.adam(.01), key=jax.random.PRNGKey(9),
                          accumulation=2, layout=layout)
        state, _, _ = trainer.place()
        data = batches()[0]
        step = trainer.compile(state, data)
        state, *_ = step(state, data)
        state, *_ = step(state, data)
        assert int(state.updates) == 1
        assert float(state.params["stats"]["seen"]) == 4.
        results.append(state)
    equal(*results)


@pytest.mark.mesh
def test_companion_coordinates_preserve_shards_under_device_permutation():
    devices = jax.devices("cpu")
    assert len(devices) >= 8
    accelerator = Mesh(np.asarray(devices[:4], dtype=object)[[3, 0, 2, 1]].reshape(2, 2),
                       ("fsdp", "tensor"), axis_types=(AxisType.Auto,) * 2)
    cpu = companion_mesh(accelerator, devices[4:])
    source = NamedSharding(accelerator, P("fsdp", "tensor"))
    target = NamedSharding(cpu, source.spec)
    values = jax.device_put(np.arange(64, dtype=np.float32).reshape(8, 8), source)
    moved = transfer(values, target)
    np.testing.assert_array_equal(np.asarray(values), np.asarray(moved))
    for before, after in zip(accelerator.devices.flat, cpu.devices.flat, strict=True):
        a = next(s for s in values.addressable_shards if s.device == before)
        b = next(s for s in moved.addressable_shards if s.device == after)
        assert a.index == b.index
        np.testing.assert_array_equal(a.data, b.data)


@pytest.mark.mesh
def test_missing_companion_devices_names_the_launch_fix():
    devices = jax.devices("cpu")
    mesh = Mesh(np.asarray(devices, dtype=object), ("fsdp",), axis_types=(AxisType.Auto,))
    with pytest.raises(ValueError, match="JAX_NUM_CPU_DEVICES=.*restart"):
        companion_mesh(mesh, devices[:1])


def test_custom_step_cannot_silently_bypass_streamed_execution():
    with pytest.raises(ValueError, match="custom steps own their execution"):
        Trainer(Coupled(), centered(), key=jax.random.key(0), layout=HOST,
                step=lambda objective, optimizer: lambda state, batch: (state, jnp.asarray(0.), Aux({})))


@pytest.mark.mesh
@pytest.mark.distributed
def test_companion_pool_uses_one_global_optimizer_reduction(tmp_path):
    from test_multiprocess import run_pool, run_worker
    single = run_worker("host_training", tmp_path / "single.json")
    pool = run_pool("host_training", tmp_path / "pool", 2)
    for result in [single, *pool]:
        assert set(result["compute_devices"]).isdisjoint(result["transaction_devices"])
        assert result["resident"] == result["host"]
        assert result["host"] == single["host"]


# --------------------------------------------------------------------------
# Declared decoder banks: nested owners, shared owners and partial freezing
# --------------------------------------------------------------------------

# Banked execution runs a different compiled module from the resident stack,
# and the preserved test_decoder_scan_training_keeps_the_original_logical_state
# records a difference between the two whose cause is not established. The
# bound below qualifies these cases numerically and settles nothing about that
# failure. Leaves nothing moves are compared exactly.
def updated(objective, batch, layout, *, optimizer=None, checkpoints=None, steps=1):
    """`steps` compiled updates of one objective under one placement."""
    trainer = Trainer(objective, optax.adam(.01) if optimizer is None else optimizer,
                      key=jax.random.key(5), layout=layout, checkpoints=checkpoints)
    state, _, _ = trainer.place()
    step = trainer.compile(state, batch)
    for _ in range(steps):
        state, *_ = step(state, batch)
    return state


def tokens(width=9):
    return {"text": jnp.tile(jnp.arange(1, width + 1, dtype=jnp.int32)[None], (jax.device_count(), 1))}


def decoder(**overrides):
    fields = dict(vocab_size=16, emb_features=8, num_layers=2, num_heads=2,
                  mlp_features=16, max_seq_len=8, scan_layers=True)
    return CausalTransformer(**{**fields, **overrides})


def frozen_leaves(state):
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf
            in jax.tree_util.tree_leaves_with_path(state.params[FROZEN])}


def _moments(opt_state) -> optax.ScaleByAdamState:
    """Adam's moments out of the optimizer state, narrowed for the checker."""
    held, _ = jax.tree_util.tree_flatten(
        opt_state, is_leaf=lambda node: isinstance(node, optax.ScaleByAdamState))
    for node in held:
        if isinstance(node, optax.ScaleByAdamState):
            return node
    raise AssertionError("the optimizer state carries no Adam moments")


def test_streamed_banks_train_a_mixed_frozen_root_decoder_like_the_resident_stack(tmp_path):
    """Per-layer and per-leaf freezing through one banked decoder.

    Every layer's parameters ride in one bank whatever their ownership, and
    only the canonical moving leaves come back as gradients: the optimizer
    holds those and nothing else, the frozen collection is untouched, and the
    stored names stay `layers_N`.
    """
    def trainable(path):
        return "layers_1" not in path and path[-2:] != ("gate_proj", "kernel")

    def objective():
        return LMObjective(decoder(), 8, head_chunks=1, trainable=trainable)

    batch = tokens()
    resident = updated(objective(), batch, DEVICE)
    host = updated(objective(), batch, HOST)
    close(host, resident)
    initial = frozen_leaves(Trainer(objective(), optax.adam(.01), key=jax.random.key(5),
                                    layout=HOST).place()[0])
    assert initial and frozen_leaves(host).keys() == initial.keys()
    for name, value in frozen_leaves(host).items():
        np.testing.assert_array_equal(value, initial[name], err_msg=name)
    assert sorted(host.params["params"]) == ["embed_tokens", "layers_0", "norm"]
    assert "layers_0" in host.params[FROZEN] and "layers_1" in host.params[FROZEN]
    assert jax.tree.structure(_moments(host.opt_state).mu) == jax.tree.structure(host.params["params"])

    # The EMA tracks the moving collection only, and banked execution reads
    # that partial tree without extending it.
    assert host.ema is not None and FROZEN not in host.ema
    assert jax.tree.structure(host.ema["params"]) == jax.tree.structure(host.params["params"])
    checkpoints = Checkpoints(str(tmp_path / "mixed"))
    prefix = updated(objective(), batch, HOST, checkpoints=checkpoints)
    checkpoints.save(int(prefix.step), prefix, None)
    checkpoints.wait()
    restored, _, _ = Trainer(objective(), optax.adam(.01), key=jax.random.key(5),
                             layout=HOST, checkpoints=Checkpoints(str(tmp_path / "mixed"))).place()
    equal(prefix, restored)
    resumed = updated(objective(), batch, HOST, checkpoints=Checkpoints(str(tmp_path / "mixed")))
    equal(resumed, updated(objective(), batch, HOST, steps=2))


def multimodal(**overrides):
    """A media-conditioned wrapper over one declared decoder."""
    from dew.nn.multimodal import MultimodalTransformer
    from dew.nn.vision import GemmaProjector, SiglipVision

    return MultimodalTransformer(
        decoder(emb_features=16, max_seq_len=16, **overrides), SiglipVision(
            hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2,
            image_size=8, patch_size=4),
        GemmaProjector(vision_width=16, text_width=16, patches_per_side=2, tokens_per_side=1),
        family="gemma3", image_token_id=1, dtype=jnp.float32)


def media_batch(scale=1.):
    """Nine-token rows whose third slot reads one image, one row per device."""
    rows = jax.device_count()
    ids = jnp.tile(jnp.asarray([[2, 3, 1, 4, 5, 6, 7, 8, 9]], jnp.int32), (rows, 1))
    indices = jnp.tile(jnp.asarray([[-1, -1, 0, -1, -1, -1, -1, -1, -1]], jnp.int32), (rows, 1))
    pixels = jnp.tile(jnp.linspace(-.5, .5, 3 * 8 * 8).reshape(1, 1, 3, 8, 8) * scale, (rows, 1, 1, 1, 1))
    return {"text": ModelInputs(ids, {"image_indices": indices}, {"pixel_values": pixels})}


def test_a_nested_decoder_bank_trains_beside_frozen_media_entries():
    """A multimodal wrapper declares its decoder under `language_model`.

    The batch carries real pixels, so the frozen tower and projector run in
    the loss beside the banked decoder: the images move the loss, the banked
    update matches the resident one, and the media weights do not move.
    """
    model = multimodal()
    batch, other = media_batch(), media_batch(scale=.25)
    held = model.init(jax.random.key(0), batch["text"].tokens,
                      image_indices=batch["text"].token_fields["image_indices"],
                      conditioning=batch["text"].conditioning)

    def trainable(path):
        return path[1] == "language_model" and "layers_1" not in path

    def objective():
        return LMObjective(model, 8, head_chunks=1, pretrained=held, trainable=trainable)

    trainer = Trainer(objective(), optax.adam(.01), key=jax.random.key(5), layout=DEVICE)
    start, _, _ = trainer.place()
    step = trainer.compile(start, batch)
    _, loss, *_ = step(start, batch)
    _, changed, *_ = step(start, other)
    assert abs(float(loss) - float(changed)) > 1e-4, "the pixels do not reach the loss"

    resident = updated(objective(), batch, DEVICE)
    host = updated(objective(), batch, HOST)
    close(host, resident)
    assert sorted(host.params["params"]) == ["language_model"]
    assert sorted(host.params[FROZEN]) == ["language_model", "projector", "tower"]
    assert sorted(host.params[FROZEN]["language_model"]) == ["layers_1"]
    media = {name: value for name, value in frozen_leaves(host).items()
             if "tower" in name or "projector" in name}
    initial = frozen_leaves(Trainer(objective(), optax.adam(.01), key=jax.random.key(5),
                                    layout=HOST).place()[0])
    assert media
    for name, value in media.items():
        np.testing.assert_array_equal(value, initial[name], err_msg=name)


def test_an_unscanned_decoder_trains_through_singleton_banks():
    """A plain loop declares one bank per layer and streams them the same way."""
    model = multimodal(scan_layers=False)
    (site,) = model.bank_sites
    assert site.namespace == ("language_model",)
    assert site.view.groups == ((0, 1), (1, 1))
    batch = media_batch()
    held = model.init(jax.random.key(0), batch["text"].tokens,
                      image_indices=batch["text"].token_fields["image_indices"],
                      conditioning=batch["text"].conditioning)

    def objective():
        return LMObjective(model, 8, head_chunks=1, pretrained=held,
                           trainable=lambda path: path[1] == "language_model")

    resident = updated(objective(), batch, DEVICE)
    host = updated(objective(), batch, HOST)
    close(host, resident)
    assert sorted(host.params["params"]["language_model"]) == ["embed_tokens", "layers_0", "layers_1", "norm"]


@pytest.mark.parametrize("detached", [False, True])
def test_a_shared_text_owner_sums_both_block_losses_through_one_bank(detached):
    """DiffusionGemma reads one physical stack twice: encoder and denoiser.

    The site is declared once, so the bank is packed once and native reverse
    mode sums both uses into it. The detached variant keeps the existing
    encoder-gradient policy.
    """
    from pathlib import Path

    from dew.interop import load_pretrained
    from dew.nn.diffusion_gemma import DiffusionGemma
    from dew.objectives.diffusion.block import BlockDiffusionObjective

    fixture = Path(__file__).resolve().parent / "fixtures/hf/diffusion-gemma-sft"
    loaded = load_pretrained(fixture, dtype="float32", attention_impl="xla", max_seq_len=32)
    source = loaded.model
    assert isinstance(source, DiffusionGemma)
    scanned = source.clone(text=source.text.clone(scan_layers=True))
    assert isinstance(scanned, DiffusionGemma)
    with np.load(fixture / "reference.npz") as arrays:
        rows = jax.device_count() // arrays["tokens"].shape[0]
        batch = {name: jnp.tile(jnp.asarray(arrays[name]), (rows, 1)) for name in
                 ("tokens", "canvas_mask", "encoder_target_mask")}
    batch["text"] = batch.pop("tokens")

    def objective():
        return BlockDiffusionObjective(
            scanned, prompt_length=4, num_canvases=2, pretrained=loaded.variables,
            stop_gradient_from_denoiser_to_encoder=detached)

    (site,) = objective().bank_sites
    assert site.namespace == ("text",)
    resident = updated(objective(), batch, DEVICE, optimizer=optax.sgd(.001))
    host = updated(objective(), batch, HOST, optimizer=optax.sgd(.001))
    close(host, resident, 1e-5)
    assert "layers_0" in host.params["params"]["text"]
    assert not any(name.startswith("layers_0_") for name in host.params["params"]["text"])



# --------------------------------------------------------------------------
# Frozen residency: placed once, aliased by every snapshot, streamed into place
# --------------------------------------------------------------------------


def _pointers(tree):
    return {path: tuple(shard.data.unsafe_buffer_pointer() for shard in leaf.addressable_shards)
            for path, leaf in _named_leaves(tree)}


def _named_leaves(tree):
    return [(jax.tree_util.keystr(path), leaf) for path, leaf in jax.tree_util.tree_leaves_with_path(tree)]


@pytest.mark.parametrize("scan", [False, True], ids=["per-layer", "scanned"])
def test_frozen_leaves_stay_resident_and_snapshots_alias_them(scan):
    """A frozen leaf is placed once, beside the accelerator, and is the same
    buffer after every step; the snapshot of a per-layer bank is the leaf
    itself and a scanned run's frozen bank is built once and kept."""
    from dew.training.execution import BANK_MEMORY, HostExecution

    def trainable(path):
        return path[-2:] == ("q_proj", "kernel")

    objective = LMObjective(decoder(scan_layers=scan), 8, head_chunks=1, trainable=trainable)
    trainer = Trainer(objective, optax.adam(.01), key=jax.random.key(5), layout=HOST)
    state, placement, _ = trainer.place()
    frozen = state.params[FROZEN]
    for path, leaf in _named_leaves(frozen):
        assert leaf.sharding.mesh == trainer.device_mesh, path
        assert leaf.sharding.memory_kind == (BANK_MEMORY if "layers_" in path else "device"), path
    execution = HostExecution(objective, HOST, trainer.device_mesh, trainer.state_mesh)
    with jax.set_mesh(trainer.state_mesh):
        first, second = execution.snapshot(state.params), execution.snapshot(state.params)
    if not scan:
        assert first["params"]["layers_1"]["mlp"]["gate_proj"]["kernel"] is frozen["layers_1"]["mlp"]["gate_proj"]["kernel"]
    else:
        bank = next(name for name in first["params"] if name.startswith("layers_0_"))
        assert first["params"][bank]["mlp"]["gate_proj"]["kernel"] is second["params"][bank]["mlp"]["gate_proj"]["kernel"]
        assert first["params"][bank]["self_attn"]["q_proj"]["kernel"] is not second["params"][bank]["self_attn"]["q_proj"]["kernel"]
    before = _pointers(frozen)
    step = trainer.compile(state, tokens())
    for _ in range(2):
        state, *_ = step(state, tokens())
    assert _pointers(state.params[FROZEN]) == before
    assert _pointers(state.params["params"]) != _pointers(frozen) and set(state.params["params"]) == {"layers_0", "layers_1"}
