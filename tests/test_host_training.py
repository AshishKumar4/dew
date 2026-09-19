"""Host-owned full-tree transactions: coupling, replay, restart and placement."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from dew.checkpoints import Checkpoints
from dew.nn.backbones.causal_transformer import CausalTransformer, Mixture
from dew.objectives.base import Aux, EMASpec, Mean, Objective
from dew.objectives.lm import LMObjective
from dew.training import Layout, Trainer
from dew.training.host import companion_mesh, transfer
from test_training_transactions import Terms, Tiny, ShortScaleTrainer, batches

HOST = Layout(host=("params",), min_shard=1, tolerance=1.)
DEVICE = Layout(min_shard=1, tolerance=1.)


def equal(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(left), jax.tree.leaves(right), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=jax.tree_util.keystr(path))


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
    equal(*states)


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
