"""Host-owned full-tree transactions: coupling, replay, restart and placement.

Streamed-bank ownership and placement are Dew execution policies; Optax oracles the updates.
"""
import dataclasses
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.core import freeze as frozen_dict
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P, SingleDeviceSharding
from steady_state import steady_state
from test_training_transactions import SHORT_SCALE, Terms, Tiny, batches

from dew.batch import FROZEN, freeze
from dew.checkpoints import Checkpoints
from dew.data import Loading, Memory
from dew.data.dataset import DataPartition
from dew.data.pipeline import DeviceStep
from dew.distributed.host import companion_mesh, transfer
from dew.inference.banks import spread
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import group_layers
from dew.nn.inputs import ModelInputs
from dew.objectives.base import Aux, EMASpec, Objective, Ratio
from dew.objectives.lm import LMObjective
from dew.objectives.routing import ExpertBalance
from dew.training import Layout, Trainer

HOST = Layout(host=("variables",), min_shard=1, tolerance=1.)
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


FUSED_PER_STEP = 8
"""Multiply-adds a step's optimizer may fuse per element on a GPU or a TPU,
where the CPU rounds the product and the sum apart: Adam's two moments,
AdamW's decay, the parameter update and the EMA, one each, and up to three
units in the last place the global clip's scale moves by, the sum of five
squares. Each fusion moves a value by at most one unit in its last place."""


def platforms_agree(resident, host, steps: int):
    """The resident layout's state and the host layout's after `steps`
    optimizer steps, the host's run on the CPU: bitwise when the accelerator
    is the CPU too. Elsewhere the resident optimizer runs where XLA fuses
    multiply-adds, so a floating leaf is held to FUSED_PER_STEP ulps a step
    of its own magnitude, and every other leaf (counts, steps, keys) is exact.
    A leaf lost or reset at the boundary is wrong in its leading digits."""
    if jax.default_backend() == "cpu":
        equal(resident, host)
        return
    for path, a, b in _leaves(resident, host):
        if not np.issubdtype(a.dtype, np.floating):
            np.testing.assert_array_equal(a, b, err_msg=path)
            continue
        bound = FUSED_PER_STEP * steps * float(np.finfo(a.dtype).eps)
        np.testing.assert_allclose(b, a, rtol=bound, atol=bound * float(np.max(np.abs(a), initial=0.0)),
                                   err_msg=path)


def close(left, right, bound=STREAMED_BOUND):
    """`left`, a CPU-owned state, agrees with `right`, a device one, leaf by
    leaf; the frozen collection is compared per layer, the host's banks
    (`dew.inference.banks.banked`) read row by row."""
    if isinstance(left, dict) and FROZEN in left:
        left = {**left, FROZEN: unbanked(left[FROZEN])}
    elif FROZEN in getattr(left, "variables", {}):
        left = dataclasses.replace(left, variables={
            **left.variables, FROZEN: unbanked(left.variables[FROZEN])})
    for path, a, b in _leaves(left, right):
        np.testing.assert_allclose(a, b, atol=bound, rtol=0, err_msg=path)


def unbanked(tree):
    """A frozen collection with every run's bank spread back over its layers, as host arrays."""
    return spread(tree, lambda leaf, count: list(np.asarray(leaf)))


class Coupled(Objective):
    ema = EMASpec(optax.constant_schedule(.5))

    def fresh_variables(self, key, held):
        return {"params": {"first": jnp.array([1., 2.]), "second": jnp.array([3., 4., 5.])}}

    def loss(self, variables, batch, step):
        p = variables["params"]
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
    platforms_agree(*states, steps=3)


@pytest.mark.parametrize("composite", [False, True])
def test_host_accumulation_replays_original_rng_and_mutable_snapshots(tmp_path, composite):
    data = batches()
    states = []
    for layout in (DEVICE, HOST):
        trainer = Trainer(
            Tiny(composite), optax.adamw(.03, weight_decay=.1), key=jax.random.PRNGKey(9),
            accumulation=2, dynamic_scale=SHORT_SCALE, layout=layout)
        state, _, _ = trainer.place()
        step = trainer.compile(state, data[0])
        prefix, *_ = step(state, data[0])
        # the step consumes the state
        prefix_params = jax.tree.map(np.asarray, prefix.variables)
        rejected, _, _, _, accepted, _ = step(prefix, data[1])
        assert not bool(accepted)
        # The rejection drops its window (`dropped_window`); the next two microbatches make the update.
        equal(prefix_params, rejected.variables)
        assert int(rejected.microstep) == 0
        state, *_ = step(rejected, data[2])
        state, *_ = step(state, data[3])
        assert int(state.updates) == 1
        assert float(state.variables["stats"]["seen"]) == 3
        states.append(state)
    platforms_agree(*states, steps=1)

    checkpoints = Checkpoints(str(tmp_path / "run"))
    trainer = Trainer(
        Tiny(composite), optax.adamw(.03, weight_decay=.1), key=jax.random.PRNGKey(9),
        accumulation=2, dynamic_scale=SHORT_SCALE, layout=HOST, checkpoints=checkpoints)
    start, _, _ = trainer.place()
    step = trainer.compile(start, data[0])
    prefix, *_ = step(start, data[0])
    checkpoints.save(1, prefix, b"position", share=DataPartition())
    checkpoints.wait()
    restored, _, position = trainer.place()
    assert position == b"position"
    equal(prefix, restored)
    resumed_step = trainer.compile(restored, data[1])
    resumed, *_ = resumed_step(restored, data[1])
    resumed, *_ = resumed_step(resumed, data[2])
    resumed, *_ = resumed_step(resumed, data[3])
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
        assert "layers_0" in state.variables["params"] and "layers_1" in state.variables["params"]
        assert "layers_0_1" not in state.variables["params"]
        states.append(state)
    close(*states)


def test_dropout_composite_replay_restarts_with_effects_and_ema_intact(tmp_path):
    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=2, num_heads=2,
                              mlp_features=16, max_seq_len=8, scan_layers=True, dropout_rate=.4,
                              mixture=Mixture(experts=4, top_k=2, bias=True))
    objective = LMObjective(model, 8, head_chunks=1, ema_decay=.9,
                            balance=ExpertBalance(balance_rate=.01, aux_loss_alpha=.2, seq_aux=False))
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
        trainer = Trainer(Tiny(), optimizer, key=jax.random.PRNGKey(9),
                                    dynamic_scale=SHORT_SCALE, layout=layout)
        initial, _, _ = trainer.place()
        data = batches()[0]
        # the step consumes the state
        assert initial.scale is not None
        start_params, start_ema, start_opt = jax.tree.map(
            np.asarray, (initial.variables, initial.ema, initial.opt_state))
        start_scale, start_fin = float(initial.scale.scale), int(initial.scale.fin_steps)
        final, _, _, finite, accepted, _ = trainer.compile(initial, data)(initial, data)
        assert bool(finite) and not bool(accepted)
        equal(start_params, final.variables)
        equal(start_ema, final.ema)
        equal(start_opt, final.opt_state)
        assert int(final.microstep) == int(final.updates) == 0
        assert int(final.step) == 1
        # Preserve the existing distinction: candidate-state rejection rolls
        # back the transaction, while the scaler tracks gradient finiteness.
        assert final.scale is not None
        assert float(final.scale.scale) == start_scale
        assert int(final.scale.fin_steps) == start_fin + 1
        results.append(final)
    equal(*results)


def test_composite_replay_does_not_apply_effects_twice():
    class Effects(Objective[Ratio | Terms, jax.Array]):
        ema = Tiny.ema

        def __init__(self):
            self.reference = Tiny(composite=True)

        def init(self, key, variables=None):
            return self.reference.init(key, variables)

        def loss(self, variables, batch, step):
            stats, aux = self.reference.loss(variables, batch, step)
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
        assert float(state.variables["stats"]["seen"]) == 4.
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
def test_a_companion_takes_the_partners_it_needs_from_more_cpus():
    """A mesh of two accelerator devices pairs with the first two of six CPUs, and moves its values
    there whole; three CPUs for a mesh of four are refused before anything is placed."""
    devices = jax.devices("cpu")
    assert len(devices) >= 8
    accelerator = Mesh(np.asarray(devices[:2], dtype=object), ("fsdp",), axis_types=(AxisType.Auto,))
    cpu = companion_mesh(accelerator, devices[2:])
    assert list(cpu.devices.flat) == list(devices[2:4])
    values = jax.device_put(np.arange(16, dtype=np.float32), NamedSharding(accelerator, P("fsdp")))
    np.testing.assert_array_equal(np.asarray(transfer(values, NamedSharding(cpu, P("fsdp")))),
                                  np.asarray(values))
    four = Mesh(np.asarray(devices[:4], dtype=object), ("fsdp",), axis_types=(AxisType.Auto,))
    with pytest.raises(ValueError):
        companion_mesh(four, devices[4:7])


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
    fields = {"vocab_size": 16, "emb_features": 8, "num_layers": 2, "num_heads": 2,
                  "mlp_features": 16, "max_seq_len": 8, "scan_layers": True}
    return CausalTransformer(**{**fields, **overrides})


def frozen_leaves(state):
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf
            in jax.tree_util.tree_leaves_with_path(state.variables[FROZEN])}


def _moments(opt_state) -> optax.ScaleByAdamState:
    """Adam's moments out of the optimizer state, narrowed for the checker."""
    held, _ = jax.tree_util.tree_flatten(
        opt_state, is_leaf=lambda node: isinstance(node, optax.ScaleByAdamState))
    for node in held:
        if isinstance(node, optax.ScaleByAdamState):
            return node
    raise AssertionError("the optimizer state carries no Adam moments")


def test_a_warm_host_run_traces_nothing():
    """Once a CPU-owned run has stepped through an accumulation window, the
    next windows run the programs it compiled: an eager vjp handed the
    pullback a new static function every realization, which traced again."""
    objective = LMObjective(decoder(), 8, head_chunks=1)
    trainer = Trainer(objective, optax.adam(.01), key=jax.random.key(5), layout=HOST, accumulation=2)
    state, _, _ = trainer.place()
    batch = tokens()
    step = trainer.compile(state, batch)
    for _ in range(2):
        state, *_ = step(state, batch)
    with steady_state(allow=("host_to_device", "device_to_host")):
        for _ in range(4):
            state, *_ = step(state, batch)
    assert int(state.updates) == 3


RESUMED = """
import json, sys
import jax, jax.numpy as jnp, numpy as np, optax
events = []
jax.monitoring.register_event_listener(lambda event, **_: events.append(event))
from dew.training.runtime import prepare_process
prepare_process(compilation_cache_dir=sys.argv[1])
from dew.checkpoints import Checkpoints
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.training import Layout, Trainer
directory, where, steps = sys.argv[2], sys.argv[3], int(sys.argv[4])
model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=2, num_heads=2, mlp_features=16,
                          max_seq_len=8, scan_layers=True)
trainer = Trainer(LMObjective(model, 8, head_chunks=1), optax.adam(.01), key=jax.random.key(5),
                  layout=Layout(host=("variables",) if where == "host" else (), min_shard=1, tolerance=1.),
                  checkpoints=None if directory == "-" else Checkpoints(directory), tracker=None)
state, _, _ = trainer.place()
started = int(state.step)
batch = {"text": jnp.tile(jnp.arange(1, 10, dtype=jnp.int32)[None], (jax.device_count(), 1))}
before = len(events)
run = trainer.compile(state, batch)
for _ in range(steps):
    state, *_ = run(state, batch)
jax.block_until_ready(state)
window = events[before:]
if directory != "-":
    trainer.checkpoints.save(int(state.step), state)
    trainer.checkpoints.wait()
print(json.dumps({"started": started, "step": int(state.step),
                  "leaves": [np.asarray(leaf, np.float64).tolist() for leaf in jax.tree.leaves(
                      (state.variables, state.opt_state))],
                  "cache": [window.count("/jax/compilation_cache/compile_requests_use_cache"),
                            window.count("/jax/compilation_cache/cache_hits")]}))
"""


@pytest.mark.parametrize("where", ["device", "host"])
def test_a_resumed_run_continues_on_the_programs_its_first_process_compiled(tmp_path, where):
    """A run resumed by a new process continues where the first stopped, to
    the parameters and optimizer state of a run that never stopped, on the
    step programs the first process compiled and the persistent cache holds:
    the restored state's leaves enter the step as the fresh state's did. The
    first process may itself hit programs its placement compiled."""
    import subprocess
    import sys

    from process_support import worker_env

    def run(directory, steps):
        command = [sys.executable, "-c", RESUMED, str(tmp_path / "cache"), str(directory), where, str(steps)]
        done = subprocess.run(command, capture_output=True, text=True, check=False, env=worker_env(1))
        assert done.returncode == 0, done.stderr[-3000:]
        return json.loads(done.stdout.strip().splitlines()[-1])

    first, resumed, whole = run(tmp_path / "run", 2), run(tmp_path / "run", 2), run("-", 4)
    assert (first["started"], resumed["started"], resumed["step"]) == (0, 2, 4)
    lookups, hits = resumed["cache"]
    assert lookups == first["cache"][0] > 0 and hits == lookups, (first["cache"], resumed["cache"])
    for got, want in zip(resumed["leaves"], whole["leaves"], strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_streamed_banks_train_a_mixed_frozen_root_decoder_like_the_resident_stack(tmp_path):
    """Per-layer and per-leaf freezing through one banked decoder.

    Every layer's parameters ride in one bank whatever their ownership, and
    only the canonical moving leaves come back as gradients: the optimizer
    holds those and nothing else, the frozen collection is untouched, and the
    stored names stay `layers_N`.
    """
    def trainable(path):
        return "layers_1" not in path and path[-2:] != ("gate_proj", "kernel")

    model = decoder()
    start = freeze(LMObjective(model, 8).init(jax.random.key(0)), trainable)

    def objective():
        # A host placement leaves its banks in the held tree's rows, so each
        # trainer starts from its own copy.
        return LMObjective(model, 8, head_chunks=1, variables=jax.tree.map(np.asarray, start),
                           ema_decay=0.999)

    batch = tokens()
    resident = updated(objective(), batch, DEVICE)
    host = updated(objective(), batch, HOST)
    close(host, resident)
    initial = frozen_leaves(Trainer(objective(), optax.adam(.01), key=jax.random.key(5),
                                    layout=HOST).place()[0])
    assert initial and frozen_leaves(host).keys() == initial.keys()
    for name, value in frozen_leaves(host).items():
        np.testing.assert_array_equal(value, initial[name], err_msg=name)
    assert sorted(host.variables["params"]) == ["embed_tokens", "layers_0", "norm"]
    # The leaf both rows froze is the run's bank; the leaves only layer 1
    # froze stay its own.
    assert sorted(host.variables[FROZEN]) == ["layers_0_1", "layers_1"]
    assert host.variables[FROZEN]["layers_0_1"]["mlp"]["gate_proj"]["kernel"].shape[0] == 2
    assert jax.tree.structure(_moments(host.opt_state).mu) == jax.tree.structure(host.variables["params"])

    # The EMA tracks the moving collection only, and banked execution reads
    # that partial tree without extending it.
    assert host.ema is not None and FROZEN not in host.ema
    assert jax.tree.structure(host.ema["params"]) == jax.tree.structure(host.variables["params"])
    checkpoints = Checkpoints(str(tmp_path / "mixed"))
    prefix = updated(objective(), batch, HOST, checkpoints=checkpoints)
    checkpoints.save(int(prefix.step), prefix, None)
    checkpoints.wait()
    restored, _, _ = Trainer(objective(), optax.adam(.01), key=jax.random.key(5),
                             layout=HOST, checkpoints=Checkpoints(str(tmp_path / "mixed"))).place()
    equal(prefix, restored)
    resumed = updated(objective(), batch, HOST, checkpoints=Checkpoints(str(tmp_path / "mixed")))
    equal(resumed, updated(objective(), batch, HOST, steps=2))


def test_a_cpu_owned_lora_run_loads_through_the_pipeline_in_a_fresh_process(tmp_path):
    """A LoRA run whose frozen base the CPU owns stores it banked; a fresh
    process loads it through `dew.pipeline` with its layers spread back, its
    logits those of the trained state, and generates from it."""
    import json
    import subprocess
    import sys

    from dew.lora import LoRA

    base = decoder()
    adapter = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(
        base, base.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32)), key=1)
    objective = LMObjective(adapter.model, 8, head_chunks=1, ema_decay=None,
                            variables=jax.tree.map(np.asarray, adapter.variables))
    checkpoints = Checkpoints(str(tmp_path / "run"))
    rows = [{"text": np.arange(1, 10, dtype=np.int32)} for _ in range(8)]
    trainer = Trainer(objective, optax.adam(.01), key=jax.random.key(5), layout=HOST, checkpoints=checkpoints)
    data = Memory(rows).shuffle().repeat().batch(jax.device_count()).read("memory").loading(
        Loading(workers=0))
    state = trainer.fit(data, steps=1, checkpoint_every=1, log_every=100)
    checkpoints.wait()
    assert any(group_layers(name) and len(group_layers(name)) > 1 for name in state.variables[FROZEN])
    prompt = [[1, 2, 3, 4]]
    script = (f"import json, dew, jax.numpy as jnp\n"
              f"task = dew.pipeline({str(tmp_path / 'run')!r}, ema=False)\n"
              f"logits = task.model.apply(task.variables, jnp.asarray({prompt}))\n"
              f"tokens = task(jnp.asarray({prompt}), max_new_tokens=2, key=0).tokens\n"
              f"print(json.dumps({{'logits': logits.tolist(), 'tokens': jnp.asarray(tokens).tolist()}}))")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False,
                          env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert done.returncode == 0, done.stderr[-3000:]
    loaded = json.loads(done.stdout.strip().splitlines()[-1])
    # Held to the fresh process's logits on its device, the CPU; the frozen base is the CPU's already.
    trained = jax.device_get({**state.variables, FROZEN: unbanked(state.variables[FROZEN])})
    with jax.default_device(jax.devices("cpu")[0]):
        expected = objective.model.apply(trained, jnp.asarray(prompt))
    np.testing.assert_allclose(np.asarray(loaded["logits"]), np.asarray(expected), atol=1e-5)
    assert np.asarray(loaded["tokens"]).shape[-1] >= 2

    # Each bank is cut into rows where the layout places its layers, in the
    # bank's memory, and released: the read leaves nothing behind but the rows.
    from dew.inference.banks import shape_rows
    from dew.training import MeshSpec

    def live_bytes():
        return sum(array.nbytes for array in jax.live_arrays())

    stored = Checkpoints(str(tmp_path / "run")).stored()["variables"][FROZEN]
    before = live_bytes()
    variables = Checkpoints(str(tmp_path / "run")).variables(ema=False)
    assert live_bytes() - before <= sum(leaf.nbytes for leaf in jax.tree.leaves(variables))
    placed = Layout().shardings(MeshSpec().build(), spread(stored, shape_rows))
    for path, leaf in jax.tree_util.tree_leaves_with_path(variables[FROZEN]):
        wanted = dict(jax.tree_util.tree_leaves_with_path(placed))[path]
        assert leaf.sharding.is_equivalent_to(wanted, leaf.ndim), jax.tree_util.keystr(path)
        assert leaf.sharding.memory_kind == wanted.memory_kind, jax.tree_util.keystr(path)


def multimodal(**overrides):
    """A media-conditioned wrapper over one declared decoder."""
    from dew.nn.multimodal import MultimodalTransformer
    from dew.nn.vision import GemmaProjector, SiglipVision

    return MultimodalTransformer(
        decoder(emb_features=16, max_seq_len=16, **overrides), SiglipVision(
            hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2,
            image_size=8, patch_size=4),
        GemmaProjector(text_width=16, patches_per_side=2, tokens_per_side=1),
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
        return LMObjective(model, 8, head_chunks=1, variables=freeze(held, trainable))

    trainer = Trainer(objective(), optax.adam(.01), key=jax.random.key(5), layout=DEVICE)
    start, _, _ = trainer.place()
    step = trainer.compile(start, batch)
    _, loss, *_ = step(start, batch)
    # Both losses must come from the same weights, and the step consumes the
    # state, so place them again instead of reusing the consumed ones.
    start, _, _ = trainer.place()
    _, changed, *_ = step(start, other)
    assert abs(float(loss) - float(changed)) > 1e-4, "the pixels do not reach the loss"

    resident = updated(objective(), batch, DEVICE)
    host = updated(objective(), batch, HOST)
    close(host, resident)
    assert sorted(host.variables["params"]) == ["language_model"]
    assert sorted(host.variables[FROZEN]) == ["language_model", "projector", "tower"]
    assert sorted(host.variables[FROZEN]["language_model"]) == ["layers_1"]
    media = {name: value for name, value in frozen_leaves(host).items()
             if "tower" in name or "projector" in name}
    initial = frozen_leaves(Trainer(objective(), optax.adam(.01), key=jax.random.key(5),
                                    layout=HOST).place()[0])
    assert media
    for name, value in media.items():
        np.testing.assert_array_equal(value, initial[name], err_msg=name)


@pytest.mark.parametrize("detached", [False, True])
def test_a_shared_text_owner_sums_both_block_losses_through_one_bank(detached):
    """DiffusionGemma reads one physical stack twice: encoder and denoiser.

    The site is declared once, so the bank is packed once and native reverse
    mode sums both uses into it. The detached variant keeps the existing
    encoder-gradient policy.
    """
    from pathlib import Path

    from dew.interop import Pretrained
    from dew.nn.diffusion_gemma import DiffusionGemma
    from dew.objectives.diffusion.block import BlockDiffusionObjective

    fixture = Path(__file__).resolve().parent / "fixtures/hf/diffusion-gemma-sft"
    loaded = Pretrained.load(fixture, dtype="float32", attention_impl="xla", max_seq_len=32)
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
            scanned, prompt_length=4, num_canvases=2, variables=loaded.variables,
            stop_gradient_from_denoiser_to_encoder=detached)

    (site,) = objective().bank_sites
    assert site.namespace == ("text",)
    resident = updated(objective(), batch, DEVICE, optimizer=optax.sgd(.001))
    host = updated(objective(), batch, HOST, optimizer=optax.sgd(.001))
    close(host, resident, 1e-5)
    assert "layers_0" in host.variables["params"]["text"]
    assert not any(name.startswith("layers_0_") for name in host.variables["params"]["text"])



# --------------------------------------------------------------------------
# Frozen residency: placed once, aliased by every snapshot, streamed into place
# --------------------------------------------------------------------------


def _named_leaves(tree):
    return [(jax.tree_util.keystr(path), leaf) for path, leaf in jax.tree_util.tree_leaves_with_path(tree)]


def test_place_streams_the_held_tree_and_releases_each_source():
    """The objective's held leaves become the placed arrays as they land, so
    the state and the objective share one array per leaf and no numpy copy
    stays beside it."""
    model = decoder()
    held = freeze(jax.tree.map(np.asarray, model.init(jax.random.key(1), jnp.zeros((1, 8), jnp.int32))),
                  lambda path: path[-2:] == ("q_proj", "kernel"))
    objective = LMObjective(model, 8, head_chunks=1, variables=held)
    trainer = Trainer(objective, optax.adam(.01), key=jax.random.key(5), layout=HOST)
    state, _, _ = trainer.place()
    for path, leaf in _named_leaves(held):
        assert isinstance(leaf, jax.Array), path
    # A leaf every row froze lives once, as the run's bank; the held tree
    # holds that bank where each row was, so no row copy stays beside it.
    bank = state.variables[FROZEN]["layers_0_1"]["mlp"]["gate_proj"]["kernel"]
    assert held[FROZEN]["layers_1"]["mlp"]["gate_proj"]["kernel"] is bank
    assert held[FROZEN]["layers_0"]["mlp"]["gate_proj"]["kernel"] is bank
    assert (
        state.variables["params"]["layers_0"]["self_attn"]["q_proj"]["kernel"]
        is held["params"]["layers_0"]["self_attn"]["q_proj"]["kernel"]
    )
    fresh = jax.tree.map(np.asarray, model.init(jax.random.key(1), jnp.zeros((1, 8), jnp.int32)))

    def queries(path):
        return path[-2:] == ("q_proj", "kernel")

    resident = updated(
        LMObjective(model, 8, head_chunks=1, variables=freeze(fresh, queries)),
        tokens(),
        DEVICE,
    )
    close(
        updated(
            LMObjective(model, 8, head_chunks=1, variables=freeze(jax.tree.map(np.asarray, fresh), queries)),
            tokens(),
            HOST,
        ).variables,
        resident.variables,
    )


def test_a_tree_of_frozen_dicts_is_placed_as_a_tree_of_dicts_is():
    """An objective given its tree as Flax `FrozenDict`s, as a loaded task holds one, holds it as
    dicts: the placed arrays stream into it as they land and its frozen rows become the run's bank,
    as a dict tree's do."""
    model = decoder()
    given = frozen_dict(freeze(
        jax.tree.map(np.asarray, model.init(jax.random.key(1), jnp.zeros((1, 8), jnp.int32))),
        lambda path: path[-2:] == ("q_proj", "kernel")))
    objective = LMObjective(model, 8, head_chunks=1, variables=given)
    state, _, _ = Trainer(objective, optax.adam(.01), key=jax.random.key(5), layout=HOST).place()
    held = objective.held_variables()
    assert held is not None
    for path, leaf in _named_leaves(held):
        assert isinstance(leaf, jax.Array), path
    bank = state.variables[FROZEN]["layers_0_1"]["mlp"]["gate_proj"]["kernel"]
    assert held[FROZEN]["layers_0"]["mlp"]["gate_proj"]["kernel"] is bank
    assert held[FROZEN]["layers_1"]["mlp"]["gate_proj"]["kernel"] is bank


class _Scaled(DeviceStep):
    """A device step whose setting its repr leaves out, as a plain class's repr does."""

    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, batch):
        return {**batch, "x": batch["x"] * self.scale}


def _scaled_fit(objective, step: DeviceStep):
    """Two steps of a host-held run of `objective` over batches that `step` runs on the device."""
    from affine_run import batches as affine_batches

    from dew.data.dataset import Loader

    data = Loader(lambda partition: affine_batches(), None, None, 8, device=(step,))
    return Trainer(objective, optax.sgd(0.1), key=jax.random.key(0), layout=HOST).fit(
        data, steps=2, log_every=1)


def test_runs_of_one_objective_with_unlike_device_steps_each_train_on_their_own():
    """A host-held run compiles its forward with the dataset's device steps inside, under the
    objective's program cache: two runs of one objective whose steps differ only in a setting
    their reprs leave out each train on their own steps, as a fresh objective does."""
    from affine_run import Regression

    shared = Regression()
    _scaled_fit(shared, _Scaled(1.0))
    reused, fresh = _scaled_fit(shared, _Scaled(3.0)), _scaled_fit(Regression(), _Scaled(3.0))
    for left, right in zip(jax.tree.leaves(reused.variables), jax.tree.leaves(fresh.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_a_later_run_of_one_objective_releases_the_device_steps_an_earlier_run_compiled():
    """The objective keeps the host forward of its latest run alone: a run's device steps, which
    its forward runs, are released once a later run of the objective compiles its own, so runs
    repeated in one process hold one forward, not one per run."""
    import gc
    import weakref

    from affine_run import Regression

    shared, first = Regression(), _Scaled(1.0)
    _scaled_fit(shared, first)
    released = weakref.ref(first)
    del first
    _scaled_fit(shared, _Scaled(1.0))
    gc.collect()
    assert released() is None


def test_a_tasks_held_weights_leave_the_callers_private_mapping_as_it_changed_it(tmp_path):
    """A task's weights are placed into a tree of their own (`held`), so the caller's host arrays stay
    as they were: a copy-on-write mapping's changed pages are not given back to the file they came from,
    which would turn them back into the file's values."""
    from dew.inference.tasks import held

    path = tmp_path / "kernel.bin"
    np.arange(4096, dtype=np.float32).tofile(path)
    kernel = np.memmap(path, np.float32, mode="c", shape=(4096,))
    kernel[:] = -1.0
    placed = held({"params": {"kernel": kernel}})
    np.testing.assert_array_equal(kernel, np.full(4096, -1.0, np.float32))
    np.testing.assert_array_equal(np.asarray(placed["params"]["kernel"]), np.full(4096, -1.0, np.float32))


def test_a_streamed_node_takes_the_one_target_its_placement_gives_it_whole():
    """A placement may give a node of several arrays one sharding, as `jax.device_put` takes a prefix:
    `stream` places each of the node's arrays there, read-only mappings of targets included, and the
    tree the sources came from (`held`) is handed the placed node, so the two share one copy."""
    import types

    from dew.distributed.host import stream

    sharding = NamedSharding(Mesh(np.asarray(jax.devices()[:1]), ("data",)), P())
    pair = (np.arange(4, dtype=np.float32), np.ones((2, 3), np.float32))
    held = {"params": {"kernel": pair}}
    placed = stream({"params": {"kernel": pair}},
                    types.MappingProxyType({"params": types.MappingProxyType({"kernel": sharding})}), held)
    for got, want in zip(placed["params"]["kernel"], pair, strict=True):
        assert isinstance(got, jax.Array) and got.sharding == sharding
        np.testing.assert_array_equal(got, want)
    assert held["params"]["kernel"] is placed["params"]["kernel"]


@pytest.mark.mesh(devices=2)
def test_a_node_two_places_share_goes_to_each_places_own_target():
    """A node held at two places of the tree it is streamed from, the tree also the one its sources
    came from, lands at each place on that place's own target; and a held tree beside a tree whose
    mapping is rebuilt (a defaultdict) is handed the placed node all the same."""
    import collections

    from dew.distributed.host import stream

    first, second = (SingleDeviceSharding(device) for device in jax.devices()[:2])
    pair = (np.arange(4, dtype=np.float32), np.ones((2, 3), np.float32))
    state = {"params": {"first": pair, "second": pair}}
    stream(state, {"params": {"first": first, "second": second}}, state)
    for name, sharding in (("first", first), ("second", second)):
        assert all(leaf.sharding == sharding for leaf in state["params"][name]), name

    held = {"params": {"kernel": pair}}
    placed = stream({"params": collections.defaultdict(dict, {"kernel": pair})},
                    {"params": {"kernel": first}}, held)
    assert held["params"]["kernel"] is placed["params"]["kernel"]
