"""Mixture of experts in the decoder: the layers that grow experts, the
expert mesh axis, and training on the simulated eight-device mesh, with the
balancing bias carried through Aux.variables.

The router and expert-block parity against transformers, the balancing
update and the grouped matmul are in tests/test_moe.py.
"""

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import PartitionSpec as P
from recording import RecordingTracker
from sharded import assert_sharded

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import GatedMLP, Mixture
from dew.nn.moe import SparseMLP
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer
from dew.training.distributed import shard_batch

VOCAB = 64
SEQ_LEN = 16
BATCH = 8
# The training models here hold thousands of parameters, orders below the
# production shard threshold, so lower it or "sharded" would mean "replicated".
TINY_SHARD = 256


# --------------------------------------------------------------------------
# The decoder that grows experts
# --------------------------------------------------------------------------

def decoder_fields(**overrides) -> dict:
    settings = {"vocab_size": VOCAB, "emb_features": 32, "num_layers": 4, "num_heads": 2,
                    "num_kv_heads": 1, "mlp_features": 64, "max_seq_len": SEQ_LEN}
    settings.update(overrides)
    return settings


def decoder(**overrides) -> CausalTransformer:
    return CausalTransformer(**decoder_fields(**overrides))


def leaf_names(variables):
    leaves, _ = jax.tree_util.tree_flatten_with_path(variables)
    return sorted('/'.join(str(entry.key) for entry in path) for path, _ in leaves)


@pytest.mark.parametrize("settings,expected", [
    ({"mixture": {"experts": 4}}, (0, 1, 2, 3)),
    ({"mixture": {"experts": 4, "layers": (0, 2)}}, (0, 2)),
    ({}, ()),
])
def test_the_sparse_layers_are_the_ones_the_mixture_names(settings, expected):
    assert models.build("causal_transformer", **decoder_fields(**settings)).sparse_layers == expected


def test_a_sparse_layer_replaces_only_its_own_feed_forward():
    """The frozen leaf names: a model with experts on one layer keeps every
    other leaf of the dense model, including the dense layers' mlp."""
    dense = decoder()
    sparse = decoder(mixture=Mixture(experts=4, top_k=2, layers=(1,)))
    tokens = jnp.zeros((2, SEQ_LEN), jnp.int32)
    dense_names = leaf_names(dense.init(jax.random.key(0), tokens))
    sparse_names = leaf_names(sparse.init(jax.random.key(0), tokens))

    assert [name for name in dense_names if 'layers_1/mlp' not in name] == \
           [name for name in sparse_names if 'layers_1/mlp' not in name]
    assert [name for name in sparse_names if 'layers_1/mlp' in name] == [
        'params/layers_1/mlp/experts/down_proj/kernel',
        'params/layers_1/mlp/experts/gate_proj/kernel',
        'params/layers_1/mlp/experts/up_proj/kernel',
        'params/layers_1/mlp/gate/kernel',
    ]


def test_the_expert_leaves_hold_every_expert_of_the_reference_layout():
    """One leaf per projection, stacked over the experts, the layout a
    checkpoint's `mlp.experts.N.gate_proj.weight` tensors translate into."""
    model = decoder(emb_features=32, mixture=Mixture(experts=8, top_k=2, layers=(0,)))
    variables = model.init(jax.random.key(0), jnp.zeros((2, SEQ_LEN), jnp.int32))
    experts = variables["params"]["layers_0"]["mlp"]["experts"]

    assert experts["gate_proj"]["kernel"].shape == (8, 32, 64)
    assert experts["up_proj"]["kernel"].shape == (8, 32, 64)
    assert experts["down_proj"]["kernel"].shape == (8, 64, 32)
    assert variables["params"]["layers_0"]["mlp"]["gate"]["kernel"].shape == (32, 8)
    dense = variables["params"]["layers_1"]["mlp"]
    assert dense["gate_proj"]["kernel"].shape == (32, 64)


def test_the_mixture_sizes_the_experts_and_the_shared_branch_apart():
    """DeepSeek's layout: routed experts at moe_intermediate_size, one dense
    shared MLP at n_shared_experts times that, and the dense layers at the
    model's own width, each a leaf where the checkpoint's tensor lands."""
    model = decoder(emb_features=32, mixture=Mixture(
        experts=8, top_k=2, layers=(0,), expert_features=24, shared_features=48))
    variables = model.init(jax.random.key(0), jnp.zeros((2, SEQ_LEN), jnp.int32))
    mlp = variables["params"]["layers_0"]["mlp"]

    assert mlp["experts"]["gate_proj"]["kernel"].shape == (8, 32, 24)
    assert mlp["experts"]["down_proj"]["kernel"].shape == (8, 24, 32)
    assert mlp["shared_experts"]["gate_proj"]["kernel"].shape == (32, 48)
    assert mlp["shared_experts"]["up_proj"]["kernel"].shape == (32, 48)
    assert mlp["shared_experts"]["down_proj"]["kernel"].shape == (48, 32)
    assert variables["params"]["layers_1"]["mlp"]["gate_proj"]["kernel"].shape == (32, 64)


@pytest.mark.parametrize("mixture, message", [
    ({"experts": 8, "expert_features": 0}, "expert_features"),
    ({"experts": 8, "shared_features": -1}, "shared_features"),
])
def test_a_misconfigured_width_is_rejected(mixture, message):
    with pytest.raises(ValueError, match=message):
        Mixture(**mixture)


def test_an_expert_layer_and_a_dense_layer_agree_at_one_expert():
    """A single expert taking every token is the dense feed-forward, which is
    what makes the router's weight the only difference between them."""
    tokens = jax.random.normal(jax.random.key(0), (2, 3, 8))
    sparse = SparseMLP(num_experts=1, top_k=1, hidden_features=16, out_features=8)
    variables = sparse.init(jax.random.key(1), tokens)
    kernels = variables["params"]["experts"]

    gated = GatedMLP(hidden_features=16, out_features=8)
    dense_variables = {"params": {
        name: {"kernel": kernels[name]["kernel"][0]}
        for name in ("gate_proj", "up_proj", "down_proj")}}

    assert np.max(np.abs(np.asarray(sparse.apply(variables, tokens))
                         - np.asarray(gated.apply(dense_variables, tokens)))) < 1e-6


def test_the_router_runs_in_fp32_under_a_bfloat16_model():
    """Routing decides which experts a token trains, so a bf16 run must not
    decide it on bf16 scores. The experts stay in the compute dtype."""
    tokens = jnp.asarray(jax.random.normal(jax.random.key(0), (2, 3, 8)), jnp.bfloat16)
    sparse = SparseMLP(num_experts=4, top_k=2, hidden_features=16, out_features=8,
                       dtype=jnp.bfloat16)
    variables = sparse.init(jax.random.key(1), tokens)
    weights, _ = sparse.apply(variables, tokens, method=lambda module, x: module.gate(x))

    assert variables["params"]["gate"]["kernel"].dtype == jnp.float32
    assert weights.dtype == jnp.float32
    assert sparse.apply(variables, tokens).dtype == jnp.bfloat16

    # The same tokens at fp32 choose the same experts and weight them the
    # same; that is the property the fp32 gate is for.
    exact = SparseMLP(num_experts=4, top_k=2, hidden_features=16, out_features=8)
    wide = exact.apply(variables, jnp.asarray(tokens, jnp.float32),
                       method=lambda module, x: module.gate(x))
    assert np.array_equal(np.asarray(weights), np.asarray(wide[0]))


@pytest.mark.parametrize("mixture,message", [
    ({"experts": 0}, "dense model has no mixture"),
    ({"experts": 4, "layers": (4,)}, "outside"),
    ({"experts": 4, "top_k": 5}, "top_k"),
    ({"experts": 4, "implementation": "megablox"}, "implementation"),
])
def test_a_misconfigured_mixture_is_rejected(mixture, message):
    """A mixture's own dials are checked where they are set, and the ones
    that read the model are checked when it builds; either way the message
    names the dial."""
    with pytest.raises(ValueError, match=message):
        models.build("causal_transformer", **decoder_fields(mixture=mixture)).init(
            jax.random.key(0), jnp.zeros((1, SEQ_LEN), jnp.int32))


def test_a_tokamax_mixture_computes_the_xla_mixtures_logits():
    """The mixture's implementation reaches every sparse layer's experts: the
    same weights through either kernel give the same logits."""
    pytest.importorskip("tokamax")
    tokens = jax.random.randint(jax.random.key(0), (2, SEQ_LEN), 0, VOCAB)
    reference = decoder(mixture=Mixture(experts=4, top_k=2, layers=(1, 3)))
    variables = reference.init(jax.random.key(1), tokens)
    expected = reference.apply(variables, tokens)
    logits = decoder(mixture=Mixture(
        experts=4, top_k=2, layers=(1, 3), implementation='tokamax')).apply(variables, tokens)

    # Observed 0 on CPU, where tokamax lowers to the same ragged_dot.
    assert np.max(np.abs(np.asarray(logits) - np.asarray(expected))) < 1e-5


def test_a_tokamax_mixture_runs_on_nothing_else(monkeypatch):
    """The experts import tokamax at their call, so a model that names it
    fails to initialise where the package cannot be imported instead of
    running the XLA kernel under the tokamax name."""
    # None in sys.modules is Python's own way to make an import fail.
    monkeypatch.setitem(sys.modules, "tokamax", None)
    model = decoder(mixture=Mixture(experts=4, top_k=2, layers=(1,), implementation='tokamax'))
    with pytest.raises(ModuleNotFoundError, match="tokamax"):
        model.init(jax.random.key(0), jnp.zeros((1, SEQ_LEN), jnp.int32))


# --------------------------------------------------------------------------
# The expert mesh axis
# --------------------------------------------------------------------------

def expert_specs(expert_size, fsdp_size, num_experts=8, min_shard_size=TINY_SHARD):
    model = SparseMLP(num_experts=num_experts, top_k=2, hidden_features=64,
                      out_features=32)
    variables = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, 4, 32)))
    mesh = MeshSpec(fsdp=fsdp_size, expert=expert_size).build()
    shardings = Layout(min_shard=min_shard_size).shardings(mesh, variables)
    return mesh, jax.tree.map(lambda sharding: sharding.spec, shardings)["params"]


@pytest.mark.mesh
@pytest.mark.parametrize("expert_size,fsdp_size", [(1, 8), (2, 4), (4, 2)])
def test_the_expert_dimension_takes_the_expert_axis(expert_size, fsdp_size):
    """Eight experts over one, two and four expert shards. The expert axis
    carries the expert dimension alone, so the widths keep fsdp."""
    _, specs = expert_specs(expert_size, fsdp_size)
    expert_axis = 'expert' if expert_size > 1 else None

    assert specs["experts"]["gate_proj"]["kernel"] == P(expert_axis, None, 'fsdp')
    assert specs["experts"]["up_proj"]["kernel"] == P(expert_axis, None, 'fsdp')
    assert specs["experts"]["down_proj"]["kernel"] == P(expert_axis, 'fsdp')
    # The router's expert dimension rides the axis too; its width keeps fsdp.
    assert specs["gate"]["kernel"] == (
        P('fsdp', 'expert') if expert_size > 1 else P('fsdp'))


@pytest.mark.mesh
def test_the_expert_axis_shards_experts_without_an_fsdp_axis():
    """Expert parallelism alone: with fsdp at one there is still a dimension
    to split, which the old two-axis rule replicated."""
    _, specs = expert_specs(expert_size=8, fsdp_size=1)

    assert specs["experts"]["gate_proj"]["kernel"] == P('expert')
    assert specs["experts"]["down_proj"]["kernel"] == P('expert')


@pytest.mark.mesh
def test_an_expert_count_the_axis_cannot_split_keeps_the_widths_sharded():
    """Six experts over four shards divides nothing, so the expert name is
    dropped and the parameter still shards on the dimension that can."""
    _, specs = expert_specs(expert_size=4, fsdp_size=2, num_experts=6)

    assert specs["experts"]["gate_proj"]["kernel"] == P(None, None, 'fsdp')


@pytest.mark.mesh
@pytest.mark.parametrize("expert_size,fsdp_size", [(1, 8), (2, 4), (4, 2)])
def test_every_expert_parallel_layout_stays_inside_the_sharding_tolerance(
        expert_size, fsdp_size):
    model = models.build("causal_transformer", **moe_config())
    variables = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    mesh = MeshSpec(fsdp=fsdp_size, expert=expert_size).build()
    layout = Layout(min_shard=TINY_SHARD)
    shardings = layout.shardings(mesh, variables)

    for (path, leaf), sharding in zip(
            jax.tree_util.tree_flatten_with_path(variables)[0],
            jax.tree.leaves(shardings), strict=True):
        for dimension, entry in enumerate(sharding.spec):
            if entry is None:
                continue
            axes = (entry,) if isinstance(entry, str) else entry
            size = int(np.prod([mesh.shape[axis] for axis in axes]))
            assert leaf.shape[dimension] % size == 0, (path, sharding.spec)
    layout.check(variables["params"], shardings["params"], mesh)


@pytest.mark.mesh
def test_a_mostly_dense_model_on_expert_only_parallelism_is_rejected():
    """Expert parallelism splits the experts alone, so a model whose experts
    are a fifth of it runs mostly replicated. The check has to see that,
    which the fsdp-only rule could not: it returned as soon as the fsdp axis
    was one.
    """
    model = models.build("causal_transformer", **moe_config())
    variables = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    mesh = MeshSpec(fsdp=1, expert=8).build()
    layout = Layout(min_shard=TINY_SHARD)
    shardings = layout.shardings(mesh, variables)

    with pytest.raises(ValueError, match="replicated"):
        layout.check(variables["params"], shardings["params"], mesh)


@pytest.mark.mesh
def test_mesh_build_rejects_an_expert_size_the_devices_cannot_hold():
    with pytest.raises(ValueError, match="expert 4"):
        MeshSpec(fsdp=4, expert=4).build()


@pytest.mark.mesh
def test_the_batch_is_split_over_the_expert_axis_too():
    """Expert parallelism must not cost data parallelism: every device holds a
    slice of the batch whichever axis it sits on."""
    mesh = MeshSpec(fsdp=2, expert=4).build()
    batch = shard_batch(mesh, np.zeros((jax.device_count(), 4), np.float32))

    assert len(batch.addressable_shards) == jax.device_count()
    assert batch.addressable_shards[0].data.shape == (1, 4)


# --------------------------------------------------------------------------
# Training on the simulated eight-device mesh
# --------------------------------------------------------------------------

def moe_config() -> dict:
    """Eight experts, so the expert dimension divides every mesh below."""
    return {"vocab_size": VOCAB, "emb_features": 32, "num_layers": 2,
            "num_heads": 2, "num_kv_heads": 1, "mlp_features": 64,
            "max_seq_len": SEQ_LEN,
            "mixture": {"experts": 8, "top_k": 2, "layers": (1,)}}


def token_batches():
    rng = np.random.default_rng(0)
    batch = {"text": rng.integers(0, VOCAB, size=(BATCH, SEQ_LEN + 1)).astype(np.int32)}
    while True:
        yield batch


class Data:
    def __init__(self, train):
        self._train, self.val, self.batch, self.records = train, None, BATCH, None

    def train(self, partition):
        return self._train()

    steps_per_epoch = None


def moe_trainer(expert_size, fsdp_size, tracker=None, bias=False):
    config = moe_config()
    model = models.build("causal_transformer",
                         **{**config, "mixture": {**config["mixture"], "bias": bias}})
    return Trainer(
        LMObjective(model, SEQ_LEN, balance_rate=0.01 if bias else None),
        optax.adam(1e-3), key=jax.random.key(0),
        mesh=MeshSpec(fsdp=fsdp_size, expert=expert_size),
        layout=Layout(min_shard=TINY_SHARD), tracker=tracker)


def run_losses(trainer, steps):
    """Per-step losses of a fit, as the tracker receives them."""
    trainer.tracker = tracker = RecordingTracker()
    state = trainer.fit(Data(token_batches), steps=steps, log_every=1)
    assert_sharded(state.variables["params"], trainer.device_mesh)
    return [scalars["train/loss"] for _, scalars in tracker.scalars if "train/loss" in scalars]


@pytest.mark.mesh
def test_the_expert_shards_train_the_same_model():
    """Fifty steps of the same sparse decoder at the same seed, with the
    experts on one shard and on four. Expert parallelism moves where the
    experts live, so the losses have to be the same run."""
    steps = 50
    one = run_losses(moe_trainer(1, 2), steps)
    four = run_losses(moe_trainer(4, 2), steps)

    assert len(one) == steps and np.all(np.isfinite(one))
    assert one[-1] < one[0] / 2, one
    difference = np.max(np.abs(np.array(one) - np.array(four)))
    # Observed equal on all 50 steps, 4.649611 down to 0.594418. The
    # tolerance is 1e-6, not zero, because a different collective order is
    # allowed to round differently.
    assert difference < 1e-6, difference


@pytest.mark.mesh
def test_the_experts_are_really_split_across_the_expert_axis():
    state = moe_trainer(expert_size=4, fsdp_size=2).fit(Data(token_batches), steps=0)
    experts = state.variables["params"]["layers_1"]["mlp"]["experts"]
    kernel = experts["gate_proj"]["kernel"]

    assert kernel.sharding.spec == P('expert', None, 'fsdp')
    assert kernel.addressable_shards[0].data.shape == (2, 32, 32)
    moments = state.opt_state[0].mu["layers_1"]["mlp"]["experts"]
    assert moments["gate_proj"]["kernel"].sharding.spec == kernel.sharding.spec


# --------------------------------------------------------------------------
# The balancing bias through Aux.variables
# --------------------------------------------------------------------------

@pytest.mark.mesh
def test_a_from_scratch_run_logs_the_load_and_moves_the_deepseek_bias():
    """The aux-loss-free balancing end to end: the routers sow their loads,
    the loss reports them and hands the bias update back through
    Aux.variables, and the trainer writes it into the `moe` collection.

    The bias starts at zero on every expert and the router is skewed by the
    tokens, so after a run the busiest experts carry a negative bias, the
    idlest a positive one, and the bias is not what a fresh init holds.
    """
    steps = 30
    tracker = RecordingTracker()
    trainer = moe_trainer(1, 2, tracker=tracker, bias=True)
    fresh = trainer.initial_state()
    assert set(fresh.variables) == {"params", "moe"}
    np.testing.assert_array_equal(
        np.asarray(fresh.variables["moe"]["layers_1"]["mlp"]["gate"]["e_score_correction_bias"]), 0.0)

    state = trainer.fit(Data(token_batches), steps=steps, log_every=1)

    bias = np.asarray(state.variables["moe"]["layers_1"]["mlp"]["gate"]["e_score_correction_bias"])
    assert np.any(bias != 0), "the bias never moved"
    assert bias.min() < 0 < bias.max(), bias
    # Every step moved every expert by the rate, one way or the other.
    np.testing.assert_allclose(np.abs(bias) / 0.01, np.round(np.abs(bias) / 0.01), atol=1e-4)
    ticks = [scalars for _, scalars in tracker.scalars if "train/moe/max_load" in scalars]
    loads = [entry["train/moe/max_load"] for entry in ticks]
    assert len(loads) == steps and all(1 / 8 <= load <= 1.0 for load in loads)
    assert all(entry["train/moe/min_load"] <= 1 / 8 for entry in ticks)
    # The bias is state, not a parameter: the optimizer holds no moment for it.
    assert "moe" not in state.opt_state[0].mu


@pytest.mark.mesh
def test_balancing_needs_a_router_with_a_bias():
    trainer = moe_trainer(1, 2)
    trainer.objective.balance_rate = 0.01
    params = trainer.initial_state().variables
    with pytest.raises(ValueError, match="bias=True"):
        trainer.objective.scalar_loss(params, next(token_batches()),
                               Step(step=jnp.asarray(0), key=jax.random.key(0), ema=None))


@pytest.mark.mesh
def test_the_balancing_bias_is_one_replicated_value_across_every_shard():
    """The bias is state every device reads, and its update counts tokens
    over the whole global batch, not over a device's slice.

    The step is `jax.jit` with in_shardings and out_shardings
    (dew/training/trainer.py), so the histogram inside it runs on the global
    array and the compiler inserts the reduction; this pins that. Every
    addressable shard has to hold the same whole bias, and the bias a mesh
    that splits the batch eight ways produces has to be the bias one device
    produces from the same batch. A per-shard count would move the bias by a
    per-shard direction and the shards would disagree.
    """
    steps = 3
    state = moe_trainer(4, 2, bias=True).fit(Data(token_batches), steps=steps)
    bias = state.variables["moe"]["layers_1"]["mlp"]["gate"]["e_score_correction_bias"]

    assert bias.sharding.spec == jax.sharding.PartitionSpec(), bias.sharding
    assert len(bias.addressable_shards) == jax.device_count()
    whole = np.asarray(jax.device_get(bias))
    for shard in bias.addressable_shards:
        assert np.array_equal(np.asarray(jax.device_get(shard.data)), whole)
    assert np.any(whole != 0), "the bias never moved"
    # Every step moved every expert by the rate, one way or the other.
    np.testing.assert_allclose(np.abs(whole) / 0.01, np.round(np.abs(whole) / 0.01),
                               atol=1e-4)
    assert np.abs(whole).max() <= steps * 0.01 + 1e-6

    single = single_device_bias(steps)
    np.testing.assert_array_equal(whole, single)


def single_device_bias(steps):
    """The same run on one device, as the value the sharded run must match.

    A fresh process is the only way to change the device count, so this
    subprocess runs the same trainer at one device and prints the bias.
    """
    script = f"""
import numpy as np, jax, optax
from dew.registry import models
import dew.nn.backbones
from dew.objectives.lm import LMObjective
from dew.training import Trainer, MeshSpec, Layout
import test_moe_decoder as suite

model = models.build("causal_transformer", **{{
    **suite.moe_config(),
    "mixture": {{**suite.moe_config()["mixture"], "bias": True}}}})
trainer = Trainer(LMObjective(model, suite.SEQ_LEN, balance_rate=0.01),
                  optax.adam(1e-3), key=jax.random.key(0),
                  mesh=MeshSpec(), layout=Layout(min_shard=suite.TINY_SHARD))
state = trainer.fit(suite.Data(suite.token_batches), steps={steps})
bias = state.variables["moe"]["layers_1"]["mlp"]["gate"]["e_score_correction_bias"]
print(",".join(repr(float(value)) for value in np.asarray(bias)))
"""
    environment = {**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
                   "JAX_PLATFORMS": "cpu",
                   "PYTHONPATH": os.pathsep.join(
                       [str(Path(__file__).resolve().parent),
                        str(Path(__file__).resolve().parents[1] / "src")])}
    finished = subprocess.run([sys.executable, "-c", script], check=True,
                              capture_output=True, text=True, env=environment)
    return np.asarray([float(value) for value in finished.stdout.strip().splitlines()[-1].split(",")])
