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
from recording import RecordingTracker
from sharded import assert_sharded
from token_support import repeated_token_batch

from dew.data.dataset import Loader
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mlp import GatedMLP
from dew.nn.moe import SparseMLP
from dew.objectives.lm import LMObjective
from dew.objectives.routing import ExpertBalance
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer

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


# --------------------------------------------------------------------------
# The expert mesh axis
# --------------------------------------------------------------------------

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
    return repeated_token_batch(VOCAB, BATCH, SEQ_LEN + 1)


def moe_trainer(expert_size, fsdp_size, tracker=None, bias=False, learning_rate=1e-3):
    config = moe_config()
    model = models.build("causal_transformer",
                         **{**config, "mixture": {**config["mixture"], "bias": bias}})
    return Trainer(
        LMObjective(model, SEQ_LEN, balance=ExpertBalance(balance_rate=0.01) if bias else None),
        optax.adam(learning_rate), key=jax.random.key(0),
        mesh=MeshSpec(fsdp=fsdp_size, expert=expert_size),
        layout=Layout(min_shard=TINY_SHARD), tracker=tracker)


def run_losses(trainer, steps):
    """Per-step losses of a fit, as the tracker receives them."""
    trainer.tracker = tracker = RecordingTracker()
    state = trainer.fit(
        Loader(lambda partition: token_batches(), None, batch=BATCH, records=None), steps=steps, log_every=1)
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
    idlest a positive one, and the bias is not what a fresh init holds. The
    parameters stay put (a zero learning rate, the optimizer's moments still
    kept), so the bias alone moves the routing of the one batch the run
    repeats: the busiest expert's load falls and the spread between it and
    the idlest closes.
    """
    steps = 30
    tracker = RecordingTracker()
    trainer = moe_trainer(1, 2, tracker=tracker, bias=True, learning_rate=0.0)
    fresh = trainer.initial_state()
    assert set(fresh.variables) == {"params", "moe"}
    np.testing.assert_array_equal(
        np.asarray(fresh.variables["moe"]["layers_1"]["mlp"]["gate"]["e_score_correction_bias"]), 0.0)

    state = trainer.fit(
        Loader(lambda partition: token_batches(), None, batch=BATCH, records=None), steps=steps, log_every=1)

    bias = np.asarray(state.variables["moe"]["layers_1"]["mlp"]["gate"]["e_score_correction_bias"])
    assert np.any(bias != 0), "the bias never moved"
    assert bias.min() < 0 < bias.max(), bias
    # Every step moved every expert by the rate, one way or the other.
    np.testing.assert_allclose(np.abs(bias) / 0.01, np.round(np.abs(bias) / 0.01), atol=1e-4)
    ticks = [scalars for _, scalars in tracker.scalars if "train/moe/max_load" in scalars]
    loads = [entry["train/moe/max_load"] for entry in ticks]
    assert len(loads) == steps and all(1 / 8 <= load <= 1.0 for load in loads)
    assert all(entry["train/moe/min_load"] <= 1 / 8 for entry in ticks)
    spread = [entry["train/moe/max_load"] - entry["train/moe/min_load"] for entry in ticks]
    assert loads[-1] < loads[0] and spread[-1] < spread[0] / 2, (loads, spread)
    # The bias is state, not a parameter: the optimizer holds no moment for it.
    assert "moe" not in state.opt_state[0].mu


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
    state = moe_trainer(4, 2, bias=True).fit(
        Loader(lambda partition: token_batches(), None, batch=BATCH, records=None), steps=steps)
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
from dew.data.dataset import Loader
from dew.registry import models
import dew.nn.backbones
from dew.objectives.lm import LMObjective
from dew.objectives.routing import ExpertBalance
from dew.training import Trainer, MeshSpec, Layout
import test_moe_decoder as suite

model = models.build("causal_transformer", **{{
    **suite.moe_config(),
    "mixture": {{**suite.moe_config()["mixture"], "bias": True}}}})
trainer = Trainer(LMObjective(model, suite.SEQ_LEN, balance=ExpertBalance(balance_rate=0.01)),
                  optax.adam(1e-3), key=jax.random.key(0),
                  mesh=MeshSpec(), layout=Layout(min_shard=suite.TINY_SHARD))
state = trainer.fit(
    Loader(lambda partition: suite.token_batches(), None, batch=suite.BATCH, records=None), steps={steps})
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
