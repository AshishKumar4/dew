"""Probe (not for landing): host offload on the CPU backend and a compile-only H100."""

import json
import statistics
import time

import jax
import jax.experimental.topologies as topologies
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from kernel_compile import target_config

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import REMAT_POLICIES
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective


def stats(compiled):
    analysis = compiled.memory_analysis()
    return {name: getattr(analysis, name, "absent") for name in dir(analysis) if name.endswith("_in_bytes")}


def test_memories():
    device = jax.devices()[0]
    print("PROBE memories", device.platform, [memory.kind for memory in device.addressable_memories()],
          device.default_memory().kind)


def step_of(remat, devices):
    model = CausalTransformer(vocab_size=64, emb_features=64, num_layers=2, num_heads=2, mlp_features=128,
                              max_seq_len=64, remat=remat)
    objective = LMObjective(model, 64)
    variables = jax.eval_shape(lambda: model.init(jax.random.key(0), jnp.ones((1, 64), jnp.int32)))
    mesh = Mesh(np.asarray(devices[:1]), ("data",))
    whole = NamedSharding(mesh, P())
    placed = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=whole), variables)
    batch = {"text": jax.ShapeDtypeStruct((8, 65), jnp.int32, sharding=whole)}

    def loss(params, batch):
        at = Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(3), ema=None)
        return objective.scalar_loss({**placed, "params": params}, batch, at)[0]

    return jax.jit(jax.grad(loss)).trace(placed["params"], batch).lower().compile()


@pytest.mark.parametrize("remat", ["minimal", "minimal_offloaded"])
def test_activation_offload_cpu(remat):
    compiled = step_of(REMAT_POLICIES[remat], jax.devices())
    print("PROBE cpu", remat, json.dumps(stats(compiled)))


@pytest.mark.parametrize("remat", ["minimal", "minimal_offloaded"])
def test_activation_offload_h100(remat):
    topology = topologies.get_topology_desc("dew-compile", "cuda", target_config=target_config("sm90"),
                                            topology="1x1x1")
    compiled = step_of(REMAT_POLICIES[remat], list(topology.devices))
    print("PROBE h100", remat, json.dumps(stats(compiled)))


def test_state_offload_cpu():
    params = {"w": jnp.ones((256, 256))}
    device = jax.devices()[0]
    mesh = Mesh(np.asarray([device]), ("data",))
    on_device, on_host = NamedSharding(mesh, P()), NamedSharding(mesh, P(), memory_kind="pinned_host")
    optimizer = optax.adam(1e-3)
    state = jax.device_put(optimizer.init(params), jax.tree.map(lambda _: on_host, optimizer.init(params)))

    def step(params, state):
        state = jax.device_put(state, jax.tree.map(lambda _: on_device, state))
        updates, state = optimizer.update(params, state, params)
        return optax.apply_updates(params, updates), state

    compiled = jax.jit(step, out_shardings=(on_device, jax.tree.map(lambda _: on_host, state))).lower(
        params, state).compile()
    print("PROBE state cpu", json.dumps(stats(compiled)))


def test_host_bandwidth_cpu():
    device = jax.devices()[0]
    mesh = Mesh(np.asarray([device]), ("data",))
    on_device, on_host = NamedSharding(mesh, P()), NamedSharding(mesh, P(), memory_kind="pinned_host")
    array = jax.device_put(np.zeros((1 << 26,), np.float32), on_device)
    seconds = []
    for _ in range(7):
        start = time.perf_counter()
        back = jax.device_put(jax.device_put(array, on_host), on_device)
        back.block_until_ready()
        seconds.append(time.perf_counter() - start)
    print("PROBE bandwidth cpu", 2 * array.nbytes / statistics.median(seconds[2:]))
