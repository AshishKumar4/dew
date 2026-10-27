"""Probe (not for landing): how XLA's memory analysis reports host-memory arguments and outputs."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from kernel_compile import compile_only_gpus


def stats(compiled):
    analysis = compiled.memory_analysis()
    return {name: int(getattr(analysis, name)) for name in dir(analysis) if name.endswith("_in_bytes")}


@pytest.mark.parametrize("target", ["cpu", "sm80"])
@pytest.mark.parametrize("donate", [True, False])
def test_probe(target, donate):
    devices = jax.devices("cpu")[:1] if target == "cpu" else list(compile_only_gpus(target))
    mesh = Mesh(np.asarray(devices), ("data",))
    device, host = NamedSharding(mesh, P()), NamedSharding(mesh, P(), memory_kind="pinned_host")
    params = {"w": jax.ShapeDtypeStruct((1024, 1024), jnp.float32, sharding=device)}
    optimizer = optax.adam(1e-3)
    for kind, where in (("resident", device), ("offloaded", host)):
        state = jax.tree.map(lambda leaf, where=where: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=where),
                             jax.eval_shape(optimizer.init, params))

        def step(params, state, where=where):
            state = jax.device_put(state, jax.tree.map(lambda _: device, state))
            updates, state = optimizer.update(params, state, params)
            return optax.apply_updates(params, updates), jax.device_put(state, jax.tree.map(lambda _: where, state))

        compiled = jax.jit(step, donate_argnums=(0, 1) if donate else (),
                           out_shardings=(device, jax.tree.map(lambda _: where, state))).trace(
            params, state).lower().compile()
        print("PROBE", target, "donate" if donate else "kept", kind, json.dumps(stats(compiled)))
