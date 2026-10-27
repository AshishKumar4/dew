"""Probe (not for landing): how XLA's memory analysis reports a step that fetches host state early or late."""

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
    return {name.removesuffix("_in_bytes"): int(getattr(analysis, name)) for name in dir(analysis)
            if name.endswith("_in_bytes") and getattr(analysis, name)}


@pytest.mark.parametrize("target", ["cpu", "sm80", "sm90"])
def test_probe(target):
    devices = jax.devices("cpu")[:1] if target == "cpu" else list(compile_only_gpus(target))
    mesh = Mesh(np.asarray(devices), ("data",))
    device, host = NamedSharding(mesh, P()), NamedSharding(mesh, P(), memory_kind="pinned_host")
    width, rows = 2048, 4096
    params = {f"w{index}": jax.ShapeDtypeStruct((width, width), jnp.float32, sharding=device) for index in range(4)}
    optimizer = optax.adam(1e-3)
    batch = jax.ShapeDtypeStruct((rows, width), jnp.float32, sharding=device)

    def loss(params, x):
        for index in range(4):
            x = jnp.tanh(x @ params[f"w{index}"])
        return jnp.mean(x ** 2)

    for kind in ("resident", "early", "late"):
        where = device if kind == "resident" else host
        state = jax.tree.map(lambda leaf, where=where: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=where),
                             jax.eval_shape(optimizer.init, params))
        on_device = jax.tree.map(lambda _: device, state)

        def step(params, state, x, kind=kind, on_device=on_device):
            if kind == "early":
                state = jax.device_put(state, on_device)
            grads = jax.grad(loss)(params, x)
            if kind == "late":
                grads, state = jax.lax.optimization_barrier((grads, state))
                state = jax.device_put(state, on_device)
            updates, state = optimizer.update(grads, state, params)
            return optax.apply_updates(params, updates), state

        compiled = jax.jit(step, donate_argnums=(0, 1),
                           out_shardings=(device, jax.tree.map(lambda _: where, state))).trace(
            params, state, batch).lower().compile()
        print("PROBE", target, kind, json.dumps(stats(compiled)))
