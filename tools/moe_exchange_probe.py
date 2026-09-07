#!/usr/bin/env python3
"""Measure the working memory and CPU time of exchange dispatch against
global dispatch on identical placements.

Reference environment: JAX/jaxlib 0.11.1, NumPy 2.5.2. Run from the checkout
whose source is being measured:

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        PYTHONPATH=src python tools/moe_exchange_probe.py

Both dispatches share `moe.expert_projection`'s precision contract, which
tests/test_moe_precision.py checks against float64 oracles; this measures
cost, and asserts the two agree on the way.
"""

import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.nn.moe import ExpertMLP
from dew.training import MeshSpec, build_mesh


def measure_exchange() -> None:
    mesh = build_mesh(MeshSpec(expert=4, fsdp=2))
    model = ExpertMLP(8, 128, 64)
    x = jax.random.normal(jax.random.key(1), (256, 64))
    ids = jax.lax.top_k(jax.random.normal(jax.random.key(2), (256, 8)), 2)[1]
    weights = jnp.full((256, 2), 0.5, jnp.float32)
    parameters = model.init(jax.random.key(3), x, weights, ids)
    columns = NamedSharding(mesh, P('expert', None, 'fsdp'))
    parameter_specs = {'params': {
        'gate_proj': {'kernel': columns}, 'up_proj': {'kernel': columns},
        'down_proj': {'kernel': NamedSharding(mesh, P('expert', 'fsdp'))}}}
    tokens = NamedSharding(mesh, P(('expert', 'fsdp')))
    parameters = jax.device_put(parameters, parameter_specs)
    x, weights, ids = (jax.device_put(value, tokens) for value in (x, weights, ids))
    baseline = {}
    for dispatch in ('global', 'exchange'):
        module = model.clone(dispatch=dispatch)

        def loss(parameters, x, weights, ids):
            y = jnp.asarray(module.apply(parameters, x, weights, ids))
            return jnp.mean(y**2), y

        derivative = jax.value_and_grad(loss, (0, 1, 2), has_aux=True)
        derivative_specs = ((NamedSharding(mesh, P()), tokens),
                            (parameter_specs, tokens, tokens))
        for phase, fn, spec in (('forward', module.apply, tokens),
                                ('backward', derivative, derivative_specs)):
            with jax.set_mesh(mesh):
                compiled = jax.jit(fn, out_shardings=spec).lower(parameters, x, weights, ids).compile()
            value = jax.block_until_ready(compiled(parameters, x, weights, ids))
            if dispatch == 'global':
                baseline[phase] = value
            errors = [float(jnp.max(jnp.abs(a - b))) for a, b in zip(
                jax.tree.leaves(value), jax.tree.leaves(baseline[phase]), strict=True)]
            for a, b in zip(jax.tree.leaves(value), jax.tree.leaves(baseline[phase]), strict=True):
                np.testing.assert_allclose(a, b, atol=3e-5, rtol=3e-5)
            durations = []
            for _ in range(10):
                start = time.perf_counter()
                jax.block_until_ready(compiled(parameters, x, weights, ids))
                durations.append(time.perf_counter() - start)
            memory = compiled.memory_analysis()
            if memory is None:
                raise RuntimeError('the backend did not report compiled memory usage')
            print(json.dumps({'dispatch': dispatch, 'phase': phase,
                              'median_seconds': statistics.median(durations),
                              'temporary_bytes': memory.temp_size_in_bytes,
                              'output_bytes': memory.output_size_in_bytes,
                              'max_errors': errors}), flush=True)


if __name__ == '__main__':
    print(json.dumps({'jax': jax.__version__, 'numpy': np.__version__,
                      'device': jax.devices()[0].device_kind}))
    measure_exchange()
