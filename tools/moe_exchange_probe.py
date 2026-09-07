#!/usr/bin/env python3
"""Measure bounded fp32 exchange and the unresolved bf16 gradient boundary.

Reference environment: JAX/jaxlib 0.11.1, NumPy 2.5.2, ml_dtypes 0.6.0.
Run from the checkout whose source is being measured:

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        PYTHONPATH=src python tools/moe_exchange_probe.py

The precision oracle uses NumPy float64 products of explicitly bf16-rounded
operands, independently of JAX's dot and autodiff. It reports both unrounded
fp32 gradients and gradients rounded once to bf16 before returning to fp32.
No tolerance is widened to make these disagreeing global paths equivalent.
"""

import json
import statistics
import time

import jax
import jax.numpy as jnp
import ml_dtypes
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


def probe_precision() -> None:
    rng = np.random.default_rng(271)
    x = rng.normal(size=(24, 8)).astype(np.float32)
    kernel = rng.normal(size=(8, 8, 16)).astype(np.float32)
    cotangent = rng.normal(size=(24, 16)).astype(np.float32)
    sizes = np.full(8, 3, np.int32)

    def bf16(value):
        return np.asarray(value).astype(ml_dtypes.bfloat16).astype(np.float64)

    reference = np.stack([bf16(x)[3*e:3*(e+1)].T @ bf16(cotangent)[3*e:3*(e+1)]
                          for e in range(8)]).astype(np.float32)
    rounded = bf16(reference).astype(np.float32)

    def loss(kernel, x, cotangent, sizes):
        y = jax.lax.ragged_dot(x.astype(jnp.bfloat16), kernel.astype(jnp.bfloat16), sizes,
                               preferred_element_type=jnp.bfloat16)
        return jnp.sum(y.astype(jnp.float32) * cotangent)

    def report(label, gradient):
        gradient = np.asarray(gradient)
        print(json.dumps({'precision_case': label,
                          'fp64_operand_oracle_error': float(np.max(abs(gradient - reference))),
                          'bf16_gradient_oracle_error': float(np.max(abs(gradient - rounded))),
                          'non_bf16_elements': int(np.count_nonzero(gradient != bf16(gradient)))}), flush=True)

    arrays = tuple(map(jnp.asarray, (kernel, x, cotangent, sizes)))
    report('unplaced_jit', jax.jit(jax.grad(loss))(*arrays))
    report('unplaced_eager', jax.grad(loss)(*arrays))
    for expert, fsdp in ((2, 4), (4, 2), (8, 1)):
        mesh = build_mesh(MeshSpec(expert=expert, fsdp=fsdp))
        kernels = NamedSharding(mesh, P('expert', None, 'fsdp'))
        tokens = NamedSharding(mesh, P(('expert', 'fsdp')))
        replicated = NamedSharding(mesh, P())
        arguments = tuple(jax.device_put(value, spec) for value, spec in zip(
            arrays, (kernels, tokens, tokens, replicated), strict=True))
        with jax.set_mesh(mesh):
            for spec, placement in ((kernels, 'placed'), (replicated, 'replicated-output')):
                report(f'expert{expert}/fsdp{fsdp}/{placement}',
                       jax.jit(jax.grad(loss), out_shardings=spec)(*arguments))


if __name__ == '__main__':
    print(json.dumps({'jax': jax.__version__, 'numpy': np.__version__,
                      'ml_dtypes': ml_dtypes.__version__, 'device': jax.devices()[0].device_kind}))
    measure_exchange()
    probe_precision()
