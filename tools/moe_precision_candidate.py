#!/usr/bin/env python3
"""Opt-in precision-bugfix investigation; production defaults are unchanged.

Proposed shared contract: grouped projections use the configured activation
dtype, accumulate master-kernel cotangents in the original parameter dtype,
and return activation cotangents at their declared input dtype. Exact GELU
uses at least fp32 for its transcendental expression and derivative, then
returns to the activation dtype. Neither contract depends on placement.

Reference environment: JAX/jaxlib 0.11.1, NumPy 2.5.2, ml_dtypes 0.6.0,
SciPy 1.18.1.

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        PYTHONPATH=src python tools/moe_precision_candidate.py

For two real processes, start ranks 0 and 1 concurrently with the same
coordinator address as their second argument. Each process needs
`XLA_FLAGS=--xla_force_host_platform_device_count=4`; for example,
`python tools/moe_precision_candidate.py 0 127.0.0.1:12358` and rank 1
in the other terminal, both with `JAX_PLATFORMS=cpu PYTHONPATH=src`.

This proposal needs independent numerical review before any production
precision/default change. NumPy float64 operand sums and SciPy's float64
GELU equation supply independent oracles; no tolerance is widened for bf16.
The candidate primitives currently define reverse-mode rules only; retaining
the existing forward-mode interface is also a production-review requirement.
"""

import functools
import json
import statistics
import time

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import optax
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax.sharding import NamedSharding, PartitionSpec as P
from scipy.special import erfc

from dew.nn.moe import ExpertMLP, grouped_matmul
from dew.training import MeshSpec, build_mesh


@jax.custom_vjp
def exact_gelu(x: jax.Array) -> jax.Array:
    # Declare the activation's dtype boundary before the wider expression.
    x = jax.lax.optimization_barrier(x)
    work = x.astype(jnp.promote_types(x.dtype, jnp.float32))
    return nn.gelu(work, approximate=False).astype(x.dtype)


def exact_gelu_forward(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return exact_gelu(x), x


def exact_gelu_backward(x: jax.Array, dy: jax.Array) -> tuple[jax.Array]:
    work = x.astype(jnp.promote_types(x.dtype, jnp.float32))
    # The cotangent entering and leaving this bf16 activation is bf16 even
    # when its producer/consumer could otherwise fuse at greater precision.
    dy = jax.lax.optimization_barrier(dy).astype(work.dtype)
    _, pullback = jax.vjp(lambda value: nn.gelu(value, approximate=False), work)
    return (jax.lax.optimization_barrier(pullback(dy)[0].astype(x.dtype)),)


exact_gelu.defvjp(exact_gelu_forward, exact_gelu_backward)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def projection(x: jax.Array, kernel: jax.Array, sizes: jax.Array,
               dtype: Dtype | None, implementation: str, precision: PrecisionLike) -> jax.Array:
    x, kernel = promote_dtype(x, kernel, dtype=dtype)
    return grouped_matmul(x, kernel, sizes, implementation=implementation,
                          precision=precision, preferred_element_type=x.dtype)


def projection_forward(x: jax.Array, kernel: jax.Array, sizes: jax.Array,
                       dtype: Dtype | None, implementation: str, precision: PrecisionLike
                       ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    output = jnp.asarray(projection(x, kernel, sizes, dtype, implementation, precision))
    return output, (x, kernel, sizes)


def projection_backward(dtype: Dtype | None, implementation: str, precision: PrecisionLike,
                        residual: tuple[jax.Array, jax.Array, jax.Array], dy: jax.Array
                        ) -> tuple[jax.Array, jax.Array, None]:
    x, kernel, sizes = residual
    inputs, matrix = promote_dtype(x, kernel, dtype=dtype)
    dx = grouped_matmul(dy, matrix.swapaxes(1, 2), sizes, implementation=implementation,
                        precision=precision, preferred_element_type=inputs.dtype).astype(x.dtype)
    dk = jax.lax.ragged_dot_general(
        inputs, dy, sizes,
        jax.lax.RaggedDotDimensionNumbers((((0,), (0,)), ((), ())), (0,), ()),
        precision=precision, preferred_element_type=kernel.dtype)
    return dx, dk, None


projection.defvjp(projection_forward, projection_backward)



class MasterGradientExperts(ExpertMLP):
    """Candidate precision rules shared by the existing and exchanged execution."""

    def _project(self, tokens: jax.Array, sizes: jax.Array,
                 kernels: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        linear = functools.partial(projection, dtype=self.dtype,
                                   implementation=self.implementation, precision=self.precision)
        gate = jnp.asarray(linear(tokens, kernels[0], sizes))
        up = jnp.asarray(linear(tokens, kernels[1], sizes))
        if self.swiglu_limit is not None:
            gate = jnp.minimum(gate, self.swiglu_limit)
            up = jnp.clip(up, -self.swiglu_limit, self.swiglu_limit)
        if self.activation == 'swiglu':
            gate = nn.silu(gate)
        elif self.activation == 'geglu':
            gate = nn.gelu(gate, approximate=True)
        else:
            gate = exact_gelu(gate)
        return jnp.asarray(linear(gate * up, kernels[2], sizes))

    def exchange(self, x: jax.Array, weights: jax.Array, indices: jax.Array) -> jax.Array:
        # Invoke the real bounded transport, without lifting its production
        # bf16 refusal. This candidate remains an explicitly invoked method.
        return self._exchange(x, weights, indices,
                              (self.gate_proj.kernel, self.up_proj.kernel, self.down_proj.kernel))


def bf16(value):
    return np.asarray(value).astype(ml_dtypes.bfloat16).astype(np.float64)


def check_activation() -> None:
    values = bf16(np.linspace(-8, 8, 257))
    cdf = 0.5 * erfc(-values / np.sqrt(2))
    oracle = bf16(values * cdf).astype(np.float32)
    derivative = bf16(cdf + values * np.exp(-values**2 / 2) / np.sqrt(2*np.pi)).astype(np.float32)
    inputs = jnp.asarray(values, jnp.bfloat16)
    actual = jax.jit(exact_gelu)(inputs).astype(jnp.float32)
    gradient = jax.jit(jax.grad(lambda x: exact_gelu(x).astype(jnp.float32).sum()))(inputs)
    np.testing.assert_allclose(actual, oracle, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(gradient.astype(jnp.float32), derivative, atol=3e-5, rtol=3e-5)
    print(json.dumps({'activation': 'bf16 exact GELU',
                      'forward_oracle_error': float(np.max(abs(actual - oracle))),
                      'gradient_oracle_error': float(np.max(abs(gradient.astype(jnp.float32) - derivative)))}))


def check_projection() -> None:
    rng = np.random.default_rng(271)
    x = rng.normal(size=(24, 8)).astype(np.float32)
    kernel = rng.normal(size=(8, 8, 16)).astype(np.float32)
    dy = rng.normal(size=(24, 16)).astype(np.float32)
    sizes = np.full(8, 3, np.int32)
    dk_oracle = np.stack([bf16(x)[3*e:3*(e+1)].T @ bf16(dy)[3*e:3*(e+1)] for e in range(8)])
    dx_oracle = bf16(np.concatenate([bf16(dy)[3*e:3*(e+1)] @ bf16(kernel)[e].T
                                    for e in range(8)])).astype(np.float32)

    def loss(kernel, x, dy, sizes):
        projected = jnp.asarray(projection(x, kernel, sizes, jnp.bfloat16, 'xla', None))
        return jnp.sum(projected.astype(jnp.float32) * dy)

    for master in (jnp.float32, jnp.float64):
        arrays = (jnp.asarray(kernel, master), jnp.asarray(x, jnp.bfloat16),
                  jnp.asarray(dy), jnp.asarray(sizes))
        for expert, fsdp in ((2, 4), (4, 2), (8, 1)):
            mesh = build_mesh(MeshSpec(expert=expert, fsdp=fsdp))
            kernels = NamedSharding(mesh, P('expert', None, 'fsdp'))
            tokens = NamedSharding(mesh, P(('expert', 'fsdp')))
            replicated = NamedSharding(mesh, P())
            arguments = tuple(jax.device_put(value, spec) for value, spec in zip(
                arrays, (kernels, tokens, tokens, replicated), strict=True))
            with jax.set_mesh(mesh):
                for spec, placement in ((kernels, 'placed'), (replicated, 'replicated-output')):
                    dk, dx = jax.jit(jax.grad(loss, (0, 1)), out_shardings=(spec, tokens))(*arguments)
                    assert dk.dtype == master and dx.dtype == jnp.bfloat16
                    np.testing.assert_allclose(dk, dk_oracle, atol=3e-5, rtol=3e-5)
                    np.testing.assert_allclose(dx.astype(jnp.float32), dx_oracle, atol=3e-5, rtol=3e-5)
                    print(json.dumps({'projection': f'{jnp.dtype(master).name}/expert{expert}/fsdp{fsdp}/{placement}',
                                      'kernel_oracle_error': float(np.max(abs(np.asarray(dk) - dk_oracle))),
                                      'input_oracle_error': float(np.max(abs(np.asarray(dx.astype(jnp.float32)) - dx_oracle)))}), flush=True)


def place(value: jax.Array, sharding: NamedSharding) -> jax.Array:
    if value.is_fully_addressable:
        host = np.asarray(value)
        return jax.make_array_from_callback(host.shape, sharding, lambda index: host[index])
    return jax.device_put(value, sharding)



def compare_updates(*, skewed: bool = False, scale_inputs: bool = False,
                    limit: float | None = None) -> None:
    rng = np.random.default_rng(341)
    inputs = bf16(rng.normal(size=(24, 8)))
    weights = rng.uniform(.1, .9, size=(24, 2)).astype(np.float32)
    indices = (np.tile([0, 1], (24, 1)) if skewed else
               np.argsort(rng.normal(size=(24, 8)), axis=-1)[:, :2]).astype(np.int32)
    for activation in ('swiglu', 'geglu', 'geglu_exact'):
        for expert, fsdp in ((2, 4), (4, 2), (8, 1)):
            model = MasterGradientExperts(8, 16, 8, dtype=jnp.bfloat16, activation=activation,
                                           scale_inputs=scale_inputs, swiglu_limit=limit)
            x, w, ids = jnp.asarray(inputs, jnp.bfloat16), jnp.asarray(weights), jnp.asarray(indices)
            parameters = model.init(jax.random.key(19), x, w, ids)
            mesh = build_mesh(MeshSpec(expert=expert, fsdp=fsdp))
            columns = NamedSharding(mesh, P('expert', None, 'fsdp'))
            specs = {'params': {'gate_proj': {'kernel': columns}, 'up_proj': {'kernel': columns},
                               'down_proj': {'kernel': NamedSharding(mesh, P('expert', 'fsdp'))}}}
            tokens = NamedSharding(mesh, P(('expert', 'fsdp')))
            parameters = jax.tree.map(place, parameters, specs)
            x, w, ids = (place(value, tokens) for value in (x, w, ids))
            optimizer = optax.adam(1e-3)
            state = optimizer.init(parameters)
            state_specs = jax.tree.map(lambda value: NamedSharding(mesh, P())
                                       if value.ndim == 0 else value.sharding, state)
            state = jax.tree.map(place, state, state_specs)
            legacy = ExpertMLP(8, 16, 8, dtype=jnp.bfloat16, activation=activation,
                               scale_inputs=scale_inputs, swiglu_limit=limit)
            baselines = {'legacy-global': [], 'global': []}
            for label, network, method in (('legacy-global', legacy, None),
                                            ('global', model, None),
                                            ('exchange', model, model.exchange)):
                def objective(p, x, w, ids):
                    y = jnp.asarray(network.apply(p, x, w, ids, method=method))
                    return jnp.mean(jnp.sin(y.astype(jnp.float32))), y

                def step(p, state, x, w, ids):
                    result, gradients = jax.value_and_grad(objective, (0, 1, 2), has_aux=True)(p, x, w, ids)
                    updates, state = optimizer.update(gradients[0], state, p)
                    return result, gradients, optax.apply_updates(p, updates), state

                outputs = ((NamedSharding(mesh, P()), tokens), (specs, tokens, tokens), specs, state_specs)
                with jax.set_mesh(mesh):
                    compiled = jax.jit(step, out_shardings=outputs).lower(parameters, state, x, w, ids).compile()
                p, s = parameters, state
                maxima = []
                comparison = 'global' if label == 'exchange' else 'legacy-global'
                for number in range(3):
                    result = jax.block_until_ready(compiled(p, s, x, w, ids))
                    if label in baselines:
                        baselines[label].append(result)
                    if label != 'legacy-global':
                        differences = []
                        for a, b in zip(jax.tree.leaves(result),
                                        jax.tree.leaves(baselines[comparison][number]), strict=True):
                            for first, second in zip(a.addressable_shards, b.addressable_shards, strict=True):
                                left = np.asarray(first.data).astype(np.float64)
                                right = np.asarray(second.data).astype(np.float64)
                                if label == 'exchange':
                                    np.testing.assert_allclose(left, right, atol=3e-5, rtol=3e-5)
                                differences.append(float(np.max(abs(left - right))))
                        maxima.append(max(differences))
                    p, s = result[2:]
                times = []
                for _ in range(5):
                    start = time.perf_counter()
                    jax.block_until_ready(compiled(parameters, state, x, w, ids))
                    times.append(time.perf_counter() - start)
                memory = compiled.memory_analysis()
                if memory is None:
                    raise RuntimeError('backend memory analysis unavailable')
                print(json.dumps({'updates': f'{activation}/expert{expert}/fsdp{fsdp}',
                                  'process': jax.process_index(), 'processes': jax.process_count(),
                                  'skewed': skewed, 'scale_inputs': scale_inputs, 'limit': limit,
                                  'dispatch': label, 'comparison': comparison,
                                  'three_step_errors': maxima, 'temporary_bytes': memory.temp_size_in_bytes,
                                  'median_seconds': statistics.median(times)}), flush=True)


if __name__ == '__main__':
    import sys
    distributed = len(sys.argv) == 3
    if distributed:
        jax.distributed.initialize(coordinator_address=sys.argv[2], num_processes=2,
                                   process_id=int(sys.argv[1]), local_device_ids=[0, 1, 2, 3],
                                   initialization_timeout=30)
    else:
        print(json.dumps({'jax': jax.__version__, 'numpy': np.__version__, 'ml_dtypes': ml_dtypes.__version__}))
        check_activation()
        with jax.enable_x64():
            check_projection()
    try:
        for skewed in (False, True):
            compare_updates(skewed=skewed, scale_inputs=skewed, limit=.7 if skewed else None)
    finally:
        if distributed:
            jax.distributed.shutdown()
