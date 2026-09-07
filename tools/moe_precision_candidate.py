#!/usr/bin/env python3
"""Opt-in precision-bugfix investigation; production defaults are unchanged.

Proposed shared contract: grouped projections use the configured operand
dtype, accumulate the complete forward contraction at least in fp32, then
round once to activation dtype. Derivative contractions use a common work
dtype covering the configured operands and both original input/master
dtypes, with an fp32 floor; cotangents return to their original dtype only
after complete reductions. Exact GELU evaluates at least in fp32 before
returning to activation dtype. These rules do not depend on placement.
The tangent law is delta_x @ Q(kernel) + Q(x) @ delta_kernel, where Q keeps
the configured rounded operand values but carries straight-through tangents
in the original input/master dtype. Its automatic transpose follows the
same gradient boundaries; higher-order differentiation retains this law.

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
Custom JVP rules support both differentiation directions. Independent checks
cover primal values, JVPs, VJPs and both mixed second-order directions, as
well as full expert-transport JVPs and real optimizer updates.
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


@jax.custom_jvp
def exact_gelu(x: jax.Array) -> jax.Array:
    x = jax.lax.optimization_barrier(x)
    work = x.astype(jnp.promote_types(x.dtype, jnp.float32))
    return nn.gelu(work, approximate=False).astype(x.dtype)


@exact_gelu.defjvp
def exact_gelu_jvp(primals: tuple[jax.Array], tangents: tuple[jax.Array]
                   ) -> tuple[jax.Array, jax.Array]:
    x, = primals
    dx, = tangents
    output = exact_gelu(x)
    work = x.astype(jnp.promote_types(x.dtype, jnp.float32))
    _, derivative = jax.jvp(lambda value: nn.gelu(value, approximate=False),
                            (work,), (jnp.ones_like(work),))
    # These barriers declare activation/cotangent dtype boundaries in both
    # differentiation directions, independently of producer/consumer fusion.
    tangent = derivative * jax.lax.optimization_barrier(dx).astype(work.dtype)
    return output, jax.lax.optimization_barrier(tangent.astype(x.dtype))


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def rounded_operand(x: jax.Array, dtype: Dtype) -> jax.Array:
    # Keep rounded values in the original dtype so their straight-through
    # master tangent does not acquire a second low-precision rounding step.
    return jax.lax.optimization_barrier(x.astype(dtype)).astype(x.dtype)


@rounded_operand.defjvp
def rounded_operand_jvp(dtype: Dtype, primals: tuple[jax.Array], tangents: tuple[jax.Array]
                        ) -> tuple[jax.Array, jax.Array]:
    return jnp.asarray(rounded_operand(primals[0], dtype)), tangents[0]


@functools.partial(jax.custom_jvp, nondiff_argnums=(3, 4, 5))
def projection(x: jax.Array, kernel: jax.Array, sizes: jax.Array,
               dtype: Dtype | None, implementation: str, precision: PrecisionLike) -> jax.Array:
    x, kernel = promote_dtype(x, kernel, dtype=dtype)
    accumulated = grouped_matmul(x, kernel, sizes, implementation=implementation,
                                  precision=precision,
                                  preferred_element_type=jnp.promote_types(x.dtype, jnp.float32))
    return accumulated.astype(x.dtype)


@projection.defjvp
def projection_jvp(dtype: Dtype | None, implementation: str, precision: PrecisionLike,
                   primals: tuple[jax.Array, jax.Array, jax.Array],
                   tangents: tuple[jax.Array, jax.Array, jax.Array]
                   ) -> tuple[jax.Array, jax.Array]:
    x, kernel, sizes = primals
    dx, dk, _ = tangents
    dx, dk = jax.lax.optimization_barrier((dx, dk))
    output = jnp.asarray(projection(x, kernel, sizes, dtype, implementation, precision))
    work = jnp.result_type(output.dtype, x.dtype, kernel.dtype, jnp.float32)
    inputs = jnp.asarray(rounded_operand(x, output.dtype))
    matrix = jnp.asarray(rounded_operand(kernel, output.dtype))
    input_term = grouped_matmul(dx.astype(work), matrix.astype(work), sizes,
                                implementation=implementation, precision=precision,
                                preferred_element_type=work)
    kernel_term = grouped_matmul(inputs.astype(work), dk.astype(work), sizes,
                                 implementation=implementation, precision=precision,
                                 preferred_element_type=work)
    tangent = jax.lax.optimization_barrier((input_term + kernel_term).astype(output.dtype))
    return output, tangent



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
    random_case = ('random', rng.normal(size=(24, 16)).astype(np.float32),
                   rng.normal(size=(8, 16, 16)).astype(np.float32),
                   rng.normal(size=(24, 16)).astype(np.float32))
    forward_kernel = np.zeros((8, 16, 8), np.float32)
    forward_kernel[:, 0, :] = 1
    forward_kernel[:, (1, 8), :] = 2**-8
    input_kernel = np.zeros((8, 8, 16), np.float32)
    input_kernel[:, :, 0] = 1
    input_kernel[:, :, (1, 8)] = 2**-8
    cases = (random_case,
             ('forward-round-once', np.ones((24, 16), np.float32), forward_kernel,
              np.ones((24, 8), np.float32)),
             ('input-gradient-round-once', np.ones((24, 8), np.float32), input_kernel,
              np.ones((24, 16), np.float32)))
    sizes = np.full(8, 3, np.int32)

    def loss(kernel, x, dy, sizes):
        projected = jnp.asarray(projection(x, kernel, sizes, jnp.bfloat16, 'xla', None))
        return jnp.sum(projected.astype(jnp.float32) * dy), projected

    for case, x, kernel, dy in cases:
        forward_oracle = bf16(np.concatenate([
            bf16(x)[3*e:3*(e+1)] @ bf16(kernel)[e] for e in range(8)]))
        dk_oracle = np.stack([bf16(x)[3*e:3*(e+1)].T @ bf16(dy)[3*e:3*(e+1)]
                              for e in range(8)])
        dx_sum = np.concatenate([bf16(dy)[3*e:3*(e+1)] @ bf16(kernel)[e].T for e in range(8)])
        dtype_cases = ((jnp.float32, jnp.bfloat16),)
        if case == 'random':
            dtype_cases += ((jnp.float64, jnp.bfloat16), (jnp.float32, jnp.float32),
                            (jnp.float64, jnp.float64))
        for master, input_dtype in dtype_cases:
            dx_oracle = dx_sum.astype(input_dtype).astype(np.float64)
            arrays = (jnp.asarray(kernel, master), jnp.asarray(x, input_dtype),
                      jnp.asarray(dy), jnp.asarray(sizes))
            for expert, fsdp in ((2, 4), (4, 2), (8, 1)):
                mesh = build_mesh(MeshSpec(expert=expert, fsdp=fsdp))
                replicated = NamedSharding(mesh, P())
                for orientation in ('rows', 'columns'):
                    rows = orientation == 'rows'
                    kernels = NamedSharding(mesh, P('expert', 'fsdp', None) if rows
                                            else P('expert', None, 'fsdp'))
                    inputs = NamedSharding(mesh, P('expert', 'fsdp') if rows else P('expert'))
                    outputs = NamedSharding(mesh, P('expert') if rows else P('expert', 'fsdp'))
                    arguments = tuple(jax.device_put(value, spec) for value, spec in zip(
                        arrays, (kernels, inputs, outputs, replicated), strict=True))
                    with jax.set_mesh(mesh):
                        for spec, placement in ((kernels, 'placed'), (replicated, 'replicated-gradient')):
                            (_, y), (dk, dx) = jax.jit(
                                jax.value_and_grad(loss, (0, 1), has_aux=True),
                                out_shardings=((replicated, outputs), (spec, inputs)))(*arguments)
                            assert dk.dtype == master and dx.dtype == input_dtype
                            actual = (np.asarray(y).astype(np.float64), np.asarray(dk),
                                      np.asarray(dx).astype(np.float64))
                            expected = (forward_oracle, dk_oracle, dx_oracle)
                            for a, b in zip(actual, expected, strict=True):
                                np.testing.assert_allclose(a, b, atol=3e-5, rtol=3e-5)
                            print(json.dumps({
                                'projection': f'{case}/{jnp.dtype(master).name}/{jnp.dtype(input_dtype).name}'
                                              f'/expert{expert}/fsdp{fsdp}/{orientation}/{placement}',
                                'forward_oracle_error': float(np.max(abs(actual[0] - expected[0]))),
                                'kernel_oracle_error': float(np.max(abs(actual[1] - expected[1]))),
                                'input_oracle_error': float(np.max(abs(actual[2] - expected[2])))}), flush=True)


def check_mixed_precision_cancellation() -> None:
    epsilon = 2.0**-30
    x = jnp.asarray([[1 + epsilon], [-1]], jnp.float64)
    kernel = jnp.ones((1, 1, 1), jnp.float32)
    sizes = jnp.asarray([2], jnp.int32)

    def inferred(x, kernel):
        return jnp.asarray(projection(x, kernel, sizes, None, 'xla', None))

    y, tangent = jax.jit(lambda x, kernel: jax.jvp(
        inferred, (x, kernel), (jnp.zeros_like(x), jnp.ones_like(kernel))))(x, kernel)
    dk = jax.jit(jax.grad(lambda kernel: inferred(x, kernel).sum()))(kernel)
    np.testing.assert_array_equal(y, np.asarray(x))
    np.testing.assert_array_equal(tangent, np.asarray(x))
    np.testing.assert_array_equal(dk, np.full(kernel.shape, epsilon, np.float32))

    x = jnp.ones((1, 1), jnp.float32)
    kernel = jnp.asarray([[[1 + epsilon, -1]]], jnp.float64)
    sizes = jnp.asarray([1], jnp.int32)

    def explicit(x, kernel):
        return jnp.asarray(projection(x, kernel, sizes, jnp.float64, 'xla', None))

    y, tangent = jax.jit(lambda x, kernel: jax.jvp(
        explicit, (x, kernel), (jnp.ones_like(x), jnp.zeros_like(kernel))))(x, kernel)
    dx = jax.jit(jax.grad(lambda x: explicit(x, kernel).sum()))(x)
    np.testing.assert_array_equal(y, np.asarray(kernel[0]))
    np.testing.assert_array_equal(tangent, np.asarray(kernel[0]))
    np.testing.assert_array_equal(dx, np.full(x.shape, epsilon, np.float32))

    kernel = jnp.asarray([[[1., -1.]]], jnp.float64)
    direction = jnp.asarray([[[1 + epsilon, -1]]], jnp.float64)

    def scalar(x, kernel):
        return jnp.asarray(projection(x, kernel, sizes, jnp.bfloat16, 'xla', None)).astype(jnp.float64).sum()

    def mixed(x, kernel, direction):
        _, forward_reverse = jax.jvp(jax.grad(scalar, (0, 1)), (x, kernel),
                                     (jnp.zeros_like(x), direction))

        def directional(x, kernel):
            return jax.jvp(scalar, (x, kernel), (jnp.zeros_like(x), direction))[1]

        return forward_reverse, jax.grad(directional, (0, 1))(x, kernel)

    for dx, dk in jax.jit(mixed)(x, kernel, direction):
        # The residue is exactly representable; a generic fp32 atol would
        # admit the old zero result and miss this mixed-order regression.
        np.testing.assert_array_equal(dx, np.full(x.shape, epsilon, np.float32))
        np.testing.assert_array_equal(dk, np.zeros(kernel.shape, np.float64))
    print(json.dumps({'mixed_precision_cancellation': 'all three exact-residue regressions passed',
                      'residue': epsilon}), flush=True)



def check_projection_higher_order() -> None:
    rng = np.random.default_rng(943)
    values = rng.normal(size=(24, 16))
    matrices = rng.normal(size=(8, 16, 16))
    input_direction = rng.normal(scale=.3, size=values.shape)
    kernel_direction = rng.normal(scale=.3, size=matrices.shape)
    cotangent = rng.normal(size=(24, 16))
    sizes = jnp.full(8, 3, jnp.int32)

    def forward(a, b):
        return (np.asarray(a, np.float64).reshape(8, 3, 16) @ np.asarray(b, np.float64)).reshape(24, 16)

    def backward(a, b):
        return forward(a, np.asarray(b).swapaxes(1, 2))

    def parameters(a, b):
        return (np.asarray(a, np.float64).reshape(8, 3, 16).swapaxes(1, 2)
                @ np.asarray(b, np.float64).reshape(8, 3, 16))

    for compute, input_dtype, master in (
            (jnp.bfloat16, jnp.bfloat16, jnp.float32),
            (jnp.bfloat16, jnp.float32, jnp.float64),
            (jnp.float32, jnp.float32, jnp.float32),
            (jnp.float64, jnp.float64, jnp.float64)):
        x, kernel = values.astype(input_dtype), matrices.astype(master)
        dx, dk = input_direction.astype(input_dtype), kernel_direction.astype(master)

        def typed(a, dtype):
            return np.asarray(a, dtype=dtype).astype(np.float64)

        qx, qk, qc = (typed(a, compute) for a in (x, kernel, cotangent))
        y_oracle = typed(forward(qx, qk), compute)
        tangent_oracle = typed(forward(dx, qk) + forward(qx, dk), compute)
        gradient_oracle = (typed(backward(qc, qk), input_dtype), typed(parameters(qx, qc), master))
        mixed_oracle = (typed(backward(qc, dk), input_dtype), typed(parameters(dx, qc), master))
        expected = (y_oracle, tangent_oracle, gradient_oracle, mixed_oracle, mixed_oracle)

        def evaluate(x, kernel, dx, dk, cotangent, sizes):
            def fn(x, kernel):
                return jnp.asarray(projection(x, kernel, sizes, compute, 'xla', None))
            y, tangent = jax.jvp(fn, (x, kernel), (dx, dk))

            def scalar(x, kernel):
                return jnp.sum(fn(x, kernel).astype(jnp.float64) * cotangent)

            gradient = jax.grad(scalar, (0, 1))(x, kernel)
            _, forward_reverse = jax.jvp(jax.grad(scalar, (0, 1)), (x, kernel), (dx, dk))

            def directional(x, kernel):
                _, direction = jax.jvp(fn, (x, kernel), (dx, dk))
                return jnp.sum(direction.astype(jnp.float64) * cotangent)

            reverse_forward = jax.grad(directional, (0, 1))(x, kernel)
            return y, tangent, gradient, forward_reverse, reverse_forward

        for expert, fsdp in ((2, 4), (4, 2), (8, 1)):
            mesh = build_mesh(MeshSpec(expert=expert, fsdp=fsdp))
            for orientation in ('rows', 'columns'):
                rows = orientation == 'rows'
                kernels = NamedSharding(mesh, P('expert', 'fsdp', None) if rows
                                        else P('expert', None, 'fsdp'))
                inputs = NamedSharding(mesh, P('expert', 'fsdp') if rows else P('expert'))
                outputs = NamedSharding(mesh, P('expert') if rows else P('expert', 'fsdp'))
                specs = (inputs, kernels, inputs, kernels, outputs, NamedSharding(mesh, P()))
                args = tuple(jax.device_put(jnp.asarray(value), spec) for value, spec in zip(
                    (x, kernel, dx, dk, cotangent, sizes), specs, strict=True))
                derivative_specs = (inputs, kernels)
                with jax.set_mesh(mesh):
                    actual = jax.jit(evaluate, out_shardings=(
                        outputs, outputs, derivative_specs, derivative_specs, derivative_specs))(*args)
                errors = []
                for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
                    a = np.asarray(a).astype(np.float64)
                    np.testing.assert_allclose(a, b, atol=3e-5, rtol=3e-5)
                    errors.append(float(np.max(abs(a - b))))
                print(json.dumps({'higher_projection':
                                  f'{jnp.dtype(compute).name}/{jnp.dtype(input_dtype).name}'
                                  f'/{jnp.dtype(master).name}/expert{expert}/fsdp{fsdp}/{orientation}',
                                  'primal_jvp_vjp_forward_reverse_reverse_forward_errors': errors}), flush=True)


def check_activation_higher_order() -> None:
    rng = np.random.default_rng(1041)
    values, cotangents, directions = rng.normal(size=(3, 1024))
    values *= 3
    directions *= .2
    for dtype in (jnp.bfloat16, jnp.float32, jnp.float64):
        x, ct, dx = (array.astype(dtype) for array in (values, cotangents, directions))
        v, c, t = (array.astype(np.float64) for array in (x, ct, dx))
        density = np.exp(-v*v/2) / np.sqrt(2*np.pi)
        first = .5 * erfc(-v / np.sqrt(2)) + v * density
        second = (2 - v*v) * density
        expected = tuple(array.astype(dtype).astype(np.float64) for array in (
            .5 * v * erfc(-v / np.sqrt(2)), first*t, first*c, second*c*t, second*t*c))

        def evaluate(x, dx, ct):
            y, tangent = jax.jvp(exact_gelu, (x,), (dx,))

            def scalar(x):
                return jnp.sum(exact_gelu(x).astype(jnp.float64) * ct.astype(jnp.float64))

            gradient = jax.grad(scalar)(x)
            _, forward_reverse = jax.jvp(jax.grad(scalar), (x,), (dx,))

            def directional(x):
                _, tangent = jax.jvp(exact_gelu, (x,), (dx,))
                return jnp.sum(tangent.astype(jnp.float64) * ct.astype(jnp.float64))

            return y, tangent, gradient, forward_reverse, jax.grad(directional)(x)

        actual = jax.jit(evaluate)(jnp.asarray(x), jnp.asarray(dx), jnp.asarray(ct))
        errors = []
        for a, b in zip(actual, expected, strict=True):
            a = np.asarray(a).astype(np.float64)
            np.testing.assert_allclose(a, b, atol=3e-5, rtol=3e-5)
            errors.append(float(np.max(abs(a - b))))
        print(json.dumps({'higher_activation': jnp.dtype(dtype).name,
                          'primal_jvp_vjp_forward_reverse_reverse_forward_errors': errors}), flush=True)



def place(value: jax.Array, sharding: NamedSharding) -> jax.Array:
    if value.is_fully_addressable:
        host = np.asarray(value)
        return jax.make_array_from_callback(host.shape, sharding, lambda index: host[index])
    return jax.device_put(value, sharding)



def check_transport_forward_mode() -> None:
    rng = np.random.default_rng(523)
    mesh = build_mesh(MeshSpec(expert=4, fsdp=2))
    columns = NamedSharding(mesh, P('expert', None, 'fsdp'))
    parameter_specs = {'params': {
        'gate_proj': {'kernel': columns}, 'up_proj': {'kernel': columns},
        'down_proj': {'kernel': NamedSharding(mesh, P('expert', 'fsdp'))}}}
    tokens = NamedSharding(mesh, P(('expert', 'fsdp')))
    for activation in ('swiglu', 'geglu', 'geglu_exact'):
        model = MasterGradientExperts(8, 16, 8, dtype=jnp.bfloat16, activation=activation)
        x = jnp.asarray(rng.normal(size=(24, 8)), jnp.bfloat16)
        weights = jnp.asarray(rng.uniform(.1, .9, size=(24, 2)), jnp.float32)
        ids = jnp.asarray(np.argsort(rng.normal(size=(24, 8)), axis=-1)[:, :2], jnp.int32)
        parameters = model.init(jax.random.key(19), x, weights, ids)
        dp = jax.tree.map(lambda value: jnp.asarray(rng.normal(scale=.05, size=value.shape),
                                                   value.dtype), parameters)
        dx = jnp.asarray(rng.normal(scale=.05, size=x.shape), x.dtype)
        dw = jnp.asarray(rng.normal(scale=.05, size=weights.shape), weights.dtype)
        parameters, dp = (jax.tree.map(place, value, parameter_specs) for value in (parameters, dp))
        x, weights, ids, dx, dw = (place(value, tokens) for value in (x, weights, ids, dx, dw))
        results = []
        for method in (None, model.exchange):
            def run(parameters, x, weights, dp, dx, dw, ids):
                def forward(parameters, x, weights):
                    return jnp.asarray(model.apply(parameters, x, weights, ids, method=method))
                return jax.jvp(forward, (parameters, x, weights), (dp, dx, dw))
            with jax.set_mesh(mesh):
                results.append(jax.jit(run, out_shardings=(tokens, tokens))(
                    parameters, x, weights, dp, dx, dw, ids))
        errors = []
        for first, second in zip(*results, strict=True):
            maximum = 0.0
            for a, b in zip(first.addressable_shards, second.addressable_shards, strict=True):
                left, right = np.asarray(a.data).astype(np.float64), np.asarray(b.data).astype(np.float64)
                np.testing.assert_allclose(left, right, atol=3e-5, rtol=3e-5)
                maximum = max(maximum, float(np.max(abs(left - right))))
            errors.append(maximum)
        print(json.dumps({'transport_jvp': activation, 'process': jax.process_index(),
                          'processes': jax.process_count(), 'primal_jvp_errors': errors}), flush=True)



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
            check_projection_higher_order()
            check_mixed_precision_cancellation()
            check_activation_higher_order()
    try:
        check_transport_forward_mode()
        for skewed in (False, True):
            compare_updates(skewed=skewed, scale_inputs=skewed, limit=.7 if skewed else None)
    finally:
        if distributed:
            jax.distributed.shutdown()
