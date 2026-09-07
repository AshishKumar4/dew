"""The precision contract routed experts train under, against independent
float64 arithmetic on the rounded operands.

Every check here would pass a gradient computed one way and fail it another:
a partial sum rounded per shard, a kernel cotangent rounded to bf16, a
tangent that dropped a representable 2^-30 residue. The Adam checks compare
the exchange dispatch with the global one, whose gradients the oracle checks.
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import optax
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P
from scipy.special import erfc

from dew.nn.moe import ExpertMLP, exact_gelu, expert_projection
from dew.nn.gpt_oss import GptOssExperts
from dew.training import MeshSpec, build_mesh

# The multi-device layouts need the eight simulated CPU devices conftest
# configures; a 1x1 mesh runs on the GPU lane's single device, where the
# contract meets cuBLAS.
MESH_LAYOUTS = tuple(pytest.param(expert, fsdp, marks=pytest.mark.mesh)
                     for expert, fsdp in ((2, 4), (4, 2), (8, 1)))
LAYOUTS = (pytest.param(1, 1, id='1-1'), *MESH_LAYOUTS)
TOLERANCE = 3e-5


def rounded(value, dtype=ml_dtypes.bfloat16) -> np.ndarray:
    return np.asarray(value).astype(dtype).astype(np.float64)


def grouped(a, b) -> np.ndarray:
    """`a @ b` per expert for `a` of three rows per expert."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a.reshape(8, 3, a.shape[-1]) @ b).reshape(24, -1)


def gathered(a, b) -> np.ndarray:
    """`a.T @ b` per expert, the kernel cotangent."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return a.reshape(8, 3, a.shape[-1]).swapaxes(1, 2) @ b.reshape(8, 3, b.shape[-1])


def shardings(mesh, rows: bool):
    kernels = NamedSharding(mesh, P('expert', 'fsdp', None) if rows else P('expert', None, 'fsdp'))
    inputs = NamedSharding(mesh, P('expert', 'fsdp') if rows else P('expert'))
    outputs = NamedSharding(mesh, P('expert') if rows else P('expert', 'fsdp'))
    return kernels, inputs, outputs


@pytest.fixture(scope='module')
def x64():
    with jax.enable_x64():
        yield


ROUND_ONCE_FORWARD_KERNEL = np.zeros((8, 16, 8), np.float32)
ROUND_ONCE_FORWARD_KERNEL[:, 0, :] = 1
ROUND_ONCE_FORWARD_KERNEL[:, (1, 8), :] = 2**-8
ROUND_ONCE_INPUT_KERNEL = np.zeros((8, 8, 16), np.float32)
ROUND_ONCE_INPUT_KERNEL[:, :, 0] = 1
ROUND_ONCE_INPUT_KERNEL[:, :, (1, 8)] = 2**-8


def cases():
    rng = np.random.default_rng(271)
    yield 'random', rng.normal(size=(24, 16)).astype(np.float32), \
        rng.normal(size=(8, 16, 16)).astype(np.float32), rng.normal(size=(24, 16)).astype(np.float32)
    # 1 + 2^-8 + 2^-8 rounds to 1 in bf16 if either half is rounded first
    # and to 1 + 2^-7 when the whole width is summed before rounding.
    yield 'forward-round-once', np.ones((24, 16), np.float32), ROUND_ONCE_FORWARD_KERNEL, \
        np.ones((24, 8), np.float32)
    yield 'input-gradient-round-once', np.ones((24, 8), np.float32), ROUND_ONCE_INPUT_KERNEL, \
        np.ones((24, 16), np.float32)


@pytest.mark.usefixtures('x64')
@pytest.mark.parametrize('case,master,input_dtype', [
    ('random', jnp.float32, jnp.bfloat16), ('random', jnp.float64, jnp.bfloat16),
    ('random', jnp.float32, jnp.float32), ('random', jnp.float64, jnp.float64),
    ('forward-round-once', jnp.float32, jnp.bfloat16),
    ('input-gradient-round-once', jnp.float32, jnp.bfloat16)])
@pytest.mark.parametrize('expert,fsdp', LAYOUTS)
@pytest.mark.parametrize('rows', [True, False], ids=['rows', 'columns'])
def test_a_projection_rounds_whole_contractions_once(case, master, input_dtype, expert, fsdp, rows):
    """Forward, kernel cotangent and input cotangent against float64 sums of
    the bf16 operands, with the kernel split on either of its dimensions and
    the gradient returned placed or replicated."""
    x, kernel, dy = next(values for name, *values in cases() if name == case)
    sizes = np.full(8, 3, np.int32)
    forward = rounded(grouped(rounded(x), rounded(kernel)))
    kernel_oracle = gathered(rounded(x), rounded(dy))
    input_oracle = rounded(grouped(rounded(dy), rounded(kernel).swapaxes(1, 2)), input_dtype)

    def loss(kernel, x, dy, sizes):
        projected = jnp.asarray(expert_projection(x, kernel, sizes, jnp.bfloat16, 'xla', None))
        return jnp.sum(projected.astype(jnp.float32) * dy), projected

    mesh = build_mesh(MeshSpec(expert=expert, fsdp=fsdp))
    kernels, inputs, outputs = shardings(mesh, rows)
    replicated = NamedSharding(mesh, P())
    arrays = (jnp.asarray(kernel, master), jnp.asarray(x, input_dtype), jnp.asarray(dy), jnp.asarray(sizes))
    arguments = tuple(jax.device_put(value, spec) for value, spec in zip(
        arrays, (kernels, inputs, outputs, replicated), strict=True))
    for spec in (kernels, replicated):
        with jax.set_mesh(mesh):
            (_, y), (dk, dx) = jax.jit(jax.value_and_grad(loss, (0, 1), has_aux=True),
                                       out_shardings=((replicated, outputs), (spec, inputs)))(*arguments)
        assert dk.dtype == master and dx.dtype == input_dtype
        np.testing.assert_allclose(np.asarray(y, np.float64), forward, atol=TOLERANCE, rtol=TOLERANCE)
        np.testing.assert_allclose(dk, kernel_oracle, atol=TOLERANCE, rtol=TOLERANCE)
        np.testing.assert_allclose(np.asarray(dx, np.float64), input_oracle, atol=TOLERANCE, rtol=TOLERANCE)


@pytest.mark.usefixtures('x64')
@pytest.mark.parametrize('compute,input_dtype,master', [
    (jnp.bfloat16, jnp.bfloat16, jnp.float32), (jnp.bfloat16, jnp.float32, jnp.float64),
    (jnp.float32, jnp.float32, jnp.float32), (jnp.float64, jnp.float64, jnp.float64)])
@pytest.mark.parametrize('expert,fsdp', LAYOUTS)
@pytest.mark.parametrize('rows', [True, False], ids=['rows', 'columns'])
def test_a_projection_differentiates_the_same_law_in_every_direction(
        compute, input_dtype, master, expert, fsdp, rows):
    """JVP, VJP, forward-over-reverse and reverse-over-forward against the
    tangent law `dx @ Q(kernel) + Q(x) @ dkernel` in float64."""
    rng = np.random.default_rng(943)
    x = rng.normal(size=(24, 16)).astype(input_dtype)
    kernel = rng.normal(size=(8, 16, 16)).astype(master)
    dx = rng.normal(scale=.3, size=x.shape).astype(input_dtype)
    dkernel = rng.normal(scale=.3, size=kernel.shape).astype(master)
    cotangent = rng.normal(size=(24, 16))
    sizes = jnp.full(8, 3, jnp.int32)
    qx, qk, qc = (rounded(value, compute) for value in (x, kernel, cotangent))
    expected = (
        rounded(grouped(qx, qk), compute),
        rounded(grouped(dx, qk) + grouped(qx, dkernel), compute),
        (rounded(grouped(qc, qk.swapaxes(1, 2)), input_dtype), rounded(gathered(qx, qc), master)),
        (rounded(grouped(qc, np.asarray(dkernel).swapaxes(1, 2)), input_dtype),
         rounded(gathered(dx, qc), master)))

    def evaluate(x, kernel, dx, dkernel, cotangent, sizes):
        def fn(x, kernel):
            return jnp.asarray(expert_projection(x, kernel, sizes, compute, 'xla', None))

        def scalar(x, kernel):
            return jnp.sum(fn(x, kernel).astype(jnp.float64) * cotangent)

        def directional(x, kernel):
            return jnp.sum(jax.jvp(fn, (x, kernel), (dx, dkernel))[1].astype(jnp.float64) * cotangent)

        y, tangent = jax.jvp(fn, (x, kernel), (dx, dkernel))
        return (y, tangent, jax.grad(scalar, (0, 1))(x, kernel),
                jax.jvp(jax.grad(scalar, (0, 1)), (x, kernel), (dx, dkernel))[1],
                jax.grad(directional, (0, 1))(x, kernel))

    mesh = build_mesh(MeshSpec(expert=expert, fsdp=fsdp))
    kernels, inputs, outputs = shardings(mesh, rows)
    specs = (inputs, kernels, inputs, kernels, outputs, NamedSharding(mesh, P()))
    arguments = tuple(jax.device_put(jnp.asarray(value), spec) for value, spec in zip(
        (x, kernel, dx, dkernel, cotangent, sizes), specs, strict=True))
    with jax.set_mesh(mesh):
        actual = jax.jit(evaluate, out_shardings=(
            outputs, outputs, (inputs, kernels), (inputs, kernels), (inputs, kernels)))(*arguments)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected + expected[-1:]), strict=True):
        np.testing.assert_allclose(np.asarray(a, np.float64), b, atol=TOLERANCE, rtol=TOLERANCE)


@pytest.mark.usefixtures('x64')
def test_a_tangent_keeps_the_residue_of_a_wider_operand():
    """A 2^-30 that only fp64 holds survives the projection's derivatives in
    every mode: implicit promotion, explicit fp64 compute, and both mixed
    orders under bf16 compute. The residue is representable, so these are
    exact; an fp32 tolerance would accept the zero of a rounded contraction."""
    epsilon = 2.0**-30
    x = jnp.asarray([[1 + epsilon], [-1]], jnp.float64)
    kernel = jnp.ones((1, 1, 1), jnp.float32)
    sizes = jnp.asarray([2], jnp.int32)

    def inferred(x, kernel):
        return jnp.asarray(expert_projection(x, kernel, sizes, None, 'xla', None))

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
        return jnp.asarray(expert_projection(x, kernel, sizes, jnp.float64, 'xla', None))

    y, tangent = jax.jit(lambda x, kernel: jax.jvp(
        explicit, (x, kernel), (jnp.ones_like(x), jnp.zeros_like(kernel))))(x, kernel)
    dx = jax.jit(jax.grad(lambda x: explicit(x, kernel).sum()))(x)
    np.testing.assert_array_equal(y, np.asarray(kernel[0]))
    np.testing.assert_array_equal(tangent, np.asarray(kernel[0]))
    np.testing.assert_array_equal(dx, np.full(x.shape, epsilon, np.float32))

    kernel = jnp.asarray([[[1., -1.]]], jnp.float64)
    direction = jnp.asarray([[[1 + epsilon, -1]]], jnp.float64)

    def scalar(x, kernel):
        return jnp.asarray(expert_projection(
            x, kernel, sizes, jnp.bfloat16, 'xla', None)).astype(jnp.float64).sum()

    def mixed(x, kernel, direction):
        forward_reverse = jax.jvp(jax.grad(scalar, (0, 1)), (x, kernel),
                                  (jnp.zeros_like(x), direction))[1]

        def directional(x, kernel):
            return jax.jvp(scalar, (x, kernel), (jnp.zeros_like(x), direction))[1]

        return forward_reverse, jax.grad(directional, (0, 1))(x, kernel)

    for dx, dk in jax.jit(mixed)(x, kernel, direction):
        np.testing.assert_array_equal(dx, np.full(x.shape, epsilon, np.float32))
        np.testing.assert_array_equal(dk, np.zeros(kernel.shape, np.float64))


@pytest.mark.usefixtures('x64')
def test_gpt_oss_infers_one_compute_dtype_from_all_expert_operands():
    x = jnp.asarray([[2**24, 1, -2**24]], jnp.float32)
    variables = {'params': {
        'gate_up_proj': jnp.asarray([[[1, 0], [1, 0], [1, 0]]], jnp.float32),
        'gate_up_proj_bias': jnp.zeros((1, 2), jnp.float32),
        'down_proj': jnp.ones((1, 1, 3), jnp.float64),
        'down_proj_bias': jnp.zeros((1, 3), jnp.float32)}}
    model = GptOssExperts(3, 1, 1)
    actual = jax.jit(model.apply)(variables, x, jnp.ones((1, 1), jnp.float32),
                                 jnp.zeros((1, 1), jnp.int32))
    # The fp64 down matrix promotes the gate too: 2^24 + 1 - 2^24 is 1,
    # so the gated intermediate is sigmoid(1.702), not the fp32 zero.
    expected = np.full((1, 3), 1 / (1 + np.exp(-1.702)), np.float64)
    assert actual.dtype == jnp.float64
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)



@pytest.mark.usefixtures('x64')
@pytest.mark.parametrize('dtype', [jnp.bfloat16, jnp.float32, jnp.float64])
def test_the_exact_gelu_rounds_once_in_every_differentiation_mode(dtype):
    """Value, JVP, VJP and both mixed second derivatives against SciPy's
    erfc, each rounded once to the activation dtype. Observed: bf16 exact,
    fp32 within 4.8e-7, fp64 within 4.5e-16."""
    rng = np.random.default_rng(1041)
    values, cotangents, directions = rng.normal(size=(3, 1024))
    x, ct, dx = ((values * 3).astype(dtype), cotangents.astype(dtype), (directions * .2).astype(dtype))
    v, c, t = (array.astype(np.float64) for array in (x, ct, dx))
    density = np.exp(-v * v / 2) / np.sqrt(2 * np.pi)
    first = .5 * erfc(-v / np.sqrt(2)) + v * density
    second = (2 - v * v) * density
    expected = [rounded(array, dtype) for array in (
        .5 * v * erfc(-v / np.sqrt(2)), first * t, first * c, second * c * t, second * t * c)]

    def evaluate(x, dx, ct):
        def scalar(x):
            return jnp.sum(exact_gelu(x).astype(jnp.float64) * ct.astype(jnp.float64))

        def directional(x):
            return jnp.sum(jax.jvp(exact_gelu, (x,), (dx,))[1].astype(jnp.float64) * ct.astype(jnp.float64))

        y, tangent = jax.jvp(exact_gelu, (x,), (dx,))
        return (y, tangent, jax.grad(scalar)(x), jax.jvp(jax.grad(scalar), (x,), (dx,))[1],
                jax.grad(directional)(x))

    actual = jax.jit(evaluate)(jnp.asarray(x), jnp.asarray(dx), jnp.asarray(ct))
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(np.asarray(a, np.float64), b, atol=TOLERANCE, rtol=TOLERANCE)


def placed_experts(mesh, activation, skewed, scale_inputs, limit):
    rng = np.random.default_rng(341)
    model = ExpertMLP(8, 16, 8, dtype=jnp.bfloat16, activation=activation,
                      scale_inputs=scale_inputs, swiglu_limit=limit)
    x = jnp.asarray(rounded(rng.normal(size=(24, 8))), jnp.bfloat16)
    weights = jnp.asarray(rng.uniform(.1, .9, size=(24, 2)), jnp.float32)
    choices = (np.tile([0, 1], (24, 1)) if skewed
               else np.argsort(rng.normal(size=(24, 8)), axis=-1)[:, :2]).astype(np.int32)
    parameters = model.init(jax.random.key(19), x, weights, choices)
    columns = NamedSharding(mesh, P('expert', None, 'fsdp'))
    specs = {'params': {'gate_proj': {'kernel': columns}, 'up_proj': {'kernel': columns},
                        'down_proj': {'kernel': NamedSharding(mesh, P('expert', 'fsdp'))}}}
    tokens = NamedSharding(mesh, P(('expert', 'fsdp')))
    parameters = jax.device_put(parameters, specs)
    x, weights, choices = (jax.device_put(value, tokens) for value in (x, weights, jnp.asarray(choices)))
    return model, parameters, specs, tokens, x, weights, choices


@pytest.mark.parametrize('activation', ['swiglu', 'geglu', 'geglu_exact'])
@pytest.mark.parametrize('skewed,scale_inputs,limit', [(False, False, None), (True, True, .7)])
@pytest.mark.parametrize('expert,fsdp', MESH_LAYOUTS)
def test_both_dispatches_take_the_same_adam_steps_in_bf16(activation, skewed, scale_inputs, limit,
                                                          expert, fsdp):
    """Three Adam steps of a bf16 routed layer: outputs, gradients, parameters
    and moments agree between global and exchange dispatch. Observed maxima
    1.5e-8 random and 6e-8 skewed."""
    mesh = build_mesh(MeshSpec(expert=expert, fsdp=fsdp))
    model, parameters, specs, tokens, x, weights, choices = placed_experts(
        mesh, activation, skewed, scale_inputs, limit)
    optimizer = optax.adam(1e-3)
    state = optimizer.init(parameters)
    state_specs = jax.tree.map(
        lambda value: NamedSharding(mesh, P()) if value.ndim == 0 else value.sharding, state)
    outputs = ((NamedSharding(mesh, P()), tokens), (specs, tokens, tokens), specs, state_specs)
    trajectories = []
    for dispatch in ('global', 'exchange'):
        network = model.clone(dispatch=dispatch)

        def objective(parameters, x, weights, choices):
            y = jnp.asarray(network.apply(parameters, x, weights, choices))
            return jnp.mean(jnp.sin(y.astype(jnp.float32))), y

        def step(parameters, state, x, weights, choices):
            result, gradients = jax.value_and_grad(objective, (0, 1, 2), has_aux=True)(
                parameters, x, weights, choices)
            updates, state = optimizer.update(gradients[0], state, parameters)
            return result, gradients, optax.apply_updates(parameters, updates), state

        with jax.set_mesh(mesh):
            compiled = jax.jit(step, out_shardings=outputs)
            trajectory, (p, s) = [], (parameters, state)
            for _ in range(3):
                result = compiled(p, s, x, weights, choices)
                trajectory.append(result)
                p, s = result[2:]
        trajectories.append(trajectory)
    for first, second in zip(*trajectories, strict=True):
        for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
            np.testing.assert_allclose(np.asarray(a, np.float64), np.asarray(b, np.float64), atol=TOLERANCE, rtol=TOLERANCE)


@pytest.mark.mesh
@pytest.mark.parametrize('activation', ['swiglu', 'geglu', 'geglu_exact'])
def test_both_dispatches_carry_the_same_tangents(activation):
    """A JVP through the whole routed layer, parameters, tokens and router
    weights all perturbed, agrees exactly between the dispatches."""
    mesh = build_mesh(MeshSpec(expert=4, fsdp=2))
    model, parameters, specs, tokens, x, weights, choices = placed_experts(
        mesh, activation, False, False, None)
    rng = np.random.default_rng(523)
    dp = jax.device_put(jax.tree.map(
        lambda value: jnp.asarray(rng.normal(scale=.05, size=value.shape), value.dtype), parameters), specs)
    dx = jax.device_put(jnp.asarray(rng.normal(scale=.05, size=x.shape), x.dtype), tokens)
    dw = jax.device_put(jnp.asarray(rng.normal(scale=.05, size=weights.shape), weights.dtype), tokens)
    results = []
    for dispatch in ('global', 'exchange'):
        network = model.clone(dispatch=dispatch)

        def run(parameters, x, weights, dp, dx, dw, choices):
            return jax.jvp(lambda parameters, x, weights: jnp.asarray(
                network.apply(parameters, x, weights, choices)), (parameters, x, weights), (dp, dx, dw))

        with jax.set_mesh(mesh):
            results.append(jax.jit(run, out_shardings=(tokens, tokens))(
                parameters, x, weights, dp, dx, dw, choices))
    for a, b in zip(*results, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
