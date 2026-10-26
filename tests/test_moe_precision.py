"""The precision contract routed experts train under, against independent
float64 arithmetic on the rounded operands.

Every check here would pass a gradient computed one way and fail it another:
a partial sum rounded per shard, a kernel cotangent rounded to bf16, a
tangent that dropped a representable 2^-30 residue. The Adam checks compare
the exchange dispatch with the global one, whose gradients the oracle checks.
"""

import types

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import optax
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P
from reference_error import assert_as_exact, with_gradients
from scipy.special import erfc
from tuned import each_config, taken

from dew.nn.activations import exact_gelu
from dew.nn.gpt_oss import GptOssExperts
from dew.nn.kernels import generation, grouped_matmul, tuning
from dew.nn.kernels.generation import chosen as kernel_choices, gpu_target, routes, triton_runs
from dew.nn.moe import ExpertMLP, expert_dispatch, expert_projection
from dew.training import MeshSpec
from tools import grouped_oracle as oracle

pytestmark = pytest.mark.kernel

# The multi-device layouts need the eight simulated CPU devices conftest
# configures; a 1x1 mesh runs on the GPU lane's single device, where the
# contract meets cuBLAS.
MESH_LAYOUTS = tuple(pytest.param(expert, fsdp, marks=pytest.mark.mesh)
                     for expert, fsdp in ((2, 4), (4, 2), (8, 1)))
LAYOUTS = (pytest.param(1, 1, id='1-1'), *MESH_LAYOUTS)
TOLERANCE = 3e-5
# A mesh whose data axis splits the tokens as well as the expert axis.
DATA_LAYOUT = pytest.param(2, 2, marks=pytest.mark.mesh, id='2-2-data2')


def rounded(value, dtype=ml_dtypes.bfloat16) -> np.ndarray:
    return np.asarray(value).astype(dtype).astype(np.float64)


THREE_EACH = np.full(8, 3)
"""Three rows for each of eight experts, the projections' equal routing."""


def shardings(mesh, rows: bool):
    kernels = NamedSharding(mesh, P('expert', 'fsdp', None) if rows else P('expert', None, 'fsdp'))
    inputs = NamedSharding(mesh, P('expert', 'fsdp') if rows else P('expert'))
    outputs = NamedSharding(mesh, P('expert') if rows else P('expert', 'fsdp'))
    return kernels, inputs, outputs


@pytest.fixture
def x64():
    # XLA's TPU rewrites float64 into pairs of float32 op by op, and has no
    # rewrite for a ragged dot ("While rewriting computation to not contain
    # X64 element types ... ragged-dot"); the fp64 oracle cases run on CPU
    # and GPU.
    if jax.default_backend() == "tpu":
        pytest.skip("XLA's TPU has no float64 ragged dot")
    with jax.enable_x64():
        yield


@pytest.fixture(autouse=True)
def float64_needs_a_backend_that_has_it(request):
    """A float64 case computes in float64 on the default device, which a TPU
    does not have ("X64 element types" at compile); it runs on CPU and GPU."""
    params = getattr(getattr(request.node, "callspec", None), "params", {})
    if jax.default_backend() == "tpu" and any(value is jnp.float64 for value in params.values()):
        pytest.skip("a TPU has no float64 arithmetic")


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
    forward = rounded(oracle.forward(rounded(x), rounded(kernel), THREE_EACH))
    kernel_oracle = oracle.kernel_gradient(rounded(x), rounded(dy), THREE_EACH)
    input_oracle = rounded(oracle.input_gradient(rounded(dy), rounded(kernel), THREE_EACH), input_dtype)

    def loss(kernel, x, dy, sizes):
        projected = jnp.asarray(expert_projection(x, kernel, sizes, jnp.bfloat16, 'xla', None))
        return jnp.sum(projected.astype(jnp.float32) * dy), projected

    mesh = MeshSpec(expert=expert, fsdp=fsdp).build()
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


@pytest.mark.parametrize('sizes', [
    np.full(8, 3), np.array([0, 5, 0, 0, 12, 1, 0, 3]), np.array([0, 0, 0, 20, 0, 0, 0, 0]),
    np.array([0, 3, 3, 3, 3, 3, 3, 3]), np.zeros(8)],
    ids=['even', 'ragged', 'one-expert', 'fewer-rows', 'all-empty'])
@pytest.mark.parametrize('input_dtype', [jnp.bfloat16, jnp.float32])
def test_the_pallas_kernels_hold_the_contract_on_one_device(sizes, input_dtype):
    """Forward, input and kernel cotangents of 'pallas' against float64
    sums of the bf16 operands, on the kernels (interpreted on a CPU): rows
    past the routed ones, and every row when no expert is routed, come back
    zero in the output and the input gradient."""
    rng = np.random.default_rng(88)
    x = rng.normal(size=(24, 16)).astype(np.float32)
    kernel = rng.normal(size=(8, 16, 16)).astype(np.float32)
    dy = rng.normal(size=(24, 16))
    sizes = sizes.astype(np.int32)
    qx, qk = rounded(x), rounded(kernel)
    qy = rounded(dy)
    forward, input_oracle, kernel_oracle = oracle.products(qx, qk, qy, sizes)

    def loss(x, kernel):
        projected = jnp.asarray(expert_projection(x, kernel, jnp.asarray(sizes), jnp.bfloat16,
                                                  'pallas', None))
        return jnp.sum(projected.astype(jnp.float32) * jnp.asarray(qy, jnp.float32)), projected

    (_, y), (dx, dk) = jax.jit(jax.value_and_grad(loss, (0, 1), has_aux=True))(
        jnp.asarray(x, input_dtype), jnp.asarray(kernel))
    assert dx.dtype == input_dtype and dk.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(y, np.float64), rounded(forward), atol=TOLERANCE,
                               rtol=TOLERANCE)
    np.testing.assert_allclose(np.asarray(dx, np.float64), rounded(input_oracle, input_dtype),
                               atol=TOLERANCE, rtol=TOLERANCE)
    np.testing.assert_allclose(np.asarray(dk, np.float64), kernel_oracle, atol=TOLERANCE,
                               rtol=TOLERANCE)


HOPPER = pytest.mark.skipif(not (triton_runs() and gpu_target().wgmma),
                            reason="Mosaic GPU's grouped launches need Hopper's warpgroup MMA")


@pytest.mark.gpu(reason="the tuner's space is a GPU's")
@pytest.mark.skipif(not triton_runs(), reason="needs a GPU the Pallas kernels compile for")
@pytest.mark.parametrize("kernel,width,features", [
    (grouped_matmul.Gmm, 192, 320), (grouped_matmul.Tgmm, 192, 320),
    pytest.param(grouped_matmul.MosaicGmm, 768, 768, marks=HOPPER),
    pytest.param(grouped_matmul.MosaicTgmm, 768, 768, marks=HOPPER)])
def test_every_tile_of_the_grouped_kernels_holds_the_float64_gate(kernel, width, features):
    """Each config a launch may take (`each_config`: the tile it derives and
    the variants an opt-in tuning run times), of the implementation the
    launch is forced to, over groups ragged against every tile size, one
    empty and one of a single row: the output and both gradients of a bf16
    projection over an fp32 master hold check 1 against XLA's ragged dot,
    from float64 sums of the same bf16 operands, over the site's draws.
    Triton's widths are ragged against its tiles, which mask them; Mosaic's
    divide by every tile width it derives (64 to 256, and a pair's 256
    rows), which it needs."""
    gate = _tile_gate(kernel, width, features)
    with kernel_choices(grouped_matmul=kernel.name):
        configs = each_config(kernel, _tile_gate_arm('pallas'), *gate.draw(0))
    assert configs
    for config in configs:
        gate.check(config)


def test_the_tile_gate_fails_a_launch_off_by_a_rounding():
    """The tile gate on the Triton launch, interpreted on the CPU as an A100's, passes it and fails it
    with its forward off by one part in 2^7, an error no rounding of its sums makes: the reference arm is
    XLA's, not the launch under test."""
    with generation.assumed("sm80"):
        gate = _tile_gate(grouped_matmul.Gmm, 64, 128)
        with kernel_choices(grouped_matmul='triton'):
            config, *_ = taken(grouped_matmul.Gmm, _tile_gate_arm('pallas'), *gate.draw(0))
        gate.check(config)
        with pytest.raises(AssertionError):
            gate.check(config, fault=1 + 2.0 ** -7)


TILE_SIZES = np.array([0, 70, 1, 129, 100, 0], np.int32)
"""The tile gate's groups: ragged against every tile, one empty, one of a single row."""


def _tile_gate_arm(implementation, fault=1.0):
    """A bf16 projection's output and both gradients through `implementation` ('pallas' with the CPU's
    interpreter), its output times `fault`."""
    def run(x, kernel_matrix, dy):
        y, *gradients = with_gradients(lambda x, k: expert_projection(
            x, k, jnp.asarray(TILE_SIZES), jnp.bfloat16, implementation, None))(x, kernel_matrix, dy)
        return ((y.astype(jnp.float32) * fault).astype(y.dtype), *gradients)
    return run


def _tile_gate(kernel, width, features):
    """The tile gate of `kernel`'s launch at one shape: its draws, and `check(config, fault)`, check 1 of
    the launch at `config` against XLA's ragged dot, which runs under its own choice (its route checked
    once) so the forced kernel never reaches it."""
    rows = int(TILE_SIZES.sum())

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(rows, width)), jnp.bfloat16),
                jnp.asarray(rng.normal(size=(len(TILE_SIZES), width, features)) / np.sqrt(width),
                            jnp.float32),
                jnp.asarray(rng.normal(size=(rows, features)), jnp.bfloat16))

    def truth(x, kernel_matrix, dy):
        # The products of the operands as the projection rounds them, in float64.
        return oracle.products(x, rounded(kernel_matrix), dy, TILE_SIZES)

    xla = jax.jit(_tile_gate_arm('xla'))
    with kernel_choices(grouped_matmul='xla'), routes() as taken:
        xla(*draw(0))
    assert {route.kernel for route in taken if route.choice == 'grouped_matmul'} == {'xla'}

    def reference(*inputs):
        with kernel_choices(grouped_matmul='xla'):
            return xla(*inputs)

    def check(config, fault=1.0):
        with kernel_choices(grouped_matmul=kernel.name), tuning.pinned({kernel: config}):
            assert_as_exact(jax.jit(_tile_gate_arm('pallas', fault)), reference, draw,
                            f"{kernel.__name__} at {config}", seeds=TILE_SEEDS, truth_fn=truth)

    return types.SimpleNamespace(draw=draw, check=check)


TILE_SEEDS = 8
"""The tile gate's declared draws (`reference_error.seeds_for` of XLA's ragged dot on them)."""


@pytest.mark.gpu(reason="needs a GPU the Pallas kernels compile for")
@pytest.mark.skipif(not triton_runs(), reason="needs a GPU the Pallas kernels compile for")
@pytest.mark.parametrize('implementation', ['auto', 'pallas'])
def test_a_projection_on_the_cpu_of_a_gpu_host_runs_without_the_triton_kernels(implementation):
    """The kernels are chosen where the call lowers, not by the process's
    default backend. On a GPU host, a projection placed on the CPU runs
    `jax.lax.ragged_dot` ('auto') or the interpreted kernels ('pallas'), and
    its forward is the float64 sum of the bf16 operands either way."""
    rng = np.random.default_rng(88)
    x = rng.normal(size=(24, 16)).astype(np.float32)
    kernel = rng.normal(size=(8, 16, 16)).astype(np.float32)
    cpu = jax.devices('cpu')[0]
    project = jax.jit(lambda x, kernel, sizes: expert_projection(
        x, kernel, sizes, jnp.bfloat16, implementation, None))

    y = project(*jax.device_put((x, kernel, np.full(8, 3, np.int32)), cpu))

    assert y.devices() == {cpu}
    np.testing.assert_allclose(np.asarray(y, np.float64),
                               rounded(oracle.forward(rounded(x), rounded(kernel), THREE_EACH)),
                               atol=TOLERANCE, rtol=TOLERANCE)


@pytest.mark.parametrize('sizes', [np.full(8, 3), np.array([0, 5, 0, 0, 12, 1, 0, 3])],
                         ids=['even', 'ragged'])
def test_a_bf16_projection_at_the_highest_precision_holds_the_contract(sizes):
    """bf16 operands multiply exactly into fp32 sums at any precision, so
    'highest' asks for the product the default gives: forward and both
    cotangents are float64 sums of the rounded operands, on every backend.
    XLA's TPU ragged-dot kernel refuses bf16 operands at fp32 contract
    precision (Mosaic "Bad lhs type" on a v6e, jax 0.11.2), which any run
    that sets jax_default_matmul_precision to 'highest' reaches."""
    rng = np.random.default_rng(89)
    x = rounded(rng.normal(size=(24, 16)))
    kernel = rounded(rng.normal(size=(8, 16, 16)))
    dy = rounded(rng.normal(size=(24, 16)))
    sizes = sizes.astype(np.int32)
    # Rows past the routed ones belong to no expert and come back zero.
    forward, input_oracle, kernel_oracle = oracle.products(x, kernel, dy, sizes)

    def loss(x, kernel):
        projected = expert_projection(x, kernel, jnp.asarray(sizes), jnp.bfloat16, 'xla', 'highest')
        return jnp.sum(projected.astype(jnp.float32) * jnp.asarray(dy, jnp.float32)), projected

    # A float32 master kernel, as training keeps it: the projection rounds it
    # to bf16 for the product, and its gradient comes back in float32.
    (_, y), (dx, dk) = jax.jit(jax.value_and_grad(loss, (0, 1), has_aux=True))(
        jnp.asarray(x, jnp.bfloat16), jnp.asarray(kernel, jnp.float32))
    assert dx.dtype == jnp.bfloat16 and dk.dtype == jnp.float32
    for got, want in ((y, rounded(forward)), (dx, rounded(input_oracle)), (dk, kernel_oracle)):
        np.testing.assert_allclose(np.asarray(got, np.float64), want, atol=TOLERANCE, rtol=TOLERANCE)


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
        rounded(oracle.forward(qx, qk, THREE_EACH), compute),
        rounded(oracle.forward(dx, qk, THREE_EACH) + oracle.forward(qx, dkernel, THREE_EACH), compute),
        (rounded(oracle.input_gradient(qc, qk, THREE_EACH), input_dtype),
         rounded(oracle.kernel_gradient(qx, qc, THREE_EACH), master)),
        (rounded(oracle.input_gradient(qc, dkernel, THREE_EACH), input_dtype),
         rounded(oracle.kernel_gradient(dx, qc, THREE_EACH), master)))

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

    mesh = MeshSpec(expert=expert, fsdp=fsdp).build()
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
@pytest.mark.parametrize('implementation', ['xla', 'pallas'])
def test_a_tangent_keeps_the_residue_of_a_wider_operand(implementation):
    """A 2^-30 that only fp64 holds survives the projection's derivatives in
    every mode: implicit promotion, explicit fp64 compute, and both mixed
    orders under bf16 compute. The residue is representable, so these are
    exact; an fp32 tolerance would accept the zero of a rounded contraction."""
    epsilon = 2.0**-30
    x = jnp.asarray([[1 + epsilon], [-1]], jnp.float64)
    kernel = jnp.ones((1, 1, 1), jnp.float32)
    sizes = jnp.asarray([2], jnp.int32)

    def inferred(x, kernel):
        return jnp.asarray(expert_projection(x, kernel, sizes, None, implementation, None))

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
        return jnp.asarray(expert_projection(x, kernel, sizes, jnp.float64, implementation, None))

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
            x, kernel, sizes, jnp.bfloat16, implementation, None)).astype(jnp.float64).sum()

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


def placed_experts(mesh, activation, skewed, scale_inputs, limit, implementation='xla'):
    rng = np.random.default_rng(341)
    model = ExpertMLP(8, 16, 8, dtype=jnp.bfloat16, activation=activation,
                      scale_inputs=scale_inputs, swiglu_limit=limit,
                      implementation=implementation)
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


def test_a_bf16_combine_weighs_its_experts_in_fp32_and_rounds_once():
    """Two experts answer 1 and -1, weighted 1/2 + 2^-11 and 1/2 - 2^-11. In
    fp32 the weighted sum is 2^-10, which bf16 holds; bf16 weights are both
    1/2 and cancel to 0. The layer's bf16 output is the float64 dot of its
    slots and weights, rounded once to bf16."""
    layer = ExpertMLP(num_experts=2, hidden_features=1, out_features=1, activation="relu2",
                      implementation="xla", dtype=jnp.bfloat16, precision="highest")
    x = jnp.ones((1, 1), jnp.float32)
    weights = jnp.asarray([[0.5 + 2 ** -11, 0.5 - 2 ** -11]], jnp.float32)
    indices = jnp.asarray([[0, 1]], jnp.int32)
    params = {"up_proj": {"kernel": jnp.ones((2, 1, 1), jnp.float32)},
              "down_proj": {"kernel": jnp.asarray([[[1.0]], [[-1.0]]], jnp.float32)}}
    initialized = layer.init(jax.random.key(0), x, weights, indices)["params"]
    assert jax.tree.map(jnp.shape, initialized) == jax.tree.map(jnp.shape, params)
    expected = np.asarray(np.dot([1.0, -1.0], np.asarray(weights[0], np.float64)), ml_dtypes.bfloat16)
    for apply in (layer.apply, jax.jit(layer.apply)):
        combined = apply({"params": params}, x, weights, indices)
        assert combined.dtype == jnp.bfloat16
        assert float(combined[0, 0]) == float(expected) == 2.0 ** -10


@pytest.mark.parametrize('activation', ['swiglu', 'geglu', 'geglu_exact'])
@pytest.mark.parametrize('skewed,scale_inputs,limit', [(False, False, None), (True, True, .7)])
@pytest.mark.parametrize('expert,fsdp', (*MESH_LAYOUTS, DATA_LAYOUT))
@pytest.mark.parametrize('implementation', ['xla', 'pallas'])
def test_both_dispatches_take_the_same_adam_steps_in_bf16(activation, skewed, scale_inputs, limit,
                                                          expert, fsdp, implementation):
    """Three Adam steps of a bf16 routed layer: outputs, gradients, parameters
    and moments agree between global and exchange dispatch. Observed maxima
    1.5e-8 random and 6e-8 skewed. 'pallas' runs its kernels (interpreted on
    a CPU) on every device's own rows, the exchange included on a mesh whose
    only split axis is the expert one."""
    mesh = MeshSpec(expert=expert, fsdp=fsdp).build()
    model, parameters, specs, tokens, x, weights, choices = placed_experts(
        mesh, activation, skewed, scale_inputs, limit, implementation)
    optimizer = optax.adam(1e-3)
    state = optimizer.init(parameters)
    state_specs = jax.tree.map(
        lambda value: NamedSharding(mesh, P()) if value.ndim == 0 else value.sharding, state)
    outputs = ((NamedSharding(mesh, P()), tokens), (specs, tokens, tokens), specs, state_specs)
    trajectories = []
    for dispatch in ('global', 'exchange'):
        network = model.clone(dispatch=dispatch)

        def objective(parameters, x, weights, choices, *, network=network):
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
            np.testing.assert_allclose(
                np.asarray(a, np.float64), np.asarray(b, np.float64), atol=TOLERANCE, rtol=TOLERANCE
            )


@pytest.mark.mesh
@pytest.mark.parametrize('activation', ['swiglu', 'geglu', 'geglu_exact'])
def test_both_dispatches_carry_the_same_tangents(activation):
    """A JVP through the whole routed layer, parameters, tokens and router
    weights all perturbed, agrees exactly between the dispatches."""
    mesh = MeshSpec(expert=4, fsdp=2).build()
    model, parameters, specs, tokens, x, weights, choices = placed_experts(
        mesh, activation, skewed=False, scale_inputs=False, limit=None)
    rng = np.random.default_rng(523)
    dp = jax.device_put(jax.tree.map(
        lambda value: jnp.asarray(rng.normal(scale=.05, size=value.shape), value.dtype), parameters), specs)
    dx = jax.device_put(jnp.asarray(rng.normal(scale=.05, size=x.shape), x.dtype), tokens)
    dw = jax.device_put(jnp.asarray(rng.normal(scale=.05, size=weights.shape), weights.dtype), tokens)
    results = []
    for dispatch in ('global', 'exchange'):
        network = model.clone(dispatch=dispatch)

        def run(parameters, x, weights, dp, dx, dw, choices, *, network=network):
            return jax.jvp(lambda parameters, x, weights, network=network: jnp.asarray(
                network.apply(parameters, x, weights, choices)), (parameters, x, weights), (dp, dx, dw))

        with jax.set_mesh(mesh):
            results.append(jax.jit(run, out_shardings=(tokens, tokens))(
                parameters, x, weights, dp, dx, dw, choices))
    for a, b in zip(*results, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def exchange_residue_case():
    """Sixteen tokens all routed to expert 0, in values whose every product and
    partial sum is exact in fp32: the kernel gradient is exact before its one
    rounding. Two cotangent columns cancel to 2^-8, which a bf16 partial sum
    of 1 + 2^-8 rounds away (it ties to 1). In column 0 the -1 is token 1,
    which follows token 0 on its device, so a four-way exchange sends it a
    round later; in column 1 it is token 8, which four devices hold apart
    from token 0, so their partial sums meet across devices."""
    rng = np.random.default_rng(29)
    x = rng.integers(-2, 3, size=(16, 8)) * 2.0**-2
    x[:, 0] = 1
    kernel = rng.integers(-8, 9, size=(8, 8, 8)) * 2.0**-4
    dy = rng.integers(-40, 41, size=(16, 8)) * 2.0**-8
    dy[:, :2] = 0
    dy[0, 0], dy[1, 0], dy[2, 0] = 1, -1, 2.0**-8
    dy[0, 1], dy[2, 1], dy[8, 1] = 1, 2.0**-8, -1
    return x, kernel, dy


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize('spec', [MeshSpec(expert=4), MeshSpec(expert=2),
                                  MeshSpec(expert=2, fsdp=2)],
                         ids=['expert4', 'data2-expert2', 'expert2-fsdp2'])
@pytest.mark.parametrize('dispatch', ['global', 'exchange'])
def test_a_bf16_master_sums_its_expert_gradient_before_rounding(spec, dispatch):
    """A bf16 kernel's cotangent sums every device's rows and every exchange
    round in fp32 and rounds once, as one device rounds it: the gradient is
    the correctly rounded exact sum, bit for bit, on every layout. So are
    the output and the token gradient, each one rounding of an exact sum."""
    x, kernel, dy = exchange_residue_case()
    expected = (rounded(x @ kernel[0]), rounded(np.einsum('ti,to->io', x, dy)),
                rounded(dy @ kernel[0].T))
    mesh = spec.build(jax.devices()[:4])
    rows = NamedSharding(mesh, P(tuple(axis for axis in ('data', 'expert', 'fsdp')
                                       if mesh.shape[axis] > 1)))

    def project(tokens, sizes, _ids, kernel):
        return jnp.asarray(expert_projection(tokens, kernel, sizes, jnp.bfloat16, 'xla', None))

    def loss(kernel, x, dy, indices):
        out = expert_dispatch(project, x, indices, kernel, [('exp', 'embed', 'mlp')], num_experts=8,
                              dispatch=dispatch, output_dtype=jnp.bfloat16)[:, 0]
        return jnp.sum(out.astype(jnp.float32) * dy), out

    arguments = (jax.device_put(jnp.asarray(kernel, jnp.bfloat16), NamedSharding(mesh, P('expert'))),
                 *(jax.device_put(value, rows) for value in (
                     jnp.asarray(x, jnp.bfloat16), jnp.asarray(dy, jnp.float32),
                     jnp.zeros((16, 1), jnp.int32))))
    with jax.set_mesh(mesh):
        (_, out), (d_kernel, d_x) = jax.jit(jax.value_and_grad(loss, (0, 1), has_aux=True))(
            *arguments)
    assert d_kernel.dtype == jnp.bfloat16 and d_x.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(out, np.float64), expected[0])
    np.testing.assert_array_equal(np.asarray(d_kernel[0], np.float64), expected[1])
    np.testing.assert_array_equal(np.asarray(d_kernel[1:], np.float64), 0)
    np.testing.assert_array_equal(np.asarray(d_x, np.float64), expected[2])
