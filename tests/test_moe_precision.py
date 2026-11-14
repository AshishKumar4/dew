"""The precision contract routed experts train under, against independent
float64 arithmetic on the rounded operands.

Every check-1 site here declares the fewest seeds (`FEWEST_SEEDS`), each sized by
`seeds_for` on armada's CPUs (2026-11-14, at 9b89b74e338).

Every check here would pass a gradient computed one way and fail it another:
a partial sum rounded per shard, a kernel cotangent rounded to bf16, a
tangent that dropped a representable 2^-30 residue. The Adam checks compare
the exchange dispatch with the global one, whose gradients the oracle checks.
"""

import contextlib
import types

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import optax
import pytest
from flax import linen as nn
from generation_support import interpreted_projection
from jax.sharding import NamedSharding, PartitionSpec as P
from reference_error import (
    FEWEST_SEEDS,
    as_given,
    assert_as_exact,
    assert_computes_the_oracle,
    chain_roundings,
    with_gradients,
    without_ynn,
)
from scipy.special import erfc
from tuned import each_config, taken

from dew.nn.activations import exact_gelu, gated_product, swiglu_clamped
from dew.nn.gpt_oss import GptOssExperts
from dew.nn.kernels import generation, grouped_matmul, tuning
from dew.nn.kernels.generation import chosen as kernel_choices
from dew.nn.moe import EXPERT_AXES, ExpertMLP, Layered, expert_dispatch, expert_projection, paired_projections
from dew.training import MeshSpec
from tools import grouped_oracle as oracle

pytestmark = pytest.mark.kernel

# The multi-device layouts need the eight simulated CPU devices conftest
# configures; a 1x1 mesh runs on the GPU lane's single device, where the
# contract meets cuBLAS.
MESH_LAYOUTS = tuple(pytest.param(expert, fsdp, marks=pytest.mark.mesh)
                     for expert, fsdp in ((2, 4), (4, 2), (8, 1)))
LAYOUTS = (pytest.param(1, 1, id='1-1'), *MESH_LAYOUTS)
# A mesh whose data axis splits the tokens as well as the expert axis.
DATA_LAYOUT = pytest.param(2, 2, marks=pytest.mark.mesh, id='2-2-data2')


def rounded(value, dtype=ml_dtypes.bfloat16) -> np.ndarray:
    return np.asarray(value).astype(dtype).astype(np.float64)


THREE_EACH = np.full(8, 3)
"""Three rows for each of eight experts, the projections' equal routing."""

def projection_draw(input_dtype, master, rows: int = 24, depth: int = 16, width: int = 16, stream: int = 271):
    """A projection's drawn operands: tokens `[rows, depth]` in `input_dtype`, eight experts' matrices
    `[8, depth, width]` in `master` and an fp32 cotangent `[rows, width]`."""
    def draw(seed):
        rng = np.random.default_rng([stream, seed])
        return (jnp.asarray(rng.normal(size=(rows, depth)), input_dtype),
                jnp.asarray(rng.normal(size=(8, depth, width)), master),
                jnp.asarray(rng.normal(size=(rows, width)), jnp.float32))
    return draw


def projected_products(sizes, cotangent_dtype=ml_dtypes.bfloat16):
    """A bf16 projection's forward, kernel gradient and input gradient in float64 (check 1's truth): the
    float64 sums of its operands as the projection rounds them, the cotangent rounded to
    `cotangent_dtype` (`grouped_oracle`)."""
    @as_given
    def truth(x, kernel, dy):
        qx, qk, qy = rounded(x), rounded(kernel), rounded(dy, cotangent_dtype)
        forward, input_gradient, kernel_gradient = oracle.products(qx, qk, qy, sizes)
        return forward, kernel_gradient, input_gradient
    return truth


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
    """Forward, kernel cotangent and input cotangent, with the kernel split on either of its dimensions
    and the gradient returned placed or replicated: on values whose every product and sum is exact
    (`cases`'s round-once cases), the float64 sums of the bf16 operands rounded once, to the bit; on
    drawn values, on par with the projection on one device by check 1, truth those float64 sums."""
    sizes = np.full(8, 3, np.int32)

    def loss(kernel, x, dy, sizes):
        projected = jnp.asarray(expert_projection(x, kernel, sizes, jnp.bfloat16, 'xla', None))
        return jnp.sum(projected.astype(jnp.float32) * dy), projected

    mesh = MeshSpec(expert=expert, fsdp=fsdp).build()
    kernels, inputs, outputs = shardings(mesh, rows)
    replicated = NamedSharding(mesh, P())

    def meshed(spec):
        compiled = jax.jit(jax.value_and_grad(loss, (0, 1), has_aux=True),
                           out_shardings=((replicated, outputs), (spec, inputs)))

        @as_given
        def run(x, kernel, dy):
            arguments = tuple(jax.device_put(value, placed) for value, placed in zip(
                (kernel, x, dy, jnp.asarray(sizes)), (kernels, inputs, outputs, replicated), strict=True))
            with jax.set_mesh(mesh):
                (_, y), (dk, dx) = compiled(*arguments)
            return y, dk, dx
        return run

    @as_given
    def alone(x, kernel, dy):
        (_, y), (dk, dx) = jax.jit(jax.value_and_grad(loss, (0, 1), has_aux=True))(kernel, x, dy,
                                                                                     jnp.asarray(sizes))
        return y, dk, dx

    for spec in (kernels, replicated):
        if case != 'random':
            x, kernel, dy = next(values for name, *values in cases() if name == case)
            operands = (jnp.asarray(x, input_dtype), jnp.asarray(kernel, master), jnp.asarray(dy))
            y, dk, dx = meshed(spec)(*operands)
            assert dk.dtype == master and dx.dtype == input_dtype
            forward, kernel_gradient, input_gradient = projected_products(sizes)(*operands)
            np.testing.assert_array_equal(np.asarray(y, np.float64), rounded(forward))
            np.testing.assert_array_equal(np.asarray(dk, np.float64), kernel_gradient)
            np.testing.assert_array_equal(np.asarray(dx, np.float64), rounded(input_gradient, input_dtype))
            continue
        label = (f"a projection over expert {expert} fsdp {fsdp}, {'rows' if rows else 'columns'} split, "
                 f"{jnp.dtype(master).name} master, {jnp.dtype(input_dtype).name} tokens, gradient "
                 f"{'placed' if spec is kernels else 'replicated'}")
        assert_as_exact(meshed(spec), alone, projection_draw(input_dtype, master), label, seeds=FEWEST_SEEDS,
                        truth_fn=projected_products(sizes))


@pytest.mark.parametrize('sizes', [
    np.full(8, 3), np.array([0, 5, 0, 0, 12, 1, 0, 3]), np.array([0, 0, 0, 20, 0, 0, 0, 0]),
    np.array([0, 3, 3, 3, 3, 3, 3, 3]), np.zeros(8)],
    ids=['even', 'ragged', 'one-expert', 'fewer-rows', 'all-empty'])
@pytest.mark.parametrize('input_dtype', [jnp.bfloat16, jnp.float32])
def test_the_pallas_kernels_hold_the_contract_on_one_device(sizes, input_dtype):
    """Forward, kernel and input cotangents of the projection's kernels (interpreted on a CPU,
    `interpreted_projection`) on par with XLA's ragged dot by check 1, truth the float64 sums of the bf16
    operands: rows past the routed ones, and every row when no expert is routed, come back zero in the
    output and the input gradient, to the bit."""
    sizes = jnp.asarray(sizes, jnp.int32)
    routed = int(sizes.sum())

    def arm(implementation):
        def loss(x, kernel, dy):
            projected = jnp.asarray(implementation(x, kernel, sizes, jnp.bfloat16))
            cotangent = dy.astype(jnp.bfloat16).astype(jnp.float32)
            return jnp.sum(projected.astype(jnp.float32) * cotangent), projected

        @as_given
        def run(x, kernel, dy):
            # A CPU interprets the kernels as the A100 it states; a GPU runs its own.
            interpreted = jax.default_backend() == "cpu"
            with generation.assumed("sm80") if interpreted else contextlib.nullcontext():
                (_, y), (dx, dk) = jax.jit(jax.value_and_grad(loss, (0, 1), has_aux=True))(x, kernel, dy)
            return y, dk, dx
        return run

    kernels = arm(interpreted_projection)
    draw = projection_draw(input_dtype, jnp.float32, stream=88)
    y, dk, dx = kernels(*draw(0))
    assert dx.dtype == input_dtype and dk.dtype == jnp.float32
    assert not np.any(np.asarray(y)[routed:]) and not np.any(np.asarray(dx)[routed:])
    if routed:
        xla = arm(lambda x, kernel, sizes, dtype: expert_projection(x, kernel, sizes, dtype, 'xla', None))
        label = (f"the projection's kernels on one device, routed {sizes.tolist()}, "
                 f"{jnp.dtype(input_dtype).name}")
        assert_as_exact(kernels, xla, draw, label, seeds=FEWEST_SEEDS, truth_fn=projected_products(sizes))




def every_tile_of_the_grouped_kernels_holds_the_float64_gate(kernel, width, features):
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
    """A bf16 projection's output and both gradients through `implementation` ('pallas': the kernels,
    interpreted on a CPU, `interpreted_projection`), its output times `fault`."""
    def project(x, k):
        if implementation == 'pallas':
            return interpreted_projection(x, k, jnp.asarray(TILE_SIZES), jnp.bfloat16)
        return expert_projection(x, k, jnp.asarray(TILE_SIZES), jnp.bfloat16, implementation, None)

    def run(x, kernel_matrix, dy):
        y, *gradients = with_gradients(project)(x, kernel_matrix, dy)
        return ((y.astype(jnp.float32) * fault).astype(y.dtype), *gradients)
    return run


def _tile_gate(kernel, width, features):
    """The tile gate of `kernel`'s launch at one shape: its draws, and `check(config, fault)`, check 1 of
    the launch at `config` against XLA's ragged dot, which runs under its own choice so the forced kernel
    never reaches it."""
    rows = int(TILE_SIZES.sum())

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(rows, width)), jnp.bfloat16),
                jnp.asarray(rng.normal(size=(len(TILE_SIZES), width, features)) / np.sqrt(width),
                            jnp.float32),
                jnp.asarray(rng.normal(size=(rows, features)), jnp.bfloat16))

    @as_given
    def truth(x, kernel_matrix, dy):
        # The products of the operands as the projection rounds them, in float64.
        return oracle.products(x, rounded(kernel_matrix), dy, TILE_SIZES)

    xla = jax.jit(_tile_gate_arm('xla'))

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


def a_projection_on_the_cpu_of_a_gpu_host_runs_without_the_triton_kernels(implementation):
    """The kernels are chosen where the call lowers, not by the process's
    default backend. On a GPU host, a projection placed on the CPU runs
    XLA's program ('auto' and 'pallas' alike): its forward is XLA's ragged
    dot's on the CPU, to the bit."""
    rng = np.random.default_rng(88)
    x = rng.normal(size=(24, 16)).astype(np.float32)
    kernel = rng.normal(size=(8, 16, 16)).astype(np.float32)
    cpu = jax.devices('cpu')[0]
    def project(implementation):
        return jax.jit(lambda x, kernel, sizes: expert_projection(
            x, kernel, sizes, jnp.bfloat16, implementation, None))

    operands = jax.device_put((x, kernel, np.full(8, 3, np.int32)), cpu)
    y = project(implementation)(*operands)

    assert y.devices() == {cpu}
    np.testing.assert_array_equal(np.asarray(y), np.asarray(project('xla')(*operands)))


@pytest.mark.parametrize('sizes', [np.full(8, 3), np.array([0, 5, 0, 0, 12, 1, 0, 3])],
                         ids=['even', 'ragged'])
def test_a_bf16_projection_at_the_highest_precision_holds_the_contract(sizes):
    """bf16 operands multiply exactly into fp32 sums at any precision, so
    'highest' asks for the product the default gives: forward and both
    cotangents on par with the default's by check 1, truth the float64 sums of
    the rounded operands, on every backend.
    XLA's TPU ragged-dot kernel refuses bf16 operands at fp32 contract
    precision (Mosaic "Bad lhs type" on a v6e, jax 0.11.2), which any run
    that sets jax_default_matmul_precision to 'highest' reaches."""
    sizes = jnp.asarray(sizes, jnp.int32)

    def at(precision):
        def loss(x, kernel, dy):
            projected = expert_projection(x, kernel, sizes, jnp.bfloat16, 'xla', precision)
            cotangent = dy.astype(jnp.bfloat16).astype(jnp.float32)
            return jnp.sum(projected.astype(jnp.float32) * cotangent), projected

        @as_given
        def run(x, kernel, dy):
            (_, y), (dx, dk) = jax.jit(jax.value_and_grad(loss, (0, 1), has_aux=True))(x, kernel, dy)
            return y, dk, dx
        return run

    # A float32 master kernel, as training keeps it: the projection rounds it
    # to bf16 for the product, and its gradient comes back in float32.
    draw = projection_draw(jnp.bfloat16, jnp.float32, stream=89)
    _, dk, dx = at('highest')(*draw(0))
    assert dx.dtype == jnp.bfloat16 and dk.dtype == jnp.float32
    label = f"a bf16 projection at the highest precision, routed {sizes.tolist()}"
    assert_as_exact(at('highest'), at(None), draw, label, seeds=FEWEST_SEEDS,
                    truth_fn=projected_products(sizes))


@pytest.mark.usefixtures('x64')
@pytest.mark.parametrize('compute,input_dtype,master', [
    (jnp.bfloat16, jnp.bfloat16, jnp.float32), (jnp.bfloat16, jnp.float32, jnp.float64),
    (jnp.float32, jnp.float32, jnp.float32), (jnp.float64, jnp.float64, jnp.float64)])
@pytest.mark.parametrize('expert,fsdp', LAYOUTS)
@pytest.mark.parametrize('rows', [True, False], ids=['rows', 'columns'])
def test_a_projection_differentiates_the_same_law_in_every_direction(
        compute, input_dtype, master, expert, fsdp, rows):
    """JVP, VJP, forward-over-reverse and reverse-over-forward on the mesh on par with one device's by
    check 1, truth the tangent law `dx @ Q(kernel) + Q(x) @ dkernel` in float64."""
    sizes = jnp.full(8, 3, jnp.int32)

    def draw(seed):
        rng = np.random.default_rng([943, seed])
        x = jnp.asarray(rng.normal(size=(24, 16)), input_dtype)
        kernel = jnp.asarray(rng.normal(size=(8, 16, 16)), master)
        dx = jnp.asarray(rng.normal(scale=.3, size=x.shape), input_dtype)
        dkernel = jnp.asarray(rng.normal(scale=.3, size=kernel.shape), master)
        return x, kernel, dx, dkernel, jnp.asarray(rng.normal(size=(24, 16)), jnp.float64)

    @as_given
    def law(x, kernel, dx, dkernel, cotangent):
        qx, qk, qc = (rounded(value, compute) for value in (x, kernel, cotangent))
        reverse_of_forward = (oracle.input_gradient(qc, dkernel, THREE_EACH),
                              oracle.kernel_gradient(dx, qc, THREE_EACH))
        return (oracle.forward(qx, qk, THREE_EACH),
                oracle.forward(dx, qk, THREE_EACH) + oracle.forward(qx, dkernel, THREE_EACH),
                (oracle.input_gradient(qc, qk, THREE_EACH), oracle.kernel_gradient(qx, qc, THREE_EACH)),
                reverse_of_forward, reverse_of_forward)

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
    compiled = jax.jit(evaluate, out_shardings=(
        outputs, outputs, (inputs, kernels), (inputs, kernels), (inputs, kernels)))

    @as_given
    def meshed(*operands):
        arguments = tuple(jax.device_put(value, spec) for value, spec in zip(
            (*operands, sizes), specs, strict=True))
        with jax.set_mesh(mesh):
            return compiled(*arguments)

    label = (f"a projection's derivatives over expert {expert} fsdp {fsdp}, {'rows' if rows else 'columns'} "
             f"split, {jnp.dtype(compute).name} compute, {jnp.dtype(input_dtype).name} tokens, "
             f"{jnp.dtype(master).name} master")
    assert_as_exact(meshed, jax.jit(lambda *operands: evaluate(*operands, sizes)), draw, label,
                    seeds=FEWEST_SEEDS, truth_fn=law)


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
    roundings = chain_roundings(jax.make_jaxpr(model.apply)(variables, x, jnp.ones((1, 1), jnp.float32),
                                                             jnp.zeros((1, 1), jnp.int32)))
    assert_computes_the_oracle(np.asarray(actual), expected, "gpt-oss's promoted gate", roundings=roundings)



@pytest.mark.usefixtures('x64')
@pytest.mark.parametrize('dtype', [jnp.bfloat16, jnp.float32, jnp.float64])
def test_the_exact_gelu_rounds_once_in_every_differentiation_mode(dtype):
    """Value, JVP, VJP and both mixed second derivatives against SciPy's
    erfc in float64: in bf16 and fp32 on par with XLA's erf GELU by check 1
    over drawn values, and in float64 within float64's rounding of the
    computation (`assert_computes_the_oracle`)."""
    def draw(seed):
        values, cotangents, directions = np.random.default_rng([1041, seed]).normal(size=(3, 1024))
        return (jnp.asarray(values * 3, dtype), jnp.asarray(directions * .2, dtype),
                jnp.asarray(cotangents, dtype))

    @as_given
    def scipy(x, dx, ct):
        v, t, c = (np.asarray(array, np.float64) for array in (x, dx, ct))
        density = np.exp(-v * v / 2) / np.sqrt(2 * np.pi)
        first = .5 * erfc(-v / np.sqrt(2)) + v * density
        second = (2 - v * v) * density
        return .5 * v * erfc(-v / np.sqrt(2)), first * t, first * c, second * c * t, second * t * c

    def evaluated(gelu):
        def evaluate(x, dx, ct):
            def scalar(x):
                return jnp.sum(gelu(x).astype(jnp.float64) * ct.astype(jnp.float64))

            def directional(x):
                return jnp.sum(jax.jvp(gelu, (x,), (dx,))[1].astype(jnp.float64) * ct.astype(jnp.float64))

            y, tangent = jax.jvp(gelu, (x,), (dx,))
            return (y, tangent, jax.grad(scalar)(x), jax.jvp(jax.grad(scalar), (x,), (dx,))[1],
                    jax.grad(directional)(x))
        return jax.jit(evaluate)

    if dtype == jnp.float64:
        operands = draw(0)
        roundings = chain_roundings(jax.make_jaxpr(evaluated(exact_gelu))(*operands))
        names = ("value", "jvp", "vjp", "forward over reverse", "reverse over forward")
        for name, mine, theirs in zip(names, evaluated(exact_gelu)(*operands), scipy(*operands), strict=True):
            assert_computes_the_oracle(np.asarray(mine), theirs, f"the exact gelu's {name}",
                                       roundings=roundings)
        return
    label = f"the exact gelu's derivatives, {jnp.dtype(dtype).name}"
    assert_as_exact(evaluated(exact_gelu), evaluated(lambda x: jax.nn.gelu(x, approximate=False)), draw,
                    label, seeds=FEWEST_SEEDS, truth_fn=scipy)


def placed_experts(mesh, activation, skewed, scale_inputs, limit, implementation='xla'):
    rng = np.random.default_rng(341)
    model = ExpertMLP(8, 16, 8, dtype=jnp.bfloat16, activation=activation,
                      scale_inputs=scale_inputs, swiglu_limit=limit,
                      implementation=implementation)
    x = jnp.asarray(rounded(rng.normal(size=(24, 8))), jnp.bfloat16)
    weights = jnp.asarray(rng.uniform(.1, .9, size=(24, 2)), jnp.float32)
    choices = (np.tile([0, 1], (24, 1)) if skewed
               else np.argsort(rng.normal(size=(24, 8)), axis=-1)[:, :2]).astype(np.int32)
    parameters = nn.unbox(model.init(jax.random.key(19), x, weights, choices))
    columns = NamedSharding(mesh, P('expert', None, 'fsdp'))
    specs = {'params': {'gate_proj': {'kernel': columns}, 'up_proj': {'kernel': columns},
                        'down_proj': {'kernel': NamedSharding(mesh, P('expert', 'fsdp'))}}}
    tokens = NamedSharding(mesh, P(('expert', 'fsdp')))
    parameters = jax.device_put(parameters, specs)
    x, weights, choices = (jax.device_put(value, tokens) for value in (x, weights, jnp.asarray(choices)))
    return model, parameters, specs, tokens, x, weights, choices


PAIRED_CHOICES = np.argsort(np.random.default_rng(3699).normal(size=(24, 8)) - np.arange(8) / 4,
                            axis=-1)[:, :2].astype(np.int32)
"""Each of 24 tokens' two experts, the earlier experts more often, so the groups differ in size."""


@pytest.mark.usefixtures('x64')
@pytest.mark.parametrize('activation, limit', [('swiglu', None), ('swiglu', .7), ('geglu', None)])
def test_a_layers_gate_and_up_as_one_paired_product_hold_check_1_against_the_two_products(activation, limit):
    """A routed layer multiplies its rows by its gate's and up projection's matrices as one paired
    product (`moe.paired_projections`), which reads the rows once where its launch takes the pair: the
    layer's output and the gradients of its parameters, tokens and router weights on par from float64
    with the two products computed apart over the same dispatch (check 1, ledger 3699), clamped
    (`swiglu_limit`) and not, and under each gate."""
    model = ExpertMLP(8, 16, 8, dtype=jnp.bfloat16, activation=activation, swiglu_limit=limit)
    names = tuple(EXPERT_AXES)

    def apart(parameters, x, weights, dtype=jnp.bfloat16):
        """The layer as two products: its gate's and up projection's apart, then its down projection
        and its experts' outputs summed in fp32 under the router weights and rounded once."""
        def project(tokens, sizes, _ids, kernels):
            def linear(rows, kernel):
                return jnp.asarray(expert_projection(rows, kernel, sizes, dtype, 'auto', None))
            gate, up = swiglu_clamped(linear(tokens, kernels[0]), linear(tokens, kernels[1]), limit)
            return linear(gated_product(activation)(gate, up), kernels[2])

        kernels = tuple(parameters['params'][name]['kernel'] for name in names)
        slots = expert_dispatch(project, x, jnp.asarray(PAIRED_CHOICES), kernels,
                                tuple(EXPERT_AXES[name] for name in names), num_experts=8, dispatch='global',
                                output_dtype=dtype)
        wide = jnp.promote_types(dtype, jnp.float32)
        return jnp.einsum('tke,tk->te', slots.astype(wide), weights.astype(wide)).astype(dtype)

    def paired(parameters, x, weights):
        return jnp.asarray(model.apply(parameters, x, weights, jnp.asarray(PAIRED_CHOICES)))

    def truth(parameters, x, weights):
        return apart(*jax.tree.map(lambda value: value.astype(jnp.float64), (parameters, x, weights)),
                     dtype=jnp.float64)

    def draw(seed):
        rng = np.random.default_rng(seed)
        x = jnp.asarray(rng.normal(size=(24, 8)), jnp.bfloat16)
        weights = jnp.asarray(rng.uniform(.1, .9, size=(24, 2)), jnp.float32)
        parameters = nn.unbox(model.init(jax.random.key(seed), x, weights, jnp.asarray(PAIRED_CHOICES)))
        return parameters, x, weights, jnp.asarray(rng.normal(size=(24, 8)), jnp.bfloat16)

    label = f"paired gate and up, {activation}{'' if limit is None else ' clamped'}"
    assert_as_exact(jax.jit(with_gradients(paired)), jax.jit(with_gradients(apart)), draw, label, seeds=8,
                    truth_fn=jax.jit(with_gradients(truth)))


def test_paired_projections_take_one_compute_dtype_however_a_matrix_is_held():
    """The gate's and up projection's products take the rows' and both matrices' one compute dtype
    (`paired_projections`), a matrix read in place from a bank (`Layered`, multiplied apart) as one held
    whole: a bf16 gate beside a float32 up projection gives two float32 products either way, where each
    product promoting alone gave the bf16 gate's a bf16 one."""
    rng = np.random.default_rng(3699)
    x = jnp.asarray(rng.normal(size=(6, 8)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(size=(2, 8, 4)), jnp.bfloat16)
    up = jnp.asarray(rng.normal(size=(2, 8, 4)), jnp.float32)
    sizes = jnp.asarray([4, 2], jnp.int32)
    for first in (gate, Layered(gate[None], jnp.asarray(0))):
        products = paired_projections(x, first, up, sizes, None, 'auto', None)
        assert [product.dtype for product in products] == [jnp.float32, jnp.float32]


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
    initialized = nn.unbox(layer.init(jax.random.key(0), x, weights, indices))["params"]
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
    and moments of the exchange dispatch on par with the global one's by
    check 1 over drawn tokens and weights, truth the global dispatch's float64
    layer on the CPU. On a GPU 'pallas' runs its kernels on every device's own
    rows, the exchange included on a mesh whose only split axis is the expert
    one; a CPU runs XLA's program."""
    mesh = MeshSpec(expert=expert, fsdp=fsdp).build()
    model, _, specs, tokens, *_ = placed_experts(mesh, activation, skewed, scale_inputs, limit,
                                                  implementation)
    optimizer = optax.adam(1e-3)

    def trajectory(network):
        def objective(parameters, x, weights, choices):
            y = jnp.asarray(network.apply(parameters, x, weights, choices))
            return jnp.mean(jnp.sin(y.astype(jnp.promote_types(y.dtype, jnp.float32)))), y

        def step(parameters, state, x, weights, choices):
            result, gradients = jax.value_and_grad(objective, (0, 1, 2), has_aux=True)(
                parameters, x, weights, choices)
            updates, state = optimizer.update(gradients[0], state, parameters)
            return result, gradients, optax.apply_updates(parameters, updates), state

        compiled = jax.jit(step)

        def run(parameters, x, weights, choices):
            steps, (p, s) = [], (parameters, optimizer.init(parameters))
            for _ in range(3):
                result = compiled(p, s, x, weights, choices)
                steps.append(result)
                p, s = result[2:]
            return steps
        return run

    def dispatched(dispatch):
        run = trajectory(model.clone(dispatch=dispatch))

        @as_given
        def placed(parameters, x, weights, choices):
            with jax.set_mesh(mesh):
                return run(jax.device_put(parameters, specs), *jax.device_put((x, weights, choices), tokens))
        return placed

    def draw(seed):
        rng = np.random.default_rng([341, seed])
        x = jnp.asarray(rounded(rng.normal(size=(24, 8))), jnp.bfloat16)
        weights = jnp.asarray(rng.uniform(.1, .9, size=(24, 2)), jnp.float32)
        choices = jnp.asarray(np.tile([0, 1], (24, 1)) if skewed
                              else np.argsort(rng.normal(size=(24, 8)), axis=-1)[:, :2], jnp.int32)
        return nn.unbox(model.init(jax.random.key(seed), x, weights, choices)), x, weights, choices

    host = jax.devices("cpu")[0]
    wide = trajectory(model.clone(dispatch='global', dtype=jnp.float64))

    @as_given
    def truth(*inputs):
        with jax.default_device(host):
            return wide(*jax.device_put(inputs, host))

    label = (f"three Adam steps of the exchange dispatch, {activation}, {'skewed' if skewed else 'random'}, "
             f"expert {expert} fsdp {fsdp}, {implementation}")
    assert_as_exact(dispatched('exchange'), dispatched('global'), draw, label, seeds=FEWEST_SEEDS,
                    truth_fn=truth)


DISPATCH_TANGENT_SEEDS = 8
"""Check 1's declared seeds for the exchange dispatch's JVP against the global one's, sized by `seeds_for`
on armada."""


@pytest.mark.mesh
@pytest.mark.parametrize('activation', ['swiglu', 'geglu', 'geglu_exact'])
def test_both_dispatches_carry_the_same_tangents(activation):
    """A JVP through the whole routed layer, parameters, tokens and router weights all perturbed: the
    exchange dispatch's output and tangent on par with the global dispatch's from the float64 layer's
    over the declared draws (check 1), that truth on a CPU device, which has the float64 ragged dot a
    TPU has none of. The dispatches are two programs, the exchange splitting each expert's rows over
    the devices, whose floats XLA keeps no common bits between (ledger 3178)."""
    mesh = MeshSpec(expert=4, fsdp=2).build()
    model, _, specs, tokens, *_ = placed_experts(mesh, activation, skewed=False, scale_inputs=False,
                                                 limit=None)

    def jvp(network):
        def run(parameters, x, weights, choices, dp, dx, dw):
            return jax.jvp(lambda parameters, x, weights: jnp.asarray(
                network.apply(parameters, x, weights, choices)), (parameters, x, weights), (dp, dx, dw))
        return run

    def dispatched(dispatch):
        """The bf16 layer's JVP under `dispatch` on the mesh, compiled as the gate compiles two Dew arms
        (`without_ynn`) and run as given."""
        compiled = without_ynn(jvp(model.clone(dispatch=dispatch)), out_shardings=(tokens, tokens))

        @as_given
        def placed(*inputs):
            with jax.set_mesh(mesh):
                return compiled(*jax.device_put(inputs, (specs, *(tokens,) * 3, specs, tokens, tokens)))
        return placed

    def draw(seed):
        rng = np.random.default_rng([0x3178, seed])
        x = jnp.asarray(rounded(rng.normal(size=(24, 8))), jnp.bfloat16)
        weights = jnp.asarray(rng.uniform(.1, .9, size=(24, 2)), jnp.float32)
        choices = jnp.asarray(np.argsort(rng.normal(size=(24, 8)), axis=-1)[:, :2], jnp.int32)
        drawn = nn.unbox(model.init(jax.random.key(seed), x, weights, choices))

        def tangent(value):
            return jnp.asarray(rng.normal(scale=.05, size=value.shape), value.dtype)
        return drawn, x, weights, choices, *jax.tree.map(tangent, (drawn, x, weights))

    host = jax.devices("cpu")[0]
    wide = jax.jit(jvp(model.clone(dispatch='global', dtype=jnp.float64)))

    @as_given
    def truth(*inputs):
        """Its inputs moved to the host, then `wide` run there: a placement, which the gate runs as given."""
        with jax.default_device(host):
            return wide(*jax.device_put(inputs, host))

    assert_as_exact(dispatched('exchange'), dispatched('global'), draw,
                    f"exchange dispatch's JVP, {activation}", seeds=DISPATCH_TANGENT_SEEDS, truth_fn=truth)


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
