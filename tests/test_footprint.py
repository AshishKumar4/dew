"""The residual storage a pullback names before compiling, not a claim about XLA's scheduled peak."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew.training.footprint import device_bytes, gradient_peak, residual_bytes


def test_resident_device_bytes_do_not_include_host_state():
    """A state leaf placed on pinned host memory is not HBM, even though it has the same global
    shape and partition as a device leaf. Counting the two as resident would charge offload twice."""
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",))
    sharding = NamedSharding(mesh, P())
    device = jax.ShapeDtypeStruct((8, 16), jnp.float32, sharding=sharding)
    host = jax.ShapeDtypeStruct((8, 16), jnp.float32, sharding=sharding.with_memory_kind("pinned_host"))
    assert device_bytes({"params": device, "optimizer": host}) == 8 * 16 * 4


def test_a_pullback_counts_new_residuals_not_input_storage():
    """On a CPU the exponential needed to differentiate exp is new storage, an expensive value XLA keeps.
    A dot product needs the other operand, already resident as an input, which must not count again as
    residual storage; nor does the cosine sine's derivative needs, which XLA recomputes from its input
    in the backward's fusion, a sine's cosine being cheap (XLA's `InstructionFusion::IsExpensive`)."""
    shape = jax.ShapeDtypeStruct((8, 16), jnp.float32)

    def exponential(params):
        return jnp.exp(params).sum(), ()

    def sine(params):
        return jnp.sin(params).sum(), ()

    def product(params, values):
        return jnp.vdot(params, values), ()

    assert residual_bytes(exponential, shape, (), platform="cpu") == 8 * 16 * 4
    assert residual_bytes(sine, shape, (), platform="cpu") == 0
    assert residual_bytes(product, shape, (shape,), platform="cpu") == 0


def test_a_pullback_counts_a_cheap_chain_as_the_product_it_is_recomputed_from():
    """Residuals XLA recomputes in the backward's fusions, each a cheap elementwise value (a select, a
    multiply) of a product, count as that product, once: a rectified, scaled product squared keeps the
    product's bytes, not the two cheap values JAX's pullback closes over."""
    weights, rows = jax.ShapeDtypeStruct((16, 32), jnp.float32), jax.ShapeDtypeStruct((8, 16), jnp.float32)

    def loss(weights, rows):
        return jnp.sum(jnp.square(jnp.maximum(rows @ weights, 0.0) * 2.0 + 1.0)), ()

    assert residual_bytes(loss, weights, (rows,), platform="cpu") == 8 * 32 * 4


def test_a_custom_pullback_counts_one_buffer_shared_by_two_residual_slots():
    """A custom VJP retains the same intermediate twice and its input once. Only the one new
    intermediate's storage counts; neither repeating its reference nor retaining an input doubles it."""
    @jax.custom_vjp
    def summed(params):
        return jnp.sin(params).sum()

    def forward(params):
        # An exponential, which XLA keeps (a cosine of real values it would recompute from the input).
        exponential = jnp.exp(params)
        return summed(params), (exponential, exponential, params)

    def backward(saved, cotangent):
        first, second, inputs = saved
        return ((first + second) * cotangent / 2 + inputs * 0,)

    summed.defvjp(forward, backward)

    def loss(params):
        return summed(params), ()

    shape = jax.ShapeDtypeStruct((8, 16), jnp.float32)
    assert residual_bytes(loss, shape, (), platform="cpu") == 8 * 16 * 4


def test_a_gpu_pullback_counts_a_transcendental_chain_as_the_product_it_is_recomputed_from():
    """On a GPU XLA's priority fusion recomputes every elementwise value in its consumers, a softmax's
    exponentials and quotients among them, from the product they are computed from: an attention's
    probabilities keep its scores' bytes there, and on a CPU, whose fusion keeps the expensive values
    (`InstructionFusion::IsExpensive`), the exponentials and quotients as well."""
    keys, queries = jax.ShapeDtypeStruct((16, 32), jnp.float32), jax.ShapeDtypeStruct((8, 32), jnp.float32)

    def attended(keys, queries):
        scores = queries @ keys.T
        weights = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
        return jnp.sum((weights / jnp.sum(weights, axis=-1, keepdims=True)) @ keys), ()

    scores = 8 * 16 * 4
    gpu = residual_bytes(attended, keys, (queries,), platform="gpu")
    assert scores <= gpu < 2 * scores < residual_bytes(attended, keys, (queries,), platform="cpu")


def test_a_gradients_peak_holds_the_gradient_beside_the_cotangent_each_residual_freed_after_its_read():
    """The most a gradient program holds at once, in its traced order: through two products, the second
    weight's gradient and the first layer's cotangent, then the first weight's gradient beside them, each
    residual (the first layer's output, the second's) freed after its last read, which comes before; and
    beside each matrix product the workspace the device gives its GEMMs, one at a time."""
    rows = jax.ShapeDtypeStruct((8, 16), jnp.float32)
    params = {"first": jax.ShapeDtypeStruct((16, 64), jnp.float32),
              "second": jax.ShapeDtypeStruct((64, 16), jnp.float32)}

    def loss(params, rows):
        return jnp.sum(jnp.exp((rows @ params["first"]) @ params["second"])), ()

    # Both gradients, 64 x 16 floats each, and the first layer's cotangent, 8 x 64.
    held = (2 * 64 * 16 + 8 * 64) * 4
    assert gradient_peak(loss, params, (rows,), platform="gpu", workspace=0).bytes == held
    assert gradient_peak(loss, params, (rows,), platform="gpu", workspace=1 << 20).bytes == held + (1 << 20)


def test_a_fused_value_a_product_reads_is_held_while_it_multiplies():
    """A value XLA's fusion would recompute is held where a matrix product reads it, which takes its
    operands from memory: the exponentials of the rows, a product's operand, and the cotangent of the
    product's hyperbolic tangent, the weight's gradient's operand, are both held beside that gradient as
    it is computed, with the product the cotangent is recomputed from."""
    rows, weights = jax.ShapeDtypeStruct((8, 16), jnp.float32), jax.ShapeDtypeStruct((16, 32), jnp.float32)

    def loss(weights, rows):
        return jnp.sum(jnp.tanh(jnp.exp(rows) @ weights)), ()

    # The gradient (16 x 32), the exponentials (8 x 16), the cotangent and the product (8 x 32 each).
    assert gradient_peak(loss, weights, (rows,), platform="gpu", workspace=0).bytes == (
        16 * 32 + 8 * 16 + 2 * 8 * 32) * 4


def test_a_gradient_a_fused_chain_computes_is_held():
    """A gradient the program returns is held whatever computes it: a sine's, its cosine, which no product
    reads, is the program's output."""
    shape = jax.ShapeDtypeStruct((8, 16), jnp.float32)

    def sine(params):
        return jnp.sin(params).sum(), ()

    for platform in ("gpu", "cpu"):
        assert gradient_peak(sine, shape, (), platform=platform, workspace=0).bytes == 8 * 16 * 4


def test_a_called_program_holds_what_it_would_inline():
    """A program called within the gradient (a jitted function's) holds at its peak what it would hold
    inlined: its results once, as the caller's."""
    rows, weights = jax.ShapeDtypeStruct((8, 16), jnp.float32), jax.ShapeDtypeStruct((16, 32), jnp.float32)

    def inline(weights, rows):
        return jnp.sum(jnp.tanh(rows @ weights)), ()

    def called(weights, rows):
        return jnp.sum(jnp.tanh(jax.jit(jnp.matmul)(rows, weights))), ()

    assert gradient_peak(called, weights, (rows,), platform="gpu", workspace=0) == gradient_peak(
        inline, weights, (rows,), platform="gpu", workspace=0)


@pytest.mark.parametrize(("generation", "workspace"), [("sm80", 4 << 20), ("sm90", 32 << 20)])
def test_a_gpu_gives_each_gemm_the_workspace_xla_hands_it(generation, workspace):
    """XLA hands each cuBLAS matmul a workspace (gemm_rewriter's `GemmWorkspaceRewriteVisitor`): 32 MiB from
    Hopper on, 4 MiB before (`GemmConfig::kHopperWorkspace`, `kDefaultWorkspace`); a CPU's take none."""
    from dew.training.ahead import compile_only_gpus
    from dew.training.footprint import gemm_workspace

    (gpu,) = compile_only_gpus(generation, 1, 1 << 30)
    assert gemm_workspace(gpu) == workspace
    assert gemm_workspace(jax.devices("cpu")[0]) == 0


def test_a_loops_stacked_results_are_held_beside_each_turns_own():
    """A loop's stacked results are held whole beside what each turn holds while it runs: ten turns, each
    multiplying twice, hold the ten results and a turn's two products at once."""
    from dew.training.footprint import peak_bytes

    square = jax.ShapeDtypeStruct((8, 8), jnp.float32)

    def stacked(rows, weights):
        def turn(carried, _):
            return carried, (carried @ weights) @ weights
        return jax.lax.scan(turn, rows, None, length=10)[1]

    program = jax.make_jaxpr(stacked)(square, square).jaxpr
    assert peak_bytes(program, "gpu", 0) == (10 + 2) * 8 * 8 * 4


def test_a_transposed_input_returned_flat_is_a_copy():
    """A transpose of an input whose layout is fixed reorders its elements into a buffer of their own,
    which a flattening of it returned holds, where a transpose of a value the program computes is laid
    out by XLA as its reader needs (a returned gradient's)."""
    from dew.training.footprint import peak_bytes

    rows = jax.ShapeDtypeStruct((8, 16), jnp.float32)
    program = jax.make_jaxpr(lambda rows: rows.T.reshape(-1))(rows).jaxpr
    assert peak_bytes(program, "gpu", 0) == 8 * 16 * 4


def test_a_product_too_small_for_cublas_takes_no_workspace():
    """XLA hands a cuBLAS workspace to the products it rewrites to cuBLAS, not to one too small for it,
    whose non-contracting sizes summed times its contracting size fall under
    `xla_gpu_gemm_rewrite_size_threshold` (100 by default), which its own emitters compute."""
    small, large = jax.ShapeDtypeStruct((2, 4), jnp.float32), jax.ShapeDtypeStruct((64, 64), jnp.float32)

    def product(params, rows):
        return jnp.sum(rows @ params), ()

    for shape, workspace in ((small, 0), (large, 1 << 20)):
        rows = jax.ShapeDtypeStruct((shape.shape[0], shape.shape[0]), jnp.float32)
        with_workspace = gradient_peak(product, shape, (rows,), platform="gpu", workspace=1 << 20).bytes
        without = gradient_peak(product, shape, (rows,), platform="gpu", workspace=0).bytes
        assert with_workspace - without == workspace


def test_a_computed_value_read_in_two_orders_is_copied_for_the_second():
    """XLA lays a computed value out once: a product returned flat both as it is and transposed holds the
    product and a reordered copy, where its transpose alone would be the product laid out for it."""
    from dew.training.footprint import peak_bytes

    rows, weights = jax.ShapeDtypeStruct((8, 16), jnp.float32), jax.ShapeDtypeStruct((16, 32), jnp.float32)

    def flattened(rows, weights):
        product = rows @ weights
        return product.reshape(-1), product.T.reshape(-1)

    program = jax.make_jaxpr(flattened)(rows, weights).jaxpr
    assert peak_bytes(program, "gpu", 0) == 2 * 8 * 32 * 4


def test_an_inner_product_of_vectors_takes_no_workspace():
    """XLA hands cuBLAS no product of two vectors (`IsCublasSupportedMatMul`: neither operand a matrix),
    however long, so it takes no workspace, where a matrix times that vector does."""
    vector, matrix = jax.ShapeDtypeStruct((64,), jnp.float32), jax.ShapeDtypeStruct((64, 64), jnp.float32)

    def inner(params, other):
        return params @ other, ()

    def matrix_vector(params, other):
        return jnp.sum(other @ params), ()

    for loss, other, workspace in ((inner, vector, 0), (matrix_vector, matrix, 1 << 20)):
        with_workspace = gradient_peak(loss, vector, (other,), platform="gpu", workspace=1 << 20).bytes
        without = gradient_peak(loss, vector, (other,), platform="gpu", workspace=0).bytes
        assert with_workspace - without == workspace


def test_a_ragged_product_on_a_gpu_holds_at_least_its_output():
    """A ragged product on a GPU holds at least its output while it runs; what more XLA makes of it is its
    own choice by target, dtype and flags (a grouped kernel, or `RaggedDotRewriter`'s expansion over
    its groups, xla/hlo/transforms/expanders/ragged_dot_rewriter.cc, beside which layout assignment
    and fusion choose by their cost models), so the program states its peak a bound from below
    (`bounded_below`); on a CPU it is the program's own."""
    from dew.training.footprint import bounded_below, peak_bytes

    rows, sizes = jax.ShapeDtypeStruct((64, 16), jnp.float32), jax.ShapeDtypeStruct((4,), jnp.int32)
    weights = jax.ShapeDtypeStruct((4, 16, 32), jnp.float32)
    program = jax.make_jaxpr(lambda rows, weights, sizes: jax.lax.ragged_dot(rows, weights, sizes))(
        rows, weights, sizes).jaxpr
    for platform in ("gpu", "cpu"):
        assert peak_bytes(program, platform, 0) == 64 * 32 * 4
    assert bounded_below(program, "gpu") and not bounded_below(program, "cpu")


def test_a_gradient_states_its_peak_a_bound_from_below_where_a_nested_ragged_product_runs():
    """A gradient whose program runs a ragged product on a GPU, in a loop's body as an expert stack's
    scan holds one, states its peak a bound from below (`Peak.exact` False); on a CPU, or with no ragged
    product, its peak is the program's own."""
    rows, sizes = jax.ShapeDtypeStruct((64, 16), jnp.float32), jax.ShapeDtypeStruct((4,), jnp.int32)
    weights = jax.ShapeDtypeStruct((2, 4, 16, 16), jnp.float32)

    def experts(weights, rows, sizes):
        def layer(carried, kernel):
            return jax.lax.ragged_dot(carried, kernel, sizes), None
        return jnp.sum(jax.lax.scan(layer, rows, weights)[0]), ()

    def dense(weights, rows, sizes):
        return jnp.sum(rows @ weights[0, 0]), ()

    assert not gradient_peak(experts, weights, (rows, sizes), platform="gpu", workspace=0).exact
    assert gradient_peak(experts, weights, (rows, sizes), platform="cpu", workspace=0).exact
    assert gradient_peak(dense, weights, (rows, sizes), platform="gpu", workspace=0).exact
