"""The residual storage a pullback names before compiling, not a claim about XLA's scheduled peak."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew.training.footprint import Workspace, device_bytes, gradient_peak, residual_bytes


def test_resident_device_bytes_do_not_include_host_state():
    """A state leaf placed on pinned host memory is not HBM, even though it has the same global
    shape and partition as a device leaf. Counting the two as resident would charge offload twice."""
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",))
    sharding = NamedSharding(mesh, P())
    device = jax.ShapeDtypeStruct((8, 16), jnp.float32, sharding=sharding)
    host = jax.ShapeDtypeStruct((8, 16), jnp.float32, sharding=sharding.with_memory_kind("pinned_host"))
    assert device_bytes({"params": device, "optimizer": host}) == 8 * 16 * 4


def test_a_marked_spans_peak_is_its_bodys_as_xla_inlines_it():
    """A marked span (`dew.nn.derivatives.marked`, an error-free transform's `Protected` among them:
    `precision.two_sum`) lowers to its body in the caller, as a call does (`dew.nn.programs.CALLS`): its
    derived peak and residuals are the inline body's (`two_sum.__wrapped__`), its sum and its
    compensation fused into their reader, where read as an opaque op each held a buffer of its own (row
    3830)."""
    from dew.nn.precision import two_sum
    from dew.training.footprint import derived_terms

    def loss(weights, x, *, span):
        total, error = span(jnp.tanh(x @ weights), jnp.sin(x @ weights))
        return jnp.sum(total * total + error), ()

    weights = jax.ShapeDtypeStruct((64, 1024), jnp.float32)
    x = jax.ShapeDtypeStruct((256, 64), jnp.float32)

    def derived(span):
        residuals, peak = derived_terms(lambda weights, x: loss(weights, x, span=span), weights, (x,),
                                        platform="gpu", workspace=Workspace(0, 0))
        return residuals, peak.bytes

    assert derived(two_sum) == derived(two_sum.__wrapped__)


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
    for size in (0, 1 << 20):
        workspace = Workspace(size, size)
        assert gradient_peak(loss, params, (rows,), platform="gpu", workspace=workspace).bytes == held + size


def test_a_fused_value_a_product_reads_is_held_while_it_multiplies():
    """A value XLA's fusion would recompute is held where a matrix product reads it, which takes its
    operands from memory: the exponentials of the rows, a product's operand, and the cotangent of the
    product's hyperbolic tangent, the weight's gradient's operand, are both held beside that gradient as
    it is computed, with the product the cotangent is recomputed from."""
    rows, weights = jax.ShapeDtypeStruct((8, 16), jnp.float32), jax.ShapeDtypeStruct((16, 32), jnp.float32)

    def loss(weights, rows):
        return jnp.sum(jnp.tanh(jnp.exp(rows) @ weights)), ()

    # The gradient (16 x 32), the exponentials (8 x 16), the cotangent and the product (8 x 32 each).
    assert gradient_peak(loss, weights, (rows,), platform="gpu", workspace=Workspace(0, 0)).bytes == (
        16 * 32 + 8 * 16 + 2 * 8 * 32) * 4


def test_a_gradient_a_fused_chain_computes_is_held():
    """A gradient the program returns is held whatever computes it: a sine's, its cosine, which no product
    reads, is the program's output."""
    shape = jax.ShapeDtypeStruct((8, 16), jnp.float32)

    def sine(params):
        return jnp.sin(params).sum(), ()

    for platform in ("gpu", "cpu"):
        none = Workspace(0, 0)
        assert gradient_peak(sine, shape, (), platform=platform, workspace=none).bytes == 8 * 16 * 4


def test_a_called_product_in_a_loop_is_priced_at_its_fused_workspace():
    """A loop's turn reads a slice of its stacked weights, which XLA fuses with the cuBLAS product reading it
    (`Slicing.fused`), the product keeping the GEMM rewriter's workspace (`Workspace.sliced`): priced so
    where the product runs in a called program (a wide sum's, `precision.accumulated`, traced once for its
    types), as XLA inlines it (row 3193)."""
    from dew.nn.precision import accumulated

    def loss(weights, rows):
        def turn(carry, weight):
            return accumulated(jnp.dot, carry, weight), None

        return jax.lax.scan(turn, rows, weights)[0].sum(), None

    weights = jax.ShapeDtypeStruct((3, 64, 64), jnp.float32)
    rows = jax.ShapeDtypeStruct((128, 64), jnp.float32)
    free = gradient_peak(loss, weights, (rows,), platform="gpu", workspace=Workspace(0, 0)).bytes
    fused = gradient_peak(loss, weights, (rows,), platform="gpu", workspace=Workspace(0, 64 << 20)).bytes
    assert fused - free >= 32 << 20, (free, fused)


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
    assert peak_bytes(program, "gpu", Workspace(0, 0)) == (10 + 2) * 8 * 8 * 4


def test_a_loops_carry_read_and_written_through_an_axis_of_one_holds_one_buffer():
    """A loop that reads its carry by unstacking an axis of one and writes it back by stacking holds the
    carry's buffer alone, each a view of it, as one that reshapes it does: a scanned head's weight
    gradient read and written so cost the derivation two copies more (Qwen3.5-4B's fused tiled head,
    2 x 2.37 GiB, row 4089)."""
    from dew.training.footprint import peak_bytes

    carry = jax.ShapeDtypeStruct((1, 64, 64), jnp.float32)
    rows = jax.ShapeDtypeStruct((10, 64, 64), jnp.float32)

    def unstacked(carry, rows):
        def turn(held, row):
            (current,) = jnp.unstack(held, axis=0)
            return jnp.stack([current + row]), None
        return jax.lax.scan(turn, carry, rows)[0]

    def reshaped(carry, rows):
        def turn(held, row):
            return (held.reshape(64, 64) + row).reshape(1, 64, 64), None
        return jax.lax.scan(turn, carry, rows)[0]

    programs = [jax.make_jaxpr(function)(carry, rows).jaxpr for function in (unstacked, reshaped)]
    body = programs[0].eqns[-1].params["jaxpr"].jaxpr
    assert {"stack", "unstack"} <= {equation.primitive.name for equation in body.eqns}
    each = [peak_bytes(program, "gpu", Workspace(0, 0)) for program in programs]
    assert each[0] == each[1], each


def test_a_zero_gradient_is_held_from_the_programs_start_where_its_gradients_are_held():
    """The gradient of a parameter the loss reads none of is a broadcast zero JAX writes at the backward's
    end, which XLA makes ahead of the work, held to its reader: where the step holds every gradient to its
    end (`consumed` false, a global clip's), it is held through the backward's peak, one buffer of the
    parameter's size more than where the parameter is read (row 4089: Qwen3.5-4B's multi-token head's 25
    zero gradients, 0.48 GiB, at its step's peak)."""
    # Activations of 8 MiB a layer make the backward's peak, past which the gradients alone are held.
    rows = jax.ShapeDtypeStruct((4096, 512), jnp.float32)
    params = {"used": jax.ShapeDtypeStruct((512, 512), jnp.float32),
              "unread": jax.ShapeDtypeStruct((128, 128), jnp.float32)}

    def loss(params, rows):
        return jnp.sum(jnp.tanh(jnp.tanh(rows @ params["used"]) @ params["used"])), ()

    held = gradient_peak(loss, params, (rows,), platform="gpu", workspace=Workspace(0, 0)).bytes
    read = gradient_peak(loss, {"used": params["used"]}, (rows,), platform="gpu",
                         workspace=Workspace(0, 0)).bytes
    assert held == read + 128 * 128 * 4, (held, read)


def test_a_transposed_input_returned_flat_is_a_copy():
    """A transpose of an input whose layout is fixed reorders its elements into a buffer of their own,
    which a flattening of it returned holds, where a transpose of a value the program computes is laid
    out by XLA as its reader needs (a returned gradient's)."""
    from dew.training.footprint import peak_bytes

    rows = jax.ShapeDtypeStruct((8, 16), jnp.float32)
    program = jax.make_jaxpr(lambda rows: rows.T.reshape(-1))(rows).jaxpr
    assert peak_bytes(program, "gpu", Workspace(0, 0)) == 8 * 16 * 4


def test_a_product_too_small_for_cublas_takes_no_workspace():
    """XLA hands a cuBLAS workspace to the products it rewrites to cuBLAS, not to one too small for it,
    whose non-contracting sizes summed times its contracting size fall under
    `xla_gpu_gemm_rewrite_size_threshold` (100 by default), which its own emitters compute."""
    small, large = jax.ShapeDtypeStruct((2, 4), jnp.float32), jax.ShapeDtypeStruct((64, 64), jnp.float32)

    def product(params, rows):
        return jnp.sum(rows @ params), ()

    for shape, workspace in ((small, 0), (large, 1 << 20)):
        rows = jax.ShapeDtypeStruct((shape.shape[0], shape.shape[0]), jnp.float32)
        megabyte = Workspace(1 << 20, 1 << 20)
        with_workspace = gradient_peak(product, shape, (rows,), platform="gpu", workspace=megabyte).bytes
        without = gradient_peak(product, shape, (rows,), platform="gpu", workspace=Workspace(0, 0)).bytes
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
    assert peak_bytes(program, "gpu", Workspace(0, 0)) == 2 * 8 * 32 * 4


def test_an_inner_product_of_vectors_takes_no_workspace():
    """XLA hands cuBLAS no product of two vectors (`IsCublasSupportedMatMul`: neither operand a matrix),
    however long, so it takes no workspace, where a matrix times that vector does."""
    vector, matrix = jax.ShapeDtypeStruct((64,), jnp.float32), jax.ShapeDtypeStruct((64, 64), jnp.float32)

    def inner(params, other):
        return params @ other, ()

    def matrix_vector(params, other):
        return jnp.sum(other @ params), ()

    for loss, other, workspace in ((inner, vector, 0), (matrix_vector, matrix, 1 << 20)):
        megabyte = Workspace(1 << 20, 1 << 20)
        with_workspace = gradient_peak(loss, vector, (other,), platform="gpu", workspace=megabyte).bytes
        without = gradient_peak(loss, vector, (other,), platform="gpu", workspace=Workspace(0, 0)).bytes
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
        assert peak_bytes(program, platform, Workspace(0, 0)) == 64 * 32 * 4
    assert bounded_below(program, "gpu") and not bounded_below(program, "cpu")


def test_a_gradient_states_its_peak_a_bound_from_below_where_a_nested_ragged_product_runs():
    """A gradient whose program runs a ragged product on a GPU, in a loop's body as an expert stack's
    scan holds one, or in an interpreted Pallas kernel, whose interpreter's program XLA compiles, states
    its peak a bound from below (`Peak.exact` False); on a CPU, or with no ragged product, its peak is
    the program's own."""
    rows, sizes = jax.ShapeDtypeStruct((64, 16), jnp.float32), jax.ShapeDtypeStruct((4,), jnp.int32)
    weights = jax.ShapeDtypeStruct((2, 4, 16, 16), jnp.float32)

    def experts(weights, rows, sizes):
        def layer(carried, kernel):
            return jax.lax.ragged_dot(carried, kernel, sizes), None
        return jnp.sum(jax.lax.scan(layer, rows, weights)[0]), ()

    def dense(weights, rows, sizes):
        return jnp.sum(rows @ weights[0, 0]), ()

    def interpreted(weights, rows, sizes):
        """A ragged product of the rows, no parameter's, in an interpreted kernel the loss reads."""
        from jax.experimental import pallas as pl

        def kernel(rows, kernels, sizes, out):
            out[...] = jax.lax.ragged_dot(rows[...], kernels[...], sizes[...])
        ragged = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((64, 16), jnp.float32), interpret=True)
        return jnp.sum(ragged(rows, jnp.ones((4, 16, 16)), sizes) * (rows @ weights[0, 0])), ()

    none = Workspace(0, 0)
    for ragged in (experts, interpreted):
        assert not gradient_peak(ragged, weights, (rows, sizes), platform="gpu", workspace=none).exact
    assert gradient_peak(experts, weights, (rows, sizes), platform="cpu", workspace=Workspace(0, 0)).exact
    assert gradient_peak(dense, weights, (rows, sizes), platform="gpu", workspace=Workspace(0, 0)).exact


def test_a_derived_peak_is_of_the_program_the_algebra_pass_makes_of_the_step():
    """The algebra pass rewrites the program a step compiles (`algebra.rewritten_program`,
    docs/design/algebra.md §6.7): two products of one operand summed become one product of it with the
    summed others, so the forward holds one 256 x 1024 product, not two, and the backward reads one. The
    derived peak and residuals are that program's (row 3445): below the unrewritten program's, which the
    pass turned off (`algebra.passes`) derives."""
    from dew import algebra
    from dew.training.footprint import derived_terms

    def loss(weights, x):
        shared = jnp.tanh(x)
        products = shared @ jnp.tanh(weights["first"]) + shared @ jnp.tanh(weights["second"])
        return jnp.sum(products * products), ()

    weights = {name: jax.ShapeDtypeStruct((64, 1024), jnp.float32) for name in ("first", "second")}
    x = jax.ShapeDtypeStruct((256, 64), jnp.float32)

    def derived():
        residuals, peak = derived_terms(loss, weights, (x,), platform="gpu", workspace=Workspace(0, 0))
        return residuals, peak.bytes

    rewritten = derived()
    with algebra.passes(on=False):
        plain = derived()
    product = 256 * 1024 * 4
    assert rewritten[0] <= plain[0] - product and rewritten[1] <= plain[1] - product, (rewritten, plain)

