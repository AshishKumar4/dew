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


def test_a_zero_gradient_is_held_from_the_programs_start_where_its_gradients_are_held():
    """The gradient of a parameter the loss reads none of is a broadcast zero JAX writes at the backward's
    end, which XLA makes ahead of the work, held to its reader: where the step holds every gradient to its
    end (`consumed` false, a global clip's), it is held through the backward's peak, one buffer of the
    parameter's size more than where the parameter is read, one for every parameter of its shape, as
    XLA computes equal constants once (row 4089: Qwen3.5-4B's multi-token head's and vision tower's zero
    gradients, 312 of 25 shapes, held as 0.48 GiB at its step's peak)."""
    # Activations of 8 MiB a layer make the backward's peak, past which the gradients alone are held.
    rows = jax.ShapeDtypeStruct((4096, 512), jnp.float32)
    # Two unread parameters of one shape: their zero gradients are one constant, which XLA computes once.
    params = {"used": jax.ShapeDtypeStruct((512, 512), jnp.float32),
              "unread": jax.ShapeDtypeStruct((128, 128), jnp.float32),
              "also_unread": jax.ShapeDtypeStruct((128, 128), jnp.float32)}

    def loss(params, rows):
        return jnp.sum(jnp.tanh(jnp.tanh(rows @ params["used"]) @ params["used"])), ()

    held = gradient_peak(loss, params, (rows,), platform="gpu", workspace=Workspace(0, 0)).bytes
    read = gradient_peak(loss, {"used": params["used"]}, (rows,), platform="gpu",
                         workspace=Workspace(0, 0)).bytes
    assert held == read + 128 * 128 * 4, (held, read)

    # So where the weight gradients run after the backward (`sunk`, `late_weight_gradients`), its peak at a
    # cotangent of 64 MiB no weight gradient reads, which the zero is held through as well.
    def widened(params, rows):
        return jnp.sum(jnp.tanh(jnp.tile(rows @ params["used"], (1, 8)))), ()

    held = gradient_peak(widened, params, (rows,), platform="gpu", workspace=Workspace(0, 0),
                         deferred=True).bytes
    read = gradient_peak(widened, {"used": params["used"]}, (rows,), platform="gpu",
                         workspace=Workspace(0, 0), deferred=True).bytes
    assert held == read + 128 * 128 * 4, (held, read)


def test_a_loops_zero_carry_is_made_where_the_loop_needs_it():
    """A constant the program's work reads, a loop's zero carry, is made where the loop needs it, as XLA
    makes it, not ahead of the work as a constant the program returns alone (`hoisted`): a 64 MiB carry
    the derivation held from the start counted beside the activations before the loop (row 4089)."""
    from dew.training.footprint import hoisted

    rows = jax.ShapeDtypeStruct((4096, 4096), jnp.float32)

    def looped(rows):
        warm = jnp.tanh(rows) @ rows
        total, _ = jax.lax.scan(lambda held, row: (held + row, None), jnp.zeros((4096,), jnp.float32), warm)
        return total

    program = jax.make_jaxpr(looped)(rows).jaxpr
    assert [equation.primitive for equation in hoisted(program).eqns] == [
        equation.primitive for equation in program.eqns]


def test_a_tied_weights_cast_twice_and_held_by_a_barrier_holds_one_cast():
    """A tied head's weights cast for the embedding, held through an optimization barrier, and cast again
    for the head are one cast, as XLA computes equal casts once (`HloCSE`), held in one buffer the barrier
    and the head share: one bf16 copy of the weights where the derivation held two (Qwen3.5-4B's fused
    tiled head, 1.18 GiB, row 4089)."""
    from dew.training.compiled_program import compiled
    from dew.training.footprint import deduplicated, peak_bytes

    # Weights read untransposed, so no layout of theirs is a product's to fold.
    weights = jax.ShapeDtypeStruct((1024, 4096), jnp.float32)
    rows = jax.ShapeDtypeStruct((64, 1024), jnp.bfloat16)

    def tied(weights, rows):
        embedded = jax.lax.optimization_barrier(weights.astype(jnp.bfloat16))
        head = weights.astype(jnp.bfloat16)
        return (rows @ embedded) + ((2 * rows) @ head)

    def once(weights, rows):
        cast = jax.lax.optimization_barrier(weights.astype(jnp.bfloat16))
        return (rows @ cast) + ((2 * rows) @ cast)

    peaks = [peak_bytes(deduplicated(compiled(jax.make_jaxpr(function)(weights, rows).jaxpr, "cuda")), "gpu",
                        Workspace(0, 0)) for function in (tied, once)]
    assert peaks[0] == peaks[1], peaks


def test_two_callbacks_of_one_factory_are_each_their_own():
    """Equal equations are one where XLA computes them once, which it never does for two callbacks: JAX
    lowers each to a custom call of its own host callback's index (`emit_python_callback`,
    jax/_src/callback.py), and two closures of one factory share the module and qualified name their
    parameters are keyed by (`parameter_key`), so each callback holds its own result (review of row 4089)."""
    from dew.training.compiled_program import compiled
    from dew.training.footprint import deduplicated

    def scaled(scale):
        def callback(rows):
            return rows * scale
        return callback

    rows = jax.ShapeDtypeStruct((256, 256), jnp.float32)

    def two(rows):
        first = jax.pure_callback(scaled(2.0), rows, rows)
        second = jax.pure_callback(scaled(3.0), rows, rows)
        return first @ second

    program = compiled(jax.make_jaxpr(two)(rows).jaxpr, "cuda")
    assert [equation.primitive.name for equation in program.eqns].count("pure_callback") == 2
    assert len(deduplicated(program).eqns) == len(program.eqns)


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


def test_a_scatter_into_zeros_holds_its_result_alone():
    """A scatter fuses the operands it reads, as XLA fuses a scatter's producers into it (an input fusion,
    `IsInputFusibleScatter`, xla/service/gpu/gpu_fusible.cc), its base computed into its result: an
    embedding's gradient, a scatter-add of its rows' cotangents into zeros, holds the gradient alone, where
    the derivation held the zeros beside it (Qwen3.5-4B's tied embedding under 'full', 2.37 GiB over XLA's
    4.76 at its step's peak: row 4089)."""
    from dew.training.footprint import peak_bytes

    vocab, width, tokens = 4096, 256, 512
    ids = jax.ShapeDtypeStruct((tokens,), jnp.int32)
    cotangents = jax.ShapeDtypeStruct((tokens, width), jnp.float32)

    def gradient(ids, cotangents):
        return jnp.zeros((vocab, width), jnp.float32).at[ids].add(2.0 * cotangents)

    program = jax.make_jaxpr(gradient)(ids, cotangents).jaxpr
    assert "scatter-add" in {equation.primitive.name for equation in program.eqns}
    assert peak_bytes(program, "gpu", Workspace(0, 0)) == vocab * width * 4


def test_a_tied_embeddings_gradient_holds_its_heads_product_alone():
    """A tied embedding's gradient, its head's product summed with the scatter of its rows' cotangents
    into zeros, is the product's buffer alone: XLA folds the sum into the scatter (`A + scatter(B, I, U) =>
    scatter(A + B, I, U)`, `A + 0 => A`, `AlgebraicSimplifierVisitor::HandleAdd`) and the scatter writes
    in the product's buffer (`CanShareOperandBufferWithUser`), so the gradient holds what the product
    alone does, where the derivation held the product, the scatter and their sum (Qwen3.5-4B's step under
    'full', 7.1 GiB against XLA's 4.4 at its end: row 4089)."""
    from dew.training.footprint import peak_bytes, simplified

    vocab, width, tokens = 4096, 256, 512
    shapes = (jax.ShapeDtypeStruct((tokens,), jnp.int32), jax.ShapeDtypeStruct((tokens, width), jnp.float32),
              jax.ShapeDtypeStruct((tokens, vocab), jnp.float32),
              jax.ShapeDtypeStruct((tokens, width), jnp.float32))

    def head(ids, hidden, logits, rows):
        return jax.lax.dot_general(logits, hidden, (((0,), (0,)), ((), ())))

    def tied(ids, hidden, logits, rows):
        return head(ids, hidden, logits, rows) + jnp.zeros((vocab, width), jnp.float32).at[ids].add(rows)

    # And where the head's product is read through a view of its buffer (a scanned head's stacked carry).
    def viewed(ids, hidden, logits, rows):
        product = jax.lax.squeeze(jax.lax.reshape(head(ids, hidden, logits, rows), (1, vocab, width)), (0,))
        return product + jnp.zeros((vocab, width), jnp.float32).at[ids].add(rows)

    def peak(function):
        return peak_bytes(simplified(jax.make_jaxpr(function)(*shapes).jaxpr, "gpu"), "gpu", Workspace(0, 0))

    assert peak(tied) == peak(viewed) == peak(head)


def test_a_gradient_added_to_a_zero_carry_holds_the_gradient_alone():
    """A sum of a value and a zero is that value, as XLA's simplifier makes it (`A + 0 => A`,
    `AlgebraicSimplifierVisitor::HandleAdd`): a tiled head of one tile adds its weight gradient to its
    loop's zero carry, which its body runs inline, and holds the gradient alone, where the derivation held
    the sum beside it (row 4089)."""
    from dew.training.footprint import peak_bytes, simplified

    shapes = (jax.ShapeDtypeStruct((512, 4096), jnp.float32), jax.ShapeDtypeStruct((512, 256), jnp.float32))

    def product(logits, hidden):
        return jax.lax.dot_general(logits, hidden, (((0,), (0,)), ((), ())))

    def carried(logits, hidden):
        return jnp.zeros((4096, 256), jnp.float32) + product(logits, hidden)

    # And a zero read through a view and a cast, as a loop of one turn reads its zero carry unstacked.
    def unstacked(logits, hidden):
        (zeros,) = jnp.unstack(jnp.zeros((1, 4096, 256), jnp.bfloat16).astype(jnp.float32), axis=0)
        return zeros + product(logits, hidden)

    def peak(function):
        return peak_bytes(simplified(jax.make_jaxpr(function)(*shapes).jaxpr, "gpu"), "gpu", Workspace(0, 0))

    assert peak(carried) == peak(unstacked) == peak(product)


def test_a_scatter_whose_updates_read_its_base_holds_them_apart():
    """A scatter writes in its base's buffer only where its updates read none of it, and its updates
    computed from its base are held apart, as XLA fuses no producer reading a scatter's in-place operand
    into it (`CanEmitInputFusedScatter`, xla/service/gpu/gpu_fusible.cc): a product scattered into itself
    through a view holds the product and the scatter's result, and one scattered into itself ten times
    over holds the updates' ten copies beside them (review of row 4089)."""
    from dew.training.footprint import peak_bytes, simplified

    size = 4096 * 256
    shapes = (jax.ShapeDtypeStruct((512, size // 256), jnp.float32),
              jax.ShapeDtypeStruct((512, 256), jnp.float32))

    def product(logits, hidden):
        return jax.lax.dot_general(logits, hidden, (((0,), (0,)), ((), ())))

    def viewed(logits, hidden):
        flat = product(logits, hidden).reshape(size)
        indices = jnp.arange(size // 256, dtype=jnp.int32)
        return flat.at[indices * 256].add(flat.reshape(size // 256, 256)[:, 0])

    def repeated(logits, hidden):
        flat = product(logits, hidden).reshape(size)
        indices = jnp.arange(10 * size, dtype=jnp.int32) % size
        return flat.at[indices].add(jnp.concatenate([flat] * 10))

    def peak(function, *specs):
        return peak_bytes(simplified(jax.make_jaxpr(function)(*(specs or shapes)).jaxpr, "gpu"), "gpu",
                          Workspace(0, 0))

    assert peak(viewed) >= 2 * size * 4, peak(viewed)
    assert peak(repeated) >= (1 + 10 + 1) * size * 4, peak(repeated)

    # So where the base is the program's input, or a view of it: the updates' ten copies and the result.
    def into_input(flat):
        indices = jnp.arange(10 * size, dtype=jnp.int32) % size
        return flat.at[indices].add(jnp.concatenate([flat] * 10))

    def into_view(square):
        flat = square.reshape(size)
        indices = jnp.arange(10 * size, dtype=jnp.int32) % size
        return flat.at[indices].add(jnp.concatenate([flat] * 10))

    assert peak(into_input, jax.ShapeDtypeStruct((size,), jnp.float32)) >= (10 + 1) * size * 4
    assert peak(into_view, jax.ShapeDtypeStruct((size // 256, 256), jnp.float32)) >= (10 + 1) * size * 4


def test_a_loops_carry_summed_with_a_product_holds_no_product_beside_it():
    """A sum of a cuBLAS product and a value of its type is the product's own, as XLA's GEMM rewriter folds
    the sum into the GEMM as its bias (`FuseMatrixBiasAdd`, its beta 1), in a loop's body too: a tiled
    head's weight gradient adding each tile's product to its loop's carry holds what a loop writing each
    product into its carry holds, where the derivation held each tile's product beside the carry
    (Qwen3.5-4B's step under 'full' at two microbatches, 2.37 GiB a tile: row 4089)."""
    from dew.training.footprint import peak_bytes, simplified

    width, vocab, tokens, tiles = 256, 4096, 512, 4
    shapes = (jax.ShapeDtypeStruct((tiles, tokens, width), jnp.float32),
              jax.ShapeDtypeStruct((tiles, tokens, vocab), jnp.float32))

    def product(tile):
        hidden, cotangents = tile
        return jax.lax.dot_general(hidden, cotangents, (((0,), (0,)), ((), ())))

    def summed(hidden, cotangents):
        return jax.lax.scan(lambda carried, tile: (carried + product(tile), None),
                            jnp.zeros((width, vocab), jnp.float32), (hidden, cotangents))[0]

    def written(hidden, cotangents):
        return jax.lax.scan(lambda carried, tile: (product(tile), None),
                            jnp.zeros((width, vocab), jnp.float32), (hidden, cotangents))[0]

    def peak(function):
        return peak_bytes(simplified(jax.make_jaxpr(function)(*shapes).jaxpr, "gpu"), "gpu", Workspace(0, 0))

    assert peak(summed) == peak(written)


def test_a_products_operand_computed_from_a_held_value_frees_it():
    """A product's operand that a fused chain computes from a value held behind a barrier is a kernel of its
    own, run before the product, past which the held value no other instruction reads is freed: a head's
    weight gradient reading the logits' cotangent, computed from the logits a recomputed head holds,
    holds what one reading a cotangent held itself holds, where the derivation held the logits and the
    cotangent at once (Qwen3.5-4B's step at four microbatches, 0.95 GiB over XLA's 7.42: row 4089)."""
    from dew.training.footprint import peak_bytes, simplified

    tokens, width, vocab = 256, 512, 4096
    shapes = (jax.ShapeDtypeStruct((tokens, width), jnp.float32),
              jax.ShapeDtypeStruct((width, vocab), jnp.float32))

    def cotangent(logits):
        return (jnp.exp(logits.astype(jnp.float32)) * 2.0).astype(jnp.bfloat16)

    def gradient(cotangents, hidden):
        return jax.lax.dot_general(cotangents, hidden.astype(jnp.bfloat16), (((0,), (0,)), ((), ())),
                                   preferred_element_type=jnp.float32)

    def from_logits(hidden, weights):
        logits = jax.lax.optimization_barrier((hidden @ weights).astype(jnp.bfloat16))
        return gradient(cotangent(logits), hidden)

    def held_cotangent(hidden, weights):
        return gradient(jax.lax.optimization_barrier(cotangent(hidden @ weights)), hidden)

    def peak(function):
        return peak_bytes(simplified(jax.make_jaxpr(function)(*shapes).jaxpr, "gpu"), "gpu", Workspace(0, 0))

    assert peak(from_logits) == peak(held_cotangent)


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

