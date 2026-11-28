"""The GEMM template's derivation at its edges: each degenerate input (no, one or odd rows, rows, a depth or a
width no tile or step divides, a shallow depth, a narrow width, one column, no depth or width) a cell in each
form of call through it (a plain product, matrices side by side, a pair, a prologue and an epilogue on their
sides), dense and grouped, on each GPU family's derivation as `generation.assumed` routes it (interpreted),
its outputs and its forward's, input's and matrices' gradients bitwise XLA's (`grouped_matmul='xla'`) over
small integers, whose sums every route makes exactly."""

import functools
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import runnable

from dew.nn.kernels import gemm, generation
from dew.nn.kernels.templates.hooks import Hook, Side

EDGES = {
    "no rows": (0, 64, 64),
    "one row": (1, 64, 64),
    "odd rows": (7, 64, 64),
    "rows no tile divides": (72, 64, 64),
    "depth no step divides": (64, 72, 64),
    "shallow depth": (64, 16, 64),
    "no depth": (64, 0, 64),
    "width no tile divides": (64, 64, 72),
    "narrow width": (64, 64, 8),
    "one column": (64, 64, 1),
    "no width": (64, 64, 0),
}
"""Each degenerate input of the derivation: its rows, depth and width."""

RUNS = tuple(name for name in ("sm80", "sm90", "sm100") if runnable(name))
"""The GPUs whose derivations a cell runs, each this process can (`generation_support.runnable`)."""

FORMS = ("plain", "beside", "paired", "hooked")
"""The forms of call a cell takes through its input: a product, matrices side by side (a second of half the
width, at least one column), a pair into two outputs, and a prologue on the depth with an epilogue on the
rows and the columns."""


def _scaled(x, scale):
    return x * scale[None, :]


def _biased(acc, bias, row_scale):
    return (acc + bias[None, :]) * row_scale[:, None]


def _groups(rows: int, dense: bool):
    """A grouped call's sizes over `rows`: three groups, the middle one empty and the first a third (every
    group empty where there are no rows); a dense call's none."""
    if dense:
        return None
    first = rows // 3
    return jnp.asarray([first, 0, rows - first], jnp.int32)


def _operands(rows: int, depth: int, width: int, groups: int, form: str, seed: int) -> dict:
    """Small integers in bf16 (rows, matrices, their sides and the outputs' cotangents), exact in every
    product and sum a route makes."""
    rng = np.random.default_rng(seed)

    def ints(*shape, low=-2, high=3):
        return jnp.asarray(rng.integers(low, high, size=shape), jnp.float32)

    widths = (width, max(width // 2, 1)) if form in ("beside", "paired") else (width,)
    if form == "paired":
        widths = (width, width)
    matrices = tuple(ints(groups, depth, each).astype(jnp.bfloat16) for each in widths)
    return {"rows": ints(rows, depth).astype(jnp.bfloat16), "matrices": matrices,
            "scale": ints(depth, low=1, high=3), "bias": ints(width), "row_scale": ints(rows, low=1, high=3),
            "cotangents": tuple(ints(rows, each) for each in widths)}


def _call(form: str, sizes, dense: bool):
    """The cell's call of `form`: its outputs from the rows, the matrices and the sides."""
    def call(rows, matrices, scale, bias, row_scale):
        held = tuple(matrix[0] for matrix in matrices) if dense else matrices
        options = {"out_dtype": jnp.float32, "interpret_on_cpu": True}
        if form == "plain":
            return (gemm.gemm(rows, held[0], sizes, **options),)
        if form == "beside":
            return tuple(gemm.gemm(rows, held[0], sizes, beside=held[1:], **options))
        if form == "paired":
            return tuple(gemm.gemm(rows, held[0], sizes, paired=held[1], out_dtypes=(jnp.float32,) * 2,
                                   **options))
        return (gemm.gemm(rows, held[0], sizes, prologue=Hook(_scaled, (Side(scale, ("depth",)),)),
                          epilogue=Hook(_biased, (Side(bias, ("columns",)), Side(row_scale, ("rows",)))),
                          **options),)

    return call


def _run(form: str, operands: dict, sizes, dense: bool, kernel: str | None, gpu: str):
    """The call's outputs and every operand's gradient under the outputs' cotangents, on `gpu` as `kernel`
    routes it (None: as derived)."""
    call = _call(form, sizes, dense)
    differentiated = (operands["rows"], operands["matrices"], operands["scale"], operands["bias"],
                      operands["row_scale"])
    chosen = {} if kernel is None else {"grouped_matmul": kernel}
    with generation.assumed(gpu), generation.chosen(**chosen):
        outs, pull = jax.vjp(call, *differentiated)
        return outs, pull(operands["cotangents"][:len(outs)])


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("edge", list(EDGES))
def test_a_degenerate_call_is_xlas(edge, form):
    """`edge`'s rows, depth and width in `form` (`FORMS`), dense and grouped (`_groups`), on an A100, an H100
    and a B200: the outputs and every gradient bitwise XLA's program's, the call taken by whichever route
    each GPU's derivation gives (a tile, or none and the next route)."""
    rows, depth, width = EDGES[edge]
    for dense in (True, False):
        sizes = _groups(rows, dense)
        label = f"{edge} {form} {'dense' if dense else 'grouped'}"
        operands = _operands(rows, depth, width, 1 if dense else 3, form, seed=zlib.crc32(label.encode()))
        reference = _run(form, operands, sizes, dense, "xla", "sm90")
        for gpu in RUNS:
            derived = _run(form, operands, sizes, dense, None, gpu)
            for got, expected in zip(jax.tree.leaves(derived), jax.tree.leaves(reference), strict=True):
                assert (got.shape, got.dtype) == (expected.shape, expected.dtype), f"{label} on {gpu}"
                np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(expected, np.float32),
                                              err_msg=f"{label} on {gpu}")


def test_a_product_of_no_groups_is_its_zeros_and_compiles_for_a_gpu():
    """A grouped product of a bank of no matrices (`[0, depth, width]`, group sizes `[0]`) over no rows,
    forward and both gradients: its zeros on this CPU, and its program compiled by XLA for a compile-only
    H100, where a ragged dot of no groups failed its compile (row 14112: a forward's legalization left its
    result unmaterialized, the kernel gradient's contraction aborted the process; a bank of no experts on a
    GH200). `precision.accumulated` forms no ragged dot of no groups."""
    from kernel_compile import lowered_for

    x, w = jnp.ones((0, 512), jnp.bfloat16), jnp.ones((0, 512, 128), jnp.bfloat16)
    sizes = jnp.zeros((0,), jnp.int32)

    def pulled(x, w):
        out, pull = jax.vjp(lambda x, w: gemm.gemm(x, w, sizes, out_dtype=jnp.float32), x, w)
        return (out, *pull(jnp.ones_like(out)))

    out, d_x, d_w = jax.jit(pulled)(x, w)
    for found, shape in ((out, (0, 128)), (d_x, (0, 512)), (d_w, (0, 512, 128))):
        assert found.shape == shape
        np.testing.assert_array_equal(np.asarray(found, np.float32), np.zeros(shape, np.float32))
    specs = tuple(jax.ShapeDtypeStruct(operand.shape, operand.dtype) for operand in (x, w))
    lowered_for(pulled, specs, "sm90").compile()


def test_a_float64_product_of_no_groups_stays_float64_outside_x64():
    """A ragged product of no groups (`precision.accumulated`'s, as `xla_products.ragged_product` binds it)
    over float64 rows and an empty float64 bank made in x64, multiplied after x64's scope ends, no preferred
    dtype: float64 zeros, as JAX types the product, not float32 ones its scope would have asked for (row
    14112's review)."""
    from dew.nn import precision
    from dew.nn.kernels import xla_products

    with jax.enable_x64(new_val=True):
        rows, bank = jnp.ones((8, 4), jnp.float64), jnp.ones((0, 4, 128), jnp.float64)
    sizes = jnp.zeros((0,), jnp.int32)
    out = precision.accumulated(xla_products.ragged_product(rows, bank), rows, bank, sizes)
    assert out.dtype == np.dtype(np.float64) and out.shape == (8, 128)
    np.testing.assert_array_equal(np.asarray(out), np.zeros((8, 128)))


@pytest.mark.mesh(devices=2)
def test_a_product_of_no_groups_keeps_its_explicit_placement_beside_an_auto_axis():
    """A ragged product of no groups whose empty bank an Explicit 'tensor' axis splits by its width, on a mesh
    that holds an Auto axis too, jitted: its zeros split by the width as the product would be, where a
    shape's evaluation dropped the placement beside the Auto axis (row 14112's review)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    from dew.nn.kernels import xla_products

    mesh = jax.make_mesh((2, 1), ("tensor", "data"), axis_types=(AxisType.Explicit, AxisType.Auto),
                         devices=jax.devices()[:2])
    with jax.set_mesh(mesh):
        bank = jax.device_put(jnp.ones((0, 4, 128), jnp.float32),
                              NamedSharding(mesh, PartitionSpec(None, None, "tensor")))
        out = jax.jit(lambda rows, bank: xla_products.xla_ragged_dot(rows, bank, jnp.zeros((0,), jnp.int32)))(
            jnp.ones((8, 4), jnp.float32), bank)
    assert out.sharding.spec == PartitionSpec(None, "tensor")
    np.testing.assert_array_equal(np.asarray(out), np.zeros((8, 128), np.float32))


def test_a_batched_ragged_product_of_no_groups_is_its_zeros_and_compiles_for_a_gpu():
    """A ragged dot batched over a leading axis (its group sizes `[batch, groups]`) of no groups, through
    `precision.accumulated`: its zeros, compiled by XLA for a compile-only H100, whose compile of the batched
    ragged dot of no groups failed as the plain one's did (row 14112's review: its group count the sizes'
    last axis)."""
    from kernel_compile import lowered_for

    from dew.nn import precision

    numbers = jax.lax.RaggedDotDimensionNumbers(dot_dimension_numbers=(((2,), (2,)), ((0,), (1,))),
                                                lhs_ragged_dimensions=(1,), rhs_group_dimensions=(0,))
    product = functools.partial(jax.lax.ragged_dot_general, ragged_dot_dimension_numbers=numbers)

    def batched(rows, bank):
        return precision.accumulated(product, rows, bank, jnp.zeros((2, 0), jnp.int32),
                                     preferred_element_type=jnp.float32)

    rows, bank = jnp.ones((2, 8, 4), jnp.float32), jnp.ones((0, 2, 4, 128), jnp.float32)
    np.testing.assert_array_equal(np.asarray(jax.jit(batched)(rows, bank)), np.zeros((2, 8, 128)))
    specs = tuple(jax.ShapeDtypeStruct(operand.shape, operand.dtype) for operand in (rows, bank))
    lowered_for(batched, specs, "sm90").compile()


def test_a_ragged_batch_product_of_no_groups_is_its_batched_dot():
    """A ragged dot whose rows' ragged axis is a batch axis (JAX's ragged-batch mode, a batched dot that reads
    no group sizes) at sizes of no groups, through `precision.accumulated` at F32_F32_F32: two [2, 1, 1]
    operands of 2s and 3s multiply to two 6s, and their gradients are 3s and 2s, not a product of no groups'
    zeros (train 1478's review of row 14112)."""
    from dew.nn import precision

    numbers = jax.lax.RaggedDotDimensionNumbers(dot_dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                                                lhs_ragged_dimensions=(0,), rhs_group_dimensions=())
    product = functools.partial(jax.lax.ragged_dot_general, ragged_dot_dimension_numbers=numbers,
                                precision=jax.lax.DotAlgorithmPreset.F32_F32_F32)

    def batched(rows, bank):
        return precision.accumulated(product, rows, bank, jnp.zeros((0,), jnp.int32),
                                     preferred_element_type=jnp.float32)

    rows, bank = jnp.full((2, 1, 1), 2.0, jnp.float32), jnp.full((2, 1, 1), 3.0, jnp.float32)
    def summed(*pair):
        return batched(*pair).sum()

    out, gradients = jax.jit(lambda *pair: (batched(*pair), jax.grad(summed, (0, 1))(*pair)))(rows, bank)
    np.testing.assert_array_equal(np.asarray(out), np.full((2, 1, 1), 6.0, np.float32))
    np.testing.assert_array_equal(np.asarray(gradients[0]), np.full((2, 1, 1), 3.0, np.float32))
    np.testing.assert_array_equal(np.asarray(gradients[1]), np.full((2, 1, 1), 2.0, np.float32))


def test_a_jitted_ragged_product_of_no_groups_reads_its_mode():
    """A ragged dot jitted before it reaches `precision.ragged` (its primitive bound inside the jit's program)
    at sizes of no groups: in its non-contracting mode its zeros, in its ragged-batch mode its batched dot,
    two 6s of two [2, 1, 1] operands of 2s and 3s (row 14112's review: its dimension numbers read at any
    depth of its program)."""
    from dew.nn import precision

    def jitted(numbers):
        return jax.jit(functools.partial(jax.lax.ragged_dot_general, ragged_dot_dimension_numbers=numbers),
                       static_argnames="preferred_element_type")

    rows_mode = jax.lax.RaggedDotDimensionNumbers(dot_dimension_numbers=(((1,), (1,)), ((), ())),
                                                  lhs_ragged_dimensions=(0,), rhs_group_dimensions=(0,))
    zeros = precision.ragged(jitted(rows_mode), jnp.ones((8, 4), jnp.float32),
                             jnp.ones((0, 4, 128), jnp.float32), jnp.zeros((0,), jnp.int32),
                             preferred_element_type=jnp.float32)
    np.testing.assert_array_equal(np.asarray(zeros), np.zeros((8, 128), np.float32))
    batch_mode = jax.lax.RaggedDotDimensionNumbers(dot_dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                                                   lhs_ragged_dimensions=(0,), rhs_group_dimensions=())
    batched = precision.ragged(jitted(batch_mode), jnp.full((2, 1, 1), 2.0, jnp.float32),
                               jnp.full((2, 1, 1), 3.0, jnp.float32), jnp.zeros((0,), jnp.int32),
                               preferred_element_type=jnp.float32)
    np.testing.assert_array_equal(np.asarray(batched), np.full((2, 1, 1), 6.0, np.float32))


def test_a_weakly_typed_product_of_no_groups_stays_weak():
    """A ragged product of no groups over weakly typed rows, bank and sizes (Python scalars' fills) outside
    x64, through `precision.accumulated`: its zeros weakly typed float32 as JAX types the product, so beside a
    strong bfloat16 value the sum is bfloat16, not float32 (row 14112's second review); the same operands
    multiplied inside x64 a weak float32 still, not the weak float64 a Python zero's fill took there (its
    third)."""
    from dew.nn import precision
    from dew.nn.kernels import xla_products

    rows, bank, sizes = jnp.full((1, 1), 1.0), jnp.full((0, 1, 1), 1.0), jnp.full((0,), 0)
    out = precision.accumulated(xla_products.ragged_product(rows, bank), rows, bank, sizes)
    assert out.shape == (1, 1)
    assert (out + jnp.ones((1, 1), jnp.bfloat16)).dtype == jnp.bfloat16
    # The same operands multiplied inside x64: still a weak float32, as the product's aval is.
    with jax.enable_x64(new_val=True):
        wide = precision.accumulated(xla_products.ragged_product(rows, bank), rows, bank, sizes)
    assert wide.dtype == jnp.float32 and wide.weak_type


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("weak", [False, True], ids=["strong", "weak"])
def test_an_eager_product_of_no_groups_lands_on_its_operands_mesh(weak):
    """A ragged product of no groups called eagerly outside any mesh context over a bank its two Explicit
    devices split by its width, the rows and sizes committed to the same mesh
    (`xla_products.placed_ragged_dot`): its zeros made on those devices at the product's placement, its
    columns split as the bank's (train 1484's review: zeros allocated at the abstract mesh were refused);
    over weakly typed operands, weak as JAX types the product, so beside bfloat16 the sum stays bfloat16
    (its confirm)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    from dew.nn.kernels import xla_products

    mesh = jax.make_mesh((2,), ("tensor",), (AxisType.Explicit,), devices=jax.devices()[:2])

    def filled(shape, value):
        return jnp.full(shape, value) if weak else jnp.full(shape, value, jnp.float32)

    rows = jax.device_put(filled((8, 4), 1.0), NamedSharding(mesh, PartitionSpec()))
    bank = jax.device_put(filled((0, 4, 128), 1.0), NamedSharding(mesh, PartitionSpec(None, None, "tensor")))
    sizes = jax.device_put(jnp.zeros((0,), jnp.int32), NamedSharding(mesh, PartitionSpec()))
    out = xla_products.placed_ragged_dot(rows, bank, sizes,
                                         preferred_element_type=None if weak else jnp.float32)
    if weak:
        assert out.weak_type and (out + jnp.ones((8, 128), jnp.bfloat16)).dtype == jnp.bfloat16
    placed = out.sharding
    assert isinstance(placed, NamedSharding) and placed.mesh == mesh and placed.spec[-1] == "tensor"
    np.testing.assert_array_equal(np.asarray(out), np.zeros((8, 128), np.float32))


@pytest.mark.mesh(devices=2)
def test_an_eager_product_of_no_groups_lands_on_its_operands_device():
    """A ragged product of no groups called eagerly over operands committed to the second of two devices,
    the first the default (`xla_products.placed_ragged_dot`): its zeros committed to its operands' device,
    where JAX places a product of them (row 14112's second review: they were made on the default device;
    its third: uncommitted there)."""
    from dew.nn.kernels import xla_products

    device = jax.devices()[1]
    rows, bank, sizes = jax.device_put((jnp.ones((8, 4), jnp.float32), jnp.ones((0, 4, 128), jnp.float32),
                                        jnp.zeros((0,), jnp.int32)), device)
    out = xla_products.placed_ragged_dot(rows, bank, sizes, preferred_element_type=jnp.float32)
    assert out.devices() == {device} and out.committed
    np.testing.assert_array_equal(np.asarray(out), np.zeros((8, 128), np.float32))
