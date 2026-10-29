"""Check 2's native certificates for the fused template kernels (ledger row 2084, kernel-templates.md §9):
each native kernel's output held element by element to the exact products of its operands, on the GPU
that compiles it, never to another implementation's error. Link 1: the grouped GEMM template's plain
forward and input gradient, each a sum of exact 16-bit products a tensor core accumulates in fp32 by
truncating adds and rounds once to its output (`certified.native_dot_holds`). Link 2: the rest of the
template's native launches, Mosaic GPU's and Triton's (CUTLASS's wheel is not installed on the lanes'
GPUs): the kernel gradient; fp32 operands at the default precision, whose products are of their TF32
values (`certified.tf32_operand`); and the exact sums a precision above the default declares for an fp32
output, held a depth step at a time (`certified.promoted_step_holds`), each step's output the launch's own
on its operands zeroed from the next step on. The rule's validation: its nulls (Gaussian and same-sign
draws certify) and its faults, each refused where it lands: a lost block of products, a row read under the
wrong group, a partial sum rounded to bf16 mid-depth; bf16 products where TF32 is declared and the
reverse, and operands truncated to TF32 where the tensor cores round them; and a step's partial added by
the tensor cores' truncating adds where IEEE sums are declared."""

import itertools

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from certified import (
    assert_refused,
    native_dot_ends,
    native_dot_holds,
    promoted_step_ends,
    promoted_step_holds,
    separated,
    tf32_operand,
    truncating_step_ends,
)
from generation_support import generations, runs
from op_certificates import BFLOAT16, FLOAT32, TFLOAT32, rounded, truncated

from dew.nn.kernels import generation, grouped_matmul, tuning
from dew.nn.kernels.templates.gemm import template

pytestmark = pytest.mark.gpu(reason="a native kernel, certified on the GPU that compiles it")

SIZES = (0, 70, 1, 129, 100, 0)
ROWS, DEPTH, WIDTH = sum(SIZES), 128, 128
SHIFTED = (0, 71, 0, 129, 100, 0)
"""`SIZES` with group 2's one row read under group 1: a misrouted row."""
LOST = slice(136, 152)
"""Sixteen of group 3's rows, all in its second depth step on either kernel: a lost block."""
SEEDS = 4
"""The draws each certificate holds: check 2 is per element, so a draw is a thirty-thousand-element witness
and a few cover the kernel's tiles, group boundaries and both signs."""
FALL = 2.0 ** -12
"""How far a falling draw's operand sits below its first depth step past it (`scaled`): each later step's
partial a few thousandths of the accumulator, so the exact-sums witness of its add, whose partial's
allowance (γ of a step's 64 products at 2^-23, of their magnitudes) falls below the accumulator's last
place, decides the add's rounding."""


LOST_SHARE = 0.9
"""The least share of the elements a lost block of sixteen rows or depth columns reaches whose interval on
the call's products is separated from that on the products the kernel was given (`assert_refused`)."""
MOVED_SHARE = 0.9
"""The same of the elements a row read under its neighbour group reaches."""
NARROW_SHARE = 0.75
"""The same of an fp32 forward's elements where a partial sum over half the depth is rounded to bf16 before
the rest is added, on Gaussian draws, whose partials sometimes cancel to a bf16 value's neighbourhood."""
TRUNCATED_SHARE = 0.4
"""The same of a kernel gradient's elements past each group's first depth step, on a falling draw of one sign
(`FALL`), where its default launch adds each step's products by truncating adds and exact sums are
declared."""
TF32_SHARES = {True: (0.9, 0.9, 0.8), False: (0.9, 0.9, 0.95)}
"""The same, on Gaussian draws and on draws of one sign, of an fp32 forward's elements where its operands'
bf16 values are multiplied and TF32 is declared, its TF32 values where bf16 is, and its truncated TF32 values
where they are rounded."""


def cells(*pairs) -> list:
    """`(kernel, generation)` pairs as parameters, each marked `runs` for its generation."""
    return [pytest.param(kernel, name, marks=runs(name), id=f"{kernel}-{name}") for kernel, name in pairs]


PLAIN = cells(*((kernel, name) for kernel in ("mosaic_gpu", "triton") for name in ("sm80", "sm90")))
"""The template's native launches of 16-bit products on the lanes' GPUs, each forced (`generation.chosen`),
refusing rather than falling back to another route."""
TRITON = cells(("triton", "sm80"), ("triton", "sm90"))
"""The launches of fp32 operands, Triton's alone (Mosaic GPU's take 16-bit ones), and of a forward's exact
sums (Mosaic GPU's forward takes none)."""
EXACT = cells(("mosaic_gpu", "sm90"), ("triton", "sm80"), ("triton", "sm90"))
"""The launches of a kernel gradient's exact sums: Triton's, and Mosaic GPU's on Hopper alone (Ampere's
bundle counts no running sum in its registers, `mosaic.Family.counts_hooks`, so it refuses them)."""
GRADIENTS = {"mosaic_gpu": grouped_matmul.MosaicTgmm, "triton": grouped_matmul.Tgmm}
"""Each kernel's kernel-gradient launch."""
STEPS = {grouped_matmul.MosaicTgmm: ("block_k", True), grouped_matmul.Tgmm: ("block_m", False),
         grouped_matmul.Gmm: ("block_k", False)}
"""Each launch of exact sums: its tile's field that is its depth step's width, and whether its steps are
the global blocks of that many rows a group's rows meet (Mosaic GPU's kernel gradient,
`mosaic_gpu._group_blocks`) rather than runs from the contraction's first (Triton's: a kernel gradient's
from its group's first row, `ragged_dot.tgmm`; a forward's from the first column, `ragged_dot.gmm`)."""


def bf16(array) -> np.ndarray:
    """`array`'s nearest bf16 values, as float64."""
    return np.asarray(np.asarray(array).astype(ml_dtypes.bfloat16), np.float64)


def draw(seed: int, signed: bool, dtype=ml_dtypes.bfloat16):
    """Rows, each group's matrix and an output cotangent, every value one of `dtype`'s: Gaussian, or of one
    sign, whose truncating adds all err one way."""
    rng = np.random.default_rng([2084, seed])

    def held(array):
        return np.asarray(np.asarray(array).astype(dtype), np.float64)

    def values(shape):
        return held(rng.normal(size=shape) if signed else rng.uniform(0.5, 2.0, size=shape))

    # The matrices and the cotangent scaled before they are rounded: the products certified are of the values
    # the kernel reads, and their grid is finer than the rows', so no sum of one sign is exact by the grids
    # alone (rows and cotangent of [0.5, 2) sum a kernel gradient's products exactly in fp32).
    return (values((ROWS, DEPTH)), held(values((len(SIZES), DEPTH, WIDTH)) / np.sqrt(DEPTH)),
            held(values((ROWS, WIDTH)) / np.sqrt(WIDTH)))


def groups(sizes) -> np.ndarray:
    """Each row's group."""
    return np.repeat(np.arange(len(sizes)), sizes)


def spans(sizes) -> list[tuple[int, int]]:
    """Each group's first row and its end."""
    return list(itertools.pairwise(np.cumsum((0, *sizes)).tolist()))


def forward_products(x, w, sizes) -> np.ndarray:
    """`[rows, width, depth]`: each output element's exact products, its row times its group's column."""
    return x[:, None, :] * np.swapaxes(w[groups(sizes)], 1, 2)


def input_gradient_products(dy, w, sizes) -> np.ndarray:
    """`[rows, depth, width]`: each input gradient element's exact products, the row's cotangent times its
    group's matrix row."""
    return dy[:, None, :] * w[groups(sizes)]


def rows_products(x, dy) -> np.ndarray:
    """`[depth, width, rows]`: the exact products of rows of `x` and of the cotangent `dy` a kernel gradient
    sums, each element's its rows' column of `x` times theirs of `dy`."""
    return x.T[:, None, :] * dy.T[None, :, :]


def kernel_gradient_products(x, dy, sizes) -> list[np.ndarray]:
    """Each group's kernel gradient's exact products (`rows_products` of its rows)."""
    return [rows_products(x[start:end], dy[start:end]) for start, end in spans(sizes)]


def kernel_gradient_ends(products: list) -> tuple[np.ndarray, np.ndarray]:
    """`[groups, depth, width]`: the least and the greatest fp32 value of each element of a kernel gradient
    its groups' exact `products` admit (`native_dot_ends`)."""
    ends = [native_dot_ends(group, FLOAT32, flushes=False) for group in products]
    return np.stack([low for low, _ in ends]), np.stack([high for _, high in ends])


def products_apart(declared, given, out) -> np.ndarray:
    """Where the native-dot intervals in `out` (`native_dot_ends`) of a call's `declared` products and of the
    products a kernel run around a defect was `given` are separated (`separated`)."""
    return separated(*(native_dot_ends(products, out, flushes=False) for products in (declared, given)))


def kernel_gradient_holds(native, products: list, out) -> np.ndarray:
    """`[groups, depth, width]`: whether each element of a native kernel gradient is a value its group's exact
    `products` admit (`native_dot_holds`)."""
    return np.stack([native_dot_holds(each, group, out, flushes=False)
                     for each, group in zip(native, products, strict=True)])


def forward_of(x, w, sizes, out_dtype=jnp.bfloat16, *, dtype=jnp.bfloat16, precision=None) -> np.ndarray:
    """The template's native forward of the rows and matrices in `dtype`, its output in `out_dtype`, its
    product declared at `precision`, traced at JAX's own default precision as a model runs it (`as_run`)."""
    with as_run():
        return np.asarray(jax.jit(lambda rows: template.gemm(
            rows, jnp.asarray(w, dtype), jnp.asarray(sizes, jnp.int32), out_dtype=out_dtype,
            precision=precision))(jnp.asarray(x, dtype)), np.float64)


def as_run():
    """JAX's own default matmul precision, as a model calls the template: the test lane sets 'highest' for
    every product that states none, at which the template takes no fp32 output, and the certificate is of
    the kernel a model runs."""
    return jax.default_matmul_precision(None)


def input_gradient_of(x, w, dy, sizes, *, dtype=jnp.bfloat16) -> np.ndarray:
    """The template's native input gradient of the cotangent in `dtype`, through its own derivative."""
    def pulled(rows, cotangent):
        _, pull = jax.vjp(lambda rows: template.gemm(rows, jnp.asarray(w, dtype),
                                                     jnp.asarray(sizes, jnp.int32), out_dtype=dtype),
                          rows)
        return pull(cotangent)[0]

    with as_run():
        pulled = jax.jit(pulled)(jnp.asarray(x, dtype), jnp.asarray(dy, dtype))
    return np.asarray(pulled, np.float64)


def kernel_gradient(sizes, out_dtype=jnp.bfloat16, *, dtype=jnp.bfloat16, precision=None):
    """The template's kernel gradient of rows and a cotangent in `dtype`, through its own derivative: of a
    matrix in `out_dtype`, the gradient's (an fp32 master's of 16-bit products), the product declared at
    `precision` (above the default, an fp32 gradient's exact sums)."""
    def pulled(rows, cotangent):
        matrices = jnp.zeros((len(sizes), rows.shape[1], cotangent.shape[1]), out_dtype)
        _, pull = jax.vjp(lambda matrices: template.gemm(rows, matrices, jnp.asarray(sizes, jnp.int32),
                                                         out_dtype=dtype, precision=precision), matrices)
        return pull(cotangent)[0]

    return pulled


def kernel_gradient_of(x, dy, sizes, out_dtype=jnp.bfloat16, *, dtype=jnp.bfloat16,
                       precision=None) -> np.ndarray:
    """The template's native kernel gradient (`kernel_gradient`) of `x` and `dy`, traced as a model runs it
    (`as_run`)."""
    with as_run():
        native = jax.jit(kernel_gradient(sizes, out_dtype, dtype=dtype, precision=precision))(
            jnp.asarray(x, dtype), jnp.asarray(dy, dtype))
    return np.asarray(native, np.float64)


def assert_products_certify(x, w, dy, read, outputs: tuple, label) -> None:
    """Every element of the native forward and kernel gradient of `x`, `w` and `dy`, in each of `outputs`
    (a dtype and its format, the first the operands' own), and of the input gradient, in the operands' own,
    is a value the exact products of the operands' values as the tensor cores read them (`read`) admit
    (`native_dot_holds`)."""
    dtype = outputs[0][0]
    rx, rw, rdy = (read(value) for value in (x, w, dy))
    forward, gradient = forward_products(rx, rw, SIZES), kernel_gradient_products(rx, rdy, SIZES)
    for out_dtype, out in outputs:
        assert np.all(native_dot_holds(forward_of(x, w, SIZES, out_dtype, dtype=dtype), forward, out,
                                       flushes=False)), (label, out.name)
        assert np.all(kernel_gradient_holds(kernel_gradient_of(x, dy, SIZES, out_dtype, dtype=dtype),
                                            gradient, out)), (label, out.name)
    assert np.all(native_dot_holds(input_gradient_of(x, w, dy, SIZES, dtype=dtype),
                                   input_gradient_products(rdy, rw, SIZES), outputs[0][1],
                                   flushes=False)), label


def depth_step(launch: type, fn, *operands) -> tuple[int, bool]:
    """The width of each depth step of the one `launch` of `fn(*operands)` and whether its steps are aligned
    (`STEPS`), from the tile it takes as it traces (`tuning.seen`): a power of two from 16, which the kernels
    keep, and unsplit, as exact sums are."""
    with as_run(), tuning.seen() as launches:
        jax.jit(fn).trace(*operands)
    [tile] = [config for (op, _), (config, _) in zip(launches, launches.taken, strict=True)
              if isinstance(op, launch)]
    field, aligned = STEPS[launch]
    width = getattr(tile, field)
    assert width >= 16 and width & (width - 1) == 0 and getattr(tile, "splits", 1) == 1, tile
    return width, aligned


def steps_of(sizes, width: int, aligned: bool) -> list[list[slice]]:
    """Each group's depth steps, each the rows it sums: `width` rows each from the group's first, or
    (`aligned`) the global blocks of `width` rows its rows meet, clipped to them; none for an empty group."""
    return [[slice(max(start, at), min(end, at + width))
             for at in range(start // width * width if aligned else start, end, width)]
            if end > start else [] for start, end in spans(sizes)]


def gradient_steps(kernel: str, precision="highest") -> list[list[slice]]:
    """Each group's depth steps (`steps_of`) of the fp32 kernel gradient `kernel` launches on `SIZES` at
    `precision` (above the default, of exact sums)."""
    def rows(width):
        return jax.ShapeDtypeStruct((ROWS, width), jnp.bfloat16)

    gradient = kernel_gradient(SIZES, jnp.float32, precision=precision)
    return steps_of(SIZES, *depth_step(GRADIENTS[kernel], gradient, rows(DEPTH), rows(WIDTH)))


def scaled(x, steps, first: int, factor: float) -> np.ndarray:
    """`x` with each group's operand from its depth step `first` on (`steps`: each group's steps' indices into
    it) scaled by `factor`, exactly: to +0, or by a power of two (`FALL`)."""
    x = x.copy()
    for taken in steps:
        for step in taken[first:]:
            x[step] = x[step] * factor if factor else 0.0
    return x


def states_of(run, x, steps) -> list[np.ndarray]:
    """The native outputs after each group's first j depth steps, from j = 0 (zeros) to the most any group
    takes (the output): `run` of `x` with each group's operand from its step j on zeroed, the same launch on
    the same shapes, as a step of exact zeros leaves its accumulator bit-identical
    (`test_a_depth_step_of_exact_zeros_leaves_the_native_accumulator_bit_identical`)."""
    outputs = [run(scaled(x, steps, j, 0.0)) for j in range(1, max(map(len, steps)) + 1)]
    return [np.zeros_like(outputs[-1]), *outputs]


def kernel_gradient_states(x, dy, steps, *, sizes=SIZES, precision="highest") -> list[np.ndarray]:
    """The native fp32 kernel gradient of `x` and `dy` launched on `sizes` at `precision` after each depth
    step of `steps` (`states_of`)."""
    def run(rows):
        return kernel_gradient_of(rows, dy, sizes, jnp.float32, precision=precision)

    return states_of(run, x, steps)


def stepped_products(x, dy):
    """A kernel gradient's depth step's exact products, of the rows of `x` and `dy` it sums
    (`rows_products`), by the step's group and rows."""
    return lambda _, rows: rows_products(x[rows], dy[rows])


def stepwise(states, steps, contract) -> list[list[np.ndarray]]:
    """[group][step]: `contract(before, after, group, j, step)` at each group's depth step j,
    `states[j][group]` the output after its first j steps (`states_of`)."""
    return [[contract(states[j][group], states[j + 1][group], group, j, step) for j, step in enumerate(taken)]
            for group, taken in enumerate(steps)]


def witnessed(states, steps, products_of) -> list[list[np.ndarray]]:
    """[group][step]: whether each element of the native output after the group's depth step is the IEEE sum
    of the one before it and the step's own partial (`promoted_step_holds`), `products_of(group, step)` the
    step's exact products."""
    return stepwise(states, steps, lambda before, after, group, _, step: promoted_step_holds(
        before, after, products_of(group, step), flushes=False))


def apart_from(states, steps, products_of, fault_ends) -> list[list[np.ndarray]]:
    """[group][step]: where the step witness's interval on the call's products (`products_of(group, step)`,
    `promoted_step_ends`) is separated from a fault's, `fault_ends(before, group, j, step)` (`separated`)."""
    return stepwise(states, steps, lambda before, _, group, j, step: separated(
        promoted_step_ends(before, before, products_of(group, step), flushes=False),
        fault_ends(before, group, j, step)))


def forward_witnessed(x, w, steps) -> list:
    """The step witness (`witnessed`) of the native fp32 forward of `x` and `w` with exact sums, against the
    exact products of each depth step's columns (`steps`: one group's, of the whole depth)."""
    def run(rows):
        return forward_of(rows, w, SIZES, jnp.float32, precision="highest")[None]

    return witnessed(states_of(run, x, steps), steps,
                     lambda _, step: forward_products(x[step], w[:, step[1]], SIZES))


def flat(lists, first: int = 0) -> np.ndarray:
    """[group][step] arrays as one, each group's from its step `first` on."""
    return np.concatenate([step.ravel() for taken in lists for step in taken[first:]])


def everywhere(held, but=()) -> bool:
    """Whether every witness of `held` ([group][step]) holds at every element but those at `but`'s steps."""
    return all(np.all(step) for group, taken in enumerate(held) for j, step in enumerate(taken)
               if (group, j) not in but)


@pytest.mark.parametrize("kernel, generation_name", PLAIN)
@pytest.mark.parametrize("signed", [True, False], ids=["gaussian", "same sign"])
def test_the_grouped_gemm_templates_products_certify_natively(kernel, generation_name, signed):
    """Every element of the native forward, input gradient and kernel gradient of 16-bit operands is a value
    its exact products admit (`native_dot_holds`), the forward's and the kernel gradient's in bf16 and in
    fp32: the null of row 2084's links 1 and 2."""
    with generation.assumed(generation_name), generation.chosen(grouped_matmul=kernel):
        for seed in range(SEEDS):
            assert_products_certify(*draw(seed, signed), lambda value: value,
                                    ((jnp.bfloat16, BFLOAT16), (jnp.float32, FLOAT32)), seed)


@pytest.mark.parametrize("kernel, generation_name", TRITON)
@pytest.mark.parametrize("signed", [True, False], ids=["gaussian", "same sign"])
def test_fp32_operands_certify_natively_as_products_of_their_tf32_values(kernel, generation_name, signed):
    """At the default precision an fp32 product is TF32 (src/dew/nn/kernels/templates/product.py): every
    element of the native forward, input gradient and kernel gradient of fp32 operands is a value the exact
    products of their TF32 values admit (`tf32_operand`, `native_dot_holds`)."""
    with generation.assumed(generation_name), generation.chosen(grouped_matmul=kernel):
        for seed in range(SEEDS):
            assert_products_certify(*draw(seed, signed, np.float32), tf32_operand, ((jnp.float32, FLOAT32),),
                                    seed)


@pytest.mark.parametrize("kernel, generation_name", EXACT)
def test_a_depth_step_of_exact_zeros_leaves_the_native_accumulator_bit_identical(kernel, generation_name):
    """What `states_of` reads each step's output by: group 3's native exact-sums kernel gradient with its rows
    from a depth step on zeroed is bit for bit its gradient with the group ending at that step (the rows past
    it read under group 4), the same launch on the same shapes, at each of its later steps."""
    with generation.assumed(generation_name), generation.chosen(grouped_matmul=kernel):
        steps = gradient_steps(kernel)
        x, _, dy = draw(0, signed=True)
        start, end = spans(SIZES)[3]
        for step in steps[3][1:]:
            ending = (*SIZES[:3], step.start - start, SIZES[4] + end - step.start, SIZES[5])
            zeroed = scaled(x, [[slice(step.start, end)]], 0, 0.0)
            native, ended = (kernel_gradient_of(rows, dy, sizes, jnp.float32, precision="highest")[3]
                             for rows, sizes in ((zeroed, SIZES), (x, ending)))
            assert np.array_equal(np.float32(native).view(np.uint32), np.float32(ended).view(np.uint32)), step


@pytest.mark.parametrize("kernel, generation_name", EXACT)
@pytest.mark.parametrize("signed", [True, False], ids=["gaussian", "same sign"])
def test_the_kernel_gradients_exact_sums_certify_natively_a_step_at_a_time(kernel, generation_name, signed):
    """Above the default precision an fp32 kernel gradient's sums are exact (product.py's `promoted`): each
    group's native gradient after each depth step is the IEEE sum of the one before and the step's own partial
    (`promoted_step_holds`), at the step the launch takes (64 rows on both kernels), on draws at one scale and
    falling past each group's first step (`FALL`), where the witness decides each add's rounding."""
    with generation.assumed(generation_name), generation.chosen(grouped_matmul=kernel):
        steps = gradient_steps(kernel)
        for seed in range(SEEDS):
            x, _, dy = draw(seed, signed)
            for rows in (x, scaled(x, steps, 1, FALL)):
                held = witnessed(kernel_gradient_states(rows, dy, steps), steps, stepped_products(rows, dy))
                assert everywhere(held), seed


@pytest.mark.parametrize("kernel, generation_name", TRITON)
@pytest.mark.parametrize("signed", [True, False], ids=["gaussian", "same sign"])
def test_the_forwards_exact_sums_certify_natively_a_depth_step_at_a_time(kernel, generation_name, signed):
    """Triton's forward takes exact sums too: each native fp32 output after each depth step (64 of the depth)
    is the IEEE sum of the one before and the step's own partial, on draws at one scale and falling past the
    first step."""
    def forward(x, w):
        return template.gemm(x, w, jnp.asarray(SIZES, jnp.int32), out_dtype=jnp.float32, precision="highest")

    with generation.assumed(generation_name), generation.chosen(grouped_matmul=kernel):
        width, _ = depth_step(grouped_matmul.Gmm, forward, jax.ShapeDtypeStruct((ROWS, DEPTH), jnp.bfloat16),
                              jax.ShapeDtypeStruct((len(SIZES), DEPTH, WIDTH), jnp.bfloat16))
        steps = [[np.s_[:, at:at + width] for at in range(0, DEPTH, width)]]
        for seed in range(SEEDS):
            x, w, _ = draw(seed, signed)
            for rows in (x, scaled(x, steps, 1, FALL)):
                assert everywhere(forward_witnessed(rows, w, steps)), seed


@pytest.mark.parametrize("kernel, generation_name", PLAIN)
def test_the_kernel_gradients_certificate_refuses_a_lost_block_and_a_misrouted_row(kernel, generation_name):
    """The kernel gradient's faults, in an fp32 output, whose grid shows them, each the native kernel run
    around a defect, certified against the true operands and refused where it lands (`assert_refused`): a
    lost block of sixteen of group 3's rows, at more than `LOST_SHARE` of that group's gradient, and a row
    read under its neighbour group, at more than `MOVED_SHARE` of both groups' gradients; every other group's
    holds."""
    with generation.assumed(generation_name), generation.chosen(grouped_matmul=kernel):
        x, _, dy = draw(0, signed=True)
        products = kernel_gradient_products(x, dy, SIZES)
        for ran, sizes, faulty, share in ((scaled(x, [[LOST]], 0, 0.0), SIZES, [3], LOST_SHARE),
                                          (x, SHIFTED, [1, 2], MOVED_SHARE)):
            held = kernel_gradient_holds(kernel_gradient_of(ran, dy, sizes, jnp.float32), products, FLOAT32)
            apart = separated(kernel_gradient_ends(products),
                              kernel_gradient_ends(kernel_gradient_products(ran, dy, sizes)))
            assert_refused(held[faulty], apart[faulty], share)
            assert np.all(np.delete(held, faulty, axis=0))


@pytest.mark.parametrize("kernel, generation_name", EXACT)
def test_the_step_witness_refuses_a_truncating_add_a_lost_block_and_a_misrouted_row(kernel, generation_name):
    """The exact-sums witness's faults, each the native kernel gradient run around a defect, witnessed a step
    at a time against the true operands and refused where it lands (`assert_refused`): the default launch,
    whose tensor cores add each step's products into the accumulator itself by truncating adds
    (`truncating_step_ends`), where exact sums are declared, on a falling draw of one sign (`FALL`), at more
    than `TRUNCATED_SHARE` of the elements past each group's first step, about those whose exact sum lies
    past the midpoint between two fp32 values, which a truncation leaves below and a rounding above; a lost
    block of sixteen of group 3's rows, at more than `LOST_SHARE` of the step that loses them, and nowhere
    else; and a row read under its neighbour group, at more than `MOVED_SHARE` of both groups' first steps
    (its groups each summing their own rows, in whatever steps, within the native-dot contract, exact sums'
    adds within its truncating ones), and nowhere past those groups."""
    with generation.assumed(generation_name), generation.chosen(grouped_matmul=kernel):
        steps = gradient_steps(kernel)
        # The default launch takes the same steps, so its outputs a step at a time are its accumulator's.
        assert gradient_steps(kernel, precision=None) == steps
        x, _, dy = draw(0, signed=False)
        falling = scaled(x, steps, 1, FALL)
        products_of = stepped_products(falling, dy)
        states = kernel_gradient_states(falling, dy, steps, precision=None)
        held = witnessed(states, steps, products_of)
        apart = apart_from(states, steps, products_of, lambda before, group, _, step: truncating_step_ends(
            before, products_of(group, step), flushes=False))
        assert not np.any(flat(held) & flat(apart))
        assert_refused(flat(held, 1), flat(apart, 1), TRUNCATED_SHARE)
        x, _, dy = draw(0, signed=True)
        products_of, lost = stepped_products(x, dy), scaled(x, [[LOST]], 0, 0.0)
        states = kernel_gradient_states(lost, dy, steps)
        held = witnessed(states, steps, products_of)
        apart = apart_from(states, steps, products_of, lambda before, group, _, step: promoted_step_ends(
            before, before, stepped_products(lost, dy)(group, step), flushes=False))
        [losing] = [j for j, step in enumerate(steps[3])
                    if step.start <= LOST.start and LOST.stop <= step.stop]
        assert_refused(held[3][losing], apart[3][losing], LOST_SHARE)
        assert everywhere(held, but={(3, losing)})
        states = kernel_gradient_states(x, dy, steps, sizes=SHIFTED)
        held = witnessed(states, steps, products_of)
        kept = [kernel_gradient_products(scaled(x, steps, j, 0.0), dy, SHIFTED)
                for j in range(1, len(states))]
        apart = apart_from(states, steps, products_of, lambda _, group, j, __: native_dot_ends(
            kept[j][group], FLOAT32, flushes=False))
        assert not np.any(flat(held) & flat(apart))
        assert_refused(np.stack([held[1][0], held[2][0]]), np.stack([apart[1][0], apart[2][0]]), MOVED_SHARE)
        assert everywhere(held[3:])


@pytest.mark.parametrize("kernel, generation_name", TRITON)
@pytest.mark.parametrize("signed", [True, False], ids=["gaussian", "same sign"])
def test_the_tf32_certificate_refuses_bf16_products_tf32_where_bf16_is_declared_and_a_truncated_read(
        kernel, generation_name, signed):
    """The TF32 contract's faults, each the native forward of fp32 operands certified against products the
    call does not take and refused where it lands (`assert_refused`, `TF32_SHARES`): the operands rounded to
    bf16 first (bf16 products where TF32 is declared); the operands themselves (TF32 products) against their
    bf16 values' products, which a bf16 product declares (`gemm`'s `compute`, the MoE projection's); and the
    operands truncated to TF32 first, where the tensor cores round them to nearest. bf16 moves a product by
    up to 2^-8 of it, either way, and a truncation by up to 2^-10 of it toward zero, where the allowance is
    γ_128 at 2^-23 of the products' magnitudes: on draws of one sign, whose errors all go one way, the
    truncation is refused nearly everywhere."""
    with generation.assumed(generation_name), generation.chosen(grouped_matmul=kernel):
        x, w, _ = draw(0, signed, np.float32)

        def forward(*operands):
            return forward_of(*operands, SIZES, jnp.float32, dtype=jnp.float32)

        def read(left, right):
            """The products the kernel takes of operands it is given: of their TF32 values."""
            return forward_products(tf32_operand(left), tf32_operand(right), SIZES)

        tf32, wide = read(x, w), forward_products(bf16(x), bf16(w), SIZES)
        cases = (((bf16(x), bf16(w)), tf32), ((x, w), wide),
                 ((truncated(x, TFLOAT32), truncated(w, TFLOAT32)), tf32))
        for (operands, declared), share in zip(cases, TF32_SHARES[signed], strict=True):
            held = native_dot_holds(forward(*operands), declared, FLOAT32, flushes=False)
            assert_refused(held, products_apart(declared, read(*operands), FLOAT32), share)


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_the_native_certificate_refuses_a_lost_block_a_misrouted_row_and_a_narrow_partial(generation_name):
    """The rule's faults, each the native kernel run around a defect, certified against the true operands
    and refused where it lands (`assert_refused`): a lost block of sixteen products (one MMA's depth) in one
    group's rows, at more than `LOST_SHARE` of those rows, every other row holding; a row read under its
    neighbour group's matrix, at more than `MOVED_SHARE` of that row alone; and, in an fp32 output, whose
    grid shows it, a partial sum over half the depth rounded to bf16 before the rest is added, at more than
    `NARROW_SHARE` of the elements of Gaussian draws, wherever that rounding outweighs the sum's own."""
    # The Mosaic template itself, refusing rather than falling back to another route.
    with generation.assumed(generation_name), generation.chosen(grouped_matmul="mosaic_gpu"):
        x, w, _ = draw(0, signed=True)
        products = forward_products(x, w, SIZES)
        rows = groups(SIZES) == 3
        lost = x.copy()
        lost[np.ix_(rows, np.arange(32, 48))] = 0
        held = native_dot_holds(forward_of(lost, w, SIZES), products, BFLOAT16, flushes=False)
        assert_refused(held[rows], products_apart(products, forward_products(lost, w, SIZES), BFLOAT16)[rows],
                       LOST_SHARE)
        assert np.all(held[~rows])
        held = native_dot_holds(forward_of(x, w, SHIFTED), products, BFLOAT16, flushes=False)
        moved = groups(SIZES) != groups(SHIFTED)
        apart = products_apart(products, forward_products(x, w, SHIFTED), BFLOAT16)
        assert moved.sum() == 1 and np.all(held[~moved])
        assert_refused(held[moved], apart[moved], MOVED_SHARE)
        halves = tuple(np.concatenate([x[:, :64] * keep, x[:, 64:] * (1 - keep)], axis=1)
                       for keep in (1.0, 0.0))
        first, rest = (forward_of(half, w, SIZES, jnp.float32) for half in halves)
        narrow = np.asarray(np.float32(bf16(first)) + np.float32(rest), np.float64)
        held = native_dot_holds(narrow, products, FLOAT32, flushes=False)
        # The first half's bf16 value lies between its ends' bf16 values; the IEEE add is monotone in both.
        halved = native_dot_ends(forward_products(halves[0], w, SIZES), FLOAT32, flushes=False)
        narrowed = promoted_step_ends(*(rounded(end, np.zeros_like(end), BFLOAT16) for end in halved),
                                      forward_products(halves[1], w, SIZES), flushes=False)
        truth = native_dot_ends(products, FLOAT32, flushes=False)
        assert_refused(held, separated(truth, narrowed), NARROW_SHARE)
