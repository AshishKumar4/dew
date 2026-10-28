"""Check 2's native certificates for the fused template kernels (ledger row 2084, kernel-templates.md §9):
each native kernel's output held element by element to the exact products of its operands, on the GPU
that compiles it, never to another implementation's error. Link 1: the grouped GEMM template's plain
forward and input gradient, each a sum of exact 16-bit products a tensor core accumulates in fp32 by
truncating adds and rounds once to its output (`certified.native_dot_holds`). The rule's validation: its
nulls (Gaussian and same-sign draws certify) and its faults (a lost block of products, a row read under
the wrong group, a partial sum rounded to bf16 mid-depth), each refused where it lands."""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from certified import native_dot_holds
from generation_support import generations
from op_certificates import BFLOAT16, FLOAT32

from dew.nn.kernels import generation
from dew.nn.kernels.templates.gemm import template

pytestmark = pytest.mark.gpu(reason="a native kernel, certified on the GPU that compiles it")

SIZES = (0, 70, 1, 129, 100, 0)
ROWS, DEPTH, WIDTH = sum(SIZES), 128, 128
SEEDS = 4
"""The draws each certificate holds: check 2 is per element, so a draw is a thirty-thousand-element witness
and a few cover the kernel's tiles, group boundaries and both signs."""


def bf16(array) -> np.ndarray:
    """`array`'s nearest bf16 values, as float64."""
    return np.asarray(np.asarray(array).astype(ml_dtypes.bfloat16), np.float64)


def draw(seed: int, signed: bool):
    """Rows, each group's matrix and an output cotangent, every value a bf16 one: Gaussian, or of one sign,
    whose truncating adds all err one way."""
    rng = np.random.default_rng([2084, seed])

    def values(shape):
        return bf16(rng.normal(size=shape) if signed else rng.uniform(0.5, 2.0, size=shape))

    return (values((ROWS, DEPTH)), values((len(SIZES), DEPTH, WIDTH)) / np.sqrt(DEPTH),
            values((ROWS, WIDTH)))


def groups(sizes) -> np.ndarray:
    """Each row's group."""
    return np.repeat(np.arange(len(sizes)), sizes)


def forward_products(x, w, sizes) -> np.ndarray:
    """`[rows, width, depth]`: each output element's exact products, its row times its group's column."""
    return x[:, None, :] * np.swapaxes(w[groups(sizes)], 1, 2)


def input_gradient_products(dy, w, sizes) -> np.ndarray:
    """`[rows, depth, width]`: each input gradient element's exact products, the row's cotangent times its
    group's matrix row."""
    return dy[:, None, :] * w[groups(sizes)]


def forward_of(x, w, sizes, out_dtype=jnp.bfloat16) -> np.ndarray:
    """The template's native forward of the bf16 rows and matrices, its output in `out_dtype`."""
    return np.asarray(jax.jit(lambda rows: template.gemm(
        rows, jnp.asarray(w, jnp.bfloat16), jnp.asarray(sizes, jnp.int32), out_dtype=out_dtype))(
        jnp.asarray(x, jnp.bfloat16)), np.float64)


def input_gradient_of(x, w, dy, sizes) -> np.ndarray:
    """The template's native input gradient of the bf16 cotangent, through its own derivative."""
    def pulled(rows, cotangent):
        _, pull = jax.vjp(lambda rows: template.gemm(rows, jnp.asarray(w, jnp.bfloat16),
                                                     jnp.asarray(sizes, jnp.int32), out_dtype=jnp.bfloat16),
                          rows)
        return pull(cotangent)[0]

    pulled = jax.jit(pulled)(jnp.asarray(x, jnp.bfloat16), jnp.asarray(dy, jnp.bfloat16))
    return np.asarray(pulled, np.float64)


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
@pytest.mark.parametrize("signed", [True, False], ids=["gaussian", "same sign"])
def test_the_grouped_gemm_templates_forward_and_input_gradient_certify_natively(generation_name, signed):
    """Every element of the native forward and input gradient is a value its exact products admit
    (`native_dot_holds`), the forward's in bf16 and in fp32: the null of row 2084's link 1."""
    with generation.assumed(generation_name):
        for seed in range(SEEDS):
            x, w, dy = draw(seed, signed)
            products = forward_products(x, w, SIZES)
            for out_dtype, out in ((jnp.bfloat16, BFLOAT16), (jnp.float32, FLOAT32)):
                assert np.all(native_dot_holds(forward_of(x, w, SIZES, out_dtype), products, out,
                                               flushes=False)), (seed, out.name)
            assert np.all(native_dot_holds(input_gradient_of(x, w, dy, SIZES),
                                           input_gradient_products(dy, w, SIZES), BFLOAT16,
                                           flushes=False)), seed


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_the_native_certificate_refuses_a_lost_block_a_misrouted_row_and_a_narrow_partial(generation_name):
    """The rule's faults, each the native kernel run around a defect and certified against the true
    operands: a lost block of sixteen products (one MMA's depth) in one group's rows, refused at those
    rows and nowhere else; a row read under its neighbour group's matrix, refused at that row alone; and,
    in an fp32 output, whose grid shows it, a partial sum over half the depth rounded to bf16 before the
    rest is added, refused nearly everywhere."""
    with generation.assumed(generation_name):
        x, w, _ = draw(0, signed=True)
        products = forward_products(x, w, SIZES)
        rows = groups(SIZES) == 3
        lost = x.copy()
        lost[np.ix_(rows, np.arange(32, 48))] = 0
        held = native_dot_holds(forward_of(lost, w, SIZES), products, BFLOAT16, flushes=False)
        assert np.mean(held[rows]) < 0.05 and np.all(held[~rows])
        shifted = (0, 71, 0, 129, 100, 0)
        held = native_dot_holds(forward_of(x, w, shifted), products, BFLOAT16, flushes=False)
        moved = groups(SIZES) != groups(shifted)
        assert moved.sum() == 1 and np.mean(held[moved]) < 0.05 and np.all(held[~moved])
        halves = (np.concatenate([x[:, :64] * keep, x[:, 64:] * (1 - keep)], axis=1) for keep in (1.0, 0.0))
        first, rest = (forward_of(half, w, SIZES, jnp.float32) for half in halves)
        narrow = np.asarray(np.float32(bf16(first)) + np.float32(rest), np.float64)
        held = native_dot_holds(narrow, products, FLOAT32, flushes=False)
        assert np.mean(held) < 0.05
