"""A gated MLP's gate and up projections as one product over one wide matrix
(`templates.gemm.wide.gated_product`, formulation (C)): their columns interleaved a run at a time, one
accumulator, the epilogue writing their gated product alone at half its tile's width
(`quantized.Form.narrowed`)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import COMPILES_ON_CPU, INTERPRETS
from reference_error import assert_as_exact, with_gradients

from dew.nn import activations
from dew.nn.kernels import generation
from dew.nn.kernels.templates.gemm import quantized, tiling, wide

SEEDS = 8
ROWS, DEPTH, HIDDEN = 128, 128, 256


def _swiglu(gate, up):
    """The gated MLP's SwiGLU hook (`activations.gated_hook`), as `gated_mlp` traces it."""
    return activations.gated_hook("swiglu", None)(gate, up)


def _draw(seed):
    rng = np.random.default_rng(seed)
    bf16 = functools.partial(jnp.asarray, dtype=jnp.bfloat16)
    return (bf16(rng.standard_normal((ROWS, DEPTH))), bf16(rng.standard_normal((DEPTH, HIDDEN)) / 8),
            bf16(rng.standard_normal((DEPTH, HIDDEN)) / 8), bf16(rng.standard_normal((ROWS, HIDDEN))))


def _apart(x, gate, up, wide_dtype=jnp.float32, out=jnp.bfloat16):
    """The gate and up projections apart, each rounded to the rows' dtype, and their SwiGLU: the unfused
    program the wide product stands in for."""
    def projected(matrix):
        return jnp.dot(x.astype(wide_dtype), matrix.astype(wide_dtype)).astype(
            x.dtype if out == jnp.bfloat16 else out)
    return _swiglu(projected(gate), projected(up)).astype(out)


def test_a_wide_matrix_holds_its_gate_and_up_columns_a_run_at_a_time():
    """`wide.interleaved` lays a gate's and an up projection's columns alternately a `GROUP` at a time, the
    gate's first, over any leading stack, and `wide.deinterleaved` gives both back."""
    rng = np.random.default_rng(0)
    gate, up = (jnp.asarray(rng.standard_normal((3, 16, 4 * wide.GROUP)), jnp.float32) for _ in range(2))
    joined = wide.interleaved(gate, up)
    assert joined.shape == (3, 16, 8 * wide.GROUP)
    for run in range(4):
        np.testing.assert_array_equal(joined[..., 2 * run * wide.GROUP:(2 * run + 1) * wide.GROUP],
                                      gate[..., run * wide.GROUP:(run + 1) * wide.GROUP])
        np.testing.assert_array_equal(joined[..., (2 * run + 1) * wide.GROUP:(2 * run + 2) * wide.GROUP],
                                      up[..., run * wide.GROUP:(run + 1) * wide.GROUP])
    back = wide.deinterleaved(joined)
    np.testing.assert_array_equal(back[0], gate)
    np.testing.assert_array_equal(back[1], up)
    with pytest.raises(ValueError):
        wide.interleaved(gate[..., :wide.GROUP - 1], up[..., :wide.GROUP - 1])


@INTERPRETS
@pytest.mark.parametrize("generation_name", ["sm80", "sm90", "sm100"])
@pytest.mark.parametrize("saved", [False, True], ids=["recomputed", "saved"])
def test_a_wide_gated_product_holds_check_1_against_its_projections_apart(generation_name, saved):
    """The wide product through the template named, interpreted as each GPU family runs it: its gated
    product (and, saved, its rounded wide product) and the rows', gate's and up's gradients hold check 1
    against the two projections apart and their SwiGLU in XLA, from float64; the launch the template's."""
    def dew(x, gate, up):
        out = wide.gated_product(x, wide.interleaved(gate, up), _swiglu, out_dtype=jnp.bfloat16,
                                 saved=saved, name="gate_up" if saved else None, interpret_on_cpu=True)
        return out[1] if saved else out

    plain = _apart
    truth = functools.partial(_apart, wide_dtype=jnp.float64, out=jnp.float64)
    with generation.assumed(generation_name), generation.chosen(grouped_matmul="mosaic"):
        with generation.routes() as seen:
            jax.eval_shape(dew, *_draw(0)[:3])
        assert ("grouped_matmul", "mosaic") in {(route.choice, route.kernel) for route in seen}, seen
        assert_as_exact(with_gradients(dew), jax.jit(with_gradients(plain)), _draw,
                        f"{generation_name} wide gated {ROWS}x{DEPTH}x{2 * HIDDEN}", seeds=SEEDS,
                        truth_fn=with_gradients(truth))


def test_a_saved_wide_product_is_its_gate_and_up_interleaved():
    """A saved wide product's first output (`wide.gated_product`'s `saved`) is the product of the rows and
    the wide matrix rounded, its gate's and up's columns interleaved as the matrix's are, and its second their
    SwiGLU, so the reverse pass reads the hook's operands from the first."""
    x, gate, up, _ = _draw(1)
    joined = wide.interleaved(gate, up)
    kept, hidden = wide.gated_product(x, joined, _swiglu, out_dtype=jnp.bfloat16, saved=True)
    np.testing.assert_array_equal(kept, jnp.dot(x, joined, preferred_element_type=jnp.float32).astype(
        jnp.bfloat16))
    np.testing.assert_array_equal(hidden, _swiglu(*wide.deinterleaved(kept)).astype(jnp.bfloat16))


@pytest.mark.parametrize(("family", "block_n", "refused"), [
    ("wgmma", 128, False), ("wgmma", 192, False), ("wgmma", 96, True), ("mma.sync", 64, False),
    ("tcgen05", 256, False)])
def test_a_narrowed_output_takes_tiles_holding_whole_pairs_of_runs(family, block_n, refused):
    """A launch stores the wide product's narrowed output (`quantized.forms_refusal`) at a tile whose
    columns, and every piece its issuer reads out at once (tcgen05's 64), hold whole pairs of runs
    (`wide.UNIT`), on a family that places a narrowed output (`stores_narrowed`); else it refuses the tile,
    which the derivation passes over."""
    tile = tiling.Tiles(128, block_n, 64, 3, block_n, family=family)
    refusal = quantized.forms_refusal((quantized.Form(2, narrowed=True),), tile, dense=True)
    assert (refusal is not None) is refused, refusal


def test_a_tpu_launch_stores_no_narrowed_output():
    """A TPU core's launch stages each output tile whole and places none narrower (`stores_narrowed`), so it
    refuses the wide product's narrowed output, which then runs XLA's program."""
    tile = tiling.Tiles(128, 128, 128, 2, 128, family="mxu")
    assert quantized.forms_refusal((quantized.Form(2, narrowed=True),), tile, dense=True) is not None


def _wide_gated_4b(saved):
    """Qwen3.5-4B's gate and up as one wide product (8192 x 2560 by 2560 x 18432) through the template
    named, its rounded product stored beside the gated one where `saved` (a training forward's)."""
    def product(x, joined):
        with generation.chosen(grouped_matmul="mosaic"):
            return wide.gated_product(x, joined, _swiglu, out_dtype=jnp.bfloat16, saved=saved)

    return product, (jax.ShapeDtypeStruct((8192, 2560), jnp.bfloat16),
                     jax.ShapeDtypeStruct((2560, 2 * 9216), jnp.bfloat16))


@COMPILES_ON_CPU
@pytest.mark.parametrize("generation_name", ["sm80", "sm90", "sm100"])
@pytest.mark.parametrize("saved", [False, True], ids=["forward", "saved"])
def test_a_wide_gated_product_compiles_within_each_familys_resources(generation_name, saved):
    """The 4B's wide gated product (`_wide_gated_4b`), its narrowed output alone or beside the whole rounded
    product, compiled by Mosaic for each compile-only GPU family through ptxas (tests/kernel_compile.py): a
    template kernel within its shared memory, no register spilled, its epilogue's split of the accumulator a
    selection of registers."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    found = compile_mosaic_kernels([KernelCase("wide gated 4b", lambda: _wide_gated_4b(saved))],
                                   generation_name)
    assert found and not mosaic_failures(found), "\n".join(mosaic_failures(found))
    assert all(record.get("kernels") for record in found), found
