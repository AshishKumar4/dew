"""An MLP's NVFP4 projections chained under a delayed-scaling recipe (`projections.nvfp4_chain`,
`projections.recipe_chain`, row 12721): the up projection's epilogue scales its sums by both tensor scales
and quantizes their activation at the hidden activation's delayed tensor scale as the down projection's rows
(`gemm.nvfp4_block_finished`, `quantized.nvfp4_epilogue`), bitwise the projections apart at the same
scales."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import COMPILES_ON_CPU, INTERPRETS
from reference_error import with_gradients

from dew.algebra import bits
from dew.nn.kernels import formats, gemm, generation, projections
from dew.nn.kernels.templates.gemm import quantized


def _swiglu(gate, up):
    """A gated MLP's SwiGLU of its rounded gate and up, in fp32."""
    return jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)


def _same(fused, separate, names) -> None:
    for name, got, want in zip(names, fused, separate, strict=True):
        assert got.dtype == want.dtype, (name, got.dtype, want.dtype)
        unsigned = np.dtype(f"uint{8 * got.dtype.itemsize}")
        np.testing.assert_array_equal(np.asarray(got).view(unsigned), np.asarray(want).view(unsigned),
                                      err_msg=name)


def _operands(seed: int, rows: int, gated: bool):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.normal(size=(rows, 256)), jnp.bfloat16)
    matrices = [jnp.asarray(rng.normal(size=(256, 384)) / 16, jnp.bfloat16) for _ in range(1 + gated)]
    down = jnp.asarray(rng.normal(size=(384, 256)) / 16, jnp.bfloat16)
    cotangent = jnp.asarray(rng.normal(size=(rows, 256)), jnp.bfloat16)
    return x, matrices, down, cotangent


def _scales(x):
    """Delayed tensor scales a step's histories hold, not the step's own: the rows' at twice their largest
    magnitude, the hidden activation's at a scale that clips its largest blocks (delayed scaling's)."""
    return (bits.nvfp4_tensor_scale(2 * projections.nvfp4_amax(x)), jnp.float32(2.0 ** -9))


@pytest.mark.parametrize("gated", [False, True], ids=["ungated", "gated"])
@pytest.mark.parametrize("rows", [100, 256])
def test_a_delayed_nvfp4_recipe_chain_is_its_recipe_projections_apart(rows, gated):
    """Under a run's NVFP4 recipe with delayed tensor scaling, an MLP's chain (`projections.recipe_chain`, its
    `tensor_scales` the rows' and the hidden activation's) of any rows count, laid in whole 128-row blocks and
    the padding dropped: its value and every gradient bitwise the `recipe_projection` calls apart at the same
    scales with the activation between them in XLA (row 12721); a delayed chain refuses to run without its
    scales, and a current-scaled NVFP4 recipe takes no chain (None), which the projections apart take."""
    from dew.nn.precision import DTYPE_PRESETS, NVFP4, LowPrecision, Numerics, projection_role

    x, matrices, down, cotangent = _operands(12721 + rows, rows, gated)
    scales = _scales(x)
    activation = _swiglu if gated else jax.nn.gelu

    def chained(x, *matrices):
        *gate, up, down = matrices
        return projections.recipe_chain(x, up, down, activation, out_dtype=jnp.bfloat16,
                                        gate=gate[0] if gate else None, tensor_scales=scales)

    def apart(x, *matrices):
        *firsts, down = matrices

        def projected(rows, matrix, scale):
            return projections.recipe_projection(rows, matrix, out_dtype=jnp.bfloat16, tensor_scale=scale)

        hidden = activation(*(projected(x, matrix, scales[0]) for matrix in firsts)).astype(jnp.bfloat16)
        return projected(hidden, down, scales[1])

    delayed = Numerics(DTYPE_PRESETS["bf16-mixed"],
                       low_precision=LowPrecision(recipe="nvfp4", tensor_scaling="delayed"))
    with generation.run_precision(delayed), projection_role("mlp"):
        fused = jax.jit(with_gradients(chained))(x, *matrices, down, cotangent)
        separate = jax.jit(with_gradients(apart))(x, *matrices, down, cotangent)
        with pytest.raises(ValueError):
            projections.recipe_chain(x, matrices[-1], down, activation, out_dtype=jnp.bfloat16)
    names = ("value", "d_x", *(f"d_{name}" for name in ("gate", "up")[2 - len(matrices):]), "d_down")
    _same(fused, separate, names)
    with generation.run_precision(NVFP4), projection_role("mlp"):
        assert projections.recipe_chain(x, matrices[-1], down, activation, out_dtype=jnp.bfloat16) is None


@INTERPRETS
@pytest.mark.parametrize("gated", [False, True], ids=["ungated", "gated"])
def test_a_delayed_nvfp4_chain_through_the_template_is_its_projections_apart(gated):
    """An MLP's NVFP4 projections chained at delayed tensor scales (`projections.nvfp4_chain`) through the
    template named, interpreted as a B200's: the up projection's block-scaled tcgen05 product taking an
    epilogue that scales its sums by both tensor scales and quantizes their activation (with a gate, its
    rounded product a side tile) as E2M1 codes and E4M3 block scales, four 16-column blocks a 64-column
    chunk stored transposed, which the down projection reads as stored; every NVFP4 product the template's,
    and its value and every gradient bitwise the projections apart (`nvfp4_projection`) at the same scales
    with the activation between them in XLA (row 12721)."""
    x, matrices, down, cotangent = _operands(12731 + gated, 256, gated)
    matrices, down = [matrix[None] for matrix in matrices], down[None]
    scales = _scales(x)
    activation = _swiglu if gated else jax.nn.gelu
    group_blocks = jnp.asarray([2], jnp.int32)

    def chained(x, *matrices):
        *gate, up, down = matrices
        return projections.nvfp4_chain(x, up, down, activation, out_dtype=jnp.bfloat16, tensor_scales=scales,
                                       gate=gate[0] if gate else None, interpret_on_cpu=True)

    def apart(x, *matrices):
        *firsts, down = matrices

        def projected(rows, matrix, scale):
            return projections.nvfp4_projection(rows, matrix, group_blocks, out_dtype=jnp.bfloat16,
                                                tensor_scale=scale, interpret_on_cpu=True)

        hidden = activation(*(projected(x, matrix, scales[0]) for matrix in firsts)).astype(jnp.bfloat16)
        return projected(hidden, down, scales[1])

    with generation.assumed("sm100"), generation.chosen(nvfp4_block="mosaic"), generation.routes() as routes:
        fused = jax.jit(with_gradients(chained))(x, *matrices, down, cotangent)
        separate = jax.jit(with_gradients(apart))(x, *matrices, down, cotangent)
    names = ("value", "d_x", *(f"d_{name}" for name in ("gate", "up")[2 - len(matrices):]), "d_down")
    _same(fused, separate, names)
    assert {route.kernel for route in routes if route.choice == "nvfp4_block"} == {"mosaic"}


def _finished_up_projection():
    """A gated MLP's NVFP4 up projection at delayed tensor scales (Llama 3 8B's 4096 x 14336, 1024 rows)
    through the template named: its sums scaled, SwiGLU of its gate's rounded product beside them, and the
    hidden activation quantized in its epilogue (`gemm.nvfp4_block_finished`, `quantized.nvfp4_epilogue`)."""
    def product(x, gate, w_up):
        with generation.chosen(nvfp4_block="mosaic"):
            rows = formats.NvFp4Block.quantized(x, jnp.float32(2.0 ** -6))
            return gemm.nvfp4_block_finished(
                rows, formats.NvFp4Block.quantized(w_up), jnp.asarray([8], jnp.int32),
                epilogue=quantized.nvfp4_epilogue(_swiglu, jnp.bfloat16),
                out_dtypes=(jnp.bfloat16, jnp.uint8, jnp.float8_e4m3fn), epilogue_operands=(gate,),
                epilogue_scalars=(jnp.float32(2.0 ** -9),), out_forms=(None, None, (16, True)), dense=True)

    return product, (jax.ShapeDtypeStruct((1024, 4096), jnp.float32),
                     jax.ShapeDtypeStruct((1024, 14336), jnp.bfloat16),
                     jax.ShapeDtypeStruct((1, 14336, 4096), jnp.float32))


@COMPILES_ON_CPU
def test_a_delayed_nvfp4_up_projection_compiles_for_a_b200_within_its_resources():
    """The fused up projection of a delayed NVFP4 chain (its block-scaled tcgen05 product's epilogue scaling
    its sums, taking a gate's tile and quantizing the hidden activation, its block scales four 16-column
    blocks a 64-column chunk stored transposed), compiled by Mosaic for a compile-only B200 on this CPU
    through ptxas (tests/kernel_compile.py): a template kernel within its shared and tensor memory, no
    register spilled (row 12721)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    case = KernelCase("nvfp4 chain up projection", _finished_up_projection)
    found = compile_mosaic_kernels([case], "sm100")
    assert found and all(record.get("kernels") for record in found), found
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))
