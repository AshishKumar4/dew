"""A kernel call's outputs stated as its operands' axes (`dew.nn.kernels.templates.outputs`), read back off
its lowered call (`kernel_compile.stated_outputs`): what a compile's live buffers read of Dew's own
kernels. The toolkit's own launch's is in tests/test_mosaic_toolkit.py beside the walks it lowers."""

import jax
import jax.numpy as jnp
from kernel_compile import stated_outputs

from dew.nn.kernels import gemm, generation


def test_a_gemms_output_states_its_rows_and_width_as_its_operands_axes():
    """A dense GEMM lowered for an H100 on any host states its output's dimensions on its call
    (`outputs.OUTPUT_DIMS`): its rows its rows operand's first axis, its width its matrix's last, which a
    compile's live buffers read back to the step's tokens and the layer's width."""
    x = jax.ShapeDtypeStruct((4096, 256), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((256, 512), jnp.bfloat16)
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        lowered = jax.jit(lambda a, b: gemm.gemm(a, b, None, out_dtype=jnp.bfloat16)).trace(x, w).lower(
            lowering_platforms=("cuda",))
    ((_, (stated,)),) = stated_outputs(lowered)
    rows, width = stated
    assert rows == [[0, 0]] and [operand for operand, _ in width] == [1], stated
