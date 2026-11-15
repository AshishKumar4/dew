"""A kernel call's outputs stated as its operands' axes (`dew.nn.kernels.templates.outputs`), read back off
its lowered call (`generation_support.declared_outputs`): what a compile's live buffers read of Dew's own
kernels. The toolkit's own launch's is in tests/test_mosaic_toolkit.py beside the walks it lowers."""

import jax
import jax.numpy as jnp
import numpy as np
from generation_support import declared_outputs
from jax.sharding import Mesh

from dew.nn.kernels import gemm, generation, launches
from dew.nn.kernels.templates import outputs


def test_a_pallas_calls_outputs_stated_as_its_operands_axes_are_stated_on_its_call():
    """A Pallas call (`launches.pallas_call`, Triton's) whose output its operand's axes state, transposed,
    lowered for an H100 on any host: its shape is theirs, and its call states each dimension's axis
    (`outputs.OUTPUT_DIMS`), as a Mosaic launch's does."""
    def copied(x):
        def body(x_ref, o_ref):
            o_ref[...] = x_ref[...].T

        transposed = outputs.output(x.dtype, outputs.axis(x, 1), outputs.axis(x, 0))
        return launches.pallas_call(body, name="copied", grid=(1,), out_shape=transposed)(x)

    x = jax.ShapeDtypeStruct((16, 32), jnp.float32)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(copied).trace(x).lower(lowering_platforms=("cuda",))
    assert lowered.out_info.shape == (32, 16)
    assert declared_outputs(lowered) == [[[[[0, 1]], [[0, 0]]]]]


def test_a_gemms_output_states_its_rows_and_width_as_its_operands_axes():
    """A dense GEMM lowered for an H100 on any host states its output's dimensions on its call
    (`outputs.OUTPUT_DIMS`): its rows its rows operand's first axis, its width its matrix's last, which a
    compile's live buffers read back to the step's tokens and the layer's width."""
    x = jax.ShapeDtypeStruct((4096, 256), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((256, 512), jnp.bfloat16)
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        lowered = jax.jit(lambda a, b: gemm.gemm(a, b, None, out_dtype=jnp.bfloat16)).trace(x, w).lower(
            lowering_platforms=("cuda",))
    ((stated,),) = declared_outputs(lowered)
    rows, width = stated
    assert rows == [[0, 0]] and [operand for operand, _ in width] == [1], stated
