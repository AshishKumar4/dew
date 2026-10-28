import os
os.environ["MOSAIC_GPU_DUMP_CONSTRAINT_SYSTEM"] = "1"
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from dew.nn.kernels import generation
from dew.nn.kernels.templates import mosaic_sm80
from jax.experimental.mosaic.gpu import utils as mgutils
mgutils.dump_to_file_or_stdout = lambda content, *args: print("DUMP", args[0], len(str(content).splitlines()))
from jax.sharding import Mesh
family = mosaic_sm80.Mma(jnp.dtype(jnp.bfloat16))
mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm80")
def run(x, fixed, count):
    def entry(x_ref, out_ref):
        def body(smem):
            part = family.rows_of(x_ref)
            if fixed:
                mosaic_sm80.plgpu.store(out_ref, part, optimized=False)
            else:
                family.stored((part,), (out_ref,), smem=smem)
        pl.run_scoped(body, mosaic_sm80.plgpu.SMEM((count,), jnp.float32))
    return family.kernel(entry, out_type=jax.ShapeDtypeStruct((count,), jnp.float32),
                         grid=(1,), grid_names=("blocks",), consumers=1, interpret=False,
                         name=f"vector_{fixed}")(x)
for count, fixed in ((64, False), (64, True), (128, False)):
    try:
        with generation.assumed("sm80"), jax.sharding.use_abstract_mesh(mesh):
            jax.jit(lambda x: run(x, fixed, count)).trace(jax.ShapeDtypeStruct((count,), jnp.float32)).lower(lowering_platforms=("cuda",))
        print("LOWERED", count, fixed)
    except Exception as e:
        print("FAILED", count, fixed, type(e).__name__, str(e))
