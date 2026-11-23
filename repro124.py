import jax, jax.numpy as jnp, jaxlib
print("jax", jax.__version__, "jaxlib", jaxlib.__version__)
@jax.jit
def f(x):
    r = jax.lax.reduce(x, jnp.float32(0.0), lambda a, b: jnp.where(a > b, a, b), (1,))
    return jnp.pad(r[:0], (1, 0)), x.sum()
padded, total = f(jnp.ones((2, 2), jnp.float32))
assert padded.tolist() == [0.0] and float(total) == 4.0, (padded, total)
print("dead reduce with a called combiner: compiled and computed")
from jax._src.lib.mosaic_gpu import _mosaic_gpu_ext
print("assembler:", _mosaic_gpu_ext._get_assembler())
