"""The programs the compile-ahead GPU cell builds where no GPU is visible and runs on the GPU, and the
arguments it runs them on, declared once for the build, the run and the jitted reference
(tests/test_gpu_compile_ahead.py)."""

import jax
import jax.numpy as jnp

from dew.nn.attention import scaled_dot_product_attention


def attention(q, k, v):
    return scaled_dot_product_attention(q, k, v, implementation="template", causal=True)


def programs(generation: str) -> dict:
    """Each program's function and argument shapes, by its name, for a GPU of `generation`: the template's
    forward on Hopper alone, where it is Mosaic GPU's, the route the pilot ran ahead."""
    square = jax.ShapeDtypeStruct((128, 128), jnp.float32)
    found = {"dot": (lambda x, y: x @ y, [square, square])}
    if generation == "sm90":
        query = jax.ShapeDtypeStruct((1, 256, 8, 128), jnp.bfloat16)
        key = jax.ShapeDtypeStruct((1, 256, 2, 128), jnp.bfloat16)
        found["attention"] = (attention, [query, key, key])
    return found


def arguments(generation: str) -> dict:
    """Each program's arguments, drawn for its shapes (`programs`)."""
    drawn = {}
    for index, (name, (_, shapes)) in enumerate(programs(generation).items()):
        keys = jax.random.split(jax.random.key(index), len(shapes))
        drawn[name] = [jax.random.normal(key, shape.shape, shape.dtype)
                       for key, shape in zip(keys, shapes, strict=True)]
    return drawn
