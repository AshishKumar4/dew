"""Programs compiled ahead for this GPU's generation, on no device of it, load here and run with no trace,
lowering or compile, bit for bit as the same programs jitted (ledger 2716: a GPU node spends its time on its
device's work alone; 2026-11-02, an H100 ran a CPU host's sm90 build of a dot and the template attention's
forward so)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.attention import scaled_dot_product_attention
from dew.training.ahead import BuildTarget, compile_programs, executable_only

pytestmark = pytest.mark.gpu(reason="a program compiled ahead runs on the GPU it was built for")


def attention(q, k, v):
    return scaled_dot_product_attention(q, k, v, implementation="template", causal=True)


def test_programs_compiled_ahead_for_this_gpu_run_bit_for_bit_as_jitted():
    generation = BuildTarget.of(jax.devices()).generation
    keys = jax.random.split(jax.random.key(0), 5)
    dot_args = [jax.random.normal(key, (128, 128), jnp.float32) for key in keys[:2]]
    attention_args = [jax.random.normal(keys[2], (1, 256, 8, 128), jnp.bfloat16),
                      *(jax.random.normal(key, (1, 256, 2, 128), jnp.bfloat16) for key in keys[3:])]
    functions = {"dot": (lambda x, y: x @ y, dot_args), "attention": (attention, attention_args)}
    # The template's forward is Mosaic GPU's on Hopper alone, the route the pilot ran ahead.
    if generation != "sm90":
        functions.pop("attention")
    built = compile_programs({name: (function, [jax.ShapeDtypeStruct(arg.shape, arg.dtype) for arg in args])
                              for name, (function, args) in functions.items()}, generation)
    loaded = built.loaded()
    with executable_only():
        ahead = {name: np.asarray(jax.block_until_ready(loaded[name](*args)))
                 for name, (_, args) in functions.items()}
    for name, (function, args) in functions.items():
        jitted = np.asarray(jax.jit(function)(*args))
        assert ahead[name].dtype == jitted.dtype and ahead[name].tobytes() == jitted.tobytes(), name
