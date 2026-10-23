"""The attention template's Mosaic GPU mainloop on the wgmma family
(dew.nn.kernels.templates.attention_mosaic): its forward and its backward in both forms, check 1
(`reference_error.assert_as_exact`) against the template's reference over ordinary draws, under Mosaic GPU's
interpreter on the CPU and compiled on an H100. The interpreter is coverage, not proof (it treats wgmma_wait
as a no-op and takes the register operand's product as a dot); the GPU gate runs the compiled kernels. Its
backward reads the Triton mainloop's training forward's residuals (its split sums), interpreted on the
CPU."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS
from jax.sharding import Mesh
from reference_error import assert_as_exact

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template, attention_mosaic, attention_triton

pytestmark = pytest.mark.kernel

ON_HOPPER = pytest.mark.skipif(generation.device_generation() != "sm90",
                               reason="runs the sm90 mainloop compiled, on an H100")
CALLS = {"full": {}, "causal": {"causal": True}}
"""The call forms the mainloop takes."""
SHAPES = {"interpreted": (1, 160, 2, 1), "compiled": (2, 400, 4, 2)}
"""Rows, tokens (not whole tiles, so the call pads, over more than one key tile past the keys where a program's
two consumers pad it to 128), query and key heads (grouped), interpreted on the CPU and compiled on an H100."""
SEEDS = 8
"""Draws a case takes, check 1's fewest (`reference_error.seeds_for`): it bounds the worst row over them
against the reference's own."""


def _drawn(shape: tuple[int, int, int, int], dtype, width: int = 128):
    """A draw `(query, key, value, dout)` of unit Gaussians at `shape` in `dtype`, by seed."""
    rows, tokens, heads, kv_heads = shape

    def draw(seed):
        streams = jax.random.split(jax.random.key(seed), 4)
        dims = ((heads, tokens), (kv_heads, tokens), (kv_heads, tokens), (heads, tokens))
        return tuple(jax.random.normal(stream, (rows, count, heads_, width), jnp.float32).astype(dtype)
                     for stream, (heads_, count) in zip(streams, dims, strict=True))

    return draw


def _check_forward(call: str, dtype, shape, *, natural: bool, interpret: bool):
    spans = template.spans(shape[1], shape[1], shape[0], **CALLS[call])
    draw = _drawn(shape, dtype)

    def ours(query, key, value, dout):
        del dout
        return attention_mosaic.forward(query, key, value, spans, natural=natural, interpret=interpret)

    def reference(query, key, value, dout):
        del dout
        return template.reference(query, key, value, spans)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw,
                    f"mosaic forward {call} {jnp.dtype(dtype).name} natural {natural}", seeds=SEEDS)


def _check_backward(call: str, shape, *, accurate: bool, interpret: bool):
    spans = template.spans(shape[1], shape[1], shape[0], **CALLS[call])
    draw = _drawn(shape, jnp.bfloat16)

    def ours(query, key, value, dout):
        residuals = attention_triton.forward(query, key, value, spans, residuals=True,
                                             interpret=interpret)[1:]
        grads = attention_mosaic.backward(query, key, value, spans, None, dout, residuals=residuals,
                                          accurate=accurate, interpret=interpret)
        return grads.query, grads.key, grads.value

    def reference(query, key, value, dout):
        _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans), query, key, value)
        return pull(dout)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw,
                    f"mosaic backward {call} accurate {accurate}", seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("call", CALLS)
def test_the_forward_is_on_par_interpreted(call, dtype):
    """The forward, the base-2 fold, under Mosaic GPU's interpreter."""
    _check_forward(call, dtype, SHAPES["interpreted"], natural=False, interpret=True)


@INTERPRETS
@pytest.mark.parametrize("call", CALLS)
def test_the_natural_forward_is_on_par_interpreted(call):
    """The forward with natural logits, fp16's (bf16's takes a headroom guard the mainloop has not yet), under
    the interpreter."""
    _check_forward(call, jnp.float16, SHAPES["interpreted"], natural=True, interpret=True)


@INTERPRETS
@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
@pytest.mark.parametrize("call", CALLS)
def test_the_backward_is_on_par_interpreted(call, accurate):
    """The backward in both forms under the interpreter."""
    _check_backward(call, SHAPES["interpreted"], accurate=accurate, interpret=True)


@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
def test_the_launches_lower_for_an_h100(accurate):
    """The forward and the backward's launches lower for an H100 on any host, as the GPU gate runs them."""
    rows, tokens, heads, kv_heads = SHAPES["compiled"]
    spans = template.spans(tokens, tokens, rows, causal=True)
    q = jax.ShapeDtypeStruct((rows, tokens, heads, 128), jnp.bfloat16)
    kv = jax.ShapeDtypeStruct((rows, tokens, kv_heads, 128), jnp.bfloat16)
    rows_shape = jax.ShapeDtypeStruct((rows, heads, tokens), jnp.float32)
    others = jax.ShapeDtypeStruct((rows, heads, tokens, 128), jnp.float32)
    split = template.Split(others, rows_shape, rows_shape, rows_shape)
    residuals = (rows_shape, rows_shape, jax.ShapeDtypeStruct((rows, heads, tokens), jnp.int32), split)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(functools.partial(attention_mosaic.forward, spans=spans)).trace(q, kv, kv).lower(
            lowering_platforms=("cuda",))
        jax.jit(lambda q, k, v, dout, residuals: attention_mosaic.backward(
            q, k, v, spans, None, dout, residuals=residuals, accurate=accurate)).trace(
            q, kv, kv, q, residuals).lower(lowering_platforms=("cuda",))


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("call", CALLS)
def test_the_forward_is_on_par_on_an_h100(call, dtype):
    """The forward, the base-2 fold, compiled on an H100."""
    _check_forward(call, dtype, SHAPES["compiled"], natural=False, interpret=False)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize("call", CALLS)
def test_the_natural_forward_is_on_par_on_an_h100(call):
    """The forward with natural logits, fp16's, compiled on an H100."""
    _check_forward(call, jnp.float16, SHAPES["compiled"], natural=True, interpret=False)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
@pytest.mark.parametrize("call", CALLS)
def test_the_backward_is_on_par_on_an_h100(call, accurate):
    """The backward in both forms compiled on an H100."""
    _check_backward(call, SHAPES["compiled"], accurate=accurate, interpret=False)
