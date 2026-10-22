"""The recurrence template's mainloops against each other (the template design's §6, test 1): at the same
tiles the portable mainloop (`dew.nn.kernels.templates.recurrence.xla`) computes what the GPU mainloop
interpreted computes, bit for bit, every value each launch stores: the inverse, `w`, `u`, the memories
the chunks enter, the corrected values, the output and the final memory. Both call the algebra's tile
functions (`recurrence_algebra`); a difference is a drift in what one mainloop loads, stores or issues.

The cases: fp32 and bf16 operands; a row of one sequence and packed documents (a document ending inside
a chunk, one starting on a chunk's first token, the row's padding a sequence of its own); value heads
grouped over a key head; keys 64 and 128 wide, the memory in one and two key slices; a ragged tail.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.kernels import fla_gdn, fla_kda, recurrence
from dew.nn.kernels.chunked_launch import _dtype, _wide, sequences, slots
from dew.nn.kernels.templates.recurrence import xla

CHUNK = 64

LAUNCHES = {"solve": fla_gdn.KktSolve, "weighted": fla_gdn.RecomputeWU, "states": fla_gdn.ChunkFwdH,
            "output": fla_gdn.ChunkFwdO}
"""Each portable launch's GPU launch, whose heuristic tiles both mainloops take."""

KEY_LAUNCHES = {"rows": fla_kda.IntraRows, "weighted": fla_kda.RecomputeWU, "states": fla_gdn.ChunkFwdH,
                "output": fla_kda.ChunkFwdO}
"""The same under a decay a key dimension (Kimi delta attention's launches)."""


def operands(dtype, width: int, seed: int = 0, *, per_key: bool = False):
    """Two rows of 150 tokens, a key head over two value heads, `width`-wide keys and 64-wide values; a
    decay a head, or a key dimension's no steeper than -5 a token (Kimi delta attention's floor)."""
    keys = jax.random.split(jax.random.key(seed), 6)
    batch, length, heads = 2, 150, 2
    q = jax.random.normal(keys[0], (batch, length, 1, width)).astype(dtype)
    k = jax.random.normal(keys[1], (batch, length, 1, width))
    k = (k / jnp.linalg.norm(k, axis=-1, keepdims=True)).astype(dtype)
    v = jax.random.normal(keys[2], (batch, length, heads, 64)).astype(dtype)
    g = -jax.nn.softplus(jax.random.normal(keys[3], (batch, length, heads, *((width,) if per_key else ()))))
    g = jnp.maximum(g, -5.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, length, heads))).astype(dtype)
    h0 = jax.random.normal(keys[5], (batch, heads, width, 64))
    return q, k, v, g, beta, h0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("packed", [False, True], ids=["rows", "packed"])
@pytest.mark.parametrize("width", [64, 128])
def test_the_portable_mainloop_is_the_gpu_mainloop_interpreted_bit_for_bit(dtype, packed, width):
    """Every value a launch stores, the portable mainloop's equal to the interpreted GPU mainloop's."""
    q, k, v, g, beta, h0 = operands(dtype, width)
    batch, length = q.shape[:2]
    cu_seqlens = jnp.asarray([0, 40, 150, 192, 300], jnp.int32) if packed else None
    spans = sequences(batch, length, cu_seqlens, chunk=CHUNK)
    scale = width ** -0.5
    prefix = recurrence.cumulated(g, scale=fla_gdn.LOG2_E, chunk=CHUNK, spans=spans, interpret=True,
                                  compensated=True)
    tiles = {name: launch.PICKS[_dtype(k), _wide()] for name, launch in LAUNCHES.items()}

    gpu = jax.jit(lambda *a: fla_gdn.cumulated_fwd(*a, spans, scale, CHUNK, interpret=True))
    (out, final), residual = gpu(q, k, v, prefix, beta, h0)
    w, h, corrected = residual[7]
    portable = jax.jit(lambda *a: xla.forward(*a, spans, scale=scale, chunk=CHUNK,
                                              held=slots(spans, batch, length, CHUNK), tiles=tiles))
    (out_x, final_x), (a_x, w_x, _, h_x, corrected_x) = portable(q, k, v, prefix, beta, h0)

    real = np.zeros(batch * length, bool)
    for start, count in np.asarray(spans.spans)[:, :2]:
        real[start:start + count] = True
    real = real.reshape(batch, length)
    for name, gpu_value, portable_value in (("inverse", residual[5], a_x), ("w", w, w_x),
                                            ("corrected", corrected, corrected_x), ("output", out, out_x)):
        np.testing.assert_array_equal(np.asarray(portable_value)[real], np.asarray(gpu_value)[real],
                                      err_msg=name)
    chunks = np.asarray(spans.chunks)
    entered = np.unique(chunks[chunks[:, 1] > 0, 2])
    np.testing.assert_array_equal(np.asarray(h_x)[entered], np.asarray(h)[entered],
                                  err_msg="memories entered")
    np.testing.assert_array_equal(np.asarray(final_x), np.asarray(final), err_msg="final memory")


def real_tokens(spans, batch: int, length: int):
    """The tokens some sequence of `spans` holds, `[B, T]`: a kernel stores no other."""
    real = np.zeros(batch * length, bool)
    for start, count in np.asarray(spans.spans)[:, :2]:
        real[start:start + count] = True
    return real.reshape(batch, length)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("packed", [False, True], ids=["rows", "packed"])
@pytest.mark.parametrize("width", [64, 128])
def test_the_portable_mainloop_takes_a_decay_a_key_dimension_as_the_gpu_mainloop_bit_for_bit(dtype, packed,
                                                                                              width):
    """Kimi delta attention's launches: every value stored, the portable mainloop's equal to the interpreted
    GPU mainloop's."""
    q, k, v, g, beta, h0 = operands(dtype, width, per_key=True)
    batch, length = q.shape[:2]
    cu_seqlens = jnp.asarray([0, 40, 150, 192, 300], jnp.int32) if packed else None
    spans = sequences(batch, length, cu_seqlens, chunk=CHUNK)
    scale = width ** -0.5
    prefix = recurrence.cumulated(g, scale=fla_gdn.LOG2_E, chunk=CHUNK, spans=spans, interpret=True,
                                  compensated=True)
    tiles = {name: launch.PICKS[_dtype(k), _wide()] for name, launch in KEY_LAUNCHES.items()}
    heads = v.shape[2]
    rows = tiles["rows"]
    tiles["rows"] = type(rows)(rows.bk, rows.bv, rows.warps, rows.stages,
                               math.gcd(rows.bh, heads) if heads == k.shape[2] else 1)

    gpu = jax.jit(lambda *a: fla_kda.cumulated_fwd(*a, spans, scale, CHUNK, interpret=True))
    (out, final), residual = gpu(q, k, v, prefix, beta, h0)
    w, kg, qg, h, corrected = residual[8]
    portable = jax.jit(lambda *a: xla.key_forward(*a, spans, scale=scale, chunk=CHUNK,
                                                  held=slots(spans, batch, length, CHUNK), tiles=tiles))
    (out_x, final_x), stored = portable(q, k, v, prefix, beta, h0)
    aqk_x, akk_x, w_x, _, kg_x, qg_x, h_x, corrected_x = stored

    real = real_tokens(spans, batch, length)
    for name, gpu_value, portable_value in (
            ("scores", residual[5], aqk_x), ("inverse", residual[6], akk_x), ("w", w, w_x),
            ("gated keys", kg, kg_x), ("gated queries", qg, qg_x), ("corrected", corrected, corrected_x),
            ("output", out, out_x)):
        np.testing.assert_array_equal(np.asarray(portable_value)[real], np.asarray(gpu_value)[real],
                                      err_msg=name)
    chunks = np.asarray(spans.chunks)
    entered = np.unique(chunks[chunks[:, 1] > 0, 2])
    np.testing.assert_array_equal(np.asarray(h_x)[entered], np.asarray(h)[entered],
                                  err_msg="memories entered")
    np.testing.assert_array_equal(np.asarray(final_x), np.asarray(final), err_msg="final memory")
