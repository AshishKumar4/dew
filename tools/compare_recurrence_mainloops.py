"""Compare the recurrence template's portable mainloop (`dew.nn.kernels.templates.recurrence.xla`) with the
GPU mainloop run interpreted, value by value: a debugging aid, not a gate (the accuracy gate and the
scoreboard judge the template). Both call the algebra's tile functions (`recurrence_algebra`), so where they
differ, one mainloop loads, stores or issues something the other does not. Prints, for each launch's stored
value (the inverse, `w`, `u`, the memories entered, the corrected values, the output, the final memory), the
largest difference and where it sits, for a decay a head (the gated delta rule) and a key dimension (Kimi
delta attention), fp32 and bf16, one sequence a row and packed documents, keys 64 and 128 wide.

    PYTHONPATH=src:tests python tools/compare_recurrence_mainloops.py
"""


import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np

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


def gated(dtype, packed: bool, width: int) -> dict:
    """Each stored value's largest difference between the mainloops, a decay a head."""
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

    real, found = real_tokens(spans, batch, length), {}
    for name, gpu_value, portable_value in (("inverse", residual[5], a_x), ("w", w, w_x),
                                            ("corrected", corrected, corrected_x), ("output", out, out_x)):
        found[name] = apart(np.asarray(portable_value)[real], np.asarray(gpu_value)[real])
    chunks = np.asarray(spans.chunks)
    entered = np.unique(chunks[chunks[:, 1] > 0, 2])
    found["memories entered"] = apart(np.asarray(h_x)[entered], np.asarray(h)[entered])
    found["final memory"] = apart(np.asarray(final_x), np.asarray(final))
    return found


def real_tokens(spans, batch: int, length: int):
    """The tokens some sequence of `spans` holds, `[B, T]`: a kernel stores no other."""
    real = np.zeros(batch * length, bool)
    for start, count in np.asarray(spans.spans)[:, :2]:
        real[start:start + count] = True
    return real.reshape(batch, length)


def key_decayed(dtype, packed: bool, width: int) -> dict:
    """Each stored value's largest difference between the mainloops, a decay a key dimension (Kimi delta
    attention's launches)."""
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

    real, found = real_tokens(spans, batch, length), {}
    for name, gpu_value, portable_value in (
            ("scores", residual[5], aqk_x), ("inverse", residual[6], akk_x), ("w", w, w_x),
            ("gated keys", kg, kg_x), ("gated queries", qg, qg_x), ("corrected", corrected, corrected_x),
            ("output", out, out_x)):
        found[name] = apart(np.asarray(portable_value)[real], np.asarray(gpu_value)[real])
    chunks = np.asarray(spans.chunks)
    entered = np.unique(chunks[chunks[:, 1] > 0, 2])
    found["memories entered"] = apart(np.asarray(h_x)[entered], np.asarray(h)[entered])
    found["final memory"] = apart(np.asarray(final_x), np.asarray(final))
    return found


def apart(mine: np.ndarray, theirs: np.ndarray) -> tuple[float, int]:
    """The largest absolute difference of two arrays (as fp32) and how many elements differ."""
    mine, theirs = mine.astype(np.float32), theirs.astype(np.float32)
    return float(np.max(np.abs(mine - theirs), initial=0.0)), int(np.sum(mine != theirs))


def main() -> None:
    cases = itertools.product((("a decay a head", gated), ("a decay a key dimension", key_decayed)),
                              (jnp.float32, jnp.bfloat16), (False, True), (64, 128))
    for (name, compare), dtype, packed, width in cases:
        found = compare(dtype, packed, width)
        differing = {value: f"{most:.2e} at {count}" for value, (most, count) in found.items() if count}
        print(f"{name}, {jnp.dtype(dtype).name}, {'packed' if packed else 'rows'}, keys {width}: "
              f"{differing or 'equal'}")


if __name__ == "__main__":
    main()
