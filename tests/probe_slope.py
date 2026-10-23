"""Measurement script (never lands): where the slope-2e38 witness's NaN arises."""
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from dew.nn.kernels import tuning
from dew.nn.kernels.templates import attention as template
from dew.nn.kernels.templates import attention_portable, attention_triton
from test_attention_algebra import TILES
from test_attention_triton import joint

query = jnp.zeros((1, 1, 1, 16), jnp.float32)
key = jnp.zeros((1, 2, 1, 16), jnp.float32).at[:, :, :, 0].set(0.25)
value = jnp.zeros((1, 2, 1, 16), jnp.float32).at[..., 0].set(1.0)
dout = jnp.zeros((1, 1, 1, 16), jnp.float32).at[..., 0].set(1.0)
spans = template.spans(1, 2, 1)
hooks = template.Hooks(score_mod=lambda s, q_pos, k_pos: 2e38 * s)
tiles = {"block_q": 32, "block_k": 32, "pad": attention_triton.BLOCK}
portable = attention_portable.backward(query, key, value, spans, hooks, dout, scale=4.0, **tiles)
print("portable dq", np.asarray(portable.query)[0, 0, 0, :2])
with tuning.pinned({attention_triton.TritonBackward: joint(TILES[1])}):
    triton = attention_triton.backward(query, key, value, spans, hooks, dout, scale=4.0, interpret=True)
    print("triton dq", np.asarray(triton.query)[0, 0, 0, :2])
    for split in (True, False):
        g = attention_triton.backward(query, key, value, spans, hooks, dout, scale=4.0, split=split, interpret=True)
        print("triton split", split, np.asarray(g.query)[0, 0, 0, :2], np.asarray(g.key)[0, :, 0, 0])
    fwd = attention_triton.forward(query, key, value, spans, hooks, residuals=True, interpret=True)
    for i, leaf in enumerate(jax.tree.leaves(fwd)):
        print("fwd leaf", i, np.asarray(leaf).ravel()[:4])
jax.config.update("jax_debug_nans", True)
try:
    with tuning.pinned({attention_triton.TritonBackward: joint(TILES[1])}):
        attention_triton.backward(query, key, value, spans, hooks, dout, scale=4.0, interpret=True)
except FloatingPointError:
    print("\n".join(line for line in traceback.format_exc().split("\n") if "dew/" in line)[-3000:])
