"""Measurement script (never lands): the slope witness's NaN, eager against jit, piece by piece."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from dew.nn.kernels.templates import attention as template
from dew.nn.kernels.templates import attention_algebra as algebra
from dew.nn.kernels.templates import attention_portable, attention_triton

query = jnp.zeros((1, 1, 1, 16), jnp.float32)
key = jnp.zeros((1, 2, 1, 16), jnp.float32).at[:, :, :, 0].set(0.25)
value = jnp.zeros((1, 2, 1, 16), jnp.float32).at[..., 0].set(1.0)
dout = jnp.zeros((1, 1, 1, 16), jnp.float32).at[..., 0].set(1.0)
spans = template.spans(1, 2, 1)
hooks = template.Hooks(score_mod=lambda s, q_pos, k_pos, slope: slope * s,
                       score_params=(template.Param(jnp.full((1,), 2e38, jnp.float32), template.Kind.HEAD),))
tiles = {"block_q": 32, "block_k": 32, "pad": attention_triton.BLOCK}
run = functools.partial(attention_portable.backward, spans=spans, hooks=hooks, scale=4.0, **tiles)
with jax.disable_jit():
    eager = run(query, key, value, dout=dout)
print("eager", np.asarray(eager.query)[0, 0, 0, :2], [np.asarray(p).ravel()[:2] for p in jax.tree.leaves(eager.score_params)] if hasattr(eager, "score_params") else "")
jitted = jax.jit(lambda q, k, v, d: run(q, k, v, dout=d))(query, key, value, dout)
print("jit", np.asarray(jitted.query)[0, 0, 0, :2])
# The pieces under jit.
w = jnp.asarray([[0.5, 0.5]], jnp.float32)
slopes = (jnp.full((1, 2), 2e38, jnp.float32), jnp.full((1, 2), 0.0, jnp.float32))
for name, fn in (("split 1e38", lambda x: algebra._split(x)), ("power2 1e38 by -127", lambda x: algebra.power2(x, jnp.float32(-127.0))),
                 ("inverse 2", lambda x: algebra.inverse(x))):
    x = jnp.asarray([1e38], jnp.float32) if "inverse" not in name else jnp.asarray([2.0], jnp.float32)
    print(name, "jit", jax.jit(fn)(x), "eager", fn(x))
