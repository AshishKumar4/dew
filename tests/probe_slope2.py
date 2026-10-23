"""Measurement script (never lands): eager against jit for the slope witness, and power2 under jit."""
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
hooks = template.Hooks(score_mod=lambda s, q_pos, k_pos: 2e38 * s)
tiles = {"block_q": 32, "block_k": 32, "pad": attention_triton.BLOCK}
with jax.disable_jit():
    eager = attention_portable.backward(query, key, value, spans, hooks, dout, scale=4.0, **tiles)
print("eager dq", np.asarray(eager.query)[0, 0, 0, :2], "dk", np.asarray(eager.key)[0, :, 0, 0])
for k in (100.0, 200.0, 254.0, -200.0):
    print("power2 jit 0 by", k, float(jax.jit(algebra.power2)(jnp.float32(0.0), jnp.float32(k))),
          "1 by", float(jax.jit(algebra.power2)(jnp.float32(2.0 ** -100), jnp.float32(k))))
x = jnp.asarray([0.0, 2e38, 0.5], jnp.float32)
print("split jit", jax.jit(algebra._split)(x))
slope = jax.jvp(lambda s: 2e38 * s, (jnp.float32(0.0),), (jnp.float32(1.0),))[1]
print("slope", float(slope), "slope x 4", float(slope * 4))
