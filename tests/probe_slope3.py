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
def traced(slope):
    hooked = template.Hooks(score_mod=lambda s, q_pos, k_pos, slope: slope * s,
                            score_params=(template.Param(slope, template.Kind.HEAD),))
    return attention_portable.backward(query, key, value, spans, hooked, dout, scale=4.0, **tiles).query


print("traced param jit dq", np.asarray(jax.jit(traced)(jnp.full((1,), 2e38, jnp.float32)))[0, 0, 0, :2])


def traced_reference(slope):
    hooked = template.Hooks(score_mod=lambda s, q_pos, k_pos, slope: slope * s,
                            score_params=(template.Param(slope, template.Kind.HEAD),))
    return jax.vjp(lambda q: template.reference(q, key, value, spans, hooked, scale=4.0), query)[1](dout)[0]


# print("traced param reference dq", np.asarray(jax.jit(traced_reference)(jnp.full((1,), 2e38, jnp.float32)))[0, 0, 0, :2])
