import numpy as np, jax, jax.numpy as jnp
from dew.nn.kernels.recurrence_chunk import chunk_gated_delta_rule
rng = np.random.default_rng(0)
B, S, H, Dk, Dv = 1, 10, 2, 8, 8
def arr(*shape): return jnp.asarray(rng.normal(size=shape), jnp.float32)
q, k, v = arr(B, S, H, Dk), arr(B, S, H, Dk), arr(B, S, H, Dv)
k = k / jnp.linalg.norm(k, axis=-1, keepdims=True); q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
g = -jnp.abs(arr(B, S, H)) * 0.1; beta = jax.nn.sigmoid(arr(B, S, H))
state = arr(B, 3, H, Dk, Dv) * 0.1
for name, starts_at, own in (("doc at 0, masked tail", [0], range(0, 5)), ("masked head, doc at 3", [3], range(3, 8)),
                             ("two docs, second followed by masked", [0, 3], range(3, 6))):
    valid = np.zeros((B, S), bool)
    starts = np.zeros((B, S), bool)
    for s in starts_at: starts[0, s] = True
    if name.startswith("two"):
        valid[0, 0:6] = True
    else:
        valid[0, list(own)] = True
    gm = jnp.where(valid[..., None], g, 0.0); bm = jnp.where(valid[..., None], beta, 0.0)
    entered = jnp.asarray([[2, 3, 0][:3]], jnp.int32) if name.startswith("two") else jnp.asarray([[2, 0, 0]], jnp.int32)
    out, fin = chunk_gated_delta_rule(q, k, v, gm, bm, state, 64, starts=jnp.asarray(starts), max_segments=3,
                                      write_bound=1.0, entered=entered)
    o = list(own)
    slot = 2 if name.startswith("two") else 1
    alone, _ = chunk_gated_delta_rule(q[:, o], k[:, o], v[:, o], g[:, o], beta[:, o], state[:, slot], 64,
                                      write_bound=1.0)
    print(name, "%.2e" % float(jnp.max(jnp.abs(out[:, o] - alone))))
