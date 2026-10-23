"""Independent log-decay inputs expose cancellation between chunk-algebra paths."""
import sys

import jax
import jax.numpy as jnp
import numpy as np
import test_recurrence_stability as st
from reference_error import widened

from dew.nn.kernels import recurrence_algebra as a, recurrence_chunk as c

NAMES = ("inverse_pairs", "local_pairs", "write_pairs", "apply_grown", "memory_grown", "carry_grown")


def paths(op, gs):
    q, k, v, beta = op.query, op.key, op.value, op.beta
    b, s, h, dv = v.shape
    dk, size = k.shape[-1], st.CHUNK
    q, k = (jnp.repeat(x, h // x.shape[2], axis=2) for x in (q, k))
    padding = (-s) % size

    def block(x):
        x = jnp.pad(x, [(0, 0), (0, padding)] + [(0, 0)] * (x.ndim - 2))
        x = jnp.moveaxis(x, 2, 1)
        return x.reshape(b, h, -1, size, x.shape[-1])

    query, key, keyed, valued = (block(x) for x in (q * dk**-.5, k, k * beta[..., None],
                                                    v * beta[..., None]))
    decay = [a.decays(block(g[..., None])) for g in gs]
    inverse = a.factor(a.intra(key, keyed, decay[0][1]))
    w, u = a.apply(inverse, keyed, valued, decay[3][0])
    kept = jnp.ones((1, 1, query.shape[2], 1, 1), v.dtype)
    entered, corrected, state = c.xla_chunk_states(key, w, u, decay[5][0],
                                                   decay[2][1][..., -1, :, :], op.state, kept)
    out = a.readout(query, decay[4][0], entered, key, decay[1][1], corrected)
    return jnp.moveaxis(out.reshape(b, h, -1, dv), 1, 2)[:, :s], state


def rms(x):
    return np.sqrt(np.mean(np.asarray(x, np.float64) ** 2))


def run(seed):
    with jax.enable_x64(False):
        op, cot = st.cotangents(st.SITES["scalar delta"], "aligned keys", seed)
    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def gradients(op, cot):
        out, pull = jax.vjp(lambda gs: paths(op, gs), (op.log_decay,) * len(NAMES))
        return out, pull(cot)[0]

    _, grads32 = gradients(op, cot)
    _, grads64 = gradients(*widened((op, cot)))
    sum32 = sum(np.asarray(x, np.float64) for x in grads32)
    sum64 = sum(np.asarray(x) for x in grads64)
    print(f"sum|magnitude={rms(sum64):.10g}|error={rms(sum32-sum64):.10g}")
    for name, g32, g64 in zip(NAMES, grads32, grads64, strict=True):
        print(f"{name}|magnitude={rms(g64):.10g}|error={rms(np.asarray(g32)-np.asarray(g64)):.10g}"
              f"|correlation={np.corrcoef(np.asarray(g64).ravel(),sum64.ravel())[0,1]:.6g}")
    print("sum_abs_paths_over_abs_total", sum(rms(g) for g in grads64) / rms(sum64))
    print("sum32jax_error", rms(np.asarray(sum(grads32)) - sum64))


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
