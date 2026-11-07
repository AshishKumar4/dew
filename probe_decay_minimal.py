import sys; sys.path[:0]=["src","tests","."]
import os
import functools
import lane_environment
os.environ.update(lane_environment.configured(os.environ))
import jax
import jax.numpy as jnp
import numpy as np
from dew.algebra import scan
from dew.nn.precision import compensated_add
from test_recurrence_stability import SITES, stressed

(x,) = stressed(SITES["rated"], "decays near zero", 0)
rate = x.rate[0]
beta = x.beta[0, :16, 0, None]
z = np.asarray(rate) * np.asarray(beta)
truth = np.cumsum(z.astype(np.float64), axis=0)
for mode in ("base", "barrier", "normalized"):
    def prefix(r, dt):
        g = r * dt
        if mode == "barrier":
            g = jax.lax.optimization_barrier(g)
        if mode == "normalized":
            return jax.lax.associative_scan(compensated_add, (g, jnp.zeros_like(g)), axis=0)
        return scan._compensated(g)
    hi, lo = jax.jit(prefix)(rate, beta)
    error = np.asarray(hi).astype(np.float64) + np.asarray(lo).astype(np.float64) - truth
    print("K prefix", mode, "max absolute error", np.max(np.abs(error)), "errors", error[:, 0], flush=True)
    def decay(r, dt):
        g = r * dt
        if mode == "barrier":
            g = jax.lax.optimization_barrier(g)
        return scan._decays_value(g, False)
    ds = jax.jit(decay)(rate, beta)
    adjacent = np.asarray(ds.between)[np.arange(1, 16), np.arange(15), 0]
    ref = np.exp(z[1:, 0].astype(np.float64))
    relative = np.abs(adjacent/ref-1.)
    print("K adjacent", mode, "max relative error", relative.max(),
          "worst token", relative.argmax()+1, "rate", float(rate), flush=True)

for seed in range(4):
    (x,) = stressed(SITES["rated"], "decays near zero", seed, value_heads=64)
    r = x.rate
    dt = x.beta
    def batch_decay(r, dt, barrier):
        g = (r * dt).reshape(2, 16, 16, 64)
        g = jnp.transpose(g, (0, 3, 1, 2)).reshape(-1, 16, 1)
        if barrier:
            g = jax.lax.optimization_barrier(g)
        return jax.vmap(lambda v: scan._decays_value(v, False))(g)
    z = np.asarray(r) * np.asarray(dt)
    z = z.reshape(2, 16, 16, 64).transpose(0, 3, 1, 2).reshape(-1, 16, 1)
    for barrier in (False, True):
        ds = jax.jit(functools.partial(batch_decay, barrier=barrier))(r, dt)
        adjacent = np.asarray(ds.between)[:, np.arange(1,16), np.arange(15), 0]
        relative = np.abs(adjacent/np.exp(z[:, 1:, 0].astype(np.float64))-1.)
        print("K batched adjacent", seed, barrier, "max", relative.max(), "rms", np.sqrt(np.mean(relative**2)), flush=True)
