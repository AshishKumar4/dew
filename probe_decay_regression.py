import sys; sys.path[:0]=["src","tests","."]
import os
import lane_environment
os.environ.update(lane_environment.configured(os.environ))
import jax
import jax.numpy as jnp
import numpy as np
from dew.algebra import scan
from test_recurrence_stability import SITES, stressed

(x,) = stressed(SITES["rated"], "decays near zero", 0, value_heads=64)
def probe(rate, beta):
    exponents = (rate * beta).reshape(2, 16, 16, 64)
    exponents = jnp.transpose(exponents, (0, 3, 1, 2)).reshape(-1, 16, 1)
    decays = jax.vmap(lambda z: scan._decays_value(z, False))(exponents)
    def backward(d):
        cot = scan._Decays(jnp.zeros_like(d.entering), jnp.zeros_like(d.to_last),
                          jnp.zeros_like(d.whole),
                          jnp.eye(16, k=-1, dtype=jnp.float32)[:, :, None])
        return scan._decays_backward(False, d, cot)[0]
    return decays.between[:, jnp.arange(1,16), jnp.arange(15), 0], jax.vmap(backward)(decays)[:, 1:, 0]
adjacent, gradient = jax.jit(probe)(x.rate, x.beta)
exponents = (np.asarray(x.rate) * np.asarray(x.beta)).reshape(2,16,16,64)
exponents = exponents.transpose(0,3,1,2).reshape(-1,16)
truth = np.exp(exponents[:, 1:].astype(np.float64))
errors = [np.max(np.abs(np.asarray(value)/truth-1)) for value in (adjacent, gradient)]
print("K max relative adjacent decay and exponent gradient errors", errors, flush=True)
assert max(errors) < 8e-6, errors
print("K regression PASS", flush=True)
