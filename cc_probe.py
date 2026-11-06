"""Paired timing on one host: N programs lowered in one thread's MLIR context, compiled in turn and all at once
(report.attempts), alternating, R rounds."""
import time, sys
import jax, jax.numpy as jnp
from dew.nn.kernels.templates import report

def program(seed, width=512, depth=24):
    def f(x, ws):
        for w in ws:
            x = jnp.tanh(x @ w) + x
        return jnp.sum(x * seed)
    x = jnp.ones((256, width)); ws = [jnp.ones((width, width)) * (i + seed) for i in range(depth)]
    return jax.jit(jax.grad(f)).lower(x, ws)

N, R = int(sys.argv[1]), int(sys.argv[2])
for r in range(R):
    for mode in (("turn", "once") if r % 2 == 0 else ("once", "turn")):
        lowered = [program(s + 100 * r + (0 if mode == "turn" else 50)) for s in range(N)]
        start = time.perf_counter()
        if mode == "turn":
            [report.attempted(low, None) for low in lowered]
        else:
            report.attempts([(low, None) for low in lowered])
        print(r, mode, round(time.perf_counter() - start, 2), flush=True)
