"""Measurement script (never lands): where the fp32 edge family's check-1 time goes, one seed."""
import time

import jax
import numpy as np
import test_attention_stability as suite
from reference_error import with_gradients

name = "a weight at the exponent's edge on a large value, float32"
family = suite.FAMILIES[name]
args = suite._cast(family, family.draw)(0)
arms = {"reference": jax.jit(with_gradients(suite._reference(family, anchored=False))),
        **{loop: jax.jit(with_gradients(fn)) for loop, fn in suite._mainloops(family, suite.FORMS["default"]).items()}}
for arm, fn in arms.items():
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    first = time.perf_counter() - start
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    print(arm, "first", round(first, 1), "again", round(time.perf_counter() - start, 1), flush=True)
truth = with_gradients(suite._reference(family, anchored=True))
start = time.perf_counter()
with jax.enable_x64(True):
    wide = [None if a is None else np.asarray(a, np.float64) for a in args]
    jax.block_until_ready(truth(*wide))
print("truth", round(time.perf_counter() - start, 1), flush=True)
