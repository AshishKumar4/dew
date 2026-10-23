"""Debug (not for landing): does x64 change a dropout mask, and do split and fold_in keys coincide."""
import jax
import jax.numpy as jnp
import numpy as np

key = jax.random.key(5)
narrow = np.asarray(jax.random.bernoulli(key, p=1.0 - 0.5, shape=(4096,)))
with jax.enable_x64(True):
    wide = np.asarray(jax.random.bernoulli(key, p=1.0 - 0.5, shape=(4096,)))
    pinned = np.asarray(jax.random.bernoulli(key, p=1.0 - np.float32(0.5), shape=(4096,)))
print("MASK python-float x64 equal", bool(np.array_equal(narrow, wide)), "differ", int(np.sum(narrow != wide)))
print("MASK float32-rate x64 equal", bool(np.array_equal(narrow, pinned)))
keys = jax.random.split(jax.random.key(9), 7)
same = [bool(np.array_equal(jax.random.key_data(keys[i]), jax.random.key_data(jax.random.fold_in(jax.random.key(9), i))))
        for i in range(7)]
print("KEYS split[i]==fold_in(i)", same, "partitionable", jax.config.jax_threefry_partitionable)
