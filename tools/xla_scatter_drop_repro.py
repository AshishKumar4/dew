"""XLA's deterministic GPU scatter writes a dropped column index into the next row.

    XLA_FLAGS=--xla_gpu_deterministic_ops=true JAX_PLATFORMS=cuda python tools/xla_scatter_drop_repro.py

Row 0 writes column 0 and, at column 4 (past the end), a value that must be
dropped; row 1 writes 3 at column 0. The expected output is [[1, 0, 0, 0],
[3, 0, 0, 0]]. With the flag, XLA on GPU returns [[1, 0, 0, 0], [2, 0, 0, 0]]:
the dropped (0, 4) is linearized to position 4, which is (1, 0). Without the
flag, or with the dropped index at 8 (past the whole array), it is correct.
"""
import jax
import jax.numpy as jnp

scatter = jax.jit(lambda buffer, rows, columns, values: buffer.at[rows, columns].set(values, mode="drop"))
print(scatter(jnp.zeros((2, 4), jnp.int32), jnp.arange(2)[:, None], jnp.array([[0, 4], [0, 4]]),
              jnp.array([[1, 2], [3, 4]], jnp.int32)).tolist())
